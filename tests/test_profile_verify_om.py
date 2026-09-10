"""Host-only checks of capture orchestration and CSV arithmetic, not NPU timings."""
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/profile_verify_om.py"
spec = importlib.util.spec_from_file_location("profile_verify_om", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
import profile_om as unified
from msprof_summary import summarize_windows


def operator_csv(path, rows, header=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header or ["Op Name", "OP Type", "Task Type", "OP State",
                                   "Task Duration(us)", "Input Shapes", "Input Data Types"])
        writer.writerows(rows)


def test_summary_units_grouping_and_separate_exports(tmp_path):
    # Synthetic task durations: no device execution or performance evidence.
    capture, output = tmp_path / "capture", tmp_path / "reports"
    output.mkdir()
    operator_csv(capture / "one/op_summary_0.csv", [
        ["TEST_1", "GDR", "AI_CORE", "static", 1000, "1;16", "FLOAT16"],
        ["TEST_2", "GDR", "AI_CORE", "static", 3000, "1;16", "FLOAT16"],
        ["TEST_3", "GDR", "AI_CPU", "dynamic", 9000, "1;16", "FLOAT16"],
        *[["BAD", "GDR", "AI_CORE", "static", value, "", ""]
          for value in ("N/A", "nan", "inf", -1)],
    ])
    operator_csv(capture / "two/op_summary_0.csv", [
        ["TEST_4", "GDR", "AI_CORE", "static", 10000, "1;16", "FLOAT16"],
    ])
    text = profile.summarize(capture, output)
    with (output / "operator-types.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3
    assert [(row["source_csv"], row["task_type"], float(row["total_ms"])) for row in rows] == [
        ("one/op_summary_0.csv", "AI_CPU", 9),
        ("one/op_summary_0.csv", "AI_CORE", 4),
        ("two/op_summary_0.csv", "AI_CORE", 10),
    ]
    assert rows[1]["count"] == "2"
    assert float(rows[1]["mean_ms"]) == 2
    assert float(rows[1]["max_ms"]) == 3
    assert "skipped invalid durations: 4" in text
    assert "not stage wall time" in text
    assert "exports are not added together" in text
    with (output / "operator-tasks.csv").open(newline="") as stream:
        tasks = list(csv.DictReader(stream))
    assert [row["op_name"] for row in tasks] == ["TEST_3", "TEST_2", "TEST_1", "TEST_4"]
    assert tasks[0]["input_data_types"] == "FLOAT16"


@pytest.mark.parametrize("kind", ["missing", "wrong_unit", "empty", "invalid"])
def test_unusable_export_is_not_a_successful_timing_report(tmp_path, kind):
    if kind == "wrong_unit":
        operator_csv(tmp_path / "op_summary.csv", [["TEST", 10]], ["Op Name", "Task Duration(ms)"])
    elif kind in {"empty", "invalid"}:
        operator_csv(tmp_path / "op_summary.csv", [] if kind == "empty" else [["TEST", "nan"]],
                     ["Op Name", "Task Duration(us)"])
    with pytest.raises(ValueError):
        profile.summarize(tmp_path, tmp_path)
    assert not (tmp_path / "hotspots.txt").exists()


@pytest.mark.parametrize("value", [[], [True], [-1], [1.5], "1,,2", "1,2.5"])
def test_invalid_token_ids(value):
    with pytest.raises(ValueError):
        profile.token_csv(value)


def setup_run(tmp_path):
    run = tmp_path / "run with spaces"
    (run / "artifacts").mkdir(parents=True)
    (run / "reports").mkdir()
    (run / "artifacts/deployment-manifest.json").write_text("{}")
    (run / "reports/cpp-paired.json").write_text(json.dumps({
        "prompt_token_ids": [10, 20, 30], "eos_token_ids": [100, 101],
    }))
    (run / "prompt-ids.csv").write_text("40,50\n")
    runner = run / "test runner"
    runner.write_text("#!/bin/sh\nexit 0\n")
    runner.chmod(0o755)
    return run, profile.parser().parse_args(["--run-dir", str(run), "--runner", str(runner)])


def test_prompt_report_preferred_and_overrides_explicit(tmp_path):
    run, args = setup_run(tmp_path)
    assert profile.prompt_and_eos(args, run)[:2] == ("10,20,30", "100,101")
    args.prompt_token_ids, args.eos_token_ids = "60, 70\n", "248044"
    assert profile.prompt_and_eos(args, run)[:2] == ("60,70", "248044")
    args.prompt_token_ids = args.eos_token_ids = None
    (run / "reports/cpp-paired.json").unlink()
    assert profile.prompt_and_eos(args, run)[:2] == ("40,50", "248044")
    args.prompt_report = run / "missing.json"
    with pytest.raises(FileNotFoundError):
        profile.prompt_and_eos(args, run)


@pytest.mark.parametrize("failure", [None, "prepare", "capture"])
def test_one_verify_capture_uses_fresh_plan_and_propagates_failure(tmp_path, monkeypatch, failure):
    run, args = setup_run(tmp_path)
    calls = []
    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path / "different inherited run"))
    monkeypatch.setattr(profile.shutil, "which", lambda _: "/test/msprof")

    def invoke(command, *, check, env, cwd):
        assert check and cwd == SCRIPT.parents[1]
        assert env["AI_RUN_DIR"] == str(run)
        assert env["PYTHONPATH"].split(":")[:2] == [str(cwd / "framework/python"), str(cwd)]
        calls.append(command)
        if "prepare-chunk-plan" in command:
            if failure == "prepare":
                raise subprocess.CalledProcessError(2, command)
            Path(command[command.index("--output") + 1]).write_text("TEST_ONLY_PLAN\n")
        else:
            before, after = command[:command.index("--")], command[command.index("--") + 1:]
            for flag, value in (("--profile-backend", "cpp"), ("--profile-mode", "dflash"),
                                ("--profile-stage", "verify"), ("--profile-warmup", "1"),
                                ("--python", sys.executable)):
                assert before[before.index(flag) + 1] == value
            assert after[after.index("--prompt-token-ids") + 1] == "10,20,30"
            assert after[after.index("--eos-token-ids") + 1] == "100,101"
            plan = Path(after[after.index("--model") + 1])
            assert after[after.index("--model-sha256") + 1] == hashlib.sha256(plan.read_bytes()).hexdigest()
            if failure == "capture":
                raise subprocess.CalledProcessError(7, command)
            output = Path(before[before.index("--output-dir") + 1])
            operator_csv(output / "raw/op_summary.csv", [["TEST_ONLY", 1000]],
                         ["Op Name", "Task Duration(us)"])
            # run_msprof.sh owns the shared post-capture summary as well.
            summarize_windows([{"stage": "verify", "profile_mode": "dflash",
                                "profile_backend": "cpp", "profile_output": str(output / "raw")}],
                              output, prefix="verify")

    monkeypatch.setattr(profile.subprocess, "run", invoke)
    if failure:
        with pytest.raises(subprocess.CalledProcessError):
            profile.run_profile(args)
        assert not list(run.rglob("*hotspots.txt"))
        assert len(calls) == (1 if failure == "prepare" else 2)
    else:
        first = profile.run_profile(args)
        second = profile.run_profile(args)
        assert first != second
        assert first.is_relative_to(run) and second.is_relative_to(run)
        assert (first / "capture/verify-hotspots.txt").is_file()
        assert (second / "capture/verify-hotspots.txt").is_file()
        assert len(calls) == 4


def test_stages_and_precision_groups_never_merge(tmp_path):
    windows = []
    for stage in ("prefill", "verify"):
        capture = tmp_path / stage
        operator_csv(capture / "one/op_summary.csv", [
            ["M16", "BatchMatMul", "AI_CORE", "static", 1000, "1,16,16", "FLOAT16;FLOAT16"],
            ["M32", "BatchMatMul", "AI_CORE", "static", 9000, "1,16,16", "FLOAT;FLOAT"],
            ["KV", "CacheUpdate", "AI_CORE", "static", 10, "3,2,64,16", "FLOAT16;FLOAT16;INT32;INT32"],
        ])
        operator_csv(capture / "two/op_summary.csv", [
            ["M16", "BatchMatMul", "AI_CORE", "static", 2000, "1,16,16", "FLOAT16;FLOAT16"],
        ])
        windows.append({"stage": stage, "profile_mode": "dflash",
                        "profile_backend": "cpp", "profile_output": str(capture)})
    summarize_windows(windows, tmp_path, prefix="all")
    with (tmp_path / "all-operator-types.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 8
    for stage in ("prefill", "verify"):
        current = [r for r in rows if r["stage"] == stage]
        assert [float(r["total_ms"]) for r in current] == [9, 1, 0.01, 2]
        assert current[0]["input_data_types"] == "FLOAT;FLOAT"
        assert current[1]["input_data_types"] == "FLOAT16;FLOAT16"
        assert current[2]["op_type"] == "CacheUpdate" and current[2]["count"] == "1"
    with pytest.raises(ValueError, match="already exists"):
        summarize_windows(windows, tmp_path, prefix="all")


@pytest.mark.parametrize("mode,stage", [
    ("ordinary", "draft"), ("ordinary", "verify"), ("dflash", "decode"),
])
def test_invalid_mode_stage_fails_before_creating_output(tmp_path, mode, stage):
    run, args = setup_run(tmp_path)
    args.profile_mode, args.profile_stage = mode, stage
    with pytest.raises(ValueError, match="stages:"):
        unified.run_profile(args)
    assert not (run / "msprof").exists()


def test_shared_entry_defaults_and_compatibility():
    assert unified.parser().parse_args([]).profile_stage == "all"
    assert profile.parser().parse_args([]).profile_stage == "verify"
    assert profile.run_profile is unified.run_profile
