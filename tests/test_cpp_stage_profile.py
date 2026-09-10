"""Real C++ socket barriers + real controller, fake ACL/msprof. No device evidence."""

import csv
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_incremental_air_om import chunk_bundle  # noqa: F401
from test_msprof_stage_script import sandbox  # noqa: F401
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file

SOURCE = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "mode,stage,failure,draft_limit,new_tokens",
    [
        ("ordinary", "prefill", "", 15, 32),
        ("ordinary", "decode", "", 15, 32),
        ("ordinary", "all", "", 15, 32),
        ("dflash", "draft", "", 15, 32),
        ("dflash", "verify", "", 15, 32),
        ("dflash", "all", "", 15, 32),
        ("ordinary", "decode", "stop", 15, 32),
        ("dflash", "all", "execute", 15, 32),
        ("ordinary", "decode", "empty", 15, 32),
        ("ordinary", "decode", "eos", 15, 32),
        ("dflash", "draft", "", 2, 32),
        ("dflash", "verify", "", 2, 32),
        ("dflash", "all", "", 15, 4),
    ],
)
def test_cpp_stage_windows(chunk_bundle, sandbox, mode, stage, failure, draft_limit, new_tokens):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL runner")
    root = sandbox["tmp"]
    plan, _, _ = write_incremental_plan(chunk_bundle, root / "plan.txt", mode=mode)
    # The C++ entry must work with standard-library Python and no torch/pyACL.
    (root / "stubs/torch.py").write_text(
        'raise AssertionError("C++ profiling must not import torch")\n'
    )
    (root / "stubs/torch_npu.py").write_text(
        'raise AssertionError("C++ profiling must not import torch_npu")\n'
    )
    events = root / "acl-events.jsonl"
    sandbox["env"]["QWEN35_FAKE_EVENT_LOG"] = str(events)
    if failure == "execute":
        sandbox["env"]["QWEN35_FAKE_FAIL_GRAPH"] = "target_verify"
    elif failure != "eos":
        sandbox["env"]["TEST_FAILURE"] = failure
    output = root / "profile-run"
    result = subprocess.run(
        [
            "bash",
            str(SOURCE / "tools/run_msprof.sh"),
            "--label",
            "cpp",
            "--output-dir",
            str(output),
            "--python",
            sys.executable,
            "--msprof-bin",
            sandbox["msprof"],
            "--profile-backend",
            "cpp",
            "--profile-mode",
            mode,
            "--profile-stage",
            stage,
            "--profile-warmup",
            "0" if failure else "1",
            "--profile-timeout",
            "5",
            "--",
            runner,
            "--model-kind",
            "chunk",
            "--model",
            str(plan),
            "--model-sha256",
            sha256_file(plan),
            "--prompt-token-ids",
            ",".join(["4"] * 65),
            "--eos-token-ids",
            "5" if failure == "eos" else "",
            "--max-draft-tokens",
            str(draft_limit),
            "--max-new-tokens",
            str(new_tokens),
        ],
        env=sandbox["env"],
        capture_output=True,
        text=True,
        timeout=40,
    )
    manifest = json.loads((output / "manifest/cpp.json").read_text())
    if failure:
        assert result.returncode != 0, result.stdout + result.stderr
        assert manifest["status"] == "FAIL"
        if failure != "empty":
            assert not (output / "cpp-stage-report.json").exists()
        return
    assert result.returncode == 0, result.stdout + result.stderr
    assert manifest["status"] == "PASS"
    report = json.loads((output / "cpp-stage-report.json").read_text())
    assert "fake-acl" in report["runner_version"]
    stages = (
        (
            ["prefill", "decode"]
            if mode == "ordinary"
            else ["prefill", "draft", "verify"]
        )
        if stage == "all"
        else [stage]
    )
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    captured = [row[0] for row in rows if row[1]]
    expected = []
    for selected in stages:
        expected += ["draft" if selected == "draft" else "target_" + selected] * (
            2 if selected == "prefill" else 1
        )
    assert captured == expected
    assert report["capture_windows"] == len(stages)
    control = json.loads((output / "manifest/cpp-control.json").read_text())
    assert control["status"] == "PASS_CONTROL"
    assert control["profile_backend"] == "cpp" and control["profile_mode"] == mode
    reports = report["captures"] if stage == "all" else [report]
    count = min(15, draft_limit, new_tokens - 1)
    for item in reports:
        assert item["proposal_count"] == (count if item["profile_stage"] in {"draft", "verify"} else 0)
    for role, measured, start, valid in rows:
        if role == "target_verify":
            assert valid == count + 1
    assert all(r["warmup_output_match"] for r in reports)
    with Path(manifest["artifacts"]["operator_types"]).open(newline="") as stream:
        operators = list(csv.DictReader(stream))
    assert [row["stage"] for row in operators] == stages
    assert all(row["profile_mode"] == mode and row["profile_backend"] == "cpp" for row in operators)
    if stage == "decode":
        assert all(not r[1] for r in rows if r[0] == "target_prefill")
    if mode == "ordinary":
        assert not any(r[0] in {"draft", "target_verify"} for r in rows)


@pytest.mark.parametrize("mode", ["ordinary", "dflash"])
def test_shared_om_entry_builds_matching_plan_and_profiles_all(chunk_bundle, sandbox, mode):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL runner")
    root = sandbox["tmp"]
    # Preparing a plan uses the model-side CLI (and its Torch imports).
    # The C++ collector itself remains standard-library-only, tested above.
    (root / "stubs/torch.py").unlink()
    for module in ("torch_npu", "acl"):
        (root / ("stubs/" + module + ".py")).write_text(
            'raise AssertionError("OM plan/capture must not initialize NPU Python bindings")\n')
    events = root / "all-events.jsonl"
    sandbox["env"]["QWEN35_FAKE_EVENT_LOG"] = str(events)
    result = subprocess.run([
        sys.executable, "-B", str(SOURCE / "tools/profile_om.py"),
        "--run-dir", str(root), "--runner", runner,
        "--deployment-manifest", str(chunk_bundle),
        "--prompt-token-ids", ",".join(["4"] * 65), "--eos-token-ids", "248044",
        "--profile-mode", mode, "--profile-stage", "all", "--profile-timeout", "5",
        "--msprof-bin", sandbox["msprof"],
    ], env=sandbox["env"], text=True, capture_output=True, timeout=40)
    assert result.returncode == 0, result.stdout + result.stderr
    outputs = list((root / "msprof").glob(mode + "-all-*"))
    assert len(outputs) == 1
    output = outputs[0]
    request = json.loads((output / "profile-request.json").read_text())
    assert request["deployment_manifest"]["sha256"] == sha256_file(chunk_bundle)
    assert request["runner"]["sha256"] == sha256_file(runner)
    control = json.loads((output / "capture/manifest/all-control.json").read_text())
    assert control["status"] == "PASS_CONTROL"
    stages = ["prefill", "decode"] if mode == "ordinary" else ["prefill", "draft", "verify"]
    assert control["stages"] == request["stages"] == stages
    assert len(control["captures"]) == len(stages)
    assert all(f'--pid={control["application_pid"]}' in item["msprof_arguments"]
               for item in control["captures"])
    with (output / "capture/all-operator-types.csv").open(newline="") as stream:
        assert [row["stage"] for row in csv.DictReader(stream)] == stages
