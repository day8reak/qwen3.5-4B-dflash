"""Real C++ socket barriers + real controller, fake ACL/msprof. No device evidence."""

import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import struct
import sys

import pytest

from test_incremental_air_om import chunk_bundle  # noqa: F401
from test_msprof_stage_script import sandbox  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file

SOURCE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


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


@pytest.mark.parametrize("stage,variation,warmup,draft_limit,new_tokens,eos,field", [
    ("verify", "padding", 1, 2, 32, "", None),
    ("verify", "padding", 1, 15, 3, "", None),
    ("verify", "padding", 1, 15, 32, "7", None),
    ("all", "padding", 1, 2, 32, "", None),
    ("verify", "verify_output", 1, 15, 32, "", "output_token_ids"),
    ("verify", "draft_input", 1, 15, 32, "", "input_token_ids"),
    ("verify", "warmup_middle", 3, 15, 32, "", "output_token_ids"),
    ("decode", "target_decode_output", 1, 15, 32, "", "output_token_ids"),
    ("prefill", "target_prefill_output", 1, 15, 32, "", "output_token_ids"),
    ("draft", "draft_output", 1, 15, 32, "", "output_token_ids"),
    ("all", "draft_output", 1, 15, 32, "", "output_token_ids"),
    ("verify", "padding", 0, 2, 32, "", None),
])
def test_cpp_profile_checks_logical_outputs_and_records_differences(
    chunk_bundle, sandbox, stage, variation, warmup, draft_limit, new_tokens, eos, field,
):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL runner")
    root = sandbox["tmp"]
    mode = "ordinary" if stage in {"prefill", "decode"} else "dflash"
    plan, _, _ = write_incremental_plan(chunk_bundle, root / "plan.txt", mode=mode)
    sandbox["env"]["QWEN35_FAKE_PROFILE_VARIATION"] = variation
    output = root / "profile-run"
    result = subprocess.run([
        "bash", str(SOURCE / "tools/run_msprof.sh"), "--label", "check",
        "--output-dir", str(output), "--python", sys.executable,
        "--msprof-bin", sandbox["msprof"], "--profile-backend", "cpp",
        "--profile-mode", mode, "--profile-stage", stage,
        "--profile-warmup", str(warmup), "--profile-timeout", "5", "--",
        runner, "--model-kind", "chunk", "--model", str(plan),
        "--model-sha256", sha256_file(plan), "--prompt-token-ids", ",".join(["4"] * 17),
        "--eos-token-ids", eos, "--max-draft-tokens", str(draft_limit),
        "--max-new-tokens", str(new_tokens),
    ], env=sandbox["env"], capture_output=True, text=True, timeout=40)
    log = result.stdout + result.stderr
    manifest = json.loads((output / "manifest/check.json").read_text())
    if field:
        assert result.returncode != 0 and manifest["status"] == "FAIL", log
        assert f"field={field}" in log
        assert "index=" in log and "reference=" in log and "actual=" in log
        assert not (output / "check-stage-report.json").exists()
    else:
        assert result.returncode == 0 and manifest["status"] == "PASS", log
    trace = output / "profile/msprof/check.iterations.jsonl"
    assert manifest["artifacts"]["iteration_trace"] == str(trace)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    completed = [r for r in rows if r["event"] == "completed"]
    assert completed and all(r["input_state_comparison"] == "NOT_RUN" for r in completed)
    assert all(r["output_token_ids"] == r["raw_output_token_ids"][:len(r["output_token_ids"])]
               for r in completed)
    if field:
        failure = completed[-1]
        assert failure["check"]["status"] == "FAIL"
        assert failure["check"]["first_difference"]["field"] == field
        assert failure["measured"] is (variation != "warmup_middle")
        if variation == "warmup_middle":
            assert failure["iteration"] == 1
            assert "msprof_start_sent" not in log
        if variation == "draft_input":
            assert failure["check"]["input_token_ids_match"] is False
        if stage == "all":
            assert failure["profile_stage"] == "draft"
            assert log.index("application_error stage=draft") < log.index("cleanup_begin")
            assert "cleanup_end" in log
            assert "preparing dflash stage=verify" not in log
    else:
        measured = [r for r in completed if r["measured"]]
        assert len(measured) == (3 if stage == "all" else 1)
        assert all(r["check"]["status"] == ("PASS" if warmup else "NO_WARMUP") for r in measured)
        verify = next(r for r in measured if r["profile_stage"] == "verify")
        assert verify["verify_valid_rows"] == 3
        assert verify["padding_output_rows"] == 13
        assert verify["raw_output_token_ids"][3:] == [43] * 13
        assert verify["check"]["padding_token_ids_match"] is (False if warmup else None)


@pytest.mark.parametrize("variation", ["", "prefill_features", "draft_output"])
def test_draft_input_audit_distinguishes_features_from_output_variation(chunk_bundle, sandbox, variation):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL runner")
    root = sandbox["tmp"]
    plan, _, _ = write_incremental_plan(chunk_bundle, root / "plan.txt", mode="dflash")
    copies = root / "copies.jsonl"
    sandbox["env"].update(QWEN35_FAKE_PROFILE_VARIATION=variation, QWEN35_FAKE_COPY_LOG=str(copies))
    output = root / "profile-run"
    result = subprocess.run([
        "bash", str(SOURCE / "tools/run_msprof.sh"), "--label", "audit",
        "--output-dir", str(output), "--python", sys.executable,
        "--msprof-bin", sandbox["msprof"], "--profile-backend", "cpp",
        "--profile-mode", "dflash", "--profile-stage", "draft", "--profile-warmup", "1",
        "--profile-timeout", "5", "--", runner, "--model-kind", "chunk", "--model", str(plan),
        "--model-sha256", sha256_file(plan), "--prompt-token-ids", ",".join(["4"] * 17),
        "--eos-token-ids", "", "--max-draft-tokens", "15", "--max-new-tokens", "32",
        "--profile-audit-draft-inputs", "true",
    ], env=sandbox["env"], capture_output=True, text=True, timeout=40)
    log = result.stdout + result.stderr
    assert (result.returncode != 0) == (variation == "draft_output"), log
    assert "cleanup_end" in log and "errors=0" in log
    trace = output / "profile/msprof/audit.iterations.jsonl"
    samples = [json.loads(line) for line in trace.read_text().splitlines()]
    reference, measured = [s for s in samples if s["event"] == "completed"]
    a, b = reference["draft_input_sha256"], measured["draft_input_sha256"]
    assert a.keys() == b.keys() and "features" in a and any(k.startswith("d0_") for k in a)
    for name, fmt, value in (("anchor", "q", 5), ("valid_rows", "h", 17),
                             ("start_position", "q", 0), ("proposal_count", "h", 15)):
        assert a[name] == b[name] == hashlib.sha256(struct.pack(fmt, value)).hexdigest()
    assert reference["input_state_comparison"] == "REFERENCE_SHA256"
    differences = [key for key in a if a[key] != b[key]]
    assert differences == (["features"] if variation == "prefill_features" else [])
    assert measured["input_state_comparison"] == (
        "DIFFERENT_SHA256" if differences else "MATCH_SHA256")
    assert measured["check"]["status"] == ("FAIL" if variation == "draft_output" else "PASS")
    assert measured["check"]["input_token_ids_match"] is True
    transfers = [json.loads(line) for line in copies.read_text().splitlines()]
    # Normal stage I/O contains only small scalars/top1. The audit's tensor
    # readbacks must all precede msprof start, even on the measured iteration.
    tensor_readbacks = [t for t in transfers if t[0] > 120 and t[1] == 2]
    assert tensor_readbacks and all(not t[2] for t in tensor_readbacks)
