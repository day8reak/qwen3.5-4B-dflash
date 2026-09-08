"""Real C++ socket barriers + real controller, fake ACL/msprof. No device evidence."""

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
    "mode,stage,failure",
    [
        ("ordinary", "prefill", ""),
        ("ordinary", "decode", ""),
        ("ordinary", "all", ""),
        ("dflash", "draft", ""),
        ("dflash", "verify", ""),
        ("dflash", "all", ""),
        ("ordinary", "decode", "stop"),
        ("dflash", "all", "execute"),
        ("ordinary", "decode", "empty"),
        ("ordinary", "decode", "eos"),
    ],
)
def test_cpp_stage_windows(chunk_bundle, sandbox, mode, stage, failure):
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
    assert all(r["warmup_output_match"] for r in reports)
    if stage == "decode":
        assert all(not r[1] for r in rows if r[0] == "target_prefill")
    if mode == "ordinary":
        assert not any(r[0] in {"draft", "target_verify"} for r in rows)
