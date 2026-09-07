"""Exercise wrapper dispatch/export without hardware or a real collector."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/run_msprof.sh"


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


@pytest.mark.parametrize("stage,failure", [
    (stage, failure)
    for stage in ("prefill", "draft-verify")
    for failure in (None, "application", "export", "empty", "invalid-report")
] + [(None, None), (None, "application")])
def test_collector_dispatch_and_export(tmp_path, stage, failure):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text('''
__version__ = "HOST_TEST_STUB"
class Npu:
    def is_available(self): return True
    def set_device(self, value): pass
    def current_device(self): return 0
    def get_device_name(self, value): return "HOST_TEST_NO_DEVICE"
npu = Npu()
''')
    (stubs / "torch_npu.py").write_text('__version__ = "HOST_TEST_STUB"\n')
    executable(stubs / "npu-smi", 'print("HOST TEST ONLY")\n')
    event_path = tmp_path / "events.jsonl"
    environment = dict(os.environ)
    environment.pop("ASCEND310P_SIMULATION_ONLY", None)
    environment.update({
        "PATH": str(stubs) + os.pathsep + environment["PATH"],
        "PYTHONPATH": str(stubs), "TEST_EVENTS": str(event_path),
        "TEST_FAILURE": failure or "", "PYTHONDONTWRITEBYTECODE": "1",
    })
    app = executable(stubs / "fake_application", '''
import json, os, pathlib, sys
args = sys.argv[1:]
with open(os.environ["TEST_EVENTS"], "a") as stream:
    stream.write(json.dumps(["application", args]) + "\\n")
if "--profile-stage" not in args:
    assert os.environ["DFLASH_MSPROF_PROCESS_CAPTURE"] == "1"
    sys.exit(7 if os.environ["TEST_FAILURE"] == "application" else 0)
assert "DFLASH_MSPROF_PROCESS_CAPTURE" not in os.environ
if os.environ["TEST_FAILURE"] == "application": sys.exit(7)
def value(flag): return args[args.index(flag) + 1]
stage = value("--profile-stage")
root = pathlib.Path(value("--profile-output"))
root.mkdir(parents=True)
report = {
    "status": "PASS_CAPTURE", "profile_stage": stage, "capture_windows": 1,
    "profile_output": str(root), "operator_fallback_enabled": False,
    "captured_calls": {"prefill": int(stage == "prefill"),
        "draft": int(stage == "draft-verify"), "target_verify": int(stage == "draft-verify")},
}
if os.environ["TEST_FAILURE"] == "invalid-report": report["capture_windows"] = 2
pathlib.Path(value("--report")).write_text(json.dumps(report))
''')
    msprof = executable(stubs / "msprof", '''
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
if args == ["--version"]:
    print("HOST_TEST_STUB")
    sys.exit(0)
with open(os.environ["TEST_EVENTS"], "a") as stream:
    stream.write(json.dumps(["msprof", args]) + "\\n")
if "--export=on" not in args:
    index = next(i for i, arg in enumerate(args) if arg.endswith("fake_application"))
    assert "--task-time=on" in args[:index] and "--ascendcl=on" in args[:index]
    sys.exit(subprocess.run(args[index:]).returncode)
assert "--export=on" in args and len(args) == 3, args
if os.environ["TEST_FAILURE"] == "export": sys.exit(8)
if os.environ["TEST_FAILURE"] != "empty":
    root = pathlib.Path(next(a.split("=", 1)[1] for a in args if a.startswith("--output=")))
    (root / "op_summary_0.csv").write_text("Op Name,Task Duration(us)\\nTEST_ONLY,1\\n")
''')
    output = tmp_path / "result"
    stage_args = ["--profile-stage", stage, "--profile-warmup", "0"] if stage else []
    completed = subprocess.run([
        "bash", str(SCRIPT), "--label", "stage", "--output-dir", str(output),
        "--python", sys.executable, "--msprof-bin", msprof,
        *stage_args, "--",
        app, "-m", "models.dflash_v1.run_npu", "--device", "npu:0",
    ], cwd=tmp_path, env=environment, capture_output=True, text=True)
    events = [json.loads(line) for line in event_path.read_text().splitlines()]
    if stage is None:
        assert [e[0] for e in events] == ["msprof", "application"]
        assert "--profile-stage" not in events[1][1]
    else:
        assert events[0][0] == "application"
        application_args = events[0][1]
        assert application_args[application_args.index("--profile-stage") + 1] == stage
        assert application_args[application_args.index("--profile-warmup") + 1] == "0"
        if failure == "application":
            assert len(events) == 1
        else:
            assert [e[0] for e in events] == ["application", "msprof"]
    manifest = json.loads((output / "manifest/stage.json").read_text())
    assert manifest["msprof"]["collector"] == ("pyACL stage API" if stage else "msprof process")
    assert manifest["msprof"]["profile_stage"] == stage
    if failure is None:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert manifest["status"] == "PASS"
    else:
        assert completed.returncode != 0
        assert manifest["status"] == "FAIL"


def test_stage_cannot_silently_wrap_an_unsupported_application(tmp_path):
    environment = dict(os.environ)
    environment.pop("ASCEND310P_SIMULATION_ONLY", None)
    output = tmp_path / "result"
    completed = subprocess.run([
        "bash", str(SCRIPT), "--label", "bad", "--output-dir", str(output),
        "--profile-stage", "prefill", "--", "/bin/true",
    ], env=environment, capture_output=True, text=True)
    assert completed.returncode == 2
    assert "requires python -m" in completed.stderr
    assert not output.exists()
