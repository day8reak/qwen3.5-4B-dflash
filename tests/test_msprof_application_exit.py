"""Keep the application's real failure when a stage socket closes before cleanup."""

import json
import subprocess
import sys

import pytest

from test_msprof_stage_script import (  # noqa: F401
    CONTROLLER, assert_processes_reaped, executable, sandbox,
)


@pytest.mark.parametrize("exit_code,cleanup_delay", [(7, 0.3), (0, 0.3), (7, 30)])
def test_stage_eof_drains_cleanup_and_never_passes_incomplete_capture(
    sandbox, exit_code, cleanup_delay,
):
    root = sandbox["tmp"]
    app = executable(root / "cleanup_application", r'''
import json, os, socket, sys, time
sock = socket.socket(fileno=int(os.environ["DFLASH_MSPROF_CONTROL_FD"]))
reader = sock.makefile("rb")
for stage in ("prefill", "draft"):
    sock.sendall((json.dumps({"event": "ready", "pid": os.getpid(), "stage": stage,
        "output": sys.argv[1] + "/" + stage, "device_id": 0, "metrics": "Memory"}) + "\n").encode())
    assert json.loads(reader.readline()) == {"event": "started"}
    sock.sendall(b'{"event":"done","success":true}\n')
    assert json.loads(reader.readline()) == {"event": "stopped"}
if sys.argv[2] != "0":
    # Print in fragments, as real process pipes can split an exception line.
    print("[stage-profile] application_error stage=draft ", end="", flush=True)
    print("message=profile output differs field=output_token_ids index=7 reference=2972 actual=13", flush=True)
reader.close()
sock.close()
print("cleanup_begin", flush=True)
time.sleep(float(sys.argv[3]))
# The useful error must survive a long driver cleanup log as well as EOF.
print("cleanup detail\n" * 6000, end="", flush=True)
print("cleanup_end", flush=True)
sys.exit(int(sys.argv[2]))
''')
    report = root / "control.json"
    result = subprocess.run([
        sys.executable, "-B", str(CONTROLLER), "--msprof-bin", sandbox["msprof"],
        "--output", str(root / "raw"), "--stage", "all", "--backend", "cpp",
        "--metrics", "Memory", "--timeout", "1", "--control-report", str(report),
        "--", app, str(root / "raw"), str(exit_code), str(cleanup_delay),
    ], env=sandbox["env"], capture_output=True, text=True, timeout=15)
    control = json.loads(report.read_text())
    log = result.stdout + result.stderr
    assert result.returncode != 0 and control["status"] == "FAIL", log
    assert not control["capture_completed"]
    failure = control["application_failure"]
    assert failure["waiting_stage"] == "verify"
    assert failure["waiting_event"] == "ready"
    assert failure["last_stopped_stage"] == "draft"
    assert len(control["application_output_tail"].encode()) <= 16384
    if cleanup_delay < 1:
        assert "cleanup_end" in log
        assert control["application_exit_code"] == exit_code
        assert failure["exit_wait_timed_out"] is False
        assert f"exit={exit_code}" in control["error"]
    else:
        assert "cleanup_end" not in log
        assert control["application_exit_code"] < 0
        assert failure["exit_wait_timed_out"] is True
    if exit_code:
        assert "index=7 reference=2972 actual=13" in control["application_error"]
        assert control["application_error"] in control["error"]
    assert [c["capture_completed"] for c in control["captures"]] == [True, True, False]
    assert_processes_reaped(control)
    for capture in control["captures"]:
        assert_processes_reaped(capture)
    assert not sandbox["active"].exists()
