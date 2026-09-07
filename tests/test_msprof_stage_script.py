"""Host-only tests of the real controller with a simulated interactive msprof.

The stub never produces device measurements; its CSV is labelled TEST_ONLY.
"""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import pytest

SOURCE = Path(__file__).resolve().parents[1]
SCRIPT = SOURCE / "tools/run_msprof.sh"
CONTROLLER = SOURCE / "models/dflash_v1/msprof_cli.py"


def executable(path, body):
    path.write_text(f"#!{sys.executable}\n" + body, encoding="utf-8")
    path.chmod(0o755)
    return str(path)


@pytest.fixture(params=("legacy", "pid"))
def sandbox(tmp_path, request):
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    (stubs / "torch.py").write_text("""
__version__ = "HOST_TEST_STUB"
class Npu:
    def is_available(self): return True
    def set_device(self, value): pass
    def current_device(self): return 0
    def get_device_name(self, value): return "HOST_TEST_NO_DEVICE"
npu = Npu()
""")
    (stubs / "torch_npu.py").write_text('__version__ = "HOST_TEST_STUB"\n')
    executable(stubs / "npu-smi", 'print("HOST TEST ONLY")\n')
    (stubs / "acl.py").write_text('raise AssertionError("pyACL must never be imported")\n')
    events, active = tmp_path / "events.jsonl", tmp_path / "active"
    environment = dict(os.environ)
    environment.pop("ASCEND310P_SIMULATION_ONLY", None)
    environment.update({
        "PATH": str(stubs) + os.pathsep + environment["PATH"],
        "PYTHONPATH": str(stubs), "TEST_EVENTS": str(events),
        "TEST_ACTIVE": str(active), "TEST_CONTROLLER": str(CONTROLLER),
        "TEST_ACK_FORMAT": request.param,
        "TEST_FAILURE": "", "PYTHONDONTWRITEBYTECODE": "1",
    })
    common = r'''
import json, os, pathlib, sys, time
failure = os.environ["TEST_FAILURE"]
active = pathlib.Path(os.environ["TEST_ACTIVE"])
def event(name, value=None):
    fd = os.open(os.environ["TEST_EVENTS"], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try: os.write(fd, (json.dumps([name, value]) + "\n").encode())
    finally: os.close(fd)
'''
    app = executable(stubs / "fake_application", common + r'''
import importlib.util
args = sys.argv[1:]
event("application", args)
event("app-pid", os.getpid())
if "--profile-stage" not in args:
    assert os.environ["DFLASH_MSPROF_PROCESS_CAPTURE"] == "1"
    sys.exit(7 if failure == "application" else 0)
assert os.environ["PROFILING_MODE"] == "dynamic"
assert "PROFILING_OPTIONS" not in os.environ
assert "DFLASH_MSPROF_PROCESS_CAPTURE" not in os.environ
if failure == "application": sys.exit(7)
def value(flag): return args[args.index(flag) + 1]
stage = value("--profile-stage")
root = pathlib.Path(value("--profile-output"))
spec = importlib.util.spec_from_file_location("real_msprof_cli", os.environ["TEST_CONTROLLER"])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
work_done = False
def synchronize():
    event("sync", active.exists())
    if work_done and failure == "sync": raise RuntimeError("injected device sync failure")
with module.MsprofStageProfiler(
    str(root), 0, value("--profile-aic-metrics"), synchronize,
    stage="incorrect" if failure == "mismatched-ready" else stage,
) as profiler:
    assert not active.exists()
    if failure == "control-eof":
        profiler.close()
        time.sleep(30)
    event("warmup-outside")
    with profiler.capture():
        assert active.exists(), "stage ran before collector start acknowledgement"
        event("captured-work")
        work_done = True
        if failure == "application-capture": raise RuntimeError("injected application failure")
        if failure == "application-hang": time.sleep(30)
    assert not active.exists(), "postprocessing ran before collector stop"
    event("postprocess-outside")
    if failure == "second-window":
        with profiler.capture(): event("unexpected-second-window")
report = {
    "status": "PASS_CAPTURE", "collector": module.COLLECTOR,
    "profile_stage": stage, "capture_windows": 1,
    "profile_output": str(root), "operator_fallback_enabled": False,
    "captured_calls": {"prefill": int(stage == "prefill"),
        "draft": int(stage == "draft-verify"), "target_verify": int(stage == "draft-verify")},
}
if failure == "invalid-report": report["capture_windows"] = 2
pathlib.Path(value("--report")).write_text(json.dumps(report))
''')
    msprof = executable(stubs / "msprof", common + r'''
import subprocess
args = sys.argv[1:]
if args == ["--version"]:
    print("HOST_TEST_STUB")
    sys.exit(0)
event("msprof", args)
if "--export=on" in args:
    assert len(args) == 3, args
    if failure == "export": sys.exit(8)
    if failure != "empty":
        root = pathlib.Path(next(a.split("=", 1)[1] for a in args if a.startswith("--output=")))
        (root / "op_summary_0.csv").write_text("Op Name,Task Duration(us)\nTEST_ONLY,1\n")
    sys.exit(0)
if "--dynamic=on" not in args:
    index = next(i for i, arg in enumerate(args) if arg.endswith("fake_application"))
    assert "--task-time=on" in args[:index] and "--ascendcl=on" in args[:index]
    sys.exit(subprocess.run(args[index:]).returncode)
assert sys.stdin.isatty()
assert "PROFILING_MODE" not in os.environ
assert "--task-time=on" in args and "--runtime-api=on" in args
pid = int(next(a.split("=")[1] for a in args if a.startswith("--pid=")))
os.kill(pid, 0)
root = pathlib.Path(next(a.split("=", 1)[1] for a in args if a.startswith("--output=")))
pid_format = os.environ["TEST_ACK_FORMAT"] == "pid"
prefix = "dynamic profiling" + (f" for pid {pid}" if pid_format else "")
prompt = "> " if pid_format else "(msprof) "
print(prompt, end="", flush=True)
for line in sys.stdin:
    command = line.strip()
    event("command", command)
    if failure == "prof-exit": sys.exit(9)
    if failure == command:
        print(prefix + " " + command + " failed", flush=True)
        continue
    if failure == command + "-wrong-pid":
        print(f"dynamic profiling for pid {pid + 1} {command} success", flush=True)
        continue
    if failure == command + "-no-ack":
        # Prompts and startup logs are not server acknowledgements.
        print(command + "\nStart profiling...\ndynamic profiling " + command + " success\n(msprof) ", flush=True)
        continue
    if command == "start":
        root.mkdir(parents=True)
        active.touch()
    elif command in ("stop", "quit"):
        active.unlink(missing_ok=True)
    else:
        raise AssertionError(command)
    # Delayed split writes exercise buffering. Sleeps are confined to the stub.
    print(prefix + " " + command[:2], end="", flush=True)
    time.sleep(0.01)
    print(command[2:] + (" success" if pid_format else " success......"), flush=True)
    if command == "quit": sys.exit(10 if failure == "quit-exit" else 0)
    print(prompt, end="", flush=True)
''')
    return dict(tmp=tmp_path, env=environment, events=events, active=active, app=app, msprof=msprof)


def read_events(sandbox):
    return [json.loads(line) for line in sandbox["events"].read_text().splitlines()]


def assert_processes_reaped(control):
    for key in ("application_pid", "msprof_pid"):
        if control.get(key) is not None:
            with pytest.raises(ProcessLookupError):
                os.kill(control[key], 0)


@pytest.mark.parametrize("stage,failure", [
    (stage, failure)
    for stage in ("prefill", "draft-verify")
    for failure in (None, "application", "export", "empty", "invalid-report")
] + [
    ("draft-verify", failure) for failure in (
        "start", "start-no-ack", "stop", "stop-no-ack", "quit", "quit-exit",
        "prof-exit", "application-capture", "application-hang", "control-eof",
        "mismatched-ready", "second-window", "sync",
        "start-wrong-pid", "stop-wrong-pid", "quit-wrong-pid",
    )
] + [(None, None), (None, "application")])
def test_collector_dispatch_and_export(sandbox, stage, failure):
    sandbox["env"]["TEST_FAILURE"] = failure or ""
    sandbox["env"]["PROFILING_OPTIONS"] = '{"training_trace":"on"}'
    output = sandbox["tmp"] / "result"
    stage_args = [
        "--profile-stage", stage, "--profile-warmup", "0",
        "--profile-timeout", "0.5" if failure and ("no-ack" in failure or "hang" in failure) else "5",
    ] if stage else []
    completed = subprocess.run([
        "bash", str(SCRIPT), "--label", "stage", "--output-dir", str(output),
        "--python", sys.executable, "--msprof-bin", sandbox["msprof"],
        *stage_args, "--",
        sandbox["app"], "-m", "models.dflash_v1.run_npu", "--device", "npu:0",
    ], cwd=sandbox["tmp"], env=sandbox["env"], capture_output=True, text=True, timeout=25)
    events = read_events(sandbox)
    names = [e[0] for e in events]
    manifest = json.loads((output / "manifest/stage.json").read_text())
    assert manifest["msprof"]["collector"] == ("msprof dynamic CLI" if stage else "msprof process")
    assert manifest["msprof"]["profile_stage"] == stage
    if stage is None:
        assert names == ["msprof", "application", "app-pid"]
        assert "--profile-stage" not in events[1][1]
    else:
        assert names[0] == "application"
        application_args = events[0][1]
        assert application_args[application_args.index("--profile-stage") + 1] == stage
        assert application_args[application_args.index("--profile-warmup") + 1] == "0"
        control = json.loads((output / "manifest/stage-control.json").read_text())
        assert_processes_reaped(control)
        commands = [e[1] for e in events if e[0] == "command"]
        assert commands.count("start") <= 1
        exported = any("--export=on" in e[1] for e in events if e[0] == "msprof")
        if failure in (None, "export", "empty", "invalid-report"):
            assert commands == ["start", "stop", "quit"]
            assert names.count("captured-work") == 1 and exported
            assert control["status"] == "PASS_CONTROL"
            assert all(control[name + "_acknowledged"] for name in ("start", "stop", "quit"))
            assert set(control["acknowledgements"]) == {"start", "stop", "quit"}
            if sandbox["env"]["TEST_ACK_FORMAT"] == "pid":
                assert all(
                    f'for pid {control["application_pid"]}' in value
                    for value in control["acknowledgements"].values()
                )
            sequence = [e["event"] for e in control["events"]]
            for before, after in (
                ("application_ready", "msprof_start_sent"),
                ("msprof_start_acknowledged", "application_started"),
                ("application_done", "msprof_stop_sent"),
                ("msprof_stop_acknowledged", "msprof_quit_sent"),
                ("msprof_quit_acknowledged", "application_stopped"),
            ):
                assert sequence.index(before) < sequence.index(after)
            attach = next(e[1] for e in events if e[0] == "msprof")
            assert f'--pid={control["application_pid"]}' in attach
            assert control["msprof_arguments"][1:] == attach
        else:
            assert control["status"] == "FAIL" and not exported
            assert control["capture_completed"] is False
        if failure in ("application", "control-eof", "mismatched-ready", "start", "start-no-ack", "start-wrong-pid", "prof-exit"):
            assert "captured-work" not in names
        if failure in ("stop", "stop-no-ack", "stop-wrong-pid", "quit", "quit-exit", "quit-wrong-pid"):
            assert "postprocess-outside" not in names
        assert "unexpected-second-window" not in names
    if failure is None:
        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert manifest["status"] == "PASS"
    else:
        assert completed.returncode != 0, completed.stdout + completed.stderr
        assert manifest["status"] == "FAIL"


def test_interrupt_cleans_up_both_owned_processes(sandbox):
    sandbox["env"]["TEST_FAILURE"] = "application-hang"
    output = sandbox["tmp"]
    control_path = output / "control.json"
    with (output / "interrupt.log").open("w") as log:
        process = subprocess.Popen([
            sys.executable, "-B", str(CONTROLLER),
            "--msprof-bin", sandbox["msprof"], "--output", str(output / "raw"),
            "--stage", "prefill", "--metrics", "Memory", "--timeout", "5",
            "--control-report", str(control_path),
            "--", sandbox["app"], "--profile-stage", "prefill",
            "--profile-output", str(output / "raw"), "--profile-aic-metrics", "Memory",
            "--report", str(output / "stage.json"),
        ], env=sandbox["env"], stdout=log, stderr=subprocess.STDOUT)
        try:
            deadline = time.monotonic() + 10
            while not sandbox["active"].exists():
                assert process.poll() is None
                assert time.monotonic() < deadline
                time.sleep(0.01)
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=10) != 0
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()
    control = json.loads(control_path.read_text())
    assert control["status"] == "FAIL" and not control["capture_completed"]
    assert_processes_reaped(control)
    assert not sandbox["active"].exists()


@pytest.mark.parametrize("option", [
    ["--profile-stage", "prefill"], ["--profile-timeout", "0"], ["--profile-timeout", "1"],
])
def test_invalid_stage_wrapper_request_fails_before_output(tmp_path, option):
    environment = dict(os.environ)
    environment.pop("ASCEND310P_SIMULATION_ONLY", None)
    output = tmp_path / "result"
    completed = subprocess.run([
        "bash", str(SCRIPT), "--label", "bad", "--output-dir", str(output),
        *option, "--", "/bin/true",
    ], env=environment, capture_output=True, text=True)
    assert completed.returncode == 2
    assert not output.exists()
