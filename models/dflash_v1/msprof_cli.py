"""Single-stage capture using only the msprof dynamic CLI and a local socket.

The application drains the NPU and waits. The controller releases it only after
msprof acknowledges start, and releases postprocessing only after stop and quit.
No profiling API, fixed-delay capture, or process-wide SIGSTOP is used.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import errno
import json
import math
import os
from pathlib import Path
import pty
import re
import selectors
import signal
import socket
import subprocess
import sys
import termios
from time import monotonic, perf_counter


CONTROL_FD_ENV = "DFLASH_MSPROF_CONTROL_FD"
TIMEOUT_ENV = "DFLASH_MSPROF_CONTROL_TIMEOUT"
COLLECTOR = "msprof dynamic CLI"
DEFAULT_TIMEOUT = 600.0
SINGLE_STAGES = (
    "prefill", "feature-project", "draft", "verify-input", "verify",
    "target-top1", "accept-commit", "draft-verify", "decode-round",
)
ORDINARY_STAGES = ("prefill", "decode")
CPP_DFLASH_STAGES = ("prefill", "draft", "verify")
PROFILE_STAGES = (*SINGLE_STAGES, "decode", "all")


def stages_for_mode(mode="dflash", backend="python"):
    if mode not in {"ordinary", "dflash"} or backend not in {"python", "cpp"}:
        raise ValueError("invalid profiling mode or backend")
    return ORDINARY_STAGES if mode == "ordinary" else (
        CPP_DFLASH_STAGES if backend == "cpp" else SINGLE_STAGES
    )


def captured_calls(stage, mode="dflash"):
    result = {
        "prefill": int(stage == "prefill"),
        "draft": int(stage in {"draft", "draft-verify", "decode-round"}),
        "target_verify": int(stage in {"verify", "draft-verify", "decode-round"}),
    }
    if mode == "ordinary":
        result["target_decode"] = int(stage == "decode")
    return result


def positive_timeout(value) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("profile timeout must be finite and positive")
    return seconds


class MsprofStageProfiler:
    """Application-side barriers; the parent process owns msprof collection."""

    def __init__(self, output, device_id, metrics, synchronize, *, stage, mode="dflash"):
        self.output, self.device_id, self.metrics = output, device_id, metrics
        self.synchronize, self.stage = synchronize, stage
        self.mode = mode
        self.stages = stages_for_mode(mode)
        self.windows = 0
        self.elapsed_ms = None
        self._channel = None
        self._reader = None
        self._children = []

    def for_stage(self, stage):
        """Share one application channel while keeping each window independent."""
        if self.stage != "all" or self._channel is None:
            raise RuntimeError("stage views require an active all-stage profiler")
        index = len(self._children)
        if index >= len(self.stages) or stage != self.stages[index]:
            raise RuntimeError("all-stage captures must follow the declared stage order")
        if self._children and self._children[-1].windows != 1:
            raise RuntimeError("previous stage did not capture exactly one window")
        child = MsprofStageProfiler(
            str(Path(self.output) / stage), self.device_id, self.metrics,
            self.synchronize, stage=stage, mode=self.mode,
        )
        child._channel, child._reader = self._channel, self._reader
        self._children.append(child)
        return child

    def __enter__(self):
        if os.environ.get("PROFILING_MODE") != "dynamic" or CONTROL_FD_ENV not in os.environ:
            raise RuntimeError("single-stage profiling must be launched by tools/run_msprof.sh --profile-stage")
        self._channel = socket.socket(fileno=int(os.environ.pop(CONTROL_FD_ENV)))
        try:
            self._channel.set_inheritable(False)
            self._channel.settimeout(positive_timeout(os.environ.get(TIMEOUT_ENV, DEFAULT_TIMEOUT)))
            self._reader = self._channel.makefile("rb")
        except BaseException:
            self.close()
            raise
        return self

    def _exchange(self, message, expected):
        self._channel.sendall(json.dumps(message).encode() + b"\n")
        try:
            raw = self._reader.readline(4097)
        except TimeoutError as error:
            raise RuntimeError(f"timed out waiting for msprof controller: {expected}") from error
        if not raw or len(raw) > 4096 or not raw.endswith(b"\n"):
            raise RuntimeError("msprof controller disconnected or sent an invalid response")
        response = json.loads(raw)
        if response != {"event": expected}:
            raise RuntimeError(f"expected msprof {expected!r}, received {response!r}")

    @contextmanager
    def capture(self):
        if self._channel is None or self.windows or self.stage == "all":
            raise RuntimeError("stage profiler requires exactly one capture window")
        self.synchronize()
        self._exchange({
            "event": "ready", "pid": os.getpid(), "stage": self.stage,
            "output": self.output, "device_id": self.device_id, "metrics": self.metrics,
        }, "started")
        self.windows += 1
        started = perf_counter()
        success = False
        try:
            yield
            success = True
        finally:
            try:
                self.synchronize()
                self.elapsed_ms = (perf_counter() - started) * 1000
            except BaseException:
                success = False
                raise
            finally:
                self._exchange({"event": "done", "success": success}, "stopped")

    def close(self):
        if self._reader is not None:
            self._reader.close()
            self._reader = None
        if self._channel is not None:
            self._channel.close()
            self._channel = None

    def __exit__(self, *_exc):
        self.close()


class DynamicCapture:
    """Own the application, msprof, and their acknowledged start/stop protocol."""

    # CANN runtime's DynProfClient prints these only after the server response.
    # https://gitcode.com/cann/runtime/blob/68752f679cfb68365e472eec38855ec4fd4721f6/src/dfx/msprof/collector/dvvp/msprof/dynamic_profiling/src/dyn_prof_client.cpp
    ACK = {name: re.compile(rb"dynamic profiling " + name.encode() + rb" success\.{2,}", re.I)
           for name in ("start", "stop", "quit")}
    # Receiver msprof also emits: "> dynamic profiling for pid 408845 start success".
    # Require the complete command reply and the attached PID, including when
    # the reply is split across PTY reads. A bare prompt/startup log is not an ACK.
    PID_ACK = {
        name: re.compile(
            rb"\bdynamic[ \t]+profiling[ \t]+for[ \t]+pid[ \t]+(?P<pid>[0-9]+)[ \t]+"
            + name.encode() + rb"[ \t]+success(?:\.{2,})?[ \t]*(?=[\r\n>])", re.I,
        ) for name in ("start", "stop", "quit")
    }
    FAILURE = re.compile(
        rb"\[ERROR\]|dynamic profiling(?: for pid [0-9]+)? "
        rb"(?:already started|has not started|device has not been set up|(?:start|stop|quit) failed)"
        rb"|cannot connect to server|invalid option",
        re.I,
    )

    @classmethod
    def acknowledgement(cls, name, output, application_pid):
        match = cls.PID_ACK[name].search(output)
        if match is not None:
            if int(match["pid"]) != application_pid:
                raise RuntimeError(
                    f"msprof {name} acknowledged unexpected PID {match['pid'].decode()}; "
                    f"expected application PID {application_pid}"
                )
        else:
            match = cls.ACK[name].search(output)
        return None if match is None else match[0].decode("utf-8", errors="replace").strip()

    def __init__(self, args):
        self.args = args
        self.stages = stages_for_mode(getattr(args, "mode", "dflash"), getattr(args, "backend", "python"))
        self.app = self.prof = None
        self.channel = None
        self.terminal = None
        self.selector = selectors.DefaultSelector()
        self.control_buffer = bytearray()
        self.prof_buffer = bytearray()
        self.messages = []
        self.control_closed = False
        self.evidence = {
            "schema_version": 1, "status": "RUNNING", "collector": COLLECTOR,
            "profile_stage": args.stage, "profile_output": str(Path(args.output).resolve()),
            "timeout_seconds": args.timeout, "start_acknowledged": False,
            "stop_acknowledged": False, "quit_acknowledged": False,
            "acknowledgements": {},
            "capture_completed": False, "events": [],
        }
        self.capture_evidence = self.evidence
        self.current_stage = args.stage
        if args.stage == "all":
            self.evidence.update(schema_version=2, stages=list(self.stages), captures=[])
        self.evidence.update(profile_mode=getattr(args, "mode", "dflash"),
                             profile_backend=getattr(args, "backend", "python"))

    def event(self, name):
        event = {"event": name, "monotonic_seconds": monotonic()}
        if self.args.stage == "all":
            event["stage"] = self.current_stage
        self.evidence["events"].append(event)
        if self.capture_evidence is not self.evidence:
            self.capture_evidence["events"].append(dict(event))
        label = f"stage={self.current_stage} " if self.args.stage == "all" else ""
        print(f"[stage-profile] {label}{name}", flush=True)

    def pump(self, delay):
        for key, _ in self.selector.select(delay):
            try:
                data = os.read(key.fd, 8192)
            except OSError as error:
                if key.data == "msprof" and error.errno == errno.EIO:
                    data = b""  # PTY peer exited.
                else:
                    raise
            if not data:
                self.selector.unregister(key.fileobj)
                if key.data == "control":
                    self.control_closed = True
                    self.event("application_control_closed")
                continue
            if key.data == "control":
                self.control_buffer.extend(data)
                while b"\n" in self.control_buffer:
                    line, _, rest = self.control_buffer.partition(b"\n")
                    self.control_buffer = bytearray(rest)
                    if len(line) > 4096:
                        raise RuntimeError("oversized stage control message")
                    message = json.loads(line)
                    if not isinstance(message, dict):
                        raise RuntimeError("invalid stage control message")
                    self.messages.append(message)
                if len(self.control_buffer) > 4096:
                    raise RuntimeError("oversized stage control message")
            else:
                sys.stdout.buffer.write(data)
                sys.stdout.buffer.flush()
                if key.data == "msprof":
                    self.prof_buffer.extend(data)
                    if len(self.prof_buffer) > 1048576:
                        raise RuntimeError("msprof emitted excessive output without completing a command")

    def check_processes(self):
        if self.app.poll() is not None:
            raise RuntimeError(f"application exited before handshake completed: {self.app.returncode}")
        if self.prof is not None and self.prof.poll() is not None:
            raise RuntimeError(f"msprof exited before handshake completed: {self.prof.returncode}")
        if self.FAILURE.search(self.prof_buffer):
            raise RuntimeError("msprof rejected dynamic collection; see its output above")

    def receive(self, event):
        deadline = monotonic() + self.args.timeout
        while not self.messages:
            if self.control_closed:
                raise RuntimeError(f"application control disconnected before {event}")
            self.check_processes()
            remaining = deadline - monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for application {event}")
            self.pump(min(remaining, 0.2))
        message = self.messages.pop(0)
        if message.get("event") != event:
            raise RuntimeError(f"expected application {event!r}, received {message!r}")
        return message

    def send(self, event):
        self.channel.sendall(json.dumps({"event": event}).encode() + b"\n")
        self.event("application_" + event)

    def command(self, name):
        # No other command is pending; discard only output already consumed.
        self.prof_buffer.clear()
        os.write(self.terminal, name.encode() + b"\n")
        self.event("msprof_" + name + "_sent")
        deadline = monotonic() + self.args.timeout
        while True:
            if self.FAILURE.search(self.prof_buffer):
                raise RuntimeError(f"msprof {name} failed; see its output above")
            acknowledgement = self.acknowledgement(name, self.prof_buffer, self.app.pid)
            if acknowledgement is not None:
                self.capture_evidence[name + "_acknowledged"] = True
                self.capture_evidence["acknowledgements"][name] = acknowledgement
                self.event("msprof_" + name + "_acknowledged")
                return
            remaining = deadline - monotonic()
            if remaining <= 0:
                tail = bytes(self.prof_buffer[-500:]).decode("utf-8", errors="replace")
                raise TimeoutError(
                    f"msprof did not acknowledge {name} for application PID {self.app.pid}; "
                    f"no stage is released without success. Last collector output: {tail!r}"
                )
            # Drain an exit's final acknowledgement before checking return codes.
            self.pump(min(remaining, 0.2))
            if self.acknowledgement(name, self.prof_buffer, self.app.pid) is None:
                self.check_processes()

    def wait_exit(self, process, description):
        deadline = monotonic() + self.args.timeout
        while process.poll() is None:
            if monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for {description} to exit")
            self.pump(0.2)
        self.pump(0)
        if process.returncode != 0:
            raise RuntimeError(f"{description} failed: exit={process.returncode}")

    def run(self):
        parent, child = socket.socketpair()
        self.channel = parent
        environment = dict(os.environ)
        environment.update({
            "PROFILING_MODE": "dynamic", CONTROL_FD_ENV: str(child.fileno()),
            # done -> stopped includes stop, quit, and collector exit deadlines.
            TIMEOUT_ENV: str(4 * self.args.timeout), "PYTHONUNBUFFERED": "1",
        })
        environment.pop("PROFILING_OPTIONS", None)
        environment.pop("DFLASH_MSPROF_PROCESS_CAPTURE", None)
        try:
            self.app = subprocess.Popen(
                self.args.application, env=environment, pass_fds=(child.fileno(),),
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, start_new_session=True,
            )
        finally:
            child.close()
        self.evidence["application_pid"] = self.app.pid
        self.selector.register(parent, selectors.EVENT_READ, "control")
        self.selector.register(self.app.stdout, selectors.EVENT_READ, "application")
        stages = self.stages if self.args.stage == "all" else (self.args.stage,)
        for stage in stages:
            output = str(Path(self.args.output) / stage) if self.args.stage == "all" else self.args.output
            self.current_stage = stage
            if self.args.stage == "all":
                self.capture_evidence = {
                    "profile_stage": stage, "profile_output": str(Path(output).resolve()),
                    "start_acknowledged": False, "stop_acknowledged": False,
                    "quit_acknowledged": False, "acknowledgements": {},
                    "capture_completed": False, "events": [],
                }
                self.evidence["captures"].append(self.capture_evidence)
            self.run_capture(stage, output)
        self.wait_exit(self.app, "application")
        if self.messages:
            raise RuntimeError("application sent extra stage control messages")
        if self.args.stage == "all":
            for key in ("start_acknowledged", "stop_acknowledged", "quit_acknowledged"):
                self.evidence[key] = all(c[key] for c in self.evidence["captures"])
        self.evidence["capture_completed"] = True
        self.evidence["status"] = "PASS_CONTROL"

    def run_capture(self, stage, output):
        ready = self.receive("ready")
        if (ready.get("pid") != self.app.pid or ready.get("stage") != stage
                or ready.get("metrics") != self.args.metrics
                or Path(ready.get("output", "")).resolve() != Path(output).resolve()):
            raise RuntimeError("application ready message does not match the requested capture")
        self.event("application_ready")
        self.capture_evidence["device_id"] = ready.get("device_id")
        argv = [
            self.args.msprof_bin, "--dynamic=on", f"--pid={self.app.pid}",
            f"--output={output}", "--ascendcl=on", "--runtime-api=on",
            "--task-time=on", "--aicpu=on", "--ai-core=on", "--aic-mode=task-based",
            f"--aic-metrics={self.args.metrics}",
        ]
        self.capture_evidence["msprof_arguments"] = argv
        self.prof_buffer.clear()
        master, slave = pty.openpty()
        self.terminal = master
        settings = termios.tcgetattr(slave)
        settings[3] &= ~termios.ECHO
        termios.tcsetattr(slave, termios.TCSANOW, settings)
        tool_environment = dict(os.environ)
        for key in ("PROFILING_MODE", "PROFILING_OPTIONS", CONTROL_FD_ENV, TIMEOUT_ENV):
            tool_environment.pop(key, None)
        try:
            self.prof = subprocess.Popen(
                argv, env=tool_environment, stdin=slave, stdout=slave, stderr=slave,
                start_new_session=True,
            )
        finally:
            os.close(slave)
        self.capture_evidence["msprof_pid"] = self.prof.pid
        self.selector.register(master, selectors.EVENT_READ, "msprof")
        self.command("start")
        self.send("started")
        done = self.receive("done")
        self.event("application_done")
        self.command("stop")
        self.command("quit")
        self.wait_exit(self.prof, "msprof")
        self.capture_evidence["msprof_exit_code"] = self.prof.returncode
        # The next stage reattaches to the same application with its own output.
        # Release the old PTY before receiving the next ready message.
        try:
            self.selector.unregister(self.terminal)
        except KeyError:
            pass
        os.close(self.terminal)
        self.terminal = None
        self.prof = None
        self.prof_buffer.clear()
        self.send("stopped")
        if done.get("success") is not True:
            raise RuntimeError("application did not complete a successful stage")
        if self.args.stage == "all":
            self.capture_evidence["capture_completed"] = True

    def close(self):
        # On failure keep the application alive briefly so msprof can stop its
        # server via quit. Never freeze that server using SIGSTOP.
        if self.prof is not None and self.prof.poll() is None:
            try:
                os.write(self.terminal, b"quit\n")
                self.prof.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                pass
        for process in (self.prof, self.app):
            if process is not None and process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait(timeout=3)
                except ProcessLookupError:
                    pass
        if self.app is not None and self.app.stdout is not None:
            self.app.stdout.close()
        self.selector.close()
        if self.channel is not None:
            self.channel.close()
        if self.terminal is not None:
            os.close(self.terminal)
        self.evidence["application_exit_code"] = None if self.app is None else self.app.returncode
        if self.prof is not None:
            self.capture_evidence["msprof_exit_code"] = self.prof.returncode
        elif self.args.stage != "all":
            self.evidence.setdefault("msprof_exit_code", None)
        if self.args.stage == "all":
            self.evidence["msprof_exit_code"] = (
                0 if len(self.evidence["captures"]) == len(self.stages)
                and all(c.get("msprof_exit_code") == 0 for c in self.evidence["captures"])
                else None
            )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--msprof-bin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stage", required=True, choices=PROFILE_STAGES)
    parser.add_argument("--mode", choices=("ordinary", "dflash"), default="dflash")
    parser.add_argument("--backend", choices=("python", "cpp"), default="python")
    parser.add_argument("--metrics", default="PipeUtilization", choices=("PipeUtilization", "Memory", "MemoryUB"))
    parser.add_argument("--timeout", type=positive_timeout, default=DEFAULT_TIMEOUT)
    parser.add_argument("--control-report", required=True)
    parser.add_argument("application", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.stage != "all" and args.stage not in stages_for_mode(args.mode, args.backend):
        parser.error("stage is unavailable for the selected mode/backend")
    if args.application[:1] == ["--"]:
        args.application.pop(0)
    if not args.application:
        parser.error("an application command is required after --")
    report = Path(args.control_report)
    if report.exists() or report.is_symlink():
        parser.error("control report must be a new file")
    controller = DynamicCapture(args)

    def interrupted(signum, _frame):
        raise KeyboardInterrupt(f"received signal {signum}")

    previous = {sig: signal.signal(sig, interrupted) for sig in (signal.SIGINT, signal.SIGTERM)}
    status = 0
    try:
        controller.run()
    except (Exception, KeyboardInterrupt) as error:
        status = 1
        controller.evidence.update(status="FAIL", error=str(error))
        print(f"[stage-profile] FAIL: {error}", file=sys.stderr, flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        try:
            controller.close()
        except Exception as error:
            status = 1
            controller.evidence.update(status="FAIL", cleanup_error=str(error))
            print(f"[stage-profile] cleanup failed: {error}", file=sys.stderr, flush=True)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(controller.evidence, indent=2) + "\n", encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
