"""Replay receiver log replies and reject ambiguous or mismatched acknowledgements."""

import pytest

from models.dflash_v1.msprof_cli import DynamicCapture


@pytest.mark.parametrize("command", ("start", "stop", "quit"))
@pytest.mark.parametrize("tail", ("\r\n> ", " > ", "......\r\n> "))
def test_receiver_pid_reply_waits_for_a_complete_message(command, tail):
    # The start variant reproduces the exact receiver log in the bug report.
    prefix = f"[INFO] Start profiling....\r\n> dynamic profiling for pid 408845 {command} success"
    response = (prefix + tail).encode()
    for offset in range(1, len(prefix) + 1):
        assert DynamicCapture.acknowledgement(command, response[:offset], 408845) is None
    acknowledgement = DynamicCapture.acknowledgement(command, response, 408845)
    assert acknowledgement.startswith(f"dynamic profiling for pid 408845 {command} success")


@pytest.mark.parametrize("command", ("start", "stop", "quit"))
def test_legacy_reply_is_still_supported(command):
    response = f"(msprof) dynamic profiling {command} success......\r\n(msprof) ".encode()
    assert DynamicCapture.acknowledgement(command, response, 408845) is not None


@pytest.mark.parametrize("command", ("start", "stop", "quit"))
@pytest.mark.parametrize("pid", (40884, 4088450, 408846))
def test_receiver_reply_cannot_acknowledge_another_pid(command, pid):
    response = f"> dynamic profiling for pid {pid} {command} success\r\n> ".encode()
    with pytest.raises(RuntimeError, match="unexpected PID"):
        DynamicCapture.acknowledgement(command, response, 408845)


@pytest.mark.parametrize("response", (
    b"[INFO] Start profiling....\r\n> ",
    b"> start\r\n> ",
    b"> dynamic profiling start success\r\n> ",
    b"> dynamic profiling for pid 408845 start successful\r\n> ",
    b"> dynamic profiling for pid 408845 start success pending\r\n> ",
    b"> dynamic profiling for pid 408845 stop success\r\n> ",
    b"> dynamic profiling for pid 408845 start failed\r\n> ",
    b"> dynamic profiling for pid 408845 start success",
))
def test_prompt_partial_wrong_command_or_failure_cannot_release_work(response):
    assert DynamicCapture.acknowledgement("start", response, 408845) is None


@pytest.mark.parametrize("command", ("start", "stop", "quit"))
def test_pid_failures_are_reported_immediately(command):
    response = f"> dynamic profiling for pid 408845 {command} failed\r\n> ".encode()
    assert DynamicCapture.FAILURE.search(response)
