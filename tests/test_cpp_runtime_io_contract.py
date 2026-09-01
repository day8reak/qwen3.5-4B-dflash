from __future__ import annotations

import copy
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.cpp_runtime import (  # noqa: E402
    CPP_RUNNER_ID,
    validate_cpp_runner_report,
)


def _mode(generation_mode: str) -> dict[str, object]:
    measurement = {
        "generated_token_ids": [11, 12],
        "stop_reason": "length",
    }
    return {
        "status": "PASS",
        "generation_mode": generation_mode,
        "warmup": 3,
        "repetitions": 10,
        "stable_generated_token_ids": [11, 12],
        "stable_stop_reason": "length",
        "measurements": [copy.deepcopy(measurement) for _ in range(10)],
    }


def _report() -> dict[str, object]:
    return {
        "status": "PASS",
        "runner_id": CPP_RUNNER_ID,
        "cpu_fallback": False,
        "device_id": 0,
        "model": {"sha256": "a" * 64},
        "prompt_token_ids": [10],
        "limits": {"max_new_tokens": 2, "max_draft_tokens": 15},
        "protocol": {
            "warmup": 3,
            "repetitions": 10,
            "device_memory_allocation_policy": "normal-only",
        },
        "abi": {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["target_top1", "draft_top1"],
            "dtype": "int64",
            "sequence_length": 64,
            "draft_width": 15,
        },
        "model_memory_query": {
            "source": "aclmdlQuerySize",
            "work_bytes": 64,
            "weight_bytes": 256,
        },
        "execution_io_counters": {
            "input_policy": (
                "persistent device mirror plus changed contiguous ranges"
            ),
            "target_output_policy": (
                "download only the last draft_width_plus_one rows needed by "
                "proposal or verify"
            ),
            "model_executions": 20,
            "stream_synchronizations": 20,
            "host_to_device_bytes": 256,
            "full_host_to_device_bytes": 2048,
            "host_to_device_bytes_avoided": 1792,
            "device_to_host_bytes": 512,
            "full_device_to_host_bytes": 4096,
            "device_to_host_bytes_avoided": 3584,
            "maximum_target_elements_per_call": 16,
        },
        "ordinary": _mode("ordinary-greedy"),
        "dflash": _mode("dflash-strict-greedy"),
        "ordinary_parity": {
            "status": "PASS",
            "token_id_mismatches": 0,
            "eos_mismatches": 0,
        },
    }


def _validate(report: dict[str, object]) -> None:
    validate_cpp_runner_report(
        report,
        prompt_token_ids=[10],
        om_sha256="a" * 64,
        device_id=0,
        max_new_tokens=2,
        max_draft_tokens=15,
    )


def test_cpp_runner_accepts_locked_ranged_io_evidence() -> None:
    _validate(_report())


def test_cpp_runner_accepts_huge_first_build_identity() -> None:
    report = _report()
    report["protocol"]["device_memory_allocation_policy"] = "huge-first"
    _validate(report)


@pytest.mark.parametrize("value", [None, "unknown"])
def test_cpp_runner_rejects_unknown_device_memory_policy(
    value: object,
) -> None:
    report = _report()
    report["protocol"]["device_memory_allocation_policy"] = value
    with pytest.raises(RuntimeError, match="device memory policy"):
        _validate(report)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_synchronizations", 19),
        ("maximum_target_elements_per_call", 17),
        ("host_to_device_bytes_avoided", 1791),
        ("device_to_host_bytes", 4097),
    ],
)
def test_cpp_runner_rejects_inconsistent_io_evidence(
    field: str,
    value: int,
) -> None:
    report = _report()
    report["execution_io_counters"][field] = value
    with pytest.raises(RuntimeError):
        _validate(report)
