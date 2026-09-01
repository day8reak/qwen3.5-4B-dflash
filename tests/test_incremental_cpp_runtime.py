from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.cpp_runtime import (  # noqa: E402
    INCREMENTAL_CPP_RUNNER_ID,
    INCREMENTAL_STATE_POLICY,
    _INCREMENTAL_GRAPH_ABI,
    _resolve_incremental_oms,
    validate_cpp_runner_options,
    validate_incremental_cpp_runner_report,
)
from qwen35_dflash.ascend310p import cpp_runtime  # noqa: E402


def _mode(generation_mode: str) -> dict[str, object]:
    measurement = {
        "generated_token_ids": [11, 12, 13, 14, 15, 16],
        "stop_reason": "length",
    }
    return {
        "status": "PASS",
        "generation_mode": generation_mode,
        "warmup": 3,
        "repetitions": 10,
        "stable_generated_token_ids": [11, 12, 13, 14, 15, 16],
        "stable_stop_reason": "length",
        "measurements": [copy.deepcopy(measurement) for _ in range(10)],
    }


def _hashes() -> dict[str, str]:
    return {
        role: hashlib.sha256(role.encode("utf-8")).hexdigest()
        for role in _INCREMENTAL_GRAPH_ABI
    }


def _report() -> dict[str, object]:
    hashes = _hashes()
    models = [
        {
            "role": role,
            "sha256": hashes[role],
            "work_bytes": 64,
            "weight_bytes": 256,
        }
        for role in _INCREMENTAL_GRAPH_ABI
    ]
    return {
        "status": "PASS",
        "runner_id": INCREMENTAL_CPP_RUNNER_ID,
        "candidate_status": "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE",
        "cpu_fallback": False,
        "device_id": 0,
        "models": models,
        "prompt_token_ids": [10],
        "limits": {"max_new_tokens": 6, "max_draft_tokens": 3},
        "protocol": {
            "warmup": 3,
            "repetitions": 10,
            "state_reset_policy": (
                "async clear queued inside first prefill; no reset-only barrier"
            ),
            "state_reset_device_work_included_by_prefill_barrier": True,
        },
        "abi": {
            "id": (
                "qwen35-4b-dflash-ascend310p-incremental-performance-v2"
            ),
            "state_policy": "explicit device-resident ping-pong",
            "proposal_width": 15,
            "verify_width": 16,
        },
        "model_memory_query": {
            "source": "aclmdlQuerySize",
            "sum_work_bytes": 256,
            "max_work_bytes": 64,
            "sum_weight_bytes": 1024,
            "state_device_bytes": 2048,
            "carrier_device_bytes": 512,
            "explicit_allocated_device_bytes_excluding_runtime": 3648,
            "load_policy": (
                "four aclmdlLoadFromFileWithMem sessions; one max-sized serial "
                "workspace; separate per-artifact weights; no cross-OM weight "
                "sharing assumed"
            ),
        },
        "execution_io_counters": {
            "model_executions": 156,
            "target_prefill_executions": 26,
            "target_decode1_executions": 65,
            "draft_propose_executions": 39,
            "target_verify_commit_executions": 26,
            "stream_synchronizations": 117,
            "state_resets": 26,
            "state_memset_operations": 52,
            "state_memset_bytes": 4096,
            "host_to_device_operations": 80,
            "host_to_device_bytes": 4096,
            "device_to_host_operations": 117,
            "device_to_host_bytes": 8192,
            "state_device_bytes": 2048,
            "carrier_device_bytes": 512,
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
    validate_incremental_cpp_runner_report(
        report,
        prompt_token_ids=[10],
        om_sha256_by_role=_hashes(),
        device_id=0,
        max_new_tokens=6,
        max_draft_tokens=3,
    )


def test_incremental_runner_report_closes_state_and_transaction_counters() -> None:
    _validate(_report())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("stream_synchronizations", 118),
        ("device_to_host_operations", 118),
        ("state_resets", 25),
        ("model_executions", 157),
    ],
)
def test_incremental_runner_rejects_inconsistent_execution_counters(
    field: str,
    value: int,
) -> None:
    report = _report()
    report["execution_io_counters"][field] = value
    with pytest.raises(RuntimeError):
        _validate(report)


def test_incremental_runner_config_is_explicit() -> None:
    identity = validate_cpp_runner_options(
        {
            "device_model": "Ascend310P3",
            "cann": "test-cann",
            "driver": "test-driver",
            "firmware": "test-firmware",
            "runtime": "AscendCL",
            "state_policy": INCREMENTAL_STATE_POLICY,
            "pad_token_id": 0,
        },
        0,
    )
    assert identity["state_policy"] == INCREMENTAL_STATE_POLICY


def test_resolve_incremental_oms_locks_all_four_abis_and_hashes(
    tmp_path: Path,
) -> None:
    graphs = []
    for role, (inputs, outputs) in _INCREMENTAL_GRAPH_ABI.items():
        om = tmp_path / f"{role}.om"
        om.write_bytes(role.encode("utf-8"))
        graphs.append(
            {
                "name": role,
                "role": role,
                "input_names": inputs,
                "output_names": outputs,
                "om": {
                    "path": om.name,
                    "sha256": hashlib.sha256(om.read_bytes()).hexdigest(),
                },
            }
        )
    manifest_path = tmp_path / "deployment.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
                "status": "PASS",
                "graphs": graphs,
            }
        ),
        encoding="utf-8",
    )
    resolved, _ = _resolve_incremental_oms(manifest_path)
    assert list(resolved) == list(_INCREMENTAL_GRAPH_ABI)
    assert all(item[0].is_file() for item in resolved.values())

    graphs[2]["output_names"] = ["wrong"]
    manifest_path.write_text(json.dumps({
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "status": "PASS",
        "graphs": graphs,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="output order"):
        _resolve_incremental_oms(manifest_path)


def test_run_cpp_pair_routes_all_four_hash_locked_oms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_root = tmp_path / "run"
    bundle = run_root / "bundle"
    bundle.mkdir(parents=True)
    monkeypatch.setenv("AI_RUN_DIR", str(run_root))
    air_manifest = bundle / "air-manifest.json"
    air_manifest.write_text("{}", encoding="utf-8")
    graphs = []
    for role, (inputs, outputs) in _INCREMENTAL_GRAPH_ABI.items():
        om = bundle / f"{role}.om"
        om.write_bytes(role.encode("utf-8"))
        graphs.append(
            {
                "name": role,
                "role": role,
                "input_names": inputs,
                "output_names": outputs,
                "om": {
                    "path": om.name,
                    "sha256": hashlib.sha256(om.read_bytes()).hexdigest(),
                },
            }
        )
    deployment = bundle / "deployment-manifest.json"
    deployment.write_text(
        json.dumps(
            {
                "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
                "status": "PASS",
                "air_manifest": {
                    "path": air_manifest.name,
                    "sha256": hashlib.sha256(
                        air_manifest.read_bytes()
                    ).hexdigest(),
                },
                "compiler": {"identity": "fake"},
                "target": {"soc_version": "Ascend310P3"},
                "graphs": graphs,
            }
        ),
        encoding="utf-8",
    )
    runner = run_root / "fake-incremental-runner"
    runner.write_text("fake", encoding="utf-8")
    runner.chmod(0o755)
    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", lambda _: runner)
    captured: dict[str, list[str]] = {}

    def execute(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        output = Path(command[command.index("--output") + 1])
        output.write_text(json.dumps(_report()), encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="fake PASS\n")

    payload = cpp_runtime.run_cpp_pair(
        deployment_manifest=deployment,
        runner=runner,
        runner_options={
            "device_model": "Ascend310P3",
            "cann": "fake-cann",
            "driver": "fake-driver",
            "firmware": "fake-firmware",
            "runtime": "fake-AscendCL",
            "state_policy": INCREMENTAL_STATE_POLICY,
            "pad_token_id": 0,
        },
        prompt_token_ids=[10],
        eos_token_ids=[],
        device_id=0,
        max_new_tokens=6,
        max_draft_tokens=3,
        raw_output=run_root / "out" / "raw.json",
        log_output=run_root / "log" / "runner.log",
        progress=False,
        execute=execute,
    )
    command = captured["command"]
    for role in _INCREMENTAL_GRAPH_ABI:
        assert f"--{role}" in command
        assert f"--{role}-sha256" in command
    assert "--model" not in command
    assert payload["backend_metadata"]["state_policy"] == (
        INCREMENTAL_STATE_POLICY
    )
    assert payload["backend_metadata"]["state_implementation"] == (
        "explicit Target/Draft device state"
    )
