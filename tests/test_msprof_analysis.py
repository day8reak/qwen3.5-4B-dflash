from __future__ import annotations

import csv
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.msprof_analysis import (  # noqa: E402
    MsprofAnalysisError,
    analyze_incremental_msprof,
)


def _runner_report() -> dict[str, object]:
    models = [
        {"role": "target-prefill", "model_id": 1},
        {"role": "target-prefill-head", "model_id": 2},
        {"role": "draft-propose", "model_id": 3},
        {"role": "target-verify-commit", "model_id": 4},
    ]
    trace = [
        {"ordinal": 0, "model_id": 1, "physical_rows": 64},
        {"ordinal": 1, "model_id": 1, "physical_rows": 64},
        {"ordinal": 2, "model_id": 2, "physical_rows": 1},
        {"ordinal": 3, "model_id": 4, "physical_rows": 1},
        {"ordinal": 4, "model_id": 3, "physical_rows": 128},
        {"ordinal": 5, "model_id": 4, "physical_rows": 4},
        {"ordinal": 6, "model_id": 3, "physical_rows": 4},
        {"ordinal": 7, "model_id": 4, "physical_rows": 1},
    ]
    return {
        "schema_version": 7,
        "status": "PASS",
        "runner_id": "qwen35-dflash-ascendcl-cpp-incremental-v3",
        "runner_version": "1.15.0",
        "cpu_fallback": False,
        "device_id": 0,
        "models": models,
        "abi": {
            "physical_topology": (
                "split-prefill-head-four-resident-unified-target-step-v1"
            ),
            "prefill_width": 64,
            "verify_width": 16,
        },
        "protocol": {
            "kind": "profile",
            "formal_latency_evidence": False,
            "profile_model_execution_trace_enabled": True,
            "dflash_sync_window": 1,
            "draft_feature_policy": "committed-prefix",
            "prefill_completion_policy": "separate",
        },
        "execution_io_counters": {
            "model_executions": 8,
            "target_prefill_executions": 2,
            "target_prefill_head_executions": 1,
            "target_decode1_executions": 2,
            "draft_propose_executions": 2,
            "target_verify_commit_executions": 1,
            "prefill_draft_propose_executions": 1,
            "prefill_feature_rows_batched": 128,
            "draft_verify_feature_input_rows": 4,
            "draft_verify_full_width_equivalent_rows": 16,
            "draft_verify_feature_rows_elided": 12,
            "draft_verify_fixed_width_executions": 0,
            "draft_verify_committed_prefix_executions": 1,
            "draft_verify_pending_upper_bound_executions": 0,
            "draft_dynamic_gear_count": 18,
            "draft_verify_dynamic_gear_count": 16,
            "draft_prefill_dynamic_gear_count": 2,
            "host_to_device_operations": 3,
            "host_to_device_bytes": 1024,
            "device_to_host_operations": 4,
            "device_to_host_bytes": 1028,
            "decode_id_device_compaction_operations": 1,
            "decode_id_device_compaction_bytes": 8,
            "state_memset_operations": 2,
            "state_initialization_memset_operations": 0,
            "stream_synchronizations": 4,
            "speculative_sync_windows": 1,
            "speculative_synchronizations_elided": 0,
            "speculative_d2h_operations_elided": 0,
            "speculative_d2h_padding_bytes": 0,
            "compact_slot_bytes": 512,
            "compact_ordinary_result_bytes": 257,
            "compact_verify_result_bytes": 452,
            "prefill_verify_coalesced_windows": 0,
            "prefill_verify_synchronizations_elided": 0,
            "prefill_verify_d2h_operations_elided": 0,
            "prefill_verify_d2h_padding_bytes": 0,
            "prefill_verify_prefill_slot0_windows": 0,
            "prefill_verify_prefill_slot1_windows": 0,
            "prefill_completion_synchronizations": 1,
            "state_initialization_stream_synchronizations": 0,
        },
        "profile_model_execution_trace": trace,
    }


def _schema_nine_runner_report() -> dict[str, object]:
    report = _runner_report()
    report["schema_version"] = 9
    report["runner_version"] = "1.17.0"
    protocol = report["protocol"]
    protocol["device_memory_allocation_policy"] = "normal-only"
    protocol["maximum_supported_dflash_sync_window"] = 8
    counters = report["execution_io_counters"]
    counters["speculative_window_staging_operations"] = 0
    counters["speculative_window_staging_bytes"] = 0
    counters["speculative_window_staging_device_bytes"] = 4096
    counters["speculative_window_staging_pinned_host_bytes"] = 4096
    return report


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _op_rows_for_invocations(
    invocations: list[tuple[int, int]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    timestamp = 100.0
    for index, (model_id, infer_id) in enumerate(invocations, start=1):
        rows.extend(
            [
                {
                    "Model ID": model_id,
                    "Infer ID": infer_id,
                    "Op Name": f"MatMul_{model_id}",
                    "OP Type": "MatMul",
                    "Task Start Time(us)": timestamp,
                    "Task Duration(us)": 10.0 * index,
                },
                {
                    "Model ID": model_id,
                    "Infer ID": infer_id,
                    "Op Name": f"Custom_{model_id}",
                    "OP Type": "Custom",
                    "Task Start Time(us)": timestamp + 10.0 * index,
                    "Task Duration(us)": 2.0 * index,
                },
            ]
        )
        timestamp += 100.0
    return rows


def _op_rows() -> list[dict[str, object]]:
    return _op_rows_for_invocations(
        [
            (1, 1),
            (1, 2),
            (2, 1),
            (4, 1),
            (3, 1),
            (4, 2),
            (3, 2),
            (4, 3),
        ]
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_api_csv(
    path: Path,
    *,
    execute_count: int = 8,
    memcpy_count: int = 8,
    synchronize_count: int = 4,
) -> None:
    rows = [
        {
            "Device_id": "host",
            "Level": "AscendCL",
            "API Name": "aclmdlExecuteAsync",
            "Time(us)": 80.0,
            "Count": execute_count,
        },
        {
            "Device_id": "host",
            "Level": "AscendCL",
            "API Name": "aclrtMemcpyAsync",
            "Time(us)": 40.0,
            "Count": memcpy_count,
        },
        {
            "Device_id": "host",
            "Level": "AscendCL",
            "API Name": "aclrtMemsetAsync",
            "Time(us)": 10.0,
            "Count": 2,
        },
        {
            "Device_id": "host",
            "Level": "AscendCL",
            "API Name": "aclrtSynchronizeStream",
            "Time(us)": 200.0,
            "Count": synchronize_count,
        },
    ]
    _write_csv(path, rows)


def _case(tmp_path: Path) -> tuple[Path, Path]:
    report_path = tmp_path / "runner-report.json"
    _write_json(report_path, _runner_report())
    profile = tmp_path / "PROF_case" / "mindstudio_profiler_output"
    profile.mkdir(parents=True)
    _write_csv(profile / "op_summary_case.csv", _op_rows())
    _write_api_csv(profile / "api_statistic_case.csv")
    return report_path, profile.parent


def test_msprof_analysis_attributes_every_role_and_dynamic_gear(
    tmp_path: Path,
) -> None:
    report, profile = _case(tmp_path)
    payload = analyze_incremental_msprof(
        profile_dir=profile,
        runner_report=report,
    )

    assert payload["status"] == "PASS"
    assert payload["formal_latency_evidence"] is False
    assert payload["coverage"]["observed_model_executions"] == 8
    assert payload["coverage"]["observed_by_role"] == {
        "target-prefill": 2,
        "target-prefill-head": 1,
        "target-verify-commit": 3,
        "draft-propose": 2,
    }
    assert payload["by_role_and_physical_rows"][
        "target-verify-commit:T=1"
    ]["invocation_task_duration"]["count"] == 2
    assert payload["by_role_and_physical_rows"][
        "target-verify-commit:T=4"
    ]["invocation_task_duration"]["count"] == 1
    assert payload["by_role_and_physical_rows"][
        "draft-propose:T=4"
    ]["invocation_task_duration"]["count"] == 1
    assert payload["expected_draft_feature_signature"] == {
        "policy": "committed-prefix",
        "physical_verify_rows": 4,
        "full_width_equivalent_rows": 16,
        "elided_rows": 12,
        "fixed_width_executions": 0,
        "committed_prefix_executions": 1,
        "pending_upper_bound_executions": 0,
        "trace_gate": (
            "sum draft-propose T<=16 physical_rows equals "
            "draft_verify_feature_input_rows"
        ),
    }
    assert payload["api_count_gates"]["aclmdlExecuteAsync"] == {
        "status": "PASS",
        "expected": 8,
        "observed": 8,
    }
    assert payload["expected_memcpy_signature"]["host_to_device"] == {
        "operations": 3,
        "bytes": 1024,
    }
    assert payload["expected_synchronization_signature"] == {
        "prefill_completion_policy": "separate",
        "stream_synchronizations": 4,
        "speculative_transactions": 1,
        "speculative_sync_windows": 1,
        "speculative_synchronizations_elided": 0,
        "speculative_d2h_operations_elided": 0,
        "speculative_d2h_padding_bytes": 0,
        "speculative_window_staging_operations": 0,
        "speculative_window_staging_bytes": 0,
        "prefill_verify_coalesced_windows": 0,
        "prefill_verify_synchronizations_elided": 0,
        "prefill_verify_d2h_operations_elided": 0,
        "prefill_verify_d2h_padding_bytes": 0,
        "closure": (
            "speculative_sync_windows + "
            "speculative_synchronizations_elided + "
            "prefill_verify_coalesced_windows == "
            "speculative_transactions"
        ),
        "device_to_host_closure": (
            "device_to_host_operations + "
            "speculative_d2h_operations_elided + "
            "prefill_verify_d2h_operations_elided == "
            "prefill_completion_synchronizations + "
            "target_decode1_executions + speculative_transactions"
        ),
    }


def test_msprof_analysis_accepts_legacy_ids_from_filename(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "runner-report.json"
    _write_json(report_path, _runner_report())
    profile = tmp_path / "PROF_case" / "mindstudio_profiler_output"
    profile.mkdir(parents=True)
    rows_by_group: dict[tuple[int, int], list[dict[str, object]]] = {}
    for row in _op_rows():
        key = (int(row["Model ID"]), int(row["Infer ID"]))
        legacy = dict(row)
        legacy.pop("Model ID")
        legacy.pop("Infer ID")
        rows_by_group.setdefault(key, []).append(legacy)
    for (model_id, infer_id), rows in rows_by_group.items():
        _write_csv(
            profile / f"op_summary_0_{model_id}_{infer_id}.csv",
            rows,
        )
    _write_api_csv(profile / "api_statistic_case.csv")

    payload = analyze_incremental_msprof(
        profile_dir=profile.parent,
        runner_report=report_path,
    )
    assert payload["coverage"]["status"] == "PASS"
    assert len(payload["input_files"]["op_summary"]) == 8


def test_msprof_analysis_rejects_default_single_model_export(
    tmp_path: Path,
) -> None:
    report, profile = _case(tmp_path)
    op_summary = next(profile.rglob("op_summary_*.csv"))
    _write_csv(
        op_summary,
        [row for row in _op_rows() if int(row["Model ID"]) == 1],
    )
    with pytest.raises(MsprofAnalysisError, match="export every model ID"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report,
        )


def test_msprof_analysis_rejects_api_count_drift(tmp_path: Path) -> None:
    report, profile = _case(tmp_path)
    api_path = next(profile.rglob("api_statistic_*.csv"))
    _write_api_csv(api_path, execute_count=7)
    with pytest.raises(MsprofAnalysisError, match="aclmdlExecuteAsync"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report,
        )


def test_msprof_analysis_rejects_sync_window_counter_drift(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    report = _runner_report()
    report["execution_io_counters"][
        "speculative_synchronizations_elided"
    ] = 1
    _write_json(report_path, report)
    with pytest.raises(MsprofAnalysisError, match="do not close"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report_path,
        )


def test_msprof_analysis_accepts_schema_nine_staging_contract(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    _write_json(report_path, _schema_nine_runner_report())

    payload = analyze_incremental_msprof(
        profile_dir=profile,
        runner_report=report_path,
    )

    assert payload["status"] == "PASS"
    assert payload["expected_memcpy_signature"][
        "device_to_device_speculative_staging"
    ] == {"operations": 0, "bytes": 0}


def test_msprof_analysis_counts_extended_window_staging_memcpy(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    report = _schema_nine_runner_report()
    report["protocol"]["dflash_sync_window"] = 8
    counters = report["execution_io_counters"]
    counters.update(
        {
            "model_executions": 9,
            "target_decode1_executions": 0,
            "draft_propose_executions": 3,
            "target_verify_commit_executions": 3,
            "draft_verify_feature_input_rows": 16,
            "draft_verify_full_width_equivalent_rows": 32,
            "draft_verify_feature_rows_elided": 16,
            "draft_verify_committed_prefix_executions": 0,
            "draft_verify_pending_upper_bound_executions": 2,
            "device_to_host_operations": 2,
            "decode_id_device_compaction_operations": 0,
            "decode_id_device_compaction_bytes": 0,
            "stream_synchronizations": 2,
            "speculative_sync_windows": 1,
            "speculative_synchronizations_elided": 2,
            "speculative_d2h_operations_elided": 2,
            "speculative_d2h_padding_bytes": 120,
            "speculative_window_staging_operations": 3,
            "speculative_window_staging_bytes": 1356,
        }
    )
    report["profile_model_execution_trace"] = [
        {"ordinal": 0, "model_id": 1, "physical_rows": 64},
        {"ordinal": 1, "model_id": 1, "physical_rows": 64},
        {"ordinal": 2, "model_id": 2, "physical_rows": 1},
        {"ordinal": 3, "model_id": 3, "physical_rows": 128},
        {"ordinal": 4, "model_id": 4, "physical_rows": 8},
        {"ordinal": 5, "model_id": 3, "physical_rows": 8},
        {"ordinal": 6, "model_id": 4, "physical_rows": 8},
        {"ordinal": 7, "model_id": 3, "physical_rows": 8},
        {"ordinal": 8, "model_id": 4, "physical_rows": 5},
    ]
    _write_json(report_path, report)
    op_path = next(profile.rglob("op_summary_*.csv"))
    _write_csv(
        op_path,
        _op_rows_for_invocations(
            [
                (1, 1),
                (1, 2),
                (2, 1),
                (3, 1),
                (4, 1),
                (3, 2),
                (4, 2),
                (3, 3),
                (4, 3),
            ]
        ),
    )
    api_path = next(profile.rglob("api_statistic_*.csv"))
    _write_api_csv(
        api_path,
        execute_count=9,
        memcpy_count=8,
        synchronize_count=2,
    )

    payload = analyze_incremental_msprof(
        profile_dir=profile,
        runner_report=report_path,
    )

    assert payload["api_count_gates"]["aclrtMemcpyAsync"] == {
        "status": "PASS",
        "expected": 8,
        "observed": 8,
    }
    assert payload["expected_memcpy_signature"][
        "device_to_device_speculative_staging"
    ] == {"operations": 3, "bytes": 1356}


def test_msprof_analysis_rejects_staging_for_window_one(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    report = _schema_nine_runner_report()
    counters = report["execution_io_counters"]
    counters["speculative_window_staging_operations"] = 1
    counters["speculative_window_staging_bytes"] = 452
    _write_json(report_path, report)

    with pytest.raises(MsprofAnalysisError, match="do not close"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report_path,
        )


def test_msprof_analysis_accepts_prefill_first_verify_coalescing(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    report = _runner_report()
    report["protocol"]["prefill_completion_policy"] = (
        "coalesce-first-verify"
    )
    counters = report["execution_io_counters"]
    counters["speculative_sync_windows"] = 0
    counters["stream_synchronizations"] = 3
    counters["device_to_host_operations"] = 3
    counters["prefill_verify_coalesced_windows"] = 1
    counters["prefill_verify_synchronizations_elided"] = 1
    counters["prefill_verify_d2h_operations_elided"] = 1
    counters["prefill_verify_d2h_padding_bytes"] = 255
    counters["prefill_verify_prefill_slot0_windows"] = 1
    _write_json(report_path, report)
    api_path = next(profile.rglob("api_statistic_*.csv"))
    _write_api_csv(api_path, memcpy_count=7, synchronize_count=3)

    payload = analyze_incremental_msprof(
        profile_dir=profile,
        runner_report=report_path,
    )
    signature = payload["expected_synchronization_signature"]
    assert signature["prefill_completion_policy"] == "coalesce-first-verify"
    assert signature["prefill_verify_coalesced_windows"] == 1
    assert signature["prefill_verify_d2h_operations_elided"] == 1


def test_msprof_analysis_rejects_coalesced_d2h_padding_drift(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    report = _runner_report()
    report["execution_io_counters"][
        "speculative_d2h_padding_bytes"
    ] = 1
    _write_json(report_path, report)
    with pytest.raises(MsprofAnalysisError, match="do not close"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report_path,
        )


def test_msprof_analysis_rejects_formal_evidence_report(tmp_path: Path) -> None:
    report_path, profile = _case(tmp_path)
    report = _runner_report()
    report["protocol"]["kind"] = "evidence"
    report["protocol"]["formal_latency_evidence"] = True
    report["protocol"]["profile_model_execution_trace_enabled"] = False
    _write_json(report_path, report)
    with pytest.raises(MsprofAnalysisError, match="diagnostic profile"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report_path,
        )


def test_msprof_analysis_requires_memory_policy_for_schema_eight(
    tmp_path: Path,
) -> None:
    report_path, profile = _case(tmp_path)
    report = _runner_report()
    report["schema_version"] = 8
    _write_json(report_path, report)
    with pytest.raises(MsprofAnalysisError, match="memory allocation policy"):
        analyze_incremental_msprof(
            profile_dir=profile,
            runner_report=report_path,
        )
