"""Exact runtime lifecycle tests; fabricated reports are not NPU evidence."""
import pytest

from qwen35_dflash.ascend310p.cpp_runtime import _runtime_identity
from test_incremental_cpp_runtime import _static_fused_report, _validate


def _phase_report():
    report = _static_fused_report()
    memory = report["model_memory_query"]
    allocated = max(m["weight_bytes"] for m in report["models"])
    saved = memory["sum_weight_bytes"] - allocated
    memory.update(
        allocated_weight_bytes=allocated, weight_bytes_elided=saved,
        load_policy=(
            "phase-resident static fused; one live OM; synchronized unload before "
            "reusing max-sized workspace and weight arena; hot decode stays loaded"
        ),
    )
    memory["explicit_allocated_device_bytes_excluding_runtime"] -= saved
    report["abi"]["physical_topology"] = "split-prefill-head-four-artifact-phase-resident-fused-v1"
    report["protocol"]["model_load_excluded_from_latency"] = False
    # Synthetic lifetime counts within this fixture's real-call upper bound.
    report["model_residency"] = dict(
        policy="phase-resident", model_id_scope="startup-metadata-inspection",
        peak_resident_models=1, model_loads=82, model_unloads=81,
        model_switches=78, model_switch_synchronizations=52,
        model_switch_wall_ms=10.0,
        timing_scope=(
            "model switch synchronization/unload/load/ABI validation is included "
            "in generation and benchmark wall times; startup inspection is separate"
        ),
    )
    report["execution_io_counters"]["stream_synchronizations"] += 52
    return report


def _check(report):
    _validate(report, fused_speculative_step=True, fused_static_feature_rows=64,
              model_residency_policy="phase-resident")


def test_phase_residency_report_closes_memory_and_additional_synchronizations():
    _check(_phase_report())


def test_phase_inspection_ids_may_be_reused_after_unload_not_live_resident_ids():
    report = _phase_report()
    for model in report["models"]:
        model["model_id"] = 2147483648
    _check(report)
    resident = _static_fused_report()
    for model in resident["models"]:
        model["model_id"] = 2147483648
    with pytest.raises(RuntimeError, match="model IDs"):
        _validate(resident, fused_speculative_step=True, fused_static_feature_rows=64)


@pytest.mark.parametrize("field,value", [
    ("policy", "all-resident"), ("model_id_scope", "resident-model"),
    ("peak_resident_models", 2), ("model_loads", 83), ("model_unloads", 0),
    ("model_switches", 0), ("model_switches", 99999),
    ("model_switch_synchronizations", 79), ("model_switch_synchronizations", True),
    ("model_switch_wall_ms", -1), ("model_switch_wall_ms", float("nan")),
    ("model_switch_wall_ms", float("inf")), ("timing_scope", "steady-state only"),
])
def test_phase_report_rejects_invalid_lifetimes_and_timings(field, value):
    report = _phase_report()
    report["model_residency"][field] = value
    with pytest.raises(RuntimeError, match="residency|phase-resident"):
        _check(report)


@pytest.mark.parametrize("damage", ["missing", "sum-weights", "elided", "total", "sync", "timing"])
def test_phase_report_rejects_old_memory_or_timing_accounting(damage):
    report = _phase_report()
    memory = report["model_memory_query"]
    if damage == "missing":
        del report["model_residency"]
    elif damage == "sum-weights":
        memory["allocated_weight_bytes"] = memory["sum_weight_bytes"]
    elif damage == "elided":
        memory["weight_bytes_elided"] = 0
    elif damage == "total":
        memory["explicit_allocated_device_bytes_excluding_runtime"] += 1
    elif damage == "sync":
        report["execution_io_counters"]["stream_synchronizations"] -= 52
    else:
        report["protocol"]["model_load_excluded_from_latency"] = True
    with pytest.raises(RuntimeError, match="residency|phase-resident|allocation|synchronization"):
        _check(report)


def test_residency_is_explicit_and_cannot_relabel_an_old_report():
    with pytest.raises(RuntimeError, match="topology"):
        _validate(_phase_report(), fused_speculative_step=True, fused_static_feature_rows=64)
    with pytest.raises(RuntimeError, match="timing"):
        _check(_static_fused_report())
    with pytest.raises(RuntimeError, match="static fused"):
        _validate(_phase_report(), fused_speculative_step=True, model_residency_policy="phase-resident")


@pytest.mark.parametrize("state,residency,error", [
    ("incremental-explicit-state-v2", "phase-resident", None),
    ("incremental-explicit-state-v2", "all-resident", None),
    ("incremental-explicit-state-v2", "typo", "model_residency_policy"),
    ("recompute-committed-prefixes", "phase-resident", "incremental static fused"),
])
def test_residency_config_admission(state, residency, error):
    options = dict(device_model="Ascend310P3", cann="test", driver="test",
                   firmware="test", runtime="fake", state_policy=state,
                   model_residency_policy=residency)
    if error:
        with pytest.raises(ValueError, match=error):
            _runtime_identity(options, 0)
    else:
        assert _runtime_identity(options, 0)["model_residency_policy"] == residency
