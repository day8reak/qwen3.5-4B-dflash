"""CPU fixtures for diagnostic plumbing, not NPU operator accuracy."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qwen35_dflash.ascend310p import operator_diagnostics as diag
from qwen35_dflash.ascend310p import cli
from test_cpp_runtime_progress import _infer_cpp_args


@pytest.fixture(autouse=True)
def run_scope(tmp_path, monkeypatch):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("ASCEND_DUMP_PATH", raising=False)
    monkeypatch.delenv("NPU_COLLECT_PATH", raising=False)


def save(path, value):
    np.save(path, np.ascontiguousarray(value), allow_pickle=False)
    return path


def config(tmp_path):
    graph = tmp_path / "graph.json"
    graph.write_text(json.dumps({"name": "compiled_name", "graph": [{"op": [
        {"name": "gdr.0", "type": "GatedDeltaRuleMTP"},
        {"name": "gdr.1", "type": "GatedDeltaRuleMTP"},
        {"name": "norm.0", "type": "RmsNorm"},
    ]}]}))
    output = tmp_path / "acl.json"
    diag.prepare_dump_config([graph], op_types=["GatedDeltaRuleMTP"],
                             output=output, dump_dir=tmp_path / "dump")
    return output


def test_dump_selection_uses_node_names_not_types_and_supports_other_ops(tmp_path):
    output = config(tmp_path)
    assert diag.validate_acl_dump_config(output) == output
    payload = json.loads(output.read_text())
    assert payload["dump"]["dump_list"] == [{"layer": ["gdr.0"]}]
    diag.prepare_dump_config([tmp_path / "graph.json"], op_types=["RmsNorm"],
                             output=tmp_path / "norm.json", dump_dir=tmp_path / "norm")
    norm = json.loads((tmp_path / "norm.json").read_text())
    assert norm["dump"]["dump_list"] == [{"layer": ["norm.0"]}]
    assert norm["dump"]["dump_mode"] == "all"
    with pytest.raises(FileExistsError):
        diag.prepare_dump_config([tmp_path / "graph.json"], op_types=["RmsNorm"],
                                 output=tmp_path / "norm.json", dump_dir=tmp_path / "norm2")
    with pytest.raises(ValueError, match="no selected"):
        diag.prepare_dump_config([tmp_path / "graph.json"], op_types=["Unknown"],
                                 output=tmp_path / "empty.json", dump_dir=tmp_path / "empty")
    assert not (tmp_path / "empty").exists()


@pytest.mark.parametrize("fault", ["all", "empty", "path", "relative", "used", "watch", "profile", "env"])
def test_dump_config_rejects_ambiguous_or_unbounded_outputs(tmp_path, monkeypatch, fault):
    output = config(tmp_path)
    payload = json.loads(output.read_text())
    if fault == "all": payload["dump"]["dump_list"] = [{}]
    if fault == "empty": payload["dump"]["dump_list"] = []
    if fault == "path": payload["dump"]["dump_path"] = str(tmp_path.parent)
    if fault == "relative": payload["dump"]["dump_path"] = "dump"
    if fault == "used": (tmp_path / "dump" / "old").touch()
    if fault == "watch": payload["dump"]["dump_scene"] = "watcher"
    if fault == "profile": payload["profiler"] = {}
    if fault == "env": monkeypatch.setenv("ASCEND_DUMP_PATH", "/elsewhere")
    output.write_text(json.dumps(payload))
    with pytest.raises((ValueError, RuntimeError)):
        diag.validate_acl_dump_config(output)


def test_cli_dump_requires_explicit_config_and_defaults_off():
    args = cli.build_parser().parse_args(_infer_cpp_args())
    assert args.acl_dump_config is None
    args = cli.build_parser().parse_args(_infer_cpp_args("--diagnose-target-parity", "--acl-dump-config", "/x/acl.json"))
    assert args.acl_dump_config == Path("/x/acl.json")


def test_tensor_comparison_coordinates_signed_zero_and_nonfinite(tmp_path):
    a = np.zeros((2, 3, 4), np.float32)
    b = a.copy()
    b[1, 2, 3] = 0.25
    ref, got = save(tmp_path / "a.npy", a), save(tmp_path / "b.npy", b)
    result = diag.compare_tensor(ref, got)
    assert result["status"] == "FAIL"
    assert result["first_difference"]["coordinates"] == [1, 2, 3]
    assert result["max_finite_abs_error"] == .25
    assert diag.compare_tensor(ref, got, atol=.3)["status"] == "PASS"
    b[:] = -0.
    save(got, b)
    result = diag.compare_tensor(ref, got)
    assert result["numeric_different_elements"] == 0
    assert result["bitwise_different_elements"] == 24
    b[0, 0, 0] = np.nan
    save(got, b)
    result = diag.compare_tensor(got, got)
    assert result["status"] == "FAIL"
    assert result["actual_nonfinite"] == 1
    assert result["first_difference"]["actual"] is None
    json.dumps(result, allow_nan=False)


def test_integer_selector_and_large_token_ids_do_not_round_to_float(tmp_path):
    ref = save(tmp_path / "a.npy", np.array([2**60, -2**63], np.int64))
    got = save(tmp_path / "b.npy", np.array([2**60 + 1, 2**63 - 1], np.int64))
    result = diag.compare_tensor(ref, got, atol=1e30)
    assert result["outside_tolerance_elements"] == 2
    assert result["max_finite_abs_error"] == 2**64 - 1


def test_float64_error_overflow_is_reported_without_invalid_json(tmp_path):
    ref = save(tmp_path / "a.npy", np.array([-np.finfo(np.float64).max]))
    got = save(tmp_path / "b.npy", np.array([np.finfo(np.float64).max]))
    result = diag.compare_tensor(ref, got)
    assert result["status"] == "FAIL"
    assert result["finite_error_overflowed_float64"] is True
    assert result["max_finite_abs_error"] is None
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("fault", ["shape", "dtype", "object", "fortran", "empty_map"])
def test_no_implicit_layout_dtype_or_empty_mapping_pass(tmp_path, fault):
    ref = save(tmp_path / "a.npy", np.ones((2, 3), np.float32))
    got = tmp_path / "b.npy"
    if fault == "shape": np.save(got, np.ones((3, 2), np.float32))
    if fault == "dtype": np.save(got, np.ones((2, 3), np.float16))
    if fault == "object": np.save(got, np.array([object()]))
    if fault == "fortran": np.save(got, np.asfortranarray(np.ones((2, 3), np.float32)))
    if fault in ("shape", "dtype"):
        assert diag.compare_tensor(ref, got)["status"] == "FAIL"
    elif fault in ("object", "fortran"):
        with pytest.raises(ValueError): diag.compare_tensor(ref, got)
    else:
        path = tmp_path / "pairs.json"
        path.write_text('{"schema_version":1,"operators":[]}')
        with pytest.raises(ValueError):
            diag.compare_operators(path, tmp_path / "result.json")


def test_operator_order_explicit_inputs_and_missing_tensor_fail_closed(tmp_path):
    a = save(tmp_path / "a.npy", np.zeros(2, np.float16))
    b = save(tmp_path / "b.npy", np.ones(2, np.float16))
    good = {"name": "out", "reference": str(a), "actual": str(a)}
    bad = {**good, "actual": str(b)}
    mapping = tmp_path / "pairs.json"
    payload = {"schema_version": 1, "operators": [
        {"id": "z-first-in-causal-order", "inputs": [bad], "outputs": [good]},
        {"id": "a-later", "inputs": [good], "outputs": [bad]},
    ]}
    mapping.write_text(json.dumps(payload))
    result = diag.compare_operators(mapping, tmp_path / "cmp.json")
    assert result["first_failing_operator"] == "z-first-in-causal-order"
    assert result["operators"][0]["localization"] == "upstream-or-input-mapping"
    assert result["operators"][1]["mapped_inputs_bitwise_equal"] is True
    assert result["formal_latency_evidence"] is False
    payload["operators"][1]["outputs"][0]["actual"] = "missing.npy"
    mapping.write_text(json.dumps(payload))
    with pytest.raises(FileNotFoundError):
        diag.compare_operators(mapping, tmp_path / "missing.json")
    assert not (tmp_path / "missing.json").exists()


def mtp_case(tmp_path):
    shapes = {"query": ((1, 2, 1, 2), np.float16),
              "key": ((1, 2, 1, 2), np.float16),
              "value": ((1, 2, 1, 3), np.float16),
              "g": ((1, 2, 1), np.float32),
              "beta": ((1, 2, 1), np.float16),
              "initial_state": ((1, 2, 1, 2, 3), np.float32),
              "accepted_tokens": ((1,), np.int8),
              "core_attn": ((1, 2, 1, 3), np.float16),
              "last_recurrent_state": ((1, 2, 1, 2, 3), np.float32)}
    for name, (shape, dtype) in shapes.items():
        save(tmp_path / (name + ".npy"), np.zeros(shape, dtype))
    payload = {"schema_version": 1, "layout": "logical-ND", "op_type": "GatedDeltaRuleMTP",
               "provenance": {"source": "CPU fixture"},
               "attrs": dict(chunk_size=64, output_final_state=True, use_qk_l2norm_in_kernel=True),
               "inputs": {k: k + ".npy" for k in diag.MTP_INPUTS},
               "outputs": {k: k + ".npy" for k in diag.MTP_OUTPUTS}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload))
    return path, payload


@pytest.mark.parametrize("fault", [None, "selector", "missing_input", "dtype", "layout", "attrs"])
def test_mtp_case_requires_complete_exact_inputs(tmp_path, fault):
    path, payload = mtp_case(tmp_path)
    if fault == "selector": save(tmp_path / "accepted_tokens.npy", np.array([2], np.int8))
    if fault == "missing_input": del payload["inputs"]["g"]
    if fault == "dtype": save(tmp_path / "g.npy", np.zeros((1, 2, 1), np.float16))
    if fault == "layout": payload["layout"] = "FRACTAL_NZ"
    if fault == "attrs": del payload["attrs"]["chunk_size"]
    path.write_text(json.dumps(payload))
    if fault:
        with pytest.raises(ValueError): diag._mtp_case(path)
    else:
        _, arrays = diag._mtp_case(path)
        assert set(arrays) == set(diag.MTP_INPUTS)


@pytest.fixture
def cpu_transport_fixture(monkeypatch):
    # Fake ONLY the transport/runtime in this unit test. Production has no CPU route.
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(__version__="CPU-fixture"))
    monkeypatch.setattr(torch, "npu", SimpleNamespace(
        is_available=lambda: True, set_device=lambda x: None,
        get_device_name=lambda x: "CPU-FIXTURE-NOT-NPU", synchronize=lambda: None), raising=False)
    original_bind = diag._bind_argument
    monkeypatch.setattr(diag, "_bind_argument",
                        lambda value, parent, device, identities: original_bind(value, parent, "cpu", identities))
    monkeypatch.setattr(diag, "_require_npu_tensor", lambda tensor, device_id: None)


@pytest.mark.parametrize("op_name,overload", [("add", "Tensor"), ("mul", "Tensor"), ("mm", "default")])
def test_generic_registered_operator_replay_cpu_fixture_not_gdr_specific(tmp_path, cpu_transport_fixture, op_name, overload):
    a, b = np.arange(4, dtype=np.float32).reshape(2, 2), np.ones((2, 2), np.float32)
    golden = a + b if op_name == "add" else a * b if op_name == "mul" else a @ b
    save(tmp_path / "x.npy", a)
    save(tmp_path / "y.npy", b)
    save(tmp_path / "om.npy", golden)
    rhs = "mat2" if op_name == "mm" else "other"
    payload = {"schema_version": 1, "torch_op": "aten::" + op_name, "overload": overload,
               "layout": "logical-ND", "provenance": {"source": "CPU fixture"},
               "arguments": {"self": {"tensor": "x.npy"}, rhs: {"tensor": "y.npy"}},
               "outputs": {"result": {"index": [], "tensor": "om.npy"}}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload))
    result = diag.replay_operator(path, tmp_path / "replay")
    assert result["status"] == "PASS"
    assert result["formal_latency_evidence"] is False
    assert np.array_equal(np.load(tmp_path / "x.npy"), a)


@pytest.mark.parametrize("fault", [None, "missing_output", "duplicate_output", "unknown_argument", "missing_argument"])
def test_generic_multi_output_operator_requires_complete_mapping(tmp_path, cpu_transport_fixture, fault):
    x = np.array([[3, 1], [1, 2]], np.float32)
    save(tmp_path / "x.npy", x)
    save(tmp_path / "values.npy", np.sort(x, axis=-1))
    save(tmp_path / "indices.npy", np.argsort(x, axis=-1).astype(np.int64))
    payload = {"schema_version": 1, "torch_op": "aten::sort", "overload": "default",
               "layout": "logical-ND", "provenance": {"source": "CPU fixture"},
               "arguments": {"self": {"tensor": "x.npy"}, "dim": {"value": -1},
                             "descending": {"value": False}},
               "outputs": {"values": {"index": [0], "tensor": "values.npy"},
                           "indices": {"index": [1], "tensor": "indices.npy"}}}
    if fault == "missing_output": del payload["outputs"]["indices"]
    if fault == "duplicate_output": payload["outputs"]["indices"]["index"] = [0]
    if fault == "unknown_argument": payload["arguments"]["invented"] = {"value": True}
    if fault == "missing_argument": del payload["arguments"]["self"]
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload))
    if fault:
        with pytest.raises(ValueError): diag.replay_operator(path, tmp_path / "replay")
        assert not (tmp_path / "replay" / "comparison.json").exists()
    else:
        assert diag.replay_operator(path, tmp_path / "replay")["status"] == "PASS"


def test_generic_cast_uses_explicit_dispatcher_dtype(tmp_path, cpu_transport_fixture):
    x = np.array([0.01, 1.01], np.float32)
    save(tmp_path / "x.npy", x)
    save(tmp_path / "om.npy", x.astype(np.float16))
    payload = {"schema_version": 1, "torch_op": "aten::_to_copy", "overload": "default",
               "layout": "logical-ND", "provenance": {"source": "CPU fixture"},
               "arguments": {"self": {"tensor": "x.npy"}, "dtype": {"dtype": "float16"}},
               "outputs": {"y": {"index": [], "tensor": "om.npy"}}}
    path = tmp_path / "case.json"
    path.write_text(json.dumps(payload))
    assert diag.replay_operator(path, tmp_path / "replay")["status"] == "PASS"


def test_native_replay_refuses_absent_device_before_output_creation(tmp_path, monkeypatch):
    path, _ = mtp_case(tmp_path)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=lambda: False), raising=False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        diag.replay_gdr_mtp(path, tmp_path / "replay")
    assert not (tmp_path / "replay").exists()


def test_native_tensor_check_rejects_cpu_outputs():
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        diag._require_npu_tensor(torch.zeros(1), 0)
