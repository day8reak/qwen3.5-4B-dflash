"""CPU metadata/graph-capture checks; these do not execute receiver kernels."""
from __future__ import annotations

import copy
import importlib
import importlib.machinery
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from qwen35_dflash.ascend310p import custom_op_export as custom
from qwen35_dflash.ascend310p.custom_op_export import _validate_npu_quant_matmul_meta
from qwen35_dflash.ascend310p.custom_op_export import (
    ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
    ADN_FUSED_INFER_ATTENTION_TORCH_OP,
    ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
    ADN_RMS_NORM_TORCH_OP,
    FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP,
    FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
    NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
    NPU_CACHE_UPDATE_TORCH_OP,
    NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
    NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
    NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
    NPU_DYNAMIC_QUANT_TORCH_OP,
    NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
    NPU_QUANT_MATMUL_TORCH_OP,
    NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE,
    NPU_SCATTER_ND_UPDATE_TORCH_OP,
    CustomOpExportSession,
    audit_custom_op_export,
    prepare_custom_op_export,
    validate_adn_attention_ge_prototype_environment,
    validate_gdr_ge_prototype_environment,
)
from qwen35_dflash.ascend310p.standard_op_export import (
    prepare_aten_softplus_export, audit_aten_softplus_export,
    ATEN_SOFTPLUS_TORCH_TARGET, ATEN_SOFTPLUS_GE_OP_TYPE,
)
from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
from qwen35_dflash.ascend310p.compiler import _validated_custom_op_audit
from qwen35_dflash.ascend310p.quant_factory import (
    _target_custom_op_exports, _enable_target_quant_matmul_export_mode,
)
from qwen35_dflash.ascend310p.exporter import export_air_bundle

@pytest.mark.parametrize("ge_type,input_name", [("RmsNorm", "x"), ("AdnRmsNorm", "self")])
def test_rms_named_input_matches_selected_ge_type(ge_type, input_name):
    op = _ensure_target_test_schema("adn_rms_norm")
    ta = _FakeTorchAir()
    prepare_custom_op_export(CustomOpExportSpec(ADN_RMS_NORM_TORCH_OP, ge_type), ta)
    x, gamma = object(), object()
    ta.converters[op](x, gamma)
    assert ta.ge.calls[-1][2]["inputs"] == {input_name: x, "gamma": gamma}


def _quant_incremental_specs(monkeypatch):
    from test_incremental_air_om import TinyTarget, draft_model, rotary
    from qwen35_dflash.ascend310p.incremental import incremental_graph_specs

    operations = _ensure_all_target_test_schemas()
    module = ModuleType("torch_npu")
    module.__spec__ = importlib.machinery.ModuleSpec("torch_npu", loader=None)
    for name, op in operations.items():
        setattr(module, name, op)
    monkeypatch.setitem(sys.modules, "torch_npu", module)
    modeling = importlib.import_module("models.modeling_qwen3_5_hiai_nd")
    monkeypatch.setattr(modeling, "torch_npu", module)
    target = TinyTarget().eval()
    body = target.dflash_execution_model.language_model
    # Exercise production QLinear and RMSNorm inside the actual incremental
    # graph classes. Tiny weights limit this evidence to graph contracts.
    for parent in list(body.modules()):
        for name, layer in list(parent.named_children()):
            if isinstance(layer, nn.Linear):
                setattr(parent, name, modeling.QLinear(
                    torch.ones(layer.in_features, layer.out_features, dtype=torch.int8),
                    torch.ones(layer.out_features, dtype=torch.float32), 0))
    for layer in body.layers:
        layer.input_layernorm = modeling.Qwen3_5RMSNorm(32)
        layer.post_attention_layernorm = modeling.Qwen3_5RMSNorm(32)
        if layer.block_type == "linear_attention":
            layer.linear_attn.norm = modeling.Qwen3_5RMSNormGated(16)
        else:
            layer.self_attn.q_norm = modeling.Qwen3_5RMSNorm(16)
            layer.self_attn.k_norm = modeling.Qwen3_5RMSNorm(16)
    body.norm = modeling.Qwen3_5RMSNorm(32)
    assert _enable_target_quant_matmul_export_mode(target) == 13
    contracts = _target_custom_op_exports({}, incremental=True)
    metadata = {
        "custom_op_export_contracts": [
            {"torch_target": op.torch_target, "ge_op_type": op.ge_op_type,
             "minimum_occurrences": op.minimum_occurrences} for op in contracts],
        "standard_op_export_contracts": [
            {"torch_target": "aten.softplus.default", "ge_op_type": "SoftplusV2",
             "minimum_occurrences": 1}],
    }
    return incremental_graph_specs(
        target, draft_model(), capacity=128, metadata=metadata,
        gdr=operations["npu_chunk_gated_delta_rule"],
        attention=operations["adn_fused_infer_attention"], rotary=rotary,
        custom_ops=contracts, include_ordinary_decode=True)


def test_four_incremental_graphs_capture_and_audit_every_custom_op(tmp_path, monkeypatch):
    specs = _quant_incremental_specs(monkeypatch)

    class CaptureTorchAir(_FakeTorchAir):
        def __init__(self):
            super().__init__()
            self.captures = {}

        def dynamo_export(self, *args, model, export_path, export_name, dynamic):
            assert not dynamic
            exported = torch.export.export(model, args, strict=True)
            # Also functionalize using the AOT path: missing alias support must
            # fail here, before a receiver attempts GE conversion.
            exported = exported.run_decompositions({})
            self.captures[export_name] = exported
            start = len(self.ge.calls)
            for node in exported.graph.nodes:
                if node.op != "call_function":
                    continue
                # FakeTensor/AOT success alone does not prove TorchAir support.
                # These standard-op converters raise NotImplementedError.
                if str(node.target) in {
                    "aten.unfold.default", "aten.unfold_copy.default",
                    "aten.index_copy.default",
                    "aten.amin.default", "aten.min.dim", "aten.cumprod.default",
                }:
                    raise AssertionError(f"unsupported AIR operator: {node.target}")
                converter = self.converters.get(node.target)
                if converter is not None:
                    converter(*node.args, **node.kwargs)
                elif node.target is torch.ops.npu.npu_dynamic_quant.default:
                    self.ge.custom_op("DynamicQuant", outputs=["y", "scale"])
                elif str(node.target).startswith(("npu.", "qwen35_dflash.")):
                    raise AssertionError(f"uncovered custom operator: {node.target}")
            root = Path(export_path)
            (root / (export_name + ".air")).write_bytes(b"host-fixture-not-real-air")
            (root / "dynamo.pbtxt").write_text("".join(
                'op { op: "' + name + '" }\n'
                for name, _, _ in self.ge.calls[start:]))

    ta = CaptureTorchAir()
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    manifest = export_air_bundle(lambda config: specs, {}, tmp_path / "bundle",
                                 torchair_module=ta)
    assert len(manifest["graphs"]) == 4
    for graph in manifest["graphs"]:
        name = graph["name"]
        audit = _validated_custom_op_audit(graph)
        exported = ta.captures[name]
        nodes = list(exported.graph.nodes)
        cache_writes = [node for node in nodes if str(node.target) == "aten.scatter.src"]
        assert len(cache_writes) == (4 if name == "draft" else 2)
        # Cache indices use static Tile repeats, matching the receiver-tested
        # quant branch and avoiding dynamic BroadcastTo auto-tiling failures.
        assert all(str(node.args[2].target) == "aten.repeat.default" for node in cache_writes)
        scans = [node for node in nodes if str(node.target) == "aten.cumsum.default"]
        assert len(scans) == (1 if name == "target_verify" else 0)
        assert all(node.meta["val"].dtype == torch.int32 for node in scans)
        calls = [node for node in nodes if str(node.target) == "npu.npu_chunk_gated_delta_rule.default"]
        if name == "draft":
            assert audit == [] and not calls
            assert "custom_op_export_contracts" not in graph["metadata"]
            assert graph["standard_op_overrides"] == []
            # Each of the two fixture layers repeats both K and V heads.
            head_repeats = [node for node in nodes if str(node.target) == "aten.repeat.default"
                            and node.meta["val"].ndim == 5]
            assert len(head_repeats) == 4
        else:
            assert len(audit) == 5
            assert len(calls) == (2 if name == "target_verify" else 1)
            rows = 1 if name == "target_decode" else 16 if name == "target_verify" else 64
            assert all(node.args[0].meta["val"].shape[1] == rows for node in calls)
            assert graph["standard_op_overrides"][0]["ge_node_occurrences"] == 1
            attention = next(node for node in nodes
                             if str(node.target) == "npu.adn_fused_infer_attention.default")
            assert attention.kwargs["pse_shift"].meta["val"].dtype == torch.int64
            quant_nodes = [node for node in nodes if str(node.target) ==
                           "qwen35_dflash.npu_quant_matmul_v4444.default"]
            assert len(quant_nodes) == 13
            assert all(node.args[2].meta["val"].dtype == torch.float32 for node in quant_nodes)
            assert all(node.meta["val"].dtype == torch.float16 for node in quant_nodes)


def test_production_preflight_checks_custom_ops_before_loading_weights(tmp_path, monkeypatch):
    from qwen35_dflash.ascend310p import quant_factory

    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    checked = []
    def missing(spec, torchair):
        checked.append(spec.torch_target)
        raise RuntimeError("receiver schema missing")
    monkeypatch.setattr(quant_factory, "prepare_custom_op_export", missing)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    # Empty configuration would fail factory validation if checkpoint preparation
    # were reached. The missing dispatcher must fail first.
    with pytest.raises(RuntimeError, match="receiver schema missing"):
        export_air_bundle(quant_factory.create_quant_incremental_graphs, {},
                          tmp_path / "bundle", torchair_module=_FakeTorchAir())
    assert checked == ["npu.adn_rms_norm.default"]


@pytest.mark.parametrize("field,value", [
    ("status", "FAIL"), ("ge_node_occurrences", 0),
    ("converter_calls", 2), ("ge_op_type", "Softplus"),
])
def test_compiler_rejects_invalid_softplus_evidence(field, value):
    from qwen35_dflash.ascend310p.compiler import _validated_standard_op_overrides
    contract = {"torch_target": "aten.softplus.default", "ge_op_type": "SoftplusV2",
                "minimum_occurrences": 1}
    graph = {"metadata": {"standard_op_export_contracts": [contract]},
             "standard_op_overrides": [{**contract, "status": "PASS",
                 "converter_policy": "framework-registered-ge-ir",
                 "converter_calls": 1, "ge_node_occurrences": 1}]}
    assert len(_validated_standard_op_overrides(graph)) == 1
    graph["standard_op_overrides"][0][field] = value
    with pytest.raises(ValueError, match="SoftplusV2"):
        _validated_standard_op_overrides(graph)


def test_compiler_rejects_missing_softplus_evidence():
    from qwen35_dflash.ascend310p.compiler import _validated_standard_op_overrides
    with pytest.raises(ValueError, match="SoftplusV2"):
        _validated_standard_op_overrides({"metadata": {
            "standard_op_export_contracts": [{"torch_target": "aten.softplus.default",
                "ge_op_type": "SoftplusV2", "minimum_occurrences": 1}]}})


_TEST_OPERATOR_LIBRARIES: list[torch.library.Library] = []

_TARGET_TEST_SCHEMAS = {
    "adn_fused_infer_attention": (
        "adn_fused_infer_attention("
        "Tensor query, Tensor[] key, Tensor[] value, *, "
        "Tensor? pse_shift=None, Tensor? atten_mask=None, "
        "SymInt[]? all_seq_lengths_q=None, "
        "SymInt[]? actual_seq_lengths_q=None, "
        "SymInt[]? actual_seq_lengths_kv=None, "
        "Tensor? dequant_scale1=None, Tensor? quant_scale1=None, "
        "Tensor? dequant_scale2=None, Tensor? quant_scale2=None, "
        "Tensor? quant_offset2=None, Tensor? antiquant_scale=None, "
        "Tensor? antiquant_offset=None, Tensor? block_table=None, "
        "Tensor? kv_padding_size=None, int num_heads=1, "
        'float scale_value=1., str input_layout="BSH", '
        "int num_key_value_heads=0, int block_size=0, int inner_precise=1"
        ") -> Tensor"
    ),
    "adn_rms_norm": (
        "adn_rms_norm(Tensor input, Tensor gamma, float epsilon=1e-6) "
        "-> (Tensor, Tensor)"
    ),
    "npu_cache_update_": (
        "npu_cache_update_(Tensor(a!) input, Tensor updates, "
        "Tensor target_block, Tensor offset_in_block) -> Tensor(a!)"
    ),
    "npu_chunk_gated_delta_rule": (
        "npu_chunk_gated_delta_rule("
        "Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, "
        "Tensor effective_length, int chunk_size=64, "
        "Tensor? initial_state=None, bool output_final_state=False, "
        "bool use_qk_l2norm_in_kernel=False) -> (Tensor, Tensor)"
    ),
    "npu_dynamic_quant": (
        "npu_dynamic_quant(Tensor input, *, Tensor? smooth_scales=None, "
        "Tensor? group_index=None, ScalarType? dst_type=None) "
        "-> (Tensor, Tensor)"
    ),
    "npu_quant_matmul": (
        "npu_quant_matmul(Tensor x1, Tensor x2, Tensor scale, *, "
        "Tensor? offset=None, Tensor? pertoken_scale=None, Tensor? bias=None, "
        "ScalarType? output_dtype=None, SymInt[]? group_sizes=None) -> Tensor"
    ),
    "npu_scatter_nd_update_": (
        "npu_scatter_nd_update_(Tensor(a!) input, Tensor indices, "
        "Tensor updates) -> Tensor(a!)"
    ),
}


def _ensure_target_test_schema(name: str) -> object:
    try:
        return getattr(getattr(torch.ops.npu, name), "default")
    except AttributeError:
        library = torch.library.Library("npu", "FRAGMENT")
        library.define(_TARGET_TEST_SCHEMAS[name])
        _TEST_OPERATOR_LIBRARIES.append(library)
        return getattr(getattr(torch.ops.npu, name), "default")


def _ensure_all_target_test_schemas() -> dict[str, object]:
    return {
        name: _ensure_target_test_schema(name)
        for name in _TARGET_TEST_SCHEMAS
    }


def _ensure_cache_update_cpu_impl() -> None:
    if torch._C._dispatch_has_kernel_for_dispatch_key(
        NPU_CACHE_UPDATE_TORCH_OP, "CPU"
    ):
        return
    library = torch.library.Library("npu", "IMPL", "CPU")

    def cache_update(
        input: torch.Tensor,
        updates: torch.Tensor,
        target_block: torch.Tensor,
        offset_in_block: torch.Tensor,
    ) -> torch.Tensor:
        del target_block, offset_in_block
        input.copy_(updates)
        return input

    library.impl("npu_cache_update_", cache_update)
    _TEST_OPERATOR_LIBRARIES.append(library)


class _FakeTorchAirGeAttr:
    @staticmethod
    def Float(value: float) -> tuple[str, float]:
        return ("float", value)

    @staticmethod
    def Int(value: int) -> tuple[str, int]:
        return ("int", value)

    @staticmethod
    def Bool(value: bool) -> tuple[str, bool]:
        return ("bool", value)

    @staticmethod
    def Str(value: str) -> tuple[str, str]:
        return ("str", value)


class _FakeTorchAirGeDataType:
    DT_FLOAT16 = 1
    DT_INT64 = "DT_INT64"


class _FakeTorchAirGe:
    attr = _FakeTorchAirGeAttr()
    DataType = _FakeTorchAirGeDataType()

    def __init__(self) -> None:
        self.calls: list[
            tuple[str, tuple[object, ...], dict[str, object]]
        ] = []

    @staticmethod
    def Const(value: object, *, dtype: object) -> tuple[str, object, object]:
        return ("const", value, dtype)

    def custom_op(self, op_type: str, **kwargs: object):
        self.calls.append((op_type, (), dict(kwargs)))
        output_count = len(kwargs.get("outputs", ()))
        if output_count:
            values = tuple(object() for _ in range(output_count))
            return values[0] if output_count == 1 else values
        if op_type in {
            ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
            NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
        }:
            return object(), object()
        return object()


def _write_gdr_ge_prototype(root: Path, *, effective_length: bool) -> Path:
    header = root / "op_proto" / "inc" / "op_proto.h"
    header.parent.mkdir(parents=True)
    effective_input = (
        "    .INPUT(effective_length, ge::TensorType::ALL())\n"
        if effective_length
        else ""
    )
    header.write_text(
        "REG_OP(ChunkGatedDeltaRule)\n"
        "    .INPUT(query, ge::TensorType::ALL())\n"
        "    .INPUT(key, ge::TensorType::ALL())\n"
        "    .INPUT(value, ge::TensorType::ALL())\n"
        "    .INPUT(g, ge::TensorType::ALL())\n"
        "    .INPUT(beta, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(initial_state, ge::TensorType::ALL())\n"
        f"{effective_input}"
        "    .OUTPUT(core_attn, ge::TensorType::ALL())\n"
        "    .OUTPUT(last_recurrent_state, ge::TensorType::ALL())\n"
        "    .ATTR(chunk_size, Int, 64)\n"
        "    .ATTR(output_final_state, Bool, false)\n"
        "    .ATTR(use_qk_l2norm_in_kernel, Bool, false)\n"
        "    .OP_END_FACTORY_REG(ChunkGatedDeltaRule);\n",
        encoding="utf-8",
    )
    return header


def _write_adn_attention_ge_package(
    root: Path,
    *,
    include_actual_q_back: bool = True,
    include_kernel: bool = True,
) -> Path:
    header = (
        root
        / "op_proto"
        / "inc"
        / "adn_fused_infer_attention_proto.h"
    )
    header.parent.mkdir(parents=True)
    actual_q_back = (
        "    .OPTIONAL_INPUT(actual_seq_lengths_q_back, "
        "ge::TensorType::ALL())\n"
        if include_actual_q_back
        else ""
    )
    header.write_text(
        "REG_OP(AdnFusedInferAttention)\n"
        "    .INPUT(query, ge::TensorType::ALL())\n"
        "    .DYNAMIC_INPUT(key, ge::TensorType::ALL())\n"
        "    .DYNAMIC_INPUT(value, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(pse_shift, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(atten_mask, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(actual_seq_lengths_q, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(actual_seq_lengths_kv, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(dequant_scale1, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(quant_scale1, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(dequant_scale2, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(quant_scale2, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(quant_offset2, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(antiquant_scale, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(antiquant_offset, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(block_table, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(kv_padding_size, ge::TensorType::ALL())\n"
        "    .OPTIONAL_INPUT(all_seq_lengths_q, ge::TensorType::ALL())\n"
        f"{actual_q_back}"
        "    .OUTPUT(attention_out, ge::TensorType::ALL())\n"
        "    .REQUIRED_ATTR(num_heads, Int)\n"
        "    .ATTR(scale_value, Float, 1)\n"
        '    .ATTR(input_layout, String, "BSH")\n'
        "    .ATTR(num_key_value_heads, Int, 0)\n"
        "    .ATTR(block_size, Int, 0)\n"
        "    .ATTR(inner_precise, Int, 1)\n"
        "    .OP_END_FACTORY_REG(AdnFusedInferAttention);\n",
        encoding="utf-8",
    )
    if include_kernel:
        kernel_dir = (
            root
            / "op_impl"
            / "ai_core"
            / "tbe"
            / "kernel"
            / "ascend310p"
            / "adn_fused_infer_attention"
        )
        kernel_dir.mkdir(parents=True)
        (kernel_dir / "AdnFusedInferAttention_test.o").write_bytes(b"kernel")
        (kernel_dir / "AdnFusedInferAttention_test.json").write_text(
            "{}",
            encoding="utf-8",
        )
    return header


class _FakeTorchAir:
    __version__ = "test"
    def __init__(self):
        self.ge = _FakeTorchAirGe()
        self.converters = {}
    def register_fx_node_ge_converter(self, operation):
        def register(converter):
            self.converters[operation] = converter
            return converter
        return register

def test_adn_attention_ge_preflight_accepts_receiver_310p_package(
    tmp_path: Path,
) -> None:
    vendor = tmp_path / "adn"
    header = _write_adn_attention_ge_package(vendor)

    result = validate_adn_attention_ge_prototype_environment(
        ascend_custom_opp_path=str(vendor),
        ld_library_path=str(vendor / "op_api" / "lib"),
    )

    assert result["status"] == "PASS"
    assert result["ge_op_type"] == "AdnFusedInferAttention"
    assert result["abi"] == "receiver-adn-attention-v1-named-inputs"
    assert result["prototype_path"] == str(header.resolve())
    assert len(result["prototype_sha256"]) == 64
    assert result["kernel_object_count"] == 1
    assert result["kernel_metadata_count"] == 1
    assert result["environment_sources"] == [
        "ASCEND_CUSTOM_OPP_PATH",
        "LD_LIBRARY_PATH",
    ]


def test_adn_attention_ge_preflight_rejects_missing_active_package(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="absent from the active"):
        validate_adn_attention_ge_prototype_environment(
            ascend_custom_opp_path=str(tmp_path / "other_vendor"),
            ld_library_path="",
        )


def test_adn_attention_ge_preflight_rejects_schema_or_kernel_drift(
    tmp_path: Path,
) -> None:
    stale = tmp_path / "stale_adn"
    _write_adn_attention_ge_package(stale, include_actual_q_back=False)
    with pytest.raises(RuntimeError, match="incompatible AdnFusedInferAttention"):
        validate_adn_attention_ge_prototype_environment(
            ascend_custom_opp_path=str(stale),
            ld_library_path="",
        )

    missing_kernel = tmp_path / "missing_kernel"
    _write_adn_attention_ge_package(missing_kernel, include_kernel=False)
    with pytest.raises(RuntimeError, match="prebuilt .o/.json kernel pair"):
        validate_adn_attention_ge_prototype_environment(
            ascend_custom_opp_path=str(missing_kernel),
            ld_library_path="",
        )


def test_gdr_ge_prototype_preflight_accepts_one_effective_length_v2(
    tmp_path: Path,
) -> None:
    vendor = tmp_path / "current_gdr"
    header = _write_gdr_ge_prototype(vendor, effective_length=True)

    result = validate_gdr_ge_prototype_environment(
        ascend_custom_opp_path=str(vendor),
        ld_library_path=str(vendor / "op_api" / "lib"),
    )

    assert result["status"] == "PASS"
    assert result["abi"] == "effective-length-v2-named-inputs"
    assert result["prototype_path"] == str(header.resolve())
    assert len(result["prototype_sha256"]) == 64
    assert result["environment_sources"] == [
        "ASCEND_CUSTOM_OPP_PATH",
        "LD_LIBRARY_PATH",
    ]


def test_gdr_ge_prototype_preflight_rejects_duplicate_vendor_roots(
    tmp_path: Path,
) -> None:
    current = tmp_path / "current_gdr"
    legacy = tmp_path / "legacy_gdr"
    _write_gdr_ge_prototype(current, effective_length=True)
    _write_gdr_ge_prototype(legacy, effective_length=False)

    with pytest.raises(RuntimeError, match="multiple ChunkGatedDeltaRule"):
        validate_gdr_ge_prototype_environment(
            ascend_custom_opp_path=os.pathsep.join((str(current), str(legacy))),
            ld_library_path="",
        )


def test_gdr_ge_prototype_preflight_rejects_legacy_abi(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "legacy_gdr"
    _write_gdr_ge_prototype(legacy, effective_length=False)

    with pytest.raises(RuntimeError, match="incompatible ChunkGatedDeltaRule"):
        validate_gdr_ge_prototype_environment(
            ascend_custom_opp_path=str(legacy),
            ld_library_path="",
        )


def test_aten_softplus_converter_emits_one_softplus_v2_node() -> None:
    torchair = _FakeTorchAir()
    session = prepare_aten_softplus_export(torchair)
    converter = torchair.converters[torch.ops.aten.softplus.default]
    input_node = object()

    output = converter(
        input_node,
        beta=1,
        threshold=20,
        meta_outputs=object(),
    )

    assert output is not None
    assert session.torch_target == ATEN_SOFTPLUS_TORCH_TARGET
    assert session.ge_op_type == ATEN_SOFTPLUS_GE_OP_TYPE
    assert session.converter_calls == 1
    assert torchair.ge.calls == [
        (
            ATEN_SOFTPLUS_GE_OP_TYPE,
            (),
            {
                "inputs": {"x": input_node},
                "outputs": ["y"],
                "attrs": {
                    "beta": ("float", 1.0),
                    "threshold": ("float", 20.0),
                },
            },
        )
    ]


@pytest.mark.parametrize("argument", [torch.tensor(1.0), True, float("inf")])
def test_aten_softplus_converter_rejects_non_static_finite_attributes(
    argument: object,
) -> None:
    torchair = _FakeTorchAir()
    prepare_aten_softplus_export(torchair)
    converter = torchair.converters[torch.ops.aten.softplus.default]

    with pytest.raises((TypeError, ValueError), match="compile-time|finite"):
        converter(object(), beta=argument)


def test_aten_softplus_audit_rejects_missing_ge_node(tmp_path: Path) -> None:
    torchair = _FakeTorchAir()
    session = prepare_aten_softplus_export(torchair)
    converter = torchair.converters[torch.ops.aten.softplus.default]
    converter(object())
    graph_dir = tmp_path / "air" / "missing"
    graph_dir.mkdir(parents=True)
    (graph_dir / "dynamo.pbtxt").write_text(
        'op {\n  name: "wrong"\n  type: "Exp"\n}\n',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="SoftplusV2 nodes"):
        audit_aten_softplus_export(
            session,
            graph_dir,
            calls_before=0,
            relative_to=tmp_path,
        )


def test_aten_softplus_audit_accepts_op_field_without_double_counting(
    tmp_path: Path,
) -> None:
    torchair = _FakeTorchAir()
    session = prepare_aten_softplus_export(torchair)
    converter = torchair.converters[torch.ops.aten.softplus.default]
    converter(object())
    converter(object())
    graph_dir = tmp_path / "air" / "op-field"
    graph_dir.mkdir(parents=True)
    (graph_dir / "dynamo.pbtxt").write_text(
        'op {\n  name: "softplus_0"\n  op: "SoftplusV2"\n}\n'
        'op {\n  name: "softplus_1"\n  op: "SoftplusV2"\n'
        '  type: "SoftplusV2"\n}\n',
        encoding="utf-8",
    )

    audit = audit_aten_softplus_export(
        session,
        graph_dir,
        calls_before=0,
        relative_to=tmp_path,
    )
    assert audit["ge_node_occurrences"] == 2
    assert audit["minimum_occurrences"] == 1
    assert audit["converter_policy"] == "framework-registered-ge-ir"


def test_aten_softplus_audit_accepts_existing_ge_lowering_without_counter(
    tmp_path: Path,
) -> None:
    session = prepare_aten_softplus_export(_FakeTorchAir())
    graph_dir = tmp_path / "air" / "target-prefill"
    graph_dir.mkdir(parents=True)
    (graph_dir / "dynamo.pbtxt").write_text(
        'op {\n  name: "softplus"\n  type: "SoftplusV2"\n}\n',
        encoding="utf-8",
    )

    audit = audit_aten_softplus_export(
        session,
        graph_dir,
        calls_before=0,
        relative_to=tmp_path,
        minimum_occurrences=1,
    )

    assert audit["status"] == "PASS"
    assert audit["converter_calls"] == 0
    assert audit["ge_node_occurrences"] == 1
    assert audit["converter_policy"] == "torchair-existing-ge-ir"
    assert audit["observed_in_graph"] is True


def test_gdr_fake_keeps_frontend_operator_in_strict_export() -> None:
    operation = _ensure_target_test_schema("npu_chunk_gated_delta_rule")
    prepare_custom_op_export(
        CustomOpExportSpec(
            NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
            NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
        ),
        _FakeTorchAir(),
    )

    class UsesGdr(nn.Module):
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            gate: torch.Tensor,
            beta: torch.Tensor,
            effective_length: torch.Tensor,
            state: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            return operation(
                query,
                key,
                value,
                gate,
                beta,
                effective_length,
                64,
                state,
                True,
                False,
            )

    query = torch.randn(1, 64, 2, 8, dtype=torch.float16)
    exported = torch.export.export(
        UsesGdr(),
        (
            query,
            query.clone(),
            torch.randn(1, 64, 2, 16, dtype=torch.float16),
            torch.randn(1, 64, 2, dtype=torch.float32),
            torch.randn(1, 64, 2, dtype=torch.float16),
            torch.tensor([64], dtype=torch.int16),
            torch.randn(1, 2, 8, 16, dtype=torch.float32),
        ),
        strict=True,
    )
    targets = [str(node.target) for node in exported.graph.nodes]
    assert "npu.npu_chunk_gated_delta_rule.default" in targets


def test_fused_attention_fake_keeps_frontend_operator_in_strict_export() -> None:
    operation = _ensure_target_test_schema("adn_fused_infer_attention")
    prepare_custom_op_export(
        CustomOpExportSpec(
            ADN_FUSED_INFER_ATTENTION_TORCH_OP,
            ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
        ),
        _FakeTorchAir(),
    )

    class UsesFusedAttention(nn.Module):
        def forward(
            self,
            query: torch.Tensor,
            key: torch.Tensor,
            value: torch.Tensor,
            mask: torch.Tensor,
            block_table: torch.Tensor,
        ) -> torch.Tensor:
            return operation(
                query,
                [key],
                [value],
                atten_mask=mask,
                all_seq_lengths_q=[3],
                actual_seq_lengths_q=[3],
                actual_seq_lengths_kv=[64],
                block_table=block_table,
                num_heads=16,
                scale_value=0.125,
                input_layout="BNSD",
                num_key_value_heads=4,
                block_size=64,
                inner_precise=2,
            )

    exported = torch.export.export(
        UsesFusedAttention(),
        (
            torch.randn(1, 256, 3, 16, dtype=torch.float16),
            torch.randn(1, 64, 64, 16, dtype=torch.float16),
            torch.randn(1, 64, 64, 16, dtype=torch.float16),
            torch.zeros(1, 1, 3, 64, dtype=torch.float16),
            torch.zeros(1, 1, dtype=torch.int32),
        ),
        strict=True,
    )
    targets = [str(node.target) for node in exported.graph.nodes]
    assert "npu.adn_fused_infer_attention.default" in targets


def test_w8a8_fakes_keep_dynamic_quant_and_quant_matmul_in_export() -> None:
    dynamic_quant = _ensure_target_test_schema("npu_dynamic_quant")
    _ensure_target_test_schema("npu_quant_matmul")
    torchair = _FakeTorchAir()
    prepare_custom_op_export(
        CustomOpExportSpec(
            NPU_DYNAMIC_QUANT_TORCH_OP,
            NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    quant_matmul = torch.ops.qwen35_dflash.npu_quant_matmul_v4444.default

    class UsesW8A8Ops(nn.Module):
        def forward(
            self,
            value: torch.Tensor,
            weight: torch.Tensor,
            scale: torch.Tensor,
        ) -> torch.Tensor:
            quantized, pertoken = dynamic_quant(value)
            return quant_matmul(
                quantized,
                weight,
                scale,
                pertoken_scale=pertoken.reshape(-1),
                output_dtype=torch.float16,
            )

    exported = torch.export.export(
        UsesW8A8Ops(),
        (
            torch.randn(1, 3, 8, dtype=torch.float16),
            torch.randint(-8, 8, (8, 5), dtype=torch.int8),
            torch.randn(5, dtype=torch.float32),
        ),
        strict=True,
    )
    targets = [str(node.target) for node in exported.graph.nodes]
    assert "npu.npu_dynamic_quant.default" in targets
    assert "qwen35_dflash.npu_quant_matmul_v4444.default" in targets


def test_v4444_frontend_does_not_overwrite_torchair_builtin_v3_converter() -> None:
    upstream = _ensure_target_test_schema("npu_quant_matmul")
    torchair = _FakeTorchAir()
    builtin_v3_converter = object()
    torchair.converters[upstream] = builtin_v3_converter

    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )

    private_frontend = (
        torch.ops.qwen35_dflash.npu_quant_matmul_v4444.default
    )
    assert torchair.converters[upstream] is builtin_v3_converter
    assert callable(torchair.converters[private_frontend])
    assert private_frontend is not upstream


def test_qlinear_export_mode_captures_private_frontend_only_while_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dynamic_quant = _ensure_target_test_schema("npu_dynamic_quant")
    _ensure_target_test_schema("npu_quant_matmul")
    torchair = _FakeTorchAir()
    prepare_custom_op_export(
        CustomOpExportSpec(
            NPU_DYNAMIC_QUANT_TORCH_OP,
            NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.__spec__ = importlib.machinery.ModuleSpec(
        "torch_npu",
        loader=None,
    )

    def unexpected_eager_call(*args: object, **kwargs: object) -> torch.Tensor:
        del args, kwargs
        raise AssertionError("AIR capture entered the eager QuantMatmul ABI")

    fake_torch_npu.npu_dynamic_quant = dynamic_quant
    fake_torch_npu.npu_trans_quant_param = unexpected_eager_call
    fake_torch_npu.npu_quant_matmul = unexpected_eager_call
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    modeling = importlib.import_module("models.modeling_qwen3_5_hiai_nd")
    monkeypatch.setattr(modeling, "torch_npu", fake_torch_npu)

    layer = modeling.QLinear(
        W_q=torch.ones((4, 3), dtype=torch.int8),
        scale=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32),
        idx=0,
    ).set_quant_matmul_export_mode(True)
    exported = torch.export.export(
        layer,
        (torch.ones((2, 4), dtype=torch.float16),),
        strict=True,
    )
    targets = [str(node.target) for node in exported.graph.nodes]
    assert "npu.npu_dynamic_quant.default" in targets
    assert "qwen35_dflash.npu_quant_matmul_v4444.default" in targets
    assert "npu.npu_quant_matmul.default" not in targets


def test_qlinear_eager_preserves_quant_branch_float_scale_outside_air(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_target_test_schema("npu_quant_matmul")
    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        _FakeTorchAir(),
    )
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.__spec__ = importlib.machinery.ModuleSpec(
        "torch_npu",
        loader=None,
    )
    observed_scales: list[torch.Tensor] = []

    def dynamic_quant(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return value.to(torch.int8), torch.ones(
            value.shape[:-1], dtype=torch.float32, device=value.device
        )

    def unexpected_trans_quant_param(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("eager QLinear must preserve the quant FP32 scale")

    def quant_matmul(
        x1: torch.Tensor,
        x2: torch.Tensor,
        scale: torch.Tensor,
        *,
        pertoken_scale: torch.Tensor,
        output_dtype: torch.dtype,
    ) -> torch.Tensor:
        del pertoken_scale
        observed_scales.append(scale)
        return torch.zeros(
            (*x1.shape[:-1], x2.shape[-1]),
            dtype=output_dtype,
            device=x1.device,
        )

    fake_torch_npu.npu_dynamic_quant = dynamic_quant
    fake_torch_npu.npu_trans_quant_param = unexpected_trans_quant_param
    fake_torch_npu.npu_quant_matmul = quant_matmul
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    modeling = importlib.import_module("models.modeling_qwen3_5_hiai_nd")
    monkeypatch.setattr(modeling, "torch_npu", fake_torch_npu)

    layer = modeling.QLinear(
        W_q=torch.ones((4, 3), dtype=torch.int8),
        scale=torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32),
        idx=0,
    ).set_quant_matmul_export_mode(True)
    value = torch.ones((2, 4), dtype=torch.float16)
    first = layer(value)
    second = layer(value)

    assert first.shape == second.shape == (2, 3)
    assert len(observed_scales) == 2
    assert all(scale.dtype == torch.float32 for scale in observed_scales)
    assert all(torch.equal(scale, layer.scale) for scale in observed_scales)
    assert layer.scale.dtype == torch.float32
    assert observed_scales[0] is observed_scales[1]


def test_air_factory_enables_private_quant_frontend_on_every_qlinear() -> None:
    class ExportAwareLinear(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.enabled = False

        def set_quant_matmul_export_mode(self, enabled: bool = True) -> None:
            self.enabled = enabled

    class ExportTarget(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.execution = nn.Sequential(
                ExportAwareLinear(),
                ExportAwareLinear(),
            )
            self.dflash_target_quantization_audit = {"qlinear_count": 2}

        @property
        def dflash_execution_model(self) -> nn.Module:
            return self.execution

    target = ExportTarget()
    assert _enable_target_quant_matmul_export_mode(target) == 2
    assert all(layer.enabled for layer in target.execution)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"output_dtype": torch.int32}, "output_dtype=torch.float16"),
        (
            {"output_dtype": torch.float16, "group_sizes": [128]},
            "group_sizes=None",
        ),
    ),
)
def test_quant_matmul_v4444_converter_rejects_unlocked_routes(
    kwargs: dict[str, object],
    message: str,
) -> None:
    _ensure_target_test_schema("npu_quant_matmul")
    torchair = _FakeTorchAir()
    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    operation = torch.ops.qwen35_dflash.npu_quant_matmul_v4444.default
    with pytest.raises(RuntimeError, match=message):
        torchair.converters[operation](object(), object(), object(), **kwargs)


def test_quant_matmul_meta_probe_uses_the_m_dimension_for_pertoken_scale() -> None:
    observed: dict[str, tuple[int, ...]] = {}

    def strict_upstream_meta(
        x1: torch.Tensor,
        x2: torch.Tensor,
        scale: torch.Tensor,
        *,
        offset: torch.Tensor | None = None,
        pertoken_scale: torch.Tensor | None = None,
        bias: torch.Tensor | None = None,
        output_dtype: torch.dtype | None = None,
        group_sizes: list[int] | None = None,
    ) -> torch.Tensor:
        del scale, offset, bias, group_sizes
        assert pertoken_scale is not None
        observed["x1"] = tuple(x1.shape)
        observed["pertoken_scale"] = tuple(pertoken_scale.shape)
        if pertoken_scale.shape[0] != x1.shape[-2]:
            raise RuntimeError(
                "the pertoken_scale 1st dim value must be x1 m dim value"
            )
        return x1.new_empty(
            (*x1.shape[:-1], x2.shape[-1]),
            dtype=output_dtype,
        )

    _validate_npu_quant_matmul_meta(strict_upstream_meta)
    assert observed == {
        "x1": (1, 64, 2560),
        "pertoken_scale": (64,),
    }


def test_quant_matmul_meta_rejects_flattened_batch_times_m_scale() -> None:
    _ensure_target_test_schema("npu_quant_matmul")
    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        _FakeTorchAir(),
    )
    operation = torch.ops.qwen35_dflash.npu_quant_matmul_v4444.default
    with pytest.raises(RuntimeError, match="x1 m dim value"):
        operation(
            torch.empty((2, 3, 8), dtype=torch.int8, device="meta"),
            torch.empty((8, 5), dtype=torch.int8, device="meta"),
            torch.empty((5,), dtype=torch.float32, device="meta"),
            pertoken_scale=torch.empty(
                (6,), dtype=torch.float32, device="meta"
            ),
            output_dtype=torch.float16,
        )


def test_native_cache_update_uses_copy_free_frontend_for_air_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_update = _ensure_target_test_schema("npu_cache_update_")
    _ensure_cache_update_cpu_impl()
    scatter_update = _ensure_target_test_schema("npu_scatter_nd_update_")
    torchair = _FakeTorchAir()
    prepare_custom_op_export(
        CustomOpExportSpec(
            FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP,
            NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    prepare_custom_op_export(
        CustomOpExportSpec(
            NPU_SCATTER_ND_UPDATE_TORCH_OP,
            NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE,
            minimum_occurrences=0,
        ),
        torchair,
    )
    fake_torch_npu = ModuleType("torch_npu")
    fake_torch_npu.__spec__ = importlib.machinery.ModuleSpec(
        "torch_npu",
        loader=None,
    )
    fake_torch_npu.npu_cache_update_ = cache_update
    monkeypatch.setitem(sys.modules, "torch_npu", fake_torch_npu)
    modeling = importlib.import_module("models.modeling_qwen3_5_hiai_nd")
    monkeypatch.setattr(modeling, "torch_npu", fake_torch_npu)

    eager_cache = torch.zeros(4, dtype=torch.float32)
    eager_result = modeling._npu_cache_update(
        eager_cache,
        torch.ones_like(eager_cache),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros((), dtype=torch.int32),
    )
    assert eager_result is eager_cache
    torch.testing.assert_close(eager_cache, torch.ones_like(eager_cache))

    class UsesCacheUpdate(nn.Module):
        def forward(
            self,
            cache: torch.Tensor,
            updates: torch.Tensor,
            block: torch.Tensor,
            offset: torch.Tensor,
        ) -> torch.Tensor:
            cache = modeling._npu_cache_update(cache, updates, block, offset, use_export_frontend=True)
            cache = modeling._npu_cache_update(
                cache, updates + 1, block, offset, use_export_frontend=True
            )
            return cache

    class UsesScatterUpdate(nn.Module):
        def forward(
            self,
            cache: torch.Tensor,
            indices: torch.Tensor,
            updates: torch.Tensor,
        ) -> torch.Tensor:
            scatter_update(cache, indices, updates)
            return cache

    cache_export = torch.export.export(
        UsesCacheUpdate(),
        (
            torch.zeros(4, 64, 64, 16, dtype=torch.float16),
            torch.zeros(1, 64, 16, dtype=torch.float16),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros((), dtype=torch.int32),
        ),
        strict=True,
    )
    compiled_input = torch.zeros(4, dtype=torch.float32)
    aot_targets: list[str] = []

    def capture_aot_graph(
        graph: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        del example_inputs
        aot_targets.extend(str(node.target) for node in graph.graph.nodes)
        return graph.forward

    from torch._dynamo.backends.common import aot_autograd

    compiled_cache_update = torch.compile(
        UsesCacheUpdate(),
        backend=aot_autograd(fw_compiler=capture_aot_graph),
        fullgraph=True,
    )
    compiled_result = compiled_cache_update(
        compiled_input,
        torch.ones_like(compiled_input),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros((), dtype=torch.int32),
    )
    expected = torch.full_like(compiled_input, 2)
    torch.testing.assert_close(compiled_result, expected)
    torch.testing.assert_close(compiled_input, torch.zeros_like(compiled_input))
    assert aot_targets.count("qwen35_dflash.npu_cache_update.default") == 2
    assert "npu.npu_cache_update_.default" not in aot_targets
    assert "aten.copy.default" not in aot_targets
    scatter_export = torch.export.export(
        UsesScatterUpdate(),
        (
            torch.zeros(8, 2, 4, dtype=torch.float16),
            torch.zeros(2, dtype=torch.int64),
            torch.zeros(2, 2, 4, dtype=torch.float16),
        ),
        strict=True,
    )
    cache_targets = {
        str(node.target) for node in cache_export.graph.nodes
    }
    assert "qwen35_dflash.npu_cache_update.default" in cache_targets
    assert "npu.npu_cache_update_.default" not in cache_targets
    assert "aten.copy.default" not in cache_targets
    assert "npu.npu_scatter_nd_update_.default" in {
        str(node.target) for node in scatter_export.graph.nodes
    }


def test_custom_op_audit_accepts_builtin_op_field_without_double_counting(
    tmp_path: Path,
) -> None:
    _ensure_target_test_schema("npu_dynamic_quant")
    session = prepare_custom_op_export(
        CustomOpExportSpec(
            torch_op=NPU_DYNAMIC_QUANT_TORCH_OP,
            ge_op_type=NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
        ),
        _FakeTorchAir(),
    )
    graph_dir = tmp_path / "air" / "op-field"
    graph_dir.mkdir(parents=True)
    (graph_dir / "dynamo.pbtxt").write_text(
        'op {\n  name: "dynamic_quant"\n  op: "DynamicQuant"\n}\n',
        encoding="utf-8",
    )

    audit = audit_custom_op_export(
        (session,), graph_dir, relative_to=tmp_path
    )
    assert audit[0]["ge_node_occurrences"] == 1
    assert audit[0]["observed_in_graph"] is True


def test_compiler_rejects_incomplete_declared_custom_op_audit() -> None:
    graph = {
        "metadata": {
            "custom_op_export_contracts": [
                {
                    "torch_target": "npu.adn_rms_norm.default",
                    "ge_op_type": "AdnRmsNorm",
                    "minimum_occurrences": 1,
                },
                {
                    "torch_target": "npu.npu_chunk_gated_delta_rule.default",
                    "ge_op_type": "ChunkGatedDeltaRule",
                    "minimum_occurrences": 1,
                },
            ]
        },
        "custom_op_audit": [
            {
                "status": "PASS",
                "torch_target": "npu.adn_rms_norm.default",
                "ge_op_type": "AdnRmsNorm",
                "minimum_occurrences": 1,
                "converter_policy": "framework-registered-ge-ir",
                "converter_calls": 1,
                "ge_node_occurrences": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="every declared contract"):
        _validated_custom_op_audit(graph)


def test_all_target_custom_ops_have_exact_meta_and_lowering_policy() -> None:
    operations = _ensure_all_target_test_schemas()
    torchair = _FakeTorchAir()
    specs = (
        CustomOpExportSpec(
            ADN_FUSED_INFER_ATTENTION_TORCH_OP,
            ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
        ),
        CustomOpExportSpec(
            ADN_RMS_NORM_TORCH_OP,
            ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
        ),
        CustomOpExportSpec(
            FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP,
            NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
        ),
        CustomOpExportSpec(
            NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
            NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
        ),
        CustomOpExportSpec(
            NPU_DYNAMIC_QUANT_TORCH_OP,
            NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
        ),
        CustomOpExportSpec(
            FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        CustomOpExportSpec(
            NPU_SCATTER_ND_UPDATE_TORCH_OP,
            NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE,
            minimum_occurrences=0,
        ),
    )
    sessions = tuple(
        prepare_custom_op_export(spec, torchair) for spec in specs
    )
    assert all(
        session.fake_kernel
        in {"framework-registered-fake", "preexisting-meta-kernel"}
        for session in sessions
    )
    policies = {
        session.spec.torch_op: session.converter_policy
        for session in sessions
    }
    assert policies == {
        ADN_FUSED_INFER_ATTENTION_TORCH_OP: "framework-registered-ge-ir",
        ADN_RMS_NORM_TORCH_OP: "framework-registered-ge-ir",
        FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP: "framework-registered-ge-ir",
        NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP: "framework-registered-ge-ir",
        NPU_DYNAMIC_QUANT_TORCH_OP: "torchair-builtin",
        FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP: "framework-registered-ge-ir",
        NPU_SCATTER_ND_UPDATE_TORCH_OP: "torchair-builtin",
    }
    functional_cache_update = torch.ops.qwen35_dflash.npu_cache_update.default
    functional_quant_matmul = (
        torch.ops.qwen35_dflash.npu_quant_matmul_v4444.default
    )
    assert set(torchair.converters) == {
        operations["adn_fused_infer_attention"],
        operations["adn_rms_norm"],
        functional_cache_update,
        operations["npu_chunk_gated_delta_rule"],
        functional_quant_matmul,
    }

    placeholder = object()
    query, key, value, gate, beta, effective_length, initial_state = (
        object() for _ in range(7)
    )
    gdr_session = next(
        session
        for session in sessions
        if session.spec.torch_op == NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP
    )
    rms_session = next(
        session
        for session in sessions
        if session.spec.torch_op == ADN_RMS_NORM_TORCH_OP
    )
    cache_session = next(
        session
        for session in sessions
        if session.spec.torch_op == FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP
    )
    quant_matmul_session = next(
        session
        for session in sessions
        if session.spec.torch_op == FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP
    )
    attention_session = next(
        session
        for session in sessions
        if session.spec.torch_op == ADN_FUSED_INFER_ATTENTION_TORCH_OP
    )
    rms_input, rms_gamma = object(), object()
    torchair.converters[operations["adn_rms_norm"]](
        rms_input,
        rms_gamma,
        1e-6,
        meta_outputs=(placeholder, placeholder),
    )
    torchair.converters[operations["npu_chunk_gated_delta_rule"]](
        query,
        key,
        value,
        gate,
        beta,
        effective_length,
        64,
        initial_state,
        True,
        True,
        meta_outputs=(placeholder, placeholder),
    )
    torchair.converters[functional_cache_update](*([placeholder] * 4))
    quant_x1, quant_x2, scale, pertoken_scale = (
        object() for _ in range(4)
    )
    torchair.converters[functional_quant_matmul](
        quant_x1,
        quant_x2,
        scale,
        pertoken_scale=pertoken_scale,
        output_dtype=torch.float16,
    )
    attention_mask, block_table, logical_end = object(), object(), object()
    torchair.converters[operations["adn_fused_infer_attention"]](
        placeholder,
        [placeholder],
        [placeholder],
        atten_mask=attention_mask,
        pse_shift=logical_end,
        all_seq_lengths_q=[5],
        actual_seq_lengths_q=[3],
        actual_seq_lengths_kv=[64],
        block_table=block_table,
        num_heads=16,
        scale_value=0.125,
        input_layout="BNSD",
        num_key_value_heads=4,
        block_size=64,
        inner_precise=2,
    )
    assert {
        call[0] for call in torchair.ge.calls
    } >= {
        ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
        ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
        NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
        NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
        NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
    }
    rms_call = next(
        call
        for call in torchair.ge.calls
        if call[0] == ADN_RMS_NORM_DEFAULT_GE_OP_TYPE
    )
    assert rms_call[1] == ()
    assert rms_call[2] == {
        "inputs": {"x": rms_input, "gamma": rms_gamma},
        "outputs": ["y", "rstd"],
        "attrs": {"epsilon": ("float", 1e-6)},
    }
    assert rms_session.converter_mode == "named-only"
    gdr_call = next(
        call
        for call in torchair.ge.calls
        if call[0] == NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE
    )
    assert gdr_call[1] == ()
    assert gdr_call[2] == {
        "inputs": {
            "query": query,
            "key": key,
            "value": value,
            "g": gate,
            "beta": beta,
            "initial_state": initial_state,
            "effective_length": effective_length,
        },
        "outputs": ["core_attn", "last_recurrent_state"],
        "attrs": {
            "chunk_size": ("int", 64),
            "output_final_state": ("bool", True),
            "use_qk_l2norm_in_kernel": ("bool", True),
        },
    }
    assert gdr_session.converter_mode == "named-gdr-effective-length-v2"
    cache_call = next(
        call
        for call in torchair.ge.calls
        if call[0] == NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE
    )
    assert cache_call[1] == ()
    assert cache_call[2] == {
        "inputs": {
            "x": placeholder,
            "updates": placeholder,
            "targetBlock": placeholder,
            "offsetInBlock": placeholder,
        },
        "outputs": ["x"],
    }
    assert cache_session.converter_mode == "named-cache-update-x-v1"
    quant_matmul_call = next(
        call
        for call in torchair.ge.calls
        if call[0] == NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE
    )
    assert quant_matmul_call[1] == ()
    assert quant_matmul_call[2] == {
        "inputs": {
            "x1": quant_x1,
            "x2": quant_x2,
            "scale": scale,
            "offset": None,
            "bias": None,
            "pertoken_scale": pertoken_scale,
        },
        "outputs": ["y"],
        "attrs": {
            "dtype": ("int", 1),
            "transpose_x1": ("bool", False),
            "transpose_x2": ("bool", False),
            "group_size": ("int", 0),
        },
    }
    assert quant_matmul_session.converter_mode == (
        "named-quant-batch-matmul-v4444-fp16"
    )
    attention_call = next(
        call
        for call in torchair.ge.calls
        if call[0] == ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE
    )
    assert attention_call[1] == ()
    assert attention_call[2] == {
        "inputs": {
            "query": placeholder,
            "key": [placeholder],
            "value": [placeholder],
            "pse_shift": logical_end,
            "atten_mask": attention_mask,
            "actual_seq_lengths_q": ("const", [3], "DT_INT64"),
            "actual_seq_lengths_kv": ("const", [64], "DT_INT64"),
            "dequant_scale1": None,
            "quant_scale1": None,
            "dequant_scale2": None,
            "quant_scale2": None,
            "quant_offset2": None,
            "antiquant_scale": None,
            "antiquant_offset": None,
            "block_table": block_table,
            "kv_padding_size": None,
            "all_seq_lengths_q": ("const", [5], "DT_INT64"),
            "actual_seq_lengths_q_back": None,
        },
        "outputs": ["attention_out"],
        "attrs": {
            "num_heads": ("int", 16),
            "scale_value": ("float", 0.125),
            "input_layout": ("str", "BNSD"),
            "num_key_value_heads": ("int", 4),
            "inner_precise": ("int", 2),
            "block_size": ("int", 64),
        },
    }
    assert attention_session.converter_mode == "named-adn-attention-v1"
