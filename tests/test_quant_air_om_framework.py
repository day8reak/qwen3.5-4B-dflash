from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.compiler import (
    _validated_custom_op_audit,
    compile_air_bundle,
)
from qwen35_dflash.ascend310p.contracts import AirGraphSpec, CustomOpExportSpec
from qwen35_dflash.ascend310p.custom_op_export import (
    ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
    ADN_FUSED_INFER_ATTENTION_TORCH_OP,
    ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
    ADN_RMS_NORM_TORCH_OP,
    FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP,
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
    _validate_npu_quant_matmul_meta,
    audit_custom_op_export,
    prepare_custom_op_export,
    validate_gdr_ge_prototype_environment,
)
from qwen35_dflash.ascend310p.exporter import export_air_bundle
from qwen35_dflash.ascend310p.input_manifest import (
    build_quant_input_manifest,
    verify_quant_input_manifest,
)
from qwen35_dflash.ascend310p.quant_factory import (
    AirDFlashOps,
    QUANT_BASE_REVISION,
    QuantFullPrefixExportTarget,
    create_quant_recompute_graph,
)
from qwen35_dflash.ascend310p.standard_op_export import (
    ATEN_SOFTPLUS_GE_OP_TYPE,
    ATEN_SOFTPLUS_TORCH_TARGET,
    audit_aten_softplus_export,
    prepare_aten_softplus_export,
)
from qwen35_dflash.ascend310p.utils import sha256_file
from qwen35_dflash.ascend310p.workflow import DEFAULT_GRAPH_FACTORY
from models.dflash_v1 import dflash_ascend310p_ops as golden_ops


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


def _ensure_adn_rms_norm_test_schema() -> object:
    return _ensure_target_test_schema("adn_rms_norm")


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

    def custom_op(self, op_type: str, *args: object, **kwargs: object):
        self.calls.append((op_type, args, dict(kwargs)))
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


class _FakeTorchAir:
    __version__ = "test"

    def __init__(self) -> None:
        self.ge = _FakeTorchAirGe()
        self.converters: dict[object, object] = {}

    @property
    def converter(self):
        return self.converters.get(torch.ops.npu.adn_rms_norm.default)

    def register_fx_node_ge_converter(self, operation: object):
        def register(converter):
            self.converters[operation] = converter
            return converter

        return register

    def dynamo_export(
        self,
        *args: object,
        model: nn.Module,
        export_path: str,
        export_name: str,
        dynamic: bool,
        **kwargs: object,
    ) -> None:
        del args, model, dynamic, kwargs
        converter = self.converter
        assert converter is not None
        converter(object(), object(), 1e-6, meta_outputs=(object(), object()))
        softplus_converter = self.converters.get(torch.ops.aten.softplus.default)
        assert softplus_converter is not None
        softplus_converter(object(), 1, 20, meta_outputs=object())
        root = Path(export_path)
        (root / f"{export_name}.air").write_bytes(b"air")
        (root / "dynamo.pbtxt").write_text(
            'op {\n  name: "rms"\n  type: "RmsNorm"\n}\n'
            'op {\n  name: "softplus"\n  type: "SoftplusV2"\n}\n',
            encoding="utf-8",
        )


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


class _FakeExecutionModel(nn.Module):
    def __init__(self, embedding: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        # Keep references outside Module registration: the owning fake target
        # already registers both weights.
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "lm_head", lm_head)
        self.last_gdr_effective_length: torch.Tensor | None = None
        self.last_export_flag: bool | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor,
        output_dflash_features: bool,
        gdr_effective_length: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids
        assert output_dflash_features is True
        self.last_export_flag = bool(kwargs.pop("export_flag"))
        self.last_gdr_effective_length = gdr_effective_length.detach().clone()
        return self.lm_head(inputs_embeds), torch.cat(
            (inputs_embeds, inputs_embeds), dim=-1
        )


class _FakeQuantTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.lm_head = nn.Linear(4, 32, bias=False)
        self.execution = _FakeExecutionModel(self.embedding, self.lm_head)
        self.public_forward_calls = 0
        self._target_quantized_embedding = self.embedding
        self.dflash_target_quantization_audit = {
            "status": "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM",
            "scheme": "w8a8_dynamic",
        }

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @property
    def dflash_execution_model(self) -> nn.Module:
        return self.execution

    def _fresh_attention_mask(self, sequence_length: int) -> torch.Tensor:
        return torch.zeros(1, 1, sequence_length, sequence_length)

    def _fresh_hybrid_cache(
        self, *, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.zeros(batch_size, 1, 1),
                torch.zeros(batch_size, 1, 1, 1),
            )
        ]

    def forward(self, input_ids: torch.Tensor, **kwargs: object):
        del input_ids, kwargs
        self.public_forward_calls += 1
        raise AssertionError("AIR adapter must bypass stateful public forward")


class _FakeDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            block_size=16,
            mask_token_id=31,
            vocab_size=32,
            hidden_size=4,
        )

    def embed_block(
        self, block_ids: torch.Tensor, embedding_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.embedding(block_ids, embedding_weight)

    def draft_top1(
        self,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del target_hidden, position_ids, lm_head_weight, attention_mask
        return torch.arange(
            1,
            noise_embedding.shape[1],
            dtype=torch.long,
            device=noise_embedding.device,
        ).unsqueeze(0)


def test_adn_rms_norm_fake_keeps_frontend_operator_in_export() -> None:
    operation = _ensure_adn_rms_norm_test_schema()
    torchair = _FakeTorchAir()
    spec = CustomOpExportSpec(
        torch_op=ADN_RMS_NORM_TORCH_OP,
        ge_op_type=ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
    )
    session = prepare_custom_op_export(spec, torchair)

    class UsesAdnRmsNorm(nn.Module):
        def forward(
            self, input_tensor: torch.Tensor, gamma: torch.Tensor
        ) -> torch.Tensor:
            return operation(input_tensor, gamma, 1e-6)[0]

    exported = torch.export.export(
        UsesAdnRmsNorm(),
        (torch.randn(2, 4, 8), torch.ones(8)),
        strict=True,
    )
    targets = [str(node.target) for node in exported.graph.nodes]
    assert "npu.adn_rms_norm.default" in targets
    assert session.fake_kernel in {
        "framework-registered-fake",
        "preexisting-meta-kernel",
    }


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
            NPU_QUANT_MATMUL_TORCH_OP,
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
        NPU_QUANT_MATMUL_TORCH_OP: "framework-registered-ge-ir",
        NPU_SCATTER_ND_UPDATE_TORCH_OP: "torchair-builtin",
    }
    functional_cache_update = torch.ops.qwen35_dflash.npu_cache_update.default
    assert set(torchair.converters) == {
        operations["adn_fused_infer_attention"],
        operations["adn_rms_norm"],
        functional_cache_update,
        operations["npu_chunk_gated_delta_rule"],
        operations["npu_quant_matmul"],
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
    cache_session = next(
        session
        for session in sessions
        if session.spec.torch_op == FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP
    )
    quant_matmul_session = next(
        session
        for session in sessions
        if session.spec.torch_op == NPU_QUANT_MATMUL_TORCH_OP
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
    torchair.converters[operations["npu_quant_matmul"]](
        quant_x1,
        quant_x2,
        scale,
        pertoken_scale=pertoken_scale,
        output_dtype=torch.float16,
    )
    torchair.converters[operations["adn_fused_infer_attention"]](
        placeholder,
        [placeholder],
        [placeholder],
        all_seq_lengths_q=[3],
        actual_seq_lengths_q=[3],
        actual_seq_lengths_kv=[64],
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
        NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
        NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
        NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
    }
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
    quant_matmul = _ensure_target_test_schema("npu_quant_matmul")
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
            NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )

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
    assert "npu.npu_quant_matmul.default" in targets


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
    operation = _ensure_target_test_schema("npu_quant_matmul")
    torchair = _FakeTorchAir()
    prepare_custom_op_export(
        CustomOpExportSpec(
            NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    with pytest.raises(RuntimeError, match=message):
        torchair.converters[operation](object(), object(), object(), **kwargs)


def test_quant_matmul_rejects_legacy_v3_ge_type() -> None:
    _ensure_target_test_schema("npu_quant_matmul")
    with pytest.raises(RuntimeError, match="Update npu_quant_matmul_ge_op_type"):
        prepare_custom_op_export(
            CustomOpExportSpec(
                NPU_QUANT_MATMUL_TORCH_OP,
                "QuantBatchMatmulV3",
            ),
            _FakeTorchAir(),
        )


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
    operation = _ensure_target_test_schema("npu_quant_matmul")
    prepare_custom_op_export(
        CustomOpExportSpec(
            NPU_QUANT_MATMUL_TORCH_OP,
            NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
        ),
        _FakeTorchAir(),
    )
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


def test_functional_cache_update_survives_aot_and_keeps_ge_custom_op() -> None:
    _ensure_target_test_schema("npu_cache_update_")
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
    cache_update = torch.ops.qwen35_dflash.npu_cache_update.default

    class UsesCacheUpdate(nn.Module):
        def forward(
            self,
            cache: torch.Tensor,
            updates: torch.Tensor,
            block: torch.Tensor,
            offset: torch.Tensor,
        ) -> torch.Tensor:
            return cache_update(cache, updates, block, offset)

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
    eager_cache = torch.zeros(4, dtype=torch.float32)
    aot_targets: set[str] = set()

    def capture_aot_graph(
        graph: torch.fx.GraphModule,
        example_inputs: list[torch.Tensor],
    ):
        del example_inputs
        aot_targets.update(str(node.target) for node in graph.graph.nodes)
        return graph.forward

    from torch._dynamo.backends.common import aot_autograd

    compiled_cache_update = torch.compile(
        UsesCacheUpdate(),
        backend=aot_autograd(fw_compiler=capture_aot_graph),
        fullgraph=True,
    )
    compiled_result = compiled_cache_update(
        eager_cache,
        torch.ones_like(eager_cache),
        torch.zeros(1, dtype=torch.int32),
        torch.zeros((), dtype=torch.int32),
    )
    torch.testing.assert_close(compiled_result, torch.ones_like(eager_cache))
    torch.testing.assert_close(eager_cache, torch.zeros_like(eager_cache))
    assert "qwen35_dflash.npu_cache_update.default" in aot_targets
    assert "npu.npu_cache_update_.default" not in aot_targets
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
    assert "npu.npu_scatter_nd_update_.default" in {
        str(node.target) for node in scatter_export.graph.nodes
    }


def test_air_export_audits_retained_adn_rms_norm(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _ensure_adn_rms_norm_test_schema()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    torchair = _FakeTorchAir()

    def factory(config: object) -> tuple[AirGraphSpec, ...]:
        del config
        return (
            AirGraphSpec(
                name="retained_custom_op",
                role="generation-recompute",
                model=nn.Identity(),
                example_args=(torch.ones(1),),
                custom_ops=(
                    CustomOpExportSpec(
                        torch_op=ADN_RMS_NORM_TORCH_OP,
                        ge_op_type=ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
                    ),
                ),
            ),
        )

    result = export_air_bundle(
        factory,
        {},
        run_dir / "bundle",
        torchair_module=torchair,
    )
    graph = result["graphs"][0]
    audit = graph["custom_op_audit"][0]
    standard_override = graph["standard_op_overrides"][0]
    assert result["schema_version"] == 3
    assert audit["status"] == "PASS"
    assert audit["torch_target"] == "npu.adn_rms_norm.default"
    assert audit["ge_op_type"] == "RmsNorm"
    assert audit["converter_calls"] == 1
    assert audit["ge_node_occurrences"] == 1
    assert standard_override["status"] == "PASS"
    assert standard_override["torch_target"] == ATEN_SOFTPLUS_TORCH_TARGET
    assert standard_override["ge_op_type"] == ATEN_SOFTPLUS_GE_OP_TYPE
    assert standard_override["converter_calls"] == 1
    assert standard_override["ge_node_occurrences"] == 1
    assert standard_override["lowering"] == (
        "one GE node per call; beta and threshold preserved"
    )
    assert torchair.ge.calls[0][0] == "RmsNorm"
    assert any(
        item["path"].endswith("dynamo.pbtxt")
        for item in graph["payload_files"]
    )


def test_custom_op_audit_rejects_missing_ge_node(tmp_path: Path) -> None:
    _ensure_adn_rms_norm_test_schema()
    torchair = _FakeTorchAir()
    session = prepare_custom_op_export(
        CustomOpExportSpec(
            torch_op=ADN_RMS_NORM_TORCH_OP,
            ge_op_type=ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
        ),
        torchair,
    )
    assert torchair.converter is not None
    torchair.converter(object(), object(), 1e-6, meta_outputs=(object(), object()))
    graph_dir = tmp_path / "air" / "missing"
    graph_dir.mkdir(parents=True)
    (graph_dir / "dynamo.pbtxt").write_text(
        'op {\n  name: "wrong"\n  type: "Mul"\n}\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="TorchAir IR contains"):
        audit_custom_op_export((session,), graph_dir, relative_to=tmp_path)


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
                    "ge_op_type": "RmsNorm",
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
                "ge_op_type": "RmsNorm",
                "minimum_occurrences": 1,
                "converter_policy": "framework-registered-ge-ir",
                "converter_calls": 1,
                "ge_node_occurrences": 1,
            }
        ],
    }
    with pytest.raises(ValueError, match="every declared contract"):
        _validated_custom_op_audit(graph)


def test_default_factory_is_quant_branch_factory() -> None:
    assert DEFAULT_GRAPH_FACTORY.endswith("quant_factory:create_quant_recompute_graph")
    assert QUANT_BASE_REVISION == "28f93e784a2beed87020a80bd93c8788754eab1c"


def test_quant_graph_rejects_non_chunk_aligned_gear_before_weight_load() -> None:
    with pytest.raises(ValueError, match="divisible"):
        create_quant_recompute_graph({"max_sequence_length": 65})


def test_quant_graph_rejects_gear_outside_gdr_int16_abi() -> None:
    with pytest.raises(ValueError, match="INT16"):
        create_quant_recompute_graph({"max_sequence_length": 32768})


def test_quant_target_adapter_bypasses_eager_guards_and_sync() -> None:
    target = _FakeQuantTarget().eval()
    adapter = QuantFullPrefixExportTarget(target).eval()
    input_ids = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    logits, features = adapter(
        input_ids,
        mask,
        use_cache=False,
        return_dict=True,
        output_dflash_features=True,
    )
    assert logits.shape == (1, 4, 32)
    assert features.shape == (1, 4, 8)
    assert target.public_forward_calls == 0
    assert target.execution.last_gdr_effective_length is not None
    assert target.execution.last_gdr_effective_length.dtype == torch.int16
    assert target.execution.last_gdr_effective_length.tolist() == [2]
    assert target.execution.last_export_flag is True


def test_both_hiai_models_route_only_air_cache_updates_through_functional_op() -> None:
    ordinary = (ROOT / "models" / "modeling_qwen3_5_hiai_nd.py").read_text(
        encoding="utf-8"
    )
    rollback = (
        ROOT / "models" / "modeling_qwen3_5_hiai_nd_dflash_rollback.py"
    ).read_text(encoding="utf-8")

    assert "def _cache_update_for_export(" in ordinary
    assert "torch.ops.qwen35_dflash.npu_cache_update.default(" in ordinary
    assert "torch_npu.npu_cache_update_(" in ordinary
    assert ordinary.count("export_flag=export_flag") >= 3
    assert "QLinear, _cache_update_for_export" in rollback
    assert rollback.count("_cache_update_for_export(") >= 2
    assert rollback.count("export_flag=export_flag") >= 6


def test_both_hiai_models_route_query_lengths_to_attention_length_abi() -> None:
    sources = (
        ROOT / "models" / "modeling_qwen3_5_hiai_nd.py",
        ROOT / "models" / "modeling_qwen3_5_hiai_nd_dflash_rollback.py",
    )
    for source in sources:
        text = source.read_text(encoding="utf-8")
        assert 'attn_params["pse_shift"] = allQLen' not in text, source
        assert (
            text.count('attn_params["all_seq_lengths_q"] = allQLen') == 2
        ), source


def test_air_ops_match_quant_branch_decomposed_golden() -> None:
    torch.manual_seed(7)
    ops = AirDFlashOps()
    value = torch.randn(1, 3, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    torch.testing.assert_close(
        ops.rms_norm(value, weight, 1e-6),
        golden_ops.rms_norm(value, weight, 1e-6),
        rtol=0,
        atol=0,
    )
    linear_weight = torch.randn(5, 8, dtype=torch.float32)
    torch.testing.assert_close(
        ops.linear(value, linear_weight),
        golden_ops.linear(value, linear_weight),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        ops.top1(value, linear_weight),
        golden_ops.top1(value, linear_weight),
    )

    query = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    key = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    val = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    visible = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    visible[..., 0, 4] = False
    torch.testing.assert_close(
        ops.attention(query, key, val, visible, 0.125, 2),
        golden_ops.attention(query, key, val, visible, 0.125, 2),
        rtol=0,
        atol=0,
    )


def test_padded_draft_context_matches_compact_quant_draft() -> None:
    from qwen35_dflash.ascend310p.integrated import enable_padded_draft_context
    from models.dflash_v1.dflash_config import Qwen35DFlashConfig
    from models.dflash_v1.modeling_dflash import DFlashDraftModel

    config = Qwen35DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_target_layers=2,
        target_layer_ids=(0,),
        layer_types=("sliding_attention", "full_attention"),
        block_size=4,
        mask_token_id=31,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=32,
        sliding_window=4,
        use_sliding_window=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype="float32",
    )
    torch.manual_seed(19)
    compact = DFlashDraftModel(
        config,
        ops=AirDFlashOps(),
        device="cpu",
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in compact.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    padded = copy.deepcopy(compact).eval()
    enable_padded_draft_context(padded)

    context = torch.randn(1, 3, config.hidden_size)
    padding = torch.randn(1, 3, config.hidden_size)
    block = torch.randn(1, config.block_size, config.hidden_size)
    compact_positions = torch.arange(7, dtype=torch.long).unsqueeze(0)
    padded_positions = torch.tensor(
        [[0, 1, 2, 0, 0, 0, 3, 4, 5, 6]], dtype=torch.long
    )
    context_valid = torch.tensor(
        [[True, True, True, False, False, False]], dtype=torch.bool
    )
    expected = compact.forward_projected(
        context,
        block,
        compact_positions,
    )
    actual = padded.forward_projected(
        torch.cat((context, padding), dim=1),
        block,
        padded_positions,
        context_valid,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    exported = torch.export.export(
        padded,
        (
            torch.randn(1, 6, config.feature_size),
            block,
            padded_positions,
            context_valid,
        ),
        strict=True,
    )
    assert len(list(exported.graph.nodes)) > 100


def test_compile_uses_air_framework_and_hash_locks_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    bundle = run_dir / "bundle"
    graph_dir = bundle / "air" / "quant_dflash_recompute"
    graph_dir.mkdir(parents=True)
    air = graph_dir / "quant_dflash_recompute.air"
    air.write_bytes(b"quant-air")
    pbtxt = graph_dir / "dynamo.pbtxt"
    pbtxt.write_text('op { type: "RmsNorm" }\n', encoding="utf-8")
    manifest = {
        "schema_version": 3,
        "artifact_kind": "qwen35-dflash-torchair-bundle",
        "status": "PASS",
        "graphs": [
            {
                "name": "quant_dflash_recompute",
                "role": "generation-recompute",
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["target_top1", "draft_top1"],
                "metadata": {
                    "custom_op_export_contracts": [
                        {
                            "torch_target": "npu.adn_rms_norm.default",
                            "ge_op_type": "RmsNorm",
                        },
                        {
                            "torch_target": "npu.npu_scatter_nd_update_.default",
                            "ge_op_type": "ScatterNdUpdate",
                            "minimum_occurrences": 0,
                        },
                    ]
                },
                "custom_op_audit": [
                    {
                        "status": "PASS",
                        "torch_target": "npu.adn_rms_norm.default",
                        "ge_op_type": "RmsNorm",
                        "minimum_occurrences": 1,
                        "converter_policy": "framework-registered-ge-ir",
                        "converter_calls": 1,
                        "ge_node_occurrences": 1,
                    },
                    {
                        "status": "PASS",
                        "torch_target": "npu.npu_scatter_nd_update_.default",
                        "ge_op_type": "ScatterNdUpdate",
                        "minimum_occurrences": 0,
                        "converter_policy": "torchair-builtin",
                        "converter_calls": None,
                        "ge_node_occurrences": 0,
                    },
                ],
                "air": {
                    "path": "air/quant_dflash_recompute/quant_dflash_recompute.air",
                    "bytes": air.stat().st_size,
                    "sha256": sha256_file(air),
                },
                "payload_files": [
                    {
                        "path": "air/quant_dflash_recompute/quant_dflash_recompute.air",
                        "bytes": air.stat().st_size,
                        "sha256": sha256_file(air),
                    },
                    {
                        "path": "air/quant_dflash_recompute/dynamo.pbtxt",
                        "bytes": pbtxt.stat().st_size,
                        "sha256": sha256_file(pbtxt),
                    },
                ],
            }
        ],
    }
    manifest_path = bundle / "air-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    atc = tmp_path / "atc"
    atc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    atc.chmod(0o755)
    commands: list[list[str]] = []

    def runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        output = next(item.split("=", 1)[1] for item in command if item.startswith("--output="))
        Path(output + ".om").write_bytes(b"quant-om")
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    result = compile_air_bundle(
        manifest_path,
        soc_version="Ascend310P3",
        atc_bin=atc,
        runner=runner,
        atc_identity="fake-atc",
    )
    assert result["status"] == "PASS"
    assert len(result["graphs"][0]["custom_op_audit"]) == 2
    assert result["graphs"][0]["custom_op_audit"][0]["status"] == "PASS"
    assert commands[0][1:3] == ["--mode=0", "--framework=1"]
    assert commands[0][-1] == "--soc_version=Ascend310P3"


def test_quant_input_manifest_hashes_and_rechecks_external_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    quant = tmp_path / "quant"
    receiver = tmp_path / "receiver" / "models"
    for directory in (target, draft, quant, receiver):
        directory.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"target")
    (draft / "config.json").write_text("{}", encoding="utf-8")
    (draft / "model.safetensors").write_bytes(b"draft")
    (quant / "data-00001.safetensors").write_bytes(b"quant")
    embedding_weight = tmp_path / "embedding.bin"
    embedding_scale = tmp_path / "embedding-scale.bin"
    embedding_weight.write_bytes(b"weight")
    embedding_scale.write_bytes(b"scale")
    (receiver / "export_model_wrapper_qwen3_5.py").write_text(
        "# receiver\n", encoding="utf-8"
    )
    quant_config = tmp_path / "quant.yaml"
    quant_config.write_text(
        "\n".join(
            (
                f"quanted_pth: {quant}",
                f"embedding_weight_path: {embedding_weight}",
                f"embedding_scale_path: {embedding_scale}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = run_dir / "quant-input-manifest.json"
    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    built = build_quant_input_manifest(
        target_dir=target,
        draft_dir=draft,
        quant_config=quant_config,
        receiver_models_dir=receiver,
        output=output,
    )
    verified = verify_quant_input_manifest(output)
    assert verified["manifest_sha256"] == built["manifest_sha256"]
    assert verified["roots"]["target_checkpoint"] == target.resolve()
    (target / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        verify_quant_input_manifest(output)


def test_quant_factory_builds_graph_from_quant_branch_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    quant_dir = tmp_path / "quant"
    receiver = tmp_path / "receiver" / "models"
    for directory in (target_dir, draft_dir, quant_dir, receiver):
        directory.mkdir(parents=True)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    (quant_dir / "data.safetensors").write_bytes(b"q")
    embedding_weight = tmp_path / "embedding.bin"
    embedding_scale = tmp_path / "embedding-scale.bin"
    embedding_weight.write_bytes(b"w")
    embedding_scale.write_bytes(b"s")
    (receiver / "export_model_wrapper_qwen3_5.py").write_text(
        "# receiver\n", encoding="utf-8"
    )
    quant_config = tmp_path / "quant.yaml"
    quant_config.write_text(
        f"quanted_pth: {quant_dir}\n"
        f"embedding_weight_path: {embedding_weight}\n"
        f"embedding_scale_path: {embedding_scale}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    manifest = run_dir / "inputs.json"
    build_quant_input_manifest(
        target_dir=target_dir,
        draft_dir=draft_dir,
        quant_config=quant_config,
        receiver_models_dir=receiver,
        output=manifest,
    )

    import models
    from qwen35_dflash.ascend310p import quant_factory
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec
    from qwen35_dflash.ascend310p.integrated import IntegratedDFlashRecomputeGraph
    from models.dflash_v1 import modeling_dflash
    from models import internal_dflash_bridge

    monkeypatch.setattr(models, "__path__", list(models.__path__))
    fake_target = _FakeQuantTarget().eval()
    fake_draft = _FakeDraft().eval()
    monkeypatch.setattr(
        internal_dflash_bridge,
        "load_qwen35_target",
        lambda *args, **kwargs: fake_target,
    )
    monkeypatch.setattr(
        modeling_dflash.DFlashDraftModel,
        "from_pretrained",
        lambda *args, **kwargs: fake_draft,
    )
    def cpu_graph_spec(
        target_model: nn.Module,
        draft_model: nn.Module,
        **kwargs: object,
    ) -> AirGraphSpec:
        graph = IntegratedDFlashRecomputeGraph(target_model, draft_model).eval()
        length = int(kwargs["max_sequence_length"])
        ids = torch.zeros((1, length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        mask[:, : int(kwargs["example_sequence_length"])] = 1
        return AirGraphSpec(
            name=str(kwargs["name"]),
            role="generation-recompute",
            model=graph,
            example_args=(ids, mask),
            input_names=("input_ids", "attention_mask"),
            output_names=("target_top1", "draft_top1"),
            metadata=dict(kwargs["metadata"]),
            custom_ops=tuple(kwargs["custom_ops"]),
        )

    monkeypatch.setattr(
        quant_factory,
        "integrated_recompute_graph_spec",
        cpu_graph_spec,
    )
    monkeypatch.setattr(
        quant_factory,
        "enable_padded_draft_context",
        lambda model: model,
    )
    real_torch_device = torch.device
    monkeypatch.setattr(
        quant_factory.torch,
        "device",
        lambda value: (
            real_torch_device("cpu")
            if str(value).startswith("npu")
            else real_torch_device(value)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    specs = create_quant_recompute_graph(
        {
            "target_dir": str(target_dir),
            "draft_dir": str(draft_dir),
            "quant_config": str(quant_config),
            "input_manifest": str(manifest),
            "receiver_models_dir": str(receiver),
            "max_sequence_length": 64,
            "example_sequence_length": 2,
            "dtype": "float16",
            "device": "npu:0",
        }
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "quant_dflash_recompute"
    assert spec.metadata["target_quant_mode"] == "w8a8_dynamic"
    assert spec.metadata["gdr_effective_length_contract"] == (
        "INT16[B] call-local valid rows derived from attention_mask"
    )
    assert spec.metadata["quant_source_lock"]["verified_file_count"] >= 10
    assert spec.metadata["target_checkpoint_manifest_sha256"]
    assert len(spec.custom_ops) == 7
    assert {
        item.torch_target: (item.ge_op_type, item.minimum_occurrences)
        for item in spec.custom_ops
    } == {
        "npu.npu_dynamic_quant.default": ("DynamicQuant", 1),
        "npu.npu_quant_matmul.default": ("QuantBatchMatmulV4444", 1),
        "npu.adn_rms_norm.default": ("RmsNorm", 1),
        "npu.npu_chunk_gated_delta_rule.default": (
            "ChunkGatedDeltaRule",
            1,
        ),
        "qwen35_dflash.npu_cache_update.default": ("CacheUpdate", 1),
        "npu.adn_fused_infer_attention.default": (
            "FusedInferAttentionScore",
            1,
        ),
        "npu.npu_scatter_nd_update_.default": ("ScatterNdUpdate", 0),
    }
    assert len(spec.metadata["custom_op_export_contracts"]) == 7
    assert spec.input_names == ("input_ids", "attention_mask")
    assert spec.output_names == ("target_top1", "draft_top1")


def test_cpp_runtime_is_bundled_under_quant_branch() -> None:
    source = ROOT / "framework" / "runtime" / "cpp"
    assert (source / "CMakeLists.txt").is_file()
    assert (source / "src" / "acl_executor.cpp").is_file()
    text = (source / "src" / "acl_executor.cpp").read_text(encoding="utf-8")
    assert "aclmdlExecuteAsync" in text
    assert "aclrtSynchronizeStream" in text
