"""Exact Fake/Meta and TorchAir support for the Target's NPU operators.

Fake kernels in this module describe tensor metadata only.  They never run the
operator numerics and therefore cannot replace an NPU kernel.  Operators that
TorchAir already supports keep their built-in converter; receiver-private
operators get one explicit converter to a registered GE IR.  The exporter
audits the resulting ``dynamo.pbtxt`` before declaring an AIR bundle passing.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import re
import threading
from typing import Any, Callable, Literal, Sequence

import torch

from .contracts import CustomOpExportSpec


ADN_FUSED_INFER_ATTENTION_TORCH_OP = "npu::adn_fused_infer_attention"
ADN_RMS_NORM_TORCH_OP = "npu::adn_rms_norm"
NPU_CACHE_UPDATE_TORCH_OP = "npu::npu_cache_update_"
NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP = "npu::npu_chunk_gated_delta_rule"
NPU_DYNAMIC_QUANT_TORCH_OP = "npu::npu_dynamic_quant"
NPU_QUANT_MATMUL_TORCH_OP = "npu::npu_quant_matmul"
NPU_SCATTER_ND_UPDATE_TORCH_OP = "npu::npu_scatter_nd_update_"

ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE = "FusedInferAttentionScore"
ADN_RMS_NORM_DEFAULT_GE_OP_TYPE = "RmsNorm"
NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE = "CacheUpdate"
NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE = "ChunkGatedDeltaRule"
NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE = "DynamicQuant"
NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE = "QuantBatchMatmulV3"
NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE = "ScatterNdUpdate"

_GE_TYPE_FIELD = re.compile(r'\btype:\s*"([A-Za-z_][A-Za-z0-9_]*)"')
_FAKE_REGISTRATION_LOCK = threading.Lock()
_FRAMEWORK_CONVERTER = "framework-registered-ge-ir"
_TORCHAIR_BUILTIN_CONVERTER = "torchair-builtin"


@dataclass(frozen=True)
class _OperatorAdapter:
    torch_op: str
    argument_names: tuple[str | tuple[str, ...], ...]
    argument_types: tuple[str, ...]
    kwarg_only: tuple[bool, ...]
    return_types: tuple[str, ...]
    fake_kernel: Callable[..., Any]
    validate_meta: Callable[[Any], None]
    converter_policy: Literal[
        "framework-registered-ge-ir", "torchair-builtin"
    ]
    inplace_alias: bool = False


@dataclass
class CustomOpExportSession:
    """Evidence collected while one front-end operator is lowered to GE."""

    spec: CustomOpExportSpec
    schema: str
    fake_kernel: str
    converter_mode: str
    converter_policy: str
    converter_calls: int = 0


def _fake_adn_rms_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the receiver-observed RMSNorm metadata."""

    del gamma, epsilon
    output = torch.empty_like(input)
    rstd = input.new_empty((*input.shape[:-1], 1), dtype=torch.float32)
    return output, rstd


def _fake_npu_chunk_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    effective_length: torch.Tensor,
    chunk_size: int = 64,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Describe the locked Qwen3.5 GDR output and FP32 recurrent state."""

    del g, beta, effective_length, chunk_size, use_qk_l2norm_in_kernel
    if not output_final_state:
        raise RuntimeError(
            "the locked Qwen3.5 GDR export requires output_final_state=True"
        )
    output = value.new_empty(value.shape, dtype=query.dtype)
    if initial_state is None:
        final_shape = (
            query.shape[0],
            query.shape[2],
            key.shape[-1],
            value.shape[-1],
        )
        final_state = query.new_empty(final_shape, dtype=torch.float32)
    else:
        final_state = initial_state.new_empty(
            initial_state.shape,
            dtype=torch.float32,
        )
    return output, final_state


def _first_tensor(values: Sequence[torch.Tensor], name: str) -> torch.Tensor:
    if not values:
        raise RuntimeError(f"{name} must contain at least one tensor")
    return values[0]


def _fake_adn_fused_infer_attention(
    query: torch.Tensor,
    key: Sequence[torch.Tensor],
    value: Sequence[torch.Tensor],
    *,
    pse_shift: torch.Tensor | None = None,
    atten_mask: torch.Tensor | None = None,
    all_seq_lengths_q: Sequence[int] | None = None,
    actual_seq_lengths_q: Sequence[int] | None = None,
    actual_seq_lengths_kv: Sequence[int] | None = None,
    dequant_scale1: torch.Tensor | None = None,
    quant_scale1: torch.Tensor | None = None,
    dequant_scale2: torch.Tensor | None = None,
    quant_scale2: torch.Tensor | None = None,
    quant_offset2: torch.Tensor | None = None,
    antiquant_scale: torch.Tensor | None = None,
    antiquant_offset: torch.Tensor | None = None,
    block_table: torch.Tensor | None = None,
    kv_padding_size: torch.Tensor | None = None,
    num_heads: int = 1,
    scale_value: float = 1.0,
    input_layout: str = "BSH",
    num_key_value_heads: int = 0,
    block_size: int = 0,
    inner_precise: int = 1,
) -> torch.Tensor:
    """Mirror the supported FusedInferAttentionScore output metadata."""

    del (
        pse_shift,
        atten_mask,
        all_seq_lengths_q,
        actual_seq_lengths_q,
        actual_seq_lengths_kv,
        dequant_scale1,
        quant_scale1,
        dequant_scale2,
        quant_offset2,
        antiquant_scale,
        antiquant_offset,
        kv_padding_size,
        scale_value,
        num_key_value_heads,
        block_size,
        inner_precise,
    )
    value_tensor = _first_tensor(value, "value")
    _first_tensor(key, "key")
    if input_layout in {"BSH", "BSND", "NSD", "TND"}:
        output_shape = query.shape
    elif input_layout == "BNSD":
        if query.dim() != 4:
            raise RuntimeError("BNSD fused attention query must be rank 4")
        output_shape = (
            query.shape
            if block_table is not None
            else (*query.shape[:-1], value_tensor.shape[-1])
        )
    elif input_layout == "BNSD_BSND":
        output_shape = (
            query.shape[0], query.shape[2], query.shape[1], query.shape[3]
        )
    elif input_layout == "BNSD_NBSD":
        output_shape = (
            query.shape[1], query.shape[0], query.shape[2], query.shape[3]
        )
    elif input_layout == "BSND_NBSD":
        output_shape = (
            query.shape[2], query.shape[0], query.shape[1], query.shape[3]
        )
    elif input_layout == "BSH_NBSD":
        output_shape = (
            num_heads,
            query.shape[0],
            query.shape[1],
            query.shape[2] // num_heads,
        )
    elif input_layout == "TND_NTD":
        output_shape = (query.shape[1], query.shape[0], query.shape[2])
    elif input_layout == "NTD_TND":
        output_shape = (query.shape[1], query.shape[0], value_tensor.shape[2])
    else:
        raise RuntimeError(
            f"unsupported adn_fused_infer_attention input_layout: {input_layout!r}"
        )
    if quant_scale2 is not None:
        output_dtype = torch.int8
    elif query.dtype == torch.int8:
        output_dtype = torch.float16
    else:
        output_dtype = query.dtype
    return query.new_empty(output_shape, dtype=output_dtype)


def _fake_npu_cache_update_(
    input: torch.Tensor,
    updates: torch.Tensor,
    target_block: torch.Tensor,
    offset_in_block: torch.Tensor,
) -> torch.Tensor:
    """Preserve the mutable ``Tensor(a!) -> Tensor(a!)`` alias."""

    del updates, target_block, offset_in_block
    return input


def _fake_npu_dynamic_quant(
    input: torch.Tensor,
    *,
    smooth_scales: torch.Tensor | None = None,
    group_index: torch.Tensor | None = None,
    dst_type: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fallback for the current per-token INT8 dynamic-quant route."""

    del smooth_scales, group_index
    if dst_type not in {None, torch.int8}:
        raise RuntimeError("the locked W8A8 route requires dynamic-quant INT8 output")
    output = input.new_empty(input.shape, dtype=torch.int8)
    scale = input.new_empty(input.shape[:-1], dtype=torch.float32)
    return output, scale


def _fake_npu_quant_matmul(
    x1: torch.Tensor,
    x2: torch.Tensor,
    scale: torch.Tensor,
    *,
    offset: torch.Tensor | None = None,
    pertoken_scale: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    output_dtype: torch.dtype | None = None,
    group_sizes: Sequence[int] | None = None,
) -> torch.Tensor:
    """Fallback metadata for the locked A8W8 matmul invocation."""

    del scale, offset, pertoken_scale, bias, group_sizes
    if x1.dim() < 2 or x2.dim() < 2:
        raise RuntimeError("npu_quant_matmul inputs must be rank 2 or greater")
    batch_shape = torch.broadcast_shapes(x1.shape[:-2], x2.shape[:-2])
    output_shape = (*batch_shape, x1.shape[-2], x2.shape[-1])
    dtype = torch.int8 if output_dtype is None else output_dtype
    if dtype not in {torch.int8, torch.int32, torch.float16, torch.bfloat16}:
        raise RuntimeError(f"unsupported npu_quant_matmul output dtype: {dtype}")
    return x1.new_empty(output_shape, dtype=dtype)


def _fake_npu_scatter_nd_update_(
    input: torch.Tensor,
    indices: torch.Tensor,
    updates: torch.Tensor,
) -> torch.Tensor:
    """Preserve the mutable ScatterNdUpdate alias."""

    del indices, updates
    return input


def _expect_tensor(
    value: Any,
    *,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    label: str,
) -> None:
    if (
        not isinstance(value, torch.Tensor)
        or tuple(value.shape) != shape
        or value.dtype != dtype
        or value.device.type != "meta"
    ):
        raise RuntimeError(f"{label} Meta contract mismatch")


def _validate_adn_rms_norm_meta(operation: Any) -> None:
    for dtype in (torch.float16, torch.float32):
        input_tensor = torch.empty((2, 3, 8), dtype=dtype, device="meta")
        gamma = torch.empty((8,), dtype=dtype, device="meta")
        result = operation(input_tensor, gamma, 1e-6)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError(
                "npu::adn_rms_norm Meta kernel must return two tensors"
            )
        _expect_tensor(
            result[0], shape=(2, 3, 8), dtype=dtype,
            label="npu::adn_rms_norm output[0]",
        )
        _expect_tensor(
            result[1], shape=(2, 3, 1), dtype=torch.float32,
            label="npu::adn_rms_norm output[1]",
        )


def _validate_npu_chunk_gated_delta_rule_meta(operation: Any) -> None:
    query = torch.empty((1, 64, 32, 128), dtype=torch.float16, device="meta")
    key = torch.empty_like(query)
    value = torch.empty_like(query)
    gate = torch.empty((1, 64, 32), dtype=torch.float32, device="meta")
    beta = torch.empty((1, 64, 32), dtype=torch.float16, device="meta")
    effective_length = torch.empty((1,), dtype=torch.int16, device="meta")
    state = torch.empty((1, 32, 128, 128), dtype=torch.float32, device="meta")
    result = operation(
        query, key, value, gate, beta, effective_length, 64, state, True, True
    )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError(
            "npu::npu_chunk_gated_delta_rule Meta kernel must return two tensors"
        )
    _expect_tensor(
        result[0], shape=(1, 64, 32, 128), dtype=torch.float16,
        label="npu::npu_chunk_gated_delta_rule output[0]",
    )
    _expect_tensor(
        result[1], shape=(1, 32, 128, 128), dtype=torch.float32,
        label="npu::npu_chunk_gated_delta_rule output[1]",
    )


def _validate_adn_fused_infer_attention_meta(operation: Any) -> None:
    query = torch.empty((1, 256, 3, 16), dtype=torch.float16, device="meta")
    key = torch.empty((1, 64, 64, 16), dtype=torch.float16, device="meta")
    value = torch.empty_like(key)
    mask = torch.empty((1, 1, 3, 64), dtype=torch.float16, device="meta")
    block_table = torch.empty((1, 1), dtype=torch.int32, device="meta")
    result = operation(
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
    _expect_tensor(
        result, shape=(1, 256, 3, 16), dtype=torch.float16,
        label="npu::adn_fused_infer_attention output",
    )


def _validate_npu_cache_update_meta(operation: Any) -> None:
    input_tensor = torch.empty((4, 64, 64, 16), dtype=torch.float16, device="meta")
    updates = torch.empty((1, 64, 16), dtype=torch.float16, device="meta")
    target_block = torch.empty((1,), dtype=torch.int32, device="meta")
    offset = torch.empty((), dtype=torch.int32, device="meta")
    result = operation(input_tensor, updates, target_block, offset)
    _expect_tensor(
        result, shape=(4, 64, 64, 16), dtype=torch.float16,
        label="npu::npu_cache_update_ output",
    )
    if result is not input_tensor:
        raise RuntimeError("npu::npu_cache_update_ Meta kernel lost input alias")


def _validate_npu_dynamic_quant_meta(operation: Any) -> None:
    input_tensor = torch.empty((2, 3, 8), dtype=torch.float16, device="meta")
    result = operation(input_tensor, dst_type=torch.int8)
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise RuntimeError("npu::npu_dynamic_quant Meta kernel must return two tensors")
    _expect_tensor(
        result[0], shape=(2, 3, 8), dtype=torch.int8,
        label="npu::npu_dynamic_quant output[0]",
    )
    _expect_tensor(
        result[1], shape=(2, 3), dtype=torch.float32,
        label="npu::npu_dynamic_quant output[1]",
    )


def _validate_npu_quant_matmul_meta(operation: Any) -> None:
    x1 = torch.empty((2, 3, 8), dtype=torch.int8, device="meta")
    x2 = torch.empty((8, 5), dtype=torch.int8, device="meta")
    scale = torch.empty((5,), dtype=torch.float32, device="meta")
    pertoken = torch.empty((6,), dtype=torch.float32, device="meta")
    result = operation(
        x1, x2, scale,
        pertoken_scale=pertoken,
        output_dtype=torch.float16,
    )
    _expect_tensor(
        result, shape=(2, 3, 5), dtype=torch.float16,
        label="npu::npu_quant_matmul output",
    )


def _validate_npu_scatter_nd_update_meta(operation: Any) -> None:
    input_tensor = torch.empty((8, 2, 4), dtype=torch.float16, device="meta")
    indices = torch.empty((2,), dtype=torch.int64, device="meta")
    updates = torch.empty((2, 2, 4), dtype=torch.float16, device="meta")
    result = operation(input_tensor, indices, updates)
    _expect_tensor(
        result, shape=(8, 2, 4), dtype=torch.float16,
        label="npu::npu_scatter_nd_update_ output",
    )
    if result is not input_tensor:
        raise RuntimeError("npu::npu_scatter_nd_update_ Meta kernel lost input alias")


_ADAPTERS = {
    ADN_RMS_NORM_TORCH_OP: _OperatorAdapter(
        torch_op=ADN_RMS_NORM_TORCH_OP,
        argument_names=(("input", "self"), "gamma", "epsilon"),
        argument_types=("Tensor", "Tensor", "float"),
        kwarg_only=(False, False, False),
        return_types=("Tensor", "Tensor"),
        fake_kernel=_fake_adn_rms_norm,
        validate_meta=_validate_adn_rms_norm_meta,
        converter_policy=_FRAMEWORK_CONVERTER,
    ),
    NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP: _OperatorAdapter(
        torch_op=NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
        argument_names=(
            "query", "key", "value", "g", "beta", "effective_length",
            "chunk_size", "initial_state", "output_final_state",
            "use_qk_l2norm_in_kernel",
        ),
        argument_types=(
            "Tensor", "Tensor", "Tensor", "Tensor", "Tensor", "Tensor",
            "int", "Optional[Tensor]", "bool", "bool",
        ),
        kwarg_only=(False,) * 10,
        return_types=("Tensor", "Tensor"),
        fake_kernel=_fake_npu_chunk_gated_delta_rule,
        validate_meta=_validate_npu_chunk_gated_delta_rule_meta,
        converter_policy=_FRAMEWORK_CONVERTER,
    ),
    ADN_FUSED_INFER_ATTENTION_TORCH_OP: _OperatorAdapter(
        torch_op=ADN_FUSED_INFER_ATTENTION_TORCH_OP,
        argument_names=(
            "query", "key", "value", "pse_shift", "atten_mask",
            "all_seq_lengths_q", "actual_seq_lengths_q",
            "actual_seq_lengths_kv", "dequant_scale1", "quant_scale1",
            "dequant_scale2", "quant_scale2", "quant_offset2",
            "antiquant_scale", "antiquant_offset", "block_table",
            "kv_padding_size", "num_heads", "scale_value", "input_layout",
            "num_key_value_heads", "block_size", "inner_precise",
        ),
        argument_types=(
            "Tensor", "List[Tensor]", "List[Tensor]", "Optional[Tensor]",
            "Optional[Tensor]", "Optional[List[int]]", "Optional[List[int]]",
            "Optional[List[int]]", "Optional[Tensor]", "Optional[Tensor]",
            "Optional[Tensor]", "Optional[Tensor]", "Optional[Tensor]",
            "Optional[Tensor]", "Optional[Tensor]", "Optional[Tensor]",
            "Optional[Tensor]", "int", "float", "str", "int", "int", "int",
        ),
        kwarg_only=(False, False, False) + (True,) * 20,
        return_types=("Tensor",),
        fake_kernel=_fake_adn_fused_infer_attention,
        validate_meta=_validate_adn_fused_infer_attention_meta,
        converter_policy=_FRAMEWORK_CONVERTER,
    ),
    NPU_CACHE_UPDATE_TORCH_OP: _OperatorAdapter(
        torch_op=NPU_CACHE_UPDATE_TORCH_OP,
        argument_names=(
            ("input", "self"), "updates", "target_block", "offset_in_block"
        ),
        argument_types=("Tensor", "Tensor", "Tensor", "Tensor"),
        kwarg_only=(False,) * 4,
        return_types=("Tensor",),
        fake_kernel=_fake_npu_cache_update_,
        validate_meta=_validate_npu_cache_update_meta,
        converter_policy=_FRAMEWORK_CONVERTER,
        inplace_alias=True,
    ),
    NPU_DYNAMIC_QUANT_TORCH_OP: _OperatorAdapter(
        torch_op=NPU_DYNAMIC_QUANT_TORCH_OP,
        argument_names=(
            ("input", "input_data", "input_dummy"), "smooth_scales",
            "group_index", "dst_type",
        ),
        argument_types=(
            "Tensor", "Optional[Tensor]", "Optional[Tensor]", "Optional[int]"
        ),
        kwarg_only=(False, True, True, True),
        return_types=("Tensor", "Tensor"),
        fake_kernel=_fake_npu_dynamic_quant,
        validate_meta=_validate_npu_dynamic_quant_meta,
        converter_policy=_TORCHAIR_BUILTIN_CONVERTER,
    ),
    NPU_QUANT_MATMUL_TORCH_OP: _OperatorAdapter(
        torch_op=NPU_QUANT_MATMUL_TORCH_OP,
        argument_names=(
            "x1", "x2", "scale", "offset", "pertoken_scale", "bias",
            "output_dtype", "group_sizes",
        ),
        argument_types=(
            "Tensor", "Tensor", "Tensor", "Optional[Tensor]",
            "Optional[Tensor]", "Optional[Tensor]", "Optional[int]",
            "Optional[List[int]]",
        ),
        kwarg_only=(False, False, False, True, True, True, True, True),
        return_types=("Tensor",),
        fake_kernel=_fake_npu_quant_matmul,
        validate_meta=_validate_npu_quant_matmul_meta,
        converter_policy=_TORCHAIR_BUILTIN_CONVERTER,
    ),
    NPU_SCATTER_ND_UPDATE_TORCH_OP: _OperatorAdapter(
        torch_op=NPU_SCATTER_ND_UPDATE_TORCH_OP,
        argument_names=(("input", "self"), "indices", "updates"),
        argument_types=("Tensor", "Tensor", "Tensor"),
        kwarg_only=(False, False, False),
        return_types=("Tensor",),
        fake_kernel=_fake_npu_scatter_nd_update_,
        validate_meta=_validate_npu_scatter_nd_update_meta,
        converter_policy=_TORCHAIR_BUILTIN_CONVERTER,
        inplace_alias=True,
    ),
}


def _resolve_operation(spec: CustomOpExportSpec) -> Any:
    namespace, operation_name = spec.torch_op.split("::", 1)
    namespace_object = getattr(torch.ops, namespace, None)
    if namespace_object is None:
        raise RuntimeError(
            f"required custom-operator namespace is not registered: {namespace}"
        )
    packet = getattr(namespace_object, operation_name, None)
    if packet is None:
        raise RuntimeError(f"required custom operator is not registered: {spec.torch_op}")
    operation = getattr(packet, spec.overload, None)
    if operation is None:
        raise RuntimeError(
            f"required custom-operator overload is not registered: {spec.torch_target}"
        )
    return operation


def _argument_name_matches(actual: str, expected: str | tuple[str, ...]) -> bool:
    return actual == expected if isinstance(expected, str) else actual in expected


def _validate_inplace_alias(schema: Any, torch_op: str) -> None:
    input_alias = schema.arguments[0].alias_info
    output_alias = schema.returns[0].alias_info
    if (
        input_alias is None
        or output_alias is None
        or not input_alias.is_write
        or not output_alias.is_write
        or not input_alias.after_set
        or input_alias.after_set != output_alias.after_set
    ):
        raise RuntimeError(
            f"{torch_op} must retain one writable input/output alias: {schema}"
        )


def _validate_schema(operation: Any, adapter: _OperatorAdapter) -> str:
    schema = getattr(operation, "_schema", None)
    if schema is None:
        raise RuntimeError(f"{adapter.torch_op} does not expose a dispatcher schema")
    names = tuple(item.name for item in schema.arguments)
    types = tuple(str(item.type) for item in schema.arguments)
    kwarg_only = tuple(bool(item.kwarg_only) for item in schema.arguments)
    return_types = tuple(str(item.type) for item in schema.returns)
    names_match = len(names) == len(adapter.argument_names) and all(
        _argument_name_matches(actual, expected)
        for actual, expected in zip(names, adapter.argument_names)
    )
    if (
        getattr(schema, "name", None) != adapter.torch_op
        or not names_match
        or types != adapter.argument_types
        or kwarg_only != adapter.kwarg_only
        or return_types != adapter.return_types
    ):
        raise RuntimeError(
            f"{adapter.torch_op} schema drifted from the locked export contract: {schema}"
        )
    if adapter.inplace_alias:
        _validate_inplace_alias(schema, adapter.torch_op)
    return str(schema)


def _has_meta_kernel(torch_op: str) -> bool:
    query = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if not callable(query):
        raise RuntimeError("this PyTorch build cannot query custom-op Meta kernels")
    return bool(query(torch_op, "Meta"))


def _ensure_fake(adapter: _OperatorAdapter, operation: Any) -> str:
    with _FAKE_REGISTRATION_LOCK:
        if _has_meta_kernel(adapter.torch_op):
            status = "preexisting-meta-kernel"
        else:
            register_fake = getattr(torch.library, "register_fake", None)
            if not callable(register_fake):
                raise RuntimeError(
                    "PyTorch torch.library.register_fake is required to export "
                    f"{adapter.torch_op}"
                )
            register_fake(adapter.torch_op)(adapter.fake_kernel)
            if not _has_meta_kernel(adapter.torch_op):
                raise RuntimeError(
                    f"{adapter.torch_op} Fake registration did not install Meta"
                )
            status = "framework-registered-fake"
    adapter.validate_meta(operation)
    return status


def _custom_op_call_mode(custom_op: Callable[..., Any]) -> str:
    try:
        signature = inspect.signature(custom_op)
    except (TypeError, ValueError):
        return "registered-ir-positional"
    if any(
        parameter.kind is inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    ):
        return "registered-ir-positional"
    return "named-only"


def _require_ge_attrs(ge_api: Any, names: Sequence[str]) -> None:
    attr_api = getattr(ge_api, "attr", None)
    missing = [name for name in names if not callable(getattr(attr_api, name, None))]
    if missing:
        raise RuntimeError(
            "TorchAir lacks GE attribute constructors: " + ", ".join(missing)
        )


def _static_lengths_match(left: Any, right: Any) -> bool:
    if isinstance(left, (tuple, list)) and isinstance(right, (tuple, list)):
        return tuple(left) == tuple(right)
    return left is right


def _ge_int64_tensor(ge_api: Any, value: Any) -> Any:
    if value is None or not isinstance(value, (tuple, list)):
        return value
    const = getattr(ge_api, "Const", None)
    data_type = getattr(getattr(ge_api, "DataType", None), "DT_INT64", None)
    if not callable(const) or data_type is None:
        raise RuntimeError(
            "TorchAir ge.Const/DataType.DT_INT64 is required for fused attention"
        )
    return const(list(value), dtype=data_type)


def _register_framework_converter(
    adapter: _OperatorAdapter,
    operation: Any,
    spec: CustomOpExportSpec,
    torchair_module: Any,
    session: CustomOpExportSession,
) -> None:
    registrar = getattr(torchair_module, "register_fx_node_ge_converter", None)
    ge_api = getattr(torchair_module, "ge", None)
    custom_op = getattr(ge_api, "custom_op", None)
    if not callable(registrar) or not callable(custom_op):
        raise RuntimeError(
            "TorchAir custom-op export requires register_fx_node_ge_converter "
            "and torchair.ge.custom_op"
        )
    call_mode = _custom_op_call_mode(custom_op)
    session.converter_mode = call_mode

    def emit_positional(*args: Any) -> Any:
        if call_mode != "registered-ir-positional":
            raise RuntimeError(
                f"TorchAir positional ge.custom_op is required for {spec.torch_target}"
            )
        session.converter_calls += 1
        return custom_op(spec.ge_op_type, *args)

    if adapter.torch_op == ADN_RMS_NORM_TORCH_OP:
        if call_mode == "named-only":
            _require_ge_attrs(ge_api, ("Float",))

        def converter(
            input: Any,
            gamma: Any,
            epsilon: float = 1e-6,
            meta_outputs: Any = None,
        ) -> Any:
            del meta_outputs
            session.converter_calls += 1
            if call_mode == "registered-ir-positional":
                return custom_op(spec.ge_op_type, input, gamma, epsilon)
            if spec.ge_op_type != ADN_RMS_NORM_DEFAULT_GE_OP_TYPE:
                raise RuntimeError(
                    "named-only TorchAir can lower adn_rms_norm only to RmsNorm"
                )
            return custom_op(
                spec.ge_op_type,
                inputs={"x": input, "gamma": gamma},
                outputs=["y", "rstd"],
                attrs={"epsilon": ge_api.attr.Float(epsilon)},
            )

        converter.__name__ = "convert_npu_adn_rms_norm_default"
    elif adapter.torch_op == NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP:

        def converter(
            query: Any,
            key: Any,
            value: Any,
            g: Any,
            beta: Any,
            effective_length: Any,
            chunk_size: int = 64,
            initial_state: Any = None,
            output_final_state: bool = False,
            use_qk_l2norm_in_kernel: bool = False,
            meta_outputs: Any = None,
        ) -> Any:
            del meta_outputs
            return emit_positional(
                query, key, value, g, beta, effective_length, chunk_size,
                initial_state, output_final_state, use_qk_l2norm_in_kernel,
            )

        converter.__name__ = "convert_npu_chunk_gated_delta_rule_default"
    elif adapter.torch_op == NPU_CACHE_UPDATE_TORCH_OP:

        def converter(
            input: Any,
            updates: Any,
            target_block: Any,
            offset_in_block: Any,
            meta_outputs: Any = None,
        ) -> Any:
            del meta_outputs
            return emit_positional(input, updates, target_block, offset_in_block)

        converter.__name__ = "convert_npu_cache_update_default"
    elif adapter.torch_op == ADN_FUSED_INFER_ATTENTION_TORCH_OP:
        if spec.ge_op_type != ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE:
            raise RuntimeError(
                "adn_fused_infer_attention currently has an exact lowering only to "
                f"{ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE}"
            )
        _require_ge_attrs(ge_api, ("Float", "Int", "Str"))

        def converter(
            query: Any,
            key: Any,
            value: Any,
            *,
            pse_shift: Any = None,
            atten_mask: Any = None,
            all_seq_lengths_q: Any = None,
            actual_seq_lengths_q: Any = None,
            actual_seq_lengths_kv: Any = None,
            dequant_scale1: Any = None,
            quant_scale1: Any = None,
            dequant_scale2: Any = None,
            quant_scale2: Any = None,
            quant_offset2: Any = None,
            antiquant_scale: Any = None,
            antiquant_offset: Any = None,
            block_table: Any = None,
            kv_padding_size: Any = None,
            num_heads: int = 1,
            scale_value: float = 1.0,
            input_layout: str = "BSH",
            num_key_value_heads: int = 0,
            block_size: int = 0,
            inner_precise: int = 1,
            meta_outputs: Any = None,
        ) -> Any:
            del meta_outputs
            if all_seq_lengths_q is not None and not _static_lengths_match(
                all_seq_lengths_q, actual_seq_lengths_q
            ):
                raise RuntimeError(
                    "the recompute AIR route requires all_seq_lengths_q to equal "
                    "actual_seq_lengths_q before lowering to FusedInferAttentionScore"
                )
            actual_q = _ge_int64_tensor(ge_api, actual_seq_lengths_q)
            actual_kv = _ge_int64_tensor(ge_api, actual_seq_lengths_kv)
            session.converter_calls += 1
            result = custom_op(
                spec.ge_op_type,
                inputs={
                    "query": query,
                    "key": key,
                    "value": value,
                    "pse_shift": pse_shift,
                    "atten_mask": atten_mask,
                    "actual_seq_lengths": actual_q,
                    "actual_seq_lengths_kv": actual_kv,
                    "dequant_scale1": dequant_scale1,
                    "quant_scale1": quant_scale1,
                    "dequant_scale2": dequant_scale2,
                    "quant_scale2": quant_scale2,
                    "quant_offset2": quant_offset2,
                    "antiquant_scale": antiquant_scale,
                    "antiquant_offset": antiquant_offset,
                    "block_table": block_table,
                    "query_padding_size": None,
                    "kv_padding_size": kv_padding_size,
                    "key_antiquant_scale": None,
                    "key_antiquant_offset": None,
                    "value_antiquant_scale": None,
                    "value_antiquant_offset": None,
                    "key_shared_prefix": None,
                    "value_shared_prefix": None,
                    "actual_shared_prefix_len": None,
                    "query_rope": None,
                    "key_rope": None,
                    "key_rope_antiquant_scale": None,
                    "dequant_scale_query": None,
                    "learnable_sink": None,
                    "q_start_idx": None,
                    "kv_start_idx": None,
                },
                outputs=["attention_out", "softmax_lse"],
                attrs={
                    "num_heads": ge_api.attr.Int(num_heads),
                    "scale": ge_api.attr.Float(scale_value),
                    "input_layout": ge_api.attr.Str(input_layout),
                    "num_key_value_heads": ge_api.attr.Int(num_key_value_heads),
                    "inner_precise": ge_api.attr.Int(inner_precise),
                    "block_size": ge_api.attr.Int(block_size),
                },
            )
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise RuntimeError(
                    "FusedInferAttentionScore GE IR must return attention_out and softmax_lse"
                )
            return result[0]

        converter.__name__ = "convert_npu_adn_fused_infer_attention_default"
        session.converter_mode = "named-fused-infer-current-recompute-route"
    else:  # pragma: no cover - registry and dispatch are kept exhaustive
        raise NotImplementedError(
            f"no framework converter is implemented for {adapter.torch_op}"
        )

    registrar(operation)(converter)


def prepare_custom_op_export(
    spec: CustomOpExportSpec,
    torchair_module: Any,
) -> CustomOpExportSession:
    """Validate schema/Meta and prepare one exact front-end lowering contract."""

    if spec.overload != "default":
        raise NotImplementedError(
            f"no exact export adapter is implemented for {spec.torch_target}"
        )
    adapter = _ADAPTERS.get(spec.torch_op)
    if adapter is None:
        raise NotImplementedError(
            f"no exact export adapter is implemented for {spec.torch_target}"
        )
    operation = _resolve_operation(spec)
    schema = _validate_schema(operation, adapter)
    fake_kernel = _ensure_fake(adapter, operation)
    session = CustomOpExportSession(
        spec=spec,
        schema=schema,
        fake_kernel=fake_kernel,
        converter_mode=_TORCHAIR_BUILTIN_CONVERTER,
        converter_policy=adapter.converter_policy,
    )
    if adapter.converter_policy == _FRAMEWORK_CONVERTER:
        _register_framework_converter(
            adapter, operation, spec, torchair_module, session
        )
    return session


def audit_custom_op_export(
    sessions: Sequence[CustomOpExportSession],
    graph_dir: Path,
    *,
    relative_to: Path,
) -> list[dict[str, Any]]:
    """Prove required custom operators reached TorchAir's GE IR."""

    if not sessions:
        return []
    pbtxt_paths = sorted(graph_dir.rglob("dynamo.pbtxt"))
    if not pbtxt_paths:
        raise RuntimeError(
            "TorchAir produced no dynamo.pbtxt; custom-op preservation cannot be audited"
        )
    ge_type_counts: dict[str, int] = {}
    for path in pbtxt_paths:
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                for match in _GE_TYPE_FIELD.finditer(line):
                    op_type = match.group(1)
                    ge_type_counts[op_type] = ge_type_counts.get(op_type, 0) + 1

    records: list[dict[str, Any]] = []
    for session in sessions:
        spec = session.spec
        ge_occurrences = ge_type_counts.get(spec.ge_op_type, 0)
        if ge_occurrences < spec.minimum_occurrences:
            raise RuntimeError(
                f"TorchAir IR contains {ge_occurrences} {spec.ge_op_type} nodes for "
                f"{spec.torch_target}; expected at least {spec.minimum_occurrences}"
            )
        converter_calls: int | None
        if session.converter_policy == _FRAMEWORK_CONVERTER:
            converter_calls = session.converter_calls
            if converter_calls < spec.minimum_occurrences:
                raise RuntimeError(
                    f"custom-op converter for {spec.torch_target} ran "
                    f"{converter_calls} times; expected at least "
                    f"{spec.minimum_occurrences}"
                )
            if ge_occurrences < converter_calls:
                raise RuntimeError(
                    f"TorchAir IR contains {ge_occurrences} {spec.ge_op_type} nodes for "
                    f"{converter_calls} converter calls to {spec.torch_target}"
                )
        else:
            converter_calls = None
        observed = ge_occurrences > 0
        records.append(
            {
                "status": "PASS",
                "torch_op": spec.torch_op,
                "torch_target": spec.torch_target,
                "torch_schema": session.schema,
                "fake_kernel": session.fake_kernel,
                "converter_policy": session.converter_policy,
                "converter_mode": session.converter_mode,
                "converter_calls": converter_calls,
                "ge_op_type": spec.ge_op_type,
                "ge_node_occurrences": ge_occurrences,
                "minimum_occurrences": spec.minimum_occurrences,
                "observed_in_graph": observed,
                "preservation": (
                    "one registered GE operator; no Tensor decomposition"
                    if observed
                    else "optional source path absent; metadata contract validated only"
                ),
                "pbtxt_files": [
                    path.relative_to(relative_to).as_posix() for path in pbtxt_paths
                ],
            }
        )
    return records


__all__ = [
    "ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE",
    "ADN_FUSED_INFER_ATTENTION_TORCH_OP",
    "ADN_RMS_NORM_DEFAULT_GE_OP_TYPE",
    "ADN_RMS_NORM_TORCH_OP",
    "NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE",
    "NPU_CACHE_UPDATE_TORCH_OP",
    "NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE",
    "NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP",
    "NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE",
    "NPU_DYNAMIC_QUANT_TORCH_OP",
    "NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE",
    "NPU_QUANT_MATMUL_TORCH_OP",
    "NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE",
    "NPU_SCATTER_ND_UPDATE_TORCH_OP",
    "CustomOpExportSession",
    "audit_custom_op_export",
    "prepare_custom_op_export",
]
