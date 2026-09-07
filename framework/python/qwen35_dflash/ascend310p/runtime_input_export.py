"""Preserve the public tensor ABI across TorchDynamo/TorchAir capture.

Logical argument names do not control Dynamo's placeholder order. Match the
actual serialization inputs to the original tensors before weight conversion,
then order surviving Data/RefData nodes by the public ABI. No scalar values
are guessed: Python floats are specialized by Dynamo, and any remaining
unbound symbol or tensor makes export fail before an AIR is published.
"""

from __future__ import annotations

from contextlib import contextmanager
import copy
import importlib
from typing import Any, Iterator, Sequence

import torch

from .torchair_compat import _PATCH_LOCK, _indexed_graph_inputs


def _tensor_identity(value: Any) -> tuple[Any, ...] | None:
    if not isinstance(value, torch.Tensor) or value.device.type == "meta":
        return None
    return (
        str(value.device), value.dtype, value.untyped_storage().data_ptr(),
        value.storage_offset(), tuple(value.shape), tuple(value.stride()),
    )


def _public_bindings(
    inputs: Sequence[Any], graph: Any, weight_names: dict[int, str],
    public_inputs: Sequence[torch.Tensor], public_names: Sequence[str],
) -> list[tuple[int, Any]]:
    keys = [_tensor_identity(value) for value in public_inputs]
    if any(key is None for key in keys) or len(set(keys)) != len(keys):
        raise RuntimeError("public AIR inputs must be distinct concrete tensor views")
    by_identity = {key: index for index, key in enumerate(keys)}
    resolved: dict[int, Any] = {}
    for runtime_index, node in _indexed_graph_inputs(graph).items():
        if runtime_index < 0 or runtime_index >= len(inputs):
            raise RuntimeError(f"AIR Data index {runtime_index} is outside runtime inputs")
        value = inputs[runtime_index]
        if id(value) in weight_names:
            continue
        public_index = by_identity.get(_tensor_identity(value))
        if public_index is None:
            shape = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise RuntimeError(
                "unbound AIR runtime input: "
                f"index={runtime_index}, name={node.name!r}, "
                f"dtype={getattr(value, 'dtype', type(value).__name__)}, shape={shape}; "
                "refusing to freeze or fill an unknown scalar/tensor"
            )
        if public_index in resolved:
            raise RuntimeError(f"multiple AIR Data nodes bind {public_names[public_index]!r}")
        resolved[public_index] = node
    if len(resolved) != len(public_inputs):
        missing = [name for index, name in enumerate(public_names) if index not in resolved]
        raise RuntimeError(f"public AIR tensor inputs disappeared during capture: {missing}")
    return sorted(resolved.items())


def _normalize_public_nodes(graph: Any, bindings: list[tuple[int, Any]]) -> None:
    positions = [i for i, op in enumerate(graph.op) if op.type in {"Data", "RefData"}]
    if len(positions) != len(bindings):
        raise RuntimeError("AIR runtime input count changed during weight conversion")
    copies = []
    for public_index, node in bindings:
        if node.type not in {"Data", "RefData"}:
            raise RuntimeError("weight conversion overwrote a public AIR input")
        node.attr["index"].i = public_index
        copies.append(copy.deepcopy(node))
    # Preserve every name/edge and every non-Data node. Also canonicalize the
    # physical Data order for serializers that walk nodes rather than indexes.
    for position, node in zip(positions, copies):
        graph.op[position].Clear()
        graph.op[position].MergeFrom(node)


@contextmanager
def canonical_runtime_input_abi(
    torchair: Any, *, public_inputs: Sequence[torch.Tensor],
    public_names: Sequence[str], explicit_test_double: bool = False,
) -> Iterator[dict[str, Any]]:
    audit: dict[str, Any] = {
        "policy": "public-tensor-storage-identity-v1",
        "status": "ARMED",
        "python_float_policy": "dynamo-specialize-float",
        "logical_input_names": list(public_names),
        "bindings": [],
        "calls": 0,
    }
    if explicit_test_double:
        audit["status"] = "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE"
        yield audit
        return
    if len(public_inputs) != len(public_names) or not public_names:
        raise ValueError("AIR public names must describe every positional tensor input")
    if len(set(public_names)) != len(public_names):
        raise ValueError("AIR public input names must be unique")
    if torchair is not importlib.import_module("torchair"):
        raise RuntimeError("public AIR ABI normalization requires canonical TorchAir")
    export_utils = importlib.import_module("torchair._utils.export_utils")
    dynamo_config = importlib.import_module("torch._dynamo.config")
    if not hasattr(dynamo_config, "specialize_float"):
        raise RuntimeError("this TorchDynamo lacks the required specialize_float policy")

    with _PATCH_LOCK:
        original = export_utils._convert_data_to_const

        def convert(inputs, export_graph, file_path, weight_name):
            bindings = _public_bindings(
                inputs, export_graph, weight_name, public_inputs, public_names,
            )
            records = [
                {"index": index, "logical_name": public_names[index],
                 "data_node_name": node.name,
                 "dtype": str(public_inputs[index].dtype).removeprefix("torch."),
                 "example_shape": list(public_inputs[index].shape)}
                for index, node in bindings
            ]
            result = original(inputs, export_graph, file_path, weight_name)
            _normalize_public_nodes(export_graph, bindings)
            audit["calls"] += 1
            audit["bindings"] = records
            return result

        export_utils._convert_data_to_const = convert
        try:
            # Specializes Python model attributes, not tensor dimensions or
            # data-derived values. Tensor/SymInt shapes remain dynamic.
            with dynamo_config.patch(specialize_float=True):
                yield audit
            if audit["calls"] != 1:
                raise RuntimeError("expected one canonical public ABI serialization per AIR graph")
            audit["status"] = "PASS"
        finally:
            export_utils._convert_data_to_const = original
