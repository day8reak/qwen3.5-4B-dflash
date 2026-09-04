"""Narrow compatibility fixes for receiver TorchAir AIR serialization.

The receiver TorchAir release converts captured parameters from ``Data``
nodes to ``Const``/``FileConstant`` nodes by indexing ``GraphDef.op`` with the
runtime input ordinal.  That is only valid while all ``Data`` nodes happen to
be the leading protobuf nodes.  Dynamic-shape conversion may emit helper
``Gather``/``Pack`` nodes before later placeholders, so the positional lookup
can overwrite a shape helper while leaving its consumers connected to a model
weight.

This module changes only that lookup: a parameter at runtime input ``i`` is
resolved through the ``Data.index == i`` attribute which TorchAir itself
assigns in ``parse_input``.  Weight bytes, node names, dtypes, shapes, and the
remaining runtime input order are preserved.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import importlib
import inspect
from threading import RLock
from typing import Any, Iterator, Mapping, Sequence


_PATCH_LOCK = RLock()
_CONVERTER_PARAMETERS = (
    "inputs",
    "export_graph",
    "file_path",
    "weight_name",
)


@dataclass
class ExternalWeightMappingAudit:
    """Evidence collected while one AIR graph is serialized."""

    required: bool
    status: str
    torchair_runtime_type: str
    policy: str = "data-index-v1"
    converter_calls: int = 0
    used_weight_inputs: int = 0
    converted_weight_inputs: int = 0
    weight_externalized: bool | None = None
    converted_samples: list[dict[str, Any]] = field(default_factory=list)

    def record_conversion(
        self,
        *,
        input_index: int,
        node_name: str,
        output_type: str,
    ) -> None:
        self.converted_weight_inputs += 1
        # A bounded sample is enough to diagnose mapping without inflating the
        # AIR manifest with every parameter in the model.
        if len(self.converted_samples) < 8:
            self.converted_samples.append(
                {
                    "runtime_input_index": input_index,
                    "data_node_name": node_name,
                    "output_type": output_type,
                }
            )

    def as_manifest_record(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "required": self.required,
            "torchair_runtime_type": self.torchair_runtime_type,
            "policy": self.policy,
            "mapping_key": "GraphDef Data.index == runtime input index",
            "converter_calls": self.converter_calls,
            "used_weight_inputs": self.used_weight_inputs,
            "converted_weight_inputs": self.converted_weight_inputs,
            "weight_externalized": self.weight_externalized,
            "converted_samples": list(self.converted_samples),
            "invariant": (
                "only indexed Data parameter placeholders may be replaced; "
                "dynamic Gather/Pack shape helpers remain unchanged"
            ),
        }


def _indexed_graph_inputs(export_graph: Any) -> dict[int, Any]:
    indexed: dict[int, Any] = {}
    for op in export_graph.op:
        if op.type not in {"Data", "RefData"}:
            continue
        if "index" not in op.attr:
            raise RuntimeError(
                f"TorchAir {op.type} node {op.name!r} has no input index"
            )
        input_index = int(op.attr["index"].i)
        previous = indexed.get(input_index)
        if previous is not None:
            raise RuntimeError(
                "TorchAir graph has duplicate runtime input index "
                f"{input_index}: {previous.name!r} and {op.name!r}"
            )
        indexed[input_index] = op
    return indexed


def _convert_data_to_const_by_index(
    export_utils: Any,
    audit: ExternalWeightMappingAudit,
    inputs: Sequence[Any],
    export_graph: Any,
    file_path: str,
    weight_name: Mapping[int, str],
) -> tuple[bool, int]:
    """TorchAir's converter with an index-safe parameter-node lookup."""

    weight_externalized, used_weight_num = export_utils._is_weight_externalized(
        inputs, weight_name, export_graph
    )
    audit.converter_calls += 1
    audit.used_weight_inputs += int(used_weight_num)
    audit.weight_externalized = bool(weight_externalized)
    if used_weight_num == 0:
        return weight_externalized, used_weight_num

    weighted_inputs = [
        (input_index, value, weight_name[id(value)])
        for input_index, value in enumerate(inputs)
        if id(value) in weight_name
    ]
    if len(weighted_inputs) != used_weight_num:
        raise RuntimeError(
            "TorchAir weight accounting changed during AIR serialization: "
            f"expected {used_weight_num}, resolved {len(weighted_inputs)}"
        )

    indexed_inputs = _indexed_graph_inputs(export_graph)
    replacements: list[tuple[int, Any, str, Any]] = []
    # Validate the complete mapping before mutating the protobuf.  A mismatch
    # therefore cannot leave a partially rewritten AIR graph behind.
    for input_index, value, file_id in weighted_inputs:
        data_op = indexed_inputs.get(input_index)
        if data_op is None:
            raise RuntimeError(
                "TorchAir parameter input has no matching indexed Data node: "
                f"runtime input {input_index}, weight {file_id!r}"
            )
        if data_op.type != "Data":
            raise RuntimeError(
                "TorchAir attempted to externalize a mutable RefData input: "
                f"runtime input {input_index}, node {data_op.name!r}"
            )
        if not value.is_contiguous():
            raise AssertionError("The value of inputs must be contiguous.")
        replacements.append((input_index, value, file_id, data_op))

    for input_index, value, file_id, data_op in replacements:
        op_name = data_op.name
        export_utils.logger.debug(
            f"  Weight {input_index} dtype: {value.dtype} shape: {value.shape}"
        )
        with export_utils.GeGraph():
            if weight_externalized:
                converted = export_utils.ge.FileConstant(
                    shape=list(value.shape),
                    dtype=export_utils.torch_type_to_ge_type(value.dtype),
                    file_path=file_path + "/" + file_id.replace(".", "_"),
                    node_name=op_name,
                )
            else:
                converted = export_utils._make_const_node(value, op_name)
        data_op.Clear()
        data_op.MergeFrom(converted.node)
        audit.record_conversion(
            input_index=input_index,
            node_name=op_name,
            output_type=str(data_op.type),
        )

    if audit.converted_weight_inputs < audit.used_weight_inputs:
        raise RuntimeError(
            "TorchAir did not convert every resolved parameter Data node: "
            f"converted {audit.converted_weight_inputs}, "
            f"expected at least {audit.used_weight_inputs}"
        )
    export_utils._sort_graph_data_index(export_graph)
    return weight_externalized, used_weight_num


def _validate_export_utils(export_utils: Any) -> None:
    required = (
        "_convert_data_to_const",
        "_is_weight_externalized",
        "_make_const_node",
        "_sort_graph_data_index",
        "ge",
        "torch_type_to_ge_type",
        "GeGraph",
        "logger",
    )
    missing = [name for name in required if not hasattr(export_utils, name)]
    if missing:
        raise RuntimeError(
            "receiver TorchAir export internals are incompatible with the "
            "dynamic external-weight mapping fix; missing: " + ", ".join(missing)
        )
    try:
        parameters = tuple(
            inspect.signature(export_utils._convert_data_to_const).parameters
        )
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "cannot inspect receiver TorchAir _convert_data_to_const ABI"
        ) from error
    if parameters != _CONVERTER_PARAMETERS:
        raise RuntimeError(
            "receiver TorchAir _convert_data_to_const ABI drifted: "
            f"expected {_CONVERTER_PARAMETERS}, observed {parameters}"
        )


@contextmanager
def index_safe_external_weight_conversion(
    torchair: Any,
    *,
    required: bool,
    explicit_test_double: bool = False,
) -> Iterator[ExternalWeightMappingAudit]:
    """Patch one dynamic export and restore TorchAir immediately afterwards."""

    runtime_type = f"{type(torchair).__module__}.{type(torchair).__qualname__}"

    if not required:
        audit = ExternalWeightMappingAudit(
            required=False,
            status="NOT_REQUIRED_STATIC_GRAPH",
            torchair_runtime_type=runtime_type,
        )
        yield audit
        return

    # Only a caller that explicitly injected the repository's marked test
    # double may bypass receiver internals.  Never infer this from Python type:
    # receiver distributions may expose the canonical package through a lazy
    # module proxy rather than types.ModuleType.
    if explicit_test_double:
        audit = ExternalWeightMappingAudit(
            required=True,
            status="NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE",
            torchair_runtime_type=runtime_type,
        )
        yield audit
        return

    try:
        canonical_torchair = importlib.import_module("torchair")
    except ImportError as error:
        raise RuntimeError(
            "dynamic AIR export cannot import the canonical torchair package"
        ) from error
    if torchair is not canonical_torchair:
        raise RuntimeError(
            "dynamic AIR export received a non-canonical TorchAir object; "
            "injectable test doubles must carry the repository test marker"
        )

    export_utils = importlib.import_module("torchair._utils.export_utils")
    _validate_export_utils(export_utils)
    original = export_utils._convert_data_to_const
    audit = ExternalWeightMappingAudit(
        required=True,
        status="ARMED",
        torchair_runtime_type=runtime_type,
    )

    def convert_data_to_const(inputs, export_graph, file_path, weight_name):
        return _convert_data_to_const_by_index(
            export_utils,
            audit,
            inputs,
            export_graph,
            file_path,
            weight_name,
        )

    with _PATCH_LOCK:
        export_utils._convert_data_to_const = convert_data_to_const
        try:
            yield audit
            if audit.converter_calls < 1:
                raise RuntimeError(
                    "TorchAir dynamic export completed without invoking the "
                    "index-safe external-weight converter"
                )
            if audit.converted_weight_inputs != audit.used_weight_inputs:
                raise RuntimeError(
                    "TorchAir dynamic export external-weight count mismatch: "
                    f"converted {audit.converted_weight_inputs}, "
                    f"used {audit.used_weight_inputs}"
                )
            audit.status = "PASS"
        finally:
            export_utils._convert_data_to_const = original
