"""Exact TorchDynamo/TorchAir export support for retained custom operators.

Fake kernels in this module describe metadata only.  They never implement the
operator numerics and therefore cannot silently replace an NPU kernel.  A
separate TorchAir converter emits one registered GE operator, and the exporter
then checks ``dynamo.pbtxt`` before declaring the AIR bundle passing.
"""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from pathlib import Path
import re
import threading
from typing import Any, Callable, Sequence

import torch

from .contracts import CustomOpExportSpec


ADN_RMS_NORM_TORCH_OP = "npu::adn_rms_norm"
ADN_RMS_NORM_DEFAULT_GE_OP_TYPE = "RmsNorm"
_GE_TYPE_FIELD = re.compile(r'\btype:\s*"([A-Za-z_][A-Za-z0-9_]*)"')
_FAKE_REGISTRATION_LOCK = threading.Lock()


@dataclass
class CustomOpExportSession:
    """Mutable evidence collected while one graph is lowered to GE."""

    spec: CustomOpExportSpec
    schema: str
    fake_kernel: str
    converter_mode: str
    converter_calls: int = 0


def _fake_adn_rms_norm(
    input: torch.Tensor,
    gamma: torch.Tensor,
    epsilon: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the receiver-observed metadata without performing RMSNorm."""

    del gamma, epsilon
    output = torch.empty_like(input)
    rstd = input.new_empty((*input.shape[:-1], 1), dtype=torch.float32)
    return output, rstd


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


def _validate_adn_rms_norm_schema(operation: Any) -> str:
    schema = getattr(operation, "_schema", None)
    if schema is None:
        raise RuntimeError("npu::adn_rms_norm does not expose a dispatcher schema")
    argument_names = tuple(item.name for item in schema.arguments)
    argument_types = tuple(str(item.type) for item in schema.arguments)
    return_types = tuple(str(item.type) for item in schema.returns)
    valid_first_name = bool(argument_names) and argument_names[0] in {"input", "self"}
    if (
        getattr(schema, "name", None) != ADN_RMS_NORM_TORCH_OP
        or argument_names[1:] != ("gamma", "epsilon")
        or not valid_first_name
        or argument_types != ("Tensor", "Tensor", "float")
        or return_types != ("Tensor", "Tensor")
    ):
        raise RuntimeError(
            "npu::adn_rms_norm schema drifted from "
            "(Tensor input, Tensor gamma, float epsilon) -> (Tensor, Tensor): "
            f"{schema}"
        )
    return str(schema)


def _has_meta_kernel(torch_op: str) -> bool:
    query = getattr(torch._C, "_dispatch_has_kernel_for_dispatch_key", None)
    if not callable(query):
        raise RuntimeError("this PyTorch build cannot query custom-op Meta kernels")
    return bool(query(torch_op, "Meta"))


def _validate_adn_rms_norm_meta(operation: Any) -> None:
    for dtype in (torch.float16, torch.float32):
        input_tensor = torch.empty((2, 3, 8), dtype=dtype, device="meta")
        gamma = torch.empty((8,), dtype=dtype, device="meta")
        result = operation(input_tensor, gamma, 1e-6)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError("npu::adn_rms_norm Meta kernel must return two tensors")
        output, rstd = result
        if (
            tuple(output.shape) != (2, 3, 8)
            or output.dtype != dtype
            or output.device.type != "meta"
        ):
            raise RuntimeError("npu::adn_rms_norm Meta output[0] contract mismatch")
        if (
            tuple(rstd.shape) != (2, 3, 1)
            or rstd.dtype != torch.float32
            or rstd.device.type != "meta"
        ):
            raise RuntimeError("npu::adn_rms_norm Meta output[1] contract mismatch")


def _ensure_adn_rms_norm_fake(operation: Any) -> str:
    with _FAKE_REGISTRATION_LOCK:
        if _has_meta_kernel(ADN_RMS_NORM_TORCH_OP):
            status = "preexisting-meta-kernel"
        else:
            register_fake = getattr(torch.library, "register_fake", None)
            if not callable(register_fake):
                raise RuntimeError(
                    "PyTorch torch.library.register_fake is required to export "
                    "npu::adn_rms_norm"
                )
            register_fake(ADN_RMS_NORM_TORCH_OP)(_fake_adn_rms_norm)
            if not _has_meta_kernel(ADN_RMS_NORM_TORCH_OP):
                raise RuntimeError("npu::adn_rms_norm Fake registration did not install Meta")
            status = "framework-registered-fake"
    _validate_adn_rms_norm_meta(operation)
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
    return "named-rmsnorm-compat"


def prepare_custom_op_export(
    spec: CustomOpExportSpec,
    torchair_module: Any,
) -> CustomOpExportSession:
    """Install metadata and FX-to-GE support for one exact custom-op contract."""

    if spec.torch_op != ADN_RMS_NORM_TORCH_OP or spec.overload != "default":
        raise NotImplementedError(
            f"no exact export adapter is implemented for {spec.torch_target}"
        )
    operation = _resolve_operation(spec)
    schema = _validate_adn_rms_norm_schema(operation)
    fake_kernel = _ensure_adn_rms_norm_fake(operation)

    registrar = getattr(torchair_module, "register_fx_node_ge_converter", None)
    ge_api = getattr(torchair_module, "ge", None)
    custom_op = getattr(ge_api, "custom_op", None)
    if not callable(registrar) or not callable(custom_op):
        raise RuntimeError(
            "TorchAir custom-op export requires register_fx_node_ge_converter "
            "and torchair.ge.custom_op"
        )
    converter_mode = _custom_op_call_mode(custom_op)
    if converter_mode == "named-rmsnorm-compat":
        attr_api = getattr(ge_api, "attr", None)
        if not callable(getattr(attr_api, "Float", None)):
            raise RuntimeError("TorchAir does not expose ge.attr.Float")

    session = CustomOpExportSession(
        spec=spec,
        schema=schema,
        fake_kernel=fake_kernel,
        converter_mode=converter_mode,
    )

    def convert_adn_rms_norm(
        input: Any,
        gamma: Any,
        epsilon: float = 1e-6,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        session.converter_calls += 1
        if session.converter_mode == "registered-ir-positional":
            return custom_op(spec.ge_op_type, input, gamma, epsilon)
        if spec.ge_op_type != ADN_RMS_NORM_DEFAULT_GE_OP_TYPE:
            raise RuntimeError(
                "this TorchAir version needs a positional custom_op API to use "
                f"the overridden GE type {spec.ge_op_type!r}"
            )
        return custom_op(
            spec.ge_op_type,
            inputs={"x": input, "gamma": gamma},
            outputs=["y", "rstd"],
            attrs={"epsilon": ge_api.attr.Float(epsilon)},
        )

    convert_adn_rms_norm.__name__ = "convert_npu_adn_rms_norm_default"
    registrar(operation)(convert_adn_rms_norm)
    return session


def audit_custom_op_export(
    sessions: Sequence[CustomOpExportSession],
    graph_dir: Path,
    *,
    relative_to: Path,
) -> list[dict[str, Any]]:
    """Prove each converter ran and its retained GE node reached TorchAir IR."""

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
        if session.converter_calls < spec.minimum_occurrences:
            raise RuntimeError(
                f"custom-op converter for {spec.torch_target} ran "
                f"{session.converter_calls} times; expected at least "
                f"{spec.minimum_occurrences}"
            )
        if ge_occurrences < session.converter_calls:
            raise RuntimeError(
                f"TorchAir IR contains {ge_occurrences} {spec.ge_op_type} nodes for "
                f"{session.converter_calls} converter calls to {spec.torch_target}"
            )
        records.append(
            {
                "status": "PASS",
                "torch_op": spec.torch_op,
                "torch_target": spec.torch_target,
                "torch_schema": session.schema,
                "fake_kernel": session.fake_kernel,
                "converter_mode": session.converter_mode,
                "converter_calls": session.converter_calls,
                "ge_op_type": spec.ge_op_type,
                "ge_node_occurrences": ge_occurrences,
                "minimum_occurrences": spec.minimum_occurrences,
                "preservation": "one registered GE operator; no Tensor decomposition",
                "pbtxt_files": [
                    path.relative_to(relative_to).as_posix() for path in pbtxt_paths
                ],
            }
        )
    return records


__all__ = [
    "ADN_RMS_NORM_DEFAULT_GE_OP_TYPE",
    "ADN_RMS_NORM_TORCH_OP",
    "CustomOpExportSession",
    "audit_custom_op_export",
    "prepare_custom_op_export",
]
