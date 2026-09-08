"""Exact TorchAir overrides for unsupported standard PyTorch operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import Any

import torch

from .utils import count_ge_ir_nodes


ATEN_SOFTPLUS_TORCH_TARGET = "aten.softplus.default"
ATEN_SOFTPLUS_GE_OP_TYPE = "SoftplusV2"


@dataclass
class StandardOpExportSession:
    """Evidence that a framework-owned standard-op converter was exercised."""

    torch_target: str
    ge_op_type: str
    converter_calls: int = 0


def _finite_static_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(
            f"aten.softplus.default {name} must be a compile-time numeric scalar"
        )
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"aten.softplus.default {name} must be finite")
    return result


def prepare_aten_softplus_export(
    torchair_module: Any,
) -> StandardOpExportSession:
    """Lower ``aten.softplus.default`` to one exact GE ``SoftplusV2`` node.

    Older receiver TorchAir releases register the ATen converter but raise
    ``NotImplementedError``; newer releases may already lower the same target.
    ``SoftplusV2`` implements the beta and threshold attributes, so retaining
    one node avoids both a graph break and an inaccurate/more expensive Tensor
    decomposition.
    """

    registrar = getattr(torchair_module, "register_fx_node_ge_converter", None)
    ge_api = getattr(torchair_module, "ge", None)
    custom_op = getattr(ge_api, "custom_op", None)
    attr_api = getattr(ge_api, "attr", None)
    float_attr = getattr(attr_api, "Float", None)
    if not callable(registrar) or not callable(custom_op) or not callable(float_attr):
        raise RuntimeError(
            "TorchAir aten.softplus export requires "
            "register_fx_node_ge_converter, ge.custom_op, and ge.attr.Float"
        )

    session = StandardOpExportSession(
        torch_target=ATEN_SOFTPLUS_TORCH_TARGET,
        ge_op_type=ATEN_SOFTPLUS_GE_OP_TYPE,
    )

    def converter(
        self: Any,
        beta: Real = 1,
        threshold: Real = 20,
        meta_outputs: Any = None,
    ) -> Any:
        del meta_outputs
        beta_value = _finite_static_float(beta, "beta")
        threshold_value = _finite_static_float(threshold, "threshold")
        session.converter_calls += 1
        return custom_op(
            ATEN_SOFTPLUS_GE_OP_TYPE,
            inputs={"x": self},
            outputs=["y"],
            attrs={
                "beta": float_attr(beta_value),
                "threshold": float_attr(threshold_value),
            },
        )

    converter.__name__ = "convert_aten_softplus_default_to_softplus_v2"
    registrar(torch.ops.aten.softplus.default)(converter)
    return session


def audit_aten_softplus_export(
    session: StandardOpExportSession,
    graph_dir: Path,
    *,
    calls_before: int,
    relative_to: Path,
    minimum_occurrences: int = 1,
) -> dict[str, Any]:
    """Require the graph's declared ``SoftplusV2`` contract in GE IR.

    TorchAir releases differ in converter import order.  Some execute the
    framework override registered above, while newer releases may already
    lower ``aten.softplus.default`` to the same ``SoftplusV2`` node before the
    override is reached.  The serialized GE graph is therefore authoritative;
    the framework call counter remains a consistency check when it is nonzero.

    Graphs such as the split prefill head do not execute a Gated DeltaNet layer
    and legitimately declare ``minimum_occurrences=0``.
    """

    if (
        isinstance(minimum_occurrences, bool)
        or not isinstance(minimum_occurrences, int)
        or minimum_occurrences < 0
    ):
        raise ValueError("SoftplusV2 minimum_occurrences must be non-negative")

    converter_calls = session.converter_calls - calls_before
    if converter_calls < 0:
        raise RuntimeError("aten.softplus.default converter call counter regressed")
    pbtxt_paths = sorted(graph_dir.rglob("dynamo.pbtxt"))
    if not pbtxt_paths:
        raise RuntimeError(
            "TorchAir produced no dynamo.pbtxt; SoftplusV2 lowering cannot be audited"
        )
    ge_occurrences = count_ge_ir_nodes(pbtxt_paths).get(session.ge_op_type, 0)
    if ge_occurrences < minimum_occurrences:
        raise RuntimeError(
            f"TorchAir IR for {graph_dir.name!r} contains {ge_occurrences} "
            f"{session.ge_op_type} nodes; expected at least "
            f"{minimum_occurrences} (framework converter calls: "
            f"{converter_calls})"
        )
    if ge_occurrences < converter_calls:
        raise RuntimeError(
            f"TorchAir IR contains {ge_occurrences} {session.ge_op_type} nodes for "
            f"{converter_calls} aten.softplus.default converter calls"
        )
    if converter_calls:
        converter_policy = "framework-registered-ge-ir"
    elif ge_occurrences:
        converter_policy = "torchair-existing-ge-ir"
    else:
        converter_policy = "not-present-in-graph"
    return {
        "status": "PASS",
        "torch_target": session.torch_target,
        "ge_op_type": session.ge_op_type,
        "minimum_occurrences": minimum_occurrences,
        "converter_calls": converter_calls,
        "converter_policy": converter_policy,
        "ge_node_occurrences": ge_occurrences,
        "observed_in_graph": bool(ge_occurrences),
        "lowering": (
            "required GE SoftplusV2 nodes retained; framework-emitted nodes "
            "preserve beta and threshold"
        ),
        "pbtxt_files": [
            path.relative_to(relative_to).as_posix() for path in pbtxt_paths
        ],
    }


__all__ = [
    "ATEN_SOFTPLUS_GE_OP_TYPE",
    "ATEN_SOFTPLUS_TORCH_TARGET",
    "StandardOpExportSession",
    "audit_aten_softplus_export",
    "prepare_aten_softplus_export",
]
