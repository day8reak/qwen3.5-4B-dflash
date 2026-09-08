"""Fixed-carrier Draft and legacy fused OM contracts (no dynamic shape APIs)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def _validated_static_shape(
    graph: Mapping[str, Any], *, air: bool = False,
    allow_test_double: bool = False,
    role: str = "fused-speculative-step", key: str = "fused_static_shape",
    binding_count: int = 15, kv_index: int = 12,
) -> dict[str, Any] | None:
    source = graph.get("metadata", {}) if air else graph
    label = "static fused" if role == "fused-speculative-step" else "static Draft"
    raw = source.get(key) if isinstance(source, Mapping) else None
    if raw is None:
        if graph.get("role") == role and graph.get("dynamic") is False:
            raise ValueError(f"static {role} OM requires a {key} contract")
        return None
    if (
        graph.get("role") != role
        or graph.get("dynamic") is not False
        or graph.get("input_dim_gears") != {}
        or not isinstance(raw, Mapping)
    ):
        raise ValueError(f"{label} shape contract conflicts with role/dynamic/gears")
    result = dict(raw)
    for field in ("feature_rows", "feature_width", "verify_rows", "kv_capacity"):
        value = result.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{label} {field} must be a positive integer")
    if (
        result["feature_rows"] % 64 or result["kv_capacity"] % 64
        or result["feature_rows"] > result["kv_capacity"]
        or result["verify_rows"] != 16
        or result.get("padding_policy") != "zero-tail-on-stream-v1"
        or result.get("capacity_policy") != "reserve-full-static-write-v1"
    ):
        raise ValueError(f"{label} shape/padding/capacity policy differs")
    if air:
        audit = graph.get("runtime_input_abi", {})
        if not isinstance(audit, Mapping):
            raise ValueError(f"{label} AIR input audit must be an object")
        if allow_test_double and audit.get("status") == "NOT_APPLICABLE_EXPLICIT_TEST_DOUBLE":
            return result
        bindings = audit.get("bindings", [])
        if audit.get("status") != "PASS" or not isinstance(bindings, list) or len(bindings) != binding_count:
            raise ValueError(f"static {role} AIR requires {binding_count} audited static tensor bindings")
        for index, binding in enumerate(bindings):
            if not isinstance(binding, Mapping):
                raise ValueError(f"{label} AIR input binding must be an object")
            shape = binding.get("serialized_shape")
            if (
                binding.get("index") != index or not isinstance(shape, list)
                or not shape or any(
                    isinstance(d, bool) or not isinstance(d, int) or d <= 0 for d in shape
                )
                or shape != binding.get("example_shape")
            ):
                raise ValueError(f"{label} AIR input has an unaudited/dynamic shape")
        if bindings[0]["serialized_shape"] != [
            1, result["feature_rows"], result["feature_width"]
        ] or bindings[0].get("dtype") != "float16":
            raise ValueError(f"{label} AIR feature binding differs from contract")
        if bindings[kv_index]["serialized_shape"][3:4] != [result["kv_capacity"]]:
            raise ValueError(f"{label} AIR KV capacity differs from contract")
    return result


def validated_fused_static_shape(
    graph: Mapping[str, Any], *, air: bool = False, allow_test_double: bool = False,
) -> dict[str, Any] | None:
    return _validated_static_shape(graph, air=air, allow_test_double=allow_test_double)


def validated_static_feature_shape(
    graph: Mapping[str, Any], *, air: bool = False, allow_test_double: bool = False,
) -> dict[str, Any] | None:
    fused = validated_fused_static_shape(graph, air=air, allow_test_double=allow_test_double)
    draft = _validated_static_shape(
        graph, air=air, allow_test_double=allow_test_double,
        role="draft-propose", key="draft_static_shape", binding_count=8, kv_index=5,
    )
    return fused if fused is not None else draft


def validate_static_request(
    shape: Mapping[str, Any], prompt_rows: int, max_new_tokens: int,
) -> None:
    rows = shape["feature_rows"]
    if prompt_rows > rows:
        raise ValueError(
            f"static fused/Draft OM accepts at most {rows} prompt tokens; got {prompt_rows}. "
            "Export a larger static feature carrier; no truncation is allowed."
        )
    # The current cache scatter writes the entire physical carrier. Reserving
    # N rows keeps all indices unique and prevents padding from clamping onto
    # a live last KV slot. This is a conservative static-baseline admission gate.
    if prompt_rows + max_new_tokens > shape["kv_capacity"] - rows:
        raise ValueError(
            "static fused generation budget must leave feature_rows free KV slots "
            "(prompt_tokens + max_new_tokens + feature_rows <= kv_capacity)"
        )
