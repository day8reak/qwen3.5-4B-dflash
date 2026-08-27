"""Validate a directly integrated HIAI DFlash feature route.

This module is deliberately read-only.  It never rewrites
``modeling_qwen3_5_hiai_nd.py`` and has no patch/apply operation.  The target
model source is expected to contain the feature collector already; this check
only rejects common deployment mistakes before model weights are loaded.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


FEATURE_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"
FEATURE_SOURCE = "package_local:modeling_qwen3_5_hiai_nd.py"
ROLLBACK_FEATURE_SOURCE = (
    "package_local:modeling_qwen3_5_hiai_nd_dflash_rollback.py"
)
CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
FEATURE_WIDTH = 20480
EXPECTED_SOURCE_BASENAME = "modeling_qwen3_5_hiai_nd.py"
SUPPORTED_SOURCE_BASENAMES = {
    EXPECTED_SOURCE_BASENAME: FEATURE_SOURCE,
    "modeling_qwen3_5_hiai_nd_dflash_rollback.py": ROLLBACK_FEATURE_SOURCE,
}


class HiaiFeatureContractError(RuntimeError):
    """The directly integrated HIAI source does not satisfy the V1 ABI."""


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _top_level_class(module: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise HiaiFeatureContractError(
            f"expected one top-level {name}, found {len(matches)}"
        )
    return matches[0]


def _method(owner: ast.ClassDef, name: str) -> ast.FunctionDef:
    matches = [
        node
        for node in owner.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == name
    ]
    if len(matches) != 1 or not isinstance(matches[0], ast.FunctionDef):
        raise HiaiFeatureContractError(
            f"{owner.name} must define one synchronous {name} method"
        )
    return matches[0]


def _literal_class_attribute(owner: ast.ClassDef, name: str) -> object:
    values: list[object] = []
    for node in owner.body:
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            if isinstance(target, ast.Name) and target.id == name:
                value = node.value
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name) and node.target.id == name:
                value = node.value
        if value is not None:
            try:
                values.append(ast.literal_eval(value))
            except (ValueError, TypeError) as error:
                raise HiaiFeatureContractError(
                    f"{owner.name}.{name} must be a literal"
                ) from error
    if len(values) != 1:
        raise HiaiFeatureContractError(
            f"{owner.name} must declare exactly one {name}"
        )
    return values[0]


def _require_metadata(owner: ast.ClassDef, *, feature_source: str) -> None:
    expected = {
        "dflash_feature_contract_id": FEATURE_CONTRACT_ID,
        "dflash_feature_source": feature_source,
        "dflash_feature_capture_point": CAPTURE_POINT,
    }
    for name, value in expected.items():
        actual = _literal_class_attribute(owner, name)
        if actual != value:
            raise HiaiFeatureContractError(
                f"{owner.name}.{name} must be {value!r}, got {actual!r}"
            )


def _argument_default(function: ast.FunctionDef, name: str) -> ast.expr | None:
    positional = [*function.args.posonlyargs, *function.args.args]
    defaults = [None] * (len(positional) - len(function.args.defaults)) + list(
        function.args.defaults
    )
    for argument, default in zip(positional, defaults, strict=True):
        if argument.arg == name:
            return default
    for argument, default in zip(
        function.args.kwonlyargs,
        function.args.kw_defaults,
        strict=True,
    ):
        if argument.arg == name:
            return default
    raise HiaiFeatureContractError(
        f"{function.name} must explicitly consume {name}"
    )


def _require_disabled_default(function: ast.FunctionDef) -> None:
    default = _argument_default(function, "output_dflash_features")
    if not isinstance(default, ast.Constant) or default.value is not False:
        raise HiaiFeatureContractError(
            f"{function.name}.output_dflash_features must default to False"
        )


def _calls(function: ast.FunctionDef, name: str) -> list[ast.Call]:
    result: list[ast.Call] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name) and node.func.id == name:
            result.append(node)
        elif isinstance(node.func, ast.Attribute) and node.func.attr == name:
            result.append(node)
    return result


def _require_imports(module: ast.Module) -> None:
    target_symbols: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module == "dflash_v1.dflash_target_features":
            target_symbols.update(
                alias.name for alias in node.names if alias.asname is None
            )
        elif node.module == "dflash_v1.dflash_hiai_feature_runtime":
            raise HiaiFeatureContractError(
                "Tensor-returning HIAI source must not use the retired "
                "ModelOutput sidecar runtime"
            )
        elif node.module == "dflash_v1.dflash_hiai_feature_patch":
            raise HiaiFeatureContractError(
                "direct HIAI source must not import the retired patcher"
            )
    required_target = {
        "DFlashFeatureCollector",
        "QWEN35_4B_DFLASH_TARGET_FEATURES",
    }
    if not required_target.issubset(target_symbols):
        missing = sorted(required_target - target_symbols)
        raise HiaiFeatureContractError(
            "HIAI source is missing direct feature imports: " + ", ".join(missing)
        )


def _is_name(node: ast.AST | None, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def _is_name_tuple(node: ast.AST | None, names: tuple[str, ...]) -> bool:
    return (
        isinstance(node, ast.Tuple)
        and len(node.elts) == len(names)
        and all(_is_name(item, name) for item, name in zip(node.elts, names))
    )


def _assigned_call(
    function: ast.FunctionDef,
    *,
    target_name: str,
    call_name: str,
) -> list[ast.Assign]:
    result: list[ast.Assign] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        if not _is_name(node.targets[0], target_name):
            continue
        if not isinstance(node.value, ast.Call):
            continue
        if (
            isinstance(node.value.func, ast.Name)
            and node.value.func.id == call_name
        ) or (
            isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == call_name
        ):
            result.append(node)
    return result


def _require_collector_constructor(function: ast.FunctionDef) -> None:
    calls = _calls(function, "DFlashFeatureCollector")
    if len(calls) != 1:
        raise HiaiFeatureContractError(
            "Qwen3_5TextModel.forward must construct one DFlashFeatureCollector"
        )
    call = calls[0]
    if not call.args or not _is_name(
        call.args[0], "QWEN35_4B_DFLASH_TARGET_FEATURES"
    ):
        raise HiaiFeatureContractError(
            "DFlashFeatureCollector must use QWEN35_4B_DFLASH_TARGET_FEATURES"
        )
    expected_keywords = {"enabled": True, "detach": True, "clone": True}
    actual: dict[str, object] = {}
    for keyword in call.keywords:
        if keyword.arg is None:
            raise HiaiFeatureContractError(
                "DFlashFeatureCollector must not receive expanded keyword arguments"
            )
        if keyword.arg in expected_keywords:
            try:
                actual[keyword.arg] = ast.literal_eval(keyword.value)
            except (TypeError, ValueError) as error:
                raise HiaiFeatureContractError(
                    f"DFlashFeatureCollector.{keyword.arg} must be a bool literal"
                ) from error
    if actual != expected_keywords:
        raise HiaiFeatureContractError(
            "DFlashFeatureCollector must set enabled=True, detach=True, clone=True"
        )


def _require_capture_position(function: ast.FunctionDef) -> None:
    capture_calls = _calls(function, "capture")
    if len(capture_calls) != 1:
        raise HiaiFeatureContractError(
            "Qwen3_5TextModel.forward must capture exactly once in the decoder loop"
        )
    capture = capture_calls[0]
    if len(capture.args) != 2 or not _is_name(capture.args[0], "idx") or not _is_name(
        capture.args[1], "hidden_states"
    ):
        raise HiaiFeatureContractError(
            "collector.capture must receive (idx, hidden_states)"
        )

    decoder_assignments: list[ast.Assign] = []
    decoder_loops: list[ast.For] = []
    for node in ast.walk(function):
        if not isinstance(node, ast.For):
            continue
        assignments = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Assign)
            and len(child.targets) == 1
            and _is_name(child.targets[0], "hidden_states")
            and isinstance(child.value, ast.Subscript)
            and _is_name(child.value.value, "layer_outputs")
        ]
        if assignments and capture in set(ast.walk(node)):
            decoder_loops.append(node)
            decoder_assignments.extend(assignments)
    if len(decoder_loops) != 1 or len(decoder_assignments) != 1:
        raise HiaiFeatureContractError(
            "capture must follow exactly one hidden_states = layer_outputs[0] "
            "assignment in the decoder loop"
        )
    decoder_assignment = decoder_assignments[0]
    if capture.lineno <= decoder_assignment.lineno:
        raise HiaiFeatureContractError(
            "DFlash capture must occur after the decoder layer output assignment"
        )

    norm_assignments = _assigned_call(
        function,
        target_name="hidden_states",
        call_name="norm",
    )
    if len(norm_assignments) != 1 or capture.lineno >= norm_assignments[0].lineno:
        raise HiaiFeatureContractError(
            "DFlash capture must occur before the one final target norm"
        )


def _require_tensor_tuple_returns(
    function: ast.FunctionDef,
    *,
    ordinary_name: str,
    feature_names: tuple[str, str],
) -> None:
    ordinary = 0
    feature = 0
    for node in ast.walk(function):
        if not isinstance(node, ast.Return):
            continue
        if _is_name(node.value, ordinary_name):
            ordinary += 1
        elif _is_name_tuple(node.value, feature_names):
            feature += 1
    if ordinary != 1 or feature != 1:
        raise HiaiFeatureContractError(
            f"{function.name} must have one ordinary Tensor return and one "
            "feature-enabled two-Tensor return"
        )


def _require_text_route(function: ast.FunctionDef) -> None:
    _require_disabled_default(function)
    _require_collector_constructor(function)
    expected_counts = {
        "capture": 1,
        "finalize": 1,
        "attach_dflash_features": 0,
    }
    for name, expected in expected_counts.items():
        actual = len(_calls(function, name))
        if actual != expected:
            raise HiaiFeatureContractError(
                f"Qwen3_5TextModel.forward must call {name} exactly {expected} "
                f"time(s), found {actual}"
            )
    _require_capture_position(function)
    finalizers = _assigned_call(
        function,
        target_name="dflash_features",
        call_name="finalize",
    )
    if len(finalizers) != 1:
        raise HiaiFeatureContractError(
            "collector.finalize() must be assigned once to dflash_features"
        )
    _require_tensor_tuple_returns(
        function,
        ordinary_name="hidden_states",
        feature_names=("hidden_states", "dflash_features"),
    )


def _require_causal_route(function: ast.FunctionDef) -> None:
    _require_disabled_default(function)
    if _calls(function, "attach_dflash_features"):
        raise HiaiFeatureContractError(
            "Tensor-returning Qwen3_5ForCausalLM must not attach a sidecar"
        )
    forwarded = False
    for call in ast.walk(function):
        if not isinstance(call, ast.Call):
            continue
        for keyword in call.keywords:
            if (
                keyword.arg == "output_dflash_features"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "output_dflash_features"
            ):
                forwarded = True
    if not forwarded:
        raise HiaiFeatureContractError(
            "causal forward must pass output_dflash_features explicitly"
        )
    _require_tensor_tuple_returns(
        function,
        ordinary_name="logits",
        feature_names=("logits", "dflash_features"),
    )


def verify_direct_source(
    source: str,
    *,
    source_sha256: str | None = None,
    feature_source: str = FEATURE_SOURCE,
) -> dict[str, Any]:
    """Verify one already modified source string without changing it."""

    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise HiaiFeatureContractError("HIAI source is not valid Python") from error
    _require_imports(module)
    text_model = _top_level_class(module, "Qwen3_5TextModel")
    causal_model = _top_level_class(module, "Qwen3_5ForCausalLM")
    if feature_source not in SUPPORTED_SOURCE_BASENAMES.values():
        raise HiaiFeatureContractError(
            f"unsupported HIAI feature source: {feature_source!r}"
        )
    _require_metadata(text_model, feature_source=feature_source)
    _require_metadata(causal_model, feature_source=feature_source)
    _require_text_route(_method(text_model, "forward"))
    _require_causal_route(_method(causal_model, "forward"))
    digest = source_sha256 or _sha256_bytes(source.encode("utf-8"))
    return {
        "status": "PASS_DIRECT_SOURCE_CONTRACT",
        "contract_id": FEATURE_CONTRACT_ID,
        "feature_source": feature_source,
        "capture_point": CAPTURE_POINT,
        "feature_width": FEATURE_WIDTH,
        "output_abi": "Tensor | tuple[Tensor, Tensor]",
        "source_sha256": digest,
        "source_modified": False,
    }


def verify_direct_source_file(path: str | Path) -> dict[str, Any]:
    """Verify a package-local, regular HIAI source file."""

    source_path = Path(path)
    feature_source = SUPPORTED_SOURCE_BASENAMES.get(source_path.name)
    if feature_source is None:
        raise HiaiFeatureContractError(
            "expected one of "
            f"{sorted(SUPPORTED_SOURCE_BASENAMES)}, got {source_path.name}"
        )
    if source_path.is_symlink() or not source_path.is_file():
        raise HiaiFeatureContractError("HIAI source must be a regular file")
    payload = source_path.read_bytes()
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HiaiFeatureContractError("HIAI source must be UTF-8") from error
    return verify_direct_source(
        source,
        source_sha256=_sha256_bytes(payload),
        feature_source=feature_source,
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the read-only source check from the command line."""

    parser = argparse.ArgumentParser(
        description="Check an already integrated HIAI DFlash feature route"
    )
    parser.add_argument("--source", required=True)
    args = parser.parse_args(argv)
    report = verify_direct_source_file(args.source)
    report["source"] = str(Path(args.source).expanduser().resolve())
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


__all__ = [
    "CAPTURE_POINT",
    "FEATURE_CONTRACT_ID",
    "FEATURE_SOURCE",
    "ROLLBACK_FEATURE_SOURCE",
    "FEATURE_WIDTH",
    "HiaiFeatureContractError",
    "verify_direct_source",
    "verify_direct_source_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
