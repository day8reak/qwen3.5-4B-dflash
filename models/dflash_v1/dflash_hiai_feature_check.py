"""Validate a directly integrated HIAI DFlash feature route.

This module is deliberately read-only.  It never rewrites
``modeling_qwen3_5_hiai_nd.py`` and has no patch/apply operation.  The internal
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
FEATURE_SOURCE = "receiver_owned:modeling_qwen3_5_hiai_nd.py"
CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
FEATURE_WIDTH = 20480
EXPECTED_SOURCE_BASENAME = "modeling_qwen3_5_hiai_nd.py"


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


def _require_metadata(owner: ast.ClassDef) -> None:
    expected = {
        "dflash_feature_contract_id": FEATURE_CONTRACT_ID,
        "dflash_feature_source": FEATURE_SOURCE,
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
    runtime_symbols: set[str] = set()
    for node in module.body:
        if not isinstance(node, ast.ImportFrom) or node.level != 1:
            continue
        if node.module == "dflash_v1.dflash_target_features":
            target_symbols.update(
                alias.name for alias in node.names if alias.asname is None
            )
        elif node.module == "dflash_v1.dflash_hiai_feature_runtime":
            runtime_symbols.update(
                alias.name for alias in node.names if alias.asname is None
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
    if "attach_dflash_features" not in runtime_symbols:
        raise HiaiFeatureContractError(
            "HIAI source must directly import attach_dflash_features"
        )


def _require_text_route(function: ast.FunctionDef) -> None:
    _require_disabled_default(function)
    expected_counts = {
        "DFlashFeatureCollector": 1,
        "capture": 1,
        "finalize": 1,
        "attach_dflash_features": 1,
    }
    for name, expected in expected_counts.items():
        actual = len(_calls(function, name))
        if actual != expected:
            raise HiaiFeatureContractError(
                f"Qwen3_5TextModel.forward must call {name} exactly {expected} "
                f"time(s), found {actual}"
            )
    if not any(
        isinstance(node, ast.Constant) and node.value == FEATURE_WIDTH
        for node in ast.walk(function)
    ):
        raise HiaiFeatureContractError(
            f"text feature route must validate width {FEATURE_WIDTH}"
        )


def _require_causal_route(function: ast.FunctionDef) -> None:
    _require_disabled_default(function)
    if len(_calls(function, "attach_dflash_features")) != 1:
        raise HiaiFeatureContractError(
            "Qwen3_5ForCausalLM.forward must attach features exactly once"
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


def verify_direct_source(source: str, *, source_sha256: str | None = None) -> dict[str, Any]:
    """Verify one already modified source string without changing it."""

    try:
        module = ast.parse(source)
    except SyntaxError as error:
        raise HiaiFeatureContractError("HIAI source is not valid Python") from error
    _require_imports(module)
    text_model = _top_level_class(module, "Qwen3_5TextModel")
    causal_model = _top_level_class(module, "Qwen3_5ForCausalLM")
    _require_metadata(text_model)
    _require_metadata(causal_model)
    _require_text_route(_method(text_model, "forward"))
    _require_causal_route(_method(causal_model, "forward"))
    digest = source_sha256 or _sha256_bytes(source.encode("utf-8"))
    return {
        "status": "PASS_DIRECT_SOURCE_CONTRACT",
        "contract_id": FEATURE_CONTRACT_ID,
        "feature_source": FEATURE_SOURCE,
        "capture_point": CAPTURE_POINT,
        "feature_width": FEATURE_WIDTH,
        "source_sha256": digest,
        "source_modified": False,
    }


def verify_direct_source_file(path: str | Path) -> dict[str, Any]:
    """Verify a package-local, regular HIAI source file."""

    source_path = Path(path)
    if source_path.name != EXPECTED_SOURCE_BASENAME:
        raise HiaiFeatureContractError(
            f"expected {EXPECTED_SOURCE_BASENAME}, got {source_path.name}"
        )
    if source_path.is_symlink() or not source_path.is_file():
        raise HiaiFeatureContractError("HIAI source must be a regular file")
    payload = source_path.read_bytes()
    try:
        source = payload.decode("utf-8")
    except UnicodeDecodeError as error:
        raise HiaiFeatureContractError("HIAI source must be UTF-8") from error
    return verify_direct_source(source, source_sha256=_sha256_bytes(payload))


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
    "FEATURE_WIDTH",
    "HiaiFeatureContractError",
    "verify_direct_source",
    "verify_direct_source_file",
]


if __name__ == "__main__":
    raise SystemExit(main())
