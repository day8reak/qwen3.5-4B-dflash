"""Fail-closed source patcher for the receiver-owned HIAI Qwen3.5 target.

The private ``modeling_qwen3_5_hiai_nd.py`` file is intentionally not part of
this delivery.  This module validates semantic AST anchors and then inserts a
small, opt-in DFlash feature route without replacing attention, GDN, cache, or
custom-operator calls.  Unknown or ambiguous source layouts are rejected.

The patched runtime imports :mod:`.dflash_target_features` and
:mod:`.dflash_hiai_feature_runtime`; both helpers must be installed beside the
receiver-owned modeling file.
"""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Iterable, Sequence


PATCH_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"
EXPECTED_SOURCE_BASENAME = "modeling_qwen3_5_hiai_nd.py"
FEATURE_SOURCE = f"receiver_owned:{EXPECTED_SOURCE_BASENAME}"
CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
FEATURE_LAYER_IDS = (1, 5, 9, 13, 17, 21, 25, 29)
TARGET_HIDDEN_SIZE = 2560
FEATURE_WIDTH = 20480

_IMPORT_SYMBOLS = (
    "DFlashFeatureCollector",
    "QWEN35_4B_DFLASH_TARGET_FEATURES",
)
_RUNTIME_IMPORT_SYMBOLS = ("attach_dflash_features",)
_PROTECTED_HELPER_SYMBOLS = frozenset(
    (*_IMPORT_SYMBOLS, *_RUNTIME_IMPORT_SYMBOLS)
)
_PATCH_OWNED_LOCALS = frozenset(
    (
        "_dflash_collector",
        "_dflash_features",
        "_dflash_base_output",
        "_dflash_causal_output",
        "output_dflash_features",
    )
)
_CANONICAL_HELPER_IMPORTS = (
    ("dflash_target_features", _IMPORT_SYMBOLS),
    ("dflash_hiai_feature_runtime", _RUNTIME_IMPORT_SYMBOLS),
)

_MARKERS = (
    "# DFLASH_HIAI_V1:IMPORT_BEGIN",
    "# DFLASH_HIAI_V1:IMPORT_END",
    "# DFLASH_HIAI_V1:TEXT_METADATA_BEGIN",
    "# DFLASH_HIAI_V1:TEXT_METADATA_END",
    "# DFLASH_HIAI_V1:TEXT_FLAG",
    "# DFLASH_HIAI_V1:COLLECTOR_BEGIN",
    "# DFLASH_HIAI_V1:COLLECTOR_END",
    "# DFLASH_HIAI_V1:CAPTURE_BEGIN",
    "# DFLASH_HIAI_V1:CAPTURE_END",
    "# DFLASH_HIAI_V1:FINALIZE_BEGIN",
    "# DFLASH_HIAI_V1:FINALIZE_END",
    "# DFLASH_HIAI_V1:TEXT_OUTPUT_BEGIN",
    "# DFLASH_HIAI_V1:TEXT_OUTPUT_END",
    "# DFLASH_HIAI_V1:CAUSAL_METADATA_BEGIN",
    "# DFLASH_HIAI_V1:CAUSAL_METADATA_END",
    "# DFLASH_HIAI_V1:CAUSAL_FLAG",
    "# DFLASH_HIAI_V1:CAUSAL_FORWARD_BEGIN",
    "# DFLASH_HIAI_V1:CAUSAL_FORWARD_END",
    "# DFLASH_HIAI_V1:CAUSAL_OUTPUT_BEGIN",
    "# DFLASH_HIAI_V1:CAUSAL_OUTPUT_END",
)


class HiaiFeaturePatchError(RuntimeError):
    """The private source did not satisfy the reviewed patch contract."""


@dataclass(frozen=True)
class PatchResult:
    """In-memory patch result; callers decide whether and where to write it."""

    source: str
    changed: bool
    report: dict[str, object]


@dataclass(frozen=True)
class _Insertion:
    """Text inserted before the zero-based line index ``position``."""

    position: int
    label: str
    text: str


@dataclass(frozen=True)
class _TextAnchors:
    cls: ast.ClassDef
    forward: ast.FunctionDef
    loop: ast.For
    layer_index_name: str
    decoder_layer_name: str
    decoder_assignment: ast.Assign | ast.AnnAssign
    norm_assignment: ast.Assign | ast.AnnAssign
    default_return: ast.Return


@dataclass(frozen=True)
class _CausalAnchors:
    cls: ast.ClassDef
    forward: ast.FunctionDef
    model_assignment: ast.Assign | ast.AnnAssign
    model_call: ast.Call
    outputs_name: str
    default_return: ast.Return


def _sha256_text(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _parse(source: str) -> ast.Module:
    try:
        return ast.parse(source)
    except SyntaxError as exc:
        raise HiaiFeaturePatchError(
            f"source is not valid Python: line {exc.lineno}: {exc.msg}"
        ) from exc


def _direct_class(module: ast.Module, name: str) -> ast.ClassDef:
    matches = [
        node
        for node in module.body
        if isinstance(node, ast.ClassDef) and node.name == name
    ]
    if len(matches) != 1:
        raise HiaiFeaturePatchError(
            f"expected exactly one top-level {name}, found {len(matches)}"
        )
    return matches[0]


def _direct_forward(cls: ast.ClassDef) -> ast.FunctionDef:
    async_matches = [
        node
        for node in cls.body
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "forward"
    ]
    if async_matches:
        raise HiaiFeaturePatchError(f"{cls.name}.forward must not be async")
    matches = [
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    ]
    if len(matches) != 1:
        raise HiaiFeaturePatchError(
            f"expected exactly one {cls.name}.forward, found {len(matches)}"
        )
    return matches[0]


def _assigned_name(node: ast.Assign | ast.AnnAssign) -> str | None:
    targets: list[ast.expr]
    if isinstance(node, ast.Assign):
        targets = list(node.targets)
    else:
        targets = [node.target]
    if len(targets) == 1 and isinstance(targets[0], ast.Name):
        return targets[0].id
    return None


def _assignment_value(node: ast.Assign | ast.AnnAssign) -> ast.expr | None:
    return node.value


def _contains_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(item, ast.Name) and item.id == name for item in ast.walk(node))


def _is_self_attribute(node: ast.AST, attribute: str) -> bool:
    return (
        isinstance(node, ast.Attribute)
        and node.attr == attribute
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    )


def _contains_self_layers(node: ast.AST) -> bool:
    return any(_is_self_attribute(item, "layers") for item in ast.walk(node))


def _for_names(node: ast.For) -> tuple[str, str] | None:
    target = node.target
    if not isinstance(target, (ast.Tuple, ast.List)) or len(target.elts) != 2:
        return None
    if not all(isinstance(item, ast.Name) for item in target.elts):
        return None
    return target.elts[0].id, target.elts[1].id


def _is_decoder_enumerate(node: ast.For) -> bool:
    return (
        isinstance(node.iter, ast.Call)
        and isinstance(node.iter.func, ast.Name)
        and node.iter.func.id == "enumerate"
        and len(node.iter.args) == 1
        and _contains_self_layers(node.iter.args[0])
    )


def _is_final_norm_assignment(node: ast.stmt) -> bool:
    if not isinstance(node, (ast.Assign, ast.AnnAssign)):
        return False
    if _assigned_name(node) != "hidden_states":
        return False
    value = _assignment_value(node)
    return (
        isinstance(value, ast.Call)
        and _is_self_attribute(value.func, "norm")
        and bool(value.args)
        and isinstance(value.args[0], ast.Name)
        and value.args[0].id == "hidden_states"
    )


def _direct_returns(forward: ast.FunctionDef) -> list[ast.Return]:
    return [node for node in forward.body if isinstance(node, ast.Return)]


def _locate_text_anchors(module: ast.Module) -> _TextAnchors:
    cls = _direct_class(module, "Qwen3_5TextModel")
    forward = _direct_forward(cls)

    loops: list[tuple[ast.For, str, str]] = []
    for node in ast.walk(forward):
        if isinstance(node, ast.For) and _is_decoder_enumerate(node):
            names = _for_names(node)
            if names is not None:
                loops.append((node, names[0], names[1]))
    if len(loops) != 1:
        raise HiaiFeaturePatchError(
            "Qwen3_5TextModel.forward must contain exactly one "
            f"enumerate(self.layers...) decoder loop, found {len(loops)}"
        )
    loop, index_name, layer_name = loops[0]

    decoder_assignments = []
    for node in loop.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        value = _assignment_value(node)
        if (
            _assigned_name(node) == "hidden_states"
            and value is not None
            and isinstance(value, ast.Call)
            and _contains_name(value, layer_name)
        ):
            decoder_assignments.append(node)
    if len(decoder_assignments) != 1:
        raise HiaiFeaturePatchError(
            "decoder loop must directly assign exactly one decoder-layer call "
            f"to hidden_states, found {len(decoder_assignments)}"
        )
    decoder_assignment = decoder_assignments[0]

    norm_assignments = [
        node
        for node in forward.body
        if _is_final_norm_assignment(node)
        and node.lineno > (loop.end_lineno or loop.lineno)
    ]
    if len(norm_assignments) != 1:
        raise HiaiFeaturePatchError(
            "text forward must have exactly one post-loop "
            f"hidden_states = self.norm(hidden_states), found {len(norm_assignments)}"
        )
    norm_assignment = norm_assignments[0]

    returns = [
        node
        for node in _direct_returns(forward)
        if node.lineno > (norm_assignment.end_lineno or norm_assignment.lineno)
    ]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        raise HiaiFeaturePatchError(
            "text forward must have one direct output-constructor return after final norm"
        )
    return _TextAnchors(
        cls=cls,
        forward=forward,
        loop=loop,
        layer_index_name=index_name,
        decoder_layer_name=layer_name,
        decoder_assignment=decoder_assignment,
        norm_assignment=norm_assignment,
        default_return=returns[0],
    )


def _self_model_call(node: ast.AST) -> ast.Call | None:
    if not isinstance(node, ast.Call):
        return None
    if _is_self_attribute(node.func, "model"):
        return node
    return None


def _locate_causal_anchors(module: ast.Module) -> _CausalAnchors:
    cls = _direct_class(module, "Qwen3_5ForCausalLM")
    forward = _direct_forward(cls)
    assignments: list[tuple[ast.Assign | ast.AnnAssign, ast.Call, str]] = []
    for node in forward.body:
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        name = _assigned_name(node)
        value = _assignment_value(node)
        call = _self_model_call(value) if value is not None else None
        if name is not None and call is not None:
            assignments.append((node, call, name))
    if len(assignments) != 1:
        raise HiaiFeaturePatchError(
            "Qwen3_5ForCausalLM.forward must directly assign exactly one "
            f"self.model(...) call, found {len(assignments)}"
        )
    model_assignment, model_call, outputs_name = assignments[0]

    returns = [
        node
        for node in _direct_returns(forward)
        if node.lineno > (model_assignment.end_lineno or model_assignment.lineno)
    ]
    if len(returns) != 1 or not isinstance(returns[0].value, ast.Call):
        raise HiaiFeaturePatchError(
            "causal-LM forward must have one direct output-constructor return"
        )
    return _CausalAnchors(
        cls=cls,
        forward=forward,
        model_assignment=model_assignment,
        model_call=model_call,
        outputs_name=outputs_name,
        default_return=returns[0],
    )


def _argument_default(forward: ast.FunctionDef, name: str) -> ast.expr | None:
    positional = [*forward.args.posonlyargs, *forward.args.args]
    defaults: list[ast.expr | None] = [None] * (len(positional) - len(forward.args.defaults))
    defaults.extend(forward.args.defaults)
    for argument, default in zip(positional, defaults):
        if argument.arg == name:
            return default
    for argument, default in zip(forward.args.kwonlyargs, forward.args.kw_defaults):
        if argument.arg == name:
            return default
    return None


def _has_argument(forward: ast.FunctionDef, name: str) -> bool:
    arguments = [
        *forward.args.posonlyargs,
        *forward.args.args,
        *forward.args.kwonlyargs,
    ]
    return any(argument.arg == name for argument in arguments)


def _require_multiline_kwargs(forward: ast.FunctionDef, lines: Sequence[str]) -> int:
    argument = forward.args.kwarg
    if argument is None:
        raise HiaiFeaturePatchError(
            f"{forward.name} must expose **kwargs so the feature flag can be consumed explicitly"
        )
    line_index = argument.lineno - 1
    if not lines[line_index].lstrip().startswith(f"**{argument.arg}"):
        raise HiaiFeaturePatchError(
            "**kwargs must be on its own signature line; refusing a formatting guess"
        )
    return line_index


def _class_body_insertion_line(cls: ast.ClassDef) -> int:
    if not cls.body:
        raise HiaiFeaturePatchError(f"{cls.name} has no body")
    first = cls.body[0]
    if (
        isinstance(first, ast.Expr)
        and isinstance(first.value, ast.Constant)
        and isinstance(first.value.value, str)
    ):
        return first.end_lineno or first.lineno
    return first.lineno - 1


def _leading_indent(line: str) -> str:
    return line[: len(line) - len(line.lstrip())]


def _import_insertion_line(module: ast.Module) -> int:
    body = list(module.body)
    index = 0
    if (
        body
        and isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        index = 1
    last_import_end: int | None = None
    while index < len(body) and isinstance(body[index], (ast.Import, ast.ImportFrom)):
        last_import_end = body[index].end_lineno or body[index].lineno
        index += 1
    if last_import_end is None:
        raise HiaiFeaturePatchError(
            "module must have a conventional top-level import block"
        )
    return last_import_end


def _call_kwargs_line(call: ast.Call, forward: ast.FunctionDef, lines: Sequence[str]) -> int:
    kwargs_name = forward.args.kwarg.arg if forward.args.kwarg is not None else None
    matches = [
        keyword
        for keyword in call.keywords
        if keyword.arg is None
        and kwargs_name is not None
        and isinstance(keyword.value, ast.Name)
        and keyword.value.id == kwargs_name
    ]
    if len(matches) != 1:
        raise HiaiFeaturePatchError(
            "self.model(...) must forward its local **kwargs exactly once"
        )
    line_index = matches[0].value.lineno - 1
    if not lines[line_index].lstrip().startswith(f"**{kwargs_name}"):
        raise HiaiFeaturePatchError(
            "the self.model **kwargs expansion must be on its own line"
        )
    return line_index


def _expression_assignment(indent: str, name: str, expression: ast.expr) -> str:
    assignment = f"{name} = {ast.unparse(expression)}"
    return "\n".join(indent + line if line else line for line in assignment.splitlines())


def _metadata_block(indent: str, *, role: str) -> str:
    prefix = "TEXT" if role == "text" else "CAUSAL"
    return (
        f"{indent}# DFLASH_HIAI_V1:{prefix}_METADATA_BEGIN\n"
        f'{indent}dflash_feature_contract_id = "{PATCH_CONTRACT_ID}"\n'
        f'{indent}dflash_feature_source = "{FEATURE_SOURCE}"\n'
        f'{indent}dflash_feature_capture_point = "{CAPTURE_POINT}"\n'
        f"{indent}# DFLASH_HIAI_V1:{prefix}_METADATA_END\n"
    )


def _build_insertions(
    source: str,
    module: ast.Module,
    text: _TextAnchors,
    causal: _CausalAnchors,
) -> list[_Insertion]:
    lines = source.splitlines(keepends=True)
    if not lines or not source.endswith(("\n", "\r")):
        raise HiaiFeaturePatchError("source must end with a newline")
    if _has_argument(text.forward, "output_dflash_features"):
        raise HiaiFeaturePatchError(
            "text forward contains an unowned output_dflash_features declaration "
            "without this patch identity"
        )
    if _has_argument(causal.forward, "output_dflash_features"):
        raise HiaiFeaturePatchError(
            "causal forward contains an unowned output_dflash_features declaration "
            "without this patch identity"
        )

    import_line = _import_insertion_line(module)
    text_flag_line = _require_multiline_kwargs(text.forward, lines)
    causal_flag_line = _require_multiline_kwargs(causal.forward, lines)
    causal_call_kwargs_line = _call_kwargs_line(
        causal.model_call, causal.forward, lines
    )

    loop_indent = _leading_indent(lines[text.loop.lineno - 1])
    decoder_indent = _leading_indent(lines[text.decoder_assignment.lineno - 1])
    norm_indent = _leading_indent(lines[text.norm_assignment.lineno - 1])
    text_return_indent = _leading_indent(lines[text.default_return.lineno - 1])
    causal_return_indent = _leading_indent(lines[causal.default_return.lineno - 1])
    causal_call_indent = _leading_indent(lines[causal_call_kwargs_line])
    text_class_indent = _leading_indent(lines[text.cls.body[0].lineno - 1])
    causal_class_indent = _leading_indent(lines[causal.cls.body[0].lineno - 1])

    text_expression = text.default_return.value
    causal_expression = causal.default_return.value
    if not isinstance(text_expression, ast.Call) or not isinstance(causal_expression, ast.Call):
        raise AssertionError("output return validation and construction diverged")

    text_feature_branch = (
        f"{text_return_indent}# DFLASH_HIAI_V1:TEXT_OUTPUT_BEGIN\n"
        f"{text_return_indent}if output_dflash_features:\n"
        + _expression_assignment(
            text_return_indent + "    ",
            "_dflash_base_output",
            text_expression,
        )
        + "\n"
        f"{text_return_indent}    return attach_dflash_features(\n"
        f"{text_return_indent}        _dflash_base_output,\n"
        f"{text_return_indent}        _dflash_features,\n"
        f"{text_return_indent}        required_fields=(\"last_hidden_state\",),\n"
        f"{text_return_indent}    )\n"
        f"{text_return_indent}# DFLASH_HIAI_V1:TEXT_OUTPUT_END\n"
    )

    causal_feature_branch = (
        f"{causal_return_indent}# DFLASH_HIAI_V1:CAUSAL_OUTPUT_BEGIN\n"
        f"{causal_return_indent}if output_dflash_features:\n"
        + _expression_assignment(
            causal_return_indent + "    ",
            "_dflash_causal_output",
            causal_expression,
        )
        + "\n"
        f"{causal_return_indent}    _dflash_features = getattr({causal.outputs_name}, \"dflash_features\", None)\n"
        f"{causal_return_indent}    if _dflash_features is None:\n"
        f"{causal_return_indent}        raise RuntimeError(\"text target returned no dflash_features\")\n"
        f"{causal_return_indent}    return attach_dflash_features(\n"
        f"{causal_return_indent}        _dflash_causal_output,\n"
        f"{causal_return_indent}        _dflash_features,\n"
        f"{causal_return_indent}        required_fields=(\"logits\",),\n"
        f"{causal_return_indent}    )\n"
        f"{causal_return_indent}# DFLASH_HIAI_V1:CAUSAL_OUTPUT_END\n"
    )

    return [
        _Insertion(
            import_line,
            "feature helper import",
            "\n# DFLASH_HIAI_V1:IMPORT_BEGIN\n"
            "from .dflash_target_features import (\n"
            "    DFlashFeatureCollector,\n"
            "    QWEN35_4B_DFLASH_TARGET_FEATURES,\n"
            ")\n"
            "from .dflash_hiai_feature_runtime import attach_dflash_features\n"
            "# DFLASH_HIAI_V1:IMPORT_END\n",
        ),
        _Insertion(
            _class_body_insertion_line(text.cls),
            "text feature metadata",
            _metadata_block(text_class_indent, role="text"),
        ),
        _Insertion(
            text_flag_line,
            "text feature flag",
            f"{_leading_indent(lines[text_flag_line])}# DFLASH_HIAI_V1:TEXT_FLAG\n"
            f"{_leading_indent(lines[text_flag_line])}output_dflash_features: bool = False,\n",
        ),
        _Insertion(
            text.loop.lineno - 1,
            "collector construction",
            f"{loop_indent}# DFLASH_HIAI_V1:COLLECTOR_BEGIN\n"
            f"{loop_indent}_dflash_collector = (\n"
            f"{loop_indent}    DFlashFeatureCollector(\n"
            f"{loop_indent}        QWEN35_4B_DFLASH_TARGET_FEATURES,\n"
            f"{loop_indent}        enabled=True,\n"
            f"{loop_indent}        detach=True,\n"
            f"{loop_indent}        clone=True,\n"
            f"{loop_indent}    )\n"
            f"{loop_indent}    if output_dflash_features\n"
            f"{loop_indent}    else None\n"
            f"{loop_indent})\n"
            f"{loop_indent}# DFLASH_HIAI_V1:COLLECTOR_END\n",
        ),
        _Insertion(
            text.decoder_assignment.end_lineno or text.decoder_assignment.lineno,
            "post-layer capture",
            f"{decoder_indent}# DFLASH_HIAI_V1:CAPTURE_BEGIN\n"
            f"{decoder_indent}if _dflash_collector is not None:\n"
            f"{decoder_indent}    _dflash_collector.capture({text.layer_index_name}, hidden_states)\n"
            f"{decoder_indent}# DFLASH_HIAI_V1:CAPTURE_END\n",
        ),
        _Insertion(
            text.norm_assignment.lineno - 1,
            "pre-norm finalization",
            f"{norm_indent}# DFLASH_HIAI_V1:FINALIZE_BEGIN\n"
            f"{norm_indent}_dflash_features = (\n"
            f"{norm_indent}    _dflash_collector.finalize()\n"
            f"{norm_indent}    if _dflash_collector is not None\n"
            f"{norm_indent}    else None\n"
            f"{norm_indent})\n"
            f"{norm_indent}if _dflash_features is not None:\n"
            f"{norm_indent}    if _dflash_features.shape != (*hidden_states.shape[:2], {FEATURE_WIDTH}):\n"
            f"{norm_indent}        raise RuntimeError(\"invalid DFlash feature shape\")\n"
            f"{norm_indent}    if _dflash_features.dtype != hidden_states.dtype:\n"
            f"{norm_indent}        raise RuntimeError(\"DFlash feature dtype changed during capture\")\n"
            f"{norm_indent}    if _dflash_features.device != hidden_states.device:\n"
            f"{norm_indent}        raise RuntimeError(\"DFlash feature device changed during capture\")\n"
            f"{norm_indent}# DFLASH_HIAI_V1:FINALIZE_END\n",
        ),
        _Insertion(
            text.default_return.lineno - 1,
            "opt-in text output",
            text_feature_branch,
        ),
        _Insertion(
            _class_body_insertion_line(causal.cls),
            "causal feature metadata",
            _metadata_block(causal_class_indent, role="causal"),
        ),
        _Insertion(
            causal_flag_line,
            "causal feature flag",
            f"{_leading_indent(lines[causal_flag_line])}# DFLASH_HIAI_V1:CAUSAL_FLAG\n"
            f"{_leading_indent(lines[causal_flag_line])}output_dflash_features: bool = False,\n",
        ),
        _Insertion(
            causal_call_kwargs_line,
            "causal-to-text feature forwarding",
            f"{causal_call_indent}# DFLASH_HIAI_V1:CAUSAL_FORWARD_BEGIN\n"
            f"{causal_call_indent}output_dflash_features=output_dflash_features,\n"
            f"{causal_call_indent}# DFLASH_HIAI_V1:CAUSAL_FORWARD_END\n",
        ),
        _Insertion(
            causal.default_return.lineno - 1,
            "opt-in causal output",
            causal_feature_branch,
        ),
    ]


def _apply_insertions(source: str, insertions: Iterable[_Insertion]) -> str:
    lines = source.splitlines(keepends=True)
    prepared = list(insertions)
    grouped: dict[int, list[_Insertion]] = {}
    for insertion in prepared:
        if insertion.position < 0 or insertion.position > len(lines):
            raise HiaiFeaturePatchError(
                f"invalid insertion position for {insertion.label}: {insertion.position}"
            )
        if not insertion.text.endswith("\n"):
            raise AssertionError(f"insertion {insertion.label} has no trailing newline")
        grouped.setdefault(insertion.position, []).append(insertion)
    # Two semantic anchors can legitimately share a physical boundary.  The
    # common case is a multi-line decoder assignment whose closing line is
    # immediately followed by the final norm: the post-layer (loop-indented)
    # capture must precede the pre-norm (function-indented) finalization.
    for position in sorted(grouped, reverse=True):
        combined = "".join(insertion.text for insertion in grouped[position])
        lines[position:position] = combined.splitlines(keepends=True)
    return "".join(lines)


def _import_bound_name(alias: ast.alias, *, from_import: bool) -> str:
    if alias.asname is not None:
        return alias.asname
    return alias.name if from_import else alias.name.split(".", 1)[0]


def _protected_helper_rebindings(
    module: ast.Module,
    *,
    allowed_import_node_ids: frozenset[int] = frozenset(),
) -> list[tuple[str, str, int]]:
    """Return every protected-name binding outside canonical helper imports."""

    events: list[tuple[str, str, int]] = []
    for node in ast.walk(module):
        lineno = int(getattr(node, "lineno", 0) or 0)
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            if node.id in _PROTECTED_HELPER_SYMBOLS:
                events.append((node.id, type(node.ctx).__name__, lineno))
        elif isinstance(node, ast.arg) and node.arg in _PROTECTED_HELPER_SYMBOLS:
            events.append((node.arg, "parameter", lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in _PROTECTED_HELPER_SYMBOLS:
                events.append((node.name, type(node).__name__, lineno))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            if id(node) in allowed_import_node_ids:
                continue
            from_import = isinstance(node, ast.ImportFrom)
            for alias in node.names:
                bound_name = _import_bound_name(alias, from_import=from_import)
                if bound_name in _PROTECTED_HELPER_SYMBOLS:
                    events.append((bound_name, "noncanonical import", lineno))
        elif isinstance(node, ast.ExceptHandler):
            if isinstance(node.name, str) and node.name in _PROTECTED_HELPER_SYMBOLS:
                events.append((node.name, "exception target", lineno))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                if name in _PROTECTED_HELPER_SYMBOLS:
                    events.append((name, type(node).__name__, lineno))
        elif isinstance(node, ast.MatchAs):
            if node.name in _PROTECTED_HELPER_SYMBOLS:
                events.append((node.name, "pattern target", lineno))
        elif isinstance(node, ast.MatchStar):
            if node.name in _PROTECTED_HELPER_SYMBOLS:
                events.append((node.name, "pattern star target", lineno))
        elif isinstance(node, ast.MatchMapping):
            if node.rest in _PROTECTED_HELPER_SYMBOLS:
                events.append((node.rest, "pattern rest target", lineno))
    return events


def _verify_canonical_helper_imports(module: ast.Module) -> None:
    canonical_ids: set[int] = set()
    for relative_module, expected_symbols in _CANONICAL_HELPER_IMPORTS:
        matches = [
            node
            for node in ast.walk(module)
            if isinstance(node, ast.ImportFrom)
            and node.level == 1
            and node.module == relative_module
        ]
        if len(matches) != 1 or matches[0] not in module.body:
            raise HiaiFeaturePatchError(
                f"expected one top-level canonical helper import from .{relative_module}"
            )
        helper_import = matches[0]
        actual_aliases = tuple(
            (alias.name, alias.asname) for alias in helper_import.names
        )
        expected_aliases = tuple((name, None) for name in expected_symbols)
        if actual_aliases != expected_aliases:
            raise HiaiFeaturePatchError(
                f"canonical helper import from .{relative_module} must be exactly "
                f"{expected_aliases!r}, got {actual_aliases!r}"
            )
        canonical_ids.add(id(helper_import))

    rebindings = _protected_helper_rebindings(
        module, allowed_import_node_ids=frozenset(canonical_ids)
    )
    if rebindings:
        details = ", ".join(
            f"{name} ({kind}, line {lineno})"
            for name, kind, lineno in rebindings
        )
        raise HiaiFeaturePatchError(
            f"protected helper symbol is rebound outside canonical imports: {details}"
        )


def _identifier_occurs(module: ast.Module, name: str) -> bool:
    for node in ast.walk(module):
        if isinstance(node, ast.Name) and node.id == name:
            return True
        if isinstance(node, ast.arg) and node.arg == name:
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return True
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            from_import = isinstance(node, ast.ImportFrom)
            if any(
                _import_bound_name(alias, from_import=from_import) == name
                for alias in node.names
            ):
                return True
        if isinstance(node, ast.ExceptHandler) and node.name == name:
            return True
        if isinstance(node, (ast.Global, ast.Nonlocal)) and name in node.names:
            return True
        if isinstance(node, (ast.MatchAs, ast.MatchStar)) and node.name == name:
            return True
        if isinstance(node, ast.MatchMapping) and node.rest == name:
            return True
    return False


def _argument_node(forward: ast.FunctionDef, name: str) -> ast.arg:
    arguments = [
        *forward.args.posonlyargs,
        *forward.args.args,
        *forward.args.kwonlyargs,
    ]
    matches = [argument for argument in arguments if argument.arg == name]
    if len(matches) != 1:
        raise HiaiFeaturePatchError(
            f"{forward.name} must have exactly one argument named {name}"
        )
    return matches[0]


def _verify_patch_owned_identifier_uses(
    module: ast.Module,
    *,
    allowed_roots: Sequence[ast.AST],
) -> None:
    """Reject any patch-owned name outside already verified AST subtrees."""

    allowed_ids = {
        id(node)
        for root in allowed_roots
        for node in ast.walk(root)
    }
    violations: list[tuple[str, str, int]] = []
    for node in ast.walk(module):
        lineno = int(getattr(node, "lineno", 0) or 0)
        if isinstance(node, ast.Name) and node.id in _PATCH_OWNED_LOCALS:
            if id(node) not in allowed_ids:
                violations.append((node.id, type(node.ctx).__name__, lineno))
        elif isinstance(node, ast.arg) and node.arg in _PATCH_OWNED_LOCALS:
            if id(node) not in allowed_ids:
                violations.append((node.arg, "parameter", lineno))
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name in _PATCH_OWNED_LOCALS:
                violations.append((node.name, type(node).__name__, lineno))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            from_import = isinstance(node, ast.ImportFrom)
            for alias in node.names:
                bound_name = _import_bound_name(alias, from_import=from_import)
                if bound_name in _PATCH_OWNED_LOCALS:
                    violations.append((bound_name, "import", lineno))
        elif isinstance(node, ast.ExceptHandler):
            if isinstance(node.name, str) and node.name in _PATCH_OWNED_LOCALS:
                violations.append((node.name, "exception target", lineno))
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            for name in node.names:
                if name in _PATCH_OWNED_LOCALS:
                    violations.append((name, type(node).__name__, lineno))
        elif isinstance(node, ast.MatchAs):
            if node.name in _PATCH_OWNED_LOCALS:
                violations.append((node.name, "pattern target", lineno))
        elif isinstance(node, ast.MatchStar):
            if node.name in _PATCH_OWNED_LOCALS:
                violations.append((node.name, "pattern star target", lineno))
        elif isinstance(node, ast.MatchMapping):
            if node.rest in _PATCH_OWNED_LOCALS:
                violations.append((node.rest, "pattern rest target", lineno))
        elif (
            isinstance(node, ast.keyword)
            and node.arg == "output_dflash_features"
            and id(node) not in allowed_ids
        ):
            violations.append((node.arg, "keyword", lineno))

    if violations:
        details = ", ".join(
            f"{name} ({kind}, line {lineno})"
            for name, kind, lineno in violations
        )
        raise HiaiFeaturePatchError(
            "patch-owned identifier is used outside locked statements: " + details
        )

    # An alias-free mutation can otherwise reach the same tensor through the
    # text output object (for example ``outputs.dflash_features.zero_()``).
    for node in ast.walk(module):
        if id(node) in allowed_ids:
            continue
        lineno = int(getattr(node, "lineno", 0) or 0)
        if isinstance(node, ast.Attribute) and node.attr == "dflash_features":
            raise HiaiFeaturePatchError(
                "dflash_features is accessed outside the locked sidecar route "
                f"at line {lineno}"
            )
        if isinstance(node, ast.Constant) and node.value == "dflash_features":
            raise HiaiFeaturePatchError(
                "dflash_features is accessed dynamically outside the locked "
                f"sidecar route at line {lineno}"
            )


def _verify_explicit_false(forward: ast.FunctionDef, owner: str) -> None:
    if not _has_argument(forward, "output_dflash_features"):
        raise HiaiFeaturePatchError(f"{owner} does not consume output_dflash_features")
    default = _argument_default(forward, "output_dflash_features")
    if not isinstance(default, ast.Constant) or default.value is not False:
        raise HiaiFeaturePatchError(
            f"{owner}.output_dflash_features must default to False"
        )


def _calls_named(node: ast.AST, name: str) -> list[ast.Call]:
    return [
        item
        for item in ast.walk(node)
        if isinstance(item, ast.Call)
        and (
            (isinstance(item.func, ast.Name) and item.func.id == name)
            or (isinstance(item.func, ast.Attribute) and item.func.attr == name)
        )
    ]


def _direct_if_for_flag(forward: ast.FunctionDef) -> list[ast.If]:
    return [
        node
        for node in forward.body
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "output_dflash_features"
    ]


def _same_ast(actual: ast.AST, expected: ast.AST) -> bool:
    """Compare executable structure while ignoring source coordinates."""

    return ast.dump(actual, include_attributes=False) == ast.dump(
        expected, include_attributes=False
    )


def _parsed_statement(source: str) -> ast.stmt:
    body = ast.parse(source).body
    if len(body) != 1:
        raise AssertionError("internal expected-source snippet has multiple statements")
    return body[0]


def _expected_collector_assignment() -> ast.stmt:
    return _parsed_statement(
        """\
_dflash_collector = (
    DFlashFeatureCollector(
        QWEN35_4B_DFLASH_TARGET_FEATURES,
        enabled=True,
        detach=True,
        clone=True,
    )
    if output_dflash_features
    else None
)
"""
    )


def _expected_capture_guard(layer_index_name: str) -> ast.stmt:
    return _parsed_statement(
        f"""\
if _dflash_collector is not None:
    _dflash_collector.capture({layer_index_name}, hidden_states)
"""
    )


def _expected_finalize_assignment() -> ast.stmt:
    return _parsed_statement(
        """\
_dflash_features = (
    _dflash_collector.finalize()
    if _dflash_collector is not None
    else None
)
"""
    )


def _expected_feature_validation() -> ast.stmt:
    return _parsed_statement(
        f"""\
if _dflash_features is not None:
    if _dflash_features.shape != (*hidden_states.shape[:2], {FEATURE_WIDTH}):
        raise RuntimeError("invalid DFlash feature shape")
    if _dflash_features.dtype != hidden_states.dtype:
        raise RuntimeError("DFlash feature dtype changed during capture")
    if _dflash_features.device != hidden_states.device:
        raise RuntimeError("DFlash feature device changed during capture")
"""
    )


def _verify_attach_call(
    call: ast.Call,
    *,
    base_name: str,
    required_field: str,
    owner: str,
) -> None:
    if not isinstance(call.func, ast.Name) or call.func.id != "attach_dflash_features":
        raise HiaiFeaturePatchError(f"{owner} does not return attach_dflash_features")
    if not (
        len(call.args) == 2
        and isinstance(call.args[0], ast.Name)
        and call.args[0].id == base_name
        and isinstance(call.args[1], ast.Name)
        and call.args[1].id == "_dflash_features"
    ):
        raise HiaiFeaturePatchError(
            f"{owner} passthrough must attach _dflash_features to {base_name}"
        )
    if len(call.keywords) != 1 or call.keywords[0].arg != "required_fields":
        raise HiaiFeaturePatchError(
            f"{owner} passthrough must declare only required_fields"
        )
    expected_fields = ast.Tuple(
        elts=[ast.Constant(value=required_field)], ctx=ast.Load()
    )
    if not _same_ast(call.keywords[0].value, expected_fields):
        raise HiaiFeaturePatchError(
            f"{owner} passthrough required_fields must be ({required_field!r},)"
        )


def _verify_text_feature_branch(branch: ast.If, default_return: ast.Return) -> None:
    if branch.orelse or len(branch.body) != 2:
        raise HiaiFeaturePatchError(
            "text feature branch must contain only base-output construction and return"
        )
    base_assignment, sidecar_return = branch.body
    if not (
        isinstance(base_assignment, (ast.Assign, ast.AnnAssign))
        and _assigned_name(base_assignment) == "_dflash_base_output"
        and default_return.value is not None
        and _assignment_value(base_assignment) is not None
        and _same_ast(_assignment_value(base_assignment), default_return.value)
    ):
        raise HiaiFeaturePatchError(
            "text feature branch must preserve the default output constructor"
        )
    if not isinstance(sidecar_return, ast.Return) or not isinstance(
        sidecar_return.value, ast.Call
    ):
        raise HiaiFeaturePatchError("text feature branch has no sidecar return")
    _verify_attach_call(
        sidecar_return.value,
        base_name="_dflash_base_output",
        required_field="last_hidden_state",
        owner="text feature branch",
    )


def _verify_causal_feature_branch(
    branch: ast.If,
    default_return: ast.Return,
    outputs_name: str,
) -> None:
    if branch.orelse or len(branch.body) != 4:
        raise HiaiFeaturePatchError(
            "causal feature branch must contain the locked four-statement sidecar route"
        )
    base_assignment, feature_assignment, missing_guard, sidecar_return = branch.body
    if not (
        isinstance(base_assignment, (ast.Assign, ast.AnnAssign))
        and _assigned_name(base_assignment) == "_dflash_causal_output"
        and default_return.value is not None
        and _assignment_value(base_assignment) is not None
        and _same_ast(_assignment_value(base_assignment), default_return.value)
    ):
        raise HiaiFeaturePatchError(
            "causal feature branch must preserve the default output constructor"
        )
    expected_feature_assignment = _parsed_statement(
        f'_dflash_features = getattr({outputs_name}, "dflash_features", None)\n'
    )
    if not _same_ast(feature_assignment, expected_feature_assignment):
        raise HiaiFeaturePatchError(
            "causal feature branch must read dflash_features from text-model outputs"
        )
    expected_missing_guard = _parsed_statement(
        """\
if _dflash_features is None:
    raise RuntimeError("text target returned no dflash_features")
"""
    )
    if not _same_ast(missing_guard, expected_missing_guard):
        raise HiaiFeaturePatchError(
            "causal feature branch must fail when text features are absent"
        )
    if not isinstance(sidecar_return, ast.Return) or not isinstance(
        sidecar_return.value, ast.Call
    ):
        raise HiaiFeaturePatchError("causal feature branch has no sidecar return")
    _verify_attach_call(
        sidecar_return.value,
        base_name="_dflash_causal_output",
        required_field="logits",
        owner="causal feature branch",
    )


def verify_source(source: str) -> dict[str, object]:
    """Verify a complete patch using markers and semantic source anchors."""

    marker_counts = {marker: source.count(marker) for marker in _MARKERS}
    invalid = {marker: count for marker, count in marker_counts.items() if count != 1}
    if invalid:
        details = ", ".join(f"{marker}={count}" for marker, count in invalid.items())
        raise HiaiFeaturePatchError(f"patch marker identity is incomplete: {details}")

    module = _parse(source)
    _verify_canonical_helper_imports(module)

    text = _locate_text_anchors(module)
    causal = _locate_causal_anchors(module)
    _verify_explicit_false(text.forward, "Qwen3_5TextModel.forward")
    _verify_explicit_false(causal.forward, "Qwen3_5ForCausalLM.forward")

    collector_calls = _calls_named(text.forward, "DFlashFeatureCollector")
    if len(collector_calls) != 1:
        raise HiaiFeaturePatchError(
            f"expected one DFlashFeatureCollector construction, found {len(collector_calls)}"
        )
    loop_index = text.forward.body.index(text.loop)
    if loop_index < 1:
        raise HiaiFeaturePatchError("collector construction is missing before decoder loop")
    collector_assignment = text.forward.body[loop_index - 1]
    if not (
        isinstance(collector_assignment, (ast.Assign, ast.AnnAssign))
        and _assigned_name(collector_assignment) == "_dflash_collector"
        and collector_calls[0] in list(ast.walk(collector_assignment))
    ):
        raise HiaiFeaturePatchError(
            "collector construction must be immediately before decoder loop"
        )
    if not _same_ast(collector_assignment, _expected_collector_assignment()):
        raise HiaiFeaturePatchError(
            "collector construction does not match the opt-in locked configuration"
        )
    collector_call = collector_calls[0]
    if not (
        len(collector_call.args) == 1
        and isinstance(collector_call.args[0], ast.Name)
        and collector_call.args[0].id == "QWEN35_4B_DFLASH_TARGET_FEATURES"
    ):
        raise HiaiFeaturePatchError("collector does not use the locked Qwen3.5-4B spec")
    collector_keywords = {keyword.arg: keyword.value for keyword in collector_call.keywords}
    for keyword_name in ("enabled", "detach", "clone"):
        value = collector_keywords.get(keyword_name)
        if not isinstance(value, ast.Constant) or value.value is not True:
            raise HiaiFeaturePatchError(
                f"collector keyword {keyword_name} must be explicitly True"
            )
    capture_calls = _calls_named(text.loop, "capture")
    if len(capture_calls) != 1:
        raise HiaiFeaturePatchError(
            f"expected one capture call in decoder loop, found {len(capture_calls)}"
        )
    capture_call = capture_calls[0]
    if capture_call.lineno <= (
        text.decoder_assignment.end_lineno or text.decoder_assignment.lineno
    ):
        raise HiaiFeaturePatchError("capture is not after the decoder-layer assignment")
    if len(capture_call.args) != 2:
        raise HiaiFeaturePatchError("capture must receive layer index and hidden_states")
    if not (
        isinstance(capture_call.args[0], ast.Name)
        and capture_call.args[0].id == text.layer_index_name
        and isinstance(capture_call.args[1], ast.Name)
        and capture_call.args[1].id == "hidden_states"
    ):
        raise HiaiFeaturePatchError("capture arguments do not match the decoder loop")
    decoder_index = text.loop.body.index(text.decoder_assignment)
    if decoder_index + 1 >= len(text.loop.body):
        raise HiaiFeaturePatchError("capture block is missing after decoder assignment")
    adjacent_capture = text.loop.body[decoder_index + 1]
    if not isinstance(adjacent_capture, ast.If) or capture_call not in list(
        ast.walk(adjacent_capture)
    ):
        raise HiaiFeaturePatchError(
            "capture must be the immediate statement after decoder assignment"
        )
    if not _same_ast(
        adjacent_capture, _expected_capture_guard(text.layer_index_name)
    ):
        raise HiaiFeaturePatchError(
            "post-layer capture guard does not match the locked semantics"
        )

    finalize_calls = _calls_named(text.forward, "finalize")
    if len(finalize_calls) != 1:
        raise HiaiFeaturePatchError(
            f"expected one feature finalization, found {len(finalize_calls)}"
        )
    if finalize_calls[0].lineno >= text.norm_assignment.lineno:
        raise HiaiFeaturePatchError("feature finalization must be before final norm")
    norm_index = text.forward.body.index(text.norm_assignment)
    if norm_index < 2:
        raise HiaiFeaturePatchError("feature finalization sequence is missing before norm")
    finalize_assignment = text.forward.body[norm_index - 2]
    finalize_validation = text.forward.body[norm_index - 1]
    if not (
        isinstance(finalize_assignment, (ast.Assign, ast.AnnAssign))
        and _assigned_name(finalize_assignment) == "_dflash_features"
        and finalize_calls[0] in list(ast.walk(finalize_assignment))
        and isinstance(finalize_validation, ast.If)
    ):
        raise HiaiFeaturePatchError(
            "feature finalization/validation must be immediately before final norm"
        )
    if not _same_ast(finalize_assignment, _expected_finalize_assignment()):
        raise HiaiFeaturePatchError(
            "feature finalization assignment does not match the locked semantics"
        )
    if not _same_ast(finalize_validation, _expected_feature_validation()):
        raise HiaiFeaturePatchError(
            "feature shape/dtype/device guards do not match the locked semantics"
        )

    text_feature_ifs = _direct_if_for_flag(text.forward)
    if len(text_feature_ifs) != 1:
        raise HiaiFeaturePatchError("text forward must have one opt-in output branch")
    _verify_text_feature_branch(text_feature_ifs[0], text.default_return)
    text_return_index = text.forward.body.index(text.default_return)
    if text_return_index < 1 or text.forward.body[text_return_index - 1] is not text_feature_ifs[0]:
        raise HiaiFeaturePatchError(
            "text feature output branch must immediately precede the default return"
        )

    forwarded = [
        keyword
        for keyword in causal.model_call.keywords
        if keyword.arg == "output_dflash_features"
    ]
    if len(forwarded) != 1 or not (
        isinstance(forwarded[0].value, ast.Name)
        and forwarded[0].value.id == "output_dflash_features"
    ):
        raise HiaiFeaturePatchError(
            "causal-LM must forward the explicit feature flag exactly once"
        )
    causal_feature_ifs = _direct_if_for_flag(causal.forward)
    if len(causal_feature_ifs) != 1:
        raise HiaiFeaturePatchError("causal forward must have one opt-in output branch")
    _verify_causal_feature_branch(
        causal_feature_ifs[0], causal.default_return, causal.outputs_name
    )
    causal_return_index = causal.forward.body.index(causal.default_return)
    if (
        causal_return_index < 1
        or causal.forward.body[causal_return_index - 1] is not causal_feature_ifs[0]
    ):
        raise HiaiFeaturePatchError(
            "causal feature output branch must immediately precede the default return"
        )

    _verify_patch_owned_identifier_uses(
        module,
        allowed_roots=(
            _argument_node(text.forward, "output_dflash_features"),
            collector_assignment,
            adjacent_capture,
            finalize_assignment,
            finalize_validation,
            text_feature_ifs[0],
            _argument_node(causal.forward, "output_dflash_features"),
            forwarded[0],
            causal_feature_ifs[0],
        ),
    )

    for cls in (text.cls, causal.cls):
        class_assignments = {
            _assigned_name(node): _assignment_value(node)
            for node in cls.body
            if isinstance(node, (ast.Assign, ast.AnnAssign))
        }
        expected = {
            "dflash_feature_contract_id": PATCH_CONTRACT_ID,
            "dflash_feature_source": FEATURE_SOURCE,
            "dflash_feature_capture_point": CAPTURE_POINT,
        }
        for name, value in expected.items():
            actual = class_assignments.get(name)
            if not isinstance(actual, ast.Constant) or actual.value != value:
                raise HiaiFeaturePatchError(f"{cls.name}.{name} metadata is invalid")

    source_sha256 = _sha256_text(source)
    return {
        "status": "verified",
        "patch_contract_id": PATCH_CONTRACT_ID,
        "feature_source": FEATURE_SOURCE,
        "capture_point": CAPTURE_POINT,
        "source_sha256": source_sha256,
        "dflash_feature_source": FEATURE_SOURCE,
        "dflash_feature_capture_point": CAPTURE_POINT,
        "dflash_feature_patch_sha256": source_sha256,
        "layer_ids": list(FEATURE_LAYER_IDS),
        "target_hidden_size": TARGET_HIDDEN_SIZE,
        "feature_width": FEATURE_WIDTH,
        "feature_shape": "[B,S,20480]",
        "feature_dtype": "preserve_target_hidden_dtype",
        "feature_device": "preserve_target_hidden_device",
        "formal_route_dtype": "float16",
        "default_abi_preserved": True,
        "kwargs_leak_blocked": True,
        "custom_operators_modified": False,
        "runtime_relative_imports": [
            ".dflash_target_features",
            ".dflash_hiai_feature_runtime",
        ],
        "anchors": {
            "decoder_loop_line": text.loop.lineno,
            "post_layer_capture_line": capture_call.lineno,
            "feature_finalize_line": finalize_calls[0].lineno,
            "final_norm_line": text.norm_assignment.lineno,
            "causal_model_call_line": causal.model_call.lineno,
        },
    }


def patch_source(source: str) -> PatchResult:
    """Patch source in memory, returning an idempotent verified result."""

    marker_presence = [marker in source for marker in _MARKERS]
    if any(marker_presence):
        if not all(marker_presence):
            raise HiaiFeaturePatchError(
                "source contains a partial DFlash HIAI V1 patch; refusing to repair it"
            )
        report = verify_source(source)
        report = dict(report)
        report.update(
            {
                "changed": False,
                "source_sha256_before": _sha256_text(source),
                "source_sha256_after": _sha256_text(source),
            }
        )
        return PatchResult(source=source, changed=False, report=report)

    module = _parse(source)
    conflicting = sorted(
        symbol
        for symbol in (
            *_IMPORT_SYMBOLS,
            *_RUNTIME_IMPORT_SYMBOLS,
            *_PATCH_OWNED_LOCALS,
        )
        if _identifier_occurs(module, symbol)
    )
    if conflicting:
        raise HiaiFeaturePatchError(
            "source contains unowned DFlash patch symbols without the expected identity: "
            + ", ".join(conflicting)
        )
    text = _locate_text_anchors(module)
    causal = _locate_causal_anchors(module)
    insertions = _build_insertions(source, module, text, causal)
    patched = _apply_insertions(source, insertions)
    report = verify_source(patched)
    report = dict(report)
    report.update(
        {
            "changed": True,
            "source_sha256_before": _sha256_text(source),
            "source_sha256_after": _sha256_text(patched),
        }
    )
    return PatchResult(source=patched, changed=True, report=report)


def check_file(path: str | os.PathLike[str]) -> dict[str, object]:
    source_path = Path(path)
    source = source_path.read_text(encoding="utf-8")
    report = dict(verify_source(source))
    report["path"] = str(source_path.resolve())
    return report


def _atomic_write(path: Path, content: str, *, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _enforce_basename(path: Path, allow_noncanonical: bool) -> None:
    if not allow_noncanonical and path.name != EXPECTED_SOURCE_BASENAME:
        raise HiaiFeaturePatchError(
            f"expected receiver source basename {EXPECTED_SOURCE_BASENAME!r}, got {path.name!r}"
        )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Patch/verify receiver-owned Qwen3.5 HIAI DFlash V1 features"
    )
    parser.add_argument("--source", required=True, type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true")
    actions.add_argument("--check", action="store_true")
    actions.add_argument("--output", type=Path)
    actions.add_argument("--in-place", action="store_true")
    parser.add_argument("--force", action="store_true", help="replace --output if it exists")
    parser.add_argument("--backup-suffix", default=".pre-dflash-v1")
    parser.add_argument("--show-diff", action="store_true")
    parser.add_argument("--allow-noncanonical-basename", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        _enforce_basename(args.source, args.allow_noncanonical_basename)
        original = args.source.read_text(encoding="utf-8")
        if args.check:
            report = dict(verify_source(original))
            report["path"] = str(args.source.resolve())
        else:
            result = patch_source(original)
            report = dict(result.report)
            report["source_path"] = str(args.source.resolve())
            if args.show_diff:
                diff = difflib.unified_diff(
                    original.splitlines(keepends=True),
                    result.source.splitlines(keepends=True),
                    fromfile=str(args.source),
                    tofile=str(args.output or args.source) + ".dflash-v1",
                )
                sys.stderr.writelines(diff)

            if args.output is not None:
                output = args.output
                if output.resolve() == args.source.resolve():
                    raise HiaiFeaturePatchError("use --in-place to replace the source")
                if output.exists() and not args.force:
                    raise HiaiFeaturePatchError(
                        f"output exists (pass --force to replace it): {output}"
                    )
                mode = args.source.stat().st_mode & 0o777
                _atomic_write(output, result.source, mode=mode)
                report["output_path"] = str(output.resolve())
            elif args.in_place and result.changed:
                if not args.backup_suffix or "/" in args.backup_suffix:
                    raise HiaiFeaturePatchError("backup suffix must be a non-empty filename suffix")
                backup = args.source.with_name(args.source.name + args.backup_suffix)
                if backup.exists():
                    raise HiaiFeaturePatchError(
                        f"backup already exists; refusing to replace it: {backup}"
                    )
                shutil.copy2(args.source, backup)
                mode = args.source.stat().st_mode & 0o777
                _atomic_write(args.source, result.source, mode=mode)
                report["backup_path"] = str(backup.resolve())
                report["output_path"] = str(args.source.resolve())
            elif args.in_place:
                report["output_path"] = str(args.source.resolve())

        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (HiaiFeaturePatchError, OSError, UnicodeError) as exc:
        print(
            json.dumps(
                {
                    "status": "failed_closed",
                    "patch_contract_id": PATCH_CONTRACT_ID,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
