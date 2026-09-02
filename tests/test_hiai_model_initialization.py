from __future__ import annotations

import ast
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODEL_SOURCES = (
    ROOT / "models" / "modeling_qwen3_5_hiai_nd.py",
    ROOT / "models" / "modeling_qwen3_5_hiai_nd_dflash_rollback.py",
)


def _method(source: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{source}: missing {class_name}.{method_name}")


def _called_attributes(method: ast.FunctionDef) -> list[str]:
    return [
        node.func.attr
        for node in ast.walk(method)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    ]


@pytest.mark.parametrize("source", MODEL_SOURCES, ids=lambda path: path.stem)
def test_missing_weight_initialization_does_not_rebuild_rope_cache(
    source: Path,
) -> None:
    method = _method(source, "Qwen3_5PreTrainedModel", "_init_weights")

    assert "_set_cos_sin_cache" not in _called_attributes(method)


@pytest.mark.parametrize("source", MODEL_SOURCES, ids=lambda path: path.stem)
def test_cached_rotary_constructor_still_builds_rope_cache(source: Path) -> None:
    method = _method(source, "Qwen3_5RotaryEmbedding1", "__init__")

    assert _called_attributes(method).count("_set_cos_sin_cache") == 1
