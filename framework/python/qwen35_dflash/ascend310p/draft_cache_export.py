"""Audit the serialized Draft KV scatter-index lowering, not generated names.

The receiver incident is a positive infer-shape followed by BroadcastTo tiling
failure. This gate proves only the intended AIR lowering; it is not an OM/device
correctness claim and must not ban unrelated BroadcastTo operations.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Mapping

DRAFT_CACHE_INDEX_POLICY = "static-repeat-tile-v1"

_NODE_START = re.compile(r"(?m)^[ \t]*(?:node|op)[ \t]*\{")
_TOKEN = re.compile(
    r'"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|\#[^\n]*|[{}:]|[A-Za-z_][A-Za-z_0-9]*'
)


def _nodes(text: str) -> dict[str, dict[str, Any]]:
    """Read node-level name/type/input fields in both TorchAir dump dialects.

    Quoted nested tensor descriptors are single tokens: their embedded node-like
    text must never be mistaken for graph edges. Unsupported/ambiguous dumps
    fail closed instead of making a name/count-only assertion.
    """
    starts = list(_NODE_START.finditer(text))
    nodes: dict[str, dict[str, Any]] = {}
    for offset, start in enumerate(starts):
        end = starts[offset + 1].start() if offset + 1 < len(starts) else len(text)
        fields: dict[str, list[str]] = {}
        depth = 0
        field = None
        for match in _TOKEN.finditer(text, start.start(), end):
            token = match.group()
            if token.startswith("#"):
                continue
            if token == "{":
                depth += 1
                field = None
            elif token == "}":
                depth -= 1
                field = None
                if depth == 0:
                    break
            elif depth == 1:
                if token in {"name", "op", "type", "input"}:
                    field = token
                elif token == ":":
                    continue
                elif token.startswith(('"', "'")) and field is not None:
                    fields.setdefault(field, []).append(ast.literal_eval(token))
                    field = None
                else:
                    field = None
        names = fields.get("name", [])
        kinds = fields.get("op", []) + fields.get("type", [])
        if depth != 0 or len(names) != 1 or not kinds or len(set(kinds)) != 1:
            raise ValueError("Draft cache index audit cannot parse GE node fields")
        name = names[0]
        if name in nodes:
            raise ValueError(f"Draft cache index audit found duplicate node {name!r}")
        nodes[name] = {
            "type": kinds[0], "inputs": fields.get("input", []),
            "text": text[start.start():match.end()],
        }
    return nodes


def _layer_count(metadata: Mapping[str, Any]) -> int | None:
    policy = metadata.get("draft_cache_index_policy")
    if policy is None:
        return None
    layers = metadata.get("draft_cache_index_layers")
    if policy != DRAFT_CACHE_INDEX_POLICY or type(layers) is not int or layers <= 0:
        raise ValueError("invalid Draft cache index policy/layer contract")
    return layers


def audit_draft_cache_index_export(
    metadata: Mapping[str, Any], graph_dir: Path, *, relative_to: Path,
) -> dict[str, Any] | None:
    layers = _layer_count(metadata)
    if layers is None:
        return None
    files = []
    for path in sorted(graph_dir.rglob("dynamo.pbtxt")):
        nodes = _nodes(path.read_text(encoding="utf-8"))
        scatters = [(name, node) for name, node in nodes.items()
                    if node["type"] == "ScatterElements"]
        if len(scatters) != 2 * layers:
            raise ValueError(
                f"Draft cache index audit expected {2 * layers} ScatterElements "
                f"nodes, found {len(scatters)} in {path}; retain AIR diagnostics"
            )
        bindings = []
        for name, scatter in scatters:
            inputs = scatter["inputs"]
            if len(inputs) < 3:
                raise ValueError(f"Draft cache index audit: {name} has no indices")
            index_name, port = _edge(inputs[1])
            index = nodes.get(index_name, {})
            if port != "0" or index.get("type") != "Tile":
                raise ValueError(
                    f"Draft cache index audit: {name} indices come from "
                    f"{index.get('type', 'missing')} {index_name!r}, expected Tile; "
                    "do not compile the old dynamic BroadcastTo scatter path"
                )
            tile_inputs = index.get("inputs", [])
            multiples_name, multiples_port = (
                _edge(tile_inputs[1]) if len(tile_inputs) >= 2 else ("", "")
            )
            multiples = nodes.get(multiples_name, {})
            if multiples_port != "0" or multiples.get("type") != "Const":
                raise ValueError(
                    f"Draft cache index audit: {index_name} repeats must be Const"
                )
            bindings.append({"scatter": name, "index": index_name})
        files.append({
            "path": path.relative_to(relative_to).as_posix(),
            "scatter_count": len(scatters),
            "bindings": bindings,
        })
    if not files:
        raise ValueError("Draft cache index audit requires dynamo.pbtxt")
    return {
        "status": "PASS",
        "policy": DRAFT_CACHE_INDEX_POLICY,
        "layers": layers,
        "index_ge_op": "Tile",
        "repeats_ge_op": "Const",
        "files": files,
        "claim_boundary": "Serialized AIR index lowering only; not OM/device validation.",
    }


def _edge(value: str) -> tuple[str, str]:
    parts = value.rsplit(":", 1)
    return (parts[0], parts[1]) if len(parts) == 2 else (value, "0")


def validated_draft_cache_index_audit(
    graph: Mapping[str, Any],
) -> dict[str, Any] | None:
    metadata = graph.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("AIR graph metadata must be an object")
    layers = _layer_count(metadata)
    record = graph.get("draft_cache_index_audit")
    if layers is None and record is None:
        return None  # Legacy bundles remain available for explicit rollback.
    if not isinstance(record, Mapping) or layers is None:
        raise ValueError("AIR requires a Draft cache index audit; re-export AIR")
    if (record.get("status") != "PASS" or
            record.get("policy") != DRAFT_CACHE_INDEX_POLICY or
            record.get("layers") != layers or
            record.get("index_ge_op") != "Tile" or
            record.get("repeats_ge_op") != "Const"):
        raise ValueError("AIR Draft cache index audit is not passing")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("AIR Draft cache index audit has no GE evidence")
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("AIR Draft cache index file evidence is invalid")
        bindings = item.get("bindings")
        if (not isinstance(item.get("path"), str) or not item["path"] or
                item.get("scatter_count") != 2 * layers or
                not isinstance(bindings, list) or len(bindings) != 2 * layers or
                any(not isinstance(binding, Mapping) or
                    not isinstance(binding.get("scatter"), str) or
                    not binding["scatter"] or not isinstance(binding.get("index"), str) or
                    not binding["index"] for binding in bindings)):
            raise ValueError("AIR Draft cache index bindings differ from the contract")
        if len({binding["scatter"] for binding in bindings}) != 2 * layers:
            raise ValueError("AIR Draft cache index audit contains duplicate scatter bindings")
    return dict(record)
