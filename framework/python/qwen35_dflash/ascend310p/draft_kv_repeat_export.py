"""Gate cached Draft K/V head replication using graph edges, not node names.

Only the GQA path from a cache ScatterElements through Concat/Unsqueeze is in
scope. Other BroadcastTo operations are permitted. These are AIR lowering
checks, not evidence that the compiled OM executes on the receiver.
"""
from __future__ import annotations

import ast
from pathlib import Path
import re
import struct
from typing import Any, Mapping

from .draft_cache_export import _edge, _nodes

DRAFT_KV_REPEAT_POLICY = "gqa-head-repeat-tile-v1"

# Parse selected small Const/Unsqueeze nodes, including the string-encoded
# AttrDef dialect emitted by TorchAir's node/op debug dumps. Never eval a dump.
_PROTO_TOKEN = re.compile(
    r'\s+|\#[^\n]*|"(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\'|'
    r'-?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9.]*|[{}:<>;,]'
)


def _message(text: str) -> dict[str, list[Any]]:
    tokens = []
    end = 0
    for match in _PROTO_TOKEN.finditer(text):
        if match.start() != end:
            raise ValueError("Draft KV repeat audit cannot parse GE attribute")
        end = match.end()
        token = match.group()
        if not token.isspace() and not token.startswith("#") and token not in {";", ","}:
            tokens.append(token)
    if end != len(text):
        raise ValueError("Draft KV repeat audit cannot parse GE attribute")
    position = 0

    def parse(close=None):
        nonlocal position
        result = {}
        while position < len(tokens):
            name = tokens[position]
            position += 1
            if name == close:
                return result
            if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9.]*", name):
                raise ValueError("Draft KV repeat audit has invalid GE field")
            if position < len(tokens) and tokens[position] == ":":
                position += 1
            if position == len(tokens):
                raise ValueError("Draft KV repeat audit has incomplete GE field")
            value = tokens[position]
            position += 1
            if value in {"{", "<"}:
                value = parse("}" if value == "{" else ">")
            elif value.startswith(("'", '"')):
                value = ast.literal_eval(value)
                while position < len(tokens) and tokens[position].startswith(("'", '"')):
                    value += ast.literal_eval(tokens[position])
                    position += 1
            elif re.fullmatch(r"-?\d+", value):
                value = int(value)
            elif value in {"}", ">", ":"}:
                raise ValueError("Draft KV repeat audit has invalid GE value")
            result.setdefault(name, []).append(value)
        if close:
            raise ValueError("Draft KV repeat audit has incomplete GE message")
        return result

    return parse()


def _one(message: Mapping[str, Any], field: str) -> Any:
    values = message.get(field, [])
    if len(values) != 1:
        raise ValueError(f"Draft KV repeat audit requires one GE {field}")
    return values[0]


def _attribute(node: Mapping[str, Any], name: str) -> dict[str, Any]:
    root = _message(node["text"])
    message = _one(root, "node" if "node" in root else "op")
    found = [item for item in message.get("attr", []) if _one(item, "key") == name]
    if len(found) != 1:
        raise ValueError(f"Draft KV repeat audit requires one GE {name} attribute")
    value = _one(found[0], "value")
    if "s" in value:  # node/op debug representation serializes AttrDef as text.
        value = _message(_one(value, "s"))
    return value


def _const_ints(node: Mapping[str, Any]) -> list[int]:
    if node.get("type") != "Const":
        raise ValueError("Draft KV repeat audit requires Const repeats")
    tensor = _one(_attribute(node, "value"), "t")
    desc = _one(tensor, "desc")
    dtype = _one(desc, "dtype")
    formats = {"DT_INT64": ("q", 8), "DT_INT32": ("i", 4)}
    if dtype not in formats:
        raise ValueError("Draft KV repeat audit requires INT32/INT64 Const repeats")
    dims = _one(desc, "shape").get("dim", [])
    if len(dims) != 1 or type(dims[0]) is not int or not 1 <= dims[0] <= 5:
        raise ValueError("Draft KV repeat audit has invalid Const repeats shape")
    raw = _one(tensor, "data")
    if not isinstance(raw, str):
        raise ValueError("Draft KV repeat audit has invalid Const data")
    data = raw.encode("latin1")
    code, width = formats[dtype]
    if len(data) != width * dims[0]:
        raise ValueError("Draft KV repeat audit has invalid Const repeats bytes")
    return list(struct.unpack("<" + code * dims[0], data))


def _producer(nodes, edge):
    """Follow only shape/value-preserving Identity nodes; reject cyclic IR."""
    visited = set()
    while True:
        name, port = _edge(edge)
        if port != "0" or name in visited:
            return None, {}
        visited.add(name)
        node = nodes.get(name, {})
        if node.get("type") != "Identity":
            return name, node
        inputs = node["inputs"]
        if len(inputs) != 1:
            return None, {}
        edge = inputs[0]


def _cache_source(nodes, repeat):
    inputs = repeat["inputs"]
    if not inputs:
        return None
    _, unsqueeze = _producer(nodes, inputs[0])
    if unsqueeze.get("type") != "Unsqueeze" or len(unsqueeze["inputs"]) != 1:
        return None
    _, concat = _producer(nodes, unsqueeze["inputs"][0])
    if concat.get("type") not in {"ConcatD", "ConcatV2"}:
        return None
    writes = []
    for edge in concat["inputs"]:
        name, producer = _producer(nodes, edge)
        if producer.get("type") == "ScatterElements":
            writes.append(name)
    if not writes:
        return None  # unrelated mask/state replication
    if len(writes) != 1:
        raise ValueError("Draft KV repeat audit has ambiguous cache source")
    axes = _one(_attribute(unsqueeze, "axes"), "list").get("i", [])
    if axes != [2]:
        raise ValueError("Draft KV repeat audit requires singleton group axis 2")
    return writes[0]


def _has_reshape_consumer(nodes, repeat_name):
    return any(
        node["type"] == "Reshape" and node["inputs"]
        and _producer(nodes, node["inputs"][0])[0] == repeat_name
        for node in nodes.values()
    )


def _contract(metadata: Mapping[str, Any]) -> tuple[int, int] | None:
    policy = metadata.get("draft_kv_repeat_policy")
    fields = ("draft_kv_repeat_layers", "draft_kv_repeat_groups")
    if policy is None and not any(key in metadata for key in fields):
        return None
    values = tuple(metadata.get(key) for key in fields)
    if policy != DRAFT_KV_REPEAT_POLICY or any(type(v) is not int or v <= 0 for v in values):
        raise ValueError("invalid Draft KV repeat policy/layer/group contract")
    return values


def audit_draft_kv_repeat_export(
    metadata: Mapping[str, Any], graph_dir: Path, *, relative_to: Path,
) -> dict[str, Any] | None:
    contract = _contract(metadata)
    if contract is None:
        return None
    layers, groups = contract
    repeats = [1, 1, groups, 1, 1]
    files = []
    for path in sorted(graph_dir.rglob("dynamo.pbtxt")):
        nodes = _nodes(path.read_text(encoding="utf-8"))
        writes = {name for name, node in nodes.items() if node["type"] == "ScatterElements"}
        if len(writes) != 2 * layers:
            raise ValueError(f"Draft KV repeat audit expected {2 * layers} cache writes")
        bindings = []
        for name, node in nodes.items():
            if node["type"] not in {"Tile", "BroadcastTo"}:
                continue
            write = _cache_source(nodes, node)
            if write is None:
                continue
            if groups == 1 or node["type"] != "Tile":
                raise ValueError(f"Draft KV repeat audit: {name} must use GQA Tile, not BroadcastTo")
            inputs = node["inputs"]
            if len(inputs) != 2:
                raise ValueError("Draft KV repeat audit requires two Tile inputs")
            const_name, port = _edge(inputs[1])
            if port != "0" or _const_ints(nodes.get(const_name, {})) != repeats:
                raise ValueError("Draft KV repeat audit requires exact Const [1,1,groups,1,1]")
            if not _has_reshape_consumer(nodes, name):
                raise ValueError("Draft KV repeat audit requires a Reshape consumer")
            bindings.append({"cache_write": write, "repeat": name, "multiples": const_name})
        expected = 2 * layers if groups > 1 else 0
        if len(bindings) != expected or (groups > 1 and
                {item["cache_write"] for item in bindings} != writes):
            raise ValueError(
                "Draft KV repeat audit: not every cache K/V write has one head Tile; "
                "retain AIR diagnostics and re-export, do not compile the old BroadcastTo path"
            )
        files.append({"path": path.relative_to(relative_to).as_posix(),
                      "cache_write_count": len(writes), "bindings": bindings})
    if not files:
        raise ValueError("Draft KV repeat audit requires dynamo.pbtxt")
    return {"status": "PASS", "policy": DRAFT_KV_REPEAT_POLICY,
            "layers": layers, "groups": groups, "repeats": repeats,
            "files": files, "claim_boundary": "Serialized AIR K/V head replication only; not OM/device validation."}


def validated_draft_kv_repeat_audit(graph: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = graph.get("metadata", {})
    if not isinstance(metadata, Mapping):
        raise ValueError("Draft KV repeat metadata must be an object")
    contract = _contract(metadata)
    record = graph.get("draft_kv_repeat_audit")
    if contract is None and record is None:
        return None  # Legacy rollback bundles do not claim this fix.
    if not isinstance(record, Mapping) or contract is None:
        raise ValueError("AIR requires a Draft KV repeat audit; re-export AIR")
    layers, groups = contract
    repeats = record.get("repeats")
    if (record.get("status") != "PASS" or record.get("policy") != DRAFT_KV_REPEAT_POLICY
            or type(record.get("layers")) is not int or record["layers"] != layers
            or type(record.get("groups")) is not int or record["groups"] != groups
            or not isinstance(repeats, list) or any(type(v) is not int for v in repeats)
            or repeats != [1, 1, groups, 1, 1]):
        raise ValueError("AIR Draft KV repeat audit differs from contract")
    files = record.get("files")
    if not isinstance(files, list) or not files:
        raise ValueError("AIR Draft KV repeat audit has no GE evidence")
    paths = set()
    for item in files:
        if not isinstance(item, Mapping):
            raise ValueError("AIR Draft KV repeat file evidence is invalid")
        path = item.get("path")
        if (not isinstance(path, str) or not path or Path(path).is_absolute()
                or ".." in Path(path).parts or path in paths):
            raise ValueError("AIR Draft KV repeat evidence path is invalid")
        paths.add(path)
        bindings = item.get("bindings")
        expected = 2 * layers if groups > 1 else 0
        if (type(item.get("cache_write_count")) is not int or item["cache_write_count"] != 2 * layers
                or not isinstance(bindings, list) or len(bindings) != expected
                or any(not isinstance(b, Mapping) or any(
                    not isinstance(b.get(key), str) or not b[key]
                    for key in ("cache_write", "repeat", "multiples")) for b in bindings)):
            raise ValueError("AIR Draft KV repeat bindings differ from contract")
        if any(len({b[key] for b in bindings}) != expected for key in ("cache_write", "repeat")):
            raise ValueError("AIR Draft KV repeat audit contains duplicate bindings")
    return dict(record)
