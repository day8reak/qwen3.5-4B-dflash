"""Complete chunk ABI validation and hash-locked native C++ loading plans."""

from __future__ import annotations

import json
from pathlib import Path

from .utils import contained_path, load_json_object, require_run_output, sha256_file

ABI = "qwen35-dflash-chunk-v1"
ATTENTION_EXPORT_POLICY = "receiver_adn_all_seq_lengths_q_static_capacity_causal_mask"
ROLES = ("target_prefill", "target_decode", "target_verify", "draft")
DTYPES = {"int64": 8, "int16": 2, "float16": 2, "float32": 4}


def descriptor(name, dtype, shape):
    return {"name": name, "dtype": dtype, "shape": shape}


def _validate_tensor(tensor):
    if not isinstance(tensor, dict) or set(tensor) != {"name", "dtype", "shape"}:
        raise ValueError("invalid tensor ABI descriptor")
    if not isinstance(tensor["name"], str) or not tensor["name"].isidentifier():
        raise ValueError("invalid tensor name")
    if tensor["dtype"] not in DTYPES:
        raise ValueError("unsupported tensor dtype")
    shape = tensor["shape"]
    if (
        not isinstance(shape, list)
        or not shape
        or len(shape) > 8
        or any(type(d) is not int or d <= 0 for d in shape)
    ):
        raise ValueError("tensor ABI requires positive static dimensions")
    size = DTYPES[tensor["dtype"]]
    for dim in shape:
        size *= dim
    if size > 2**40:
        raise ValueError("tensor ABI exceeds the size limit")


def expected_signatures(c):
    states, drafts = c["target_states"], c["draft_states"]
    start, valid = (
        descriptor("start_position", "int64", [1]),
        descriptor("valid_rows", "int16", [1]),
    )
    feature = lambda rows: descriptor(
        "features", "float16", [1, rows, c["feature_width"]]
    )
    result = {}
    for name, rows, verify in (
        ("target_prefill", 64, False),
        ("target_decode", 1, False),
        ("target_verify", 16, True),
    ):
        outputs = [descriptor("target_top1", "int64", [1, rows if verify else 1])]
        if verify:
            outputs.append(descriptor("accepted_count", "int64", [1]))
        if rows != 1:
            outputs.append(feature(64))
        outputs += states
        result[name] = {
            "inputs": [
                descriptor("input_ids", "int64", [1, rows]),
                start,
                valid,
                *states,
            ],
            "outputs": outputs,
        }
    result["draft"] = {
        "inputs": [
            feature(64),
            start,
            valid,
            descriptor("anchor", "int64", [1]),
            *drafts,
        ],
        "outputs": [descriptor("draft_top1", "int64", [1, 15]), *drafts],
    }
    return result


def validate_incremental_bundle(graphs):
    candidates = [
        g for g in graphs if g.get("metadata", {}).get("incremental_contract")
    ]
    if not candidates:
        return None
    names = {g["name"] for g in graphs}
    required = set(ROLES) - {"target_decode"}
    if (
        len(candidates) != len(graphs)
        or len(names) != len(graphs)
        or names not in (set(ROLES), required)
    ):
        raise ValueError(
            "incremental bundle needs prefill, verify, draft and optional ordinary decode (four graphs at most)"
        )
    c = candidates[0]["metadata"]["incremental_contract"]
    if c.get("abi") != ABI or c.get("block_size") != 16 or c.get("prefill_rows") != 64:
        raise ValueError("unsupported incremental ABI")
    if c.get("attention_export") != ATTENTION_EXPORT_POLICY:
        raise ValueError(
            "unsupported attention export ABI: regenerate AIR with "
            "all_seq_lengths_q and no integer pse_shift in a new bundle directory"
        )
    capacity = c.get("capacity")
    if (
        type(capacity) is not int
        or capacity < 64
        or capacity % 64
        or capacity > 32704
        or c.get("cache_capacity") != capacity + 64
    ):
        raise ValueError("invalid logical/scratch cache capacity")
    for key in ("vocab_size", "feature_width"):
        if type(c.get(key)) is not int or c[key] <= 0:
            raise ValueError(f"invalid {key}")
    for key in ("target_states", "draft_states", "capsules"):
        tensors = c.get(key)
        if not isinstance(tensors, list) or not tensors:
            raise ValueError(f"missing {key}")
        for tensor in tensors:
            _validate_tensor(tensor)
        if len({t["name"] for t in tensors}) != len(tensors):
            raise ValueError(f"duplicate tensor in {key}")
    names = {s["name"] for s in c["target_states"]}
    gdn, kv = c["gdn_states"], c["kv_states"]
    if (
        not gdn
        or not kv
        or len(gdn) % 2
        or len(kv) % 2
        or len(gdn) + len(kv) != len(names)
        or set(gdn) & set(kv)
        or set(gdn) | set(kv) != names
    ):
        raise ValueError("state partition is inconsistent")
    if len(c["capsules"]) != len(gdn) // 2 * 7:
        raise ValueError("GDR capsule count differs from linear layer count")
    expected = expected_signatures(c)
    for graph in graphs:
        if graph["metadata"]["incremental_contract"] != c:
            raise ValueError("incremental graph contracts differ")
        signature = graph["metadata"].get("tensor_abi")
        if signature != expected[graph["name"]]:
            raise ValueError(f"incremental tensor ABI differs: {graph['name']}")
        if graph.get("role") != graph["name"].replace("_", "-"):
            raise ValueError("incremental graph role differs")
        for direction in ("input", "output"):
            tensors = signature[direction + "s"]
            for tensor in tensors:
                _validate_tensor(tensor)
            if list(graph[direction + "_names"]) != [t["name"] for t in tensors]:
                raise ValueError("ordered graph tensor names differ from ABI")
    return c


def write_incremental_plan(deployment_manifest, output, *, mode="paired"):
    if mode not in {"paired", "ordinary", "dflash"}:
        raise ValueError("mode must be paired, ordinary or dflash")
    path = Path(deployment_manifest).resolve()
    manifest = load_json_object(path)
    if (
        manifest.get("status") != "PASS"
        or manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle"
    ):
        raise ValueError("incremental runner requires a passing OM bundle")
    graphs = manifest.get("graphs", [])
    c = validate_incremental_bundle(graphs)
    if c is None:
        raise ValueError("deployment is not an incremental chunk bundle")
    if mode != "dflash" and not any(g["name"] == "target_decode" for g in graphs):
        raise ValueError("ordinary/paired mode requires an exported target_decode OM")
    record = manifest["air_manifest"]
    air = contained_path(path.parent, record["path"])
    if sha256_file(air) != record["sha256"]:
        raise ValueError("AIR manifest hash differs")
    if validate_incremental_bundle(load_json_object(air)["graphs"]) != c:
        raise ValueError("deployment contract differs from AIR")
    output = require_run_output(output)
    if output.exists():
        raise FileExistsError(output)
    lines = [ABI, f"capacity {c['capacity']} {c['vocab_size']}"]
    for name in ROLES:
        if name == "target_decode" and mode == "dflash":
            continue
        graph = next(g for g in graphs if g["name"] == name)
        om = contained_path(path.parent, graph["om"]["path"])
        if (
            om.stat().st_size != graph["om"]["bytes"]
            or sha256_file(om) != graph["om"]["sha256"]
        ):
            raise ValueError(f"OM integrity check failed: {name}")
        if any(ch in str(om) for ch in "\r\n\t"):
            raise ValueError("OM path contains a control character")
        lines.append(
            f"graph {name} {json.dumps(str(om), ensure_ascii=False)} {graph['om']['sha256']}"
        )
        for direction, marker in (("inputs", "I"), ("outputs", "O")):
            for tensor in graph["metadata"]["tensor_abi"][direction]:
                lines.append(
                    " ".join(
                        (
                            marker,
                            tensor["name"],
                            tensor["dtype"],
                            str(len(tensor["shape"])),
                            *(str(d) for d in tensor["shape"]),
                        )
                    )
                )
        lines.append("end")
    lines.append("done")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output, manifest, c
