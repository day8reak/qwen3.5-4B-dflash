"""Exact CPU and serialized-IR regressions for the receiver BroadcastTo incident.

These are not Ascend/ATC tests. In particular duplicate clamped scatter indices
retain the old operation and values; CPU equality does not establish NPU order.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework" / "python"))

from qwen35_dflash.ascend310p.draft_cache_export import (
    DRAFT_CACHE_INDEX_POLICY,
    audit_draft_cache_index_export,
    validated_draft_cache_index_audit,
)
from qwen35_dflash.ascend310p.incremental_graphs import _FixedDraftCache


GEARS = (*range(1, 17), *range(64, 2049, 64))
METADATA = {
    "draft_cache_index_policy": DRAFT_CACHE_INDEX_POLICY,
    "draft_cache_index_layers": 6,
}


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


class _Update(nn.Module):
    def forward(self, key, value, cursor, context_key, context_value, block_key, block_value):
        cache = _FixedDraftCache(
            key, value, cursor, context_capacity=context_key.shape[-2], block_rows=16,
        )
        joined = []
        for layer in range(key.shape[0]):
            joined.extend(cache.update(
                layer,
                torch.cat((context_key[layer], block_key[layer]), dim=2),
                torch.cat((context_value[layer], block_value[layer]), dim=2),
            ))
        return (*cache.grouped(), *joined)


def _inputs(n, cursor=0, layers=1):
    generator = torch.Generator().manual_seed(101 + n)
    def rand(rows):
        return torch.randn(layers, 1, 8, rows, 128, generator=generator, dtype=torch.float16)
    return (rand(2048), rand(2048), torch.tensor([cursor], dtype=torch.int64),
            rand(n), rand(n), rand(16), rand(16))


def _old_expand_oracle(args):
    key, value, cursor, context_key, context_value, block_key, block_value = args
    n = context_key.shape[-2]
    positions = (cursor.reshape(-1, 1) + torch.arange(n).reshape(1, -1)).clamp(max=2047)
    # Intentionally retain the pre-fix expand+scatter, including padded writes
    # and repeated index 2047. This is not an index_copy reference.
    index = positions[:, None, :, None].expand_as(context_key[0])
    next_keys, next_values, joined = [], [], []
    for layer in range(key.shape[0]):
        next_key = key[layer].scatter(2, index, context_key[layer])
        next_value = value[layer].scatter(2, index, context_value[layer])
        next_keys.append(next_key)
        next_values.append(next_value)
        joined.extend((torch.cat((next_key, block_key[layer]), dim=2),
                       torch.cat((next_value, block_value[layer]), dim=2)))
    return (torch.stack(next_keys), torch.stack(next_values), *joined)


@pytest.mark.parametrize("n", [1, 2, 15, 16, 64, 128, 2048])
@pytest.mark.parametrize("cursor", [0, 7, 2047])
def test_real_cache_update_matches_old_indices_and_all_outputs(n, cursor):
    args = _inputs(n, cursor)
    snapshots = tuple(value.clone() for value in args)
    positions = (args[2].reshape(-1, 1) + torch.arange(n).reshape(1, -1)).clamp(max=2047)
    assert torch.equal(positions[:, None, :, None].repeat(1, 8, 1, 128),
                       positions[:, None, :, None].expand_as(args[3][0]))
    for actual, expected in zip(_Update()(*args), _old_expand_oracle(args)):
        assert torch.equal(actual, expected)
    assert all(torch.equal(value, old) for value, old in zip(args, snapshots))


def _export(layers):
    n = torch.export.Dim("feature_rows", min=1, max=2048)
    return torch.export.export(
        _Update(), _inputs(64, layers=layers), strict=True,
        dynamic_shapes=({}, {}, {}, {3: n}, {3: n}, {}, {}),
    )


@pytest.fixture(scope="module")
def dynamic_update():
    return _export(1).module()


@pytest.mark.parametrize("n", GEARS)
def test_exported_update_accepts_every_advertised_physical_width(dynamic_update, n):
    # Includes N=1 decode tails, T16 tails, first-prefill N64, and N2048.
    args = _inputs(n, cursor=7)
    for actual, expected in zip(dynamic_update(*args), _old_expand_oracle(args)):
        assert torch.equal(actual, expected)


def test_all_six_layers_lower_indices_to_static_repeats():
    exported = _export(6)
    scatters = [node for node in exported.graph.nodes
                if node.target == torch.ops.aten.scatter.src]
    assert len(scatters) == 12
    for scatter in scatters:
        assert scatter.args[1] == 2
        index = scatter.args[2]
        assert index.target == torch.ops.aten.repeat.default
        assert list(index.args[1]) == [1, 8, 1, 128]
        # K/V of a layer must consume the same index, not independent policies.
    assert all(scatters[i].args[2] is scatters[i + 1].args[2]
               for i in range(0, 12, 2))
    args = _inputs(16, cursor=2047, layers=6)
    for actual, expected in zip(exported.module()(*args), _old_expand_oracle(args)):
        assert torch.equal(actual, expected)


def _pbtxt(dialect="node", *, index_type="Tile", repeats_type="Const", count=12):
    kind = "op" if dialect == "node" else "type"
    def node(name, op, inputs=(), extra=""):
        fields = [f'name: {json.dumps(name)}', f'{kind}: {json.dumps(op)}']
        fields += [f'input: {json.dumps(edge)}' for edge in inputs]
        return f'{dialect} {{ ' + " ".join(fields) + extra + " }\n"
    # CSE can share one index across all six layers; generated names are opaque.
    return (
        node("static_multiples", repeats_type)
        + node("renamed_index_42", index_type, ("positions:0", "static_multiples:0"))
        + node("unrelated_mask_broadcast", "BroadcastTo", ("mask:0", "shape:0"))
        + "".join(node(f"write_{i}", "ScatterElements",
                       (f"cache_{i}:0", "renamed_index_42:0", f"updates_{i}:0"),
                       ' attr { key: "descriptor" value { s: "op { name: \\\"fake\\\" }" } }')
                  for i in range(count))
    )


def _audit(tmp_path, text=None, metadata=None):
    path = tmp_path / "air" / "draft" / "dynamo.pbtxt"
    path.parent.mkdir(parents=True, exist_ok=True)
    if text is not None:
        path.write_text(text, encoding="utf-8")
    return audit_draft_cache_index_export(
        metadata or METADATA, path.parent, relative_to=tmp_path,
    )


@pytest.mark.parametrize("dialect", ["node", "op"])
def test_ge_audit_follows_index_edges_not_names_or_global_broadcast_count(tmp_path, dialect):
    record = _audit(tmp_path, _pbtxt(dialect))
    assert record["status"] == "PASS"
    assert record["files"][0]["scatter_count"] == 12
    assert {binding["index"] for binding in record["files"][0]["bindings"]} == {"renamed_index_42"}
    assert validated_draft_cache_index_audit(
        {"metadata": METADATA, "draft_cache_index_audit": record},
    ) == record


@pytest.mark.parametrize("text, error", [
    (_pbtxt(index_type="BroadcastTo"), "expected Tile"),
    (_pbtxt(repeats_type="Pack"), "repeats must be Const"),
    (_pbtxt().replace('"renamed_index_42:0"', '"renamed_index_42:1"'), "expected Tile"),
    (_pbtxt().replace('"static_multiples:0"', '"static_multiples:1"'), "repeats must be Const"),
    (_pbtxt().replace('"static_multiples:0"', '"missing:0"'), "repeats must be Const"),
    (_pbtxt(count=10), "expected 12 ScatterElements"),
    (_pbtxt(count=14), "expected 12 ScatterElements"),
    (_pbtxt().replace('"write_11"', '"write_10"'), "duplicate node"),
    (_pbtxt() + 'node { name: "incomplete" op: "Tile"', "cannot parse"),
    (None, "requires dynamo.pbtxt"),
])
def test_ge_audit_rejects_related_lowering_regressions(tmp_path, text, error):
    with pytest.raises(ValueError, match=error):
        _audit(tmp_path, text)


@pytest.mark.parametrize("field, bad", [
    ("status", "FAIL"), ("layers", 5), ("policy", "broadcast"),
    ("index_ge_op", "BroadcastTo"), ("repeats_ge_op", "Pack"), ("files", []),
])
def test_compile_contract_rejects_missing_or_failed_evidence(tmp_path, field, bad):
    record = _audit(tmp_path, _pbtxt())
    record[field] = bad
    with pytest.raises(ValueError, match="Draft cache index"):
        validated_draft_cache_index_audit({"metadata": METADATA, "draft_cache_index_audit": record})


def test_compile_contract_rejects_duplicate_bindings_and_requires_declared_audit(tmp_path):
    record = _audit(tmp_path, _pbtxt())
    record["files"][0]["bindings"][-1] = copy.deepcopy(record["files"][0]["bindings"][0])
    with pytest.raises(ValueError, match="duplicate scatter"):
        validated_draft_cache_index_audit({"metadata": METADATA, "draft_cache_index_audit": record})
    with pytest.raises(ValueError, match="re-export AIR"):
        validated_draft_cache_index_audit({"metadata": METADATA})
    # Historical manifests do not claim this fix and remain usable for rollback.
    assert validated_draft_cache_index_audit({}) is None
    assert audit_draft_cache_index_export({}, tmp_path, relative_to=tmp_path) is None


@pytest.mark.parametrize("layers", [0, -1, True, "6"])
def test_invalid_layer_contract_fails_closed(tmp_path, layers):
    with pytest.raises(ValueError, match="policy/layer contract"):
        _audit(tmp_path, _pbtxt(), {**METADATA, "draft_cache_index_layers": layers})
