"""CPU/FX and serialized-IR regression gates; not CANN/OM/NPU validation."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import struct
import subprocess
import sys

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from qwen35_dflash.ascend310p.draft_kv_repeat_export import (
    DRAFT_KV_REPEAT_POLICY,
    audit_draft_kv_repeat_export,
    validated_draft_kv_repeat_audit,
)
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps, _repeat_kv

METADATA = {"draft_kv_repeat_policy": DRAFT_KV_REPEAT_POLICY,
            "draft_kv_repeat_layers": 6, "draft_kv_repeat_groups": 4}


@pytest.fixture(scope="module", autouse=True)
def _bounded_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _old_repeat(states, groups):
    if groups == 1:
        return states
    b, h, s, d = states.shape
    return states[:, :, None, :, :].expand(b, h, groups, s, d).reshape(b, h * groups, s, d)


@pytest.mark.parametrize("shape", [(1, 8, 2064, 128), (2, 3, 17, 8), (1, 8, 1, 128)])
@pytest.mark.parametrize("groups", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "transposed"])
def test_replication_preserves_exact_values_and_inputs(shape, groups, dtype, layout):
    b, h, s, d = shape
    generator = torch.Generator().manual_seed(39)
    if layout == "transposed":
        states = torch.randn(b, s, h, d, dtype=dtype, generator=generator).transpose(1, 2)
    else:
        states = torch.randn(b, h, s, d * (2 if layout == "strided" else 1),
                             dtype=dtype, generator=generator)
        if layout == "strided":
            states = states[..., ::2]
    before = states.clone()
    result = _repeat_kv(states, groups)
    expected = _old_repeat(states, groups)
    assert torch.equal(result, expected)
    assert torch.equal(states, before)
    if groups == 1:
        assert result is states


def test_adjacent_head_order_and_special_float_bits():
    states = torch.arange(8, dtype=torch.float16).reshape(1, 8, 1, 1)
    result = _repeat_kv(states, 4)
    assert result.flatten().tolist() == [float(h) for h in range(8) for _ in range(4)]
    assert not torch.equal(result, states.repeat(1, 4, 1, 1))
    for dtype in (torch.float16, torch.float32):
        special = torch.tensor([0.0, -0.0, float("inf"), float("-inf"), float("nan")],
                               dtype=dtype).reshape(1, 1, 5, 1)
        assert torch.equal(_repeat_kv(special, 4).contiguous().view(torch.uint8),
                           _old_repeat(special, 4).contiguous().view(torch.uint8))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("groups", [1, 2, 4])
def test_attention_matches_prefixed_masked_golden(dtype, groups):
    generator = torch.Generator().manual_seed(123)
    q = torch.randn(1, 8 * groups, 16, 128, generator=generator, dtype=dtype)
    # Same cache+block geometry as the receiver; mask padding and causal block.
    k = torch.randn(1, 8, 2064, 128, generator=generator, dtype=dtype)
    v = torch.randn(1, 8, 2064, 128, generator=generator, dtype=dtype)
    visible = torch.zeros(1, 1, 16, 2064, dtype=torch.bool)
    visible[..., :17] = True
    visible[..., 2048:] = torch.ones(16, 16, dtype=torch.bool).tril()
    scale = 128 ** -0.5
    scores = torch.matmul(q.float(), _old_repeat(k, groups).float().transpose(-2, -1))
    scores = (scores * scale).masked_fill(~visible, float("-inf"))
    expected = torch.matmul(torch.softmax(scores, dim=-1, dtype=torch.float32),
                            _old_repeat(v, groups).float()).to(dtype)
    actual = AirDFlashOps().attention(q, k, v, visible, scale, groups)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


class _SixLayerHeads(nn.Module):
    def __init__(self, groups):
        super().__init__()
        self.groups = groups

    def forward(self, key, value):
        return tuple(_repeat_kv(states[layer], self.groups)
                     for layer in range(6) for states in (key, value))


@pytest.mark.parametrize("groups", [1, 4])
def test_static_export_covers_all_six_layers_key_and_value(groups):
    args = (torch.randn(6, 1, 8, 80, 128, dtype=torch.float16),
            torch.randn(6, 1, 8, 80, 128, dtype=torch.float16))
    exported = torch.export.export(_SixLayerHeads(groups), args, strict=True)
    repeats = [n for n in exported.graph.nodes if n.target == torch.ops.aten.repeat.default]
    assert len(repeats) == (12 if groups > 1 else 0)
    assert not any(n.target == torch.ops.aten.expand.default for n in exported.graph.nodes)
    for repeat in repeats:
        assert list(repeat.args[1]) == [1, 1, groups, 1, 1]
        assert repeat.args[0].target == torch.ops.aten.unsqueeze.default
        assert repeat.args[0].args[1] == 2
    expected = tuple(_old_repeat(states[layer], groups)
                     for layer in range(6) for states in args)
    for actual, reference in zip(exported.module()(*args), expected):
        assert torch.equal(actual, reference)


def _pbtxt(dialect="node", *, serialized=True, groups=4, count=12, kind="Tile",
           repeats=None, constant_kind="Const", axis=2, identity=False,
           dtype="DT_INT64", bad_bytes=False):
    """Keep both GE dump dialects and the receiver's encoded TensorDef format."""
    field = "op" if dialect == "node" else "type"

    def node(name, op, inputs=(), attrs=""):
        return (f'{dialect} {{ name: "{name}" {field}: "{op}" '
                + " ".join(f'input: "{edge}"' for edge in inputs) + " " + attrs + " }\n")

    def attr(name, value):
        encoded = "s: " + (repr(value) if dialect == "node" else json.dumps(value)) if serialized else value
        return f'attr {{ key: "{name}" value {{ {encoded} }} }}'

    repeats = repeats if repeats is not None else [1, 1, groups, 1, 1]
    data = struct.pack("<" + ("i" if dtype == "DT_INT32" else "q") * len(repeats), *repeats)
    if bad_bytes:
        data = data[:-1]
    octal = "".join("\\" + format(byte, "03o") for byte in data)
    tensor = (f't {{ desc {{ dtype: {dtype} shape {{ dim: {len(repeats)} }} layout: "ND" }} '
              f'data: "{octal}" }}')
    text = node("shared_repeats", constant_kind, attrs=(
        attr("value", tensor) + attr("_readable_value", 's: "[1,1,4,1,1]"')
    ))
    # Matching readable text cannot cover for invalid binary constants.
    text += node("unrelated", "BroadcastTo", ("mask:0", "mask_shape:0"))
    for index in range(12):
        text += node(f"write_{index}", "ScatterElements",
                     (f"cache_{index}:0", "cache_index:0", f"update_{index}:0"))
        text += node(f"joined_{index}", "ConcatV2", (f"write_{index}:0", f"block_{index}:0"))
        text += node(f"group_{index}", "Unsqueeze", (f"joined_{index}:0",),
                     attr("axes", f"list {{ i: {axis} }}"))
        if index < count and groups > 1:
            # Generated names intentionally do not encode layer or role.
            text += node(f"replica_{index}", kind, (f"group_{index}:0", "shared_repeats:0"))
            parent = f"replica_{index}"
            if identity:
                text += node(f"copy_{index}", "Identity", (parent + ":0",))
                parent = f"copy_{index}"
            text += node(f"flat_{index}", "Reshape", (parent + ":0", "reshape_shape:0"))
    return text


def _audit(tmp_path, text, metadata=None):
    directory = tmp_path / "air/fused-speculative-step"
    directory.mkdir(parents=True, exist_ok=True)
    if text is not None:
        (directory / "dynamo.pbtxt").write_text(text, encoding="utf-8")
    return audit_draft_kv_repeat_export(metadata or METADATA, directory, relative_to=tmp_path)


@pytest.mark.parametrize("dialect, serialized", [("node", True), ("op", True), ("op", False)])
@pytest.mark.parametrize("identity", [False, True])
@pytest.mark.parametrize("dtype", ["DT_INT64", "DT_INT32"])
def test_ir_audit_follows_each_cache_head_and_decodes_const_data(tmp_path, dialect, serialized, identity, dtype):
    record = _audit(tmp_path, _pbtxt(dialect, serialized=serialized, identity=identity, dtype=dtype))
    assert len(record["files"][0]["bindings"]) == 12
    assert {b["cache_write"] for b in record["files"][0]["bindings"]} == {f"write_{i}" for i in range(12)}
    assert validated_draft_kv_repeat_audit(
        {"metadata": METADATA, "draft_kv_repeat_audit": record}) == record


@pytest.mark.parametrize("text, error", [
    (_pbtxt(kind="BroadcastTo"), "must use GQA Tile"),
    (_pbtxt(constant_kind="Pack"), "Const repeats"),
    (_pbtxt(count=11), "not every cache K/V"),
    (_pbtxt(repeats=[1, 4, 1, 1, 1]), "exact Const"),
    (_pbtxt(repeats=[1, 1, 3, 1, 1]), "exact Const"),
    (_pbtxt(axis=1), "group axis 2"),
    (_pbtxt(dtype="DT_FLOAT"), "INT32/INT64"),
    (_pbtxt(bad_bytes=True), "Const repeats bytes"),
    (_pbtxt().replace('"shared_repeats:0"', '"shared_repeats:1"'), "exact Const"),
    (_pbtxt().replace('"write_11:0"', '"write_10:0"'), "not every cache K/V"),
    (_pbtxt().replace('"write_11"', '"write_10"'), "duplicate node"),
    (_pbtxt().replace('op: "Reshape"', 'op: "Identity"'), "Reshape consumer"),
    (_pbtxt().replace('"joined_11:0"', '"missing:0"'), "not every cache K/V"),
    (_pbtxt() + 'node { name: "dangling" op: "Tile"', "cannot parse"),
    (None, "requires dynamo.pbtxt"),
])
def test_ir_regressions_fail_closed(tmp_path, text, error):
    with pytest.raises(ValueError, match=error):
        _audit(tmp_path, text)


def test_single_group_and_legacy_contracts(tmp_path):
    metadata = {**METADATA, "draft_kv_repeat_groups": 1}
    record = _audit(tmp_path, _pbtxt(groups=1), metadata)
    assert record["files"][0]["bindings"] == []
    assert validated_draft_kv_repeat_audit(
        {"metadata": metadata, "draft_kv_repeat_audit": record}) == record
    assert validated_draft_kv_repeat_audit({}) is None
    assert audit_draft_kv_repeat_export({}, tmp_path, relative_to=tmp_path) is None


@pytest.mark.parametrize("field, value", [
    ("status", "FAIL"), ("policy", "expand"), ("layers", True), ("groups", 3),
    ("repeats", [True, 1, 4, 1, 1]), ("files", []),
])
def test_compile_gate_rejects_bad_evidence(tmp_path, field, value):
    record = _audit(tmp_path, _pbtxt())
    record[field] = value
    with pytest.raises(ValueError, match="Draft KV repeat"):
        validated_draft_kv_repeat_audit({"metadata": METADATA, "draft_kv_repeat_audit": record})


def test_compile_gate_rejects_missing_duplicate_or_unbound_evidence(tmp_path):
    record = _audit(tmp_path, _pbtxt())
    with pytest.raises(ValueError, match="re-export AIR"):
        validated_draft_kv_repeat_audit({"metadata": METADATA})
    with pytest.raises(ValueError, match="re-export AIR"):
        validated_draft_kv_repeat_audit({"draft_kv_repeat_audit": record})
    duplicate = copy.deepcopy(record)
    duplicate["files"][0]["bindings"][-1] = copy.deepcopy(duplicate["files"][0]["bindings"][0])
    with pytest.raises(ValueError, match="duplicate bindings"):
        validated_draft_kv_repeat_audit({"metadata": METADATA, "draft_kv_repeat_audit": duplicate})
    record["files"][0]["path"] = "../another-run/dynamo.pbtxt"
    with pytest.raises(ValueError, match="evidence path"):
        validated_draft_kv_repeat_audit({"metadata": METADATA, "draft_kv_repeat_audit": record})


@pytest.mark.parametrize("field, value", [
    ("draft_kv_repeat_policy", "expand"), ("draft_kv_repeat_layers", 0),
    ("draft_kv_repeat_groups", True), ("draft_kv_repeat_groups", -1),
])
def test_bad_metadata_is_not_a_legacy_bundle(tmp_path, field, value):
    with pytest.raises(ValueError, match="policy/layer/group contract"):
        _audit(tmp_path, _pbtxt(), {**METADATA, field: value})


@pytest.mark.parametrize("kind", ["Tile", "BroadcastTo"])
def test_export_compile_propagates_gate_before_atc(tmp_path, monkeypatch, kind):
    from test_quant_air_om_framework import _FakeTorchAir, _ensure_adn_rms_norm_test_schema
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec, CustomOpExportSpec
    from qwen35_dflash.ascend310p.custom_op_export import ADN_RMS_NORM_TORCH_OP
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle

    _ensure_adn_rms_norm_test_schema()
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    monkeypatch.delenv("ASCEND_CUSTOM_OPP_PATH", raising=False)

    class FakeKVHeadsTorchAir(_FakeTorchAir):
        def dynamo_export(self, *args, **kwargs):
            super().dynamo_export(*args, **kwargs)
            path = Path(kwargs["export_path"]) / "dynamo.pbtxt"
            path.write_text(path.read_text(encoding="utf-8") + _pbtxt(kind=kind), encoding="utf-8")

    def factory(config):
        return (AirGraphSpec(name="draft-propose", role="draft-propose",
                             model=nn.Identity(), example_args=(torch.ones(1),),
                             metadata=METADATA,
                             custom_ops=(CustomOpExportSpec(
                                 torch_op=ADN_RMS_NORM_TORCH_OP,
                                 ge_op_type="AdnRmsNorm",
                             ),)),)

    bundle = tmp_path / "bundle"
    if kind == "BroadcastTo":
        with pytest.raises(ValueError, match="must use GQA Tile"):
            export_air_bundle(factory, {}, bundle, torchair_module=FakeKVHeadsTorchAir())
        assert not (bundle / "air-manifest.json").exists()
        assert (bundle / "air/draft-propose/dynamo.pbtxt").is_file()
        return
    manifest = export_air_bundle(factory, {}, bundle, torchair_module=FakeKVHeadsTorchAir())
    path = bundle / "air-manifest.json"
    bad = copy.deepcopy(manifest)
    del bad["graphs"][0]["draft_kv_repeat_audit"]
    path.write_text(json.dumps(bad), encoding="utf-8")

    def unexpected(command, cwd):
        pytest.fail("ATC must not start with missing K/V head repeat evidence")

    with pytest.raises(ValueError, match="re-export AIR"):
        compile_air_bundle(path, soc_version="Ascend310P3", atc_bin=tmp_path / "absent-atc", runner=unexpected)
    assert not (bundle / "om").exists()
    path.write_text(json.dumps(manifest), encoding="utf-8")
    atc = tmp_path / "atc"
    atc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    atc.chmod(0o755)

    def runner(command, cwd):
        prefix = next(arg.split("=", 1)[1] for arg in command if arg.startswith("--output="))
        Path(prefix + "_linux_aarch64.om").write_bytes(b"explicit-fake-kv-head-tile-om")
        return subprocess.CompletedProcess(command, 0, stdout="explicit-test-double")

    deployment = compile_air_bundle(path, soc_version="Ascend310P3", atc_bin=atc,
                                    runner=runner, atc_identity="explicit-test-double")
    assert deployment["graphs"][0]["draft_kv_repeat_audit"] == manifest["graphs"][0]["draft_kv_repeat_audit"]
