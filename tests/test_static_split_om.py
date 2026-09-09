from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen35_dflash.ascend310p import cpp_runtime, quant_factory, workflow
from qwen35_dflash.ascend310p.incremental_graphs import (
    TargetPrefillHeadGraph, TargetPrefillStateGraph, incremental_state_graph_specs,
)
from qwen35_dflash.ascend310p.static_shape import validated_static_feature_shape
from test_incremental_om_graphs import _FakeTarget, _FakeDraft
from test_static_fused_om import _MaskReadingLayer, _compile_static_fixture


def test_static_split_export_accepts_real_repository_source_lock():
    # Exercise the production preflight without mocking it or loading weights.
    # Documentation-only commits can also invalidate a locked *_file payload.
    identity = quant_factory._verify_quant_source_lock()
    lock = Path(__file__).resolve().parents[1] / "SOURCE_LOCK.json"
    assert Path(identity["path"]) == lock
    assert identity["sha256"] == quant_factory._sha256(lock)
    assert identity["verified_file_count"] >= 10


def test_static_split_export_rejects_changed_locked_document(monkeypatch):
    document = Path(__file__).resolve().parents[1] / "docs/DFLASH_RUN_AND_VALIDATE.md"
    original_sha256 = quant_factory._sha256

    def changed_document_sha256(path):
        return "0" * 64 if path == document else original_sha256(path)

    monkeypatch.setattr(quant_factory, "_sha256", changed_document_sha256)
    with pytest.raises(
        ValueError,
        match=r"quant source differs from SOURCE_LOCK: docs/DFLASH_RUN_AND_VALIDATE\.md",
    ):
        quant_factory._verify_quant_source_lock()


def _specs(rows=64, capacity=256):
    draft = _FakeDraft().eval()
    draft.layers = nn.ModuleList([_MaskReadingLayer(0), _MaskReadingLayer(1)])
    return incremental_state_graph_specs(
        _FakeTarget().eval(), draft, kv_cache_max_len=capacity,
        device="cpu", dtype=torch.float16, eos_table_width=4,
        ordinary_custom_ops=(), head_custom_ops=(), verify_custom_ops=(),
        merged_prefill=True, draft_static_feature_rows=rows,
    )


@pytest.mark.parametrize("length", [1, 17, 64])
@pytest.mark.parametrize("eos_count", [0, 1])
def test_merged_prefill_equals_body_plus_head(length, eos_count):
    prefill = _specs()[0]
    args = list(prefill.example_args)
    args[0] = torch.arange(64).reshape(1, 64)
    args[1] = torch.tensor([length], dtype=torch.int16)
    args[6] = torch.tensor([64], dtype=torch.int64)
    args[7] = torch.tensor([length, 0, 0, 0])
    args[8] = torch.tensor([eos_count], dtype=torch.int32)
    actual = prefill.model(*args)
    target = _FakeTarget().eval()
    body = TargetPrefillStateGraph(target, kv_cache_max_len=256)(*args[:7])
    expected = (*body, *TargetPrefillHeadGraph(target)(body[0], *args[7:]))
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=0, atol=0)
    assert actual[8][0, 0].item() == length
    assert actual[9].item() == 1
    assert actual[10].item() == bool(eos_count)
    assert actual[7].item() == 64 + length


@pytest.mark.parametrize("rows", [64, 128])
def test_four_static_graphs_export_no_symbols_or_head_artifact(rows):
    specs = _specs(rows)
    assert [s.role for s in specs] == list(cpp_runtime._MERGED_STATIC_GRAPH_ABI)
    for spec in specs:
        inputs, outputs = cpp_runtime._MERGED_STATIC_GRAPH_ABI[spec.role]
        assert list(spec.input_names) == inputs
        assert list(spec.output_names) == outputs
        assert not spec.dynamic and not spec.input_dim_gears
        assert spec.metadata["physical_topology"] == "merged-prefill-four-static-split-v1"
        captured = torch.export.export(spec.model, spec.example_args, strict=True)
        assert not captured.range_constraints
        assert all(not isinstance(n.meta.get("val"), torch.SymInt)
                   for n in captured.graph.nodes if n.op == "placeholder")
        for a, e in zip(captured.module()(*spec.example_args), spec.model(*spec.example_args)):
            torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("count", [1, 2, 16, 17, 63, 64])
@pytest.mark.parametrize("cursor", [0, 48, 192])
def test_static_draft_preserves_exact_valid_prefix_and_visible_kv(count, cursor):
    draft = _specs()[2]
    args = list(draft.example_args)
    args[0] = torch.arange(256, dtype=torch.float16).reshape(1, 64, 4) / 128
    args[0][:, count:] = 0
    args[1] = torch.tensor([count], dtype=torch.int32)
    args[5] = torch.full_like(args[5], 0.25)
    args[6] = torch.full_like(args[6], 0.5)
    args[7] = torch.tensor([cursor], dtype=torch.int64)
    actual = draft.model(*args)
    args[0] = args[0][:, :count].contiguous()
    expected = draft.model(*args)
    for i, (a, e) in enumerate(zip(actual, expected)):
        if i in (1, 2):
            a, e = a[..., :cursor + count, :], e[..., :cursor + count, :]
        torch.testing.assert_close(a, e, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [0, True, -64, 16, 65, 512, "64"])
def test_static_split_rejects_invalid_carriers(rows):
    with pytest.raises((TypeError, ValueError)):
        _specs(rows)


def test_compile_and_resolve_four_static_split_graphs(tmp_path, monkeypatch):
    deployed, commands = _compile_static_fixture(tmp_path, monkeypatch, _specs())
    assert len(commands) == 4
    assert deployed["compiler"]["precision_policy"] == "preserve_graph_dtypes"
    assert all(c.count("--precision_mode=must_keep_origin_dtype") == 1 for c in commands)
    assert [g["atc_command"] for g in deployed["graphs"]] == commands
    assert all(not any(a.startswith(("--dynamic", "--input_shape")) for a in c) for c in commands)
    resolved, _ = cpp_runtime._resolve_incremental_oms(deployed["manifest_path"])
    assert list(resolved) == list(cpp_runtime._MERGED_STATIC_GRAPH_ABI)
    assert resolved["draft-propose"][1]["draft_static_shape"]["feature_rows"] == 64
    assert "fused_static_shape" not in resolved["draft-propose"][1]


@pytest.mark.parametrize("damage", ["missing", "dynamic", "gears", "head-abi", "extra-om"])
def test_split_manifest_rejects_stale_or_mixed_artifacts(tmp_path, monkeypatch, damage):
    deployed, _ = _compile_static_fixture(tmp_path, monkeypatch, _specs())
    manifest = Path(deployed["manifest_path"])
    value = json.loads(manifest.read_text())
    if damage == "missing":
        del value["graphs"][2]["draft_static_shape"]
    if damage == "dynamic":
        value["graphs"][0]["dynamic"] = True
    if damage == "gears":
        value["graphs"][2]["input_dim_gears"] = {"0": {"1": [16, 64]}}
    if damage == "head-abi":
        value["graphs"][0]["input_names"] = cpp_runtime._INCREMENTAL_GRAPH_ABI["target-prefill"][0]
    if damage == "extra-om":
        value["graphs"].append(copy.deepcopy(value["graphs"][1]))
    manifest.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cpp_runtime._resolve_incremental_oms(manifest)


@pytest.mark.parametrize("damage", [None, "symbol", "bindings", "capacity", "wrong-key"])
def test_static_draft_air_audit_checks_all_eight_tensor_bindings(damage):
    spec = _specs()[2]
    graph = {
        "role": spec.role, "dynamic": False, "input_dim_gears": {},
        "metadata": copy.deepcopy(spec.metadata),
        "runtime_input_abi": {"status": "PASS", "bindings": [
            {"index": i, "dtype": str(a.dtype).removeprefix("torch."),
             "example_shape": list(a.shape), "serialized_shape": list(a.shape)}
            for i, a in enumerate(spec.example_args)
        ]},
    }
    if damage == "symbol":
        graph["runtime_input_abi"]["bindings"][0]["serialized_shape"][1] = -1
    if damage == "bindings":
        graph["runtime_input_abi"]["bindings"].pop()
    if damage == "capacity":
        graph["metadata"]["draft_static_shape"]["kv_capacity"] = 128
    if damage == "wrong-key":
        graph["metadata"]["fused_static_shape"] = graph["metadata"].pop("draft_static_shape")
    if damage is None:
        assert validated_static_feature_shape(graph, air=True)["feature_rows"] == 64
    else:
        with pytest.raises(ValueError):
            validated_static_feature_shape(graph, air=True)


@pytest.mark.parametrize("policy", [None, "all-resident", "phase-resident"])
def test_split_launch_selects_group_residency_and_static_shape(tmp_path, monkeypatch, policy):
    deployed, _ = _compile_static_fixture(tmp_path, monkeypatch, _specs())
    commands = []
    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", lambda *a, **k: Path(sys.executable))
    def execute(command, **kwargs):
        commands.append(command)
        return subprocess.CompletedProcess(command, 96)
    options = dict(device_model="Ascend310P3", cann="test", driver="test",
                   firmware="test", runtime="test", state_policy=cpp_runtime.INCREMENTAL_STATE_POLICY)
    if policy is not None:
        options["model_residency_policy"] = policy
    with pytest.raises(RuntimeError, match="exit 96"):
        cpp_runtime.run_cpp_pair(
            deployment_manifest=deployed["manifest_path"], runner=sys.executable,
            runner_options=options, prompt_token_ids=[10] * 17, eos_token_ids=[],
            device_id=0, max_new_tokens=32, max_draft_tokens=15,
            raw_output=tmp_path / "raw.json", log_output=tmp_path / "log.txt",
            execute=execute, progress=False,
        )
    command, = commands
    assert "--target-prefill-head" not in command and "--fused-speculative-step" not in command
    assert command[command.index("--draft-static-feature-rows") + 1] == "64"
    assert command[command.index("--model-residency-policy") + 1] == (policy or "phase-resident")


@pytest.mark.parametrize("config,merged,rows", [
    ({}, True, 64),
    ({"draft_static_feature_rows": 128}, True, 128),
    ({"merged_prefill": False}, False, 0),
    ({"fused_speculative_step": True, "fused_static_feature_rows": 64}, False, 0),
    ({"unified_target_step": True}, False, 0),
])
def test_public_factory_defaults_to_static_split_and_keeps_opt_in_rollback(monkeypatch, config, merged, rows):
    assert workflow.DEFAULT_CPP_GRAPH_FACTORY == workflow.QUANT_INCREMENTAL_GRAPH_FACTORY
    monkeypatch.setattr(quant_factory.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(torch.ops.npu, "npu_gated_delta_rule_mtp",
                        SimpleNamespace(default=SimpleNamespace(_schema="fixture")), raising=False)
    monkeypatch.setattr(torch.ops.npu, "npu_chunk_gated_delta_rule",
                        SimpleNamespace(default=SimpleNamespace(_schema="fixture")), raising=False)
    identity = {"locked_inputs": {"group_sha256": {
        name: "fixture" for name in ("target_checkpoint", "draft_checkpoint", "quant_linear_weights", "quant_embedding")},
        "manifest_sha256": "fixture"}, "quant_source_lock": {}, "target_dir": "target",
        "draft_dir": "draft", "receiver_models_dir": "models", "quant_config": {},
        "quant_matmul_export_qlinear_count": 1}
    monkeypatch.setattr(quant_factory, "_load_quant_components", lambda *a, **k: (None, None, identity))
    monkeypatch.setattr(quant_factory, "_target_custom_op_exports", lambda *a, **k: tuple(
        SimpleNamespace(torch_op=op) for op in (
            quant_factory.NPU_DYNAMIC_QUANT_TORCH_OP, quant_factory.FUNCTIONAL_NPU_QUANT_MATMUL_TORCH_OP)))
    monkeypatch.setattr(quant_factory, "incremental_state_graph_specs", lambda *a, **k: k)
    selected = quant_factory.create_quant_incremental_state_graphs({"max_sequence_length": 256, **config})
    assert selected["merged_prefill"] is merged
    assert selected["draft_static_feature_rows"] == rows
    assert selected["target_verify_gdr_policy"] == "mtp-block-v1"
