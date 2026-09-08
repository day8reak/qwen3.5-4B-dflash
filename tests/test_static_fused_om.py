from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from qwen35_dflash.ascend310p.incremental_graphs import incremental_state_graph_specs
from qwen35_dflash.ascend310p.static_shape import (
    validated_fused_static_shape, validate_static_request,
)
from qwen35_dflash.ascend310p.cpp_runtime import (
    _resolve_incremental_oms, _FUSED_SPECULATIVE_STEP_GRAPH_ABI,
)
from qwen35_dflash.ascend310p import cpp_runtime
from qwen35_dflash.ascend310p.compiler import compile_air_bundle
from qwen35_dflash.ascend310p.utils import file_record
from test_incremental_om_graphs import _FakeTarget, _FakeDraft


class _MaskReadingLayer(nn.Module):
    """Unlike the basic wiring fixture, make Draft outputs depend on KV/masks."""

    def __init__(self, index):
        super().__init__()
        self.index = index
        self.self_attn = SimpleNamespace(is_causal=True, sliding_window=None)

    def forward_cached(self, hidden, projected, cosine, sine, cache, attention_mask):
        combined = torch.cat((projected, hidden), dim=1).unsqueeze(1)
        key, value = cache.update(self.index, combined, combined * 0.1)
        return hidden + torch.nn.functional.scaled_dot_product_attention(
            hidden.unsqueeze(1), key, value, attn_mask=attention_mask,
        ).squeeze(1)


def _specs(rows=64, capacity=256):
    draft = _FakeDraft().eval()
    draft.layers = nn.ModuleList([_MaskReadingLayer(0), _MaskReadingLayer(1)])
    return incremental_state_graph_specs(
        _FakeTarget().eval(), draft, kv_cache_max_len=capacity,
        device="cpu", dtype=torch.float16, eos_table_width=4,
        ordinary_custom_ops=(), head_custom_ops=(), verify_custom_ops=(),
        fused_speculative_step=True, fused_static_feature_rows=rows,
    )


@pytest.mark.parametrize("rows", [64, 128])
def test_static_four_graphs_have_fixed_carriers_and_live_counts(rows):
    specs = _specs(rows)
    assert len(specs) == 4
    assert all(s.dynamic is False and s.input_dim_gears == {} for s in specs)
    fused = specs[-1]
    assert tuple(fused.example_args[0].shape) == (1, rows, 4)
    assert fused.metadata["fused_static_shape"]["feature_rows"] == rows
    exported = torch.export.export(fused.model, fused.example_args, strict=True)
    assert not exported.range_constraints
    inputs = [n for n in exported.graph.nodes if n.op == "placeholder"]
    assert not any(isinstance(n.meta.get("val"), torch.SymInt) for n in inputs)
    # One static capture must still respond to changing runtime count values.
    for count in (1, 4, 16):
        args = list(fused.example_args)
        args[1] = torch.tensor([count], dtype=torch.int32)
        result = exported.module()(*args)
        assert result[15].item() == count
        assert result[7].shape[1] == 16


@pytest.mark.parametrize("count", [1, 2, 16, 17, 63, 64])
@pytest.mark.parametrize("cursor", [0, 48, 192])
def test_static_padding_matches_exact_prefix_and_visible_cache(count, cursor):
    fused = _specs()[-1]
    args = list(fused.example_args)
    args[0] = torch.arange(256, dtype=torch.float16).reshape(1, 64, 4) / 128
    args[0][:, count:] = 0
    args[1] = torch.tensor([count], dtype=torch.int32)
    args[14] = torch.tensor([cursor], dtype=torch.int64)
    args[12] = torch.full_like(args[12], 0.25)
    args[13] = torch.full_like(args[13], 0.5)
    static_result = fused.model(*args)
    args[0] = args[0][:, :count].contiguous()
    exact_result = fused.model(*args)
    for index, (actual, reference) in enumerate(zip(static_result, exact_result)):
        if index in (13, 14):
            # Uncommitted KV tail is intentionally not a semantic output.
            actual, reference = actual[..., :cursor + count, :], reference[..., :cursor + count, :]
        torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("rows", [True, -64, 1, 16, 65, 512, "64"])
def test_invalid_static_carriers_fail_before_export(rows):
    with pytest.raises((ValueError, TypeError), match="fused_static_feature_rows"):
        _specs(rows)


def _graph():
    spec = _specs()[-1]
    return {
        "role": spec.role, "dynamic": False, "input_dim_gears": {},
        "metadata": spec.metadata,
        "fused_static_shape": spec.metadata["fused_static_shape"],
        "runtime_input_abi": {
            "status": "PASS",
            "bindings": [
                {"index": i, "dtype": str(a.dtype).removeprefix("torch."),
                 "example_shape": list(a.shape), "serialized_shape": list(a.shape)}
                for i, a in enumerate(spec.example_args)
            ],
        },
    }


@pytest.mark.parametrize("damage", [None, "dynamic", "gears", "missing", "symbol",
                                    "width", "capacity", "padding", "bindings"])
def test_static_air_contract_requires_audited_fixed_serialized_inputs(damage):
    graph = _graph()
    if damage == "dynamic": graph["dynamic"] = True
    if damage == "gears": graph["input_dim_gears"] = {"0": {"1": [16, 64]}}
    if damage == "missing": del graph["metadata"]["fused_static_shape"]
    if damage == "symbol": graph["runtime_input_abi"]["bindings"][0]["serialized_shape"][1] = -1
    if damage == "width": graph["metadata"]["fused_static_shape"]["feature_width"] = 8
    if damage == "capacity": graph["metadata"]["fused_static_shape"]["kv_capacity"] = 128
    if damage == "padding": graph["metadata"]["fused_static_shape"]["padding_policy"] = "none"
    if damage == "bindings": graph["runtime_input_abi"]["bindings"].pop()
    if damage is None:
        assert validated_fused_static_shape(graph, air=True)["feature_rows"] == 64
    else:
        with pytest.raises(ValueError, match="static fused"):
            validated_fused_static_shape(graph, air=True)


def test_static_manifest_accepts_fixed_om_and_rejects_mixed_dynamic_bundle(tmp_path):
    graphs = []
    for role, (inputs, outputs) in _FUSED_SPECULATIVE_STEP_GRAPH_ABI.items():
        path = tmp_path / (role + ".om")
        path.write_bytes(role.encode())
        graphs.append({
            "name": role, "role": role, "input_names": inputs, "output_names": outputs,
            "dynamic": False, "input_dim_gears": {},
            "om": {"path": path.name, "sha256": hashlib.sha256(role.encode()).hexdigest()},
        })
    graphs[-1]["fused_static_shape"] = _graph()["fused_static_shape"]
    manifest = tmp_path / "deployment.json"
    payload = {"status": "PASS", "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
               "graphs": graphs}
    manifest.write_text(json.dumps(payload))
    assert len(_resolve_incremental_oms(manifest)[0]) == 4
    graphs[0]["dynamic"] = True
    manifest.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="four static OMs"):
        _resolve_incremental_oms(manifest)


def test_static_request_does_not_truncate_or_allow_clamped_padding_to_hit_live_kv():
    shape = _graph()["fused_static_shape"]
    validate_static_request(shape, 17, 32)
    validate_static_request(shape, 64, 128)
    with pytest.raises(ValueError, match="prompt tokens"):
        validate_static_request(shape, 65, 1)
    with pytest.raises(ValueError, match="free KV slots"):
        validate_static_request(shape, 64, 129)


def _compile_static_fixture(tmp_path, monkeypatch, specs=None):
    """Control-plane fixture only: no actual AIR graph or ATC/device execution."""
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    graphs = []
    for spec in _specs() if specs is None else specs:
        path = bundle / (spec.role + ".air")
        path.write_bytes(b"static control-plane fixture: " + spec.role.encode())
        graph = {
            "name": spec.role, "role": spec.role, "dynamic": False,
            "input_dim_gears": {}, "input_names": list(spec.input_names),
            "output_names": list(spec.output_names),
            "air": file_record(path, relative_to=bundle),
            "payload_files": [file_record(path, relative_to=bundle)],
            "runtime_input_abi": {
                "status": "PASS", "calls": 1,
                "policy": "public-tensor-storage-identity-v1",
                "python_float_policy": "dynamo-specialize-float",
                "logical_input_names": list(spec.input_names),
                "bindings": [
                    {"index": i, "logical_name": name, "data_node_name": f"arg{i}",
                     "dtype": str(arg.dtype).removeprefix("torch."),
                     "example_shape": list(arg.shape), "serialized_shape": list(arg.shape)}
                    for i, (name, arg) in enumerate(zip(spec.input_names, spec.example_args))
                ],
            },
        }
        if spec.role == "fused-speculative-step":
            graph["metadata"] = {"fused_static_shape": spec.metadata["fused_static_shape"]}
        if spec.role == "draft-propose":
            graph["metadata"] = {"draft_static_shape": spec.metadata["draft_static_shape"]}
        graphs.append(graph)
    air_manifest = bundle / "air-manifest.json"
    air_manifest.write_text(json.dumps({
        "schema_version": 4, "status": "PASS",
        "artifact_kind": "qwen35-dflash-torchair-bundle", "graphs": graphs,
    }))
    commands = []

    def fake_atc(command, cwd):
        commands.append(command)
        prefix = next(a.split("=", 1)[1] for a in command if a.startswith("--output="))
        Path(prefix + "_linux_aarch64.om").write_bytes(b"fake static OM")
        return subprocess.CompletedProcess(command, 0, stdout="fixture only")

    deployment = compile_air_bundle(
        air_manifest, soc_version="Ascend310P3", atc_bin=sys.executable,
        runner=fake_atc, atc_identity="CONTROL_PLANE_TEST_DOUBLE",
    )
    return deployment, commands


def test_static_contract_survives_compile_to_runtime_manifest(tmp_path, monkeypatch):
    deployment, commands = _compile_static_fixture(tmp_path, monkeypatch)
    assert len(commands) == 4
    assert all(not any(a.startswith(("--dynamic", "--input_shape")) for a in c)
               for c in commands)
    assert all(g["dynamic"] is False and g["input_dim_gears"] == {}
               for g in deployment["graphs"])
    resolved, _ = _resolve_incremental_oms(deployment["manifest_path"])
    assert len(resolved) == 4
    om, fused, _ = resolved["fused-speculative-step"]
    assert om.name.endswith("_linux_aarch64.om")
    assert fused["fused_static_shape"] == _graph()["fused_static_shape"]
    assert fused["runtime_input_abi"]["bindings"][0]["serialized_shape"] == [1, 64, 4]


@pytest.mark.parametrize("residency", ["all-resident", "phase-resident"])
@pytest.mark.parametrize("prompt_rows,max_tokens,error", [
    (17, 32, "exit 96"), (65, 1, "prompt tokens"), (64, 129, "free KV slots"),
])
def test_static_control_plane_forwards_explicit_carrier_and_rejects_before_launch(
    tmp_path, monkeypatch, capsys, prompt_rows, max_tokens, error, residency,
):
    deployment, _ = _compile_static_fixture(tmp_path, monkeypatch)
    commands = []

    def preflight(*args, **kwargs):
        assert prompt_rows == 17, "inadmissible request reached runner preflight"
        return Path(sys.executable)

    def execute(command, **kwargs):
        commands.append(command)
        # Stop at the handoff: this fixture does not produce an inference report.
        return subprocess.CompletedProcess(command, 96, stdout="fixture launch observed")

    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", preflight)
    with pytest.raises((ValueError, RuntimeError), match=error):
        cpp_runtime.run_cpp_pair(
            deployment_manifest=deployment["manifest_path"], runner=sys.executable,
            runner_options={
                "device_model": "Ascend310P3", "cann": "fake", "driver": "fake",
                "firmware": "fake", "runtime": "fake",
                "state_policy": cpp_runtime.INCREMENTAL_STATE_POLICY,
                "model_residency_policy": residency,
            },
            prompt_token_ids=[10] * prompt_rows, eos_token_ids=[], device_id=0,
            max_new_tokens=max_tokens, max_draft_tokens=15,
            raw_output=tmp_path / "raw.json", log_output=tmp_path / "runner.log",
            execute=execute, progress=True,
        )
    if prompt_rows == 17:
        stderr = capsys.readouterr().err
        assert "stage=fused-manifest mode=static feature_rows=64" in stderr
        assert "sha256=" in stderr
        command, = commands
        assert command[command.index("--fused-static-feature-rows") + 1] == "64"
        if residency == "phase-resident":
            assert command[command.index("--model-residency-policy") + 1] == residency
        else:
            assert "--model-residency-policy" not in command
        for role in _FUSED_SPECULATIVE_STEP_GRAPH_ABI:
            assert f"--{role}" in command
        assert "--draft-propose" not in command and "--target-verify-commit" not in command
    else:
        assert not commands
