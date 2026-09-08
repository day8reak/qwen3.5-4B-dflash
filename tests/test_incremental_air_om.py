"""Host contract tests. Small fixtures are not checkpoint/device evidence."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))
from qwen35_dflash.ascend310p.incremental import (
    DraftGraph,
    TargetRowsGraph,
    conv_chunk,
    copy_cache_rows,
    incremental_graph_specs,
    prefix_state,
    update_paged,
)
from qwen35_dflash.ascend310p.incremental_plan import validate_incremental_bundle
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.modeling_dflash import DFlashDraftModel


@pytest.fixture(autouse=True)
def small_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def gdr(query, key, value, *, g, beta, effective_length, initial_state, **kwargs):
    del kwargs
    state = initial_state
    outputs = []
    query, key, value = query.float(), key.float(), value.float()
    query, key = F.normalize(query, dim=-1), F.normalize(key, dim=-1)
    for i in range(query.shape[1]):
        decayed = state * g[:, i].exp()[..., None, None]
        residual = value[:, i] - (key[:, i, :, None, :] @ decayed).squeeze(-2)
        candidate = (
            decayed
            + key[:, i, :, :, None] * (residual * beta[:, i, :, None])[..., None, :]
        )
        state = torch.where(
            (i < effective_length)[:, None, None, None], candidate, state
        )
        outputs.append((query[:, i, :, None, :] @ state).squeeze(-2))
    return torch.stack(outputs, dim=1).half(), state


class GatedNorm(nn.Module):
    def forward(self, x, z):
        return x * F.silu(z)


class Gdn(nn.Module):
    def __init__(self):
        super().__init__()
        self.key_dim = self.value_dim = self.head_k_dim = self.head_v_dim = 16
        self.num_v_heads = self.num_k_heads = 1
        self.conv_dim, self.conv_kernel_size = 48, 4
        self.in_proj_qkv = nn.Linear(32, 48, bias=False)
        self.in_proj_z = nn.Linear(32, 16, bias=False)
        self.in_proj_b = nn.Linear(32, 1, bias=False)
        self.in_proj_a = nn.Linear(32, 1, bias=False)
        self.conv1d = nn.Conv1d(48, 48, 4, groups=48, bias=False)
        self.A_log, self.dt_bias = (
            nn.Parameter(torch.zeros(1)),
            nn.Parameter(torch.zeros(1)),
        )
        self.norm = GatedNorm()
        self.out_proj = nn.Linear(16, 32, bias=False)


class Rope(nn.Module):
    def forward(self, x, positions):
        return torch.ones(
            (*positions.shape, 16), device=x.device, dtype=x.dtype
        ), torch.zeros((*positions.shape, 16), device=x.device, dtype=x.dtype)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.head_dim, self.num_heads, self.num_key_value_heads = 16, 2, 1
        self.kv_max_len, self.block_size, self.scaling = 192, 64, 0.25
        self.q_proj, self.k_proj, self.v_proj, self.o_proj = (
            nn.Linear(32, width, bias=False) for width in (64, 16, 16, 32)
        )
        self.q_norm, self.k_norm, self.rotary_emb = nn.Identity(), nn.Identity(), Rope()
        self.register_buffer("block_table", torch.arange(3, dtype=torch.int32)[None])

    def transform_nd_2_nz(self, x):
        b, n, s, d = x.shape
        return (
            x.reshape(b, n, s, d // 16, 16)
            .transpose(2, 3)
            .contiguous()
            .reshape(b, n, s, d)
        )

    def transform_nz_2_nd(self, x):
        b, n, s, d = x.shape
        return (
            x.reshape(b, n, d // 16, s, 16)
            .transpose(2, 3)
            .contiguous()
            .reshape(b, n, s, d)
        )


def attention_op(*, query, key, value, atten_mask, **kwargs):
    del kwargs
    rows = query.shape[-2]
    q = query.reshape(1, 2, rows, 16)
    k, v = (
        item[0].permute(0, 2, 1, 3).reshape(1, 1, 192, 16).expand(1, 2, 192, 16)
        for item in (key, value)
    )
    scores = q.float() @ k.float().transpose(-1, -2) * 0.25 + atten_mask.float()
    return (scores.softmax(-1) @ v.float()).half().reshape_as(query)


def rotary(q, k, cosine, sine):
    del cosine, sine
    return q, k


class TinyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        self.requested_device, self.kv_cache_max_len = torch.device("cpu"), 192
        layers = nn.ModuleList()
        for kind in ("linear_attention", "full_attention"):
            layer = nn.Module()
            layer.block_type = kind
            if kind == "linear_attention":
                layer.linear_attn = Gdn()
            else:
                layer.self_attn = Attention()
            layer.input_layernorm = layer.post_attention_layernorm = nn.Identity()
            layer.mlp = nn.Sequential(
                nn.Linear(32, 64, bias=False), nn.SiLU(), nn.Linear(64, 32, bias=False)
            )
            layers.append(layer)
        self.dflash_execution_model = nn.Module()
        self.dflash_execution_model.language_model = nn.Module()
        self.dflash_execution_model.language_model.layers = layers
        self.dflash_execution_model.language_model.norm = nn.Identity()
        self.embedding, self.head = nn.Embedding(64, 32), nn.Linear(32, 64, bias=False)
        self.half()

    def get_input_embeddings(self):
        return self.embedding

    def get_output_embeddings(self):
        return self.head

    def _fresh_hybrid_cache(self, batch_size):
        assert batch_size == 1
        return [
            (torch.zeros(1, 48, 4).half(), torch.zeros(1, 1, 16, 16).half()),
            (torch.zeros(3, 1, 64, 16).half(), torch.zeros(3, 1, 64, 16).half()),
        ]


def draft_model():
    config = Qwen35DFlashConfig(
        hidden_size=32,
        intermediate_size=64,
        vocab_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        num_target_layers=2,
        target_layer_ids=(0, 1),
        layer_types=("sliding_attention", "full_attention"),
        block_size=16,
        mask_token_id=63,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=256,
        sliding_window=32,
        use_sliding_window=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype="float16",
    )
    draft = DFlashDraftModel(config, ops=AirDFlashOps(), dtype=torch.float16).eval()
    with torch.no_grad():
        for parameter in draft.parameters():
            parameter.normal_(0, 0.05)
    return draft


def specs(include_ordinary_decode=True):
    torch.manual_seed(42)
    target, draft = TinyTarget().eval(), draft_model()
    return incremental_graph_specs(
        target,
        draft,
        capacity=128,
        metadata={},
        gdr=gdr,
        attention=attention_op,
        rotary=rotary,
        include_ordinary_decode=include_ordinary_decode,
    )


def manifest_graphs(values):
    return [
        {
            "name": s.name,
            "role": s.role,
            "metadata": dict(s.metadata),
            "input_names": list(s.input_names),
            "output_names": list(s.output_names),
        }
        for s in values
    ]


@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("with_bias", [False, True])
def test_conv_chunk_preserves_output_and_every_committed_prefix(rows, dtype, with_bias):
    generator = torch.Generator().manual_seed(812)
    batch, channels, width = 2, 8, 4
    # Real GDN projections are transposed into [B,C,R], so include noncontiguous x.
    x = torch.randn(batch, rows, channels, generator=generator, dtype=dtype).transpose(1, 2)
    state = torch.randn(batch, channels, width, generator=generator, dtype=dtype)
    weight = torch.randn(channels, width, generator=generator, dtype=dtype)
    bias = torch.randn(channels, generator=generator, dtype=dtype) if with_bias else None
    original_x, original_state = x.clone(), state.clone()
    history = torch.cat((state, x), dim=-1).to(weight.dtype)
    reference_output = F.silu(
        F.conv1d(history, weight.unsqueeze(1), bias, groups=channels)
    )[..., -rows:].to(dtype)
    output, bank = conv_chunk(x, state, weight, bias)
    torch.testing.assert_close(output, reference_output, rtol=0, atol=0)
    assert bank.shape == (batch, rows, channels, width)
    assert bank.dtype == dtype and bank.is_contiguous()
    # Independent sequential cache update: each prefix must keep exactly its
    # last K values, including tails and zero-acceptance verify (anchor only).
    window = state.clone()
    for valid in range(1, rows + 1):
        window = torch.cat((window[..., 1:], x[..., valid - 1 : valid]), dim=-1)
        torch.testing.assert_close(bank[:, valid - 1], window, rtol=0, atol=0)
        committed = prefix_state(bank, torch.tensor([valid], dtype=torch.int16))
        torch.testing.assert_close(committed, window, rtol=0, atol=0)
    torch.testing.assert_close(x, original_x, rtol=0, atol=0)
    torch.testing.assert_close(state, original_state, rtol=0, atol=0)


def test_exactly_four_graphs_and_complete_signatures():
    values = specs()
    assert {s.name for s in values} == {
        "target_prefill",
        "target_decode",
        "target_verify",
        "draft",
    }
    assert validate_incremental_bundle(manifest_graphs(values))["capacity"] == 128
    with torch.inference_mode():
        for spec in values:
            outputs = spec.model(*spec.example_args)
            expected = spec.metadata["tensor_abi"]["outputs"]
            assert len(outputs) == len(expected)
            for tensor, desc in zip(outputs, expected):
                assert list(tensor.shape) == desc["shape"]
                assert str(tensor.dtype) == "torch." + desc["dtype"]
    bad = copy.deepcopy(manifest_graphs(values))
    bad[2]["metadata"]["tensor_abi"]["outputs"][1]["dtype"] = "int16"
    with pytest.raises(ValueError, match="ABI differs"):
        validate_incremental_bundle(bad)
    with pytest.raises(ValueError, match="four"):
        validate_incremental_bundle(manifest_graphs(values)[:-1])
    pure = [g for g in manifest_graphs(values) if g["name"] != "target_decode"]
    assert validate_incremental_bundle(pure)["capacity"] == 128
    assert len(specs(include_ordinary_decode=False)) == 3


@pytest.mark.parametrize("accepted", [0, 3, 15])
def test_fused_verify_accepts_prefix_and_commits_only_that_prefix(accepted):
    values = {s.name: s for s in specs()}
    verify, decode = values["target_verify"], values["target_decode"]
    initial = tuple(t.clone() for t in verify.example_args[3:])
    ids = [4]
    state = initial
    # Causal greedy proposals obtained using the same Target projections.
    with torch.inference_mode():
        for i in range(15):
            out = decode.model(
                torch.tensor([[ids[-1]]]),
                torch.tensor([i]),
                torch.tensor([1], dtype=torch.int16),
                *state,
            )
            ids.append(int(out[0][0, 0]))
            state = out[1:]
        if accepted < 15:
            ids[accepted + 1] = (ids[accepted + 1] + 1) % 64
        out = verify.model(
            torch.tensor([ids]),
            torch.tensor([0]),
            torch.tensor([16], dtype=torch.int16),
            *initial,
        )
        assert int(out[1][0]) == accepted
        # A fresh compact call over exactly anchor + accepted proposals is the
        # reference for the committed scalar GDN state and visible KV prefix.
        reference = copy.copy(decode.model)
        reference.rows = accepted + 1
        ref = reference(
            torch.tensor([ids[: accepted + 1]]),
            torch.tensor([0]),
            torch.tensor([accepted + 1], dtype=torch.int16),
            *initial,
        )
        torch.testing.assert_close(out[3], ref[1], atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(out[4], ref[2], atol=2e-3, rtol=2e-3)
        for actual, expected in zip(out[5:], ref[3:]):
            torch.testing.assert_close(
                actual[:, :, : accepted + 1],
                expected[:, :, : accepted + 1],
                atol=2e-3,
                rtol=2e-3,
            )
    for actual, expected in zip(initial, verify.example_args[3:]):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("dim,shape", [(0, (192, 2, 16)), (2, (1, 2, 192, 16))])
@pytest.mark.parametrize("rows", [1, 16, 64])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_cache_rows_match_index_copy_without_mutating_input(dim, shape, rows, dtype):
    generator = torch.Generator().manual_seed(813)
    cache = torch.randn(shape, generator=generator, dtype=dtype)
    original = cache.clone()
    value_shape = list(shape)
    value_shape[dim] = rows
    # Keep the value layout noncontiguous to exercise transposed Draft K/V.
    values = torch.randn(value_shape, generator=generator, dtype=dtype)
    values = values.transpose(-1, -2).contiguous().transpose(-1, -2)
    for start in (0, 63, shape[dim] - rows):
        positions = torch.arange(start, start + rows)
        expected = torch.index_copy(cache, dim, positions, values)
        actual = copy_cache_rows(cache, dim, positions, values)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(cache, original, rtol=0, atol=0)
        assert actual.data_ptr() != cache.data_ptr()


def test_cache_write_crosses_block_boundary_without_touching_prefix():
    cache = torch.arange(3 * 2 * 64 * 16).reshape(3, 2, 64, 16).half()
    old = cache.clone()
    values = torch.randn(1, 16, 2, 16).half()
    result = update_paged(cache, values, torch.arange(63, 79))
    flattened = result.permute(0, 2, 1, 3).reshape(192, 2, 16)
    torch.testing.assert_close(flattened[63:79], values[0])
    torch.testing.assert_close(
        flattened[:63], old.permute(0, 2, 1, 3).reshape(192, 2, 16)[:63]
    )
    torch.testing.assert_close(cache, old)


def test_draft_fused_context_matches_original_cached_draft_with_padding():
    torch.manual_seed(10)
    draft, target = draft_model(), TinyTarget().eval()
    graph = DraftGraph(draft, target.embedding, target.head)
    cache = draft.new_kv_cache(max_length=192)
    states = tuple(torch.zeros(1, 1, 192, 16).half() for _ in range(4))
    start = 0
    with torch.inference_mode():
        for rows in (37, 4, 1, 16):
            features = torch.randn(1, rows, 64).half()
            padded = F.pad(features, (0, 0, 0, 64 - rows), value=float("nan"))
            block = torch.tensor([[4] + [63] * 15])
            position_ids = torch.arange(start, start + rows + 16)[None]
            expected = draft.draft_top1_cached_projected(
                draft.project_target_hidden(features),
                target.embedding(block),
                position_ids,
                cache,
                target.head.weight,
            )
            actual = graph(
                padded,
                torch.tensor([start]),
                torch.tensor([rows], dtype=torch.int16),
                torch.tensor([4]),
                *states,
            )
            torch.testing.assert_close(actual[0], expected)
            states = actual[1:]
            start += rows
            for index in range(2):
                torch.testing.assert_close(
                    states[2 * index][:, :, :start],
                    cache._keys[index],
                    atol=1e-3,
                    rtol=1e-3,
                )
                torch.testing.assert_close(
                    states[2 * index + 1][:, :, :start],
                    cache._values[index],
                    atol=1e-3,
                    rtol=1e-3,
                )


def test_tensor_export_keeps_runtime_acceptance_and_state_inputs():
    spec = next(s for s in specs() if s.name == "target_verify")
    with torch.inference_mode():
        spec.model.head.weight.zero_()
        exported = torch.export.export(spec.model, spec.example_args).module()
        args = list(spec.example_args)
        args[2] = torch.tensor([4], dtype=torch.int16)
        actual = exported(*args)
        expected = spec.model(*args)
        assert int(actual[1][0]) == 3
        for lhs, rhs in zip(actual, expected):
            torch.testing.assert_close(lhs, rhs)
        args[0] = args[0].clone()
        args[0][0, 2] = 1
        assert int(exported(*args)[1][0]) == 1


def test_undefined_gdr_padding_cannot_pollute_later_kv():
    def poisoned(*args, **kwargs):
        output, state = gdr(*args, **kwargs)
        active = torch.arange(output.shape[1]) < kwargs["effective_length"]
        return torch.where(active[None, :, None, None], output, float("nan")), state

    target = TinyTarget().eval()
    graph = TargetRowsGraph(
        target,
        rows=64,
        verify=False,
        feature_layers=(0, 1),
        gdr=poisoned,
        attention=attention_op,
        rotary=rotary,
    )
    state = tuple(t for pair in target._fresh_hybrid_cache(1) for t in pair)
    with torch.inference_mode():
        outputs = graph(
            torch.zeros(1, 64, dtype=torch.long),
            torch.tensor([0]),
            torch.tensor([3], dtype=torch.int16),
            *state,
        )
    assert all(torch.isfinite(t).all() for t in outputs)


def test_draft_graph_exports_with_dynamic_context_length():
    spec = next(s for s in specs() if s.name == "draft")
    with torch.inference_mode():
        exported = torch.export.export(spec.model, spec.example_args).module()
        args = list(spec.example_args)
        args[2] = torch.tensor([37], dtype=torch.int16)
        for lhs, rhs in zip(exported(*args), spec.model(*args)):
            torch.testing.assert_close(lhs, rhs)


@pytest.fixture
def chunk_bundle(tmp_path, monkeypatch):
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle

    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    values = specs()
    by_name = {s.name: s for s in values}

    class FakeTorchAir:
        def dynamo_export(self, *args, model, export_path, export_name, **kwargs):
            assert len(model(*args)) == len(by_name[export_name].output_names)
            Path(export_path, export_name + ".air").write_text(
                json.dumps(by_name[export_name].metadata["tensor_abi"])
            )

    air = export_air_bundle(
        lambda config: values, {}, tmp_path / "bundle", torchair_module=FakeTorchAir()
    )

    def atc(command, cwd):
        prefix = Path(
            next(s.split("=", 1)[1] for s in command if s.startswith("--output="))
        )
        signature = by_name[prefix.name].metadata["tensor_abi"]
        lines = ["FAKE_CHUNK " + prefix.name]
        for key, marker in (("inputs", "I"), ("outputs", "O")):
            for t in signature[key]:
                lines.append(
                    " ".join(
                        (
                            marker,
                            t["name"],
                            t["dtype"],
                            str(len(t["shape"])),
                            *(str(d) for d in t["shape"]),
                        )
                    )
                )
        Path(str(prefix) + ".om").write_text("\n".join(lines))
        return subprocess.CompletedProcess(command, 0, "host fake ATC only")

    deployment = compile_air_bundle(
        air["manifest_path"],
        soc_version="Ascend310P3",
        atc_bin="/bin/true",
        runner=atc,
        atc_identity="fake-host-test",
    )
    return Path(deployment["manifest_path"])


def test_fake_conversion_preserves_tensor_abi_and_hashes(chunk_bundle, tmp_path):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan

    plan, deployment, contract = write_incremental_plan(
        chunk_bundle, tmp_path / "chunk-plan.txt"
    )
    assert plan.read_text().count("\ngraph ") == 4
    assert len(deployment["graphs"]) == 4
    assert contract["commit_capsules"] == "internal_to_target_verify_not_external_OM_IO"
    om = chunk_bundle.parent / deployment["graphs"][0]["om"]["path"]
    om.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        write_incremental_plan(chunk_bundle, tmp_path / "bad-plan.txt")


@pytest.mark.parametrize("failure", ["empty", "duplicate", "invalid-name"])
def test_compiler_rejects_invalid_graph_sets_before_atc(chunk_bundle, failure):
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle

    deployment = json.loads(chunk_bundle.read_text())
    air_path = chunk_bundle.parent / deployment["air_manifest"]["path"]
    air = json.loads(air_path.read_text())
    if failure == "empty":
        air["graphs"] = []
    elif failure == "duplicate":
        air["graphs"].append(air["graphs"][0])
    else:
        air["graphs"][0]["name"] = "../escape"
    air_path.write_text(json.dumps(air))
    calls = []
    with pytest.raises(ValueError):
        compile_air_bundle(
            air_path,
            soc_version="Ascend310P3",
            atc_bin="/bin/true",
            runner=lambda *args: calls.append(args),
            atc_identity="host-test",
        )
    assert calls == []


@pytest.mark.parametrize("accepted,eos", [(0, []), (3, []), (15, []), (15, [7])])
def test_cpp_four_om_roundtrip_with_fake_acl(
    chunk_bundle, tmp_path, monkeypatch, accepted, eos
):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the CMake fake ACL executable")
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", str(accepted))
    result = run_cpp_pair(
        deployment_manifest=chunk_bundle,
        runner=runner,
        runner_options={
            "device_model": "Ascend310P3-host-fixture",
            "cann": "fake",
            "driver": "fake",
            "firmware": "fake",
            "runtime": "fake-acl",
        },
        prompt_token_ids=[4] * 65,
        eos_token_ids=eos,
        device_id=0,
        max_new_tokens=40,
        max_draft_tokens=15,
        raw_output=tmp_path / "cpp.json",
        log_output=tmp_path / "cpp.log",
    )
    assert result["ordinary_parity"]["token_id_mismatches"] == 0
    assert result["abi"]["graph_count"] == 4
    for row in result["dflash"]["measurements"]:
        assert "target_decode" not in row["stage_ms"]
        assert "target_verify" in row["stage_ms"]


@pytest.mark.parametrize(
    "failure", ["QWEN35_FAKE_FAIL_GRAPH", "QWEN35_FAKE_BAD_ACCEPT"]
)
def test_cpp_rejects_failed_or_inconsistent_verify(
    chunk_bundle, tmp_path, monkeypatch, failure
):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    monkeypatch.setenv(failure, "target_verify" if failure.endswith("GRAPH") else "1")
    output = tmp_path / "fail.json"
    result = subprocess.run(
        [
            runner,
            "--model",
            str(plan),
            "--model-sha256",
            sha256_file(plan),
            "--model-kind",
            "chunk",
            "--mode",
            "dflash",
            "--prompt-token-ids",
            "4",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert not output.exists()
    if failure == "QWEN35_FAKE_BAD_ACCEPT":
        assert "host acceptance disagrees" in result.stderr


def test_pure_dflash_does_not_load_or_require_decode_om(chunk_bundle, tmp_path):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, deployment, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    decode = next(g for g in deployment["graphs"] if g["name"] == "target_decode")
    (chunk_bundle.parent / decode["om"]["path"]).unlink()
    output = tmp_path / "dflash.json"
    result = subprocess.run(
        [
            runner,
            "--model",
            str(plan),
            "--model-sha256",
            sha256_file(plan),
            "--model-kind",
            "chunk",
            "--mode",
            "dflash",
            "--prompt-token-ids",
            "4",
            "--output",
            str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "loaded_models=3" in result.stderr
    assert json.loads(output.read_text())["ordinary_parity"]["status"] == "NOT_RUN"
