"""Host contract tests. Small fixtures are not checkpoint/device evidence."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
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
    accepted_prefix_length,
    conv_chunk,
    copy_cache_rows,
    incremental_graph_specs,
    prefix_state,
    update_paged,
)
from qwen35_dflash.ascend310p.incremental_plan import validate_incremental_bundle
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps, _repeat_kv
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.modeling_dflash import DFlashDraftModel, DFlashRMSNorm
from models.dflash_v1.dflash_ops import TorchDFlashOps
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


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
        self.dflash_execution_model.lm_head = self.head
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


def cache_update_reference(cache, updates, target_block, offset):
    """CPU model of the receiver's single-page ABI, not an NPU fallback."""
    assert target_block.dtype == torch.int32 and target_block.shape == (1,)
    assert offset.dtype == torch.int32 and offset.shape == ()
    block, row = int(target_block.item()), int(offset.item())
    assert 0 <= block < cache.shape[0]
    assert 0 <= row < row + updates.shape[0] <= cache.shape[2]
    result = cache.clone()
    result[block, :, row : row + updates.shape[0], :] = updates.transpose(0, 1)
    return result


def specs(include_ordinary_decode=True, cache_update=None):
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
        cache_update=cache_update,
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


@pytest.mark.parametrize(
    "valid_rows,accepted",
    [(16, 0), (16, 3), (16, 15), (1, 0), (4, 0), (4, 1), (4, 3)],
)
@pytest.mark.parametrize("cache_update", [None, cache_update_reference], ids=["scatter", "CacheUpdate"])
@pytest.mark.parametrize("start", [0, 63])
def test_fused_verify_accepts_prefix_and_commits_only_that_prefix(
    valid_rows, accepted, cache_update, start,
):
    values = {s.name: s for s in specs(cache_update=cache_update)}
    verify, decode = values["target_verify"], values["target_decode"]
    initial = tuple(t.clone() for t in verify.example_args[3:])
    ids = [4]
    state = initial
    # Causal greedy proposals obtained using the same Target projections.
    with torch.inference_mode():
        for i in range(15):
            out = decode.model(
                torch.tensor([[ids[-1]]]),
                torch.tensor([start + i]),
                torch.tensor([1], dtype=torch.int16),
                *state,
            )
            ids.append(int(out[0][0, 0]))
            state = out[1:]
        if accepted < 15:
            ids[accepted + 1] = (ids[accepted + 1] + 1) % 64
        # Invalid tail tokens deliberately mismatch; they cannot reduce the
        # accepted count or enter the committed GDN/visible KV prefix.
        ids[valid_rows:] = [63] * (16 - valid_rows)
        out = verify.model(
            torch.tensor([ids]),
            torch.tensor([start]),
            torch.tensor([valid_rows], dtype=torch.int16),
            *initial,
        )
        assert int(out[1][0]) == accepted
        # A fresh compact call over exactly anchor + accepted proposals is the
        # reference for the committed scalar GDN state and visible KV prefix.
        reference = copy.deepcopy(decode.model)
        reference.rows = accepted + 1
        for block in reference.blocks:
            if hasattr(block, "cache_update"):
                block.cache_update = None
        ref = reference(
            torch.tensor([ids[: accepted + 1]]),
            torch.tensor([start]),
            torch.tensor([accepted + 1], dtype=torch.int16),
            *initial,
        )
        torch.testing.assert_close(out[3], ref[1], atol=2e-3, rtol=2e-3)
        torch.testing.assert_close(out[4], ref[2], atol=2e-3, rtol=2e-3)
        # Remaining outputs after the four cache tensors are discard-only.
        for actual, expected in zip(out[5:7], ref[3:]):
            actual = actual.permute(0, 2, 1, 3).flatten(0, 1)
            expected = expected.permute(0, 2, 1, 3).flatten(0, 1)
            torch.testing.assert_close(
                actual[: start + accepted + 1],
                expected[: start + accepted + 1],
                atol=2e-3,
                rtol=2e-3,
            )
        # Rejected physical KV rows remain present but invisible. A subsequent
        # decode must overwrite the next slot and match the compact reference.
        next_ids = out[0][:, accepted : accepted + 1]
        next_args = (next_ids, torch.tensor([start + accepted + 1]),
                     torch.tensor([1], dtype=torch.int16))
        reference.rows = 1
        actual_next = decode.model(*next_args, *out[3:7])
        expected_next = reference(*next_args, *ref[1:])
        torch.testing.assert_close(actual_next[0], expected_next[0], rtol=0, atol=0)
    for actual, expected in zip(initial, verify.example_args[3:]):
        torch.testing.assert_close(actual, expected)


@pytest.fixture(scope="module")
def gdr_output_fixture():
    lib = torch.library.Library("verify_state_fixture", "DEF")
    lib.define("gdr(Tensor query, Tensor key, Tensor value, Tensor g, Tensor beta, "
               "Tensor effective_length, Tensor initial_state, int chunk_size, "
               "bool output_final_state, bool use_qk_l2norm_in_kernel) -> (Tensor, Tensor)")

    def operation(query, key, value, g, beta, effective_length, initial_state,
                  chunk_size, output_final_state, use_qk_l2norm_in_kernel):
        assert chunk_size == 64 and output_final_state and use_qk_l2norm_in_kernel
        return torch.zeros_like(value), initial_state + effective_length.float() * 0.0137

    lib.impl("gdr", operation, "CPU")
    lib.impl("gdr", lambda q, k, v, g, b, length, state, *attrs:
             (torch.empty_like(v), torch.empty_like(state)), "Meta")
    yield torch.ops.verify_state_fixture.gdr.default
    lib._destroy()


@pytest.mark.parametrize("valid_rows,accepted", [(16, 0), (16, 7), (16, 15), (8, 3), (1, 0)])
def test_verify_retains_raw_first_pass_outputs_but_commits_second_pass(valid_rows, accepted, gdr_output_fixture):
    import operator

    # Two GDN layers separated by attention detect index/order mixups. This
    # opaque host op tests output liveness through torch.export, not GDR math.
    target = TinyTarget().eval()
    target.dflash_execution_model.language_model.layers.append(
        copy.deepcopy(target.dflash_execution_model.language_model.layers[0])
    )
    graph = TargetRowsGraph(
        target, rows=16, verify=True, feature_layers=(0, 2),
        gdr=gdr_output_fixture, attention=attention_op, rotary=rotary,
    )
    state = [t for pair in target._fresh_hybrid_cache(1) for t in pair]
    state += [t.clone() for t in state[:2]]
    state[1].fill_(0.1372)
    state[5].fill_(0.2734)
    frozen = [t.clone() for t in state]
    ids = torch.zeros(1, 16, dtype=torch.int64)
    if accepted + 1 < valid_rows:
        ids[0, accepted + 1] = 1
    args = (ids, torch.tensor([0]), torch.tensor([valid_rows], dtype=torch.int16), *state)
    with torch.inference_mode():
        target.head.weight.zero_()
        eager = graph(*args)
        ep = torch.export.export(graph, args)
        actual = ep.module()(*args)
    assert len(actual) == 11  # Top1, acceptance, features, six caches, two discards
    assert int(actual[1][0]) == accepted
    for a, b in zip(actual, eager):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for index, layer in enumerate((0, 2)):
        raw = actual[9 + index]
        committed = actual[3 + 2 * layer + 1]
        initial = state[2 * layer + 1].float()
        assert raw.dtype == torch.float32 and committed.dtype == torch.float16
        torch.testing.assert_close(raw, initial + valid_rows * 0.0137, rtol=0, atol=0)
        torch.testing.assert_close(committed, (initial + (accepted + 1) * 0.0137).half(),
                                   rtol=0, atol=0)
    for a, b in zip(state, frozen):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    calls = [n for n in ep.graph.nodes if n.target == gdr_output_fixture]
    outputs = next(n for n in ep.graph.nodes if n.op == "output").args[0]
    assert len(calls) == 4
    for first, second, output in zip(calls[:2], calls[2:], outputs[-2:]):
        assert output.target == operator.getitem and output.args == (first, 1)
        # Both calls read the round-start state; only effective_length changes.
        assert first.args[6] is second.args[6]
        assert first.args[5] is not second.args[5]
        assert first.args[7:] == second.args[7:] == (64, True, True)

@pytest.mark.parametrize("damage", ["missing", "fp16", "shape", "cache_alias", "input", "policy", "stale"])
def test_verify_discard_contract_rejects_incompatible_artifacts(damage):
    graphs = manifest_graphs(specs())
    c = graphs[0]["metadata"]["incremental_contract"]
    verify = next(g for g in graphs if g["name"] == "target_verify")
    if damage == "missing":
        verify["metadata"]["tensor_abi"]["outputs"].pop()
        verify["output_names"].pop()
    elif damage in {"fp16", "shape", "cache_alias"}:
        tensor = c["verify_discard_states"][0]
        if damage == "fp16":
            tensor["dtype"] = "float16"
        elif damage == "shape":
            tensor["shape"] = [1]
        else:
            tensor["name"] = c["gdn_states"][1]
    elif damage == "input":
        verify["metadata"]["tensor_abi"]["inputs"].append(c["verify_discard_states"][0])
    elif damage == "policy":
        c.pop("verify_state_output_policy")
    else:
        c["abi"] = "qwen35-dflash-chunk-v2"
    with pytest.raises(ValueError):
        validate_incremental_bundle(graphs)


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


@pytest.mark.parametrize("rows,start", [(1, 0), (1, 63), (1, 191),
                                      (16, 0), (16, 63), (16, 176),
                                      (64, 0), (64, 64), (64, 128)])
def test_cache_update_matches_scatter_with_page_boundaries_and_scratch(rows, start):
    generator = torch.Generator().manual_seed(813)
    cache = torch.randn(3, 2, 64, 16, generator=generator).half()
    original = cache.clone()
    # RoPE K arrives transposed, V is contiguous: test both packing paths.
    values = torch.randn(1, 2, rows, 16, generator=generator).half().transpose(1, 2)
    positions = torch.arange(start, start + rows)
    calls = []

    def operation(cache, updates, target_block, offset):
        calls.append((int(target_block.item()), int(offset.item()), updates.shape[0]))
        return cache_update_reference(cache, updates, target_block, offset)

    expected = update_paged(cache, values, positions)
    actual = update_paged(cache, values, positions, cache_update=operation,
                          aligned_prefill=rows == 64)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(cache, original, rtol=0, atol=0)
    assert calls == ([(start // 64, 0, 64)] if rows == 64 else
                     [(p // 64, p % 64, 1) for p in range(start, start + rows)])


@pytest.mark.parametrize("valid_rows", [1, 17, 64])
def test_cache_update_prefill_padding_matches_reference_and_next_decode(valid_rows):
    values = {s.name: s for s in specs(cache_update=cache_update_reference)}
    prefill, decode = values["target_prefill"], values["target_decode"]
    reference = copy.deepcopy(prefill.model)
    for block in reference.blocks:
        if hasattr(block, "cache_update"):
            block.cache_update = None
    # Two aligned chunks exercise prefix preservation and a short final gear.
    state, reference_state = prefill.example_args[3:], prefill.example_args[3:]
    with torch.inference_mode():
        for start, valid in [(0, 64), (64, valid_rows)]:
            ids = torch.arange(64)[None] % 63
            args = (ids, torch.tensor([start]), torch.tensor([valid], dtype=torch.int16))
            out = prefill.model(*args, *state)
            ref = reference(*args, *reference_state)
            for actual, expected in zip(out, ref):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state, reference_state = out[2:], ref[2:]
        reference = copy.deepcopy(decode.model)
        for block in reference.blocks:
            if hasattr(block, "cache_update"):
                block.cache_update = None
        args = (out[0], torch.tensor([64 + valid_rows]), torch.ones(1, dtype=torch.int16))
        out = decode.model(*args, *state)
        ref = reference(*args, *reference_state)
        for actual, expected in zip(out, ref):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("proposal_count", [1, 2, 7, 13, 15])
def test_draft_fused_context_matches_original_cached_draft_with_padding(proposal_count):
    torch.manual_seed(10)
    draft, target = draft_model(), TinyTarget().eval()
    # Unit RMS weights keep attention contributions material. Small random
    # norm weights can hide an incorrect full-width non-causal Draft block.
    with torch.no_grad():
        for module in draft.modules():
            if isinstance(module, DFlashRMSNorm):
                module.weight.fill_(1)
    native = copy.deepcopy(draft)
    native.set_ops(TorchDFlashOps())
    graph = DraftGraph(draft, target.embedding, target.head)
    cache = native.new_kv_cache(max_length=192)
    states = tuple(torch.zeros(1, 1, 192, 16).half() for _ in range(4))
    start = 0
    with torch.inference_mode():
        for rows in (37, 4, 1, 16):
            features = torch.randn(1, rows, 64).half()
            padded = F.pad(features, (0, 0, 0, 64 - rows), value=float("nan"))
            block = torch.tensor([[4] + [63] * proposal_count])
            position_ids = torch.arange(start, start + rows + proposal_count + 1)[None]
            expected = native.draft_top1_cached_projected(
                native.project_target_hidden(features),
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
                torch.tensor([proposal_count], dtype=torch.int16),
                *states,
            )
            torch.testing.assert_close(actual[0][:, :proposal_count], expected)
            assert torch.count_nonzero(actual[0][:, proposal_count:]) == 0
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


@pytest.mark.parametrize("proposal_count", [1, 2, 7, 13, 15])
def test_short_block_excludes_padding_from_noncausal_attention(proposal_count):
    # Uniform final-layer attention exposes the logical K denominator.
    # Context V points at channel 0, all noise V at channel 1; including
    # fifteen masks instead of one changes the winning vocabulary ID.
    draft, target = draft_model(), TinyTarget().eval()
    with torch.no_grad():
        for parameter in draft.parameters():
            parameter.zero_()
        for module in draft.modules():
            if isinstance(module, DFlashRMSNorm):
                module.weight.fill_(1)
        draft.fc.weight[:, :32].copy_(torch.eye(32).half())
        attention = draft.layers[-1].self_attn
        attention.v_proj.weight[:, :16].copy_(torch.eye(16).half())
        attention.o_proj.weight.copy_(torch.eye(32).half())
        target.embedding.weight.zero_()
        target.embedding.weight[:, 1] = 0.1
        target.head.weight.zero_()
        target.head.weight[0, 0] = 1
        target.head.weight[1, 1] = 1
    native = copy.deepcopy(draft)
    native.set_ops(TorchDFlashOps())
    graph = DraftGraph(draft, target.embedding, target.head)
    features = torch.zeros(1, 4, 64).half()
    features[..., 0] = 1
    padded = F.pad(features, (0, 0, 0, 60), value=float("nan"))
    states = tuple(torch.zeros(1, 1, 192, 16).half() for _ in range(4))
    args = (padded, torch.tensor([0]), torch.tensor([4], dtype=torch.int16),
            torch.tensor([4]))
    with torch.inference_mode():
        expected = native.draft_top1_cached_projected(
            native.project_target_hidden(features),
            target.embedding(torch.tensor([[4] + [63] * proposal_count])),
            torch.arange(4 + proposal_count + 1)[None],
            native.new_kv_cache(max_length=192), target.head.weight,
        )
        actual = graph(*args, torch.tensor([proposal_count], dtype=torch.int16), *states)[0]
        torch.testing.assert_close(actual[:, :proposal_count], expected, rtol=0, atol=0)
        if proposal_count <= 2:
            legacy = graph(*args, torch.tensor([15], dtype=torch.int16), *states)[0]
            assert torch.all(expected == 0)
            assert torch.all(legacy[:, :proposal_count] == 1)


class AcceptanceGraph(nn.Module):
    def forward(self, input_ids, top1, valid_rows):
        return accepted_prefix_length(input_ids, top1, valid_rows)


@pytest.mark.parametrize("capture", ["eager", "export", "aot"])
def test_acceptance_all_mismatch_patterns_and_valid_lengths(capture):
    # Exhaust all 2**15 match/mismatch patterns for every valid_rows in 1..16.
    # The oracle uses integer bits, independently of Tensor scans/reductions.
    width, batch = 15, 1 << 15
    patterns = torch.arange(batch, dtype=torch.long)
    bits = (patterns[:, None] >> torch.arange(width)) & 1
    top1 = torch.arange(width + 1)[None, :].expand(batch, -1) + 248000
    input_ids = torch.cat((torch.full((batch, 1), 42), top1[:, :-1] ^ bits), dim=1)
    valid = torch.full((batch,), 16, dtype=torch.int16)
    graph = AcceptanceGraph()
    captured_graphs = []
    if capture == "export":
        exported = torch.export.export(graph, (input_ids, top1, valid), strict=True)
        exported = exported.run_decompositions({})
        captured_graphs.append(exported.graph_module)
        run = exported.module()
    elif capture == "aot":
        from torch._dynamo.backends.common import aot_autograd

        def compiler(module, args):
            captured_graphs.append(module)
            return module.forward

        run = torch.compile(graph, backend=aot_autograd(fw_compiler=compiler),
                            fullgraph=True, dynamic=False)
    else:
        run = graph
    with torch.inference_mode():
        for length in range(1, width + 2):
            valid = torch.full((batch,), length, dtype=torch.int16)
            actual = run(input_ids, top1, valid)
            expected = []
            for pattern in range(batch):
                mismatches = pattern & ((1 << (length - 1)) - 1)
                expected.append((mismatches & -mismatches).bit_length() - 1
                                if mismatches else length - 1)
            assert actual.dtype == torch.int64 and actual.shape == (batch,)
            torch.testing.assert_close(actual, torch.tensor(expected), rtol=0, atol=0)
    if capture != "eager":
        # Changing runtime valid_rows must reuse one graph and retain integer
        # scan/reduction. PyTorch's implicit default would promote to INT64.
        assert len(captured_graphs) == 1
        nodes = [n for n in captured_graphs[0].graph.nodes if n.op == "call_function"]
        targets = {str(n.target) for n in nodes}
        assert not targets & {"aten.amin.default", "aten.min.dim", "aten.cumprod.default"}
        for target in ("aten.cumsum.default", "aten.sum.dim_IntList"):
            reductions = [n for n in nodes if str(n.target) == target]
            assert len(reductions) == 1
            assert reductions[0].kwargs["dtype"] == torch.int32
            assert reductions[0].meta["val"].dtype == torch.int32


@pytest.mark.parametrize("repetitions", [1, 2, 4])
@pytest.mark.parametrize("sequence", [16, 2064])
def test_draft_kv_head_repeat_preserves_group_order(repetitions, sequence):
    # Multiple distinct heads and noncontiguous values detect repetition of
    # the head axis instead of the inserted group axis, even at full KV width.
    states = (torch.arange(2 * 8 * sequence * 16) % 2048).half()
    states = states.reshape(2, sequence, 8, 16).transpose(1, 2)
    actual = _repeat_kv(states, repetitions)
    expected = torch.stack([states[:, head]
                            for head in range(8) for _ in range(repetitions)], dim=1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if repetitions == 1:
        assert actual is states


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


def test_target_gears_use_execution_head_and_draft_keeps_checkpoint_head():
    target = TinyTarget().eval()
    # The receiver retains one FP16 head for Draft while replacing the Target
    # head during quantization. Distinct Top1s expose selecting the wrong one.
    execution_head = nn.Linear(32, 64, bias=True).half()
    draft_head = nn.Linear(32, 64, bias=True).half()
    with torch.no_grad():
        for head, token in ((execution_head, 7), (draft_head, 25)):
            head.weight.zero_()
            head.bias.zero_()
            head.bias[token] = 1
    target.head = draft_head
    target.dflash_execution_model.lm_head = execution_head
    graphs = incremental_graph_specs(
        target, draft_model(), capacity=128, metadata={},
        gdr=gdr, attention=attention_op, rotary=rotary, include_ordinary_decode=True)
    with torch.inference_mode():
        for spec in graphs:
            if spec.name == "draft":
                assert spec.model.propose.head is draft_head
                assert spec.model.propose.head.weight.dtype == torch.float16
            else:
                assert torch.all(spec.model(*spec.example_args)[0] == 7)


def test_target_missing_execution_head_fails_before_export():
    target = TinyTarget().eval()
    del target.dflash_execution_model.lm_head
    with pytest.raises(TypeError, match="execution-model lm_head"):
        TargetRowsGraph(target, rows=64, verify=False, feature_layers=(0, 1),
                        gdr=gdr, attention=attention_op, rotary=rotary)


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


@pytest.mark.parametrize("rows,start,valid", [
    (64, 0, 1), (64, 63, 64), (64, 2049, 7), (64, 4083, 13),
    (16, 63, 1), (16, 2049, 16), (16, 4091, 5),
    (1, 2049, 1), (1, 4095, 1),
])
def test_attention_lengths_and_mask_preserve_runtime_prefix(rows, start, valid):
    # Positions beyond 2048 include integers FP16 cannot represent exactly.
    # Lengths stay in integer inputs; FP16 carries only the exact 0/-inf mask.
    capacity = 4160
    target = TinyTarget().eval()
    target.kv_cache_max_len = capacity
    base = target.dflash_execution_model.language_model.layers[1].self_attn
    base.kv_max_len = capacity
    base.block_table = torch.arange(capacity // 64, dtype=torch.int32)[None]
    calls = []
    def attention(**kwargs):
        calls.append(kwargs)
        return torch.zeros_like(kwargs["query"])
    graph = TargetRowsGraph(target, rows=rows, verify=rows == 16,
                            feature_layers=(), gdr=gdr, attention=attention, rotary=rotary)
    conv, recurrent = target._fresh_hybrid_cache(1)[0]
    key, value = (torch.zeros(capacity // 64, 1, 64, 16).half() for _ in range(2))
    with torch.inference_mode():
        graph(torch.zeros(1, rows, dtype=torch.long), torch.tensor([start]),
              torch.tensor([valid], dtype=torch.int16), conv, recurrent, key, value)
    assert len(calls) == 1
    call = calls[0]
    assert call.get("pse_shift") is None
    assert call["all_seq_lengths_q"] == [capacity]
    assert call["actual_seq_lengths_q"] == [rows]
    assert call["actual_seq_lengths_kv"] == [capacity]
    mask = call["atten_mask"]
    assert mask.dtype == torch.float16 and mask.shape == (1, 1, rows, capacity)
    for row in range(rows):
        visible_end = start + min(row + 1, valid)
        assert torch.equal(mask[0, 0, row, :visible_end], torch.zeros(visible_end).half())
        assert torch.isneginf(mask[0, 0, row, visible_end:]).all()
    assert call["block_table"].dtype == torch.int32


def test_draft_graph_exports_with_dynamic_context_length():
    spec = next(s for s in specs() if s.name == "draft")
    with torch.inference_mode():
        program = torch.export.export(spec.model, spec.example_args).run_decompositions()
        # Include projections and head as well as QK/PV: the complete Draft
        # ATen graph must not contain FP32 x FP32 mm/bmm in the default mode.
        matmuls = [n for n in program.graph.nodes if n.target in (
            torch.ops.aten.mm.default, torch.ops.aten.bmm.default)]
        bmms = [n for n in matmuls if n.target == torch.ops.aten.bmm.default]
        assert len(bmms) == 2 * len(spec.model.propose.draft.layers)
        assert len(matmuls) > len(bmms)
        for node in matmuls:
            assert node.args[0].meta["val"].dtype == torch.float16, node
            assert node.args[1].meta["val"].dtype == torch.float16, node
        softmaxes = [n for n in program.graph.nodes
                     if n.target == torch.ops.aten._softmax.default]
        assert len(softmaxes) == len(spec.model.propose.draft.layers)
        assert all(n.args[0].meta["val"].dtype == torch.float32 for n in softmaxes)
        exported = program.module()
        args = list(spec.example_args)
        args[2] = torch.tensor([37], dtype=torch.int16)
        for count in (1, 7, 15):
            args[4] = torch.tensor([count], dtype=torch.int16)
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
    assert deployment["compiler"]["precision_policy"] == "preserve_graph_dtypes"
    for graph in deployment["graphs"]:
        assert "--precision_mode=must_keep_origin_dtype" in graph["atc_command"]
    air = json.loads((chunk_bundle.parent / deployment["air_manifest"]["path"]).read_text())
    assert [g["runtime_input_abi"] for g in deployment["graphs"]] == [
        g["runtime_input_abi"] for g in air["graphs"]
    ]
    assert contract["commit_capsules"] == "internal_to_target_verify_not_external_OM_IO"
    om = chunk_bundle.parent / deployment["graphs"][0]["om"]["path"]
    om.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="integrity"):
        write_incremental_plan(chunk_bundle, tmp_path / "bad-plan.txt")


@pytest.mark.parametrize("failure", ["empty", "duplicate", "invalid-name", "attention-abi", "missing-attention-abi",
                                      "stale-chunk-abi", "missing-draft-length-policy",
                                      "missing-input-abi", "failed-input-abi"])
def test_compiler_rejects_invalid_graph_sets_before_atc(chunk_bundle, failure):
    from qwen35_dflash.ascend310p.compiler import compile_air_bundle

    deployment = json.loads(chunk_bundle.read_text())
    air_path = chunk_bundle.parent / deployment["air_manifest"]["path"]
    air = json.loads(air_path.read_text())
    if failure == "empty":
        air["graphs"] = []
    elif failure == "duplicate":
        air["graphs"].append(air["graphs"][0])
    elif failure == "invalid-name":
        air["graphs"][0]["name"] = "../escape"
    elif failure == "missing-input-abi":
        air["graphs"][-1].pop("runtime_input_abi")
    elif failure == "failed-input-abi":
        air["graphs"][-1]["runtime_input_abi"]["status"] = "FAIL"
    else:
        for graph in air["graphs"]:
            contract = graph["metadata"]["incremental_contract"]
            if failure == "attention-abi":
                contract["attention_export"] = "receiver_adn_fused_infer_attention_pse_shift_int64_logical_end"
            elif failure == "stale-chunk-abi":
                contract["abi"] = "qwen35-dflash-chunk-v1"
            elif failure == "missing-draft-length-policy":
                contract.pop("draft_length_policy", None)
            else:
                contract.pop("attention_export", None)
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


@pytest.mark.parametrize("failure", ["input-order", *[
    f"{direction}-{field}" for direction in ("input", "output")
    for field in ("dtype", "bytes", "rank", "shape", "count")
]])
def test_cpp_reports_actual_om_descriptors_before_execute(
    chunk_bundle, tmp_path, monkeypatch, failure,
):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the CMake fake ACL executable")
    monkeypatch.setenv("QWEN35_FAKE_CHUNK_IO_FAULT", failure)
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    output, log = tmp_path / "cpp.json", tmp_path / "cpp.log"
    with pytest.raises(RuntimeError, match="graph=draft") as caught:
        run_cpp_pair(
            deployment_manifest=chunk_bundle, runner=runner,
            runner_options={"device_model": "host-fixture", "cann": "fake",
                            "driver": "fake", "firmware": "fake", "runtime": "fake-acl"},
            prompt_token_ids=[4], eos_token_ids=[], device_id=0,
            max_new_tokens=16, max_draft_tokens=15, raw_output=output, log_output=log,
        )
    text = log.read_text()
    assert "OM I/O descriptors graph=draft" in text
    assert "inputs: plan=9 om=" in text and "outputs: plan=5 om=" in text
    assert not output.exists() and not events.exists()
    detail = str(caught.value)
    if failure == "input-order":
        assert 'input[1] expected={name="start_position" dtype=int64(9) bytes=8 rank=1 shape=[1]}' in detail
        assert 'actual={name="valid_rows" dtype=int16(6) bytes=2 rank=1 shape=[1]' in detail
    elif failure.endswith("count"):
        assert "OM tensor count differs" in detail
    else:
        assert f'{failure.split("-")[0]}[0] expected={{' in detail
        assert all(word in detail for word in ("dtype=", "bytes=", "rank=", "shape=", "actual={"))


@pytest.mark.parametrize("accepted,eos", [(0, []), (3, []), (15, []), (15, [7])])
@pytest.mark.parametrize("low_memory", [False, True])
def test_cpp_four_om_roundtrip_with_fake_acl(
    chunk_bundle, tmp_path, monkeypatch, accepted, eos, low_memory
):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the CMake fake ACL executable")
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", str(accepted))
    workspace = tmp_path / "workspace.jsonl"
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_WORKSPACE_LOG", str(workspace))
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
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
        trace_rounds=True,
        low_memory=low_memory,
    )
    assert result["ordinary_parity"]["token_id_mismatches"] == 0
    assert result["abi"]["graph_count"] == 4
    assert result["protocol"]["low_memory"] is low_memory
    assert result["protocol"]["dflash_speculation_policy"] == "always_on"
    assert result["protocol"]["max_resident_models"] == (3 if low_memory else 4)
    assert result["protocol"]["order"] == (
        "ordinary then DFlash with model unload between modes" if low_memory
        else "alternating ordinary/DFlash in one loaded process"
    )
    log = (tmp_path / "cpp.log").read_text()
    assert_cpp_resources_released(cleanup, log)
    loaded = [json.loads(line) for line in workspace.read_text().splitlines()]
    assert [r[0] for r in loaded] == (
        ["target_decode", "target_prefill", "draft", "target_prefill", "target_verify"]
        if low_memory else ["draft", "target_decode", "target_prefill", "target_verify"]
    )
    assert [r[3] for r in loaded] == ([1, 2, 1, 2, 3] if low_memory else [1, 2, 3, 4])
    groups = [loaded[:2], loaded[2:]] if low_memory else [loaded]
    for group in groups:
        assert len({r[1] for r in group}) == 1  # All models borrow one work buffer.
        assert {r[2] for r in group} == {3145728}  # Maximum, not sum.
    if low_memory:
        assert log.index("unload graph=target_prefill") < log.index("load graph=draft")
        assert result["startup_ms"]["mode_switch_unload"] > 0
    else:
        assert result["startup_ms"]["mode_switch_unload"] == 0
    for row in result["dflash"]["measurements"]:
        assert "target_decode" not in row["stage_ms"]
        assert "target_verify" in row["stage_ms"]
        assert row["counters"]["speculation_disable_events"] == 0
        assert row["counters"]["target_only_fallback_rounds"] == 0
        assert all(r["proposed_token_ids"] for r in row["rounds"][1:])
        if accepted == 0:
            assert len(row["stage_ms"]["target_verify"]) == 39
            # One additional Draft call primes KV for the first 64 prompt rows.
            assert len(row["stage_ms"]["draft"]) == 40
            assert row["counters"]["drafted_tokens"] == sum(min(15, n) for n in range(1, 40))
            assert row["counters"]["accepted_draft_tokens"] == 0
    assert result["protocol"]["round_trace_enabled"] is True
    for mode in ("ordinary", "dflash"):
        for row in result[mode]["measurements"]:
            emitted = []
            for round in row["rounds"]:
                assert round["committed_prefix_length"] == 65 + len(emitted)
                proposed, verified = round["proposed_token_ids"], round["target_token_ids"]
                accepted_ids = round["accepted_draft_token_ids"]
                assert accepted_ids == proposed[:len(accepted_ids)] == verified[:len(accepted_ids)]
                assert len(verified) == len(proposed) + 1
                expected = list(accepted_ids)
                if round["fallback_token_id"] is not None:
                    expected.append(round["fallback_token_id"])
                    assert round["fallback_token_id"] == verified[len(accepted_ids)]
                assert round["emitted_token_ids"] == expected
                emitted.extend(expected)
            assert emitted == row["generated_token_ids"]


def assert_cpp_resources_released(path, stderr):
    resources = json.loads(path.read_text())
    # ACL test-double counters cover every managed device/host buffer, model,
    # descriptor, dataset, stream and context, plus allocations during decoding.
    assert resources and all(value == 0 for value in resources.values()), resources
    match = re.search(
        r"cleanup_end released_models=\d+ allocated_device_bytes=(\d+) "
        r"freed_device_bytes=(\d+) errors=0", stderr,
    )
    assert match and match[1] == match[2], stderr


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
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
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
    assert_cpp_resources_released(cleanup, result.stderr)
    if failure == "QWEN35_FAKE_BAD_ACCEPT":
        assert "host acceptance disagrees" in result.stderr


@pytest.mark.parametrize("failed_graph", ["draft", "target_verify"])
def test_cpp_load_failure_reports_memory_before_any_execution(
    chunk_bundle, tmp_path, monkeypatch, failed_graph,
):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    monkeypatch.setenv("QWEN35_FAKE_FAIL_LOAD_GRAPH", failed_graph)
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    events = tmp_path / "execute.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    output = tmp_path / "failed.json"
    result = subprocess.run(
        [runner, "--model", str(plan), "--model-sha256", sha256_file(plan),
         "--model-kind", "chunk", "--mode", "paired", "--prompt-token-ids", "4",
         "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert not output.exists() and not events.exists()
    assert f"aclmdlLoadFromFileWithMem failed: 245000 graph={failed_graph}" in result.stderr
    assert f"phase=load_failed graph={failed_graph}" in result.stderr
    assert "loaded_models=" + ("0" if failed_graph == "draft" else "3") in result.stderr
    # Even an OOM on the first model leaves the full selected set diagnosable.
    before_load = result.stderr.split("[chunk-runtime] load graph=", 1)[0]
    for name in ("draft", "target_decode", "target_prefill", "target_verify"):
        assert f"om-memory graph={name} query_status=0" in before_load
    assert "weight_bytes=5001682944 work_bytes=1048576" in before_load
    assert "pool=DDR query_status=0 free_bytes=unavailable" in result.stderr
    assert_cpp_resources_released(cleanup, result.stderr)


@pytest.mark.parametrize("operation", ["aclmdlUnload", "aclrtFree", "aclrtResetDevice"])
def test_cpp_cleanup_logs_errors_and_continues_teardown(
    chunk_bundle, tmp_path, monkeypatch, operation,
):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    monkeypatch.setenv("QWEN35_FAKE_FAIL_LOAD_GRAPH", "target_verify")
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_FAIL", operation)
    output = tmp_path / "failed.json"
    result = subprocess.run(
        [runner, "--model", str(plan), "--model-sha256", sha256_file(plan),
         "--model-kind", "chunk", "--mode", "paired", "--prompt-token-ids", "4",
         "--output", str(output)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0 and not output.exists()
    assert "aclmdlLoadFromFileWithMem failed: 245000 graph=target_verify" in result.stderr
    assert f"cleanup-error operation={operation} status=38" in result.stderr
    assert re.search(r"cleanup_end .* errors=[1-9]\d*", result.stderr)
    # Reaching fake aclFinalize after an unload/free/reset failure proves that
    # cleanup attempts the remaining operations and preserves the initial error.
    resources = json.loads(cleanup.read_text())
    for name, count in resources.items():
        if name == "live_models" and operation == "aclmdlUnload":
            assert count == 3
        elif name == "live_device_buffers" and operation == "aclmdlUnload":
            assert count == 1  # The fake driver refuses to free borrowed work memory.
        elif name == "live_device_buffers" and operation == "aclrtFree":
            assert count > 0
        else:
            assert count == 0, (name, count)


@pytest.mark.parametrize("unavailable", [False, True])
def test_pure_dflash_does_not_load_or_require_decode_om(
    chunk_bundle, tmp_path, monkeypatch, unavailable,
):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "0")
    if unavailable:
        monkeypatch.setenv("QWEN35_FAKE_MEM_INFO_FAIL", "1")
        monkeypatch.setenv("QWEN35_FAKE_QUERY_SIZE_FAIL", "1")
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
    assert "om-memory graph=target_decode" not in result.stderr
    assert "selected_models=3" in result.stderr
    if unavailable:
        assert "query_status=36 weight_bytes=unavailable work_bytes=unavailable" in result.stderr
        assert "pool=HBM query_status=35 free_bytes=unavailable" in result.stderr
    report = json.loads(output.read_text())
    assert report["ordinary_parity"]["status"] == "NOT_RUN"
    for row in report["benchmark"]["measurements"]:
        assert "target_decode" not in row["stage_ms"]
        assert len(row["stage_ms"]["target_verify"]) == 31
        assert len(row["stage_ms"]["draft"]) == 31
        assert row["counters"]["target_only_fallback_rounds"] == 0
        assert row["counters"]["speculation_disable_events"] == 0
        assert "rounds" not in row


@pytest.mark.parametrize("mode", ["paired", "dflash", "ordinary"])
def test_cpp_discard_buffers_are_private_device_only_and_never_committed(
    chunk_bundle, tmp_path, monkeypatch, mode,
):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, _, c = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt", mode=mode)
    log = tmp_path / "memory.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_MEMORY_LOG", str(log))
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "7")
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    report = tmp_path / "report.json"
    result = subprocess.run(
        [runner, "--model", str(plan), "--model-sha256", sha256_file(plan),
         "--model-kind", "chunk", "--mode", mode, "--prompt-token-ids", "4",
         "--max-new-tokens", "32", "--output", str(report)],
        capture_output=True, text=True,
    )
    # The fake ACL rejects any reuse/transfer/reset of discard buffers and
    # validates committed cursors on every call, across 3+10 repetitions.
    assert result.returncode == 0, result.stderr
    records = [json.loads(line) for line in log.read_text().splitlines()]
    discards = [r for r in records if r[0] == "discard"]
    expected_bytes = 1 * 1 * 16 * 16 * 4  # Tiny fixture, not device evidence
    assert all(r[1] != expected_bytes for r in records if r[0] == "host_alloc")
    if mode == "ordinary":
        assert not discards and "verify_discard_buffer_bytes=0" in result.stderr
    else:
        assert len(discards) > 13  # buffer is reused across rounds and requests
        assert {r[1] for r in discards} == {t["name"] for t in c["verify_discard_states"]}
        assert {r[2] for r in discards} == {expected_bytes}
        assert len({r[3] for r in discards}) == 1
        assert f"verify_discard_buffer_bytes={expected_bytes}" in result.stderr
    payload = json.loads(report.read_text())
    assert payload["status"] == "PASS"
    assert_cpp_resources_released(cleanup, result.stderr)


@pytest.mark.parametrize("low_memory", [False, True])
@pytest.mark.parametrize("operation", ["aclmdlUnload", "aclrtFree", "aclrtResetDevice"])
def test_cpp_cleanup_failure_never_publishes_passing_benchmark(
    chunk_bundle, tmp_path, monkeypatch, low_memory, operation,
):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_FAIL", operation)
    output = tmp_path / "report.json"
    command = [
        runner, "--model", str(plan), "--model-sha256", sha256_file(plan),
        "--model-kind", "chunk", "--prompt-token-ids", "4",
        "--max-new-tokens", "8", "--output", str(output),
    ]
    if low_memory:
        command.append("--low-memory")
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode != 0 and not output.exists()
    assert f"cleanup-error operation={operation} status=38" in result.stderr
    if low_memory and operation != "aclrtResetDevice":
        assert "refusing to load the next mode" in result.stderr
        assert "[chunk-runtime] load graph=draft" not in result.stderr
    else:
        assert "chunk runner cleanup failed" in result.stderr


@pytest.mark.parametrize("damage", ["missing", "dtype", "shape", "input", "order", "stale"])
def test_cpp_rejects_invalid_discard_plan_before_loading(chunk_bundle, tmp_path, monkeypatch, damage):
    from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
    from qwen35_dflash.ascend310p.utils import sha256_file

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt")
    lines = plan.read_text().splitlines()
    index = next(i for i, s in enumerate(lines) if s.startswith("O verify_discard_"))
    if damage == "missing":
        del lines[index]
    elif damage == "dtype":
        lines[index] = lines[index].replace("float32", "float16")
    elif damage == "shape":
        lines[index] = lines[index].replace("4 1 1 16 16", "4 1 1 16 32")
    elif damage == "input":
        lines[index] = "I " + lines[index][2:]
    elif damage == "order":
        lines[index - 1], lines[index] = lines[index], lines[index - 1]
    else:
        lines[0] = "qwen35-dflash-chunk-v2"
    plan.write_text("\n".join(lines) + "\n")
    result = subprocess.run(
        [runner, "--model", str(plan), "--model-sha256", sha256_file(plan),
         "--model-kind", "chunk", "--mode", "dflash", "--prompt-token-ids", "4",
         "--output", str(tmp_path / "report.json")],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "discard" in result.stderr if damage != "stale" else "ABI" in result.stderr
    assert "[chunk-runtime] load graph=" not in result.stderr


@pytest.mark.parametrize("extra,expected", [
    ([], ["--precision_mode=must_keep_origin_dtype"]),
    (["--log=info"], ["--log=info", "--precision_mode=must_keep_origin_dtype"]),
    (["--precision_mode=must_keep_origin_dtype"], ["--precision_mode=must_keep_origin_dtype"]),
    (["--precision_mode_v2=origin"], ["--precision_mode_v2=origin"]),
])
def test_chunk_compilation_preserves_native_precision(extra, expected):
    from qwen35_dflash.ascend310p.compiler import _chunk_precision_args
    assert _chunk_precision_args(extra, incremental=True) == expected
    assert _chunk_precision_args(extra, incremental=False) == extra


@pytest.mark.parametrize("extra", [
    ["--precision_mode=force_fp16"],
    ["--precision_mode=allow_fp32_to_fp16"],
    ["--precision_mode_v2=fp16"],
    ["--precision_mode_v2=origin", "--precision_mode=must_keep_origin_dtype"],
    ["--precision_mode=must_keep_origin_dtype", "--precision_mode=must_keep_origin_dtype"],
])
def test_chunk_compilation_rejects_precision_drift(extra):
    from qwen35_dflash.ascend310p.compiler import _chunk_precision_args
    with pytest.raises(ValueError, match="original graph precision"):
        _chunk_precision_args(extra, incremental=True)


@pytest.mark.parametrize("max_draft", [2, 7, 15])
def test_cpp_passes_request_and_remaining_budget_to_draft(
    chunk_bundle, tmp_path, monkeypatch, max_draft,
):
    from qwen35_dflash.ascend310p.cpp_runtime import run_cpp_pair
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    log = tmp_path / "proposals.log"
    monkeypatch.setenv("QWEN35_FAKE_PROPOSAL_LOG", str(log))
    result = run_cpp_pair(
        deployment_manifest=chunk_bundle, runner=runner,
        runner_options={"device_model":"host-fixture", "cann":"fake",
                        "driver":"fake", "firmware":"fake", "runtime":"fake-acl"},
        prompt_token_ids=[4] * 17, eos_token_ids=[], device_id=0,
        max_new_tokens=20, max_draft_tokens=max_draft,
        raw_output=tmp_path / "report.json", log_output=tmp_path / "runner.log",
        trace_rounds=True,
    )
    observed = [tuple(map(int, line.split())) for line in log.read_text().splitlines()]
    assert observed
    for start, feature_rows, count in observed:
        generated = start + feature_rows - 17 + 1
        assert count == min(max_draft, 20 - generated)
    assert min(count for _, _, count in observed) < max_draft
    assert result["ordinary_parity"]["token_id_mismatches"] == 0
