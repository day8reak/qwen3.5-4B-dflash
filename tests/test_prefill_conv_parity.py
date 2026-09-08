"""Production conv-state parity at the static/compact prefill boundary.

CPU semantic tests, not a real-checkpoint or Ascend OM accuracy claim.
"""
from __future__ import annotations

import ast

import pytest
import torch
from torch import nn

from test_dflash_rollback_helpers import SOURCE, load_helpers


HELPERS = load_helpers()
UPDATE = HELPERS["torch_causal_conv1d_update"]
LOGICAL_UPDATE = HELPERS["causal_conv1d_update_logical"]


@pytest.fixture(autouse=True)
def _threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


class _StaticConv(nn.Module):
    def __init__(self, dtype):
        super().__init__()
        self.register_buffer("weight", torch.tensor(
            [[0.25, -0.5, 0.75, 1.0], [1.0, 0.5, -0.25, 0.75]], dtype=dtype
        ))

    def forward(self, hidden, state, effective_length):
        state = state.clone()
        out = LOGICAL_UPDATE(hidden, state, self.weight, None, "silu",
                             effective_length, UPDATE)
        return out, state


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("rows", [1, 3, 4, 17, 63, 64])
@pytest.mark.parametrize("prior_chunks", [0, 1])
def test_partial_prefill_state_and_next_decode_match_compact(dtype, rows, prior_chunks):
    graph = _StaticConv(dtype)
    state = torch.arange(8, dtype=dtype).reshape(1, 2, 4) / 8
    # Nonzero initial history and 64+N multi-chunk prompts must both work.
    for _ in range(prior_chunks):
        full = torch.arange(128, dtype=dtype).reshape(1, 2, 64) / 128
        UPDATE(full, state, graph.weight, None, "silu")
    hidden = torch.arange(2 * rows, dtype=dtype).reshape(1, 2, rows) / 64
    # Adversarial padding: fixing this by simply zeroing feature outputs fails.
    padded = torch.full((1, 2, 64), 7.0, dtype=dtype)
    padded[..., :rows] = hidden
    expected_state = state.clone()
    expected = UPDATE(hidden, expected_state, graph.weight, None, "silu")
    actual, actual_state = graph(padded, state, torch.tensor([rows], dtype=torch.int16))
    torch.testing.assert_close(actual[..., :rows], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)
    # Check multiple subsequent steps; checking only prefill logits missed this.
    for value in (0.5, -1.0, 2.0):
        token = torch.full((1, 2, 1), value, dtype=dtype)
        eager = UPDATE(token, expected_state, graph.weight, None, "silu")
        actual, actual_state = graph(token, actual_state, torch.tensor([1], dtype=torch.int16))
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_logical_commit_preserves_selected_updater_and_batch_lengths(dtype):
    hidden = torch.arange(256, dtype=dtype).reshape(2, 2, 64) / 128
    state = torch.ones(2, 2, 4, dtype=dtype)
    weight = _StaticConv(dtype).weight
    calls = []

    def native_fixture(x, s, w, b, activation):
        calls.append(tuple(x.shape))
        result = UPDATE(x, s, w, b, activation)
        # The wrapper must not replace the chosen updater's physical outputs.
        return result + 1

    actual = LOGICAL_UPDATE(hidden, state, weight, None, "silu",
                            torch.tensor([1, 17], dtype=torch.int16), native_fixture)
    assert calls == [(2, 2, 64)]
    for batch, rows in enumerate((1, 17)):
        ref_state = torch.ones(1, 2, 4, dtype=dtype)
        expected = UPDATE(hidden[batch:batch+1, :, :rows], ref_state, weight, None, "silu")
        torch.testing.assert_close(actual[batch:batch+1, :, :rows], expected + 1, rtol=0, atol=0)
        torch.testing.assert_close(state[batch:batch+1], ref_state, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_static_export_reuses_one_graph_for_effective_lengths(dtype):
    graph = _StaticConv(dtype)
    hidden = torch.arange(128, dtype=dtype).reshape(1, 2, 64) / 64
    state = torch.ones(1, 2, 4, dtype=dtype)
    exported = torch.export.export(graph, (hidden, state, torch.tensor([17], dtype=torch.int16)), strict=True)
    assert not exported.range_constraints
    for rows in (1, 3, 4, 17, 63, 64):
        actual, actual_state = exported.module()(hidden, state, torch.tensor([rows], dtype=torch.int16))
        ref_state = state.clone()
        expected = UPDATE(hidden[..., :rows], ref_state, graph.weight, None, "silu")
        torch.testing.assert_close(actual[..., :rows], expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_state, ref_state, rtol=0, atol=0)


def test_gdn_wires_effective_length_into_conv_update():
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    gdn = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5GatedDeltaNet")
    calls = [node for node in ast.walk(gdn) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "causal_conv1d_update_logical"]
    assert len(calls) == 1
    assert ast.unparse(calls[0].args[-2]) == "gdr_effective_length"
    assert ast.unparse(calls[0].args[-1]) == "self.causal_conv1d_update"


@pytest.mark.parametrize("rows", [1, 3, 4, 17, 63, 64])
@pytest.mark.parametrize("proposal_count", [0, 1, 15])
def test_production_prefill_boundary_carries_logical_state_into_verify(rows, proposal_count):
    from qwen35_dflash.ascend310p.incremental_graphs import (
        TargetPrefillStateGraph, TargetVerifyCommitStateGraph,
    )
    from test_incremental_om_graphs import _FakeLanguageModel, _FakeTarget, _target_state, _eos

    class ConvLanguage(_FakeLanguageModel):
        def forward(self, *, past_key_values, inputs_embeds, accepted_tokens,
                    output_dflash_features, gdr_effective_length, **kwargs):
            conv, recurrent = past_key_values[0]
            x = inputs_embeds[..., :2].transpose(1, 2)
            weight = torch.tensor([[0.25, 0.5], [-0.5, 1.0]], dtype=x.dtype)
            if accepted_tokens is None:
                if x.shape[-1] == 64:
                    out = LOGICAL_UPDATE(x, conv, weight, None, "silu", gdr_effective_length, UPDATE)
                else:
                    out = UPDATE(x, conv, weight, None, "silu")
            else:
                out, conv = HELPERS["torch_dflash_causal_conv1d_mtp"](
                    x, conv, weight, None, accepted_tokens, "silu")
                recurrent = recurrent.unsqueeze(1).repeat(1, x.shape[-1], 1, 1, 1)
            past_key_values[0] = (conv, recurrent)
            hidden = torch.cat((out.transpose(1, 2), inputs_embeds[..., 2:]), -1)
            return (hidden, hidden * 2) if output_dflash_features else hidden

    target = _FakeTarget().eval()
    target.dflash_execution_model.language_model = ConvLanguage()
    prefill = TargetPrefillStateGraph(target, kv_cache_max_len=128)
    state = _target_state()
    state[0].fill_(0.75)
    before = state[0].clone()
    ids = torch.arange(64).reshape(1, 64) % 7
    length = torch.tensor([rows], dtype=torch.int16)
    actual = prefill(ids, length, *state)
    compact = prefill(ids[:, :rows].contiguous(), length, *state)
    torch.testing.assert_close(state[0], before, rtol=0, atol=0)
    for index in (0, 2, 3, 4, 5, 6, 7):
        torch.testing.assert_close(actual[index], compact[index], rtol=0, atol=0)
    torch.testing.assert_close(actual[1][:, :rows], compact[1], rtol=0, atol=0)
    assert torch.count_nonzero(actual[1][:, rows:]) == 0
    verify = TargetVerifyCommitStateGraph(target, kv_cache_max_len=128)
    block = torch.arange(16).reshape(1, 16) % 7
    control = (block, torch.tensor([proposal_count], dtype=torch.int32), *_eos())
    for _ in range(3):
        got = verify(*control, *actual[3:])
        ref = verify(*control, *compact[3:])
        for value, expected in zip(got, ref):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        if proposal_count == 0:
            assert got[1].item() == 1
            assert got[3].item() == got[4].item() == 0
            assert got[9].item() == actual[7].item() + 1
            assert got[6].dtype == torch.float32
        # Reconstruct the public committed state, not a provisional bank.
        actual = (None, None, None, got[5], got[6], got[11], got[12], got[9])
        compact = (None, None, None, ref[5], ref[6], ref[11], ref[12], ref[9])
