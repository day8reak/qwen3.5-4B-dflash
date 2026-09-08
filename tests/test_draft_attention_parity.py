"""Independent compact eager vs fixed-carrier Draft semantics, without NPU claims.

Use the production DFlash model and the official five-sliding/one-full layer
mix with reduced random weights. Comparing two copies of the export wrapper
would miss a shared mask bug; the oracle must construct its own per-layer mask.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1 import dflash_ascend310p_ops
from models.dflash_v1.dflash_ops import ModuleDFlashOps
from models.dflash_v1.modeling_dflash import DFlashDraftModel
from qwen35_dflash.ascend310p.incremental_graphs import DraftProposeStateGraph
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps


class _TraceOps(AirDFlashOps):
    def __init__(self):
        self.clear()

    def clear(self):
        self.attention_outputs = []
        self.masks = []
        self.head_input = None

    def attention(self, query, key, value, attention_mask, scale, groups):
        result = super().attention(query, key, value, attention_mask, scale, groups)
        self.attention_outputs.append(result.detach().clone())
        self.masks.append(None if attention_mask is None else attention_mask.clone())
        return result

    def top1(self, hidden, weight):
        self.head_input = hidden.detach().clone()
        return super().top1(hidden, weight)


@pytest.fixture(autouse=True)
def _small_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.random.fork_rng(devices=[]):
            yield
    finally:
        torch.set_num_threads(previous)


def _model(*, window=4096, dtype=torch.float32, trace=True):
    # Seed 4 exposes proposal-ID differences with the old all-causal wrapper,
    # not just a hidden-state tolerance failure.
    torch.manual_seed(4)
    config = Qwen35DFlashConfig.from_dict({
        "hidden_size": 32, "intermediate_size": 64, "vocab_size": 96,
        "num_hidden_layers": 6, "num_attention_heads": 4,
        "num_key_value_heads": 2, "head_dim": 8, "num_target_layers": 32,
        "layer_types": ["sliding_attention"] * 5 + ["full_attention"],
        "use_sliding_window": True, "sliding_window": window,
        "max_position_embeddings": 4096,
        "rope_parameters": {"rope_theta": 10000000.0},
        "dflash_config": {"block_size": 16, "mask_token_id": 95,
                          "target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29]},
    })
    ops = _TraceOps() if trace else AirDFlashOps()
    draft = DFlashDraftModel(config, ops=ops, dtype=dtype).eval()
    with torch.no_grad():
        for parameter in draft.parameters():
            if parameter.ndim == 1:
                parameter.fill_(1.0)
            else:
                parameter.normal_(std=0.08)
    embedding = nn.Embedding(96, 32, dtype=dtype).eval()
    output = nn.Linear(32, 96, bias=False, dtype=dtype).eval()
    graph = DraftProposeStateGraph(draft, embedding, output, kv_cache_max_len=128).eval()
    return graph


def _args(graph, count=17, proposals=15, previous_count=1):
    dtype = graph.input_embedding.weight.dtype
    features = torch.randn(1, count, graph.draft.config.feature_size, dtype=dtype)
    padded = torch.zeros(1, 64, features.shape[-1], dtype=dtype)
    padded[:, :count] = features
    previous_ids = torch.arange(16, dtype=torch.int64).view(1, 16) + 3
    cache_shape = (6, 1, 2, 128, 8)
    return (
        padded, torch.tensor([count], dtype=torch.int32), previous_ids,
        torch.tensor([previous_count], dtype=torch.int32),
        torch.tensor([proposals], dtype=torch.int32),
        torch.zeros(cache_shape, dtype=dtype), torch.zeros(cache_shape, dtype=dtype),
        torch.tensor([0], dtype=torch.int64),
    )


def _eager_round(graph, args, cache):
    count, proposals, previous_count = args[1].item(), args[4].item(), args[3].item()
    anchor = args[2][:, previous_count - 1:previous_count]
    ids = torch.cat((anchor, torch.full((1, proposals), 95, dtype=torch.int64)), dim=1)
    positions = torch.arange(cache.committed_length,
                             cache.committed_length + count + proposals + 1).view(1, -1)
    # No attention_mask argument: production eager builds it independently.
    hidden = graph.draft.forward_cached(
        args[0][:, :count].contiguous(),
        graph.draft.embed_block(ids, graph.input_embedding.weight), positions, cache,
    )
    return graph.draft.ops.top1(hidden[:, 1:], graph.output_embedding.weight), hidden


@pytest.mark.parametrize("proposals", [1, 3, 7, 15])
@pytest.mark.parametrize("window", [1, 8, 4096])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fixed_draft_matches_compact_eager_per_layer_and_live_kv(proposals, window, dtype):
    graph = _model(window=window, dtype=dtype)
    ops = graph.draft.ops
    cache = graph.draft.new_kv_cache(max_length=128)
    next_state = None
    # Initial prompt followed by zero/partial/all-accept sized committed tails.
    # One cache crosses the sliding boundary and repeatedly overwrites scratch.
    for count, previous_count in ((17, 1), (1, 1), (4, 4), (16, 16)):
        args = list(_args(graph, count, proposals, previous_count))
        if next_state is not None:
            args[5:] = next_state
        ops.clear()
        with torch.no_grad():
            expected_ids, expected_hidden = _eager_round(graph, args, cache)
            expected_outputs, expected_masks = ops.attention_outputs, ops.masks
            ops.clear()
            actual = graph(*args)
        assert torch.equal(actual[0][:, 1:proposals + 1], expected_ids)
        assert actual[0][0, 0] == args[2][0, previous_count - 1]
        assert actual[3].item() == cache.committed_length
        tolerance = 0.004 if dtype == torch.float16 else 3e-6
        torch.testing.assert_close(ops.head_input[:, :proposals], expected_hidden[:, 1:],
                                   atol=tolerance, rtol=tolerance)
        assert torch.isfinite(ops.head_input).all(), "padded queries must not produce NaNs"
        for i, (expected_output, expected_mask) in enumerate(zip(expected_outputs, expected_masks)):
            torch.testing.assert_close(ops.attention_outputs[i][..., :proposals + 1, :],
                                       expected_output, atol=tolerance, rtol=tolerance)
            mask = ops.masks[i]
            logical_mask = torch.cat((mask[..., :proposals + 1, :cache.committed_length],
                                      mask[..., :proposals + 1, 128:128 + proposals + 1]), dim=-1)
            reference = torch.ones_like(logical_mask) if expected_mask is None else expected_mask
            assert torch.equal(logical_mask, reference), f"wrong layer {i} attention policy"
            assert not mask[..., cache.committed_length:128].any()
            assert not mask[..., 128 + proposals + 1:].any()
            assert mask.any(dim=-1).all(), "every physical query needs a visible live key"
            torch.testing.assert_close(actual[1][i, ..., :cache.committed_length, :],
                                       cache._keys[i], atol=tolerance, rtol=tolerance)
            torch.testing.assert_close(actual[2][i, ..., :cache.committed_length, :],
                                       cache._values[i], atol=tolerance, rtol=tolerance)
        next_state = list(actual[1:])


def test_full_layer_sees_future_live_proposals_not_physical_padding():
    graph = _model()
    args = _args(graph, proposals=3)
    with torch.no_grad():
        graph(*args)
    masks = graph.draft.ops.masks
    assert not masks[0][0, 0, 0, 129]  # Causal layer: anchor cannot see proposal 0.
    assert masks[-1][0, 0, 0, 129:132].all()  # Full layer sees all three proposals.
    assert not masks[-1][..., 132:].any()  # No influence from twelve padded masks.


def test_static_real_draft_export_keeps_counts_live_and_matches_eager():
    graph = _model(trace=False)
    args = _args(graph)
    captured = torch.export.export(graph, args, strict=True)
    assert not captured.range_constraints
    assert all(not isinstance(n.meta.get("val"), torch.SymInt)
               for n in captured.graph.nodes if n.op == "placeholder")
    exported = captured.module()
    for count, proposals in ((1, 1), (17, 3), (64, 15)):
        current = _args(graph, count=count, proposals=proposals)
        with torch.no_grad():
            expected, _ = _eager_round(graph, current, graph.draft.new_kv_cache(max_length=128))
            actual = exported(*current)
        assert actual[0].shape == (1, 16)
        assert actual[3].item() == count
        assert torch.equal(actual[0][:, 1:proposals + 1], expected)


@pytest.mark.parametrize("causal,window", [(True, None), (True, 1), (False, None), (False, 2)])
@pytest.mark.parametrize("proposals", [0, 1, 15])
def test_effective_layer_policy_matches_eager_mask_at_cache_boundary(causal, window, proposals):
    graph = _model()
    attention = graph.draft.layers[0].self_attn
    # Effective policy, not config.layer_types or an assumed final-layer index.
    attention.is_causal, attention.sliding_window = causal, window
    graph = DraftProposeStateGraph(graph.draft, graph.input_embedding, graph.output_embedding,
                                  kv_cache_max_len=128)
    assert graph.attention_policies[0] == (causal, window)
    for context in (0, 1, 112):
        actual = graph._attention_mask(
            torch.tensor([context]), torch.tensor([0]), torch.tensor([proposals]),
            is_causal=causal, sliding_window=window, device=torch.device("cpu"),
        )
        expected = attention._attention_mask(proposals + 1, context, device=torch.device("cpu"))
        logical = torch.cat((actual[..., :proposals + 1, :context],
                             actual[..., :proposals + 1, 128:128 + proposals + 1]), dim=-1)
        assert torch.equal(logical, torch.ones_like(logical) if expected is None else expected)
        assert not actual[..., context:128].any()
        assert not actual[..., 128 + proposals + 1:].any()
        assert actual.any(dim=-1).all()


@pytest.mark.parametrize("damage", ["missing", "causal", "window", "bool-window"])
def test_unknown_layer_policy_is_rejected_instead_of_assumed_causal(damage):
    graph = _model()
    attention = graph.draft.layers[0].self_attn
    if damage == "missing":
        del attention.sliding_window
    elif damage == "causal":
        attention.is_causal = 1
    else:
        attention.sliding_window = True if damage == "bool-window" else 0
    with pytest.raises(ValueError, match="Draft layer 0"):
        DraftProposeStateGraph(graph.draft, graph.input_embedding, graph.output_embedding,
                               kv_cache_max_len=128)


def test_uncommitted_feature_and_cache_scratch_cannot_change_live_proposals():
    graph = _model()
    args = list(_args(graph, count=3, proposals=3))
    with torch.no_grad():
        expected = graph(*args)
        args[0][:, 3:] = 10.0
        args[5].fill_(10.0)
        args[6].fill_(-10.0)
        actual = graph(*args)
    assert torch.equal(actual[0][:, :4], expected[0][:, :4])
    for index in (1, 2):
        torch.testing.assert_close(actual[index][..., :3, :], expected[index][..., :3, :])


@pytest.mark.parametrize("proposals", [1, 3, 15])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_fixed_air_draft_matches_current_strict_npu_ops_in_cpu_simulation(proposals, dtype):
    graph = _model(dtype=dtype, trace=False)
    args = _args(graph, proposals=proposals)
    with torch.no_grad():
        # Execute the actual current-branch NPU primitive module, on CPU, with
        # fallback disabled. This is NOT a torch_npu/OM/device execution claim.
        graph.draft.set_ops(ModuleDFlashOps(dflash_ascend310p_ops, strict=True))
        expected, _ = _eager_round(graph, args, graph.draft.new_kv_cache(max_length=128))
        graph.draft.set_ops(AirDFlashOps())
        actual = graph(*args)
    assert torch.equal(actual[0][:, 1:proposals + 1], expected)
