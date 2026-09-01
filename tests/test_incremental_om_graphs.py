from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from qwen35_dflash.ascend310p.incremental_graphs import (
    DraftProposeStateGraph,
    TargetDecodeOneStateGraph,
    TargetPrefillHeadGraph,
    TargetPrefillStateGraph,
    TargetStepStateGraph,
    TargetVerifyCommitStateGraph,
    incremental_state_graph_specs,
)


class _TokenEmbedding(nn.Module):
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        first = (input_ids + 1).to(torch.float16).unsqueeze(-1)
        return torch.cat((first, torch.zeros_like(first).expand(-1, -1, 3)), dim=-1)


class _TokenHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()), requires_grad=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        token = (hidden[..., 0] * self.scale).to(torch.long).clamp(0, 127)
        return torch.nn.functional.one_hot(token, num_classes=128).to(torch.float32)


class _FakeLanguageModel(nn.Module):
    dflash_scalar_state_seed_policy = "per-linear-layer-jit-v1"
    dflash_cache_index_policy = "once-per-verify-v1"

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        inputs_embeds: torch.Tensor,
        output_dflash_features: bool,
        accepted_tokens: torch.Tensor | None,
        **kwargs: object,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        del input_ids
        target_blocks = kwargs.pop("dflash_cache_target_blocks", None)
        offsets = kwargs.pop("dflash_cache_offsets", None)
        del kwargs
        conv, recurrent = past_key_values[0]
        cache_index_zero: torch.Tensor | int = 0
        if accepted_tokens is None:
            assert target_blocks is None
            assert offsets is None
            past_key_values[0] = (conv + 1, recurrent + 1)
        else:
            # The graph boundary must pass committed scalar states.  The real
            # rollback GDN expands them one linear layer at a time.
            assert conv.ndim == 3
            assert recurrent.ndim == 4
            assert target_blocks is not None
            assert offsets is not None
            assert target_blocks.dtype == torch.int32
            assert offsets.dtype == torch.int32
            assert target_blocks.shape == offsets.shape == (inputs_embeds.shape[1],)
            cache_index_zero = (target_blocks.sum() + offsets.sum()).to(
                inputs_embeds.dtype
            ) * 0
            slots = torch.arange(
                inputs_embeds.shape[1], dtype=conv.dtype, device=conv.device
            ).reshape(1, -1, 1, 1)
            recurrent_slots = slots.reshape(1, -1, 1, 1, 1)
            past_key_values[0] = (
                conv.unsqueeze(1) + slots,
                recurrent.unsqueeze(1) + recurrent_slots,
            )
        key, value = past_key_values[1]
        past_key_values[1] = (
            key + 1 + cache_index_zero,
            value + 1 + cache_index_zero,
        )
        if output_dflash_features:
            return inputs_embeds, inputs_embeds * 2
        return inputs_embeds


class _FakeExecution(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.language_model = _FakeLanguageModel()
        self.lm_head = _TokenHead()
        self.config = SimpleNamespace(
            layer_types=("linear_attention", "full_attention"),
            linear_num_key_heads=1,
            linear_num_value_heads=1,
            linear_key_head_dim=2,
            linear_value_head_dim=2,
            linear_conv_kernel_dim=2,
            num_key_value_heads=1,
            head_dim=16,
            hidden_size=4,
        )


class _FakeTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dflash_execution_model = _FakeExecution()
        self._target_quantized_embedding = _TokenEmbedding()
        self.draft_input_embedding = nn.Embedding(128, 2, dtype=torch.float16)
        self.draft_output_embedding = nn.Linear(
            2, 128, bias=False, dtype=torch.float16
        )

    @property
    def config(self) -> object:
        return self.dflash_execution_model.config

    def get_input_embeddings(self) -> nn.Module:
        return self.draft_input_embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.draft_output_embedding


def _target_state() -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(1, 1, 2, 2, dtype=torch.float16),
        torch.zeros(1, 1, 1, 2, 2, dtype=torch.float32),
        torch.zeros(1, 1, 2, 2, dtype=torch.float16),
        torch.zeros(1, 1, 2, 2, dtype=torch.float16),
        torch.tensor([0], dtype=torch.long),
    )


def _eos() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.tensor([127, 0, 0, 0]), torch.tensor([1], dtype=torch.int32)


def test_target_prefill_decode_and_verify_have_explicit_scalar_state() -> None:
    target = _FakeTarget().eval()
    conv, recurrent, key, value, cursor = _target_state()
    eos_ids, eos_count = _eos()
    prefill = TargetPrefillStateGraph(target, kv_cache_max_len=64).eval()
    prompt = torch.zeros(1, 64, dtype=torch.long)
    prompt[:, :3] = torch.tensor([1, 2, 3])
    prefill_output = prefill(
        prompt,
        torch.tensor([3], dtype=torch.int16),
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    assert prefill_output[0][0, 0, 0].item() == 4
    assert torch.count_nonzero(prefill_output[1][:, 3:]) == 0
    assert prefill_output[2].tolist() == [3]
    assert prefill_output[7].tolist() == [3]
    assert prefill_output[3].shape == conv.shape
    assert prefill_output[4].dtype == torch.float32

    head = TargetPrefillHeadGraph(target).eval()
    head_output = head(prefill_output[0], eos_ids, eos_count)
    assert head_output[0][0, 0].item() == 4
    assert head_output[1].tolist() == [1]
    assert head_output[2].tolist() == [False]

    decode = TargetDecodeOneStateGraph(target, kv_cache_max_len=64).eval()
    decoded = decode(
        torch.tensor([[4]]),
        eos_ids,
        eos_count,
        prefill_output[3],
        prefill_output[4],
        prefill_output[5],
        prefill_output[6],
        prefill_output[7],
    )
    assert decoded[0][0, 0].item() == 5
    assert decoded[7].tolist() == [4]

    verify = TargetVerifyCommitStateGraph(target, kv_cache_max_len=64).eval()
    block = torch.arange(10, 26, dtype=torch.long).reshape(1, 16)
    verified = verify(
        block,
        torch.tensor([15], dtype=torch.int32),
        eos_ids,
        eos_count,
        prefill_output[3],
        prefill_output[4],
        prefill_output[5],
        prefill_output[6],
        prefill_output[7],
    )
    # Fake Target predicts input+1, so all 15 proposals match and a bonus is
    # emitted. Only scalar slot 15 leaves the graph.
    assert verified[0][0].tolist() == list(range(11, 27))
    assert verified[1].tolist() == [16]
    assert verified[3].tolist() == [15]
    assert verified[5].shape == conv.shape
    assert verified[6].shape == recurrent.shape
    assert verified[8].tolist() == [16]
    assert verified[9].tolist() == [19]
    assert verified[10].tolist() == [False]


def test_target_graphs_are_torch_export_capture_safe() -> None:
    target = _FakeTarget().eval()
    conv, recurrent, key, value, cursor = _target_state()
    eos_ids, eos_count = _eos()
    graph = TargetVerifyCommitStateGraph(target, kv_cache_max_len=64).eval()
    args = (
        torch.arange(10, 26, dtype=torch.long).reshape(1, 16),
        torch.tensor([7], dtype=torch.int32),
        eos_ids,
        eos_count,
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    exported = torch.export.export(graph, args, strict=True)
    eager = graph(*args)
    captured = exported.module()(*args)
    assert len(eager) == len(captured) == 13
    for actual, expected in zip(captured, eager):
        torch.testing.assert_close(actual, expected)
    targets = [str(node.target) for node in exported.graph.nodes]
    assert sum("floor_divide" in target for target in targets) == 1
    assert sum("remainder" in target for target in targets) == 1


def test_dynamic_target_step_matches_decode_and_fixed_verify() -> None:
    target = _FakeTarget().eval()
    conv, recurrent, key, value, cursor = _target_state()
    eos_ids, eos_count = _eos()
    step = TargetStepStateGraph(target, kv_cache_max_len=64).eval()
    decode = TargetDecodeOneStateGraph(target, kv_cache_max_len=64).eval()
    fixed = TargetVerifyCommitStateGraph(target, kv_cache_max_len=64).eval()

    one = torch.tensor([[10]], dtype=torch.long)
    step_one = step(
        one,
        torch.tensor([0], dtype=torch.int32),
        eos_ids,
        eos_count,
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    decode_one = decode(
        one, eos_ids, eos_count, conv, recurrent, key, value, cursor
    )
    assert step_one[0][0, 0].item() == decode_one[0][0, 0].item() == 11
    assert step_one[1].tolist() == decode_one[1].tolist() == [1]
    assert step_one[2:5][0].tolist() == [0]
    assert step_one[3].tolist() == step_one[4].tolist() == [0]
    assert step_one[5].shape == conv.shape
    assert step_one[6].shape == recurrent.shape
    assert step_one[7].shape == (1, 16, 4)
    assert step_one[8].tolist() == [1]
    assert step_one[9].tolist() == [1]

    block = torch.arange(10, 26, dtype=torch.long).reshape(1, 16)
    dynamic_full = step(
        block,
        torch.tensor([15], dtype=torch.int32),
        eos_ids,
        eos_count,
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    fixed_full = fixed(
        block,
        torch.tensor([15], dtype=torch.int32),
        eos_ids,
        eos_count,
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    for actual, expected in zip(dynamic_full, fixed_full):
        torch.testing.assert_close(actual, expected)


def test_dynamic_target_step_torch_export_covers_t1_to_t16() -> None:
    target = _FakeTarget().eval()
    conv, recurrent, key, value, cursor = _target_state()
    eos_ids, eos_count = _eos()
    graph = TargetStepStateGraph(target, kv_cache_max_len=64).eval()
    args = (
        torch.arange(10, 26, dtype=torch.long).reshape(1, 16),
        torch.tensor([15], dtype=torch.int32),
        eos_ids,
        eos_count,
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    rows = torch.export.Dim("target_step_rows", min=1, max=16)
    dynamic_shapes = ({1: rows},) + (None,) * (len(args) - 1)
    exported = torch.export.export(
        graph, args, dynamic_shapes=dynamic_shapes, strict=True
    )
    captured = exported.module()
    for physical_rows in (1, 2, 7, 16):
        call_args = (
            args[0][:, :physical_rows],
            torch.tensor([physical_rows - 1], dtype=torch.int32),
            *args[2:],
        )
        eager = graph(*call_args)
        actual = captured(*call_args)
        assert len(actual) == 13
        for captured_value, eager_value in zip(actual, eager):
            torch.testing.assert_close(captured_value, eager_value)


def test_prefill_body_and_head_export_without_cross_retaining_weights() -> None:
    target = _FakeTarget().eval()
    conv, recurrent, key, value, cursor = _target_state()
    eos_ids, eos_count = _eos()
    body = TargetPrefillStateGraph(target, kv_cache_max_len=64).eval()
    body_args = (
        torch.zeros((1, 64), dtype=torch.long),
        torch.tensor([3], dtype=torch.int16),
        conv,
        recurrent,
        key,
        value,
        cursor,
    )
    body_export = torch.export.export(body, body_args, strict=True)
    body_targets = {
        str(item.target)
        for item in body_export.graph_signature.input_specs
        if item.target is not None
    }
    assert not any("lm_head" in item for item in body_targets)

    body_output = body(*body_args)
    head = TargetPrefillHeadGraph(target).eval()
    head_export = torch.export.export(
        head,
        (body_output[0], eos_ids, eos_count),
        strict=True,
    )
    head_targets = {
        str(item.target)
        for item in head_export.graph_signature.input_specs
        if item.target is not None
    }
    assert head_targets
    assert all("lm_head" in item for item in head_targets)


class _FakeDraftOps:
    @staticmethod
    def top1(hidden: torch.Tensor, lm_head_weight: torch.Tensor) -> torch.Tensor:
        del lm_head_weight
        return hidden[..., 0].to(torch.long)


class _FakeDraftLayer(nn.Module):
    def __init__(self, layer_index: int) -> None:
        super().__init__()
        self.layer_index = layer_index

    def forward_cached(
        self,
        hidden: torch.Tensor,
        projected: torch.Tensor,
        cosine: torch.Tensor,
        sine: torch.Tensor,
        cache: object,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        del cosine, sine, attention_mask
        combined = torch.cat((projected, hidden), dim=1).unsqueeze(1)
        cache.update(self.layer_index, combined, combined + 1)
        return hidden


class _FakeDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            block_size=16,
            mask_token_id=99,
            num_key_value_heads=1,
            head_dim=2,
            feature_size=4,
        )
        self.layers = nn.ModuleList([_FakeDraftLayer(0), _FakeDraftLayer(1)])
        self.norm = nn.Identity()
        self.ops = _FakeDraftOps()

    @staticmethod
    def project_target_hidden(features: torch.Tensor) -> torch.Tensor:
        return features[..., :2]

    @staticmethod
    def embed_block(ids: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.embedding(ids, weight)

    @staticmethod
    def rotary(positions: torch.Tensor, dtype: torch.dtype):
        carrier = positions.to(dtype).unsqueeze(-1).expand(-1, -1, 2)
        return carrier, carrier


def test_draft_graph_keeps_kv_fixed_and_returns_device_verify_carrier() -> None:
    draft = _FakeDraft().eval()
    input_embedding = nn.Embedding(128, 2, dtype=torch.float16)
    output_embedding = nn.Linear(2, 128, bias=False, dtype=torch.float16)
    with torch.no_grad():
        input_embedding.weight[:, 0] = torch.arange(128)
        input_embedding.weight[:, 1] = 0
    graph = DraftProposeStateGraph(
        draft,
        input_embedding,
        output_embedding,
        kv_cache_max_len=32,
    ).eval()
    features = torch.arange(16, dtype=torch.float16).reshape(1, 4, 4)
    args = (
        features,
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([[40, 41] + [0] * 14], dtype=torch.long),
        torch.tensor([2], dtype=torch.int32),
        torch.tensor([5], dtype=torch.int32),
        torch.zeros(2, 1, 1, 32, 2, dtype=torch.float16),
        torch.zeros(2, 1, 1, 32, 2, dtype=torch.float16),
        torch.tensor([3], dtype=torch.long),
    )
    verify_ids, next_key, next_value, next_cursor = graph(*args)
    assert verify_ids.shape == (1, 16)
    assert verify_ids[0, 0].item() == 41
    assert verify_ids[0, 1:].tolist() == [99] * 15
    assert next_key.shape == args[5].shape
    assert next_value.shape == args[6].shape
    assert next_cursor.tolist() == [5]
    assert torch.count_nonzero(next_key[..., 3:7, :]) > 0

    exported = torch.export.export(graph, args, strict=True)
    captured = exported.module()(*args)
    for actual, expected in zip(captured, (verify_ids, next_key, next_value, next_cursor)):
        torch.testing.assert_close(actual, expected)


def test_draft_graph_batched_prompt_features_match_sequential_chunk_state() -> None:
    draft = _FakeDraft().eval()
    input_embedding = nn.Embedding(128, 2, dtype=torch.float16)
    output_embedding = nn.Linear(2, 128, bias=False, dtype=torch.float16)
    with torch.no_grad():
        input_embedding.weight[:, 0] = torch.arange(128)
        input_embedding.weight[:, 1] = 0
    graph = DraftProposeStateGraph(
        draft,
        input_embedding,
        output_embedding,
        kv_cache_max_len=128,
    ).eval()
    first = torch.arange(256, dtype=torch.float16).reshape(1, 64, 4)
    final = torch.zeros((1, 64, 4), dtype=torch.float16)
    final[:, :6, :] = torch.arange(24, dtype=torch.float16).reshape(1, 6, 4)
    key = torch.zeros(2, 1, 1, 128, 2, dtype=torch.float16)
    value = torch.zeros_like(key)
    proposal_count = torch.tensor([5], dtype=torch.int32)

    first_result = graph(
        first,
        torch.tensor([64], dtype=torch.int32),
        torch.tensor([[40] + [0] * 15], dtype=torch.long),
        torch.tensor([1], dtype=torch.int32),
        proposal_count,
        key,
        value,
        torch.tensor([0], dtype=torch.long),
    )
    sequential = graph(
        final,
        torch.tensor([6], dtype=torch.int32),
        torch.tensor([[50] + [0] * 15], dtype=torch.long),
        torch.tensor([1], dtype=torch.int32),
        proposal_count,
        first_result[1],
        first_result[2],
        first_result[3],
    )
    batched = graph(
        torch.cat((first, final), dim=1),
        torch.tensor([70], dtype=torch.int32),
        torch.tensor([[50] + [0] * 15], dtype=torch.long),
        torch.tensor([1], dtype=torch.int32),
        proposal_count,
        key,
        value,
        torch.tensor([0], dtype=torch.long),
    )
    for actual, expected in zip(batched, sequential):
        torch.testing.assert_close(actual, expected)


def test_five_physical_specs_freeze_binding_order_and_reuse_state_examples() -> None:
    target = _FakeTarget().eval()
    draft = _FakeDraft().eval()
    specs = incremental_state_graph_specs(
        target,
        draft,
        kv_cache_max_len=64,
        device="cpu",
        dtype=torch.float16,
        eos_table_width=4,
        ordinary_custom_ops=(),
        head_custom_ops=(),
        verify_custom_ops=(),
        metadata={"test_identity": "reduced"},
    )
    assert [item.name for item in specs] == [
        "target-prefill",
        "target-prefill-head",
        "target-decode1",
        "draft-propose",
        "target-verify-commit",
    ]
    assert [item.role for item in specs] == [item.name for item in specs]
    assert specs[3].dynamic is True
    assert specs[3].input_dim_gears == {0: {1: (16, 64)}}
    assert specs[0].example_args[3].dtype == torch.float32
    assert specs[0].example_args[4].shape == (1, 1, 1, 64, 16)
    assert specs[3].example_args[5].shape == (2, 1, 1, 64, 2)
    assert specs[4].input_names[0:2] == (
        "verify_input_ids",
        "logical_proposal_count",
    )
    assert specs[4].output_names[0:5] == (
        "committed_token_ids",
        "commit_count",
        "drafted_count",
        "accepted_count",
        "rejected_count",
    )
    assert all(
        item.metadata["status"] == "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE"
        for item in specs
    )
    assert all(
        item.metadata["verify_cache_index_policy"] == "once-per-verify-v1"
        for item in specs
    )


def test_draft_air_gears_cover_every_prompt_batch_capacity() -> None:
    specs = incremental_state_graph_specs(
        _FakeTarget().eval(),
        _FakeDraft().eval(),
        kv_cache_max_len=256,
        device="cpu",
        dtype=torch.float16,
        eos_table_width=4,
        ordinary_custom_ops=(),
        head_custom_ops=(),
        verify_custom_ops=(),
    )
    assert specs[3].input_dim_gears == {
        0: {1: (16, 64, 128, 192, 256)}
    }


def test_four_physical_specs_use_dynamic_unified_target_step() -> None:
    specs = incremental_state_graph_specs(
        _FakeTarget().eval(),
        _FakeDraft().eval(),
        kv_cache_max_len=64,
        device="cpu",
        dtype=torch.float16,
        eos_table_width=4,
        ordinary_custom_ops=(),
        head_custom_ops=(),
        verify_custom_ops=(),
        unified_target_step=True,
    )
    assert [item.name for item in specs] == [
        "target-prefill",
        "target-prefill-head",
        "draft-propose",
        "target-verify-commit",
    ]
    target_step = specs[-1]
    assert isinstance(target_step.model, TargetStepStateGraph)
    assert target_step.dynamic is True
    assert target_step.input_dim_gears == {
        0: {1: tuple(range(1, 17))}
    }
    assert target_step.metadata["target_step_fixed_output_rows"] == 16
