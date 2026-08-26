#!/usr/bin/env python3
"""Reduced-shape CPU contract check for the persistent HIAI bridge."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.internal_dflash_bridge import (  # noqa: E402
    DFLASH_MAX_VERIFY_TOKENS,
    FEATURE_WIDTH,
    InternalDFlashTarget,
    VOCAB_SIZE,
)


class FakePagedAttention(nn.Module):
    def __init__(self, config: SimpleNamespace) -> None:
        super().__init__()
        self.config = config
        self.kv_max_len = 0
        self.register_buffer(
            "block_table",
            torch.empty((1, 0), dtype=torch.int32),
            persistent=False,
        )

    def _rebuild_block_table(self) -> None:
        self.kv_max_len = int(self.config.kv_cache_max_len)
        blocks = self.kv_max_len // 64
        self.block_table = torch.arange(blocks, dtype=torch.int32).view(1, -1)


class FakeHIAIModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            layer_types=["linear_attention", "full_attention"],
            num_hidden_layers=2,
            linear_num_value_heads=1,
            linear_num_key_heads=1,
            linear_key_head_dim=1,
            linear_value_head_dim=1,
            linear_conv_kernel_dim=2,
            num_key_value_heads=1,
            head_dim=16,
            vocab_size=VOCAB_SIZE,
            pad_token_id=0,
            kv_cache_max_len=64,
        )
        self.embedding = nn.Embedding(VOCAB_SIZE, 2, dtype=torch.float16)
        self.lm_head = nn.Linear(2, VOCAB_SIZE, bias=False, dtype=torch.float16)
        self.paged_attention = FakePagedAttention(self.config)
        self.calls: list[dict[str, object]] = []

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values,
        new_kv_cache_pos: torch.Tensor,
        allQLen,
        output_dflash_features: bool,
        accepted_tokens: torch.Tensor | None,
        **kwargs,
    ):
        del kwargs
        self.calls.append(
            {
                "positions": tuple(int(value) for value in new_kv_cache_pos.tolist()),
                "all_q_len": tuple(int(value) for value in allQLen),
                "accepted": (
                    None
                    if accepted_tokens is None
                    else tuple(int(value) for value in accepted_tokens.tolist())
                ),
            }
        )
        conv_state, recurrent_state = past_key_values[0]
        token_values = input_ids.to(torch.float16)
        if accepted_tokens is None:
            conv_state.add_(token_values.sum())
            recurrent_state.add_(token_values.sum() * 10)
        else:
            selected = int(accepted_tokens[0])
            conv_base = conv_state[:, selected].clone()
            recurrent_base = recurrent_state[:, selected].clone()
            running_conv = conv_base
            running_recurrent = recurrent_base
            for row in range(input_ids.shape[1]):
                value = token_values[:, row].view(1, 1, 1)
                running_conv = running_conv + value
                running_recurrent = running_recurrent + value.view(1, 1, 1, 1) * 10
                conv_state[:, row].copy_(running_conv)
                recurrent_state[:, row].copy_(running_recurrent)

        rows = input_ids.shape[1]
        logits = torch.zeros((1, rows, 1), dtype=torch.float16).expand(
            1,
            rows,
            VOCAB_SIZE,
        )
        if not output_dflash_features:
            return logits
        features = torch.zeros((1, rows, 1), dtype=torch.float16).expand(
            1,
            rows,
            FEATURE_WIDTH,
        )
        return logits, features


class FakeWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model


def main() -> None:
    assert DFLASH_MAX_VERIFY_TOKENS == 16
    model = FakeHIAIModel().eval()
    bridge = InternalDFlashTarget(
        FakeWrapper(model).eval(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        kv_cache_max_len=64,
        rollback_enabled=True,
    ).eval()
    bridge.begin_rollback(torch.tensor([[1, 2]], dtype=torch.long))
    assert bridge.dflash_rollback_audit["persistent_cursor"] == 2
    assert model.calls[-1]["positions"] == (1,)

    try:
        bridge._prepare_rollback_state(17)
    except ValueError as error:
        assert "1..16 rows" in str(error)
    else:
        raise AssertionError("17-row rollback verify block was not rejected")

    bridge.verify_rollback(torch.tensor([[3, 4, 99]], dtype=torch.long))
    state = bridge._persistent_state
    assert state is not None
    assert tuple(state[0][0].shape) == (1, 3, 3, 2)
    assert float(state[0][0][0, 1, 0, 0]) == 10.0
    bridge.commit_rollback(1)
    assert bridge.dflash_rollback_audit["persistent_cursor"] == 4
    assert bridge.dflash_rollback_audit["previous_accepted"] == 1

    # K changes from 2 proposals to 1. The bridge must select old slot 1,
    # rebase to two slots, and overwrite the rejected physical tail at pos 4.
    bridge.verify_rollback(torch.tensor([[5, 6]], dtype=torch.long))
    assert model.calls[-1] == {
        "positions": (4, 5),
        "all_q_len": (6,),
        "accepted": (0,),
    }
    state = bridge._persistent_state
    assert state is not None
    assert tuple(state[0][0].shape) == (1, 2, 3, 2)
    assert float(state[0][0][0, 0, 0, 0]) == 15.0
    bridge.commit_rollback(0)
    audit = bridge.dflash_rollback_audit
    assert audit["persistent_cursor"] == 5
    assert audit["rollback_verify_calls"] == 2
    assert audit["rollback_commit_calls"] == 2
    assert audit["historical_prefix_replay_during_verify"] is False
    bridge.verify_rollback(torch.tensor([[6, 7]], dtype=torch.long))
    bridge.abort_rollback()
    failed_audit = bridge.dflash_rollback_audit
    assert failed_audit["session_invalid"] is True
    assert failed_audit["pending_verify_rows"] is None
    assert bridge._persistent_state is None
    print("PASS: HIAI bridge state-bank selection, rebase, and logical KV cursor")


if __name__ == "__main__":
    main()
