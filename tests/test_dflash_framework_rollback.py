#!/usr/bin/env python3
"""CPU check for DynamicCache rollback and bounded commit replay."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn
from transformers import PreTrainedConfig


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.dflash_v1.dflash_rollback_adapter import (  # noqa: E402
    FrameworkDFlashRollbackTarget,
)


class TinyHybridConfig(PreTrainedConfig):
    model_type = "tiny_hybrid_rollback_test"

    def __init__(self) -> None:
        super().__init__()
        self.num_hidden_layers = 2
        self.layer_types = ["full_attention", "linear_attention"]


class TinyHybridTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = TinyHybridConfig()
        self.embedding = nn.Embedding(128, 2)
        self.head = nn.Linear(2, 128, bias=False)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.head

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values,
        output_dflash_features: bool,
        **kwargs,
    ):
        del kwargs
        values = input_ids.to(torch.float32)
        kv = values.view(1, 1, -1, 1)
        past_key_values.update(kv, -kv, 0)

        linear = past_key_values.layers[1]
        delta = values.sum().view(1, 1, 1)
        if linear.conv_states is None:
            linear.conv_states = torch.zeros_like(delta)
            linear.recurrent_states = torch.zeros((1, 1, 1, 1))
            linear.is_conv_states_initialized = True
            linear.is_recurrent_states_initialized = True
        linear.conv_states.add_(delta)
        linear.recurrent_states.add_(delta.view(1, 1, 1, 1) * 10)
        linear.has_previous_state = True

        token_ids = ((input_ids + 1) % 128).to(torch.long)
        logits = torch.full((1, input_ids.shape[1], 128), -10.0)
        logits.scatter_(2, token_ids.unsqueeze(-1), 10.0)
        result = {
            "logits": logits,
            "past_key_values": past_key_values,
        }
        if output_dflash_features:
            result["dflash_features"] = values.unsqueeze(-1).repeat(1, 1, 3)
        return result


def main() -> None:
    target = TinyHybridTarget().eval()
    controller = FrameworkDFlashRollbackTarget(target).eval()
    controller.begin_rollback(torch.tensor([[1, 2]], dtype=torch.long))
    assert controller._cache is not None
    assert controller._cache.get_seq_length() == 2
    linear = controller._cache.layers[1]
    assert float(linear.conv_states.item()) == 3.0
    assert float(linear.recurrent_states.item()) == 30.0

    # The rejected tail is deliberately large, making a failed state restore
    # obvious. accepted=1 must commit only rows [3,4], never row 99.
    controller.verify_rollback(torch.tensor([[3, 4, 99]], dtype=torch.long))
    assert controller._cache.get_seq_length() == 5
    assert float(linear.conv_states.item()) == 109.0
    committed = controller.commit_rollback(1)
    assert controller._cache.get_seq_length() == 4
    assert float(linear.conv_states.item()) == 10.0
    assert float(linear.recurrent_states.item()) == 100.0
    assert tuple(committed["logits"].shape) == (1, 2, 128)
    assert tuple(committed["dflash_features"].shape) == (1, 2, 3)
    audit = controller.dflash_rollback_audit
    assert audit["historical_prefix_replay_during_verify"] is False
    assert audit["rollback_commit_transactions"] == 1
    assert audit["rollback_commit_replay_calls"] == 2
    assert audit["cache_sequence_length"] == 4

    failed = FrameworkDFlashRollbackTarget(TinyHybridTarget().eval()).eval()
    failed.begin_rollback(torch.tensor([[1, 2]], dtype=torch.long))
    failed.verify_rollback(torch.tensor([[3, 99]], dtype=torch.long))
    failed.abort_rollback()
    failed_audit = failed.dflash_rollback_audit
    assert failed_audit["pending_transaction"] is False
    assert failed_audit["cache_sequence_length"] is None
    assert failed_audit["rollback_aborts"] == 1
    print("PASS: framework KV/GDN restore and anchor+accepted commit replay")


if __name__ == "__main__":
    main()
