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
from models.dflash_v1.target_quant import (  # noqa: E402
    QUANT_MODE_W8A8_DYNAMIC,
    OriginalQuantizedEmbedding,
    TargetQuantizationRequest,
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
        self.commit_rows: list[int] = []
        self._pending_chunk: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def discard_dflash_chunk_state(self) -> None:
        self._pending_chunk = None

    def commit_dflash_chunk_state(
        self,
        committed_rows: int,
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        if self._pending_chunk is None:
            raise RuntimeError("no fake chunk is pending")
        conv_base, recurrent_base, token_values = self._pending_chunk
        if not 1 <= committed_rows <= token_values.shape[1]:
            raise ValueError("invalid fake commit length")
        delta = token_values[:, :committed_rows].sum()
        self.commit_rows.append(committed_rows)
        self._pending_chunk = None
        return {
            0: (
                conv_base + delta,
                recurrent_base.float() + delta.float() * 10,
            )
        }

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        past_key_values,
        new_kv_cache_pos: torch.Tensor,
        allQLen,
        output_dflash_features: bool,
        gdr_effective_length: torch.Tensor,
        dflash_chunk_verify: bool = False,
        **kwargs,
    ):
        skip_lm_head = bool(kwargs.pop("dflash_skip_lm_head", False))
        last_token_only = bool(kwargs.pop("dflash_last_token_only", False))
        del kwargs
        assert gdr_effective_length.dtype == torch.int16
        assert tuple(gdr_effective_length.shape) == (input_ids.shape[0],)
        assert gdr_effective_length.device == input_ids.device
        assert bool((gdr_effective_length >= 1).all())
        assert bool((gdr_effective_length <= input_ids.shape[1]).all())
        self.calls.append(
            {
                "positions": tuple(int(value) for value in new_kv_cache_pos.tolist()),
                "all_q_len": tuple(int(value) for value in allQLen),
                "gdr_effective_length": tuple(
                    int(value) for value in gdr_effective_length.tolist()
                ),
                "chunk_verify": dflash_chunk_verify,
                "skip_lm_head": skip_lm_head,
                "last_token_only": last_token_only,
            }
        )
        conv_state, recurrent_state = past_key_values[0]
        token_values = input_ids.to(torch.float16)
        if not dflash_chunk_verify:
            conv_state.add_(token_values.sum())
            recurrent_state.add_(token_values.sum() * 10)
        else:
            if self._pending_chunk is not None:
                raise RuntimeError("fake chunk already pending")
            self._pending_chunk = (
                conv_state.clone(),
                recurrent_state.clone(),
                token_values.clone(),
            )
            conv_state.add_(token_values.sum())
            past_key_values[0] = (
                conv_state,
                recurrent_state.float() + token_values.sum().float() * 10,
            )

        rows = input_ids.shape[1]
        logit_rows = 0 if skip_lm_head else (1 if last_token_only else rows)
        logits = torch.zeros((1, logit_rows, 1), dtype=torch.float16).expand(
            1,
            logit_rows,
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

    def commit_dflash_chunk_state(self, committed_rows: int):
        return self.model.commit_dflash_chunk_state(committed_rows)

    def discard_dflash_chunk_state(self) -> None:
        self.model.discard_dflash_chunk_state()


class QuantInputFakeHIAIModel(FakeHIAIModel):
    def __init__(self) -> None:
        super().__init__()
        self.config.hidden_size = 2
        self.quant_input_markers: list[float] = []

    def forward(self, **kwargs):
        inputs_embeds = kwargs.get("inputs_embeds")
        assert isinstance(inputs_embeds, torch.Tensor)
        self.quant_input_markers.append(float(inputs_embeds[0, 0, 0]))
        return super().forward(**kwargs)


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
    bootstrap = bridge.begin_rollback(torch.tensor([[1, 2]], dtype=torch.long))
    assert tuple(bootstrap["logits"].shape) == (1, 1, VOCAB_SIZE)
    assert len(model.calls) == 1
    assert model.calls[0] == {
        "positions": (0, 1),
        "all_q_len": (2,),
        "gdr_effective_length": (2,),
        "chunk_verify": False,
        "skip_lm_head": False,
        "last_token_only": True,
    }
    assert bridge.dflash_rollback_audit["persistent_cursor"] == 2
    assert bridge.dflash_rollback_audit["rollback_prefill_lm_head_skips"] == 1
    assert bridge.dflash_rollback_audit["rollback_prefill_token_calls"] == 1
    assert bridge.dflash_rollback_audit["prefill_execution_mode"] == (
        "block_aligned_real_token_chunks_original_gdr"
    )
    assert bridge.dflash_rollback_audit[
        "ordinary_gdr_effective_length_contract"
    ] == "int16_batch_call_local_valid_rows"

    try:
        bridge._prepare_rollback_state(17)
    except ValueError as error:
        assert "1..16 rows" in str(error)
    else:
        raise AssertionError("17-row rollback verify block was not rejected")

    persistent_before = bridge._persistent_state
    assert persistent_before is not None
    provisional = bridge._prepare_rollback_state(3)
    assert provisional[0][0] is not persistent_before[0][0]
    assert provisional[0][1] is not persistent_before[0][1]
    assert provisional[1][0] is persistent_before[1][0]
    assert provisional[1][1] is persistent_before[1][1]

    bridge.verify_rollback(torch.tensor([[3, 4, 99]], dtype=torch.long))
    state = bridge._persistent_state
    assert state is not None
    assert tuple(state[0][0].shape) == (1, 3, 2)
    assert float(state[0][0][0, 0, 0]) == 3.0
    bridge.commit_rollback(1)
    assert bridge.dflash_rollback_audit["persistent_cursor"] == 4
    assert bridge.dflash_rollback_audit["last_committed_rows"] == 2
    assert model.commit_rows == [2]
    state = bridge._persistent_state
    assert state is not None
    assert float(state[0][0][0, 0, 0]) == 10.0
    assert float(state[0][1][0, 0, 0, 0]) == 100.0
    assert state[0][1].dtype == torch.float16

    # K changes from 2 proposals to 1. The next verify starts from the scalar
    # state committed by the second original-GDR chunk call.
    bridge.verify_rollback(torch.tensor([[5, 6]], dtype=torch.long))
    assert model.calls[-1] == {
        "positions": (4, 5),
        "all_q_len": (6,),
        "gdr_effective_length": (2,),
        "chunk_verify": True,
        "skip_lm_head": False,
        "last_token_only": False,
    }
    state = bridge._persistent_state
    assert state is not None
    assert tuple(state[0][0].shape) == (1, 3, 2)
    assert float(state[0][0][0, 0, 0]) == 10.0
    bridge.commit_rollback(0)
    audit = bridge.dflash_rollback_audit
    assert audit["persistent_cursor"] == 5
    assert audit["rollback_verify_calls"] == 2
    assert audit["rollback_commit_calls"] == 2
    assert audit["rollback_gdr_verify_layer_calls"] == 2
    assert audit["rollback_gdr_commit_layer_calls"] == 2
    assert audit["last_committed_rows"] == 1
    assert audit["gdr_backend"] == "npu_chunk_gated_delta_rule_two_pass"
    assert audit["custom_gdr_mtp_required"] is False
    assert model.commit_rows == [2, 1]
    assert audit["historical_prefix_replay_during_verify"] is False
    assert audit["persistent_call_synchronization_policy"] == (
        "same_device_stream_dependencies_no_per_call_host_barrier"
    )
    assert bridge.dflash_full_prefix_bridge_audit["device_synchronizations"] == 0
    bridge.verify_rollback(torch.tensor([[6, 7]], dtype=torch.long))
    bridge.abort_rollback()
    failed_audit = bridge.dflash_rollback_audit
    assert failed_audit["session_invalid"] is True
    assert failed_audit["pending_verify_rows"] is None
    assert bridge._persistent_state is None

    # Prefill must preserve real-token boundaries around the receiver's S=64
    # chunk gear, never issue one call per prompt token, and never commit padding.
    expected_chunks = {
        1: (1,),
        63: (63,),
        64: (64,),
        65: (64, 1),
    }
    for prompt_length, chunk_lengths in expected_chunks.items():
        chunk_model = FakeHIAIModel().eval()
        chunk_bridge = InternalDFlashTarget(
            FakeWrapper(chunk_model).eval(),
            device=torch.device("cpu"),
            dtype=torch.float16,
            kv_cache_max_len=64 if prompt_length <= 64 else 128,
            rollback_enabled=True,
        ).eval()
        prompt = torch.arange(1, prompt_length + 1, dtype=torch.long).view(1, -1)
        chunk_bootstrap = chunk_bridge.begin_rollback(prompt)
        assert tuple(chunk_bootstrap["logits"].shape) == (1, 1, VOCAB_SIZE)
        assert tuple(chunk_bootstrap["dflash_features"].shape) == (
            1,
            prompt_length,
            FEATURE_WIDTH,
        )
        expected_positions: list[tuple[int, ...]] = []
        cursor = 0
        for chunk_length in chunk_lengths:
            expected_positions.append(tuple(range(cursor, cursor + chunk_length)))
            cursor += chunk_length
        assert [call["positions"] for call in chunk_model.calls] == expected_positions
        assert [
            call["gdr_effective_length"] for call in chunk_model.calls
        ] == [(chunk_length,) for chunk_length in chunk_lengths]
        assert [call["skip_lm_head"] for call in chunk_model.calls] == [
            *([True] * (len(chunk_lengths) - 1)),
            False,
        ]
        assert [call["last_token_only"] for call in chunk_model.calls] == [
            *([False] * (len(chunk_lengths) - 1)),
            True,
        ]
        chunk_audit = chunk_bridge.dflash_rollback_audit
        assert chunk_audit["persistent_cursor"] == prompt_length
        assert chunk_audit["rollback_prefill_token_calls"] == len(chunk_lengths)
        assert chunk_audit["rollback_prefill_lm_head_skips"] == prompt_length - 1

    ordinary_model = FakeHIAIModel().eval()
    ordinary_bridge = InternalDFlashTarget(
        FakeWrapper(ordinary_model).eval(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        kv_cache_max_len=64,
        rollback_enabled=True,
    ).eval()
    ordinary = ordinary_bridge.begin_ordinary(
        torch.tensor([[7, 8, 9]], dtype=torch.long)
    )
    assert tuple(ordinary["logits"].shape) == (1, 1, VOCAB_SIZE)
    assert ordinary_model.calls == [
        {
            "positions": (0, 1, 2),
            "all_q_len": (3,),
            "gdr_effective_length": (3,),
            "chunk_verify": False,
            "skip_lm_head": False,
            "last_token_only": True,
        }
    ]
    ordinary_audit = ordinary_bridge.dflash_rollback_audit
    assert ordinary_audit["persistent_cursor"] == 3
    assert ordinary_audit["ordinary_prefill_token_calls"] == 1
    assert ordinary_audit["ordinary_prefill_lm_head_skips"] == 2

    # The full-prefix oracle keeps a fixed 64-row physical GDR gear.  Its
    # effective length must remain the real prefix length rather than allQLen
    # or the padded execution length.
    full_prefix_model = FakeHIAIModel().eval()
    full_prefix_bridge = InternalDFlashTarget(
        FakeWrapper(full_prefix_model).eval(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        kv_cache_max_len=64,
        rollback_enabled=False,
    ).eval()
    real_prefix = torch.arange(1, 38, dtype=torch.long).view(1, -1)
    full_prefix_bridge.prepare_dflash_full_prefix_call(
        input_ids=real_prefix,
        sequence_length=37,
        output_dflash_features=True,
        logits_to_keep=1,
        call_index=0,
    )
    full_prefix_output = full_prefix_bridge(
        real_prefix,
        use_cache=False,
        return_dict=True,
        output_hidden_states=False,
        output_dflash_features=True,
        logits_to_keep=1,
    )
    assert tuple(full_prefix_output["logits"].shape) == (1, 1, VOCAB_SIZE)
    assert tuple(full_prefix_output["dflash_features"].shape) == (
        1,
        37,
        FEATURE_WIDTH,
    )
    assert full_prefix_model.calls[0]["positions"] == tuple(range(64))
    assert full_prefix_model.calls[0]["all_q_len"] == (37,)
    assert full_prefix_model.calls[0]["gdr_effective_length"] == (37,)
    full_prefix_audit = full_prefix_bridge.dflash_full_prefix_bridge_audit
    assert full_prefix_audit["last_requested_sequence_length"] == 37
    assert full_prefix_audit["last_execution_sequence_length"] == 64
    assert full_prefix_audit["gdr_effective_length_contract"] == (
        "int16_batch_call_local_valid_rows"
    )
    print("PASS: HIAI bridge two-pass chunk GDR and logical KV cursor")


def test_bridge_rollback_contract() -> None:
    main()


def test_quant_embedding_covers_chunk_prefill_and_verify() -> None:
    request = TargetQuantizationRequest(
        mode=QUANT_MODE_W8A8_DYNAMIC,
        config_path=Path("qwen3.5.yaml"),
        quant_weight_path=Path("quant.safetensors"),
        embedding_weight_path=Path("embedding.safetensors"),
        embedding_scale_path=Path("embedding-scale.safetensors"),
    )
    quantized_embedding = OriginalQuantizedEmbedding(
        torch.full((VOCAB_SIZE, 2), 6, dtype=torch.int8),
        torch.full((VOCAB_SIZE, 1), 0.5, dtype=torch.float32),
        output_dtype=torch.float16,
    ).eval()
    model = QuantInputFakeHIAIModel().eval()
    bridge = InternalDFlashTarget(
        FakeWrapper(model).eval(),
        device=torch.device("cpu"),
        dtype=torch.float16,
        kv_cache_max_len=128,
        rollback_enabled=True,
        quantization_request=request,
        quantized_embedding=quantized_embedding,
        target_quantization_audit={
            "status": "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM",
            "scheme": QUANT_MODE_W8A8_DYNAMIC,
            "qlinear_count": 1,
        },
    ).eval()
    prompt = torch.arange(1, 66, dtype=torch.long).view(1, -1)
    bridge.begin_rollback(prompt)
    bridge.verify_rollback(torch.tensor([[66, 67]], dtype=torch.long))
    bridge.commit_rollback(0)

    assert model.quant_input_markers == [3.0, 3.0, 3.0]
    quant_audit = bridge.dflash_target_quantization_audit
    assert quant_audit["scheme"] == QUANT_MODE_W8A8_DYNAMIC
    assert quant_audit["embedding_lookup_calls"] == 3
    assert quant_audit["embedding_lookup_successes"] == 3
    assert quant_audit["embedding_lookup_failures"] == 0
    assert bridge.dflash_rollback_audit["target_quantization"] == quant_audit


if __name__ == "__main__":
    main()
