"""Exact reduced-shape checks for the rollback hot-path optimizations."""

from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models.dflash_v1.dflash_ascend310p_ops import (
    EXHAUSTIVE_CHECKS_ENV,
    exhaustive_value_checks_enabled,
)
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.dflash_rollback_adapter import Qwen35DFlashRollbackAdapter
from models.dflash_v1.modeling_dflash import (
    DFlashDraftKVCache,
    DFlashDraftModel,
)


def tiny_config() -> Qwen35DFlashConfig:
    return Qwen35DFlashConfig(
        hidden_size=4,
        intermediate_size=8,
        vocab_size=32,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        head_dim=4,
        num_target_layers=2,
        target_layer_ids=(0, 1),
        layer_types=("full_attention",),
        block_size=4,
        mask_token_id=31,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=32,
        sliding_window=8,
        use_sliding_window=False,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype="float32",
    )


def initialized_tiny_model() -> DFlashDraftModel:
    model = DFlashDraftModel(tiny_config(), dtype=torch.float32).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


def initialized_mixed_attention_model() -> DFlashDraftModel:
    config = replace(
        tiny_config(),
        num_hidden_layers=2,
        layer_types=("sliding_attention", "full_attention"),
        use_sliding_window=True,
    )
    model = DFlashDraftModel(config, dtype=torch.float32).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.uniform_(-0.1, 0.1)
    return model


class TinyTransactionalTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.head = nn.Linear(4, 32, bias=False)
        self.pending: torch.Tensor | None = None

    @property
    def dflash_rollback_audit(self):
        return {"historical_prefix_replay_during_verify": False}

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.head

    def _output(self, input_ids: torch.Tensor):
        hidden = self.embedding(input_ids)
        return {
            "logits": self.head(hidden),
            "dflash_features": torch.cat((hidden, hidden), dim=-1),
        }

    def begin_ordinary(self, prompt_ids: torch.Tensor):
        return {"logits": self._output(prompt_ids)["logits"]}

    def advance_ordinary(self, input_ids: torch.Tensor):
        return {"logits": self._output(input_ids)["logits"]}

    def begin_rollback(self, prompt_ids: torch.Tensor):
        self.pending = None
        return self._output(prompt_ids)

    def verify_rollback(self, block_ids: torch.Tensor):
        self.pending = block_ids.detach().clone()
        return self._output(block_ids)

    def commit_rollback(self, accepted_draft_tokens: int):
        if self.pending is None:
            raise RuntimeError("no tiny transaction")
        committed = self.pending[:, : accepted_draft_tokens + 1]
        self.pending = None
        return self._output(committed)

    def abort_rollback(self) -> None:
        self.pending = None


class DFlashRuntimeOptimizationTest(unittest.TestCase):
    def test_projected_feature_path_matches_uncached_path(self) -> None:
        torch.manual_seed(20260827)
        model = initialized_tiny_model()
        target_hidden = torch.randn(1, 3, 8)
        noise_embedding = torch.randn(1, 4, 4)
        position_ids = torch.arange(7, dtype=torch.long).unsqueeze(0)

        with torch.inference_mode():
            expected = model(target_hidden, noise_embedding, position_ids)
            projected = model.project_target_hidden(target_hidden)
            actual = model.forward_projected(
                projected,
                noise_embedding,
                position_ids,
            )

        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_projection_can_be_appended_without_reprojecting_history(self) -> None:
        torch.manual_seed(20260827)
        model = initialized_tiny_model()
        first = torch.randn(1, 3, 8)
        second = torch.randn(1, 2, 8)

        with torch.inference_mode():
            full = model.project_target_hidden(torch.cat((first, second), dim=1))
            incremental = torch.cat(
                (
                    model.project_target_hidden(first),
                    model.project_target_hidden(second),
                ),
                dim=1,
            )

        # Splitting a GEMM by token can change the last FP32 bit even though
        # each row is independent.  Draft proposals remain target-verified, so
        # this gate checks numerical equivalence rather than bitwise tiling.
        torch.testing.assert_close(incremental, full, rtol=1e-6, atol=1e-7)

    def test_incremental_draft_kv_matches_cache_free_golden_across_rounds(
        self,
    ) -> None:
        torch.manual_seed(20260827)
        model = initialized_tiny_model()
        first = torch.randn(1, 3, 8)
        second = torch.randn(1, 2, 8)
        first_noise = torch.randn(1, 4, 4)
        second_noise = torch.randn(1, 3, 4)
        first_positions = torch.arange(7, dtype=torch.long).unsqueeze(0)
        second_positions = torch.arange(8, dtype=torch.long).unsqueeze(0)
        cache = model.new_kv_cache()

        with torch.inference_mode():
            projected_first = model.project_target_hidden(first)
            expected_first = model.forward_projected(
                projected_first,
                first_noise,
                first_positions,
            )
            actual_first = model.forward_cached_projected(
                projected_first,
                first_noise,
                first_positions,
                cache,
            )
            projected_second = model.project_target_hidden(second)
            expected_second = model.forward_projected(
                torch.cat((projected_first, projected_second), dim=1),
                second_noise,
                second_positions,
            )
            actual_second = model.forward_cached_projected(
                projected_second,
                second_noise,
                second_positions[:, 3:],
                cache,
            )

        torch.testing.assert_close(actual_first, expected_first, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(actual_second, expected_second, rtol=1e-5, atol=1e-6)
        self.assertEqual(cache.committed_length, 5)
        self.assertEqual(
            cache.audit,
            {
                "enabled": True,
                "mode": "upstream_equivalent_append_then_crop",
                "num_layers": 1,
                "max_length": 32,
                "committed_length": 5,
                "active_round": False,
                "rounds": 2,
                "aborted_rounds": 0,
                "crop_calls": 0,
                "tokens_appended": 5,
                "tokens_reused": 3,
                "peak_committed_length": 5,
                "logical_bytes": 160,
            },
        )

    def test_incremental_cache_matches_mixed_sliding_and_full_attention(self) -> None:
        torch.manual_seed(20260827)
        model = initialized_mixed_attention_model()
        first = model.project_target_hidden(torch.randn(1, 4, 8))
        second = model.project_target_hidden(torch.randn(1, 2, 8))
        first_noise = torch.randn(1, 4, 4)
        second_noise = torch.randn(1, 4, 4)
        cache = model.new_kv_cache()

        with torch.inference_mode():
            first_expected = model.forward_projected(
                first,
                first_noise,
                torch.arange(8, dtype=torch.long).unsqueeze(0),
            )
            first_actual = model.forward_cached_projected(
                first,
                first_noise,
                torch.arange(8, dtype=torch.long).unsqueeze(0),
                cache,
            )
            second_expected = model.forward_projected(
                torch.cat((first, second), dim=1),
                second_noise,
                torch.arange(10, dtype=torch.long).unsqueeze(0),
            )
            second_actual = model.forward_cached_projected(
                second,
                second_noise,
                torch.arange(4, 10, dtype=torch.long).unsqueeze(0),
                cache,
            )

        torch.testing.assert_close(first_actual, first_expected, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(second_actual, second_expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(cache.audit["num_layers"], 2)
        self.assertEqual(cache.committed_length, 6)

    def test_draft_cache_abort_preserves_committed_context(self) -> None:
        cache = DFlashDraftKVCache(num_layers=1, max_length=16)
        first_key = torch.randn(1, 1, 3, 4)
        first_value = torch.randn(1, 1, 3, 4)
        cache.begin_round(new_context_length=2, block_length=1)
        cache.update(0, first_key, first_value)
        cache.finish_round()
        self.assertEqual(cache.committed_length, 2)

        cache.begin_round(new_context_length=1, block_length=2)
        cache.update(
            0,
            torch.randn(1, 1, 3, 4),
            torch.randn(1, 1, 3, 4),
        )
        cache.abort_round()

        self.assertEqual(cache.committed_length, 2)
        self.assertFalse(cache.audit["active_round"])
        self.assertEqual(cache.audit["rounds"], 1)
        self.assertEqual(cache.audit["aborted_rounds"], 1)

        # A valid next round must still reuse the original committed prefix.
        cache.begin_round(new_context_length=1, block_length=1)
        key, value = cache.update(
            0,
            torch.randn(1, 1, 2, 4),
            torch.randn(1, 1, 2, 4),
        )
        self.assertEqual(tuple(key.shape), (1, 1, 4, 4))
        self.assertEqual(tuple(value.shape), (1, 1, 4, 4))
        cache.finish_round()
        self.assertEqual(cache.committed_length, 3)

    def test_cached_forward_aborts_round_when_a_layer_fails(self) -> None:
        torch.manual_seed(20260827)
        model = initialized_tiny_model()
        cache = model.new_kv_cache()
        projected = model.project_target_hidden(torch.randn(1, 2, 8))
        noise = torch.randn(1, 2, 4)
        positions = torch.arange(4, dtype=torch.long).unsqueeze(0)

        with patch.object(
            model.norm,
            "forward",
            side_effect=RuntimeError("injected final norm failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "injected final norm failure"):
                model.forward_cached_projected(
                    projected,
                    noise,
                    positions,
                    cache,
                )

        self.assertEqual(cache.committed_length, 0)
        self.assertFalse(cache.audit["active_round"])
        self.assertEqual(cache.audit["rounds"], 0)
        self.assertEqual(cache.audit["aborted_rounds"], 1)

    def test_draft_cache_crop_and_clear_use_logical_committed_rows(self) -> None:
        cache = DFlashDraftKVCache(num_layers=1, max_length=16)
        cache.begin_round(new_context_length=4, block_length=2)
        cache.update(
            0,
            torch.randn(1, 1, 6, 4),
            torch.randn(1, 1, 6, 4),
        )
        cache.finish_round()
        cache.crop(2)
        self.assertEqual(cache.committed_length, 2)
        self.assertEqual(cache.audit["logical_bytes"], 64)
        cache.clear()
        self.assertEqual(cache.committed_length, 0)
        self.assertEqual(cache.audit["logical_bytes"], 0)

    def test_rollback_adapter_projects_and_appends_only_new_context(self) -> None:
        torch.manual_seed(20260827)
        target = TinyTransactionalTarget().eval()
        draft = initialized_tiny_model()
        adapter = Qwen35DFlashRollbackAdapter(
            target,
            draft,
            require_official_config=False,
        )
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        adapter.begin_rollback(prompt)

        first_prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        first_proposals = adapter.propose_rollback(first_prefix, 2)
        self.assertEqual(tuple(first_proposals.shape), (1, 2))
        self.assertEqual(adapter.dflash_draft_cache_audit["committed_length"], 3)
        self.assertEqual(
            adapter.dflash_draft_cache_audit["pending_projected_tokens"],
            0,
        )

        adapter.verify_rollback(
            torch.tensor([[4, int(first_proposals[0, 0])]], dtype=torch.long)
        )
        adapter.commit_rollback(0)
        self.assertEqual(
            adapter.dflash_draft_cache_audit["pending_projected_tokens"],
            1,
        )

        second_prefix = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        second_proposals = adapter.propose_rollback(second_prefix, 2)
        self.assertEqual(tuple(second_proposals.shape), (1, 2))
        stats = adapter.snapshot_rollback_stats()
        self.assertEqual(stats.draft_kv_cache_rounds, 2)
        self.assertEqual(stats.draft_kv_cache_tokens_appended, 4)
        self.assertEqual(stats.draft_kv_cache_tokens_reused, 3)
        self.assertEqual(stats.draft_kv_cache_peak_tokens, 4)
        self.assertEqual(stats.draft_feature_projection_calls, 2)
        self.assertEqual(stats.draft_feature_tokens_projected, 4)
        self.assertEqual(adapter.dflash_draft_cache_audit["committed_length"], 4)

    def test_accelerator_checks_default_to_boundary_only(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(EXHAUSTIVE_CHECKS_ENV, None)
            self.assertFalse(exhaustive_value_checks_enabled("npu:0"))
            self.assertTrue(exhaustive_value_checks_enabled("cpu"))
        with patch.dict(os.environ, {EXHAUSTIVE_CHECKS_ENV: "1"}):
            self.assertTrue(exhaustive_value_checks_enabled("npu:0"))


if __name__ == "__main__":
    unittest.main()
