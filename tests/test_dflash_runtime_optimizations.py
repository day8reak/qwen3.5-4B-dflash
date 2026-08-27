"""Exact reduced-shape checks for the rollback hot-path optimizations."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import torch

from models.dflash_v1.dflash_ascend310p_ops import (
    EXHAUSTIVE_CHECKS_ENV,
    exhaustive_value_checks_enabled,
)
from models.dflash_v1.dflash_config import Qwen35DFlashConfig
from models.dflash_v1.modeling_dflash import DFlashDraftModel


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

    def test_accelerator_checks_default_to_boundary_only(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop(EXHAUSTIVE_CHECKS_ENV, None)
            self.assertFalse(exhaustive_value_checks_enabled("npu:0"))
            self.assertTrue(exhaustive_value_checks_enabled("cpu"))
        with patch.dict(os.environ, {EXHAUSTIVE_CHECKS_ENV: "1"}):
            self.assertTrue(exhaustive_value_checks_enabled("npu:0"))


if __name__ == "__main__":
    unittest.main()
