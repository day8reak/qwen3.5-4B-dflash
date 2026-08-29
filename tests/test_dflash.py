import types
import unittest
from dataclasses import replace

import torch

from qwen35_dflash.config import (
    OFFICIAL_QWEN35_4B_DFLASH,
    Qwen35DFlashConfig,
    audit_official_4b_dflash_config,
)
from qwen35_dflash.model import DFlashDraftModel, extract_context_feature
from qwen35_dflash.ops import ModuleDFlashOps, TorchDFlashOps


def tiny_config() -> Qwen35DFlashConfig:
    return Qwen35DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_target_layers=4,
        target_layer_ids=(0, 2),
        layer_types=("sliding_attention", "full_attention"),
        block_size=4,
        mask_token_id=31,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=64,
        sliding_window=4,
        use_sliding_window=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype="float32",
    )


class DFlashGoldenTest(unittest.TestCase):
    def _model(self) -> DFlashDraftModel:
        torch.manual_seed(7)
        model = DFlashDraftModel(tiny_config(), dtype=torch.float32)
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight") or "layernorm.weight" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.05)
        return model.eval()

    def test_official_config_contract(self):
        raw = {
            "hidden_size": 2560,
            "intermediate_size": 9216,
            "vocab_size": 248320,
            "num_hidden_layers": 6,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "head_dim": 128,
            "num_target_layers": 32,
            "layer_types": OFFICIAL_QWEN35_4B_DFLASH["layer_types"],
            "attention_bias": False,
            "attention_dropout": 0.0,
            "hidden_act": "silu",
            "dtype": "bfloat16",
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 262144,
            "sliding_window": 4096,
            "use_sliding_window": True,
            "rope_parameters": {"rope_theta": 10000000, "rope_type": "default"},
            "dflash_config": {
                "block_size": 16,
                "mask_token_id": 248077,
                "target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29],
            },
        }
        config = Qwen35DFlashConfig.from_dict(raw)
        self.assertEqual(audit_official_4b_dflash_config(config), [])
        self.assertEqual(len(config.required_tensor_shapes()), 69)
        self.assertEqual(config.parameter_count, 634425856)

    def test_target_hidden_offset_and_concat(self):
        hidden = [torch.full((1, 3, 2), float(index)) for index in range(5)]
        result = extract_context_feature(hidden, [0, 2])
        self.assertEqual(tuple(result.shape), (1, 3, 4))
        torch.testing.assert_close(result[..., :2], hidden[1])
        torch.testing.assert_close(result[..., 2:], hidden[3])

    def test_cache_free_forward_and_top1_shapes(self):
        model = self._model()
        config = model.config
        target = torch.randn(1, 3, config.feature_size)
        noise = torch.randn(1, config.block_size, config.hidden_size)
        positions = torch.arange(7).unsqueeze(0)
        head = torch.randn(config.vocab_size, config.hidden_size)
        with torch.inference_mode():
            hidden = model(target, noise, positions)
            top1 = model.draft_top1(target, noise, positions, head)
        self.assertEqual(tuple(hidden.shape), (1, 4, 8))
        self.assertEqual(tuple(top1.shape), (1, 3))
        self.assertTrue(torch.isfinite(hidden).all())

    def test_cpu_fp16_simulation_is_finite(self):
        torch.manual_seed(19)
        model = DFlashDraftModel(tiny_config(), dtype=torch.float16).eval()
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if name.endswith("norm.weight") or "layernorm.weight" in name:
                    parameter.fill_(1.0)
                else:
                    parameter.normal_(mean=0.0, std=0.05)
            hidden = model(
                torch.randn(1, 3, model.config.feature_size, dtype=torch.float16),
                torch.randn(1, 4, model.config.hidden_size, dtype=torch.float16),
                torch.arange(7).unsqueeze(0),
            )
        self.assertEqual(hidden.dtype, torch.float16)
        self.assertTrue(torch.isfinite(hidden).all())

    def test_sliding_and_full_masks_differ(self):
        model = self._model()
        sliding = model.layers[0].self_attn._attention_mask(4, 3, device=torch.device("cpu"))
        full = model.layers[1].self_attn._attention_mask(4, 3, device=torch.device("cpu"))
        self.assertIsNone(full)
        self.assertEqual(tuple(sliding.shape), (1, 1, 4, 7))
        self.assertFalse(bool(sliding[0, 0, 0, -1]))
        self.assertTrue(bool(sliding[0, 0, -1, -1]))
        self.assertFalse(bool(sliding[0, 0, -1, 0]))

    def test_disabling_sliding_window_keeps_layer_causal(self):
        config = replace(tiny_config(), use_sliding_window=False)
        attention = DFlashDraftModel(config).layers[0].self_attn
        mask = attention._attention_mask(4, 3, device=torch.device("cpu"))
        self.assertIsNone(attention.sliding_window)
        self.assertFalse(bool(mask[0, 0, 0, -1]))
        self.assertTrue(bool(mask[0, 0, -1, -1]))
        self.assertTrue(bool(mask[0, 0, -1, 0]))

    def test_right_padded_context_mask_matches_unpadded_draft(self):
        model = self._model()
        target = torch.randn(1, 2, model.config.feature_size)
        padded_target = torch.cat(
            (target, torch.randn(1, 2, model.config.feature_size) * 100.0), dim=1
        )
        noise = torch.randn(1, 4, model.config.hidden_size)
        unpadded_positions = torch.arange(6).unsqueeze(0)
        padded_positions = torch.tensor([[0, 1, 0, 0, 2, 3, 4, 5]])
        with torch.inference_mode():
            expected = model(
                target,
                noise,
                unpadded_positions,
                context_attention_mask=torch.ones(1, 2, dtype=torch.bool),
            )
            actual = model(
                padded_target,
                noise,
                padded_positions,
                context_attention_mask=torch.tensor([[True, True, False, False]]),
            )
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_module_ops_strict_and_fallback(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            ModuleDFlashOps(types.ModuleType("empty"), strict=True)
        fallback = ModuleDFlashOps(types.ModuleType("empty"), strict=False)
        x = torch.tensor([[[3.0, 4.0]]])
        weight = torch.ones(2)
        actual = fallback.rms_norm(x, weight, 0.0)
        expected = TorchDFlashOps().rms_norm(x, weight, 0.0)
        torch.testing.assert_close(actual, expected)

    def test_complete_module_ops_are_drop_in_equivalent(self):
        torch_ops = TorchDFlashOps()
        module = types.ModuleType("complete")
        for name in ModuleDFlashOps.required_operations:
            setattr(module, name, getattr(torch_ops, name))
        model = self._model()
        target = torch.randn(1, 3, model.config.feature_size)
        noise = torch.randn(1, 4, model.config.hidden_size)
        positions = torch.arange(7).unsqueeze(0)
        with torch.inference_mode():
            expected = model(target, noise, positions)
            model.set_ops(ModuleDFlashOps(module, strict=True))
            actual = model(target, noise, positions)
        torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)

    def test_position_contract_rejects_short_vector(self):
        model = self._model()
        target = torch.randn(1, 3, model.config.feature_size)
        noise = torch.randn(1, 4, model.config.hidden_size)
        with self.assertRaisesRegex(ValueError, "position_ids"):
            model(target, noise, torch.arange(6).unsqueeze(0))


if __name__ == "__main__":
    unittest.main()
