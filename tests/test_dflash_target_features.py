import unittest

import torch
from torch import nn
import torch.nn.functional as F

from qwen35_dflash.target_features import (
    DFlashBaseModelOutputWithPast,
    DFlashCausalLMOutputWithPast,
    DFlashFeatureCollector,
    DFlashTargetFeatureSpec,
    QWEN35_4B_DFLASH_TARGET_FEATURES,
)
from qwen35_dflash.config import DFLASH_CONFIG


class _ToyTarget(nn.Module):
    """A decoder-loop fixture with no attention, GDN or cache changes."""

    def __init__(self, hidden_size: int = 2560, vocab_size: int = 17) -> None:
        super().__init__()
        torch.manual_seed(31)
        self.register_buffer("head", torch.randn(vocab_size, hidden_size))

    def forward(self, inputs, *, output_dflash_features=False):
        collector = DFlashFeatureCollector(
            QWEN35_4B_DFLASH_TARGET_FEATURES,
            enabled=output_dflash_features,
            detach=True,
        )
        hidden_states = inputs
        for layer_index in range(32):
            hidden_states = hidden_states + float(layer_index + 1)
            collector.capture(layer_index, hidden_states)
        logits = F.linear(hidden_states, self.head)
        return logits, collector.finalize()


class DFlashTargetFeatureTest(unittest.TestCase):
    def test_qwen35_contract(self):
        spec = QWEN35_4B_DFLASH_TARGET_FEATURES
        self.assertEqual(spec.layer_ids, (1, 5, 9, 13, 17, 21, 25, 29))
        self.assertEqual(spec.feature_size, 20480)
        self.assertEqual(DFLASH_CONFIG["feature_layers"], list(spec.layer_ids))
        self.assertEqual(DFLASH_CONFIG["feature_dim"], spec.feature_size)

    def test_feature_mode_does_not_change_target_logits(self):
        model = _ToyTarget().eval()
        inputs = torch.randn(1, 64, 2560)
        with torch.inference_mode():
            ordinary_logits, ordinary_features = model(inputs)
            feature_logits, features = model(
                inputs,
                output_dflash_features=True,
            )
        self.assertIsNone(ordinary_features)
        self.assertTrue(torch.equal(feature_logits, ordinary_logits))
        self.assertEqual(tuple(features.shape), (1, 64, 20480))

    def test_feature_order_is_post_layer_checkpoint_order(self):
        model = _ToyTarget().eval()
        inputs = torch.zeros(1, 1, 2560)
        with torch.inference_mode():
            _, features = model(inputs, output_dflash_features=True)
        for feature_index, layer_index in enumerate(
            QWEN35_4B_DFLASH_TARGET_FEATURES.layer_ids
        ):
            start = feature_index * 2560
            expected = float((layer_index + 1) * (layer_index + 2) // 2)
            self.assertTrue(
                torch.equal(
                    features[..., start : start + 2560],
                    torch.full((1, 1, 2560), expected),
                )
            )

    def test_detach_is_auxiliary_only(self):
        spec = DFlashTargetFeatureSpec(
            layer_ids=(0, 2),
            hidden_size=4,
            num_hidden_layers=3,
        )
        source = torch.randn(1, 2, 4, requires_grad=True)
        detached = DFlashFeatureCollector(spec, enabled=True, detach=True)
        attached = DFlashFeatureCollector(spec, enabled=True, detach=False)
        for index in spec.layer_ids:
            hidden = source + float(index)
            detached.capture(index, hidden)
            attached.capture(index, hidden)
        self.assertFalse(detached.finalize().requires_grad)
        self.assertTrue(attached.finalize().requires_grad)
        self.assertTrue(source.requires_grad)

    def test_clone_protects_against_in_place_target_code(self):
        spec = DFlashTargetFeatureSpec(
            layer_ids=(0,),
            hidden_size=2,
            num_hidden_layers=1,
        )
        hidden = torch.zeros(1, 1, 2)
        collector = DFlashFeatureCollector(
            spec,
            enabled=True,
            detach=True,
            clone=True,
        )
        collector.capture(0, hidden)
        hidden.add_(1.0)
        self.assertTrue(torch.equal(collector.finalize(), torch.zeros_like(hidden)))

    def test_missing_layer_fails_closed(self):
        spec = DFlashTargetFeatureSpec(
            layer_ids=(0, 2),
            hidden_size=4,
            num_hidden_layers=3,
        )
        collector = DFlashFeatureCollector(spec, enabled=True)
        collector.capture(0, torch.zeros(1, 1, 4))
        with self.assertRaisesRegex(RuntimeError, "not captured"):
            collector.finalize()

    def test_opt_in_output_types_expose_feature_field(self):
        hidden = torch.randn(1, 2, 4)
        features = torch.randn(1, 2, 8)
        base = DFlashBaseModelOutputWithPast(
            last_hidden_state=hidden,
            dflash_features=features,
        )
        causal = DFlashCausalLMOutputWithPast(
            logits=torch.randn(1, 2, 5),
            dflash_features=features,
        )
        self.assertIs(base.dflash_features, features)
        self.assertIs(causal.dflash_features, features)


if __name__ == "__main__":
    unittest.main()
