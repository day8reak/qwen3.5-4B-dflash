from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from qwen35_mtp.precision import (
    audit_fp16_conversion,
    fp16_conversion_is_admissible,
    metric_within,
    project_logits_chunked,
    stable_top2,
    tensor_error_metrics,
)


class PrecisionTest(unittest.TestCase):
    def test_exact_tensor_metrics(self):
        value = torch.tensor([[1.0, -2.0]], dtype=torch.bfloat16)
        metrics = tensor_error_metrics(value, value.to(torch.float16))
        self.assertEqual(metrics["max_abs"], 0.0)
        self.assertEqual(metrics["relative_l2"], 0.0)
        self.assertAlmostEqual(metrics["cosine"], 1.0)
        self.assertTrue(
            metric_within(metrics, max_relative_l2=0.0, min_cosine=1.0)
        )

    def test_non_finite_candidate_fails_gate(self):
        metrics = tensor_error_metrics(
            torch.tensor([1.0]), torch.tensor([float("inf")])
        )
        self.assertFalse(
            metric_within(metrics, max_relative_l2=1.0, min_cosine=-1.0)
        )

    def test_chunked_projection_and_stable_tie(self):
        hidden = torch.tensor([1.0, 0.0])
        weight = torch.tensor(
            [[2.0, 0.0], [2.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
        )
        chunked = project_logits_chunked(
            hidden, weight, compute_dtype=torch.float32, chunk_size=2
        )
        torch.testing.assert_close(chunked, F.linear(hidden, weight))
        top2 = stable_top2(chunked)
        self.assertEqual(top2["token_ids"], [0, 1])
        self.assertEqual(top2["margin"], 0.0)

    def test_fp16_conversion_audit_detects_range_loss(self):
        report = audit_fp16_conversion(
            [("safe", torch.tensor([1.0], dtype=torch.bfloat16))]
        )
        self.assertTrue(report["safe_for_fp16_range"])
        self.assertTrue(report["exact_roundtrip"])
        self.assertTrue(fp16_conversion_is_admissible(report))
        underflow = audit_fp16_conversion(
            [("underflow", torch.tensor([1.0e-30], dtype=torch.bfloat16))]
        )
        self.assertGreater(underflow["underflow_to_zero_count"], 0)
        self.assertFalse(underflow["exact_roundtrip"])
        self.assertTrue(fp16_conversion_is_admissible(underflow))
        overflow = audit_fp16_conversion(
            [("overflow", torch.tensor([1.0e10], dtype=torch.bfloat16))]
        )
        self.assertEqual(overflow["overflow_count"], 1)
        self.assertFalse(overflow["safe_for_fp16_range"])
        self.assertFalse(fp16_conversion_is_admissible(overflow))


if __name__ == "__main__":
    unittest.main()
