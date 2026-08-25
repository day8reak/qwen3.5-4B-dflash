from __future__ import annotations

from types import SimpleNamespace
import unittest

import torch

from qwen35_mtp.ops import ModuleMtpOps, TorchMtpOps


class OpsTest(unittest.TestCase):
    def test_strict_module_rejects_missing_target_operations(self):
        with self.assertRaisesRegex(ValueError, "missing"):
            ModuleMtpOps(SimpleNamespace(linear=lambda x, w: x), strict=True)

    def test_explicit_simulation_fallback_matches_torch(self):
        fallback = ModuleMtpOps(SimpleNamespace(), strict=False)
        reference = TorchMtpOps()
        x = torch.tensor([[[1.0, -2.0]]])
        weight = torch.tensor([0.1, -0.2])
        torch.testing.assert_close(
            fallback.rms_norm(x, weight, 1e-6),
            reference.rms_norm(x, weight, 1e-6),
        )

    def test_top1_tie_uses_lowest_token_id(self):
        hidden = torch.tensor([[1.0, 0.0]])
        lm_head = torch.tensor(
            [
                [2.0, 0.0],
                [2.0, 0.0],
                [1.0, 0.0],
            ]
        )
        token = TorchMtpOps().top1(hidden, lm_head)
        self.assertEqual(int(token.item()), 0)

    def test_top1_rejects_non_finite_values(self):
        with self.assertRaises(FloatingPointError):
            TorchMtpOps().top1(
                torch.tensor([[float("nan"), 0.0]]), torch.eye(2)
            )


if __name__ == "__main__":
    unittest.main()
