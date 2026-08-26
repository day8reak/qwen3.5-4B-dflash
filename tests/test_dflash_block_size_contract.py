"""Regression tests for the locked upstream DFlash block-size convention."""

from __future__ import annotations

import unittest

import torch

from models.dflash_v1.dflash_config import (
    OFFICIAL_DFLASH_BLOCK_SIZE,
    OFFICIAL_DFLASH_PROPOSAL_CAPACITY,
    OFFICIAL_DFLASH_PROPOSAL_SWEEP,
)
from models.dflash_v1.dflash_reference_decode_v1 import (
    dflash_full_prefix_greedy,
)
from models.dflash_v1.diagnose_acceptance import parse_proposal_counts


class DFlashBlockSizeContractTest(unittest.TestCase):
    def test_official_limits_and_sweep_share_one_conversion(self) -> None:
        self.assertEqual(OFFICIAL_DFLASH_BLOCK_SIZE, 16)
        self.assertEqual(OFFICIAL_DFLASH_PROPOSAL_CAPACITY, 15)
        self.assertEqual(OFFICIAL_DFLASH_PROPOSAL_SWEEP, (1, 3, 5, 7, 15))
        self.assertEqual(
            tuple(count + 1 for count in OFFICIAL_DFLASH_PROPOSAL_SWEEP),
            (2, 4, 6, 8, 16),
        )

    def test_block_size_includes_anchor_row(self) -> None:
        proposal_limits: list[int] = []

        def target(input_ids: torch.Tensor) -> torch.Tensor:
            vocab_size = 32
            next_ids = (input_ids + 1) % vocab_size
            logits = torch.full(
                (*input_ids.shape, vocab_size),
                -1000.0,
                dtype=torch.float32,
                device=input_ids.device,
            )
            return logits.scatter(-1, next_ids.unsqueeze(-1), 1000.0)

        def draft(prefix_ids: torch.Tensor, proposal_limit: int) -> torch.Tensor:
            proposal_limits.append(proposal_limit)
            first = int(prefix_ids[0, -1]) + 1
            return torch.arange(
                first,
                first + proposal_limit,
                dtype=torch.long,
                device=prefix_ids.device,
            ).unsqueeze(0)

        result = dflash_full_prefix_greedy(
            target,
            draft,
            [1],
            max_new_tokens=4,
            block_size=4,
        )

        self.assertEqual(proposal_limits, [3])
        self.assertEqual(result.generated_token_ids, (2, 3, 4, 5))

    def test_block_size_requires_anchor_and_proposal(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 2"):
            dflash_full_prefix_greedy(
                lambda input_ids: torch.zeros((*input_ids.shape, 2)),
                lambda prefix_ids, proposal_limit: [0],
                [0],
                max_new_tokens=1,
                block_size=1,
            )

        with self.assertRaisesRegex(ValueError, "upstream maximum"):
            dflash_full_prefix_greedy(
                lambda input_ids: torch.zeros((*input_ids.shape, 2)),
                lambda prefix_ids, proposal_limit: [0],
                [0],
                max_new_tokens=1,
                block_size=17,
            )

    def test_acceptance_diagnostic_caps_explicit_k_at_fifteen(self) -> None:
        self.assertEqual(
            parse_proposal_counts("1,3,5,7,15"),
            (1, 3, 5, 7, 15),
        )
        with self.assertRaisesRegex(ValueError, r"\[1,15\]"):
            parse_proposal_counts("16")


if __name__ == "__main__":
    unittest.main()
