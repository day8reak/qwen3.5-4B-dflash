from __future__ import annotations

import unittest

import torch

from qwen35_mtp.backends import MainEvaluation
from qwen35_mtp.generation import (
    assert_exact_match,
    ordinary_generate,
    speculative_generate,
)


class ToyMainBackend:
    backend_id = "toy-main"

    @staticmethod
    def target(token: int) -> int:
        return (token * 3 + 1) % 97

    def evaluate(self, input_ids, top1_positions):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        values = [self.target(int(input_ids[0, position])) for position in top1_positions]
        top1 = torch.tensor([values], dtype=torch.long)
        return MainEvaluation(hidden_states=hidden, top1_token_ids=top1)


class ToyDraftBackend:
    backend_id = "toy-draft"

    def __init__(self, mismatch_index: int | None):
        self.mismatch_index = mismatch_index

    def propose(
        self,
        prefix_ids,
        main_hidden_states,
        max_draft_tokens,
        *,
        eos_token_ids=(),
    ):
        del main_hidden_states, eos_token_ids
        token = int(prefix_ids[0, -1])
        result = []
        for index in range(max_draft_tokens):
            token = ToyMainBackend.target(token)
            proposal = token
            if self.mismatch_index == index:
                proposal = (proposal + 7) % 97
            result.append(proposal)
            # Subsequent target proposals are based on the draft token actually
            # supplied to the target verifier.
            token = proposal
        return result


class GenerationTest(unittest.TestCase):
    def test_all_acceptance_branches_preserve_greedy_tokens(self):
        main = ToyMainBackend()
        for mismatch_index in (0, 1, None):
            with self.subTest(mismatch_index=mismatch_index):
                ordinary = ordinary_generate(
                    main, [2, 5], max_new_tokens=9, eos_token_ids=[]
                )
                mtp = speculative_generate(
                    main,
                    ToyDraftBackend(mismatch_index),
                    [2, 5],
                    max_new_tokens=9,
                    max_draft_tokens=2,
                    eos_token_ids=[],
                )
                assert_exact_match(ordinary, mtp)
                self.assertGreater(mtp.stats.drafted_tokens, 0)
                if mismatch_index == 0:
                    self.assertEqual(mtp.stats.accepted_draft_tokens, 0)
                elif mismatch_index == 1:
                    self.assertGreater(mtp.stats.accepted_draft_tokens, 0)
                    self.assertGreater(mtp.stats.rejected_draft_tokens, 0)
                else:
                    self.assertEqual(
                        mtp.stats.accepted_draft_tokens, mtp.stats.drafted_tokens
                    )

    def test_eos_and_length_limit_match(self):
        main = ToyMainBackend()
        eos = {ToyMainBackend.target(5)}
        ordinary = ordinary_generate(main, [5], max_new_tokens=8, eos_token_ids=eos)
        mtp = speculative_generate(
            main,
            ToyDraftBackend(None),
            [5],
            max_new_tokens=8,
            max_draft_tokens=2,
            eos_token_ids=eos,
        )
        assert_exact_match(ordinary, mtp)
        self.assertTrue(ordinary.reached_eos)
        self.assertEqual(len(ordinary.generated_token_ids), 1)

    def test_zero_length_generation_does_not_call_backends(self):
        main = ToyMainBackend()
        ordinary = ordinary_generate(main, [1], max_new_tokens=0)
        mtp = speculative_generate(
            main, ToyDraftBackend(None), [1], max_new_tokens=0
        )
        assert_exact_match(ordinary, mtp)
        self.assertEqual(ordinary.stats.main_calls, 0)
        self.assertEqual(mtp.stats.main_calls, 0)


if __name__ == "__main__":
    unittest.main()
