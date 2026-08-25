from __future__ import annotations

import unittest

import torch

from qwen35_mtp.backends import MainEvaluation
from qwen35_mtp.benchmark import BenchmarkConfig, run_benchmark


class ResettableMain:
    backend_id = "resettable-main"

    def __init__(self):
        self.reset_calls = 0

    def reset_benchmark_state(self):
        self.reset_calls += 1

    def evaluate(self, input_ids, top1_positions):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        values = [
            (int(input_ids[0, position]) * 3 + 1) % 97
            for position in top1_positions
        ]
        return MainEvaluation(
            hidden_states=hidden,
            top1_token_ids=torch.tensor([values], dtype=torch.long),
        )


class ResettableDraft:
    backend_id = "resettable-draft"

    def __init__(self):
        self.reset_calls = 0

    def reset_state(self):
        self.reset_calls += 1

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
        for _index in range(max_draft_tokens):
            token = (token * 3 + 1) % 97
            result.append(token)
        return result


class UnstableMain:
    backend_id = "unstable-main"

    def __init__(self):
        self.calls = 0

    def evaluate(self, input_ids, top1_positions):
        self.calls += 1
        hidden = input_ids.float().unsqueeze(-1)
        value = self.calls % 2
        return MainEvaluation(
            hidden_states=hidden,
            top1_token_ids=torch.tensor(
                [[value for _position in top1_positions]], dtype=torch.long
            ),
        )


class BenchmarkTest(unittest.TestCase):
    def test_mtp_benchmark_warms_resets_syncs_and_retains_raw_samples(self):
        main = ResettableMain()
        draft = ResettableDraft()
        sync_calls = []

        report = run_benchmark(
            main,
            [2, 5],
            draft=draft,
            config=BenchmarkConfig(
                mode="mtp",
                warmup=1,
                repetitions=3,
                max_new_tokens=5,
                max_draft_tokens=2,
            ),
            synchronize=lambda: sync_calls.append(None),
            synchronization_source="test-sync",
        )

        self.assertEqual(report["status"], "PASS")
        self.assertEqual(len(report["measurements"]), 3)
        self.assertEqual(report["summary"]["count"], 3)
        self.assertEqual(report["synchronization"]["source"], "test-sync")
        self.assertEqual(report["state_reset_hooks"]["main"], "reset_benchmark_state")
        self.assertEqual(report["state_reset_hooks"]["draft"], "reset_state")
        self.assertEqual(main.reset_calls, 4)
        self.assertEqual(draft.reset_calls, 4)
        self.assertEqual(len(sync_calls), 8)
        token_runs = [item["generated_token_ids"] for item in report["measurements"]]
        self.assertTrue(all(tokens == token_runs[0] for tokens in token_runs))

    def test_repetitions_with_different_tokens_fail(self):
        with self.assertRaisesRegex(RuntimeError, "different token IDs"):
            run_benchmark(
                UnstableMain(),
                [1],
                config=BenchmarkConfig(
                    mode="ordinary",
                    warmup=0,
                    repetitions=2,
                    max_new_tokens=1,
                ),
            )

    def test_invalid_counts_fail_before_execution(self):
        with self.assertRaisesRegex(ValueError, "repetitions must be positive"):
            run_benchmark(
                ResettableMain(),
                [1],
                config=BenchmarkConfig(mode="ordinary", repetitions=0),
            )


if __name__ == "__main__":
    unittest.main()
