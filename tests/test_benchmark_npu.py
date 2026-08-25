from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import unittest

import torch

from models.dflash_v1.benchmark_npu import (
    BenchmarkConfig,
    BenchmarkInvocation,
    _dflash_invocation,
    _ordinary_invocation,
    run_benchmark,
)
from models.dflash_v1.dflash_qwen_adapter_v1 import (
    Qwen35FullPrefixAdapterStats,
)
from models.dflash_v1.dflash_reference_decode_v1 import (
    ReplayDecodeResult,
    ReplayDecodeStats,
)


def _result(tokens: tuple[int, ...]) -> ReplayDecodeResult:
    return ReplayDecodeResult(
        mode="test",
        prompt_token_ids=(1,),
        generated_token_ids=tokens,
        reached_eos=False,
        stop_reason="max_new_tokens",
        stats=ReplayDecodeStats(target_calls=len(tokens)),
    )


class _DeterministicAdapter:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.stats = Qwen35FullPrefixAdapterStats()

    def reset_stats(self) -> None:
        self.stats = Qwen35FullPrefixAdapterStats()

    def snapshot_stats(self) -> Qwen35FullPrefixAdapterStats:
        return replace(self.stats)

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        vocab_size = 32
        logits = torch.full(
            (1, input_ids.shape[1], vocab_size),
            -1000.0,
            dtype=torch.float32,
        )
        for index, token in enumerate(input_ids[0].tolist()):
            logits[0, index, (int(token) + 1) % vocab_size] = 1.0
        self.stats.target_logit_calls += 1
        self.stats.target_logit_tokens_recomputed += int(input_ids.shape[1])
        return logits

    def propose(self, prefix_ids: torch.Tensor, limit: int) -> torch.Tensor:
        start = int(prefix_ids[0, -1].item())
        values = [(start + offset + 1) % 32 for offset in range(limit)]
        self.stats.target_feature_calls += 1
        self.stats.target_feature_tokens_recomputed += max(
            0, int(prefix_ids.shape[1]) - 1
        )
        self.stats.draft_calls += 1
        self.stats.proposed_tokens += limit
        return torch.tensor([values], dtype=torch.long)


class BenchmarkHarnessTests(unittest.TestCase):
    def test_synchronized_measurements_and_ranges(self) -> None:
        expected = _result((2, 3, 4))
        sync_calls: list[int] = []
        ranges: list[str] = []

        @contextmanager
        def record_range(label: str):
            ranges.append(label)
            yield

        report = run_benchmark(
            lambda: BenchmarkInvocation(
                result=expected,
                adapter_stats={"target_logit_calls": 1, "target_feature_calls": 1},
            ),
            expected=expected,
            config=BenchmarkConfig(
                mode="dflash",
                warmup=1,
                repetitions=2,
                max_new_tokens=3,
                max_draft_tokens=2,
            ),
            synchronize=lambda: sync_calls.append(1),
            synchronization_source="test",
            range_factory=record_range,
            range_source="test-range",
        )

        self.assertEqual(len(sync_calls), 6)
        self.assertEqual(
            ranges,
            [
                "qwen35/dflash/warmup/0",
                "qwen35/dflash/measure/0",
                "qwen35/dflash/measure/1",
            ],
        )
        self.assertEqual(report["target_forward_calls"], 6)
        self.assertEqual(report["summary"]["count"], 2)
        self.assertGreater(
            report["summary"]["aggregate_output_tokens_per_second"],
            0,
        )

    def test_output_change_fails_closed(self) -> None:
        expected = _result((2, 3))
        results = iter((_result((2, 3)), _result((2, 9))))
        with self.assertRaisesRegex(RuntimeError, "output changed"):
            run_benchmark(
                lambda: BenchmarkInvocation(
                    result=next(results),
                    adapter_stats={"target_logit_calls": 1},
                ),
                expected=expected,
                config=BenchmarkConfig(
                    mode="ordinary",
                    warmup=1,
                    repetitions=1,
                    max_new_tokens=2,
                    max_draft_tokens=1,
                ),
                synchronize=lambda: None,
                synchronization_source="test",
            )

    def test_current_dflash_route_matches_ordinary(self) -> None:
        adapter = _DeterministicAdapter()
        ordinary = _ordinary_invocation(
            adapter,
            [1],
            max_new_tokens=4,
            eos_token_ids=(),
        )
        dflash = _dflash_invocation(
            adapter,
            [1],
            max_new_tokens=4,
            max_draft_tokens=2,
            eos_token_ids=(),
        )
        self.assertEqual(
            ordinary.result.generated_token_ids,
            dflash.result.generated_token_ids,
        )
        self.assertEqual(dflash.result.generated_token_ids, (2, 3, 4, 5))
        self.assertGreater(dflash.adapter_stats["draft_calls"], 0)
        self.assertGreater(dflash.adapter_stats["target_feature_calls"], 0)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            BenchmarkConfig("ordinary", warmup=-1),
            BenchmarkConfig("ordinary", repetitions=0),
            BenchmarkConfig("ordinary", max_new_tokens=1),
            BenchmarkConfig("ordinary", max_draft_tokens=17),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
