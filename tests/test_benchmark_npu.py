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
from models.dflash_v1.dflash_reference_decode_v1 import (
    ReplayDecodeResult,
    ReplayDecodeStats,
)
from models.dflash_v1.dflash_rollback_adapter import Qwen35RollbackAdapterStats


def _result(tokens: tuple[int, ...]) -> ReplayDecodeResult:
    return ReplayDecodeResult(
        mode="test",
        prompt_token_ids=(1,),
        generated_token_ids=tokens,
        reached_eos=False,
        stop_reason="max_new_tokens",
        stats=ReplayDecodeStats(target_calls=len(tokens)),
    )


def _logits(tokens: list[int], vocab_size: int = 32) -> torch.Tensor:
    result = torch.full((1, len(tokens), vocab_size), -1000.0)
    for row, token in enumerate(tokens):
        result[0, row, token % vocab_size] = 1.0
    return result


class _DeterministicRollbackAdapter:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.target = self
        self.stats = Qwen35RollbackAdapterStats()
        self.audit = {
            "ordinary_prefill_token_calls": 0,
            "ordinary_decode_calls": 0,
            "rollback_prefill_token_calls": 0,
            "rollback_verify_calls": 0,
        }

    @property
    def dflash_rollback_audit(self):
        return dict(self.audit)

    def reset_rollback_stats(self) -> None:
        self.stats = Qwen35RollbackAdapterStats()

    def snapshot_rollback_stats(self) -> Qwen35RollbackAdapterStats:
        return replace(self.stats)

    def begin_ordinary(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        self.stats.ordinary_prefill_calls += 1
        self.audit["ordinary_prefill_token_calls"] += int(prompt_ids.shape[1])
        return _logits([int(prompt_ids[0, -1]) + 1])

    def advance_ordinary(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.stats.ordinary_decode_calls += 1
        self.audit["ordinary_decode_calls"] += 1
        return _logits([int(input_ids[0, -1]) + 1])

    def begin_rollback(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        self.stats.rollback_prefill_calls += 1
        self.audit["rollback_prefill_token_calls"] += int(prompt_ids.shape[1])
        return _logits([int(prompt_ids[0, -1]) + 1])

    def propose_rollback(
        self,
        prefix_ids: torch.Tensor,
        proposal_limit: int,
    ) -> torch.Tensor:
        anchor = int(prefix_ids[0, -1])
        self.stats.draft_calls += 1
        self.stats.proposed_tokens += proposal_limit
        return torch.tensor(
            [[anchor + index + 1 for index in range(proposal_limit)]],
            dtype=torch.long,
        )

    def verify_rollback(self, block_ids: torch.Tensor) -> torch.Tensor:
        self.stats.rollback_verify_calls += 1
        self.audit["rollback_verify_calls"] += 1
        return _logits([int(token) + 1 for token in block_ids[0]])

    def commit_rollback(self, accepted_draft_tokens: int) -> None:
        self.stats.rollback_commit_calls += 1
        self.stats.rollback_committed_input_tokens += accepted_draft_tokens + 1

    def abort_rollback(self) -> None:
        raise AssertionError("deterministic benchmark adapter must not abort")


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
                adapter_stats={
                    "ordinary_prefill_calls": 1,
                    "ordinary_decode_calls": 1,
                },
            ),
            expected=expected,
            config=BenchmarkConfig(
                mode="ordinary",
                warmup=1,
                repetitions=2,
                max_new_tokens=3,
                block_size=4,
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
                "qwen35/ordinary/warmup/0",
                "qwen35/ordinary/measure/0",
                "qwen35/ordinary/measure/1",
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
                    adapter_stats={"ordinary_prefill_calls": 1},
                ),
                expected=expected,
                config=BenchmarkConfig(
                    mode="ordinary",
                    warmup=1,
                    repetitions=1,
                    max_new_tokens=2,
                    block_size=2,
                ),
                synchronize=lambda: None,
                synchronization_source="test",
            )

    def test_rollback_invocations_match_and_report_audit_delta(self) -> None:
        adapter = _DeterministicRollbackAdapter()
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
            block_size=4,
            eos_token_ids=(),
        )
        self.assertEqual(ordinary.result.generated_token_ids, (2, 3, 4, 5))
        self.assertEqual(
            ordinary.result.generated_token_ids,
            dflash.result.generated_token_ids,
        )
        self.assertEqual(ordinary.target_forward_calls, 4)
        self.assertEqual(dflash.target_forward_calls, 2)
        self.assertEqual(dflash.adapter_stats["rollback_verify_calls"], 1)
        self.assertGreater(dflash.adapter_stats["draft_calls"], 0)

    def test_invalid_configuration_is_rejected(self) -> None:
        invalid = (
            BenchmarkConfig("ordinary", warmup=-1),
            BenchmarkConfig("ordinary", repetitions=0),
            BenchmarkConfig("ordinary", max_new_tokens=1),
            BenchmarkConfig("ordinary", block_size=1),
            BenchmarkConfig("ordinary", block_size=17),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()


if __name__ == "__main__":
    unittest.main()
