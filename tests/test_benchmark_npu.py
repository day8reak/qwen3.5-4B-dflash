from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from models.dflash_v1.benchmark_npu import (
    BenchmarkConfig,
    BenchmarkInvocation,
    _benchmark_range_label,
    _dflash_invocation,
    _draft_kv_cache_invocation_audit,
    _mstx_range_factory,
    _ordinary_invocation,
    _target_audit_delta,
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
                "qwen35_ordinary_warmup_0",
                "qwen35_ordinary_measure_0",
                "qwen35_ordinary_measure_1",
            ],
        )
        self.assertEqual(report["target_forward_calls"], 6)
        self.assertEqual(report["summary"]["count"], 2)
        self.assertGreater(
            report["summary"]["aggregate_output_tokens_per_second"],
            0,
        )

    def test_mstx_labels_use_conservative_characters_and_close_ranges(self) -> None:
        calls: list[tuple[str, object]] = []
        ended: list[int] = []
        fake_mstx = SimpleNamespace(
            range_start=lambda label, stream: (
                calls.append((label, stream)) or 17
            ),
            range_end=lambda range_id: ended.append(range_id),
        )
        with patch.dict(sys.modules, {"mstx": fake_mstx}):
            factory, source = _mstx_range_factory(True)
            assert factory is not None
            label = _benchmark_range_label("dflash", "measure", 3)
            with factory(label):
                pass

        self.assertEqual(label, "qwen35_dflash_measure_3")
        self.assertRegex(label, r"^[A-Za-z0-9_]+$")
        self.assertEqual(calls, [(label, None)])
        self.assertEqual(ended, [17])
        self.assertEqual(source, "mstx.range_start/range_end")

    def test_mstx_marker_failure_explains_no_msproftx_fallback(self) -> None:
        fake_mstx = SimpleNamespace(
            range_start=lambda _label, _stream: 0,
            range_end=lambda _range_id: None,
        )
        with patch.dict(sys.modules, {"mstx": fake_mstx}):
            factory, _ = _mstx_range_factory(True)
            assert factory is not None
            with self.assertRaisesRegex(RuntimeError, "--no-msproftx"):
                with factory("qwen35_dflash_warmup_0"):
                    pass

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

    def test_session_cursor_reset_is_not_treated_as_counter_regression(self) -> None:
        counter_fields = (
            "ordinary_prefill_token_calls",
            "ordinary_decode_calls",
        )
        before = {
            "cumulative_counter_fields": counter_fields,
            "ordinary_prefill_token_calls": 100,
            "ordinary_decode_calls": 30,
            "persistent_mode": "rollback",
            "persistent_cursor": 128,
        }
        after = {
            "cumulative_counter_fields": counter_fields,
            "ordinary_prefill_token_calls": 112,
            "ordinary_decode_calls": 62,
            "persistent_mode": "ordinary",
            "persistent_cursor": 44,
        }

        delta = _target_audit_delta(before, after)

        self.assertEqual(delta["ordinary_prefill_token_calls"], 12)
        self.assertEqual(delta["ordinary_decode_calls"], 32)
        self.assertEqual(delta["target_execution_calls"], 44)
        self.assertEqual(
            delta["session_state_after"],
            {"persistent_mode": "ordinary", "persistent_cursor": 44},
        )

    def test_real_audit_counter_regression_still_fails_closed(self) -> None:
        before = {
            "cumulative_counter_fields": ("rollback_verify_calls",),
            "rollback_verify_calls": 9,
            "persistent_cursor": 128,
        }
        after = {
            "cumulative_counter_fields": ("rollback_verify_calls",),
            "rollback_verify_calls": 8,
            "persistent_cursor": 44,
        }
        with self.assertRaisesRegex(
            RuntimeError,
            "counter moved backwards: rollback_verify_calls",
        ):
            _target_audit_delta(before, after)

    def test_draft_cache_invocation_reports_counter_delta_and_state(self) -> None:
        before = {
            "rounds": 4,
            "aborted_rounds": 1,
            "crop_calls": 0,
            "tokens_appended": 20,
            "tokens_reused": 30,
            "context_tokens": 12,
        }
        after = {
            "rounds": 6,
            "aborted_rounds": 1,
            "crop_calls": 0,
            "tokens_appended": 25,
            "tokens_reused": 54,
            "context_tokens": 17,
        }

        audit = _draft_kv_cache_invocation_audit(before, after)

        self.assertEqual(audit["context_tokens"], 17)
        self.assertEqual(
            audit["invocation_counter_delta"],
            {
                "rounds": 2,
                "aborted_rounds": 0,
                "crop_calls": 0,
                "tokens_appended": 5,
                "tokens_reused": 24,
            },
        )

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
