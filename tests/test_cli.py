from __future__ import annotations

from types import SimpleNamespace
import unittest

from qwen35_mtp.cli import _benchmark_synchronizer, build_parser


class CliTest(unittest.TestCase):
    def test_compare_accepts_deterministic_token_ids(self):
        args = build_parser().parse_args(
            [
                "compare",
                "--model-dir",
                "/checkpoint",
                "--prompt-token-ids",
                "1,2",
            ]
        )
        self.assertEqual(args.prompt_token_ids, "1,2")
        self.assertEqual(args.max_draft_tokens, 2)

    def test_prompt_and_token_ids_are_mutually_exclusive(self):
        with self.assertRaises(SystemExit):
            build_parser().parse_args(
                [
                    "ordinary",
                    "--model-dir",
                    "/checkpoint",
                    "--prompt",
                    "hello",
                    "--prompt-token-ids",
                    "1,2",
                ]
            )

    def test_benchmark_defaults_to_frozen_target_measurement_counts(self):
        args = build_parser().parse_args(
            [
                "benchmark",
                "--mode",
                "mtp",
                "--model-dir",
                "/checkpoint",
                "--prompt-token-ids",
                "1,2",
                "--device",
                "npu:0",
                "--output",
                "/run/out/performance/mtp.json",
            ]
        )
        self.assertEqual(args.warmup, 3)
        self.assertEqual(args.repetitions, 10)
        self.assertEqual(args.dtype, "float16")
        self.assertFalse(hasattr(args, "allow_op_fallback"))

    def test_cpu_backend_hook_does_not_bypass_simulation_approval(self):
        backend = SimpleNamespace(synchronize=lambda: None)
        args = SimpleNamespace(device="cpu", allow_cpu_simulation=False)
        with self.assertRaisesRegex(RuntimeError, "simulation only"):
            _benchmark_synchronizer(args, backend, None)


if __name__ == "__main__":
    unittest.main()
