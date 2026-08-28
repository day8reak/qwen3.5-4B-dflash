from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class BenchmarkSourceLockTests(unittest.TestCase):
    def _assert_locked_files(
        self,
        section: dict[str, object],
        pairs: tuple[tuple[str, str], ...],
    ) -> None:
        for path_key, hash_key in pairs:
            path_value = section[path_key]
            expected_hash = section[hash_key]
            self.assertIsInstance(path_value, str)
            self.assertIsInstance(expected_hash, str)
            path = REPOSITORY / str(path_value)
            with self.subTest(path=path_value):
                self.assertTrue(path.is_file())
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, expected_hash)

    def test_package_layout_files_match_source_lock(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        layout = lock["package_layout"]
        pairs = (
            ("hiai_source_file", "hiai_source_sha256"),
            ("hiai_bridge_file", "hiai_bridge_sha256"),
            ("namespace_compatibility_file", "namespace_compatibility_sha256"),
            ("compatibility_entry_file", "compatibility_entry_sha256"),
        )
        self._assert_locked_files(layout, pairs)

    def test_npu_runtime_and_target_quant_files_match_source_lock(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        runtime = lock["npu_embedded_runtime"]
        pairs = (
            ("runner_file", "runner_sha256"),
            ("loader_file", "loader_sha256"),
            ("loader_contract_file", "loader_contract_sha256"),
            ("hiai_source_check_file", "hiai_source_check_sha256"),
            ("hiai_runtime_file", "hiai_runtime_sha256"),
            ("ascend_ops_file", "ascend_ops_sha256"),
            ("target_quant_contract_file", "target_quant_contract_sha256"),
            ("original_quant_file", "original_quant_sha256"),
            ("target_quant_preflight_file", "target_quant_preflight_sha256"),
            ("w8a8_emulation_file", "w8a8_emulation_sha256"),
            ("w8a8_cpu_validator_file", "w8a8_cpu_validator_sha256"),
        )
        self._assert_locked_files(runtime, pairs)

    def test_v1_runtime_files_match_source_lock(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        runtime = lock["v1_runtime"]
        pairs = (
            ("package_init_file", "package_init_sha256"),
            ("config_file", "config_sha256"),
            ("ops_file", "ops_sha256"),
            ("adapter_file", "adapter_sha256"),
            ("scheduler_file", "scheduler_sha256"),
            ("draft_model_file", "draft_model_sha256"),
            ("weight_loader_file", "weight_loader_sha256"),
            ("acceptance_diagnostic_file", "acceptance_diagnostic_sha256"),
        )
        self._assert_locked_files(runtime, pairs)

    def test_benchmark_files_match_source_lock(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        benchmark = lock["npu_benchmark"]
        pairs = (
            ("runner_file", "runner_sha256"),
            ("msprof_wrapper_file", "msprof_wrapper_sha256"),
            ("contract_file", "contract_sha256"),
            ("documentation_file", "documentation_sha256"),
        )
        self._assert_locked_files(benchmark, pairs)

    def test_rollback_runtime_files_match_source_lock(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        runtime = lock["rollback_runtime"]
        pairs = (
            ("runner_file", "runner_sha256"),
            ("adapter_file", "adapter_sha256"),
            ("scheduler_file", "scheduler_sha256"),
            ("hiai_modeling_file", "hiai_modeling_sha256"),
            ("wrapper_file", "wrapper_sha256"),
            ("shared_qlinear_source_file", "shared_qlinear_source_sha256"),
        )
        self._assert_locked_files(runtime, pairs)

    def test_lock_declares_rollback_target_only_quant_scope(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        self.assertEqual(lock["schema_version"], 6)
        self.assertIn("Target-only W8A8", lock["purpose"])
        self.assertIn("original multi-token GDR", lock["rollback_runtime"]["policy"])
        self.assertIn("Draft stays FP16", lock["rollback_runtime"]["policy"])


if __name__ == "__main__":
    unittest.main()
