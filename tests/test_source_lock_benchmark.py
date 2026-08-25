from __future__ import annotations

import hashlib
import json
from pathlib import Path
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]


class BenchmarkSourceLockTests(unittest.TestCase):
    def test_benchmark_files_match_source_lock(self) -> None:
        lock = json.loads((REPOSITORY / "SOURCE_LOCK.json").read_text("utf-8"))
        benchmark = lock["npu_benchmark"]
        pairs = (
            ("runner_file", "runner_sha256"),
            ("msprof_wrapper_file", "msprof_wrapper_sha256"),
            ("contract_file", "contract_sha256"),
            ("documentation_file", "documentation_sha256"),
        )
        for path_key, hash_key in pairs:
            with self.subTest(path=benchmark[path_key]):
                path = REPOSITORY / benchmark[path_key]
                self.assertTrue(path.is_file())
                actual = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(actual, benchmark[hash_key])


if __name__ == "__main__":
    unittest.main()
