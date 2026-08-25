from __future__ import annotations

import os
from pathlib import Path
import unittest

from qwen35_mtp.config import OFFICIAL_QWEN35_4B, Qwen35MTPConfig
from qwen35_mtp.weights import audit_checkpoint


class CheckpointContractTest(unittest.TestCase):
    def test_locked_checkpoint_metadata(self):
        raw = os.environ.get("QWEN35_MODEL_DIR")
        if not raw:
            self.skipTest("QWEN35_MODEL_DIR is not set")
        model_dir = Path(raw)
        report = audit_checkpoint(model_dir)
        self.assertEqual(report["status"], "PASS", report["errors"])
        self.assertEqual(report["actual_mtp_tensor_count"], 15)
        config = Qwen35MTPConfig.from_pretrained(model_dir).to_dict()
        for key, expected in OFFICIAL_QWEN35_4B.items():
            self.assertEqual(config[key], expected)


if __name__ == "__main__":
    unittest.main()
