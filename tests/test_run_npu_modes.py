from __future__ import annotations

import unittest
from unittest.mock import patch

from models.dflash_v1 import run_npu


class RunNpuModeTests(unittest.TestCase):
    def test_dflash_only_mode_is_forwarded_to_shared_runner(self) -> None:
        with patch.object(run_npu, "_adapter_main", return_value=0) as adapter_main:
            status = run_npu.main(
                [
                    "--target-dir",
                    "/model/target",
                    "--draft-dir",
                    "/model/draft",
                    "--kv-cache-max-len",
                    "64",
                    "--prompt-ids",
                    "1,2",
                    "--max-new-tokens",
                    "2",
                    "--block-size",
                    "2",
                    "--execution-mode",
                    "dflash",
                    "--device",
                    "npu:0",
                    "--no-progress",
                ]
            )

        self.assertEqual(status, 0)
        forwarded = adapter_main.call_args.args[0]
        mode_index = forwarded.index("--execution-mode")
        self.assertEqual(forwarded[mode_index + 1], "dflash")
        self.assertIn("--no-progress", forwarded)


if __name__ == "__main__":
    unittest.main()
