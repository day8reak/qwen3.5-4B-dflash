from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools" / "run_msprof.sh"
RUN_DOCUMENT = REPOSITORY / "docs" / "DFLASH_RUN_AND_VALIDATE.md"
QUANT_DOCUMENT = REPOSITORY / "docs" / "QUANT_AIR_OM_FRAMEWORK.md"


class MsprofScriptTests(unittest.TestCase):
    def test_original_main_profile_uses_external_unquantized_inference(self) -> None:
        document = RUN_DOCUMENT.read_text(encoding="utf-8")
        section = document.split("### 7.1 原 main 非 DFlash 模型", 1)[1].split(
            "### 7.2 rollback 内部 ordinary 控制组", 1
        )[0]
        command = section.split("~~~bash", 1)[1].split("~~~", 1)[0]

        self.assertIn("python3 inference.py", command)
        self.assertIn("--config ./config/qwen3.5.ymal", command)
        self.assertIn("--max_token 32", command)
        self.assertIn("--no-msproftx", command)
        self.assertNotIn("--quant_mode", command)
        self.assertNotIn("--max_token 10", document)
        self.assertNotIn("--max-new-tokens 10", document)
        self.assertIn("不在本仓库中", document)
        self.assertIn("并不是原 main 非 DFlash 模型", document)

    def test_wrapper_requires_no_git_checkout_or_vcs_metadata(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        forbidden = (
            "git -C",
            "git_commit",
            "git_branch",
            "git_dirty",
            '"repository":',
        )
        for fragment in forbidden:
            with self.subTest(fragment=fragment):
                self.assertNotIn(fragment, source)
        self.assertIn("content_hash_without_vcs_metadata", source)
        self.assertIn("copied source tree", source)
        self.assertIn('root / "framework"', source)
        self.assertIn(
            'root / "docs" / "QUANT_AIR_OM_FRAMEWORK.md"', source
        )

    def test_quant_om_profile_workflow_is_documented(self) -> None:
        document = QUANT_DOCUMENT.read_text(encoding="utf-8")
        section = document.split(
            "### 11.4 用 msprof 单独分析当前 OM", 1
        )[1].split("## 12. 常见失败定位", 1)[0]

        self.assertIn("quant_dflash_recompute.om", section)
        self.assertIn('"$DFLASH_SOURCE/tools/run_msprof.sh"', section)
        self.assertIn("--max-new-tokens 1", section)
        self.assertIn("--warmup 3", section)
        self.assertIn("--repetitions 10", section)
        self.assertIn("2 × (3 + 10) = 26", section)
        self.assertIn("PipeUtilization Memory MemoryUB", section)
        self.assertIn("msprof --query=on", section)
        self.assertIn("msprof --export=on", section)
        self.assertIn("analyze-msprof", section)
        for report in (
            "op_summary_*.csv",
            "op_statistic_*.csv",
            "api_statistic_*.csv",
            "task_time_*.csv",
        ):
            self.assertIn(report, section)
        self.assertIn("2 input/2 output ABI", section)

    def test_mstx_is_compatibility_opt_in(self) -> None:
        source = SCRIPT.read_text(encoding="utf-8")
        self.assertIn('msproftx="off"', source)
        self.assertIn("--msproftx)", source)
        self.assertIn("--no-msproftx)", source)

    def test_shell_syntax_and_help(self) -> None:
        syntax = subprocess.run(
            ["bash", "-n", str(SCRIPT)],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_result = subprocess.run(
            ["bash", str(SCRIPT), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--output-dir", help_result.stdout)
        self.assertIn("--msproftx", help_result.stdout)
        self.assertIn("default", help_result.stdout)

    def test_simulation_profile_is_rejected_before_writes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "profile"
            environment = dict(os.environ)
            environment["ASCEND310P_SIMULATION_ONLY"] = "1"
            result = subprocess.run(
                [
                    "bash",
                    str(SCRIPT),
                    "--label",
                    "simulation",
                    "--output-dir",
                    str(output),
                    "--",
                    "/bin/true",
                ],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("simulation-only", result.stderr)
            self.assertFalse(output.exists())

    def test_cpu_device_and_fallback_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            environment = dict(os.environ)
            environment.pop("ASCEND310P_SIMULATION_ONLY", None)
            base = [
                "bash",
                str(SCRIPT),
                "--label",
                "invalid",
                "--output-dir",
                str(Path(temporary) / "profile"),
                "--",
                "/bin/true",
            ]
            cpu = subprocess.run(
                [*base, "--device", "cpu"],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(cpu.returncode, 2)
            self.assertIn("non-NPU device", cpu.stderr)

            fallback = subprocess.run(
                [*base, "--allow-op-fallback"],
                env=environment,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(fallback.returncode, 2)
            self.assertIn("fallback", fallback.stderr)


if __name__ == "__main__":
    unittest.main()
