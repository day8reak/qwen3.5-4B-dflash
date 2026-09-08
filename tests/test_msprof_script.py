from __future__ import annotations

import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools" / "run_msprof.sh"
RUN_DOCUMENT = REPOSITORY / "docs" / "DFLASH_RUN_AND_VALIDATE.md"


class MsprofScriptTests(unittest.TestCase):
    def test_documented_ordinary_profile_uses_the_declared_model_entry(self) -> None:
        document = RUN_DOCUMENT.read_text(encoding="utf-8")
        commands = [block for block in re.findall(r"```bash\n(.*?)```", document, re.S)
                    if "--profile-mode ordinary" in block]
        self.assertTrue(commands, "ordinary profiling needs an executable example")
        for command in commands:
            with self.subTest(command=command):
                self.assertIn("--profile-backend python", command)
                self.assertIn("--profile-stage", command)
                self.assertIn('"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu', command)
                self.assertIn('"${NPU_ARGS[@]}" "${QUANT_ARGS[@]}"', command)
        self.assertTrue(any("--profile-stage all" in command for command in commands))
        self.assertTrue(any("PROFILE_STAGE=decode" in command for command in commands))

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
