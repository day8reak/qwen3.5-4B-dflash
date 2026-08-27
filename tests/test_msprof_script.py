from __future__ import annotations

import os
from pathlib import Path
import subprocess
import tempfile
import unittest


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "tools" / "run_msprof.sh"


class MsprofScriptTests(unittest.TestCase):
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
