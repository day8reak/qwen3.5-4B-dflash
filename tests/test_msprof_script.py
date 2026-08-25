from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest


MODEL_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = MODEL_ROOT / "targets/ascend310p/scripts/run_msprof.sh"


class MsprofScriptTest(unittest.TestCase):
    def _environment(self, root: Path) -> tuple[dict[str, str], Path]:
        run_dir = root / "run"
        run_dir.mkdir()
        fake_msprof = root / "msprof"
        fake_msprof.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env python3
                from pathlib import Path
                import subprocess
                import sys

                if sys.argv[1:] == ["--version"]:
                    print("fake-msprof 1.0")
                    raise SystemExit(0)
                arguments = sys.argv[1:]
                output = next(
                    item.split("=", 1)[1]
                    for item in arguments
                    if item.startswith("--output=")
                )
                profile = Path(output) / "PROF_FAKE"
                profile.mkdir(parents=True)
                command_index = next(
                    index for index, item in enumerate(arguments) if not item.startswith("-")
                )
                raise SystemExit(subprocess.run(arguments[command_index:]).returncode)
                """
            ),
            encoding="utf-8",
        )
        fake_msprof.chmod(0o755)
        preflight = root / "preflight.sh"
        preflight.write_text("#!/usr/bin/env bash\necho real-device-preflight\n", encoding="utf-8")
        preflight.chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "AI_RUN_DIR": str(run_dir),
                "AI_MODEL_ROOT": str(MODEL_ROOT),
                "AI_MODEL_PYTHON": sys.executable,
                "AI_TARGET_PROFILE": str(root / "real-ascend310p-profile"),
                "AI_TARGET_PROFILE_ID": "ascend310p-test",
                "AI_TARGET_PREFLIGHT": str(preflight),
            }
        )
        return environment, fake_msprof

    def test_wraps_application_and_writes_run_scoped_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment, fake_msprof = self._environment(root)
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--label",
                    "ordinary-pipe",
                    "--msprof-bin",
                    str(fake_msprof),
                    "--",
                    sys.executable,
                    "-c",
                    "print('target-ran')",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_path = (
                Path(environment["AI_RUN_DIR"])
                / "out/performance/msprof/ordinary-pipe.manifest.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "PASS")
            self.assertFalse(manifest["target"]["cpu_fallback_allowed"])
            self.assertEqual(manifest["application"][0], sys.executable)
            self.assertIn("--msproftx=on", manifest["msprof"]["arguments"])
            self.assertEqual(len(manifest["model_source"]["source_tree_sha256"]), 64)
            self.assertGreater(manifest["model_source"]["source_files"], 0)
            self.assertTrue(
                (
                    Path(environment["AI_RUN_DIR"])
                    / "profile/msprof/ordinary-pipe/PROF_FAKE"
                ).is_dir()
            )

    def test_rejects_simulation_profile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment, fake_msprof = self._environment(root)
            environment["AI_TARGET_PROFILE"] = str(root / "simulation")
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--label",
                    "invalid",
                    "--msprof-bin",
                    str(fake_msprof),
                    "--",
                    sys.executable,
                    "-c",
                    "pass",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("simulation-only", result.stderr)

    def test_rejects_explicit_operator_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment, fake_msprof = self._environment(root)
            result = subprocess.run(
                [
                    str(SCRIPT),
                    "--label",
                    "invalid",
                    "--msprof-bin",
                    str(fake_msprof),
                    "--",
                    sys.executable,
                    "runner.py",
                    "--allow-op-fallback",
                ],
                env=environment,
                text=True,
                capture_output=True,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("forbidden", result.stderr)


if __name__ == "__main__":
    unittest.main()
