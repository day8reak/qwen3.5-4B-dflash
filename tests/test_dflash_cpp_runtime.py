import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest
from unittest import mock

from qwen35_dflash.ascend310p.cli import build_parser
from qwen35_dflash.ascend310p.cpp_runtime import (
    CPP_RUNNER_ID,
    run_cpp_pair,
    validate_cpp_runner_report,
)


_COMPARE_PATH = (
    Path(__file__).resolve().parents[1]
    / "targets"
    / "ascend310p"
    / "scripts"
    / "compare_cpp_closed_runtime.py"
)
_COMPARE_SPEC = importlib.util.spec_from_file_location(
    "compare_cpp_closed_runtime", _COMPARE_PATH
)
assert _COMPARE_SPEC is not None and _COMPARE_SPEC.loader is not None
_COMPARE_MODULE = importlib.util.module_from_spec(_COMPARE_SPEC)
_COMPARE_SPEC.loader.exec_module(_COMPARE_MODULE)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mode_report(mode: str, tokens=None):
    stable = [11, 12, 13] if tokens is None else list(tokens)
    measurements = [
        {
            "repetition": index,
            "generated_token_ids": stable,
            "stop_reason": "length",
            "counters": {
                "graph_calls": 3,
                "drafted_tokens": 2 if mode.startswith("dflash") else 0,
                "accepted_draft_tokens": 2 if mode.startswith("dflash") else 0,
                "rejected_draft_tokens": 0,
                "decode_iterations": 2,
            },
            "latency_ms": {"prefill": 1.0, "decode": 2.0, "model_total": 3.0},
            "decode_iteration_ms": [1.0, 1.0],
        }
        for index in range(10)
    ]
    distribution = {
        "count": 10,
        "min": 1.0,
        "max": 1.0,
        "mean": 1.0,
        "median": 1.0,
        "p90": 1.0,
        "population_stdev": 0.0,
    }
    return {
        "status": "PASS",
        "generation_mode": mode,
        "warmup": 3,
        "repetitions": 10,
        "stable_generated_token_ids": stable,
        "stable_stop_reason": "length",
        "latency_ms": {
            "prefill": distribution,
            "decode": distribution,
            "model_total": distribution,
        },
        "totals": {},
        "acceptance_rate": 1.0 if mode.startswith("dflash") else 0.0,
        "generated_tokens_per_second": 100.0,
        "measurements": measurements,
    }


def _runner_report(om_hash: str, *, dflash_tokens=None):
    return {
        "schema_version": 1,
        "status": "PASS",
        "scope": "AscendCL C++ paired OM model loop",
        "runner_id": CPP_RUNNER_ID,
        "runner_version": "test",
        "cpu_fallback": False,
        "device_id": 0,
        "model": {"path": "/run/model.om", "sha256": om_hash},
        "abi": {
            "input_names": ["input_ids", "attention_mask"],
            "output_names": ["target_top1", "draft_top1"],
            "dtype": "int64",
            "sequence_length": 32,
            "draft_width": 15,
        },
        "protocol": {"warmup": 3, "repetitions": 10},
        "prompt_token_ids": [1, 2],
        "eos_token_ids": [99],
        "limits": {"max_new_tokens": 3, "max_draft_tokens": 15},
        "ordinary": _mode_report("ordinary-greedy"),
        "dflash": _mode_report("dflash-strict-greedy", dflash_tokens),
        "ordinary_parity": {
            "status": "PASS",
            "token_id_mismatches": 0,
            "eos_mismatches": 0,
        },
    }


class CppRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.run = Path(self.temporary.name).resolve()
        self.environment = mock.patch.dict(os.environ, {"AI_RUN_DIR": str(self.run)})
        self.environment.start()
        self.bundle = self.run / "out" / "bundle"
        om_dir = self.bundle / "om"
        om_dir.mkdir(parents=True)
        self.om = om_dir / "dflash_recompute.om"
        self.om.write_bytes(b"fake-om")
        self.om_hash = _sha256(self.om)
        self.manifest = self.bundle / "deployment-manifest.json"
        self.manifest.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
                    "status": "PASS",
                    "target": {"target_id": "ascend310p", "soc_version": "Ascend310P3"},
                    "compiler": {"identity": "fake-atc", "framework": 1},
                    "graphs": [
                        {
                            "name": "dflash_recompute",
                            "role": "generation-recompute",
                            "input_names": ["input_ids", "attention_mask"],
                            "output_names": ["target_top1", "draft_top1"],
                            "om": {
                                "path": "om/dflash_recompute.om",
                                "bytes": self.om.stat().st_size,
                                "sha256": self.om_hash,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.runner = self.run / "build" / "qwen35_dflash_acl_runner"
        self.runner.parent.mkdir(parents=True)
        self.runner.write_text("fake runner", encoding="utf-8")
        self.runner.chmod(self.runner.stat().st_mode | stat.S_IXUSR)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def _execute(self, command, **_kwargs):
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            json.dumps(_runner_report(self.om_hash)), encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="fake runner pass\n")

    def test_cpp_pair_validates_manifest_and_adds_target_identity(self):
        payload = run_cpp_pair(
            deployment_manifest=self.manifest,
            runner=self.runner,
            runner_options={
                "device_model": "Atlas 300I Pro / Ascend310P3",
                "cann": "8.0.test",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "ascendcl-cpp-test",
                "ordinary_only": "ignored",
            },
            prompt_token_ids=[1, 2],
            eos_token_ids=[99],
            device_id=0,
            max_new_tokens=3,
            max_draft_tokens=15,
            raw_output=self.run / "out" / "cpp-raw.json",
            log_output=self.run / "log" / "cpp.log",
            execute=self._execute,
        )
        self.assertEqual(payload["ordinary_parity"]["token_id_mismatches"], 0)
        self.assertEqual(
            payload["backend_metadata"]["host_hot_path"], "AscendCL C++"
        )
        self.assertFalse(payload["backend_metadata"]["cpu_fallback"])
        self.assertEqual(
            payload["backend_metadata"]["artifacts"]["dflash_recompute"],
            self.om_hash,
        )

    def test_cpp_report_rejects_dflash_token_mismatch(self):
        report = _runner_report(self.om_hash, dflash_tokens=[11, 12, 14])
        with self.assertRaisesRegex(RuntimeError, "tokens differ"):
            validate_cpp_runner_report(
                report,
                prompt_token_ids=[1, 2],
                om_sha256=self.om_hash,
                device_id=0,
                max_new_tokens=3,
                max_draft_tokens=15,
            )

    def test_cpp_pair_rejects_generic_device_identity_before_launch(self):
        with self.assertRaisesRegex(ValueError, "concrete 310P"):
            run_cpp_pair(
                deployment_manifest=self.manifest,
                runner=self.runner,
                runner_options={
                    "device_model": "Ascend310P",
                    "cann": "8",
                    "driver": "driver",
                    "firmware": "firmware",
                    "runtime": "runtime",
                },
                prompt_token_ids=[1, 2],
                eos_token_ids=[],
                device_id=0,
                max_new_tokens=3,
                max_draft_tokens=15,
                raw_output=self.run / "out" / "bad.json",
                log_output=self.run / "log" / "bad.log",
                execute=self._execute,
            )

    def test_cli_exposes_cpp_hot_path_without_simulation_switch(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "infer-cpp",
                "--deployment-manifest",
                str(self.manifest),
                "--runner",
                str(self.runner),
                "--runner-config",
                str(self.run / "runner.json"),
                "--prompt",
                "hello",
                "--output",
                str(self.run / "out" / "report.json"),
            ]
        )
        self.assertEqual(args.max_draft_tokens, 15)
        self.assertFalse(hasattr(args, "allow_simulation"))

    def test_closed_runtime_comparison_requires_same_tokens_and_meets_explicit_gate(self):
        air_manifest = self.run / "out" / "air-manifest.json"
        air_manifest.write_text(
            json.dumps(
                {
                    "status": "PASS",
                    "graphs": [
                        {
                            "metadata": {
                                "target_checkpoint_manifest_sha256": "a" * 64,
                                "draft_checkpoint_manifest_sha256": "b" * 64,
                                "dtype": "float16",
                            }
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        cpp = _runner_report(self.om_hash)
        cpp.update(
            {
                "report_kind": "cpp-ascendcl-paired-target",
                "prompt": "hello",
                "chat": False,
                "output": {"token_ids": [11, 12, 13], "stop_reason": "length"},
                "backend_metadata": {
                    "cpu_fallback": False,
                    "device": {
                        "target_id": "ascend310p",
                        "model": "Atlas 300I Pro / Ascend310P3",
                        "device_id": 0,
                    },
                    "cann": "8.0.test",
                    "driver": "test-driver",
                    "firmware": "test-firmware",
                    "runtime": "ascendcl-cpp-test",
                },
                "control_plane": {
                    "air_manifest": {
                        "path": air_manifest.relative_to(self.run).as_posix(),
                        "bytes": air_manifest.stat().st_size,
                        "sha256": _sha256(air_manifest),
                    }
                },
            }
        )
        closed = {
            "status": "PASS",
            "device": cpp["backend_metadata"]["device"],
            "runtime_identity": {
                "cann": "8.0.test",
                "driver": "test-driver",
                "firmware": "test-firmware",
                "runtime": "closed-runtime-test",
            },
            "model_identity": {
                "target_checkpoint_manifest_sha256": "a" * 64,
                "draft_checkpoint_manifest_sha256": "b" * 64,
                "dtype": "float16",
                "sequence_length": 32,
            },
            "prompt": "hello",
            "chat": False,
            "prompt_token_ids": [1, 2],
            "output": {"token_ids": [11, 12, 13], "stop_reason": "length"},
            "measurement_protocol": {
                "warmup": 3,
                "repetitions": 10,
                "concurrency": 1,
                "model_load_excluded": True,
            },
            "latency_ms": {"model_total": {"raw": [1.0] * 10}},
        }
        result = _COMPARE_MODULE.compare(
            cpp,
            closed,
            run=self.run,
            max_median_ratio=3.1,
            max_p90_ratio=3.1,
        )
        self.assertEqual(result["status"], "PASS")
        closed["output"]["token_ids"] = [11, 12, 14]
        failed = _COMPARE_MODULE.compare(
            cpp,
            closed,
            run=self.run,
            max_median_ratio=3.1,
            max_p90_ratio=3.1,
        )
        self.assertEqual(failed["status"], "FAIL")
        self.assertIn("output.token_ids", failed["identity_or_accuracy_mismatches"])


if __name__ == "__main__":
    unittest.main()
