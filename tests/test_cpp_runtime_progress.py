from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.cli import build_parser
from qwen35_dflash.ascend310p import cpp_runtime
from qwen35_dflash.ascend310p.cpp_runtime import (
    INCREMENTAL_STATE_POLICY,
    RECOMPUTE_STATE_POLICY,
    _execute_streaming,
    preflight_cpp_runner,
)
from qwen35_dflash.ascend310p.workflow import (
    DEFAULT_CPP_GRAPH_FACTORY,
    QUANT_RECOMPUTE_GRAPH_FACTORY,
    run_cpp_target_pipeline,
)


def _infer_cpp_args(*extra: str) -> list[str]:
    return [
        "infer-cpp",
        "--deployment-manifest",
        "/tmp/deployment.json",
        "--runner",
        "/tmp/runner",
        "--runner-config",
        "/tmp/runner.json",
        "--model-dir",
        "/tmp/model",
        "--prompt",
        "hello",
        "--output",
        "/tmp/report.json",
        *extra,
    ]


def test_infer_cpp_progress_is_on_by_default_and_can_be_disabled() -> None:
    parser = build_parser()
    assert parser.parse_args(_infer_cpp_args()).progress is True
    assert parser.parse_args(_infer_cpp_args("--no-progress")).progress is False


def test_run_e2e_cpp_defaults_to_fused_four_om_factory() -> None:
    args = build_parser().parse_args(
        [
            "run-e2e-cpp",
            "--factory-config",
            "/tmp/factory.json",
            "--bundle-dir",
            "/tmp/bundle",
            "--soc-version",
            "Ascend310P3",
            "--runner",
            "/tmp/incremental-runner",
            "--runner-config",
            "/tmp/runner.json",
            "--model-dir",
            "/tmp/model",
            "--prompt",
            "hello",
            "--report-dir",
            "/tmp/report",
        ]
    )

    assert args.factory == DEFAULT_CPP_GRAPH_FACTORY
    assert args.factory.endswith("create_quant_fused_speculative_step_graphs")


def test_runner_preflight_rejects_the_wrong_binary_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = tmp_path / "runner"
    runner.write_text("fake", encoding="utf-8")
    runner.chmod(0o755)

    def fake_help(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=(
                "Usage: qwen35_dflash_acl_runner [options]\n"
                "  --model PATH\n"
                "  --progress true|false\n"
            ),
        )

    monkeypatch.setattr(cpp_runtime.subprocess, "run", fake_help)
    assert preflight_cpp_runner(
        runner, state_policy=RECOMPUTE_STATE_POLICY
    ) == runner.resolve()
    with pytest.raises(RuntimeError, match="wrong runner family"):
        preflight_cpp_runner(runner, state_policy=INCREMENTAL_STATE_POLICY)


@pytest.mark.parametrize(
    ("factory", "state_policy"),
    [
        (DEFAULT_CPP_GRAPH_FACTORY, RECOMPUTE_STATE_POLICY),
        (QUANT_RECOMPUTE_GRAPH_FACTORY, INCREMENTAL_STATE_POLICY),
    ],
)
def test_cpp_pipeline_rejects_factory_runner_topology_mismatch_before_export(
    factory: str, state_policy: str
) -> None:
    with pytest.raises(ValueError, match="different OM topologies"):
        run_cpp_target_pipeline(
            factory=factory,
            factory_config={},
            bundle_dir="/not-reached/bundle",
            soc_version="Ascend310P3",
            atc_bin=None,
            atc_args=[],
            runner="/not-reached/runner",
            runner_options={
                "device_model": "Ascend310P3",
                "cann": "fake-cann",
                "driver": "fake-driver",
                "firmware": "fake-firmware",
                "runtime": "fake-AscendCL",
                "state_policy": state_policy,
            },
            report_dir="/not-reached/report",
            prompt="hello",
        )


def test_recompute_runner_command_does_not_receive_incremental_only_options(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_root = tmp_path / "run"
    bundle = run_root / "bundle"
    bundle.mkdir(parents=True)
    monkeypatch.setenv("AI_RUN_DIR", str(run_root))
    air_manifest = bundle / "air-manifest.json"
    air_manifest.write_text("{}", encoding="utf-8")
    om = bundle / "quant_dflash_recompute.om"
    om.write_bytes(b"recompute")
    deployment = bundle / "deployment-manifest.json"
    deployment.write_text(
        json.dumps(
            {
                "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
                "status": "PASS",
                "air_manifest": {
                    "path": air_manifest.name,
                    "sha256": hashlib.sha256(
                        air_manifest.read_bytes()
                    ).hexdigest(),
                },
                "compiler": {"identity": "fake"},
                "target": {"soc_version": "Ascend310P3"},
                "graphs": [
                    {
                        "name": "quant_dflash_recompute",
                        "role": "generation-recompute",
                        "input_names": ["input_ids", "attention_mask"],
                        "output_names": ["target_top1", "draft_top1"],
                        "om": {
                            "path": om.name,
                            "sha256": hashlib.sha256(
                                om.read_bytes()
                            ).hexdigest(),
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    runner = run_root / "recompute-runner"
    runner.write_text("fake", encoding="utf-8")
    runner.chmod(0o755)
    policies: list[str | None] = []

    def fake_preflight(
        _: str | Path, *, state_policy: str | None = None
    ) -> Path:
        policies.append(state_policy)
        return runner

    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", fake_preflight)
    monkeypatch.setattr(cpp_runtime, "validate_cpp_runner_report", lambda *_, **__: None)
    captured: dict[str, list[str]] = {}

    def execute(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        captured["command"] = command
        Path(command[command.index("--output") + 1]).write_text(
            "{}", encoding="utf-8"
        )
        return subprocess.CompletedProcess(command, 0, stdout="fake PASS\n")

    cpp_runtime.run_cpp_pair(
        deployment_manifest=deployment,
        runner=runner,
        runner_options={
            "device_model": "Ascend310P3",
            "cann": "fake-cann",
            "driver": "fake-driver",
            "firmware": "fake-firmware",
            "runtime": "fake-AscendCL",
            "state_policy": RECOMPUTE_STATE_POLICY,
            "graph_name": "quant_dflash_recompute",
            "pad_token_id": 0,
        },
        prompt_token_ids=[10],
        eos_token_ids=[],
        device_id=0,
        max_new_tokens=2,
        max_draft_tokens=1,
        raw_output=run_root / "out" / "raw.json",
        log_output=run_root / "log" / "runner.log",
        progress=False,
        execute=execute,
    )

    command = captured["command"]
    assert policies == [RECOMPUTE_STATE_POLICY]
    assert "--model" in command
    assert "--measurement-protocol" not in command
    assert "--state-reset-policy" not in command


def test_streaming_executor_tees_child_output(tmp_path, capsys) -> None:
    log = tmp_path / "runner.log"
    result = _execute_streaming(
        [
            sys.executable,
            "-c",
            "print('[qwen35-dflash] stage=one', flush=True); "
            "print('[qwen35-dflash] stage=two', flush=True)",
        ],
        log_path=log,
        echo=True,
    )

    assert result.returncode == 0
    assert "stage=one" in result.stdout
    assert "stage=two" in result.stdout
    assert log.read_text(encoding="utf-8") == result.stdout
    captured = capsys.readouterr()
    assert "stage=one" in captured.err
    assert "stage=two" in captured.err
