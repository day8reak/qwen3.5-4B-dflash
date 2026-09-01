from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.cli import build_parser
from qwen35_dflash.ascend310p.cpp_runtime import _execute_streaming


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
