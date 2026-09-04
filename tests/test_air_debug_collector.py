from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile


ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / "framework" / "scripts" / "collect_air_debug.py"


def test_collector_runs_from_a_plain_source_copy_without_git(
    tmp_path: Path,
) -> None:
    factory_config = tmp_path / "factory.json"
    factory_config.write_text("{}\n", encoding="utf-8")
    output_dir = tmp_path / "reports"
    empty_bin = tmp_path / "empty-bin"
    empty_bin.mkdir()
    environment = os.environ.copy()
    environment.pop("AI_RUN_DIR", None)
    environment["PATH"] = str(empty_bin)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"

    result = subprocess.run(
        [
            sys.executable,
            str(COLLECTOR),
            "--factory-config",
            str(factory_config),
            "--output-dir",
            str(output_dir),
            "--skip-export",
        ],
        cwd=ROOT,
        env=environment,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert result.returncode == 0, result.stdout

    archives = list(output_dir.glob("air-debug-*.tar.gz"))
    assert len(archives) == 1
    with tarfile.open(archives[0], "r:gz") as stream:
        names = set(stream.getnames())
        summary_name = next(name for name in names if name.endswith("/summary.json"))
        source_name = next(
            name for name in names if name.endswith("/source-identity.json")
        )
        prototype_name = next(
            name for name in names if name.endswith("/ge-prototypes.json")
        )
        operator_name = next(
            name for name in names if name.endswith("/operator-dispatch.json")
        )
        summary_member = stream.extractfile(summary_name)
        source_member = stream.extractfile(source_name)
        prototype_member = stream.extractfile(prototype_name)
        operator_member = stream.extractfile(operator_name)
        assert summary_member is not None
        assert source_member is not None
        assert prototype_member is not None
        assert operator_member is not None
        summary = json.load(summary_member)
        source_identity = json.load(source_member)
        prototypes = json.load(prototype_member)
        operator_dispatch = json.load(operator_member)

    assert summary["status"] == "COLLECTED"
    assert summary["uses_git_metadata"] is False
    assert summary["copies_model_weights"] is False
    assert summary["export"]["status"] == "SKIPPED"
    assert prototypes["status"] == "COLLECTED"
    assert set(prototypes) == {
        "status",
        "gdr",
        "gdr_mtp",
        "adn_attention",
    }
    source_paths = {item["path"] for item in source_identity}
    assert "models/modeling_qwen3_5_hiai_nd.py" in source_paths
    assert "models/modeling_qwen3_5_hiai_nd_dflash_rollback.py" in source_paths
    operator_names = {item["name"] for item in operator_dispatch["operators"]}
    assert "npu::npu_trans_quant_param" in operator_names
    assert any(name.endswith("/source-snapshot/SOURCE_LOCK.json") for name in names)


def test_collector_has_no_git_command_dependency() -> None:
    source = COLLECTOR.read_text(encoding="utf-8")
    assert "git diff" not in source
    assert "git status" not in source
    assert "git rev-parse" not in source
