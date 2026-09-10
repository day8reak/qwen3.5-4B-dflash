"""Host compiler/manifest regression; fake ATC is not device evidence."""
import json
from pathlib import Path
import subprocess

import pytest

from test_incremental_air_om import chunk_bundle  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401
from qwen35_dflash.ascend310p.compiler import (
    AtcCompileError, _graph_atc_args, compile_air_bundle, recompile_draft_om,
)
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


def test_full_bundle_enables_draft_only(chunk_bundle):
    deployment = json.loads(chunk_bundle.read_text())
    for graph in deployment["graphs"]:
        flags = [x for x in graph["atc_command"] if x.startswith("--deterministic")]
        assert flags == (["--deterministic=1"] if graph["name"] == "draft" else [])
        assert "--precision_mode=must_keep_origin_dtype" in graph["atc_command"]
        assert all(x in graph["atc_command"] for x in deployment["compiler"]["graph_extra_args"][graph["name"]])


@pytest.mark.parametrize("flags", [["--deterministic=0"], ["--deterministic=1"]])
def test_explicit_diagnostic_mode_is_not_silently_overridden(flags):
    assert _graph_atc_args(flags, name="draft", incremental=True) == flags
    assert _graph_atc_args(flags, name="target_verify", incremental=True) == flags
    assert _graph_atc_args([], name="draft", incremental=False) == []


@pytest.mark.parametrize("flags", [["--deterministic"], ["--deterministic=true"],
                                  ["--deterministic=1", "--deterministic=0"]])
def test_ambiguous_mode_stops_full_bundle_before_any_atc(chunk_bundle, flags):
    calls = []
    with pytest.raises(ValueError, match="deterministic"):
        compile_air_bundle(chunk_bundle.parent / "air-manifest.json", soc_version="Ascend310P3",
            atc_bin="/bin/true", extra_args=flags, runner=lambda *x: calls.append(x), atc_identity="HOST_TEST")
    assert not calls


def frozen_files(directory):
    return {path: path.read_bytes() for path in directory.rglob("*") if path.is_file()}


@pytest.mark.parametrize("old_mode", [None, "--deterministic=0", "--deterministic=1"])
def test_recompile_only_draft_reuses_targets_and_creates_loadable_manifest(chunk_bundle, tmp_path, old_mode):
    old = json.loads(chunk_bundle.read_text())
    draft = next(graph for graph in old["graphs"] if graph["name"] == "draft")
    draft["atc_command"] = [s for s in draft["atc_command"] if not s.startswith("--deterministic")]
    draft["atc_command"].append("--log=info")
    if old_mode:
        draft["atc_command"].append(old_mode)
    chunk_bundle.write_text(json.dumps(old))
    before = frozen_files(chunk_bundle.parent)
    output = chunk_bundle.with_name("deployment-manifest-deterministic.json")
    calls = []

    def fake_atc(command, cwd):
        calls.append((command, cwd))
        prefix = Path(next(x.split("=", 1)[1] for x in command if x.startswith("--output=")))
        Path(str(prefix) + ".om").write_bytes(b"HOST_ONLY_DETERMINISTIC_DRAFT_OM")
        return subprocess.CompletedProcess(command, 0, "host test ATC")

    result = recompile_draft_om(chunk_bundle, output=output, atc_bin="/bin/true",
                                runner=fake_atc, atc_identity="HOST_TEST")
    assert len(calls) == 1
    command, cwd = calls[0]
    assert command.count("--deterministic=1") == 1 and "--deterministic=0" not in command
    assert "--precision_mode=must_keep_origin_dtype" in command and "--log=info" in command
    assert f"--model={chunk_bundle.parent / draft['air']['path']}" in command
    assert cwd == (chunk_bundle.parent / draft['air']['path']).parent
    assert all(path.read_bytes() == contents for path, contents in before.items())
    for original, current in zip(old["graphs"], result["graphs"], strict=True):
        if original["name"] != "draft":
            assert original == current
        else:
            assert original["metadata"] == current["metadata"]
            assert original["air"] == current["air"]
            assert original["om"]["path"] != current["om"]["path"]
            assert current["om"]["sha256"] == sha256_file(chunk_bundle.parent / current["om"]["path"])
    assert result["recompilation"]["parent_manifest"]["sha256"] == sha256_file(chunk_bundle)
    assert result["recompilation"]["ordinary_parity"] == "NOT_RUN"
    assert result["recompilation"]["formal_latency_evidence"] is False
    plan, loaded, _ = write_incremental_plan(output, tmp_path / "new-plan.txt")
    assert plan.read_text().count("\ngraph ") == 4
    assert loaded["graphs"] == result["graphs"]
    with pytest.raises(FileExistsError):
        recompile_draft_om(chunk_bundle, output=output, atc_bin="/bin/true", runner=fake_atc)
    assert len(calls) == 1


@pytest.mark.parametrize("damage", ["target_om", "air_payload", "air_manifest", "metadata", "outside_output"])
def test_recompile_rejects_invalid_reuse_before_atc(chunk_bundle, tmp_path, damage):
    deployment = json.loads(chunk_bundle.read_text())
    draft = next(graph for graph in deployment["graphs"] if graph["name"] == "draft")
    output = chunk_bundle.with_name("deployment-new.json")
    if damage == "target_om":
        (chunk_bundle.parent / deployment["graphs"][0]["om"]["path"]).write_bytes(b"corrupt")
    elif damage == "air_payload":
        (chunk_bundle.parent / draft["air"]["path"]).write_bytes(b"corrupt")
    elif damage == "air_manifest":
        (chunk_bundle.parent / deployment["air_manifest"]["path"]).write_text("{}")
    elif damage == "metadata":
        draft["metadata"]["unrecorded_change"] = True
        chunk_bundle.write_text(json.dumps(deployment))
    else:
        output = tmp_path / "different-root.json"
    calls = []
    with pytest.raises(ValueError):
        recompile_draft_om(chunk_bundle, output=output, atc_bin="/bin/true",
                          runner=lambda *x: calls.append(x), atc_identity="HOST_TEST")
    assert not calls and not output.exists()


def test_atc_failure_retains_original_bundle_and_publishes_no_manifest(chunk_bundle):
    before = frozen_files(chunk_bundle.parent)
    output = chunk_bundle.with_name("deployment-failed.json")

    def fail(command, cwd):
        return subprocess.CompletedProcess(command, 255, "Unsupported_Operator(EZ3002): test failure")

    with pytest.raises(AtcCompileError, match="test failure"):
        recompile_draft_om(chunk_bundle, output=output, atc_bin="/bin/true", runner=fail, atc_identity="HOST_TEST")
    assert not output.exists()
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert list(chunk_bundle.parent.glob("draft-deterministic-*/log/draft.log"))


def test_cli_accepts_draft_recompile_without_requiring_export_or_soc(monkeypatch):
    from qwen35_dflash.ascend310p.cli import build_parser, command_recompile_draft
    monkeypatch.delenv("SOC_VERSION", raising=False)
    args = build_parser().parse_args(["recompile-draft-om", "--deployment-manifest", "original.json",
                                      "--output", "new.json", "--atc", "/declared/atc"])
    assert args.handler is command_recompile_draft
    assert args.atc == Path("/declared/atc")
