"""Host-only compiler-policy tests; fake AIR/ATC are not NPU evidence."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from qwen35_dflash.ascend310p import cli, compiler, workflow
from qwen35_dflash.ascend310p.utils import file_record


ORIGIN = "--precision_mode=must_keep_origin_dtype"
ORIGIN_V2 = "--precision_mode_v2=origin"


@pytest.mark.parametrize("extra", [[], ["--log=info", "--op_debug_level=2"]])
def test_default_precision_is_explicit_idempotent_and_does_not_mutate(extra):
    before = list(extra)
    result = compiler.validate_atc_args(extra)
    assert extra == before
    assert result == [*extra, ORIGIN]
    assert compiler.validate_atc_args(result) == result


@pytest.mark.parametrize("argument,canonical", [
    (ORIGIN, ORIGIN),
    (ORIGIN_V2, ORIGIN_V2),
    ("--precision-mode=must_keep_origin_dtype", ORIGIN),
    ("--precision-mode-v2=origin", ORIGIN_V2),
    ("--precision_mode-v2=origin", ORIGIN_V2),
])
def test_only_one_explicit_origin_mode_is_retained(argument, canonical):
    assert compiler.validate_atc_args(["--log=info", argument]) == ["--log=info", canonical]


@pytest.mark.parametrize("argument", [
    "--precision_mode=force_fp16",
    "--precision_mode=allow_fp32_to_fp16",
    "--precision_mode=allow_mix_precision",
    "--precision_mode=force_fp32",
    "--precision_mode_v2=fp16",
    "--precision_mode_v2=mixed_float16",
    "--precision_mode_v2=cube_fp16in_fp32out",
    "--precision-mode=force_fp16",
    "--precision-mode-v2=fp16",
    "--precision_mode=",
    "--precision_mode_v2=",
    "--precision_mode",
    "--precision_mode_v2",
    "--precision_mode=must_keep_origin_dtype ",
    "--precision_mode_v2=origin=fp16",
])
def test_changing_or_omitting_graph_precision_is_rejected(argument):
    with pytest.raises(ValueError, match="preserve_graph_dtypes"):
        compiler.validate_atc_args([argument])


@pytest.mark.parametrize("arguments", [
    [ORIGIN, ORIGIN],
    [ORIGIN_V2, ORIGIN_V2],
    [ORIGIN, ORIGIN_V2],
    [ORIGIN_V2, ORIGIN],
    [ORIGIN, "--precision-mode=must_keep_origin_dtype"],
])
def test_duplicate_and_mixed_precision_flags_are_rejected(arguments):
    with pytest.raises(ValueError, match="mutually exclusive"):
        compiler.validate_atc_args(arguments)


@pytest.mark.parametrize("argument", [
    "--customize_dtypes=override.cfg",
    "--customize-dtypes=override.cfg",
    "--input_fp16_nodes=recurrent_state",
    "--input-fp16-nodes=recurrent_state",
])
def test_higher_priority_dtype_overrides_cannot_bypass_origin_policy(argument):
    with pytest.raises(ValueError, match="conflicts with preserve_graph_dtypes"):
        compiler.validate_atc_args([ORIGIN, argument])


@pytest.mark.parametrize("argument", [
    "--soc-version=Ascend310P1",
    "--dynamic-dims=16;64",
    "--input-shape=state:1,32",
    "--output-type=FP16",
])
def test_hyphenated_atc_options_cannot_override_core_contract(argument):
    with pytest.raises(ValueError, match="core option"):
        compiler.validate_atc_args([argument])


def _bundle(tmp_path, monkeypatch, *, dynamic=False):
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    air = bundle / "verify.air"
    air.write_bytes(b"host policy test only: not a real AIR model")
    audit = {
        "status": "PASS", "torch_target": "npu.npu_gated_delta_rule_mtp.default",
        "ge_op_type": "GatedDeltaRuleMTP", "minimum_occurrences": 24,
        "converter_policy": "framework-registered-ge-ir",
        "converter_calls": 24, "ge_node_occurrences": 24,
    }
    graph = {
        "name": "target-verify-commit", "role": "target-verify-commit",
        "dynamic": dynamic, "input_dim_gears": {},
        "air": file_record(air, relative_to=bundle),
        "payload_files": [file_record(air, relative_to=bundle)],
        "custom_op_audit": [audit],
    }
    if dynamic:
        graph["torchair_external_weight_mapping"] = {
            "status": "PASS", "required": True, "policy": "data-index-v1",
            "mapping_key": "GraphDef Data.index == runtime input index",
            "converter_calls": 1, "used_weight_inputs": 1,
            "converted_weight_inputs": 1,
        }
    manifest = bundle / "air-manifest.json"
    manifest.write_text(json.dumps({
        "schema_version": 3, "status": "PASS",
        "artifact_kind": "qwen35-dflash-torchair-bundle", "graphs": [graph],
    }), encoding="utf-8")
    return manifest, graph


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize("extra,expected", [
    ([], ORIGIN),
    ([ORIGIN], ORIGIN),
    ([ORIGIN_V2], ORIGIN_V2),
    (["--precision-mode-v2=origin"], ORIGIN_V2),
])
def test_compile_preserves_mtp_audit_and_records_actual_precision(
    tmp_path, monkeypatch, dynamic, extra, expected,
):
    manifest, graph = _bundle(tmp_path, monkeypatch, dynamic=dynamic)
    manifest_before = manifest.read_bytes()
    graph_before = copy.deepcopy(graph)
    commands = []

    def fake_atc(command, cwd):
        commands.append(list(command))
        output = next(a.split("=", 1)[1] for a in command if a.startswith("--output="))
        Path(output + ".om").write_bytes(b"host test only: not a real OM")
        return subprocess.CompletedProcess(command, 0, stdout="policy fixture")

    result = compiler.compile_air_bundle(
        manifest, soc_version="Ascend310P3", atc_bin=sys.executable,
        extra_args=extra, runner=fake_atc, atc_identity="TEST_DOUBLE",
    )
    assert len(commands) == 1
    assert [a for a in commands[0] if a.startswith("--precision")] == [expected]
    assert manifest.read_bytes() == manifest_before
    assert graph == graph_before
    assert result["compiler"]["extra_args"] == [expected]
    assert result["compiler"]["precision_policy"] == "preserve_graph_dtypes"
    deployed = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert deployed["compiler"] == result["compiler"]
    assert deployed["graphs"][0]["atc_command"] == commands[0]
    assert deployed["graphs"][0]["custom_op_audit"] == graph["custom_op_audit"]


def test_compile_failure_never_retries_with_lower_precision(tmp_path, monkeypatch):
    manifest, _ = _bundle(tmp_path, monkeypatch)
    commands = []

    def unsupported_dtype(command, cwd):
        commands.append(list(command))
        return subprocess.CompletedProcess(command, 1, stdout="unsupported original dtype")

    with pytest.raises(compiler.AtcCompileError, match="no precision fallback attempted"):
        compiler.compile_air_bundle(
            manifest, soc_version="Ascend310P3", atc_bin=sys.executable,
            runner=unsupported_dtype, atc_identity="TEST_DOUBLE",
        )
    assert len(commands) == 1 and commands[0][-1] == ORIGIN
    assert not (manifest.parent / "deployment-manifest.json").exists()
    assert (tmp_path / "log/dflash-atc/target-verify-commit.log").read_text() == (
        "unsupported original dtype"
    )


def test_compile_rejects_bad_precision_before_atc_or_output_creation(tmp_path, monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid precision must be rejected before ATC resolution")

    monkeypatch.setattr(compiler, "resolve_atc_executable", unexpected)
    with pytest.raises(ValueError, match="preserve_graph_dtypes"):
        compiler.compile_air_bundle(
            tmp_path / "bundle/air-manifest.json", soc_version="Ascend310P3",
            extra_args=["--precision_mode=force_fp16"],
        )
    assert not list(tmp_path.iterdir())


def test_build_rejects_bad_precision_before_checkpoint_loading(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid precision must be rejected before toolchain/export work")

    monkeypatch.setattr(cli, "resolve_atc_executable", unexpected)
    monkeypatch.setattr(cli, "export_air_bundle", unexpected)
    with pytest.raises(ValueError, match="preserve_graph_dtypes"):
        cli.command_build(SimpleNamespace(atc_arg=["--precision_mode_v2=fp16"]))


@pytest.mark.parametrize("cpp", [False, True])
def test_e2e_rejects_bad_precision_before_checkpoint_loading(tmp_path, monkeypatch, cpp):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid precision must be rejected before toolchain/export work")

    monkeypatch.setattr(workflow, "preflight_target_pipeline", unexpected)
    monkeypatch.setattr(workflow, "preflight_cpp_runner", unexpected)
    monkeypatch.setattr(workflow, "export_air_bundle", unexpected)
    common = {
        "factory": "unused", "factory_config": {},
        "bundle_dir": tmp_path / "bundle", "report_dir": tmp_path / "report",
        "soc_version": "Ascend310P3", "atc_bin": None,
        "atc_args": ["--precision_mode=allow_fp32_to_fp16"], "prompt": "unused",
    }
    with pytest.raises(ValueError, match="preserve_graph_dtypes"):
        if cpp:
            workflow.run_cpp_target_pipeline(**common, runner="unused", runner_options={})
        else:
            workflow.run_target_pipeline(
                **common, backend_factory="unused",
                ordinary_backend_options={}, dflash_backend_options={},
            )
    assert not list(tmp_path.iterdir())
