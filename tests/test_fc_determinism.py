"""Host-only tests for FC A/B isolation, immutable inputs and interpretation."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.debug_draft_context import fc_determinism as fc
from tools.debug_draft_context import run as probe


def previous_probe(tmp_path):
    previous = tmp_path / "previous"
    previous.mkdir()
    array = np.linspace(-1, 1, 64 * 32, dtype="<f2").reshape(1, 64, 32)
    inp = probe.save_array(previous, "inputs/features.bin", array)
    (previous / "fc.air").write_bytes(b"HOST_TEST_AIR_FIXTURE")
    graph = {"input": inp, "air": "fc.air", "sha256": probe.digest(previous / "fc.air"),
             "outputs": [{"name": "fc_output", "dtype": "float16", "shape": [1, 64, 16],
                          "bytes": 2048, "valid_bytes": 17 * 16 * 2}]}
    probe.save_json(previous / "air.json", {"valid_rows": 17, "graphs": {"fc": graph},
        "checkpoint": {"config": {"feature_size": 32, "hidden_size": 16}, "model_sha256": "a" * 64}})
    probe.save_json(previous / "request.json", {"draft_dir": str(tmp_path / "external_weights")})
    args = SimpleNamespace(run_dir=tmp_path, probe_dir=previous, repetitions=20,
                           device_id=0, atc=None, ascendcl_root=None)
    return args, graph


def test_prepare_reuses_exact_air_and_input_without_writing_old_run(tmp_path, monkeypatch):
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    args, graph = previous_probe(tmp_path)
    before = {p.relative_to(args.probe_dir): probe.digest(p) for p in args.probe_dir.rglob("*") if p.is_file()}
    root, request = fc.prepare(args)
    after = {p.relative_to(args.probe_dir): probe.digest(p) for p in args.probe_dir.rglob("*") if p.is_file()}
    assert before == after
    assert root.parent == tmp_path and root != args.probe_dir
    assert probe.digest(root / "air/fc.air") == graph["sha256"]
    assert request["graph"]["input"]["sha256"] == graph["input"]["sha256"]
    assert request["expected_checkpoint_sha256"] == "a" * 64


@pytest.mark.parametrize("field", ["air", "input", "valid", "output"])
def test_source_tampering_or_wrong_abi_stops_before_compilation(tmp_path, field):
    args, _ = previous_probe(tmp_path)
    if field == "air":
        (args.probe_dir / "fc.air").write_bytes(b"MODIFIED")
    elif field == "input":
        (args.probe_dir / "inputs/features.bin").write_bytes(b"MODIFIED")
    else:
        path = args.probe_dir / "air.json"
        air = json.loads(path.read_text())
        if field == "valid":
            air["valid_rows"] = 0
        else:
            air["graphs"]["fc"]["outputs"][0]["valid_bytes"] = 2
        path.write_text(json.dumps(air))
    with pytest.raises(ValueError):
        fc.prepare(args)
    assert not list(tmp_path.glob("debug-fc-determinism-*"))


@pytest.mark.parametrize("mode", [0, 1])
def test_native_policy_uses_real_torch_getters_and_disallows_warn_only(mode):
    old = torch.are_deterministic_algorithms_enabled()
    old_warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
        record = fc.configure_native(torch, mode)
        assert record == {"requested": mode, "algorithms_enabled": bool(mode), "warn_only": False,
                          "pid": os.getpid()}
    finally:
        torch.use_deterministic_algorithms(old, warn_only=old_warn)


def output_case(native=0, om=0, cross_difference=0, nonfinite=0):
    return {"outputs": {"fc_output": {"native_mismatch_iterations": native, "mismatch_iterations": om,
             "native_vs_om_reference": {"changed_elements": cross_difference, "actual_nonfinite_elements": nonfinite}}}}


@pytest.mark.parametrize("off,on,expected", [
    (output_case(19, 19), output_case(), "STABLE_WITH_DETERMINISTIC"),
    (output_case(19, 19), output_case(0, 1), "VARIATION_WITH_DETERMINISTIC"),
    (output_case(19, 19), output_case(1, 0), "VARIATION_WITH_DETERMINISTIC"),
    (output_case(), output_case(), "BASELINE_NOT_REPRODUCED"),
    (output_case(19, 19), output_case(cross_difference=7), "STABLE_WITH_DETERMINISTIC"),
    (output_case(19, 19), output_case(nonfinite=1), "VARIATION_WITH_DETERMINISTIC"),
])
def test_cross_mode_rounding_is_separate_from_repeatability(off, on, expected):
    assert fc.verdict([off, on]) == expected


def test_orchestration_uses_fresh_native_and_acl_jobs_with_same_air(tmp_path, monkeypatch):
    # All program/OM records below are explicit host fixtures, not NPU execution.
    from qwen35_dflash.ascend310p import compiler
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    args, _ = previous_probe(tmp_path)
    atc = tmp_path / "atc-fixture"
    atc.write_bytes(b"HOST_TEST_NOT_ATC")
    monkeypatch.setattr(compiler, "resolve_atc_executable", lambda _: atc)
    commands = []

    def fake_call(command, log, allowed=(0,)):
        cmd = list(map(str, command))
        commands.append(cmd)
        root = log.parent.parent
        request = json.loads((root / "request.json").read_text())
        if "native" in cmd:
            mode = int(cmd[cmd.index("--mode") + 1])
            probe.save_json(root / f"native-det{mode}.json", {
                "cpu_fallback": False, "fixture": "HOST_TEST_METADATA_ONLY", "replay": {},
                "determinism": {"algorithms_enabled": bool(mode), "warn_only": False, "pid": 100 + mode}})
        elif cmd[0] == str(atc):
            prefix = next(x.split("=", 1)[1] for x in cmd if x.startswith("--output="))
            Path(prefix + ".om").write_bytes(b"HOST_TEST_NOT_OM")
        elif cmd[:2] == ["cmake", "--build"]:
            path = root / "build/gdr_debug_runner"
            path.parent.mkdir()
            path.write_bytes(b"HOST_TEST_NOT_ACL")
        elif cmd[0].endswith("gdr_debug_runner"):
            mode = int(Path(cmd[1]).stem[-1])
            path = root / "om-results" / f"fc_det{mode}" / "fc_output-reference.bin"
            path.parent.mkdir(parents=True)
            # Different stable references are not a stability failure.
            np.full(request["graph"]["outputs"][0]["shape"], mode, dtype="<f2").tofile(path)
        return 0

    monkeypatch.setattr(probe, "call", fake_call)
    monkeypatch.setattr(probe, "summarize_case", lambda root, case, *args: {
        **output_case(19 if case.endswith("0") else 0, 19 if case.endswith("0") else 0),
        "case": case, "cpu_fallback": False, "fixture": "HOST_TEST_METADATA_ONLY"})
    assert fc.run(args) == 0
    native = [c for c in commands if "native" in c]
    assert len(native) == 2 and [c[-1] for c in native] == ["0", "1"]
    assert all(c[:3] == [sys.executable, "-B", fc.__file__] for c in native)
    compiles = [c for c in commands if c[0] == str(atc)]
    assert [c[-1] for c in compiles] == ["--deterministic=0", "--deterministic=1"]
    air_arguments = [next(x for x in c if x.startswith("--model=")) for c in compiles]
    assert air_arguments[0] == air_arguments[1]
    assert all("--precision_mode=must_keep_origin_dtype" in c for c in compiles)
    assert len([c for c in commands if c[0].endswith("gdr_debug_runner")]) == 2
    root = next(tmp_path.glob("debug-fc-determinism-*"))
    report = json.loads((root / "summary.json").read_text())
    assert report["status"] == "STABLE_WITH_DETERMINISTIC"
    assert report["om_det0_vs_det1_reference"]["changed_elements"] == 17 * 16
    assert report["ordinary_parity"] == "NOT_RUN" and report["formal_latency_evidence"] is False


def test_native_child_rejects_simulation_without_touching_device(tmp_path):
    path = tmp_path / "child"
    path.mkdir()
    env = {**os.environ, "AI_RUN_DIR": str(tmp_path), "ASCEND310P_SIMULATION_ONLY": "1"}
    result = subprocess.run([sys.executable, "-B", fc.__file__, "native", "--work-dir", str(path), "--mode", "1"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 2 and "real Ascend310P" in result.stderr
    assert not list(path.iterdir())
