"""Probe graph/byte-contract tests; CPU and fake ACL are never NPU evidence."""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from tools.debug_draft_context import run as probe
from tools.debug_gdr.test_debug_gdr import fake_runner  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401


def weights():
    g = torch.Generator().manual_seed(37)
    return dict(zip(probe.WEIGHTS, [torch.randn(s, generator=g).half()
                                  for s in [(16, 32), (16,), (8, 16)]]))


@pytest.mark.parametrize("case", probe.CASES)
def test_probe_graph_keeps_rounding_and_fp16_linear_inputs(case, adn_rms_norm_cpu):
    w = weights()
    model = probe.module(torch, w, 1e-6, case)
    x = torch.linspace(-3, 2, 64 * (16 if case.startswith("norm_") or case == "vproj" else 32))
    x = x.reshape(1, 64, -1).half()
    fc = torch.nn.functional.linear(x, w[probe.WEIGHTS[0]]) if probe.input_name(case) == "features" else x
    norm = (fc.float() * torch.rsqrt(fc.float().square().mean(-1, keepdim=True) + 1e-6)).half() * w[probe.WEIGHTS[1]]
    expected = ((fc,) if case == "fc" else
                (torch.nn.functional.linear(x, w[probe.WEIGHTS[2]]),) if case == "vproj" else
                (norm,) if case.startswith("norm_") else
                (fc, norm, torch.nn.functional.linear(norm, w[probe.WEIGHTS[2]])))
    actual = model(x)
    for ref, got in zip(expected, actual, strict=True):
        torch.testing.assert_close(got, ref, rtol=0, atol=0)
    assert len(adn_rms_norm_cpu) == int(case.endswith("_adn"))
    if adn_rms_norm_cpu:
        call = adn_rms_norm_cpu[0]
        assert call["input_dtype"] == torch.float32
        assert torch.equal(call["gamma"], torch.ones(16))
    exported = torch.export.export(model, (x,), strict=True)
    ops = [n for n in exported.graph.nodes if n.op == "call_function"]
    norms = [n for n in ops if str(n.target) == "npu.adn_rms_norm.default"]
    assert len(norms) == int(case.endswith("_adn"))
    linears = [n for n in ops if str(n.target) == "aten.linear.default"]
    assert len(linears) == (2 if case.startswith("chain_") else 0 if case.startswith("norm_") else 1)
    assert all(arg.meta["val"].dtype == torch.float16 for n in linears for arg in n.args[:2])
    if norms:
        assert all(arg.meta["val"].dtype == torch.float32 for arg in norms[0].args[:2])


def replay_fixture(root, *, valid=17):
    directory = root / "snapshot inputs"
    directory.mkdir()
    arrays = {"features": np.ones((1, 64, 32), dtype="<f2"),
              "valid_rows": np.array([valid], dtype="<i2"),
              "start_position": np.array([0], dtype="<i8")}
    arrays["features"][:, valid:] = np.nan
    hashes = {}
    for name, array in arrays.items():
        path = directory / (name + ".bin")
        path.write_bytes(array.tobytes())
        hashes[name] = probe.digest(path)
    report = root / "private.json"
    probe.save_json(report, {"input_directory": str(directory), "snapshot_sha256": hashes,
                             "draft_om_sha256": "a" * 64})
    return report, directory


def test_frozen_replay_uses_actual_bytes_and_model_padding_mask(tmp_path):
    report, _ = replay_fixture(tmp_path)
    data, valid, source = probe.frozen_features(report, 32)
    assert valid == 17 and np.all(data[:, :valid] == 1) and not data[:, valid:].any()
    assert source["report_sha256"] == probe.digest(report)
    assert probe.input_name("norm_adn") == probe.input_name("norm_tensor") == "projected"
    assert probe.input_name("vproj") == "normalized"


@pytest.mark.parametrize("bad", ["features", "valid_rows", "start_position", "analysis"])
def test_replay_source_mismatch_fails_before_export(tmp_path, bad):
    report, directory = replay_fixture(tmp_path)
    if bad == "analysis":
        report.write_text('{"status":"ANALYZED_SAVED_BYTES"}')
    else:
        (directory / (bad + ".bin")).write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch|pass the replay"):
        probe.frozen_features(report, 32)


def test_saved_input_hash_and_abi_are_checked(tmp_path):
    array = np.ones((1, 64, 16), dtype="<f2")
    record = probe.save_array(tmp_path, "input.bin", array)
    assert np.array_equal(probe.read_array(tmp_path, record), array)
    with pytest.raises(ValueError, match="complete FP16"):
        probe.read_array(tmp_path, {**record, "shape": [1, 16, 64]})
    (tmp_path / "input.bin").write_bytes(array.tobytes()[:-2])
    with pytest.raises(ValueError, match="hash mismatch"):
        probe.read_array(tmp_path, record)


def test_native_replay_retains_first_change_and_never_feeds_back(tmp_path, monkeypatch):
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=lambda: None), raising=False)
    array = np.ones((1, 64, 16), dtype="<f2")
    inputs = []

    def model(x):
        inputs.append(x.clone())
        y = x.clone()
        if len(inputs) == 2:
            y[0, 1, 2] = 2
        return (y,)

    result = probe.native_replay(torch, model, array, ("norm_output",), 17, 4, "cpu", tmp_path, "norm_adn")
    assert result["mismatch_iterations"] == {"norm_output": 1}
    assert all(torch.equal(x, torch.ones_like(x)) for x in inputs)
    a = probe.read_array(tmp_path, result["reference"]["norm_output"])
    b = probe.read_array(tmp_path, result["first_difference"]["norm_output"])
    diff = probe.array_comparison(a, b, 17)
    assert diff["changed_elements"] == 1
    assert diff["first_difference"]["coordinate_bsf"] == [0, 1, 2]


def fake_case(tmp_path, fake_runner, monkeypatch, case="chain_adn"):
    root = tmp_path / "probe with spaces 测试"
    root.mkdir()
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    inp = probe.save_array(root, "inputs/frozen.bin", np.ones((1, 64, 16), dtype="<f2"))
    specs = [{"name": name, "shape": [1, 64, 16], "dtype": "float16", "bytes": 2048,
              "valid_bytes": 17 * 16 * 2} for name in probe.output_names(case)]
    graph = {"input": inp, "outputs": specs}
    om = root / "probe.om"
    om.write_text("FAKE_DRAFT_PROBE 16 " + str(len(specs)) + " 16" * len(specs))
    plan, out = probe.plan(root, case, graph, om, {"device_id": 0, "repetitions": 4})
    env = dict(os.environ)
    env.pop("ASCEND310P_SIMULATION_ONLY", None)
    return root, graph, [str(fake_runner), str(plan)], env, out


@pytest.mark.parametrize("case", ["norm_adn", "chain_adn"])
@pytest.mark.parametrize("flag", [None, "PROBE_TEST_UNSTABLE", "PROBE_TEST_PADDING_UNSTABLE"])
def test_acl_probe_saves_first_valid_difference_not_just_last_call(tmp_path, fake_runner, monkeypatch, case, flag):
    root, graph, command, env, out = fake_case(tmp_path, fake_runner, monkeypatch, case)
    if flag:
        env[flag] = "1"
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    unstable = flag == "PROBE_TEST_UNSTABLE"
    assert proc.returncode == (2 if unstable else 0), proc.stderr
    report = json.loads((out / "report.json").read_text())
    assert report["cpu_fallback"] is True  # This runner cannot claim NPU verification.
    assert report["stable"] == (not unstable)
    assert len(report["samples"]) == 4
    native = {"reference": {}, "samples": [None] * 4, "mismatch_iterations": {},
              "input_sha256": graph["input"]["sha256"], "first_difference": {}}
    for spec in graph["outputs"]:
        name = spec["name"]
        reference = out / (name + "-reference.bin")
        assert reference.read_bytes() == np.ones((1, 64, 16), dtype="<f2").tobytes()
        # The last call is stable: losing an earlier differing call would conceal the fault.
        assert (out / (name + ".bin")).read_bytes() == reference.read_bytes()
        assert (out / (name + "-first-diff.bin")).exists() == unstable
        native["reference"][name] = probe.save_array(root, f"native/{name}.bin", np.ones((1, 64, 16), dtype="<f2"))
        native["mismatch_iterations"][name] = 0
    summary = probe.summarize_case(root, case, graph, 17, native, 4)
    for item in summary["outputs"].values():
        assert item["mismatch_iterations"] == int(unstable)
        assert item["native_vs_om_reference"]["changed_elements"] == 0
        if unstable:
            assert item["first_changed_example"]["first_difference"]["coordinate_bsf"] == [0, 0, 1]
    with pytest.raises(ValueError, match="all unprofiled replay"):
        probe.summarize_case(root, case, graph, 17, native, 5)


def test_acl_probe_input_mutation_is_not_silent(tmp_path, fake_runner, monkeypatch):
    _, _, command, env, _ = fake_case(tmp_path, fake_runner, monkeypatch)
    env["GDR_TEST_MUTATE"] = "1"
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    assert proc.returncode == 1 and "modified declared read-only input" in proc.stderr


def test_acl_probe_rejects_actual_om_abi_mismatch(tmp_path, fake_runner, monkeypatch):
    root, _, command, env, _ = fake_case(tmp_path, fake_runner, monkeypatch)
    (root / "probe.om").write_text("FAKE_DRAFT_PROBE 8 3 16 16 16")
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    assert proc.returncode == 1 and "OM dtype/size mismatch" in proc.stderr


def test_acl_probe_rejects_profiling_environment(tmp_path, fake_runner, monkeypatch):
    _, _, command, env, _ = fake_case(tmp_path, fake_runner, monkeypatch)
    env["PROFILING_MODE"] = "dynamic"
    proc = subprocess.run(command, env=env, capture_output=True, text=True)
    assert proc.returncode == 1 and "outside msprof" in proc.stderr


def test_probe_cannot_run_inside_simulation_profile(tmp_path, monkeypatch):
    (tmp_path / "request.json").write_text("{}")
    monkeypatch.setenv("ASCEND310P_SIMULATION_ONLY", "1")
    with pytest.raises(RuntimeError, match="actual Ascend310P"):
        probe.prepare_export(tmp_path)


def test_all_command_reaches_device_preflight_in_managed_output(tmp_path):
    config = tmp_path / "factory.json"
    probe.save_json(config, {"draft_dir": str(tmp_path / "external-weights"),
                             "adn_rms_norm_ge_op_type": "AdnRmsNorm"})
    env = {**os.environ, "ASCEND310P_SIMULATION_ONLY": "1"}
    env.pop("PROFILING_MODE", None)
    proc = subprocess.run([sys.executable, "-B", str(probe.HERE / "run.py"), "all",
                           "--run-dir", str(tmp_path), "--replay-report", str(tmp_path / "private.json")],
                          env=env, capture_output=True, text=True)
    assert proc.returncode == 2 and "actual Ascend310P" in proc.stderr
    roots = list(tmp_path.glob("debug-draft-context-*"))
    assert len(roots) == 1
    request = json.loads((roots[0] / "request.json").read_text())
    assert request["rms_ge_op_type"] == "AdnRmsNorm"
    assert request["draft_dir"] == str(tmp_path / "external-weights")
    assert not (roots[0] / "air").exists()
