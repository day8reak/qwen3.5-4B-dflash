"""Frozen Draft boundary replay through real C++ control code and fake ACL."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import numpy as np

from test_incremental_air_om import chunk_bundle, assert_cpp_resources_released  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file
from tools.analyze_draft_replay import analyze, metrics

SOURCE = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


@pytest.fixture
def replay_case(chunk_bundle, tmp_path, monkeypatch):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to fake ACL runner")
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    monkeypatch.delenv("TEST_ACTIVE", raising=False)
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "plan.txt", mode="dflash")
    events, cleanup = tmp_path / "acl-events.jsonl", tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_EVENT_LOG", str(events))
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))

    def invoke(name="report", policy="shared", prompt_count=17, count=15, inputs=None, extra=()):
        report = tmp_path / (name + ".json")
        command = [
            runner, "--model-kind", "chunk", "--mode", "dflash",
            "--model", str(plan), "--model-sha256", sha256_file(plan),
            "--prompt-token-ids", ",".join(["4"] * prompt_count),
            "--max-draft-tokens", str(count), "--output", str(report),
            "--debug-draft-replay", "4", "--debug-draft-workspace", policy,
        ]
        if inputs:
            command += ["--debug-draft-inputs", str(inputs)]
        proc = subprocess.run(command + list(extra), capture_output=True, text=True, timeout=30)
        data = json.loads(report.read_text()) if report.exists() else None
        rows = []
        trace = Path(str(report) + ".replay") / "iterations.jsonl"
        if trace.exists():
            rows = [json.loads(line) for line in trace.read_text().splitlines()]
        return proc, data, rows

    return invoke, events, cleanup


@pytest.mark.parametrize("policy", ["shared", "private"])
@pytest.mark.parametrize("prompt_count,count", [(17, 15), (65, 2)])
def test_replay_freezes_boundary_never_commits_outputs(replay_case, policy, prompt_count, count):
    invoke, events, cleanup = replay_case
    proc, data, rows = invoke(policy=policy, prompt_count=prompt_count, count=count)
    assert proc.returncode == 0, proc.stderr
    assert data["fake_acl"] is True and not data["formal_latency_evidence"]
    assert not data["draft_outputs_committed"] and data["ordinary_parity"] == "NOT_RUN"
    assert data["actual_workspace_policy"] == ("shared_serial" if policy == "shared" else "per_model")
    completed = [r for r in rows if r["event"] == "completed"]
    assert len(rows) == 16 and len(completed) == 8
    assert [r["phase"] for r in completed] == ["isolated"] * 4 + ["interleaved_prefill"] * 4
    for row in completed:
        assert row["status"] == "PASS" and row["inputs_unchanged"]
        assert row["input_sha256"] == row["input_after_sha256"] == data["snapshot_sha256"]
        assert row["valid_tokens_match"] and row["all_output_bytes_match"]
        assert row["valid_kv_matches"]
        assert len(row["output_token_ids"]) == count
        assert row["first_token_difference"] is None and not row["profiled"]
        for name, address in row["input_device_addresses"].items():
            if name.startswith("d"):
                assert row["output_device_addresses"][name] != address
    assert all(r["input_device_addresses"] == completed[0]["input_device_addresses"]
               for r in completed[:4])
    snapshot = Path(data["input_directory"])
    for name, digest in data["snapshot_sha256"].items():
        assert hashlib.sha256((snapshot / (name + ".bin")).read_bytes()).hexdigest() == digest
    audit = data["kv_output_audit"]
    assert audit["layout"] == "B,H,S,D" and audit["saved_differences"] == []
    assert audit["row_regions"]["valid_prefix"] == [0, prompt_count]
    # This fixture's request capacity is 128; physical KV has an extra 64 rows.
    assert audit["row_regions"]["untouched_tail"][1] == 192
    report = snapshot.parent.with_suffix("")
    analysis = analyze(report)
    assert analysis["examples"] == [] and analysis["fake_acl"]
    assert all(phase["valid_kv_mismatch_iterations"] == 0 for phase in analysis["phase_counts"])
    calls = [json.loads(line) for line in events.read_text().splitlines()]
    assert not any(row[1] or row[0] in ("target_verify", "target_decode") for row in calls)
    # Interleaved setup runs Prefill again; a long prompt also initializes old Draft KV.
    prefix = ["target_prefill"] if prompt_count == 17 else ["target_prefill", "draft", "target_prefill"]
    assert [r[0] for r in calls] == prefix + ["draft"] * 4 + (prefix + ["draft"]) * 4
    assert_cpp_resources_released(cleanup, proc.stderr)


@pytest.mark.parametrize("variation,field", [
    ("draft_rejected_tail", "token_mismatch_iterations"),
    ("draft_input_mutation", "input_mutation_iterations"),
    ("draft_output_bytes", "valid_kv_mismatch_iterations"),
    ("draft_output_padding", "full_output_hash_mismatch_iterations"),
    ("draft_output_tail", "full_output_hash_mismatch_iterations"),
])
def test_replay_retains_drift_and_runs_remaining_iterations(replay_case, monkeypatch, variation, field):
    invoke, _, cleanup = replay_case
    monkeypatch.setenv("QWEN35_FAKE_PROFILE_VARIATION", variation)
    proc, data, rows = invoke()
    # Valid KV drift fails even with identical tokens. Padding remains diagnostic.
    padding_only = variation in ("draft_output_padding", "draft_output_tail")
    assert proc.returncode == (0 if padding_only else 1), proc.stderr
    completed = [r for r in rows if r["event"] == "completed"]
    assert len(completed) == 8
    assert completed[2]["input_matches_snapshot"]
    assert data["phases"][0][field] == 1
    assert data["phases"][1][field] == 0
    if variation == "draft_rejected_tail":
        assert completed[2]["first_token_difference"]["index"] == 7
        assert completed[2]["inputs_unchanged"]
    elif variation == "draft_input_mutation":
        assert not completed[2]["inputs_unchanged"]
        assert completed[3]["inputs_unchanged"]  # Restored, not carried into next call.
    else:
        region, row, channel = {
            "draft_output_bytes": ("valid_prefix", 0, 1),
            "draft_output_padding": ("written_padding", 17, 0),
            "draft_output_tail": ("untouched_tail", 64, 0),
        }[variation]
        assert completed[2]["valid_tokens_match"] and completed[2]["inputs_unchanged"]
        assert completed[2]["valid_kv_matches"] is padding_only
        changes = data["phases"][0]["kv_region_mismatch_iterations"]
        assert changes["d0_key"] == {r: int(r == region) for r in data["kv_output_audit"]["row_regions"]}
        report = Path(data["trace"]).parent.parent / "report.json"
        analysis = analyze(report)
        assert len(analysis["examples"]) == 1
        example = analysis["examples"][0]
        assert example["region"] == region
        diff = example["comparison"]
        assert diff["changed_elements"] == 1 and diff["max_fp16_ulp_finite"] == 1
        assert diff["max_abs_finite"] == 2 ** -24
        assert diff["first_difference"]["coordinate_bhsd"] == [0, 0, row, channel]
        assert example["outside_write_vs_input"]["untouched_tail"]["changed_elements"] == int(row == 64)
        # Evidence is hash locked; corruption must not yield numerical results.
        saved = Path(data["trace"]).parent / example["path"]
        raw = bytearray(saved.read_bytes())
        raw[0] ^= 1
        saved.write_bytes(raw)
        with pytest.raises(ValueError, match="hash mismatch"):
            analyze(report)
    assert_cpp_resources_released(cleanup, proc.stderr)


def test_snapshot_import_is_exact_and_checks_damage(replay_case, monkeypatch):
    invoke, _, _ = replay_case
    _, original, _ = invoke(name="original")
    inputs = Path(original["input_directory"])
    # The second process's Prefill features differ; import must override them.
    monkeypatch.setenv("QWEN35_FAKE_PROFILE_VARIATION", "prefill_features")
    proc, imported, _ = invoke(name="imported", policy="private", inputs=inputs)
    assert proc.returncode == 0, proc.stderr
    assert imported["snapshot_sha256"] == original["snapshot_sha256"]
    proc, data, _ = invoke(name="wrong-contract", inputs=inputs, count=2)
    assert proc.returncode != 0 and data is None and "contract" in proc.stderr
    path = inputs / "features.bin"
    raw = bytearray(path.read_bytes())
    raw[2] ^= 1
    path.write_bytes(raw)
    proc, data, _ = invoke(name="damaged", inputs=inputs)
    assert proc.returncode != 0 and data is None and "hash mismatch" in proc.stderr


def test_replay_never_overwrites_previous_evidence(replay_case):
    invoke, _, _ = replay_case
    proc, original, _ = invoke()
    assert proc.returncode == 0
    proc, retained, _ = invoke()
    assert proc.returncode != 0 and "report must be new" in proc.stderr
    assert retained == original


@pytest.mark.parametrize("extra", [
    ("--warmup", "2"),  # Formal benchmark requirements stay intact.
    ("--profile-stage", "draft"),
])
def test_debug_cannot_relax_benchmark_or_mix_with_profiler(replay_case, extra):
    invoke, events, _ = replay_case
    proc, data, _ = invoke(extra=extra)
    assert proc.returncode != 0 and data is None
    assert not events.exists()


@pytest.mark.parametrize("variation,query_fail", [
    ("", False), ("draft_rejected_tail", False), ("draft_private_output", False),
    ("draft_private_kv", False), ("", True),
])
def test_wrapper_compares_same_snapshot_in_two_processes(
    chunk_bundle, tmp_path, monkeypatch, variation, query_fail,
):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER")
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    monkeypatch.setenv("QWEN35_FAKE_PROFILE_VARIATION", variation)
    if query_fail:
        monkeypatch.setenv("QWEN35_FAKE_QUERY_SIZE_FAIL", "1")
    stubs = tmp_path / "stubs"
    stubs.mkdir()
    # The existing plan CLI imports CPU Torch; it runs and exits before ACL.
    # No Python NPU or pyACL execution belongs to this diagnostic.
    for module in ("torch_npu", "acl"):
        (stubs / (module + ".py")).write_text('raise AssertionError("debug CLI must not load device Python")\n')
    monkeypatch.setenv("PYTHONPATH", str(stubs))
    proc = subprocess.run([
        sys.executable, "-B", str(SOURCE / "tools/debug_draft_om.py"),
        "--run-dir", str(tmp_path), "--runner", runner,
        "--deployment-manifest", str(chunk_bundle), "--prompt-token-ids", "4,4,4",
        "--repetitions", "4", "--workspace", "both",
    ], text=True, capture_output=True, timeout=40)
    assert proc.returncode == (1 if variation or query_fail else 0), proc.stdout + proc.stderr
    output = next(tmp_path.glob("debug-draft-*/comparison.json"))
    comparison = json.loads(output.read_text())
    assert len(comparison["runs"]) == 2
    assert comparison["same_frozen_inputs_across_processes"]
    assert comparison["distinct_workspace_policies_exercised"] is (not query_fail)
    assert comparison["reference_tokens_match_across_processes"] is (variation != "draft_private_output")
    assert comparison["reference_valid_kv_match_across_processes"] is (variation != "draft_private_kv")
    if variation in ("draft_private_output", "draft_private_kv"):
        # Both processes are internally stable, but disagree for identical inputs.
        assert all(row["status"] == "PASS_REPLAY_CHECKS" for row in comparison["runs"])
        assert comparison["status"] == "FAIL_OR_INCOMPLETE"
    for row in comparison["runs"]:
        assert row["fake_acl"] is True
        assert json.loads(Path(row["kv_analysis"]).read_text())["fake_acl"] is True
        report = json.loads(Path(row["report"]).read_text())
        trace = [json.loads(line) for line in Path(report["trace"]).read_text().splitlines()]
        assert len([r for r in trace if r["event"] == "completed"]) == 8


def test_kv_metrics_slice_each_head_and_keep_physical_coordinates():
    reference = np.zeros((2, 3, 80, 4), dtype="<f2")
    actual = reference.copy()
    actual[0, 0, 60, 0] = 10  # Outside [5,17), despite a small flat offset.
    actual[1, 2, 9, 3] = 1
    result = metrics(reference, actual, 5, 17)
    assert result["elements"] == 2 * 3 * 12 * 4
    assert result["changed_elements"] == 1 and result["max_abs_finite"] == 1
    assert result["mean_abs_finite"] == 1 / result["elements"]
    assert result["first_difference"]["coordinate_bhsd"] == [1, 2, 9, 3]
    empty = metrics(reference, actual, 17, 17)
    assert empty["elements"] == 0 and empty["first_difference"] is None
    assert empty["max_abs_finite"] is None


def test_kv_metrics_fp16_bit_distance_and_nonfinite_json():
    reference = np.array([1, -1, 0, 0], dtype="<f2").reshape(1, 1, 1, 4)
    actual = np.nextafter(reference, np.float16(np.inf))
    actual[0, 0, 0, 3] = -0.0
    result = metrics(reference, actual, 0, 1)
    assert result["changed_elements"] == 4 and result["max_fp16_ulp_finite"] == 1
    assert result["max_abs_finite"] == 2 ** -10
    reference.fill(np.inf)
    actual.fill(np.nan)
    result = metrics(reference, actual, 0, 1)
    assert result["reference_nonfinite_elements"] == result["actual_nonfinite_elements"] == 4
    assert result["first_difference"]["reference"] is None
    assert result["first_difference"]["actual"] is None
    assert result["max_abs_finite"] is None and result["max_fp16_ulp_finite"] is None
    json.dumps(result, allow_nan=False)


def test_analyzer_requires_output_evidence(tmp_path):
    report = tmp_path / "old.json"
    report.write_text('{"schema_version": 1}')
    with pytest.raises(ValueError, match="no KV output snapshots"):
        analyze(report)
