"""Synthetic saved-report tests; no device accuracy or latency evidence."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_dflash.ascend310p.cpp_runtime import validate_cpp_runner_report
from qwen35_dflash.ascend310p.utils import sha256_file
from tools import benchmark_prompts as suite


def round_row(prefix, proposed, target, accepted, emitted, fallback, stage="target_verify"):
    return dict(committed_prefix_length=prefix, proposed_token_ids=proposed,
                target_token_ids=target, accepted_draft_token_ids=accepted,
                emitted_token_ids=emitted, fallback_token_id=fallback, stage=stage)


def saved_report():
    draft_tokens = [6, 7, 8, 42, 43, 44, 45, 46]
    ordinary_tokens = list(range(6, 14))
    rounds = [round_row(2, [], [6], [], [6], 6, "target_prefill"),
              round_row(3, [7, 8, 99], [7, 8, 42, 43], [7, 8], [7, 8, 42], 42),
              round_row(6, [43, 44, 45, 46], [43, 44, 45, 46, 47],
                        [43, 44, 45, 46], [43, 44, 45, 46], None)]

    def benchmark(tokens, draft):
        counters = dict(drafted_tokens=7 if draft else 0, accepted_draft_tokens=6 if draft else 0,
                        rejected_draft_tokens=1 if draft else 0,
                        target_only_fallback_rounds=0, speculation_disable_events=0)
        ms = 2.0 if draft else 4.0
        measurement = dict(generated_token_ids=tokens, stop_reason="max_new_tokens",
                           latency_ms=dict(model_total=ms), counters=counters)
        if draft:
            measurement["rounds"] = rounds
        return dict(status="PASS", generation_mode="dflash-strict-greedy" if draft else "ordinary-greedy",
                    warmup=3, repetitions=10, stable_generated_token_ids=tokens,
                    stable_stop_reason="max_new_tokens", measurements=[copy.deepcopy(measurement) for _ in range(10)],
                    latency_ms=dict(model_total=dict(median=ms)), generated_tokens_per_second=len(tokens) / ms * 1000)

    return dict(schema_version=1, status="FAIL", failure_stage="ordinary_dflash_parity", error="outputs differ",
                runner_id="qwen35-dflash-ascendcl-cpp-v1", runner_version="synthetic-fixture", cpu_fallback=False,
                device_id=0, model=dict(sha256="plan-hash"), prompt_token_ids=[4, 5], eos_token_ids=[63],
                limits=dict(max_new_tokens=8, max_draft_tokens=15),
                protocol=dict(warmup=3, repetitions=10, low_memory=False, round_trace_enabled=True),
                abi=dict(id="qwen35-dflash-chunk-v3", graph_count=4),
                ordinary=benchmark(ordinary_tokens, False), dflash=benchmark(draft_tokens, True),
                ordinary_parity=dict(status="FAIL", token_id_mismatches=5, eos_mismatches=0,
                                     first_difference=dict(field="generated_token_ids", index=3,
                                                           ordinary_token_id=9, dflash_token_id=42)))


def validate(report, allow=True):
    return validate_cpp_runner_report(report, prompt_token_ids=[4, 5], om_sha256=report["model"]["sha256"],
        device_id=0, max_new_tokens=8, max_draft_tokens=15, chunk_abi=True, allow_output_differences=allow)


def test_output_difference_policy_preserves_strict_default_and_raw_failure():
    report = saved_report()
    before = copy.deepcopy(report)
    with pytest.raises(RuntimeError, match="passing known report"):
        validate(report, False)
    validate(report)
    assert report == before


@pytest.mark.parametrize("corruption", ["unstable_tokens", "unstable_stop", "few_repeats", "missing_measurement",
    "warmup", "mode_failed", "other_failure", "cpu_fallback", "wrong_device", "wrong_parity", "wrong_status"])
def test_allowing_output_difference_does_not_allow_other_failures(corruption):
    report = saved_report()
    draft = report["dflash"]
    if corruption == "unstable_tokens":
        draft["measurements"][3]["generated_token_ids"][2] = 100
    elif corruption == "unstable_stop":
        draft["measurements"][3]["stop_reason"] = "eos"
    elif corruption == "few_repeats":
        draft["repetitions"] = 9
    elif corruption == "missing_measurement":
        draft["measurements"].pop()
    elif corruption == "warmup":
        draft["warmup"] = 2
    elif corruption == "mode_failed":
        draft["status"] = "FAIL"
    elif corruption == "other_failure":
        report["failure_stage"] = "dflash_benchmark"
    elif corruption == "cpu_fallback":
        report["cpu_fallback"] = True
    elif corruption == "wrong_device":
        report["device_id"] = 1
    elif corruption == "wrong_parity":
        report["ordinary_parity"]["token_id_mismatches"] = 0
    else:
        report["status"] = "PASS"
    with pytest.raises(RuntimeError):
        validate(report)


def test_position_bins_use_whole_round_start_and_reconcile_counters():
    report = saved_report()
    bins = suite.acceptance_by_position(report, window=2)
    assert [b["generated_index_begin"] for b in bins] == [0, 4]  # First round spans output positions 1..3.
    assert bins[0]["rounds"] == 10 and bins[0]["drafted_tokens"] == 30
    assert bins[0]["accepted_draft_tokens"] == 20 and bins[0]["emitted_tokens"] == 30
    assert bins[1]["acceptance_rate"] == 1 and bins[1]["mean_proposed_tokens"] == 4
    assert sum(b["accepted_draft_tokens"] for b in bins) == 60
    report["dflash"]["measurements"][0]["counters"]["accepted_draft_tokens"] += 1
    with pytest.raises(ValueError, match="counters"):
        suite.acceptance_by_position(report)


def test_position_bins_capture_zero_acceptance_then_recovery():
    report = saved_report()
    draft = report["dflash"]
    for m in draft["measurements"]:
        m["rounds"] = [round_row(2, [], [6], [], [6], 6, "target_prefill"),
            round_row(3, [99], [7, 8], [], [7], 7),
            round_row(4, [99], [8, 42], [], [8], 8),
            round_row(5, [42, 43, 44, 45, 46], [42, 43, 44, 45, 46, 47],
                      [42, 43, 44, 45, 46], [42, 43, 44, 45, 46], None)]
        m["counters"].update(drafted_tokens=7, accepted_draft_tokens=5)
    bins = suite.acceptance_by_position(report, window=2)
    assert bins[0]["zero_accept_rate"] == 1 and bins[0]["tokens_per_round"] == 1
    assert bins[1]["zero_accept_rate"] == 0.5 and bins[1]["tokens_per_round"] == 3
    assert bins[1]["mean_proposed_tokens"] == 3


def write_saved_suite(root, report=None):
    report = report or saved_report()
    root.mkdir()
    plan = root / "chunk-plan.txt"
    plan.write_text("qwen35-dflash-chunk-v3\nsynthetic plan; no model execution\n")
    batch = root / "prompts.txt"
    batch.write_text('QWEN35_PROMPT_BATCH_V1\np "4,5"\n')
    report["model"]["sha256"] = sha256_file(plan)
    index = root / "runner-batch.json"
    cases = root / "runner-batch.json.cases"
    cases.mkdir()
    case_path = cases / "p.json"
    case_path.write_text(json.dumps(report))
    index.write_text(json.dumps(dict(schema_version=1, status="FAIL", fake_acl=False,
        runner_version="synthetic-fixture", prompt_batch_sha256=sha256_file(batch), model_sha256=sha256_file(plan),
        dflash_speculation_policy="always_on", models_reused_across_prompts=True, low_memory=False,
        cases=[dict(id="p", status="FAIL", report=str(case_path))])))
    command = ["no-runner", "--model", str(plan), "--model-sha256", sha256_file(plan),
        "--prompt-batch", str(batch), "--prompt-batch-sha256", sha256_file(batch),
        "--device-id", "0", "--max-new-tokens", "8", "--max-draft-tokens", "15",
        "--eos-token-ids", ",".join(map(str, report["eos_token_ids"]))]
    request = dict(prompts=[dict(id="p", prompt="synthetic", prompt_token_ids=[4, 5])],
                   command=command, max_new_tokens=8, max_draft_tokens=15, eos_token_ids=report["eos_token_ids"])
    (root / "request.json").write_text(json.dumps(request))
    return index


def offline_args(root, index, allow=True):
    return argparse.Namespace(run_dir=root, summarize_existing=index, model_dir=None,
                              allow_output_differences=allow)


@pytest.mark.parametrize("allow", [False, True])
def test_offline_summary_needs_no_runner_or_model_and_keeps_sources(tmp_path, allow):
    index = write_saved_suite(tmp_path / "saved")
    before = {p: p.read_bytes() for p in index.parent.rglob("*") if p.is_file()}
    # Uses the declared model Python dependencies, but no runner or model files.
    command = [sys.executable, "-I", "-B", str(suite.REPO / "tools/benchmark_prompts.py"),
               "--run-dir", str(tmp_path), "--summarize-existing", str(index)]
    if allow:
        command.append("--allow-output-differences")
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == (0 if allow else 1), proc.stderr
    output, = tmp_path.glob("prompt-summary-*/summary.json")
    result = json.loads(output.read_text())
    assert result["ordinary_parity"] == "FAIL" and result["quality_evaluation"] == "NOT_RUN"
    assert result["formal_latency_evidence"] is False
    assert result["reanalysis"]["device_execution"] == "NOT_RUN"
    if allow:
        assert result["status"] == "PASS_WITH_DIFFERENCES"
        row, = result["cases"]
        assert row["generated_tokens"] == 8  # Saved request, never the CLI default of 128.
        assert row["raw_status"] == "FAIL" and row["first_difference"]["index"] == 3
        assert row["speedup"] == row["throughput_speedup"] == 2
        assert row["draft_token_share_of_output"] == 0.75
        assert result["aggregate"]["allowed_difference_prompts"] == 1
        assert result["aggregate"]["weighted_acceptance_rate"] == 6 / 7
        assert "Acceptance by generation position" in proc.stdout
        assert "no validated acceptance" not in proc.stdout
    else:
        assert result["status"] == "FAIL_OR_INCOMPLETE"
        assert result["aggregate"]["weighted_acceptance_rate"] is None
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize("kind", ["fake", "load_failure", "unstable", "trace", "nan", "eos", "identity"])
def test_offline_allow_policy_rejects_unusable_measurements(tmp_path, kind):
    report = saved_report()
    if kind == "load_failure":
        report["failure_stage"] = "load_dflash"
    elif kind == "unstable":
        report["dflash"]["measurements"][4]["generated_token_ids"][1] = 42
    elif kind == "trace":
        report["dflash"]["measurements"][0]["rounds"][1]["committed_prefix_length"] = 9
    elif kind == "nan":
        report["dflash"]["measurements"][0]["latency_ms"]["model_total"] = float("nan")
    elif kind == "eos":
        report["eos_token_ids"] = [6]
    elif kind == "identity":
        report["device_id"] = 1
    index = write_saved_suite(tmp_path / "saved", report)
    if kind == "fake":
        raw = json.loads(index.read_text())
        raw["fake_acl"] = True
        index.write_text(json.dumps(raw))
    assert suite.summarize_existing(offline_args(tmp_path, index)) == 1
    output, = tmp_path.glob("prompt-summary-*/summary.json")
    result = json.loads(output.read_text())
    assert result["aggregate"]["measured_prompts"] == 0
    assert result["aggregate"]["weighted_acceptance_rate"] is None


def test_offline_hash_mismatch_stops_before_writing_a_summary(tmp_path):
    index = write_saved_suite(tmp_path / "saved")
    (index.parent / "chunk-plan.txt").write_text("changed\n")
    with pytest.raises(ValueError, match="hashes"):
        suite.summarize_existing(offline_args(tmp_path, index))
    assert not list(tmp_path.glob("prompt-summary-*"))


def test_allowing_cross_mode_eos_difference_still_records_both_stop_reasons(tmp_path):
    report = saved_report()
    report["eos_token_ids"] = [46]
    report["dflash"]["stable_stop_reason"] = "eos"
    for measurement in report["dflash"]["measurements"]:
        measurement["stop_reason"] = "eos"
    report["ordinary_parity"]["eos_mismatches"] = 1
    index = write_saved_suite(tmp_path / "saved", report)
    assert suite.summarize_existing(offline_args(tmp_path, index)) == 0
    output, = tmp_path.glob("prompt-summary-*/summary.json")
    row, = json.loads(output.read_text())["cases"]
    assert row["status"] == "PASS_WITH_DIFFERENCES"
    assert row["stop_reason"] == "eos" and row["ordinary_stop_reason"] == "max_new_tokens"
    assert row["ordinary_parity"]["eos_mismatches"] == 1
