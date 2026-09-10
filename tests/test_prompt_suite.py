"""Multi-prompt scheduling and reporting tests; fake ACL is host evidence only."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess

import pytest

from test_incremental_air_om import chunk_bundle, assert_cpp_resources_released, small_threads  # noqa: F401
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401
from qwen35_dflash.ascend310p.cpp_runtime import validate_cpp_runner_report
from qwen35_dflash.ascend310p.incremental_plan import write_incremental_plan
from qwen35_dflash.ascend310p.utils import sha256_file
from tools import benchmark_prompts as suite
from tools.decode_outputs import saved_cases

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu")


def batch_command(manifest, root, prompts, low_memory=False):
    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("set QWEN35_CPP_TEST_RUNNER to the fake ACL binary")
    plan, _, _ = write_incremental_plan(manifest, root / "plan.txt")
    batch = root / "prompts.txt"
    batch.write_text("QWEN35_PROMPT_BATCH_V1\n" + "".join(
        f'{name} "{",".join(map(str, tokens))}"\n' for name, tokens in prompts))
    output = root / "batch.json"
    command = [runner, "--model-kind", "chunk", "--mode", "paired",
        "--model", str(plan), "--model-sha256", sha256_file(plan),
        "--prompt-batch", str(batch), "--prompt-batch-sha256", sha256_file(batch),
        "--output", str(output), "--max-new-tokens", "20", "--max-draft-tokens", "15",
        "--trace-rounds", "--warmup", "3", "--repetitions", "10"]
    if low_memory:
        command.append("--low-memory")
    return command, output, plan


@pytest.mark.parametrize("low_memory", [False, True])
def test_batch_reuses_models_resets_prompts_and_preserves_parity(
    chunk_bundle, tmp_path, monkeypatch, low_memory
):
    prompts = [("short", [4, 5]), ("long", [4] * 65), ("different", [8, 9, 10])]
    workspace, cleanup = tmp_path / "loads.jsonl", tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_WORKSPACE_LOG", str(workspace))
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "3")
    command, output, plan = batch_command(chunk_bundle, tmp_path, prompts, low_memory)
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    index = json.loads(output.read_text())
    assert index["status"] == "PASS" and index["fake_acl"] is True
    assert index["models_reused_across_prompts"] is True
    assert index["low_memory"] is low_memory
    assert [c["id"] for c in index["cases"]] == [p[0] for p in prompts]
    loaded = [json.loads(line) for line in workspace.read_text().splitlines()]
    assert [r[0] for r in loaded] == (
        ["target_decode", "target_prefill", "draft", "target_prefill", "target_verify"]
        if low_memory else ["draft", "target_decode", "target_prefill", "target_verify"])
    assert_cpp_resources_released(cleanup, proc.stderr)
    reports = []
    for case, (_, tokens) in zip(index["cases"], prompts):
        report = json.loads(Path(case["report"]).read_text())
        validate_cpp_runner_report(report, prompt_token_ids=tokens, om_sha256=sha256_file(plan),
            device_id=0, max_new_tokens=20, max_draft_tokens=15, chunk_abi=True, low_memory=low_memory)
        assert report["startup_ms"]["acl_and_model_load"] == 0
        stats = suite.summarize_prompt(report)
        assert stats["speculative_rounds"] > 0
        assert 0 < stats["acceptance_rate"] < 1
        for mode in ("ordinary", "dflash"):
            for measurement in report[mode]["measurements"]:
                assert measurement["rounds"][0]["committed_prefix_length"] == len(tokens)
                assert measurement["rounds"][0]["stage"] == "target_prefill"
        reports.append(report)
    # A fresh process must give exactly the same tokens and round counters as a
    # later prompt in the shared process; model loading is not N-times repeated.
    monkeypatch.delenv("QWEN35_FAKE_WORKSPACE_LOG")
    single = tmp_path / "single.json"
    command = [command[0], "--model-kind", "chunk", "--mode", "paired", "--model", str(plan),
        "--model-sha256", sha256_file(plan), "--prompt-token-ids", "8,9,10",
        "--output", str(single), "--max-new-tokens", "20", "--trace-rounds"]
    if low_memory:
        command.append("--low-memory")
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    isolated = json.loads(single.read_text())
    for mode in ("ordinary", "dflash"):
        assert isolated[mode]["stable_generated_token_ids"] == reports[-1][mode]["stable_generated_token_ids"]
        for a, b in zip(isolated[mode]["measurements"], reports[-1][mode]["measurements"]):
            assert a["counters"] == b["counters"]
            assert a["rounds"] == b["rounds"]


@pytest.mark.parametrize("low_memory", [False, True])
def test_one_failed_prompt_is_retained_and_next_prompt_runs(chunk_bundle, tmp_path, monkeypatch, low_memory):
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    command, output, _ = batch_command(chunk_bundle, tmp_path,
        [("too_long", [4] * 129), ("valid", [4, 5])], low_memory)
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    index = json.loads(output.read_text())
    assert index["status"] == "FAIL"
    assert [c["status"] for c in index["cases"]] == ["FAIL", "PASS"]
    assert json.loads(Path(index["cases"][0]["report"]).read_text())["error"]
    assert_cpp_resources_released(cleanup, proc.stderr)


def test_changed_batch_is_rejected_before_model_loading(chunk_bundle, tmp_path, monkeypatch):
    workspace = tmp_path / "loads.jsonl"
    monkeypatch.setenv("QWEN35_FAKE_WORKSPACE_LOG", str(workspace))
    command, output, _ = batch_command(chunk_bundle, tmp_path, [("p", [4])])
    (tmp_path / "prompts.txt").write_text('QWEN35_PROMPT_BATCH_V1\np "5"\n')
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 1 and "batch SHA-256 differs" in proc.stderr
    assert not workspace.exists() and not output.exists()


def test_weighted_acceptance_keeps_failures_and_empty_denominator_visible():
    rows = [
        dict(id="a", status="PASS", drafted_tokens=10, accepted_draft_tokens=9,
             ordinary_total_measured_ms=100, dflash_total_measured_ms=50),
        dict(id="b", status="PASS", drafted_tokens=90, accepted_draft_tokens=9,
             ordinary_total_measured_ms=200, dflash_total_measured_ms=250),
        dict(id="bad", status="FAIL", error="token mismatch"),
    ]
    result = suite.aggregate(rows)
    assert result["weighted_acceptance_rate"] == 0.18  # Not (90% + 10%) / 2.
    assert result["total_model_time_speedup"] == 1
    assert result["passed_prompts"] == 2 and result["failed_prompts"] == 1
    assert suite.aggregate(rows[-1:])["weighted_acceptance_rate"] is None
    rows[0].update(drafted_tokens=0, accepted_draft_tokens=0)
    assert suite.aggregate(rows[:1])["weighted_acceptance_rate"] is None


def test_default_and_custom_prompts(tmp_path):
    assert len(suite.load_prompts(None)) == 8
    assert [p["id"] for p in suite.load_prompts(None, ["code", "math"])] == ["math", "code"]
    with pytest.raises(ValueError, match="unknown prompt"):
        suite.load_prompts(None, ["absent"])
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(["一个问题", {"id": "code", "prompt": "Write code", "category": "code"}]))
    result = suite.load_prompts(custom)
    assert [r["id"] for r in result] == ["prompt_01", "code"]
    for bad in ([42], [{"id": "../escape", "prompt": "x"}],
                [{"id": "x", "prompt": "x"}, {"id": "x", "prompt": "y"}], []):
        custom.write_text(json.dumps(bad))
        with pytest.raises(ValueError):
            suite.load_prompts(custom)


def test_wrapper_records_requests_and_rejects_fake_acl_as_device_evidence(chunk_bundle, tmp_path, monkeypatch):
    from qwen35_dflash.ascend310p import workflow

    runner = os.environ.get("QWEN35_CPP_TEST_RUNNER")
    if not runner:
        pytest.skip("fake ACL binary required")
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return [4, len(messages[0]["content"]) % 32]

        def decode(self, tokens, **kwargs):
            return str(tokens)

    monkeypatch.delenv("ASCEND310P_SIMULATION_ONLY", raising=False)  # This test explicitly exercises fake rejection.
    monkeypatch.delenv("PROFILING_MODE", raising=False)
    monkeypatch.setattr(workflow, "load_tokenizer", lambda **kwargs: (Tokenizer(), "host-test-tokenizer"))
    config = tmp_path / "runner.json"
    config.write_text(json.dumps(dict(device_model="Ascend310P3-host-fixture", cann="fake",
                                      driver="fake", firmware="fake", runtime="fake-acl")))
    custom = tmp_path / "custom.json"
    custom.write_text(json.dumps(["one", "longer question"]))
    args = argparse.Namespace(run_dir=tmp_path, runner=Path(runner), deployment_manifest=chunk_bundle,
        runner_config=config, prompts=custom, model_dir=tmp_path, chat=True, eos_token_id=[63],
        device_id=0, max_new_tokens=20, max_draft_tokens=15, low_memory=False)
    assert suite.run(args) == 1
    root, = list(tmp_path.glob("prompt-suite-*"))
    request = json.loads((root / "request.json").read_text())
    assert [p["prompt_token_ids"] for p in request["prompts"]] == [[4, 3], [4, 15]]
    assert "--deterministic=1" in request["draft_atc_command"]
    summary = json.loads((root / "summary.json").read_text())
    assert summary["status"] == "FAIL_OR_INCOMPLETE"
    assert summary["aggregate"]["failed_prompts"] == 2
    assert all("fake ACL" in r["error"] for r in summary["cases"])
    assert "| prompt_01 | FAIL |" in (root / "summary.md").read_text()


@pytest.mark.parametrize("low_memory", [False, True])
def test_stable_parity_failure_keeps_both_outputs_and_locates_first_different_round(
    chunk_bundle, tmp_path, monkeypatch, low_memory
):
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    monkeypatch.setenv("QWEN35_FAKE_ACCEPT", "15")
    monkeypatch.setenv("QWEN35_FAKE_VERIFY_DRIFT_ROW", "2")
    command, output, plan = batch_command(chunk_bundle, tmp_path, [("p", [4, 5])], low_memory)
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    index = json.loads(output.read_text())
    assert index["status"] == "FAIL" and index["cases"][0]["status"] == "FAIL"
    report = json.loads(Path(index["cases"][0]["report"]).read_text())
    assert report["failure_stage"] == "ordinary_dflash_parity"
    assert report["formal_latency_evidence"] is False
    assert report["dflash_speedup_over_ordinary_model_total_median"] is None
    assert report["ordinary_parity"]["token_id_mismatches"] > 0
    for mode in ("ordinary", "dflash"):
        assert report[mode]["status"] == "PASS"  # Within-mode repeatability only.
        assert len(report[mode]["measurements"]) == 10
        assert report[mode]["measurements"][0]["rounds"]
    diff = report["ordinary_parity"]["first_difference"]
    assert diff["index"] == 3 and diff["absolute_token_position"] == 5
    assert diff["ordinary_token_id"] == 9 and diff["dflash_token_id"] == 16
    assert diff["ordinary_location"]["round_index"] == 3
    assert diff["dflash_location"]["round_index"] == 1
    assert diff["dflash_location"]["emitted_index"] == 2
    assert diff["dflash_location"]["round"]["accepted_draft_token_ids"] == [7, 8]
    assert diff["dflash_location"]["previous_round"]["stage"] == "target_prefill"
    assert "generated_index=3 ordinary=9 dflash=16" in proc.stderr
    with pytest.raises(RuntimeError, match="passing known report"):
        validate_cpp_runner_report(report, prompt_token_ids=[4, 5], om_sha256=sha256_file(plan),
            device_id=0, max_new_tokens=20, max_draft_tokens=15, chunk_abi=True, low_memory=low_memory)
    assert_cpp_resources_released(cleanup, proc.stderr)
    # The single-prompt entry must preserve the same failure and still exit 1.
    single = tmp_path / "single-failed.json"
    single_cmd = [command[0], "--model-kind", "chunk", "--mode", "paired", "--model", str(plan),
                  "--model-sha256", sha256_file(plan), "--prompt-token-ids", "4,5",
                  "--output", str(single), "--max-new-tokens", "20", "--trace-rounds"]
    if low_memory:
        single_cmd.append("--low-memory")
    proc = subprocess.run(single_cmd, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    assert json.loads(single.read_text())["ordinary_parity"] == report["ordinary_parity"]
    assert_cpp_resources_released(cleanup, proc.stderr)


def test_mode_load_failure_reports_dflash_not_run_and_keeps_ordinary(chunk_bundle, tmp_path, monkeypatch):
    cleanup = tmp_path / "cleanup.json"
    monkeypatch.setenv("QWEN35_FAKE_CLEANUP_LOG", str(cleanup))
    monkeypatch.setenv("QWEN35_FAKE_FAIL_LOAD_GRAPH", "target_verify")
    command, output, _ = batch_command(chunk_bundle, tmp_path, [("a", [4, 5]), ("b", [7, 8])], True)
    proc = subprocess.run(command, capture_output=True, text=True)
    assert proc.returncode == 1, proc.stderr
    index = json.loads(output.read_text())
    assert index["status"] == "FAIL" and "245000" in index["error"]
    assert all(c["status"] == "NOT_RUN" for c in index["cases"])
    for case in index["cases"]:
        report = json.loads(Path(case["report"]).read_text())
        assert report["failure_stage"] == "load_dflash"
        assert report["ordinary_parity"]["status"] == "NOT_RUN"
        assert len(report["ordinary"]["measurements"]) == 10
        assert "dflash" not in report
    assert "start mode=dflash" not in proc.stderr
    assert_cpp_resources_released(cleanup, proc.stderr)


class TextTokenizer:
    def decode(self, tokens, *, skip_special_tokens):
        assert skip_special_tokens is False
        return "".join({1: "<think>", 2: "你好", 3: "世界", 4: "！"}[t] for t in tokens)


def test_decoding_failed_pair_keeps_special_tokens_and_both_texts():
    report = {"status": "FAIL", "ordinary": {"stable_generated_token_ids": [1, 2, 3]},
              "dflash": {"stable_generated_token_ids": [1, 2, 4]},
              "ordinary_parity": {"first_difference": {"field": "generated_token_ids", "index": 2,
                  "ordinary_token_id": 3, "dflash_token_id": 4}}}
    decoded = suite.decode_outputs(report, TextTokenizer())
    assert decoded["ordinary"]["text"] == "<think>你好世界"
    assert decoded["dflash"]["text"] == "<think>你好！"
    assert decoded["dflash"]["difference_context"]["token_ids"] == [1, 2, 4]
    row = dict(id="p", status="FAIL", first_difference=report["ordinary_parity"]["first_difference"],
               decoded_outputs=decoded, error="parity failed", failure_stage="ordinary_dflash_parity")
    text = suite.render_outputs([row])
    assert "| FAIL" in text and "<think>你好世界" in text and "<think>你好！" in text
    rows = [row, dict(id="missing", status="NOT_RUN", error="load failed", failure_stage="load_dflash")]
    md = suite.markdown(dict(cases=rows, aggregate=suite.aggregate(rows)))
    assert "failed: 1; not run: 1" in md and "| missing | NOT_RUN |" in md
    assert "Each passing prompt" not in md and "Weighted acceptance: N/A" in md
    assert "ordinary_dflash_parity" in md


def test_legacy_error_report_decodes_only_saved_ordinary(tmp_path):
    case = tmp_path / "p.json"
    case.write_text(json.dumps(dict(status="FAIL", error="parity failed")))
    ordinary = {"status": "PASS", "generation_mode": "ordinary-greedy", "stable_generated_token_ids": [1, 2]}
    (tmp_path / "p.ordinary.json").write_text(json.dumps(ordinary))
    (_, report), = saved_cases(case)
    decoded = suite.decode_outputs(report, TextTokenizer())
    assert decoded["ordinary"]["text"] == "<think>你好" and not decoded["dflash"]["available"]
    text = suite.render_outputs([dict(status="FAIL", decoded_outputs=decoded)])
    assert "dflash: output token IDs unavailable" in text
    (_, report), = saved_cases(tmp_path / "p.ordinary.json")
    assert suite.decode_outputs(report, TextTokenizer())["ordinary"]["available"]


def test_decode_cli_selects_failed_batch_case_without_inference(tmp_path, monkeypatch, capsys):
    from qwen35_dflash.ascend310p import workflow
    from tools import decode_outputs as decoder

    directory = tmp_path / "runner-batch.json.cases"
    directory.mkdir()
    index = tmp_path / "runner-batch.json"
    report = {"status": "FAIL", "prompt_token_ids": [2], "error": "parity failed",
              "ordinary": {"stable_generated_token_ids": [1, 2, 3]},
              "dflash": {"stable_generated_token_ids": [1, 2, 4]}}
    cases = []
    for name in ("a", "b"):
        path = directory / (name + ".json")
        path.write_text(json.dumps(report))
        cases.append(dict(id=name, status="FAIL", report=str(path)))
    index.write_text(json.dumps(dict(status="FAIL", cases=cases)))
    before = {p: p.read_bytes() for p in (index, *(directory.glob("*.json")))}
    monkeypatch.setattr(workflow, "load_tokenizer", lambda **kwargs: (TextTokenizer(), "host-test"))
    monkeypatch.setattr("sys.argv", ["decode_outputs.py", "--model-dir", str(tmp_path),
                                    "--report", str(tmp_path), "--prompt-id", "b"])
    assert decoder.main() == 0  # Text conversion, not a parity pass.
    output = capsys.readouterr().out
    assert "=== b | FAIL ===" in output and "=== a" not in output
    assert "Prompt:\n你好" in output and "<think>你好世界" in output and "<think>你好！" in output
    assert all(p.read_bytes() == content for p, content in before.items())
