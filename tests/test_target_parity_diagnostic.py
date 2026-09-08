"""Opt-in diagnostic control plane; these fixtures do not execute an OM."""
import copy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from qwen35_dflash.ascend310p import cpp_runtime, cli
from test_cpp_runtime_progress import _infer_cpp_args
from test_static_split_om import _specs
from test_static_fused_om import _compile_static_fixture


def _report(artifacts):
    token_check = {"equal": True, "mismatches": []}
    return {
        "schema_version": 1,
        "report_kind": "cpp-ascendcl-target-parity-diagnostic",
        "status": "DIAGNOSTIC", "formal_latency_evidence": False,
        "model_hashes_verified": True, "device_id": 0,
        "prompt_token_ids": [10], "eos_token_ids": [99],
        "max_new_tokens": 32, "max_draft_tokens": 15, "artifacts": artifacts,
        "diagnostic": {
            "max_transactions": 2, "captured_transactions": 1,
            "raw_logits_available": False, "token_parity": "PASS", "cursor_parity": "PASS",
            "transactions": [{
                "verify_cursor_transition_consistent": True,
                "chained_decode1_tokens": copy.deepcopy(token_check),
                "same_input_state_decode1_tokens": copy.deepcopy(token_check),
            }],
        },
    }


def test_cli_opt_in_is_off_by_default():
    parser = cli.build_parser()
    defaults = parser.parse_args(_infer_cpp_args())
    assert defaults.diagnose_target_parity is False
    assert defaults.diagnostic_max_transactions == 2
    selected = parser.parse_args(_infer_cpp_args("--diagnose-target-parity", "--diagnostic-max-transactions", "3"))
    assert selected.diagnose_target_parity is True
    assert selected.diagnostic_max_transactions == 3
    with pytest.raises(SystemExit):
        parser.parse_args(_infer_cpp_args("--diagnostic-max-transactions", "5"))


@pytest.mark.parametrize("failure", [False, True])
def test_diagnostic_command_preserves_request_and_separates_report(tmp_path, monkeypatch, failure):
    deployed, _ = _compile_static_fixture(tmp_path, monkeypatch, _specs())
    monkeypatch.setattr(cpp_runtime, "preflight_cpp_runner", lambda *a, **k: Path(sys.executable))
    resolved, _ = cpp_runtime._resolve_incremental_oms(deployed["manifest_path"])
    artifacts = {role: item[2]["sha256"] for role, item in resolved.items()}
    report = _report(artifacts)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        raw = Path(command[command.index("--output") + 1])
        invocation = json.loads(Path(str(raw) + ".invocation.json").read_text())
        assert invocation["command"] == command
        raw.write_text(json.dumps(report))
        if failure:
            Path(str(raw) + ".failure.json").write_text(json.dumps({
                "report_kind": "cpp-ascendcl-generation-failure", "status": "FAIL",
                "error": "Target parity diagnostic found token/cursor mismatch",
            }))
        return subprocess.CompletedProcess(command, int(failure), stdout="diagnostic fixture")

    options = dict(device_model="Ascend310P3", cann="test", driver="test",
                   firmware="test", runtime="test", state_policy=cpp_runtime.INCREMENTAL_STATE_POLICY)
    arguments = dict(
        deployment_manifest=deployed["manifest_path"], runner=sys.executable,
        runner_options=options, prompt_token_ids=[10], eos_token_ids=[99],
        device_id=0, max_new_tokens=32, max_draft_tokens=15,
        raw_output=tmp_path / "raw.json", log_output=tmp_path / "log.txt",
        execute=execute, progress=False, diagnose_target_parity=True,
    )
    if failure:
        with pytest.raises(RuntimeError, match="target_parity_report=.*raw.json"):
            cpp_runtime.run_cpp_pair(**arguments)
        assert arguments["raw_output"].is_file()
    else:
        payload = cpp_runtime.run_cpp_pair(**arguments)
        assert payload["status"] == "DIAGNOSTIC"
        assert payload["formal_latency_evidence"] is False
        assert "dflash" not in payload and "ordinary_parity" not in payload
        assert payload["control_plane"]["runner_invocation"]
    command, = calls
    for flag, expected in {
        "--diagnose-target-parity": "true", "--diagnostic-max-transactions": "2",
        "--measurement-protocol": "profile", "--warmup": "1", "--repetitions": "1",
        "--max-new-tokens": "32", "--max-draft-tokens": "15", "--eos-token-ids": "99",
        "--zero-accept-fallback-policy": "request-target-only",
    }.items():
        assert command[command.index(flag) + 1] == expected


@pytest.mark.parametrize("damage", ["formal", "pass-kind", "hash", "count", "mismatch", "false-summary", "cursor"])
def test_diagnostic_validator_fails_closed(damage):
    report = _report({"target-prefill": "a" * 64})
    args = dict(prompt_token_ids=[10], artifacts=report["artifacts"], device_id=0,
                max_new_tokens=32, max_draft_tokens=15, max_transactions=2)
    cpp_runtime.validate_target_parity_diagnostic(report, **args)
    broken = copy.deepcopy(report)
    if damage == "formal": broken["formal_latency_evidence"] = True
    if damage == "pass-kind": broken["status"] = "PASS"
    if damage == "hash": broken["artifacts"]["target-prefill"] = "b" * 64
    if damage == "count": broken["diagnostic"]["captured_transactions"] = 3
    if damage == "mismatch": broken["diagnostic"]["token_parity"] = "FAIL"
    if damage == "false-summary":
        broken["diagnostic"]["transactions"][0]["chained_decode1_tokens"]["equal"] = False
    if damage == "cursor":
        broken["diagnostic"]["transactions"][0]["verify_cursor_transition_consistent"] = False
    with pytest.raises(RuntimeError):
        cpp_runtime.validate_target_parity_diagnostic(broken, **args)


@pytest.mark.parametrize("extra", [{"dflash_sync_window": 2}, {"prefill_completion_policy": "coalesce-first-verify"}])
def test_diagnostic_rejects_unsupported_policy_without_silently_changing_it(tmp_path, monkeypatch, extra):
    deployed, _ = _compile_static_fixture(tmp_path, monkeypatch, _specs())
    options = dict(device_model="Ascend310P3", cann="test", driver="test",
                   firmware="test", runtime="test", state_policy=cpp_runtime.INCREMENTAL_STATE_POLICY, **extra)
    with pytest.raises(ValueError, match="sync-window=1 and separate prefill"):
        cpp_runtime.run_cpp_pair(
            deployment_manifest=deployed["manifest_path"], runner=sys.executable,
            runner_options=options, prompt_token_ids=[10], eos_token_ids=[99],
            device_id=0, max_new_tokens=32, max_draft_tokens=15,
            raw_output=tmp_path / "raw.json", log_output=tmp_path / "log.txt",
            diagnose_target_parity=True, progress=False,
            execute=lambda *a, **k: pytest.fail("must not launch incompatible diagnosis"),
        )


def test_diagnostic_cli_does_not_detokenize_or_write_paired_pass(tmp_path, monkeypatch):
    args = cli.build_parser().parse_args(_infer_cpp_args("--diagnose-target-parity"))
    args.output = tmp_path / "diagnostic.json"
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    log = tmp_path / "preflight.log"
    log.write_text("fixture")
    monkeypatch.setattr(cli, "run_declared_target_preflight", lambda: log)
    monkeypatch.setattr(cli, "load_tokenizer", lambda **kw: (object(), {}))
    monkeypatch.setattr(cli, "tokenize_prompt", lambda *a, **kw: [10])
    monkeypatch.setattr(cli, "_config", lambda *a: {})
    def run(**kwargs):
        assert kwargs["diagnose_target_parity"] is True
        return {**_report({}), "control_plane": {}}
    monkeypatch.setattr(cli, "run_cpp_pair", run)
    monkeypatch.setattr(cli, "write_cpp_prompt_report", lambda **kw: pytest.fail("diagnostic must not be paired PASS"))
    assert cli.command_infer_cpp(args) == 0
    assert json.loads(args.output.read_text())["status"] == "DIAGNOSTIC"
