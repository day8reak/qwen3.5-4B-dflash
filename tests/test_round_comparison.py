"""Report comparisons must not confuse final-token parity with round parity."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework/python"))
from qwen35_dflash.ascend310p.compare_rounds import compare_npu_cpp_rounds, main


def _round(prefix, proposed, target, accepted, emitted, fallback):
    return dict(committed_prefix_length=prefix, proposed_token_ids=proposed,
                target_token_ids=target, accepted_draft_token_ids=accepted,
                emitted_token_ids=emitted, fallback_token_id=fallback)


def _reports():
    rows = [
        _round(1, [], [10], [], [10], 10),
        _round(2, [11, 99], [11, 12, 13], [11], [11, 12], 12),
        _round(4, [13], [13, 14], [13], [13], None),
    ]
    dflash = dict(prompt_token_ids=[1], generated_token_ids=[10, 11, 12, 13],
                  stop_reason="max_new_tokens", rounds=rows)
    native = dict(dflash=dflash, request={"eos_token_ids": [50]})
    cpp = dict(prompt_token_ids=[1], eos_token_ids=[50],
               dflash={"measurements": [copy.deepcopy(dflash), copy.deepcopy(dflash)]})
    return native, cpp


def test_every_cpp_repetition_must_match():
    native, cpp = _reports()
    assert compare_npu_cpp_rounds(native, cpp)["status"] == "MATCH"
    # Same final tokens and acceptance; rejected proposal alone changes a round.
    cpp["dflash"]["measurements"][1]["rounds"][1]["proposed_token_ids"][1] = 98
    report = compare_npu_cpp_rounds(native, cpp)
    assert report["status"] == "NOT_MATCHED"
    assert report["measurements"][0]["rounds"]["status"] == "MATCH"
    difference = report["measurements"][1]["rounds"]["first_difference"]
    assert difference["committed_prefix_length"] == 2
    assert difference["fields"] == {
        "proposed_token_ids": {"native": [11, 99], "cpp": [11, 98]}}


def test_different_acceptance_boundaries_with_identical_final_tokens():
    native, cpp = _reports()
    for measurement in cpp["dflash"]["measurements"]:
        measurement["rounds"] = [
            _round(1, [], [10], [], [10], 10),
            _round(2, [99], [11, 12], [], [11], 11),
            _round(3, [], [12], [], [12], 12),
            _round(4, [], [13], [], [13], 13),
        ]
    report = compare_npu_cpp_rounds(native, cpp)
    assert report["status"] == "NOT_MATCHED"
    for measurement in report["measurements"]:
        assert measurement["final_tokens"] == "MATCH"
        rounds = measurement["rounds"]
        assert rounds["status"] == "DIFFERENT"
        assert rounds["same_prefix_rounds_compared"] == 3
        assert rounds["cpp_only_prefix_lengths"] == [3]
        assert rounds["first_difference"]["committed_prefix_length"] == 2
        assert "accepted_draft_token_ids" in rounds["first_difference"]["fields"]


def test_equal_prefix_lengths_with_different_tokens_are_not_aligned():
    native, cpp = _reports()
    measurement = cpp["dflash"]["measurements"][0]
    measurement["generated_token_ids"][0] = 9
    measurement["rounds"][0] = _round(1, [], [9], [], [9], 9)
    rounds = compare_npu_cpp_rounds(native, cpp)["measurements"][0]["rounds"]
    assert rounds["same_prefix_rounds_compared"] == 1
    assert rounds["different_context_prefix_lengths"] == [2, 4]
    assert [d["committed_prefix_length"] for d in rounds["differences"]] == [1]


@pytest.mark.parametrize("missing", ["native", "cpp"])
def test_reports_without_rounds_do_not_claim_round_parity(missing):
    native, cpp = _reports()
    if missing == "native":
        native["dflash"].pop("rounds")
    else:
        cpp["dflash"]["measurements"][0].pop("rounds")
    result = compare_npu_cpp_rounds(native, cpp)
    assert result["status"] == "NOT_MATCHED"
    assert result["measurements"][0]["rounds"]["status"] == "NOT_AVAILABLE"


def test_eos_policy_and_prompt_are_checked_separately():
    native, cpp = _reports()
    cpp["eos_token_ids"] = [51]
    report = compare_npu_cpp_rounds(native, cpp)
    assert report["eos_policy"]["status"] == "DIFFERENT"
    assert report["measurements"][0]["rounds"]["status"] == "MATCH"
    assert report["status"] == "NOT_MATCHED"
    cpp["prompt_token_ids"] = [2]
    report = compare_npu_cpp_rounds(native, cpp)
    assert report["prompt_tokens"] == "DIFFERENT"
    assert report["measurements"][0]["rounds"]["status"] == "NOT_COMPARABLE"


@pytest.mark.parametrize("corrupt", ["prefix", "tokens", "fallback", "emitted", "accepted", "rows"])
def test_corrupt_traces_cannot_pass(corrupt):
    native, cpp = _reports()
    measurement = cpp["dflash"]["measurements"][0]
    round = measurement["rounds"][1]
    if corrupt == "prefix":
        round["committed_prefix_length"] = 9
    elif corrupt == "tokens":
        measurement["generated_token_ids"][-1] = 9
    elif corrupt == "fallback":
        round["fallback_token_id"] = 9
    elif corrupt == "emitted":
        round["emitted_token_ids"] = []
    elif corrupt == "accepted":
        round["accepted_draft_token_ids"] = [9]
    else:
        round["target_token_ids"] = []
    with pytest.raises(ValueError):
        compare_npu_cpp_rounds(native, cpp)


def test_comparison_cli_records_differences_and_refuses_overwrite(tmp_path, monkeypatch):
    native, cpp = _reports()
    cpp["eos_token_ids"] = [51]
    for name, payload in (("native", native), ("cpp", cpp)):
        (tmp_path / f"{name}.json").write_text(json.dumps(payload))
    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    output = tmp_path / "comparison.json"
    args = ["--native", str(tmp_path / "native.json"),
            "--cpp", str(tmp_path / "cpp.json"), "--output", str(output)]
    assert main(args) == 1
    assert json.loads(output.read_text())["eos_policy"]["status"] == "DIFFERENT"
    with pytest.raises(FileExistsError):
        main(args)
