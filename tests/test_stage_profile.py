"""Host tests of capture ownership, real round boundaries, and cleanup."""

from __future__ import annotations

import json
import socket
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.dflash_v1 import run_npu, run_rollback
from models.dflash_v1.stage_profile import (
    MsprofStageProfiler, profile_all_stages, profile_one_stage, validate_profile_request,
)
from models.dflash_v1.msprof_cli import SINGLE_STAGES, PROFILE_STAGES


def logits(tokens):
    result = torch.full((1, len(tokens), 128), -10.0)
    for i, token in enumerate(tokens):
        result[0, i, token] = 10
    return result


class SocketController:
    """Host-only peer for the actual application-side stage barrier."""

    def __init__(self, events, channel):
        self.events, self.channel = events, channel
        self.active = False
        self.messages = []
        self.errors = []

    def run(self):
        try:
            with self.channel, self.channel.makefile("rb") as reader:
                for line in reader:
                    message = json.loads(line)
                    self.messages.append(message)
                    if message["event"] == "ready":
                        assert not self.active
                        self.events.append(("start",))
                        self.active, response = True, "started"
                    else:
                        assert message["event"] == "done" and self.active
                        self.events.append(("stop",))
                        self.active, response = False, "stopped"
                    self.channel.sendall(json.dumps({"event": response}).encode() + b"\n")
        except BaseException as error:
            self.errors.append(error)


class FakeTarget:
    def __init__(self, adapter):
        self.adapter = adapter

    def begin_rollback(self, ids):
        a = self.adapter
        a.cursor = ids.shape[1]
        a.pending = None
        a.requests += 1
        for start in range(0, ids.shape[1], 64):
            a.events.append(("prefill-chunk", a.collector.active, min(64, ids.shape[1] - start)))
        return {"logits": logits([a.anchor]), "features": torch.ones(1, ids.shape[1], 4)}


class FakeDraft:
    def __init__(self, adapter):
        self.adapter = adapter

    def project_target_hidden(self, features):
        a = self.adapter
        a.events.append(("projection", a.collector.active))
        return features[..., :2] + 1


class FakeAdapter:
    device = torch.device("cpu")
    max_block_size = 16

    def __init__(self, collector, events, *, anchor=10, fail_verify=False, reject=False,
                 fail_draft=False, empty_draft=False):
        self.collector, self.events = collector, events
        self.target = FakeTarget(self)
        self.draft = FakeDraft(self)
        self.anchor = anchor
        self.fail_verify = fail_verify
        self.reject = reject
        self.fail_draft, self.empty_draft = fail_draft, empty_draft
        self.cursor = 0
        self.pending = None
        self.requests = 0

    def _validated_output(self, output, *, rows, features):
        return output["logits"], output["features"] if features else None

    def begin_rollback(self, ids):
        output = self.target.begin_rollback(ids)
        self.events.append(("projection", self.collector.active))
        return output

    def propose_rollback(self, prefix_ids, proposal_limit):
        assert self.cursor == prefix_ids.shape[1] - 1
        assert self.pending is None
        self.events.append(("draft", self.collector.active, proposal_limit, self.cursor))
        if self.fail_draft:
            raise RuntimeError("injected draft failure")
        if self.empty_draft:
            return torch.empty((1, 0), dtype=torch.long)
        return torch.tensor([list(range(11, 11 + proposal_limit))])

    def verify_rollback(self, ids):
        self.events.append(("verify", self.collector.active, ids.tolist(), self.cursor))
        assert self.pending is None
        self.pending = ids.tolist()[0]
        if self.fail_verify:
            raise RuntimeError("injected verify failure")
        tokens = list(range(11, 11 + ids.shape[1]))
        if self.reject:
            tokens[0] = 77
        return logits(tokens)

    def disable_speculation(self):
        self.events.append(("disable", self.collector.active))

    def commit_rollback(self, accepted):
        assert self.pending is not None
        self.events.append(("commit", self.collector.active, accepted))
        self.cursor += accepted + 1
        self.pending = None

    def abort_rollback(self):
        self.events.append(("abort", self.collector.active))
        self.pending = None


@pytest.fixture
def setup_profiler(tmp_path, monkeypatch):
    controllers = []

    def create(stage="draft-verify"):
        events = []
        server, client = socket.socketpair()
        collector = SocketController(events, server)
        monkeypatch.setenv("PROFILING_MODE", "dynamic")
        monkeypatch.setenv("DFLASH_MSPROF_CONTROL_FD", str(client.detach()))
        monkeypatch.setenv("DFLASH_MSPROF_CONTROL_TIMEOUT", "3")
        profiler = MsprofStageProfiler(
            str(tmp_path / "raw"), 2, "MemoryUB",
            lambda: events.append(("sync", collector.active)), stage=stage,
        )
        thread = threading.Thread(target=collector.run, daemon=True)
        thread.start()
        controllers.append((collector, thread))
        return events, collector, profiler

    yield create
    for collector, thread in controllers:
        thread.join(timeout=3)
        assert not thread.is_alive(), "stage client did not close its control channel"
        assert not collector.errors


@pytest.mark.parametrize("stage", ["prefill", "draft", "verify", "draft-verify"])
def test_only_one_stage_is_collected_after_fresh_warmup(setup_profiler, stage):
    events, collector, profiler = setup_profiler(stage)
    adapter = FakeAdapter(collector, events)
    with profiler:
        result = profile_one_stage(
            adapter, [1] * 129, stage=stage, block_size=16,
            eos_token_ids=[99], warmup=2, profiler=profiler,
        )
    start = events.index(("start",))
    stop = events.index(("stop",))
    captured = events[start + 1:stop]
    assert events[start - 1] == ("sync", False)
    assert captured[-1] == ("sync", True)
    assert events.count(("start",)) == events.count(("stop",)) == 1
    assert collector.messages[0]["device_id"] == 2
    assert collector.messages[0]["metrics"] == "MemoryUB"
    assert collector.messages[-1] == {"event": "done", "success": True}
    assert adapter.requests == 3
    assert result["warmup_output_match"] is True
    assert result["strict_greedy_exact_match"] is None
    assert result["formal_latency_evidence"] is False
    assert result["profiled_elapsed_ms"] >= 0
    assert result["captured_calls"] == {
        "prefill": int(stage == "prefill"),
        "draft": int(stage in {"draft", "draft-verify"}),
        "target_verify": int(stage in {"verify", "draft-verify"}),
    }
    if stage == "prefill":
        assert [event[0] for event in captured] == ["prefill-chunk"] * 3 + ["sync"]
        assert [event[2] for event in captured[:-1]] == [64, 64, 1]
        assert not any(e[0] in {"draft", "verify", "projection", "commit"} for e in events)
    elif stage == "draft":
        assert captured == [("draft", True, 15, 129), ("sync", True)]
        assert not any(e[0] in {"verify", "commit"} for e in events)
        assert events[-1] == ("abort", False)
        assert adapter.pending is None and adapter.cursor == 129
        assert result["result"]["verify_rows"] == 0
        assert result["result"]["proposal_token_ids"] == list(range(11, 26))
    else:
        expected_ops = ["verify", "sync"] if stage == "verify" else ["draft", "verify", "sync"]
        assert [event[0] for event in captured] == expected_ops
        assert captured[-2][2] == [list(range(10, 26))]
        assert result["result"]["verify_rows"] == 16
        assert all(not e[1] for e in events if e[0] in {"prefill-chunk", "projection", "commit"})
        assert all(e[1] == (stage == "draft-verify") for e in captured if e[0] == "draft")
        if stage == "verify":
            assert all(not e[1] for e in events if e[0] == "draft")
        assert adapter.cursor == 129 + 16


@pytest.mark.parametrize("stage", ["verify", "draft-verify"])
def test_verify_error_stops_capture_and_aborts_after_stop(setup_profiler, stage):
    events, collector, profiler = setup_profiler(stage)
    adapter = FakeAdapter(collector, events, fail_verify=True)
    with pytest.raises(RuntimeError, match="injected verify failure"):
        with profiler:
            profile_one_stage(
                adapter, [1], stage=stage, block_size=16,
                eos_token_ids=[99], warmup=0, profiler=profiler,
            )
    assert events[-3:] == [("sync", True), ("stop",), ("abort", False)]
    assert collector.messages[-1] == {"event": "done", "success": False}
    assert not any(e[0] == "commit" for e in events)


@pytest.mark.parametrize("stage", ["draft", "verify", "draft-verify"])
def test_proposal_eos_reports_actual_verify_rows(setup_profiler, stage):
    events, collector, profiler = setup_profiler(stage)
    with profiler:
        result = profile_one_stage(
            FakeAdapter(collector, events), [1], stage=stage, block_size=16,
            eos_token_ids=[12], warmup=0, profiler=profiler,
        )
    assert result["result"]["proposal_token_ids"] == [11, 12]
    assert result["result"]["verify_input_token_ids"] == ([] if stage == "draft" else [10, 11, 12])
    assert result["result"]["verify_rows"] == (0 if stage == "draft" else 3)
    assert result["warmup_output_match"] is None
    assert events.count(("start",)) == events.count(("stop",)) == 1


@pytest.mark.parametrize("stage", ["draft", "verify", "draft-verify"])
def test_immediate_eos_does_not_emit_an_empty_capture(setup_profiler, stage):
    events, collector, profiler = setup_profiler(stage)
    with pytest.raises(RuntimeError, match="anchor is EOS"):
        with profiler:
            profile_one_stage(
                FakeAdapter(collector, events, anchor=99), [1],
                stage=stage, block_size=16,
                eos_token_ids=[99], warmup=0, profiler=profiler,
            )
    assert ("start",) not in events
    assert not collector.messages


@pytest.mark.parametrize("stage", ["verify", "draft-verify"])
def test_zero_acceptance_commit_remains_outside_capture(setup_profiler, stage):
    events, collector, profiler = setup_profiler(stage)
    with profiler:
        result = profile_one_stage(
            FakeAdapter(collector, events, reject=True), [1],
            stage=stage, block_size=16,
            eos_token_ids=[99], warmup=1, profiler=profiler,
        )
    assert result["result"]["accepted_draft_tokens"] == 0
    assert events[events.index(("stop",)) + 1:][:2] == [("disable", False), ("commit", False, 0)]


@pytest.mark.parametrize("stage", ["draft", "verify", "draft-verify"])
def test_separate_windows_exclude_proposal_normalization_and_input_upload(setup_profiler, stage):
    from models.dflash_v1 import stage_profile as module

    events, collector, profiler = setup_profiler(stage)
    normalize, input_ids = module._normalize_proposals, module._input_ids

    def normalize_observed(*args, **kwargs):
        events.append(("normalize", collector.active))
        return normalize(*args, **kwargs)

    def input_ids_observed(tokens, device):
        events.append(("input-ids", collector.active, len(tokens)))
        return input_ids(tokens, device)

    with profiler, patch.object(module, "_normalize_proposals", side_effect=normalize_observed), \
            patch.object(module, "_input_ids", side_effect=input_ids_observed):
        profile_one_stage(
            FakeAdapter(collector, events), [1], stage=stage, block_size=16,
            eos_token_ids=[], warmup=0, profiler=profiler,
        )
    assert [e for e in events if e[0] == "normalize"] == [("normalize", stage == "draft-verify")]
    block_inputs = [e for e in events if e[0] == "input-ids" and e[2] == 16]
    assert block_inputs == ([] if stage == "draft" else [("input-ids", stage == "draft-verify", 16)])


@pytest.mark.parametrize("stage", ["draft", "verify"])
@pytest.mark.parametrize("failure", ["exception", "empty"])
def test_draft_failures_abort_without_an_empty_verify_capture(setup_profiler, stage, failure):
    events, collector, profiler = setup_profiler(stage)
    adapter = FakeAdapter(
        collector, events, fail_draft=failure == "exception", empty_draft=failure == "empty",
    )
    with profiler, pytest.raises(RuntimeError, match="draft failure|no proposals"):
        profile_one_stage(
            adapter, [1], stage=stage, block_size=16,
            eos_token_ids=[], warmup=0, profiler=profiler,
        )
    assert profiler.windows == int(stage == "draft")
    assert not collector.active and adapter.pending is None
    assert events[-1] == ("abort", False)
    assert not any(e[0] in {"verify", "commit"} for e in events)
    if stage == "draft":
        assert events.count(("stop",)) == 1
        assert collector.messages[-1] == {"event": "done", "success": failure != "exception"}


@pytest.mark.parametrize("stage", PROFILE_STAGES)
def test_stage_arguments_forward_without_shrinking_block(tmp_path, stage):
    with patch.object(run_npu, "_adapter_main", return_value=0) as run:
        assert run_npu.main([
            "--target-dir", "/model/target", "--draft-dir", "/model/draft",
            "--kv-cache-max-len", "256", "--prompt-ids", "1,2",
            "--max-new-tokens", "1", "--block-size", "16",
            "--profile-stage", stage, "--profile-output", str(tmp_path / "raw"),
            "--profile-warmup", "0", "--profile-aic-metrics", "Memory",
        ]) == 0
    args = run_rollback._parser().parse_args(run.call_args.args[0])
    assert (args.profile_stage, args.profile_warmup, args.profile_aic_metrics) == (stage, 0, "Memory")
    assert args.block_size == 16 and args.max_new_tokens == 1


@pytest.mark.parametrize("device, env, error", [
    ("npu:0", {}, "must be launched"),
    ("cpu", {}, "real NPU"),
    ("npu:0", {"ASCEND310P_SIMULATION_ONLY": "1"}, "simulation-only"),
    ("npu:0", {"DFLASH_MSPROF_PROCESS_CAPTURE": "1"}, "before --"),
])
def test_invalid_collection_environment_fails_before_writes(tmp_path, device, env, error):
    args = SimpleNamespace(
        device=device, profile_stage="prefill", profile_output=str(tmp_path / "raw"),
        profile_warmup=1, report=None, target_dir="/model/target", draft_dir="/model/draft",
    )
    with patch.dict("os.environ", env, clear=True):
        with pytest.raises(ValueError, match=error):
            validate_profile_request(args, source_root=Path(__file__).parents[1])
    assert not (tmp_path / "raw").exists()


@pytest.mark.parametrize("location", ["raw-source", "report-source", "report-in-raw", "report-ancestor"])
def test_stage_destinations_do_not_pollute_sources_or_raw_capture(tmp_path, location):
    source = tmp_path / "source"
    raw = source / "raw" if location == "raw-source" else tmp_path / "run" / "raw"
    reports = {
        "raw-source": None,
        "report-source": source / "report.json",
        "report-in-raw": raw / "report.json",
        "report-ancestor": raw.parent,
    }
    args = SimpleNamespace(
        device="npu:0", profile_stage="prefill", profile_output=str(raw),
        profile_warmup=1, report=reports[location],
        target_dir=tmp_path / "target", draft_dir=tmp_path / "draft",
    )
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="outside source|must not overlap"):
            validate_profile_request(args, source_root=source)
    assert not source.exists() and not raw.exists()


@pytest.mark.parametrize("stage,expected", [
    ("feature-project", ["projection"]),
    ("verify-input", ["normalize", "input-ids"]),
    ("target-top1", ["top1"]),
    ("accept-commit", ["commit"]),
    ("decode-round", ["input-ids", "draft", "normalize", "input-ids", "verify", "top1", "commit"]),
])
def test_auxiliary_stage_boundaries(setup_profiler, stage, expected):
    from models.dflash_v1 import stage_profile as module

    events, collector, profiler = setup_profiler(stage)
    def observed(name, function):
        def call(*args, **kwargs):
            events.append((name, collector.active))
            return function(*args, **kwargs)
        return call
    with profiler, patch.object(module, "_normalize_proposals",
            side_effect=observed("normalize", module._normalize_proposals)), \
            patch.object(module, "_input_ids", side_effect=observed("input-ids", module._input_ids)), \
            patch.object(module, "_top1_rows", side_effect=observed("top1", module._top1_rows)):
        report = profile_one_stage(
            FakeAdapter(collector, events), [1, 2], stage=stage, block_size=16,
            eos_token_ids=[], warmup=1, profiler=profiler,
        )
    captured = events[events.index(("start",)) + 1:events.index(("stop",))]
    assert [e[0] for e in captured] == expected + ["sync"]
    assert all(e[1] is True for e in captured)
    assert report["warmup_output_match"] is True
    if stage == "feature-project":
        assert report["result"]["projection_shape"] == [1, 2, 2]
        assert not any(e[0] in {"draft", "verify", "commit"} for e in events)


def test_all_captures_each_stage_once_in_order_with_fresh_state(setup_profiler):
    events, collector, profiler = setup_profiler("all")
    adapter = FakeAdapter(collector, events)
    with profiler:
        report = profile_all_stages(
            adapter, [1, 2], block_size=16, eos_token_ids=[], warmup=1, profiler=profiler,
        )
    assert report["model_loads"] == 1
    assert report["capture_windows"] == len(SINGLE_STAGES)
    assert adapter.requests == 2 * len(SINGLE_STAGES)
    assert [m["stage"] for m in collector.messages if m["event"] == "ready"] == list(SINGLE_STAGES)
    outputs = [m["output"] for m in collector.messages if m["event"] == "ready"]
    assert len(set(outputs)) == len(SINGLE_STAGES)
    assert [Path(p).name for p in outputs] == list(SINGLE_STAGES)
    assert all(c["warmup_output_match"] for c in report["captures"])
    assert events.count(("start",)) == events.count(("stop",)) == len(SINGLE_STAGES)


def test_all_stops_before_later_stages_after_verify_failure(setup_profiler):
    events, collector, profiler = setup_profiler("all")
    with profiler, pytest.raises(RuntimeError, match="injected verify failure"):
        profile_all_stages(
            FakeAdapter(collector, events, fail_verify=True), [1],
            block_size=16, eos_token_ids=[], warmup=0, profiler=profiler,
        )
    # verify-input postprocessing verifies outside collection and propagates failure.
    assert [m["stage"] for m in collector.messages if m["event"] == "ready"] == [
        "prefill", "feature-project", "draft", "verify-input",
    ]
    assert events[-1] == ("abort", False)


def test_zero_acceptance_commit_can_have_no_device_operators(setup_profiler):
    events, collector, profiler = setup_profiler("accept-commit")
    with profiler:
        report = profile_one_stage(
            FakeAdapter(collector, events, reject=True), [1], stage="accept-commit",
            block_size=16, eos_token_ids=[], warmup=0, profiler=profiler,
        )
    assert report["result"]["accepted_draft_tokens"] == 0
    assert report["operator_rows_required"] is False
    captured = events[events.index(("start",)) + 1:events.index(("stop",))]
    assert captured == [("disable", True), ("commit", True, 0), ("sync", True)]


def test_projection_values_must_match_warmup(setup_profiler):
    events, collector, profiler = setup_profiler("feature-project")
    adapter = FakeAdapter(collector, events)
    with profiler, patch.object(adapter.draft, "project_target_hidden", side_effect=[
            torch.zeros(1, 1, 2), torch.ones(1, 1, 2),
    ]), pytest.raises(RuntimeError, match="outputs differ"):
        profile_one_stage(
            adapter, [1], stage="feature-project", block_size=16,
            eos_token_ids=[], warmup=1, profiler=profiler,
        )
    assert events.count(("start",)) == events.count(("stop",)) == 1
