"""Host tests of capture ownership, real round boundaries, and cleanup."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from models.dflash_v1 import run_npu, run_rollback
from models.dflash_v1.stage_profile import (
    AclStageProfiler, profile_one_stage, validate_profile_request,
)


def logits(tokens):
    result = torch.full((1, len(tokens), 128), -10.0)
    for i, token in enumerate(tokens):
        result[0, i, token] = 10
    return result


class FakeAclProf:
    def __init__(self, events, *, fail=None):
        self.events = events
        self.fail = fail
        self.active = False

    def init(self, path):
        self.events.append(("init", path))
        return 0

    def create_config(self, devices, metric, reserved, flags):
        self.events.append(("config", devices, metric, reserved, flags))
        return None if self.fail == "config" else 1234

    def start(self, config):
        assert config == 1234
        self.events.append(("start",))
        if self.fail == "start":
            return 17
        assert not self.active
        self.active = True
        return 0

    def stop(self, config):
        assert config == 1234 and self.active
        self.events.append(("stop",))
        self.active = False
        return 18 if self.fail == "stop" else 0

    def destroy_config(self, config):
        assert config == 1234 and not self.active
        self.events.append(("destroy",))
        return 19 if self.fail == "destroy_config" else 0

    def finalize(self):
        self.events.append(("finalize",))
        return 20 if self.fail == "finalize" else 0


class FakeTarget:
    def __init__(self, adapter):
        self.adapter = adapter

    def begin_rollback(self, ids):
        a = self.adapter
        a.cursor = ids.shape[1]
        a.pending = None
        a.requests += 1
        for start in range(0, ids.shape[1], 64):
            a.events.append(("prefill-chunk", a.acl.active, min(64, ids.shape[1] - start)))
        return {"logits": logits([a.anchor])}


class FakeAdapter:
    device = torch.device("cpu")
    max_block_size = 16

    def __init__(self, acl, events, *, anchor=10, fail_verify=False, reject=False):
        self.acl, self.events = acl, events
        self.target = FakeTarget(self)
        self.anchor = anchor
        self.fail_verify = fail_verify
        self.reject = reject
        self.cursor = 0
        self.pending = None
        self.requests = 0

    def begin_rollback(self, ids):
        output = self.target.begin_rollback(ids)
        self.events.append(("projection", self.acl.active))
        return output

    def propose_rollback(self, prefix_ids, proposal_limit):
        assert self.cursor == prefix_ids.shape[1] - 1
        assert self.pending is None
        self.events.append(("draft", self.acl.active, proposal_limit, self.cursor))
        return torch.tensor([list(range(11, 11 + proposal_limit))])

    def verify_rollback(self, ids):
        self.events.append(("verify", self.acl.active, ids.tolist(), self.cursor))
        assert self.pending is None
        self.pending = ids.tolist()[0]
        if self.fail_verify:
            raise RuntimeError("injected verify failure")
        tokens = list(range(11, 11 + ids.shape[1]))
        if self.reject:
            tokens[0] = 77
        return logits(tokens)

    def disable_speculation(self):
        self.events.append(("disable", self.acl.active))

    def commit_rollback(self, accepted):
        assert self.pending is not None
        self.events.append(("commit", self.acl.active, accepted))
        self.cursor += accepted + 1
        self.pending = None

    def abort_rollback(self):
        self.events.append(("abort", self.acl.active))
        self.pending = None


def setup_profiler(tmp_path, *, fail=None):
    events = []
    acl = FakeAclProf(events, fail=fail)
    profiler = AclStageProfiler(
        str(tmp_path / "raw"), 2, "MemoryUB",
        lambda: events.append(("sync", acl.active)),
    )
    return events, acl, profiler


@pytest.mark.parametrize("stage", ["prefill", "draft-verify"])
def test_only_one_stage_is_collected_after_fresh_warmup(tmp_path, stage):
    events, acl, profiler = setup_profiler(tmp_path)
    adapter = FakeAdapter(acl, events)
    with patch("models.dflash_v1.stage_profile.importlib.import_module",
               return_value=SimpleNamespace(prof=acl)):
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
    assert events[-2:] == [("destroy",), ("finalize",)]
    assert ("config", [2], 5, 0, 7) in events
    assert adapter.requests == 3
    assert result["warmup_output_match"] is True
    assert result["strict_greedy_exact_match"] is None
    assert result["formal_latency_evidence"] is False
    if stage == "prefill":
        assert [event[0] for event in captured] == ["prefill-chunk"] * 3 + ["sync"]
        assert [event[2] for event in captured[:-1]] == [64, 64, 1]
        assert not any(e[0] in {"draft", "verify", "projection", "commit"} for e in events)
    else:
        assert [event[0] for event in captured] == ["draft", "verify", "sync"]
        assert captured[0] == ("draft", True, 15, 129)
        assert captured[1][2] == [list(range(10, 26))]
        assert result["result"]["verify_rows"] == 16
        assert all(not e[1] for e in events if e[0] in {"prefill-chunk", "projection", "commit"})
        assert adapter.cursor == 129 + 16


def test_verify_error_stops_and_finalizes_capture(tmp_path):
    events, acl, profiler = setup_profiler(tmp_path)
    adapter = FakeAdapter(acl, events, fail_verify=True)
    with patch("models.dflash_v1.stage_profile.importlib.import_module",
               return_value=SimpleNamespace(prof=acl)):
        with pytest.raises(RuntimeError, match="injected verify failure"):
            with profiler:
                profile_one_stage(
                    adapter, [1], stage="draft-verify", block_size=16,
                    eos_token_ids=[99], warmup=0, profiler=profiler,
                )
    assert events[-5:] == [
        ("sync", True), ("stop",), ("abort", False), ("destroy",), ("finalize",),
    ]
    assert not any(e[0] == "commit" for e in events)


@pytest.mark.parametrize("fail", ["config", "start", "stop", "destroy_config", "finalize"])
def test_profiling_api_errors_are_not_reported_as_success(tmp_path, fail):
    events, acl, profiler = setup_profiler(tmp_path, fail=fail)
    with patch("models.dflash_v1.stage_profile.importlib.import_module",
               return_value=SimpleNamespace(prof=acl)):
        with pytest.raises(RuntimeError, match="config|start|stop|finalize"):
            with profiler:
                with profiler.capture():
                    events.append(("work",))
    assert events[-1] == ("finalize",)
    assert not acl.active


def test_proposal_eos_reports_actual_verify_rows(tmp_path):
    events, acl, profiler = setup_profiler(tmp_path)
    with patch("models.dflash_v1.stage_profile.importlib.import_module",
               return_value=SimpleNamespace(prof=acl)):
        with profiler:
            result = profile_one_stage(
                FakeAdapter(acl, events), [1], stage="draft-verify", block_size=16,
                eos_token_ids=[12], warmup=0, profiler=profiler,
            )
    assert result["result"]["proposal_token_ids"] == [11, 12]
    assert result["result"]["verify_input_token_ids"] == [10, 11, 12]
    assert result["result"]["verify_rows"] == 3
    assert result["warmup_output_match"] is None
    assert events.count(("start",)) == events.count(("stop",)) == 1


def test_immediate_eos_does_not_emit_an_empty_capture(tmp_path):
    events, acl, profiler = setup_profiler(tmp_path)
    with patch("models.dflash_v1.stage_profile.importlib.import_module",
               return_value=SimpleNamespace(prof=acl)):
        with pytest.raises(RuntimeError, match="anchor is EOS"):
            with profiler:
                profile_one_stage(
                    FakeAdapter(acl, events, anchor=99), [1],
                    stage="draft-verify", block_size=16,
                    eos_token_ids=[99], warmup=0, profiler=profiler,
                )
    assert ("start",) not in events
    assert events[-2:] == [("destroy",), ("finalize",)]


def test_zero_acceptance_commit_remains_outside_capture(tmp_path):
    events, acl, profiler = setup_profiler(tmp_path)
    with patch("models.dflash_v1.stage_profile.importlib.import_module",
               return_value=SimpleNamespace(prof=acl)):
        with profiler:
            result = profile_one_stage(
                FakeAdapter(acl, events, reject=True), [1],
                stage="draft-verify", block_size=16,
                eos_token_ids=[99], warmup=1, profiler=profiler,
            )
    assert result["result"]["accepted_draft_tokens"] == 0
    assert events[events.index(("stop",)) + 1:][:2] == [("disable", False), ("commit", False, 0)]


def test_stage_arguments_forward_without_shrinking_block(tmp_path):
    with patch.object(run_npu, "_adapter_main", return_value=0) as run:
        assert run_npu.main([
            "--target-dir", "/model/target", "--draft-dir", "/model/draft",
            "--kv-cache-max-len", "256", "--prompt-ids", "1,2",
            "--max-new-tokens", "1", "--block-size", "16",
            "--profile-stage", "draft-verify", "--profile-output", str(tmp_path / "raw"),
            "--profile-warmup", "0", "--profile-aic-metrics", "Memory",
        ]) == 0
    args = run_rollback._parser().parse_args(run.call_args.args[0])
    assert (args.profile_stage, args.profile_warmup, args.profile_aic_metrics) == ("draft-verify", 0, "Memory")
    assert args.block_size == 16 and args.max_new_tokens == 1


@pytest.mark.parametrize("device, env, error", [
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
