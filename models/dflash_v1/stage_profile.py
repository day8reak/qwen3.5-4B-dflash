"""Independent first-round stage windows through the msprof dynamic CLI.

The wrapper attaches msprof after warmup and exports after acknowledged stop/quit.
Warmups rebuild the same first-round state and never start the collector.
The all mode reuses loaded models and collects each stage into its own directory.
This is a diagnostic protocol, not the whole-generation latency benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import hashlib
import os
from pathlib import Path

import torch

from .dflash_rollback_decode import (
    _input_ids,
    _normalize_eos,
    _normalize_prompt,
    _normalize_proposals,
    _top1_rows,
)
from .msprof_cli import (
    COLLECTOR, CONTROL_FD_ENV, PROFILE_STAGES, SINGLE_STAGES,
    ORDINARY_STAGES, MsprofStageProfiler, captured_calls, stages_for_mode,
)
from .target_quant import (
    TARGET_EMBEDDING_SCALE_PATH_ENV,
    TARGET_EMBEDDING_WEIGHT_PATH_ENV,
    TARGET_QUANT_CONFIG_ENV,
    TARGET_QUANT_WEIGHT_PATH_ENV,
)


AIC_METRICS = ("PipeUtilization", "Memory", "MemoryUB")
GDR_BACKEND = "npu_chunk_gated_delta_rule_two_pass"
STAGE_SCOPES = {
    "prefill": (
        "one complete Target.begin_rollback: fresh-cache reset, all real "
        "prompt chunks, feature capture and final-row LM head; excludes "
        "Draft feature projection and anchor Top1"
    ),
    "draft": (
        "one first-round Draft proposal including initial Draft KV construction "
        "and Draft Top1; excludes prefill, prompt feature projection, anchor Top1, "
        "proposal normalization and request cleanup; no Target verify is executed"
    ),
    "verify": (
        "one first-round Target verify including scalar GDN state clones and the "
        "first original chunk GDR pass with effective_length=T; uses the "
        "real prefill state and Draft tokens prepared before start; excludes "
        "prefill, Draft, proposal normalization/block construction, Target Top1, "
        "accept and the second chunk GDR state commit"
    ),
    "draft-verify": (
        "first Draft proposal, proposal normalization/block construction "
        "and Target verify including scalar GDN state clones and the first "
        "original chunk GDR pass; excludes prefill, prompt feature projection, "
        "Target Top1, accept and the second chunk GDR state commit"
    ),
    "feature-project": (
        "one prompt Target-to-Draft fc + hidden_norm projection using real prefill "
        "features; excludes Target prefill, output validation and fingerprint transfer"
    ),
    "verify-input": (
        "one Draft proposal normalization including token-ID readback and EOS "
        "truncation, plus verify block construction/upload; excludes Draft and verify"
    ),
    "target-top1": (
        "one post-verify Target finite-value check, argmax and token-ID readback; "
        "excludes Target LM head (inside verify) and prefill anchor Top1"
    ),
    "accept-commit": (
        "one acceptance comparison and adapter "
        "commit: second original chunk GDR pass from saved round-start state with "
        "effective_length=accepted+1 (also when accepted=0), conv state selection, "
        "persistent-state dtype conversion and logical KV commit, plus next-round "
        "feature projection including zero acceptance; excludes Target Top1"
    ),
    "decode-round": (
        "one complete first Draft/verify transaction: prefix tensor, Draft, verify "
        "input preparation, Target verify, Target Top1, acceptance and commit "
        "including both original chunk GDR passes; "
        "excludes prefill, prompt projection/anchor Top1, outer generation-loop "
        "bookkeeping, output callbacks and detokenization"
    ),
}


def _snapshot_gdr_calls(target):
    audit = target.dflash_rollback_audit
    if audit.get("gdr_backend") != GDR_BACKEND:
        raise RuntimeError("stage profiling on this branch requires the two-pass chunk-GDR target")
    counts = {}
    for step in ("verify", "commit"):
        value = audit.get(f"rollback_gdr_{step}_layer_calls")
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RuntimeError(f"missing or invalid chunk-GDR {step} layer counter")
        counts[step] = value
    return counts


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile-mode", choices=("ordinary", "dflash"), default="dflash",
                        help="diagnostic execution route (default: dflash)")
    parser.add_argument(
        "--profile-stage", choices=PROFILE_STAGES,
        help="NPU diagnostic: collect one selected stage, or all stages separately in one process",
    )
    parser.add_argument("--profile-output", help="new directory for raw msprof data")
    parser.add_argument(
        "--profile-warmup", type=int, default=1,
        help="unprofiled warmups with fresh session state (default: 1)",
    )
    parser.add_argument(
        "--profile-aic-metrics", choices=tuple(AIC_METRICS), default="PipeUtilization",
    )


def validate_profile_request(args, *, source_root: Path) -> None:
    if args.profile_stage is None:
        if args.profile_output is not None or getattr(args, "profile_mode", "dflash") != "dflash":
            raise ValueError("--profile-output/--profile-mode requires --profile-stage")
        return
    if args.profile_stage != "all" and args.profile_stage not in stages_for_mode(getattr(args, "profile_mode", "dflash")):
        raise ValueError("stage is unavailable for the selected --profile-mode")
    if str(args.device).split(":", 1)[0] != "npu":
        raise ValueError("--profile-stage requires a real NPU")
    if os.environ.get("ASCEND310P_SIMULATION_ONLY") == "1":
        raise ValueError("simulation-only profiles cannot collect NPU performance")
    if getattr(args, "allow_op_fallback", False):
        raise ValueError("--profile-stage forbids operator fallback")
    if args.profile_warmup < 0:
        raise ValueError("--profile-warmup must be non-negative")
    if args.profile_output is None:
        raise ValueError("--profile-stage requires --profile-output")
    # Do not nest a dynamic collector inside the existing process-level wrapper.
    if os.environ.get("DFLASH_MSPROF_PROCESS_CAPTURE") == "1":
        raise ValueError(
            "put --profile-stage on run_msprof.sh before --; "
            "process-level msprof and the dynamic stage collector cannot run together"
        )
    destination = Path(args.profile_output).expanduser()
    if destination.is_symlink() or destination.exists():
        raise ValueError("--profile-output must be a new directory")
    resolved = destination.resolve()
    protected = [source_root, Path(args.target_dir), Path(args.draft_dir)]
    for name in (
        TARGET_QUANT_CONFIG_ENV, TARGET_QUANT_WEIGHT_PATH_ENV,
        TARGET_EMBEDDING_WEIGHT_PATH_ENV, TARGET_EMBEDDING_SCALE_PATH_ENV,
    ):
        if os.environ.get(name):
            protected.append(Path(os.environ[name]))
    protected = [path.expanduser().resolve() for path in protected]
    if any(resolved.is_relative_to(path) for path in protected):
        raise ValueError("--profile-output must be outside source and checkpoint roots")
    args.profile_output = str(resolved)
    if args.report is None:
        args.report = str(resolved.parent / (resolved.name + "-report.json"))
    destination = Path(args.report).expanduser()
    if destination.is_symlink():
        raise ValueError("stage --report must not be a symlink")
    report = destination.resolve()
    if report.is_relative_to(resolved) or resolved.is_relative_to(report):
        raise ValueError("--report must not overlap the raw --profile-output directory")
    if any(report.is_relative_to(path) for path in protected):
        raise ValueError("stage --report must be outside source and checkpoint roots")
    if report.exists():
        raise ValueError("stage --report must be a new file")
    args.report = str(report)

    if os.environ.get("PROFILING_MODE") != "dynamic" or CONTROL_FD_ENV not in os.environ:
        raise ValueError(
            "--profile-stage must be launched by tools/run_msprof.sh --profile-stage"
        )


def profile_one_stage(
    adapter, prompt_token_ids, *, stage: str, block_size: int,
    eos_token_ids, warmup: int, profiler: MsprofStageProfiler,
) -> dict[str, object]:
    """Replay the real bootstrap; never shorten K using max_new_tokens."""
    if stage not in SINGLE_STAGES or warmup < 0:
        raise ValueError("invalid stage or warmup count")
    if not 2 <= block_size <= adapter.max_block_size:
        raise ValueError("invalid profile block_size")
    prompt, device = _normalize_prompt(prompt_token_ids, adapter.device)
    prompt_ids = _input_ids(prompt, device)
    eos = _normalize_eos(eos_token_ids)
    captured_gdr_calls = None

    @contextmanager
    def measured_capture():
        nonlocal captured_gdr_calls
        before = _snapshot_gdr_calls(adapter.target)
        with profiler.capture():
            yield
        after = _snapshot_gdr_calls(adapter.target)
        captured_gdr_calls = {key: after[key] - before[key] for key in before}

    def once(measured: bool):
        def capture(selected_stage: str):
            return measured_capture() if measured and stage == selected_stage else nullcontext()

        if stage == "prefill":
            with capture("prefill"):
                # The adapter's begin also projects Draft features. Keep that
                # outside this pure Target-prefill diagnostic.
                output = adapter.target.begin_rollback(prompt_ids)
            anchor = _top1_rows(output, expected_rows=None, source="profile prefill")[-1]
            return {"anchor_token_id": anchor}

        if stage == "feature-project":
            output = adapter.target.begin_rollback(prompt_ids)
            _, features = adapter._validated_output(output, rows=None, features=True)
            if features is None or features.shape[1] != len(prompt):
                raise RuntimeError("prefill did not return complete prompt features")
            features = features.detach()
            with torch.inference_mode(), capture("feature-project"):
                projected = adapter.draft.project_target_hidden(features)
            # Validate repeatability outside the window, including values rather
            # than only comparing the projected tensor's shape.
            fingerprint = hashlib.sha256(
                projected.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            ).hexdigest()
            return {
                "projection_shape": list(projected.shape),
                "projection_dtype": str(projected.dtype),
                "projection_sha256": fingerprint,
            }

        output = adapter.begin_rollback(prompt_ids)
        anchor = _top1_rows(output, expected_rows=None, source="profile bootstrap")[-1]
        if anchor in eos:
            raise RuntimeError("prefill anchor is EOS; no Draft/verify round to profile")
        try:
            # Exactly one scope collects. Combined scopes add no internal barrier.
            with capture("decode-round"):
                prefix_ids = _input_ids([*prompt, anchor], device)
                with capture("draft-verify"):
                    with capture("draft"):
                        raw_proposals = adapter.propose_rollback(prefix_ids, block_size - 1)
                    with capture("verify-input"):
                        proposals = _normalize_proposals(
                            raw_proposals,
                            proposal_limit=block_size - 1, eos_token_ids=eos,
                        )
                        if not proposals:
                            raise RuntimeError("Draft returned no proposals; no Draft/verify capture")
                        if stage != "draft":
                            block = [anchor, *proposals]
                            block_ids = _input_ids(block, device)
                    if stage != "draft":
                        with capture("verify"):
                            verify_output = adapter.verify_rollback(block_ids)
                if stage == "draft":
                    adapter.abort_rollback()
                    return {
                        "anchor_token_id": anchor,
                        "proposal_token_ids": proposals,
                        "verify_input_token_ids": [],
                        "target_top1_token_ids": [],
                        "verify_rows": 0,
                        "accepted_draft_tokens": None,
                    }
                with capture("target-top1"):
                    target_tokens = _top1_rows(
                        verify_output, expected_rows=len(block), source="profile verify",
                    )
                with capture("accept-commit"):
                    accepted = next(
                        (i for i, token in enumerate(proposals) if token != target_tokens[i]),
                        len(proposals),
                    )
                    adapter.commit_rollback(accepted)
        except Exception:
            adapter.abort_rollback()
            raise
        return {
            "anchor_token_id": anchor,
            "proposal_token_ids": proposals,
            "verify_input_token_ids": block,
            "target_top1_token_ids": target_tokens,
            "verify_rows": len(block),
            "accepted_draft_tokens": accepted,
        }

    reference = None
    for _ in range(warmup):
        reference = once(False)
        profiler.synchronize()
    measured = once(True)
    stable = None if reference is None else reference == measured
    if stable is False:
        raise RuntimeError("single-stage outputs differ from the unprofiled warmup")
    if profiler.windows != 1:
        raise RuntimeError("single-stage profiling did not capture exactly one window")
    expected_gdr = {
        "verify": stage in {"verify", "draft-verify", "decode-round"},
        "commit": stage in {"accept-commit", "decode-round"},
    }
    if captured_gdr_calls is None or any(
        count < 0 or (count > 0) != expected_gdr[key]
        for key, count in captured_gdr_calls.items()
    ):
        raise RuntimeError("captured chunk-GDR layer calls do not match the selected stage")
    return {
        "schema_version": 3,
        "route": "qwen3.5-dflash-single-stage-profile",
        "profile_mode": "dflash", "profile_backend": "python",
        "status": "PASS_CAPTURE",
        "formal_latency_evidence": False,
        "strict_greedy_exact_match": None,
        "correctness_gate": {"status": "NOT_RUN_STAGE_DIAGNOSTIC"},
        "profile_stage": stage,
        "profile_output": profiler.output,
        "collector": COLLECTOR,
        "aic_metrics": profiler.metrics,
        "capture_windows": profiler.windows,
        "captured_calls": captured_calls(stage),
        "gdr_backend": GDR_BACKEND,
        "captured_gdr_layer_calls": captured_gdr_calls,
        # This branch always runs a second chunk GDR pass on commit, including
        # accepted=0. An empty commit export is a failed capture.
        "operator_rows_required": True,
        "warmup_iterations": warmup,
        "warmup_output_match": stable,
        "profiled_elapsed_ms": profiler.elapsed_ms,
        "profiled_elapsed_excludes": "msprof attach/start/stop/quit and controller waits",
        "prompt_tokens": len(prompt),
        "block_size": block_size,
        "proposal_capacity": block_size - 1,
        "max_new_tokens_applies": False,
        "stage_scope": STAGE_SCOPES[stage],
        "synchronization": "before/after the selected stage; no added Draft/verify barrier in joint mode",
        "result": measured,
    }


def profile_all_stages(adapter, prompt_token_ids, *, block_size, eos_token_ids, warmup, profiler):
    """Load models once; replay identical fresh request state for each window."""
    reports = []
    for stage in SINGLE_STAGES:
        print(f"[stage-profile] preparing stage={stage} warmup={warmup}", flush=True)
        child = profiler.for_stage(stage)
        report = profile_one_stage(
            adapter, prompt_token_ids, stage=stage, block_size=block_size,
            eos_token_ids=eos_token_ids, warmup=warmup, profiler=child,
        )
        reports.append(report)
        print(
            f"[stage-profile] captured stage={stage} elapsed_ms={child.elapsed_ms:.3f}",
            flush=True,
        )
    return {
        "schema_version": 4, "route": "qwen3.5-dflash-all-stage-profile",
        "profile_mode": "dflash", "profile_backend": "python",
        "status": "PASS_CAPTURE", "profile_stage": "all",
        "profile_output": profiler.output, "collector": COLLECTOR,
        "aic_metrics": profiler.metrics, "capture_windows": len(reports),
        "gdr_backend": GDR_BACKEND,
        "stages": list(SINGLE_STAGES), "captures": reports,
        "formal_latency_evidence": False, "strict_greedy_exact_match": None,
        "correctness_gate": {"status": "NOT_RUN_STAGE_DIAGNOSTIC"},
        "model_loads": 1, "warmup_iterations_per_stage": warmup,
        "state_policy": "same prompt, fresh first-round state per warmup and capture",
        "max_new_tokens_applies": False,
    }


def profile_ordinary_stage(target, prompt_token_ids, *, stage, device, eos_token_ids, warmup, profiler):
    """One native ordinary prefill or one real single-token decode, no Draft load."""
    if stage not in ORDINARY_STAGES or warmup < 0:
        raise ValueError("invalid ordinary stage or warmup count")
    prompt, device = _normalize_prompt(prompt_token_ids, device)
    prompt_ids = _input_ids(prompt, device)
    eos = _normalize_eos(eos_token_ids)
    measured_counts = None
    measured_gdr = None

    def snapshot():
        audit = target.dflash_rollback_audit
        result = {key: audit.get(key) for key in ("ordinary_prefill_token_calls", "ordinary_decode_calls")}
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in result.values()):
            raise RuntimeError("ordinary target is missing execution counters")
        return result

    @contextmanager
    def capture(measured):
        nonlocal measured_counts, measured_gdr
        if not measured:
            yield
            return
        before, gdr_before = snapshot(), _snapshot_gdr_calls(target)
        with profiler.capture():
            yield
        after, gdr_after = snapshot(), _snapshot_gdr_calls(target)
        measured_counts = {k: after[k] - before[k] for k in before}
        measured_gdr = {k: gdr_after[k] - gdr_before[k] for k in gdr_before}

    @torch.inference_mode()
    def once(measured):
        if stage == "prefill":
            with capture(measured):
                output = target.begin_ordinary(prompt_ids)
            return {"anchor_token_id": _top1_rows(output, expected_rows=None, source="ordinary prefill")[-1]}
        output = target.begin_ordinary(prompt_ids)
        anchor = _top1_rows(output, expected_rows=None, source="ordinary bootstrap")[-1]
        if anchor in eos:
            raise RuntimeError("prefill anchor is EOS; no ordinary decode to profile")
        token_ids = _input_ids([anchor], device)
        with capture(measured):
            output = target.advance_ordinary(token_ids)
        return {"anchor_token_id": anchor,
                "next_token_id": _top1_rows(output, expected_rows=1, source="ordinary decode")[0]}

    reference = None
    for _ in range(warmup):
        reference = once(False)
        profiler.synchronize()
    measured = once(True)
    stable = None if reference is None else reference == measured
    if stable is False or profiler.windows != 1:
        raise RuntimeError("ordinary capture must be one repeatable window")
    expected = {"ordinary_prefill_token_calls": (len(prompt) + 63) // 64 if stage == "prefill" else 0,
                "ordinary_decode_calls": int(stage == "decode")}
    if measured_counts != expected or measured_gdr != {"verify": 0, "commit": 0}:
        raise RuntimeError("captured ordinary calls differ from the selected stage")
    return {
        "schema_version": 3, "route": "qwen3.5-ordinary-single-stage-profile",
        "status": "PASS_CAPTURE", "profile_mode": "ordinary", "profile_backend": "python",
        "profile_stage": stage, "profile_output": profiler.output,
        "collector": COLLECTOR, "aic_metrics": profiler.metrics, "capture_windows": 1,
        "captured_calls": captured_calls(stage, "ordinary"),
        "captured_ordinary_calls": measured_counts, "captured_gdr_layer_calls": measured_gdr,
        "gdr_backend": GDR_BACKEND, "operator_rows_required": True,
        "warmup_iterations": warmup, "warmup_output_match": stable,
        "profiled_elapsed_ms": profiler.elapsed_ms, "prompt_tokens": len(prompt),
        "stage_scope": (
            "one complete Target.begin_ordinary: cache reset, all real prompt chunks and final LM head; "
            "excludes input upload and anchor Top1; no Draft feature collection or projection"
            if stage == "prefill" else
            "one Target.advance_ordinary with [1,1] input: single-row target forward, LM head and state update; "
            "excludes prefill/cache preparation, input upload, anchor and next-token Top1"
        ),
        "profiled_elapsed_excludes": "msprof attach/start/stop/quit and controller waits",
        "formal_latency_evidence": False, "strict_greedy_exact_match": None,
        "correctness_gate": {"status": "NOT_RUN_STAGE_DIAGNOSTIC"},
        "max_new_tokens_applies": False, "draft_model_loaded": False, "result": measured,
    }


def profile_ordinary_all(target, prompt_token_ids, *, device, eos_token_ids, warmup, profiler):
    reports = []
    for stage in ORDINARY_STAGES:
        print(f"[stage-profile] preparing ordinary stage={stage} warmup={warmup}", flush=True)
        reports.append(profile_ordinary_stage(
            target, prompt_token_ids, stage=stage, device=device, eos_token_ids=eos_token_ids,
            warmup=warmup, profiler=profiler.for_stage(stage),
        ))
    return {
        "schema_version": 4, "route": "qwen3.5-ordinary-all-stage-profile",
        "status": "PASS_CAPTURE", "profile_mode": "ordinary", "profile_backend": "python",
        "profile_stage": "all", "profile_output": profiler.output, "collector": COLLECTOR,
        "aic_metrics": profiler.metrics, "capture_windows": len(reports),
        "stages": list(ORDINARY_STAGES), "captures": reports,
        "formal_latency_evidence": False, "strict_greedy_exact_match": None,
        "correctness_gate": {"status": "NOT_RUN_STAGE_DIAGNOSTIC"},
        "model_loads": 1, "draft_model_loaded": False, "warmup_iterations_per_stage": warmup,
        "state_policy": "same prompt, fresh ordinary cache per warmup and capture",
        "max_new_tokens_applies": False,
    }
