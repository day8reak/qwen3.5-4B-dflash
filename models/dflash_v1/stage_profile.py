"""One real prefill or first Draft/verify round, collected through the msprof dynamic CLI.

The wrapper attaches msprof after warmup and exports after acknowledged stop/quit.
Warmups rebuild the same first-round state and never start the collector.
This is a diagnostic protocol, not the whole-generation latency benchmark.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path

from .dflash_rollback_decode import (
    _input_ids,
    _normalize_eos,
    _normalize_prompt,
    _normalize_proposals,
    _top1_rows,
)
from .msprof_cli import COLLECTOR, CONTROL_FD_ENV, MsprofStageProfiler
from .target_quant import (
    TARGET_EMBEDDING_SCALE_PATH_ENV,
    TARGET_EMBEDDING_WEIGHT_PATH_ENV,
    TARGET_QUANT_CONFIG_ENV,
    TARGET_QUANT_WEIGHT_PATH_ENV,
)


PROFILE_STAGES = ("prefill", "draft-verify")
AIC_METRICS = ("PipeUtilization", "Memory", "MemoryUB")


def add_profile_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--profile-stage", choices=PROFILE_STAGES,
        help="NPU diagnostic: collect exactly one prefill or first Draft+verify round",
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
        if args.profile_output is not None:
            raise ValueError("--profile-output requires --profile-stage")
        return
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
    if stage not in PROFILE_STAGES or warmup < 0:
        raise ValueError("invalid stage or warmup count")
    if not 2 <= block_size <= adapter.max_block_size:
        raise ValueError("invalid profile block_size")
    prompt, device = _normalize_prompt(prompt_token_ids, adapter.device)
    prompt_ids = _input_ids(prompt, device)
    eos = _normalize_eos(eos_token_ids)

    def once(measured: bool):
        capture = profiler.capture if measured else nullcontext
        if stage == "prefill":
            with capture():
                # The adapter's begin also projects Draft features. Keep that
                # outside this pure Target-prefill diagnostic.
                output = adapter.target.begin_rollback(prompt_ids)
            anchor = _top1_rows(output, expected_rows=None, source="profile prefill")[-1]
            return {"anchor_token_id": anchor}

        output = adapter.begin_rollback(prompt_ids)
        anchor = _top1_rows(output, expected_rows=None, source="profile bootstrap")[-1]
        if anchor in eos:
            raise RuntimeError("prefill anchor is EOS; no Draft/verify round to profile")
        prefix_ids = _input_ids([*prompt, anchor], device)
        try:
            with capture():
                proposals = _normalize_proposals(
                    adapter.propose_rollback(prefix_ids, block_size - 1),
                    proposal_limit=block_size - 1, eos_token_ids=eos,
                )
                if not proposals:
                    raise RuntimeError("Draft returned no proposals; no Draft/verify capture")
                block = [anchor, *proposals]
                verify_output = adapter.verify_rollback(_input_ids(block, device))
            # Target Top1, accept and commit are deliberately outside collection.
            # Keep ordinary production acceptance/disable semantics after the call.
            target_tokens = _top1_rows(
                verify_output, expected_rows=len(block), source="profile verify",
            )
            accepted = next(
                (i for i, token in enumerate(proposals) if token != target_tokens[i]),
                len(proposals),
            )
            if accepted == 0:
                adapter.disable_speculation()
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
    return {
        "schema_version": 2,
        "route": "qwen3.5-dflash-single-stage-profile",
        "status": "PASS_CAPTURE",
        "formal_latency_evidence": False,
        "strict_greedy_exact_match": None,
        "correctness_gate": {"status": "NOT_RUN_STAGE_DIAGNOSTIC"},
        "profile_stage": stage,
        "profile_output": profiler.output,
        "collector": COLLECTOR,
        "aic_metrics": profiler.metrics,
        "capture_windows": profiler.windows,
        "captured_calls": {
            "prefill": int(stage == "prefill"),
            "draft": int(stage == "draft-verify"),
            "target_verify": int(stage == "draft-verify"),
        },
        "warmup_iterations": warmup,
        "warmup_output_match": stable,
        "profiled_elapsed_ms": profiler.elapsed_ms,
        "profiled_elapsed_excludes": "msprof attach/start/stop/quit and controller waits",
        "prompt_tokens": len(prompt),
        "block_size": block_size,
        "proposal_capacity": block_size - 1,
        "max_new_tokens_applies": False,
        "stage_scope": (
            "one complete Target.begin_rollback: fresh-cache reset, all real "
            "prompt chunks, feature capture and final-row LM head; excludes "
            "Draft feature projection and anchor Top1"
            if stage == "prefill" else
            "first Draft proposal, proposal normalization/block construction "
            "and Target verify including first-round state-bank preparation; "
            "excludes prefill, prompt feature projection, Target Top1, accept and commit"
        ),
        "synchronization": "before start and before stop; no added Draft/verify barrier",
        "result": measured,
    }
