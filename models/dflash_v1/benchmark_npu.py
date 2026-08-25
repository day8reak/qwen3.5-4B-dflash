"""Device-synchronized NPU benchmark for the packaged DFlash V1 runtime.

The benchmark keeps model loading, checkpoint verification, and the mandatory
ordinary/DFlash strict-greedy correctness gate outside the timed interval.  A
timed iteration covers one complete generation call and an explicit device
synchronization.  Operator attribution belongs to msprof; the host clock is
used only for synchronized end-to-end latency and output throughput.

This measures the repository's current correctness-first routes:

* ``ordinary`` is ordinary full-prefix greedy replay;
* ``dflash`` is target bootstrap plus sequential full-prefix DFlash replay.

It must not be described as the future incremental/vectorized DFlash runtime.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import ContextManager, Literal

sys.dont_write_bytecode = True

import torch

from .dflash_qwen_adapter_v1 import (
    Qwen35DFlashFullPrefixAdapter,
    _audit_target_config,
    _configure_embedded_npu_inputs,
    _decode_payload,
    _dflash_execution_gate,
    _draft_device_memory_preflight,
    _dtype,
    _emit_progress,
    _load_target,
    _prepare_device_backend,
    _request_payload,
    _resolve_prompt,
    _runtime_identity,
    _select_draft_ops,
    _target_forward_reconciliation,
    _target_integration_audit,
    _validate_experiment_dtype,
    _validate_formal_cli_inputs,
    _validate_ops_backend_request,
    validate_qwen35_dflash_strict_greedy,
)
from .dflash_reference_decode_v1 import (
    ReplayDecodeResult,
    ReplayDecodeStats,
    ReplayRound,
    dflash_full_prefix_greedy,
    ordinary_full_prefix_greedy,
)
from .dflash_weights import require_official_dflash_checkpoint
from .internal_target_loader import (
    DECODE_CHUNK_SIZE_ENV,
    PREFILL_CHUNK_SIZE_ENV,
)
from .modeling_dflash import DFlashDraftModel
from .run_npu import DEFAULT_TARGET_FACTORY, KV_CACHE_MAX_LEN_ENV


BenchmarkMode = Literal["ordinary", "dflash"]
Synchronize = Callable[[], None]
RangeFactory = Callable[[str], ContextManager[None]]


@dataclass(frozen=True)
class BenchmarkConfig:
    """Controls for one separately executed benchmark mode."""

    mode: BenchmarkMode
    warmup: int = 3
    repetitions: int = 10
    max_new_tokens: int = 32
    max_draft_tokens: int = 16

    def validate(self) -> None:
        if self.mode not in {"ordinary", "dflash"}:
            raise ValueError(f"unsupported benchmark mode: {self.mode!r}")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.max_new_tokens < 2:
            raise ValueError(
                "NPU benchmark max_new_tokens must be at least 2 so the "
                "correctness gate executes a post-bootstrap DFlash round"
            )
        if not 1 <= self.max_draft_tokens <= 16:
            raise ValueError("max_draft_tokens must be between 1 and 16")


@dataclass(frozen=True)
class BenchmarkInvocation:
    """One generation result and the adapter work that produced it."""

    result: ReplayDecodeResult
    adapter_stats: Mapping[str, object]

    @property
    def target_forward_calls(self) -> int:
        return int(self.adapter_stats.get("target_logit_calls", 0)) + int(
            self.adapter_stats.get("target_feature_calls", 0)
        )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("a percentile requires at least one value")
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * percentile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    fraction = rank - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def _result_identity(result: ReplayDecodeResult) -> dict[str, object]:
    generated = [int(token) for token in result.generated_token_ids]
    return {
        "generated_token_ids": generated,
        "generated_token_ids_sha256": hashlib.sha256(
            json.dumps(generated, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "generated_tokens": len(generated),
        "reached_eos": bool(result.reached_eos),
        "stop_reason": result.stop_reason,
    }


def _require_expected_result(
    expected: ReplayDecodeResult,
    actual: ReplayDecodeResult,
    *,
    phase: str,
    iteration: int,
) -> None:
    fields = (
        "prompt_token_ids",
        "generated_token_ids",
        "reached_eos",
        "stop_reason",
    )
    mismatches = [
        field for field in fields if getattr(expected, field) != getattr(actual, field)
    ]
    if mismatches:
        raise RuntimeError(
            "benchmark output changed from the strict-greedy correctness gate: "
            f"phase={phase}, iteration={iteration}, fields={mismatches}"
        )


def _replay_stats(result: ReplayDecodeResult) -> dict[str, object]:
    payload = asdict(result.stats)
    payload["acceptance_rate"] = result.stats.acceptance_rate
    return payload


def run_benchmark(
    run_once: Callable[[], BenchmarkInvocation],
    *,
    expected: ReplayDecodeResult,
    config: BenchmarkConfig,
    synchronize: Synchronize,
    synchronization_source: str,
    range_factory: RangeFactory | None = None,
    range_source: str = "disabled",
) -> dict[str, object]:
    """Run stable warmups and synchronized whole-generation measurements."""

    config.validate()
    make_range = range_factory or (lambda _label: nullcontext())
    total_target_forward_calls = 0
    warmup_target_forward_calls = 0

    for index in range(config.warmup):
        synchronize()
        with make_range(f"qwen35/{config.mode}/warmup/{index}"):
            invocation = run_once()
            synchronize()
        _require_expected_result(
            expected,
            invocation.result,
            phase="warmup",
            iteration=index,
        )
        warmup_target_forward_calls += invocation.target_forward_calls
        total_target_forward_calls += invocation.target_forward_calls

    elapsed_seconds: list[float] = []
    measurements: list[dict[str, object]] = []
    measured_target_forward_calls = 0
    for index in range(config.repetitions):
        synchronize()
        with make_range(f"qwen35/{config.mode}/measure/{index}"):
            started_ns = time.perf_counter_ns()
            invocation = run_once()
            synchronize()
            elapsed = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        _require_expected_result(
            expected,
            invocation.result,
            phase="measure",
            iteration=index,
        )
        if elapsed <= 0:
            raise RuntimeError("benchmark clock returned a non-positive duration")
        generated_tokens = len(invocation.result.generated_token_ids)
        elapsed_seconds.append(elapsed)
        measured_target_forward_calls += invocation.target_forward_calls
        total_target_forward_calls += invocation.target_forward_calls
        measurements.append(
            {
                "iteration": index,
                "elapsed_seconds": elapsed,
                "elapsed_milliseconds": elapsed * 1000,
                "output_tokens_per_second": generated_tokens / elapsed,
                "result": _result_identity(invocation.result),
                "replay_stats": _replay_stats(invocation.result),
                "adapter_stats": dict(invocation.adapter_stats),
                "target_forward_calls": invocation.target_forward_calls,
            }
        )

    generated_tokens = len(expected.generated_token_ids)
    total_seconds = sum(elapsed_seconds)
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": config.mode,
        "configuration": {
            "warmup": config.warmup,
            "repetitions": config.repetitions,
            "max_new_tokens": config.max_new_tokens,
            "max_draft_tokens": config.max_draft_tokens,
        },
        "timing_scope": (
            "complete generation after target/draft load and correctness gate; "
            "includes host orchestration and final device synchronization; "
            "excludes model loading and prompt tokenization"
        ),
        "synchronization": {
            "source": synchronization_source,
            "applied_before_and_after_each_iteration": True,
        },
        "ranges": {
            "source": range_source,
            "warmup_pattern": f"qwen35/{config.mode}/warmup/<index>",
            "measurement_pattern": f"qwen35/{config.mode}/measure/<index>",
        },
        "expected_result": _result_identity(expected),
        "warmup": {
            "count": config.warmup,
            "stable_result_checked": True,
            "target_forward_calls": warmup_target_forward_calls,
        },
        "measurements": measurements,
        "summary": {
            "count": len(elapsed_seconds),
            "latency_ms": {
                "min": min(elapsed_seconds) * 1000,
                "max": max(elapsed_seconds) * 1000,
                "mean": statistics.fmean(elapsed_seconds) * 1000,
                "median": statistics.median(elapsed_seconds) * 1000,
                "p90": _percentile(elapsed_seconds, 0.90) * 1000,
                "population_stdev": statistics.pstdev(elapsed_seconds) * 1000,
            },
            "aggregate_output_tokens_per_second": (
                generated_tokens * len(elapsed_seconds) / total_seconds
            ),
            "generated_tokens_per_iteration": generated_tokens,
            "measured_target_forward_calls": measured_target_forward_calls,
        },
        "target_forward_calls": total_target_forward_calls,
        "route_boundary": (
            "ordinary is full-prefix greedy replay; dflash is sequential "
            "full-prefix DFlash V1, not incremental/vectorized DFlash"
        ),
    }


def _ordinary_invocation(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
) -> BenchmarkInvocation:
    adapter.reset_stats()
    result = ordinary_full_prefix_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    return BenchmarkInvocation(result=result, adapter_stats=asdict(adapter.snapshot_stats()))


def _dflash_invocation(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    max_draft_tokens: int,
    eos_token_ids: Sequence[int],
) -> BenchmarkInvocation:
    """Execute the same target-bootstrap DFlash route used by validation."""

    adapter.reset_stats()
    bootstrap = ordinary_full_prefix_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=min(max_new_tokens, 1),
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    bootstrap_token = (
        bootstrap.generated_token_ids[0] if bootstrap.generated_token_ids else None
    )
    if bootstrap_token is None or bootstrap.reached_eos:
        tail = None
    else:
        tail = dflash_full_prefix_greedy(
            adapter,
            adapter,
            (*bootstrap.prompt_token_ids, bootstrap_token),
            max_new_tokens=max_new_tokens - 1,
            max_draft_tokens=max_draft_tokens,
            eos_token_ids=eos_token_ids,
            input_device=adapter.device,
            verification_mode="sequential",
        )

    tail_stats = ReplayDecodeStats() if tail is None else tail.stats
    bootstrap_trace = (
        ()
        if bootstrap_token is None
        else (
            ReplayRound(
                committed_prefix_length=len(bootstrap.prompt_token_ids),
                proposed_token_ids=(),
                target_token_ids=(bootstrap_token,),
                accepted_draft_token_ids=(),
                fallback_token_id=bootstrap_token,
                emitted_token_ids=(bootstrap_token,),
            ),
        )
    )
    replay_stats = ReplayDecodeStats(
        target_calls=bootstrap.stats.target_calls + tail_stats.target_calls,
        target_verify_calls=(
            bootstrap.stats.target_verify_calls + tail_stats.target_verify_calls
        ),
        target_input_tokens_recomputed=(
            bootstrap.stats.target_input_tokens_recomputed
            + tail_stats.target_input_tokens_recomputed
        ),
        target_rows_read=(
            bootstrap.stats.target_rows_read + tail_stats.target_rows_read
        ),
        draft_calls=tail_stats.draft_calls,
        drafted_tokens=tail_stats.drafted_tokens,
        accepted_draft_tokens=tail_stats.accepted_draft_tokens,
        rejected_draft_tokens=tail_stats.rejected_draft_tokens,
        fallback_tokens=(1 if bootstrap_token is not None else 0)
        + tail_stats.fallback_tokens,
    )
    tail_generated = () if tail is None else tail.generated_token_ids
    generated = (
        () if bootstrap_token is None else (bootstrap_token, *tail_generated)
    )
    reached_eos = bootstrap.reached_eos or (
        False if tail is None else tail.reached_eos
    )
    result = ReplayDecodeResult(
        mode="qwen3.5-dflash-v1-target-bootstrap-sequential-full-prefix-replay",
        prompt_token_ids=bootstrap.prompt_token_ids,
        generated_token_ids=generated,
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        stats=replay_stats,
        rounds=bootstrap_trace + (() if tail is None else tail.rounds),
    )
    return BenchmarkInvocation(result=result, adapter_stats=asdict(adapter.snapshot_stats()))


def _device_synchronize(device: torch.device) -> None:
    backend = getattr(torch, device.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(f"torch.{device.type}.synchronize is unavailable")
    try:
        synchronize(device)
    except TypeError:
        synchronize()


def _mstx_range_factory(enabled: bool) -> tuple[RangeFactory | None, str]:
    if not enabled:
        return None, "disabled"
    try:
        import mstx  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "MSTX ranges were requested but the mstx Python package is unavailable"
        ) from error

    @contextmanager
    def traced(label: str):
        range_id = mstx.range_start(label, None)
        if range_id in (None, 0):
            raise RuntimeError(f"mstx.range_start failed for {label!r}")
        try:
            yield
        finally:
            mstx.range_end(range_id)

    return traced, "mstx.range_start/range_end"


def _call_memory_method(
    backend: object,
    method_name: str,
    device: torch.device,
) -> int | None:
    method = getattr(backend, method_name, None)
    if not callable(method):
        return None
    try:
        value = method(device)
    except TypeError:
        value = method()
    return int(value)


def _memory_snapshot(device: torch.device) -> dict[str, object]:
    backend = getattr(torch, device.type, None)
    if backend is None:
        return {"status": "UNAVAILABLE_BACKEND"}
    values = {
        name: _call_memory_method(backend, name, device)
        for name in (
            "memory_allocated",
            "max_memory_allocated",
            "memory_reserved",
            "max_memory_reserved",
        )
    }
    return {
        "status": (
            "PASS" if any(value is not None for value in values.values()) else "UNAVAILABLE"
        ),
        **values,
    }


def _reset_peak_memory(device: torch.device) -> dict[str, object]:
    _device_synchronize(device)
    baseline = _memory_snapshot(device)
    backend = getattr(torch, device.type, None)
    reset = getattr(backend, "reset_peak_memory_stats", None)
    if callable(reset):
        try:
            reset(device)
        except TypeError:
            reset()
        reset_status = "PASS"
    else:
        reset_status = "UNAVAILABLE"
    return {"baseline_after_correctness_gate": baseline, "peak_reset": reset_status}


def _source_identity() -> dict[str, object]:
    repository = Path(__file__).resolve().parents[2]
    paths = (
        Path(__file__).resolve(),
        repository / "tools" / "run_msprof.sh",
        repository / "docs" / "NPU_BENCHMARK.md",
        repository / "config" / "npu_benchmark_v1.json",
        repository / "SOURCE_LOCK.json",
    )
    files: dict[str, str] = {}
    for path in paths:
        if path.is_file():
            files[str(path.relative_to(repository))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    return {"repository": str(repository), "file_sha256": files}


def _npu_smi_identity() -> dict[str, object]:
    try:
        completed = subprocess.run(
            ["npu-smi", "info"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except FileNotFoundError as error:
        raise RuntimeError("npu-smi is required for benchmark identity") from error
    except subprocess.TimeoutExpired as error:
        raise RuntimeError("npu-smi info timed out") from error
    output = completed.stdout + completed.stderr
    if completed.returncode != 0:
        raise RuntimeError(
            f"npu-smi info failed with exit code {completed.returncode}: {output}"
        )
    return {
        "command": ["npu-smi", "info"],
        "exit_code": completed.returncode,
        "output": output,
        "output_sha256": hashlib.sha256(output.encode("utf-8")).hexdigest(),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the packaged Qwen3.5-4B ordinary or DFlash V1 route "
            "on a real NPU after a strict-greedy correctness gate"
        )
    )
    parser.add_argument("--mode", choices=("ordinary", "dflash"), required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument(
        "--target-factory",
        default=DEFAULT_TARGET_FACTORY,
        help="advanced override for the packaged HIAI target factory",
    )
    parser.add_argument("--reset-hook", help="advanced target state reset hook")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-ids", help="comma-separated token IDs")
    prompt.add_argument("--prompt-json", help="JSON token list or input_ids object")
    prompt.add_argument("--prompt", help="UTF-8 prompt text")
    prompt.add_argument("--prompt-file", help="path to a UTF-8 prompt text file")
    parser.add_argument("--prompt-mode", choices=("chat", "raw"), default="chat")
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--max-draft-tokens", type=int, default=16)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--kv-cache-max-len", type=int, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--decode-chunk-size", type=int, default=1)
    parser.add_argument("--report", required=True)
    parser.add_argument(
        "--mstx",
        action=argparse.BooleanOptionalAction,
        default=os.environ.get("DFLASH_BENCHMARK_MSTX") == "1",
        help="emit MSTX ranges around warmup/measurement iterations",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _configure_formal_args(args: argparse.Namespace) -> BenchmarkConfig:
    config = BenchmarkConfig(
        mode=args.mode,
        warmup=args.warmup,
        repetitions=args.repetitions,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
    )
    config.validate()
    if str(args.device).split(":", 1)[0].lower() != "npu":
        raise ValueError("benchmark_npu requires --device npu or npu:N")
    for name in (
        "kv_cache_max_len",
        "prefill_chunk_size",
        "decode_chunk_size",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")

    args.target_loader = None
    args.hiai_source = None
    args.npu_layout = "embedded"
    args.dtype = "float16"
    args.eos_token_id = [248044]
    args.ops_backend = None
    args.allow_op_fallback = False
    args.allow_download = False
    args.trust_remote_code = False
    os.environ[PREFILL_CHUNK_SIZE_ENV] = str(args.prefill_chunk_size)
    os.environ[DECODE_CHUNK_SIZE_ENV] = str(args.decode_chunk_size)
    os.environ[KV_CACHE_MAX_LEN_ENV] = str(args.kv_cache_max_len)
    _configure_embedded_npu_inputs(args)
    return config


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.benchmark-tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError(f"temporary report path already exists: {temporary}")
    try:
        temporary.write_text(serialized + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _configure_formal_args(args)
    _validate_ops_backend_request(args.device, args.ops_backend)
    initial_runtime_preflight = _validate_formal_cli_inputs(args)
    if initial_runtime_preflight is None:
        raise AssertionError("formal NPU preflight unexpectedly returned None")

    dtype = _dtype(args.dtype)
    _prepare_device_backend(args.device)
    _validate_experiment_dtype(args.device, dtype)
    npu_smi_identity = _npu_smi_identity()
    target_root = Path(args.target_dir).expanduser().resolve()
    target_checkpoint = _audit_target_config(target_root)
    prompt_ids, tokenizer = _resolve_prompt(args, target_root=target_root)

    _emit_progress(args.progress, "benchmark_draft_checkpoint_audit_begin", {})
    draft_checkpoint = require_official_dflash_checkpoint(
        args.draft_dir,
        verify_model_hash=True,
    )
    _emit_progress(args.progress, "benchmark_target_load_begin", {})
    target = _load_target(
        args.target_dir,
        target_loader=args.target_loader,
        hiai_source=args.hiai_source,
        device=args.device,
        dtype=dtype,
        allow_download=False,
        trust_remote_code=False,
    )
    initial_target_integration = _target_integration_audit(
        target,
        device=args.device,
        target_loader=args.target_loader,
        require_completed_forward=False,
    )
    draft_memory_preflight = _draft_device_memory_preflight(
        args.device,
        dtype,
        draft_checkpoint,
    )
    ops, backend = _select_draft_ops(
        device=args.device,
        ops_backend=None,
        allow_op_fallback=False,
    )
    _emit_progress(args.progress, "benchmark_draft_load_begin", {"backend": backend})
    draft = DFlashDraftModel.from_pretrained(
        args.draft_dir,
        ops=ops,
        device=args.device,
        dtype=dtype,
    )
    adapter = Qwen35DFlashFullPrefixAdapter(target, draft)
    _emit_progress(args.progress, "benchmark_correctness_gate_begin", {})
    validation = validate_qwen35_dflash_strict_greedy(
        adapter,
        prompt_ids,
        max_new_tokens=config.max_new_tokens,
        max_draft_tokens=config.max_draft_tokens,
        eos_token_ids=args.eos_token_id,
        progress_callback=lambda event, fields: _emit_progress(
            args.progress,
            event,
            fields,
        ),
    )
    execution_gate = _dflash_execution_gate(validation, formal_npu=True)
    correctness_target_integration = _target_integration_audit(
        target,
        device=args.device,
        target_loader=args.target_loader,
        require_completed_forward=True,
    )
    expected_correctness_calls = (
        validation.predecode_gate_target_calls
        + validation.ordinary_adapter_stats.target_logit_calls
        + validation.dflash_adapter_stats.target_logit_calls
        + validation.dflash_adapter_stats.target_feature_calls
    )
    correctness_target_integration["validation_call_reconciliation"] = (
        _target_forward_reconciliation(
            initial_target_integration,
            correctness_target_integration,
            expected_validation_calls=expected_correctness_calls,
            formal_npu=True,
        )
    )
    _emit_progress(
        args.progress,
        "benchmark_correctness_gate_end",
        {"status": "PASS", "strict_greedy_exact_match": True},
    )

    expected = validation.ordinary if config.mode == "ordinary" else validation.dflash
    if config.mode == "ordinary":
        run_once = lambda: _ordinary_invocation(
            adapter,
            prompt_ids,
            max_new_tokens=config.max_new_tokens,
            eos_token_ids=args.eos_token_id,
        )
    else:
        run_once = lambda: _dflash_invocation(
            adapter,
            prompt_ids,
            max_new_tokens=config.max_new_tokens,
            max_draft_tokens=config.max_draft_tokens,
            eos_token_ids=args.eos_token_id,
        )

    range_factory, range_source = _mstx_range_factory(bool(args.mstx))
    device = adapter.device
    memory = _reset_peak_memory(device)
    _emit_progress(
        args.progress,
        "benchmark_measurement_begin",
        {
            "mode": config.mode,
            "warmup": config.warmup,
            "repetitions": config.repetitions,
            "mstx": bool(args.mstx),
        },
    )
    benchmark = run_benchmark(
        run_once,
        expected=expected,
        config=config,
        synchronize=lambda: _device_synchronize(device),
        synchronization_source=f"torch.{device.type}.synchronize",
        range_factory=range_factory,
        range_source=range_source,
    )
    _device_synchronize(device)
    memory["after_measurement"] = _memory_snapshot(device)
    _emit_progress(args.progress, "benchmark_measurement_end", {"status": "PASS"})

    final_target_integration = _target_integration_audit(
        target,
        device=args.device,
        target_loader=args.target_loader,
        require_completed_forward=True,
    )
    final_target_integration["benchmark_call_reconciliation"] = (
        _target_forward_reconciliation(
            correctness_target_integration,
            final_target_integration,
            expected_validation_calls=int(benchmark["target_forward_calls"]),
            formal_npu=True,
        )
    )
    final_runtime_preflight = _validate_formal_cli_inputs(args)
    if final_runtime_preflight != initial_runtime_preflight:
        raise RuntimeError("embedded runtime identity changed during benchmark")

    report = {
        "schema_version": 1,
        "status": "PASS",
        "route": "qwen3.5-dflash-v1-npu-whole-generation-benchmark",
        "classification": "real NPU synchronized execution",
        "mode": config.mode,
        "strict_greedy_exact_match": True,
        "request": _request_payload(
            args,
            effective_max_draft_tokens=config.max_draft_tokens,
            prompt_token_ids=prompt_ids,
        ),
        "runtime_identity": _runtime_identity(device),
        "npu_smi_identity": npu_smi_identity,
        "runtime_preflight": final_runtime_preflight,
        "source_identity": _source_identity(),
        "target_checkpoint": target_checkpoint,
        "draft_checkpoint": draft_checkpoint,
        "draft_memory_preflight": draft_memory_preflight,
        "ops_backend": backend,
        "operator_fallback_enabled": False,
        "target_integration": final_target_integration,
        "correctness_gate": {
            "status": "PASS",
            "dflash_execution": execution_gate,
            "target_integration": correctness_target_integration,
            "ordinary": _decode_payload(validation.ordinary, tokenizer=tokenizer),
            "dflash": _decode_payload(validation.dflash, tokenizer=tokenizer),
        },
        "benchmark": benchmark,
        "accelerator_memory": memory,
        "claim_boundary": (
            "host timings are valid only for this locked current full-prefix "
            "implementation and device identity; msprof runs are diagnostic, "
            "while performance comparisons require separate unprofiled "
            "ordinary/dflash processes with 3 warmups and 10 stable repetitions"
        ),
    }
    destination = Path(args.report).expanduser().resolve()
    _write_report(destination, report)
    postwrite_runtime_preflight = _validate_formal_cli_inputs(args)
    if postwrite_runtime_preflight != initial_runtime_preflight:
        raise RuntimeError("embedded runtime identity changed while writing report")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
