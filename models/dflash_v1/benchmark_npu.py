"""Synchronized whole-generation benchmark for the NPU rollback runtime.

The benchmark branch originally measured the full-prefix V1 route.  This
version preserves its measurement contract (separate mode processes, strict
correctness gate, warmups, ten retained measurements, device synchronization,
memory evidence, and optional MSTX ranges) but invokes the transactional
rollback scheduler used by :mod:`models.dflash_v1.run_npu`.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass, field
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

from . import dflash_qwen_adapter_v1 as _legacy
from .dflash_config import DFLASH_MIN_BLOCK_SIZE, OFFICIAL_DFLASH_BLOCK_SIZE
from .dflash_reference_decode_v1 import ReplayDecodeResult
from .dflash_rollback_adapter import (
    Qwen35DFlashRollbackAdapter,
    validate_qwen35_dflash_rollback,
)
from .dflash_rollback_decode import (
    dflash_rollback_greedy,
    ordinary_incremental_greedy,
)
from .dflash_weights import require_official_dflash_checkpoint
from .internal_target_loader import DECODE_CHUNK_SIZE_ENV, PREFILL_CHUNK_SIZE_ENV
from .modeling_dflash import DFlashDraftModel
from .run_npu import (
    KV_CACHE_MAX_LEN_ENV,
    ORIGINAL_QUANT_DISABLE,
    ORIGINAL_QUANT_ENABLE,
    _configure_target_quantization,
)
from .run_rollback import (
    DEFAULT_NPU_TARGET_FACTORY,
    _atomic_report,
    _load_transactional_target,
    _rollback_runtime_identity,
    _synchronize_device,
)


BenchmarkMode = Literal["ordinary", "dflash"]
Synchronize = Callable[[], None]
RangeFactory = Callable[[str], ContextManager[None]]

_TARGET_AUDIT_COUNTER_FIELDS = frozenset(
    {
        # HIAI receiver counters.
        "ordinary_prefill_token_calls",
        "ordinary_prefill_lm_head_skips",
        "ordinary_decode_calls",
        "rollback_prefill_token_calls",
        "rollback_prefill_lm_head_skips",
        "rollback_verify_calls",
        "rollback_commit_calls",
        "rollback_aborts",
        # Framework transaction counters.
        "ordinary_prefill_calls",
        "rollback_prefill_calls",
        "rollback_commit_transactions",
        "rollback_commit_replay_calls",
    }
)
_TARGET_AUDIT_SESSION_FIELDS = (
    "persistent_mode",
    "persistent_cursor",
    "previous_accepted",
    "pending_verify_rows",
    "session_invalid",
    "pending_transaction",
    "cache_sequence_length",
)
_DRAFT_KV_CACHE_COUNTER_FIELDS = (
    "rounds",
    "aborted_rounds",
    "crop_calls",
    "tokens_appended",
    "tokens_reused",
)


def _benchmark_range_label(
    mode: BenchmarkMode,
    phase: Literal["warmup", "measure"],
    index: int,
) -> str:
    """Build a conservative MSTX message accepted by older CANN releases."""

    if mode not in {"ordinary", "dflash"}:
        raise ValueError(f"unsupported benchmark range mode: {mode!r}")
    if phase not in {"warmup", "measure"}:
        raise ValueError(f"unsupported benchmark range phase: {phase!r}")
    if isinstance(index, bool) or not isinstance(index, int) or index < 0:
        raise ValueError("benchmark range index must be a non-negative integer")
    return f"qwen35_{mode}_{phase}_{index}"


@dataclass(frozen=True)
class BenchmarkConfig:
    """Controls for one separately executed benchmark mode."""

    mode: BenchmarkMode
    warmup: int = 3
    repetitions: int = 10
    max_new_tokens: int = 32
    block_size: int = OFFICIAL_DFLASH_BLOCK_SIZE

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
                "correctness gate executes a rollback draft round"
            )
        if not DFLASH_MIN_BLOCK_SIZE <= self.block_size <= OFFICIAL_DFLASH_BLOCK_SIZE:
            raise ValueError(
                "block_size must be between "
                f"{DFLASH_MIN_BLOCK_SIZE} and {OFFICIAL_DFLASH_BLOCK_SIZE}"
            )


@dataclass(frozen=True)
class BenchmarkInvocation:
    """One generation result plus its bounded runtime counters."""

    result: ReplayDecodeResult
    adapter_stats: Mapping[str, object]
    target_audit_delta: Mapping[str, object] = field(default_factory=dict)
    draft_kv_cache_audit: Mapping[str, object] = field(default_factory=dict)

    @property
    def target_forward_calls(self) -> int:
        audited = self.target_audit_delta.get("target_execution_calls")
        if isinstance(audited, int) and not isinstance(audited, bool):
            return audited
        keys = (
            "ordinary_prefill_calls",
            "ordinary_decode_calls",
            "rollback_prefill_calls",
            "rollback_verify_calls",
        )
        return sum(int(self.adapter_stats.get(name, 0)) for name in keys)


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
        name for name in fields if getattr(expected, name) != getattr(actual, name)
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
        with make_range(_benchmark_range_label(config.mode, "warmup", index)):
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
        with make_range(_benchmark_range_label(config.mode, "measure", index)):
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
                "target_audit_delta": dict(invocation.target_audit_delta),
                "draft_kv_cache_audit": dict(invocation.draft_kv_cache_audit),
                "target_forward_calls": invocation.target_forward_calls,
            }
        )

    generated_tokens = len(expected.generated_token_ids)
    total_seconds = sum(elapsed_seconds)
    return {
        "schema_version": 3,
        "status": "PASS",
        "mode": config.mode,
        "configuration": {
            "warmup": config.warmup,
            "repetitions": config.repetitions,
            "max_new_tokens": config.max_new_tokens,
            "block_size": config.block_size,
            "proposal_capacity": config.block_size - 1,
        },
        "timing_scope": (
            "complete rollback generation after target/draft load and correctness "
            "gate; includes host orchestration and final device synchronization; "
            "excludes model loading, checkpoint hashing, and prompt tokenization"
        ),
        "synchronization": {
            "source": synchronization_source,
            "applied_before_and_after_each_iteration": True,
        },
        "ranges": {
            "source": range_source,
            "message_character_policy": "ASCII letters, digits, and underscore",
            "warmup_pattern": f"qwen35_{config.mode}_warmup_<index>",
            "measurement_pattern": f"qwen35_{config.mode}_measure_<index>",
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
            "ordinary and DFlash both use persistent target state; DFlash uses "
            "T=K+1 transactional verify/rollback and never verifies by replaying "
            "the committed historical prefix"
        ),
    }


def _target_audit(target: object) -> dict[str, object]:
    raw = getattr(target, "dflash_rollback_audit", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _draft_kv_cache_audit(adapter: object) -> dict[str, object]:
    raw = getattr(adapter, "dflash_draft_cache_audit", None)
    return dict(raw) if isinstance(raw, Mapping) else {}


def _draft_kv_cache_invocation_audit(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    if not after:
        return {}
    audit = dict(after)
    counter_delta: dict[str, int] = {}
    for name in _DRAFT_KV_CACHE_COUNTER_FIELDS:
        before_value = before.get(name, 0)
        after_value = after.get(name)
        if (
            not isinstance(before_value, int)
            or isinstance(before_value, bool)
            or not isinstance(after_value, int)
            or isinstance(after_value, bool)
        ):
            raise TypeError(f"Draft KV cache audit counter must be integer: {name}")
        if after_value < before_value:
            raise RuntimeError(f"Draft KV cache counter moved backwards: {name}")
        counter_delta[name] = after_value - before_value
    audit["invocation_counter_delta"] = counter_delta
    return audit


def _target_audit_delta(
    before: Mapping[str, object],
    after: Mapping[str, object],
) -> dict[str, object]:
    """Subtract cumulative counters while snapshotting resettable session state."""

    delta: dict[str, object] = {}
    declared_after = after.get("cumulative_counter_fields")
    declared_before = before.get("cumulative_counter_fields")
    if declared_after is None:
        counter_fields = sorted(
            name
            for name in _TARGET_AUDIT_COUNTER_FIELDS
            if name in before and name in after
        )
    else:
        if not isinstance(declared_after, (tuple, list)) or not all(
            isinstance(name, str) for name in declared_after
        ):
            raise TypeError(
                "rollback audit cumulative_counter_fields must be a string sequence"
            )
        if declared_before is not None and tuple(declared_before) != tuple(
            declared_after
        ):
            raise RuntimeError("rollback audit counter schema changed during invocation")
        counter_fields = list(declared_after)

    for name in counter_fields:
        if name not in before or name not in after:
            raise RuntimeError(f"rollback audit counter is missing: {name}")
        after_value = after[name]
        before_value = before.get(name)
        if (
            not isinstance(after_value, int)
            or isinstance(after_value, bool)
            or not isinstance(before_value, int)
            or isinstance(before_value, bool)
        ):
            raise TypeError(f"rollback audit counter must be an integer: {name}")
        if after_value < before_value:
            raise RuntimeError(f"rollback audit counter moved backwards: {name}")
        delta[name] = after_value - before_value

    session_state_after = {
        name: after[name]
        for name in _TARGET_AUDIT_SESSION_FIELDS
        if name in after
    }
    if session_state_after:
        delta["session_state_after"] = session_state_after
    execution_keys = (
        "ordinary_prefill_token_calls",
        "ordinary_decode_calls",
        "rollback_prefill_token_calls",
        "rollback_verify_calls",
    )
    present = [name for name in execution_keys if name in delta]
    if present:
        delta["target_execution_calls"] = sum(int(delta[name]) for name in present)
    return delta


def _ordinary_invocation(
    adapter: Qwen35DFlashRollbackAdapter,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Sequence[int],
) -> BenchmarkInvocation:
    adapter.reset_rollback_stats()
    before = _target_audit(adapter.target)
    draft_before = _draft_kv_cache_audit(adapter)
    result = ordinary_incremental_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    after = _target_audit(adapter.target)
    draft_after = _draft_kv_cache_audit(adapter)
    return BenchmarkInvocation(
        result=result,
        adapter_stats=asdict(adapter.snapshot_rollback_stats()),
        target_audit_delta=_target_audit_delta(before, after),
        draft_kv_cache_audit=_draft_kv_cache_invocation_audit(
            draft_before,
            draft_after,
        ),
    )


def _dflash_invocation(
    adapter: Qwen35DFlashRollbackAdapter,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    block_size: int,
    eos_token_ids: Sequence[int],
) -> BenchmarkInvocation:
    adapter.reset_rollback_stats()
    before = _target_audit(adapter.target)
    draft_before = _draft_kv_cache_audit(adapter)
    result = dflash_rollback_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    after = _target_audit(adapter.target)
    draft_after = _draft_kv_cache_audit(adapter)
    return BenchmarkInvocation(
        result=result,
        adapter_stats=asdict(adapter.snapshot_rollback_stats()),
        target_audit_delta=_target_audit_delta(before, after),
        draft_kv_cache_audit=_draft_kv_cache_invocation_audit(
            draft_before,
            draft_after,
        ),
    )


def _device_synchronize(device: torch.device) -> None:
    _synchronize_device(device)


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
            raise RuntimeError(
                f"mstx.range_start failed for safe marker {label!r}; verify the "
                "CANN/mstx profiling environment or rerun run_msprof.sh with "
                "--no-msproftx"
            )
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


def _source_identity(package_dir: Path) -> dict[str, object]:
    repository = package_dir.parents[1]
    paths = (
        Path(__file__).resolve(),
        package_dir / "run_npu.py",
        package_dir / "run_rollback.py",
        package_dir / "target_quant.py",
        package_dir / "dflash_rollback_decode.py",
        package_dir / "dflash_rollback_adapter.py",
        repository / "models" / "internal_dflash_bridge.py",
        repository / "models" / "modeling_qwen3_5_hiai_nd_dflash_rollback.py",
        repository / "models" / "modeling_qwen3_5_hiai_nd.py",
        repository / "models" / "export_model_wrapper_qwen3_5_dflash_rollback.py",
        repository / "tools" / "run_msprof.sh",
        repository / "config" / "npu_benchmark_v1.json",
        repository / "docs" / "DFLASH_RUN_AND_VALIDATE.md",
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
            "Benchmark the Qwen3.5-4B ordinary or DFlash transactional rollback "
            "route on a real NPU after a strict-greedy correctness gate"
        )
    )
    parser.add_argument("--mode", choices=("ordinary", "dflash"), required=True)
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument(
        "--target-factory",
        default=DEFAULT_NPU_TARGET_FACTORY,
        help="advanced override for the packaged rollback target factory",
    )
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
    parser.add_argument(
        "--block-size",
        type=int,
        default=OFFICIAL_DFLASH_BLOCK_SIZE,
        help="total verify rows B including one anchor (official range: 2..16)",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--kv-cache-max-len", type=int, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--decode-chunk-size", type=int, default=1)
    parser.add_argument(
        "--config",
        help="original inference YAML used when --quant_mode enable",
    )
    parser.add_argument(
        "--quant_mode",
        "--quant-mode",
        dest="quant_mode",
        choices=(ORIGINAL_QUANT_ENABLE, ORIGINAL_QUANT_DISABLE),
        default=ORIGINAL_QUANT_DISABLE,
        help="same Target quantization switch as inference.py",
    )
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
        block_size=args.block_size,
    )
    config.validate()
    if str(args.device).split(":", 1)[0].lower() != "npu":
        raise ValueError("benchmark_npu requires --device npu or npu:N")
    for name in ("kv_cache_max_len", "prefill_chunk_size", "decode_chunk_size"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.prefill_chunk_size != 64 or args.decode_chunk_size != 1:
        raise ValueError(
            "rollback HIAI requires prefill-chunk-size=64 and decode-chunk-size=1"
        )
    if args.kv_cache_max_len % 64 != 0:
        raise ValueError("rollback HIAI requires kv-cache-max-len divisible by 64")

    # Compatibility fields consumed by the shared loader/report helpers.
    args.target_loader = None
    args.reset_hook = None
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
    _configure_target_quantization(args)
    return config


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _configure_formal_args(args)
    package_dir = Path(__file__).resolve().parent
    hiai_source = package_dir.parent / "modeling_qwen3_5_hiai_nd_dflash_rollback.py"
    if hiai_source.is_symlink() or not hiai_source.is_file():
        raise FileNotFoundError(f"rollback HIAI source is missing: {hiai_source}")
    _legacy._validate_report_destination(
        args,
        package_dir=package_dir,
        formal_npu=False,
    )
    repository = package_dir.parents[1].resolve()
    report_path = Path(args.report).expanduser().resolve()
    try:
        report_path.relative_to(repository)
    except ValueError:
        pass
    else:
        raise ValueError("benchmark --report must be outside the source repository")
    runtime_identity_before = _rollback_runtime_identity(package_dir)
    source_identity_before = _source_identity(package_dir)
    dtype = _legacy._dtype(args.dtype)
    _legacy._validate_ops_backend_request(args.device, args.ops_backend)
    _legacy._prepare_device_backend(args.device)
    _legacy._validate_experiment_dtype(args.device, dtype)
    npu_smi_identity = _npu_smi_identity()

    target_root = Path(args.target_dir).expanduser().resolve()
    target_checkpoint = _legacy._audit_target_config(target_root)
    prompt_ids, tokenizer = _legacy._resolve_prompt(args, target_root=target_root)
    if len(prompt_ids) + config.max_new_tokens > args.kv_cache_max_len:
        raise ValueError(
            "kv_cache_max_len must cover prompt tokens plus max_new_tokens"
        )

    _legacy._emit_progress(args.progress, "benchmark_checkpoint_audit_begin", {})
    draft_checkpoint = require_official_dflash_checkpoint(
        args.draft_dir,
        verify_model_hash=True,
    )
    _legacy._emit_progress(args.progress, "benchmark_target_load_begin", {})
    target, target_route = _load_transactional_target(args, dtype=dtype)
    draft_memory_preflight = _legacy._draft_device_memory_preflight(
        args.device,
        dtype,
        draft_checkpoint,
    )
    _legacy._emit_progress(
        args.progress,
        "benchmark_draft_load_begin",
        {"target_route": target_route},
    )
    ops, backend = _legacy._select_draft_ops(
        device=args.device,
        ops_backend=None,
        allow_op_fallback=False,
    )
    draft = DFlashDraftModel.from_pretrained(
        args.draft_dir,
        ops=ops,
        device=args.device,
        dtype=dtype,
    )
    adapter = Qwen35DFlashRollbackAdapter(target, draft)
    _device_synchronize(adapter.device)

    _legacy._emit_progress(args.progress, "benchmark_correctness_gate_begin", {})
    validation = validate_qwen35_dflash_rollback(
        adapter,
        prompt_ids,
        max_new_tokens=config.max_new_tokens,
        block_size=config.block_size,
        eos_token_ids=args.eos_token_id,
    )
    _device_synchronize(adapter.device)
    if validation.dflash.stats.draft_calls <= 0:
        raise RuntimeError("benchmark correctness gate executed no Draft round")
    _legacy._emit_progress(
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
            block_size=config.block_size,
            eos_token_ids=args.eos_token_id,
        )

    range_factory, range_source = _mstx_range_factory(bool(args.mstx))
    memory = _reset_peak_memory(adapter.device)
    _legacy._emit_progress(
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
        synchronize=lambda: _device_synchronize(adapter.device),
        synchronization_source=f"torch.{adapter.device.type}.synchronize",
        range_factory=range_factory,
        range_source=range_source,
    )
    _device_synchronize(adapter.device)
    memory["after_measurement"] = _memory_snapshot(adapter.device)

    runtime_identity_after = _rollback_runtime_identity(package_dir)
    source_identity_after = _source_identity(package_dir)
    if runtime_identity_after != runtime_identity_before:
        raise RuntimeError("rollback runtime source identity changed during benchmark")
    if source_identity_after != source_identity_before:
        raise RuntimeError("benchmark source identity changed during benchmark")

    target_quantization = getattr(target, "dflash_target_quantization_audit", None)
    if not isinstance(target_quantization, Mapping):
        raise TypeError("NPU Target must expose dflash_target_quantization_audit")
    report = {
        "schema_version": 4,
        "status": "PASS",
        "route": "qwen3.5-dflash-npu-incremental-rollback-benchmark",
        "classification": "real NPU synchronized rollback execution",
        "mode": config.mode,
        "strict_greedy_exact_match": True,
        "historical_prefix_replay_during_verify": False,
        "request": _legacy._request_payload(
            args,
            effective_block_size=config.block_size,
            prompt_token_ids=prompt_ids,
        ),
        "runtime_identity": _legacy._runtime_identity(adapter.device),
        "npu_smi_identity": npu_smi_identity,
        "rollback_runtime_identity": runtime_identity_after,
        "source_identity": source_identity_after,
        "target_checkpoint": target_checkpoint,
        "draft_checkpoint": draft_checkpoint,
        "draft_memory_preflight": draft_memory_preflight,
        "ops_backend": backend,
        "operator_fallback_enabled": False,
        "target_route": target_route,
        "target_rollback_audit": _target_audit(target),
        "target_quantization": dict(target_quantization),
        "correctness_gate": {
            "status": "PASS",
            "ordinary": _legacy._decode_payload(
                validation.ordinary,
                tokenizer=tokenizer,
            ),
            "dflash": _legacy._decode_payload(
                validation.dflash,
                tokenizer=tokenizer,
            ),
            "ordinary_adapter_stats": asdict(validation.ordinary_adapter_stats),
            "dflash_adapter_stats": asdict(validation.dflash_adapter_stats),
            "draft_kv_cache_audit": dict(validation.draft_kv_cache_audit),
        },
        "benchmark": benchmark,
        "accelerator_memory": memory,
        "claim_boundary": (
            "compare separate unprofiled ordinary and dflash processes with the "
            "same revision, checkpoints, prompt, device, block_size, three "
            "warmups, and ten measurements; msprof runs are diagnostic only"
        ),
    }
    serialized = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    _atomic_report(args.report, serialized)
    _legacy._emit_progress(args.progress, "benchmark_measurement_end", {"status": "PASS"})
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "BenchmarkConfig",
    "BenchmarkInvocation",
    "run_benchmark",
]
