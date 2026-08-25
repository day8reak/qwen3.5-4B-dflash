"""Reproducible whole-generation timing for target backends.

The benchmark deliberately measures complete ordinary or speculative generation
iterations.  Per-stage device attribution belongs to msprof; the host timer is
used only around a device-synchronized end-to-end interval.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
import math
import statistics
import time
from typing import Callable, ContextManager, Iterable, Literal, Sequence

from .backends import DraftBackend, MainBackend
from .generation import GenerationResult, ordinary_generate, speculative_generate


BenchmarkMode = Literal["ordinary", "mtp"]
Synchronize = Callable[[], None]
RangeFactory = Callable[[str], ContextManager[None]]


@dataclass(frozen=True)
class BenchmarkConfig:
    mode: BenchmarkMode
    warmup: int = 3
    repetitions: int = 10
    max_new_tokens: int = 8
    max_draft_tokens: int = 2

    def validate(self) -> None:
        if self.mode not in {"ordinary", "mtp"}:
            raise ValueError(f"unsupported benchmark mode: {self.mode!r}")
        if self.warmup < 0:
            raise ValueError("warmup must be non-negative")
        if self.repetitions <= 0:
            raise ValueError("repetitions must be positive")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive for a benchmark")
        if self.max_draft_tokens <= 0:
            raise ValueError("max_draft_tokens must be positive")


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


def _reset_backend(backend: object | None) -> str | None:
    if backend is None:
        return None
    for method_name in ("reset_benchmark_state", "reset_state", "clear_cache"):
        method = getattr(backend, method_name, None)
        if callable(method):
            method()
            return method_name
    return None


def _run_generation(
    config: BenchmarkConfig,
    main: MainBackend,
    draft: DraftBackend | None,
    prompt_token_ids: Sequence[int],
    eos_token_ids: Iterable[int],
) -> GenerationResult:
    if config.mode == "ordinary":
        return ordinary_generate(
            main,
            prompt_token_ids,
            max_new_tokens=config.max_new_tokens,
            eos_token_ids=eos_token_ids,
        )
    if draft is None:
        raise ValueError("MTP benchmark mode requires a draft backend")
    return speculative_generate(
        main,
        draft,
        prompt_token_ids,
        max_new_tokens=config.max_new_tokens,
        max_draft_tokens=config.max_draft_tokens,
        eos_token_ids=eos_token_ids,
    )


def run_benchmark(
    main: MainBackend,
    prompt_token_ids: Sequence[int],
    *,
    config: BenchmarkConfig,
    draft: DraftBackend | None = None,
    eos_token_ids: Iterable[int] = (),
    synchronize: Synchronize | None = None,
    synchronization_source: str = "caller-unspecified",
    range_factory: RangeFactory | None = None,
) -> dict:
    """Run warmups and synchronized target measurements.

    Backends may expose one of ``reset_benchmark_state()``, ``reset_state()``,
    or ``clear_cache()``.  The first available hook is invoked before every
    warmup and measured iteration.  A target caller must provide a real device
    synchronization callback; CPU tests may omit it.
    """

    config.validate()
    prompt = [int(token) for token in prompt_token_ids]
    if not prompt:
        raise ValueError("benchmark prompt_token_ids must not be empty")
    sync = synchronize or (lambda: None)
    make_range = range_factory or (lambda _label: nullcontext())
    eos = tuple(int(token) for token in eos_token_ids)
    reset_methods: dict[str, str | None] = {
        "main": None,
        "draft": None,
    }

    def prepare_iteration() -> None:
        reset_methods["main"] = _reset_backend(main)
        if draft is not main:
            reset_methods["draft"] = _reset_backend(draft)
        sync()

    for index in range(config.warmup):
        prepare_iteration()
        with make_range(f"qwen35/{config.mode}/warmup/{index}"):
            _run_generation(config, main, draft, prompt, eos)
            sync()

    measurements = []
    expected_tokens: list[int] | None = None
    elapsed_values = []
    generated_token_count: int | None = None
    for index in range(config.repetitions):
        prepare_iteration()
        with make_range(f"qwen35/{config.mode}/measure/{index}"):
            started_ns = time.perf_counter_ns()
            result = _run_generation(config, main, draft, prompt, eos)
            sync()
            elapsed_seconds = (time.perf_counter_ns() - started_ns) / 1_000_000_000
        tokens = list(result.generated_token_ids)
        if expected_tokens is None:
            expected_tokens = tokens
            generated_token_count = len(tokens)
        elif tokens != expected_tokens:
            raise RuntimeError(
                "benchmark repetitions produced different token IDs: "
                f"expected={expected_tokens}, iteration={index}, actual={tokens}"
            )
        elapsed_values.append(elapsed_seconds)
        measurements.append(
            {
                "iteration": index,
                "elapsed_seconds": elapsed_seconds,
                "generated_token_ids": tokens,
                "generated_tokens": len(tokens),
                "output_tokens_per_second": (
                    len(tokens) / elapsed_seconds if elapsed_seconds > 0 else None
                ),
                "reached_eos": result.reached_eos,
                "stop_reason": result.stop_reason,
                "generation_stats": result.stats.to_dict(),
            }
        )

    assert expected_tokens is not None
    assert generated_token_count is not None
    total_seconds = sum(elapsed_values)
    summary = {
        "count": len(elapsed_values),
        "latency_ms": {
            "min": min(elapsed_values) * 1000,
            "max": max(elapsed_values) * 1000,
            "mean": statistics.fmean(elapsed_values) * 1000,
            "median": statistics.median(elapsed_values) * 1000,
            "p90": _percentile(elapsed_values, 0.90) * 1000,
            "population_stdev": statistics.pstdev(elapsed_values) * 1000,
        },
        "aggregate_output_tokens_per_second": (
            (generated_token_count * len(elapsed_values)) / total_seconds
            if total_seconds > 0
            else None
        ),
        "generated_tokens_per_iteration": generated_token_count,
        "stable_generated_token_ids": expected_tokens,
    }
    return {
        "schema_version": 1,
        "status": "PASS",
        "mode": config.mode,
        "configuration": {
            "warmup": config.warmup,
            "repetitions": config.repetitions,
            "max_new_tokens": config.max_new_tokens,
            "max_draft_tokens": config.max_draft_tokens,
            "prompt_token_ids": prompt,
            "eos_token_ids": list(eos),
        },
        "backends": {
            "main": main.backend_id,
            "draft": None if draft is None else draft.backend_id,
        },
        "synchronization": {
            "source": synchronization_source,
            "applied_before_and_after_each_iteration": True,
        },
        "state_reset_hooks": reset_methods,
        "measurements": measurements,
        "summary": summary,
        "timing_scope": "complete generation call after model load",
        "stage_timing_note": (
            "generation_stats main_seconds/draft_seconds are host observations; "
            "use msprof task/timeline data for device-stage attribution"
        ),
    }
