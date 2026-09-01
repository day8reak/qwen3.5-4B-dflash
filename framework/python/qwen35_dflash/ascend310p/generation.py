"""Prompt-to-text generation with synchronized prefill/decode timing."""

from __future__ import annotations

import math
from pathlib import Path
import re
import statistics
import time
from typing import Any, Callable, Mapping, Sequence

from .contracts import DFlashOmBackend, GenerationStep
from .utils import contained_path, load_json_object, resolve_callable, sha256_file


def _percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return float(ordered[low])
    weight = position - low
    return float(ordered[low] * (1.0 - weight) + ordered[high] * weight)


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cannot summarize an empty latency set")
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p90": _percentile(values, 0.9),
        "population_stdev": statistics.pstdev(values),
    }


def _normalize_eos(eos_token_id: Any) -> tuple[int, ...]:
    if eos_token_id is None:
        return ()
    if isinstance(eos_token_id, int):
        return (int(eos_token_id),)
    return tuple(int(item) for item in eos_token_id)


def _extract_input_ids(values: Any) -> list[int]:
    """Normalize tokenizer outputs to the single supported token-id sequence."""

    if isinstance(values, Mapping):
        if "input_ids" not in values:
            raise ValueError("tokenizer mapping output does not contain input_ids")
        values = values["input_ids"]
    elif hasattr(values, "input_ids"):
        values = values.input_ids

    if hasattr(values, "tolist"):
        values = values.tolist()

    if isinstance(values, (list, tuple)) and values:
        if isinstance(values[0], (list, tuple)):
            if len(values) != 1:
                raise ValueError("only batch size 1 is supported")
            values = values[0]

    if values is None:
        raise ValueError("tokenizer returned no input_ids")
    try:
        return [int(token) for token in values]
    except TypeError as error:
        raise ValueError(
            f"tokenizer input_ids must be an iterable of integers, got {type(values).__name__}"
        ) from error


def tokenize_prompt(tokenizer: Any, prompt: str, *, chat: bool) -> list[int]:
    if chat:
        values = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        )
    else:
        values = tokenizer.encode(prompt, add_special_tokens=True)
    result = _extract_input_ids(values)
    if not result:
        raise ValueError("the prompt tokenized to an empty sequence")
    return result


def _validate_target_metadata(
    backend: DFlashOmBackend,
    metadata: Mapping[str, Any],
    *,
    require_target: bool,
) -> None:
    if not getattr(backend, "backend_id", None):
        raise ValueError("OM backend must expose a non-empty backend_id")
    if metadata.get("cpu_fallback") is not False:
        raise RuntimeError("target DFlash inference requires cpu_fallback=false")
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, Mapping) or not artifacts:
        raise ValueError("OM backend metadata must include artifact hashes")
    invalid_hashes = [
        str(name)
        for name, digest in artifacts.items()
        if re.fullmatch(r"[0-9a-f]{64}", str(digest).lower()) is None
    ]
    if invalid_hashes:
        raise ValueError(f"OM backend metadata has invalid SHA-256 values: {invalid_hashes}")
    if require_target:
        device = metadata.get("device")
        if not isinstance(device, Mapping):
            raise ValueError("target backend metadata must include a device object")
        target_id = str(device.get("target_id", "")).lower()
        if target_id != "ascend310p":
            raise RuntimeError(
                f"target inference requires device.target_id=ascend310p, got {target_id!r}"
            )
        device_id = device.get("device_id")
        if not isinstance(device_id, int) or device_id < 0 or not device.get("model"):
            raise ValueError("target device metadata needs concrete model and device_id")
        normalized_model = re.sub(r"[^a-z0-9]", "", str(device["model"]).lower())
        if normalized_model in {"310p", "ascend310p", "atlas310p"}:
            raise ValueError("target device metadata must name the concrete 310P product")
        missing_identity = [
            name
            for name in ("cann", "driver", "firmware", "runtime")
            if not str(metadata.get(name, "")).strip()
        ]
        if missing_identity:
            raise ValueError(
                f"target backend metadata is missing runtime identities: {missing_identity}"
            )


def _append_step(
    generated: list[int],
    step: GenerationStep,
    *,
    remaining: int,
    eos: frozenset[int],
) -> tuple[list[int], bool]:
    if len(step.token_ids) > remaining:
        raise RuntimeError(
            f"backend committed {len(step.token_ids)} tokens with only {remaining} remaining"
        )
    committed: list[int] = []
    finished = bool(step.finished)
    for token in step.token_ids:
        generated.append(token)
        committed.append(token)
        if token in eos:
            finished = True
            break
    if len(committed) != len(step.token_ids):
        raise RuntimeError("backend returned tokens after an EOS token in one commit")
    return committed, finished


def generate_prompt(
    backend: DFlashOmBackend,
    tokenizer: Any,
    prompt: str,
    *,
    chat: bool = False,
    max_new_tokens: int = 32,
    max_draft_tokens: int = 15,
    require_target: bool = True,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> dict[str, Any]:
    """Run one prompt and measure host-observed synchronized stage latency.

    ``prefill_ms`` measures exactly one backend prefill call. ``decode_ms`` is
    the sum of all subsequent DFlash draft/verify calls. ``model_total_ms`` is
    their sum. ``end_to_end_ms`` additionally includes tokenization and output
    decoding, which are reported separately so callers do not mix scopes.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive")
    if max_draft_tokens <= 0:
        raise ValueError("max_draft_tokens must be positive")
    metadata = dict(backend.metadata())
    _validate_target_metadata(backend, metadata, require_target=require_target)

    end_to_end_start = clock_ns()
    tokenize_start = clock_ns()
    prompt_ids = tokenize_prompt(tokenizer, prompt, chat=chat)
    tokenize_end = clock_ns()
    eos_ids = _normalize_eos(getattr(tokenizer, "eos_token_id", None))
    eos = frozenset(eos_ids)

    backend.synchronize()
    prefill_start = clock_ns()
    prefill = backend.prefill(
        prompt_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_ids,
    )
    backend.synchronize()
    prefill_end = clock_ns()

    generated: list[int] = []
    prefill_tokens, finished = _append_step(
        generated,
        prefill,
        remaining=max_new_tokens,
        eos=eos,
    )
    backend_finished = bool(prefill.finished)
    decode_measurements: list[dict[str, Any]] = []
    totals = {
        "drafted_tokens": int(prefill.drafted_tokens),
        "accepted_draft_tokens": int(prefill.accepted_draft_tokens),
        "rejected_draft_tokens": int(prefill.rejected_draft_tokens),
    }
    while not finished and len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        backend.synchronize()
        decode_start = clock_ns()
        step = backend.decode(
            [*prompt_ids, *generated],
            max_new_tokens=remaining,
            max_draft_tokens=min(max_draft_tokens, remaining),
            eos_token_ids=eos_ids,
        )
        backend.synchronize()
        decode_end = clock_ns()
        committed, finished = _append_step(
            generated,
            step,
            remaining=remaining,
            eos=eos,
        )
        backend_finished = bool(step.finished)
        measurement = {
            "iteration": len(decode_measurements),
            "latency_ms": (decode_end - decode_start) / 1_000_000.0,
            "committed_token_ids": committed,
            "drafted_tokens": int(step.drafted_tokens),
            "accepted_draft_tokens": int(step.accepted_draft_tokens),
            "rejected_draft_tokens": int(step.rejected_draft_tokens),
            "backend_metadata": dict(step.metadata),
        }
        decode_measurements.append(measurement)
        for key in totals:
            totals[key] += int(getattr(step, key))

    detokenize_start = clock_ns()
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)
    detokenize_end = clock_ns()
    end_to_end_end = clock_ns()
    prefill_ms = (prefill_end - prefill_start) / 1_000_000.0
    decode_ms = sum(item["latency_ms"] for item in decode_measurements)
    model_total_ms = prefill_ms + decode_ms
    return {
        "schema_version": 1,
        "status": "PASS",
        "backend_id": str(backend.backend_id),
        "backend_metadata": metadata,
        "prompt": prompt,
        "chat": bool(chat),
        "prompt_token_ids": prompt_ids,
        "generated_token_ids": generated,
        "generated_text": generated_text,
        "stop_reason": (
            "eos"
            if generated and generated[-1] in eos
            else "backend"
            if backend_finished
            else "length"
        ),
        "limits": {
            "max_new_tokens": max_new_tokens,
            "max_draft_tokens": max_draft_tokens,
        },
        "counters": {
            **totals,
            "generated_tokens": len(generated),
            "decode_iterations": len(decode_measurements),
        },
        "latency_ms": {
            "tokenize": (tokenize_end - tokenize_start) / 1_000_000.0,
            "prefill": prefill_ms,
            "decode": decode_ms,
            "model_total": model_total_ms,
            "detokenize": (detokenize_end - detokenize_start) / 1_000_000.0,
            "end_to_end": (end_to_end_end - end_to_end_start) / 1_000_000.0,
            "time_to_first_token": (
                (tokenize_end - tokenize_start) + (prefill_end - prefill_start)
            )
            / 1_000_000.0,
        },
        "prefill": {
            "committed_token_ids": prefill_tokens,
            "backend_metadata": dict(prefill.metadata),
        },
        "decode_iterations": decode_measurements,
        "timing_scope": {
            "clock": "time.perf_counter_ns",
            "device_synchronization": "before and after every prefill/decode call",
            "prefill": "one synchronized backend.prefill invocation",
            "decode": "sum of synchronized backend.decode invocations",
            "model_total": "prefill + decode",
            "end_to_end": "tokenization through detokenization",
        },
    }


def benchmark_prompt(
    backend: DFlashOmBackend,
    tokenizer: Any,
    prompt: str,
    *,
    chat: bool = False,
    max_new_tokens: int = 32,
    max_draft_tokens: int = 15,
    warmup: int = 3,
    repetitions: int = 10,
    require_target: bool = True,
) -> dict[str, Any]:
    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive")
    if require_target and (warmup != 3 or repetitions != 10):
        raise ValueError(
            "target performance evidence requires exactly 3 warmups and 10 measurements"
        )
    for _index in range(warmup):
        backend.reset()
        generate_prompt(
            backend,
            tokenizer,
            prompt,
            chat=chat,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens,
            require_target=require_target,
        )

    measurements = []
    for index in range(repetitions):
        backend.reset()
        item = generate_prompt(
            backend,
            tokenizer,
            prompt,
            chat=chat,
            max_new_tokens=max_new_tokens,
            max_draft_tokens=max_draft_tokens,
            require_target=require_target,
        )
        item["repetition"] = index
        measurements.append(item)
    reference_tokens = measurements[0]["generated_token_ids"]
    if any(item["generated_token_ids"] != reference_tokens for item in measurements[1:]):
        raise RuntimeError("measured repetitions produced different token IDs")
    reference_stop_reason = measurements[0]["stop_reason"]
    if any(item["stop_reason"] != reference_stop_reason for item in measurements[1:]):
        raise RuntimeError("measured repetitions produced different stop reasons")
    reference_text = measurements[0]["generated_text"]
    if any(item["generated_text"] != reference_text for item in measurements[1:]):
        raise RuntimeError("measured repetitions produced different generated text")
    latency_names = (
        "tokenize",
        "prefill",
        "decode",
        "model_total",
        "detokenize",
        "end_to_end",
        "time_to_first_token",
    )
    return {
        "schema_version": 1,
        "status": "PASS",
        "warmup": warmup,
        "repetitions": repetitions,
        "stable_generated_token_ids": reference_tokens,
        "stable_generated_text": reference_text,
        "stable_prompt_token_ids": measurements[0]["prompt_token_ids"],
        "stable_stop_reason": reference_stop_reason,
        "prompt": prompt,
        "chat": bool(chat),
        "limits": {
            "max_new_tokens": max_new_tokens,
            "max_draft_tokens": max_draft_tokens,
        },
        "backend_id": str(backend.backend_id),
        "backend_metadata": dict(backend.metadata()),
        "latency_ms": {
            name: _summary([item["latency_ms"][name] for item in measurements])
            for name in latency_names
        },
        "measurements": measurements,
    }


def verify_ordinary_reference(
    candidate: Mapping[str, Any],
    reference: Mapping[str, Any],
    *,
    reference_path: str | Path | None = None,
) -> dict[str, Any]:
    """Require exact DFlash token/EOS parity with an ordinary target report."""

    if reference.get("status") != "PASS":
        raise ValueError("ordinary reference report is not passing")
    reference_metadata = reference.get("backend_metadata")
    if not isinstance(reference_metadata, Mapping):
        raise ValueError("ordinary reference has no backend metadata")
    if reference_metadata.get("generation_mode") != "ordinary-greedy":
        raise ValueError("ordinary reference was not produced in ordinary-greedy mode")
    candidate_metadata = candidate.get("backend_metadata")
    if not isinstance(candidate_metadata, Mapping):
        raise ValueError("DFlash candidate has no backend metadata")
    if candidate_metadata.get("generation_mode") != "dflash-strict-greedy":
        raise ValueError("candidate was not produced in DFlash strict-greedy mode")
    if reference_metadata.get("cpu_fallback") is not False:
        raise ValueError("ordinary reference reports CPU fallback")
    if candidate_metadata.get("cpu_fallback") is not False:
        raise ValueError("DFlash candidate reports CPU fallback")
    reference_device = reference_metadata.get("device")
    if not isinstance(reference_device, Mapping) or str(
        reference_device.get("target_id", "")
    ).lower() != "ascend310p":
        raise ValueError("ordinary reference is not an Ascend 310P target report")
    if reference_device != candidate_metadata.get("device"):
        raise ValueError("ordinary and DFlash reports use different target devices")
    for name in ("cann", "driver", "firmware", "runtime"):
        if reference_metadata.get(name) != candidate_metadata.get(name):
            raise ValueError(
                f"ordinary and DFlash reports differ in runtime identity {name}"
            )
    if reference_metadata.get("artifacts") != candidate_metadata.get("artifacts"):
        raise ValueError("ordinary and DFlash reports use different OM artifacts")
    for name in ("prompt", "chat"):
        if reference.get(name) != candidate.get(name):
            raise ValueError(f"ordinary and DFlash reports differ in {name}")
    if reference.get("tokenizer_source") != candidate.get("tokenizer_source"):
        raise ValueError("ordinary and DFlash reports use different tokenizer sources")
    if reference.get("stable_prompt_token_ids") != candidate.get(
        "stable_prompt_token_ids"
    ):
        raise ValueError("ordinary and DFlash reports tokenized the prompt differently")
    reference_limits = reference.get("limits", {})
    candidate_limits = candidate.get("limits", {})
    if reference_limits.get("max_new_tokens") != candidate_limits.get(
        "max_new_tokens"
    ):
        raise ValueError("ordinary and DFlash reports use different max_new_tokens")
    for report_name, report in (("ordinary", reference), ("DFlash", candidate)):
        if report.get("warmup") != 3 or report.get("repetitions") != 10:
            raise ValueError(
                f"{report_name} target report must contain exactly 3 warmups and 10 measurements"
            )
    expected = [int(item) for item in reference.get("stable_generated_token_ids", [])]
    actual = [int(item) for item in candidate.get("stable_generated_token_ids", [])]
    if not expected:
        raise ValueError("ordinary reference contains no generated token IDs")
    mismatch_positions = [
        index
        for index in range(max(len(expected), len(actual)))
        if index >= len(expected)
        or index >= len(actual)
        or expected[index] != actual[index]
    ]
    expected_stop = reference.get("stable_stop_reason")
    actual_stop = candidate.get("stable_stop_reason")
    if mismatch_positions or expected_stop != actual_stop:
        raise RuntimeError(
            "DFlash output differs from ordinary greedy authority: "
            f"token_mismatch_positions={mismatch_positions}, "
            f"ordinary_stop={expected_stop!r}, dflash_stop={actual_stop!r}"
        )
    source = None
    if reference_path is not None:
        source_path = Path(reference_path).expanduser().resolve()
        source = {
            "path": str(source_path),
            "sha256": sha256_file(source_path),
        }
    return {
        "status": "PASS",
        "ordinary_reference": source,
        "token_id_mismatches": 0,
        "eos_mismatches": 0,
        "compared_tokens": len(actual),
    }


def load_backend(
    factory_reference: str,
    deployment_manifest_path: str | Path,
    *,
    device_id: int,
    options: Mapping[str, Any],
) -> DFlashOmBackend:
    manifest_path = Path(deployment_manifest_path).expanduser().resolve()
    manifest = load_json_object(manifest_path)
    if manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
        raise ValueError("backend requires a Qwen3.5 DFlash OM deployment manifest")
    if manifest.get("status") != "PASS":
        raise ValueError("deployment manifest is not passing")
    root = manifest_path.parent
    for graph in manifest.get("graphs", []):
        om = graph["om"]
        path = contained_path(root, str(om["path"]))
        if not path.is_file() or sha256_file(path) != om["sha256"]:
            raise ValueError(f"OM artifact integrity check failed: {graph['name']}")
    factory = resolve_callable(factory_reference)
    backend = factory(
        bundle_dir=root,
        manifest=manifest,
        device_id=int(device_id),
        options=dict(options),
    )
    required = ("metadata", "synchronize", "reset", "prefill", "decode", "close")
    missing = [name for name in required if not callable(getattr(backend, name, None))]
    if missing:
        raise TypeError(f"OM backend is missing methods: {missing}")
    if not getattr(backend, "backend_id", None):
        raise TypeError("OM backend must expose backend_id")
    expected_artifacts = {
        str(graph["name"]): str(graph["om"]["sha256"])
        for graph in manifest.get("graphs", [])
    }
    reported_metadata = dict(backend.metadata())
    reported_artifacts = reported_metadata.get("artifacts")
    if reported_artifacts != expected_artifacts:
        backend.close()
        raise ValueError(
            "OM backend artifact identities differ from the deployment manifest: "
            f"expected={expected_artifacts}, reported={reported_artifacts}"
        )
    return backend
