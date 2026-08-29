#!/usr/bin/env python3
"""Compare C++ AscendCL evidence with a same-scope closed-runtime baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
from typing import Any, Mapping, Sequence


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: Sequence[float], fraction: float) -> float:
    ordered = sorted(float(item) for item in values)
    if not ordered or any(not math.isfinite(item) or item <= 0 for item in ordered):
        raise ValueError("latency raw values must be finite and positive")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * fraction
    low = math.floor(position)
    high = math.ceil(position)
    weight = position - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def summary(values: Sequence[float]) -> dict[str, float | int]:
    clean = [float(item) for item in values]
    return {
        "count": len(clean),
        "min": min(clean),
        "max": max(clean),
        "mean": statistics.fmean(clean),
        "median": statistics.median(clean),
        "p90": percentile(clean, 0.9),
        "population_stdev": statistics.pstdev(clean),
    }


def resolve_run_record(run: Path, record: Mapping[str, Any]) -> Path:
    path = Path(str(record["path"]))
    path = path if path.is_absolute() else run / path
    resolved = path.resolve()
    if resolved != run and run not in resolved.parents:
        raise ValueError(f"evidence path escapes AI_RUN_DIR: {path}")
    if not resolved.is_file() or sha256_file(resolved) != record.get("sha256"):
        raise ValueError(f"evidence record hash mismatch: {resolved}")
    return resolved


def cpp_model_identity(run: Path, report: Mapping[str, Any]) -> dict[str, Any]:
    control = report.get("control_plane")
    if not isinstance(control, Mapping):
        raise ValueError("C++ report has no control_plane")
    record = control.get("air_manifest")
    if not isinstance(record, Mapping):
        raise ValueError("C++ report has no AIR manifest evidence")
    manifest = load_object(resolve_run_record(run, record))
    graphs = manifest.get("graphs")
    if not isinstance(graphs, list) or len(graphs) != 1:
        raise ValueError("C++ AIR manifest must contain one integrated graph")
    metadata = graphs[0].get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("C++ AIR graph has no model metadata")
    abi = report.get("abi", {})
    return {
        "target_checkpoint_manifest_sha256": metadata.get(
            "target_checkpoint_manifest_sha256"
        ),
        "draft_checkpoint_manifest_sha256": metadata.get(
            "draft_checkpoint_manifest_sha256"
        ),
        "dtype": metadata.get("dtype"),
        "sequence_length": abi.get("sequence_length"),
    }


def cpp_raw_model_total(report: Mapping[str, Any]) -> list[float]:
    candidate = report.get("dflash")
    if not isinstance(candidate, Mapping):
        raise ValueError("C++ report has no DFlash measurements")
    measurements = candidate.get("measurements")
    if not isinstance(measurements, list) or len(measurements) != 10:
        raise ValueError("C++ report must retain exactly ten DFlash measurements")
    return [float(item["latency_ms"]["model_total"]) for item in measurements]


def closed_raw_model_total(report: Mapping[str, Any]) -> list[float]:
    protocol = report.get("measurement_protocol", {})
    if int(protocol.get("warmup", -1)) < 3:
        raise ValueError("closed baseline needs at least three warmups")
    repetitions = int(protocol.get("repetitions", -1))
    if repetitions < 10:
        raise ValueError("closed baseline needs at least ten measurements")
    if int(protocol.get("concurrency", -1)) != 1:
        raise ValueError("closed baseline concurrency must be one")
    if protocol.get("model_load_excluded") is not True:
        raise ValueError("closed baseline must exclude model load")
    raw = report.get("latency_ms", {}).get("model_total", {}).get("raw")
    if not isinstance(raw, list) or len(raw) != repetitions:
        raise ValueError("closed baseline raw model_total count differs")
    return [float(item) for item in raw]


def compare(
    cpp: Mapping[str, Any],
    closed: Mapping[str, Any],
    *,
    run: Path,
    max_median_ratio: float,
    max_p90_ratio: float,
) -> dict[str, Any]:
    if max_median_ratio <= 0 or max_p90_ratio <= 0:
        raise ValueError("latency ratio thresholds must be positive")
    if cpp.get("status") != "PASS" or cpp.get("report_kind") != "cpp-ascendcl-paired-target":
        raise ValueError("C++ input is not a passing paired target report")
    if closed.get("status") != "PASS":
        raise ValueError("closed-runtime baseline is not passing")
    backend = cpp.get("backend_metadata")
    if not isinstance(backend, Mapping) or backend.get("cpu_fallback") is not False:
        raise ValueError("C++ report is not physical-target evidence")
    mismatches: list[str] = []

    comparisons = (
        ("device", backend.get("device"), closed.get("device")),
        ("prompt", cpp.get("prompt"), closed.get("prompt")),
        ("chat", cpp.get("chat"), closed.get("chat")),
        ("prompt_token_ids", cpp.get("prompt_token_ids"), closed.get("prompt_token_ids")),
        ("output.token_ids", cpp.get("output", {}).get("token_ids"), closed.get("output", {}).get("token_ids")),
        ("output.stop_reason", cpp.get("output", {}).get("stop_reason"), closed.get("output", {}).get("stop_reason")),
        ("model_identity", cpp_model_identity(run, cpp), closed.get("model_identity")),
    )
    for name, actual, expected in comparisons:
        if actual != expected:
            mismatches.append(name)
    closed_runtime = closed.get("runtime_identity")
    if not isinstance(closed_runtime, Mapping):
        raise ValueError("closed baseline has no runtime_identity")
    for name in ("cann", "driver", "firmware"):
        if backend.get(name) != closed_runtime.get(name):
            mismatches.append(f"runtime_identity.{name}")

    cpp_latency = summary(cpp_raw_model_total(cpp))
    closed_latency = summary(closed_raw_model_total(closed))
    median_ratio = float(cpp_latency["median"]) / float(closed_latency["median"])
    p90_ratio = float(cpp_latency["p90"]) / float(closed_latency["p90"])
    performance_failures = []
    if median_ratio > max_median_ratio:
        performance_failures.append("model_total_median")
    if p90_ratio > max_p90_ratio:
        performance_failures.append("model_total_p90")
    status = "PASS" if not mismatches and not performance_failures else "FAIL"
    return {
        "schema_version": 1,
        "status": status,
        "scope": "same-device pretokenized synchronized model-loop comparison",
        "identity_or_accuracy_mismatches": mismatches,
        "performance_failures": performance_failures,
        "thresholds": {
            "max_cpp_over_closed_median_ratio": max_median_ratio,
            "max_cpp_over_closed_p90_ratio": max_p90_ratio,
        },
        "latency_ms": {"cpp_dflash": cpp_latency, "closed": closed_latency},
        "ratios": {
            "cpp_over_closed_model_total_median": median_ratio,
            "cpp_over_closed_model_total_p90": p90_ratio,
        },
        "runtime_names": {
            "cpp": backend.get("runtime"),
            "closed": closed_runtime.get("runtime"),
        },
    }


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    run_value = os.environ.get("AI_RUN_DIR")
    if not run_value:
        raise RuntimeError("comparison output requires AI_RUN_DIR")
    run = Path(run_value).expanduser().resolve()
    output = path.expanduser().resolve()
    if output == run or run not in output.parents:
        raise RuntimeError("comparison output must be below AI_RUN_DIR")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpp-report", type=Path, required=True)
    parser.add_argument("--closed-report", type=Path, required=True)
    parser.add_argument("--max-median-ratio", type=float, required=True)
    parser.add_argument("--max-p90-ratio", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    result = compare(
        load_object(args.cpp_report),
        load_object(args.closed_report),
        run=run,
        max_median_ratio=args.max_median_ratio,
        max_p90_ratio=args.max_p90_ratio,
    )
    atomic_write(args.output, result)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
