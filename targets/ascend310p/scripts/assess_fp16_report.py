#!/usr/bin/env python3
"""Re-evaluate a retained raw FP16 report against the frozen approved gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from qwen35_mtp.precision import fp16_conversion_is_admissible, metric_within
from qwen35_mtp.weights import sha256_file


def _metric_gate(
    metrics: dict[str, Any], threshold: dict[str, float]
) -> dict[str, Any]:
    return {
        "passed": metric_within(
            metrics,
            max_relative_l2=threshold["max_relative_l2"],
            min_cosine=threshold["min_cosine"],
        ),
        "threshold": threshold,
    }


def _conversion_summary(report: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "tensor_count",
        "element_count",
        "source_non_finite_count",
        "candidate_non_finite_count",
        "overflow_count",
        "underflow_to_zero_count",
        "roundtrip_mismatch_count",
        "source_abs_max",
    )
    return {key: report[key] for key in keys}


def assess(raw: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    metrics = raw["metrics"]
    top2 = raw["top2"]
    thresholds = spec["metric_thresholds"]
    ordinary_conversion = raw["ordinary_weight_conversion_audit"]
    mtp_conversion = raw["mtp_weight_conversion_audit"]
    gates = {
        "ordinary_weight_conversion_finite_no_overflow": {
            "passed": fp16_conversion_is_admissible(ordinary_conversion),
            "approved_range_loss_reported": True,
        },
        "mtp_weight_conversion_finite_no_overflow": {
            "passed": fp16_conversion_is_admissible(mtp_conversion),
            "approved_range_loss_reported": True,
        },
        "ordinary_hidden_metric": _metric_gate(
            metrics["ordinary_final_hidden"], thresholds["ordinary_final_hidden"]
        ),
        "ordinary_logits_metric": _metric_gate(
            metrics["ordinary_full_vocab_logits"],
            thresholds["ordinary_full_vocab_logits"],
        ),
        "ordinary_top1_exact": {
            "passed": top2["ordinary_bf16"]["token_ids"][0]
            == top2["ordinary_fp16"]["token_ids"][0]
        },
        "mtp_isolated_hidden_metric": _metric_gate(
            metrics["mtp_isolated_hidden"], thresholds["mtp_hidden"]
        ),
        "mtp_isolated_key_cache_metric": _metric_gate(
            metrics["mtp_isolated_key_cache"], thresholds["mtp_cache"]
        ),
        "mtp_isolated_value_cache_metric": _metric_gate(
            metrics["mtp_isolated_value_cache"], thresholds["mtp_cache"]
        ),
        "mtp_isolated_logits_metric": _metric_gate(
            metrics["mtp_isolated_full_vocab_logits"],
            thresholds["mtp_full_vocab_logits"],
        ),
        "mtp_isolated_top1_exact": {
            "passed": top2["mtp_bf16"]["token_ids"][0]
            == top2["mtp_isolated_fp16"]["token_ids"][0]
        },
        "mtp_end_to_end_hidden_metric": _metric_gate(
            metrics["mtp_end_to_end_hidden"], thresholds["mtp_hidden"]
        ),
        "mtp_end_to_end_key_cache_metric": _metric_gate(
            metrics["mtp_end_to_end_key_cache"], thresholds["mtp_cache"]
        ),
        "mtp_end_to_end_value_cache_metric": _metric_gate(
            metrics["mtp_end_to_end_value_cache"], thresholds["mtp_cache"]
        ),
        "mtp_end_to_end_logits_metric": _metric_gate(
            metrics["mtp_end_to_end_full_vocab_logits"],
            thresholds["mtp_full_vocab_logits"],
        ),
        "mtp_end_to_end_top1_exact": {
            "passed": top2["mtp_bf16"]["token_ids"][0]
            == top2["mtp_end_to_end_fp16"]["token_ids"][0]
        },
    }
    passed = all(bool(item["passed"]) for item in gates.values())
    return {
        "schema_version": 1,
        "status": "PASS_CPU_CANDIDATE" if passed else "FAIL_CPU_CANDIDATE",
        "disposition": (
            "ELIGIBLE_FOR_ASCEND310P_TESTING_NOT_PROMOTED"
            if passed
            else "REJECTED_RETAIN_BF16_REFERENCE"
        ),
        "case_id": raw["case_id"],
        "committed_token_ids": raw["committed_token_ids"],
        "source": raw["source"],
        "candidate_onnx": raw.get("candidate_onnx"),
        "correction": {
            "raw_status": raw["status"],
            "reason": (
                "The v1 evaluator added an exact FP16 weight-roundtrip gate that "
                "was absent from the approved proposal and frozen spec. Approved "
                "finite underflow remains reported and is governed by the frozen "
                "hidden/logit/token gates."
            ),
            "thresholds_changed": False,
            "raw_measurements_changed": False,
        },
        "ordinary_weight_conversion": _conversion_summary(ordinary_conversion),
        "mtp_weight_conversion": _conversion_summary(mtp_conversion),
        "metrics": metrics,
        "top2": top2,
        "gates": gates,
        "timing_seconds": raw["timing_seconds"],
        "claim_boundary": raw["claim_boundary"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-report", type=Path, required=True)
    parser.add_argument("--threshold-spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.raw_report.expanduser().resolve()
    spec_path = args.threshold_spec.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    result = assess(raw, spec)
    result["raw_report"] = {"path": str(raw_path), "sha256": sha256_file(raw_path)}
    result["threshold_spec"] = {
        "path": str(spec_path),
        "sha256": sha256_file(spec_path),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS_CPU_CANDIDATE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
