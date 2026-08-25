"""Numerical helpers for the approval-gated BF16-to-FP16 experiment."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor
import torch.nn.functional as F


def tensor_error_metrics(reference: Tensor, candidate: Tensor) -> dict[str, Any]:
    """Return JSON-safe finite/error metrics without hiding shape mismatches."""

    if reference.shape != candidate.shape:
        raise ValueError(
            f"metric tensors must have the same shape: {reference.shape} != {candidate.shape}"
        )
    if reference.numel() == 0:
        raise ValueError("metric tensors must not be empty")
    reference_float = reference.detach().float().cpu()
    candidate_float = candidate.detach().float().cpu()
    reference_finite = bool(torch.isfinite(reference_float).all())
    candidate_finite = bool(torch.isfinite(candidate_float).all())
    result: dict[str, Any] = {
        "shape": list(reference.shape),
        "reference_dtype": str(reference.dtype).removeprefix("torch."),
        "candidate_dtype": str(candidate.dtype).removeprefix("torch."),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
    }
    if not reference_finite or not candidate_finite:
        result.update(
            {
                "reference_abs_max": None,
                "candidate_abs_max": None,
                "max_abs": None,
                "mean_abs": None,
                "rmse": None,
                "relative_l2": None,
                "cosine": None,
            }
        )
        return result

    difference = candidate_float - reference_float
    reference_l2 = torch.linalg.vector_norm(reference_float)
    candidate_l2 = torch.linalg.vector_norm(candidate_float)
    difference_l2 = torch.linalg.vector_norm(difference)
    denominator = max(float(reference_l2), torch.finfo(torch.float32).tiny)
    if float(reference_l2) == 0.0 and float(candidate_l2) == 0.0:
        cosine = 1.0
    elif float(reference_l2) == 0.0 or float(candidate_l2) == 0.0:
        cosine = 0.0
    else:
        cosine = float(
            torch.dot(reference_float.reshape(-1), candidate_float.reshape(-1))
            / (reference_l2 * candidate_l2)
        )
    result.update(
        {
            "reference_abs_max": float(reference_float.abs().max()),
            "candidate_abs_max": float(candidate_float.abs().max()),
            "max_abs": float(difference.abs().max()),
            "mean_abs": float(difference.abs().mean()),
            "rmse": float(torch.sqrt(torch.mean(difference.square()))),
            "relative_l2": float(difference_l2) / denominator,
            "cosine": cosine,
        }
    )
    return result


@torch.inference_mode()
def project_logits_chunked(
    hidden: Tensor,
    weight: Tensor,
    *,
    compute_dtype: torch.dtype,
    chunk_size: int = 8192,
) -> Tensor:
    """Project one hidden row over the full vocabulary with bounded memory."""

    if hidden.numel() != hidden.shape[-1]:
        raise ValueError("chunked projection requires exactly one hidden row")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[-1]:
        raise ValueError("LM-head weight is incompatible with the hidden row")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    row = hidden.reshape(1, -1).to(dtype=compute_dtype)
    chunks: list[Tensor] = []
    for start in range(0, weight.shape[0], chunk_size):
        stop = min(start + chunk_size, weight.shape[0])
        logits = F.linear(row, weight[start:stop].to(dtype=compute_dtype))
        chunks.append(logits.reshape(-1).float().cpu())
    return torch.cat(chunks)


def stable_top2(logits: Tensor) -> dict[str, Any]:
    """Return Top1/Top2 with lowest-token-ID tie breaking."""

    scores = logits.detach().float().cpu().reshape(-1)
    if scores.numel() < 2:
        raise ValueError("Top2 requires at least two logits")
    if not torch.isfinite(scores).all():
        raise FloatingPointError("non-finite value in full-vocabulary logits")
    order = torch.argsort(scores, descending=True, stable=True)[:2]
    top_scores = scores[order]
    return {
        "token_ids": [int(order[0]), int(order[1])],
        "scores": [float(top_scores[0]), float(top_scores[1])],
        "margin": float(top_scores[0] - top_scores[1]),
    }


def metric_within(
    metrics: dict[str, Any],
    *,
    max_relative_l2: float,
    min_cosine: float,
) -> bool:
    """Apply a frozen metric gate; exact token gates are evaluated separately."""

    relative_l2 = metrics.get("relative_l2")
    cosine = metrics.get("cosine")
    return bool(
        metrics.get("reference_finite")
        and metrics.get("candidate_finite")
        and relative_l2 is not None
        and cosine is not None
        and relative_l2 <= max_relative_l2
        and cosine >= min_cosine
    )


@torch.inference_mode()
def audit_fp16_conversion(
    named_tensors: Iterable[tuple[str, Tensor]],
    *,
    chunk_elements: int = 4 * 1024 * 1024,
) -> dict[str, Any]:
    """Audit explicit FP16 conversion without retaining a second model copy."""

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    totals = {
        "tensor_count": 0,
        "element_count": 0,
        "source_non_finite_count": 0,
        "candidate_non_finite_count": 0,
        "overflow_count": 0,
        "underflow_to_zero_count": 0,
        "roundtrip_mismatch_count": 0,
    }
    source_abs_max = 0.0
    affected_tensors: list[dict[str, Any]] = []
    fp16_max = torch.finfo(torch.float16).max
    for name, tensor in named_tensors:
        if not tensor.is_floating_point():
            continue
        totals["tensor_count"] += 1
        totals["element_count"] += tensor.numel()
        per_tensor = {
            "name": name,
            "source_non_finite_count": 0,
            "candidate_non_finite_count": 0,
            "overflow_count": 0,
            "underflow_to_zero_count": 0,
            "roundtrip_mismatch_count": 0,
        }
        flat = tensor.detach().reshape(-1)
        for start in range(0, flat.numel(), chunk_elements):
            source = flat[start : start + chunk_elements]
            candidate = source.to(dtype=torch.float16)
            source_finite = torch.isfinite(source)
            candidate_finite = torch.isfinite(candidate)
            source_abs = source.abs()
            if source.numel():
                finite_abs = source_abs[source_finite]
                if finite_abs.numel():
                    source_abs_max = max(source_abs_max, float(finite_abs.max()))
            counts = {
                "source_non_finite_count": int((~source_finite).sum()),
                "candidate_non_finite_count": int((~candidate_finite).sum()),
                "overflow_count": int((source_finite & (source_abs > fp16_max)).sum()),
                "underflow_to_zero_count": int(
                    (source_finite & (source != 0) & (candidate == 0)).sum()
                ),
                "roundtrip_mismatch_count": int(
                    (source_finite & (candidate.to(dtype=source.dtype) != source)).sum()
                ),
            }
            for key, value in counts.items():
                totals[key] += value
                per_tensor[key] += value
        if any(per_tensor[key] for key in per_tensor if key != "name"):
            affected_tensors.append(per_tensor)
    affected_tensor_count = len(affected_tensors)
    return {
        **totals,
        "source_abs_max": source_abs_max,
        "affected_tensor_count": affected_tensor_count,
        "affected_tensors": affected_tensors[:64],
        "affected_tensors_truncated": affected_tensor_count > 64,
        "exact_roundtrip": totals["roundtrip_mismatch_count"] == 0,
        "finite_no_overflow": (
            totals["source_non_finite_count"] == 0
            and totals["candidate_non_finite_count"] == 0
            and totals["overflow_count"] == 0
        ),
        "safe_for_fp16_range": (
            totals["source_non_finite_count"] == 0
            and totals["candidate_non_finite_count"] == 0
            and totals["overflow_count"] == 0
            and totals["underflow_to_zero_count"] == 0
        ),
    }


def fp16_conversion_is_admissible(report: dict[str, Any]) -> bool:
    """Gate only finite/overflow failures; approved underflow remains reported."""

    return bool(
        report.get("source_non_finite_count") == 0
        and report.get("candidate_non_finite_count") == 0
        and report.get("overflow_count") == 0
    )
