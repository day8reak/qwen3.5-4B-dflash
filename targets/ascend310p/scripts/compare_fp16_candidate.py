#!/usr/bin/env python3
"""Compare the approved FP16 candidate with the official BF16 CPU reference."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gc
import json
from pathlib import Path
import time
from typing import Any

import torch
from torch import Tensor, nn

from qwen35_mtp.backends import TransformersMainBackend
from qwen35_mtp.config import Qwen35MTPConfig
from qwen35_mtp.mtp import MTPKVCache, Qwen35MTPDrafter, Qwen35MTPModule
from qwen35_mtp.precision import (
    audit_fp16_conversion,
    fp16_conversion_is_admissible,
    metric_within,
    project_logits_chunked,
    stable_top2,
    tensor_error_metrics,
)
from qwen35_mtp.weights import EMBEDDING_WEIGHT, SafeTensorRepository, sha256_file


@dataclass(frozen=True)
class MainCase:
    hidden_states: Tensor
    embedding_rows: Tensor
    logits: Tensor
    top2: dict[str, Any]
    backend_top1: int
    conversion_audit: dict[str, Any] | None
    timing_seconds: dict[str, float]


def _parse_tokens(value: str) -> list[int]:
    tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(tokens) < 2:
        raise ValueError("the precision case requires at least two committed tokens")
    return tokens


def _load_main_case(
    model_dir: Path,
    input_ids: Tensor,
    *,
    dtype: torch.dtype,
    chunk_size: int,
    audit_conversion: bool,
) -> MainCase:
    started = time.perf_counter()
    backend = TransformersMainBackend.from_pretrained(model_dir, dtype=dtype)
    loaded = time.perf_counter()
    conversion = (
        audit_fp16_conversion(backend.owner.named_parameters())
        if audit_conversion
        else None
    )
    audited = time.perf_counter()
    evaluation = backend.evaluate(input_ids, [-1])
    embedding_rows = (
        backend.embedding(input_ids[:, 1:])
        .detach()
        .cpu()
        .clone()
    )
    hidden_states = evaluation.hidden_states.detach().cpu().clone()
    backend_top1 = int(evaluation.top1_token_ids[0, 0])
    forwarded = time.perf_counter()
    logits = project_logits_chunked(
        hidden_states[:, -1:, :],
        backend.lm_head.weight,
        compute_dtype=dtype,
        chunk_size=chunk_size,
    )
    top2 = stable_top2(logits)
    projected = time.perf_counter()
    if backend_top1 != top2["token_ids"][0]:
        raise AssertionError("backend Top1 disagrees with stable full-vocabulary projection")
    backend.clear_cache()
    del evaluation
    del backend
    gc.collect()
    return MainCase(
        hidden_states=hidden_states,
        embedding_rows=embedding_rows,
        logits=logits,
        top2=top2,
        backend_top1=backend_top1,
        conversion_audit=conversion,
        timing_seconds={
            "load": loaded - started,
            "conversion_audit": audited - loaded,
            "forward": forwarded - audited,
            "full_vocab_projection": projected - forwarded,
            "total": projected - started,
        },
    )


def _load_mtp_core(
    config: Qwen35MTPConfig,
    state: dict[str, Tensor],
    *,
    dtype: torch.dtype,
) -> Qwen35MTPModule:
    embedding = nn.Embedding(
        config.vocab_size,
        config.hidden_size,
        device="meta",
        dtype=dtype,
    )
    drafter = Qwen35MTPDrafter(config, embedding)
    drafter.mtp.to(dtype=dtype)
    drafter.load_official_mtp_state(state)
    return drafter.mtp


@torch.inference_mode()
def _run_mtp_core(
    core: Qwen35MTPModule,
    embedding_rows: Tensor,
    hidden_sources: Tensor,
    position_ids: Tensor,
) -> tuple[Tensor, MTPKVCache]:
    hidden, cache = core(embedding_rows, hidden_sources, position_ids)
    return hidden.detach().cpu(), MTPKVCache(
        key=cache.key.detach().cpu(), value=cache.value.detach().cpu()
    )


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


def _run(args: argparse.Namespace) -> dict[str, Any]:
    thresholds_document = json.loads(args.threshold_spec.read_text(encoding="utf-8"))
    thresholds = thresholds_document["metric_thresholds"]
    tokens = _parse_tokens(args.committed_token_ids)
    input_ids = torch.tensor([tokens], dtype=torch.long)
    torch.set_num_threads(max(1, args.threads))

    experiment_started = time.perf_counter()
    bf16_main = _load_main_case(
        args.model_dir,
        input_ids,
        dtype=torch.bfloat16,
        chunk_size=args.logit_chunk_size,
        audit_conversion=True,
    )
    fp16_main = _load_main_case(
        args.model_dir,
        input_ids,
        dtype=torch.float16,
        chunk_size=args.logit_chunk_size,
        audit_conversion=False,
    )
    main_finished = time.perf_counter()

    config = Qwen35MTPConfig.from_pretrained(args.model_dir)
    repository = SafeTensorRepository(args.model_dir)
    bf16_state = repository.load(
        config.required_tensor_shapes(), dtype=torch.bfloat16
    )
    mtp_conversion_audit = audit_fp16_conversion(bf16_state.items())
    fp16_state = {name: tensor.to(torch.float16) for name, tensor in bf16_state.items()}
    bf16_core = _load_mtp_core(config, bf16_state, dtype=torch.bfloat16)
    fp16_core = _load_mtp_core(config, fp16_state, dtype=torch.float16)
    del bf16_state
    del fp16_state

    positions = torch.arange(1, len(tokens), dtype=torch.long).view(1, -1)
    bf16_hidden, bf16_cache = _run_mtp_core(
        bf16_core,
        bf16_main.embedding_rows,
        bf16_main.hidden_states[:, :-1, :],
        positions,
    )
    isolated_fp16_hidden, isolated_fp16_cache = _run_mtp_core(
        fp16_core,
        bf16_main.embedding_rows.to(torch.float16),
        bf16_main.hidden_states[:, :-1, :].to(torch.float16),
        positions,
    )
    end_to_end_fp16_hidden, end_to_end_fp16_cache = _run_mtp_core(
        fp16_core,
        fp16_main.embedding_rows,
        fp16_main.hidden_states[:, :-1, :],
        positions,
    )
    mtp_forward_finished = time.perf_counter()

    embedding_weight = repository.load(
        [EMBEDDING_WEIGHT], dtype=torch.bfloat16
    )[EMBEDDING_WEIGHT]
    bf16_mtp_logits = project_logits_chunked(
        bf16_hidden[:, -1:, :],
        embedding_weight,
        compute_dtype=torch.bfloat16,
        chunk_size=args.logit_chunk_size,
    )
    isolated_fp16_mtp_logits = project_logits_chunked(
        isolated_fp16_hidden[:, -1:, :],
        embedding_weight,
        compute_dtype=torch.float16,
        chunk_size=args.logit_chunk_size,
    )
    end_to_end_fp16_mtp_logits = project_logits_chunked(
        end_to_end_fp16_hidden[:, -1:, :],
        embedding_weight,
        compute_dtype=torch.float16,
        chunk_size=args.logit_chunk_size,
    )
    del embedding_weight
    del bf16_core
    del fp16_core
    gc.collect()
    projection_finished = time.perf_counter()

    metrics = {
        "ordinary_embedding_rows": tensor_error_metrics(
            bf16_main.embedding_rows, fp16_main.embedding_rows
        ),
        "ordinary_final_hidden": tensor_error_metrics(
            bf16_main.hidden_states, fp16_main.hidden_states
        ),
        "ordinary_full_vocab_logits": tensor_error_metrics(
            bf16_main.logits, fp16_main.logits
        ),
        "mtp_isolated_hidden": tensor_error_metrics(
            bf16_hidden, isolated_fp16_hidden
        ),
        "mtp_isolated_key_cache": tensor_error_metrics(
            bf16_cache.key, isolated_fp16_cache.key
        ),
        "mtp_isolated_value_cache": tensor_error_metrics(
            bf16_cache.value, isolated_fp16_cache.value
        ),
        "mtp_isolated_full_vocab_logits": tensor_error_metrics(
            bf16_mtp_logits, isolated_fp16_mtp_logits
        ),
        "mtp_end_to_end_hidden": tensor_error_metrics(
            bf16_hidden, end_to_end_fp16_hidden
        ),
        "mtp_end_to_end_key_cache": tensor_error_metrics(
            bf16_cache.key, end_to_end_fp16_cache.key
        ),
        "mtp_end_to_end_value_cache": tensor_error_metrics(
            bf16_cache.value, end_to_end_fp16_cache.value
        ),
        "mtp_end_to_end_full_vocab_logits": tensor_error_metrics(
            bf16_mtp_logits, end_to_end_fp16_mtp_logits
        ),
    }
    top2 = {
        "ordinary_bf16": bf16_main.top2,
        "ordinary_fp16": fp16_main.top2,
        "mtp_bf16": stable_top2(bf16_mtp_logits),
        "mtp_isolated_fp16": stable_top2(isolated_fp16_mtp_logits),
        "mtp_end_to_end_fp16": stable_top2(end_to_end_fp16_mtp_logits),
    }

    gates = {
        "ordinary_weight_conversion_finite_no_overflow": {
            "passed": bool(
                bf16_main.conversion_audit
                and fp16_conversion_is_admissible(bf16_main.conversion_audit)
            ),
            "approved_range_loss_reported": True,
        },
        "mtp_weight_conversion_finite_no_overflow": {
            "passed": fp16_conversion_is_admissible(mtp_conversion_audit),
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
    all_passed = all(bool(gate["passed"]) for gate in gates.values())
    artifact = None
    if args.candidate_onnx is not None:
        artifact = {
            "path": str(args.candidate_onnx),
            "sha256": sha256_file(args.candidate_onnx),
        }
    return {
        "schema_version": 1,
        "status": "PASS_CPU_CANDIDATE" if all_passed else "FAIL_CPU_CANDIDATE",
        "disposition": (
            "ELIGIBLE_FOR_ASCEND310P_TESTING_NOT_PROMOTED"
            if all_passed
            else "REJECTED_RETAIN_BF16_REFERENCE"
        ),
        "case_id": args.case_id,
        "committed_token_ids": tokens,
        "source": {
            "model_dir": str(args.model_dir),
            "config_sha256": sha256_file(args.model_dir / "config.json"),
            "index_sha256": sha256_file(
                args.model_dir / "model.safetensors.index.json"
            ),
        },
        "approval_record": ".work/qwen3.5-4b/20260818T105058Z-8154-e71669/out/approvals/ascend310p-fp16-precision-approval.json",
        "threshold_spec": {
            "path": str(args.threshold_spec),
            "sha256": sha256_file(args.threshold_spec),
        },
        "candidate_onnx": artifact,
        "ordinary_weight_conversion_audit": bf16_main.conversion_audit,
        "mtp_weight_conversion_audit": mtp_conversion_audit,
        "metrics": metrics,
        "top2": top2,
        "gates": gates,
        "timing_seconds": {
            "ordinary_bf16": bf16_main.timing_seconds,
            "ordinary_fp16": fp16_main.timing_seconds,
            "mtp_load_and_forward": mtp_forward_finished - main_finished,
            "mtp_full_vocab_projection": projection_finished - mtp_forward_finished,
            "total": projection_finished - experiment_started,
        },
        "claim_boundary": (
            "This is one CPU BF16-versus-FP16 admission case. It neither proves "
            "Ascend 310P operator equivalence nor authorizes candidate promotion."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threshold-spec", type=Path, required=True)
    parser.add_argument("--candidate-onnx", type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--committed-token-ids", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--logit-chunk-size", type=int, default=8192)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.threshold_spec = args.threshold_spec.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if args.candidate_onnx is not None:
        args.candidate_onnx = args.candidate_onnx.expanduser().resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = _run(args)
    except Exception as error:
        report = {
            "schema_version": 1,
            "status": "FAIL_RUNTIME",
            "case_id": args.case_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "claim_boundary": "No FP16 accuracy conclusion is available from this failed run.",
        }
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
        raise
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS_CPU_CANDIDATE":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
