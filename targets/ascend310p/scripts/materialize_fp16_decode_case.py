#!/usr/bin/env python3
"""Materialize a real-text S1/P1 FP16 MTP core case for internal 310P runs."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn

from qwen35_mtp.backends import TransformersMainBackend
from qwen35_mtp.config import Qwen35MTPConfig
from qwen35_mtp.mtp import Qwen35MTPDrafter, Qwen35MTPModule
from qwen35_mtp.precision import (
    metric_within,
    project_logits_chunked,
    stable_top2,
    tensor_error_metrics,
)
from qwen35_mtp.weights import SafeTensorRepository, sha256_file


def _parse_tokens(value: str) -> list[int]:
    tokens = [int(item.strip()) for item in value.split(",") if item.strip()]
    if len(tokens) != 2:
        raise ValueError("the S1/P1 decode fixture requires exactly two prefix tokens")
    return tokens


def _load_core(
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


def _save_array(directory: Path, name: str, tensor: Tensor) -> dict[str, Any]:
    array = tensor.detach().cpu().contiguous().numpy()
    npy_path = directory / f"{name}.npy"
    bin_path = directory / f"{name}.bin"
    np.save(npy_path, array, allow_pickle=False)
    array.tofile(bin_path)
    return {
        "name": name,
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "npy": {"file": npy_path.name, "sha256": sha256_file(npy_path)},
        "bin": {"file": bin_path.name, "sha256": sha256_file(bin_path)},
    }


def _gate(metrics: dict[str, Any], threshold: dict[str, float]) -> bool:
    return metric_within(
        metrics,
        max_relative_l2=threshold["max_relative_l2"],
        min_cosine=threshold["min_cosine"],
    )


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    tokens = _parse_tokens(args.prefix_token_ids)
    thresholds_document = json.loads(args.threshold_spec.read_text(encoding="utf-8"))
    thresholds = thresholds_document["metric_thresholds"]
    torch.set_num_threads(max(1, args.threads))
    started = time.perf_counter()

    main = TransformersMainBackend.from_pretrained(
        args.model_dir, dtype=torch.bfloat16
    )
    loaded = time.perf_counter()
    prefix = torch.tensor([tokens], dtype=torch.long)
    evaluation = main.evaluate(prefix, [])
    main_hidden = evaluation.hidden_states.detach().cpu().clone()
    embedding_weight = main.embedding.weight.detach()
    shifted_embedding = main.embedding(prefix[:, 1:]).detach().cpu().clone()
    forwarded = time.perf_counter()
    main.clear_cache()
    del evaluation
    del main
    gc.collect()

    config = Qwen35MTPConfig.from_pretrained(args.model_dir)
    repository = SafeTensorRepository(args.model_dir)
    bf16_state = repository.load(
        config.required_tensor_shapes(), dtype=torch.bfloat16
    )
    fp16_state = {name: tensor.to(torch.float16) for name, tensor in bf16_state.items()}
    bf16_core = _load_core(config, bf16_state, dtype=torch.bfloat16)
    fp16_core = _load_core(config, fp16_state, dtype=torch.float16)
    del bf16_state
    del fp16_state

    prefill_position = torch.tensor([[1]], dtype=torch.long)
    bf16_prefill_hidden, bf16_prefill_cache = bf16_core(
        shifted_embedding,
        main_hidden[:, :1, :],
        prefill_position,
    )
    fp16_prefill_hidden, fp16_prefill_cache = fp16_core(
        shifted_embedding.to(torch.float16),
        main_hidden[:, :1, :].to(torch.float16),
        prefill_position,
    )
    bf16_first_logits = project_logits_chunked(
        bf16_prefill_hidden,
        embedding_weight,
        compute_dtype=torch.bfloat16,
        chunk_size=args.logit_chunk_size,
    )
    fp16_first_logits = project_logits_chunked(
        fp16_prefill_hidden,
        embedding_weight,
        compute_dtype=torch.float16,
        chunk_size=args.logit_chunk_size,
    )
    bf16_first_top2 = stable_top2(bf16_first_logits)
    fp16_first_top2 = stable_top2(fp16_first_logits)
    first_draft = bf16_first_top2["token_ids"][0]

    decode_position = torch.tensor([[2]], dtype=torch.long)
    bf16_decode_embedding = embedding_weight[first_draft].reshape(1, 1, -1)
    fp16_decode_embedding = bf16_decode_embedding.to(torch.float16)
    bf16_decode_hidden, bf16_decode_cache = bf16_core(
        bf16_decode_embedding,
        bf16_prefill_hidden,
        decode_position,
        past_key_values=bf16_prefill_cache,
    )
    fp16_decode_hidden, fp16_decode_cache = fp16_core(
        fp16_decode_embedding,
        fp16_prefill_hidden,
        decode_position,
        past_key_values=fp16_prefill_cache,
    )
    bf16_second_logits = project_logits_chunked(
        bf16_decode_hidden,
        embedding_weight,
        compute_dtype=torch.bfloat16,
        chunk_size=args.logit_chunk_size,
    )
    fp16_second_logits = project_logits_chunked(
        fp16_decode_hidden,
        embedding_weight,
        compute_dtype=torch.float16,
        chunk_size=args.logit_chunk_size,
    )
    bf16_second_top2 = stable_top2(bf16_second_logits)
    fp16_second_top2 = stable_top2(fp16_second_logits)
    computed = time.perf_counter()

    metrics = {
        "prefill_hidden": tensor_error_metrics(
            bf16_prefill_hidden, fp16_prefill_hidden
        ),
        "prefill_key_cache": tensor_error_metrics(
            bf16_prefill_cache.key, fp16_prefill_cache.key
        ),
        "prefill_value_cache": tensor_error_metrics(
            bf16_prefill_cache.value, fp16_prefill_cache.value
        ),
        "prefill_logits": tensor_error_metrics(
            bf16_first_logits, fp16_first_logits
        ),
        "decode_hidden": tensor_error_metrics(
            bf16_decode_hidden, fp16_decode_hidden
        ),
        "decode_key_cache": tensor_error_metrics(
            bf16_decode_cache.key, fp16_decode_cache.key
        ),
        "decode_value_cache": tensor_error_metrics(
            bf16_decode_cache.value, fp16_decode_cache.value
        ),
        "decode_logits": tensor_error_metrics(
            bf16_second_logits, fp16_second_logits
        ),
    }
    gates = {
        "prefill_hidden": _gate(metrics["prefill_hidden"], thresholds["mtp_hidden"]),
        "prefill_key_cache": _gate(metrics["prefill_key_cache"], thresholds["mtp_cache"]),
        "prefill_value_cache": _gate(metrics["prefill_value_cache"], thresholds["mtp_cache"]),
        "prefill_logits": _gate(
            metrics["prefill_logits"], thresholds["mtp_full_vocab_logits"]
        ),
        "prefill_top1_exact": bf16_first_top2["token_ids"][0]
        == fp16_first_top2["token_ids"][0],
        "decode_hidden": _gate(metrics["decode_hidden"], thresholds["mtp_hidden"]),
        "decode_key_cache": _gate(metrics["decode_key_cache"], thresholds["mtp_cache"]),
        "decode_value_cache": _gate(metrics["decode_value_cache"], thresholds["mtp_cache"]),
        "decode_logits": _gate(
            metrics["decode_logits"], thresholds["mtp_full_vocab_logits"]
        ),
        "decode_top1_exact": bf16_second_top2["token_ids"][0]
        == fp16_second_top2["token_ids"][0],
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_records = [
        _save_array(args.output_dir, "inputs_embeds", fp16_decode_embedding),
        _save_array(args.output_dir, "hidden_sources", fp16_prefill_hidden),
        _save_array(args.output_dir, "position_ids", decode_position),
        _save_array(args.output_dir, "past_key", fp16_prefill_cache.key),
        _save_array(args.output_dir, "past_value", fp16_prefill_cache.value),
    ]
    expected_records = [
        _save_array(args.output_dir, "expected_mtp_hidden_fp16", fp16_decode_hidden),
        _save_array(args.output_dir, "expected_present_key_fp16", fp16_decode_cache.key),
        _save_array(
            args.output_dir, "expected_present_value_fp16", fp16_decode_cache.value
        ),
    ]
    reference_records = [
        _save_array(
            args.output_dir, "reference_mtp_hidden_bf16_as_fp32", bf16_decode_hidden.float()
        ),
        _save_array(
            args.output_dir,
            "reference_present_key_bf16_as_fp32",
            bf16_decode_cache.key.float(),
        ),
        _save_array(
            args.output_dir,
            "reference_present_value_bf16_as_fp32",
            bf16_decode_cache.value.float(),
        ),
    ]
    passed = all(gates.values())
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "case_id": args.case_id,
        "scope": "real-text official-weight MTP second-draft S1/P1 core case",
        "prefix_token_ids": tokens,
        "first_draft_token_id": first_draft,
        "second_draft_token_id": bf16_second_top2["token_ids"][0],
        "input_order": [
            "inputs_embeds",
            "hidden_sources",
            "position_ids",
            "past_key",
            "past_value",
        ],
        "output_order": ["mtp_hidden", "present_key", "present_value"],
        "inputs": input_records,
        "expected_fp16_outputs": expected_records,
        "bf16_reference_outputs_as_fp32": reference_records,
        "top2": {
            "prefill_bf16": bf16_first_top2,
            "prefill_fp16": fp16_first_top2,
            "decode_bf16": bf16_second_top2,
            "decode_fp16": fp16_second_top2,
        },
        "metrics": metrics,
        "gates": gates,
        "timing_seconds": {
            "main_load": loaded - started,
            "main_forward": forwarded - loaded,
            "mtp_and_projection": computed - forwarded,
            "total_before_serialization": computed - started,
        },
        "threshold_spec": {
            "file": args.threshold_spec.name,
            "sha256": sha256_file(args.threshold_spec),
        },
        "claim_boundary": (
            "This fixture validates the MTP core S1/P1 ABI with a BF16 main hidden "
            "source cast to FP16. It does not validate the internal ordinary graph, "
            "ATC conversion, or Ascend 310P execution."
        ),
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--threshold-spec", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--prefix-token-ids", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--logit-chunk-size", type=int, default=8192)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.model_dir = args.model_dir.expanduser().resolve()
    args.threshold_spec = args.threshold_spec.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    run(args)


if __name__ == "__main__":
    main()
