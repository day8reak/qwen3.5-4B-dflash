#!/usr/bin/env python3
"""Export one fixed Qwen3.5-4B MTP gear using only standard PyTorch ops."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnx
import torch
from torch import nn

from qwen35_mtp.config import Qwen35MTPConfig
from qwen35_mtp.export import MTPCoreExportWrapper, export_mtp_core_onnx
from qwen35_mtp.mtp import Qwen35MTPDrafter
from qwen35_mtp.weights import SafeTensorRepository, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--past-length", type=int, default=0)
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    args = parser.parse_args()
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    config = Qwen35MTPConfig.from_pretrained(args.model_dir)
    repository = SafeTensorRepository(args.model_dir)
    # The core accepts embeddings and does not reference the table.  A meta
    # embedding preserves the module/state naming without allocating 1.27 GB.
    embedding = nn.Embedding(
        config.vocab_size, config.hidden_size, device="meta", dtype=dtype
    )
    drafter = Qwen35MTPDrafter(config, embedding)
    drafter.mtp.to(dtype=dtype)
    drafter.load_official_mtp_state(
        repository.load(config.required_tensor_shapes(), dtype=dtype)
    )
    output = export_mtp_core_onnx(
        MTPCoreExportWrapper(drafter.mtp),
        args.output,
        sequence_length=args.sequence_length,
        past_length=args.past_length,
        hidden_size=config.hidden_size,
        kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        dtype=dtype,
    )
    onnx.checker.check_model(str(output), full_check=False)
    metadata = {
        "schema_version": 1,
        "status": "PASS",
        "artifact": output.name,
        "artifact_sha256": sha256_file(output),
        "sequence_length": args.sequence_length,
        "past_length": args.past_length,
        "dtype": args.dtype,
        "inputs_are_materialized_embeddings": True,
        "lm_head_is_external_and_tied": True,
    }
    metadata_path = output.with_suffix(output.suffix + ".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
