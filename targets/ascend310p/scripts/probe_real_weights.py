#!/usr/bin/env python3
"""Run one real-weight MTP proposal/verification probe on a committed prefix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from qwen35_mtp.backends import TorchMTPDraftBackend, TransformersMainBackend
from qwen35_mtp.mtp import Qwen35MTPDrafter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--committed-token-ids",
        required=True,
        help="comma-separated target-valid prefix with at least two tokens",
    )
    parser.add_argument("--draft-tokens", type=int, default=2)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    tokens = [int(item) for item in args.committed_token_ids.split(",")]
    if len(tokens) < 2:
        raise ValueError("probe requires at least two committed tokens")
    if args.draft_tokens <= 0:
        raise ValueError("draft-tokens must be positive")
    torch.set_num_threads(max(1, args.threads))

    started = time.perf_counter()
    main_backend = TransformersMainBackend.from_pretrained(
        args.model_dir, dtype=torch.bfloat16
    )
    drafter = TorchMTPDraftBackend(
        Qwen35MTPDrafter.from_pretrained(
            args.model_dir,
            embedding=main_backend.embedding,
            dtype=torch.bfloat16,
        )
    )
    loaded = time.perf_counter()
    prefix_tensor = torch.tensor([tokens], dtype=torch.long)
    context = main_backend.evaluate(prefix_tensor, [])
    proposed_at = time.perf_counter()
    proposals = drafter.propose(
        prefix_tensor,
        context.hidden_states,
        args.draft_tokens,
    )
    drafted_at = time.perf_counter()
    verify_tokens = [*tokens, *proposals]
    first_row = len(tokens) - 1
    rows = list(range(first_row, first_row + len(proposals) + 1))
    verification = main_backend.evaluate(
        torch.tensor([verify_tokens], dtype=torch.long), rows
    )
    finished = time.perf_counter()
    targets = [int(token) for token in verification.top1_token_ids[0].tolist()]
    mismatch = next(
        (index for index, token in enumerate(proposals) if token != targets[index]),
        None,
    )
    accepted = len(proposals) if mismatch is None else mismatch
    report = {
        "schema_version": 1,
        "status": "PASS",
        "scope": "real BF16 weights, CPU proposal and main verification probe",
        "committed_token_ids": tokens,
        "draft_token_ids": proposals,
        "target_top1_token_ids": targets,
        "accepted_prefix_length": accepted,
        "correction_token_id": targets[accepted],
        "timing_seconds": {
            "load": loaded - started,
            "main_context": proposed_at - loaded,
            "draft": drafted_at - proposed_at,
            "main_verify": finished - drafted_at,
            "total": finished - started,
        },
        "claim_boundary": (
            "This exercises official weights and alignment, but is not an "
            "Ascend 310P result or a multi-prompt accuracy claim."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
