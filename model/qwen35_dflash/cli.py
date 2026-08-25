"""Command line tools for checkpoint audit and cache-free golden cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .model import DFlashDraftModel
from .ops import ModuleDFlashOps, TorchDFlashOps
from .weights import audit_dflash_checkpoint


DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _write_json(payload: dict[str, Any], output: str | None) -> None:
    text = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    if output is None:
        print(text)
    else:
        path = Path(output).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")
        print(text)


def _ops(args):
    if args.ops_backend is None:
        return TorchDFlashOps()
    return ModuleDFlashOps.from_name(
        args.ops_backend,
        strict=not args.allow_op_fallback,
    )


def command_audit(args) -> int:
    report = audit_dflash_checkpoint(
        args.draft_dir,
        verify_model_hash=args.verify_model_hash,
    )
    _write_json(report, args.output)
    return 0 if report["status"] == "PASS" else 1


def _load_case(path: str, *, device: torch.device, dtype: torch.dtype):
    with np.load(Path(path).expanduser().resolve(), allow_pickle=False) as archive:
        required = {"target_hidden", "noise_embedding", "position_ids"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"case is missing arrays: {missing}")
        target_hidden = torch.from_numpy(archive["target_hidden"]).to(
            device=device, dtype=dtype
        )
        noise_embedding = torch.from_numpy(archive["noise_embedding"]).to(
            device=device, dtype=dtype
        )
        position_ids = torch.from_numpy(archive["position_ids"]).to(
            device=device, dtype=torch.long
        )
    return target_hidden, noise_embedding, position_ids


def command_run_case(args) -> int:
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    operations = _ops(args)
    model = DFlashDraftModel.from_pretrained(
        args.draft_dir,
        ops=operations,
        device=device,
        dtype=dtype,
    )
    target_hidden, noise_embedding, position_ids = _load_case(
        args.case, device=device, dtype=dtype
    )
    with torch.inference_mode():
        hidden = model(target_hidden, noise_embedding, position_ids)
    if not torch.isfinite(hidden).all():
        raise FloatingPointError("DFlash golden produced a non-finite hidden tensor")

    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, hidden.float().cpu().numpy(), allow_pickle=False)
    report = {
        "schema_version": 1,
        "status": "PASS",
        "backend": "torch" if args.ops_backend is None else args.ops_backend,
        "strict_custom_ops": bool(args.ops_backend and not args.allow_op_fallback),
        "dtype": args.dtype,
        "device": str(device),
        "target_hidden_shape": list(target_hidden.shape),
        "noise_embedding_shape": list(noise_embedding.shape),
        "position_ids_shape": list(position_ids.shape),
        "output_shape": list(hidden.shape),
        "output": str(output),
        "finite": True,
    }
    _write_json(report, args.report)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B DFlash cache-free PyTorch golden"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit official config and 69 tensors")
    audit.add_argument("--draft-dir", required=True)
    audit.add_argument("--verify-model-hash", action="store_true")
    audit.add_argument("--output")
    audit.set_defaults(handler=command_audit)

    run = subparsers.add_parser("run-case", help="run a cache-free draft-core NPZ case")
    run.add_argument("--draft-dir", required=True)
    run.add_argument("--case", required=True)
    run.add_argument("--output", required=True, help="output .npy hidden tensor")
    run.add_argument("--report")
    run.add_argument("--dtype", choices=sorted(DTYPES), default="float32")
    run.add_argument("--device", default="cpu")
    run.add_argument("--ops-backend")
    run.add_argument("--allow-op-fallback", action="store_true")
    run.set_defaults(handler=command_run_case)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))
