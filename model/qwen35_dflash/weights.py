"""Audit and streaming load for the public Qwen3.5-4B-DFlash checkpoint."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import Qwen35DFlashConfig, audit_official_4b_dflash_config


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_path(model_dir: str | Path) -> Path:
    path = Path(model_dir).expanduser().resolve() / "model.safetensors"
    if not path.is_file():
        raise FileNotFoundError(f"DFlash checkpoint is missing: {path}")
    return path


def audit_dflash_checkpoint(
    model_dir: str | Path,
    *,
    verify_model_hash: bool = False,
) -> dict[str, Any]:
    from safetensors import safe_open

    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    config = Qwen35DFlashConfig.from_pretrained(root)
    expected = config.required_tensor_shapes()
    checkpoint = _checkpoint_path(root)
    metadata: dict[str, dict[str, Any]] = {}
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        actual_names = sorted(handle.keys())
        for name in actual_names:
            view = handle.get_slice(name)
            metadata[name] = {
                "shape": list(view.get_shape()),
                "dtype": str(view.get_dtype()),
            }

    missing = sorted(set(expected) - set(actual_names))
    extra = sorted(set(actual_names) - set(expected))
    shape_mismatches = {
        name: {"expected": list(shape), "actual": metadata[name]["shape"]}
        for name, shape in expected.items()
        if name in metadata and tuple(metadata[name]["shape"]) != shape
    }
    dtype_mismatches = {
        name: item["dtype"]
        for name, item in metadata.items()
        if item["dtype"] != "BF16"
    }
    official_mismatches = audit_official_4b_dflash_config(config)
    errors: list[str] = []
    if missing:
        errors.append(f"missing DFlash tensors: {missing}")
    if extra:
        errors.append(f"unexpected DFlash tensors: {extra}")
    if shape_mismatches:
        errors.append("one or more DFlash tensor shapes differ")
    if dtype_mismatches:
        errors.append("one or more DFlash tensors are not BF16")
    if official_mismatches:
        errors.append("config is not the locked Qwen3.5-4B-DFlash shape")

    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "model_dir": str(root),
        "config_sha256": sha256_file(config_path),
        "model_bytes": checkpoint.stat().st_size,
        "model_sha256": sha256_file(checkpoint) if verify_model_hash else None,
        "config": config.to_dict(),
        "parameter_count": config.parameter_count,
        "required_tensor_count": len(expected),
        "actual_tensor_count": len(actual_names),
        "official_config_mismatches": official_mismatches,
        "missing_tensors": missing,
        "extra_tensors": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "errors": errors,
    }


@torch.no_grad()
def load_dflash_weights(model: nn.Module, model_dir: str | Path) -> None:
    """Copy one tensor at a time so loading never duplicates the full checkpoint."""

    from safetensors import safe_open

    config = Qwen35DFlashConfig.from_pretrained(model_dir)
    expected = config.required_tensor_shapes()
    parameters = dict(model.named_parameters())
    if set(parameters) != set(expected):
        missing = sorted(set(expected) - set(parameters))
        extra = sorted(set(parameters) - set(expected))
        raise RuntimeError(
            f"model parameter contract differs: missing={missing}, extra={extra}"
        )

    checkpoint = _checkpoint_path(model_dir)
    with safe_open(checkpoint, framework="pt", device="cpu") as handle:
        actual = set(handle.keys())
        if actual != set(expected):
            missing = sorted(set(expected) - actual)
            extra = sorted(actual - set(expected))
            raise RuntimeError(
                f"checkpoint tensor contract differs: missing={missing}, extra={extra}"
            )
        for name in sorted(expected):
            source = handle.get_tensor(name)
            if tuple(source.shape) != expected[name]:
                raise RuntimeError(
                    f"{name} has shape {tuple(source.shape)}, expected {expected[name]}"
                )
            destination = parameters[name]
            destination.copy_(source.to(device=destination.device, dtype=destination.dtype))
