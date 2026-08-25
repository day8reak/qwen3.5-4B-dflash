"""Checkpoint audit and bounded SafeTensors loading for the official MTP layer."""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from .config import Qwen35MTPConfig, audit_official_4b_config


EMBEDDING_WEIGHT = "model.language_model.embed_tokens.weight"


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


class SafeTensorRepository:
    """Resolve named tensors through a Hugging Face sharded index."""

    def __init__(self, model_dir: str | Path) -> None:
        self.model_dir = Path(model_dir).expanduser().resolve()
        self.index_path = self.model_dir / "model.safetensors.index.json"
        self.index = json.loads(self.index_path.read_text(encoding="utf-8"))
        self.weight_map: dict[str, str] = self.index["weight_map"]

    def metadata(self, names: Iterable[str]) -> dict[str, dict[str, Any]]:
        from safetensors import safe_open

        requested = list(names)
        unknown = sorted(set(requested) - set(self.weight_map))
        if unknown:
            raise KeyError(f"checkpoint tensors are missing: {unknown}")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)
        result: dict[str, dict[str, Any]] = {}
        for shard, shard_names in sorted(by_shard.items()):
            with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    view = handle.get_slice(name)
                    result[name] = {
                        "shape": list(view.get_shape()),
                        "dtype": str(view.get_dtype()),
                        "shard": shard,
                    }
        return result

    def load(
        self,
        names: Iterable[str],
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> dict[str, torch.Tensor]:
        from safetensors import safe_open

        requested = list(names)
        unknown = sorted(set(requested) - set(self.weight_map))
        if unknown:
            raise KeyError(f"checkpoint tensors are missing: {unknown}")
        by_shard: dict[str, list[str]] = defaultdict(list)
        for name in requested:
            by_shard[self.weight_map[name]].append(name)
        result: dict[str, torch.Tensor] = {}
        target_device = torch.device(device)
        for shard, shard_names in sorted(by_shard.items()):
            with safe_open(self.model_dir / shard, framework="pt", device="cpu") as handle:
                for name in shard_names:
                    tensor = handle.get_tensor(name)
                    if dtype is not None and tensor.is_floating_point():
                        tensor = tensor.to(dtype=dtype)
                    if target_device.type != "cpu":
                        tensor = tensor.to(device=target_device)
                    result[name] = tensor
        return result


def audit_checkpoint(
    model_dir: str | Path,
    *,
    verify_manifest_hashes: bool = False,
) -> dict[str, Any]:
    """Audit config, all 15 MTP tensors, tied embedding, and resource identity."""

    root = Path(model_dir).expanduser().resolve()
    config_path = root / "config.json"
    index_path = root / "model.safetensors.index.json"
    config = Qwen35MTPConfig.from_pretrained(root)
    repository = SafeTensorRepository(root)
    expected = config.required_tensor_shapes()
    actual_mtp = sorted(name for name in repository.weight_map if name.startswith("mtp."))
    missing = sorted(set(expected) - set(actual_mtp))
    extra = sorted(set(actual_mtp) - set(expected))
    metadata = repository.metadata([*expected, EMBEDDING_WEIGHT])
    shape_mismatches = {
        name: {"expected": list(shape), "actual": metadata[name]["shape"]}
        for name, shape in expected.items()
        if tuple(metadata[name]["shape"]) != shape
    }
    embedding_expected = [config.vocab_size, config.hidden_size]
    if metadata[EMBEDDING_WEIGHT]["shape"] != embedding_expected:
        shape_mismatches[EMBEDDING_WEIGHT] = {
            "expected": embedding_expected,
            "actual": metadata[EMBEDDING_WEIGHT]["shape"],
        }
    dtype_mismatches = {
        name: item["dtype"]
        for name, item in metadata.items()
        if item["dtype"] != "BF16"
    }

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    tied = bool(raw_config.get("tie_word_embeddings")) and bool(
        raw_config.get("text_config", {}).get("tie_word_embeddings")
    )
    official_mismatches = audit_official_4b_config(config)
    errors: list[str] = []
    if missing:
        errors.append(f"missing MTP tensors: {missing}")
    if extra:
        errors.append(f"unexpected MTP tensors: {extra}")
    if shape_mismatches:
        errors.append("one or more MTP/embedding shapes do not match config")
    if dtype_mismatches:
        errors.append("one or more MTP/embedding tensors are not BF16")
    if official_mismatches:
        errors.append("config is not the locked official Qwen3.5-4B shape")
    if not tied:
        errors.append("checkpoint does not declare tied token embedding / LM head")

    resource_manifest = root / "manifest.json"
    manifest: dict[str, Any] | None = None
    manifest_hash_results: list[dict[str, Any]] = []
    if resource_manifest.is_file():
        manifest = json.loads(resource_manifest.read_text(encoding="utf-8"))
        if verify_manifest_hashes:
            for item in manifest.get("files", []):
                file_path = root / item["path"]
                actual_hash = sha256_file(file_path)
                passed = actual_hash == item["sha256"]
                manifest_hash_results.append(
                    {
                        "path": item["path"],
                        "expected_sha256": item["sha256"],
                        "actual_sha256": actual_hash,
                        "passed": passed,
                    }
                )
                if not passed:
                    errors.append(f"manifest hash mismatch: {item['path']}")

    return {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "model_dir": str(root),
        "config_sha256": sha256_file(config_path),
        "index_sha256": sha256_file(index_path),
        "config": config.to_dict(),
        "official_4b_config_mismatches": official_mismatches,
        "required_mtp_tensor_count": len(expected),
        "actual_mtp_tensor_count": len(actual_mtp),
        "missing_mtp_tensors": missing,
        "extra_mtp_tensors": extra,
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "tied_embedding_lm_head": tied,
        "tensor_metadata": metadata,
        "resource": None
        if manifest is None
        else {
            "asset_id": manifest.get("asset_id"),
            "source": manifest.get("source"),
            "manifest_sha256": sha256_file(resource_manifest),
            "hashes_verified": verify_manifest_hashes,
            "hash_results": manifest_hash_results,
        },
        "errors": errors,
    }
