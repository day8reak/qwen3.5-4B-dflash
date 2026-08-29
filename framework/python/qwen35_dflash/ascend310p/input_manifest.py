"""Deterministic external-input manifests for quant AIR/OM builds."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping

from .quant_factory import QUANT_BASE_REVISION
from .utils import atomic_write_json, load_json_object, sha256_file


_SCHEMA_VERSION = 1
_ARTIFACT_KIND = "qwen35-quant-air-om-input-manifest"


def _regular_file(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    return resolved


def _regular_directory(path: Path, *, label: str) -> Path:
    candidate = path.expanduser()
    if candidate.is_symlink():
        raise ValueError(f"{label} must not be a symlink: {candidate}")
    resolved = candidate.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"{label} is not a directory: {resolved}")
    return resolved


def _directory_files(root: Path) -> list[Path]:
    result: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"external input tree contains a symlink: {path}")
        if path.is_file():
            result.append(path.resolve())
    if not result:
        raise ValueError(f"external input directory contains no files: {root}")
    return result


def _group_digest(records: Iterable[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for record in records:
        digest.update(str(record["path"]).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(record["bytes"]).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(record["sha256"]).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _directory_group(root: Path) -> dict[str, Any]:
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in _directory_files(root)
    ]
    return {
        "root": str(root),
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "sha256": _group_digest(files),
        "files": files,
    }


def _file_group(paths: Iterable[Path]) -> dict[str, Any]:
    files = []
    for path in sorted((_regular_file(item, label="manifest input") for item in paths)):
        files.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise ValueError("file group must not be empty")
    return {
        "file_count": len(files),
        "bytes": sum(int(item["bytes"]) for item in files),
        "sha256": _group_digest(files),
        "files": files,
    }


def build_quant_input_manifest(
    *,
    target_dir: str | Path,
    draft_dir: str | Path,
    quant_config: str | Path,
    receiver_models_dir: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    """Hash every external model/quant input consumed by the graph factory."""

    from models.dflash_v1.target_quant import load_original_quant_config

    target = _regular_directory(Path(target_dir), label="target_dir")
    draft = _regular_directory(Path(draft_dir), label="draft_dir")
    receiver = _regular_directory(
        Path(receiver_models_dir), label="receiver_models_dir"
    )
    wrapper = _regular_file(
        receiver / "export_model_wrapper_qwen3_5.py",
        label="receiver export wrapper",
    )
    quant = load_original_quant_config(quant_config)
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "artifact_kind": _ARTIFACT_KIND,
        "status": "PASS",
        "quant_branch_base_revision": QUANT_BASE_REVISION,
        "groups": {
            "target_checkpoint": _directory_group(target),
            "draft_checkpoint": _directory_group(draft),
            "quant_linear_weights": _directory_group(quant.quant_weight_path),
            "quant_embedding": _file_group(
                (quant.embedding_weight_path, quant.embedding_scale_path)
            ),
            "receiver_wrapper": _file_group((wrapper,)),
            "quant_config": _file_group((quant.config_path,)),
        },
    }
    path = atomic_write_json(output, payload)
    payload["manifest_path"] = str(path)
    payload["manifest_sha256"] = sha256_file(path)
    return payload


def _verify_directory_group(name: str, group: Mapping[str, Any]) -> Path:
    root = _regular_directory(Path(str(group.get("root", ""))), label=name)
    records = group.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"input manifest group {name} contains no files")
    actual_records = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"input manifest group {name} has a non-object record")
        path = (root / str(record.get("path", ""))).resolve()
        if path == root or root not in path.parents:
            raise ValueError(f"input manifest path escapes {name}: {path}")
        path = _regular_file(path, label=f"{name} payload")
        actual = {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual != dict(record):
            raise ValueError(f"input manifest payload changed: {name}/{actual['path']}")
        actual_records.append(actual)
    if int(group.get("file_count", -1)) != len(actual_records):
        raise ValueError(f"input manifest file count differs for {name}")
    if int(group.get("bytes", -1)) != sum(
        int(item["bytes"]) for item in actual_records
    ):
        raise ValueError(f"input manifest byte count differs for {name}")
    if str(group.get("sha256")) != _group_digest(actual_records):
        raise ValueError(f"input manifest group digest differs for {name}")
    if len(_directory_files(root)) != len(actual_records):
        raise ValueError(f"unmanifested file appeared in {name}")
    return root


def _verify_file_group(name: str, group: Mapping[str, Any]) -> tuple[Path, ...]:
    records = group.get("files")
    if not isinstance(records, list) or not records:
        raise ValueError(f"input manifest group {name} contains no files")
    actual_records = []
    paths = []
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError(f"input manifest group {name} has a non-object record")
        path = _regular_file(Path(str(record.get("path", ""))), label=name)
        actual = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        if actual != dict(record):
            raise ValueError(f"input manifest payload changed: {path}")
        paths.append(path)
        actual_records.append(actual)
    if int(group.get("file_count", -1)) != len(actual_records):
        raise ValueError(f"input manifest file count differs for {name}")
    if int(group.get("bytes", -1)) != sum(
        int(item["bytes"]) for item in actual_records
    ):
        raise ValueError(f"input manifest byte count differs for {name}")
    if str(group.get("sha256")) != _group_digest(actual_records):
        raise ValueError(f"input manifest group digest differs for {name}")
    return tuple(paths)


def verify_quant_input_manifest(path: str | Path) -> dict[str, Any]:
    """Rehash a manifest and return normalized roots/group identities."""

    manifest_path = _regular_file(Path(path), label="input_manifest")
    payload = load_json_object(manifest_path)
    if payload.get("schema_version") != _SCHEMA_VERSION:
        raise ValueError("unsupported quant input manifest schema")
    if payload.get("artifact_kind") != _ARTIFACT_KIND or payload.get("status") != "PASS":
        raise ValueError("input_manifest is not a passing quant manifest")
    if payload.get("quant_branch_base_revision") != QUANT_BASE_REVISION:
        raise ValueError("input_manifest was made for a different quant revision")
    groups = payload.get("groups")
    if not isinstance(groups, Mapping):
        raise TypeError("input_manifest groups must be an object")
    expected = {
        "target_checkpoint",
        "draft_checkpoint",
        "quant_linear_weights",
        "quant_embedding",
        "receiver_wrapper",
        "quant_config",
    }
    if set(groups) != expected:
        raise ValueError("input_manifest group set differs from the locked contract")
    roots = {
        name: _verify_directory_group(name, groups[name])
        for name in (
            "target_checkpoint",
            "draft_checkpoint",
            "quant_linear_weights",
        )
    }
    files = {
        name: _verify_file_group(name, groups[name])
        for name in ("quant_embedding", "receiver_wrapper", "quant_config")
    }
    return {
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "roots": roots,
        "files": files,
        "group_sha256": {
            name: str(groups[name]["sha256"]) for name in sorted(expected)
        },
    }


__all__ = ["build_quant_input_manifest", "verify_quant_input_manifest"]
