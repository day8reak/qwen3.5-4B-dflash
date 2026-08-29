"""Resolve immutable model data through workspace and project manifests."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from .utils import contained_path, load_json_object, sha256_file


@dataclass(frozen=True)
class LockedDataResource:
    asset_id: str
    root: Path
    manifest_path: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]
    lock: Mapping[str, Any]

    def file(self, relative_path: str, *, verify: bool = True) -> Path:
        records = {
            str(item["path"]): item for item in self.manifest.get("files", [])
        }
        if relative_path not in records:
            raise ValueError(
                f"resource {self.asset_id!r} does not declare file {relative_path!r}"
            )
        record = records[relative_path]
        path = contained_path(self.root, relative_path)
        if not path.is_file():
            raise FileNotFoundError(f"locked resource file is missing: {path}")
        if verify:
            if path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"locked resource size mismatch: {self.asset_id}/{relative_path}")
            if sha256_file(path) != record["sha256"]:
                raise ValueError(f"locked resource hash mismatch: {self.asset_id}/{relative_path}")
        return path

    def verify_all_files(self) -> None:
        for item in self.manifest.get("files", []):
            self.file(str(item["path"]), verify=True)


def resolve_locked_data(
    asset_id: str,
    *,
    workspace_root: str | Path | None = None,
    model_root: str | Path | None = None,
) -> LockedDataResource:
    """Resolve one project-declared shared data asset and verify its manifest."""

    workspace_value = workspace_root or os.environ.get("AI_WS_ROOT")
    model_value = model_root or os.environ.get("AI_MODEL_ROOT")
    if workspace_value is None or model_value is None:
        raise RuntimeError(
            "locked data resolution requires AI_WS_ROOT and AI_MODEL_ROOT from ws env"
        )
    workspace = Path(workspace_value).expanduser().resolve()
    model = Path(model_value).expanduser().resolve()
    workspace_manifest = load_json_object(workspace / "workspace.yaml")
    project = load_json_object(model / "project.yaml")
    dependencies = project.get("dependencies", {}).get("data", [])
    if asset_id not in dependencies:
        raise ValueError(f"project does not declare data asset {asset_id!r}")

    data_lock = load_json_object(model / "specs" / "data.lock.json")
    matches = [
        item for item in data_lock.get("resources", []) if item.get("asset_id") == asset_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"data.lock.json must contain exactly one record for {asset_id!r}"
        )
    lock = matches[0]
    shared_data = str(workspace_manifest["shared"]["data"])
    data_root = contained_path(workspace, shared_data)
    resource_root = contained_path(data_root, asset_id)
    manifest_path = resource_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"data resource manifest is missing: {manifest_path}")
    manifest_hash = sha256_file(manifest_path)
    if manifest_hash != lock["manifest_sha256"]:
        raise ValueError(f"data resource manifest hash mismatch: {asset_id}")
    manifest = load_json_object(manifest_path)
    if manifest.get("asset_id") != asset_id:
        raise ValueError(f"data resource asset_id mismatch: {asset_id}")
    source_revision = manifest.get("source", {}).get("revision")
    if source_revision != lock.get("source_revision"):
        raise ValueError(f"data resource source revision mismatch: {asset_id}")
    checkpoint = manifest.get("checkpoint", {})
    expected_scalars = {
        "tensor_count": checkpoint.get("tensor_count"),
        "tensor_bytes": checkpoint.get("total_tensor_bytes"),
        "shards": checkpoint.get("shards"),
    }
    mismatches = {
        name: {"lock": lock.get(name), "manifest": value}
        for name, value in expected_scalars.items()
        if value is not None and int(lock.get(name, -1)) != int(value)
    }
    if mismatches:
        raise ValueError(f"data resource checkpoint contract mismatch: {mismatches}")
    return LockedDataResource(
        asset_id=asset_id,
        root=resource_root,
        manifest_path=manifest_path,
        manifest_sha256=manifest_hash,
        manifest=manifest,
        lock=lock,
    )
