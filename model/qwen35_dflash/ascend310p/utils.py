"""Small deployment helpers with strict run-directory and hash handling."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: str | Path, *, relative_to: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    root = Path(relative_to).expanduser().resolve()
    return {
        "path": resolved.relative_to(root).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": sha256_file(resolved),
    }


def require_run_output(path: str | Path) -> Path:
    """Resolve an output and prove it is below the active workspace run."""

    run_value = os.environ.get("AI_RUN_DIR")
    if not run_value:
        raise RuntimeError("deployment output requires an active AI_RUN_DIR")
    run_dir = Path(run_value).expanduser().resolve()
    resolved = Path(path).expanduser().resolve()
    if resolved == run_dir or run_dir not in resolved.parents:
        raise RuntimeError(
            f"deployment output must be below AI_RUN_DIR={run_dir}, got {resolved}"
        )
    return resolved


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    output = require_run_output(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return output


def load_json_object(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"JSON root must be an object: {resolved}")
    return payload


def import_symbol(reference: str) -> Any:
    module_name, separator, attribute = reference.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("symbol reference must use module.path:attribute")
    module = importlib.import_module(module_name)
    value: Any = module
    for component in attribute.split("."):
        value = getattr(value, component)
    return value


def resolve_callable(reference: str | Callable[..., Any]) -> Callable[..., Any]:
    value = import_symbol(reference) if isinstance(reference, str) else reference
    if not callable(value):
        raise TypeError(f"deployment factory is not callable: {value!r}")
    return value


def contained_path(root: str | Path, relative: str) -> Path:
    base = Path(root).expanduser().resolve()
    candidate = (base / relative).resolve()
    if candidate == base or base not in candidate.parents:
        raise ValueError(f"artifact path escapes bundle root: {relative!r}")
    return candidate
