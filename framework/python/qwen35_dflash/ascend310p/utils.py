"""Small deployment helpers with strict run-directory and hash handling."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Iterable, Mapping


_GE_IR_NODE_FIELD = re.compile(
    r'\b(?P<field>op|type):\s*"(?P<ge_type>[A-Za-z_][A-Za-z0-9_]*)"'
)


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    resolved = Path(path).expanduser().resolve()
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def count_ge_ir_nodes(paths: Iterable[str | Path]) -> dict[str, int]:
    """Count GE nodes across TorchAir ``dynamo.pbtxt`` dialects.

    Receiver TorchAir versions serialize the node kind as either ``type:`` or
    ``op:``. A dump may expose both fields for the same physical nodes, so the
    counts must not be added. Select the larger per-file count for each GE
    type, then add independent files.
    """

    totals: dict[str, int] = {}
    for value in paths:
        path = Path(value)
        field_counts: dict[str, dict[str, int]] = {"op": {}, "type": {}}
        with path.open("r", encoding="utf-8") as stream:
            for line in stream:
                for match in _GE_IR_NODE_FIELD.finditer(line):
                    field = match.group("field")
                    ge_type = match.group("ge_type")
                    counts = field_counts[field]
                    counts[ge_type] = counts.get(ge_type, 0) + 1
        for ge_type in field_counts["op"].keys() | field_counts["type"].keys():
            occurrences = max(
                field_counts["op"].get(ge_type, 0),
                field_counts["type"].get(ge_type, 0),
            )
            totals[ge_type] = totals.get(ge_type, 0) + occurrences
    return totals


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
