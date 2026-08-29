"""Compile a hash-locked TorchAir bundle into Ascend 310P OM artifacts."""

from __future__ import annotations

from pathlib import Path
import os
import re
import subprocess
from typing import Any, Callable, Mapping, Sequence

from .utils import (
    atomic_write_json,
    contained_path,
    file_record,
    load_json_object,
    require_run_output,
    sha256_file,
)


_FORBIDDEN_ATC_PREFIXES = (
    "--framework",
    "--model",
    "--output",
    "--soc_version",
    "--mode",
)


class AtcCompileError(RuntimeError):
    """ATC failed or did not produce the promised OM artifact."""


def validate_soc_version(soc_version: str) -> str:
    """Require an ATC SoC variant instead of the generic 310P family name."""

    value = str(soc_version).strip()
    if not value:
        raise ValueError("soc_version must be the exact ATC target identity")
    normalized = re.sub(r"[^a-z0-9]", "", value.lower())
    if normalized in {"310p", "ascend310p", "atlas310p"}:
        raise ValueError(
            "soc_version must identify the concrete 310P ATC variant, "
            "for example Ascend310P3; a generic Ascend310P value is not sufficient"
        )
    return value


def resolve_atc_executable(atc_bin: str | Path | None = None) -> Path:
    """Resolve ATC from the declared profile and fail before graph construction."""

    configured_atc = atc_bin or os.environ.get("ASCEND310P_ATC_BIN")
    if configured_atc is None:
        raise RuntimeError(
            "ATC is unavailable in the declared target profile; provide --atc from a CANN profile"
        )
    atc_path = Path(configured_atc).expanduser().resolve()
    if not atc_path.is_file() or not os.access(atc_path, os.X_OK):
        raise RuntimeError(f"ATC is not an executable file: {atc_path}")
    return atc_path


def _validate_extra_args(arguments: Sequence[str]) -> list[str]:
    result = []
    for argument in arguments:
        value = str(argument)
        if not value.startswith("--"):
            raise ValueError(f"ATC extra argument must start with '--': {value!r}")
        if value.startswith(_FORBIDDEN_ATC_PREFIXES):
            raise ValueError(f"ATC core option cannot be overridden: {value!r}")
        result.append(value)
    return result


def _default_runner(command: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        list(command),
        cwd=cwd,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _atc_identity(atc_bin: Path) -> str:
    result = subprocess.run(
        [str(atc_bin), "--version"],
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    text = result.stdout.strip()
    return text if text else f"exit={result.returncode} (no version text)"


def compile_air_bundle(
    air_manifest_path: str | Path,
    *,
    soc_version: str,
    atc_bin: str | Path | None = None,
    extra_args: Sequence[str] = (),
    runner: Callable[[Sequence[str], Path], subprocess.CompletedProcess[str]] | None = None,
    atc_identity: str | None = None,
) -> dict[str, Any]:
    """Compile all AIR graphs with ``framework=1`` into the same run bundle."""

    manifest_path = Path(air_manifest_path).expanduser().resolve()
    root = require_run_output(manifest_path.parent)
    air_manifest = load_json_object(manifest_path)
    if air_manifest.get("status") != "PASS":
        raise ValueError("AIR manifest is not passing")
    if air_manifest.get("artifact_kind") != "qwen35-dflash-torchair-bundle":
        raise ValueError("unexpected AIR artifact kind")
    exact_soc_version = validate_soc_version(soc_version)
    atc_path = resolve_atc_executable(atc_bin)

    arguments = _validate_extra_args(extra_args)
    execute = runner or _default_runner
    om_root = root / "om"
    if om_root.exists() and any(om_root.iterdir()):
        raise FileExistsError(f"OM output directory is not empty: {om_root}")
    om_root.mkdir(parents=True, exist_ok=True)
    run_dir = Path(os.environ["AI_RUN_DIR"]).expanduser().resolve()
    log_root = run_dir / "log" / "dflash-atc"
    log_root.mkdir(parents=True, exist_ok=True)

    compiled: list[dict[str, Any]] = []
    for graph in air_manifest.get("graphs", []):
        if not isinstance(graph, Mapping):
            raise TypeError("AIR graph manifest entry must be an object")
        name = str(graph["name"])
        air_record = graph["air"]
        payload_records = graph.get("payload_files")
        if not isinstance(payload_records, list) or not payload_records:
            raise ValueError(f"AIR graph has no payload manifest: {name}")
        for record in payload_records:
            payload_path = contained_path(root, str(record["path"]))
            if not payload_path.is_file():
                raise FileNotFoundError(f"AIR payload is missing: {payload_path}")
            if payload_path.stat().st_size != int(record["bytes"]):
                raise ValueError(f"AIR payload size mismatch before ATC: {record['path']}")
            if sha256_file(payload_path) != record["sha256"]:
                raise ValueError(f"AIR payload hash mismatch before ATC: {record['path']}")
        air_path = contained_path(root, str(air_record["path"]))
        if not air_path.is_file():
            raise FileNotFoundError(f"AIR graph is missing: {air_path}")
        actual_hash = sha256_file(air_path)
        if actual_hash != air_record["sha256"]:
            raise ValueError(f"AIR graph hash mismatch before ATC: {name}")

        output_prefix = om_root / name
        command = [
            str(atc_path),
            "--mode=0",
            "--framework=1",
            f"--model={air_path}",
            f"--output={output_prefix}",
            f"--soc_version={exact_soc_version}",
            *arguments,
        ]
        result = execute(command, air_path.parent)
        log_path = log_root / f"{name}.log"
        log_path.write_text(result.stdout or "", encoding="utf-8")
        om_path = output_prefix.with_suffix(".om")
        if result.returncode != 0:
            raise AtcCompileError(
                f"ATC failed for {name!r} with exit {result.returncode}; log={log_path}"
            )
        if not om_path.is_file() or om_path.stat().st_size == 0:
            raise AtcCompileError(
                f"ATC returned success but produced no non-empty OM for {name!r}; log={log_path}"
            )
        compiled.append(
            {
                "name": name,
                "role": graph["role"],
                "input_names": list(graph.get("input_names", [])),
                "output_names": list(graph.get("output_names", [])),
                "air": dict(air_record),
                "om": file_record(om_path, relative_to=root),
                "atc_command": command,
                "atc_log": str(log_path.relative_to(run_dir)),
            }
        )

    deployment = {
        "schema_version": 1,
        "artifact_kind": "qwen35-dflash-ascend310p-om-bundle",
        "status": "PASS",
        "target": {"target_id": "ascend310p", "soc_version": exact_soc_version},
        "air_manifest": {
            "path": manifest_path.relative_to(root).as_posix(),
            "sha256": sha256_file(manifest_path),
        },
        "compiler": {
            "path": str(atc_path),
            "identity": atc_identity or _atc_identity(atc_path),
            "framework": 1,
            "extra_args": arguments,
        },
        "graphs": compiled,
    }
    output = atomic_write_json(root / "deployment-manifest.json", deployment)
    deployment["manifest_path"] = str(output)
    return deployment
