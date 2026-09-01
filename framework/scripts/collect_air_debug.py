#!/usr/bin/env python3
"""Collect a copy-safe AIR export diagnostic bundle without Git metadata.

The target deployment often receives a plain source-tree copy rather than a
Git checkout.  This script therefore identifies the active code by SHA256 and
includes a small source snapshot instead of invoking Git.  It never copies
model checkpoints, quantized weights, AIR files, or OM files into the report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Mapping


ROOT = Path(__file__).resolve().parents[2]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))


DEFAULT_FACTORY = (
    "qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph"
)
SOURCE_FILES = (
    Path("SOURCE_LOCK.json"),
    Path("framework/FRAMEWORK_LOCK.json"),
    Path("framework/python/qwen35_dflash/ascend310p/custom_op_export.py"),
    Path("framework/python/qwen35_dflash/ascend310p/integrated.py"),
    Path("framework/python/qwen35_dflash/ascend310p/incremental.py"),
    Path("framework/python/qwen35_dflash/ascend310p/quant_factory.py"),
    Path("framework/scripts/collect_air_debug.py"),
    Path("models/modeling_qwen3_5_hiai_nd.py"),
    Path("models/modeling_qwen3_5_hiai_nd_dflash_rollback.py"),
)
CUSTOM_OPERATORS = (
    "npu::adn_fused_infer_attention",
    "npu::adn_rms_norm",
    "npu::npu_chunk_gated_delta_rule",
    "npu::npu_gated_delta_rule_mtp",
    "npu::npu_cache_update_",
    "npu::npu_dynamic_quant",
    "npu::npu_quant_matmul",
    "npu::npu_trans_quant_param",
    "npu::npu_scatter_nd_update_",
)
ENVIRONMENT_KEYS = (
    "AI_MODEL_PYTHON",
    "AI_RUN_DIR",
    "AI_TARGET_PROFILE",
    "ASCEND_AICPU_PATH",
    "ASCEND_CUSTOM_OPP_PATH",
    "ASCEND_HOME_PATH",
    "ASCEND_OPP_PATH",
    "ASCEND_RT_VISIBLE_DEVICES",
    "ASCEND_TOOLKIT_HOME",
    "CANN_INSTALL_PATH",
    "DEVICE_ID",
    "LD_LIBRARY_PATH",
    "MODEL_PYTHON",
    "PATH",
    "PYTHONPATH",
)
DIAGNOSTIC_SUFFIXES = {".json", ".log", ".pbtxt", ".txt", ".yaml", ".yml"}
MAX_COPIED_DIAGNOSTIC_BYTES = 128 * 1024 * 1024
MAX_SINGLE_DIAGNOSTIC_BYTES = 64 * 1024 * 1024


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _redact_sensitive(value: object) -> object:
    sensitive_keys = {
        "access_token",
        "api_key",
        "password",
        "private_key",
        "secret",
    }
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for key, item in value.items():
            normalized = str(key).strip().lower()
            result[str(key)] = (
                "<redacted>"
                if normalized in sensitive_keys
                else _redact_sensitive(item)
            )
        return result
    if isinstance(value, list):
        return [_redact_sensitive(item) for item in value]
    return value


def _source_snapshot(report_root: Path) -> list[dict[str, object]]:
    snapshot_root = report_root / "source-snapshot"
    records: list[dict[str, object]] = []
    for relative in SOURCE_FILES:
        source = ROOT / relative
        record: dict[str, object] = {
            "path": relative.as_posix(),
            "exists": source.is_file(),
        }
        if source.is_file():
            record.update(
                {
                    "size_bytes": source.stat().st_size,
                    "sha256": _sha256(source),
                }
            )
            destination = snapshot_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        records.append(record)
    return records


def _run_identity_command(name: str, arguments: list[str]) -> dict[str, object]:
    executable = shutil.which(name)
    if executable is None:
        return {
            "command": [name, *arguments],
            "status": "NOT_FOUND",
        }
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [executable, *arguments],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            errors="replace",
            timeout=30,
        )
        return {
            "command": [executable, *arguments],
            "status": "PASS" if completed.returncode == 0 else "FAILED",
            "returncode": completed.returncode,
            "duration_seconds": time.monotonic() - started,
            "output": completed.stdout,
        }
    except Exception as error:  # noqa: BLE001 - evidence collection must continue
        return {
            "command": [executable, *arguments],
            "status": "ERROR",
            "duration_seconds": time.monotonic() - started,
            "error": repr(error),
        }


def _module_identity(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    # Vendor imports can raise RuntimeError/OSError rather than ImportError.
    except BaseException as error:  # noqa: BLE001
        return {"name": name, "status": "IMPORT_ERROR", "error": repr(error)}
    raw_path = getattr(module, "__file__", None)
    record: dict[str, object] = {
        "name": name,
        "status": "IMPORTED",
        "version": getattr(module, "__version__", None),
        "file": raw_path,
    }
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            record["file_sha256"] = _sha256(path)
            record["file_size_bytes"] = path.stat().st_size
    return record


def _python_stack() -> dict[str, object]:
    distributions: list[dict[str, str]] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            distributions.append({"name": name, "version": distribution.version})
    distributions.sort(key=lambda item: item["name"].lower())

    modules = [
        _module_identity(name)
        for name in (
            "torch",
            "torch._C",
            "torch_npu",
            "torch_npu._C",
            "torchair",
            "transformers",
            "acl",
        )
    ]
    return {
        "executable": sys.executable,
        "version": sys.version,
        "prefix": sys.prefix,
        "modules": modules,
        "distributions": distributions,
    }


def _operator_dispatch() -> dict[str, object]:
    try:
        import torch
    except BaseException as error:  # noqa: BLE001
        return {"status": "TORCH_IMPORT_ERROR", "error": repr(error)}

    torch_npu_import: dict[str, object]
    try:
        import torch_npu

        torch_npu_import = {
            "status": "IMPORTED",
            "npu_trans_quant_param_callable": callable(
                getattr(torch_npu, "npu_trans_quant_param", None)
            ),
        }
    except BaseException as error:  # noqa: BLE001
        torch_npu_import = {"status": "IMPORT_ERROR", "error": repr(error)}

    operators: list[dict[str, object]] = []
    for qualified_name in CUSTOM_OPERATORS:
        record: dict[str, object] = {"name": qualified_name}
        try:
            handle = torch._C._dispatch_find_schema_or_throw(qualified_name, "")
            record["schema"] = str(handle.schema())
        except Exception as error:  # noqa: BLE001
            record["schema_error"] = repr(error)
        try:
            record["dispatch_table"] = torch._C._dispatch_dump_table(
                qualified_name
            )
        except Exception as error:  # noqa: BLE001
            record["dispatch_error"] = repr(error)
        operators.append(record)

    device: dict[str, object] = {}
    npu = getattr(torch, "npu", None)
    if npu is None:
        device["status"] = "torch.npu unavailable"
    else:
        for label, function in (
            ("is_available", getattr(npu, "is_available", None)),
            ("device_count", getattr(npu, "device_count", None)),
            ("current_device", getattr(npu, "current_device", None)),
        ):
            if function is None:
                continue
            try:
                device[label] = function()
            except Exception as error:  # noqa: BLE001
                device[f"{label}_error"] = repr(error)
        count = device.get("device_count")
        if isinstance(count, int):
            names: list[object] = []
            for index in range(count):
                try:
                    names.append(npu.get_device_name(index))
                except Exception as error:  # noqa: BLE001
                    names.append({"index": index, "error": repr(error)})
            device["names"] = names

    return {
        "status": "COLLECTED",
        "torch_npu_import": torch_npu_import,
        "device": device,
        "operators": operators,
    }


def _export_environment() -> dict[str, str]:
    return {
        key: os.environ[key]
        for key in ENVIRONMENT_KEYS
        if key in os.environ
    }


def _ge_prototype_preflight() -> dict[str, object]:
    try:
        from qwen35_dflash.ascend310p.custom_op_export import (
            validate_adn_attention_ge_prototype_environment,
            validate_gdr_ge_prototype_environment,
        )
    except BaseException as error:  # noqa: BLE001
        return {"status": "IMPORT_ERROR", "error": repr(error)}

    results: dict[str, object] = {"status": "COLLECTED"}
    for name, validator in (
        ("gdr", validate_gdr_ge_prototype_environment),
        ("adn_attention", validate_adn_attention_ge_prototype_environment),
    ):
        try:
            results[name] = validator()
        except BaseException as error:  # noqa: BLE001
            results[name] = {"status": "ERROR", "error": repr(error)}
    return results


def _run_export(
    *,
    factory: str,
    factory_config: Path,
    bundle_dir: Path,
    log_path: Path,
    verbose_dynamo: bool,
) -> dict[str, object]:
    command = [
        sys.executable,
        "-m",
        "qwen35_dflash.ascend310p",
        "export-air",
        "--factory",
        factory,
        "--factory-config",
        str(factory_config),
        "--bundle-dir",
        str(bundle_dir),
    ]
    environment = os.environ.copy()
    existing_python_path = environment.get("PYTHONPATH", "")
    prefixes = [str(FRAMEWORK_PYTHON), str(ROOT)]
    if existing_python_path:
        prefixes.append(existing_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(prefixes)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONFAULTHANDLER"] = "1"
    if verbose_dynamo:
        environment["TORCHDYNAMO_VERBOSE"] = "1"
        environment["TORCH_LOGS"] = "+dynamo"

    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                errors="replace",
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
            returncode = process.wait()
    except BaseException as error:  # noqa: BLE001
        return {
            "status": "LAUNCH_ERROR",
            "command": command,
            "bundle_dir": str(bundle_dir),
            "log": str(log_path),
            "duration_seconds": time.monotonic() - started,
            "error": repr(error),
            "returncode": 125,
        }
    return {
        "status": "PASS" if returncode == 0 else "FAILED",
        "command": command,
        "bundle_dir": str(bundle_dir),
        "log": str(log_path),
        "duration_seconds": time.monotonic() - started,
        "returncode": returncode,
    }


def _collect_export_diagnostics(
    bundle_dir: Path,
    report_root: Path,
) -> list[dict[str, object]]:
    if not bundle_dir.exists():
        return []
    copied_bytes = 0
    records: list[dict[str, object]] = []
    destination_root = report_root / "export-diagnostics"
    for path in sorted(bundle_dir.rglob("*")):
        relative = path.relative_to(bundle_dir)
        if path.is_symlink():
            records.append({"path": relative.as_posix(), "kind": "symlink-skipped"})
            continue
        if not path.is_file():
            continue
        size = path.stat().st_size
        record: dict[str, object] = {
            "path": relative.as_posix(),
            "kind": "file",
            "size_bytes": size,
        }
        should_copy = (
            path.suffix.lower() in DIAGNOSTIC_SUFFIXES
            and size <= MAX_SINGLE_DIAGNOSTIC_BYTES
            and copied_bytes + size <= MAX_COPIED_DIAGNOSTIC_BYTES
        )
        if should_copy:
            destination = destination_root / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied_bytes += size
            record["copied"] = True
            record["sha256"] = _sha256(path)
        else:
            record["copied"] = False
        records.append(record)
    return records


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "collect source hashes, runtime identities, custom-op dispatch "
            "contracts, and one complete AIR export log without using Git"
        )
    )
    parser.add_argument("--factory-config", type=Path, required=True)
    parser.add_argument("--factory", default=DEFAULT_FACTORY)
    parser.add_argument("--bundle-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--skip-export",
        action="store_true",
        help="collect identities only; intended for host-side smoke tests",
    )
    parser.add_argument(
        "--no-dynamo-logs",
        action="store_true",
        help="do not set TORCHDYNAMO_VERBOSE=1 and TORCH_LOGS=+dynamo",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    factory_config = args.factory_config.expanduser().resolve()
    if not factory_config.is_file():
        raise FileNotFoundError(f"factory config is missing: {factory_config}")

    stamp = _utc_stamp()
    run_dir_value = os.environ.get("AI_RUN_DIR")
    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    elif run_dir_value:
        output_dir = Path(run_dir_value).expanduser().resolve() / "reports"
    else:
        raise RuntimeError("--output-dir or AI_RUN_DIR is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    report_root = output_dir / f"air-debug-{stamp}"
    if report_root.exists():
        raise FileExistsError(f"diagnostic directory already exists: {report_root}")
    report_root.mkdir(parents=True)

    raw_factory_config = json.loads(factory_config.read_text(encoding="utf-8"))
    _write_json(
        report_root / "factory-config.json",
        _redact_sensitive(raw_factory_config),
    )
    source_records = _source_snapshot(report_root)
    _write_json(report_root / "source-identity.json", source_records)
    _write_json(report_root / "python-stack.json", _python_stack())
    _write_json(report_root / "operator-dispatch.json", _operator_dispatch())
    _write_json(report_root / "ge-prototypes.json", _ge_prototype_preflight())
    _write_json(
        report_root / "environment.json",
        {
            "timestamp_utc": stamp,
            "cwd": str(Path.cwd()),
            "repository_root": str(ROOT),
            "platform": platform.platform(),
            "uname": list(platform.uname()),
            "environment": _export_environment(),
            "atc": _run_identity_command("atc", ["--version"]),
            "npu_smi": _run_identity_command("npu-smi", ["info"]),
        },
    )

    if args.skip_export:
        export_result: dict[str, object] = {
            "status": "SKIPPED",
            "returncode": 0,
        }
    else:
        if args.bundle_dir is not None:
            bundle_dir = args.bundle_dir.expanduser().resolve()
        elif run_dir_value:
            bundle_dir = (
                Path(run_dir_value).expanduser().resolve()
                / "artifacts"
                / f"air-debug-export-{stamp}"
            )
        else:
            raise RuntimeError("--bundle-dir or AI_RUN_DIR is required for export")
        export_result = _run_export(
            factory=args.factory,
            factory_config=factory_config,
            bundle_dir=bundle_dir,
            log_path=report_root / "export-air.full.log",
            verbose_dynamo=not args.no_dynamo_logs,
        )
        export_files = _collect_export_diagnostics(bundle_dir, report_root)
        _write_json(report_root / "export-files.json", export_files)

    summary = {
        "schema_version": 1,
        "status": "COLLECTED",
        "uses_git_metadata": False,
        "copies_model_weights": False,
        "source_identity": "source-identity.json plus source-snapshot/",
        "export": export_result,
    }
    _write_json(report_root / "summary.json", summary)

    archive = output_dir / f"{report_root.name}.tar.gz"
    with tarfile.open(archive, "w:gz") as stream:
        stream.add(report_root, arcname=report_root.name, recursive=True)
    print(
        json.dumps(
            {
                "status": "COLLECTED",
                "archive": str(archive),
                "report_dir": str(report_root),
                "export_returncode": int(export_result.get("returncode", 125)),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return int(export_result.get("returncode", 125))


if __name__ == "__main__":
    raise SystemExit(main())
