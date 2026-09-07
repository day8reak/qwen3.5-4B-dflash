#!/usr/bin/env python3
"""Query one hash-locked OM's actual ABI without executing inference.

Uses the active libascendcl directly; no torch, pyACL, tokenizer, checkpoint
load, AIR export, or C++ rebuild is required. Scalar and unknown descriptors
are evidence, not errors to hide with the generation runner's ABI checks.
"""

from __future__ import annotations

import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import platform
from typing import Any


DYNAMIC_CONTROL = b"ascend_mbatch_shape_data"


class IODims(C.Structure):
    _fields_ = [
        ("name", C.c_char * 128),
        ("dimCount", C.c_size_t),
        ("dims", C.c_int64 * 128),
    ]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_model(manifest_path: Path, role: str) -> tuple[Path, dict[str, Any]]:
    manifest_path = manifest_path.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
        raise ValueError("expected a deployment-manifest.json for the OM bundle")
    graphs = [g for g in manifest.get("graphs", []) if g.get("name") == role]
    if len(graphs) != 1:
        raise ValueError(f"expected exactly one graph named {role!r}")
    graph = graphs[0]
    model_path = (manifest_path.parent / graph["om"]["path"]).resolve()
    if manifest_path.parent not in model_path.parents:
        raise ValueError("OM path escapes the deployment bundle")
    record = graph["om"]
    if model_path.stat().st_size != record["bytes"]:
        raise ValueError("OM size differs from the deployment manifest")
    print(f"Verifying {role} OM SHA256...", flush=True)
    if sha256_file(model_path) != record["sha256"]:
        raise ValueError("OM SHA256 differs from the deployment manifest")
    return model_path, {
        "deployment_manifest": str(manifest_path),
        "deployment_manifest_sha256": sha256_file(manifest_path),
        "role": role,
        "om": dict(record),
        "declared_dynamic": graph.get("dynamic"),
        "declared_input_dim_gears": graph.get("input_dim_gears"),
        "logical_input_names": graph.get("input_names", []),
        "logical_output_names": graph.get("output_names", []),
        "atc_command": graph.get("atc_command", []),
    }


def bind_acl(library: str) -> Any:
    acl = C.CDLL(library)
    ptr = C.c_void_p
    signatures = {
        "aclInit": ([C.c_char_p], C.c_int),
        "aclFinalize": ([], C.c_int),
        "aclrtSetDevice": ([C.c_int32], C.c_int),
        "aclrtResetDevice": ([C.c_int32], C.c_int),
        "aclrtCreateContext": ([C.POINTER(ptr), C.c_int32], C.c_int),
        "aclrtDestroyContext": ([ptr], C.c_int),
        "aclmdlLoadFromFile": ([C.c_char_p, C.POINTER(C.c_uint32)], C.c_int),
        "aclmdlUnload": ([C.c_uint32], C.c_int),
        "aclmdlCreateDesc": ([], ptr),
        "aclmdlDestroyDesc": ([ptr], C.c_int),
        "aclmdlGetDesc": ([ptr, C.c_uint32], C.c_int),
        "aclmdlGetNumInputs": ([ptr], C.c_size_t),
        "aclmdlGetNumOutputs": ([ptr], C.c_size_t),
        "aclmdlGetInputIndexByName": (
            [ptr, C.c_char_p, C.POINTER(C.c_size_t)], C.c_int
        ),
        "aclmdlGetInputDynamicGearCount": (
            [ptr, C.c_size_t, C.POINTER(C.c_size_t)], C.c_int
        ),
        "aclmdlGetInputDynamicDims": (
            [ptr, C.c_size_t, C.POINTER(IODims), C.c_size_t], C.c_int
        ),
    }
    for kind in ("Input", "Output"):
        signatures[f"aclmdlGet{kind}Dims"] = (
            [ptr, C.c_size_t, C.POINTER(IODims)], C.c_int
        )
        signatures[f"aclmdlGet{kind}DataType"] = ([ptr, C.c_size_t], C.c_int)
        signatures[f"aclmdlGet{kind}SizeByIndex"] = (
            [ptr, C.c_size_t], C.c_size_t
        )
    for name, (arguments, result) in signatures.items():
        function = getattr(acl, name)
        function.argtypes = arguments
        function.restype = result
    return acl


def check(code: int, operation: str) -> None:
    if code != 0:
        raise RuntimeError(f"{operation} failed with ACL error {code}")


def dims_record(dims: IODims) -> dict[str, Any]:
    rank = int(dims.dimCount)
    return {
        "name": bytes(dims.name).decode("utf-8", errors="replace"),
        "dim_count": rank,
        "shape": [int(dims.dims[i]) for i in range(min(rank, 128))],
        "rank_within_acl_capacity": rank <= 128,
    }


def describe_io(acl: Any, description: Any, kind: str) -> list[dict[str, Any]]:
    count = getattr(acl, f"aclmdlGetNum{kind}s")(description)
    records = []
    for index in range(count):
        dims = IODims()
        code = getattr(acl, f"aclmdlGet{kind}Dims")(description, index, C.byref(dims))
        record = {
            "index": index,
            "dims_status": code,
            "dtype": getattr(acl, f"aclmdlGet{kind}DataType")(description, index),
            "bytes": getattr(acl, f"aclmdlGet{kind}SizeByIndex")(description, index),
        }
        if code == 0:
            record.update(dims_record(dims))
        records.append(record)
    return records


def describe_dynamic(acl: Any, description: Any) -> dict[str, Any]:
    index = C.c_size_t()
    index_status = acl.aclmdlGetInputIndexByName(
        description, DYNAMIC_CONTROL, C.byref(index)
    )
    result: dict[str, Any] = {
        "control_name": DYNAMIC_CONTROL.decode(),
        "control_lookup_status": index_status,
        "control_index": index.value if index_status == 0 else None,
    }
    count = C.c_size_t()
    all_inputs = C.c_size_t(-1).value
    code = acl.aclmdlGetInputDynamicGearCount(description, all_inputs, C.byref(count))
    result["gear_count_status"] = code
    result["gear_count"] = count.value if code == 0 else None
    if code == 0 and 0 < count.value <= 100:
        gears = (IODims * count.value)()
        gear_status = acl.aclmdlGetInputDynamicDims(
            description, all_inputs, gears, count.value
        )
        result["gears_status"] = gear_status
        if gear_status == 0:
            result["gears"] = [dims_record(gear) for gear in gears]
    return result


def inspect_model(
    acl: Any, model: Path, device_id: int, report: dict[str, Any]
) -> None:
    initialized = False
    device_set = False
    context = C.c_void_p()
    model_id = C.c_uint32()
    loaded = False
    description = None
    try:
        check(acl.aclInit(None), "aclInit")
        initialized = True
        check(acl.aclrtSetDevice(device_id), "aclrtSetDevice")
        device_set = True
        check(acl.aclrtCreateContext(C.byref(context), device_id), "aclrtCreateContext")
        print("Loading only the selected OM for metadata inspection...", flush=True)
        check(
            acl.aclmdlLoadFromFile(os.fsencode(model), C.byref(model_id)),
            "aclmdlLoadFromFile",
        )
        loaded = True
        description = acl.aclmdlCreateDesc()
        if not description:
            raise RuntimeError("aclmdlCreateDesc returned null")
        check(acl.aclmdlGetDesc(description, model_id.value), "aclmdlGetDesc")
        report["inputs"] = describe_io(acl, description, "Input")
        report["outputs"] = describe_io(acl, description, "Output")
        report["dynamic_metadata"] = describe_dynamic(acl, description)
        report["status"] = "OBSERVED"
        report["model_id"] = model_id.value
    finally:
        cleanup = []
        if description:
            cleanup.append(("aclmdlDestroyDesc", acl.aclmdlDestroyDesc(description)))
        if loaded:
            cleanup.append(("aclmdlUnload", acl.aclmdlUnload(model_id.value)))
        if context.value:
            cleanup.append(("aclrtDestroyContext", acl.aclrtDestroyContext(context)))
        if device_set:
            cleanup.append(("aclrtResetDevice", acl.aclrtResetDevice(device_id)))
        if initialized:
            cleanup.append(("aclFinalize", acl.aclFinalize()))
        report["cleanup"] = [{"operation": op, "status": code} for op, code in cleanup]
        if any(code != 0 for _, code in cleanup):
            report["status"] = "FAILED"
            report["cleanup_failed"] = True


def output_path(value: Path) -> Path:
    run = os.environ.get("AI_RUN_DIR")
    if not run:
        raise RuntimeError("an active AI_RUN_DIR is required")
    path = value.resolve()
    if Path(run).resolve() not in path.parents:
        raise ValueError("output must be below AI_RUN_DIR")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deployment-manifest", type=Path, required=True)
    parser.add_argument("--role", default="fused-speculative-step")
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", default="libascendcl.so")
    args = parser.parse_args()
    destination = output_path(args.output)
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "FAILED",
        "device_id": args.device_id,
        "host_architecture": platform.machine(),
        "library": args.library,
        "claim_boundary": "Loaded OM metadata only; no inference or accuracy validation.",
    }
    try:
        model, identity = resolve_model(args.deployment_manifest, args.role)
        report.update(identity)
        inspect_model(bind_acl(args.library), model, args.device_id, report)
    except Exception as error:
        report["status"] = "FAILED"
        report["error"] = f"{type(error).__name__}: {error}"
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    with destination.open("x", encoding="utf-8") as stream:
        stream.write(payload)
    print(payload, end="", flush=True)
    print(f"Report: {destination}", flush=True)
    return 0 if report["status"] == "OBSERVED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
