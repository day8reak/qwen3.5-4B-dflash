#!/usr/bin/env python3
"""Compare saved Draft KV bytes by B,H,S,D row regions; no NPU execution."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def metrics(reference: np.ndarray, actual: np.ndarray, begin: int, end: int) -> dict:
    """Bitwise FP16 comparison; positions are sequence indices, not flat offsets."""
    if (reference.shape != actual.shape or reference.ndim != 4
            or reference.dtype != np.dtype("<f2") or actual.dtype != reference.dtype
            or not 0 <= begin <= end <= reference.shape[2]):
        raise ValueError("expected equal FP16 B,H,S,D arrays and valid row bounds")
    ref, got = reference[:, :, begin:end, :], actual[:, :, begin:end, :]
    ref_bits, got_bits = ref.view(np.uint16), got.view(np.uint16)
    changed = ref_bits != got_bits
    finite = np.isfinite(ref) & np.isfinite(got)
    with np.errstate(invalid="ignore"):
        error = np.abs(got.astype(np.float64) - ref.astype(np.float64))[finite]
    # Map negative/positive half encodings into monotonic integer order.
    # Signed zeros have different bits but numeric distance/ULP distance zero.
    def ordered(bits):
        bits = bits.astype(np.int32)
        return np.where(bits & 0x8000, 0x8000 - (bits & 0x7fff), 0x8000 + bits)
    ulps = np.abs(ordered(got_bits) - ordered(ref_bits))[finite]
    result = {
        "row_begin": begin, "row_end": end, "elements": ref.size,
        "changed_elements": int(changed.sum()),
        "reference_nonfinite_elements": int((~np.isfinite(ref)).sum()),
        "actual_nonfinite_elements": int((~np.isfinite(got)).sum()),
        "max_abs_finite": float(error.max()) if error.size else None,
        "mean_abs_finite": float(error.mean()) if error.size else None,
        "max_fp16_ulp_finite": int(ulps.max()) if ulps.size else None,
        "first_difference": None,
    }
    different = np.argwhere(changed)
    if different.size:
        local = tuple(different[0])
        coordinate = [int(i) for i in local]
        coordinate[2] += begin
        result["first_difference"] = {
            "coordinate_bhsd": coordinate,
            "reference": float(ref[local]) if np.isfinite(ref[local]) else None,
            "actual": float(got[local]) if np.isfinite(got[local]) else None,
            "reference_bits": f"0x{int(ref_bits[local]):04x}",
            "actual_bits": f"0x{int(got_bits[local]):04x}",
        }
    return result


def analyze(report_path: Path) -> dict:
    report_path = report_path.expanduser().resolve()
    report = json.loads(report_path.read_text())
    audit = report.get("kv_output_audit")
    if not isinstance(audit, dict) or audit.get("version") != 1:
        raise ValueError("this report has no KV output snapshots; rebuild the C++ runner and replay existing OMs")
    root = Path(report["trace"]).resolve().parent
    if root.parent != report_path.parent:
        raise ValueError("replay artifacts must be beside the report")
    loaded = {}

    def load(relative: str, digest: str, shape=None):
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("replay artifact escapes output directory")
        key = (str(path), digest)
        if key not in loaded:
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError(f"replay artifact hash mismatch: {path}")
            loaded[key] = raw
        raw = loaded[key]
        if shape is None:
            return raw
        if len(shape) != 4 or any(type(d) is not int or d <= 0 for d in shape):
            raise ValueError("invalid KV shape")
        if len(raw) != int(np.prod(shape)) * 2:
            raise ValueError("KV snapshot byte size differs from ABI")
        return np.frombuffer(raw, dtype="<f2").reshape(shape)

    start_bytes = load("inputs/start_position.bin", report["snapshot_sha256"]["start_position"])
    if len(start_bytes) != 8:
        raise ValueError("invalid start_position snapshot")
    start = int(np.frombuffer(start_bytes, dtype="<i8")[0])
    references, preserved = {}, {}
    bounds = audit["row_regions"]
    for name, spec in audit["abi"].items():
        if spec["dtype"] != "float16":
            raise ValueError("KV analysis expects float16")
        reference = load(f"outputs/reference/{name}.bin",
                         report["reference_output_sha256"][name], spec["shape"])
        references[name] = reference
        capacity = reference.shape[2]
        if (not 0 <= start <= capacity - 64
                or set(bounds) != {"valid_prefix", "written_padding", "untouched_tail"}
                or any(len(b) != 2 or any(type(x) is not int for x in b)
                       or not 0 <= b[0] <= b[1] <= capacity for b in bounds.values())
                or bounds["valid_prefix"][0] != 0
                or not start < bounds["valid_prefix"][1] <= start + 64
                or bounds["written_padding"] != [bounds["valid_prefix"][1], start + 64]
                or bounds["untouched_tail"] != [start + 64, capacity]):
            raise ValueError("invalid KV row regions")
        for region, (begin, end) in bounds.items():
            digest = hashlib.sha256(reference[:, :, begin:end, :].tobytes(order="C")).hexdigest()
            if digest != audit["reference_region_sha256"][name][region]:
                raise ValueError(f"KV region hash mismatch: {name}/{region}")
        input_value = load(f"inputs/{name}.bin", report["snapshot_sha256"][name], spec["shape"])
        # Scatter replaces [start, start+64). Outside that interval the
        # functional graph must preserve input bytes, including on iteration 0.
        preserved[name] = {
            "old_prefix": metrics(input_value, reference, 0, start),
            "untouched_tail": metrics(input_value, reference, *bounds["untouched_tail"]),
        }

    examples = []
    for record in audit["saved_differences"]:
        name, region = record["tensor"], record["region"]
        actual = load(record["path"], record["sha256"], audit["abi"][name]["shape"])
        example = dict(record)
        example["comparison"] = metrics(references[name], actual, *bounds[region])
        input_value = load(f"inputs/{name}.bin", report["snapshot_sha256"][name], audit["abi"][name]["shape"])
        example["outside_write_vs_input"] = {
            "old_prefix": metrics(input_value, actual, 0, start),
            "untouched_tail": metrics(input_value, actual, *bounds["untouched_tail"]),
        }
        examples.append(example)
    return {
        "schema_version": 1, "status": "ANALYZED_SAVED_BYTES",
        "report": str(report_path), "fake_acl": report["fake_acl"],
        "formal_latency_evidence": False, "layout": "B,H,S,D",
        "row_regions": bounds,
        "phase_counts": [
            {"phase": p["phase"], "valid_kv_mismatch_iterations": p["valid_kv_mismatch_iterations"],
             "tensor_regions": p["kv_region_mismatch_iterations"]}
            for p in report["phases"]
        ],
        "reference_outside_write_vs_input": preserved,
        "examples": examples,
        "note": "Counts cover all replay calls; numerical examples are the first change per tensor/region, "
                "not the largest error across the run. Max/mean/ULP metrics exclude nonfinite pairs. "
                "Context KV changes narrow the dataflow but do not identify a particular kernel.",
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("report", type=Path)
    p.add_argument("--output", type=Path, help="New output JSON, otherwise print to stdout")
    args = p.parse_args(argv)
    try:
        payload = json.dumps(analyze(args.report), ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(payload)
        else:
            print(payload, end="")
    except (OSError, ValueError, KeyError) as error:
        p.exit(2, f"analyze_draft_replay: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
