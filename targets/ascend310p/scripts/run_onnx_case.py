#!/usr/bin/env python3
"""Run a materialized MTP core case with ONNX Runtime CPU for export parity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
import torch

from qwen35_mtp.precision import tensor_error_metrics
from qwen35_mtp.weights import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--atol", type=float, default=0.01)
    args = parser.parse_args()
    model_path = args.model.expanduser().resolve()
    case_dir = args.case_dir.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    manifest = json.loads((case_dir / "manifest.json").read_text(encoding="utf-8"))
    inputs = {
        name: np.load(case_dir / f"{name}.npy", allow_pickle=False)
        for name in manifest["input_order"]
    }
    expected_files = {
        "mtp_hidden": "expected_mtp_hidden_fp16.npy",
        "present_key": "expected_present_key_fp16.npy",
        "present_value": "expected_present_value_fp16.npy",
    }
    started = time.perf_counter()
    session = ort.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )
    loaded = time.perf_counter()
    output_names = [item.name for item in session.get_outputs()]
    actual_values = session.run(output_names, inputs)
    finished = time.perf_counter()
    output_dir = output_path.parent / "onnxruntime_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    comparisons = {}
    passed = True
    for name, actual in zip(output_names, actual_values, strict=True):
        expected = np.load(case_dir / expected_files[name], allow_pickle=False)
        finite = bool(np.isfinite(actual).all())
        close = bool(
            finite
            and np.allclose(actual, expected, rtol=args.rtol, atol=args.atol)
        )
        actual_path = output_dir / f"{name}.npy"
        np.save(actual_path, actual, allow_pickle=False)
        comparisons[name] = {
            "passed": close,
            "finite": finite,
            "rtol": args.rtol,
            "atol": args.atol,
            "metrics": tensor_error_metrics(
                torch.from_numpy(expected), torch.from_numpy(actual)
            ),
            "actual_npy": {
                "file": str(actual_path),
                "sha256": sha256_file(actual_path),
            },
        }
        passed = passed and close
    report = {
        "schema_version": 1,
        "status": "PASS" if passed else "FAIL",
        "provider": session.get_providers(),
        "model": {"file": model_path.name, "sha256": sha256_file(model_path)},
        "case_manifest": {
            "file": str(case_dir / "manifest.json"),
            "sha256": sha256_file(case_dir / "manifest.json"),
        },
        "comparisons": comparisons,
        "timing_seconds": {
            "session_load": loaded - started,
            "inference": finished - loaded,
        },
        "claim_boundary": "ONNX Runtime CPU parity is not Ascend 310P execution.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
