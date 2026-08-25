#!/usr/bin/env python3
"""Validate a fixed-gear MTP ONNX candidate and report its portable ABI."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import onnx
from onnx import TensorProto

from qwen35_mtp.weights import sha256_file


def _value_info(value) -> dict:
    tensor_type = value.type.tensor_type
    dimensions = []
    for dimension in tensor_type.shape.dim:
        dimensions.append(
            dimension.dim_value if dimension.HasField("dim_value") else dimension.dim_param
        )
    return {
        "name": value.name,
        "dtype": TensorProto.DataType.Name(tensor_type.elem_type),
        "shape": dimensions,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model = onnx.load(str(input_path), load_external_data=True)
    onnx.checker.check_model(model, full_check=True)
    floating_types = {
        TensorProto.FLOAT,
        TensorProto.FLOAT16,
        TensorProto.BFLOAT16,
        TensorProto.DOUBLE,
    }
    floating_initializers = [
        initializer
        for initializer in model.graph.initializer
        if initializer.data_type in floating_types
    ]
    non_fp16_initializers = [
        {
            "name": initializer.name,
            "dtype": TensorProto.DataType.Name(initializer.data_type),
        }
        for initializer in floating_initializers
        if initializer.data_type != TensorProto.FLOAT16
    ]
    expected_fp16_matrices = [
        initializer
        for initializer in floating_initializers
        if initializer.data_type == TensorProto.FLOAT16 and len(initializer.dims) == 2
    ]
    expected_fp32_norm_vectors = [
        initializer
        for initializer in floating_initializers
        if initializer.data_type == TensorProto.FLOAT
        and list(initializer.dims) in ([2560], [256])
    ]
    expected_profile_names = {
        initializer.name
        for initializer in [*expected_fp16_matrices, *expected_fp32_norm_vectors]
    }
    unexpected_precision_initializers = [
        {
            "name": initializer.name,
            "dtype": TensorProto.DataType.Name(initializer.data_type),
            "shape": list(initializer.dims),
        }
        for initializer in floating_initializers
        if initializer.name not in expected_profile_names
    ]
    inputs = [_value_info(value) for value in model.graph.input]
    outputs = [_value_info(value) for value in model.graph.output]
    expected_float_inputs = {"inputs_embeds", "hidden_sources", "past_key", "past_value"}
    input_types = {item["name"]: item["dtype"] for item in inputs}
    output_types = {item["name"]: item["dtype"] for item in outputs}
    errors = []
    if any(input_types.get(name) != "FLOAT16" for name in expected_float_inputs):
        errors.append("one or more floating ABI inputs are not FLOAT16")
    if input_types.get("position_ids") != "INT64":
        errors.append("position_ids is not INT64")
    if any(dtype != "FLOAT16" for dtype in output_types.values()):
        errors.append("one or more ABI outputs are not FLOAT16")
    fp32_norm_shapes = Counter(tuple(initializer.dims) for initializer in expected_fp32_norm_vectors)
    if len(expected_fp16_matrices) != 8:
        errors.append("the graph does not contain exactly eight FP16 matrix weights")
    if fp32_norm_shapes != Counter({(2560,): 5, (256,): 2}):
        errors.append("the FP32 RMSNorm vector profile is not 5x[2560] plus 2x[256]")
    if unexpected_precision_initializers:
        errors.append("one or more floating initializers are outside the frozen precision profile")
    report = {
        "schema_version": 1,
        "status": "PASS" if not errors else "FAIL",
        "artifact": input_path.name,
        "artifact_sha256": sha256_file(input_path),
        "artifact_bytes": input_path.stat().st_size,
        "opset_imports": [
            {"domain": item.domain or "ai.onnx", "version": item.version}
            for item in model.opset_import
        ],
        "inputs": inputs,
        "outputs": outputs,
        "initializer_count": len(model.graph.initializer),
        "floating_initializer_count": len(floating_initializers),
        "floating_initializer_dtype_counts": dict(
            Counter(
                TensorProto.DataType.Name(initializer.data_type)
                for initializer in floating_initializers
            )
        ),
        "non_fp16_floating_initializers": non_fp16_initializers,
        "precision_profile": {
            "fp16_matrix_count": len(expected_fp16_matrices),
            "fp32_norm_vector_shapes": {
                "2560": fp32_norm_shapes[(2560,)],
                "256": fp32_norm_shapes[(256,)],
            },
            "reason": "MatMul/attention/SwiGLU weights are FP16; seven RMSNorm vectors are constant-folded to FP32 for the approved high-precision norm path.",
        },
        "unexpected_precision_initializers": unexpected_precision_initializers,
        "operator_counts": dict(Counter(node.op_type for node in model.graph.node)),
        "checker_full_check": True,
        "errors": errors,
        "claim_boundary": "ONNX structural validation is not ATC, CAModel, or device execution.",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if errors:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
