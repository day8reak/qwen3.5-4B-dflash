"""Deterministic CPU contract check for the W8A8 QLinear emulator.

This command deliberately uses synthetic tensors. It proves that the local
CPU implementation follows the declared dynamic-per-token INT8 formula and
that its final FP16 result matches an independent INT64-accumulator oracle.
It does not claim parity with an NPU kernel or validate a deployment artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
from typing import Sequence

import torch
from torch import Tensor

from .w8a8_emulation import (
    ACTIVATION_QMAX,
    dynamic_quantize_per_token,
    emulate_w8a8_linear,
)


_DEFAULT_SEED = 20260823
_CASES = (
    {
        "name": "hidden_projection_in2560_per_channel",
        "rows": 4,
        "in_features": 2560,
        "out_features": 257,
        "scale_layout": "per_output_channel",
    },
    {
        "name": "mlp_down_projection_in9728_per_tensor",
        "rows": 4,
        "in_features": 9728,
        "out_features": 33,
        "scale_layout": "per_tensor",
    },
)


def _reference_dynamic_quantize(x: Tensor) -> tuple[Tensor, Tensor]:
    """Small independent spelling of the public dynamic-quant contract."""

    source = x.to(torch.float32)
    scale = source.abs().amax(dim=-1) / float(ACTIVATION_QMAX)
    safe_scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.clamp(
        torch.round(source / safe_scale.unsqueeze(-1)),
        -ACTIVATION_QMAX,
        ACTIVATION_QMAX,
    ).to(torch.int8)
    return quantized, scale


def _reference_output(
    quantized: Tensor,
    activation_scale: Tensor,
    weight: Tensor,
    weight_scale: Tensor,
) -> Tensor:
    """Use INT64 accumulation to independently check the optimized INT32 path."""

    accumulator = torch.matmul(
        quantized.to(torch.int64),
        weight.to(torch.int64),
    ).to(torch.float32)
    output = accumulator * weight_scale.reshape(1, -1)
    output = output * activation_scale.reshape(-1, 1)
    return output.to(torch.float16)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _run_case(specification: dict[str, object], *, seed: int) -> dict[str, object]:
    name = str(specification["name"])
    rows = int(specification["rows"])
    in_features = int(specification["in_features"])
    out_features = int(specification["out_features"])
    scale_layout = str(specification["scale_layout"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    activation = (
        torch.randn(
            (rows, in_features),
            generator=generator,
            dtype=torch.float32,
        )
        * 0.75
    ).to(torch.float16)
    # Exercise the zero-row policy explicitly. A second non-random row makes
    # the result stable even if a future RNG implementation changes.
    activation[0].zero_()
    activation[1].fill_(0.25)
    activation[1, ::2].mul_(-1)

    weight = torch.randint(
        -ACTIVATION_QMAX,
        ACTIVATION_QMAX + 1,
        (in_features, out_features),
        generator=generator,
        dtype=torch.int8,
    )
    if scale_layout == "per_output_channel":
        weight_scale = (
            torch.rand(out_features, generator=generator, dtype=torch.float32)
            * 0.004
            + 0.0001
        )
    elif scale_layout == "per_tensor":
        weight_scale = torch.tensor([0.00125], dtype=torch.float32)
    else:
        raise AssertionError(f"unknown scale layout: {scale_layout}")

    quantized, activation_scale = dynamic_quantize_per_token(activation)
    reference_quantized, reference_activation_scale = _reference_dynamic_quantize(
        activation
    )
    _require(
        torch.equal(quantized, reference_quantized),
        f"{name}: dynamic-quant INT8 values differ from the formula",
    )
    _require(
        torch.equal(activation_scale, reference_activation_scale),
        f"{name}: per-token scales differ from the formula",
    )

    candidate = emulate_w8a8_linear(
        activation,
        weight,
        weight_scale,
        output_dtype=torch.float16,
    )
    repeated = emulate_w8a8_linear(
        activation,
        weight,
        weight_scale,
        output_dtype=torch.float16,
    )
    reference = _reference_output(
        reference_quantized,
        reference_activation_scale,
        weight,
        weight_scale,
    )
    _require(
        torch.equal(candidate, reference),
        f"{name}: CPU QLinear output differs from the INT64 oracle",
    )
    _require(
        torch.equal(candidate, repeated),
        f"{name}: CPU QLinear output is not deterministic",
    )
    _require(
        bool(torch.isfinite(candidate).all()),
        f"{name}: CPU QLinear output contains non-finite values",
    )
    _require(
        int(torch.count_nonzero(candidate[0])) == 0,
        f"{name}: zero activation row did not produce an exact zero row",
    )

    maximum_accumulator = in_features * ACTIVATION_QMAX * ACTIVATION_QMAX
    _require(
        maximum_accumulator <= torch.iinfo(torch.int32).max,
        f"{name}: declared INT32 accumulator can overflow",
    )
    return {
        "name": name,
        "status": "PASS_BITWISE_INT64_ORACLE",
        "activation_shape": [rows, in_features],
        "weight_shape": [in_features, out_features],
        "weight_scale_layout": scale_layout,
        "activation_dtype": str(activation.dtype),
        "quantized_activation_dtype": str(quantized.dtype),
        "weight_dtype": str(weight.dtype),
        "weight_scale_dtype": str(weight_scale.dtype),
        "output_dtype": str(candidate.dtype),
        "dynamic_quant_formula_match": True,
        "output_bitwise_int64_oracle": True,
        "repeatable": True,
        "finite": True,
        "zero_row_exact": True,
        "maximum_possible_int32_accumulator": maximum_accumulator,
    }


def validate_cpu_w8a8_formula(*, seed: int = _DEFAULT_SEED) -> dict[str, object]:
    """Run the bounded CPU formula suite and return a JSON-safe report."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    cases = [
        _run_case(specification, seed=seed + index)
        for index, specification in enumerate(_CASES)
    ]
    capability = getattr(torch.backends.cpu, "get_cpu_capability", None)
    return {
        "status": "PASS_CPU_W8A8_FORMULA_CONTRACT",
        "scope": "synthetic_qlinear_formula_only",
        "seed": seed,
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": "cpu",
            "torch_int_mm_available": callable(getattr(torch, "_int_mm", None)),
            "cpu_capability": capability() if callable(capability) else None,
        },
        "formula": {
            "activation_quantization": "symmetric_dynamic_per_token_int8",
            "activation_scale": "max(abs(row))/127",
            "activation_rounding": "torch.round_then_clamp_-127_127",
            "weight_layout": "K_by_N",
            "accumulator": "int32_checked_against_int64_oracle",
            "dequantization_order": (
                "accumulator_times_weight_scale_times_pertoken_scale"
            ),
            "output_dtype": "torch.float16",
        },
        "cases": cases,
        "not_proven": [
            "deployment_quant_artifact_layout_or_coverage",
            "quantized_embedding_or_input_provider_semantics",
            "real_npu_dynamic_quant_rounding_or_quant_matmul_parity",
            "whole_target_token_or_feature_parity",
            "dflash_acceptance_rate_or_performance",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic synthetic CPU checks for the DFlash W8A8 "
            "QLinear formula; no model weights or NPU are required"
        )
    )
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument(
        "--report",
        help="optional new JSON file; the command refuses to overwrite it",
    )
    return parser


def _write_new_report(path_text: str, report: dict[str, object]) -> Path:
    raw = Path(path_text).expanduser()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError("--report must name a new file")
    destination = raw.resolve()
    if not destination.parent.is_dir():
        raise FileNotFoundError("--report parent directory does not exist")
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("temporary report path already exists")
    try:
        temporary.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = validate_cpu_w8a8_formula(seed=args.seed)
    if args.report is not None:
        _write_new_report(args.report, report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "validate_cpu_w8a8_formula"]
