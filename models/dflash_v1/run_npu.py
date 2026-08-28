"""Simple NPU entry point for the embedded HIAI rollback layout.

This command derives the HIAI source and DFlash loader from the colocated
``models`` package.  It intentionally exposes only the controls needed for a
strict-greedy persistent rollback smoke or validation run.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .dflash_config import DFLASH_MIN_BLOCK_SIZE, OFFICIAL_DFLASH_BLOCK_SIZE
from .run_rollback import (
    DEFAULT_NPU_TARGET_FACTORY,
    main as _adapter_main,
)
from .internal_target_loader import (
    DECODE_CHUNK_SIZE_ENV,
    PREFILL_CHUNK_SIZE_ENV,
)
from .target_quant import (
    ORIGINAL_QUANTIZER_SPEC,
    QUANT_MODE_DISABLED,
    QUANT_MODE_W8A8_DYNAMIC,
    TARGET_EMBEDDING_SCALE_PATH_ENV,
    TARGET_EMBEDDING_WEIGHT_PATH_ENV,
    TARGET_QUANT_CONFIG_ENV,
    TARGET_QUANT_MODE_ENV,
    TARGET_QUANT_WEIGHT_PATH_ENV,
    TargetQuantizationRequest,
    load_callback,
    load_original_quant_config,
    quantizer_callback_abi,
)


DEFAULT_TARGET_FACTORY = DEFAULT_NPU_TARGET_FACTORY
KV_CACHE_MAX_LEN_ENV = "DFLASH_HIAI_KV_CACHE_MAX_LEN"
ORIGINAL_QUANT_ENABLE = "enable"
ORIGINAL_QUANT_DISABLE = "disable"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3.5-4B DFlash with persistent HIAI state-bank rollback"
        )
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument(
        "--target-factory",
        default=DEFAULT_TARGET_FACTORY,
        help=(
            "advanced override; the default loads the separate rollback "
            "modeling through the deployed wrapper adapter"
        ),
    )
    parser.add_argument(
        "--reset-hook",
        help=(
            "unsupported compatibility option; rollback owns state through "
            "begin/verify/commit/abort"
        ),
    )
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-ids", help="comma-separated token IDs")
    prompt.add_argument("--prompt-json", help="JSON token list or input_ids object")
    prompt.add_argument("--prompt", help="UTF-8 prompt text")
    prompt.add_argument("--prompt-file", help="path to a UTF-8 prompt text file")
    parser.add_argument(
        "--prompt-mode",
        choices=("chat", "raw"),
        default="chat",
        help="chat applies the local Qwen chat template; raw tokenizes text directly",
    )
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="enable Qwen thinking in chat mode (default: enabled)",
    )
    parser.add_argument("--max-new-tokens", type=int, default=2)
    parser.add_argument(
        "--execution-mode",
        choices=("validate", "dflash"),
        default="validate",
        help=(
            "validate runs ordinary plus DFlash exact-match checking; dflash "
            "runs only the production DFlash session"
        ),
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=DFLASH_MIN_BLOCK_SIZE,
        help="total draft/verify rows, including one anchor (official range: 2..16)",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--kv-cache-max-len",
        type=int,
        required=True,
        help="same kv_cache_max_len used by the existing HIAI inference YAML",
    )
    parser.add_argument(
        "--prefill-chunk-size",
        type=int,
        default=64,
        help="compatibility setting; the current receiver contract requires 64",
    )
    parser.add_argument(
        "--decode-chunk-size",
        type=int,
        default=1,
        help="compatibility setting; prompt bootstrap/decode require 1",
    )
    parser.add_argument(
        "--config",
        help=(
            "original inference YAML containing quanted_pth, "
            "embedding_weight_path, and embedding_scale_path"
        ),
    )
    parser.add_argument(
        "--quant_mode",
        "--quant-mode",
        dest="quant_mode",
        choices=(ORIGINAL_QUANT_ENABLE, ORIGINAL_QUANT_DISABLE),
        default=ORIGINAL_QUANT_DISABLE,
        help=(
            "same switch as inference.py; enable uses built-in "
            "original quant_model and the three paths from --config"
        ),
    )
    parser.add_argument("--report")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def _configure_target_quantization(args: argparse.Namespace) -> None:
    """Mirror the original YAML + ``utils.quant_model`` route."""

    environment_names = (
        TARGET_QUANT_CONFIG_ENV,
        TARGET_QUANT_WEIGHT_PATH_ENV,
        TARGET_EMBEDDING_WEIGHT_PATH_ENV,
        TARGET_EMBEDDING_SCALE_PATH_ENV,
    )
    if args.quant_mode == ORIGINAL_QUANT_DISABLE:
        os.environ[TARGET_QUANT_MODE_ENV] = QUANT_MODE_DISABLED
        for name in environment_names:
            os.environ.pop(name, None)
        return

    if args.target_factory != DEFAULT_TARGET_FACTORY:
        raise ValueError(
            "rollback Target quantization requires the packaged factory "
            f"{DEFAULT_TARGET_FACTORY}; a custom factory would bypass the "
            "QLinear and state-transaction audits"
        )
    if args.config is None:
        raise ValueError(
            "--quant_mode enable requires the original inference --config YAML"
        )
    config = load_original_quant_config(args.config)
    os.environ[TARGET_QUANT_MODE_ENV] = QUANT_MODE_W8A8_DYNAMIC
    os.environ[TARGET_QUANT_CONFIG_ENV] = str(config.config_path)
    os.environ[TARGET_QUANT_WEIGHT_PATH_ENV] = str(config.quant_weight_path)
    os.environ[TARGET_EMBEDDING_WEIGHT_PATH_ENV] = str(
        config.embedding_weight_path
    )
    os.environ[TARGET_EMBEDDING_SCALE_PATH_ENV] = str(
        config.embedding_scale_path
    )

    request = TargetQuantizationRequest.from_environment()
    assert request.enabled
    protected_paths = (
        request.config_path,
        request.quant_weight_path,
        request.embedding_weight_path,
        request.embedding_scale_path,
    )
    assert all(path is not None for path in protected_paths)
    report = Path(args.report).expanduser() if args.report is not None else None
    if report is not None:
        if report.is_symlink():
            raise ValueError("--report must not be a symlink")
        resolved_report = report.resolve()
        for protected in protected_paths:
            assert protected is not None
            overlaps = resolved_report == protected
            if protected.is_dir():
                try:
                    resolved_report.relative_to(protected)
                except ValueError:
                    pass
                else:
                    overlaps = True
            if overlaps:
                raise ValueError(
                    "--report must not overwrite or be placed inside any "
                    "Target quantization data path"
                )
    quantizer, _ = load_callback(
        ORIGINAL_QUANTIZER_SPEC,
        label="packaged original utils.quant_model",
    )
    if quantizer_callback_abi(quantizer) != "simple":
        raise TypeError(
            "packaged quant_model must accept (model, quant_weight_path)"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not str(args.device).startswith("npu"):
        raise ValueError("run_npu requires --device npu or npu:N")
    if args.reset_hook is not None:
        raise ValueError("--reset-hook is not part of the rollback transaction")
    if args.max_new_tokens < 2:
        raise ValueError("NPU DFlash smoke requires --max-new-tokens >= 2")
    if not DFLASH_MIN_BLOCK_SIZE <= args.block_size <= OFFICIAL_DFLASH_BLOCK_SIZE:
        raise ValueError(
            "--block-size must be between "
            f"{DFLASH_MIN_BLOCK_SIZE} and {OFFICIAL_DFLASH_BLOCK_SIZE}"
        )
    for name, value in (
        ("--kv-cache-max-len", args.kv_cache_max_len),
        ("--prefill-chunk-size", args.prefill_chunk_size),
        ("--decode-chunk-size", args.decode_chunk_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    if args.prefill_chunk_size != 64 or args.decode_chunk_size != 1:
        raise ValueError(
            "rollback HIAI requires prefill-chunk-size=64 and "
            "decode-chunk-size=1"
        )

    _configure_target_quantization(args)

    package_dir = Path(__file__).resolve().parent
    hiai_source = (
        package_dir.parent / "modeling_qwen3_5_hiai_nd_dflash_rollback.py"
    )
    if hiai_source.is_symlink() or not hiai_source.is_file():
        raise FileNotFoundError(
            "expected rollback HIAI source at "
            "models/modeling_qwen3_5_hiai_nd_dflash_rollback.py"
        )
    os.environ[PREFILL_CHUNK_SIZE_ENV] = str(args.prefill_chunk_size)
    os.environ[DECODE_CHUNK_SIZE_ENV] = str(args.decode_chunk_size)
    os.environ[KV_CACHE_MAX_LEN_ENV] = str(args.kv_cache_max_len)

    adapter_args = [
        "--target-dir",
        args.target_dir,
        "--draft-dir",
        args.draft_dir,
        "--target-factory",
        args.target_factory,
        "--npu-layout",
        "embedded",
        "--device",
        args.device,
        "--dtype",
        "float16",
        "--eos-token-id",
        "248044",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--execution-mode",
        args.execution_mode,
        "--block-size",
        str(args.block_size),
    ]
    if args.prompt_ids is not None:
        adapter_args.extend(["--prompt-ids", args.prompt_ids])
    elif args.prompt_json is not None:
        adapter_args.extend(["--prompt-json", args.prompt_json])
    elif args.prompt is not None:
        adapter_args.extend(["--prompt", args.prompt])
    else:
        adapter_args.extend(["--prompt-file", args.prompt_file])
    adapter_args.extend(["--prompt-mode", args.prompt_mode])
    adapter_args.append(
        "--enable-thinking" if args.enable_thinking else "--no-enable-thinking"
    )
    if args.report is not None:
        adapter_args.extend(["--report", args.report])
    adapter_args.append("--progress" if args.progress else "--no-progress")
    return _adapter_main(adapter_args)


if __name__ == "__main__":
    raise SystemExit(main())
