"""Simple NPU entry point for the embedded HIAI layout.

This command derives the HIAI source and DFlash loader from the colocated
``models`` package.  It intentionally exposes only the controls needed for a
V1 strict-greedy smoke or validation run.  The colocated source tree is
validated directly; no generated overlay report is needed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from .dflash_qwen_adapter_v1 import main as _adapter_main
from .internal_target_loader import (
    DECODE_CHUNK_SIZE_ENV,
    PREFILL_CHUNK_SIZE_ENV,
)


DEFAULT_TARGET_FACTORY = "models.internal_dflash_bridge:load_qwen35_target"
KV_CACHE_MAX_LEN_ENV = "DFLASH_HIAI_KV_CACHE_MAX_LEN"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run Qwen3.5-4B DFlash V1 with the original HIAI model in "
            "models/ and DFlash in models/dflash_v1/"
        )
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--draft-dir", required=True)
    parser.add_argument(
        "--target-factory",
        default=DEFAULT_TARGET_FACTORY,
        help=(
            "advanced override; default models.internal_dflash_bridge:"
            "load_qwen35_target reuses Qwen3_5ForCausalLMWrapper and builds "
            "fresh hybrid state for every target call"
        ),
    )
    parser.add_argument(
        "--reset-hook",
        help=(
            "advanced override for a custom target factory; the packaged "
            "internal_dflash_bridge does not need a reset hook"
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
    parser.add_argument("--max-draft-tokens", type=int, default=1)
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--kv-cache-max-len",
        type=int,
        required=True,
        help="same kv_cache_max_len used by the existing HIAI inference YAML",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--decode-chunk-size", type=int, default=1)
    parser.add_argument("--report")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not str(args.device).startswith("npu"):
        raise ValueError("run_npu requires --device npu or npu:N")
    if args.max_new_tokens < 2:
        raise ValueError("NPU DFlash smoke requires --max-new-tokens >= 2")
    if not 1 <= args.max_draft_tokens <= 16:
        raise ValueError("--max-draft-tokens must be between 1 and 16")
    for name, value in (
        ("--kv-cache-max-len", args.kv_cache_max_len),
        ("--prefill-chunk-size", args.prefill_chunk_size),
        ("--decode-chunk-size", args.decode_chunk_size),
    ):
        if value <= 0:
            raise ValueError(f"{name} must be positive")

    package_dir = Path(__file__).resolve().parent
    hiai_source = package_dir.parent / "modeling_qwen3_5_hiai_nd.py"
    if hiai_source.is_symlink() or not hiai_source.is_file():
        raise FileNotFoundError(
            "expected original HIAI source at models/modeling_qwen3_5_hiai_nd.py"
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
        "--max-draft-tokens",
        str(args.max_draft_tokens),
    ]
    if args.reset_hook is not None:
        adapter_args.extend(["--reset-hook", args.reset_hook])
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
