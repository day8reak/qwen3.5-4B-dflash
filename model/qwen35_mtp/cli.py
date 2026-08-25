"""Command-line entry points for audit, ordinary, MTP, and paired comparison."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch

from .backends import (
    TorchMTPDraftBackend,
    TransformersMainBackend,
    load_external_draft_backend,
    load_external_main_backend,
)
from .benchmark import BenchmarkConfig, run_benchmark
from .generation import assert_exact_match, ordinary_generate, speculative_generate
from .mtp import Qwen35MTPDrafter
from .ops import ModuleMtpOps, TorchMtpOps
from .weights import audit_checkpoint


def _dtype(name: str) -> torch.dtype:
    return {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _write_json(payload: dict[str, Any], output: Path | None) -> None:
    rendered = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if output is None:
        sys.stdout.write(rendered)
        return
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered, encoding="utf-8")
    print(output)


def _tokenize(
    model_dir: Path,
    prompt: str | None,
    prompt_token_ids: str | None,
    chat: bool,
):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir), local_files_only=True, trust_remote_code=False
    )
    if prompt_token_ids is not None:
        token_ids = [int(item) for item in prompt_token_ids.split(",")]
        if not token_ids:
            raise ValueError("prompt-token-ids must not be empty")
    elif chat:
        assert prompt is not None
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
        )
        token_ids = encoded[0].tolist()
    else:
        assert prompt is not None
        token_ids = tokenizer.encode(prompt, add_special_tokens=True)
    eos = tokenizer.eos_token_id
    eos_ids = [] if eos is None else ([eos] if isinstance(eos, int) else list(eos))
    return tokenizer, token_ids, eos_ids


def _ops_backend(name: str, allow_fallback: bool):
    if name == "torch":
        return TorchMtpOps()
    return ModuleMtpOps.from_name(name, strict=not allow_fallback)


def _main_backend(args, ops):
    if args.main_backend == "torch":
        return TransformersMainBackend.from_pretrained(
            args.model_dir,
            device=args.device,
            dtype=_dtype(args.dtype),
            ops=ops,
        )
    return load_external_main_backend(
        args.main_backend,
        model_dir=args.model_dir,
        options={"device": args.device, "dtype": args.dtype},
    )


def _draft_backend(args, main, ops):
    if args.draft_backend != "torch":
        return load_external_draft_backend(
            args.draft_backend,
            model_dir=args.model_dir,
            options={"device": args.device, "dtype": args.dtype},
        )
    if not isinstance(main, TransformersMainBackend):
        raise ValueError(
            "the PyTorch draft backend needs the tied embedding from the PyTorch main; "
            "provide an external draft backend with an external main"
        )
    drafter = Qwen35MTPDrafter.from_pretrained(
        args.model_dir,
        embedding=main.embedding,
        ops=ops,
        device=args.device,
        dtype=_dtype(args.dtype),
    )
    return TorchMTPDraftBackend(drafter)


def _result_payload(result, tokenizer) -> dict[str, Any]:
    payload = result.to_dict()
    payload["generated_text"] = tokenizer.decode(
        result.generated_token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return payload


def _require_run_output(output: Path) -> Path:
    run_dir_value = os.environ.get("AI_RUN_DIR")
    if not run_dir_value:
        raise RuntimeError(
            "benchmark output requires an active workspace session (AI_RUN_DIR is unset)"
        )
    run_dir = Path(run_dir_value).expanduser().resolve()
    resolved = output.expanduser().resolve()
    try:
        resolved.relative_to(run_dir)
    except ValueError as error:
        raise ValueError(
            f"benchmark output must be below AI_RUN_DIR={run_dir}, got {resolved}"
        ) from error
    return resolved


def _benchmark_synchronizer(args, main, draft):
    if _is_cpu_device(args.device) and not args.allow_cpu_simulation:
        raise RuntimeError(
            "CPU benchmark is simulation only; pass --allow-cpu-simulation explicitly"
        )
    hooks = []
    seen = set()
    for role, backend in (("main", main), ("draft", draft)):
        if backend is None or id(backend) in seen:
            continue
        seen.add(id(backend))
        method = getattr(backend, "synchronize", None)
        if callable(method):
            hooks.append((role, method))
    if hooks:
        def synchronize_backends():
            for _role, method in hooks:
                method()

        source = "backend:" + ",".join(role for role, _method in hooks)
        return synchronize_backends, source

    if _is_cpu_device(args.device):
        return (lambda: None), "cpu-synchronous-simulation"

    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError(
            "target benchmark needs backend.synchronize() or an installed torch_npu"
        ) from error
    npu = getattr(torch_npu, "npu", None)
    synchronize = getattr(npu, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError("torch_npu.npu.synchronize is unavailable")
    return synchronize, "torch_npu.npu.synchronize"


def _is_cpu_device(device: str) -> bool:
    return device.lower().split(":", 1)[0] == "cpu"


def _backend_benchmark_metadata(backend):
    if backend is None:
        return None
    method = getattr(backend, "benchmark_metadata", None)
    if not callable(method):
        return {}
    metadata = method()
    if not isinstance(metadata, dict):
        raise TypeError("backend benchmark_metadata() must return a dict")
    return metadata


def _benchmark_range_factory(enabled: bool):
    if not enabled:
        return None
    try:
        import torch_npu
    except ImportError as error:
        raise RuntimeError("--enable-mstx requires torch_npu") from error
    npu = getattr(torch_npu, "npu", None)
    mstx = getattr(npu, "mstx", None)
    current_stream = getattr(npu, "current_stream", None)
    if mstx is None or not callable(current_stream):
        raise RuntimeError("torch_npu MSTX APIs are unavailable")
    stream = current_stream()

    @contextmanager
    def marked_range(label: str):
        range_id = mstx.range_start(label, stream)
        if not range_id:
            raise RuntimeError(f"MSTX range_start failed for {label!r}")
        try:
            yield
        finally:
            mstx.range_end(range_id)

    return marked_range


def run_benchmark_command(args) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    model_dir = args.model_dir.expanduser().resolve()
    if args.prompt_token_ids is not None and args.chat:
        raise ValueError("--chat cannot be combined with --prompt-token-ids")
    tokenizer, prompt_ids, eos_ids = _tokenize(
        model_dir, args.prompt, args.prompt_token_ids, args.chat
    )
    operations = _ops_backend(args.ops_backend, allow_fallback=False)
    main = _main_backend(args, operations)
    draft = None
    if args.mode == "mtp":
        draft = _draft_backend(args, main, operations)
    synchronize, synchronization_source = _benchmark_synchronizer(
        args, main, draft
    )
    payload = run_benchmark(
        main,
        prompt_ids,
        draft=draft,
        config=BenchmarkConfig(
            mode=args.mode,
            warmup=args.warmup,
            repetitions=args.repetitions,
            max_new_tokens=args.max_new_tokens,
            max_draft_tokens=args.max_draft_tokens,
        ),
        eos_token_ids=eos_ids,
        synchronize=synchronize,
        synchronization_source=synchronization_source,
        range_factory=_benchmark_range_factory(args.enable_mstx),
    )
    payload["target"] = {
        "device": args.device,
        "dtype": args.dtype,
        "profile_id": os.environ.get("AI_TARGET_PROFILE_ID"),
        "operator_backend": args.ops_backend,
        "allow_op_fallback": False,
        "strict_custom_op_dispatch": args.ops_backend != "torch",
        "cpu_simulation": _is_cpu_device(args.device),
        "mstx_enabled": bool(args.enable_mstx),
        "backend_metadata": {
            "main": _backend_benchmark_metadata(main),
            "draft": _backend_benchmark_metadata(draft),
        },
    }
    payload["software"] = {"torch": torch.__version__}
    if not _is_cpu_device(args.device):
        try:
            import torch_npu
        except ImportError:
            pass
        else:
            payload["software"]["torch_npu"] = getattr(
                torch_npu, "__version__", "unknown"
            )
    payload["summary"]["generated_text"] = tokenizer.decode(
        payload["summary"]["stable_generated_token_ids"],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    payload["claim_boundary"] = (
        "CPU timing is simulation evidence only and is not a target-device result."
        if _is_cpu_device(args.device)
        else "Target timing observation; promotion still requires frozen accuracy gates and 10/10 stable device runs."
    )
    return payload


def run_generation(args) -> dict[str, Any]:
    torch.set_num_threads(args.threads)
    model_dir = args.model_dir.expanduser().resolve()
    if args.prompt_token_ids is not None and args.chat:
        raise ValueError("--chat cannot be combined with --prompt-token-ids")
    tokenizer, prompt_ids, eos_ids = _tokenize(
        model_dir, args.prompt, args.prompt_token_ids, args.chat
    )
    ops = _ops_backend(args.ops_backend, args.allow_op_fallback)
    main = _main_backend(args, ops)

    if args.command == "ordinary":
        result = ordinary_generate(
            main,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            eos_token_ids=eos_ids,
        )
        return _result_payload(result, tokenizer)

    draft = _draft_backend(args, main, ops)
    if args.command == "mtp":
        result = speculative_generate(
            main,
            draft,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            max_draft_tokens=args.max_draft_tokens,
            eos_token_ids=eos_ids,
        )
        return _result_payload(result, tokenizer)

    ordinary = ordinary_generate(
        main,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        eos_token_ids=eos_ids,
    )
    mtp = speculative_generate(
        main,
        draft,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        eos_token_ids=eos_ids,
    )
    assert_exact_match(ordinary, mtp)
    return {
        "status": "PASS",
        "criterion": "ordinary and MTP greedy token IDs are exactly identical",
        "ordinary": _result_payload(ordinary, tokenizer),
        "mtp": _result_payload(mtp, tokenizer),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Qwen3.5-4B official MTP accuracy-first reference"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit", help="audit official config and MTP tensors")
    audit.add_argument("--model-dir", type=Path, required=True)
    audit.add_argument("--verify-file-hashes", action="store_true")
    audit.add_argument("--output", type=Path)

    for command in ("ordinary", "mtp", "compare"):
        item = subparsers.add_parser(command)
        item.add_argument("--model-dir", type=Path, required=True)
        prompts = item.add_mutually_exclusive_group(required=True)
        prompts.add_argument("--prompt")
        prompts.add_argument(
            "--prompt-token-ids",
            help="comma-separated committed prompt token IDs for deterministic probes",
        )
        item.add_argument("--chat", action="store_true")
        item.add_argument("--max-new-tokens", type=int, default=8)
        item.add_argument("--main-backend", default="torch")
        item.add_argument("--ops-backend", default="torch")
        item.add_argument("--allow-op-fallback", action="store_true")
        item.add_argument("--device", default="cpu")
        item.add_argument(
            "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
        )
        item.add_argument("--threads", type=int, default=max(1, torch.get_num_threads()))
        item.add_argument("--output", type=Path)
        if command in {"mtp", "compare"}:
            item.add_argument("--draft-backend", default="torch")
            item.add_argument("--max-draft-tokens", type=int, default=2)

    benchmark = subparsers.add_parser(
        "benchmark", help="synchronized whole-generation target benchmark"
    )
    benchmark.add_argument("--mode", choices=("ordinary", "mtp"), required=True)
    benchmark.add_argument("--model-dir", type=Path, required=True)
    prompts = benchmark.add_mutually_exclusive_group(required=True)
    prompts.add_argument("--prompt")
    prompts.add_argument(
        "--prompt-token-ids",
        help="comma-separated committed prompt token IDs for deterministic probes",
    )
    benchmark.add_argument("--chat", action="store_true")
    benchmark.add_argument("--max-new-tokens", type=int, default=8)
    benchmark.add_argument("--max-draft-tokens", type=int, default=2)
    benchmark.add_argument("--main-backend", default="torch")
    benchmark.add_argument("--draft-backend", default="torch")
    benchmark.add_argument("--ops-backend", default="torch")
    benchmark.add_argument("--device", required=True)
    benchmark.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="float16"
    )
    benchmark.add_argument("--threads", type=int, default=max(1, torch.get_num_threads()))
    benchmark.add_argument("--warmup", type=int, default=3)
    benchmark.add_argument("--repetitions", type=int, default=10)
    benchmark.add_argument("--enable-mstx", action="store_true")
    benchmark.add_argument("--allow-cpu-simulation", action="store_true")
    benchmark.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "audit":
        payload = audit_checkpoint(
            args.model_dir,
            verify_manifest_hashes=args.verify_file_hashes,
        )
        _write_json(payload, args.output)
        return 0 if payload["status"] == "PASS" else 1
    if args.command == "benchmark":
        try:
            args.output = _require_run_output(args.output)
        except Exception as error:
            failure = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(failure, None)
            return 1
        try:
            payload = run_benchmark_command(args)
        except Exception as error:
            failure = {
                "status": "FAIL",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _write_json(failure, args.output)
            return 1
        _write_json(payload, args.output)
        return 0
    try:
        payload = run_generation(args)
    except Exception as error:
        failure = {
            "status": "FAIL",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        _write_json(failure, args.output)
        return 1
    _write_json(payload, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
