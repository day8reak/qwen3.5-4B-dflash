"""CPU, CUDA, and HIAI entry point for incremental DFlash rollback.

``validate`` compares independent ordinary and DFlash sessions. ``dflash`` runs
only the production generation path.  Both keep persistent Target state and no
verification call receives the historical prefix.  CPU/CUDA use a
``DynamicCache`` transaction with GDN-state restore plus bounded commit replay;
the HIAI route delegates two-pass chunk-GDR and logical-KV commit to the
receiver bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from time import perf_counter
from typing import Any

import torch
from torch import nn

from . import dflash_qwen_adapter_v1 as _legacy
from .dflash_config import DFLASH_MIN_BLOCK_SIZE, OFFICIAL_DFLASH_BLOCK_SIZE
from .dflash_rollback_adapter import (
    FrameworkDFlashRollbackTarget,
    Qwen35DFlashRollbackAdapter,
    validate_qwen35_dflash_rollback,
)
from .dflash_rollback_decode import dflash_rollback_greedy
from .dflash_weights import require_official_dflash_checkpoint
from .modeling_dflash import DFlashDraftModel


DEFAULT_NPU_TARGET_FACTORY = (
    "models.internal_dflash_bridge:load_qwen35_rollback_target"
)
_ROLLBACK_MODEL_SOURCE = "modeling_qwen3_5_hiai_nd_dflash_rollback.py"
_ROLLBACK_WRAPPER_SOURCE = "export_model_wrapper_qwen3_5_dflash_rollback.py"


def _parser():
    parser = _legacy._parser()
    parser.description = (
        "Run or validate Qwen3.5 DFlash with persistent CPU/CUDA/NPU target "
        "state, T=K+1 verification, rollback, and bounded commit"
    )
    parser.set_defaults(target_factory=None, hiai_source=None)
    parser.add_argument(
        "--execution-mode",
        choices=("validate", "dflash"),
        default="validate",
        help=(
            "validate runs ordinary then DFlash and requires exact tokens; "
            "dflash runs only the production DFlash session"
        ),
    )
    rollback_help = {
        "target_loader": (
            "CPU/CUDA only: optional MODULE:FUNCTION returning a framework "
            "model or an already-transactional target"
        ),
        "npu_layout": "embedded rollback source layout",
        "target_factory": (
            "NPU only: MODULE:FUNCTION returning a persistent rollback target; "
            f"default {DEFAULT_NPU_TARGET_FACTORY}"
        ),
        "reset_hook": (
            "unsupported by rollback; state is owned by begin/verify/commit/abort"
        ),
        "hiai_source": (
            "optional NPU assertion; must name the package-local rollback modeling"
        ),
    }
    for action in parser._actions:
        destination = getattr(action, "dest", None)
        if destination in rollback_help:
            action.help = rollback_help[destination]
    return parser


def _source_identity(path: Path) -> dict[str, object]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required rollback runtime source is missing: {path}")
    return {
        "path": str(path.resolve()),
        "sha256": _legacy._sha256_file(path),
    }


def _rollback_runtime_identity(package_dir: Path) -> dict[str, object]:
    parent = package_dir.parent
    files = {
        "scheduler": package_dir / "dflash_rollback_decode.py",
        "adapter": package_dir / "dflash_rollback_adapter.py",
        "draft_modeling": package_dir / "modeling_dflash.py",
        "draft_npu_ops": package_dir / "dflash_ascend310p_ops.py",
        "runner": package_dir / "run_rollback.py",
        "npu_runner": package_dir / "run_npu.py",
        "target_quant_contract": package_dir / "target_quant.py",
        "bridge": parent / "internal_dflash_bridge.py",
        "wrapper": parent / _ROLLBACK_WRAPPER_SOURCE,
        "hiai_modeling": parent / _ROLLBACK_MODEL_SOURCE,
        "shared_qlinear_modeling": parent / "modeling_qwen3_5_hiai_nd.py",
    }
    return {
        "status": "PASS_SOURCE_IDENTITY",
        "files": {name: _source_identity(path) for name, path in files.items()},
    }


def _has_transactional_contract(target: object) -> bool:
    return all(
        callable(getattr(target, name, None))
        for name in (
            "begin_ordinary",
            "advance_ordinary",
            "begin_rollback",
            "verify_rollback",
            "commit_rollback",
            "abort_rollback",
        )
    )


def _load_transactional_target(
    args: Any,
    *,
    dtype: torch.dtype,
) -> tuple[nn.Module, str]:
    device_type = str(args.device).split(":", 1)[0].lower()
    if device_type == "npu":
        factory_spec = args.target_factory or DEFAULT_NPU_TARGET_FACTORY
        factory = _legacy._load_callable(factory_spec)
        target = factory(
            args.target_dir,
            device=torch.device(args.device),
            dtype=dtype,
        )
        route = factory_spec
    else:
        if args.target_factory is not None:
            raise ValueError("--target-factory is reserved for the HIAI NPU route")
        raw_target = _legacy._load_target(
            args.target_dir,
            target_loader=args.target_loader,
            device=args.device,
            dtype=dtype,
            allow_download=args.allow_download,
            trust_remote_code=args.trust_remote_code,
        )
        if _has_transactional_contract(raw_target):
            target = raw_target
            route = args.target_loader or "custom_transactional_framework_target"
        else:
            target = FrameworkDFlashRollbackTarget(raw_target).eval()
            route = "framework_dynamic_cache_transaction"

    if not isinstance(target, nn.Module):
        raise TypeError("rollback target factory must return torch.nn.Module")
    if target.training:
        raise ValueError("rollback target must be in eval mode")
    if not _has_transactional_contract(target):
        raise TypeError("target does not implement the rollback transaction contract")
    audit = getattr(target, "dflash_rollback_audit", None)
    if not isinstance(audit, Mapping):
        raise TypeError("rollback target must expose dflash_rollback_audit")
    if bool(audit.get("historical_prefix_replay_during_verify", True)):
        raise RuntimeError("rollback target declares historical-prefix verification")
    if device_type == "npu" and audit.get("enabled") is not True:
        raise RuntimeError("HIAI target was not created by the rollback factory")
    return target, route


def _atomic_report(path: str, serialized: str) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(
        f".{destination.name}.dflash-rollback-tmp-{os.getpid()}"
    )
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("temporary rollback report path already exists")
    try:
        temporary.write_text(serialized + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _synchronize_device(device: str | torch.device) -> None:
    requested = torch.device(device)
    if requested.type == "cpu":
        return
    backend = getattr(torch, requested.type, None)
    synchronize = getattr(backend, "synchronize", None)
    if not callable(synchronize):
        raise RuntimeError(
            f"{requested.type} backend lacks synchronize(); cannot time model load"
        )
    try:
        synchronize(requested)
    except TypeError:
        synchronize()


def main(argv: Sequence[str] | None = None) -> int:
    request_started = perf_counter()
    args = _parser().parse_args(argv)
    device_type = str(args.device).split(":", 1)[0].lower()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if device_type in {"cuda", "npu"} and args.max_new_tokens < 2:
        raise ValueError(
            "accelerator rollback execution needs at least two new tokens"
        )
    if args.block_size is not None and not (
        DFLASH_MIN_BLOCK_SIZE <= args.block_size <= OFFICIAL_DFLASH_BLOCK_SIZE
    ):
        raise ValueError(
            "--block-size must be between "
            f"{DFLASH_MIN_BLOCK_SIZE} and {OFFICIAL_DFLASH_BLOCK_SIZE}"
        )
    if args.allow_op_fallback and device_type != "cpu":
        raise ValueError("operator fallback is allowed only for CPU simulation")
    if args.reset_hook is not None:
        raise ValueError(
            "--reset-hook is a full-prefix control and is unsupported by rollback"
        )
    if device_type == "npu" and args.target_loader is not None:
        raise ValueError("NPU rollback uses --target-factory, not --target-loader")
    if device_type != "npu" and args.hiai_source is not None:
        raise ValueError("--hiai-source applies only to the HIAI NPU route")
    if device_type == "npu" and tuple(args.eos_token_id) != (248044,):
        raise ValueError("the locked NPU rollback route requires EOS token 248044")

    package_dir = Path(__file__).resolve().parent
    expected_hiai_source = package_dir.parent / _ROLLBACK_MODEL_SOURCE
    if args.hiai_source is not None:
        supplied_hiai_source = Path(args.hiai_source).expanduser()
        if (
            supplied_hiai_source.is_symlink()
            or supplied_hiai_source.resolve() != expected_hiai_source.resolve()
        ):
            raise ValueError(
                "--hiai-source must match the package-local rollback modeling"
            )
    _legacy._validate_report_destination(
        args,
        package_dir=package_dir,
        formal_npu=False,
    )
    source_identity_before = _rollback_runtime_identity(package_dir)
    _legacy._validate_ops_backend_request(args.device, args.ops_backend)
    dtype = _legacy._dtype(args.dtype)
    _legacy._prepare_device_backend(args.device)
    _legacy._validate_experiment_dtype(args.device, dtype)

    target_root = Path(args.target_dir).expanduser().resolve()
    target_checkpoint = _legacy._audit_target_config(target_root)
    prompt_ids, tokenizer = _legacy._resolve_prompt(args, target_root=target_root)
    checkpoint_audit_started = perf_counter()
    _legacy._emit_progress(
        args.progress,
        "draft_checkpoint_audit_begin",
        {"verify_model_sha256": True},
    )
    draft_checkpoint = require_official_dflash_checkpoint(
        args.draft_dir,
        verify_model_hash=True,
    )
    checkpoint_audit_seconds = perf_counter() - checkpoint_audit_started
    _legacy._emit_progress(
        args.progress,
        "draft_checkpoint_audit_end",
        {
            "config_sha256": draft_checkpoint["config_sha256"],
            "model_sha256": draft_checkpoint["model_sha256"],
            "model_bytes": draft_checkpoint["model_bytes"],
        },
    )
    _legacy._emit_progress(
        args.progress,
        "target_load_begin",
        {"device": args.device, "dtype": args.dtype},
    )
    target_load_started = perf_counter()
    target, target_route = _load_transactional_target(args, dtype=dtype)
    _synchronize_device(args.device)
    target_load_seconds = perf_counter() - target_load_started
    _legacy._emit_progress(
        args.progress,
        "target_load_end",
        {"route": target_route, "transactional": True},
    )
    # Query free memory after the Target is resident.  Running this check
    # before Target load can report enough space for the Draft and then OOM
    # immediately because the multi-gigabyte Target allocation was omitted.
    draft_memory_preflight = _legacy._draft_device_memory_preflight(
        args.device,
        dtype,
        draft_checkpoint,
    )
    draft_load_started = perf_counter()
    ops, backend = _legacy._select_draft_ops(
        device=args.device,
        ops_backend=args.ops_backend,
        allow_op_fallback=args.allow_op_fallback,
    )
    _legacy._emit_progress(args.progress, "draft_ops_ready", {"backend": backend})
    draft = DFlashDraftModel.from_pretrained(
        args.draft_dir,
        ops=ops,
        device=args.device,
        dtype=dtype,
    )
    _synchronize_device(args.device)
    draft_load_seconds = perf_counter() - draft_load_started
    adapter = Qwen35DFlashRollbackAdapter(target, draft)
    effective_block_size = (
        adapter.max_block_size
        if args.block_size is None
        else args.block_size
    )

    progress_fields = {
        "execution_mode": args.execution_mode,
        "prompt_tokens": len(prompt_ids),
        "max_new_tokens": args.max_new_tokens,
        "block_size": effective_block_size,
        "proposal_capacity": effective_block_size - 1,
        "historical_prefix_replay": False,
        "draft_kv_cache": True,
    }
    _legacy._emit_progress(
        args.progress,
        "rollback_execution_begin",
        progress_fields,
    )
    if args.execution_mode == "validate":
        validation = validate_qwen35_dflash_rollback(
            adapter,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=effective_block_size,
            eos_token_ids=args.eos_token_id,
            progress_callback=lambda event, fields: _legacy._emit_progress(
                args.progress,
                event,
                fields,
            ),
        )
        ordinary_result = validation.ordinary
        ordinary_stats = validation.ordinary_adapter_stats
        ordinary_elapsed_seconds: float | None = (
            validation.ordinary_elapsed_seconds
        )
        dflash_result = validation.dflash
        dflash_stats = validation.dflash_adapter_stats
        dflash_elapsed_seconds = validation.dflash_elapsed_seconds
        validation_seconds: float | None = (
            ordinary_elapsed_seconds + dflash_elapsed_seconds
        )
        target_rollback_audit = dict(validation.target_rollback_audit)
        draft_kv_cache_audit = dict(validation.draft_kv_cache_audit)
        correctness_gate = {
            "status": "PASS",
            "ordinary_comparison_executed": True,
            "strict_greedy_exact_match": True,
        }
    else:
        adapter.reset_rollback_stats()
        _synchronize_device(adapter.device)
        dflash_started = perf_counter()
        dflash_result = dflash_rollback_greedy(
            adapter,
            prompt_ids,
            max_new_tokens=args.max_new_tokens,
            block_size=effective_block_size,
            eos_token_ids=args.eos_token_id,
            input_device=adapter.device,
            progress_callback=lambda event, fields: _legacy._emit_progress(
                args.progress,
                event,
                fields,
            ),
        )
        _synchronize_device(adapter.device)
        dflash_elapsed_seconds = perf_counter() - dflash_started
        dflash_stats = adapter.snapshot_rollback_stats()
        ordinary_result = None
        ordinary_stats = None
        ordinary_elapsed_seconds = None
        validation_seconds = None
        raw_target_audit = getattr(adapter.target, "dflash_rollback_audit", None)
        target_rollback_audit = (
            dict(raw_target_audit) if isinstance(raw_target_audit, Mapping) else {}
        )
        draft_kv_cache_audit = dict(adapter.dflash_draft_cache_audit)
        correctness_gate = {
            "status": "NOT_RUN_DFLASH_ONLY",
            "ordinary_comparison_executed": False,
            "strict_greedy_exact_match": None,
            "validation_command": "rerun with --execution-mode validate",
        }
    source_identity_after = _rollback_runtime_identity(package_dir)
    if source_identity_after != source_identity_before:
        raise RuntimeError("rollback runtime source identity changed during execution")

    draft_round_executed = dflash_result.stats.draft_calls > 0
    execution_gate = {
        "status": "PASS" if draft_round_executed else "INCONCLUSIVE_NO_DRAFT_ROUND",
        "draft_round_executed": draft_round_executed,
        "target_verify_calls": dflash_result.stats.target_verify_calls,
        "historical_prefix_replay_during_verify": False,
        "draft_kv_cache": True,
    }
    if device_type == "npu":
        raw_target_quantization = getattr(
            adapter.target,
            "dflash_target_quantization_audit",
            None,
        )
        if not isinstance(raw_target_quantization, Mapping):
            raise TypeError(
                "NPU rollback Target must expose dflash_target_quantization_audit"
            )
        target_quantization = dict(raw_target_quantization)
        state_policy = (
            "persistent scalar HIAI state; original chunk GDR verify plus "
            "accepted-prefix second chunk commit; Torch tensor causal-conv "
            "prefix states on the input NPU device; physical provisional "
            "paged-KV writes with logical-cursor commit"
        )
        target_operator_policy = {
            "gdr": "npu_chunk_gated_delta_rule_two_pass",
            "conv_bank": "torch_tensor_golden_on_input_device",
            "kv_update": "existing_npu_cache_update_per_row_correctness_fallback",
            "attention": "existing_adn_fused_infer_attention",
            "linear": (
                "existing_qlinear_w8a8_dynamic"
                if target_quantization.get("scheme") == "w8a8_dynamic"
                else "existing_fp16_linear"
            ),
        }
    else:
        target_quantization = {
            "status": "NOT_APPLICABLE_FRAMEWORK_TARGET",
            "scheme": "disabled",
        }
        state_policy = (
            "persistent Transformers DynamicCache; attention KV crop plus GDN "
            "conv/recurrent snapshot restore; commit replays only anchor and "
            "accepted proposals (at most K+1 rows)"
        )
        target_operator_policy = {
            "route": "framework_torch",
            "historical_prefix_replay_during_verify": False,
        }

    ops_module = getattr(ops, "module", None)
    exhaustive_checks = getattr(
        ops_module,
        "exhaustive_value_checks_enabled",
        None,
    )
    if callable(exhaustive_checks):
        draft_value_check_policy = {
            "mode": (
                "exhaustive_intermediate_and_boundary"
                if exhaustive_checks(adapter.device)
                else "boundary_only"
            ),
            "boundary_logits_finite_check": True,
        }
    else:
        draft_value_check_policy = {
            "mode": "backend_default",
            "boundary_logits_finite_check": True,
        }

    request_to_report_seconds = perf_counter() - request_started

    report = {
        "schema_version": 5,
        "route": "qwen3.5-dflash-incremental-rollback",
        "classification": {
            "cpu": f"CPU/framework rollback simulation ({args.execution_mode})",
            "cuda": f"CUDA/framework rollback execution ({args.execution_mode})",
            "npu": (
                "HIAI/NPU rollback execution "
                f"({args.execution_mode}); complete device gate remains external"
            ),
        }.get(device_type, "framework rollback execution"),
        "execution_mode": args.execution_mode,
        "decode_policy": "strict_greedy",
        "strict_greedy_exact_match": correctness_gate[
            "strict_greedy_exact_match"
        ],
        "correctness_gate": correctness_gate,
        "verification_mode": "incremental_transactional_rollback",
        "historical_prefix_replay_during_verify": False,
        "state_policy": state_policy,
        "device": str(adapter.device),
        "dtype": str(adapter.dtype),
        "runtime_identity": _legacy._runtime_identity(adapter.device),
        "rollback_runtime_identity": source_identity_after,
        "ops_backend": backend,
        "draft_value_check_policy": draft_value_check_policy,
        "target_route": target_route,
        "target_rollback_audit": target_rollback_audit,
        "target_quantization": target_quantization,
        "draft_kv_cache_audit": draft_kv_cache_audit,
        "target_operator_policy": target_operator_policy,
        "dflash_execution_gate": execution_gate,
        "operator_fallback_enabled": bool(args.allow_op_fallback),
        "performance_claim": (
            "NONE_CORRECTNESS_BRINGUP"
            if args.execution_mode == "validate"
            else "UNMEASURED_SINGLE_DFLASH_RUN"
        ),
        "target_dir": str(target_root),
        "target_checkpoint": target_checkpoint,
        "draft_dir": str(Path(args.draft_dir).expanduser().resolve()),
        "draft_checkpoint": draft_checkpoint,
        "draft_memory_preflight": draft_memory_preflight,
        "block_size": effective_block_size,
        "max_proposal_tokens": adapter.max_proposal_tokens,
        "request": _legacy._request_payload(
            args,
            effective_block_size=effective_block_size,
            prompt_token_ids=prompt_ids,
        ),
        "ordinary": (
            None
            if ordinary_result is None
            else _legacy._decode_payload(ordinary_result, tokenizer=tokenizer)
        ),
        "dflash": _legacy._decode_payload(dflash_result, tokenizer=tokenizer),
        "ordinary_adapter_stats": (
            None if ordinary_stats is None else asdict(ordinary_stats)
        ),
        "dflash_adapter_stats": asdict(dflash_stats),
        "timings_seconds": {
            "draft_checkpoint_audit": checkpoint_audit_seconds,
            "target_load": target_load_seconds,
            "draft_load": draft_load_seconds,
            "ordinary_decode": ordinary_elapsed_seconds,
            "dflash_decode": dflash_elapsed_seconds,
            "validation_decode_total": validation_seconds,
            "request_to_report_build": request_to_report_seconds,
        },
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        _atomic_report(args.report, serialized)
    _legacy._emit_progress(
        args.progress,
        "rollback_execution_end",
        {
            "status": "PASS",
            "execution_mode": args.execution_mode,
            "draft_round_executed": draft_round_executed,
            "strict_greedy_exact_match": correctness_gate[
                "strict_greedy_exact_match"
            ],
        },
    )
    if tokenizer is not None:
        if report["ordinary"] is not None:
            print("\n=== Ordinary Target 输出 ===", file=sys.stderr)
            print(report["ordinary"]["generated_text"], file=sys.stderr)
        print("\n=== DFlash rollback 输出 ===", file=sys.stderr)
        print(report["dflash"]["generated_text"], file=sys.stderr)
        print(file=sys.stderr)
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["DEFAULT_NPU_TARGET_FACTORY", "main"]
