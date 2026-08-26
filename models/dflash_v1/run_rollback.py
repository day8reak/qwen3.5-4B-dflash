"""CPU, CUDA, and HIAI entry point for incremental DFlash rollback.

The ordinary baseline and DFlash validation both keep persistent target state.
No formal verification call receives the historical prefix.  CPU/CUDA use a
``DynamicCache`` transaction with GDN-state restore plus bounded commit replay;
the HIAI route delegates state-bank and logical-KV commit to the receiver
bridge.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import torch
from torch import nn

from . import dflash_qwen_adapter_v1 as _legacy
from .dflash_rollback_adapter import (
    FrameworkDFlashRollbackTarget,
    Qwen35DFlashRollbackAdapter,
    validate_qwen35_dflash_rollback,
)
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
        "Validate Qwen3.5 DFlash with persistent CPU/CUDA/NPU target state, "
        "T=K+1 verification, rollback, and bounded commit"
    )
    parser.set_defaults(target_factory=None, hiai_source=None)
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
        "runner": package_dir / "run_rollback.py",
        "bridge": parent / "internal_dflash_bridge.py",
        "wrapper": parent / _ROLLBACK_WRAPPER_SOURCE,
        "hiai_modeling": parent / _ROLLBACK_MODEL_SOURCE,
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


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device_type = str(args.device).split(":", 1)[0].lower()
    if args.max_new_tokens < 0:
        raise ValueError("--max-new-tokens must be non-negative")
    if device_type in {"cuda", "npu"} and args.max_new_tokens < 2:
        raise ValueError(
            "accelerator rollback validation needs at least two new tokens"
        )
    if args.max_draft_tokens is not None and not 1 <= args.max_draft_tokens <= 16:
        raise ValueError("--max-draft-tokens must be between 1 and 16")
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
    _legacy._emit_progress(
        args.progress,
        "draft_checkpoint_audit_begin",
        {"verify_model_sha256": True},
    )
    draft_checkpoint = require_official_dflash_checkpoint(
        args.draft_dir,
        verify_model_hash=True,
    )
    _legacy._emit_progress(
        args.progress,
        "draft_checkpoint_audit_end",
        {
            "config_sha256": draft_checkpoint["config_sha256"],
            "model_sha256": draft_checkpoint["model_sha256"],
            "model_bytes": draft_checkpoint["model_bytes"],
        },
    )
    draft_memory_preflight = _legacy._draft_device_memory_preflight(
        args.device,
        dtype,
        draft_checkpoint,
    )

    _legacy._emit_progress(
        args.progress,
        "target_load_begin",
        {"device": args.device, "dtype": args.dtype},
    )
    target, target_route = _load_transactional_target(args, dtype=dtype)
    _legacy._emit_progress(
        args.progress,
        "target_load_end",
        {"route": target_route, "transactional": True},
    )
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
    adapter = Qwen35DFlashRollbackAdapter(target, draft)
    effective_k = (
        adapter.max_proposal_tokens
        if args.max_draft_tokens is None
        else args.max_draft_tokens
    )

    _legacy._emit_progress(
        args.progress,
        "rollback_validation_begin",
        {
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": args.max_new_tokens,
            "max_draft_tokens": effective_k,
            "historical_prefix_replay": False,
        },
    )
    result = validate_qwen35_dflash_rollback(
        adapter,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=effective_k,
        eos_token_ids=args.eos_token_id,
    )
    source_identity_after = _rollback_runtime_identity(package_dir)
    if source_identity_after != source_identity_before:
        raise RuntimeError("rollback runtime source identity changed during validation")

    draft_round_executed = result.dflash.stats.draft_calls > 0
    execution_gate = {
        "status": "PASS" if draft_round_executed else "INCONCLUSIVE_NO_DRAFT_ROUND",
        "draft_round_executed": draft_round_executed,
        "target_verify_calls": result.dflash.stats.target_verify_calls,
        "historical_prefix_replay_during_verify": False,
    }
    if device_type == "npu":
        state_policy = (
            "persistent HIAI state; npu_gated_delta_rule_mtp recurrent banks; "
            "Torch tensor causal-conv banks on the input NPU device; physical "
            "provisional paged-KV writes with logical-cursor commit"
        )
        target_operator_policy = {
            "gdr": "npu_gated_delta_rule_mtp",
            "conv_bank": "torch_tensor_golden_on_input_device",
            "kv_update": "existing_npu_cache_update_per_row_correctness_fallback",
            "attention": "existing_adn_fused_infer_attention",
        }
    else:
        state_policy = (
            "persistent Transformers DynamicCache; attention KV crop plus GDN "
            "conv/recurrent snapshot restore; commit replays only anchor and "
            "accepted proposals (at most K+1 rows)"
        )
        target_operator_policy = {
            "route": "framework_torch",
            "historical_prefix_replay_during_verify": False,
        }

    report = {
        "schema_version": 3,
        "route": "qwen3.5-dflash-incremental-rollback",
        "classification": {
            "cpu": "CPU/framework rollback simulation",
            "cuda": "CUDA/framework rollback validation",
            "npu": "HIAI/NPU rollback execution; complete device gate remains external",
        }.get(device_type, "framework rollback execution"),
        "strict_greedy_exact_match": True,
        "verification_mode": result.verification_mode,
        "historical_prefix_replay_during_verify": False,
        "state_policy": state_policy,
        "device": str(adapter.device),
        "dtype": str(adapter.dtype),
        "runtime_identity": _legacy._runtime_identity(adapter.device),
        "rollback_runtime_identity": source_identity_after,
        "ops_backend": backend,
        "target_route": target_route,
        "target_rollback_audit": dict(result.target_rollback_audit),
        "target_operator_policy": target_operator_policy,
        "dflash_execution_gate": execution_gate,
        "operator_fallback_enabled": bool(args.allow_op_fallback),
        "performance_claim": "NONE_CORRECTNESS_BRINGUP",
        "target_dir": str(target_root),
        "target_checkpoint": target_checkpoint,
        "draft_dir": str(Path(args.draft_dir).expanduser().resolve()),
        "draft_checkpoint": draft_checkpoint,
        "draft_memory_preflight": draft_memory_preflight,
        "max_proposal_tokens": adapter.max_proposal_tokens,
        "request": _legacy._request_payload(
            args,
            effective_max_draft_tokens=effective_k,
            prompt_token_ids=prompt_ids,
        ),
        "ordinary": _legacy._decode_payload(result.ordinary, tokenizer=tokenizer),
        "dflash": _legacy._decode_payload(result.dflash, tokenizer=tokenizer),
        "ordinary_adapter_stats": asdict(result.ordinary_adapter_stats),
        "dflash_adapter_stats": asdict(result.dflash_adapter_stats),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        _atomic_report(args.report, serialized)
    _legacy._emit_progress(
        args.progress,
        "rollback_validation_end",
        {
            "status": "PASS",
            "draft_round_executed": draft_round_executed,
            "strict_greedy_exact_match": True,
        },
    )
    if tokenizer is not None:
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
