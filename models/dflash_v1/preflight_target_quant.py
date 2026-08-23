"""Target-only NPU preflight for the experimental quant branch.

This command loads the quantized HIAI target and exercises the same fresh
full-prefix bridge used by DFlash, but deliberately does not hash or load the
Draft checkpoint.  It is the inexpensive first device gate for converter,
input-provider, feature-route, and call-local state integration.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from .dflash_qwen_adapter_v1 import (
    _compare_repeatable_tensors,
    _require_repeatability_pass,
)
from .internal_target_loader import (
    DECODE_CHUNK_SIZE_ENV,
    PREFILL_CHUNK_SIZE_ENV,
    TARGET_FACTORY_ENV,
    load_target,
)
from .run_npu import (
    DEFAULT_TARGET_FACTORY,
    KV_CACHE_MAX_LEN_ENV,
    _configure_target_quantization,
)
from .target_quant import QUANT_MODE_W8A8_DYNAMIC


_ASSEMBLY_PASS = "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Load only the W8A8 NPU target and validate its DFlash bridge; "
            "the Draft checkpoint is not read"
        )
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument(
        "--prompt-ids",
        required=True,
        help="non-empty comma-separated token IDs used by bounded target probes",
    )
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--kv-cache-max-len", type=int, required=True)
    parser.add_argument("--target-quantizer", required=True)
    parser.add_argument("--target-quant-artifact", required=True)
    parser.add_argument("--target-input-provider", required=True)
    parser.add_argument(
        "--export-w8a8-emulation-artifact",
        help=(
            "optional new directory receiving portable QLinear W_q/scale files "
            "for correctness-first CPU/CUDA formula emulation"
        ),
    )
    parser.add_argument("--report")
    return parser


def _prompt_ids(value: str) -> tuple[int, ...]:
    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("--prompt-ids must be a non-empty comma-separated list")
    try:
        result = tuple(int(piece, 10) for piece in pieces)
    except ValueError as error:
        raise ValueError("--prompt-ids must contain decimal integers") from error
    if any(token < 0 for token in result):
        raise ValueError("--prompt-ids must not contain negative token IDs")
    return result


def _field(output: object, name: str) -> Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    return value if isinstance(value, Tensor) else None


def _execute(
    target: nn.Module,
    input_ids: Tensor,
    *,
    features: bool,
) -> object:
    with torch.inference_mode():
        return target(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
            output_dflash_features=features,
            logits_to_keep=1,
        )


def _snapshot(
    target: nn.Module,
    input_ids: Tensor,
    *,
    features: bool,
    vocab_size: int,
) -> tuple[Tensor, Tensor | None]:
    output = _execute(target, input_ids, features=features)
    logits = _field(output, "logits")
    if logits is None or tuple(logits.shape) != (1, 1, vocab_size):
        actual = None if logits is None else tuple(logits.shape)
        raise ValueError(
            "quant target preflight requires last-row logits shape "
            f"(1, 1, {vocab_size}); got {actual}"
        )
    captured = _field(output, "dflash_features")
    if features:
        if captured is None:
            raise TypeError("feature-enabled target returned no dflash_features")
    elif captured is not None:
        raise ValueError("feature-disabled target unexpectedly returned features")
    return (
        logits.detach().clone(),
        None if captured is None else captured.detach().clone(),
    )


def _different_length_prefix(
    prefix: Tensor,
    *,
    vocab_size: int,
    maximum_length: int,
) -> Tensor:
    if int(prefix.shape[1]) < maximum_length:
        different_token = (prefix[0, -1] + 1) % vocab_size
        return torch.cat((prefix, different_token.reshape(1, 1)), dim=1)
    if int(prefix.shape[1]) > 1:
        return prefix[:, :-1]
    raise ValueError(
        "cannot construct a different-length prefix within kv_cache_max_len"
    )


def _run_bounded_probes(
    target: nn.Module,
    prefix: Tensor,
) -> dict[str, object]:
    config = getattr(target, "config", None)
    if config is None:
        raise TypeError("quant target must expose config")
    vocab_size = int(getattr(config, "vocab_size", 0))
    maximum_length = int(getattr(config, "kv_cache_max_len", 0))
    if vocab_size <= 1 or maximum_length <= 0:
        raise ValueError("target config has invalid vocab_size/kv_cache_max_len")
    if int(prefix.shape[1]) > maximum_length:
        raise ValueError("probe prefix exceeds kv_cache_max_len")
    if bool(((prefix < 0) | (prefix >= vocab_size)).any().item()):
        raise ValueError("probe prefix contains a token outside target vocabulary")
    different = _different_length_prefix(
        prefix,
        vocab_size=vocab_size,
        maximum_length=maximum_length,
    )

    ordinary_before, _ = _snapshot(
        target,
        prefix,
        features=False,
        vocab_size=vocab_size,
    )
    ordinary_immediate, _ = _snapshot(
        target,
        prefix,
        features=False,
        vocab_size=vocab_size,
    )
    ordinary_control = _compare_repeatable_tensors(
        ordinary_before,
        ordinary_immediate,
        require_top1=True,
    )
    _require_repeatability_pass(
        ordinary_control,
        message="quant target ordinary P-P control failed",
    )

    feature_logits_before, feature_before = _snapshot(
        target,
        prefix,
        features=True,
        vocab_size=vocab_size,
    )
    assert feature_before is not None
    zero_impact = _compare_repeatable_tensors(
        ordinary_immediate,
        feature_logits_before,
        require_top1=True,
    )
    _require_repeatability_pass(
        zero_impact,
        message="feature capture changed quant target logits",
    )

    feature_logits_immediate, feature_immediate = _snapshot(
        target,
        prefix,
        features=True,
        vocab_size=vocab_size,
    )
    assert feature_immediate is not None
    feature_logits_control = _compare_repeatable_tensors(
        feature_logits_before,
        feature_logits_immediate,
        require_top1=True,
    )
    _require_repeatability_pass(
        feature_logits_control,
        message="quant target feature-mode P-P logits control failed",
    )
    feature_control = _compare_repeatable_tensors(
        feature_before,
        feature_immediate,
        require_top1=False,
    )
    _require_repeatability_pass(
        feature_control,
        message="quant target feature-mode P-P feature control failed",
    )

    _execute(target, different, features=False)
    feature_logits_after, feature_after = _snapshot(
        target,
        prefix,
        features=True,
        vocab_size=vocab_size,
    )
    assert feature_after is not None
    feature_logits_isolation = _compare_repeatable_tensors(
        feature_logits_before,
        feature_logits_after,
        require_top1=True,
    )
    _require_repeatability_pass(
        feature_logits_isolation,
        message="quant target feature logits changed after intervening Q",
    )
    feature_isolation = _compare_repeatable_tensors(
        feature_before,
        feature_after,
        require_top1=False,
    )
    _require_repeatability_pass(
        feature_isolation,
        message="quant target features changed after intervening Q",
    )

    _execute(target, different, features=True)
    ordinary_after, _ = _snapshot(
        target,
        prefix,
        features=False,
        vocab_size=vocab_size,
    )
    ordinary_isolation = _compare_repeatable_tensors(
        ordinary_before,
        ordinary_after,
        require_top1=True,
    )
    _require_repeatability_pass(
        ordinary_isolation,
        message="quant target ordinary logits changed after intervening Q",
    )

    return {
        "status": "PASS_BOUNDED_TARGET_PROBES",
        "prefix_length": int(prefix.shape[1]),
        "intervening_prefix_length": int(different.shape[1]),
        "target_forward_calls": 8,
        "feature_zero_impact": zero_impact,
        "ordinary": {
            "immediate_p_p_control": ordinary_control,
            "p_q_p_isolation": ordinary_isolation,
        },
        "feature_mode": {
            "logits_immediate_p_p_control": feature_logits_control,
            "features_immediate_p_p_control": feature_control,
            "logits_p_q_p_isolation": feature_logits_isolation,
            "features_p_q_p_isolation": feature_isolation,
        },
        "scope": (
            "bounded full-prefix behavior only; ordinary incremental quant "
            "target parity and per-operator device trace remain pending"
        ),
    }


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_report_destination(
    destination: Path,
    *,
    target_root: Path,
    artifact: Path,
) -> Path:
    if destination.is_symlink():
        raise ValueError("--report must not be a symlink")
    resolved = destination.resolve()
    package_root = Path(__file__).resolve().parents[2]
    protected_roots = [target_root, package_root]
    if artifact.is_dir():
        protected_roots.append(artifact)
    if resolved == artifact or any(
        _is_within(resolved, root) for root in protected_roots
    ):
        raise ValueError(
            "--report must be outside the target, source package, and quant artifact"
        )
    return resolved


def _validate_export_destination(
    destination: Path,
    *,
    target_root: Path,
    artifact: Path,
    report_destination: Path | None,
) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "--export-w8a8-emulation-artifact must name a new directory"
        )
    resolved = destination.resolve()
    package_root = Path(__file__).resolve().parents[2]
    protected_roots = [target_root, package_root]
    if artifact.is_dir():
        protected_roots.append(artifact)
    if resolved == artifact or any(
        _is_within(resolved, root) for root in protected_roots
    ):
        raise ValueError(
            "W8A8 emulation export must be outside the target, source package, "
            "and source quant artifact"
        )
    if report_destination is not None and resolved == report_destination:
        raise ValueError("W8A8 emulation export and --report must be different")
    if not resolved.parent.is_dir():
        raise FileNotFoundError(
            "parent directory for --export-w8a8-emulation-artifact does not exist"
        )
    return resolved


def _write_report(
    payload: Mapping[str, object],
    destination: Path,
) -> None:
    resolved = destination.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise RuntimeError("temporary report path already exists")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, resolved)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quantization_audit(target: nn.Module) -> tuple[dict[str, object], dict[str, object]]:
    raw = getattr(target, "dflash_full_prefix_isolation_audit", None)
    if not isinstance(raw, Mapping):
        raise TypeError("target facade did not expose its isolation audit")
    audit = dict(raw)
    bridge = audit.get("bridge_runtime")
    if not isinstance(bridge, Mapping):
        raise TypeError("target facade did not expose bridge_runtime audit")
    quant = bridge.get("target_quantization")
    if not isinstance(quant, Mapping):
        raise TypeError("target bridge did not expose target_quantization audit")
    normalized = dict(quant)
    if normalized.get("status") != _ASSEMBLY_PASS:
        raise RuntimeError("quant target assembly contract did not pass")
    if normalized.get("scheme") != QUANT_MODE_W8A8_DYNAMIC:
        raise RuntimeError("target bridge did not activate W8A8 dynamic mode")
    qlinear_count = normalized.get("qlinear_count")
    if isinstance(qlinear_count, bool) or not isinstance(qlinear_count, int):
        raise TypeError("quant target qlinear_count must be an integer")
    if qlinear_count <= 0:
        raise RuntimeError("quant target contains no audited QLinear modules")
    return audit, normalized


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not str(args.device).startswith("npu"):
        raise ValueError("quant target preflight requires --device npu or npu:N")
    if args.kv_cache_max_len <= 0 or args.kv_cache_max_len % 64:
        raise ValueError("--kv-cache-max-len must be positive and divisible by 64")
    tokens = _prompt_ids(args.prompt_ids)
    target_root = Path(args.target_dir).expanduser().resolve()
    if not target_root.is_dir():
        raise FileNotFoundError(f"target directory does not exist: {target_root}")

    quant_args = argparse.Namespace(
        target_quant_mode=QUANT_MODE_W8A8_DYNAMIC,
        target_factory=DEFAULT_TARGET_FACTORY,
        target_quantizer=args.target_quantizer,
        target_quant_artifact=args.target_quant_artifact,
        target_input_provider=args.target_input_provider,
        report=args.report,
    )
    _configure_target_quantization(quant_args)
    artifact = Path(args.target_quant_artifact).expanduser().resolve()
    report_destination = (
        None
        if args.report is None
        else _validate_report_destination(
            Path(args.report).expanduser(),
            target_root=target_root,
            artifact=artifact,
        )
    )
    export_destination = (
        None
        if args.export_w8a8_emulation_artifact is None
        else _validate_export_destination(
            Path(args.export_w8a8_emulation_artifact).expanduser(),
            target_root=target_root,
            artifact=artifact,
            report_destination=report_destination,
        )
    )
    os.environ[TARGET_FACTORY_ENV] = DEFAULT_TARGET_FACTORY
    os.environ[PREFILL_CHUNK_SIZE_ENV] = "64"
    os.environ[DECODE_CHUNK_SIZE_ENV] = "1"
    os.environ[KV_CACHE_MAX_LEN_ENV] = str(args.kv_cache_max_len)

    target = load_target(
        str(target_root),
        device=args.device,
        dtype=torch.float16,
    ).eval()
    if not isinstance(target, nn.Module):
        raise TypeError("target loader did not return torch.nn.Module")
    initial_audit, initial_quant = _quantization_audit(target)
    device = torch.device(args.device)
    prefix = torch.tensor([tokens], dtype=torch.long, device=device)
    probes = _run_bounded_probes(target, prefix)
    final_audit, final_quant = _quantization_audit(target)

    expected_calls = int(probes["target_forward_calls"])
    bridge = final_audit.get("bridge_runtime")
    assert isinstance(bridge, Mapping)
    for label, value in (
        ("facade target_forward_calls", final_audit.get("target_forward_calls")),
        ("bridge full_prefix_calls", bridge.get("full_prefix_calls")),
        ("input_provider_calls", final_quant.get("input_provider_calls")),
        (
            "input_provider_successes",
            final_quant.get("input_provider_successes"),
        ),
    ):
        if value != expected_calls:
            raise RuntimeError(
                f"{label} must equal bounded probe calls {expected_calls}; got {value}"
            )
    if final_quant.get("input_provider_failures") != 0:
        raise RuntimeError("quant target input provider reported failures")

    emulation_export: dict[str, object] | None = None
    if export_destination is not None:
        from .w8a8_emulation import export_w8a8_emulation_artifact

        qlinear_paths = final_quant.get("qlinear_paths")
        if not isinstance(qlinear_paths, list) or any(
            not isinstance(path, str) for path in qlinear_paths
        ):
            raise TypeError("quant target audit did not expose qlinear_paths")
        emulation_export = export_w8a8_emulation_artifact(
            target,
            export_destination,
            expected_qlinear_paths=qlinear_paths,
        )

    payload: dict[str, Any] = {
        "schema_version": 1,
        "status": "PASS_TARGET_QUANT_ASSEMBLY_AND_BOUNDED_PREFIX_PROBES",
        "classification": "TARGET_ONLY_NO_DFLASH_DRAFT",
        "device": str(device),
        "dtype": str(torch.float16),
        "draft_checkpoint_read": False,
        "prompt_token_count": len(tokens),
        "target_quantization_initial": initial_quant,
        "target_quantization_final": final_quant,
        "target_isolation_initial": initial_audit,
        "target_isolation_final": final_audit,
        "bounded_probes": probes,
        "w8a8_emulation_export": emulation_export,
        "remaining_gates": [
            "same_activation_real_npu_qlinear_vs_cpu_cuda_formula_parity",
            "ordinary_incremental_quant_target_vs_fresh_full_prefix_parity",
            "quant_target_plus_dflash_strict_greedy_exact_match",
            "real_npu_operator_trace_and_no_fallback",
            "acceptance_and_performance_measurement",
        ],
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.report is None:
        print(serialized)
    else:
        assert report_destination is not None
        _write_report(payload, report_destination)
        print(f"TARGET_QUANT_PREFLIGHT_PASS report={report_destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
