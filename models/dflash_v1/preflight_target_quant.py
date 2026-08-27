"""Target-only NPU preflight for the experimental quant branch.

This command loads the quantized rollback HIAI target and exercises both its
fresh full-prefix diagnostic route and its persistent ordinary/rollback
transaction, but deliberately does not hash or load the Draft checkpoint.  It
is the inexpensive first device gate for converter, input-provider,
feature-route, state-bank, and logical-KV integration.
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
    parser.add_argument("--target-quant-weight-path", required=True)
    parser.add_argument("--target-input-provider", required=True)
    parser.add_argument("--target-embedding-weight-path", required=True)
    parser.add_argument("--target-embedding-scale-path", required=True)
    comparison = parser.add_mutually_exclusive_group()
    comparison.add_argument(
        "--compare-first-qlinear",
        action="store_true",
        help=(
            "capture the first QLinear activation/output from one real NPU "
            "Target call and compare it with the CPU W8A8 formula"
        ),
    )
    comparison.add_argument(
        "--compare-qlinear-path",
        action="append",
        default=[],
        help=(
            "repeatable audited QLinear module path for same-activation NPU/CPU "
            "comparison; cannot be combined with --compare-first-qlinear"
        ),
    )
    parser.add_argument(
        "--require-qlinear-bitwise",
        action="store_true",
        help=(
            "return a failing status when any requested same-activation "
            "comparison is not bitwise equal"
        ),
    )
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
            "bounded fresh full-prefix oracle sub-probe; persistent ordinary "
            "and rollback transactions are checked by rollback_target_probe, "
            "while per-operator device trace remains pending"
        ),
    }


def _run_rollback_target_probe(
    target: nn.Module,
    prefix: Tensor,
) -> dict[str, object]:
    """Compare one quantized ordinary step with the rollback state-bank step."""

    controller = getattr(target, "target", None)
    if not isinstance(controller, nn.Module):
        raise TypeError("rollback preflight requires the packaged target facade")
    required = (
        "begin_ordinary",
        "advance_ordinary",
        "begin_rollback",
        "verify_rollback",
        "commit_rollback",
        "abort_rollback",
    )
    missing = [name for name in required if not callable(getattr(controller, name, None))]
    if missing:
        raise TypeError(
            "quantized Target lacks rollback transaction methods: "
            + ", ".join(missing)
        )
    config = getattr(controller, "config", None)
    vocab_size = int(getattr(config, "vocab_size", 0))
    kv_cache_max_len = int(getattr(controller, "kv_cache_max_len", 0))
    if vocab_size <= 1:
        raise ValueError("rollback Target has an invalid vocabulary size")
    if kv_cache_max_len <= 0:
        raise ValueError("rollback Target has an invalid kv_cache_max_len")
    if int(prefix.shape[1]) + 1 > kv_cache_max_len:
        raise ValueError(
            "rollback preflight needs one decode row beyond the prompt, but "
            f"prompt length {int(prefix.shape[1])} reaches kv_cache_max_len "
            f"{kv_cache_max_len}"
        )
    next_token = ((prefix[:, -1:] + 1) % vocab_size).to(torch.long)
    before_quant = getattr(controller, "dflash_target_quantization_audit", None)
    if not isinstance(before_quant, Mapping):
        raise TypeError("rollback Target lacks a quantization audit")

    ordinary_prefill = controller.begin_ordinary(prefix)
    ordinary_prefill_logits = _field(ordinary_prefill, "logits")
    ordinary_step = controller.advance_ordinary(next_token)
    ordinary_step_logits = _field(ordinary_step, "logits")
    rollback_prefill = controller.begin_rollback(prefix)
    rollback_prefill_logits = _field(rollback_prefill, "logits")
    rollback_features = _field(rollback_prefill, "dflash_features")
    rollback_step = controller.verify_rollback(next_token)
    rollback_step_logits = _field(rollback_step, "logits")
    controller.commit_rollback(0)

    tensors = {
        "ordinary_prefill_logits": ordinary_prefill_logits,
        "ordinary_step_logits": ordinary_step_logits,
        "rollback_prefill_logits": rollback_prefill_logits,
        "rollback_step_logits": rollback_step_logits,
    }
    missing_tensors = [name for name, value in tensors.items() if value is None]
    if missing_tensors:
        raise TypeError(
            "rollback Target probe returned no " + ", ".join(missing_tensors)
        )
    assert ordinary_prefill_logits is not None
    assert ordinary_step_logits is not None
    assert rollback_prefill_logits is not None
    assert rollback_step_logits is not None
    if rollback_features is None or tuple(rollback_features.shape[:2]) != tuple(
        prefix.shape
    ):
        raise ValueError("rollback prefill did not return one feature row per token")
    prefill_parity = _compare_repeatable_tensors(
        ordinary_prefill_logits,
        rollback_prefill_logits,
        require_top1=True,
    )
    _require_repeatability_pass(
        prefill_parity,
        message="quantized ordinary/rollback prefill logits differ",
    )
    step_parity = _compare_repeatable_tensors(
        ordinary_step_logits,
        rollback_step_logits,
        require_top1=True,
    )
    _require_repeatability_pass(
        step_parity,
        message="quantized ordinary/rollback one-step logits differ",
    )

    after_quant = getattr(controller, "dflash_target_quantization_audit", None)
    rollback_audit = getattr(controller, "dflash_rollback_audit", None)
    if not isinstance(after_quant, Mapping) or not isinstance(rollback_audit, Mapping):
        raise TypeError("rollback Target audits disappeared during the probe")
    before_calls = before_quant.get("input_provider_calls")
    after_calls = after_quant.get("input_provider_calls")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in (before_calls, after_calls)
    ):
        raise TypeError("input-provider call counters must be integers")
    assert isinstance(before_calls, int) and isinstance(after_calls, int)
    prompt_chunks = (int(prefix.shape[1]) + 63) // 64
    expected_calls = 2 * prompt_chunks + 2
    if after_calls - before_calls != expected_calls:
        raise RuntimeError(
            "rollback quant input-provider calls do not cover ordinary and "
            f"rollback execution: expected {expected_calls}, got "
            f"{after_calls - before_calls}"
        )
    if rollback_audit.get("historical_prefix_replay_during_verify") is not False:
        raise RuntimeError("rollback probe unexpectedly replayed the prefix")
    return {
        "status": "PASS_QUANTIZED_ROLLBACK_TARGET_TRANSACTION",
        "input_provider_calls": expected_calls,
        "prompt_chunks_per_session": prompt_chunks,
        "ordinary_vs_rollback_prefill": prefill_parity,
        "ordinary_vs_rollback_one_step": step_parity,
        "historical_prefix_replay_during_verify": False,
        "committed_rows": 1,
    }


def _quantized_execution_model(target: nn.Module) -> nn.Module:
    controller = getattr(target, "target", None)
    execution_model = getattr(controller, "dflash_execution_model", None)
    if not isinstance(controller, nn.Module) or not isinstance(
        execution_model,
        nn.Module,
    ):
        raise TypeError(
            "same-activation comparison requires the packaged facade over "
            "InternalDFlashTarget"
        )
    return execution_model


def _same_activation_qlinear_comparison(
    target: nn.Module,
    prefix: Tensor,
    *,
    audited_paths: Sequence[str],
    compare_first: bool,
    requested_paths: Sequence[str],
) -> dict[str, object]:
    """Compare real QLinear outputs with the CPU formula on identical inputs.

    Hooks capture both tensors from one real Target forward.  This avoids the
    invalid comparison where two independent whole-model runs have already
    produced different activations before the selected QLinear.
    """

    if not compare_first and not requested_paths:
        return {
            "status": "DISABLED",
            "target_forward_calls": 0,
            "scope": "same real activation comparison was not requested",
        }
    if not audited_paths or any(
        not isinstance(path, str) or not path for path in audited_paths
    ):
        raise ValueError("quant target audit contains no valid QLinear paths")
    if len(audited_paths) != len(set(audited_paths)):
        raise ValueError("quant target audit contains duplicate QLinear paths")

    execution_model = _quantized_execution_model(target)
    modules = dict(execution_model.named_modules())
    audited_set = set(audited_paths)
    audited_module_order = [
        name
        for name, module in execution_model.named_modules()
        if name in audited_set and type(module).__name__ == "QLinear"
    ]
    if set(audited_module_order) != audited_set:
        raise RuntimeError("audited QLinear paths differ from the execution model")
    selected = [] if compare_first else list(requested_paths)
    if not compare_first and (
        not selected
        or any(not isinstance(path, str) or not path for path in selected)
    ):
        raise ValueError("same-activation QLinear paths must be non-empty strings")
    if len(selected) != len(set(selected)):
        raise ValueError("same-activation QLinear paths must not repeat")
    unknown = sorted(set(selected) - audited_set)
    if unknown:
        raise ValueError(
            "requested QLinear paths are not in the quant assembly audit: "
            + ", ".join(unknown)
        )

    hook_paths = audited_module_order if compare_first else selected
    activations: dict[str, Tensor] = {}
    outputs: dict[str, Tensor] = {}
    calls = {path: 0 for path in hook_paths}
    handles: list[torch.utils.hooks.RemovableHandle] = []
    first_runtime_path: str | None = None

    def capture_input(path: str):
        def hook(_module: nn.Module, values: tuple[object, ...]) -> None:
            nonlocal first_runtime_path
            if compare_first:
                if first_runtime_path is None:
                    first_runtime_path = path
                    selected.append(path)
                if path != first_runtime_path:
                    return
            if calls[path] != 0:
                raise RuntimeError(f"QLinear {path!r} executed more than once")
            if not values or not isinstance(values[0], Tensor):
                raise TypeError(f"QLinear {path!r} did not receive a Tensor input")
            calls[path] = 1
            activations[path] = values[0].detach().clone()

        return hook

    def capture_output(path: str):
        def hook(_module: nn.Module, _values: tuple[object, ...], value: object) -> None:
            if compare_first and path != first_runtime_path:
                return
            if not isinstance(value, Tensor):
                raise TypeError(f"QLinear {path!r} did not return a Tensor")
            outputs[path] = value.detach().clone()

        return hook

    try:
        for path in hook_paths:
            module = modules[path]
            handles.append(module.register_forward_pre_hook(capture_input(path)))
            handles.append(module.register_forward_hook(capture_output(path)))
        _execute(target, prefix, features=False)
    finally:
        for handle in handles:
            handle.remove()

    if compare_first and first_runtime_path is None:
        raise RuntimeError("the quant Target executed no audited QLinear module")

    from .w8a8_emulation import compare_formula_output, emulate_w8a8_linear

    records: list[dict[str, object]] = []
    for path in selected:
        if calls[path] != 1 or path not in activations or path not in outputs:
            raise RuntimeError(f"QLinear {path!r} was not captured exactly once")
        module = modules[path]
        weight = getattr(module, "W_q", None)
        scale = getattr(module, "scale", None)
        if not isinstance(weight, Tensor) or not isinstance(scale, Tensor):
            raise TypeError(f"QLinear {path!r} lost W_q/scale tensors")
        activation = activations.pop(path)
        npu_output = outputs.pop(path)
        if not bool(torch.isfinite(activation).all().item()):
            raise FloatingPointError(f"QLinear {path!r} activation is non-finite")
        if not bool(torch.isfinite(npu_output).all().item()):
            raise FloatingPointError(f"QLinear {path!r} output is non-finite")
        activation_cpu = activation.to(device="cpu")
        npu_output_cpu = npu_output.to(device="cpu")
        formula_output = emulate_w8a8_linear(
            activation_cpu,
            weight.detach().to(device="cpu"),
            scale.detach().to(device="cpu"),
            output_dtype=torch.float16,
        )
        comparison = compare_formula_output(npu_output_cpu, formula_output)
        records.append(
            {
                "path": path,
                "activation_shape": list(activation_cpu.shape),
                "activation_dtype": str(activation_cpu.dtype),
                "weight_shape": list(weight.shape),
                "weight_dtype": str(weight.dtype),
                "scale_shape": list(scale.shape),
                "scale_dtype": str(scale.dtype),
                "npu_output_vs_cpu_formula": comparison,
            }
        )
        del activation, npu_output, activation_cpu, npu_output_cpu, formula_output

    bitwise = all(
        bool(record["npu_output_vs_cpu_formula"]["bitwise_equal"])
        for record in records
        if isinstance(record["npu_output_vs_cpu_formula"], Mapping)
    )
    return {
        "status": (
            "PASS_BITWISE_EQUAL"
            if bitwise
            else "OBSERVED_NUMERICAL_DIFFERENCE"
        ),
        "target_forward_calls": 1,
        "all_bitwise_equal": bitwise,
        "selected_by": "first_runtime_forward" if compare_first else "explicit_paths",
        "comparisons": records,
        "scope": (
            "same NPU activation and output versus the documented CPU W8A8 "
            "formula; no whole-model or performance claim"
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
    quantization_paths: Sequence[Path],
) -> Path:
    if destination.is_symlink():
        raise ValueError("--report must not be a symlink")
    resolved = destination.resolve()
    package_root = Path(__file__).resolve().parents[2]
    protected_roots = [target_root, package_root]
    protected_roots.extend(path for path in quantization_paths if path.is_dir())
    if resolved in quantization_paths or any(
        _is_within(resolved, root) for root in protected_roots
    ):
        raise ValueError(
            "--report must be outside the target, source package, and all "
            "quantization data paths"
        )
    return resolved


def _validate_export_destination(
    destination: Path,
    *,
    target_root: Path,
    quantization_paths: Sequence[Path],
    report_destination: Path | None,
) -> Path:
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            "--export-w8a8-emulation-artifact must name a new directory"
        )
    resolved = destination.resolve()
    package_root = Path(__file__).resolve().parents[2]
    protected_roots = [target_root, package_root]
    protected_roots.extend(path for path in quantization_paths if path.is_dir())
    if resolved in quantization_paths or any(
        _is_within(resolved, root) for root in protected_roots
    ):
        raise ValueError(
            "W8A8 emulation export must be outside the target, source package, "
            "and quantization data paths"
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
    if normalized.get("linear_topology_validation") != "PASS_EXACT_PATH_SHAPE_BIAS":
        raise RuntimeError(
            "quant target did not pass exact Linear path/shape/bias validation"
        )
    if normalized.get("quantized_weight_layout") != "K_by_N":
        raise RuntimeError("quant target did not prove the QLinear K-by-N layout")
    return audit, normalized


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not str(args.device).startswith("npu"):
        raise ValueError("quant target preflight requires --device npu or npu:N")
    if args.kv_cache_max_len <= 0 or args.kv_cache_max_len % 64:
        raise ValueError("--kv-cache-max-len must be positive and divisible by 64")
    if args.require_qlinear_bitwise and not (
        args.compare_first_qlinear or args.compare_qlinear_path
    ):
        raise ValueError(
            "--require-qlinear-bitwise requires --compare-first-qlinear or "
            "--compare-qlinear-path"
        )
    tokens = _prompt_ids(args.prompt_ids)
    target_root = Path(args.target_dir).expanduser().resolve()
    if not target_root.is_dir():
        raise FileNotFoundError(f"target directory does not exist: {target_root}")

    quant_args = argparse.Namespace(
        target_quant_mode=QUANT_MODE_W8A8_DYNAMIC,
        target_factory=DEFAULT_TARGET_FACTORY,
        target_quantizer=args.target_quantizer,
        target_quant_weight_path=args.target_quant_weight_path,
        target_input_provider=args.target_input_provider,
        target_embedding_weight_path=args.target_embedding_weight_path,
        target_embedding_scale_path=args.target_embedding_scale_path,
        report=args.report,
    )
    _configure_target_quantization(quant_args)
    quantization_paths = (
        Path(args.target_quant_weight_path).expanduser().resolve(),
        Path(args.target_embedding_weight_path).expanduser().resolve(),
        Path(args.target_embedding_scale_path).expanduser().resolve(),
    )
    report_destination = (
        None
        if args.report is None
        else _validate_report_destination(
            Path(args.report).expanduser(),
            target_root=target_root,
            quantization_paths=quantization_paths,
        )
    )
    export_destination = (
        None
        if args.export_w8a8_emulation_artifact is None
        else _validate_export_destination(
            Path(args.export_w8a8_emulation_artifact).expanduser(),
            target_root=target_root,
            quantization_paths=quantization_paths,
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
    qlinear_paths = initial_quant.get("qlinear_paths")
    if not isinstance(qlinear_paths, list) or any(
        not isinstance(path, str) for path in qlinear_paths
    ):
        raise TypeError("quant target audit did not expose qlinear_paths")
    same_activation = _same_activation_qlinear_comparison(
        target,
        prefix,
        audited_paths=qlinear_paths,
        compare_first=bool(args.compare_first_qlinear),
        requested_paths=tuple(args.compare_qlinear_path),
    )
    probes = _run_bounded_probes(target, prefix)
    rollback_probe = _run_rollback_target_probe(target, prefix)
    final_audit, final_quant = _quantization_audit(target)

    expected_calls = int(probes["target_forward_calls"]) + int(
        same_activation["target_forward_calls"]
    )
    initial_bridge = initial_audit.get("bridge_runtime")
    bridge = final_audit.get("bridge_runtime")
    if not isinstance(initial_bridge, Mapping) or not isinstance(bridge, Mapping):
        raise TypeError("target facade bridge audit must be a mapping")
    for label, initial, final in (
        (
            "facade target_forward_calls",
            initial_audit.get("target_forward_calls"),
            final_audit.get("target_forward_calls"),
        ),
        (
            "bridge full_prefix_calls",
            initial_bridge.get("full_prefix_calls"),
            bridge.get("full_prefix_calls"),
        ),
    ):
        if (
            isinstance(initial, bool)
            or not isinstance(initial, int)
            or isinstance(final, bool)
            or not isinstance(final, int)
        ):
            raise TypeError(f"{label} counters must be integers")
        if final - initial != expected_calls:
            raise RuntimeError(
                f"{label} delta must equal executed target calls {expected_calls}; "
                f"got initial={initial}, final={final}"
            )
    expected_provider_calls = expected_calls + int(
        rollback_probe["input_provider_calls"]
    )
    for label in ("input_provider_calls", "input_provider_successes"):
        initial = initial_quant.get(label)
        final = final_quant.get(label)
        if (
            isinstance(initial, bool)
            or not isinstance(initial, int)
            or isinstance(final, bool)
            or not isinstance(final, int)
        ):
            raise TypeError(f"{label} counters must be integers")
        if final - initial != expected_provider_calls:
            raise RuntimeError(
                f"{label} delta must equal all full-prefix and rollback calls "
                f"{expected_provider_calls}; got initial={initial}, final={final}"
            )
    initial_failures = initial_quant.get("input_provider_failures")
    final_failures = final_quant.get("input_provider_failures")
    if (
        isinstance(initial_failures, bool)
        or not isinstance(initial_failures, int)
        or isinstance(final_failures, bool)
        or not isinstance(final_failures, int)
    ):
        raise TypeError("input_provider_failures counters must be integers")
    if final_failures - initial_failures != 0:
        raise RuntimeError("quant target input provider reported new failures")

    emulation_export: dict[str, object] | None = None
    if export_destination is not None:
        from .w8a8_emulation import export_w8a8_emulation_artifact

        emulation_export = export_w8a8_emulation_artifact(
            target,
            export_destination,
            expected_qlinear_paths=qlinear_paths,
        )

    same_activation_pass = same_activation.get("status") == "PASS_BITWISE_EQUAL"
    strict_comparison_failure = bool(
        args.require_qlinear_bitwise and not same_activation_pass
    )
    remaining_gates = [
        "quant_target_plus_dflash_strict_greedy_exact_match",
        "real_npu_operator_trace_and_no_fallback",
        "acceptance_and_performance_measurement",
    ]
    if same_activation.get("status") == "DISABLED":
        remaining_gates.insert(
            0,
            "same_activation_real_npu_qlinear_vs_cpu_cuda_formula_parity",
        )
    elif not same_activation_pass:
        remaining_gates.insert(
            0,
            "review_same_activation_qlinear_numerical_difference",
        )

    payload: dict[str, Any] = {
        "schema_version": 3,
        "status": (
            "FAIL_SAME_ACTIVATION_QLINEAR_NOT_BITWISE"
            if strict_comparison_failure
            else "PASS_TARGET_QUANT_ASSEMBLY_PREFIX_AND_ROLLBACK_PROBES"
        ),
        "classification": "TARGET_ONLY_NO_DFLASH_DRAFT",
        "device": str(device),
        "dtype": str(torch.float16),
        "draft_checkpoint_read": False,
        "prompt_token_count": len(tokens),
        "target_quantization_initial": initial_quant,
        "target_quantization_final": final_quant,
        "target_isolation_initial": initial_audit,
        "target_isolation_final": final_audit,
        "same_activation_qlinear": same_activation,
        "bounded_probes": probes,
        "rollback_target_probe": rollback_probe,
        "w8a8_emulation_export": emulation_export,
        "remaining_gates": remaining_gates,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.report is None:
        print(serialized)
    else:
        assert report_destination is not None
        _write_report(payload, report_destination)
        marker = (
            "TARGET_QUANT_PREFLIGHT_FAIL"
            if strict_comparison_failure
            else "TARGET_QUANT_PREFLIGHT_PASS"
        )
        print(f"{marker} report={report_destination}")
    return 2 if strict_comparison_failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
