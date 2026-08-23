"""Diagnose low DFlash acceptance on CPU, CUDA, or the embedded NPU route.

This command is intentionally read-only with respect to model weights and the
deployed source tree.  It answers three questions in order:

1. Does a fresh full-prefix target call produce the same last-row logits and
   DFlash features as a persistent prefill/decode path on the same device?
2. On identical ordinary-greedy prefixes, how does acceptance change for
   proposal counts K=1,4,8,16?
3. At which measured boundary do two device/dtype reports first diverge?

The first question is more fundamental.  Strict-greedy token equality alone
cannot detect a target bridge whose fresh-prefill semantics differ from the
established incremental path, because both ordinary and DFlash validation can
otherwise use the same incorrect full-prefix route.

The official V1 draft predicts all masked rows in one parallel forward.  This
tool therefore traces that single forward; it deliberately does not add an
iterative mask-replacement loop.

By default the report contains non-plaintext hashes and aggregate numbers, but
not prompt or generated token IDs.  Raw IDs are included only when
``--include-token-ids`` is explicitly supplied.  Full tensor bundles for an
independent single-round oracle are also opt-in.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn

from .dflash_ops import TorchDFlashOps
from .dflash_qwen_adapter_v1 import (
    Qwen35DFlashFullPrefixAdapter,
    _audit_target_config,
    _draft_device_memory_preflight,
    _load_target,
    _prepare_device_backend,
    _prompt_ids,
    _select_draft_ops,
    _tokenize_prompt_text,
    _validate_experiment_dtype,
)
from .dflash_weights import require_official_dflash_checkpoint, sha256_file
from .internal_target_loader import (
    DECODE_CHUNK_SIZE_ENV,
    PREFILL_CHUNK_SIZE_ENV,
    TARGET_FACTORY_ENV,
)
from .modeling_dflash import DFlashDraftModel


DEFAULT_TARGET_FACTORY = "models.internal_dflash_bridge:load_qwen35_target"
KV_CACHE_MAX_LEN_ENV = "DFLASH_HIAI_KV_CACHE_MAX_LEN"
OFFICIAL_EOS_TOKEN_ID = 248_044
SUPPORTED_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def tensor_fingerprint(tensor: Tensor) -> dict[str, object]:
    """Return a non-plaintext exact hash plus bounded numerical health data.

    The SHA-256 includes dtype and shape, so it is suitable for exact same-
    dtype GPU/NPU report comparison.  Cross-dtype runs are intentionally not
    reported as exact matches; their acceptance metrics and numerical health
    remain useful for an FP16/BF16 A/B experiment.
    """

    if not isinstance(tensor, Tensor):
        raise TypeError("tensor_fingerprint input must be torch.Tensor")
    value = tensor.detach().contiguous().to("cpu")
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(value.view(torch.uint8).numpy().tobytes())
    finite = bool(torch.isfinite(value).all().item())
    result: dict[str, object] = {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": digest.hexdigest(),
        "finite": finite,
        "numel": int(value.numel()),
    }
    if value.numel() == 0 or not finite:
        return result
    numeric = value.float()
    result.update(
        {
            "rms": float(torch.sqrt(numeric.square().mean()).item()),
            "mean": float(numeric.mean().item()),
            "std": float(numeric.std(unbiased=False).item()),
            "min": float(numeric.min().item()),
            "max": float(numeric.max().item()),
            "zero_fraction": float((numeric == 0).float().mean().item()),
        }
    )
    return result


def _feature_fingerprints(
    features: Tensor,
    layer_ids: Sequence[int],
    hidden_size: int,
) -> dict[str, object]:
    expected_width = len(layer_ids) * hidden_size
    if features.ndim != 3 or int(features.shape[-1]) != expected_width:
        raise ValueError(
            f"feature tensor must have width {expected_width}, got {tuple(features.shape)}"
        )
    return {
        str(layer_id): tensor_fingerprint(
            features[..., offset * hidden_size : (offset + 1) * hidden_size]
        )
        for offset, layer_id in enumerate(layer_ids)
    }


def parse_proposal_counts(raw: str, *, maximum: int = 16) -> tuple[int, ...]:
    """Parse sorted, unique proposal counts from a comma-separated string."""

    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        raise ValueError("maximum proposal count must be positive")
    pieces = [piece.strip() for piece in str(raw).split(",") if piece.strip()]
    if not pieces:
        raise ValueError("--proposal-counts must contain at least one integer")
    result: list[int] = []
    for piece in pieces:
        try:
            value = int(piece)
        except ValueError as error:
            raise ValueError(
                "--proposal-counts must be comma-separated integers"
            ) from error
        if not 1 <= value <= maximum:
            raise ValueError(
                f"proposal count {value} is outside the supported range [1,{maximum}]"
            )
        result.append(value)
    if len(set(result)) != len(result):
        raise ValueError("--proposal-counts must not contain duplicates")
    if result != sorted(result):
        raise ValueError("--proposal-counts must be in ascending order")
    return tuple(result)


def tensor_metrics(reference: Tensor, candidate: Tensor) -> dict[str, object]:
    """Return finite, scale, and error metrics without retaining tensor data."""

    if not isinstance(reference, Tensor) or not isinstance(candidate, Tensor):
        raise TypeError("tensor_metrics inputs must be torch.Tensor")
    if tuple(reference.shape) != tuple(candidate.shape):
        return {
            "shape_match": False,
            "reference_shape": list(reference.shape),
            "candidate_shape": list(candidate.shape),
        }
    if reference.device != candidate.device:
        return {
            "shape_match": True,
            "device_match": False,
            "reference_device": str(reference.device),
            "candidate_device": str(candidate.device),
        }

    reference_finite = bool(torch.isfinite(reference).all().item())
    candidate_finite = bool(torch.isfinite(candidate).all().item())
    result: dict[str, object] = {
        "shape_match": True,
        "device_match": True,
        "dtype_match": reference.dtype == candidate.dtype,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "reference_finite": reference_finite,
        "candidate_finite": candidate_finite,
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }
    if not reference_finite or not candidate_finite:
        return result

    left = reference.detach().float().reshape(-1)
    right = candidate.detach().float().reshape(-1)
    difference = left - right
    absolute = difference.abs()
    left_rms = torch.sqrt(torch.mean(left.square()))
    right_rms = torch.sqrt(torch.mean(right.square()))
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator.item()) == 0.0:
        cosine = 1.0 if bool(torch.equal(left, right)) else 0.0
    else:
        cosine = float((torch.dot(left, right) / denominator).item())
    rmse = float(torch.sqrt(torch.mean(difference.square())).item())
    reference_rms = float(left_rms.item())
    result.update(
        {
            "reference_rms": reference_rms,
            "candidate_rms": float(right_rms.item()),
            "max_abs_error": float(absolute.max().item()),
            "mean_abs_error": float(absolute.mean().item()),
            "rmse": rmse,
            "relative_rmse": rmse / max(reference_rms, 1.0e-12),
            "cosine_similarity": cosine,
        }
    )
    return result


def summarize_acceptance(
    records: Sequence[Mapping[str, object]],
    proposal_counts: Sequence[int],
) -> dict[str, dict[str, object]]:
    """Aggregate per-round acceptance records for each requested K."""

    result: dict[str, dict[str, object]] = {}
    for proposal_count in proposal_counts:
        selected = [
            record
            for record in records
            if int(record["requested_proposal_count"]) == int(proposal_count)
        ]
        proposed = sum(int(record["actual_proposal_count"]) for record in selected)
        accepted = sum(int(record["accepted_count"]) for record in selected)
        emitted = sum(
            int(record.get("theoretical_emitted_count", int(record["accepted_count"]) + 1))
            for record in selected
        )
        first_matches = sum(bool(record["first_proposal_match"]) for record in selected)
        full_blocks = sum(bool(record["full_block_accepted"]) for record in selected)
        histogram = Counter(int(record["accepted_count"]) for record in selected)
        rounds = len(selected)
        result[str(proposal_count)] = {
            "rounds": rounds,
            "proposed_tokens": proposed,
            "accepted_draft_tokens": accepted,
            "draft_token_acceptance_rate": (
                accepted / proposed if proposed else None
            ),
            "first_proposal_accuracy": (
                first_matches / rounds if rounds else None
            ),
            "mean_accepted_draft_tokens": (
                accepted / rounds if rounds else None
            ),
            # Every verification round emits the accepted prefix plus one
            # target correction/bonus token, unless generation stops at EOS.
            "mean_theoretical_emitted_per_verify": (
                emitted / rounds if rounds else None
            ),
            "full_block_accept_rate": full_blocks / rounds if rounds else None,
            "accepted_count_histogram": {
                str(key): histogram[key] for key in sorted(histogram)
            },
        }
    return result


def summarize_acceptance_phases(
    records: Sequence[Mapping[str, object]],
    proposal_counts: Sequence[int],
) -> dict[str, dict[str, dict[str, object]]]:
    """Split completed rounds into equal early/middle/late rank buckets."""

    round_indices = sorted({int(record["round_index"]) for record in records})
    if not round_indices:
        return {str(value): {} for value in proposal_counts}
    phase_names = ("early", "middle", "late")
    phase_by_round = {
        round_index: phase_names[min(2, rank * 3 // len(round_indices))]
        for rank, round_index in enumerate(round_indices)
    }
    result: dict[str, dict[str, dict[str, object]]] = {}
    for proposal_count in proposal_counts:
        by_phase: dict[str, dict[str, object]] = {}
        for phase in phase_names:
            selected = [
                record
                for record in records
                if int(record["requested_proposal_count"]) == int(proposal_count)
                and phase_by_round[int(record["round_index"])] == phase
            ]
            if selected:
                by_phase[phase] = summarize_acceptance(
                    selected, (int(proposal_count),)
                )[str(proposal_count)]
        result[str(proposal_count)] = by_phase
    return result


def _tensor_field(output: object, name: str) -> Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    return value if isinstance(value, Tensor) else None


def _unwrap_logits_features(
    output: object,
    *,
    require_features: bool,
) -> tuple[Tensor, Tensor | None]:
    features = _tensor_field(output, "dflash_features")
    base = getattr(output, "base_output", output)
    logits = _tensor_field(base, "logits")
    if isinstance(base, Tensor):
        logits = base
    elif isinstance(base, (tuple, list)):
        if base and isinstance(base[0], Tensor):
            logits = base[0]
        if require_features and features is None and len(base) > 1:
            candidate = base[1]
            if isinstance(candidate, Tensor):
                features = candidate
    if logits is None and isinstance(output, (tuple, list)):
        if output and isinstance(output[0], Tensor):
            logits = output[0]
        if require_features and features is None and len(output) > 1:
            candidate = output[1]
            if isinstance(candidate, Tensor):
                features = candidate
    if logits is None:
        raise TypeError("target output does not expose Tensor logits")
    if require_features and features is None:
        raise TypeError("target output does not expose dflash_features")
    return logits, features


def _controller_from_facade(target: nn.Module) -> nn.Module:
    controller = getattr(target, "target", None)
    required = (
        "_fresh_hybrid_cache",
        "_fresh_attention_mask",
        "get_input_embeddings",
        "dflash_execution_model",
    )
    if not isinstance(controller, nn.Module) or any(
        not hasattr(controller, name) for name in required
    ):
        raise TypeError(
            "acceptance diagnosis requires the packaged full-prefix facade over "
            "InternalDFlashTarget"
        )
    return controller


def _direct_target_call(
    controller: nn.Module,
    *,
    input_ids: Tensor,
    attention_mask: Tensor,
    position_ids: Tensor,
    cache_positions: Tensor,
    past_key_values: list[tuple[Tensor, Tensor]],
    all_q_len: int,
) -> tuple[Tensor, Tensor]:
    embeddings = controller.get_input_embeddings()
    weight = getattr(embeddings, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError("target embedding module does not expose Tensor weight")
    inputs_embeds = embeddings(input_ids.to(weight.device)).to(input_ids.device)
    execution_model = getattr(controller, "dflash_execution_model")
    if not isinstance(execution_model, nn.Module):
        raise TypeError("DFlash controller lost its execution model")
    with torch.inference_mode():
        output = execution_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            new_kv_cache_pos=cache_positions,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            inputs_embeds=inputs_embeds,
            embed_scale=None,
            output_pos=None,
            allQLen=[int(all_q_len)],
            output_dflash_features=True,
        )
    logits, features = _unwrap_logits_features(output, require_features=True)
    assert features is not None
    if logits.ndim != 3 or features.ndim != 3:
        raise ValueError("incremental target must return rank-3 logits/features")
    return logits, features


def _decode_attention_mask(
    controller: nn.Module,
    *,
    absolute_position: int,
    device: torch.device,
) -> Tensor:
    maximum = int(getattr(controller, "kv_cache_max_len"))
    columns = torch.arange(maximum, device=device)
    visible = columns <= int(absolute_position)
    zero = torch.zeros((), device=device, dtype=torch.float32)
    negative_infinity = torch.full(
        (), float("-inf"), device=device, dtype=torch.float32
    )
    return torch.where(visible, zero, negative_infinity).view(1, 1, 1, maximum)


def _incremental_snapshots(
    target: nn.Module,
    prompt_ids: Tensor,
    *,
    decode_steps: int,
    eos_token_id: int,
) -> list[dict[str, object]]:
    """Run one persistent prefill/decode stream and snapshot only last rows."""

    controller = _controller_from_facade(target)
    device = prompt_ids.device
    prefix = prompt_ids.detach().clone()
    maximum = int(getattr(controller, "kv_cache_max_len"))
    if int(prefix.shape[1]) + decode_steps > maximum:
        raise ValueError(
            "prompt length plus diagnostic decode steps exceeds kv_cache_max_len"
        )
    fresh_cache = getattr(controller, "_fresh_hybrid_cache")
    fresh_mask = getattr(controller, "_fresh_attention_mask")
    past_key_values = fresh_cache(batch_size=1)
    prompt_length = int(prefix.shape[1])
    logits, features = _direct_target_call(
        controller,
        input_ids=prefix,
        attention_mask=fresh_mask(prompt_length),
        position_ids=torch.arange(prompt_length, device=device).unsqueeze(0),
        cache_positions=torch.arange(prompt_length, device=device),
        past_key_values=past_key_values,
        all_q_len=prompt_length,
    )

    snapshots: list[dict[str, object]] = []
    for decode_index in range(decode_steps + 1):
        snapshots.append(
            {
                "prefix": prefix.detach().clone(),
                "incremental_logits": logits[:, -1:, :].detach().clone(),
                "incremental_features": features[:, -1:, :].detach().clone(),
                "source": "prefill" if decode_index == 0 else "decode",
            }
        )
        if decode_index == decode_steps:
            break
        next_token = logits[:, -1, :].argmax(dim=-1).to(torch.long)
        if int(next_token.item()) == eos_token_id:
            break
        absolute_position = int(prefix.shape[1])
        prefix = torch.cat((prefix, next_token.view(1, 1)), dim=1)
        logits, features = _direct_target_call(
            controller,
            input_ids=next_token.view(1, 1),
            attention_mask=_decode_attention_mask(
                controller,
                absolute_position=absolute_position,
                device=device,
            ),
            position_ids=torch.tensor(
                [[absolute_position]], dtype=torch.long, device=device
            ),
            cache_positions=torch.tensor(
                [absolute_position], dtype=torch.long, device=device
            ),
            past_key_values=past_key_values,
            all_q_len=absolute_position + 1,
        )
    return snapshots


def _output_field(output: object, name: str) -> object | None:
    """Read one field through the optional target facade output wrapper."""

    candidates = (output, getattr(output, "base_output", None))
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            value = candidate.get(name)
        else:
            value = getattr(candidate, name, None)
        if value is not None:
            return value
    return None


def _framework_incremental_snapshots(
    target: nn.Module,
    prompt_ids: Tensor,
    *,
    decode_steps: int,
    eos_token_id: int,
) -> list[dict[str, object]]:
    """Run the ordinary Transformers cache path on CPU/CUDA.

    This deliberately feeds only the newly committed token after prefill.  It
    therefore checks the same cached GDN/KV continuation boundary that a
    serving runtime uses, instead of comparing two copies of the cache-free
    full-prefix path.
    """

    prefix = prompt_ids.detach().clone()
    past_key_values: object | None = None

    def call(input_ids: Tensor, state: object | None) -> tuple[Tensor, Tensor, object]:
        with torch.inference_mode():
            output = target(
                input_ids=input_ids,
                past_key_values=state,
                use_cache=True,
                return_dict=True,
                output_hidden_states=False,
                output_dflash_features=True,
                logits_to_keep=1,
            )
        logits, features = _unwrap_logits_features(output, require_features=True)
        assert features is not None
        next_state = _output_field(output, "past_key_values")
        if next_state is None:
            raise RuntimeError(
                "framework target use_cache=True did not return past_key_values"
            )
        if logits.ndim != 3 or features.ndim != 3:
            raise ValueError("framework incremental target must return rank-3 tensors")
        return logits, features, next_state

    logits, features, past_key_values = call(prefix, past_key_values)
    snapshots: list[dict[str, object]] = []
    for decode_index in range(decode_steps + 1):
        snapshots.append(
            {
                "prefix": prefix.detach().clone(),
                "incremental_logits": logits[:, -1:, :].detach().clone(),
                "incremental_features": features[:, -1:, :].detach().clone(),
                "source": "prefill" if decode_index == 0 else "cached_decode",
            }
        )
        if decode_index == decode_steps:
            break
        next_token = logits[:, -1, :].argmax(dim=-1).to(torch.long)
        if int(next_token.item()) == eos_token_id:
            break
        prefix = torch.cat((prefix, next_token.view(1, 1)), dim=1)
        logits, features, past_key_values = call(
            next_token.view(1, 1), past_key_values
        )
    return snapshots


def _full_prefix_output(target: nn.Module, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    with torch.inference_mode():
        output = target(
            input_ids=input_ids,
            use_cache=False,
            return_dict=True,
            output_hidden_states=False,
            output_dflash_features=True,
            logits_to_keep=1,
        )
    logits, features = _unwrap_logits_features(output, require_features=True)
    assert features is not None
    return logits, features


def _hidden_state_sequence(output: object) -> tuple[Tensor, ...] | None:
    """Find a Transformers-style hidden-state tuple without assuming one wrapper."""

    candidates = (output, getattr(output, "base_output", None))
    for candidate in candidates:
        if candidate is None:
            continue
        if isinstance(candidate, Mapping):
            value = candidate.get("hidden_states")
        else:
            value = getattr(candidate, "hidden_states", None)
        if isinstance(value, (tuple, list)) and value and all(
            isinstance(item, Tensor) for item in value
        ):
            return tuple(value)
    return None


def compare_feature_collector_semantics(
    target: nn.Module,
    input_ids: Tensor,
    *,
    layer_ids: Sequence[int],
    hidden_size: int,
    device_type: str,
) -> dict[str, object]:
    """Compare the opt-in collector with the official ``hidden_states[id+1]`` rule.

    The framework target can expose both paths in one forward, making this a
    direct check of layer numbering, capture point, ordering, and concatenation.
    The compact NPU target intentionally does not expose every intermediate
    hidden state, so its equivalent check is a same-prompt report comparison
    against a framework run.
    """

    if device_type == "npu":
        return {
            "status": "NOT_APPLICABLE_COMPACT_NPU_OUTPUT",
            "indexing_contract": "decoder_output[layer_id] == hidden_states[layer_id + 1]",
            "message": (
                "compare the NPU target_feature_layer fingerprints with a same-dtype "
                "framework report"
            ),
        }

    try:
        with torch.inference_mode():
            output = target(
                input_ids=input_ids,
                use_cache=False,
                return_dict=True,
                output_hidden_states=True,
                output_dflash_features=True,
                logits_to_keep=1,
            )
    except Exception as error:
        return {
            "status": "FAIL_FEATURE_SEMANTICS_FORWARD",
            "error_type": type(error).__name__,
            "message": str(error).splitlines()[0][:300],
        }

    _logits, collected = _unwrap_logits_features(output, require_features=True)
    assert collected is not None
    hidden_states = _hidden_state_sequence(output)
    if hidden_states is None:
        return {
            "status": "FAIL_HIDDEN_STATES_UNAVAILABLE",
            "indexing_contract": "decoder_output[layer_id] == hidden_states[layer_id + 1]",
        }
    required = max(int(layer_id) for layer_id in layer_ids) + 2
    if len(hidden_states) < required:
        return {
            "status": "FAIL_HIDDEN_STATE_COUNT",
            "hidden_state_count": len(hidden_states),
            "required_hidden_state_count": required,
        }

    selected = tuple(hidden_states[int(layer_id) + 1] for layer_id in layer_ids)
    expected = torch.cat(selected, dim=-1)
    if expected.shape[-1] != len(layer_ids) * hidden_size:
        return {
            "status": "FAIL_EXPECTED_FEATURE_WIDTH",
            "expected_shape": list(expected.shape),
            "required_width": len(layer_ids) * hidden_size,
        }
    layer_metrics: dict[str, object] = {}
    for offset, (layer_id, hidden) in enumerate(zip(layer_ids, selected, strict=True)):
        start = offset * hidden_size
        layer_metrics[str(layer_id)] = tensor_metrics(
            hidden,
            collected[..., start : start + hidden_size],
        )
    combined = tensor_metrics(expected, collected)
    exact = bool(combined.get("bitwise_equal"))
    return {
        "status": "PASS_BITWISE_EQUAL" if exact else "FAIL_COLLECTOR_SEMANTICS_MISMATCH",
        "indexing_contract": "decoder_output[layer_id] == hidden_states[layer_id + 1]",
        "layer_ids": [int(layer_id) for layer_id in layer_ids],
        "hidden_state_count": len(hidden_states),
        "combined": combined,
        "layers": layer_metrics,
    }


def _compare_target_snapshots(
    target: nn.Module,
    snapshots: Sequence[Mapping[str, object]],
    *,
    layer_ids: Sequence[int],
    hidden_size: int,
    include_token_ids: bool,
    incremental_path: str,
) -> dict[str, object]:
    """Compare captured incremental rows with independent full-prefix calls."""

    records: list[dict[str, object]] = []
    for index, snapshot in enumerate(snapshots):
        prefix = snapshot["prefix"]
        assert isinstance(prefix, Tensor)
        full_logits, full_features = _full_prefix_output(target, prefix)
        incremental_logits = snapshot["incremental_logits"]
        incremental_features = snapshot["incremental_features"]
        assert isinstance(incremental_logits, Tensor)
        assert isinstance(incremental_features, Tensor)
        full_logits_row = full_logits[:, -1:, :].detach().clone()
        full_feature_row = full_features[:, -1:, :].detach().clone()
        incremental_top1 = int(incremental_logits.argmax(dim=-1).item())
        full_top1 = int(full_logits_row.argmax(dim=-1).item())
        expected_width = len(layer_ids) * hidden_size
        if int(full_feature_row.shape[-1]) != expected_width:
            raise ValueError(
                f"feature width {full_feature_row.shape[-1]} does not equal "
                f"{len(layer_ids)} x {hidden_size}"
            )
        layer_metrics: dict[str, object] = {}
        for layer_offset, layer_id in enumerate(layer_ids):
            start = layer_offset * hidden_size
            end = start + hidden_size
            layer_metrics[str(layer_id)] = tensor_metrics(
                incremental_features[..., start:end],
                full_feature_row[..., start:end],
            )
        record: dict[str, object] = {
            "comparison_index": index,
            "incremental_source": snapshot["source"],
            "prefix_length": int(prefix.shape[1]),
            "top1_match": incremental_top1 == full_top1,
            "logits": tensor_metrics(incremental_logits, full_logits_row),
            "features": tensor_metrics(incremental_features, full_feature_row),
            "feature_layers": layer_metrics,
        }
        if include_token_ids:
            record["prefix_token_ids"] = prefix.detach().cpu().reshape(-1).tolist()
            record["incremental_top1"] = incremental_top1
            record["full_prefix_top1"] = full_top1
        records.append(record)

    all_top1 = all(bool(record["top1_match"]) for record in records)
    all_feature_exact = all(
        bool(record["features"].get("bitwise_equal"))
        for record in records
        if isinstance(record["features"], Mapping)
    )
    all_logits_exact = all(
        bool(record["logits"].get("bitwise_equal"))
        for record in records
        if isinstance(record["logits"], Mapping)
    )
    if not all_top1:
        status = "FAIL_TOP1_DIVERGENCE"
    elif all_feature_exact and all_logits_exact:
        status = "PASS_BITWISE_EQUAL"
    else:
        status = "PASS_TOP1_WITH_NUMERIC_DIFFERENCE"
    feature_relative_rmse = [
        float(record["features"]["relative_rmse"])
        for record in records
        if isinstance(record.get("features"), Mapping)
        and isinstance(record["features"].get("relative_rmse"), (int, float))
    ]
    feature_cosines = [
        float(record["features"]["cosine_similarity"])
        for record in records
        if isinstance(record.get("features"), Mapping)
        and isinstance(record["features"].get("cosine_similarity"), (int, float))
    ]
    return {
        "status": status,
        "incremental_path": incremental_path,
        "comparisons": len(records),
        "all_top1_match": all_top1,
        "all_logits_bitwise_equal": all_logits_exact,
        "all_features_bitwise_equal": all_feature_exact,
        "max_feature_relative_rmse": (
            max(feature_relative_rmse) if feature_relative_rmse else None
        ),
        "min_feature_cosine_similarity": (
            min(feature_cosines) if feature_cosines else None
        ),
        "records": records,
    }


def compare_target_paths(
    target: nn.Module,
    prompt_ids: Tensor,
    *,
    decode_steps: int,
    eos_token_id: int,
    layer_ids: Sequence[int],
    hidden_size: int,
    include_token_ids: bool,
) -> dict[str, object]:
    """Compare the NPU persistent state path with fresh full-prefix replay."""

    snapshots = _incremental_snapshots(
        target,
        prompt_ids,
        decode_steps=decode_steps,
        eos_token_id=eos_token_id,
    )
    return _compare_target_snapshots(
        target,
        snapshots,
        layer_ids=layer_ids,
        hidden_size=hidden_size,
        include_token_ids=include_token_ids,
        incremental_path="receiver_hiai_prefill_decode",
    )


def compare_framework_target_paths(
    target: nn.Module,
    prompt_ids: Tensor,
    *,
    decode_steps: int,
    eos_token_id: int,
    layer_ids: Sequence[int],
    hidden_size: int,
    include_token_ids: bool,
) -> dict[str, object]:
    """Compare CPU/CUDA cached decode with independent full-prefix replay."""

    try:
        snapshots = _framework_incremental_snapshots(
            target,
            prompt_ids,
            decode_steps=decode_steps,
            eos_token_id=eos_token_id,
        )
        return _compare_target_snapshots(
            target,
            snapshots,
            layer_ids=layer_ids,
            hidden_size=hidden_size,
            include_token_ids=include_token_ids,
            incremental_path="transformers_dynamic_cache_prefill_decode",
        )
    except Exception as error:
        return {
            "status": "FAIL_FRAMEWORK_INCREMENTAL_PATH",
            "incremental_path": "transformers_dynamic_cache_prefill_decode",
            "comparisons": 0,
            "all_top1_match": False,
            "all_logits_bitwise_equal": False,
            "all_features_bitwise_equal": False,
            "records": [],
            "error_type": type(error).__name__,
            "message": str(error).splitlines()[0][:300],
        }


def _draft_attention_contract(
    adapter: Qwen35DFlashFullPrefixAdapter,
    *,
    context_length: int,
    block_length: int,
) -> dict[str, object]:
    """Check each local attention mask against the vLLM 0.27.1 rule."""

    key_length = context_length + block_length
    device = adapter.device
    query_positions = context_length + torch.arange(
        block_length, device=device
    ).view(block_length, 1)
    key_positions = torch.arange(key_length, device=device).view(1, key_length)
    records: list[dict[str, object]] = []
    all_exact = True
    for layer_index, layer in enumerate(adapter.draft.layers):
        attention = layer.self_attn
        actual = attention._attention_mask(
            block_length,
            context_length,
            device=device,
        )
        expected_is_causal = (
            adapter.draft.config.layer_types[layer_index] == "sliding_attention"
        )
        expected_window = (
            int(adapter.draft.config.sliding_window)
            if adapter.draft.config.layer_types[layer_index] == "sliding_attention"
            else None
        )
        if expected_is_causal or expected_window is not None:
            expected = torch.ones(
                (block_length, key_length), dtype=torch.bool, device=device
            )
            if expected_is_causal:
                expected &= key_positions <= query_positions
            if expected_window is not None:
                expected &= query_positions - key_positions < expected_window
                if not expected_is_causal:
                    expected &= key_positions - query_positions < expected_window
            expected = expected.view(1, 1, block_length, key_length)
            exact = isinstance(actual, Tensor) and bool(torch.equal(actual, expected))
            block_visibility = (
                actual[0, 0, :, context_length:]
                if isinstance(actual, Tensor)
                else None
            )
            mask_kind = (
                "causal_sliding" if expected_is_causal else "bidirectional_sliding"
            )
        else:
            exact = actual is None
            block_visibility = torch.ones(
                (block_length, block_length), dtype=torch.bool, device=device
            )
            mask_kind = "bidirectional_full"
        exact = bool(
            exact
            and attention.is_causal is expected_is_causal
            and attention.sliding_window == expected_window
        )
        all_exact &= exact
        records.append(
            {
                "layer_index": layer_index,
                "layer_type": adapter.draft.config.layer_types[layer_index],
                "mask_kind": mask_kind,
                "is_causal": bool(attention.is_causal),
                "sliding_window": attention.sliding_window,
                "mask_matches_pinned_rule": exact,
                "first_query_visible_block_rows": (
                    int(block_visibility[0].sum().item())
                    if isinstance(block_visibility, Tensor)
                    else None
                ),
                "last_query_visible_block_rows": (
                    int(block_visibility[-1].sum().item())
                    if isinstance(block_visibility, Tensor)
                    else None
                ),
            }
        )
    return {
        "status": (
            "PASS_VLLM_0_27_1_V1_MASKS"
            if all_exact
            else "FAIL_ATTENTION_MASK_CONTRACT"
        ),
        "semantics": "causal_sliding_layers_then_noncausal_full_layer",
        "context_length": context_length,
        "block_length": block_length,
        "layers": records,
    }


def _draft_input_contract(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    target_hidden: Tensor,
    block_ids: Tensor,
    noise_embedding: Tensor,
    position_ids: Tensor,
    proposal_count: int,
) -> dict[str, object]:
    block_length = proposal_count + 1
    expected_positions = torch.arange(
        int(target_hidden.shape[1]) + block_length,
        dtype=torch.long,
        device=adapter.device,
    ).unsqueeze(0)
    checks = {
        "context_excludes_anchor": (
            int(target_hidden.shape[1]) == int(prefix_ids.shape[1]) - 1
        ),
        "block_length_is_anchor_plus_k": tuple(block_ids.shape) == (1, block_length),
        "anchor_matches_committed_prefix_tail": bool(
            torch.equal(block_ids[:, :1], prefix_ids[:, -1:])
        ),
        "proposal_rows_are_mask_token": bool(
            torch.all(
                block_ids[:, 1:] == int(adapter.draft.config.mask_token_id)
            ).item()
        ),
        "position_ids_are_contiguous_absolute": bool(
            torch.equal(position_ids, expected_positions)
        ),
        "noise_embedding_shape": tuple(noise_embedding.shape)
        == (1, block_length, int(adapter.draft.config.hidden_size)),
        "feature_width": int(target_hidden.shape[-1])
        == int(adapter.draft.config.feature_size),
        "feature_noise_dtype_match": target_hidden.dtype == noise_embedding.dtype,
        "feature_noise_device_match": target_hidden.device == noise_embedding.device,
    }
    attention = _draft_attention_contract(
        adapter,
        context_length=int(target_hidden.shape[1]),
        block_length=block_length,
    )
    passed = (
        all(checks.values())
        and attention["status"] == "PASS_VLLM_0_27_1_V1_MASKS"
    )
    return {
        "status": "PASS_OFFICIAL_SINGLE_FORWARD_INPUT" if passed else "FAIL_DRAFT_INPUT_CONTRACT",
        "proposal_count": proposal_count,
        "block_length": block_length,
        "context_length": int(target_hidden.shape[1]),
        "mask_token_id": int(adapter.draft.config.mask_token_id),
        "checks": checks,
        "block_token_fingerprint": tensor_fingerprint(block_ids),
        "position_id_fingerprint": tensor_fingerprint(position_ids),
        "attention": attention,
    }


def _build_draft_inputs(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    target_hidden: Tensor,
    proposal_count: int,
) -> tuple[Tensor, Tensor, Tensor, dict[str, object]]:
    block_length = proposal_count + 1
    block_ids = torch.full(
        (1, block_length),
        int(adapter.draft.config.mask_token_id),
        dtype=torch.long,
        device=adapter.device,
    )
    block_ids[:, 0] = prefix_ids[:, -1]
    noise_embedding = adapter.draft.embed_block(
        block_ids, adapter.input_embedding_weight
    )
    total_positions = int(target_hidden.shape[1]) + block_length
    position_ids = torch.arange(
        total_positions, dtype=torch.long, device=adapter.device
    ).unsqueeze(0)
    contract = _draft_input_contract(
        adapter,
        prefix_ids,
        target_hidden,
        block_ids,
        noise_embedding,
        position_ids,
        proposal_count,
    )
    if contract["status"] != "PASS_OFFICIAL_SINGLE_FORWARD_INPUT":
        raise RuntimeError("DFlash draft input no longer matches the pinned V1 contract")
    return block_ids, noise_embedding, position_ids, contract


def _propose_with_features(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    target_hidden: Tensor,
    proposal_count: int,
) -> tuple[Tensor, dict[str, object]]:
    _block_ids, noise_embedding, position_ids, input_contract = _build_draft_inputs(
        adapter, prefix_ids, target_hidden, proposal_count
    )
    with torch.inference_mode():
        proposals = adapter.draft.draft_top1(
            target_hidden,
            noise_embedding,
            position_ids,
            adapter.lm_head_weight,
        )
    expected = (1, proposal_count)
    if tuple(proposals.shape) != expected:
        raise RuntimeError(
            f"DFlash returned proposals with shape {tuple(proposals.shape)}, "
            f"expected {expected}"
        )
    return proposals.to(torch.long), input_contract


def _trace_draft_forward(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    target_hidden: Tensor,
    proposal_count: int,
    *,
    capture_tensors: bool,
    target_trace: Mapping[str, object] | None = None,
) -> tuple[Tensor, dict[str, object], dict[str, Tensor]]:
    """Run the normal one-pass draft while tracing stable model boundaries."""

    block_ids, noise_embedding, position_ids, input_contract = _build_draft_inputs(
        adapter, prefix_ids, target_hidden, proposal_count
    )
    if target_trace is None:
        target_trace = {
            "target_hidden": tensor_fingerprint(target_hidden),
            "target_feature_layers": _feature_fingerprints(
                target_hidden,
                adapter.draft.config.target_layer_ids,
                adapter.draft.config.hidden_size,
            ),
        }
    trace: dict[str, object] = {
        "target_hidden": target_trace["target_hidden"],
        "target_feature_layers": target_trace["target_feature_layers"],
        "noise_embedding": tensor_fingerprint(noise_embedding),
        "position_ids": tensor_fingerprint(position_ids),
        "position_range": {
            "first": int(position_ids[0, 0].item()),
            "context_last": int(target_hidden.shape[1]) - 1,
            "block_first": int(target_hidden.shape[1]),
            "last": int(position_ids[0, -1].item()),
        },
        "input_contract": input_contract,
    }
    oracle: dict[str, Tensor] = {}
    if capture_tensors:
        oracle.update(
            {
                "input.block_ids": block_ids.detach().contiguous().cpu(),
                "input.target_hidden": target_hidden.detach().contiguous().cpu(),
                "input.noise_embedding": noise_embedding.detach().contiguous().cpu(),
                "input.position_ids": position_ids.detach().contiguous().cpu(),
            }
        )

    handles: list[object] = []

    def capture_tensor(name: str):
        def hook(_module: nn.Module, _inputs: object, output: object) -> None:
            if not isinstance(output, Tensor):
                raise TypeError(f"draft trace boundary {name} did not return Tensor")
            trace[name] = tensor_fingerprint(output)
            if capture_tensors:
                oracle[f"trace.{name}"] = output.detach().contiguous().cpu()

        return hook

    def capture_rotary(
        _module: nn.Module,
        _inputs: object,
        output: object,
    ) -> None:
        if (
            not isinstance(output, tuple)
            or len(output) != 2
            or not all(isinstance(item, Tensor) for item in output)
        ):
            raise TypeError("draft rotary trace did not return (cosine, sine)")
        cosine, sine = output
        trace["rotary_cosine"] = tensor_fingerprint(cosine)
        trace["rotary_sine"] = tensor_fingerprint(sine)
        if capture_tensors:
            oracle["trace.rotary_cosine"] = cosine.detach().contiguous().cpu()
            oracle["trace.rotary_sine"] = sine.detach().contiguous().cpu()

    handles.append(adapter.draft.fc.register_forward_hook(capture_tensor("fc_output")))
    handles.append(
        adapter.draft.hidden_norm.register_forward_hook(
            capture_tensor("projected_target_hidden")
        )
    )
    handles.append(adapter.draft.rotary.register_forward_hook(capture_rotary))
    for layer_index, layer in enumerate(adapter.draft.layers):
        handles.append(
            layer.register_forward_hook(capture_tensor(f"draft_layer_{layer_index}"))
        )
    handles.append(
        adapter.draft.norm.register_forward_hook(capture_tensor("final_norm_output"))
    )
    try:
        with torch.inference_mode():
            draft_hidden = adapter.draft.draft_hidden(
                target_hidden,
                noise_embedding,
                position_ids,
            )
            proposals = adapter.draft.ops.top1(
                draft_hidden,
                adapter.lm_head_weight,
            ).to(torch.long)
    finally:
        for handle in handles:
            handle.remove()

    expected = (1, proposal_count)
    if tuple(proposals.shape) != expected:
        raise RuntimeError(
            f"DFlash returned proposals with shape {tuple(proposals.shape)}, "
            f"expected {expected}"
        )
    trace["draft_hidden"] = tensor_fingerprint(draft_hidden)
    trace["proposal_tokens"] = tensor_fingerprint(proposals)
    if capture_tensors:
        oracle["output.draft_hidden"] = draft_hidden.detach().contiguous().cpu()
        oracle["output.top1"] = proposals.detach().contiguous().cpu()
    return proposals, trace, oracle


def _accepted_prefix_length(proposals: Tensor, target_tokens: Tensor) -> int:
    if proposals.ndim != 1 or target_tokens.ndim != 1:
        raise ValueError("acceptance comparison expects rank-1 token tensors")
    if proposals.shape != target_tokens.shape:
        raise ValueError("proposal and target token shapes differ")
    matches = proposals == target_tokens
    mismatch = torch.nonzero(~matches, as_tuple=False)
    return int(proposals.numel()) if mismatch.numel() == 0 else int(mismatch[0].item())


def acceptance_sweep(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prompt_ids: Tensor,
    *,
    proposal_counts: Sequence[int],
    rounds: int,
    eos_token_id: int,
    include_token_ids: bool,
    trace_draft_layers: bool = False,
    oracle_capture: dict[str, Tensor] | None = None,
    verification_mode: str = "sequential",
) -> dict[str, object]:
    """Evaluate all K values on the same sequence of clean target prefixes."""

    if verification_mode not in {"sequential", "vectorized"}:
        raise ValueError("verification_mode must be sequential or vectorized")

    clean_logits = adapter.forward_logits(prompt_ids)
    anchor = clean_logits[:, -1, :].argmax(dim=-1).to(torch.long)
    prefix = torch.cat((prompt_ids, anchor.view(1, 1)), dim=1)
    records: list[dict[str, object]] = []
    stopped_on_eos = int(anchor.item()) == eos_token_id

    for round_index in range(rounds):
        if stopped_on_eos:
            break
        context_hidden = adapter._replay_target_features(prefix[:, :-1])
        prefix_token_sha256 = tensor_fingerprint(prefix)["sha256"]
        round_target_trace = (
            {
                "target_hidden": tensor_fingerprint(context_hidden),
                "target_feature_layers": _feature_fingerprints(
                    context_hidden,
                    adapter.draft.config.target_layer_ids,
                    adapter.draft.config.hidden_size,
                ),
            }
            if trace_draft_layers
            else None
        )
        verifier_first_tokens: dict[str, int] = {}
        for proposal_count in proposal_counts:
            capture_oracle = (
                oracle_capture is not None
                and round_index == 0
                and proposal_count == max(proposal_counts)
            )
            draft_trace: dict[str, object] | None = None
            if trace_draft_layers or capture_oracle:
                proposed, draft_trace, oracle = _trace_draft_forward(
                    adapter,
                    prefix,
                    context_hidden,
                    proposal_count,
                    capture_tensors=capture_oracle,
                    target_trace=round_target_trace,
                )
                if capture_oracle:
                    assert oracle_capture is not None
                    oracle_capture.update(oracle)
                input_contract = draft_trace["input_contract"]
            else:
                proposed, input_contract = _propose_with_features(
                    adapter, prefix, context_hidden, proposal_count
                )
            proposals = proposed.reshape(-1)
            eos_positions = torch.nonzero(
                proposals == int(eos_token_id), as_tuple=False
            )
            if eos_positions.numel():
                proposals = proposals[: int(eos_positions[0].item()) + 1]
            verification_ids = torch.cat((prefix, proposals.view(1, -1)), dim=1)
            verification_logits = adapter.forward_logits(verification_ids)
            first_row = int(prefix.shape[1]) - 1
            row_count = int(proposals.numel())
            vectorized_target_tokens = verification_logits[
                0, first_row : first_row + row_count + 1, :
            ].argmax(dim=-1)
            if int(vectorized_target_tokens.numel()) != row_count + 1:
                raise RuntimeError("target verification did not expose the bonus row")
            if verification_mode == "sequential":
                decision_values: list[int] = []
                accepted_count = 0
                for proposal_index, proposal in enumerate(proposals):
                    isolated_prefix = torch.cat(
                        (prefix, proposals[:proposal_index].view(1, -1)),
                        dim=1,
                    )
                    isolated_logits = adapter.forward_logits(isolated_prefix)
                    isolated_token = int(
                        isolated_logits[0, -1, :].argmax(dim=-1).item()
                    )
                    decision_values.append(isolated_token)
                    if int(proposal.item()) != isolated_token:
                        break
                    accepted_count += 1
                if accepted_count == row_count:
                    isolated_bonus_input = torch.cat(
                        (prefix, proposals.view(1, -1)),
                        dim=1,
                    )
                    isolated_bonus_logits = adapter.forward_logits(
                        isolated_bonus_input
                    )
                    decision_values.append(
                        int(
                            isolated_bonus_logits[0, -1, :]
                            .argmax(dim=-1)
                            .item()
                        )
                    )
                verification_target_tokens = torch.tensor(
                    decision_values,
                    dtype=torch.long,
                    device=proposals.device,
                )
            else:
                verification_target_tokens = vectorized_target_tokens
                accepted_count = _accepted_prefix_length(
                    proposals,
                    vectorized_target_tokens[:row_count],
                )
            target_tokens = verification_target_tokens[
                : min(row_count, int(verification_target_tokens.numel()))
            ]
            verifier_first = int(verification_target_tokens[0].item())
            verifier_first_tokens[str(proposal_count)] = verifier_first
            accepted_contains_eos = bool(
                (proposals[:accepted_count] == int(eos_token_id)).any().item()
            )
            theoretical_emitted_count = accepted_count + (
                0 if accepted_contains_eos else 1
            )
            record: dict[str, object] = {
                "round_index": round_index,
                "round_role": "first_round" if round_index == 0 else "later_round",
                "prefix_length": int(prefix.shape[1]),
                "prefix_token_sha256": prefix_token_sha256,
                "requested_proposal_count": int(proposal_count),
                "actual_proposal_count": row_count,
                "accepted_count": accepted_count,
                "theoretical_emitted_count": theoretical_emitted_count,
                "first_proposal_match": bool(proposals[0] == target_tokens[0]),
                "full_block_accepted": accepted_count == row_count,
                "verification_mode": verification_mode,
                "target_decision_rows_evaluated": int(
                    verification_target_tokens.numel()
                ),
                "draft_input_contract": input_contract,
                "proposal_token_fingerprint": tensor_fingerprint(proposals),
                "target_token_fingerprint": tensor_fingerprint(target_tokens),
                "vectorized_target_token_fingerprint": tensor_fingerprint(
                    vectorized_target_tokens
                ),
                "correction_or_bonus_token_fingerprint": tensor_fingerprint(
                    verification_target_tokens[accepted_count : accepted_count + 1]
                ),
            }
            compared_rows = min(
                int(verification_target_tokens.numel()),
                int(vectorized_target_tokens.numel()),
            )
            divergence = torch.nonzero(
                verification_target_tokens[:compared_rows]
                != vectorized_target_tokens[:compared_rows],
                as_tuple=False,
            )
            record["vectorized_prefix_invariance"] = {
                "status": (
                    "PASS_EVALUATED_ROWS"
                    if divergence.numel() == 0
                    else "FAIL_PREFIX_ROW_DIVERGENCE"
                ),
                "evaluated_rows": compared_rows,
                "first_divergent_row": (
                    None
                    if divergence.numel() == 0
                    else int(divergence[0].item())
                ),
            }
            if draft_trace is not None:
                record["draft_trace"] = draft_trace
            if include_token_ids:
                record["proposal_token_ids"] = proposals.detach().cpu().tolist()
                record["target_token_ids"] = target_tokens.detach().cpu().tolist()
                record["vectorized_target_token_ids"] = (
                    vectorized_target_tokens.detach().cpu().tolist()
                )
                record["correction_or_bonus_token_id"] = int(
                    verification_target_tokens[accepted_count].item()
                )
            records.append(record)

        clean_logits = adapter.forward_logits(prefix)
        clean_next = clean_logits[:, -1, :].argmax(dim=-1).to(torch.long)
        clean_next_id = int(clean_next.item())
        if any(value != clean_next_id for value in verifier_first_tokens.values()):
            raise AssertionError(
                "target row at the committed-prefix boundary changed when draft "
                "suffix length/content changed; inspect the target causal mask"
            )
        prefix = torch.cat((prefix, clean_next.view(1, 1)), dim=1)
        stopped_on_eos = clean_next_id == eos_token_id

    result: dict[str, object] = {
        "prefix_policy": "ordinary_target_greedy_same_prefix_for_all_k",
        "rounds_requested": rounds,
        "rounds_completed": len({int(record["round_index"]) for record in records}),
        "stopped_on_eos": stopped_on_eos,
        "proposal_counts": list(proposal_counts),
        "verification_mode": verification_mode,
        "draft_layer_trace_enabled": bool(trace_draft_layers),
        "metrics_by_proposal_count": summarize_acceptance(records, proposal_counts),
        "phase_metrics_by_proposal_count": summarize_acceptance_phases(
            records, proposal_counts
        ),
        "records": records,
    }
    round_indices = sorted({int(record["round_index"]) for record in records})
    if round_indices:
        phase_names = ("early", "middle", "late")
        phase_by_round = {
            round_index: phase_names[min(2, rank * 3 // len(round_indices))]
            for rank, round_index in enumerate(round_indices)
        }
        for record in records:
            record["round_phase"] = phase_by_round[int(record["round_index"])]
    if include_token_ids:
        result["bootstrap_anchor_token_id"] = int(anchor.item())
        result["final_prefix_token_ids"] = prefix.detach().cpu().reshape(-1).tolist()
    return result


def _acceptance_records(
    report: Mapping[str, object],
) -> dict[tuple[int, int], Mapping[str, object]]:
    sweep = report.get("acceptance_sweep")
    records = sweep.get("records") if isinstance(sweep, Mapping) else None
    if not isinstance(records, Sequence):
        raise ValueError("diagnostic report does not contain acceptance records")
    result: dict[tuple[int, int], Mapping[str, object]] = {}
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("acceptance record must be an object")
        key = (int(record["round_index"]), int(record["requested_proposal_count"]))
        if key in result:
            raise ValueError(f"duplicate acceptance record {key}")
        result[key] = record
    return result


def _fingerprint_sha(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return None
    digest = value.get("sha256")
    return digest if isinstance(digest, str) else None


def _record_has_complete_trace(record: Mapping[str, object]) -> bool:
    trace = record.get("draft_trace")
    if not isinstance(trace, Mapping):
        return False
    input_contract = trace.get("input_contract")
    if not isinstance(input_contract, Mapping) or input_contract.get(
        "status"
    ) != "PASS_OFFICIAL_SINGLE_FORWARD_INPUT":
        return False
    layers = trace.get("target_feature_layers")
    if not isinstance(layers, Mapping) or not layers:
        return False
    if any(_fingerprint_sha(value) is None for value in layers.values()):
        return False
    required = (
        "position_ids",
        "target_hidden",
        "noise_embedding",
        "fc_output",
        "projected_target_hidden",
        "rotary_cosine",
        "rotary_sine",
        *(f"draft_layer_{index}" for index in range(6)),
        "final_norm_output",
        "draft_hidden",
    )
    return all(_fingerprint_sha(trace.get(name)) is not None for name in required)


def _trace_first_divergence(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
    *,
    compare_float_hashes: bool,
) -> tuple[str | None, dict[str, object]]:
    """Return the earliest observable boundary that differs in one round."""

    details: dict[str, object] = {}
    if reference.get("prefix_token_sha256") != candidate.get("prefix_token_sha256"):
        return "committed_prefix", details

    reference_trace = reference.get("draft_trace")
    candidate_trace = candidate.get("draft_trace")
    if isinstance(reference_trace, Mapping) and isinstance(candidate_trace, Mapping):
        reference_position = _fingerprint_sha(reference_trace.get("position_ids"))
        candidate_position = _fingerprint_sha(candidate_trace.get("position_ids"))
        if reference_position != candidate_position:
            details["reference_fingerprint"] = reference_trace.get("position_ids")
            details["candidate_fingerprint"] = candidate_trace.get("position_ids")
            return "position_ids", details

        if compare_float_hashes:
            reference_layers = reference_trace.get("target_feature_layers")
            candidate_layers = candidate_trace.get("target_feature_layers")
            if isinstance(reference_layers, Mapping) and isinstance(candidate_layers, Mapping):
                for layer_id in sorted(
                    set(reference_layers).union(candidate_layers), key=lambda item: int(item)
                ):
                    if _fingerprint_sha(reference_layers.get(layer_id)) != _fingerprint_sha(
                        candidate_layers.get(layer_id)
                    ):
                        details["target_feature_layer_id"] = int(layer_id)
                        details["reference_fingerprint"] = reference_layers.get(layer_id)
                        details["candidate_fingerprint"] = candidate_layers.get(layer_id)
                        return "target_feature_layer", details

            ordered_float_boundaries = [
                "target_hidden",
                "noise_embedding",
                "fc_output",
                "projected_target_hidden",
                "rotary_cosine",
                "rotary_sine",
                *[f"draft_layer_{index}" for index in range(6)],
                "final_norm_output",
                "draft_hidden",
            ]
            for boundary in ordered_float_boundaries:
                if _fingerprint_sha(reference_trace.get(boundary)) != _fingerprint_sha(
                    candidate_trace.get(boundary)
                ):
                    details["reference_fingerprint"] = reference_trace.get(boundary)
                    details["candidate_fingerprint"] = candidate_trace.get(boundary)
                    return boundary, details
    elif compare_float_hashes:
        details["message"] = "one or both reports lack --trace-draft-layers"

    if _fingerprint_sha(reference.get("proposal_token_fingerprint")) != _fingerprint_sha(
        candidate.get("proposal_token_fingerprint")
    ):
        details["reference_fingerprint"] = reference.get("proposal_token_fingerprint")
        details["candidate_fingerprint"] = candidate.get("proposal_token_fingerprint")
        return "proposal_tokens", details
    if _fingerprint_sha(reference.get("target_token_fingerprint")) != _fingerprint_sha(
        candidate.get("target_token_fingerprint")
    ):
        details["reference_fingerprint"] = reference.get("target_token_fingerprint")
        details["candidate_fingerprint"] = candidate.get("target_token_fingerprint")
        return "verifier_target_tokens", details
    if int(reference.get("accepted_count", -1)) != int(candidate.get("accepted_count", -1)):
        return "accepted_count", details
    return None, details


def _metric_deltas(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    reference_sweep = reference.get("acceptance_sweep")
    candidate_sweep = candidate.get("acceptance_sweep")
    reference_metrics = (
        reference_sweep.get("metrics_by_proposal_count")
        if isinstance(reference_sweep, Mapping)
        else None
    )
    candidate_metrics = (
        candidate_sweep.get("metrics_by_proposal_count")
        if isinstance(candidate_sweep, Mapping)
        else None
    )
    if not isinstance(reference_metrics, Mapping) or not isinstance(
        candidate_metrics, Mapping
    ):
        return {}
    result: dict[str, object] = {}
    for proposal_count in sorted(
        set(reference_metrics).intersection(candidate_metrics), key=int
    ):
        left = reference_metrics[proposal_count]
        right = candidate_metrics[proposal_count]
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            continue
        item: dict[str, object] = {}
        for name in (
            "first_proposal_accuracy",
            "mean_accepted_draft_tokens",
            "mean_theoretical_emitted_per_verify",
            "full_block_accept_rate",
        ):
            left_value = left.get(name)
            right_value = right.get(name)
            item[name] = {
                "reference": left_value,
                "candidate": right_value,
                "delta": (
                    float(right_value) - float(left_value)
                    if isinstance(left_value, (int, float))
                    and isinstance(right_value, (int, float))
                    else None
                ),
            }
        result[str(proposal_count)] = item
    return result


def compare_diagnostic_reports(
    reference: Mapping[str, object],
    candidate: Mapping[str, object],
) -> dict[str, object]:
    """Compare two reports and classify first-round versus later divergence."""

    if reference.get("diagnostic") != "qwen3.5-4b-dflash-v1-acceptance":
        raise ValueError("reference report is not a DFlash V1 acceptance diagnosis")
    if candidate.get("diagnostic") != "qwen3.5-4b-dflash-v1-acceptance":
        raise ValueError("candidate report is not a DFlash V1 acceptance diagnosis")
    if int(reference.get("schema_version", 0)) < 2 or int(
        candidate.get("schema_version", 0)
    ) < 2:
        raise ValueError(
            "cross-report localization requires schema_version >= 2; rerun both diagnostics"
        )
    reference_prompt = reference.get("prompt_token_sha256")
    candidate_prompt = candidate.get("prompt_token_sha256")
    if not isinstance(reference_prompt, str) or not isinstance(candidate_prompt, str):
        raise ValueError("both reports must contain prompt_token_sha256")
    if reference_prompt != candidate_prompt:
        raise ValueError("reports use different prompts; first-divergence comparison is invalid")
    if reference.get("eos_token_id") != candidate.get("eos_token_id"):
        raise ValueError("reports use different EOS token IDs")
    reference_checkpoint = reference.get("draft_checkpoint")
    candidate_checkpoint = candidate.get("draft_checkpoint")
    if isinstance(reference_checkpoint, Mapping) and isinstance(
        candidate_checkpoint, Mapping
    ):
        for identity in ("config_sha256", "model_sha256"):
            left = reference_checkpoint.get(identity)
            right = candidate_checkpoint.get(identity)
            if left is not None and right is not None and left != right:
                raise ValueError(f"reports use different draft checkpoint {identity}")
    reference_sweep = reference.get("acceptance_sweep")
    candidate_sweep = candidate.get("acceptance_sweep")
    reference_counts = (
        reference_sweep.get("proposal_counts")
        if isinstance(reference_sweep, Mapping)
        else None
    )
    candidate_counts = (
        candidate_sweep.get("proposal_counts")
        if isinstance(candidate_sweep, Mapping)
        else None
    )
    if reference_counts != candidate_counts:
        raise ValueError("reports use different proposal-count sweeps")
    reference_records = _acceptance_records(reference)
    candidate_records = _acceptance_records(candidate)
    common = sorted(set(reference_records).intersection(candidate_records))
    same_dtype = reference.get("dtype") == candidate.get("dtype")
    full_trace_coverage = bool(common) and all(
        _record_has_complete_trace(reference_records[key])
        and _record_has_complete_trace(candidate_records[key])
        for key in common
    )
    first: dict[str, object] | None = None
    for key in common:
        boundary, details = _trace_first_divergence(
            reference_records[key],
            candidate_records[key],
            compare_float_hashes=same_dtype and full_trace_coverage,
        )
        if boundary is not None:
            first = {
                "round_index": key[0],
                "proposal_count": key[1],
                "boundary": boundary,
                **details,
            }
            break
    if not common:
        status = "INCOMPARABLE_NO_COMMON_ROUNDS"
    elif first is None and set(reference_records) == set(candidate_records):
        status = (
            "MATCH_ON_ALL_RECORDED_ROUNDS"
            if full_trace_coverage and same_dtype
            else "MATCH_ON_ALL_RECORDED_ROUNDS_TOKEN_LEVEL_ONLY"
        )
    elif first is None:
        status = "MATCH_ON_COMMON_ROUNDS_RECORD_SET_DIFFERS"
    elif int(first["round_index"]) == 0:
        status = "DIVERGED_IN_FIRST_ROUND"
    else:
        status = "DIVERGED_AFTER_FIRST_ROUND"
    precision_hypothesis = (
        "DEMOTED_TOKEN_DECISIONS_UNCHANGED_ACROSS_DTYPES"
        if not same_dtype
        and first is None
        and set(reference_records) == set(candidate_records)
        else (
            "STILL_POSSIBLE_TOKEN_DECISIONS_DIVERGED"
            if not same_dtype and first is not None
            else "NOT_A_CROSS_DTYPE_COMPARISON"
        )
    )
    return {
        "status": status,
        "comparison_mode": (
            "exact_same_dtype_fingerprints"
            if same_dtype and full_trace_coverage
            else (
                "same_dtype_token_fingerprints_only"
                if same_dtype
                else "cross_dtype_token_and_metric_comparison"
            )
        ),
        "reference": {
            "device": reference.get("device"),
            "dtype": reference.get("dtype"),
            "draft_backend": reference.get("draft_backend"),
        },
        "candidate": {
            "device": candidate.get("device"),
            "dtype": candidate.get("dtype"),
            "draft_backend": candidate.get("draft_backend"),
        },
        "common_records": len(common),
        "full_draft_trace_coverage": full_trace_coverage,
        "precision_hypothesis": precision_hypothesis,
        "reference_only_records": len(set(reference_records) - set(candidate_records)),
        "candidate_only_records": len(set(candidate_records) - set(reference_records)),
        "first_divergence": first,
        "metric_deltas_by_proposal_count": _metric_deltas(reference, candidate),
        "interpretation": (
            "first-round divergence points to inputs/model math rather than cache evolution"
            if status == "DIVERGED_IN_FIRST_ROUND"
            else (
                "later-only divergence prioritizes prefix/state/cache evolution"
                if status == "DIVERGED_AFTER_FIRST_ROUND"
                else (
                    "float tensor hashes are intentionally not compared across dtypes"
                    if not same_dtype
                    else "no measured divergence on common records"
                )
            )
        ),
    }


def _feature_health(
    features: Tensor,
    layer_ids: Sequence[int],
    hidden_size: int,
) -> dict[str, object]:
    if features.ndim != 3 or features.shape[-1] != len(layer_ids) * hidden_size:
        raise ValueError("feature-health input has an unexpected shape")
    result: dict[str, object] = {}
    for offset, layer_id in enumerate(layer_ids):
        values = features[..., offset * hidden_size : (offset + 1) * hidden_size]
        finite = bool(torch.isfinite(values).all().item())
        item: dict[str, object] = {
            "finite": finite,
            "zero_fraction": float((values == 0).float().mean().item()),
        }
        if finite:
            values_f = values.float()
            item.update(
                {
                    "rms": float(torch.sqrt(values_f.square().mean()).item()),
                    "std": float(values_f.std(unbiased=False).item()),
                    "min": float(values_f.min().item()),
                    "max": float(values_f.max().item()),
                }
            )
        result[str(layer_id)] = item
    return result


def shadow_torch_ops(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    *,
    proposal_count: int,
) -> dict[str, object]:
    """Compare the active NPU draft backend with Torch ops on identical tensors."""

    target_hidden = adapter._replay_target_features(prefix_ids[:, :-1])
    _block_ids, noise_embedding, position_ids, _input_contract = _build_draft_inputs(
        adapter, prefix_ids, target_hidden, proposal_count
    )
    original_ops = adapter.draft.ops

    def run_once() -> tuple[Tensor, Tensor, list[Tensor]]:
        captured: list[Tensor] = []
        handles = [
            layer.register_forward_hook(
                lambda _module, _inputs, output: captured.append(output.detach().clone())
            )
            for layer in adapter.draft.layers
        ]
        try:
            with torch.inference_mode():
                hidden = adapter.draft.draft_hidden(
                    target_hidden, noise_embedding, position_ids
                )
                top1 = adapter.draft.ops.top1(hidden, adapter.lm_head_weight)
        finally:
            for handle in handles:
                handle.remove()
        return hidden.detach().clone(), top1.detach().clone(), captured

    try:
        active_hidden, active_top1, active_layers = run_once()
        adapter.draft.set_ops(TorchDFlashOps())
        torch_hidden, torch_top1, torch_layers = run_once()
    except Exception as error:  # NPU SDPA support is runtime-dependent.
        return {
            "status": "SKIPPED_TORCH_OPS_UNSUPPORTED",
            "error_type": type(error).__name__,
            "message": str(error).splitlines()[0][:300],
        }
    finally:
        adapter.draft.set_ops(original_ops)

    return {
        "status": "PASS_COMPARISON_COMPLETED",
        "proposal_count": proposal_count,
        "top1_equal": bool(torch.equal(active_top1, torch_top1)),
        "top1_mismatch_count": int((active_top1 != torch_top1).sum().item()),
        "final_hidden": tensor_metrics(torch_hidden, active_hidden),
        "layers": [
            {
                "layer_index": index,
                "active_vs_torch": tensor_metrics(torch_output, active_output),
            }
            for index, (torch_output, active_output) in enumerate(
                zip(torch_layers, active_layers, strict=True)
            )
        ],
    }


def _weight_health(adapter: Qwen35DFlashFullPrefixAdapter) -> dict[str, object]:
    input_weight = adapter.input_embedding_weight
    output_weight = adapter.lm_head_weight
    try:
        tied: bool | None = (
            input_weight.untyped_storage().data_ptr()
            == output_weight.untyped_storage().data_ptr()
        )
    except (AttributeError, NotImplementedError, RuntimeError):
        try:
            tied = input_weight.data_ptr() == output_weight.data_ptr()
        except (NotImplementedError, RuntimeError):
            tied = None
    controller = getattr(adapter.target, "target", None)
    execution_model = getattr(controller, "dflash_execution_model", adapter.target)
    if not isinstance(execution_model, nn.Module):
        raise TypeError("target execution model is not torch.nn.Module")
    quantized_linear_count = sum(
        type(module).__name__ == "QLinear" for module in execution_model.modules()
    )
    return {
        "input_embedding": {
            "shape": list(input_weight.shape),
            "dtype": str(input_weight.dtype),
            "device": str(input_weight.device),
        },
        "lm_head": {
            "shape": list(output_weight.shape),
            "dtype": str(output_weight.dtype),
            "device": str(output_weight.device),
        },
        "embedding_and_lm_head_share_storage": tied,
        "target_execution_model_type": (
            f"{type(execution_model).__module__}.{type(execution_model).__qualname__}"
        ),
        "target_qlinear_module_count": quantized_linear_count,
        "quantized_target_warning": (
            "QLinear modules were detected; establish non-quantized target parity first"
            if quantized_linear_count
            else None
        ),
    }


def diagnose_next_actions(report: Mapping[str, object]) -> list[str]:
    """Turn measured gates into an ordered, non-speculative investigation list."""

    comparison = report.get("report_comparison")
    if isinstance(comparison, Mapping):
        status = comparison.get("status")
        divergence = comparison.get("first_divergence")
        boundary = divergence.get("boundary") if isinstance(divergence, Mapping) else None
        if status == "DIVERGED_IN_FIRST_ROUND":
            return [
                f"两份报告在首轮的 {boundary} 首次分叉；首轮没有历史 draft "
                "cache，优先检查输入构造、feature、position、dtype 或对应草稿层。",
                "若是同 dtype GPU/NPU 对比，沿 target_feature_layer → "
                "projected_target_hidden → draft_layer_N → proposal_tokens 顺序定位。",
                "若是 FP16/BF16 对比，浮点 SHA 不参与精确判定；比较 K=1 首命中率及 emitted/verify 的变化。",
            ]
        if status == "DIVERGED_AFTER_FIRST_ROUND":
            return [
                f"首轮一致、后续轮在 {boundary} 首次分叉；优先检查 committed prefix、full-prefix replay 与缓存/状态演进。",
                "不要给 DFlash 增加迭代 mask 替换；官方 V1 每轮仍是一次并行 draft forward。",
                "用相同 prompt 增加轮数，确认首个分叉 round 是否稳定。",
            ]

    feature_semantics = report.get("feature_collector_semantics")
    if isinstance(feature_semantics, Mapping) and str(
        feature_semantics.get("status", "")
    ).startswith("FAIL_"):
        return [
            "先修复 framework Target feature 语义：collector 与官方 hidden_states[layer_id+1] 规则不一致。",
            "按 layers 指标找到第一个不一致的 layer_id；检查是否误取 layer 输入、final norm 后输出或 layer_id 偏移 1。",
            "feature 语义闭合前，不用接受率判断 draft 权重或设备算子。",
        ]

    parity = report.get("target_path_parity")
    if isinstance(parity, Mapping) and parity.get("status") == (
        "FAIL_FRAMEWORK_INCREMENTAL_PATH"
    ):
        return [
            "CPU/CUDA Target 的 use_cache=True 增量探针失败，尚不能排除 full-prefix feature 路径偏差。",
            f"错误: {parity.get('error_type')}: {parity.get('message')}",
            "先修复/确认 packaged Target 的 DynamicCache prefill→单 token decode，再解释 Draft 接受率。",
        ]
    if isinstance(parity, Mapping) and parity.get("all_top1_match") is False:
        return [
            "先修复 Target 路径：增量路径与 fresh full-prefix 的 Top-1 已分叉；此时接受率没有可解释性。",
            "查看首个分叉 comparison 的 8 个 feature_layer 指标；最早分叉层就是优先排查边界。",
            "核对 full-prefix 的 fresh KV/GDN state、position_ids、"
            "new_kv_cache_pos、allQLen 和 causal mask。",
        ]
    if isinstance(parity, Mapping) and parity.get("status") == (
        "PASS_TOP1_WITH_NUMERIC_DIFFERENCE"
    ):
        relative_rmse = parity.get("max_feature_relative_rmse")
        minimum_cosine = parity.get("min_feature_cosine_similarity")
        if (
            isinstance(relative_rmse, (int, float))
            and float(relative_rmse) >= 1.0e-2
        ) or (
            isinstance(minimum_cosine, (int, float))
            and float(minimum_cosine) < 0.999
        ):
            return [
                "Target Top-1 虽一致，但增量与 full-prefix 的 8 层特征有明显漂移；Draft 消费的是特征而不是 Target Top-1。",
                "本次 max_feature_relative_rmse="
                f"{relative_rmse}, min_feature_cosine_similarity={minimum_cosine}；"
                "按 records.feature_layers 找最早漂移层。",
                "先让 Draft 分别消费同一前缀的 cached-incremental 特征和 full-prefix 特征做 A/B；若 proposal/接受长度随之恢复，就不要改 Draft 权重数学。",
            ]

    health = report.get("weight_health")
    if isinstance(health, Mapping) and int(health.get("target_qlinear_module_count", 0)):
        return [
            "先用非量化 Target 重跑同一诊断；量化 Target 会把 feature 数值误差与 DFlash 本身混在一起。",
            "非量化路径闭合后，再单独测量量化对每层 feature 和接受率的影响。",
        ]

    sweep = report.get("acceptance_sweep")
    metrics = sweep.get("metrics_by_proposal_count") if isinstance(sweep, Mapping) else None
    records = sweep.get("records") if isinstance(sweep, Mapping) else None
    if isinstance(records, Sequence):
        divergent = [
            record
            for record in records
            if isinstance(record, Mapping)
            and isinstance(record.get("vectorized_prefix_invariance"), Mapping)
            and record["vectorized_prefix_invariance"].get("status")
            == "FAIL_PREFIX_ROW_DIVERGENCE"
        ]
        if divergent:
            first = divergent[0]
            invariance = first["vectorized_prefix_invariance"]
            assert isinstance(invariance, Mapping)
            return [
                "一次性 vectorized verify 与逐前缀隔离 verify 已出现 Top-1 分叉；不要把该异常归因于 Draft 接受率。",
                "首个分叉位于 round="
                f"{first.get('round_index')}, K={first.get('requested_proposal_count')}, "
                f"row={invariance.get('first_divergent_row')}。",
                "正式 V1 使用 sequential isolated-prefix verifier；另行检查 Target causal mask、长度相关 kernel 和 full-prefix 数值稳定性。",
            ]
    k1 = metrics.get("1") if isinstance(metrics, Mapping) else None
    k1_accuracy = k1.get("first_proposal_accuracy") if isinstance(k1, Mapping) else None
    if isinstance(k1_accuracy, (int, float)) and k1_accuracy < 0.5:
        actions = [
            "K=1 的首 token 命中率已低，问题发生在长 block 退化之前；优先检查 8 层 feature 内容和草稿实现语义。",
            "使用 --trace-draft-layers 记录 target feature、projection、position、"
            "6 层 draft 和 Top-1 的逐轮指纹。",
            "用 --oracle-bundle 导出首轮输入和各层输出，离线逐边界检查；首个差异边界比接受率更有定位价值。",
            "增加 --acceptance-rounds 后再判断，避免少量轮次造成偶然比例。",
        ]
        phase_metrics = (
            sweep.get("phase_metrics_by_proposal_count")
            if isinstance(sweep, Mapping)
            else None
        )
        if isinstance(phase_metrics, Mapping):
            count_keys = sorted(phase_metrics, key=int)
            largest = phase_metrics.get(count_keys[-1]) if count_keys else None
            early = largest.get("early") if isinstance(largest, Mapping) else None
            late = largest.get("late") if isinstance(largest, Mapping) else None
            early_mean = (
                early.get("mean_accepted_draft_tokens")
                if isinstance(early, Mapping)
                else None
            )
            late_mean = (
                late.get("mean_accepted_draft_tokens")
                if isinstance(late, Mapping)
                else None
            )
            if (
                isinstance(early_mean, (int, float))
                and isinstance(late_mean, (int, float))
                and float(late_mean) > float(early_mean)
            ):
                actions.insert(
                    0,
                    "接受长度呈后段上升：先区分自然文本难度与早期状态构造问题。"
                    f"当前最大 K 的 early={float(early_mean):.3f}, "
                    f"late={float(late_mean):.3f}；换 3 个不同 prompt 重跑，"
                    "若上升总绑定绝对轮次则查状态，若绑定句式/内容则更像正常难度分布。",
                )
        precision = (
            comparison.get("precision_hypothesis")
            if isinstance(comparison, Mapping)
            else None
        )
        if precision == "DEMOTED_TOKEN_DECISIONS_UNCHANGED_ACROSS_DTYPES":
            actions.insert(
                0,
                "FP16/BF16 的逐轮 proposal、verifier 和 accepted_count 决策未变；将精度从首要嫌疑降级，检查两种精度共享的 feature/输入/草稿逻辑。",
            )
        if report.get("device_type") == "npu":
            actions.insert(
                3,
                "使用 --shadow-torch-ops 比较同一 NPU tensor 上的分解 backend "
                "与 Torch draft；若 Top-1 不同，先收敛草稿算子。",
            )
        elif report.get("device_type") == "cuda" and report.get("dtype") == str(
            torch.float16
        ) and precision != "DEMOTED_TOKEN_DECISIONS_UNCHANGED_ACROSS_DTYPES":
            actions.insert(
                3,
                "若 CUDA 支持 BF16，用完全相同 prompt/K/轮数重跑 --dtype "
                "bfloat16，并用 --compare-report 对比 FP16 报告。",
            )
        return actions
    return [
        "Target 路径未见 Top-1 分叉；扩大 --acceptance-rounds 获取稳定统计。",
        "若 K=1 稳定而 K=8/16 明显退化，打开 --trace-draft-layers 检查 block attention、位置和低精度边界。",
        "mean_theoretical_emitted_per_verify 才接近吞吐收益口径，不要把 accepted/proposed 百分比直接当官方加速指标。",
    ]


def _safe_output_path(
    raw_path: str,
    *,
    protected_roots: Sequence[Path],
    option_name: str,
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_symlink():
        raise ValueError(f"{option_name} must not be a symlink")
    resolved = path.resolve()
    for root in protected_roots:
        resolved_root = root.expanduser().resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            continue
        raise ValueError(f"{option_name} must be outside source and model directories")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


def _safe_report_path(
    raw_path: str,
    *,
    protected_roots: Sequence[Path],
) -> Path:
    return _safe_output_path(
        raw_path,
        protected_roots=protected_roots,
        option_name="--report",
    )


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as stream:
            stream.write(serialized)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_report(path: str | Path) -> dict[str, object]:
    raw = Path(path).expanduser()
    if raw.is_symlink():
        raise ValueError("--compare-report must not be a symlink")
    resolved = raw.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"comparison report is not a regular file: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("comparison report root must be a JSON object")
    return payload


def _write_oracle_bundle(
    path: Path,
    tensors: Mapping[str, Tensor],
    *,
    device: str,
    dtype: torch.dtype,
    proposal_count: int,
) -> None:
    """Atomically write explicit first-round tensors for an independent oracle."""

    if path.suffix != ".safetensors":
        raise ValueError("--oracle-bundle must end in .safetensors")
    if not tensors:
        raise RuntimeError("oracle bundle capture is empty")
    from safetensors.torch import save_file

    prepared = {
        name: tensor.detach().contiguous().cpu().clone()
        for name, tensor in tensors.items()
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(file_descriptor)
    try:
        save_file(
            prepared,
            temporary_name,
            metadata={
                "schema_version": "2",
                "diagnostic": "qwen3.5-4b-dflash-v1-first-round-oracle",
                "device": str(device),
                "dtype": str(dtype),
                "round_index": "0",
                "proposal_count": str(proposal_count),
                "algorithm": "single_parallel_draft_forward",
            },
        )
        with open(temporary_name, "rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose DFlash V1 acceptance on CPU, CUDA, or NPU"
    )
    parser.add_argument("--target-dir")
    parser.add_argument("--draft-dir")
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument("--prompt-ids", help="comma-separated token IDs")
    prompt.add_argument("--prompt-json", help="JSON token list or input_ids object")
    prompt.add_argument("--prompt", help="UTF-8 text prompt")
    prompt.add_argument("--prompt-file", help="UTF-8 text file used as the prompt")
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
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=tuple(SUPPORTED_DTYPES),
        default="float16",
        help="NPU is locked to float16; CUDA BF16 is the recommended dtype A/B",
    )
    parser.add_argument(
        "--target-w8a8-emulation-artifact",
        help=(
            "CPU/CUDA only: reuse exported NPU QLinear W_q/scale and replace "
            "framework Target text linears with the exact diagnostic formula"
        ),
    )
    parser.add_argument(
        "--kv-cache-max-len",
        type=int,
        help="required for the NPU incremental/full-prefix target comparison",
    )
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--decode-chunk-size", type=int, default=1)
    parser.add_argument("--target-factory", default=DEFAULT_TARGET_FACTORY)
    parser.add_argument("--target-parity-decode-steps", type=int, default=4)
    parser.add_argument("--acceptance-rounds", type=int, default=8)
    parser.add_argument(
        "--verification-mode",
        choices=("sequential", "vectorized"),
        default="sequential",
        help=(
            "sequential verifies each proposal on its isolated prefix; "
            "vectorized retains the one-call diagnostic assumption"
        ),
    )
    parser.add_argument(
        "--proposal-counts",
        default="1,4,8,16",
        help=(
            "K proposal/mask tokens using the vLLM convention; the clean "
            "anchor is an additional query row, so K=16 uses 17 rows"
        ),
    )
    parser.add_argument("--eos-token-id", type=int, default=OFFICIAL_EOS_TOKEN_ID)
    parser.add_argument("--shadow-torch-ops", action="store_true")
    parser.add_argument(
        "--trace-draft-layers",
        action="store_true",
        help="hash target features, projection, position, each draft layer, and Top-1 per round",
    )
    parser.add_argument(
        "--compare-report",
        help="compare this run with an earlier diagnosis and locate the first divergence",
    )
    parser.add_argument(
        "--compare-reports",
        nargs=2,
        metavar=("REFERENCE", "CANDIDATE"),
        help="compare two existing JSON reports without loading model weights",
    )
    parser.add_argument(
        "--oracle-bundle",
        help="opt-in first-round .safetensors inputs/outputs for offline boundary analysis",
    )
    parser.add_argument("--include-token-ids", action="store_true")
    parser.add_argument("--report", help="optional JSON output outside source/model dirs")
    parser.add_argument(
        "--verify-draft-sha256",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verify the full official draft weight hash before diagnosis",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> tuple[int, ...]:
    if args.target_dir is None or args.draft_dir is None:
        raise ValueError("diagnosis requires --target-dir and --draft-dir")
    prompt_inputs = (
        args.prompt_ids,
        args.prompt_json,
        args.prompt,
        args.prompt_file,
    )
    if sum(value is not None for value in prompt_inputs) != 1:
        raise ValueError(
            "diagnosis requires exactly one of --prompt-ids, --prompt-json, "
            "--prompt, or --prompt-file"
        )
    device_type = str(args.device).split(":", 1)[0].lower()
    if device_type not in {"cpu", "cuda", "npu"}:
        raise ValueError("--device must be cpu, cuda, cuda:N, npu, or npu:N")
    if args.eos_token_id != OFFICIAL_EOS_TOKEN_ID:
        raise ValueError(f"this checkpoint requires --eos-token-id {OFFICIAL_EOS_TOKEN_ID}")
    for name in (
        "prefill_chunk_size",
        "decode_chunk_size",
        "acceptance_rounds",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.target_parity_decode_steps < 0:
        raise ValueError("--target-parity-decode-steps must be non-negative")
    if device_type == "npu":
        if args.kv_cache_max_len is None or args.kv_cache_max_len <= 0:
            raise ValueError("NPU diagnosis requires positive --kv-cache-max-len")
        if args.kv_cache_max_len % 64:
            raise ValueError("--kv-cache-max-len must be divisible by 64")
        if args.dtype != "float16":
            raise ValueError("NPU diagnosis requires --dtype float16")
    elif args.kv_cache_max_len is not None and args.kv_cache_max_len <= 0:
        raise ValueError("--kv-cache-max-len must be positive when supplied")
    if args.shadow_torch_ops and device_type != "npu":
        raise ValueError("--shadow-torch-ops is only meaningful for the NPU backend")
    if args.target_w8a8_emulation_artifact is not None:
        if device_type not in {"cpu", "cuda"}:
            raise ValueError(
                "--target-w8a8-emulation-artifact is supported only on CPU/CUDA"
            )
        if args.dtype != "float16":
            raise ValueError("strict NPU QLinear emulation requires --dtype float16")
        artifact = Path(args.target_w8a8_emulation_artifact).expanduser()
        if artifact.is_symlink() or not artifact.is_dir():
            raise ValueError(
                "--target-w8a8-emulation-artifact must be a real artifact directory"
            )
    return parse_proposal_counts(args.proposal_counts)


def _print_summary(report: Mapping[str, object]) -> None:
    parity = report["target_path_parity"]
    sweep = report["acceptance_sweep"]
    assert isinstance(parity, Mapping) and isinstance(sweep, Mapping)
    print("\n=== DFlash V1 接受率诊断 ===")
    print(
        "运行身份:",
        f"device={report.get('device')}",
        f"dtype={report.get('dtype')}",
        f"backend={report.get('draft_backend')}",
        f"verify={report.get('verification_mode')}",
    )
    emulation = report.get("target_w8a8_emulation")
    if isinstance(emulation, Mapping):
        print(
            "Target W8A8 仿真:",
            emulation.get("status"),
            f"qlinear={emulation.get('qlinear_count', 0)}",
        )
    print(
        "Target 增量 vs full-prefix:",
        parity["status"],
        f"({parity['comparisons']} 个前缀)",
    )
    print(
        "Target 特征路径指标:",
        f"max_relative_rmse={parity.get('max_feature_relative_rmse')}",
        f"min_cosine={parity.get('min_feature_cosine_similarity')}",
    )
    feature_semantics = report.get("feature_collector_semantics")
    if isinstance(feature_semantics, Mapping):
        print("Feature layer_id+1 语义:", feature_semantics.get("status"))
    parity_records = parity.get("records")
    if isinstance(parity_records, Sequence):
        for record in parity_records:
            if not isinstance(record, Mapping):
                continue
            feature_metrics = record.get("features")
            feature_exact = (
                feature_metrics.get("bitwise_equal")
                if isinstance(feature_metrics, Mapping)
                else None
            )
            if bool(record.get("top1_match")) and feature_exact is True:
                continue
            logits_metrics = record.get("logits")
            logits_max_abs = (
                logits_metrics.get("max_abs_error")
                if isinstance(logits_metrics, Mapping)
                else None
            )
            feature_max_abs = (
                feature_metrics.get("max_abs_error")
                if isinstance(feature_metrics, Mapping)
                else None
            )
            feature_relative_rmse = (
                feature_metrics.get("relative_rmse")
                if isinstance(feature_metrics, Mapping)
                else None
            )
            print(
                "首个差异前缀:",
                f"length={record.get('prefix_length')}",
                f"source={record.get('incremental_source')}",
                f"top1_match={record.get('top1_match')}",
                f"logits_max_abs={logits_max_abs}",
                f"feature_max_abs={feature_max_abs}",
                f"feature_relative_rmse={feature_relative_rmse}",
            )
            break
    metrics = sweep["metrics_by_proposal_count"]
    assert isinstance(metrics, Mapping)
    print("K  rounds  first-hit  mean-accepted  emitted/verify  full-block")
    for proposal_count in sweep["proposal_counts"]:
        item = metrics[str(proposal_count)]
        assert isinstance(item, Mapping)
        first = item["first_proposal_accuracy"]
        accepted = item["mean_accepted_draft_tokens"]
        emitted = item["mean_theoretical_emitted_per_verify"]
        full = item["full_block_accept_rate"]

        def percent(value: object) -> str:
            return "n/a" if value is None else f"{float(value) * 100:6.2f}%"

        def number(value: object) -> str:
            return "n/a" if value is None else f"{float(value):6.3f}"

        print(
            f"{int(proposal_count):2d} {int(item['rounds']):7d}  "
            f"{percent(first):>9}  {number(accepted):>13}  "
            f"{number(emitted):>14}  {percent(full):>10}"
        )
    phase_metrics = sweep.get("phase_metrics_by_proposal_count")
    if isinstance(phase_metrics, Mapping):
        print("\n分段 mean-accepted (early / middle / late):")
        for proposal_count in sweep["proposal_counts"]:
            by_phase = phase_metrics.get(str(proposal_count))
            if not isinstance(by_phase, Mapping):
                continue
            values: list[str] = []
            for phase in ("early", "middle", "late"):
                item = by_phase.get(phase)
                value = (
                    item.get("mean_accepted_draft_tokens")
                    if isinstance(item, Mapping)
                    else None
                )
                values.append("n/a" if value is None else f"{float(value):.3f}")
            print(f"K={int(proposal_count):2d}: " + " / ".join(values))
    records = sweep.get("records")
    if isinstance(records, Sequence) and records:
        first_record = records[0]
        if isinstance(first_record, Mapping):
            input_contract = first_record.get("draft_input_contract")
            if isinstance(input_contract, Mapping):
                attention = input_contract.get("attention")
                attention_status = (
                    attention.get("status") if isinstance(attention, Mapping) else None
                )
                print(
                    "首轮输入合同:",
                    input_contract.get("status"),
                    f"attention={attention_status}",
                )
        maximum_k = max(int(value) for value in sweep["proposal_counts"])
        maximum_records = [
            record
            for record in records
            if isinstance(record, Mapping)
            and int(record.get("requested_proposal_count", -1)) == maximum_k
        ]
        print(
            f"逐轮接受长度(K={maximum_k}):",
            " ".join(
                f"r{int(record['round_index'])}:{int(record['accepted_count'])}"
                for record in maximum_records
            ),
        )
    text_output = report.get("text_output")
    if isinstance(text_output, Mapping):
        print("\nTarget 续写文本:")
        print(text_output.get("text", ""))
    comparison = report.get("report_comparison")
    if isinstance(comparison, Mapping):
        print("\n跨报告比较:", comparison.get("status"))
        divergence = comparison.get("first_divergence")
        if isinstance(divergence, Mapping):
            print(
                "首个分叉:",
                f"round={divergence.get('round_index')}",
                f"K={divergence.get('proposal_count')}",
                f"boundary={divergence.get('boundary')}",
            )
        print("比较模式:", comparison.get("comparison_mode"))
    print("\n建议顺序：")
    for index, action in enumerate(report["next_actions"], start=1):
        print(f"{index}. {action}")


def _compare_reports_only(args: argparse.Namespace, package_dir: Path) -> int:
    if args.compare_report is not None or args.oracle_bundle is not None:
        raise ValueError(
            "--compare-reports cannot be combined with --compare-report or --oracle-bundle"
        )
    assert args.compare_reports is not None
    reference_path = Path(args.compare_reports[0]).expanduser().resolve()
    candidate_path = Path(args.compare_reports[1]).expanduser().resolve()
    reference = _read_report(reference_path)
    candidate = _read_report(candidate_path)
    comparison = compare_diagnostic_reports(reference, candidate)
    payload: dict[str, object] = {
        "schema_version": 1,
        "diagnostic": "qwen3.5-4b-dflash-v1-report-comparison",
        "reference_report": {
            "path": str(reference_path),
            "sha256": sha256_file(reference_path),
        },
        "candidate_report": {
            "path": str(candidate_path),
            "sha256": sha256_file(candidate_path),
        },
        "comparison": comparison,
    }
    print("\n=== DFlash V1 跨报告比较 ===")
    print("状态:", comparison["status"])
    print("模式:", comparison["comparison_mode"])
    divergence = comparison.get("first_divergence")
    if isinstance(divergence, Mapping):
        print(
            "首个分叉:",
            f"round={divergence.get('round_index')}",
            f"K={divergence.get('proposal_count')}",
            f"boundary={divergence.get('boundary')}",
        )
    print("解释:", comparison["interpretation"])
    if args.report is not None:
        report_path = _safe_output_path(
            args.report,
            protected_roots=(package_dir.parent.parent,),
            option_name="--report",
        )
        if report_path in {reference_path, candidate_path}:
            raise ValueError("--report must not overwrite either input report")
        _write_report(report_path, payload)
        print(f"JSON 比较报告已写入: {report_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    package_dir = Path(__file__).resolve().parent
    if args.compare_reports is not None:
        return _compare_reports_only(args, package_dir)
    proposal_counts = _validate_args(args)
    device_type = str(args.device).split(":", 1)[0].lower()
    dtype = SUPPORTED_DTYPES[args.dtype]
    source_path = package_dir.parent / "modeling_qwen3_5_hiai_nd.py"
    if device_type == "npu" and (source_path.is_symlink() or not source_path.is_file()):
        raise FileNotFoundError(
            "expected models/modeling_qwen3_5_hiai_nd.py beside dflash_v1"
        )
    target_root = Path(args.target_dir).expanduser().resolve()
    draft_root = Path(args.draft_dir).expanduser().resolve()
    if not target_root.is_dir() or not draft_root.is_dir():
        raise FileNotFoundError("--target-dir and --draft-dir must be existing directories")

    emulation_artifact = (
        None
        if args.target_w8a8_emulation_artifact is None
        else Path(args.target_w8a8_emulation_artifact).expanduser().resolve()
    )
    protected_roots = tuple(
        path
        for path in (
            package_dir.parent.parent,
            target_root,
            draft_root,
            emulation_artifact,
        )
        if path is not None
    )
    report_path = (
        None
        if args.report is None
        else _safe_report_path(args.report, protected_roots=protected_roots)
    )
    oracle_path = (
        None
        if args.oracle_bundle is None
        else _safe_output_path(
            args.oracle_bundle,
            protected_roots=protected_roots,
            option_name="--oracle-bundle",
        )
    )
    if report_path is not None and oracle_path is not None and report_path == oracle_path:
        raise ValueError("--report and --oracle-bundle must be different files")
    comparison_path = (
        None if args.compare_report is None else Path(args.compare_report).expanduser().resolve()
    )
    prompt_json_path = (
        None if args.prompt_json is None else Path(args.prompt_json).expanduser().resolve()
    )
    prompt_file_path = (
        None if args.prompt_file is None else Path(args.prompt_file).expanduser().resolve()
    )
    input_paths = {
        path
        for path in (comparison_path, prompt_json_path, prompt_file_path)
        if path is not None
    }
    if report_path is not None and report_path in input_paths:
        raise ValueError(
            "--report must not overwrite a comparison report or prompt input file"
        )
    if oracle_path is not None and oracle_path in input_paths:
        raise ValueError(
            "--oracle-bundle must not overwrite a comparison report or prompt input file"
        )
    reference_report = (
        None if args.compare_report is None else _read_report(args.compare_report)
    )

    if device_type == "npu":
        assert args.kv_cache_max_len is not None
        os.environ[TARGET_FACTORY_ENV] = args.target_factory
        os.environ[PREFILL_CHUNK_SIZE_ENV] = str(args.prefill_chunk_size)
        os.environ[DECODE_CHUNK_SIZE_ENV] = str(args.decode_chunk_size)
        os.environ[KV_CACHE_MAX_LEN_ENV] = str(args.kv_cache_max_len)

    _prepare_device_backend(args.device)
    _validate_experiment_dtype(args.device, dtype)
    target_checkpoint = _audit_target_config(target_root)
    checkpoint = require_official_dflash_checkpoint(
        draft_root, verify_model_hash=bool(args.verify_draft_sha256)
    )
    target_loader = (
        f"{__package__}.internal_target_loader:load_target"
        if device_type == "npu"
        else None
    )
    target = _load_target(
        str(target_root),
        target_loader=target_loader,
        hiai_source=str(source_path) if device_type == "npu" else None,
        device=args.device,
        dtype=dtype,
        allow_download=False,
        trust_remote_code=False,
    )
    if emulation_artifact is None:
        target_w8a8_emulation: dict[str, object] = {
            "status": "DISABLED",
            "scheme": "disabled",
            "scope": "framework_target",
        }
    else:
        from .w8a8_emulation import apply_w8a8_emulation

        target_w8a8_emulation = apply_w8a8_emulation(
            target,
            emulation_artifact,
            device=args.device,
            dtype=dtype,
        )
    draft_memory_preflight = _draft_device_memory_preflight(
        args.device,
        dtype,
        checkpoint,
    )
    ops, backend_name = _select_draft_ops(
        device=args.device, ops_backend=None, allow_op_fallback=False
    )
    draft = DFlashDraftModel.from_pretrained(
        draft_root, ops=ops, device=args.device, dtype=dtype
    )
    adapter = Qwen35DFlashFullPrefixAdapter(target, draft)
    tokenizer: object | None = None
    if args.prompt is not None or prompt_file_path is not None:
        if prompt_file_path is not None:
            if not prompt_file_path.is_file():
                raise FileNotFoundError(
                    f"--prompt-file is not a regular file: {prompt_file_path}"
                )
            prompt_text = prompt_file_path.read_text(encoding="utf-8")
        else:
            prompt_text = str(args.prompt)
        prompt_values, tokenizer = _tokenize_prompt_text(
            prompt_text,
            target_root=target_root,
            prompt_mode=args.prompt_mode,
            enable_thinking=bool(args.enable_thinking),
        )
    else:
        prompt_values = _prompt_ids(args.prompt_ids, args.prompt_json)
    if any(token < 0 or token >= adapter.vocab_size for token in prompt_values):
        raise ValueError("prompt contains a token outside the target vocabulary")
    prompt = torch.tensor(
        [prompt_values], dtype=torch.long, device=torch.device(args.device)
    )
    context_limit = int(draft.config.max_position_embeddings)
    if args.kv_cache_max_len is not None:
        context_limit = min(context_limit, int(args.kv_cache_max_len))
    if int(prompt.shape[1]) + max(proposal_counts) + args.acceptance_rounds + 1 > context_limit:
        raise ValueError("prompt plus diagnostic sweep exceeds the configured context")

    target_path_function = (
        compare_target_paths
        if device_type == "npu"
        else compare_framework_target_paths
    )
    target_path_parity = target_path_function(
        target,
        prompt,
        decode_steps=args.target_parity_decode_steps,
        eos_token_id=args.eos_token_id,
        layer_ids=draft.config.target_layer_ids,
        hidden_size=draft.config.hidden_size,
        include_token_ids=args.include_token_ids,
    )
    feature_semantics = compare_feature_collector_semantics(
        target,
        prompt,
        layer_ids=draft.config.target_layer_ids,
        hidden_size=draft.config.hidden_size,
        device_type=device_type,
    )
    first_full_logits, first_full_features = _full_prefix_output(target, prompt)
    del first_full_logits
    weight_health = _weight_health(adapter)
    feature_health = _feature_health(
        first_full_features,
        draft.config.target_layer_ids,
        draft.config.hidden_size,
    )
    oracle_capture: dict[str, Tensor] | None = {} if oracle_path is not None else None
    capture_text_ids = tokenizer is not None
    sweep = acceptance_sweep(
        adapter,
        prompt,
        proposal_counts=proposal_counts,
        rounds=args.acceptance_rounds,
        eos_token_id=args.eos_token_id,
        include_token_ids=bool(args.include_token_ids or capture_text_ids),
        trace_draft_layers=bool(args.trace_draft_layers),
        oracle_capture=oracle_capture,
        verification_mode=args.verification_mode,
    )

    text_output: dict[str, object] | None = None
    if tokenizer is not None:
        final_prefix_ids = sweep.get("final_prefix_token_ids")
        if not isinstance(final_prefix_ids, list):
            raise RuntimeError("text diagnosis did not retain the generated prefix")
        continuation_ids = final_prefix_ids[len(prompt_values) :]
        decode = getattr(tokenizer, "decode", None)
        if not callable(decode):
            raise TypeError("local tokenizer does not expose decode")
        text_output = {
            "source": "ordinary_target_greedy_prefix_sweep",
            "generated_token_count": len(continuation_ids),
            "text": decode(continuation_ids, skip_special_tokens=False),
        }
        if not args.include_token_ids:
            sweep.pop("bootstrap_anchor_token_id", None)
            sweep.pop("final_prefix_token_ids", None)
            records = sweep.get("records")
            if isinstance(records, Sequence):
                for record in records:
                    if isinstance(record, dict):
                        record.pop("proposal_token_ids", None)
                        record.pop("target_token_ids", None)
                        record.pop("vectorized_target_token_ids", None)
                        record.pop("correction_or_bonus_token_id", None)

    shadow: dict[str, object] | None = None
    if args.shadow_torch_ops and int(sweep["rounds_completed"]) > 0:
        clean_logits = adapter.forward_logits(prompt)
        anchor = clean_logits[:, -1, :].argmax(dim=-1).to(torch.long)
        shadow_prefix = torch.cat((prompt, anchor.view(1, 1)), dim=1)
        shadow = shadow_torch_ops(
            adapter, shadow_prefix, proposal_count=max(proposal_counts)
        )

    oracle_identity: dict[str, object] | None = None
    if oracle_path is not None:
        assert oracle_capture is not None
        _write_oracle_bundle(
            oracle_path,
            oracle_capture,
            device=args.device,
            dtype=dtype,
            proposal_count=max(proposal_counts),
        )
        oracle_identity = {
            "path": str(oracle_path),
            "sha256": sha256_file(oracle_path),
            "round_index": 0,
            "proposal_count": max(proposal_counts),
            "contains_raw_tensors": True,
            "algorithm": "single_parallel_draft_forward",
        }

    report: dict[str, object] = {
        "schema_version": 3,
        "diagnostic": "qwen3.5-4b-dflash-v1-acceptance",
        "device": str(args.device),
        "device_type": device_type,
        "dtype": str(dtype),
        "eos_token_id": args.eos_token_id,
        "prompt_length": len(prompt_values),
        "prompt_token_sha256": tensor_fingerprint(prompt)["sha256"],
        "token_ids_included": bool(args.include_token_ids),
        "draft_algorithm": "single_parallel_mask_block_forward",
        "draft_layer_trace_enabled": bool(args.trace_draft_layers),
        "draft_backend": backend_name,
        "draft_memory_preflight": draft_memory_preflight,
        "verification_mode": args.verification_mode,
        "enable_thinking": (
            bool(args.enable_thinking)
            if tokenizer is not None and args.prompt_mode == "chat"
            else None
        ),
        "draft_checkpoint": {
            "status": checkpoint["status"],
            "config_sha256": checkpoint["config_sha256"],
            "model_sha256": checkpoint["model_sha256"],
            "model_sha256_verified": bool(args.verify_draft_sha256),
        },
        "target_checkpoint": target_checkpoint,
        "target_w8a8_emulation": target_w8a8_emulation,
        "weight_health": weight_health,
        "feature_health": feature_health,
        "feature_collector_semantics": feature_semantics,
        "target_path_parity": target_path_parity,
        "acceptance_sweep": sweep,
        "text_output": text_output,
        "shadow_torch_ops": shadow,
        "oracle_bundle": oracle_identity,
    }
    if args.include_token_ids:
        report["prompt_token_ids"] = prompt_values
    if reference_report is not None:
        report["report_comparison"] = compare_diagnostic_reports(
            reference_report, report
        )
    else:
        report["report_comparison"] = None
    report["next_actions"] = diagnose_next_actions(report)
    _print_summary(report)

    if report_path is not None:
        _write_report(report_path, report)
        print(f"\nJSON 报告已写入: {report_path}")
    if oracle_path is not None:
        print(f"单轮 oracle tensor bundle 已写入: {oracle_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "acceptance_sweep",
    "compare_diagnostic_reports",
    "compare_framework_target_paths",
    "compare_target_paths",
    "diagnose_next_actions",
    "main",
    "parse_proposal_counts",
    "summarize_acceptance",
    "summarize_acceptance_phases",
    "tensor_fingerprint",
    "tensor_metrics",
]
