"""Diagnose low DFlash acceptance on the embedded NPU route.

This command is intentionally read-only with respect to model weights and the
deployed source tree.  It answers two questions in order:

1. Does a fresh full-prefix target call produce the same last-row logits and
   DFlash features as the target's persistent prefill/decode path?
2. On identical ordinary-greedy prefixes, how does acceptance change for
   proposal counts K=1,3,7,15?

The first question is more fundamental.  Strict-greedy token equality alone
cannot detect a target bridge whose fresh-prefill semantics differ from the
established incremental path, because both ordinary and DFlash validation can
otherwise use the same incorrect full-prefix route.

By default the command prints aggregate numbers only.  Prompt and generated
token IDs are included in an optional JSON report only when
``--include-token-ids`` is explicitly supplied.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import tempfile

import torch
from torch import Tensor, nn

from .dflash_ops import TorchDFlashOps
from .dflash_qwen_adapter_v1 import (
    Qwen35DFlashFullPrefixAdapter,
    _load_target,
    _prepare_device_backend,
    _prompt_ids,
    _select_draft_ops,
)
from .dflash_weights import require_official_dflash_checkpoint
from .internal_target_loader import (
    DECODE_CHUNK_SIZE_ENV,
    PREFILL_CHUNK_SIZE_ENV,
    TARGET_FACTORY_ENV,
)
from .modeling_dflash import DFlashDraftModel


DEFAULT_TARGET_FACTORY = "models.internal_dflash_bridge:load_qwen35_target"
KV_CACHE_MAX_LEN_ENV = "DFLASH_HIAI_KV_CACHE_MAX_LEN"
OFFICIAL_EOS_TOKEN_ID = 248_044


def parse_proposal_counts(raw: str, *, maximum: int = 15) -> tuple[int, ...]:
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
    result.update(
        {
            "reference_rms": float(left_rms.item()),
            "candidate_rms": float(right_rms.item()),
            "max_abs_error": float(absolute.max().item()),
            "mean_abs_error": float(absolute.mean().item()),
            "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
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
    """Compare persistent incremental target state with fresh full-prefix replay."""

    snapshots = _incremental_snapshots(
        target,
        prompt_ids,
        decode_steps=decode_steps,
        eos_token_id=eos_token_id,
    )
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
    return {
        "status": status,
        "comparisons": len(records),
        "all_top1_match": all_top1,
        "all_logits_bitwise_equal": all_logits_exact,
        "all_features_bitwise_equal": all_feature_exact,
        "records": records,
    }


def _build_draft_inputs(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    target_hidden: Tensor,
    proposal_count: int,
) -> tuple[Tensor, Tensor]:
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
    return noise_embedding, position_ids


def _propose_with_features(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prefix_ids: Tensor,
    target_hidden: Tensor,
    proposal_count: int,
) -> Tensor:
    noise_embedding, position_ids = _build_draft_inputs(
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
    return proposals.to(torch.long)


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
) -> dict[str, object]:
    """Evaluate all K values on the same sequence of clean target prefixes."""

    clean_logits = adapter.forward_logits(prompt_ids)
    anchor = clean_logits[:, -1, :].argmax(dim=-1).to(torch.long)
    prefix = torch.cat((prompt_ids, anchor.view(1, 1)), dim=1)
    records: list[dict[str, object]] = []
    stopped_on_eos = int(anchor.item()) == eos_token_id

    for round_index in range(rounds):
        if stopped_on_eos:
            break
        context_hidden = adapter._replay_target_features(prefix[:, :-1])
        verifier_first_tokens: dict[str, int] = {}
        for proposal_count in proposal_counts:
            proposals = _propose_with_features(
                adapter, prefix, context_hidden, proposal_count
            ).reshape(-1)
            eos_positions = torch.nonzero(
                proposals == int(eos_token_id), as_tuple=False
            )
            if eos_positions.numel():
                proposals = proposals[: int(eos_positions[0].item()) + 1]
            verification_ids = torch.cat((prefix, proposals.view(1, -1)), dim=1)
            verification_logits = adapter.forward_logits(verification_ids)
            first_row = int(prefix.shape[1]) - 1
            row_count = int(proposals.numel())
            verification_target_tokens = verification_logits[
                0, first_row : first_row + row_count + 1, :
            ].argmax(dim=-1)
            if int(verification_target_tokens.numel()) != row_count + 1:
                raise RuntimeError("target verification did not expose the bonus row")
            target_tokens = verification_target_tokens[:row_count]
            accepted_count = _accepted_prefix_length(proposals, target_tokens)
            verifier_first = int(target_tokens[0].item())
            verifier_first_tokens[str(proposal_count)] = verifier_first
            accepted_contains_eos = bool(
                (proposals[:accepted_count] == int(eos_token_id)).any().item()
            )
            theoretical_emitted_count = accepted_count + (
                0 if accepted_contains_eos else 1
            )
            record: dict[str, object] = {
                "round_index": round_index,
                "prefix_length": int(prefix.shape[1]),
                "requested_proposal_count": int(proposal_count),
                "actual_proposal_count": row_count,
                "accepted_count": accepted_count,
                "theoretical_emitted_count": theoretical_emitted_count,
                "first_proposal_match": bool(proposals[0] == target_tokens[0]),
                "full_block_accepted": accepted_count == row_count,
            }
            if include_token_ids:
                record["proposal_token_ids"] = proposals.detach().cpu().tolist()
                record["target_token_ids"] = target_tokens.detach().cpu().tolist()
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
        "rounds_requested": rounds,
        "rounds_completed": len({int(record["round_index"]) for record in records}),
        "stopped_on_eos": stopped_on_eos,
        "proposal_counts": list(proposal_counts),
        "metrics_by_proposal_count": summarize_acceptance(records, proposal_counts),
        "records": records,
    }
    if include_token_ids:
        result["bootstrap_anchor_token_id"] = int(anchor.item())
        result["final_prefix_token_ids"] = prefix.detach().cpu().reshape(-1).tolist()
    return result


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
    noise_embedding, position_ids = _build_draft_inputs(
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
    controller = _controller_from_facade(adapter.target)
    execution_model = getattr(controller, "dflash_execution_model")
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
        "target_qlinear_module_count": quantized_linear_count,
        "quantized_target_warning": (
            "QLinear modules were detected; establish non-quantized target parity first"
            if quantized_linear_count
            else None
        ),
    }


def diagnose_next_actions(report: Mapping[str, object]) -> list[str]:
    """Turn measured gates into an ordered, non-speculative investigation list."""

    parity = report.get("target_path_parity")
    if isinstance(parity, Mapping) and not bool(parity.get("all_top1_match")):
        return [
            "先修复 Target 路径：增量路径与 fresh full-prefix 的 Top-1 已分叉；此时接受率没有可解释性。",
            "查看首个分叉 comparison 的 8 个 feature_layer 指标；最早分叉层就是优先排查边界。",
            "核对 full-prefix 的 fresh KV/GDN state、position_ids、"
            "new_kv_cache_pos、allQLen 和 causal mask。",
        ]

    health = report.get("weight_health")
    if isinstance(health, Mapping) and int(health.get("target_qlinear_module_count", 0)):
        return [
            "先用非量化 Target 重跑同一诊断；量化 Target 会把 feature 数值误差与 DFlash 本身混在一起。",
            "非量化路径闭合后，再单独测量量化对每层 feature 和接受率的影响。",
        ]

    sweep = report.get("acceptance_sweep")
    metrics = sweep.get("metrics_by_proposal_count") if isinstance(sweep, Mapping) else None
    k1 = metrics.get("1") if isinstance(metrics, Mapping) else None
    k1_accuracy = k1.get("first_proposal_accuracy") if isinstance(k1, Mapping) else None
    if isinstance(k1_accuracy, (int, float)) and k1_accuracy < 0.5:
        return [
            "K=1 的首 token 命中率已低，优先检查 8 层 feature 数值、草稿 checkpoint 身份和 FP16 backend。",
            "使用 --shadow-torch-ops 比较同一 NPU tensor 上的分解 backend 与 Torch draft；若 Top-1 不同，先收敛草稿算子。",
            "增加 --acceptance-rounds 后再判断，避免少量轮次造成偶然比例。",
        ]
    return [
        "Target 路径未见 Top-1 分叉；扩大 --acceptance-rounds 获取稳定统计。",
        "若 K=1 稳定而 K=7/15 明显退化，使用 --shadow-torch-ops 检查草稿 block attention/位置和低精度误差。",
        "mean_theoretical_emitted_per_verify 才接近吞吐收益口径，不要把 accepted/proposed 百分比直接当官方加速指标。",
    ]


def _safe_report_path(
    raw_path: str,
    *,
    protected_roots: Sequence[Path],
) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_symlink():
        raise ValueError("--report must not be a symlink")
    resolved = path.resolve()
    for root in protected_roots:
        resolved_root = root.expanduser().resolve()
        try:
            resolved.relative_to(resolved_root)
        except ValueError:
            continue
        raise ValueError("--report must be outside source and model directories")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    return resolved


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


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnose DFlash V1 NPU target parity and acceptance"
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--draft-dir", required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-ids", help="comma-separated token IDs")
    prompt.add_argument("--prompt-json", help="JSON token list or input_ids object")
    parser.add_argument("--device", default="npu:0")
    parser.add_argument("--kv-cache-max-len", type=int, required=True)
    parser.add_argument("--prefill-chunk-size", type=int, default=64)
    parser.add_argument("--decode-chunk-size", type=int, default=1)
    parser.add_argument("--target-factory", default=DEFAULT_TARGET_FACTORY)
    parser.add_argument("--target-parity-decode-steps", type=int, default=4)
    parser.add_argument("--acceptance-rounds", type=int, default=8)
    parser.add_argument(
        "--proposal-counts",
        default="1,3,7,15",
        help="K proposal tokens; 1,3,7,15 correspond to total blocks 2,4,8,16",
    )
    parser.add_argument("--eos-token-id", type=int, default=OFFICIAL_EOS_TOKEN_ID)
    parser.add_argument("--shadow-torch-ops", action="store_true")
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
    if str(args.device).split(":", 1)[0].lower() != "npu":
        raise ValueError("this diagnostic currently requires --device npu or npu:N")
    if args.eos_token_id != OFFICIAL_EOS_TOKEN_ID:
        raise ValueError(f"this checkpoint requires --eos-token-id {OFFICIAL_EOS_TOKEN_ID}")
    for name in (
        "kv_cache_max_len",
        "prefill_chunk_size",
        "decode_chunk_size",
        "acceptance_rounds",
    ):
        value = getattr(args, name)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.target_parity_decode_steps < 0:
        raise ValueError("--target-parity-decode-steps must be non-negative")
    if args.kv_cache_max_len % 64:
        raise ValueError("--kv-cache-max-len must be divisible by 64")
    return parse_proposal_counts(args.proposal_counts)


def _print_summary(report: Mapping[str, object]) -> None:
    parity = report["target_path_parity"]
    sweep = report["acceptance_sweep"]
    assert isinstance(parity, Mapping) and isinstance(sweep, Mapping)
    print("\n=== DFlash V1 接受率诊断 ===")
    print(
        "Target 增量 vs full-prefix:",
        parity["status"],
        f"({parity['comparisons']} 个前缀)",
    )
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
            print(
                "首个差异前缀:",
                f"length={record.get('prefix_length')}",
                f"source={record.get('incremental_source')}",
                f"top1_match={record.get('top1_match')}",
                f"logits_max_abs={logits_max_abs}",
                f"feature_max_abs={feature_max_abs}",
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
    print("\n建议顺序：")
    for index, action in enumerate(report["next_actions"], start=1):
        print(f"{index}. {action}")


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    proposal_counts = _validate_args(args)
    package_dir = Path(__file__).resolve().parent
    source_path = package_dir.parent / "modeling_qwen3_5_hiai_nd.py"
    if source_path.is_symlink() or not source_path.is_file():
        raise FileNotFoundError(
            "expected models/modeling_qwen3_5_hiai_nd.py beside dflash_v1"
        )
    target_root = Path(args.target_dir).expanduser().resolve()
    draft_root = Path(args.draft_dir).expanduser().resolve()
    if not target_root.is_dir() or not draft_root.is_dir():
        raise FileNotFoundError("--target-dir and --draft-dir must be existing directories")

    os.environ[TARGET_FACTORY_ENV] = args.target_factory
    os.environ[PREFILL_CHUNK_SIZE_ENV] = str(args.prefill_chunk_size)
    os.environ[DECODE_CHUNK_SIZE_ENV] = str(args.decode_chunk_size)
    os.environ[KV_CACHE_MAX_LEN_ENV] = str(args.kv_cache_max_len)

    _prepare_device_backend(args.device)
    checkpoint = require_official_dflash_checkpoint(
        draft_root, verify_model_hash=bool(args.verify_draft_sha256)
    )
    target_loader = f"{__package__}.internal_target_loader:load_target"
    target = _load_target(
        str(target_root),
        target_loader=target_loader,
        hiai_source=str(source_path),
        device=args.device,
        dtype=torch.float16,
        allow_download=False,
        trust_remote_code=False,
    )
    ops, backend_name = _select_draft_ops(
        device=args.device, ops_backend=None, allow_op_fallback=False
    )
    draft = DFlashDraftModel.from_pretrained(
        draft_root, ops=ops, device=args.device, dtype=torch.float16
    )
    adapter = Qwen35DFlashFullPrefixAdapter(target, draft)
    prompt_values = _prompt_ids(args.prompt_ids, args.prompt_json)
    if any(token < 0 or token >= adapter.vocab_size for token in prompt_values):
        raise ValueError("prompt contains a token outside the target vocabulary")
    prompt = torch.tensor(
        [prompt_values], dtype=torch.long, device=torch.device(args.device)
    )
    if int(prompt.shape[1]) + max(proposal_counts) + args.acceptance_rounds + 1 > min(
        args.kv_cache_max_len, int(draft.config.max_position_embeddings)
    ):
        raise ValueError("prompt plus diagnostic sweep exceeds the configured context")

    target_path_parity = compare_target_paths(
        target,
        prompt,
        decode_steps=args.target_parity_decode_steps,
        eos_token_id=args.eos_token_id,
        layer_ids=draft.config.target_layer_ids,
        hidden_size=draft.config.hidden_size,
        include_token_ids=args.include_token_ids,
    )
    first_full_logits, first_full_features = _full_prefix_output(target, prompt)
    del first_full_logits
    weight_health = _weight_health(adapter)
    feature_health = _feature_health(
        first_full_features,
        draft.config.target_layer_ids,
        draft.config.hidden_size,
    )
    sweep = acceptance_sweep(
        adapter,
        prompt,
        proposal_counts=proposal_counts,
        rounds=args.acceptance_rounds,
        eos_token_id=args.eos_token_id,
        include_token_ids=args.include_token_ids,
    )

    shadow: dict[str, object] | None = None
    if args.shadow_torch_ops and int(sweep["rounds_completed"]) > 0:
        clean_logits = adapter.forward_logits(prompt)
        anchor = clean_logits[:, -1, :].argmax(dim=-1).to(torch.long)
        shadow_prefix = torch.cat((prompt, anchor.view(1, 1)), dim=1)
        shadow = shadow_torch_ops(
            adapter, shadow_prefix, proposal_count=max(proposal_counts)
        )

    report: dict[str, object] = {
        "schema_version": 1,
        "diagnostic": "qwen3.5-4b-dflash-v1-acceptance",
        "device": str(args.device),
        "dtype": str(torch.float16),
        "eos_token_id": args.eos_token_id,
        "prompt_length": len(prompt_values),
        "token_ids_included": bool(args.include_token_ids),
        "draft_backend": backend_name,
        "draft_checkpoint": {
            "status": checkpoint["status"],
            "config_sha256": checkpoint["config_sha256"],
            "model_sha256": checkpoint["model_sha256"],
            "model_sha256_verified": bool(args.verify_draft_sha256),
        },
        "weight_health": weight_health,
        "feature_health": feature_health,
        "target_path_parity": target_path_parity,
        "acceptance_sweep": sweep,
        "shadow_torch_ops": shadow,
    }
    if args.include_token_ids:
        report["prompt_token_ids"] = prompt_values
    report["next_actions"] = diagnose_next_actions(report)
    _print_summary(report)

    if args.report:
        report_path = _safe_report_path(
            args.report,
            protected_roots=(package_dir.parent.parent, target_root, draft_root),
        )
        _write_report(report_path, report)
        print(f"\nJSON 报告已写入: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "acceptance_sweep",
    "compare_target_paths",
    "diagnose_next_actions",
    "main",
    "parse_proposal_counts",
    "summarize_acceptance",
    "tensor_metrics",
]
