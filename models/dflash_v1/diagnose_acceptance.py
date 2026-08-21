"""Diagnose low DFlash acceptance on CPU, CUDA, or the embedded NPU route.

This command is intentionally read-only with respect to model weights and the
deployed source tree.  It answers three questions in order:

1. On NPU, does a fresh full-prefix target call produce the same last-row
   logits and DFlash features as the persistent prefill/decode path?
2. On identical ordinary-greedy prefixes, how does acceptance change for
   proposal counts K=1,3,7,15?
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
    _load_target,
    _prepare_device_backend,
    _prompt_ids,
    _select_draft_ops,
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

    noise_embedding, position_ids = _build_draft_inputs(
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
    }
    oracle: dict[str, Tensor] = {}
    if capture_tensors:
        oracle.update(
            {
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
            else:
                proposed = _propose_with_features(
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
                "round_role": "first_round" if round_index == 0 else "later_round",
                "prefix_length": int(prefix.shape[1]),
                "prefix_token_sha256": prefix_token_sha256,
                "requested_proposal_count": int(proposal_count),
                "actual_proposal_count": row_count,
                "accepted_count": accepted_count,
                "theoretical_emitted_count": theoretical_emitted_count,
                "first_proposal_match": bool(proposals[0] == target_tokens[0]),
                "full_block_accepted": accepted_count == row_count,
                "proposal_token_fingerprint": tensor_fingerprint(proposals),
                "target_token_fingerprint": tensor_fingerprint(target_tokens),
                "correction_or_bonus_token_fingerprint": tensor_fingerprint(
                    verification_target_tokens[accepted_count : accepted_count + 1]
                ),
            }
            if draft_trace is not None:
                record["draft_trace"] = draft_trace
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
        "prefix_policy": "ordinary_target_greedy_same_prefix_for_all_k",
        "rounds_requested": rounds,
        "rounds_completed": len({int(record["round_index"]) for record in records}),
        "stopped_on_eos": stopped_on_eos,
        "proposal_counts": list(proposal_counts),
        "draft_layer_trace_enabled": bool(trace_draft_layers),
        "metrics_by_proposal_count": summarize_acceptance(records, proposal_counts),
        "records": records,
    }
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

    parity = report.get("target_path_parity")
    if isinstance(parity, Mapping) and parity.get("all_top1_match") is False:
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
        actions = [
            "K=1 的首 token 命中率已低，优先检查 8 层 feature 数值、草稿 checkpoint 身份和 FP16 backend。",
            "使用 --trace-draft-layers 记录 target feature、projection、position、"
            "6 层 draft 和 Top-1 的逐轮指纹。",
            "增加 --acceptance-rounds 后再判断，避免少量轮次造成偶然比例。",
        ]
        if report.get("device_type") == "npu":
            actions.insert(
                2,
                "使用 --shadow-torch-ops 比较同一 NPU tensor 上的分解 backend "
                "与 Torch draft；若 Top-1 不同，先收敛草稿算子。",
            )
        elif report.get("device_type") == "cuda" and report.get("dtype") == str(
            torch.float16
        ):
            actions.insert(
                2,
                "若 CUDA 支持 BF16，用完全相同 prompt/K/轮数重跑 --dtype "
                "bfloat16，并用 --compare-report 对比 FP16 报告。",
            )
        return actions
    return [
        "Target 路径未见 Top-1 分叉；扩大 --acceptance-rounds 获取稳定统计。",
        "若 K=1 稳定而 K=7/15 明显退化，打开 --trace-draft-layers 检查 block attention、位置和低精度边界。",
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
                "schema_version": "1",
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
    parser.add_argument("--device", default="npu:0")
    parser.add_argument(
        "--dtype",
        choices=tuple(SUPPORTED_DTYPES),
        default="float16",
        help="NPU is locked to float16; CUDA BF16 is the recommended dtype A/B",
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
        "--proposal-counts",
        default="1,3,7,15",
        help="K proposal tokens; 1,3,7,15 correspond to total blocks 2,4,8,16",
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
        help="opt-in first-round .safetensors inputs/outputs for an independent official oracle",
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
    if (args.prompt_ids is None) == (args.prompt_json is None):
        raise ValueError("diagnosis requires exactly one of --prompt-ids/--prompt-json")
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
    )
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

    protected_roots = (package_dir.parent.parent, target_root, draft_root)
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
    input_paths = {path for path in (comparison_path, prompt_json_path) if path is not None}
    if report_path is not None and report_path in input_paths:
        raise ValueError("--report must not overwrite --compare-report or --prompt-json")
    if oracle_path is not None and oracle_path in input_paths:
        raise ValueError(
            "--oracle-bundle must not overwrite --compare-report or --prompt-json"
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
    ops, backend_name = _select_draft_ops(
        device=args.device, ops_backend=None, allow_op_fallback=False
    )
    draft = DFlashDraftModel.from_pretrained(
        draft_root, ops=ops, device=args.device, dtype=dtype
    )
    adapter = Qwen35DFlashFullPrefixAdapter(target, draft)
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

    if device_type == "npu":
        target_path_parity = compare_target_paths(
            target,
            prompt,
            decode_steps=args.target_parity_decode_steps,
            eos_token_id=args.eos_token_id,
            layer_ids=draft.config.target_layer_ids,
            hidden_size=draft.config.hidden_size,
            include_token_ids=args.include_token_ids,
        )
    else:
        target_path_parity = {
            "status": "NOT_APPLICABLE_FRAMEWORK_FULL_PREFIX_ONLY",
            "comparisons": 0,
            "records": [],
            "message": (
                "CPU/CUDA package target has no receiver persistent-state path; "
                "use same-prompt report comparison for device/dtype localization"
            ),
        }
    first_full_logits, first_full_features = _full_prefix_output(target, prompt)
    del first_full_logits
    weight_health = _weight_health(adapter)
    feature_health = _feature_health(
        first_full_features,
        draft.config.target_layer_ids,
        draft.config.hidden_size,
    )
    oracle_capture: dict[str, Tensor] | None = {} if oracle_path is not None else None
    sweep = acceptance_sweep(
        adapter,
        prompt,
        proposal_counts=proposal_counts,
        rounds=args.acceptance_rounds,
        eos_token_id=args.eos_token_id,
        include_token_ids=args.include_token_ids,
        trace_draft_layers=bool(args.trace_draft_layers),
        oracle_capture=oracle_capture,
    )

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
        "schema_version": 2,
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
    "compare_target_paths",
    "diagnose_next_actions",
    "main",
    "parse_proposal_counts",
    "summarize_acceptance",
    "tensor_fingerprint",
    "tensor_metrics",
]
