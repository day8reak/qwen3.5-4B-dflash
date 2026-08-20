"""Qwen3.5 adapter for the correctness-first DFlash replay golden.

This is the deliberately slow V1 route.  It never supplies a target cache and
does not attempt to save, commit, or roll back Qwen3.5 Gated DeltaNet state.
Every feature-capture call replays all committed tokens before the clean
anchor, and every verification call replays the complete prefix plus proposal
block.

The adapter implements both protocols consumed by
``dflash_reference_decode_v1``:

* ``forward_logits(input_ids)`` returns full target logits for verification;
* ``propose(prefix_ids, K)`` builds ``[anchor, K * mask]`` and runs the
  official six-layer DFlash draft with the target embedding and LM head.

``block_size`` includes the clean anchor.  Therefore the official Qwen3.5
checkpoint's block size 16 exposes at most 15 proposal tokens per round.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import inspect
import json
import operator
import os
from pathlib import Path
import platform
import sys
from typing import Any, Callable

sys.dont_write_bytecode = True

import torch
from torch import Tensor, nn

from .dflash_config import audit_official_4b_dflash_config
from .dflash_hiai_feature_check import verify_direct_source_file
from .dflash_ops import ModuleDFlashOps, TorchDFlashOps
from .dflash_reference_decode_v1 import (
    ReplayDecodeResult,
    ReplayDecodeStats,
    ReplayRound,
    assert_exact_greedy_match,
    dflash_full_prefix_greedy,
    ordinary_full_prefix_greedy,
)
from .dflash_weights import require_official_dflash_checkpoint
from .modeling_dflash import DFlashDraftModel


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}

_RESERVED_TARGET_KWARGS = frozenset(
    {
        "input_ids",
        "inputs_embeds",
        "use_cache",
        "past_key_values",
        "return_dict",
        "output_hidden_states",
        "output_dflash_features",
        "logits_to_keep",
        # Receiver-owned mutable state must be prepared by the target facade,
        # never injected as stale per-call kwargs by the portable adapter.
        "cache_params",
        "cache_state",
        "kv_cache",
        "conv_state",
        "recurrent_state",
        "initial_state",
        "new_kv_cache_pos",
        "allQLen",
        "token_count",
        "export_flag",
    }
)

_FORMAL_ISOLATION_MODES = frozenset(
    {"receiver_reset_hook", "fresh_instance"}
)
_HIAI_FEATURE_SOURCE = "receiver_owned:modeling_qwen3_5_hiai_nd.py"
_HIAI_CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
_HIAI_FEATURE_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"
_FACADE_CONTRACT_ID = "qwen3.5-4b-dflash-v1-full-prefix-isolation-r6"
_FORMAL_EOS_TOKEN_ID = 248044
_NPU_LAYOUT_EMBEDDED = "embedded"
_TARGET_FACTORY_ENV = "DFLASH_HIAI_TARGET_FACTORY"
_RESET_HOOK_ENV = "DFLASH_HIAI_RESET_HOOK"
_EMBEDDED_RUNTIME_FILES = frozenset(
    {
        "dflash_ascend310p_ops.py",
        "dflash_config.py",
        "dflash_hiai_feature_check.py",
        "dflash_hiai_feature_runtime.py",
        "dflash_ops.py",
        "dflash_qwen_adapter_v1.py",
        "dflash_reference_decode_v1.py",
        "dflash_target_features.py",
        "dflash_target_hook_bridge.py",
        "dflash_weights.py",
        "internal_target_loader_template.py",
        "internal_target_loader.py",
        "modeling_dflash.py",
        "modeling_qwen3_5_dflash.py",
        "run_npu.py",
    }
)
_TARGET_STATE_OUTPUT_FIELDS = (
    "past_key_values",
    "cache_params",
    "cache_state",
    "kv_cache",
    "conv_state",
    "recurrent_state",
    "initial_state",
    "new_kv_cache_pos",
    "allQLen",
    "token_count",
    "export_flag",
)


def _tensor_field(output: object, name: str) -> Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    return value if isinstance(value, Tensor) else None


def _module_weight(module: object, *, name: str) -> Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError(f"target {name} must expose a Tensor weight")
    return weight


def _target_weight(
    target: object,
    getter_name: str,
    explicit_weight: Tensor | None,
    *,
    name: str,
) -> Tensor:
    if explicit_weight is not None:
        if not isinstance(explicit_weight, Tensor):
            raise TypeError(f"explicit target {name} weight must be a Tensor")
        return explicit_weight
    getter = getattr(target, getter_name, None)
    if not callable(getter):
        raise TypeError(
            f"target must provide {getter_name}() or an explicit {name} weight"
        )
    module = getter()
    if module is None:
        raise TypeError(f"target {getter_name}() returned None")
    return _module_weight(module, name=name)


def _draft_parameter_identity(draft: nn.Module) -> tuple[torch.device, torch.dtype]:
    parameter = next(draft.parameters(), None)
    if parameter is None:
        raise ValueError("DFlash draft model has no parameters")
    if not torch.is_floating_point(parameter):
        raise TypeError("DFlash draft parameters must use a floating-point dtype")
    return parameter.device, parameter.dtype


def _validate_token_tensor(
    input_ids: Tensor,
    *,
    device: torch.device,
    vocab_size: int,
    name: str,
) -> Tensor:
    if not isinstance(input_ids, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError(f"{name} must have shape [1, sequence_length]")
    if input_ids.shape[1] == 0:
        raise ValueError(f"{name} must contain at least one token")
    if input_ids.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"{name} must use an integer dtype")
    if input_ids.device != device:
        raise ValueError(
            f"{name} is on {input_ids.device}, but target weights are on {device}"
        )
    minimum = int(input_ids.min().item())
    maximum = int(input_ids.max().item())
    if minimum < 0 or maximum >= vocab_size:
        raise ValueError(
            f"{name} contains a token outside [0, {vocab_size - 1}]"
        )
    return input_ids.to(dtype=torch.long)


def _proposal_count(value: int, *, maximum: int) -> int:
    if isinstance(value, bool):
        raise TypeError("max_draft_tokens must be an integer, not bool")
    try:
        count = int(operator.index(value))
    except TypeError as error:
        raise TypeError("max_draft_tokens must be an integer") from error
    if count <= 0:
        raise ValueError("max_draft_tokens must be positive")
    if count > maximum:
        raise ValueError(
            "max_draft_tokens exceeds the DFlash proposal capacity: "
            f"requested {count}, maximum {maximum} "
            "(checkpoint block_size includes one clean anchor)"
        )
    return count


@dataclass
class Qwen35FullPrefixAdapterStats:
    """Target work performed by the concrete Qwen3.5 adapter."""

    target_logit_calls: int = 0
    target_logit_tokens_recomputed: int = 0
    target_feature_calls: int = 0
    target_feature_tokens_recomputed: int = 0
    draft_calls: int = 0
    draft_context_tokens_recomputed: int = 0
    draft_block_tokens: int = 0
    proposed_tokens: int = 0


class Qwen35DFlashFullPrefixAdapter:
    """Bind a feature-enabled Qwen3.5 target to ``DFlashDraftModel``.

    Args:
        target: Eval-mode causal LM whose forward accepts
            ``output_dflash_features`` and whose output exposes ``logits`` and,
            for opted-in calls, ``dflash_features``.
        draft: Eval-mode official DFlash draft model.
        target_forward_kwargs: Optional immutable per-call target arguments,
            such as an attention backend control.  Cache/output control keys
            are reserved so callers cannot accidentally turn V1 into a
            stateful path.
        input_embedding_weight: Optional explicit target input embedding
            weight.  Otherwise ``target.get_input_embeddings().weight`` is
            used.
        lm_head_weight: Optional explicit target output head weight.  Otherwise
            ``target.get_output_embeddings().weight`` is used.
        check_finite_features: Fail before drafting when a captured target
            feature or block embedding is non-finite.
        require_official_config: Keep enabled outside reduced-shape unit tests;
            it rejects any draft architecture other than the locked public
            Qwen3.5-4B-DFlash checkpoint contract.
    """

    def __init__(
        self,
        target: nn.Module,
        draft: DFlashDraftModel,
        *,
        target_forward_kwargs: Mapping[str, Any] | None = None,
        input_embedding_weight: Tensor | None = None,
        lm_head_weight: Tensor | None = None,
        check_finite_features: bool = True,
        require_official_config: bool = True,
    ) -> None:
        if not isinstance(target, nn.Module):
            raise TypeError("target must be a torch.nn.Module")
        if not isinstance(draft, DFlashDraftModel):
            raise TypeError("draft must be a DFlashDraftModel")
        if target.training or draft.training:
            raise ValueError("target and draft must both be in eval mode")
        if require_official_config:
            mismatches = audit_official_4b_dflash_config(draft.config)
            if mismatches:
                raise ValueError(
                    "draft config is not the locked official Qwen3.5-4B-DFlash "
                    "contract: " + "; ".join(mismatches)
                )

        kwargs = dict(target_forward_kwargs or {})
        conflicts = sorted(_RESERVED_TARGET_KWARGS.intersection(kwargs))
        if conflicts:
            raise ValueError(
                "target_forward_kwargs contains V1-reserved keys: "
                + ", ".join(conflicts)
            )

        self.target = target
        self.draft = draft
        self.target_forward_kwargs = kwargs
        self.check_finite_features = bool(check_finite_features)
        self.input_embedding_weight = _target_weight(
            target,
            "get_input_embeddings",
            input_embedding_weight,
            name="input embedding",
        )
        self.lm_head_weight = _target_weight(
            target,
            "get_output_embeddings",
            lm_head_weight,
            name="LM head",
        )
        self._validate_weight_contract()
        self.stats = Qwen35FullPrefixAdapterStats()

    @property
    def device(self) -> torch.device:
        return self.input_embedding_weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.input_embedding_weight.dtype

    @property
    def vocab_size(self) -> int:
        return int(self.draft.config.vocab_size)

    @property
    def max_proposal_tokens(self) -> int:
        # Official DFlash's configured block contains anchor + proposals.
        return int(self.draft.config.block_size) - 1

    def reset_stats(self) -> None:
        self.stats = Qwen35FullPrefixAdapterStats()

    def snapshot_stats(self) -> Qwen35FullPrefixAdapterStats:
        return replace(self.stats)

    def _validate_weight_contract(self) -> None:
        config = self.draft.config
        expected_embedding = (int(config.vocab_size), int(config.hidden_size))
        if tuple(self.input_embedding_weight.shape) != expected_embedding:
            raise ValueError(
                "target input embedding shape does not match DFlash: "
                f"expected {expected_embedding}, got "
                f"{tuple(self.input_embedding_weight.shape)}"
            )
        if tuple(self.lm_head_weight.shape) != expected_embedding:
            raise ValueError(
                "target LM-head shape does not match DFlash: "
                f"expected {expected_embedding}, got "
                f"{tuple(self.lm_head_weight.shape)}"
            )
        if not torch.is_floating_point(self.input_embedding_weight):
            raise TypeError("target input embedding must be floating point")
        if not torch.is_floating_point(self.lm_head_weight):
            raise TypeError("target LM head must be floating point")
        if (
            self.input_embedding_weight.device != self.lm_head_weight.device
            or self.input_embedding_weight.dtype != self.lm_head_weight.dtype
        ):
            raise ValueError("target input embedding and LM head differ in device or dtype")

        draft_device, draft_dtype = _draft_parameter_identity(self.draft)
        if draft_device != self.input_embedding_weight.device:
            raise ValueError(
                f"draft is on {draft_device}, target weights are on "
                f"{self.input_embedding_weight.device}"
            )
        if draft_dtype != self.input_embedding_weight.dtype:
            raise ValueError(
                f"draft uses {draft_dtype}, target weights use "
                f"{self.input_embedding_weight.dtype}"
            )
        if self.max_proposal_tokens <= 0:
            raise ValueError("DFlash block_size must leave room for a proposal")

    def _target_forward(
        self,
        input_ids: Tensor,
        *,
        features: bool,
        logits_to_keep: int,
    ) -> object:
        kwargs = dict(self.target_forward_kwargs)
        kwargs.update(
            {
                "use_cache": False,
                "return_dict": True,
                "output_hidden_states": False,
                "output_dflash_features": features,
                "logits_to_keep": logits_to_keep,
            }
        )
        with torch.inference_mode():
            output = self.target(input_ids=input_ids, **kwargs)
        returned_state = [
            name
            for name in _TARGET_STATE_OUTPUT_FIELDS
            if (
                output.get(name)
                if isinstance(output, Mapping)
                else getattr(output, name, None)
            )
            is not None
        ]
        if returned_state:
            raise RuntimeError(
                "target returned cache/state even though V1 uses receiver-owned "
                "full-prefix isolation: " + ", ".join(returned_state)
            )
        return output

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        """Replay the complete verification sequence and return all logits."""

        input_ids = _validate_token_tensor(
            input_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="verification input_ids",
        )
        # Verification needs every row, not just the final LM-head row.
        output = self._target_forward(
            input_ids,
            features=False,
            logits_to_keep=0,
        )
        logits = _tensor_field(output, "logits")
        if logits is None:
            raise TypeError("target output does not expose Tensor logits")
        expected = (1, input_ids.shape[1], self.vocab_size)
        if tuple(logits.shape) != expected:
            raise ValueError(
                "target must return full-prefix logits for V1 verification: "
                f"expected {expected}, got {tuple(logits.shape)}"
            )
        if not torch.is_floating_point(logits):
            raise TypeError("target logits must use a floating-point dtype")
        self.stats.target_logit_calls += 1
        self.stats.target_logit_tokens_recomputed += int(input_ids.shape[1])
        return logits

    def _replay_target_features(self, context_ids: Tensor) -> Tensor:
        context_length = int(context_ids.shape[1])
        if context_length == 0:
            return self.input_embedding_weight.new_empty(
                (1, 0, int(self.draft.config.feature_size))
            )

        # Feature capture needs every decoder row but only one LM-head row.
        output = self._target_forward(
            context_ids,
            features=True,
            logits_to_keep=1,
        )
        features = _tensor_field(output, "dflash_features")
        if features is None:
            raise TypeError(
                "feature-enabled target output does not expose Tensor "
                "dflash_features; use a feature-enabled target modeling"
            )
        expected = (1, context_length, int(self.draft.config.feature_size))
        if tuple(features.shape) != expected:
            raise ValueError(
                f"target dflash_features must have shape {expected}, "
                f"got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features):
            raise TypeError("target dflash_features must be floating point")
        if features.device != self.device or features.dtype != self.dtype:
            raise ValueError(
                "target dflash_features differ from the shared embedding in "
                "device or dtype"
            )
        if self.check_finite_features and not bool(torch.isfinite(features).all()):
            raise FloatingPointError("target dflash_features contain non-finite values")
        self.stats.target_feature_calls += 1
        self.stats.target_feature_tokens_recomputed += context_length
        return features

    def validate_feature_capture_zero_impact(self, input_ids: Tensor) -> None:
        """Require opt-in feature capture to leave target logits bitwise unchanged."""

        input_ids = _validate_token_tensor(
            input_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="feature zero-impact input_ids",
        )
        ordinary_output = self._target_forward(
            input_ids,
            features=False,
            logits_to_keep=0,
        )
        ordinary_logits = _tensor_field(ordinary_output, "logits")
        if ordinary_logits is None:
            raise TypeError("ordinary zero-impact output must expose Tensor logits")
        expected_logits = (1, input_ids.shape[1], self.vocab_size)
        if tuple(ordinary_logits.shape) != expected_logits:
            raise ValueError(
                "feature zero-impact gate requires full target logits with shape "
                f"{expected_logits}"
            )
        # Closed runtimes may reuse output storage across calls.  Snapshot the
        # ordinary result before the feature-enabled forward can overwrite it.
        ordinary_logits = ordinary_logits.detach().clone()
        feature_output = self._target_forward(
            input_ids,
            features=True,
            logits_to_keep=0,
        )
        feature_logits = _tensor_field(feature_output, "logits")
        if feature_logits is None:
            raise TypeError("feature zero-impact output must expose Tensor logits")
        if tuple(feature_logits.shape) != expected_logits:
            raise ValueError(
                "feature zero-impact gate requires full target logits with shape "
                f"{expected_logits}"
            )
        feature_logits = feature_logits.detach().clone()
        if (
            ordinary_logits.dtype != feature_logits.dtype
            or ordinary_logits.device != feature_logits.device
        ):
            raise ValueError("feature capture changed target logits dtype or device")
        if not torch.equal(ordinary_logits, feature_logits):
            mismatch = ordinary_logits != feature_logits
            mismatch_count = int(mismatch.sum().item())
            maximum_error = float(
                (ordinary_logits.float() - feature_logits.float()).abs().max().item()
            )
            raise AssertionError(
                "output_dflash_features changed target logits: "
                f"{mismatch_count} elements differ, max_abs_error={maximum_error}"
            )
        features = _tensor_field(feature_output, "dflash_features")
        expected_features = (
            1,
            input_ids.shape[1],
            int(self.draft.config.feature_size),
        )
        if features is None or tuple(features.shape) != expected_features:
            actual = None if features is None else tuple(features.shape)
            raise ValueError(
                "feature zero-impact output has an invalid dflash_features shape: "
                f"expected {expected_features}, got {actual}"
            )

    def validate_full_prefix_state_isolation(self, input_ids: Tensor) -> None:
        """Run P→Q→P probes for ordinary and feature-enabled target calls.

        Receiver declarations alone cannot prove that an in-place KV/GDN
        state was reset.  The repeated P outputs must therefore be raw-bit
        identical after an intervening different prefix Q.  This is a bounded
        behavioral gate, not a replacement for a device execution trace.
        """

        prefix = _validate_token_tensor(
            input_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="full-prefix isolation probe input_ids",
        )
        if self.vocab_size <= 1:
            raise ValueError("full-prefix isolation probe requires vocab_size > 1")
        maximum_positions = int(self.draft.config.max_position_embeddings)
        if int(prefix.shape[1]) < maximum_positions:
            different_token = (prefix[0, -1] + 1) % self.vocab_size
            different = torch.cat(
                (prefix, different_token.reshape(1, 1)),
                dim=1,
            )
        elif int(prefix.shape[1]) > 1:
            different = prefix[:, :-1]
        else:
            raise ValueError(
                "full-prefix isolation probe cannot construct a different-length "
                "prefix within max_position_embeddings"
            )

        def snapshot(ids: Tensor, *, features: bool) -> tuple[Tensor, Tensor | None]:
            output = self._target_forward(
                ids,
                features=features,
                logits_to_keep=1,
            )
            logits = _tensor_field(output, "logits")
            if logits is None or logits.ndim != 3 or logits.shape[0] != 1:
                raise TypeError("state-isolation probe requires target logits")
            if logits.shape[1] not in (1, ids.shape[1]):
                raise ValueError("state-isolation probe received invalid logit rows")
            captured = _tensor_field(output, "dflash_features")
            if features:
                expected = (1, ids.shape[1], int(self.draft.config.feature_size))
                if captured is None or tuple(captured.shape) != expected:
                    raise ValueError(
                        "state-isolation probe received invalid dflash_features"
                    )
            elif captured is not None:
                raise ValueError(
                    "state-isolation probe received features while disabled"
                )
            return (
                logits.detach().clone(),
                None if captured is None else captured.detach().clone(),
            )

        ordinary_before, _ = snapshot(prefix, features=False)
        snapshot(different, features=True)
        ordinary_after, _ = snapshot(prefix, features=False)
        if not torch.equal(ordinary_before, ordinary_after):
            raise AssertionError(
                "full-prefix state isolation failed: ordinary P logits changed "
                "after an intervening Q feature call"
            )

        feature_logits_before, feature_before = snapshot(prefix, features=True)
        snapshot(different, features=False)
        feature_logits_after, feature_after = snapshot(prefix, features=True)
        if not torch.equal(feature_logits_before, feature_logits_after):
            raise AssertionError(
                "full-prefix state isolation failed: feature-mode P logits "
                "changed after an intervening Q ordinary call"
            )
        assert feature_before is not None and feature_after is not None
        if not torch.equal(feature_before, feature_after):
            raise AssertionError(
                "full-prefix state isolation failed: P features changed after "
                "an intervening Q ordinary call"
            )

    def propose(self, prefix_ids: Tensor, max_draft_tokens: int) -> Tensor:
        """Return ``K`` DFlash Top-1 proposals for one committed prefix.

        ``prefix_ids[:, -1]`` is the clean anchor.  Target features cover the
        entire causal context before that anchor, i.e. ``prefix_ids[:, :-1]``.
        This is the cache-free equivalent of the official upstream cache
        layout and avoids duplicating the anchor in target context K/V.
        """

        prefix_ids = _validate_token_tensor(
            prefix_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="draft prefix_ids",
        )
        proposal_count = _proposal_count(
            max_draft_tokens,
            maximum=self.max_proposal_tokens,
        )
        context_ids = prefix_ids[:, :-1]
        target_hidden = self._replay_target_features(context_ids)

        block_length = proposal_count + 1
        block_ids = torch.full(
            (1, block_length),
            int(self.draft.config.mask_token_id),
            dtype=torch.long,
            device=self.device,
        )
        block_ids[:, 0] = prefix_ids[:, -1]
        noise_embedding = self.draft.embed_block(
            block_ids,
            self.input_embedding_weight,
        )
        expected_noise = (1, block_length, int(self.draft.config.hidden_size))
        if tuple(noise_embedding.shape) != expected_noise:
            raise RuntimeError(
                f"DFlash block embedding must have shape {expected_noise}, "
                f"got {tuple(noise_embedding.shape)}"
            )
        if (
            noise_embedding.device != target_hidden.device
            or noise_embedding.dtype != target_hidden.dtype
        ):
            raise ValueError(
                "target features and draft block embedding differ in device or dtype"
            )
        if self.check_finite_features and not bool(
            torch.isfinite(noise_embedding).all()
        ):
            raise FloatingPointError("DFlash block embedding contains non-finite values")

        total_positions = int(context_ids.shape[1]) + block_length
        if total_positions > int(self.draft.config.max_position_embeddings):
            raise ValueError(
                "DFlash context plus block exceeds max_position_embeddings: "
                f"{total_positions} > {self.draft.config.max_position_embeddings}"
            )
        position_ids = torch.arange(
            total_positions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            proposed = self.draft.draft_top1(
                target_hidden,
                noise_embedding,
                position_ids,
                self.lm_head_weight,
            )
        expected_proposals = (1, proposal_count)
        if tuple(proposed.shape) != expected_proposals:
            raise RuntimeError(
                f"DFlash Top-1 output must have shape {expected_proposals}, "
                f"got {tuple(proposed.shape)}"
            )
        if proposed.dtype not in _INTEGER_DTYPES:
            raise TypeError("DFlash Top-1 output must use an integer dtype")
        if (
            int(proposed.min().item()) < 0
            or int(proposed.max().item()) >= self.vocab_size
        ):
            raise ValueError("DFlash proposed a token outside the target vocabulary")

        self.stats.draft_calls += 1
        self.stats.draft_context_tokens_recomputed += int(context_ids.shape[1])
        self.stats.draft_block_tokens += block_length
        self.stats.proposed_tokens += proposal_count
        return proposed


@dataclass(frozen=True)
class Qwen35GoldenValidation:
    """Ordinary and speculative outputs plus adapter-level replay evidence."""

    ordinary: ReplayDecodeResult
    dflash: ReplayDecodeResult
    ordinary_adapter_stats: Qwen35FullPrefixAdapterStats
    dflash_adapter_stats: Qwen35FullPrefixAdapterStats
    feature_capture_zero_impact: bool
    bounded_full_prefix_repeatability: bool
    predecode_gate_target_calls: int


def validate_qwen35_dflash_strict_greedy(
    adapter: Qwen35DFlashFullPrefixAdapter,
    prompt_token_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    max_draft_tokens: int | None = None,
    eos_token_ids: Iterable[int] = (),
    progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> Qwen35GoldenValidation:
    """Run both V1 paths and require zero final token-ID mismatch."""

    def notify(event: str, **fields: object) -> None:
        if progress_callback is not None:
            progress_callback(event, fields)

    proposal_count = (
        adapter.max_proposal_tokens
        if max_draft_tokens is None
        else _proposal_count(
            max_draft_tokens,
            maximum=adapter.max_proposal_tokens,
        )
    )
    if isinstance(prompt_token_ids, Tensor):
        gate_ids = prompt_token_ids
        if gate_ids.ndim == 1:
            gate_ids = gate_ids.unsqueeze(0)
        gate_ids = gate_ids.to(device=adapter.device)
    else:
        gate_ids = torch.tensor(
            [list(prompt_token_ids)],
            dtype=torch.long,
            device=adapter.device,
        )
    notify("state_isolation_gate_begin", prompt_tokens=int(gate_ids.shape[1]))
    adapter.validate_full_prefix_state_isolation(gate_ids)
    notify("state_isolation_gate_pass", target_calls=6)
    notify("feature_gate_begin", prompt_tokens=int(gate_ids.shape[1]))
    adapter.validate_feature_capture_zero_impact(gate_ids)
    notify("feature_gate_pass")
    adapter.reset_stats()
    notify("ordinary_greedy_begin", max_new_tokens=max_new_tokens)
    ordinary = ordinary_full_prefix_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    ordinary_stats = adapter.snapshot_stats()
    notify(
        "ordinary_greedy_end",
        generated_tokens=len(ordinary.generated_token_ids),
        stop_reason=ordinary.stop_reason,
    )

    # Official DFlash first asks the target for one clean token.  That token is
    # the anchor at draft-block row zero, while target features cover the full
    # prompt before it.  Starting the draft directly from the final prompt
    # token is target-greedy-correct after verification, but it is not the
    # checkpoint's trained data flow and can destroy the real acceptance rate.
    adapter.reset_stats()
    notify("target_bootstrap_begin")
    bootstrap = ordinary_full_prefix_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=min(max_new_tokens, 1),
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    bootstrap_token = (
        bootstrap.generated_token_ids[0]
        if bootstrap.generated_token_ids
        else None
    )
    notify(
        "target_bootstrap_end",
        generated_tokens=len(bootstrap.generated_token_ids),
        reached_eos=bootstrap.reached_eos,
    )
    if bootstrap_token is None or bootstrap.reached_eos:
        tail = None
    else:
        seeded_prefix = (*bootstrap.prompt_token_ids, bootstrap_token)
        notify(
            "dflash_replay_begin",
            max_new_tokens=max_new_tokens - 1,
            max_draft_tokens=proposal_count,
        )
        tail = dflash_full_prefix_greedy(
            adapter,
            adapter,
            seeded_prefix,
            max_new_tokens=max_new_tokens - 1,
            max_draft_tokens=proposal_count,
            eos_token_ids=eos_token_ids,
            input_device=adapter.device,
        )
        notify(
            "dflash_replay_end",
            generated_tokens=len(tail.generated_token_ids),
            rounds=len(tail.rounds),
            stop_reason=tail.stop_reason,
        )

    tail_stats = ReplayDecodeStats() if tail is None else tail.stats
    bootstrap_trace = (
        ()
        if bootstrap_token is None
        else (
            ReplayRound(
                committed_prefix_length=len(bootstrap.prompt_token_ids),
                proposed_token_ids=(),
                target_token_ids=(bootstrap_token,),
                accepted_draft_token_ids=(),
                fallback_token_id=bootstrap_token,
                emitted_token_ids=(bootstrap_token,),
            ),
        )
    )
    replay_stats = ReplayDecodeStats(
        target_calls=bootstrap.stats.target_calls + tail_stats.target_calls,
        target_verify_calls=(
            bootstrap.stats.target_verify_calls + tail_stats.target_verify_calls
        ),
        target_input_tokens_recomputed=(
            bootstrap.stats.target_input_tokens_recomputed
            + tail_stats.target_input_tokens_recomputed
        ),
        target_rows_read=(
            bootstrap.stats.target_rows_read + tail_stats.target_rows_read
        ),
        draft_calls=tail_stats.draft_calls,
        drafted_tokens=tail_stats.drafted_tokens,
        accepted_draft_tokens=tail_stats.accepted_draft_tokens,
        rejected_draft_tokens=tail_stats.rejected_draft_tokens,
        fallback_tokens=(1 if bootstrap_token is not None else 0)
        + tail_stats.fallback_tokens,
    )
    tail_generated = () if tail is None else tail.generated_token_ids
    generated = (
        ()
        if bootstrap_token is None
        else (bootstrap_token, *tail_generated)
    )
    reached_eos = bootstrap.reached_eos or (
        False if tail is None else tail.reached_eos
    )
    dflash = ReplayDecodeResult(
        mode="qwen3.5-dflash-v1-target-bootstrap-full-prefix-replay",
        prompt_token_ids=bootstrap.prompt_token_ids,
        generated_token_ids=generated,
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        stats=replay_stats,
        rounds=bootstrap_trace + (() if tail is None else tail.rounds),
    )
    dflash_stats = adapter.snapshot_stats()
    assert_exact_greedy_match(ordinary, dflash)
    notify(
        "exact_token_gate_pass",
        generated_tokens=len(dflash.generated_token_ids),
        stop_reason=dflash.stop_reason,
    )
    return Qwen35GoldenValidation(
        ordinary=ordinary,
        dflash=dflash,
        ordinary_adapter_stats=ordinary_stats,
        dflash_adapter_stats=dflash_stats,
        feature_capture_zero_impact=True,
        bounded_full_prefix_repeatability=True,
        predecode_gate_target_calls=8,
    )


def _dtype(name: str) -> torch.dtype:
    values = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return values[name]


def _validate_experiment_dtype(
    device: str | torch.device,
    dtype: torch.dtype,
) -> None:
    device_type = str(device).split(":", 1)[0].lower()
    if device_type == "npu" and dtype != torch.float16:
        raise ValueError(
            "the approved Ascend 310P V1 experiment requires --dtype float16"
        )
    if device_type == "cuda" and dtype == torch.bfloat16:
        is_bf16_supported = getattr(torch.cuda, "is_bf16_supported", None)
        if not callable(is_bf16_supported) or not bool(is_bf16_supported()):
            raise ValueError(
                "--dtype bfloat16 requires a CUDA device with BF16 support; "
                "use float16 otherwise"
            )


def _load_callable(specification: str) -> Callable[..., nn.Module]:
    if ":" not in specification:
        raise ValueError("target loader must use MODULE:FUNCTION syntax")
    module_name, function_name = specification.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise TypeError(f"target loader is not callable: {specification}")
    return function


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callable_source_identity(
    function: object,
    *,
    specification: str,
) -> dict[str, object]:
    identity: dict[str, object] = {
        "specification": specification,
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "source_file": None,
        "source_sha256": None,
    }
    try:
        source = inspect.getsourcefile(function)
    except (OSError, TypeError):
        source = None
    if source is not None:
        path = Path(source).expanduser().resolve()
        identity["source_file"] = str(path)
        if path.is_file():
            identity["source_sha256"] = _sha256_file(path)
    return identity


def _require_receiver_facade_type(
    target: nn.Module,
    *,
    target_loader: str,
) -> dict[str, object]:
    """Bind formal NPU execution to the facade class exported by its loader.

    The receiver-owned loader may differ from the template in its factory, so
    this is a type/source binding plus behavioral gate, not an independent
    proof that the receiver preserved every template statement.
    """

    loader_module, _function_name = target_loader.rsplit(":", 1)
    module = importlib.import_module(loader_module)
    facade_class = getattr(module, "InternalTargetFacade", None)
    if not isinstance(facade_class, type) or not issubclass(facade_class, nn.Module):
        raise TypeError(
            "formal NPU target loader module must export an nn.Module "
            "InternalTargetFacade class"
        )
    target_type = type(target)
    if target_type is not facade_class:
        raise TypeError(
            "formal NPU target loader must return its module-local "
            "InternalTargetFacade; got "
            f"{target_type.__module__}.{target_type.__name__}"
        )
    facade_identity = _callable_source_identity(
        facade_class,
        specification=f"{loader_module}:InternalTargetFacade",
    )
    if not isinstance(facade_identity.get("source_sha256"), str):
        raise RuntimeError("formal NPU facade class lacks source-file identity")
    return facade_identity


def _bind_formal_hiai_source(
    target: nn.Module,
    *,
    target_loader: str,
    hiai_source: str | None,
) -> dict[str, object]:
    """Bind receiver-declared provenance to the actual package-local source."""

    if hiai_source is None:
        raise ValueError("formal NPU V1 requires --hiai-source")
    loader_module_name, _function_name = target_loader.rsplit(":", 1)
    loader_module = importlib.import_module(loader_module_name)
    loader_file = getattr(loader_module, "__file__", None)
    if not isinstance(loader_file, str):
        raise RuntimeError("formal NPU target loader module lacks __file__")
    package_dir = Path(loader_file).resolve().parent
    raw_source = Path(hiai_source).expanduser()
    if raw_source.is_symlink():
        raise RuntimeError("formal HIAI source must not be a symlink")
    source = raw_source.resolve()
    expected = (package_dir.parent / "modeling_qwen3_5_hiai_nd.py").resolve()
    if source != expected or not source.is_file():
        raise RuntimeError(
            "formal HIAI source must match the selected NPU layout's "
            "modeling_qwen3_5_hiai_nd.py"
        )
    verification = verify_direct_source_file(source)
    actual_sha256 = _sha256_file(source)
    declared_sha256 = getattr(target, "dflash_feature_source_sha256", None)
    if (
        verification.get("status") != "PASS_DIRECT_SOURCE_CONTRACT"
        or verification.get("contract_id") != _HIAI_FEATURE_CONTRACT_ID
        or verification.get("source_sha256") != actual_sha256
        or declared_sha256 != actual_sha256
    ):
        raise RuntimeError(
            "formal HIAI source hash/contract does not match target provenance"
        )
    hiai_package = loader_module_name.rsplit(".", 2)[0]
    hiai_module_name = hiai_package + ".modeling_qwen3_5_hiai_nd"
    hiai_module = importlib.import_module(hiai_module_name)
    expected_target_class = getattr(hiai_module, "Qwen3_5ForCausalLM", None)
    target_controller = getattr(target, "target", None)
    raw_target = getattr(
        target_controller,
        "dflash_execution_model",
        target_controller,
    )
    if not isinstance(expected_target_class, type) or type(raw_target) is not expected_target_class:
        raise RuntimeError(
            "formal facade must execute the exact package-local "
            "Qwen3_5ForCausalLM class exported by modeling_qwen3_5_hiai_nd, "
            "directly or through internal_dflash_bridge"
        )
    isolation_audit = getattr(target, "dflash_full_prefix_isolation_audit", None)
    raw_target_identity = (
        isolation_audit.get("raw_target_identity")
        if isinstance(isolation_audit, Mapping)
        else None
    )
    if not isinstance(raw_target_identity, Mapping):
        raise RuntimeError("formal facade lacks raw target execution-model identity")
    raw_source_file = raw_target_identity.get("source_file")
    if (
        not isinstance(raw_source_file, str)
        or Path(raw_source_file).resolve() != source
        or raw_target_identity.get("source_sha256") != actual_sha256
    ):
        raise RuntimeError(
            "formal facade raw target type is not defined by the locked "
            "package-local HIAI source"
        )
    return {
        "status": "PASS_ACTUAL_PACKAGE_SOURCE",
        "path": str(source),
        "source_sha256": actual_sha256,
        "contract_id": verification["contract_id"],
        "feature_source": verification["feature_source"],
        "capture_point": verification["capture_point"],
        "source_integration": "direct",
        "source_modified_by_runtime": False,
        "raw_target_fqcn": raw_target_identity.get("fqcn"),
        "raw_target_class_object_bound": True,
    }


def _require_formal_target_loader_spec(
    target_loader: str,
) -> Path:
    """Require the receiver loader to be the adapter's package-local sibling."""

    expected_spec = f"{__package__}.internal_target_loader:load_target"
    if target_loader != expected_spec:
        raise ValueError(
            "formal NPU V1 requires the package-local target loader "
            f"{expected_spec!r}; got {target_loader!r}"
        )
    loader_path = Path(__file__).resolve().with_name("internal_target_loader.py")
    if loader_path.is_symlink() or not loader_path.is_file():
        raise RuntimeError(
            "formal NPU package lacks a real sibling internal_target_loader.py"
        )
    return loader_path.resolve()


def _load_target(
    target_dir: str,
    *,
    target_loader: str | None,
    hiai_source: str | None = None,
    device: str,
    dtype: torch.dtype,
    allow_download: bool,
    trust_remote_code: bool,
) -> nn.Module:
    device_type = str(device).split(":", 1)[0].lower()
    if device_type == "npu" and target_loader is None:
        raise ValueError(
            "formal NPU V1 requires --target-loader returning the receiver "
            "state-isolated HIAI facade; the package-default HF target is a "
            "CPU/framework golden route"
        )
    expected_loader_path: Path | None = None
    if device_type == "npu":
        assert target_loader is not None
        expected_loader_path = _require_formal_target_loader_spec(target_loader)
    _prepare_device_backend(device)
    if target_loader is not None:
        loader_function = _load_callable(target_loader)
        loader_identity = _callable_source_identity(
            loader_function,
            specification=target_loader,
        )
        if expected_loader_path is not None:
            loader_source = loader_identity.get("source_file")
            if (
                not isinstance(loader_source, str)
                or Path(loader_source).resolve() != expected_loader_path
            ):
                raise RuntimeError(
                    "formal NPU target loader resolved outside the adapter's "
                    "package-local internal_target_loader.py"
                )
        target = loader_function(
            target_dir,
            device=device,
            dtype=dtype,
        )
    else:
        # The packaged sibling contains the opt-in feature collector; the
        # ordinary Transformers class does not expose dflash_features.
        try:
            from .modeling_qwen3_5_dflash import (
                Qwen3_5ForConditionalGeneration,
            )
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "could not import the packaged feature-enabled Qwen3.5 target; "
                "keep the complete models/dflash_v1 package or pass "
                "--target-loader MODULE:FUNCTION"
            ) from error

        target = Qwen3_5ForConditionalGeneration.from_pretrained(
            target_dir,
            dtype=dtype,
            local_files_only=not allow_download,
            trust_remote_code=trust_remote_code,
        )
        target = target.to(device)
        loader_identity = {
            "specification": "package_default",
            "module": __name__,
            "qualname": "Qwen3_5ForConditionalGeneration.from_pretrained",
            "source_file": str(Path(__file__).resolve()),
            "source_sha256": _sha256_file(Path(__file__).resolve()),
        }
    if not isinstance(target, nn.Module):
        raise TypeError("target loader must return a torch.nn.Module")
    if device_type == "npu":
        assert target_loader is not None
        facade_identity = _require_receiver_facade_type(
            target,
            target_loader=target_loader,
        )
        loader_identity["facade_source_file"] = facade_identity["source_file"]
        loader_identity["facade_source_sha256"] = facade_identity["source_sha256"]
        hiai_source_identity = _bind_formal_hiai_source(
            target,
            target_loader=target_loader,
            hiai_source=hiai_source,
        )
        setattr(target, "_dflash_hiai_source_identity", hiai_source_identity)
    setattr(target, "_dflash_target_loader_identity", loader_identity)
    return target.eval()


def _target_integration_audit(
    target: nn.Module,
    *,
    device: str | torch.device,
    target_loader: str | None,
    require_completed_forward: bool,
) -> dict[str, object]:
    """Validate/report receiver isolation and HIAI feature-source provenance.

    CPU package-default execution remains a framework golden.  A formal NPU
    run must use the receiver facade, prepare every full-prefix call, and
    declare the directly integrated HIAI feature route.  Receiver declarations are
    recorded as declarations; the later zero-impact and token gates remain the
    behavioral evidence.
    """

    device_type = str(device).split(":", 1)[0].lower()
    formal_npu = device_type == "npu"
    raw_isolation = getattr(target, "dflash_full_prefix_isolation_audit", None)
    if raw_isolation is None:
        if formal_npu:
            raise RuntimeError(
                "formal NPU target lacks dflash_full_prefix_isolation_audit; "
                "the custom loader must return InternalTargetFacade"
            )
        isolation: dict[str, object] = {
            "status": "UNVERIFIED_FRAMEWORK_GOLDEN",
            "mode": "not_declared",
            "formal_npu": False,
            "all_calls_prepared": False,
            "prepare_calls": 0,
            "target_forward_calls": 0,
        }
    else:
        if not isinstance(raw_isolation, Mapping):
            raise TypeError("dflash_full_prefix_isolation_audit must be a mapping")
        isolation = dict(raw_isolation)
        mode = isolation.get("mode")
        if not isinstance(mode, str):
            raise TypeError("target isolation audit mode must be a string")
        for field in (
            "formal_npu",
            "all_calls_prepared",
            "prepare_forward_serialized",
        ):
            if not isinstance(isolation.get(field), bool):
                raise TypeError(f"target isolation audit {field} must be bool")
        for field in (
            "prepare_calls",
            "prepare_successes",
            "prepare_failures",
            "target_forward_calls",
            "target_forward_completions",
            "target_forward_failures",
            "output_validation_failures",
        ):
            value = isolation.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TypeError(
                    f"target isolation audit {field} must be a non-negative int"
                )
        isolation["status"] = "PASS_DECLARED_AND_INSTRUMENTED"

    loader_identity = getattr(target, "_dflash_target_loader_identity", None)
    if loader_identity is None:
        loader_identity = {
            "specification": target_loader or "package_default",
            "module": None,
            "qualname": None,
            "source_file": None,
            "source_sha256": None,
        }
    elif not isinstance(loader_identity, Mapping):
        raise TypeError("target loader identity must be a mapping")
    else:
        loader_identity = dict(loader_identity)

    feature_source = getattr(target, "dflash_feature_source", None)
    capture_point = getattr(target, "dflash_feature_capture_point", None)
    feature_contract_id = getattr(target, "dflash_feature_contract_id", None)
    source_sha256 = getattr(target, "dflash_feature_source_sha256", None)
    actual_source_identity = getattr(target, "_dflash_hiai_source_identity", None)
    if source_sha256 is not None:
        if not isinstance(source_sha256, str) or len(source_sha256) != 64:
            raise TypeError("HIAI feature source SHA-256 must be a 64-character string")
        try:
            int(source_sha256, 16)
        except ValueError as error:
            raise TypeError("HIAI feature source SHA-256 must be hexadecimal") from error
        source_sha256 = source_sha256.lower()

    feature = {
        "source": feature_source,
        "capture_point": capture_point,
        "contract_id": feature_contract_id,
        "source_sha256": source_sha256,
        "integration_mode": "direct_source",
        "evidence_authority": "receiver_declared",
        "actual_package_source": (
            dict(actual_source_identity)
            if isinstance(actual_source_identity, Mapping)
            else None
        ),
        "status": (
            "PASS_DECLARED_DIRECT_SOURCE"
            if feature_source == _HIAI_FEATURE_SOURCE
            and capture_point == _HIAI_CAPTURE_POINT
            and feature_contract_id == _HIAI_FEATURE_CONTRACT_ID
            and source_sha256 is not None
            else "UNVERIFIED_FRAMEWORK_GOLDEN"
        ),
    }

    if formal_npu:
        if target_loader is None:
            raise RuntimeError("formal NPU target integration requires a custom loader")
        if isolation.get("formal_npu") is not True:
            raise RuntimeError("target isolation audit is not marked formal_npu")
        if isolation.get("mode") not in _FORMAL_ISOLATION_MODES:
            raise RuntimeError(
                "formal NPU target isolation mode must be receiver_reset_hook, "
                "or fresh_instance; the HIAI route has known in-place KV/GDN state"
            )
        if isolation.get("facade_contract_id") != _FACADE_CONTRACT_ID:
            raise RuntimeError(
                "formal NPU target does not declare the required facade contract"
            )
        raw_target_identity = isolation.get("raw_target_identity")
        if not isinstance(raw_target_identity, Mapping) or not isinstance(
            raw_target_identity.get("source_sha256"), str
        ):
            raise RuntimeError(
                "formal NPU facade lacks raw target type/artifact identity"
            )
        hook_identity = isolation.get("isolation_hook_identity")
        if not isinstance(hook_identity, Mapping) or not isinstance(
            hook_identity.get("source_sha256"), str
        ):
            raise RuntimeError(
                "formal NPU target isolation hook lacks source-file identity"
            )
        if not isinstance(loader_identity.get("source_sha256"), str):
            raise RuntimeError("formal NPU target loader lacks source-file identity")
        if isolation.get("prepare_forward_serialized") is not True:
            raise RuntimeError("target prepare/forward sequence is not serialized")
        if isolation.get("full_prefix_execution_mode") != "fresh_prefill":
            raise RuntimeError(
                "formal NPU target must prepare every complete prefix as fresh prefill"
            )
        declared_chunks = isolation.get("declared_chunk_modes")
        if not isinstance(declared_chunks, Mapping) or dict(declared_chunks) != {
            "prefill_chunk_size": 64,
            "decode_chunk_size": 1,
        }:
            raise RuntimeError(
                "formal NPU target must declare prefill chunk_size=64 and "
                "decode chunk_size=1"
            )
        if isolation.get("prepare_failures") != 0:
            raise RuntimeError("target full-prefix isolation reported prepare failures")
        if feature["status"] != "PASS_DECLARED_DIRECT_SOURCE":
            raise RuntimeError(
                "formal NPU target must declare the directly integrated "
                "modeling_qwen3_5_hiai_nd.py feature contract and source hash"
            )
        if not isinstance(actual_source_identity, Mapping) or (
            actual_source_identity.get("status") != "PASS_ACTUAL_PACKAGE_SOURCE"
            or actual_source_identity.get("source_sha256") != source_sha256
            or actual_source_identity.get("contract_id")
            != _HIAI_FEATURE_CONTRACT_ID
        ):
            raise RuntimeError(
                "formal NPU target provenance is not bound to the actual "
                "package-local HIAI source"
            )
        if require_completed_forward:
            forward_calls = isolation["target_forward_calls"]
            if forward_calls <= 0:
                raise RuntimeError("formal NPU validation executed no target forwards")
            if isolation.get("all_calls_prepared") is not True:
                raise RuntimeError("not every target forward had a successful prepare hook")
            if isolation.get("prepare_calls") != forward_calls:
                raise RuntimeError("target prepare and forward call counts differ")
            if isolation.get("target_forward_completions") != forward_calls:
                raise RuntimeError("one or more target forwards did not complete")
            if isolation.get("target_forward_failures") != 0:
                raise RuntimeError("target forward failures were recorded")
            if isolation.get("output_validation_failures") != 0:
                raise RuntimeError("target output validation failures were recorded")

    return {
        "loader": target_loader or "package_default",
        "loader_identity": loader_identity,
        "route": "receiver_hiai" if target_loader is not None else "hf_package_default",
        "isolation": isolation,
        "feature_capture": feature,
    }


def _target_forward_reconciliation(
    initial: Mapping[str, object],
    final: Mapping[str, object],
    *,
    expected_validation_calls: int,
    formal_npu: bool,
) -> dict[str, object]:
    if isinstance(expected_validation_calls, bool) or expected_validation_calls <= 0:
        raise ValueError("expected_validation_calls must be positive")
    initial_isolation = initial.get("isolation")
    final_isolation = final.get("isolation")
    if not isinstance(initial_isolation, Mapping) or not isinstance(
        final_isolation, Mapping
    ):
        raise TypeError("target integration isolation payloads must be mappings")
    if (
        initial_isolation.get("status") == "UNVERIFIED_FRAMEWORK_GOLDEN"
        or final_isolation.get("status") == "UNVERIFIED_FRAMEWORK_GOLDEN"
    ):
        return {
            "status": "UNINSTRUMENTED_FRAMEWORK_GOLDEN",
            "expected_validation_calls": expected_validation_calls,
            "target_forward_delta": None,
            "prepare_delta": None,
            "matches": None,
        }
    initial_forwards = initial_isolation.get("target_forward_calls")
    final_forwards = final_isolation.get("target_forward_calls")
    initial_prepares = initial_isolation.get("prepare_calls")
    final_prepares = final_isolation.get("prepare_calls")
    if not all(
        isinstance(value, int) and not isinstance(value, bool)
        for value in (initial_forwards, final_forwards, initial_prepares, final_prepares)
    ):
        raise TypeError("target integration counters must be integers")
    forward_delta = final_forwards - initial_forwards
    prepare_delta = final_prepares - initial_prepares
    matches = forward_delta == prepare_delta == expected_validation_calls
    if formal_npu and not matches:
        raise RuntimeError(
            "formal NPU target facade call counts do not match the V1 scheduler: "
            f"expected={expected_validation_calls}, forwards={forward_delta}, "
            f"prepares={prepare_delta}"
        )
    return {
        "status": "PASS" if matches else "FAIL",
        "expected_validation_calls": expected_validation_calls,
        "target_forward_delta": forward_delta,
        "prepare_delta": prepare_delta,
        "matches": matches,
    }


def _dflash_execution_gate(
    result: Qwen35GoldenValidation,
    *,
    formal_npu: bool,
    require_accelerator_round: bool = False,
) -> dict[str, object]:
    draft_calls = int(result.dflash_adapter_stats.draft_calls)
    target_feature_calls = int(result.dflash_adapter_stats.target_feature_calls)
    target_verify_calls = int(result.dflash.stats.target_verify_calls)
    executed = (
        draft_calls > 0
        and target_feature_calls > 0
        and target_verify_calls > 0
    )
    if (formal_npu or require_accelerator_round) and not executed:
        raise RuntimeError(
            "accelerator validation was inconclusive: no complete DFlash "
            "draft/feature/target-verify round executed (the bootstrap may "
            "have reached EOS or max_new_tokens was too small)"
        )
    return {
        "status": "PASS" if executed else "INCONCLUSIVE_FRAMEWORK_GOLDEN",
        "draft_round_executed": executed,
        "draft_calls": draft_calls,
        "target_feature_calls": target_feature_calls,
        "target_verify_calls": target_verify_calls,
    }


def _validate_ops_backend_request(device: str, ops_backend: str | None) -> None:
    if str(device).split(":", 1)[0].lower() == "npu" and ops_backend is not None:
        raise ValueError(
            "formal NPU execution forbids an external --ops-backend; use the "
            "package-local dflash_ascend310p_ops backend"
        )


def _select_draft_ops(
    *,
    device: str,
    ops_backend: str | None,
    allow_op_fallback: bool,
) -> tuple[object, str]:
    """Select a fail-closed draft primitive route for the requested device."""

    _validate_ops_backend_request(device, ops_backend)
    if ops_backend:
        return (
            ModuleDFlashOps.from_name(
                ops_backend,
                strict=not allow_op_fallback,
            ),
            ops_backend,
        )
    device_type = str(device).split(":", 1)[0].lower()
    if device_type in {"cpu", "cuda"}:
        return TorchDFlashOps(), "torch" if device_type == "cpu" else "torch_cuda"
    if device_type == "npu":
        # This package-local backend deliberately decomposes attention into
        # matmul/mask/FP32-softmax/matmul, stays on the NPU, and has no hidden
        # CPU or SDPA fallback.  This exact package-local module is part of the
        # formal embedded-runtime identity; external backends remain a CPU-only
        # development option until they have their own oracle/provenance gate.
        module_name = f"{__package__}.dflash_ascend310p_ops"
        return ModuleDFlashOps.from_name(module_name, strict=True), module_name
    raise ValueError(
        f"device {device!r} requires an explicit --ops-backend; "
        "automatic selection is defined only for cpu, cuda, and npu"
    )


def _emit_progress(enabled: bool, event: str, fields: Mapping[str, object]) -> None:
    if not enabled:
        return
    payload = {"event": event, **dict(fields)}
    print(
        "[dflash-v1] " + json.dumps(payload, ensure_ascii=False, sort_keys=True),
        file=sys.stderr,
        flush=True,
    )


def _prepare_device_backend(device: str | torch.device) -> None:
    """Load the explicit device extension before constructing device tensors.

    ``torch_npu`` registers the ``npu`` device type at import time.  The V1/V2
    CLIs therefore cannot defer that import until the final runtime report.
    CUDA is already registered by PyTorch, but availability and device
    selection are checked before loading multi-gigabyte checkpoints.  CPU is
    left unchanged.
    """

    device_text = str(device)
    device_type = device_text.split(":", 1)[0].lower()
    if device_type == "cpu":
        return
    if device_type == "cuda":
        if getattr(torch.version, "cuda", None) is None:
            raise RuntimeError("--device cuda requires a CUDA-enabled PyTorch build")
        is_available = getattr(torch.cuda, "is_available", None)
        if not callable(is_available) or not bool(is_available()):
            raise RuntimeError("--device cuda requested but no CUDA device is available")
        set_device = getattr(torch.cuda, "set_device", None)
        device_index = torch.device(device_text).index
        if device_index is not None and not callable(set_device):
            raise RuntimeError("torch.cuda.set_device is unavailable")
        # ``--device cuda`` means the process's current CUDA device.  Calling
        # set_device with an index-less device is rejected by some PyTorch
        # versions, so only an explicit ``cuda:N`` changes the current card.
        if device_index is not None:
            set_device(device_text)
        return
    if device_type != "npu":
        return
    try:
        importlib.import_module("torch_npu")
    except ImportError as error:
        raise RuntimeError(
            "--device npu requires an importable torch_npu package"
        ) from error
    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("torch_npu imported but torch.npu is unavailable")
    is_available = getattr(npu, "is_available", None)
    if not callable(is_available):
        raise RuntimeError("torch.npu.is_available is unavailable")
    if not bool(is_available()):
        raise RuntimeError("torch_npu is installed but no NPU device is available")
    set_device = getattr(npu, "set_device", None)
    if not callable(set_device):
        raise RuntimeError("torch.npu.set_device is unavailable")
    set_device(device_text)


def _prompt_ids(raw: str | None, json_path: str | None) -> list[int]:
    if (raw is None) == (json_path is None):
        raise ValueError("provide exactly one of --prompt-ids or --prompt-json")
    if json_path is not None:
        payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
        if isinstance(payload, Mapping):
            payload = payload.get("input_ids")
        if (
            isinstance(payload, list)
            and len(payload) == 1
            and isinstance(payload[0], list)
        ):
            payload = payload[0]
        if not isinstance(payload, list):
            raise TypeError("prompt JSON must be a token list or {'input_ids': list}")
        values = payload
    else:
        assert raw is not None
        values = [piece.strip() for piece in raw.split(",") if piece.strip()]
    result: list[int] = []
    for value in values:
        if isinstance(value, bool):
            raise TypeError("prompt token IDs must be integers, not bool")
        try:
            token = int(value) if isinstance(value, str) else int(operator.index(value))
            result.append(token)
        except (TypeError, ValueError) as error:
            raise TypeError("prompt token IDs must be integers") from error
    if not result:
        raise ValueError("prompt must contain at least one token")
    return result


def _request_payload(
    args: argparse.Namespace,
    *,
    effective_max_draft_tokens: int,
) -> dict[str, object]:
    """Return non-secret controls needed to reproduce one CLI validation."""

    formal_npu = str(getattr(args, "device", "cpu")).split(":", 1)[0].lower() == "npu"
    return {
        "max_new_tokens": int(args.max_new_tokens),
        "requested_max_draft_tokens": args.max_draft_tokens,
        "effective_max_draft_tokens": int(effective_max_draft_tokens),
        "eos_token_ids": list(args.eos_token_id),
        "formal_locked_eos_token_id": (
            _FORMAL_EOS_TOKEN_ID if formal_npu else None
        ),
        "prompt_source": "inline_token_ids" if args.prompt_ids is not None else "json_file",
        "target_loader": args.target_loader or "package_default",
        "npu_layout": getattr(args, "npu_layout", None),
        "target_factory": getattr(args, "target_factory", None),
        "reset_hook": getattr(args, "reset_hook", None),
        "hiai_source": (
            None
            if getattr(args, "hiai_source", None) is None
            else str(Path(args.hiai_source).expanduser().resolve())
        ),
        "allow_download": bool(args.allow_download),
        "trust_remote_code": bool(args.trust_remote_code),
        "progress_enabled": bool(args.progress),
        "draft_checkpoint_sha256_verified": True,
    }


def _decode_payload(result: ReplayDecodeResult) -> dict[str, Any]:
    payload = asdict(result)
    payload["stats"]["acceptance_rate"] = result.stats.acceptance_rate
    return payload


def _runtime_identity(device: torch.device) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "torch_version": torch.__version__,
        "torch_git_version": getattr(torch.version, "git_version", None),
        "device": str(device),
        "device_type": device.type,
        "device_index": device.index,
        "platform": platform.platform(),
    }
    try:
        import transformers

        identity["transformers_version"] = transformers.__version__
    except ImportError:
        identity["transformers_version"] = None
    if device.type == "cpu":
        identity["device_name"] = platform.processor() or "cpu"
        return identity

    backend = getattr(torch, device.type, None)
    current_device = getattr(backend, "current_device", None)
    device_index = device.index
    if device_index is None and callable(current_device):
        device_index = int(current_device())
        identity["device_index"] = device_index
    get_device_name = getattr(backend, "get_device_name", None)
    if callable(get_device_name):
        identity["device_name"] = str(get_device_name(device_index))
    else:
        identity["device_name"] = f"{device.type}:{device_index}"
    if device.type == "cuda":
        identity["cuda_version"] = getattr(torch.version, "cuda", None)
    if device.type == "npu":
        try:
            import torch_npu  # type: ignore[import-not-found]

            identity["torch_npu_version"] = getattr(torch_npu, "__version__", None)
        except ImportError:
            identity["torch_npu_version"] = None
    return identity


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Qwen3.5 DFlash V1 against ordinary full-prefix greedy "
            "generation without scheduler-owned target or draft cache"
        )
    )
    parser.add_argument("--target-dir", required=True)
    parser.add_argument("--draft-dir", required=True)
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt-ids", help="comma-separated token IDs")
    prompt.add_argument("--prompt-json", help="JSON token list or input_ids object")
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument(
        "--max-draft-tokens",
        type=int,
        help="proposal count; default is checkpoint block_size minus one anchor",
    )
    parser.add_argument("--eos-token-id", type=int, action="append", default=[])
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="float16",
        help=(
            "CPU supports all choices; CUDA BF16 requires hardware support; "
            "the approved NPU experiment requires float16"
        ),
    )
    parser.add_argument(
        "--target-loader",
        help=(
            "MODULE:FUNCTION loader(target_dir, device=..., dtype=...); required "
            "on NPU, optional on CPU/CUDA; the framework default loads the sibling "
            "modeling_qwen3_5_dflash module from the current package namespace"
        ),
    )
    parser.add_argument(
        "--npu-layout",
        choices=(_NPU_LAYOUT_EMBEDDED,),
        default=_NPU_LAYOUT_EMBEDDED,
        help=(
            "embedded keeps modeling_qwen3_5_hiai_nd.py in the parent models "
            "package and DFlash below models.dflash_v1"
        ),
    )
    parser.add_argument(
        "--target-factory",
        help=(
            "embedded NPU only: existing inference MODULE:FUNCTION returning "
            "the raw HIAI Qwen3.5 target"
        ),
    )
    parser.add_argument(
        "--reset-hook",
        help=(
            "embedded NPU only: MODULE:FUNCTION resetting KV/GDN/request state "
            "before each complete-prefix target call; optional when the target "
            "already exposes prepare_dflash_full_prefix_call"
        ),
    )
    parser.add_argument(
        "--hiai-source",
        help=(
            "receiver package-local modeling_qwen3_5_hiai_nd.py; required on "
            "NPU and hash-bound to target provenance"
        ),
    )
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="allow the default Transformers loader to access remote files",
    )
    parser.add_argument(
        "--ops-backend",
        help=(
            "module exporting the six dflash_ops functions; default is the "
            "PyTorch oracle on CPU/CUDA and the package-local decomposed "
            "310P backend on NPU; external modules are non-formal "
            "development routes outside the formal NPU flow"
        ),
    )
    parser.add_argument(
        "--allow-op-fallback",
        action="store_true",
        help="simulation only: fall back to PyTorch when an op is missing",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="emit stage progress as JSON lines on stderr (default: enabled)",
    )
    parser.add_argument("--report", help="optional JSON report path")
    return parser


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _validate_report_destination(
    args: argparse.Namespace,
    *,
    package_dir: Path,
    formal_npu: bool,
) -> None:
    """Keep the output report outside every validated input/artifact tree."""

    raw_report = getattr(args, "report", None)
    if raw_report is None:
        return
    destination = Path(raw_report).expanduser()
    if destination.is_symlink():
        raise ValueError("--report must not be a symlink")
    resolved = destination.resolve()
    protected_files: set[Path] = set()
    if formal_npu:
        protected_files.update(
            {
                Path(args.hiai_source).expanduser().resolve(),
                (package_dir / "internal_target_loader.py").resolve(),
                (package_dir.parent / "internal_dflash_bridge.py").resolve(),
            }
        )
    prompt_json = getattr(args, "prompt_json", None)
    if prompt_json is not None:
        protected_files.add(Path(prompt_json).expanduser().resolve())
    if resolved in protected_files:
        raise ValueError("--report overlaps a validated input file")
    protected_roots = {
        package_dir.resolve(),
        Path(args.target_dir).expanduser().resolve(),
        Path(args.draft_dir).expanduser().resolve(),
    }
    if any(_is_within(resolved, root) for root in protected_roots):
        raise ValueError(
            "--report must be in a separate run directory, outside "
            "the runtime package and target/draft model directories"
        )


def _configure_embedded_npu_inputs(args: argparse.Namespace) -> None:
    """Derive the internal layout so daily NPU runs need no overlay arguments."""

    if str(args.device).split(":", 1)[0].lower() != "npu":
        return
    package_dir = Path(__file__).resolve().parent
    expected_loader = f"{__package__}.internal_target_loader:load_target"
    expected_source = package_dir.parent / "modeling_qwen3_5_hiai_nd.py"
    if args.target_loader is None:
        args.target_loader = expected_loader
    if args.hiai_source is None:
        args.hiai_source = str(expected_source)
    if args.target_factory is None:
        raise ValueError(
            "embedded NPU layout requires --target-factory MODULE:FUNCTION"
        )
    os.environ[_TARGET_FACTORY_ENV] = args.target_factory
    if args.reset_hook is not None:
        os.environ[_RESET_HOOK_ENV] = args.reset_hook
    else:
        # Do not accidentally reuse a reset function left by an earlier run in
        # the same Python process.  Omitting the option means the raw target
        # itself must expose prepare_dflash_full_prefix_call.
        os.environ.pop(_RESET_HOOK_ENV, None)


def _validate_embedded_runtime(
    *,
    package_dir: Path,
    hiai_source: Path,
    loader_path: Path,
) -> dict[str, object]:
    """Validate the colocated source tree without a generated overlay report."""

    import transformers

    if transformers.__version__ != "5.14.1":
        raise RuntimeError(
            "embedded DFlash framework requires transformers==5.14.1; got "
            f"{transformers.__version__}"
        )
    verification = verify_direct_source_file(hiai_source)
    if (
        verification.get("status") != "PASS_DIRECT_SOURCE_CONTRACT"
        or verification.get("contract_id") != _HIAI_FEATURE_CONTRACT_ID
    ):
        raise RuntimeError("embedded HIAI source does not satisfy the feature contract")
    bridge_path = package_dir.parent / "internal_dflash_bridge.py"
    if bridge_path.is_symlink() or not bridge_path.is_file():
        raise RuntimeError(
            "embedded NPU layout requires models/internal_dflash_bridge.py"
        )
    runtime_hashes: dict[str, str] = {}
    for name in sorted(_EMBEDDED_RUNTIME_FILES):
        path = package_dir / name
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"embedded DFlash runtime file is missing: {name}")
        runtime_hashes[name] = _sha256_file(path)
    return {
        "status": "PASS_EMBEDDED_RUNTIME_PREFLIGHT",
        "layout": _NPU_LAYOUT_EMBEDDED,
        "package": __package__,
        "package_dir": str(package_dir),
        "transformers_version": transformers.__version__,
        "runtime_file_sha256": runtime_hashes,
        "hiai_source": str(hiai_source),
        "hiai_source_sha256": _sha256_file(hiai_source),
        "feature_contract_id": verification["contract_id"],
        "source_integration": "direct",
        "source_modified_by_runtime": False,
        "receiver_loader_sha256": _sha256_file(loader_path),
        "internal_bridge_sha256": _sha256_file(bridge_path),
    }


def _validate_formal_cli_inputs(args: argparse.Namespace) -> dict[str, object] | None:
    """Fail before the 1.27 GB draft hash when formal receiver inputs are absent."""

    if str(args.device).split(":", 1)[0].lower() != "npu":
        return None
    if tuple(args.eos_token_id) != (_FORMAL_EOS_TOKEN_ID,):
        raise ValueError(
            "formal NPU V1 requires exactly --eos-token-id 248044"
        )
    if args.target_loader is None:
        raise ValueError("formal NPU V1 requires --target-loader")
    _require_formal_target_loader_spec(args.target_loader)
    if args.hiai_source is None:
        raise ValueError("formal NPU V1 requires --hiai-source")
    package_dir = Path(__file__).resolve().parent
    raw_hiai_source = Path(args.hiai_source).expanduser()
    expected_hiai_source = package_dir.parent / "modeling_qwen3_5_hiai_nd.py"
    if (
        raw_hiai_source.is_symlink()
        or raw_hiai_source.resolve() != expected_hiai_source.resolve()
        or not raw_hiai_source.is_file()
    ):
        raise ValueError(
            "--hiai-source does not match the selected NPU layout's "
            "modeling_qwen3_5_hiai_nd.py"
        )
    _validate_report_destination(args, package_dir=package_dir, formal_npu=True)
    if os.environ.get("PYTHONPYCACHEPREFIX") or sys.pycache_prefix is not None:
        raise RuntimeError("formal NPU V1 forbids PYTHONPYCACHEPREFIX")
    bytecode = list(package_dir.glob("*.pyc"))
    cache_dir = package_dir / "__pycache__"
    if cache_dir.exists():
        bytecode.extend(cache_dir.glob("*.pyc"))
    if bytecode:
        raise RuntimeError("formal NPU V1 package contains precompiled bytecode")
    return _validate_embedded_runtime(
        package_dir=package_dir,
        hiai_source=expected_hiai_source,
        loader_path=package_dir / "internal_target_loader.py",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    device_type = str(args.device).split(":", 1)[0].lower()
    formal_npu = device_type == "npu"
    _configure_embedded_npu_inputs(args)
    _validate_ops_backend_request(args.device, args.ops_backend)
    if not formal_npu:
        _validate_report_destination(
            args,
            package_dir=Path(__file__).resolve().parent,
            formal_npu=False,
        )
    initial_runtime_preflight = _validate_formal_cli_inputs(args)
    if args.allow_op_fallback and not str(args.device).startswith("cpu"):
        raise ValueError("operator fallback is allowed only for CPU simulation")
    if device_type in {"npu", "cuda"} and args.max_new_tokens < 2:
        raise ValueError(
            "accelerator DFlash validation requires --max-new-tokens >= 2 so "
            "at least one post-bootstrap draft round can execute"
        )
    dtype = _dtype(args.dtype)
    # Select the requested accelerator before querying device-specific dtype
    # capabilities (notably BF16 on heterogeneous multi-GPU systems).
    _prepare_device_backend(args.device)
    _validate_experiment_dtype(args.device, dtype)
    _emit_progress(
        args.progress,
        "draft_checkpoint_audit_begin",
        {"verify_model_sha256": True},
    )
    draft_checkpoint = require_official_dflash_checkpoint(
        args.draft_dir,
        verify_model_hash=True,
    )
    _emit_progress(
        args.progress,
        "draft_checkpoint_audit_end",
        {
            "config_sha256": draft_checkpoint["config_sha256"],
            "model_sha256": draft_checkpoint["model_sha256"],
            "model_bytes": draft_checkpoint["model_bytes"],
        },
    )
    _emit_progress(
        args.progress,
        "target_load_begin",
        {"device": args.device, "dtype": args.dtype},
    )
    target = _load_target(
        args.target_dir,
        target_loader=args.target_loader,
        hiai_source=args.hiai_source,
        device=args.device,
        dtype=dtype,
        allow_download=args.allow_download,
        trust_remote_code=args.trust_remote_code,
    )
    initial_target_integration = _target_integration_audit(
        target,
        device=args.device,
        target_loader=args.target_loader,
        require_completed_forward=False,
    )
    _emit_progress(
        args.progress,
        "target_load_end",
        {
            "integration_route": initial_target_integration["route"],
            "isolation_mode": initial_target_integration["isolation"]["mode"],
            "feature_source": initial_target_integration["feature_capture"]["source"],
        },
    )
    ops, backend = _select_draft_ops(
        device=args.device,
        ops_backend=args.ops_backend,
        allow_op_fallback=args.allow_op_fallback,
    )
    _emit_progress(args.progress, "draft_ops_ready", {"backend": backend})
    _emit_progress(args.progress, "draft_load_begin", {})
    draft = DFlashDraftModel.from_pretrained(
        args.draft_dir,
        ops=ops,
        device=args.device,
        dtype=dtype,
    )
    _emit_progress(args.progress, "draft_load_end", {})
    adapter = Qwen35DFlashFullPrefixAdapter(target, draft)
    prompt_ids = _prompt_ids(args.prompt_ids, args.prompt_json)
    _emit_progress(
        args.progress,
        "validation_begin",
        {
            "prompt_tokens": len(prompt_ids),
            "max_new_tokens": args.max_new_tokens,
            "max_draft_tokens": args.max_draft_tokens,
        },
    )
    result = validate_qwen35_dflash_strict_greedy(
        adapter,
        prompt_ids,
        max_new_tokens=args.max_new_tokens,
        max_draft_tokens=args.max_draft_tokens,
        eos_token_ids=args.eos_token_id,
        progress_callback=lambda event, fields: _emit_progress(
            args.progress,
            event,
            fields,
        ),
    )
    _emit_progress(args.progress, "token_validation_end", {"status": "PASS"})
    final_target_integration = _target_integration_audit(
        target,
        device=args.device,
        target_loader=args.target_loader,
        require_completed_forward=True,
    )
    dflash_execution = _dflash_execution_gate(
        result,
        formal_npu=formal_npu,
        require_accelerator_round=device_type == "cuda",
    )
    expected_target_calls = (
        result.predecode_gate_target_calls
        + result.ordinary_adapter_stats.target_logit_calls
        + result.dflash_adapter_stats.target_logit_calls
        + result.dflash_adapter_stats.target_feature_calls
    )
    final_target_integration["validation_call_reconciliation"] = (
        _target_forward_reconciliation(
            initial_target_integration,
            final_target_integration,
            expected_validation_calls=expected_target_calls,
            formal_npu=formal_npu,
        )
    )
    final_runtime_preflight = _validate_formal_cli_inputs(args)
    if formal_npu and final_runtime_preflight != initial_runtime_preflight:
        raise RuntimeError("embedded runtime identity changed during validation")
    if formal_npu:
        state_policy = (
            "the receiver declares that each full-prefix call starts from a "
            "clean KV/GDN state; a bounded P-Q-P output gate passed, while the "
            "actual per-state/chunk device trace remains pending; V1 performs "
            "no speculative state commit or rollback"
        )
        target_operator_policy = (
            "the receiver contract changes only the target model source to "
            "add an opt-in DFlash feature side output; it does not modify the "
            "receiver-owned ChunkGatedDeltaRule or CacheUpdate implementations; "
            "actual device operator trace remains pending"
        )
        known_internal_interface_use = {
            "controller_and_draft": "do not directly call receiver target ACLNN interfaces",
            "ChunkGatedDeltaRule": "receiver declares fresh call-local GDN state; controller/draft do not call it directly and actual device isolation trace is pending",
            "CacheUpdate": "receiver declares fresh call-local block-table KV updates; controller/draft do not call it directly and actual device isolation trace is pending",
            "DynamicQuant": "not called by controller/draft; receiver target use must be traced",
            "GroupedMatmul": "not called by controller/draft; receiver target use must be traced",
            "QuantBatchMatmulV4444": "not called by controller/draft; receiver target use must be traced",
        }
    else:
        state_policy = (
            "the package-local HF/PyTorch target is replayed with use_cache=False; "
            "the bounded P-Q-P output gate passed; no receiver HIAI state claim "
            "or speculative state commit/rollback is made"
        )
        target_operator_policy = (
            "framework target and draft execution only; receiver HIAI custom "
            "operator interfaces are not part of this CPU/CUDA route"
        )
        known_internal_interface_use = {
            "route": "NOT_APPLICABLE_FRAMEWORK_TARGET",
        }
    report = {
        "schema_version": 2,
        "route": "qwen3.5-dflash-v1-full-prefix-replay",
        "classification": {
            "cpu": "CPU/framework simulation",
            "cuda": "CUDA/framework full-prefix validation",
            "npu": "NPU/framework execution; complete 310P gate remains external",
        }.get(adapter.device.type, "framework device execution"),
        "strict_greedy_exact_match": True,
        "feature_capture_zero_impact": result.feature_capture_zero_impact,
        "bounded_full_prefix_repeatability": (
            result.bounded_full_prefix_repeatability
        ),
        "state_policy": state_policy,
        "device": str(adapter.device),
        "dtype": str(adapter.dtype),
        "runtime_identity": _runtime_identity(adapter.device),
        "ops_backend": backend,
        "npu_layout": args.npu_layout if formal_npu else None,
        "target_operator_policy": target_operator_policy,
        "target_integration": final_target_integration,
        "runtime_preflight": final_runtime_preflight,
        "dflash_execution_gate": dflash_execution,
        "known_internal_interface_use": known_internal_interface_use,
        "operator_fallback_enabled": bool(args.allow_op_fallback),
        "target_dir": str(Path(args.target_dir).expanduser().resolve()),
        "draft_dir": str(Path(args.draft_dir).expanduser().resolve()),
        "draft_checkpoint": draft_checkpoint,
        "max_proposal_tokens": adapter.max_proposal_tokens,
        "request": _request_payload(
            args,
            effective_max_draft_tokens=(
                adapter.max_proposal_tokens
                if args.max_draft_tokens is None
                else args.max_draft_tokens
            ),
        ),
        "ordinary": _decode_payload(result.ordinary),
        "dflash": _decode_payload(result.dflash),
        "ordinary_adapter_stats": asdict(result.ordinary_adapter_stats),
        "dflash_adapter_stats": asdict(result.dflash_adapter_stats),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.report:
        destination = Path(args.report).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary_report = destination.with_name(
            f".{destination.name}.dflash-tmp-{os.getpid()}"
        )
        if temporary_report.exists() or temporary_report.is_symlink():
            raise RuntimeError("temporary report path already exists")
        try:
            temporary_report.write_text(serialized + "\n", encoding="utf-8")
            postwrite_runtime_preflight = _validate_formal_cli_inputs(args)
            if formal_npu and postwrite_runtime_preflight != initial_runtime_preflight:
                raise RuntimeError(
                    "embedded runtime identity changed while writing the report"
                )
            os.replace(temporary_report, destination)
        finally:
            if temporary_report.exists():
                temporary_report.unlink()
    _emit_progress(
        args.progress,
        "validation_end",
        {
            "status": "PASS",
            "dflash_round_executed": dflash_execution["draft_round_executed"],
            "target_call_reconciliation": final_target_integration[
                "validation_call_reconciliation"
            ]["status"],
        },
    )
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
