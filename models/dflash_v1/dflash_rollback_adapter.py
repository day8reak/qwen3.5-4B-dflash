"""Qwen3.5 DFlash adapters for persistent CPU/CUDA/NPU rollback.

The framework controller uses a persistent Transformers ``DynamicCache``.  A
verification block is speculative: full-attention KV is cropped back to the
round-start length, GDN conv/recurrent tensors are restored from a private
snapshot, and only ``anchor + accepted proposals`` is replayed one token at a
time.  The replay is bounded by ``K + 1`` and never includes the historical
prefix.

The HIAI bridge implements the same public methods with native GDR state banks
and a logical paged-KV cursor, so the scheduler below it is backend-neutral.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from time import perf_counter
from typing import Any

import torch
from torch import Tensor, nn
from transformers.cache_utils import DynamicCache

from .dflash_qwen_adapter_v1 import (
    Qwen35DFlashFullPrefixAdapter,
    _block_size,
    _proposal_count,
    _tensor_field,
    _validate_token_tensor,
)
from .dflash_reference_decode_v1 import (
    ReplayDecodeResult,
    assert_exact_greedy_match,
)
from .dflash_rollback_decode import (
    dflash_rollback_greedy,
    ordinary_incremental_greedy,
)
from .modeling_dflash import DFlashDraftModel


@dataclass(frozen=True)
class _LinearStateSnapshot:
    layer_index: int
    conv_states: Tensor | None
    recurrent_states: Tensor | None
    is_conv_states_initialized: bool
    is_recurrent_states_initialized: bool
    has_previous_state: bool


@dataclass(frozen=True)
class _FrameworkCacheSnapshot:
    sequence_length: int
    linear_states: tuple[_LinearStateSnapshot, ...]


def _output_field(output: object, name: str) -> object | None:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _clone_optional_tensor(value: object) -> Tensor | None:
    return value.detach().clone() if isinstance(value, Tensor) else None


def _snapshot_framework_cache(cache: DynamicCache) -> _FrameworkCacheSnapshot:
    """Save only state that ``Cache.crop`` cannot restore."""

    linear_states: list[_LinearStateSnapshot] = []
    for index, layer in enumerate(cache.layers):
        if not hasattr(layer, "conv_states") and not hasattr(
            layer,
            "recurrent_states",
        ):
            continue
        linear_states.append(
            _LinearStateSnapshot(
                layer_index=index,
                conv_states=_clone_optional_tensor(
                    getattr(layer, "conv_states", None)
                ),
                recurrent_states=_clone_optional_tensor(
                    getattr(layer, "recurrent_states", None)
                ),
                is_conv_states_initialized=bool(
                    getattr(layer, "is_conv_states_initialized", False)
                ),
                is_recurrent_states_initialized=bool(
                    getattr(layer, "is_recurrent_states_initialized", False)
                ),
                has_previous_state=bool(
                    getattr(layer, "has_previous_state", False)
                ),
            )
        )
    return _FrameworkCacheSnapshot(
        sequence_length=int(cache.get_seq_length()),
        linear_states=tuple(linear_states),
    )


def _restore_tensor_attribute(
    layer: object,
    name: str,
    snapshot: Tensor | None,
) -> None:
    current = getattr(layer, name, None)
    if snapshot is None:
        setattr(layer, name, None)
        return
    if (
        isinstance(current, Tensor)
        and current.shape == snapshot.shape
        and current.dtype == snapshot.dtype
        and current.device == snapshot.device
    ):
        current.copy_(snapshot)
    else:
        setattr(layer, name, snapshot.detach().clone())


@torch.inference_mode()
def _restore_framework_cache(
    cache: DynamicCache,
    snapshot: _FrameworkCacheSnapshot,
) -> None:
    cache.crop(snapshot.sequence_length)
    if int(cache.get_seq_length()) != snapshot.sequence_length:
        raise RuntimeError("attention KV cache did not restore its round-start length")
    for state in snapshot.linear_states:
        if state.layer_index >= len(cache.layers):
            raise RuntimeError("linear state snapshot references a missing cache layer")
        layer = cache.layers[state.layer_index]
        _restore_tensor_attribute(layer, "conv_states", state.conv_states)
        _restore_tensor_attribute(
            layer,
            "recurrent_states",
            state.recurrent_states,
        )
        setattr(
            layer,
            "is_conv_states_initialized",
            state.is_conv_states_initialized,
        )
        setattr(
            layer,
            "is_recurrent_states_initialized",
            state.is_recurrent_states_initialized,
        )
        setattr(layer, "has_previous_state", state.has_previous_state)


class FrameworkDFlashRollbackTarget(nn.Module):
    """Transactional target facade for CPU and CUDA framework execution."""

    dflash_rollback_contract_id = "qwen3.5-dflash-framework-rollback-v1"
    dflash_rollback_mode = (
        "dynamic-cache-crop-linear-state-restore-bounded-token-replay"
    )

    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        if not isinstance(target, nn.Module):
            raise TypeError("framework rollback target must be torch.nn.Module")
        if target.training:
            raise ValueError("framework rollback target must be in eval mode")
        self.target = target
        self._cache: DynamicCache | None = None
        self._mode: str | None = None
        self._pending_snapshot: _FrameworkCacheSnapshot | None = None
        self._pending_block: Tensor | None = None
        self._ordinary_prefill_calls = 0
        self._ordinary_decode_calls = 0
        self._rollback_prefill_calls = 0
        self._rollback_verify_calls = 0
        self._rollback_commit_transactions = 0
        self._rollback_commit_replay_calls = 0
        self._rollback_aborts = 0

    @property
    def config(self) -> object:
        config = getattr(self.target, "config", None)
        if config is None:
            raise TypeError("framework target must expose config")
        return config

    @property
    def dflash_rollback_audit(self) -> Mapping[str, object]:
        return {
            "contract_id": self.dflash_rollback_contract_id,
            "mode": self.dflash_rollback_mode,
            "historical_prefix_replay_during_verify": False,
            "commit_replay_scope": (
                "anchor_plus_accepted_prefix_only_one_token_per_call"
            ),
            "ordinary_prefill_calls": self._ordinary_prefill_calls,
            "ordinary_decode_calls": self._ordinary_decode_calls,
            "rollback_prefill_calls": self._rollback_prefill_calls,
            "rollback_verify_calls": self._rollback_verify_calls,
            "rollback_commit_transactions": self._rollback_commit_transactions,
            "rollback_commit_replay_calls": self._rollback_commit_replay_calls,
            "rollback_aborts": self._rollback_aborts,
            "pending_transaction": self._pending_block is not None,
            "cache_sequence_length": (
                None
                if self._cache is None
                else int(self._cache.get_seq_length())
            ),
        }

    def get_input_embeddings(self) -> nn.Module:
        getter = getattr(self.target, "get_input_embeddings", None)
        if not callable(getter):
            raise TypeError("framework target lacks get_input_embeddings()")
        module = getter()
        if not isinstance(module, nn.Module):
            raise TypeError("target input embedding must be torch.nn.Module")
        return module

    def get_output_embeddings(self) -> nn.Module:
        getter = getattr(self.target, "get_output_embeddings", None)
        if not callable(getter):
            raise TypeError("framework target lacks get_output_embeddings()")
        module = getter()
        if not isinstance(module, nn.Module):
            raise TypeError("target output embedding must be torch.nn.Module")
        return module

    def _new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.config)

    def _clear_transaction(self) -> None:
        self._pending_snapshot = None
        self._pending_block = None

    def _call(self, input_ids: Tensor, *, features: bool) -> object:
        if self._cache is None:
            raise RuntimeError("framework target cache has not been initialized")
        with torch.inference_mode():
            output = self.target(
                input_ids=input_ids,
                past_key_values=self._cache,
                use_cache=True,
                return_dict=True,
                output_hidden_states=False,
                output_dflash_features=features,
                logits_to_keep=0,
            )
        returned_cache = _output_field(output, "past_key_values")
        if returned_cache is not self._cache:
            raise RuntimeError(
                "framework target replaced the persistent DynamicCache object"
            )
        logits = _tensor_field(output, "logits")
        expected_rows = int(input_ids.shape[1])
        if logits is None or tuple(logits.shape[:2]) != (1, expected_rows):
            raise ValueError("framework target returned invalid incremental logits")
        captured = _tensor_field(output, "dflash_features")
        if features:
            if captured is None or tuple(captured.shape[:2]) != (1, expected_rows):
                raise ValueError(
                    "feature-enabled framework target returned invalid features"
                )
        elif captured is not None:
            raise ValueError("framework target returned features while disabled")
        return output

    def begin_ordinary(self, prompt_ids: Tensor) -> object:
        self._clear_transaction()
        self._cache = self._new_cache()
        self._mode = "ordinary"
        output = self._call(prompt_ids, features=False)
        self._ordinary_prefill_calls += 1
        return output

    def advance_ordinary(self, input_ids: Tensor) -> object:
        if self._mode != "ordinary" or self._pending_block is not None:
            raise RuntimeError("ordinary incremental target is not active")
        if tuple(input_ids.shape) != (1, 1):
            raise ValueError("ordinary incremental advance requires [1,1] input")
        output = self._call(input_ids, features=False)
        self._ordinary_decode_calls += 1
        return output

    def begin_rollback(self, prompt_ids: Tensor) -> object:
        self._clear_transaction()
        self._cache = self._new_cache()
        self._mode = "rollback"
        output = self._call(prompt_ids, features=True)
        self._rollback_prefill_calls += 1
        return output

    def verify_rollback(self, block_ids: Tensor) -> object:
        if self._mode != "rollback" or self._cache is None:
            raise RuntimeError("rollback target is not active")
        if self._pending_block is not None:
            raise RuntimeError("a rollback verification is already pending")
        snapshot = _snapshot_framework_cache(self._cache)
        self._pending_snapshot = snapshot
        self._pending_block = block_ids.detach().clone()
        try:
            output = self._call(block_ids, features=True)
        except Exception:
            _restore_framework_cache(self._cache, snapshot)
            self._clear_transaction()
            raise
        self._rollback_verify_calls += 1
        return output

    def commit_rollback(self, accepted_draft_tokens: int) -> object:
        if self._cache is None or self._pending_block is None:
            raise RuntimeError("no rollback verification is pending")
        snapshot = self._pending_snapshot
        assert snapshot is not None
        if isinstance(accepted_draft_tokens, bool) or not isinstance(
            accepted_draft_tokens,
            int,
        ):
            raise TypeError("accepted_draft_tokens must be an integer")
        maximum = int(self._pending_block.shape[1]) - 1
        if not 0 <= accepted_draft_tokens <= maximum:
            raise ValueError(
                "accepted_draft_tokens is outside the pending verify block"
            )
        accepted = accepted_draft_tokens
        commit_rows = accepted + 1
        block = self._pending_block[:, :commit_rows].detach().clone()
        _restore_framework_cache(self._cache, snapshot)
        try:
            logits: list[Tensor] = []
            features: list[Tensor] = []
            for row in range(commit_rows):
                output = self._call(block[:, row : row + 1], features=True)
                row_logits = _tensor_field(output, "logits")
                row_features = _tensor_field(output, "dflash_features")
                assert row_logits is not None and row_features is not None
                logits.append(row_logits)
                features.append(row_features)
        except Exception:
            _restore_framework_cache(self._cache, snapshot)
            self._clear_transaction()
            raise
        expected_length = snapshot.sequence_length + commit_rows
        if int(self._cache.get_seq_length()) != expected_length:
            _restore_framework_cache(self._cache, snapshot)
            self._clear_transaction()
            raise RuntimeError("framework rollback commit advanced KV by wrong length")
        self._clear_transaction()
        self._rollback_commit_transactions += 1
        self._rollback_commit_replay_calls += commit_rows
        return {
            "logits": torch.cat(logits, dim=1),
            "dflash_features": torch.cat(features, dim=1),
            "past_key_values": self._cache,
        }

    def abort_rollback(self) -> None:
        if self._cache is not None and self._pending_snapshot is not None:
            _restore_framework_cache(self._cache, self._pending_snapshot)
            self._rollback_aborts += 1
        self._clear_transaction()
        self._cache = None
        self._mode = None


@dataclass
class Qwen35RollbackAdapterStats:
    ordinary_prefill_calls: int = 0
    ordinary_decode_calls: int = 0
    rollback_prefill_calls: int = 0
    rollback_verify_calls: int = 0
    rollback_commit_calls: int = 0
    rollback_committed_input_tokens: int = 0
    draft_calls: int = 0
    draft_context_tokens_reused: int = 0
    draft_block_tokens: int = 0
    proposed_tokens: int = 0
    draft_feature_projection_calls: int = 0
    draft_feature_tokens_projected: int = 0
    speculation_disabled: bool = False


class Qwen35DFlashRollbackAdapter(Qwen35DFlashFullPrefixAdapter):
    """Bind the official DFlash draft to a transactional target controller."""

    def __init__(
        self,
        target: nn.Module,
        draft: DFlashDraftModel,
        **kwargs: Any,
    ) -> None:
        super().__init__(target, draft, **kwargs)
        required = (
            "begin_ordinary",
            "advance_ordinary",
            "begin_rollback",
            "verify_rollback",
            "commit_rollback",
            "abort_rollback",
        )
        missing = [name for name in required if not callable(getattr(target, name, None))]
        if missing:
            raise TypeError(
                "transactional target is missing methods: " + ", ".join(missing)
            )
        self.rollback_stats = Qwen35RollbackAdapterStats()
        self._rollback_projected_features: Tensor | None = None
        self._pending_verify_rows: int | None = None
        self._drafting_disabled = False

    def reset_rollback_stats(self) -> None:
        self.rollback_stats = Qwen35RollbackAdapterStats()

    def snapshot_rollback_stats(self) -> Qwen35RollbackAdapterStats:
        return replace(self.rollback_stats)

    def _validated_output(
        self,
        output: object,
        *,
        rows: int | None,
        features: bool,
    ) -> tuple[Tensor, Tensor | None]:
        logits = _tensor_field(output, "logits")
        if logits is None or logits.ndim != 3 or logits.shape[0] != 1:
            raise TypeError("transactional target output must expose [1,T,V] logits")
        if rows is not None and logits.shape[1] != rows:
            raise ValueError("transactional target returned the wrong logit rows")
        if logits.shape[-1] != self.vocab_size:
            raise ValueError("transactional target logits use the wrong vocabulary")
        captured = _tensor_field(output, "dflash_features")
        if features:
            if captured is None or captured.ndim != 3 or captured.shape[0] != 1:
                raise TypeError("transactional target did not return DFlash features")
            if rows is not None and captured.shape[1] != rows:
                raise ValueError("transactional target returned wrong feature rows")
            if captured.shape[-1] != int(self.draft.config.feature_size):
                raise ValueError("transactional target features use the wrong width")
            if captured.device != self.device or captured.dtype != self.dtype:
                raise ValueError("transactional features differ in device or dtype")
            if self.check_finite_features and not bool(torch.isfinite(captured).all()):
                raise FloatingPointError("transactional target features are non-finite")
        elif captured is not None:
            raise ValueError("transactional target returned features while disabled")
        return logits, captured

    def begin_ordinary(self, prompt_ids: Tensor) -> Tensor:
        prompt_ids = _validate_token_tensor(
            prompt_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="ordinary incremental prompt_ids",
        )
        self._rollback_projected_features = None
        self._pending_verify_rows = None
        self._drafting_disabled = False
        output = self.target.begin_ordinary(prompt_ids)
        logits, _ = self._validated_output(
            output,
            rows=None,
            features=False,
        )
        self.rollback_stats.ordinary_prefill_calls += 1
        return logits

    def advance_ordinary(self, input_ids: Tensor) -> Tensor:
        input_ids = _validate_token_tensor(
            input_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="ordinary incremental input_ids",
        )
        if tuple(input_ids.shape) != (1, 1):
            raise ValueError("ordinary incremental input must have shape [1,1]")
        output = self.target.advance_ordinary(input_ids)
        logits, _ = self._validated_output(output, rows=1, features=False)
        self.rollback_stats.ordinary_decode_calls += 1
        return logits

    def begin_rollback(self, prompt_ids: Tensor) -> Tensor:
        prompt_ids = _validate_token_tensor(
            prompt_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="rollback prompt_ids",
        )
        self._rollback_projected_features = None
        self._pending_verify_rows = None
        self._drafting_disabled = False
        output = self.target.begin_rollback(prompt_ids)
        logits, features = self._validated_output(
            output,
            rows=None,
            features=True,
        )
        assert features is not None
        if features.shape[1] != prompt_ids.shape[1]:
            raise ValueError("rollback prefill features must cover the prompt once")
        with torch.inference_mode():
            projected = self.draft.project_target_hidden(features.detach())
        expected_projected = (
            1,
            int(prompt_ids.shape[1]),
            int(self.draft.config.hidden_size),
        )
        if tuple(projected.shape) != expected_projected:
            raise RuntimeError("Draft returned invalid projected prompt features")
        self._rollback_projected_features = projected
        self._pending_verify_rows = None
        self._drafting_disabled = False
        self.rollback_stats.rollback_prefill_calls += 1
        self.rollback_stats.draft_feature_projection_calls += 1
        self.rollback_stats.draft_feature_tokens_projected += int(
            prompt_ids.shape[1]
        )
        return logits

    def propose_rollback(self, prefix_ids: Tensor, proposal_limit: int) -> Tensor:
        prefix_ids = _validate_token_tensor(
            prefix_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="rollback draft prefix_ids",
        )
        proposal_count = _proposal_count(
            proposal_limit,
            maximum=self.max_proposal_tokens,
        )
        if self._drafting_disabled:
            raise RuntimeError("Draft proposal requested after speculation was disabled")
        target_hidden = self._rollback_projected_features
        if target_hidden is None:
            raise RuntimeError("begin_rollback() must run before drafting")
        expected_context = int(prefix_ids.shape[1]) - 1
        expected_features = (
            1,
            expected_context,
            int(self.draft.config.hidden_size),
        )
        if tuple(target_hidden.shape) != expected_features:
            raise RuntimeError(
                "committed feature history is not aligned before the current anchor: "
                f"expected {expected_features}, got {tuple(target_hidden.shape)}"
            )

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
        total_positions = expected_context + block_length
        if total_positions > int(self.draft.config.max_position_embeddings):
            raise ValueError("DFlash rollback block exceeds max_position_embeddings")
        position_ids = torch.arange(
            total_positions,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)
        with torch.inference_mode():
            proposed = self.draft.draft_top1_projected(
                target_hidden,
                noise_embedding,
                position_ids,
                self.lm_head_weight,
            )
        if tuple(proposed.shape) != (1, proposal_count):
            raise RuntimeError("DFlash rollback draft returned an invalid Top-1 shape")
        self.rollback_stats.draft_calls += 1
        self.rollback_stats.draft_context_tokens_reused += expected_context
        self.rollback_stats.draft_block_tokens += block_length
        self.rollback_stats.proposed_tokens += proposal_count
        return proposed

    def disable_speculation(self) -> None:
        """Stop maintaining Draft-only feature state after a zero-accept round."""

        self._drafting_disabled = True
        self._rollback_projected_features = None
        self.rollback_stats.speculation_disabled = True

    def verify_rollback(self, block_ids: Tensor) -> Tensor:
        block_ids = _validate_token_tensor(
            block_ids,
            device=self.device,
            vocab_size=self.vocab_size,
            name="rollback verification block_ids",
        )
        if not 1 <= block_ids.shape[1] <= self.max_block_size:
            raise ValueError(
                "rollback verification block exceeds the configured block_size"
            )
        if self._pending_verify_rows is not None:
            raise RuntimeError("a rollback verification is already pending")
        output = self.target.verify_rollback(block_ids)
        logits, _ = self._validated_output(
            output,
            rows=int(block_ids.shape[1]),
            features=True,
        )
        self._pending_verify_rows = int(block_ids.shape[1])
        self.rollback_stats.rollback_verify_calls += 1
        return logits

    def commit_rollback(self, accepted_draft_tokens: int) -> None:
        rows = self._pending_verify_rows
        if rows is None:
            raise RuntimeError("no rollback verification is pending")
        if isinstance(accepted_draft_tokens, bool) or not isinstance(
            accepted_draft_tokens,
            int,
        ):
            raise TypeError("accepted_draft_tokens must be an integer")
        if not 0 <= accepted_draft_tokens < rows:
            raise ValueError("accepted_draft_tokens is outside the pending block")
        output = self.target.commit_rollback(accepted_draft_tokens)
        committed_rows = accepted_draft_tokens + 1
        _, committed_features = self._validated_output(
            output,
            rows=committed_rows,
            features=True,
        )
        assert committed_features is not None
        if not self._drafting_disabled:
            cached = self._rollback_projected_features
            if cached is None:
                raise RuntimeError("rollback projected feature history was lost")
            with torch.inference_mode():
                projected = self.draft.project_target_hidden(
                    committed_features.detach()
                )
            self._rollback_projected_features = torch.cat(
                (cached, projected),
                dim=1,
            )
            self.rollback_stats.draft_feature_projection_calls += 1
            self.rollback_stats.draft_feature_tokens_projected += committed_rows
        self._pending_verify_rows = None
        self.rollback_stats.rollback_commit_calls += 1
        self.rollback_stats.rollback_committed_input_tokens += committed_rows

    def abort_rollback(self) -> None:
        self.target.abort_rollback()
        self._pending_verify_rows = None
        self._rollback_projected_features = None


@dataclass(frozen=True)
class Qwen35RollbackValidation:
    ordinary: ReplayDecodeResult
    dflash: ReplayDecodeResult
    ordinary_adapter_stats: Qwen35RollbackAdapterStats
    dflash_adapter_stats: Qwen35RollbackAdapterStats
    target_rollback_audit: Mapping[str, object]
    ordinary_elapsed_seconds: float
    dflash_elapsed_seconds: float
    verification_mode: str = "incremental_transactional_rollback"


def validate_qwen35_dflash_rollback(
    adapter: Qwen35DFlashRollbackAdapter,
    prompt_token_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    block_size: int | None = None,
    eos_token_ids: Iterable[int] = (),
    progress_callback: Callable[[str, Mapping[str, object]], None] | None = None,
) -> Qwen35RollbackValidation:
    """Compare ordinary and DFlash incremental streams with zero mismatch."""

    effective_block_size = (
        adapter.max_block_size
        if block_size is None
        else _block_size(block_size, maximum=adapter.max_block_size)
    )
    adapter.reset_rollback_stats()
    ordinary_started = perf_counter()
    ordinary = ordinary_incremental_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
    )
    ordinary_elapsed_seconds = perf_counter() - ordinary_started
    if progress_callback is not None:
        progress_callback(
            "ordinary_decode_end",
            {
                "elapsed_seconds": ordinary_elapsed_seconds,
                "generated_tokens": len(ordinary.generated_token_ids),
            },
        )
    ordinary_stats = adapter.snapshot_rollback_stats()
    adapter.reset_rollback_stats()
    dflash_started = perf_counter()
    dflash = dflash_rollback_greedy(
        adapter,
        prompt_token_ids,
        max_new_tokens=max_new_tokens,
        block_size=effective_block_size,
        eos_token_ids=eos_token_ids,
        input_device=adapter.device,
        progress_callback=progress_callback,
    )
    dflash_elapsed_seconds = perf_counter() - dflash_started
    if progress_callback is not None:
        progress_callback(
            "dflash_decode_end",
            {
                "elapsed_seconds": dflash_elapsed_seconds,
                "generated_tokens": len(dflash.generated_token_ids),
                "draft_rounds": dflash.stats.draft_calls,
            },
        )
    dflash_stats = adapter.snapshot_rollback_stats()
    assert_exact_greedy_match(ordinary, dflash)
    raw_audit = getattr(adapter.target, "dflash_rollback_audit", None)
    audit = dict(raw_audit) if isinstance(raw_audit, Mapping) else {}
    return Qwen35RollbackValidation(
        ordinary=ordinary,
        dflash=dflash,
        ordinary_adapter_stats=ordinary_stats,
        dflash_adapter_stats=dflash_stats,
        target_rollback_audit=audit,
        ordinary_elapsed_seconds=ordinary_elapsed_seconds,
        dflash_elapsed_seconds=dflash_elapsed_seconds,
    )


def rollback_validation_payload(
    result: Qwen35RollbackValidation,
) -> Mapping[str, object]:
    """Small serializable payload shared by tests and the CLI."""

    return {
        "verification_mode": result.verification_mode,
        "strict_greedy_exact_match": True,
        "ordinary": asdict(result.ordinary),
        "dflash": asdict(result.dflash),
        "ordinary_adapter_stats": asdict(result.ordinary_adapter_stats),
        "dflash_adapter_stats": asdict(result.dflash_adapter_stats),
        "target_rollback_audit": dict(result.target_rollback_audit),
        "timings_seconds": {
            "ordinary_decode": result.ordinary_elapsed_seconds,
            "dflash_decode": result.dflash_elapsed_seconds,
        },
    }


__all__ = [
    "FrameworkDFlashRollbackTarget",
    "Qwen35DFlashRollbackAdapter",
    "Qwen35RollbackAdapterStats",
    "Qwen35RollbackValidation",
    "rollback_validation_payload",
    "validate_qwen35_dflash_rollback",
]
