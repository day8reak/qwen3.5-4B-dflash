"""Explicit-state graph modules for the approved incremental OM candidate.

These modules expose every request state tensor to AscendCL.  They deliberately
do not claim target readiness: the fixed ``all_seq_lengths_q=max_cache`` policy
and TorchAir dynamic Draft feature-tail gear remain real-device proof gates.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .incremental import ExactAcceptCommitStateGraph, _valid_eos_matches
from .contracts import AirGraphSpec, CustomOpExportSpec


VERIFY_ROWS = 16
PROPOSAL_ROWS = VERIFY_ROWS - 1
PREFILL_ROWS = 64
SCALAR_STATE_SEED_POLICY = "per-linear-layer-jit-v1"
CACHE_INDEX_POLICY = "once-per-verify-v1"


class _ExplicitTargetGraph(nn.Module):
    def __init__(
        self,
        target: nn.Module,
        *,
        kv_cache_max_len: int,
        include_lm_head: bool = True,
    ) -> None:
        super().__init__()
        execution = getattr(target, "dflash_execution_model", None)
        embedding = getattr(target, "_target_quantized_embedding", None)
        if not isinstance(execution, nn.Module):
            raise TypeError("incremental Target graph requires an execution model")
        if not isinstance(embedding, nn.Module):
            raise TypeError("incremental quant Target graph requires INT8 embedding")
        language_model = getattr(execution, "language_model", None)
        lm_head = getattr(execution, "lm_head", None)
        if not isinstance(language_model, nn.Module) or not isinstance(
            lm_head, nn.Module
        ):
            raise TypeError("incremental Target requires language_model and lm_head")
        if (
            getattr(language_model, "dflash_scalar_state_seed_policy", None)
            != SCALAR_STATE_SEED_POLICY
        ):
            raise RuntimeError(
                "incremental Target requires rollback modeling with "
                f"{SCALAR_STATE_SEED_POLICY} scalar-state seeding"
            )
        if (
            getattr(language_model, "dflash_cache_index_policy", None)
            != CACHE_INDEX_POLICY
        ):
            raise RuntimeError(
                "incremental Target requires rollback modeling with "
                f"{CACHE_INDEX_POLICY} cache-index reuse"
            )
        config = getattr(execution, "config", None)
        layer_types = tuple(getattr(config, "layer_types", ()))
        if not layer_types or any(
            item not in {"linear_attention", "full_attention"}
            for item in layer_types
        ):
            raise ValueError("Target config has an invalid layer_types layout")
        if kv_cache_max_len <= 0 or kv_cache_max_len % 64:
            raise ValueError("kv_cache_max_len must be positive and divisible by 64")
        # Register only the modules this physical graph executes.  In
        # particular, target-prefill deliberately excludes lm_head so its AIR
        # and OM cannot retain a dead full-vocabulary weight.  The final
        # prompt chunk is completed by TargetPrefillHeadGraph.
        self.language_model = language_model
        self.lm_head = lm_head if include_lm_head else None
        self.quantized_embedding = embedding
        self.layer_types = layer_types
        self.linear_layers = layer_types.count("linear_attention")
        self.full_layers = layer_types.count("full_attention")
        self.kv_cache_max_len = kv_cache_max_len

    def _state_list(
        self,
        conv_state: Tensor,
        recurrent_state: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        *,
        verify_rows: int | None,
    ) -> list[tuple[Tensor, Tensor]]:
        state: list[tuple[Tensor, Tensor]] = []
        linear_index = 0
        full_index = 0
        for layer_type in self.layer_types:
            if layer_type == "linear_attention":
                conv = conv_state[linear_index]
                recurrent = recurrent_state[linear_index]
                if verify_rows is None:
                    # Ordinary causal-conv updates its Tensor in place.  Keep
                    # the public OM input immutable and return the clone.
                    conv = conv.clone()
                    recurrent = recurrent.clone()
                # Verify deliberately keeps the public committed state scalar
                # here.  Causal-conv consumes the scalar directly; rollback
                # modeling seeds only the recurrent fixed T-slot input just
                # before each linear-attention layer consumes it.  Expanding
                # both banks for all 24 layers at this boundary created about
                # 792 MiB of graph-entry seed tensors.  The GDR-MTP custom-op
                # ABI and its per-row output bank are unchanged.
                state.append((conv, recurrent))
                linear_index += 1
            else:
                state.append(
                    (key_cache[full_index], value_cache[full_index])
                )
                full_index += 1
        return state

    def _group_state(
        self,
        state: list[tuple[Tensor, Tensor]],
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        conv: list[Tensor] = []
        recurrent: list[Tensor] = []
        key: list[Tensor] = []
        value: list[Tensor] = []
        for layer_type, pair in zip(self.layer_types, state):
            if layer_type == "linear_attention":
                conv.append(pair[0])
                recurrent.append(pair[1].to(torch.float32))
            else:
                key.append(pair[0])
                value.append(pair[1])
        return (
            torch.stack(conv, dim=0),
            torch.stack(recurrent, dim=0),
            torch.stack(key, dim=0),
            torch.stack(value, dim=0),
        )

    def _attention_mask(self, positions: Tensor) -> Tensor:
        columns = torch.arange(
            self.kv_cache_max_len,
            dtype=positions.dtype,
            device=positions.device,
        )
        visible = columns.reshape(1, -1) <= positions.reshape(-1, 1)
        # ADN consumes FP16 masks.  Construct the exact 0/-inf values in that
        # dtype once instead of inserting the same FP32->FP16 cast in every
        # full-attention layer.
        zero = torch.zeros((), dtype=torch.float16, device=positions.device)
        negative = torch.full(
            (), float("-inf"), dtype=torch.float16, device=positions.device
        )
        return torch.where(visible, zero, negative).unsqueeze(0).unsqueeze(0)

    def _body(
        self,
        input_ids: Tensor,
        logical_target_cursor: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        *,
        output_features: bool,
        gdr_effective_length: Tensor | None,
        verify: bool,
    ) -> tuple[Tensor, Tensor | None, Tensor, Tensor, Tensor, Tensor]:
        rows = input_ids.shape[1]
        offsets = torch.arange(rows, dtype=torch.long, device=input_ids.device)
        positions = logical_target_cursor.to(torch.long).reshape(1) + offsets
        state = self._state_list(
            conv_state,
            recurrent_state,
            key_cache,
            value_cache,
            verify_rows=rows if verify else None,
        )
        accepted = (
            torch.zeros((input_ids.shape[0],), dtype=torch.int8, device=input_ids.device)
            if verify
            else None
        )
        cache_target_blocks = (
            (positions // 64).to(torch.int32) if verify else None
        )
        cache_offsets = (
            (positions % 64).to(torch.int32) if verify else None
        )
        text_output = self.language_model(
            input_ids=input_ids,
            attention_mask=self._attention_mask(positions),
            position_ids=positions.unsqueeze(0),
            past_key_values=state,
            inputs_embeds=self.quantized_embedding(input_ids),
            use_cache=True,
            new_kv_cache_pos=positions,
            # The receiver schema exposes this as SymInt[], not a runtime
            # Tensor.  A single reusable OM therefore uses the physical cache
            # extent and relies on the explicit causal mask/logical cursor.
            # Exact equivalence remains a mandatory real-device gate.
            allQLen=[self.kv_cache_max_len],
            output_dflash_features=output_features,
            accepted_tokens=accepted,
            gdr_effective_length=gdr_effective_length,
            dflash_cache_target_blocks=cache_target_blocks,
            dflash_cache_offsets=cache_offsets,
        )
        if output_features:
            hidden, features = text_output
        else:
            hidden = text_output
            features = None
        next_conv, next_recurrent, next_key, next_value = self._group_state(state)
        return hidden, features, next_conv, next_recurrent, next_key, next_value

    def _top1(self, hidden: Tensor) -> Tensor:
        if self.lm_head is None:
            raise RuntimeError("this Target graph has no LM head")
        return torch.argmax(self.lm_head(hidden), dim=-1)


class TargetPrefillStateGraph(_ExplicitTargetGraph):
    """Consume one physical prompt chunk without a full-vocabulary head."""

    def __init__(self, target: nn.Module, *, kv_cache_max_len: int) -> None:
        super().__init__(
            target,
            kv_cache_max_len=kv_cache_max_len,
            include_lm_head=False,
        )

    def forward(
        self,
        input_ids: Tensor,
        effective_length: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        logical_target_cursor: Tensor,
    ) -> tuple[Tensor, ...]:
        hidden, features, conv, recurrent, key, value = self._body(
            input_ids,
            logical_target_cursor,
            conv_state,
            recurrent_state,
            key_cache,
            value_cache,
            output_features=True,
            gdr_effective_length=effective_length,
            verify=False,
        )
        assert features is not None
        batch_size = input_ids.shape[0]
        last_index = effective_length.to(torch.long).reshape(batch_size, 1, 1) - 1
        last_hidden = torch.gather(
            hidden,
            1,
            last_index.expand(batch_size, 1, hidden.shape[-1]),
        )
        rows = torch.arange(
            input_ids.shape[1], dtype=effective_length.dtype, device=input_ids.device
        ).reshape(1, -1, 1)
        masked_features = features * (
            rows < effective_length.reshape(batch_size, 1, 1)
        ).to(features.dtype)
        next_cursor = (
            logical_target_cursor.to(torch.int64)
            + effective_length.to(torch.int64)
        )
        return (
            last_hidden,
            masked_features,
            effective_length.to(torch.int32),
            conv,
            recurrent,
            key,
            value,
            next_cursor,
        )


class TargetPrefillHeadGraph(nn.Module):
    """Run QLinear Top1/EOS only after the final physical prompt chunk."""

    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        execution = getattr(target, "dflash_execution_model", None)
        lm_head = getattr(execution, "lm_head", None)
        if not isinstance(lm_head, nn.Module):
            raise TypeError("incremental Target prefill head requires lm_head")
        # Do not retain Target body modules in this physical graph.  Together
        # with TargetPrefillStateGraph excluding lm_head, this moves (rather
        # than duplicates) the prefill QLinear weight across the two OMs.
        self.lm_head = lm_head

    def forward(
        self,
        last_hidden: Tensor,
        eos_token_ids: Tensor,
        eos_token_count: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        first_token = torch.argmax(self.lm_head(last_hidden), dim=-1)
        batch_size = last_hidden.shape[0]
        committed = torch.cat(
            (
                first_token,
                torch.zeros(
                    (batch_size, VERIFY_ROWS - 1),
                    dtype=first_token.dtype,
                    device=first_token.device,
                ),
            ),
            dim=1,
        )
        commit_count = torch.ones(
            (batch_size,), dtype=torch.int32, device=last_hidden.device
        )
        finished = _valid_eos_matches(
            first_token,
            eos_token_ids,
            eos_token_count,
        ).squeeze(1)
        return committed, commit_count, finished


class TargetDecodeOneStateGraph(_ExplicitTargetGraph):
    """Consume one ordinary input row without executing the Draft."""

    def forward(
        self,
        input_ids: Tensor,
        eos_token_ids: Tensor,
        eos_token_count: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        logical_target_cursor: Tensor,
    ) -> tuple[Tensor, ...]:
        effective = torch.ones(
            (input_ids.shape[0],), dtype=torch.int16, device=input_ids.device
        )
        hidden, _, conv, recurrent, key, value = self._body(
            input_ids,
            logical_target_cursor,
            conv_state,
            recurrent_state,
            key_cache,
            value_cache,
            output_features=False,
            gdr_effective_length=effective,
            verify=False,
        )
        token = self._top1(hidden)
        committed = torch.cat(
            (
                token,
                torch.zeros(
                    (input_ids.shape[0], VERIFY_ROWS - 1),
                    dtype=token.dtype,
                    device=token.device,
                ),
            ),
            dim=1,
        )
        commit_count = torch.ones(
            (input_ids.shape[0],), dtype=torch.int32, device=input_ids.device
        )
        finished = _valid_eos_matches(
            token,
            eos_token_ids,
            eos_token_count,
        ).squeeze(1)
        return (
            committed,
            commit_count,
            finished,
            conv,
            recurrent,
            key,
            value,
            logical_target_cursor.to(torch.int64) + 1,
        )


class TargetVerifyCommitStateGraph(_ExplicitTargetGraph):
    """Verify fixed T=16 and embed exact accept/state selection in the graph."""

    def __init__(self, target: nn.Module, *, kv_cache_max_len: int) -> None:
        super().__init__(target, kv_cache_max_len=kv_cache_max_len)
        self.transaction = ExactAcceptCommitStateGraph(PROPOSAL_ROWS)

    def forward(
        self,
        verify_input_ids: Tensor,
        logical_proposal_count: Tensor,
        eos_token_ids: Tensor,
        eos_token_count: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        logical_target_cursor: Tensor,
    ) -> tuple[Tensor, ...]:
        hidden, features, conv_bank, recurrent_bank, key, value = self._body(
            verify_input_ids,
            logical_target_cursor,
            conv_state,
            recurrent_state,
            key_cache,
            value_cache,
            output_features=True,
            gdr_effective_length=None,
            verify=True,
        )
        assert features is not None
        target_top1 = self._top1(hidden)
        transaction = self.transaction(
            verify_input_ids[:, 1:],
            target_top1,
            logical_proposal_count,
            eos_token_ids,
            eos_token_count,
            conv_bank,
            recurrent_bank,
            features,
            logical_target_cursor,
        )
        return (*transaction, key, value)


class TargetStepStateGraph(TargetVerifyCommitStateGraph):
    """Execute an exact dynamic ``T=1..16`` Target step.

    ``T=1`` has zero Draft proposals and is therefore ordinary greedy decode.
    ``T=K+1`` verifies exactly ``K`` proposals.  Only the expensive Target body
    follows the selected physical gear; the small transaction tail pads its
    three row carriers back to the frozen 16-row external ABI.  State banks are
    not padded because the selected slot is always inside the physical prefix.
    """

    @staticmethod
    def _pad_rows(value: Tensor, width: int) -> Tensor:
        return F.pad(value, (0, width - value.shape[1]))

    @staticmethod
    def _pad_feature_rows(value: Tensor, width: int) -> Tensor:
        return F.pad(value, (0, 0, 0, width - value.shape[1]))

    def forward(
        self,
        verify_input_ids: Tensor,
        logical_proposal_count: Tensor,
        eos_token_ids: Tensor,
        eos_token_count: Tensor,
        conv_state: Tensor,
        recurrent_state: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        logical_target_cursor: Tensor,
    ) -> tuple[Tensor, ...]:
        hidden, features, conv_bank, recurrent_bank, key, value = self._body(
            verify_input_ids,
            logical_target_cursor,
            conv_state,
            recurrent_state,
            key_cache,
            value_cache,
            output_features=True,
            gdr_effective_length=None,
            verify=True,
        )
        assert features is not None
        # Pad before dropping the anchor.  Padding an already empty ``T-1``
        # slice makes torch.export infer a spurious T>=3 guard and excludes the
        # required ordinary-decode T=1 gear.
        fixed_input_ids = self._pad_rows(verify_input_ids, VERIFY_ROWS)
        proposal_ids = fixed_input_ids[:, 1:]
        target_top1 = self._pad_rows(self._top1(hidden), VERIFY_ROWS)
        fixed_features = self._pad_feature_rows(features, VERIFY_ROWS)
        transaction = self.transaction(
            proposal_ids,
            target_top1,
            logical_proposal_count,
            eos_token_ids,
            eos_token_count,
            conv_bank,
            recurrent_bank,
            fixed_features,
            logical_target_cursor,
        )
        return (*transaction, key, value)


class _FixedDraftCache:
    """Graph-local fixed-capacity Draft KV facade used by six Draft layers."""

    def __init__(
        self,
        key_cache: Tensor,
        value_cache: Tensor,
        cursor: Tensor,
        *,
        context_capacity: Any,
        block_rows: int,
    ) -> None:
        self.key_cache = key_cache
        self.value_cache = value_cache
        self.cursor = cursor
        self.context_capacity = context_capacity
        self.block_rows = block_rows
        self.num_layers = key_cache.shape[0]
        self.max_length = key_cache.shape[-2]
        self.next_keys: list[Tensor] = []
        self.next_values: list[Tensor] = []

    @property
    def round_context_length(self) -> Any:
        # Attention receives an explicit logical mask, while its physical key
        # carrier is fixed max_length + block_rows.
        return self.max_length

    def update(
        self,
        layer_index: int,
        key_states: Tensor,
        value_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        context_key = key_states[..., : self.context_capacity, :]
        context_value = value_states[..., : self.context_capacity, :]
        block_key = key_states[..., self.context_capacity :, :]
        block_value = value_states[..., self.context_capacity :, :]
        old_key = self.key_cache[layer_index]
        old_value = self.value_cache[layer_index]
        positions = self.cursor.to(torch.long).reshape(-1, 1) + torch.arange(
            self.context_capacity,
            dtype=torch.long,
            device=key_states.device,
        ).reshape(1, -1)
        positions = torch.clamp(positions, max=self.max_length - 1)
        index = positions[:, None, :, None].expand_as(context_key)
        next_key = old_key.scatter(2, index, context_key)
        next_value = old_value.scatter(2, index, context_value)
        self.next_keys.append(next_key)
        self.next_values.append(next_value)
        return (
            torch.cat((next_key, block_key), dim=2),
            torch.cat((next_value, block_value), dim=2),
        )

    def grouped(self) -> tuple[Tensor, Tensor]:
        return torch.stack(self.next_keys, dim=0), torch.stack(
            self.next_values, dim=0
        )


class DraftProposeStateGraph(nn.Module):
    """Append a Target feature tail to fixed Draft KV and propose 15 IDs."""

    def __init__(
        self,
        draft: nn.Module,
        input_embedding: nn.Module,
        output_embedding: nn.Module,
        *,
        kv_cache_max_len: int,
    ) -> None:
        super().__init__()
        if not isinstance(draft, nn.Module):
            raise TypeError("draft must be a torch module")
        input_weight = getattr(input_embedding, "weight", None)
        output_weight = getattr(output_embedding, "weight", None)
        if not isinstance(input_weight, Tensor) or not isinstance(output_weight, Tensor):
            raise TypeError("Draft graph requires input/output embedding weights")
        if kv_cache_max_len <= 0:
            raise ValueError("kv_cache_max_len must be positive")
        self.draft = draft
        self.input_embedding = input_embedding
        self.output_embedding = output_embedding
        self.kv_cache_max_len = kv_cache_max_len
        if int(getattr(draft.config, "block_size", 0)) != VERIFY_ROWS:
            raise ValueError("approved Draft graph requires block_size=16")

    def _attention_mask(
        self,
        cursor: Tensor,
        context_count: Tensor,
        *,
        device: torch.device,
    ) -> Tensor:
        cache_columns = torch.arange(
            self.kv_cache_max_len, dtype=torch.long, device=device
        ).reshape(1, 1, 1, -1)
        cache_visible = cache_columns < (
            cursor.to(torch.long) + context_count.to(torch.long)
        ).reshape(-1, 1, 1, 1)
        query = torch.arange(VERIFY_ROWS, dtype=torch.long, device=device).reshape(
            1, 1, VERIFY_ROWS, 1
        )
        block = torch.arange(VERIFY_ROWS, dtype=torch.long, device=device).reshape(
            1, 1, 1, VERIFY_ROWS
        )
        block_visible = block <= query
        return torch.cat(
            (
                cache_visible.expand(-1, 1, VERIFY_ROWS, -1),
                block_visible,
            ),
            dim=-1,
        )

    def forward(
        self,
        target_feature_tail: Tensor,
        committed_input_count: Tensor,
        previous_committed_token_ids: Tensor,
        previous_commit_count: Tensor,
        logical_proposal_count: Tensor,
        key_cache: Tensor,
        value_cache: Tensor,
        logical_draft_cursor: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        batch_size = target_feature_tail.shape[0]
        last_index = previous_commit_count.to(torch.long).reshape(
            batch_size, 1
        ) - 1
        anchor = torch.gather(previous_committed_token_ids, 1, last_index)
        mask_ids = torch.full(
            (batch_size, PROPOSAL_ROWS),
            int(self.draft.config.mask_token_id),
            dtype=anchor.dtype,
            device=anchor.device,
        )
        verify_input_ids = torch.cat((anchor, mask_ids), dim=1)
        projected = self.draft.project_target_hidden(target_feature_tail)
        noise = self.draft.embed_block(
            verify_input_ids,
            self.input_embedding.weight,
        )

        context_capacity = target_feature_tail.shape[1]
        context_offsets = torch.arange(
            context_capacity, dtype=torch.long, device=anchor.device
        ).reshape(1, -1)
        context_positions = torch.clamp(
            logical_draft_cursor.to(torch.long).reshape(batch_size, 1)
            + context_offsets,
            max=self.kv_cache_max_len - 1,
        )
        block_offsets = torch.arange(
            VERIFY_ROWS, dtype=torch.long, device=anchor.device
        ).reshape(1, -1)
        raw_block_positions = (
            logical_draft_cursor.to(torch.long).reshape(batch_size, 1)
            + committed_input_count.to(torch.long).reshape(batch_size, 1)
            + block_offsets
        )
        logical_block_rows = logical_proposal_count.to(torch.long).reshape(
            batch_size, 1
        ) + 1
        block_positions = torch.where(
            block_offsets < logical_block_rows,
            raw_block_positions,
            torch.clamp(raw_block_positions, max=self.kv_cache_max_len - 1),
        )
        positions = torch.cat((context_positions, block_positions), dim=1)
        cosine, sine = self.draft.rotary(positions, noise.dtype)
        cache = _FixedDraftCache(
            key_cache,
            value_cache,
            logical_draft_cursor,
            context_capacity=context_capacity,
            block_rows=VERIFY_ROWS,
        )
        attention_mask = self._attention_mask(
            logical_draft_cursor,
            committed_input_count,
            device=anchor.device,
        )
        hidden = noise
        for layer in self.draft.layers:
            hidden = layer.forward_cached(
                hidden,
                projected,
                cosine,
                sine,
                cache,
                attention_mask,
            )
        hidden = self.draft.norm(hidden)
        proposal_ids = self.draft.ops.top1(
            hidden[:, 1:, :],
            self.output_embedding.weight,
        )
        verify_input_ids = torch.cat((anchor, proposal_ids), dim=1)
        next_key, next_value = cache.grouped()
        next_cursor = (
            logical_draft_cursor.to(torch.int64)
            + committed_input_count.to(torch.int64)
        )
        return verify_input_ids, next_key, next_value, next_cursor


class FusedSpeculativeStepStateGraph(nn.Module):
    """Run the exact Draft proposal and fixed-T16 Target transaction in one graph.

    This is the approved ``fused-speculative-step`` topology candidate.  It
    preserves the two existing modules and their state contracts verbatim; the
    only removed boundary is the device-resident ``verify_input_ids`` carrier
    between two separately launched OMs.
    """

    def __init__(
        self,
        target: nn.Module,
        draft: nn.Module,
        input_embedding: nn.Module,
        output_embedding: nn.Module,
        *,
        kv_cache_max_len: int,
    ) -> None:
        super().__init__()
        self.draft_propose = DraftProposeStateGraph(
            draft,
            input_embedding,
            output_embedding,
            kv_cache_max_len=kv_cache_max_len,
        )
        self.target_verify = TargetVerifyCommitStateGraph(
            target,
            kv_cache_max_len=kv_cache_max_len,
        )

    def forward(
        self,
        target_feature_tail: Tensor,
        committed_input_count: Tensor,
        previous_committed_token_ids: Tensor,
        previous_commit_count: Tensor,
        logical_proposal_count: Tensor,
        eos_token_ids: Tensor,
        eos_token_count: Tensor,
        target_conv_state: Tensor,
        target_recurrent_state: Tensor,
        target_key_cache: Tensor,
        target_value_cache: Tensor,
        logical_target_cursor: Tensor,
        draft_key_cache: Tensor,
        draft_value_cache: Tensor,
        logical_draft_cursor: Tensor,
    ) -> tuple[Tensor, ...]:
        (
            verify_input_ids,
            next_draft_key,
            next_draft_value,
            next_draft_cursor,
        ) = self.draft_propose(
            target_feature_tail,
            committed_input_count,
            previous_committed_token_ids,
            previous_commit_count,
            logical_proposal_count,
            draft_key_cache,
            draft_value_cache,
            logical_draft_cursor,
        )
        target_outputs = self.target_verify(
            verify_input_ids,
            logical_proposal_count,
            eos_token_ids,
            eos_token_count,
            target_conv_state,
            target_recurrent_state,
            target_key_cache,
            target_value_cache,
            logical_target_cursor,
        )
        return (
            *target_outputs,
            next_draft_key,
            next_draft_value,
            next_draft_cursor,
        )


def incremental_state_graph_specs(
    target: nn.Module,
    draft: nn.Module,
    *,
    kv_cache_max_len: int,
    device: str | torch.device,
    dtype: torch.dtype,
    eos_table_width: int,
    ordinary_custom_ops: tuple[CustomOpExportSpec, ...],
    head_custom_ops: tuple[CustomOpExportSpec, ...],
    verify_custom_ops: tuple[CustomOpExportSpec, ...],
    unified_target_step: bool = False,
    fused_speculative_step: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> tuple[AirGraphSpec, ...]:
    """Create one approved exact incremental physical-topology candidate."""

    if dtype is not torch.float16:
        raise ValueError("incremental state graphs currently require float16")
    if isinstance(eos_table_width, bool) or not isinstance(eos_table_width, int):
        raise TypeError("eos_table_width must be an integer")
    if eos_table_width <= 0:
        raise ValueError("eos_table_width must be positive")
    if not isinstance(unified_target_step, bool) or not isinstance(
        fused_speculative_step, bool
    ):
        raise TypeError("incremental topology selectors must be booleans")
    if unified_target_step and fused_speculative_step:
        raise ValueError(
            "unified_target_step and fused_speculative_step are mutually exclusive"
        )
    target_device = torch.device(device)
    config = getattr(target, "config", None)
    layer_types = tuple(getattr(config, "layer_types", ()))
    linear_layers = layer_types.count("linear_attention")
    full_layers = layer_types.count("full_attention")
    key_heads = int(getattr(config, "linear_num_key_heads"))
    value_heads = int(getattr(config, "linear_num_value_heads"))
    key_dim = int(getattr(config, "linear_key_head_dim"))
    value_dim = int(getattr(config, "linear_value_head_dim"))
    conv_window = int(getattr(config, "linear_conv_kernel_dim"))
    conv_channels = key_heads * key_dim * 2 + value_heads * value_dim
    target_kv_heads = int(getattr(config, "num_key_value_heads"))
    target_head_dim = int(getattr(config, "head_dim"))
    packed_width = target_kv_heads * target_head_dim
    if packed_width % 16:
        raise ValueError("Target packed KV width must be divisible by 16")
    target_conv = torch.zeros(
        (linear_layers, 1, conv_channels, conv_window),
        dtype=dtype,
        device=target_device,
    )
    target_recurrent = torch.zeros(
        (linear_layers, 1, value_heads, key_dim, value_dim),
        dtype=torch.float32,
        device=target_device,
    )
    target_key = torch.zeros(
        (
            full_layers,
            kv_cache_max_len // 64,
            packed_width // 16,
            64,
            16,
        ),
        dtype=dtype,
        device=target_device,
    )
    target_value = torch.zeros_like(target_key)
    target_cursor = torch.zeros((1,), dtype=torch.long, device=target_device)
    eos_ids = torch.zeros(
        (eos_table_width,), dtype=torch.long, device=target_device
    )
    eos_count = torch.zeros((1,), dtype=torch.int32, device=target_device)

    input_embedding = target.get_input_embeddings()
    output_embedding = target.get_output_embeddings()
    draft_layers = len(draft.layers)
    draft_heads = int(getattr(draft.config, "num_key_value_heads"))
    draft_head_dim = int(getattr(draft.config, "head_dim"))
    draft_key = torch.zeros(
        (draft_layers, 1, draft_heads, kv_cache_max_len, draft_head_dim),
        dtype=dtype,
        device=target_device,
    )
    draft_value = torch.zeros_like(draft_key)
    draft_cursor = torch.zeros((1,), dtype=torch.long, device=target_device)
    feature_width = int(getattr(draft.config, "feature_size"))

    shared_metadata = {
        **dict(metadata or {}),
        "incremental_abi": "qwen35-4b-dflash-ascend310p-incremental-performance-v2",
        "status": "APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE",
        "state_owner": "C++ request context device buffers",
        "target_all_q_length_policy": "fixed kv_cache_max_len plus explicit causal mask and logical cursor",
        "target_all_q_length_evidence": "PENDING_REAL_NPU_EQUIVALENCE",
        "verify_scalar_state_seed_policy": SCALAR_STATE_SEED_POLICY,
        "verify_scalar_state_seed_evidence": (
            "source graph passes committed scalar GDN state; causal-conv "
            "consumes scalar state directly and recurrent state is seeded one "
            "linear-attention layer at a time; AIR/ATC peak memory remains a "
            "real-toolchain gate"
        ),
        "verify_cache_index_policy": CACHE_INDEX_POLICY,
        "verify_cache_index_evidence": (
            "target block and in-block offset vectors are derived once from "
            "the logical cursor and reused by both K/V updates in all eight "
            "full-attention layers"
        ),
        "draft_feature_tail": (
            "TorchAir discrete dynamic N gears: committed verify prefixes "
            "N=1..16 and prompt feature batches N=64..kv_cache_max_len in "
            "64-row increments; the runtime retains fixed N=16 as rollback"
        ),
        "draft_feature_prefix_policy": "exact-leading-committed-rows-v1",
        "claim_boundary": (
            "graph-construction candidate only; AIR/ATC, custom-node, "
            "real-model parity, complete-set memory and latency remain gated"
        ),
    }

    def graph_metadata(
        role: str,
        custom_ops: tuple[CustomOpExportSpec, ...],
        aliases: list[str],
    ) -> dict[str, Any]:
        return {
            **shared_metadata,
            "role": role,
            "device_buffer_aliases": aliases,
            "custom_op_export_contracts": [
                {
                    "torch_target": item.torch_target,
                    "ge_op_type": item.ge_op_type,
                    "minimum_occurrences": item.minimum_occurrences,
                    "preservation": "one registered GE operator; no Tensor decomposition",
                }
                for item in custom_ops
            ],
        }

    prefill = TargetPrefillStateGraph(
        target, kv_cache_max_len=kv_cache_max_len
    ).eval()
    prefill_head = TargetPrefillHeadGraph(target).eval()
    decode = TargetDecodeOneStateGraph(
        target, kv_cache_max_len=kv_cache_max_len
    ).eval()
    verify_type = (
        TargetStepStateGraph if unified_target_step else TargetVerifyCommitStateGraph
    )
    verify = verify_type(target, kv_cache_max_len=kv_cache_max_len).eval()
    propose = DraftProposeStateGraph(
        draft,
        input_embedding,
        output_embedding,
        kv_cache_max_len=kv_cache_max_len,
    ).eval()
    fused = (
        FusedSpeculativeStepStateGraph(
            target,
            draft,
            input_embedding,
            output_embedding,
            kv_cache_max_len=kv_cache_max_len,
        ).eval()
        if fused_speculative_step
        else None
    )
    draft_feature_gears = tuple(range(1, VERIFY_ROWS + 1)) + tuple(
        range(PREFILL_ROWS, kv_cache_max_len + 1, PREFILL_ROWS)
    )

    state_inputs = (
        target_conv,
        target_recurrent,
        target_key,
        target_value,
        target_cursor,
    )
    prefill_spec = AirGraphSpec(
        name="target-prefill",
        role="target-prefill",
        model=prefill,
        example_args=(
            torch.zeros((1, PREFILL_ROWS), dtype=torch.long, device=target_device),
            torch.full((1,), PREFILL_ROWS, dtype=torch.int16, device=target_device),
            *state_inputs,
        ),
        input_names=(
            "input_ids", "effective_length", "target_conv_state",
            "target_recurrent_state", "target_key_cache",
            "target_value_cache", "logical_target_cursor",
        ),
        output_names=(
            "last_hidden", "target_feature_tail", "committed_input_count",
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ),
        metadata=graph_metadata(
            "target-prefill",
            ordinary_custom_ops,
            [
                "target_conv_state", "target_recurrent_state",
                "target_key_cache", "target_value_cache",
                "logical_target_cursor",
            ],
        ),
        custom_ops=ordinary_custom_ops,
    )
    prefill_head_spec = AirGraphSpec(
        name="target-prefill-head",
        role="target-prefill-head",
        model=prefill_head,
        example_args=(
            torch.zeros(
                (1, 1, int(getattr(config, "hidden_size"))),
                dtype=dtype,
                device=target_device,
            ),
            eos_ids,
            eos_count,
        ),
        input_names=(
            "last_hidden", "eos_token_ids", "eos_token_count",
        ),
        output_names=(
            "committed_token_ids", "commit_count", "finished",
        ),
        metadata=graph_metadata(
            "target-prefill-head",
            head_custom_ops,
            ["last_hidden"],
        ),
        custom_ops=head_custom_ops,
    )
    decode_spec = AirGraphSpec(
        name="target-decode1",
        role="target-decode1",
        model=decode,
        example_args=(
            torch.zeros((1, 1), dtype=torch.long, device=target_device),
            eos_ids,
            eos_count,
            *state_inputs,
        ),
        input_names=(
            "input_ids", "eos_token_ids", "eos_token_count",
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ),
        output_names=(
            "committed_token_ids", "commit_count", "finished",
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ),
        metadata=graph_metadata(
            "target-decode1",
            ordinary_custom_ops,
            [
                "target_conv_state", "target_recurrent_state",
                "target_key_cache", "target_value_cache",
                "logical_target_cursor",
            ],
        ),
        custom_ops=ordinary_custom_ops,
    )
    draft_spec = AirGraphSpec(
        name="draft-propose",
        role="draft-propose",
        model=propose,
        example_args=(
            torch.zeros((1, VERIFY_ROWS, feature_width), dtype=dtype, device=target_device),
            torch.ones((1,), dtype=torch.int32, device=target_device),
            torch.zeros((1, VERIFY_ROWS), dtype=torch.long, device=target_device),
            torch.ones((1,), dtype=torch.int32, device=target_device),
            torch.full((1,), PROPOSAL_ROWS, dtype=torch.int32, device=target_device),
            draft_key,
            draft_value,
            draft_cursor,
        ),
        input_names=(
            "target_feature_tail", "committed_input_count",
            "previous_committed_token_ids", "previous_commit_count",
            "logical_proposal_count", "draft_key_cache",
            "draft_value_cache", "logical_draft_cursor",
        ),
        output_names=(
            "verify_input_ids", "draft_key_cache", "draft_value_cache",
            "logical_draft_cursor",
        ),
        dynamic=True,
        input_dim_gears={0: {1: draft_feature_gears}},
        metadata=graph_metadata(
            "draft-propose",
            (),
            ["draft_key_cache", "draft_value_cache", "logical_draft_cursor"],
        ),
    )
    verify_metadata = graph_metadata(
        "target-verify-commit",
        verify_custom_ops,
        [
            "target_conv_state", "target_recurrent_state",
            "target_key_cache", "target_value_cache",
            "logical_target_cursor",
        ],
    )
    if unified_target_step:
        verify_metadata = {
            **verify_metadata,
            "physical_topology": "split-prefill-head-four-resident-unified-target-step-v1",
            "target_step_gears": list(range(1, VERIFY_ROWS + 1)),
            "target_step_t1_semantics": "ordinary strict-greedy decode with zero proposals",
            "target_step_fixed_output_rows": VERIFY_ROWS,
        }
    verify_spec = AirGraphSpec(
        name="target-verify-commit",
        role="target-verify-commit",
        model=verify,
        example_args=(
            torch.zeros((1, VERIFY_ROWS), dtype=torch.long, device=target_device),
            torch.full((1,), PROPOSAL_ROWS, dtype=torch.int32, device=target_device),
            eos_ids,
            eos_count,
            *state_inputs,
        ),
        input_names=(
            "verify_input_ids", "logical_proposal_count", "eos_token_ids",
            "eos_token_count", "target_conv_state",
            "target_recurrent_state", "target_key_cache",
            "target_value_cache", "logical_target_cursor",
        ),
        output_names=(
            "committed_token_ids", "commit_count", "drafted_count",
            "accepted_count", "rejected_count", "target_conv_state",
            "target_recurrent_state", "target_feature_tail",
            "committed_input_count", "logical_target_cursor", "finished",
            "target_key_cache", "target_value_cache",
        ),
        dynamic=unified_target_step,
        input_dim_gears=(
            {0: {1: tuple(range(1, VERIFY_ROWS + 1))}}
            if unified_target_step
            else {}
        ),
        metadata=verify_metadata,
        custom_ops=verify_custom_ops,
    )
    if fused_speculative_step:
        assert fused is not None
        fused_metadata = graph_metadata(
            "fused-speculative-step",
            verify_custom_ops,
            [
                "target_conv_state",
                "target_recurrent_state",
                "target_key_cache",
                "target_value_cache",
                "logical_target_cursor",
                "draft_key_cache",
                "draft_value_cache",
                "logical_draft_cursor",
            ],
        )
        fused_metadata = {
            **fused_metadata,
            "physical_topology": (
                "split-prefill-head-four-resident-fused-speculative-step-v1"
            ),
            "fused_logical_roles": [
                "draft-propose",
                "target-verify-commit",
            ],
            "verify_input_ids_externalized": False,
            "target_verify_rows": VERIFY_ROWS,
        }
        fused_spec = AirGraphSpec(
            name="fused-speculative-step",
            role="fused-speculative-step",
            model=fused,
            example_args=(
                torch.zeros(
                    (1, VERIFY_ROWS, feature_width),
                    dtype=dtype,
                    device=target_device,
                ),
                torch.ones((1,), dtype=torch.int32, device=target_device),
                torch.zeros(
                    (1, VERIFY_ROWS), dtype=torch.long, device=target_device
                ),
                torch.ones((1,), dtype=torch.int32, device=target_device),
                torch.full(
                    (1,),
                    PROPOSAL_ROWS,
                    dtype=torch.int32,
                    device=target_device,
                ),
                eos_ids,
                eos_count,
                *state_inputs,
                draft_key,
                draft_value,
                draft_cursor,
            ),
            input_names=(
                "target_feature_tail",
                "committed_input_count",
                "previous_committed_token_ids",
                "previous_commit_count",
                "logical_proposal_count",
                "eos_token_ids",
                "eos_token_count",
                "target_conv_state",
                "target_recurrent_state",
                "target_key_cache",
                "target_value_cache",
                "logical_target_cursor",
                "draft_key_cache",
                "draft_value_cache",
                "logical_draft_cursor",
            ),
            output_names=(
                "committed_token_ids",
                "commit_count",
                "drafted_count",
                "accepted_count",
                "rejected_count",
                "target_conv_state",
                "target_recurrent_state",
                "target_feature_tail",
                "committed_input_count",
                "logical_target_cursor",
                "finished",
                "target_key_cache",
                "target_value_cache",
                "draft_key_cache",
                "draft_value_cache",
                "logical_draft_cursor",
            ),
            dynamic=True,
            input_dim_gears={0: {1: draft_feature_gears}},
            metadata=fused_metadata,
            custom_ops=verify_custom_ops,
        )
        return prefill_spec, prefill_head_spec, decode_spec, fused_spec
    if unified_target_step:
        return prefill_spec, prefill_head_spec, draft_spec, verify_spec
    return prefill_spec, prefill_head_spec, decode_spec, draft_spec, verify_spec


__all__ = [
    "DraftProposeStateGraph",
    "FusedSpeculativeStepStateGraph",
    "incremental_state_graph_specs",
    "PREFILL_ROWS",
    "PROPOSAL_ROWS",
    "TargetDecodeOneStateGraph",
    "TargetPrefillHeadGraph",
    "TargetPrefillStateGraph",
    "TargetStepStateGraph",
    "TargetVerifyCommitStateGraph",
    "VERIFY_ROWS",
]
