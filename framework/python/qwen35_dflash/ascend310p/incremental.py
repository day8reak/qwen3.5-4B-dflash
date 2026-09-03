"""Exact device-side transaction tail for the incremental multi-OM route.

The five-artifact baseline Target verify graph physically executes one fixed
``K + 1`` causal block. ``logical_proposal_count`` may be smaller than ``K``;
rows after that logical prefix are scratch rows and are never selected or
committed.  The unified Target-step candidate instead executes only
``logical_proposal_count + 1`` rows and pads its transaction carriers before
entering this fixed-width tail.  Because every Target component is causal,
later padded or scratch rows cannot change logits, features or state slots
belonging to the logical prefix.

This module contains no host reads, ``Tensor.item`` calls, sampling or
approximation.  It is intended to be embedded at the end of the Target verify
AIR graph so the C++ loop downloads only a compact transaction result.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


def _valid_eos_matches(
    token_ids: Tensor,
    eos_token_ids: Tensor,
    eos_token_count: Tensor,
) -> Tensor:
    """Return ``[B,N]`` matches against the valid prefix of the EOS table."""

    eos_width = eos_token_ids.shape[0]
    eos_columns = torch.arange(
        eos_width,
        dtype=eos_token_count.dtype,
        device=eos_token_ids.device,
    )
    valid_eos = eos_columns.unsqueeze(0) < eos_token_count.reshape(-1, 1)
    matches = token_ids.unsqueeze(-1).eq(eos_token_ids.reshape(1, 1, -1))
    return (matches & valid_eos.unsqueeze(1)).any(dim=-1)


def _select_layer_state_slot(state_bank: Tensor, slot: Tensor) -> Tensor:
    """Select one ``T`` slot from a grouped ``[L,B,T,...]`` state bank."""

    layer_count, batch_size = state_bank.shape[:2]
    index_shape = (1, batch_size, 1, *((1,) * (state_bank.ndim - 3)))
    index = slot.to(torch.long).reshape(index_shape)
    index = index.expand(layer_count, batch_size, 1, *state_bank.shape[3:])
    return torch.gather(state_bank, 2, index).squeeze(2)


class ExactAcceptCommitStateGraph(nn.Module):
    """Compute strict-greedy acceptance and persist only the selected state.

    Inputs use a fixed physical proposal width.  Counts are per-batch INT32
    tensors and the production ABI fixes ``B=1``.  ``target_features`` contains
    the rows for ``[anchor, p0, ..., p(K-1)]``.  The returned feature carrier
    keeps that static shape and zeros rows outside ``anchor + accepted``;
    ``committed_input_count`` tells the Draft graph which leading rows are real.

    State banks are grouped by layer solely at this graph boundary:

    * conv: ``[linear_layers,B,K+1,C,Kc]``;
    * recurrent: ``[linear_layers,B,K+1,H,Dk,Dv]``.

    The Target body may still unbind these tensors per layer before invoking
    the existing custom operators.
    """

    def __init__(self, proposal_width: int) -> None:
        super().__init__()
        if isinstance(proposal_width, bool) or not isinstance(proposal_width, int):
            raise TypeError("proposal_width must be an integer")
        if proposal_width <= 0:
            raise ValueError("proposal_width must be positive")
        self.proposal_width = proposal_width

    def forward(
        self,
        proposal_ids: Tensor,
        target_top1: Tensor,
        logical_proposal_count: Tensor,
        eos_token_ids: Tensor,
        eos_token_count: Tensor,
        conv_state_bank: Tensor,
        recurrent_state_bank: Tensor,
        target_features: Tensor,
        logical_target_cursor: Tensor,
    ) -> tuple[Tensor, ...]:
        width = self.proposal_width
        batch_size = proposal_ids.shape[0]

        positions = torch.arange(
            width,
            dtype=logical_proposal_count.dtype,
            device=proposal_ids.device,
        ).reshape(1, width)
        logical_limit = torch.minimum(
            logical_proposal_count.reshape(batch_size, 1),
            torch.full_like(logical_proposal_count.reshape(batch_size, 1), width),
        )
        within_requested = positions < logical_limit

        proposal_is_eos = _valid_eos_matches(
            proposal_ids,
            eos_token_ids,
            eos_token_count,
        ) & within_requested
        # TorchAir does not lower aten.amin or aten.cumprod on the receiver
        # toolchain.  Keep both prefix decisions in supported INT32 Cumsum
        # form.  Before the first EOS cumulative_eos == 0 == eos_bits, and at
        # the first EOS it is 1 == 1; every later row differs.  Consequently
        # drafted_mask contains exactly the requested prefix through its first
        # EOS (inclusive), without a host read or a reduction-min operation.
        eos_bits = proposal_is_eos.to(torch.int32)
        cumulative_eos = torch.cumsum(
            eos_bits,
            dim=1,
            dtype=torch.int32,
        )
        drafted_mask = within_requested & cumulative_eos.eq(eos_bits)
        drafted_count = drafted_mask.to(torch.int32).sum(dim=1)

        matches = proposal_ids.eq(target_top1[:, :width])
        mismatch_bits = (drafted_mask & ~matches).to(torch.int32)
        cumulative_mismatches = torch.cumsum(
            mismatch_bits,
            dim=1,
            dtype=torch.int32,
        )
        accepted_mask = drafted_mask & cumulative_mismatches.eq(0)
        accepted_count = accepted_mask.to(torch.int32).sum(dim=1)
        rejected_count = drafted_count.to(torch.int32) - accepted_count

        accepted_eos = (
            proposal_is_eos
            & accepted_mask
        ).any(dim=1)
        correction = torch.gather(
            target_top1,
            1,
            accepted_count.to(torch.long).reshape(batch_size, 1),
        ).squeeze(1)

        output_positions = torch.arange(
            width + 1,
            dtype=accepted_count.dtype,
            device=proposal_ids.device,
        ).reshape(1, width + 1)
        padded_proposals = torch.cat(
            (proposal_ids, torch.zeros_like(proposal_ids[:, :1])),
            dim=1,
        )
        committed_token_ids = torch.where(
            output_positions < accepted_count.reshape(batch_size, 1),
            padded_proposals,
            torch.zeros_like(padded_proposals),
        )
        committed_token_ids = torch.where(
            output_positions.eq(accepted_count.reshape(batch_size, 1))
            & ~accepted_eos.reshape(batch_size, 1),
            correction.reshape(batch_size, 1).expand_as(committed_token_ids),
            committed_token_ids,
        )
        commit_count = accepted_count + (~accepted_eos).to(torch.int32)

        selected_conv_state = _select_layer_state_slot(
            conv_state_bank,
            accepted_count,
        )
        selected_recurrent_state = _select_layer_state_slot(
            recurrent_state_bank,
            accepted_count,
        )
        committed_input_count = accepted_count + 1
        feature_positions = torch.arange(
            width + 1,
            dtype=committed_input_count.dtype,
            device=target_features.device,
        ).reshape(1, width + 1, 1)
        committed_features = target_features * (
            feature_positions
            < committed_input_count.reshape(batch_size, 1, 1)
        ).to(target_features.dtype)
        next_target_cursor = (
            logical_target_cursor.to(torch.int64)
            + committed_input_count.to(torch.int64)
        )

        last_index = torch.maximum(
            commit_count - 1,
            torch.zeros_like(commit_count),
        )
        last_token = torch.gather(
            committed_token_ids,
            1,
            last_index.to(torch.long).reshape(batch_size, 1),
        )
        finished = _valid_eos_matches(
            last_token,
            eos_token_ids,
            eos_token_count,
        ).squeeze(1)

        return (
            committed_token_ids,
            commit_count,
            drafted_count.to(torch.int32),
            accepted_count,
            rejected_count,
            selected_conv_state,
            selected_recurrent_state,
            committed_features,
            committed_input_count,
            next_target_cursor,
            finished,
        )


__all__ = ["ExactAcceptCommitStateGraph"]
