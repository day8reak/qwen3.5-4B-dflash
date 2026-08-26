"""Incremental strict-greedy DFlash scheduling with transactional rollback.

Unlike :mod:`dflash_reference_decode_v1`, this module never asks the target to
recompute a growing committed prefix while validating proposals.  The target
adapter owns one persistent state and exposes an explicit transaction:

``begin_rollback(prompt)``
    Prefill the prompt once and return logits that produce the first anchor.
``verify_rollback([anchor, proposal...])``
    Execute one causal ``K + 1`` target block against the committed state.
``commit_rollback(a)``
    Commit the anchor plus the longest ``a`` accepted proposal tokens.  The
    correction/bonus token remains the next unprocessed anchor.

CPU/CUDA adapters may implement commit by restoring a short-block snapshot and
replaying only ``anchor + accepted proposals``.  A target with native state
banks may select the accepted slot directly.  Both are rollback paths: neither
replays the historical prefix during proposal verification.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
import operator
from typing import Any, Protocol

import torch
from torch import Tensor

from .dflash_reference_decode_v1 import (
    DraftTokens,
    ReplayDecodeResult,
    ReplayDecodeStats,
    ReplayRound,
)


_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


class IncrementalRollbackAdapter(Protocol):
    """Target/draft boundary required by the incremental scheduler."""

    def begin_ordinary(self, prompt_ids: Tensor) -> Tensor | Any: ...

    def advance_ordinary(self, input_ids: Tensor) -> Tensor | Any: ...

    def begin_rollback(self, prompt_ids: Tensor) -> Tensor | Any: ...

    def propose_rollback(
        self,
        prefix_ids: Tensor,
        max_draft_tokens: int,
    ) -> DraftTokens: ...

    def verify_rollback(self, block_ids: Tensor) -> Tensor | Any: ...

    def commit_rollback(self, accepted_draft_tokens: int) -> None: ...

    def abort_rollback(self) -> None: ...


def _token_id(value: object, *, source: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{source} must contain integer token IDs, not bool")
    try:
        token = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{source} must contain integer token IDs") from error
    if token < 0:
        raise ValueError(f"{source} token IDs must be non-negative")
    return token


def _normalize_prompt(
    prompt_token_ids: Sequence[int] | Tensor,
    input_device: str | torch.device | None,
) -> tuple[list[int], torch.device]:
    if isinstance(prompt_token_ids, Tensor):
        if prompt_token_ids.ndim == 1:
            values = prompt_token_ids.detach().cpu().tolist()
        elif prompt_token_ids.ndim == 2 and prompt_token_ids.shape[0] == 1:
            values = prompt_token_ids[0].detach().cpu().tolist()
        else:
            raise ValueError("incremental DFlash supports batch=1 token IDs only")
        if prompt_token_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("prompt_token_ids must use an integer dtype")
        default_device = prompt_token_ids.device
    else:
        values = list(prompt_token_ids)
        default_device = torch.device("cpu")
    prompt = [_token_id(value, source="prompt") for value in values]
    if not prompt:
        raise ValueError("generation requires a non-empty prompt")
    device = torch.device(input_device) if input_device is not None else default_device
    return prompt, device


def _normalize_eos(eos_token_ids: Iterable[int]) -> frozenset[int]:
    return frozenset(
        _token_id(value, source="eos_token_ids") for value in eos_token_ids
    )


def _count(value: int, *, name: str, positive: bool = False) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, not bool")
    try:
        result = int(operator.index(value))
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    minimum = 1 if positive else 0
    if result < minimum:
        qualifier = "positive" if positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _input_ids(tokens: Sequence[int], device: torch.device) -> Tensor:
    return torch.tensor([list(tokens)], dtype=torch.long, device=device)


def _extract_logits(output: object) -> Tensor:
    if isinstance(output, Tensor):
        logits = output
    elif isinstance(output, Mapping):
        logits = output.get("logits")
    else:
        logits = getattr(output, "logits", None)
    if not isinstance(logits, Tensor):
        raise TypeError("rollback target output must expose Tensor logits")
    return logits


def _top1_rows(
    output: object,
    *,
    expected_rows: int | None,
    source: str,
) -> list[int]:
    logits = _extract_logits(output)
    if logits.ndim != 3 or logits.shape[0] != 1 or logits.shape[-1] <= 0:
        raise ValueError(f"{source} logits must have shape [1,T,V]")
    if expected_rows is not None and logits.shape[1] != expected_rows:
        raise ValueError(
            f"{source} logits must contain {expected_rows} rows, "
            f"got {logits.shape[1]}"
        )
    if not torch.is_floating_point(logits):
        raise TypeError(f"{source} logits must use a floating-point dtype")
    rows = logits.detach().float().cpu()
    if not bool(torch.isfinite(rows).all()):
        raise FloatingPointError(f"{source} logits contain non-finite values")
    return [int(value) for value in rows[0].argmax(dim=-1).tolist()]


def _normalize_proposals(
    raw: DraftTokens,
    *,
    proposal_limit: int,
    eos_token_ids: frozenset[int],
) -> list[int]:
    if isinstance(raw, Tensor):
        if raw.dtype not in _INTEGER_DTYPES:
            raise TypeError("draft proposals must use an integer dtype")
        if raw.ndim == 1:
            values = raw.detach().cpu().tolist()
        elif raw.ndim == 2 and raw.shape[0] == 1:
            values = raw[0].detach().cpu().tolist()
        else:
            raise ValueError("draft proposals must have shape [K] or [1,K]")
    else:
        values = list(raw)
    proposals = [_token_id(value, source="draft proposals") for value in values]
    if len(proposals) > proposal_limit:
        raise ValueError(
            f"draft returned {len(proposals)} tokens, limit is {proposal_limit}"
        )
    for index, token in enumerate(proposals):
        if token in eos_token_ids:
            return proposals[: index + 1]
    return proposals


def ordinary_incremental_greedy(
    adapter: IncrementalRollbackAdapter,
    prompt_token_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    input_device: str | torch.device | None = None,
) -> ReplayDecodeResult:
    """Generate the authoritative stream with one persistent target state."""

    prompt, device = _normalize_prompt(prompt_token_ids, input_device)
    maximum = _count(max_new_tokens, name="max_new_tokens")
    eos = _normalize_eos(eos_token_ids)
    generated: list[int] = []
    stats = ReplayDecodeStats()
    reached_eos = False
    if maximum == 0:
        return ReplayDecodeResult(
            mode="ordinary-incremental-greedy",
            prompt_token_ids=tuple(prompt),
            generated_token_ids=(),
            reached_eos=False,
            stop_reason="max_new_tokens",
            stats=stats,
        )

    output = adapter.begin_ordinary(_input_ids(prompt, device))
    stats.target_calls += 1
    stats.target_input_tokens_recomputed += len(prompt)
    while len(generated) < maximum:
        token = _top1_rows(
            output,
            expected_rows=None,
            source="ordinary incremental target",
        )[-1]
        stats.target_rows_read += 1
        generated.append(token)
        if token in eos:
            reached_eos = True
            break
        if len(generated) == maximum:
            break
        output = adapter.advance_ordinary(_input_ids([token], device))
        stats.target_calls += 1
        stats.target_input_tokens_recomputed += 1

    return ReplayDecodeResult(
        mode="ordinary-incremental-greedy",
        prompt_token_ids=tuple(prompt),
        generated_token_ids=tuple(generated),
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        stats=stats,
    )


def dflash_rollback_greedy(
    adapter: IncrementalRollbackAdapter,
    prompt_token_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    max_draft_tokens: int,
    eos_token_ids: Iterable[int] = (),
    input_device: str | torch.device | None = None,
) -> ReplayDecodeResult:
    """Run one-block target verification with transactional state rollback."""

    prompt, device = _normalize_prompt(prompt_token_ids, input_device)
    maximum = _count(max_new_tokens, name="max_new_tokens")
    block_size = _count(
        max_draft_tokens,
        name="max_draft_tokens",
        positive=True,
    )
    eos = _normalize_eos(eos_token_ids)
    stats = ReplayDecodeStats()
    rounds: list[ReplayRound] = []
    if maximum == 0:
        return ReplayDecodeResult(
            mode="dflash-incremental-rollback-strict-greedy",
            prompt_token_ids=tuple(prompt),
            generated_token_ids=(),
            reached_eos=False,
            stop_reason="max_new_tokens",
            stats=stats,
        )

    bootstrap_output = adapter.begin_rollback(_input_ids(prompt, device))
    bootstrap_token = _top1_rows(
        bootstrap_output,
        expected_rows=None,
        source="rollback bootstrap target",
    )[-1]
    stats.target_calls += 1
    stats.target_input_tokens_recomputed += len(prompt)
    stats.target_rows_read += 1
    stats.fallback_tokens += 1
    generated = [bootstrap_token]
    committed = [*prompt, bootstrap_token]
    reached_eos = bootstrap_token in eos
    rounds.append(
        ReplayRound(
            committed_prefix_length=len(prompt),
            proposed_token_ids=(),
            target_token_ids=(bootstrap_token,),
            accepted_draft_token_ids=(),
            fallback_token_id=bootstrap_token,
            emitted_token_ids=(bootstrap_token,),
        )
    )

    while len(generated) < maximum and not reached_eos:
        remaining = maximum - len(generated)
        proposal_limit = min(block_size, remaining)
        prefix_length = len(committed)
        raw_proposals = adapter.propose_rollback(
            _input_ids(committed, device),
            proposal_limit,
        )
        stats.draft_calls += 1
        proposals = _normalize_proposals(
            raw_proposals,
            proposal_limit=proposal_limit,
            eos_token_ids=eos,
        )
        stats.drafted_tokens += len(proposals)

        block = [committed[-1], *proposals]
        try:
            verification_output = adapter.verify_rollback(
                _input_ids(block, device)
            )
            target_tokens = _top1_rows(
                verification_output,
                expected_rows=len(block),
                source="rollback verification target",
            )
            mismatch = next(
                (
                    index
                    for index, proposal in enumerate(proposals)
                    if proposal != target_tokens[index]
                ),
                None,
            )
            accepted_count = (
                len(proposals) if mismatch is None else mismatch
            )
            adapter.commit_rollback(accepted_count)
        except Exception:
            adapter.abort_rollback()
            raise

        stats.target_calls += 1
        stats.target_verify_calls += 1
        stats.target_input_tokens_recomputed += len(block)
        stats.target_rows_read += len(block)
        stats.accepted_draft_tokens += accepted_count
        stats.rejected_draft_tokens += len(proposals) - accepted_count

        accepted = proposals[:accepted_count]
        emitted_this_round: list[int] = []
        for token in accepted:
            if len(generated) >= maximum:
                raise AssertionError("accepted tokens exceeded max_new_tokens")
            generated.append(token)
            committed.append(token)
            emitted_this_round.append(token)
            if token in eos:
                reached_eos = True
                break

        fallback_token: int | None = None
        if not reached_eos and len(generated) < maximum:
            fallback_token = target_tokens[accepted_count]
            generated.append(fallback_token)
            committed.append(fallback_token)
            emitted_this_round.append(fallback_token)
            stats.fallback_tokens += 1
            reached_eos = fallback_token in eos

        if not emitted_this_round:
            raise AssertionError("a DFlash rollback round made no token progress")
        rounds.append(
            ReplayRound(
                committed_prefix_length=prefix_length,
                proposed_token_ids=tuple(proposals),
                target_token_ids=tuple(target_tokens),
                accepted_draft_token_ids=tuple(accepted),
                fallback_token_id=fallback_token,
                emitted_token_ids=tuple(emitted_this_round),
            )
        )

    return ReplayDecodeResult(
        mode="dflash-incremental-rollback-strict-greedy",
        prompt_token_ids=tuple(prompt),
        generated_token_ids=tuple(generated),
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        stats=stats,
        rounds=tuple(rounds),
    )


__all__ = [
    "IncrementalRollbackAdapter",
    "dflash_rollback_greedy",
    "ordinary_incremental_greedy",
]
