"""Correctness-first DFlash decoding with full-prefix target replay.

This module deliberately has no KV-cache or recurrent-state commit/rollback
path.  At every round the target callback receives a complete prefix.  The
correctness route verifies each proposal against its own isolated prefix,
instead of assuming that earlier logits are invariant when an unaccepted
suffix is appended to the same target call.  This is slow, but it gives the
portable golden a small and auditable strict-greedy acceptance rule.

The target callback (or ``forward_logits`` adapter method) must return logits
with shape ``[1, input_length, vocab_size]``.  The draft callback (or
``propose`` adapter method) receives ``[1, committed_length]`` token IDs and a
positive proposal limit, and returns at most that many token IDs.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import operator
from typing import Any, Protocol

import torch
from torch import Tensor

from .dflash_config import DFLASH_MIN_BLOCK_SIZE, OFFICIAL_DFLASH_BLOCK_SIZE


DraftTokens = Sequence[int] | Tensor
TargetLogitsCallback = Callable[[Tensor], Tensor | Any]
DraftProposalCallback = Callable[[Tensor, int], DraftTokens]


class TargetLogitsAdapter(Protocol):
    """Object adapter accepted by the replay decoders."""

    def forward_logits(self, input_ids: Tensor) -> Tensor | Any: ...


class DraftBlockAdapter(Protocol):
    """Object adapter accepted by :func:`dflash_full_prefix_greedy`."""

    def propose(self, prefix_ids: Tensor, proposal_limit: int) -> DraftTokens: ...


@dataclass
class ReplayDecodeStats:
    """Counters that make the deliberately repeated target work explicit."""

    target_calls: int = 0
    target_verify_calls: int = 0
    target_input_tokens_recomputed: int = 0
    target_rows_read: int = 0
    draft_calls: int = 0
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    fallback_tokens: int = 0
    speculation_disable_events: int = 0
    target_only_fallback_rounds: int = 0

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.drafted_tokens


@dataclass(frozen=True)
class ReplayRound:
    """One proposal/verification round, retained for golden diagnosis."""

    committed_prefix_length: int
    proposed_token_ids: tuple[int, ...]
    target_token_ids: tuple[int, ...]
    accepted_draft_token_ids: tuple[int, ...]
    fallback_token_id: int | None
    emitted_token_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReplayDecodeResult:
    """Token-level result of ordinary or DFlash strict-greedy generation."""

    mode: str
    prompt_token_ids: tuple[int, ...]
    generated_token_ids: tuple[int, ...]
    reached_eos: bool
    stop_reason: str
    stats: ReplayDecodeStats = field(compare=False)
    rounds: tuple[ReplayRound, ...] = field(default=(), compare=False)

    @property
    def all_token_ids(self) -> tuple[int, ...]:
        return self.prompt_token_ids + self.generated_token_ids


def _token_id(value: object, *, source: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{source} must contain integer token IDs, not bool")
    try:
        token = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{source} must contain integer token IDs") from error
    token = int(token)
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
            raise ValueError("the replay golden supports batch=1 token IDs only")
        if torch.is_floating_point(prompt_token_ids) or torch.is_complex(
            prompt_token_ids
        ):
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
    return frozenset(_token_id(value, source="eos_token_ids") for value in eos_token_ids)


def _non_negative_count(value: int, *, name: str) -> int:
    try:
        normalized = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer") from error
    if normalized < 0:
        raise ValueError(f"{name} must be non-negative")
    return int(normalized)


def _positive_count(value: int, *, name: str) -> int:
    normalized = _non_negative_count(value, name=name)
    if normalized == 0:
        raise ValueError(f"{name} must be positive")
    return normalized


def _dflash_block_size(value: int) -> int:
    """Validate upstream DFlash's total-row block-size convention."""

    normalized = _positive_count(value, name="block_size")
    if normalized < DFLASH_MIN_BLOCK_SIZE:
        raise ValueError(
            "block_size must be at least 2 (one anchor plus one proposal)"
        )
    if normalized > OFFICIAL_DFLASH_BLOCK_SIZE:
        raise ValueError(
            "block_size exceeds the locked upstream maximum: "
            f"{normalized} > {OFFICIAL_DFLASH_BLOCK_SIZE}"
        )
    return normalized


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
        raise TypeError("target callback must return a Tensor or an object with Tensor logits")
    return logits


def _call_target(
    target: TargetLogitsCallback | TargetLogitsAdapter,
    input_ids: Tensor,
) -> Tensor:
    """Call a target under the caller's full-prefix isolation contract.

    This portable primitive does not know receiver-owned KV/GDN state.  Formal
    HIAI NPU execution must enter through ``Qwen35DFlashFullPrefixAdapter`` and
    ``InternalTargetFacade``; a raw callable is for CPU/framework simulation.
    """
    forward_logits = getattr(target, "forward_logits", None)
    with torch.inference_mode():
        if callable(forward_logits):
            output = forward_logits(input_ids)
        elif callable(target):
            output = target(input_ids)
        else:
            raise TypeError("target must be callable or provide forward_logits(input_ids)")
    return _extract_logits(output)


def _target_top1(
    target: TargetLogitsCallback | TargetLogitsAdapter,
    tokens: Sequence[int],
    positions: Sequence[int],
    *,
    device: torch.device,
    stats: ReplayDecodeStats,
    verification: bool,
) -> list[int]:
    input_ids = _input_ids(tokens, device)
    logits = _call_target(target, input_ids)
    expected_prefix = (1, len(tokens))
    if logits.ndim != 3 or tuple(logits.shape[:2]) != expected_prefix:
        raise ValueError(
            "target logits must have shape [1, input_length, vocab_size]; "
            f"got {tuple(logits.shape)} for input length {len(tokens)}"
        )
    if logits.shape[-1] <= 0:
        raise ValueError("target logits vocab_size must be positive")
    if not torch.is_floating_point(logits):
        raise TypeError("target logits must use a floating-point dtype")
    if any(position < 0 or position >= len(tokens) for position in positions):
        raise IndexError("a target logit row is outside the replay input")

    # Copy only the rows that participate in the acceptance decision.  This
    # also keeps device-specific finite/argmax behavior out of the golden.
    rows = logits[0, list(positions), :].detach().float().cpu()
    if not bool(torch.isfinite(rows).all()):
        raise FloatingPointError("target returned a non-finite verification logit")
    top1 = rows.argmax(dim=-1).tolist()

    stats.target_calls += 1
    stats.target_verify_calls += int(verification)
    stats.target_input_tokens_recomputed += len(tokens)
    stats.target_rows_read += len(positions)
    return [int(token) for token in top1]


def _call_draft(
    draft: DraftProposalCallback | DraftBlockAdapter,
    prefix_ids: Tensor,
    proposal_limit: int,
) -> DraftTokens:
    propose = getattr(draft, "propose", None)
    with torch.inference_mode():
        if callable(propose):
            return propose(prefix_ids, proposal_limit)
        if callable(draft):
            return draft(prefix_ids, proposal_limit)
    raise TypeError("draft must be callable or provide propose(prefix_ids, limit)")


def _normalize_proposals(
    raw: DraftTokens,
    *,
    proposal_limit: int,
    eos_token_ids: frozenset[int],
) -> list[int]:
    if isinstance(raw, Tensor):
        if torch.is_floating_point(raw) or torch.is_complex(raw):
            raise TypeError("draft proposals must use an integer dtype")
        if raw.ndim == 1:
            values = raw.detach().cpu().tolist()
        elif raw.ndim == 2 and raw.shape[0] == 1:
            values = raw[0].detach().cpu().tolist()
        else:
            raise ValueError("draft proposals must have shape [K] or [1, K]")
    else:
        values = list(raw)
    proposals = [_token_id(value, source="draft proposals") for value in values]
    if len(proposals) > proposal_limit:
        raise ValueError(
            f"draft returned {len(proposals)} tokens, limit is {proposal_limit}"
        )

    # A fixed-width drafter may populate slots after EOS.  They are outside
    # the autoregressive sequence and must not enter target verification.
    for index, token in enumerate(proposals):
        if token in eos_token_ids:
            return proposals[: index + 1]
    return proposals


def ordinary_full_prefix_greedy(
    target: TargetLogitsCallback | TargetLogitsAdapter,
    prompt_token_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
    input_device: str | torch.device | None = None,
) -> ReplayDecodeResult:
    """Generate the authoritative one-target-call-per-token greedy stream."""

    prompt, device = _normalize_prompt(prompt_token_ids, input_device)
    maximum = _non_negative_count(max_new_tokens, name="max_new_tokens")
    eos = _normalize_eos(eos_token_ids)
    committed = list(prompt)
    generated: list[int] = []
    stats = ReplayDecodeStats()
    reached_eos = False

    while len(generated) < maximum:
        token = _target_top1(
            target,
            committed,
            [len(committed) - 1],
            device=device,
            stats=stats,
            verification=False,
        )[0]
        generated.append(token)
        committed.append(token)
        if token in eos:
            reached_eos = True
            break

    return ReplayDecodeResult(
        mode="ordinary-full-prefix-greedy",
        prompt_token_ids=tuple(prompt),
        generated_token_ids=tuple(generated),
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        stats=stats,
    )


def dflash_full_prefix_greedy(
    target: TargetLogitsCallback | TargetLogitsAdapter,
    draft: DraftProposalCallback | DraftBlockAdapter,
    prompt_token_ids: Sequence[int] | Tensor,
    *,
    max_new_tokens: int,
    block_size: int,
    eos_token_ids: Iterable[int] = (),
    input_device: str | torch.device | None = None,
    verification_mode: str = "sequential",
) -> ReplayDecodeResult:
    """Run strict-greedy DFlash verification without scheduler-owned cache.

    A target callback may have private mutable state, but its owner must make
    every invocation behave like a fresh full-prefix call.  This function does
    not reset or validate receiver-specific state; the formal 310P route adds
    those gates in the Qwen adapter/facade layer.

    In the default ``sequential`` mode, proposal ``d[i]`` is verified by a
    fresh target call on ``committed + d[:i]``.  This avoids relying on a
    vectorized target's prefix-invariance across different input lengths and
    kernel choices.  ``vectorized`` retains the one-call diagnostic route in
    which target row ``prefix_length - 1 + i`` verifies ``d[i]``.

    Only the longest contiguous matching prefix is accepted.  The next target
    Top-1 token is then emitted as the correction (on mismatch) or bonus token
    (when every proposal matches), subject to EOS and ``max_new_tokens``.
    Consequently every emitted token is target-greedy in sequential mode.
    """

    prompt, device = _normalize_prompt(prompt_token_ids, input_device)
    maximum = _non_negative_count(max_new_tokens, name="max_new_tokens")
    block_size = _dflash_block_size(block_size)
    proposal_capacity = block_size - 1
    if verification_mode not in {"sequential", "vectorized"}:
        raise ValueError(
            "verification_mode must be 'sequential' or 'vectorized'"
        )
    eos = _normalize_eos(eos_token_ids)
    committed = list(prompt)
    generated: list[int] = []
    stats = ReplayDecodeStats()
    traces: list[ReplayRound] = []
    reached_eos = False

    while len(generated) < maximum and not reached_eos:
        remaining = maximum - len(generated)
        proposal_limit = min(proposal_capacity, remaining)
        prefix_length = len(committed)
        raw_proposals = _call_draft(
            draft,
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

        if proposals:
            if verification_mode == "vectorized":
                verification_input = [*committed, *proposals]
                first_row = prefix_length - 1
                target_tokens = _target_top1(
                    target,
                    verification_input,
                    list(range(first_row, first_row + len(proposals) + 1)),
                    device=device,
                    stats=stats,
                    verification=True,
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
            else:
                # Verify only committed/accepted tokens.  In particular, a
                # mismatching proposal never becomes target context merely
                # because it occupied an earlier row in a vectorized call.
                target_tokens = []
                accepted_count = 0
                for proposal_index, proposal in enumerate(proposals):
                    verification_prefix = [
                        *committed,
                        *proposals[:proposal_index],
                    ]
                    target_token = _target_top1(
                        target,
                        verification_prefix,
                        [len(verification_prefix) - 1],
                        device=device,
                        stats=stats,
                        verification=True,
                    )[0]
                    target_tokens.append(target_token)
                    if proposal != target_token:
                        break
                    accepted_count += 1
                if accepted_count == len(proposals):
                    bonus_prefix = [*committed, *proposals]
                    target_tokens.extend(
                        _target_top1(
                            target,
                            bonus_prefix,
                            [len(bonus_prefix) - 1],
                            device=device,
                            stats=stats,
                            verification=True,
                        )
                    )
        else:
            target_tokens = _target_top1(
                target,
                committed,
                [prefix_length - 1],
                device=device,
                stats=stats,
                verification=False,
            )
            accepted_count = 0

        accepted_tokens = proposals[:accepted_count]
        stats.accepted_draft_tokens += accepted_count
        stats.rejected_draft_tokens += len(proposals) - accepted_count
        emitted_this_round: list[int] = []

        for token in accepted_tokens:
            # proposal_limit <= remaining makes this guard an invariant check,
            # not an expected truncation path.
            if len(generated) >= maximum:
                raise AssertionError("accepted draft tokens exceeded the generation limit")
            generated.append(token)
            committed.append(token)
            emitted_this_round.append(token)
            if token in eos:
                reached_eos = True
                break

        fallback_token: int | None = None
        if not reached_eos and len(generated) < maximum:
            # target_tokens[accepted_count] is valid for all branches: it is
            # the mismatch correction, the all-accepted bonus, or the ordinary
            # fallback when the draft returned an empty block.
            fallback_token = target_tokens[accepted_count]
            generated.append(fallback_token)
            committed.append(fallback_token)
            emitted_this_round.append(fallback_token)
            stats.fallback_tokens += 1
            reached_eos = fallback_token in eos

        if not emitted_this_round:
            raise AssertionError("a DFlash round made no token-level progress")
        traces.append(
            ReplayRound(
                committed_prefix_length=prefix_length,
                proposed_token_ids=tuple(proposals),
                target_token_ids=tuple(target_tokens),
                accepted_draft_token_ids=tuple(accepted_tokens),
                fallback_token_id=fallback_token,
                emitted_token_ids=tuple(emitted_this_round),
            )
        )

    return ReplayDecodeResult(
        mode=f"dflash-full-prefix-{verification_mode}-strict-greedy",
        prompt_token_ids=tuple(prompt),
        generated_token_ids=tuple(generated),
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        stats=stats,
        rounds=tuple(traces),
    )


def assert_exact_greedy_match(
    ordinary: ReplayDecodeResult,
    dflash: ReplayDecodeResult,
) -> None:
    """Raise with the first token offset when DFlash differs from greedy."""

    if ordinary.prompt_token_ids != dflash.prompt_token_ids:
        raise AssertionError("ordinary and DFlash prompts differ")
    if ordinary.generated_token_ids != dflash.generated_token_ids:
        shared = min(
            len(ordinary.generated_token_ids), len(dflash.generated_token_ids)
        )
        mismatch = next(
            (
                index
                for index in range(shared)
                if ordinary.generated_token_ids[index]
                != dflash.generated_token_ids[index]
            ),
            shared,
        )
        ordinary_token = (
            ordinary.generated_token_ids[mismatch]
            if mismatch < len(ordinary.generated_token_ids)
            else None
        )
        dflash_token = (
            dflash.generated_token_ids[mismatch]
            if mismatch < len(dflash.generated_token_ids)
            else None
        )
        raise AssertionError(
            "ordinary/DFlash mismatch at generated offset "
            f"{mismatch}: ordinary={ordinary_token}, dflash={dflash_token}"
        )
    if ordinary.reached_eos != dflash.reached_eos:
        raise AssertionError("ordinary and DFlash EOS outcomes differ")
    if ordinary.stop_reason != dflash.stop_reason:
        raise AssertionError("ordinary and DFlash stop reasons differ")
