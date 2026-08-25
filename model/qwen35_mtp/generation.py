"""Accuracy-first ordinary and strict greedy speculative generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import time
from typing import Iterable, Sequence

import torch

from .backends import DraftBackend, MainBackend


@dataclass
class GenerationStats:
    main_calls: int = 0
    main_verify_calls: int = 0
    main_rows_projected: int = 0
    main_input_tokens_recomputed: int = 0
    draft_calls: int = 0
    drafted_tokens: int = 0
    accepted_draft_tokens: int = 0
    rejected_draft_tokens: int = 0
    fallback_tokens: int = 0
    main_seconds: float = 0.0
    draft_seconds: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        if self.drafted_tokens == 0:
            return 0.0
        return self.accepted_draft_tokens / self.drafted_tokens

    def to_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["acceptance_rate"] = self.acceptance_rate
        return result


@dataclass(frozen=True)
class GenerationResult:
    mode: str
    prompt_token_ids: list[int]
    generated_token_ids: list[int]
    reached_eos: bool
    stop_reason: str
    main_backend: str
    draft_backend: str | None
    stats: GenerationStats = field(compare=False)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "prompt_token_ids": self.prompt_token_ids,
            "generated_token_ids": self.generated_token_ids,
            "reached_eos": self.reached_eos,
            "stop_reason": self.stop_reason,
            "main_backend": self.main_backend,
            "draft_backend": self.draft_backend,
            "stats": self.stats.to_dict(),
        }


def _validate_inputs(prompt_token_ids: Sequence[int], max_new_tokens: int) -> list[int]:
    prompt = [int(token) for token in prompt_token_ids]
    if not prompt:
        raise ValueError("generation requires a non-empty tokenized prompt")
    if any(token < 0 for token in prompt):
        raise ValueError("token IDs must be non-negative")
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    return prompt


def _input_tensor(tokens: Sequence[int]) -> torch.Tensor:
    return torch.tensor([list(tokens)], dtype=torch.long)


def _main_evaluate(
    main: MainBackend,
    tokens: Sequence[int],
    positions: Sequence[int],
    stats: GenerationStats,
    *,
    verify: bool = False,
):
    start = time.perf_counter()
    result = main.evaluate(_input_tensor(tokens), positions)
    stats.main_seconds += time.perf_counter() - start
    stats.main_calls += 1
    stats.main_verify_calls += int(verify)
    stats.main_rows_projected += len(positions)
    stats.main_input_tokens_recomputed += len(tokens)
    expected = (1, len(positions))
    if tuple(result.top1_token_ids.shape) != expected:
        raise ValueError(
            f"main backend returned top1 shape {tuple(result.top1_token_ids.shape)}, "
            f"expected {expected}"
        )
    if result.hidden_states.ndim != 3 or result.hidden_states.shape[:2] != (
        1,
        len(tokens),
    ):
        raise ValueError("main backend returned hidden states with an invalid shape")
    if not torch.isfinite(result.hidden_states).all():
        raise FloatingPointError("main backend returned a non-finite hidden value")
    return result


def ordinary_generate(
    main: MainBackend,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    eos_token_ids: Iterable[int] = (),
) -> GenerationResult:
    """Greedy one-token-at-a-time target-model reference."""

    prompt = _validate_inputs(prompt_token_ids, max_new_tokens)
    eos = set(int(token) for token in eos_token_ids)
    prefix = list(prompt)
    generated: list[int] = []
    stats = GenerationStats()
    reached_eos = False
    while len(generated) < max_new_tokens:
        evaluation = _main_evaluate(main, prefix, [len(prefix) - 1], stats)
        token = int(evaluation.top1_token_ids[0, 0].item())
        generated.append(token)
        prefix.append(token)
        if token in eos:
            reached_eos = True
            break
    return GenerationResult(
        mode="ordinary-greedy",
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        main_backend=main.backend_id,
        draft_backend=None,
        stats=stats,
    )


def speculative_generate(
    main: MainBackend,
    draft: DraftBackend,
    prompt_token_ids: Sequence[int],
    *,
    max_new_tokens: int,
    max_draft_tokens: int = 2,
    eos_token_ids: Iterable[int] = (),
) -> GenerationResult:
    """Strict greedy speculative decoding with longest-prefix acceptance.

    Every emitted token is either explicitly accepted against the target's
    Top-1 token or is the target correction token.  Draft errors therefore
    affect acceptance/performance, never the final greedy token stream.
    """

    prompt = _validate_inputs(prompt_token_ids, max_new_tokens)
    if max_draft_tokens <= 0:
        raise ValueError("max_draft_tokens must be positive")
    eos = set(int(token) for token in eos_token_ids)
    prefix = list(prompt)
    generated: list[int] = []
    stats = GenerationStats()
    reached_eos = False

    # Emit the first target token.  This also makes the shifted MTP prefill
    # well-defined even for a one-token prompt.
    if max_new_tokens:
        first = _main_evaluate(main, prefix, [len(prefix) - 1], stats)
        token = int(first.top1_token_ids[0, 0].item())
        generated.append(token)
        prefix.append(token)
        if token in eos:
            reached_eos = True

    while len(generated) < max_new_tokens and not reached_eos:
        remaining = max_new_tokens - len(generated)
        if remaining < 2:
            fallback = _main_evaluate(main, prefix, [len(prefix) - 1], stats)
            token = int(fallback.top1_token_ids[0, 0].item())
            stats.fallback_tokens += 1
            generated.append(token)
            prefix.append(token)
            reached_eos = token in eos
            continue

        context = _main_evaluate(main, prefix, [], stats)
        draft_limit = min(max_draft_tokens, remaining - 1)
        draft_start = time.perf_counter()
        proposals = draft.propose(
            _input_tensor(prefix).to(context.hidden_states.device),
            context.hidden_states,
            draft_limit,
            eos_token_ids=eos,
        )
        stats.draft_seconds += time.perf_counter() - draft_start
        stats.draft_calls += 1
        if not proposals:
            fallback = _main_evaluate(main, prefix, [len(prefix) - 1], stats)
            token = int(fallback.top1_token_ids[0, 0].item())
            stats.fallback_tokens += 1
            generated.append(token)
            prefix.append(token)
            reached_eos = token in eos
            continue
        if len(proposals) > draft_limit:
            raise ValueError("draft backend returned more tokens than requested")
        stats.drafted_tokens += len(proposals)

        verification_tokens = [*prefix, *proposals]
        first_row = len(prefix) - 1
        rows = list(range(first_row, first_row + len(proposals) + 1))
        verified = _main_evaluate(
            main, verification_tokens, rows, stats, verify=True
        )
        targets = [int(value) for value in verified.top1_token_ids[0].tolist()]
        mismatch = next(
            (
                index
                for index, proposal in enumerate(proposals)
                if proposal != targets[index]
            ),
            None,
        )
        accepted = len(proposals) if mismatch is None else mismatch
        stats.accepted_draft_tokens += accepted
        stats.rejected_draft_tokens += len(proposals) - accepted
        correction = targets[accepted]
        candidates = [*proposals[:accepted], correction]

        for token in candidates:
            if len(generated) >= max_new_tokens:
                break
            generated.append(int(token))
            prefix.append(int(token))
            if token in eos:
                reached_eos = True
                break

    return GenerationResult(
        mode="mtp-strict-greedy",
        prompt_token_ids=prompt,
        generated_token_ids=generated,
        reached_eos=reached_eos,
        stop_reason="eos" if reached_eos else "max_new_tokens",
        main_backend=main.backend_id,
        draft_backend=draft.backend_id,
        stats=stats,
    )


def assert_exact_match(
    ordinary: GenerationResult,
    speculative: GenerationResult,
) -> None:
    if ordinary.prompt_token_ids != speculative.prompt_token_ids:
        raise AssertionError("ordinary and MTP prompts differ")
    if ordinary.generated_token_ids != speculative.generated_token_ids:
        mismatch = next(
            (
                index
                for index, pair in enumerate(
                    zip(ordinary.generated_token_ids, speculative.generated_token_ids)
                )
                if pair[0] != pair[1]
            ),
            min(len(ordinary.generated_token_ids), len(speculative.generated_token_ids)),
        )
        raise AssertionError(
            "ordinary/MTP token mismatch at generated index "
            f"{mismatch}: ordinary={ordinary.generated_token_ids}, "
            f"mtp={speculative.generated_token_ids}"
        )
    if ordinary.reached_eos != speculative.reached_eos:
        raise AssertionError("ordinary and MTP EOS outcomes differ")
