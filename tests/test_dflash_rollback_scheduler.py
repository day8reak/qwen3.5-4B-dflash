#!/usr/bin/env python3
"""Boundary checks for strict-greedy transactional DFlash scheduling."""

from __future__ import annotations

import sys
from pathlib import Path

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

from models.dflash_v1.dflash_rollback_decode import (  # noqa: E402
    dflash_rollback_greedy,
    ordinary_incremental_greedy,
)


def logits(tokens: list[int], vocab_size: int = 128) -> torch.Tensor:
    result = torch.full((1, len(tokens), vocab_size), -10.0)
    for row, token in enumerate(tokens):
        result[0, row, token] = 10.0
    return result


class BoundaryAdapter:
    def __init__(self, *, proposals: list[int], target_rows: list[int]) -> None:
        self.proposals = proposals
        self.target_rows = target_rows
        self.ordinary_index = 0
        self.committed_inputs: list[int] = []
        self.pending_block: list[int] | None = None
        self.verify_shapes: list[tuple[int, int]] = []
        self.commits: list[int] = []
        self.abort_calls = 0

    def begin_ordinary(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        self.ordinary_index = 0
        return logits([self.target_rows[0]])

    def advance_ordinary(self, input_ids: torch.Tensor) -> torch.Tensor:
        self.ordinary_index += 1
        return logits([self.target_rows[self.ordinary_index]])

    def begin_rollback(self, prompt_ids: torch.Tensor) -> torch.Tensor:
        self.committed_inputs = [int(value) for value in prompt_ids[0].tolist()]
        self.pending_block = None
        return logits([self.target_rows[0]])

    def propose_rollback(
        self,
        prefix_ids: torch.Tensor,
        proposal_limit: int,
    ) -> torch.Tensor:
        return torch.tensor(
            [self.proposals[:proposal_limit]],
            dtype=torch.long,
        )

    def verify_rollback(self, block_ids: torch.Tensor) -> torch.Tensor:
        self.pending_block = [int(value) for value in block_ids[0].tolist()]
        self.verify_shapes.append(tuple(block_ids.shape))
        rows = len(self.pending_block)
        return logits(self.target_rows[1 : 1 + rows])

    def commit_rollback(self, accepted_draft_tokens: int) -> None:
        if self.pending_block is None:
            raise AssertionError("commit without verify")
        self.commits.append(accepted_draft_tokens)
        self.committed_inputs.extend(
            self.pending_block[: accepted_draft_tokens + 1]
        )
        self.pending_block = None

    def abort_rollback(self) -> None:
        self.pending_block = None
        self.abort_calls += 1


class FailingVerifyAdapter(BoundaryAdapter):
    def verify_rollback(self, block_ids: torch.Tensor) -> torch.Tensor:
        self.pending_block = [int(value) for value in block_ids[0].tolist()]
        raise RuntimeError("injected verify failure")


def check_acceptance_boundary(accepted: int, proposal_count: int) -> None:
    eos = 99
    anchor = 20
    correct = [anchor + index + 1 for index in range(proposal_count)]
    # Row 0 is bootstrap.  Verification row i predicts proposal i, and row K
    # is the all-accepted bonus.  Put EOS at the correction/bonus row so every
    # boundary case completes exactly one transactional round.
    target_rows = [anchor, *correct, 77]
    target_rows[accepted + 1] = eos
    proposals = list(correct)
    if accepted < proposal_count:
        proposals[accepted] = 70 + accepted
        if proposals[accepted] == eos:
            proposals[accepted] = 69
        for index in range(accepted + 1, proposal_count):
            proposals[index] = 60 + index

    ordinary_adapter = BoundaryAdapter(
        proposals=proposals,
        target_rows=target_rows,
    )
    ordinary = ordinary_incremental_greedy(
        ordinary_adapter,
        [1],
        max_new_tokens=proposal_count + 2,
        eos_token_ids=[eos],
    )
    rollback_adapter = BoundaryAdapter(
        proposals=proposals,
        target_rows=target_rows,
    )
    rollback = dflash_rollback_greedy(
        rollback_adapter,
        [1],
        max_new_tokens=proposal_count + 2,
        block_size=proposal_count + 1,
        eos_token_ids=[eos],
    )

    assert rollback.generated_token_ids == ordinary.generated_token_ids
    assert rollback.reached_eos and ordinary.reached_eos
    assert rollback_adapter.verify_shapes == [(1, proposal_count + 1)]
    assert rollback_adapter.commits == [accepted]
    expected_inputs = [1, anchor, *correct[:accepted]]
    assert rollback_adapter.committed_inputs == expected_inputs
    draft_round = rollback.rounds[1]
    assert len(draft_round.accepted_draft_token_ids) == accepted
    assert draft_round.fallback_token_id == eos
    assert rollback.stats.target_input_tokens_recomputed == proposal_count + 2


def main() -> None:
    proposal_count = 15
    for accepted in range(proposal_count + 1):
        check_acceptance_boundary(accepted, proposal_count)
    failing = FailingVerifyAdapter(
        proposals=[21],
        target_rows=[20, 21, 99],
    )
    try:
        dflash_rollback_greedy(
            failing,
            [1],
            max_new_tokens=3,
            block_size=2,
            eos_token_ids=[99],
        )
    except RuntimeError as error:
        assert "injected verify failure" in str(error)
    else:
        raise AssertionError("injected verify failure was not propagated")
    assert failing.abort_calls == 1
    assert failing.pending_block is None
    print("PASS: rollback scheduler accepted=0..K and correction/bonus alignment")


if __name__ == "__main__":
    main()
