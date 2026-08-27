#!/usr/bin/env python3
"""CPU checks for the pure tensor helpers in the HIAI rollback source."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    REPOSITORY_ROOT
    / "models"
    / "modeling_qwen3_5_hiai_nd_dflash_rollback.py"
)
HELPERS = {
    "_require_dflash_accepted_tokens",
    "_select_dflash_state_slot",
    "seed_dflash_gdn_state_banks",
    "rebase_dflash_gdn_state_banks",
    "torch_dflash_causal_conv1d_mtp",
}
CONSTANTS = {
    "DFLASH_BLOCK_SIZE",
    "DFLASH_MAX_PROPOSALS",
    "DFLASH_MAX_VERIFY_TOKENS",
}


def load_helpers() -> dict[str, object]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"), filename=str(SOURCE))
    selected: list[ast.stmt] = []
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            names = {
                target.id
                for target in (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
                if isinstance(target, ast.Name)
            }
            if names & CONSTANTS:
                selected.append(node)
        elif isinstance(node, ast.FunctionDef) and node.name in HELPERS:
            selected.append(node)
    namespace: dict[str, object] = {
        "torch": torch,
        "F": F,
        "Optional": Optional,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    missing = HELPERS - namespace.keys()
    if missing:
        raise AssertionError(f"missing helper definitions: {sorted(missing)}")
    return namespace


def sequential_reference(
    hidden: torch.Tensor,
    bank: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    accepted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, tokens = hidden.shape
    state_len = bank.shape[-1]
    outputs = torch.empty_like(hidden)
    states = torch.empty_like(bank)
    for batch_index in range(batch):
        state = bank[batch_index, int(accepted[batch_index])].clone()
        for token_index in range(tokens):
            state = torch.cat(
                (state, hidden[batch_index, :, token_index : token_index + 1]),
                dim=-1,
            )[:, -state_len:]
            raw = (state * weight).sum(dim=-1) + bias
            outputs[batch_index, :, token_index] = F.silu(raw)
            states[batch_index, token_index] = state
    return outputs, states


def main() -> None:
    helper = load_helpers()
    assert helper["DFLASH_BLOCK_SIZE"] == 16
    assert helper["DFLASH_MAX_PROPOSALS"] == 15
    assert helper["DFLASH_MAX_VERIFY_TOKENS"] == 16
    seed = helper["seed_dflash_gdn_state_banks"]
    rebase = helper["rebase_dflash_gdn_state_banks"]
    conv = helper["torch_dflash_causal_conv1d_mtp"]

    torch.manual_seed(20260826)
    committed_conv = torch.randn(2, 3, 4, dtype=torch.float16)
    committed_recurrent = torch.randn(2, 2, 3, 4, dtype=torch.float16)
    conv_bank, recurrent_bank = seed(committed_conv, committed_recurrent, 5)
    assert conv_bank.shape == (2, 5, 3, 4)
    assert recurrent_bank.shape == (2, 5, 2, 3, 4)
    assert recurrent_bank.dtype is torch.float32
    for slot in range(5):
        torch.testing.assert_close(conv_bank[:, slot], committed_conv)
        torch.testing.assert_close(
            recurrent_bank[:, slot], committed_recurrent.float()
        )

    # Give every previous slot a distinct identity so selection mistakes are
    # visible, then compare the vectorized convolution with a token loop for
    # the MTP sizes used by the device operator.
    for tokens in (2, 5, 16):
        conv_bank = torch.randn(2, tokens, 3, 4, dtype=torch.float32)
        recurrent_bank = torch.randn(2, tokens, 2, 3, 4, dtype=torch.float32)
        hidden = torch.randn(2, 3, tokens, dtype=torch.float32)
        weight = torch.randn(3, 4, dtype=torch.float32)
        bias = torch.randn(3, dtype=torch.float32)
        accepted = torch.tensor([0, tokens - 1], dtype=torch.int8)
        output, next_bank = conv(
            hidden,
            conv_bank,
            weight,
            bias,
            accepted,
            "silu",
        )
        expected_output, expected_bank = sequential_reference(
            hidden,
            conv_bank,
            weight,
            bias,
            accepted,
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(next_bank, expected_bank, rtol=0, atol=0)

    rebased_conv, rebased_recurrent = rebase(
        next_bank,
        recurrent_bank,
        accepted,
        2,
    )
    assert rebased_conv.shape == (2, 2, 3, 4)
    assert rebased_recurrent.shape == (2, 2, 2, 3, 4)
    selected_slots = accepted.to(torch.long)
    for slot in range(2):
        torch.testing.assert_close(
            rebased_conv[:, slot], expected_bank[[0, 1], selected_slots]
        )
        torch.testing.assert_close(
            rebased_recurrent[:, slot], recurrent_bank[[0, 1], selected_slots]
        )

    try:
        conv(
            hidden,
            conv_bank,
            weight,
            bias,
            torch.tensor([0, 16], dtype=torch.int8),
            "silu",
        )
    except ValueError as error:
        assert "values must be" in str(error)
    else:
        raise AssertionError("out-of-range accepted_tokens was not rejected")

    print("PASS: state-bank seed/select/rebase and causal-conv rollback helpers")


if __name__ == "__main__":
    main()
