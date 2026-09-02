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
BASE_SOURCE = REPOSITORY_ROOT / "models" / "modeling_qwen3_5_hiai_nd.py"
HELPERS = {
    "_normalize_gdr_effective_length",
    "_require_dflash_accepted_tokens",
    "_select_dflash_state_slot",
    "seed_dflash_gdn_state_banks",
    "seed_dflash_recurrent_state_bank",
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


def assert_gdr_effective_length_source_contract() -> None:
    for source in (BASE_SOURCE, SOURCE):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        ordinary_calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "npu_chunk_gated_delta_rule"
        ]
        assert len(ordinary_calls) == 1
        keyword_names = {item.arg for item in ordinary_calls[0].keywords}
        assert "effective_length" in keyword_names

    rollback_tree = ast.parse(
        SOURCE.read_text(encoding="utf-8"),
        filename=str(SOURCE),
    )
    mtp_bridge = next(
        node
        for node in rollback_tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "_npu_gated_delta_rule_mtp"
    )
    assert [argument.arg for argument in mtp_bridge.args.args] == [
        "query",
        "key",
        "value",
        "g",
        "beta",
        "initial_state",
        "accepted_tokens",
    ]

    gdn_class = next(
        node
        for node in rollback_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5GatedDeltaNet"
    )
    gdn_forward = next(
        node
        for node in gdn_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "forward"
    )
    seed_calls = [
        node
        for node in ast.walk(gdn_forward)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "seed_dflash_recurrent_state_bank"
    ]
    assert len(seed_calls) == 1

    text_model = next(
        node
        for node in rollback_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5TextModel"
    )
    policies = {
        target.id: node.value.value
        for node in text_model.body
        if isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Constant)
        for target in node.targets
        if isinstance(target, ast.Name)
        and target.id
        in {"dflash_scalar_state_seed_policy", "dflash_cache_index_policy"}
    }
    assert policies == {
        "dflash_scalar_state_seed_policy": "per-linear-layer-jit-v1",
        "dflash_cache_index_policy": "once-per-verify-v1",
    }

    attention_class = next(
        node
        for node in rollback_tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Qwen3_5Attention"
    )
    update_dflash = next(
        node
        for node in attention_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "update_dflash"
    )
    assert [argument.arg for argument in update_dflash.args.kwonlyargs] == [
        "target_blocks",
        "offsets_in_block",
    ]
    token_loop = next(
        node for node in ast.walk(update_dflash) if isinstance(node, ast.For)
    )
    repeated_division = [
        node
        for node in ast.walk(token_loop)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.FloorDiv, ast.Mod))
    ]
    assert repeated_division == []


def main() -> None:
    assert_gdr_effective_length_source_contract()
    helper = load_helpers()
    assert helper["DFLASH_BLOCK_SIZE"] == 16
    assert helper["DFLASH_MAX_PROPOSALS"] == 15
    assert helper["DFLASH_MAX_VERIFY_TOKENS"] == 16
    seed = helper["seed_dflash_gdn_state_banks"]
    seed_recurrent = helper["seed_dflash_recurrent_state_bank"]
    rebase = helper["rebase_dflash_gdn_state_banks"]
    conv = helper["torch_dflash_causal_conv1d_mtp"]
    normalize_effective_length = helper["_normalize_gdr_effective_length"]

    recurrent_seed = seed_recurrent(
        torch.zeros(2, 3, 4, 5, dtype=torch.float16),
        7,
    )
    assert recurrent_seed.shape == (2, 7, 3, 4, 5)
    assert recurrent_seed.dtype is torch.float32

    default_effective_length = normalize_effective_length(
        None,
        batch_size=2,
        physical_sequence_length=5,
        device=torch.device("cpu"),
    )
    assert default_effective_length.dtype == torch.int16
    assert default_effective_length.tolist() == [5, 5]
    explicit_effective_length = torch.tensor([3, 4], dtype=torch.int16)
    assert normalize_effective_length(
        explicit_effective_length,
        batch_size=2,
        physical_sequence_length=5,
        device=torch.device("cpu"),
    ) is explicit_effective_length
    for invalid, expected_error in (
        (torch.tensor([3, 4], dtype=torch.int64), TypeError),
        (torch.tensor([[3, 4]], dtype=torch.int16), ValueError),
        (torch.tensor([0, 4], dtype=torch.int16), ValueError),
        (torch.tensor([3, 6], dtype=torch.int16), ValueError),
    ):
        try:
            normalize_effective_length(
                invalid,
                batch_size=2,
                physical_sequence_length=5,
                device=torch.device("cpu"),
            )
        except expected_error:
            pass
        else:
            raise AssertionError(
                "invalid GDR effective_length contract was not rejected"
            )

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
        scalar_conv = torch.stack(
            [
                conv_bank[index, int(accepted[index])]
                for index in range(conv_bank.shape[0])
            ]
        )
        scalar_output, scalar_next_bank = conv(
            hidden,
            scalar_conv,
            weight,
            bias,
            accepted,
            "silu",
        )
        torch.testing.assert_close(
            scalar_output,
            expected_output,
            rtol=1e-6,
            atol=1e-6,
        )
        torch.testing.assert_close(
            scalar_next_bank,
            expected_bank,
            rtol=0,
            atol=0,
        )

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
