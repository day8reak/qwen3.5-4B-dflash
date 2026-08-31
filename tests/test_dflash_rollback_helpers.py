#!/usr/bin/env python3
"""CPU checks for the pure tensor helpers in the HIAI rollback source."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Callable, Optional

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
    "torch_dflash_causal_conv1d_chunk",
    "select_dflash_chunk_commit_state",
    "run_dflash_chunk_gdr_commit",
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
        "Callable": Callable,
        "Optional": Optional,
    }
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), "exec"), namespace)
    missing = HELPERS - namespace.keys()
    if missing:
        raise AssertionError(f"missing helper definitions: {sorted(missing)}")
    return namespace


def sequential_reference(
    hidden: torch.Tensor,
    initial_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, tokens = hidden.shape
    state_len = initial_state.shape[-1]
    outputs = torch.empty_like(hidden)
    states = torch.empty(
        (batch, tokens, channels, state_len),
        dtype=hidden.dtype,
    )
    for batch_index in range(batch):
        state = initial_state[batch_index].clone()
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
        for call in ordinary_calls:
            keyword_names = {item.arg for item in call.keywords}
            assert "effective_length" in keyword_names
            assert "initial_state" in keyword_names
            assert "output_final_state" in keyword_names

    rollback_tree = ast.parse(SOURCE.read_text("utf-8"), filename=str(SOURCE))
    assert not any(
        isinstance(node, ast.Attribute)
        and node.attr == "npu_gated_delta_rule_mtp"
        for node in ast.walk(rollback_tree)
    )
    commit_method = next(
        node
        for node in ast.walk(rollback_tree)
        if isinstance(node, ast.FunctionDef)
        and node.name == "compute_dflash_chunk_commit"
    )
    assert any(
        isinstance(node, ast.Attribute)
        and node.attr == "npu_chunk_gated_delta_rule"
        for node in ast.walk(commit_method)
    )


def main() -> None:
    assert_gdr_effective_length_source_contract()
    helper = load_helpers()
    assert helper["DFLASH_BLOCK_SIZE"] == 16
    assert helper["DFLASH_MAX_PROPOSALS"] == 15
    assert helper["DFLASH_MAX_VERIFY_TOKENS"] == 16
    conv = helper["torch_dflash_causal_conv1d_chunk"]
    select_commit = helper["select_dflash_chunk_commit_state"]
    run_commit = helper["run_dflash_chunk_gdr_commit"]
    normalize_effective_length = helper["_normalize_gdr_effective_length"]

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
    # Compare vectorized chunk convolution and every possible committed prefix
    # against a sequential scalar-state reference.
    for tokens in (1, 2, 5, 16):
        initial_state = torch.randn(2, 3, 4, dtype=torch.float32)
        hidden = torch.randn(2, 3, tokens, dtype=torch.float32)
        weight = torch.randn(3, 4, dtype=torch.float32)
        bias = torch.randn(3, dtype=torch.float32)
        output, next_bank = conv(
            hidden,
            initial_state,
            weight,
            bias,
            "silu",
        )
        expected_output, expected_bank = sequential_reference(
            hidden,
            initial_state,
            weight,
            bias,
        )
        torch.testing.assert_close(output, expected_output, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(next_bank, expected_bank, rtol=0, atol=0)
        for committed_rows in {1, tokens}:
            torch.testing.assert_close(
                select_commit(next_bank, committed_rows),
                expected_bank[:, committed_rows - 1],
                rtol=0,
                atol=0,
            )

    try:
        select_commit(next_bank, tokens + 1)
    except ValueError as error:
        assert "committed_rows must be" in str(error)
    else:
        raise AssertionError("out-of-range committed_rows was not rejected")

    calls: list[dict[str, object]] = []

    def fake_chunk_gdr(query, key, value, **kwargs):
        del key, value
        calls.append(kwargs)
        initial = kwargs["initial_state"]
        effective = kwargs["effective_length"].to(torch.float32)
        return query, initial + effective.view(-1, 1, 1, 1)

    query = torch.randn(2, 5, 3, 4, dtype=torch.float16)
    key = torch.randn_like(query)
    value = torch.randn(2, 5, 3, 6, dtype=torch.float16)
    g = torch.randn(2, 5, 3, dtype=torch.float32)
    beta = torch.randn(2, 5, 3, dtype=torch.float16)
    initial = torch.zeros(2, 3, 4, 6, dtype=torch.float32)
    committed = run_commit(
        fake_chunk_gdr,
        query,
        key,
        value,
        g,
        beta,
        initial,
        3,
    )
    assert calls[0]["effective_length"].dtype == torch.int16
    assert calls[0]["effective_length"].tolist() == [3, 3]
    assert calls[0]["chunk_size"] == 64
    assert calls[0]["initial_state"] is initial
    assert calls[0]["output_final_state"] is True
    assert calls[0]["use_qk_l2norm_in_kernel"] is True
    torch.testing.assert_close(committed, torch.full_like(initial, 3.0))

    print("PASS: two-pass chunk GDR source and causal-conv commit helpers")


if __name__ == "__main__":
    main()
