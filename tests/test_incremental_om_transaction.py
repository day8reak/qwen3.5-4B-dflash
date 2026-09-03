from __future__ import annotations

import itertools

import pytest
import torch

from qwen35_dflash.ascend310p.incremental import ExactAcceptCommitStateGraph


def _inputs(
    proposals: list[int],
    target: list[int],
    *,
    logical_count: int | None = None,
    eos: tuple[int, ...] = (),
) -> tuple[torch.Tensor, ...]:
    width = len(proposals)
    slots = width + 1
    layers = 2
    conv = torch.arange(
        layers * slots * 4,
        dtype=torch.float32,
    ).reshape(layers, 1, slots, 2, 2)
    recurrent = torch.arange(
        layers * slots * 4,
        dtype=torch.float32,
    ).reshape(layers, 1, slots, 1, 2, 2)
    features = torch.arange(slots * 3, dtype=torch.float32).reshape(1, slots, 3)
    eos_table = torch.zeros(4, dtype=torch.long)
    if eos:
        eos_table[: len(eos)] = torch.tensor(eos, dtype=torch.long)
    return (
        torch.tensor([proposals], dtype=torch.long),
        torch.tensor([target], dtype=torch.long),
        torch.tensor(
            [width if logical_count is None else logical_count],
            dtype=torch.int32,
        ),
        eos_table,
        torch.tensor([len(eos)], dtype=torch.int32),
        conv,
        recurrent,
        features,
        torch.tensor([17], dtype=torch.long),
    )


def _reference(
    proposals: list[int],
    target: list[int],
    logical_count: int,
    eos: set[int],
) -> tuple[list[int], int, int, int, bool]:
    logical = proposals[:logical_count]
    for index, token in enumerate(logical):
        if token in eos:
            logical = logical[: index + 1]
            break
    accepted = 0
    while accepted < len(logical) and logical[accepted] == target[accepted]:
        accepted += 1
    committed = logical[:accepted]
    accepted_eos = bool(committed and committed[-1] in eos)
    if not accepted_eos:
        committed.append(target[accepted])
    return (
        committed,
        len(logical),
        accepted,
        len(logical) - accepted,
        bool(committed and committed[-1] in eos),
    )


@pytest.mark.parametrize(
    ("proposals", "target", "logical_count", "eos"),
    [
        ([11, 12, 13], [11, 12, 13, 14], 3, ()),
        ([11, 12, 13], [11, 99, 13, 14], 3, ()),
        ([11, 12, 13], [99, 12, 13, 14], 3, ()),
        ([11, 12, 13], [11, 12, 13, 14], 3, (12,)),
        ([11, 12, 13], [99, 12, 13, 14], 3, (12,)),
        ([11, 12, 13], [11, 90, 13, 14], 3, (90,)),
        ([11, 777, 778], [11, 12, 13, 14], 1, ()),
        ([11, 12, 13], [11, 12, 13, 14], 2, ()),
        ([11, 12, 13], [99, 12, 13, 14], 0, ()),
        ([11, 11, 13], [11, 11, 13, 14], 3, (11,)),
    ],
)
def test_exact_accept_commit_and_state_selection(
    proposals: list[int],
    target: list[int],
    logical_count: int,
    eos: tuple[int, ...],
) -> None:
    graph = ExactAcceptCommitStateGraph(len(proposals))
    args = _inputs(
        proposals,
        target,
        logical_count=logical_count,
        eos=eos,
    )
    outputs = graph(*args)
    (
        committed_ids,
        commit_count,
        drafted_count,
        accepted_count,
        rejected_count,
        selected_conv,
        selected_recurrent,
        committed_features,
        committed_input_count,
        next_cursor,
        finished,
    ) = outputs
    expected, drafted, accepted, rejected, expected_finished = _reference(
        proposals,
        target,
        logical_count,
        set(eos),
    )
    assert commit_count.tolist() == [len(expected)]
    assert committed_ids[0, : len(expected)].tolist() == expected
    assert torch.count_nonzero(committed_ids[0, len(expected) :]) == 0
    assert drafted_count.tolist() == [drafted]
    assert accepted_count.tolist() == [accepted]
    assert rejected_count.tolist() == [rejected]
    assert finished.tolist() == [expected_finished]
    assert committed_input_count.tolist() == [accepted + 1]
    assert next_cursor.tolist() == [18 + accepted]
    torch.testing.assert_close(selected_conv, args[5][:, :, accepted])
    torch.testing.assert_close(selected_recurrent, args[6][:, :, accepted])
    torch.testing.assert_close(
        committed_features[:, : accepted + 1],
        args[7][:, : accepted + 1],
    )
    assert torch.count_nonzero(committed_features[:, accepted + 1 :]) == 0


def test_exhaustive_mismatch_eos_and_logical_tail_contract() -> None:
    width = 4
    proposals = [31, 32, 33, 34]
    graph = ExactAcceptCommitStateGraph(width)
    for logical_count, mismatch, eos_index in itertools.product(
        range(1, width + 1),
        range(width + 1),
        range(-1, width),
    ):
        target = proposals + [35]
        if mismatch < width:
            target[mismatch] = 100 + mismatch
        eos = () if eos_index < 0 else (proposals[eos_index],)
        args = _inputs(
            proposals,
            target,
            logical_count=logical_count,
            eos=eos,
        )
        result = graph(*args)
        expected, drafted, accepted, rejected, finished = _reference(
            proposals,
            target,
            logical_count,
            set(eos),
        )
        assert result[0][0, : len(expected)].tolist() == expected
        assert result[1].tolist() == [len(expected)]
        assert result[2].tolist() == [drafted]
        assert result[3].tolist() == [accepted]
        assert result[4].tolist() == [rejected]
        assert result[10].tolist() == [finished]


def test_transaction_tail_is_torch_export_capture_safe() -> None:
    graph = ExactAcceptCommitStateGraph(3).eval()
    args = _inputs([11, 12, 13], [11, 99, 13, 14], eos=(99,))
    exported = torch.export.export(graph, args)
    eager = graph(*args)
    captured = exported.module()(*args)
    assert len(captured) == len(eager) == 11
    for actual, expected in zip(captured, eager):
        torch.testing.assert_close(actual, expected)

    exported_targets = {
        node.target
        for node in exported.graph_module.graph.nodes
        if node.op == "call_function"
    }
    assert torch.ops.aten.cumsum.default in exported_targets
    assert torch.ops.aten.amin.default not in exported_targets
    assert torch.ops.aten.min.dim not in exported_targets
    assert torch.ops.aten.cumprod.default not in exported_targets


def test_proposal_width_validation() -> None:
    with pytest.raises(ValueError, match="positive"):
        ExactAcceptCommitStateGraph(0)
    with pytest.raises(TypeError, match="integer"):
        ExactAcceptCommitStateGraph(True)
