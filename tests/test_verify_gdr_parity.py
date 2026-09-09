"""CPU semantic/capture regressions, not real native GDR or OM evidence."""
from __future__ import annotations

import __future__
import ast
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from test_dflash_rollback_helpers import SOURCE, load_helpers
from test_incremental_om_graphs import _FakeTarget
from qwen35_dflash.ascend310p.contracts import CustomOpExportSpec
from qwen35_dflash.ascend310p.incremental_graphs import (
    DECODE1_GDR_POLICY, MTP_GDR_POLICY, TargetStepStateGraph,
    TargetVerifyCommitStateGraph, resolve_verify_gdr_policy,
)


def native_step_fixture(query, key, value, *, g, beta, effective_length,
                        chunk_size, initial_state, output_final_state,
                        use_qk_l2norm_in_kernel):
    """Small explicit GDR equation used only as a deterministic CPU fixture."""
    assert query.shape[1] == key.shape[1] == value.shape[1] == 1
    assert chunk_size == 1 and output_final_state and use_qk_l2norm_in_kernel
    assert effective_length.dtype == torch.int16
    assert initial_state.dtype == torch.float32
    q = torch.nn.functional.normalize(query[:, 0].float(), dim=-1)
    k = torch.nn.functional.normalize(key[:, 0].float(), dim=-1)
    state = initial_state * g[:, 0].float().exp()[..., None, None]
    delta = value[:, 0].float() - (state * k[..., None]).sum(dim=-2)
    next_state = state + k[..., None] * delta[..., None, :] * beta[:, 0].float()[..., None, None]
    output = (next_state * q[..., None]).sum(dim=-2).unsqueeze(1).to(query.dtype)
    return output, next_state


def _namespace():
    namespace = load_helpers()
    namespace["torch_npu"] = SimpleNamespace(npu_chunk_gated_delta_rule=native_step_fixture)
    # Test the production GDN forward, without importing the hardware-only
    # Transformers/NPU stack or substituting a production kernel.
    tree = ast.parse(SOURCE.read_text("utf-8"))
    gdn = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "Qwen3_5GatedDeltaNet")
    forward = next(n for n in gdn.body if isinstance(n, ast.FunctionDef) and n.name == "forward")
    exec(compile(ast.Module(body=[forward], type_ignores=[]), str(SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def _inputs(rows, batch=1):
    generator = torch.Generator().manual_seed(461)
    query = torch.randn(batch, rows, 2, 4, generator=generator).half()
    key = torch.randn(batch, rows, 2, 4, generator=generator).half()
    value = torch.randn(batch, rows, 2, 3, generator=generator).half()
    g = -torch.rand(batch, rows, 2, generator=generator)
    beta = torch.rand(batch, rows, 2, generator=generator).half()
    # Deliberately not on the FP16 grid: the entry must not be rounded before
    # its first step, and later steps must receive the rounded predecessor.
    state = torch.randn(batch, 2, 4, 3, generator=generator) * 0.731
    return query, key, value, g, beta, state


def _sequential_reference(args, *, round_feedback=True):
    query, key, value, g, beta, state = args
    outputs, states = [], []
    for row in range(query.shape[1]):
        out, state = native_step_fixture(
            query[:, row:row + 1], key[:, row:row + 1], value[:, row:row + 1],
            g=g[:, row:row + 1], beta=beta[:, row:row + 1],
            effective_length=torch.ones(query.shape[0], dtype=torch.int16),
            chunk_size=1, initial_state=state.float(),
            output_final_state=True, use_qk_l2norm_in_kernel=True,
        )
        if round_feedback:
            state = state.half().float()
        outputs.append(out)
        states.append(state)
    return torch.cat(outputs, 1), torch.stack(states, 1)


@pytest.mark.parametrize("rows", [1, 2, 6, 16])
@pytest.mark.parametrize("batch", [1, 2])
def test_bank_matches_every_decode1_step_and_does_not_mutate_inputs(rows, batch):
    args = _inputs(rows, batch)
    before = tuple(x.clone() for x in args)
    out, bank = _namespace()["_npu_decode1_gated_delta_rule_bank"](*args)
    expected_out, expected_bank = _sequential_reference(args)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(bank, expected_bank, rtol=0, atol=0)
    assert bank.dtype == torch.float32
    torch.testing.assert_close(bank, bank.half().float(), rtol=0, atol=0)
    for actual, expected in zip(args, before):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_final_cast_only_does_not_reproduce_feedback_or_later_outputs():
    args = _inputs(16)
    out, bank = _namespace()["_npu_decode1_gated_delta_rule_bank"](*args)
    uninterrupted_out, uninterrupted_bank = _sequential_reference(args, round_feedback=False)
    torch.testing.assert_close(out[:, :1], uninterrupted_out[:, :1], rtol=0, atol=0)
    assert torch.any(out[:, 1:] != uninterrupted_out[:, 1:])
    assert torch.any(bank[:, 1:] != uninterrupted_bank[:, 1:].half().float())


@pytest.mark.parametrize("accepted", [0, 1, 5, 15])
def test_reject_and_continue_uses_selected_rounded_state(accepted):
    namespace = _namespace()
    first = _inputs(16, 2)
    _, bank = namespace["_npu_decode1_gated_delta_rule_bank"](*first)
    selectors = torch.tensor([accepted, 15 - accepted], dtype=torch.int8)
    committed = namespace["_select_dflash_state_slot"](bank, selectors)
    second = (*_inputs(6, 2)[:5], committed)
    out, next_bank = namespace["_npu_decode1_gated_delta_rule_bank"](*second)
    expected_out, expected_bank = _sequential_reference(second)
    torch.testing.assert_close(out, expected_out, rtol=0, atol=0)
    torch.testing.assert_close(next_bank, expected_bank, rtol=0, atol=0)


def test_fixed_export_preserves_per_token_casts_and_no_symbolic_shapes():
    helper = _namespace()["_npu_decode1_gated_delta_rule_bank"]

    class Graph(nn.Module):
        def forward(self, q, k, v, g, beta, state):
            return helper(q, k, v, g, beta, state)

    args = _inputs(16)
    exported = torch.export.export(Graph(), args, strict=True)
    assert not exported.range_constraints
    for actual, expected in zip(exported.module()(*args), _sequential_reference(args)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    # Each returned FP32 state has its own FP16 producer, including the state
    # fed into the next recurrence. A final-bank-only cast fails this check.
    fp16_casts = [n for n in exported.graph.nodes if "to" in str(n.target)
                  and (torch.float16 in n.args or torch.float16 in n.kwargs.values())]
    assert len(fp16_casts) >= 16


class _Gate(nn.Module):
    def forward(self, x, z):
        return x * torch.sigmoid(z)


def _gdn():
    namespace = _namespace()

    class Gdn(nn.Module):
        forward = namespace["forward"]

        def __init__(self):
            super().__init__()
            self.num_v_heads = self.num_k_heads = 1
            self.head_k_dim = self.head_v_dim = self.key_dim = self.value_dim = 2
            self.conv_dim, self.conv_kernel_size, self.activation = 6, 2, "silu"
            self.in_proj_qkv = nn.Linear(4, 6, bias=False, dtype=torch.float16)
            self.in_proj_z = nn.Linear(4, 2, bias=False, dtype=torch.float16)
            self.in_proj_a = nn.Linear(4, 1, bias=False, dtype=torch.float16)
            self.in_proj_b = nn.Linear(4, 1, bias=False, dtype=torch.float16)
            self.conv1d = nn.Conv1d(6, 6, 2, groups=6, bias=False, dtype=torch.float16)
            self.A_log = nn.Parameter(torch.zeros(1))
            self.dt_bias = nn.Parameter(torch.zeros(1, dtype=torch.float16))
            self.norm = _Gate()
            self.out_proj = nn.Linear(2, 4, bias=False, dtype=torch.float16)
            self.causal_conv1d_update = namespace["torch_causal_conv1d_update"]

    with torch.random.fork_rng():
        torch.manual_seed(42)
        return Gdn().eval(), namespace


@pytest.mark.parametrize("rows", [1, 2, 6, 16])
@pytest.mark.parametrize("banked", [False, True])
def test_real_gdn_forward_selects_new_path_and_matches_ordinary_state(rows, banked):
    model, namespace = _gdn()
    x = torch.linspace(-1, 1, 2 * rows * 4).reshape(2, rows, 4).half()
    conv = torch.full((2, 6, 2), 0.2).half()
    state = torch.full((2, 1, 2, 2), 0.0012345)
    accepted = torch.zeros(2, dtype=torch.int8)
    initial = (conv.clone(), state.clone())
    if banked:
        initial = namespace["seed_dflash_gdn_state_banks"](*initial, rows)
        if rows > 1:
            # Select different previous slots per batch, not an assumed zero.
            initial[0][:, 0] += 1
            initial[1][:, 0] += 1
            accepted = torch.tensor([rows - 1, 0], dtype=torch.int8)
            conv = namespace["_select_dflash_state_slot"](initial[0], accepted)
            state = namespace["_select_dflash_state_slot"](initial[1], accepted)
    out, (conv_bank, bank) = model(
        x, cache_params=initial, accepted_tokens=accepted,
        dflash_gdr_verify_policy=DECODE1_GDR_POLICY,
    )
    expected_out, expected_conv, expected_states = [], [], []
    for row in range(rows):
        y, (conv, state) = model(
            x[:, row:row + 1], cache_params=(conv, state.float()),
            gdr_effective_length=torch.ones(2, dtype=torch.int16),
        )
        expected_out.append(y)
        expected_conv.append(conv.clone())
        expected_states.append(state.float())
    torch.testing.assert_close(out, torch.cat(expected_out, 1), rtol=0, atol=0)
    torch.testing.assert_close(conv_bank, torch.stack(expected_conv, 1), rtol=0, atol=0)
    torch.testing.assert_close(bank, torch.stack(expected_states, 1), rtol=0, atol=0)


def test_stale_receiver_and_dynamic_policy_fail_closed():
    target = _FakeTarget()
    target.dflash_execution_model.language_model.dflash_gdr_verify_policies = ()
    with pytest.raises(RuntimeError, match="update the loaded modeling"):
        TargetVerifyCommitStateGraph(target, kv_cache_max_len=64,
                                     gdr_verify_policy=DECODE1_GDR_POLICY)
    with pytest.raises(ValueError, match="fixed Verify"):
        TargetStepStateGraph(_FakeTarget(), kv_cache_max_len=64,
                             gdr_verify_policy=DECODE1_GDR_POLICY)
    with pytest.raises(ValueError, match="target_verify_gdr_policy"):
        resolve_verify_gdr_policy("typo", merged_prefill=True, unified_target_step=False)


def test_default_gdn_still_calls_mtp_and_preserves_its_fp32_bank():
    model, namespace = _gdn()
    calls = []

    def mtp_fixture(q, k, v, g, beta, state, accepted):
        calls.append((q.shape[1], accepted.clone()))
        return torch.zeros_like(v), state + 0.00012345

    def forbidden_single_step(*args, **kwargs):
        raise AssertionError("default MTP must not select the single-step reference")

    namespace["_npu_gated_delta_rule_mtp"] = mtp_fixture
    namespace["torch_npu"].npu_chunk_gated_delta_rule = forbidden_single_step
    state = torch.full((2, 1, 2, 2), 0.0034567)
    _, (_, bank) = model(
        torch.ones(2, 6, 4, dtype=torch.float16),
        cache_params=(torch.zeros(2, 6, 2, dtype=torch.float16), state),
        accepted_tokens=torch.zeros(2, dtype=torch.int8),
    )
    assert len(calls) == 1 and calls[0][0] == 6
    torch.testing.assert_close(
        bank, state.unsqueeze(1).expand_as(bank) + 0.00012345, rtol=0, atol=0,
    )
    assert torch.any(bank != bank.half().float())


def test_opt_in_static_spec_preserves_abi_and_requires_all_single_step_gdr_nodes():
    from test_incremental_om_graphs import _FakeDraft
    from qwen35_dflash.ascend310p.incremental_graphs import incremental_state_graph_specs
    specs = incremental_state_graph_specs(
        _FakeTarget(), _FakeDraft(), kv_cache_max_len=64,
        device="cpu", dtype=torch.float16, eos_table_width=4,
        ordinary_custom_ops=(), head_custom_ops=(),
        verify_custom_ops=(CustomOpExportSpec(
            torch_op="npu::npu_gated_delta_rule_mtp", ge_op_type="GatedDeltaRuleMTP",
        ),),
        merged_prefill=True, draft_static_feature_rows=64,
        target_verify_gdr_policy=DECODE1_GDR_POLICY,
    )
    verify = specs[-1]
    assert verify.metadata["target_verify_gdr_policy"] == DECODE1_GDR_POLICY
    assert verify.metadata["verify_scalar_state_seed_policy"] == "committed-scalar-no-input-bank-v1"
    assert len(verify.input_names) == 9 and len(verify.output_names) == 13
    op, = verify.custom_ops
    assert op.torch_op == "npu::npu_chunk_gated_delta_rule"
    assert op.ge_op_type == "ChunkGatedDeltaRule" and op.minimum_occurrences == 16
    assert all(not s.dynamic for s in specs)
    assert resolve_verify_gdr_policy(None, merged_prefill=True,
                                     unified_target_step=False) == MTP_GDR_POLICY
    assert resolve_verify_gdr_policy(None, merged_prefill=False,
                                     unified_target_step=True) == MTP_GDR_POLICY
