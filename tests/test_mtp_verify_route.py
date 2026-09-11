"""Host-only MTP route semantics, export ABI and artifact selection gates."""
import copy
import json

import pytest
import torch
from torch import nn

from test_incremental_air_om import specs, manifest_graphs, chunk_bundle, small_threads
from rms_norm_test_support import adn_rms_norm_cpu
from qwen35_dflash.ascend310p.incremental_plan import (
    ABI, MTP_ABI, require_verify_gdr, validate_incremental_bundle, write_incremental_plan,
)

pytestmark = pytest.mark.usefixtures("adn_rms_norm_cpu", "small_threads")


class ForcedHead(nn.Module):
    """Force a rejection boundary; recurrence and cache computations stay real."""
    def __init__(self, accepted):
        super().__init__()
        self.accepted = accepted

    def forward(self, x):
        ids = torch.where(torch.arange(x.shape[1], device=x.device) < self.accepted, 4, 5)
        return torch.nn.functional.one_hot(ids, 64).to(x.dtype).unsqueeze(0)


@pytest.mark.parametrize("valid,accepted", [(1, 0), (2, 0), (2, 1), (4, 1), (4, 3),
                                           (16, 0), (16, 1), (16, 14), (16, 15)])
@pytest.mark.parametrize("start", [0, 63])
def test_mtp_gathers_exact_prefix_and_survives_rejection(valid, accepted, start):
    graph = next(s for s in specs(verify_gdr="mtp") if s.name == "target_verify")
    model = graph.model
    model.head = ForcedHead(accepted)
    state = [t.clone() for t in graph.example_args[3:]]
    state[1].normal_(0, 0.02)  # Retain information not representable in FP16.
    ids = torch.full((1, 16), 4, dtype=torch.long)
    with torch.inference_mode():
        actual = model(ids, torch.tensor([start]), torch.tensor([valid], dtype=torch.int16), *state)
        assert actual[1].item() == accepted
        assert actual[4].dtype == torch.float32
        assert not torch.equal(actual[4], actual[4].half().float())
        assert len(actual) == 7  # No discard bank crosses the OM boundary.
        reference = copy.deepcopy(model)
        reference.verify = False
        reference.rows = accepted + 1
        expected = reference(ids[:, :accepted+1], torch.tensor([start]),
                             torch.tensor([accepted+1], dtype=torch.int16), *state)
        # Compact MTP oracle processes only committed input rows; no rejected
        # future token can enter recurrent or convolution state.
        torch.testing.assert_close(actual[3], expected[2], rtol=2e-3, atol=2e-3)
        torch.testing.assert_close(actual[4], expected[3], rtol=2e-3, atol=2e-4)
        for a, b in zip(actual[5:], expected[4:]):
            a = a.permute(0, 2, 1, 3).flatten(0, 1)
            b = b.permute(0, 2, 1, 3).flatten(0, 1)
            torch.testing.assert_close(a[:start+accepted+1], b[:start+accepted+1], rtol=2e-3, atol=2e-3)
        changed = ids.clone()
        changed[:, accepted+1:] = 61
        tail = model(changed, torch.tensor([start]), torch.tensor([valid], dtype=torch.int16), *state)
        torch.testing.assert_close(tail[3], actual[3], rtol=0, atol=0)
        torch.testing.assert_close(tail[4], actual[4], rtol=0, atol=0)
        # Commit only selected scalar states, then change K and run another
        # round across a page boundary when applicable.
        model.head = ForcedHead(1)
        next_start = torch.tensor([start+accepted+1])
        next_valid = torch.tensor([4], dtype=torch.int16)
        next_actual = model(ids, next_start, next_valid, *actual[3:])
        next_expected = model(ids, next_start, next_valid, *expected[2:])
        assert next_actual[1].item() == next_expected[1].item() == 1
        for a, b in zip(next_actual[3:5], next_expected[3:5]):
            torch.testing.assert_close(a, b, rtol=3e-3, atol=2e-3)


def test_routes_are_explicit_and_mtp_requires_fp32_state():
    for route, abi in (("chunk", ABI), ("mtp", MTP_ABI)):
        graphs = manifest_graphs(specs(verify_gdr=route))
        c = validate_incremental_bundle(graphs)
        assert c["abi"] == abi and require_verify_gdr(c, route) == route
        with pytest.raises(ValueError, match="requested verify_gdr"):
            require_verify_gdr(c, "mtp" if route == "chunk" else "chunk")
        if route == "mtp":
            c["target_states"][1]["dtype"] = "float16"
            with pytest.raises(ValueError, match="FP32"):
                validate_incremental_bundle(graphs)


def test_mtp_is_not_a_silent_fallback():
    from qwen35_dflash.ascend310p.incremental import incremental_graph_specs
    from test_incremental_air_om import TinyTarget, draft_model, gdr, attention_op, rotary
    with pytest.raises(ValueError, match="requires GDR MTP"):
        incremental_graph_specs(TinyTarget(), draft_model(), capacity=128, metadata={},
            gdr=gdr, attention=attention_op, rotary=rotary, verify_gdr="mtp")


def test_mtp_bundle_preserves_ordinary_fp16_rounding():
    chunk = {s.name: s for s in specs()}
    mtp = {s.name: s for s in specs(verify_gdr="mtp")}
    c, m = chunk["target_prefill"], mtp["target_prefill"]
    with torch.inference_mode():
        co, mo = c.model(*c.example_args), m.model(*m.example_args)
        for a, b in zip(co, mo):
            torch.testing.assert_close(a, b.to(a.dtype), rtol=0, atol=0)
        cs, ms = co[2:], mo[2:]
        for pos in range(1, 10):
            args = (torch.tensor([[pos]]), torch.tensor([pos]), torch.ones(1, dtype=torch.int16))
            co = chunk["target_decode"].model(*args, *cs)
            mo = mtp["target_decode"].model(*args, *ms)
            for a, b in zip(co, mo):
                torch.testing.assert_close(a, b.to(a.dtype), rtol=0, atol=0)
            cs, ms = co[1:], mo[1:]


@pytest.mark.parametrize("damage", [None, "wrong-output", "missing-accepted", "duplicate"])
def test_mtp_ge_prototype_is_the_receiver_named_abi(tmp_path, damage):
    from qwen35_dflash.ascend310p.custom_op_export import validate_gdr_mtp_ge_prototype_environment
    body = """
REG_OP(GatedDeltaRuleMTP)
.INPUT(query, TensorType({DT_FLOAT16}))
.INPUT(key, TensorType({DT_FLOAT16}))
.INPUT(value, TensorType({DT_FLOAT16}))
.INPUT(g, TensorType({DT_FLOAT}))
.INPUT(beta, TensorType({DT_FLOAT16}))
.INPUT(initial_state, TensorType({DT_FLOAT}))
.INPUT(accepted_tokens, TensorType({DT_INT8}))
.OUTPUT(core_attn, TensorType({DT_FLOAT16}))
.OUTPUT(last_recurrent_state, TensorType({DT_FLOAT}))
.ATTR(chunk_size, Int, 64)
.ATTR(output_final_state, Bool, false)
.ATTR(use_qk_l2norm_in_kernel, Bool, false)
.OP_END_FACTORY_REG(GatedDeltaRuleMTP)
"""
    if damage == "wrong-output":
        body = body.replace("last_recurrent_state", "state_bank")
    elif damage == "missing-accepted":
        body = body.replace(".INPUT(accepted_tokens, TensorType({DT_INT8}))", "")
    header = tmp_path / "one/op_proto/inc/mtp.h"
    header.parent.mkdir(parents=True)
    header.write_text(body)
    roots = str(tmp_path / "one")
    if damage == "duplicate":
        second = tmp_path / "two/op_proto/inc/mtp.h"
        second.parent.mkdir(parents=True)
        second.write_text(body)
        roots += ":" + str(tmp_path / "two")
    if damage:
        with pytest.raises(RuntimeError, match="incompatible|multiple"):
            validate_gdr_mtp_ge_prototype_environment(ascend_custom_opp_path=roots, ld_library_path="")
    else:
        report = validate_gdr_mtp_ge_prototype_environment(ascend_custom_opp_path=roots, ld_library_path="")
        assert report["status"] == "PASS" and report["abi"] == "receiver-gdr-mtp-v1-named-io"


@pytest.mark.parametrize("chunk_bundle", ["chunk", "mtp"], indirect=True)
def test_plan_locks_selected_route_before_writing(chunk_bundle, tmp_path):
    contract = json.loads(chunk_bundle.read_text())["graphs"][0]["metadata"]["incremental_contract"]
    route = require_verify_gdr(contract)
    wrong = "chunk" if route == "mtp" else "mtp"
    with pytest.raises(ValueError, match="requested verify_gdr"):
        write_incremental_plan(chunk_bundle, tmp_path / "wrong.txt", verify_gdr=wrong)
    assert not (tmp_path / "wrong.txt").exists()
    plan, _, _ = write_incremental_plan(chunk_bundle, tmp_path / "right.txt", verify_gdr=route)
    assert plan.read_text().splitlines()[0] == contract["abi"]


def test_cli_route_overrides_factory_config(tmp_path):
    from qwen35_dflash.ascend310p.cli import build_parser, _factory_config
    config = tmp_path / "config.json"
    config.write_text('{"verify_gdr":"chunk"}')
    args = build_parser().parse_args(["export-air", "--factory", "unused:factory",
        "--factory-config", str(config), "--bundle-dir", str(tmp_path / "out"), "--verify-gdr", "mtp"])
    assert _factory_config(args)["verify_gdr"] == "mtp"
    assert json.loads(config.read_text())["verify_gdr"] == "chunk"
