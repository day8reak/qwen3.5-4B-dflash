from __future__ import annotations

import sys
import copy
from types import ModuleType, SimpleNamespace

import pytest
import torch

from qwen35_dflash.ascend310p.runtime_input_export import (
    _normalize_public_nodes, _public_bindings, canonical_runtime_input_abi,
)
from test_torchair_compat import _Graph, _Op
from qwen35_dflash.ascend310p.compiler import _validated_runtime_input_abi


def test_public_input_order_comes_from_identity_not_placeholder_order_or_shape():
    first = torch.zeros(1, dtype=torch.int32)
    second = torch.zeros(1, dtype=torch.int32)
    graph = _Graph([
        _Op("arg9", "Data", index=0), _Op("helper", "Gather"),
        _Op("arg3", "Data", index=1),
    ])
    bindings = _public_bindings([second, first], graph, {}, [first, second], ["a", "b"])
    assert [(i, n.name) for i, n in bindings] == [(0, "arg3"), (1, "arg9")]
    _normalize_public_nodes(graph, bindings)
    assert [op.name for op in graph.op] == ["arg3", "helper", "arg9"]
    assert [op.attr["index"].i for op in graph.op if op.type == "Data"] == [0, 1]


def test_weight_slots_and_shape_helpers_are_not_public_inputs():
    weight = torch.ones(4)
    public = torch.zeros(2)
    graph = _Graph([
        _Op("helper", "Gather"), _Op("weight", "Data", index=0),
        _Op("public", "Data", index=1),
    ])
    bindings = _public_bindings([weight, public], graph, {id(weight): "w"}, [public], ["x"])
    graph.op[1].type = "FileConstant"
    _normalize_public_nodes(graph, bindings)
    assert graph.op[0].type == "Gather"
    assert graph.op[1].type == "FileConstant"
    assert graph.op[2].attr["index"].i == 0


def test_remaining_scalar_is_rejected_without_guessing_its_value():
    public = torch.zeros(2)
    lifted = torch.tensor(0.125, dtype=torch.float64)
    graph = _Graph([_Op("arg1", "Data", index=0), _Op("arg7", "Data", index=1)])
    with pytest.raises(RuntimeError, match="unbound AIR runtime input.*arg7.*float64"):
        _public_bindings([public, lifted], graph, {}, [public], ["x"])


def test_missing_and_aliased_public_inputs_are_rejected():
    public = torch.zeros(2)
    with pytest.raises(RuntimeError, match="disappeared"):
        _public_bindings([], _Graph([]), {}, [public], ["x"])
    with pytest.raises(RuntimeError, match="distinct"):
        _public_bindings([], _Graph([]), {}, [public, public], ["x", "y"])


@pytest.mark.parametrize("index", [-1, 1])
def test_invalid_data_index_cannot_alias_a_public_tensor(index):
    public = torch.zeros(2)
    graph = _Graph([_Op("public", "Data", index=index)])
    with pytest.raises(RuntimeError, match="outside runtime inputs"):
        _public_bindings([public], graph, {}, [public], ["x"])


@pytest.mark.parametrize("fail", [False, True])
def test_normalization_context_restores_converter_and_float_policy(monkeypatch, fail):
    torchair = ModuleType("torchair")
    public = [torch.zeros(1), torch.ones(2)]
    graph = _Graph([_Op("second", "Data", index=0), _Op("first", "Data", index=1)])

    def original(inputs, export_graph, file_path, weight_name):
        if fail:
            raise RuntimeError("conversion failure")
        return False, 0

    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    with torch._dynamo.config.patch(specialize_float=False):
        try:
            with canonical_runtime_input_abi(
                torchair, public_inputs=public, public_names=["a", "b"],
            ) as audit:
                assert torch._dynamo.config.specialize_float is True
                module._convert_data_to_const(public[::-1], graph, "unused", {})
        except RuntimeError as error:
            assert fail and str(error) == "conversion failure"
        assert torch._dynamo.config.specialize_float is False
    assert module._convert_data_to_const is original
    if not fail:
        assert audit["status"] == "PASS"
        assert [r["logical_name"] for r in audit["bindings"]] == ["a", "b"]


def test_dynamo_specializes_model_float_without_freezing_dynamic_tensor_rows():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 0.125

        def forward(self, x):
            return x * self.scale + x.shape[0]

    examples = []

    def backend(graph, args):
        examples.append(args)
        return graph.forward

    model = Model()
    torch._dynamo.reset()
    try:
        with torch._dynamo.config.patch(specialize_float=True):
            compiled = torch.compile(model, backend=backend, dynamic=True, fullgraph=True)
            for rows in (3, 7, 16, 64):
                x = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4)
                torch.testing.assert_close(compiled(x), model(x), rtol=0, atol=0)
        assert examples
        for args in examples:
            assert not any(isinstance(arg, (float, torch.SymFloat)) for arg in args)
            assert not any(isinstance(arg, torch.Tensor) and arg.ndim == 0
                           and arg.dtype == torch.float64 for arg in args)
        assert any(isinstance(arg, torch.SymInt) for args in examples for arg in args)
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("damage", [None, "missing", "scalar", "order", "duplicate", "calls"])
def test_compiler_checks_canonical_input_audit(damage):
    graph = {"input_names": ["x", "state"], "runtime_input_abi": {
        "policy": "public-tensor-storage-identity-v1", "status": "PASS", "calls": 1,
        "python_float_policy": "dynamo-specialize-float",
        "logical_input_names": ["x", "state"],
        "bindings": [
            {"index": 0, "logical_name": "x", "data_node_name": "arg4"},
            {"index": 1, "logical_name": "state", "data_node_name": "arg1"},
        ],
    }}
    graph = copy.deepcopy(graph)
    if damage == "missing":
        del graph["runtime_input_abi"]
    elif damage == "scalar":
        graph["runtime_input_abi"]["bindings"].append({"name": "arg7"})
    elif damage == "order":
        graph["runtime_input_abi"]["bindings"].reverse()
    elif damage == "duplicate":
        graph["runtime_input_abi"]["bindings"][1]["data_node_name"] = "arg4"
    elif damage == "calls":
        graph["runtime_input_abi"]["calls"] = 0
    if damage is None:
        assert _validated_runtime_input_abi(graph, required=True)["status"] == "PASS"
    else:
        with pytest.raises(ValueError, match="runtime_input_abi"):
            _validated_runtime_input_abi(graph, required=True)
