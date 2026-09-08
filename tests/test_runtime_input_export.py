from __future__ import annotations

import sys
import copy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "framework/python"))

from qwen35_dflash.ascend310p.runtime_input_export import (
    _normalize_public_nodes, _public_bindings, canonical_runtime_input_abi,
    validated_runtime_input_abi as _validated_runtime_input_abi,
)


class _Attr:
    def __init__(self, value: int) -> None:
        self.i = value


class _Op:
    def __init__(
        self,
        name: str,
        op_type: str,
        *,
        index: int | None = None,
        inputs: tuple[str, ...] = (),
    ) -> None:
        self.name = name
        self.type = op_type
        self.attr = {} if index is None else {"index": _Attr(index)}
        self.input = list(inputs)

    def Clear(self) -> None:
        self.name = ""
        self.type = ""
        self.attr = {}
        self.input = []
        self.output_desc = []

    def MergeFrom(self, other: "_Op") -> None:
        self.name = other.name
        self.type = other.type
        self.attr = dict(other.attr)
        self.input = list(other.input)
        self.output_desc = copy.deepcopy(getattr(other, "output_desc", []))


class _Graph:
    def __init__(self, ops: list[_Op]) -> None:
        self.op = ops

    @staticmethod
    def ByteSize() -> int:
        return 1


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


@pytest.mark.parametrize("serialized", [[1, 64, 4], [1, -1, 4], [1, 16, 4]])
@pytest.mark.parametrize("change_during_conversion", [False, True])
def test_static_export_audits_real_data_descriptors_and_restores_patch(
    monkeypatch, serialized, change_during_conversion,
):
    torchair = ModuleType("torchair")
    public = torch.zeros(1, 64, 4)
    node = _Op("feature", "Data", index=0)
    node.output_desc = [SimpleNamespace(shape=SimpleNamespace(
        dim=[1, 64, 4] if change_during_conversion else serialized,
    ))]
    graph = _Graph([node])

    def original(*args):
        if change_during_conversion:
            node.output_desc[0].shape.dim = serialized
        return False, 0

    module = SimpleNamespace(_convert_data_to_const=original)
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", module)
    if serialized == [1, 64, 4]:
        with canonical_runtime_input_abi(
            torchair, public_inputs=[public], public_names=["features"], require_static_shapes=True,
        ) as audit:
            module._convert_data_to_const([public], graph, "unused", {})
        assert audit["bindings"][0]["serialized_shape"] == serialized
    else:
        with pytest.raises(RuntimeError, match="static AIR input"):
            with canonical_runtime_input_abi(
                torchair, public_inputs=[public], public_names=["features"], require_static_shapes=True,
            ):
                module._convert_data_to_const([public], graph, "unused", {})
    assert module._convert_data_to_const is original


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


@pytest.mark.parametrize("field", ["dtype", "example_shape", "serialized_shape"])
def test_static_audit_must_match_chunk_tensor_contract(field):
    tensor = {"name": "start_position", "dtype": "int64", "shape": [1]}
    binding = {"index": 0, "logical_name": tensor["name"], "data_node_name": "arg7",
               "dtype": tensor["dtype"], "example_shape": [1], "serialized_shape": [1]}
    graph = {"input_names": [tensor["name"]], "metadata": {"tensor_abi": {"inputs": [tensor]}},
             "runtime_input_abi": {
                 "status": "PASS", "policy": "public-tensor-storage-identity-v1", "calls": 1,
                 "python_float_policy": "dynamo-specialize-float",
                 "logical_input_names": [tensor["name"]], "bindings": [binding],
             }}
    assert _validated_runtime_input_abi(graph, required=True)["status"] == "PASS"
    binding[field] = "int16" if field == "dtype" else [-1]
    with pytest.raises(ValueError, match="tensor descriptor"):
        _validated_runtime_input_abi(graph, required=True)


def test_all_chunk_exports_normalize_actual_dynamo_input_order(tmp_path, monkeypatch):
    """Real Dynamo capture + host serializer fixture; no AIR/device claim."""
    import json
    from test_incremental_air_om import specs
    from qwen35_dflash.ascend310p.exporter import export_air_bundle
    from qwen35_dflash.ascend310p.runtime_input_export import _tensor_identity

    monkeypatch.setenv("AI_RUN_DIR", str(tmp_path))
    torchair = ModuleType("torchair")
    values = specs()
    by_name = {spec.name: spec for spec in values}
    raw_orders = {}

    def original(inputs, graph, file_path, weight_name):
        del file_path
        for node in graph.op:
            if node.type == "Data" and id(inputs[node.attr["index"].i]) in weight_name:
                node.type = "Const"
        return False, len(weight_name)

    export_utils = SimpleNamespace(_convert_data_to_const=original)

    def dynamo_export(*public, model, export_path, export_name, **kwargs):
        spec = by_name[export_name]
        names = {_tensor_identity(t): n for n, t in zip(spec.input_names, public)}
        weights = {id(t): n for n, t in (*model.named_parameters(), *model.named_buffers())}

        def backend(fx, example_args):
            def run(*actual):
                raw_orders[export_name] = [names[_tensor_identity(t)] for t in actual
                                          if _tensor_identity(t) in names]
                nodes = []
                for index, tensor in enumerate(actual):
                    node = _Op(f"arg{index}", "Data", index=index)
                    node.output_desc = [SimpleNamespace(shape=SimpleNamespace(dim=list(tensor.shape)))]
                    nodes.append(node)
                # A consumer keeps its input edges when Data nodes move.
                consumer = _Op("consumer", "Add", inputs=tuple(n.name + ":0" for n in nodes))
                consumer_edges = tuple(consumer.input)
                graph = _Graph([*nodes, consumer])
                export_utils._convert_data_to_const(actual, graph, "unused", weights)
                assert tuple(graph.op[-1].input) == consumer_edges
                normalized = [n for n in graph.op if n.type == "Data"]
                assert [n.attr["index"].i for n in normalized] == list(range(len(public)))
                for index, node in enumerate(normalized):
                    captured = actual[int(node.name.removeprefix("arg"))]
                    assert _tensor_identity(captured) == _tensor_identity(public[index])
                Path(export_path, export_name + ".air").write_text(json.dumps({
                    "host_serializer_fixture": True,
                    "public_nodes": [n.name for n in normalized],
                }))
                return fx.forward(*actual)
            return run

        captured = torch.compile(model, backend=backend, dynamic=False, fullgraph=True)(*public)
        eager = model(*public)
        torch.testing.assert_close(captured, eager, rtol=0, atol=0)

    torchair.dynamo_export = dynamo_export
    monkeypatch.setitem(sys.modules, "torchair", torchair)
    monkeypatch.setitem(sys.modules, "torchair._utils.export_utils", export_utils)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch._dynamo.reset()
    try:
        report = export_air_bundle(lambda config: values, {}, tmp_path / "bundle")
    finally:
        torch._dynamo.reset()
        torch.set_num_threads(previous_threads)
    assert raw_orders["draft"] != list(by_name["draft"].input_names)
    assert raw_orders["draft"][:3] == ["features", "valid_rows", "start_position"]
    for graph in report["graphs"]:
        audit = _validated_runtime_input_abi(graph, required=True)
        assert audit["status"] == "PASS" and audit["calls"] == 1
        assert [b["logical_name"] for b in audit["bindings"]] == graph["input_names"]
    assert export_utils._convert_data_to_const is original
