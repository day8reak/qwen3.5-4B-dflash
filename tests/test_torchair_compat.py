from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.torchair_compat import (
    ExternalWeightMappingAudit,
    _convert_data_to_const_by_index,
    index_safe_external_weight_conversion,
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

    def MergeFrom(self, other: "_Op") -> None:
        self.name = other.name
        self.type = other.type
        self.attr = dict(other.attr)
        self.input = list(other.input)


class _Graph:
    def __init__(self, ops: list[_Op]) -> None:
        self.op = ops

    @staticmethod
    def ByteSize() -> int:
        return 1


class _Logger:
    @staticmethod
    def debug(message: str) -> None:
        del message


def _fake_export_utils() -> ModuleType:
    module = ModuleType("torchair._utils.export_utils")

    def original(inputs, export_graph, file_path, weight_name):
        del inputs, export_graph, file_path, weight_name
        return False, 0

    def is_weight_externalized(inputs, weight_name, export_graph):
        del export_graph
        return True, sum(id(value) in weight_name for value in inputs)

    def file_constant(*, shape, dtype, file_path, node_name):
        del shape, dtype, file_path
        return SimpleNamespace(node=_Op(node_name, "FileConstant"))

    def make_const_node(value, name):
        del value
        return SimpleNamespace(node=_Op(name, "Const"))

    def sort_graph_data_index(graph):
        index = 0
        for op in graph.op:
            if op.type == "Data":
                op.attr["index"].i = index
                index += 1
        for op in graph.op:
            if op.type == "RefData":
                op.attr["index"].i = index
                index += 1

    module._convert_data_to_const = original
    module._is_weight_externalized = is_weight_externalized
    module._make_const_node = make_const_node
    module._sort_graph_data_index = sort_graph_data_index
    module.ge = SimpleNamespace(FileConstant=file_constant)
    module.torch_type_to_ge_type = lambda dtype: dtype
    module.GeGraph = nullcontext
    module.logger = _Logger()
    return module


def _interleaved_dynamic_graph() -> _Graph:
    # This reproduces the important ordering property of the failed receiver
    # graph: symbolic-shape helpers precede a captured parameter Data node.
    return _Graph(
        [
            _Op("Gather", "Gather"),
            _Op("draft_fc_weight", "Data", index=0),
            _Op("Const_6", "Const"),
            _Op(
                "Pack",
                "Pack",
                inputs=("Gather:0", "Const_6:0"),
            ),
            _Op("runtime_feature", "Data", index=1),
        ]
    )


def test_external_weight_conversion_uses_data_index_not_op_position() -> None:
    export_utils = _fake_export_utils()
    graph = _interleaved_dynamic_graph()
    weight = torch.ones((2, 3), dtype=torch.float16)
    runtime = torch.zeros((1,), dtype=torch.float16)
    audit = ExternalWeightMappingAudit(required=True, status="ARMED")

    result = _convert_data_to_const_by_index(
        export_utils,
        audit,
        (weight, runtime),
        graph,
        "/weights",
        {id(weight): "draft.propose.draft_fc.weight"},
    )

    assert result == (True, 1)
    assert graph.op[0].name == "Gather"
    assert graph.op[0].type == "Gather"
    assert graph.op[1].name == "draft_fc_weight"
    assert graph.op[1].type == "FileConstant"
    assert graph.op[3].input == ["Gather:0", "Const_6:0"]
    assert audit.converted_weight_inputs == 1
    assert audit.converted_samples == [
        {
            "runtime_input_index": 0,
            "data_node_name": "draft_fc_weight",
            "output_type": "FileConstant",
        }
    ]


def test_external_weight_conversion_fails_before_partial_graph_mutation() -> None:
    export_utils = _fake_export_utils()
    graph = _interleaved_dynamic_graph()
    graph.op[1].attr["index"].i = 9
    weight = torch.ones((2, 3), dtype=torch.float16)
    audit = ExternalWeightMappingAudit(required=True, status="ARMED")

    with pytest.raises(RuntimeError, match="no matching indexed Data node"):
        _convert_data_to_const_by_index(
            export_utils,
            audit,
            (weight,),
            graph,
            "/weights",
            {id(weight): "draft.propose.draft_fc.weight"},
        )

    assert graph.op[0].type == "Gather"
    assert graph.op[1].type == "Data"


def test_dynamic_export_patch_is_scoped_audited_and_restored(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torchair = ModuleType("torchair")
    export_utils = _fake_export_utils()
    original = export_utils._convert_data_to_const
    monkeypatch.setitem(
        sys.modules,
        "torchair._utils.export_utils",
        export_utils,
    )
    graph = _interleaved_dynamic_graph()
    weight = torch.ones((2, 3), dtype=torch.float16)

    with index_safe_external_weight_conversion(
        torchair,
        required=True,
    ) as audit:
        assert export_utils._convert_data_to_const is not original
        export_utils._convert_data_to_const(
            (weight,),
            graph,
            "/weights",
            {id(weight): "draft.propose.draft_fc.weight"},
        )

    assert export_utils._convert_data_to_const is original
    assert audit.status == "PASS"
    assert audit.converter_calls == 1
    assert audit.used_weight_inputs == 1
    assert audit.converted_weight_inputs == 1
    assert audit.as_manifest_record()["mapping_key"] == (
        "GraphDef Data.index == runtime input index"
    )


def test_static_export_does_not_touch_torchair_internals() -> None:
    torchair = object()
    with index_safe_external_weight_conversion(
        torchair,
        required=False,
    ) as audit:
        pass
    assert audit.status == "NOT_REQUIRED_STATIC_GRAPH"
    assert audit.converter_calls == 0
