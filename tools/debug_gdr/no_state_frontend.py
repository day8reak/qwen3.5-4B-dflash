"""Temporary core-only frontend for the output_final_state=False experiment.

The production GDR Fake kernel only accepts True. This independent namespace
does not replace it: native execution calls the receiver API with False, and
the converter emits the same registered GE operator with False explicitly.
"""
from contextlib import contextmanager
import importlib
from types import SimpleNamespace

_LIBRARIES = []
_OPERATION = None


def inspect_graph(graph):
    nodes = [node for node in graph.op if node.type == "ChunkGatedDeltaRule"]
    if len(nodes) != 1:
        raise RuntimeError("no-state AIR must have exactly one GDR")
    node = nodes[0]
    if any(key not in node.attr for key in ("chunk_size", "output_final_state", "use_qk_l2norm_in_kernel")):
        raise RuntimeError("no-state GDR is missing explicit attributes")
    actual = {"chunk_size": node.attr["chunk_size"].i,
              "output_final_state": node.attr["output_final_state"].b,
              "use_qk_l2norm_in_kernel": node.attr["use_qk_l2norm_in_kernel"].b}
    if actual != {"chunk_size": 64, "output_final_state": False, "use_qk_l2norm_in_kernel": True}:
        raise RuntimeError(f"wrong serialized no-state GDR attributes: {actual}")
    consumers = [user.name for user in graph.op if f"{node.name}:1" in user.input]
    if consumers:
        raise RuntimeError(f"no-state GDR state output unexpectedly has consumers: {consumers}")
    return {"status": "PASS", "node": node.name, "attributes": actual, "state_consumers": consumers}


@contextmanager
def serialized_audit(audit):
    """Inspect the actual GE graph at the same save boundary as the ABI audit."""
    utils = importlib.import_module("torchair._utils.export_utils")
    original = utils._convert_data_to_const

    def convert(inputs, export_graph, file_path, weight_name):
        result = original(inputs, export_graph, file_path, weight_name)
        audit["serialized_ge"] = inspect_graph(export_graph)
        return result

    utils._convert_data_to_const = convert
    try:
        yield
        if audit.get("serialized_ge", {}).get("status") != "PASS":
            raise RuntimeError("no-state GDR serialization was not audited")
    finally:
        utils._convert_data_to_const = original


def register(torch, torch_npu, torchair):
    global _OPERATION
    if _OPERATION is None:
        lib = torch.library.Library("qwen35_gdr_debug", "FRAGMENT")
        lib.define("core_no_state(Tensor query, Tensor key, Tensor value, Tensor g, "
                   "Tensor beta, Tensor initial_state, Tensor effective_length) -> Tensor")

        def eager(query, key, value, g, beta, initial_state, effective_length):
            outputs = torch_npu.npu_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=initial_state,
                effective_length=effective_length, chunk_size=64,
                output_final_state=False, use_qk_l2norm_in_kernel=True)
            return outputs[0].reshape(1, 16, 32, 128)

        def fake(query, key, value, g, beta, initial_state, effective_length):
            return value.new_empty((1, 16, 32, 128), dtype=query.dtype)

        lib.impl("core_no_state", eager, "CompositeExplicitAutograd")
        lib.impl("core_no_state", fake, "Meta")
        _LIBRARIES.append(lib)
        _OPERATION = torch.ops.qwen35_gdr_debug.core_no_state.default

    audit = {"torch_op": "qwen35_gdr_debug::core_no_state", "ge_op": "ChunkGatedDeltaRule",
             "converter_calls": 0, "output_final_state": False,
             "public_outputs": ["core_attn"],
             "unused_ge_slot": "registered FP32 last_recurrent_state descriptor; never returned or read"}

    def converter(query, key, value, g, beta, initial_state, effective_length, meta_outputs=None):
        outputs = torchair.ge.custom_op(
            "ChunkGatedDeltaRule",
            inputs={"query": query, "key": key, "value": value, "g": g, "beta": beta,
                    "initial_state": initial_state, "effective_length": effective_length},
            outputs=["core_attn", "last_recurrent_state"],
            attrs={"chunk_size": torchair.ge.attr.Int(64),
                   "output_final_state": torchair.ge.attr.Bool(False),
                   "use_qk_l2norm_in_kernel": torchair.ge.attr.Bool(True)})
        if not isinstance(outputs, (tuple, list)) or len(outputs) != 2:
            raise RuntimeError("registered ChunkGatedDeltaRule requires two GE output slots")
        # The registered GE prototype has a mandatory FP32 state slot, even
        # when it has no consumer. Describe that slot without adding a graph
        # output or asserting that the native False call returns a state value.
        # TorchAir Tensor.set_meta: Ascend/torchair python/torchair/ge/_ge_graph.py.
        outputs[1].set_meta(torch.empty((1, 32, 128, 128), dtype=torch.float32, device="meta"))
        audit["converter_calls"] += 1
        return outputs[0]

    torchair.register_fx_node_ge_converter(_OPERATION)(converter)

    class CoreOnly(torch.nn.Module):
        def forward(self, query, key, value, g, beta, initial_state, effective_length):
            return (_OPERATION(query, key, value, g, beta, initial_state, effective_length),)

    return SimpleNamespace(model=CoreOnly().eval(), operation=_OPERATION, audit=audit, converter=converter)
