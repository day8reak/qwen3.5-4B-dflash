"""Opt-in, same-input operator diagnostics; never a generation/performance gate.

CANN owns raw OM dump decoding. This module consumes explicit logical .npy
tensors after conversion; it never guesses FRACTAL/NZ layouts or call pairing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .utils import atomic_write_json, load_json_object, require_run_output, sha256_file

MTP_INPUTS = ("query", "key", "value", "g", "beta", "initial_state", "accepted_tokens")
MTP_OUTPUTS = ("core_attn", "last_recurrent_state")
MTP_ATTRS = ("chunk_size", "output_final_state", "use_qk_l2norm_in_kernel")
MAX_TENSOR_BYTES = 512 * 1024 * 1024


def _new_output(path: str | Path) -> Path:
    result = require_run_output(path)
    if result.exists() or Path(str(result) + ".tmp").exists():
        raise FileExistsError(result)
    return result


def _validate_tolerance(atol: float, rtol: float) -> None:
    if not (math.isfinite(atol) and math.isfinite(rtol)) or min(atol, rtol) < 0:
        raise ValueError("atol/rtol must be finite and nonnegative")


def _identity(path: Path) -> dict[str, Any]:
    return {"path": str(path.resolve()), "bytes": path.stat().st_size,
            "sha256": sha256_file(path)}


def validate_acl_dump_config(path: str | Path) -> Path:
    """Reject unbounded dumps and outputs outside the active run before ACL."""
    source = Path(path).expanduser().resolve()
    if source.stat().st_size > 1024 * 1024:
        raise ValueError("dump config exceeds 1 MiB")
    payload = load_json_object(source)
    if set(payload) != {"dump"} or not isinstance(payload["dump"], dict):
        raise ValueError("diagnostic ACL config must contain only a dump object")
    dump = payload["dump"]
    allowed = {"dump_list", "dump_path", "dump_mode", "dump_data",
               "dump_op_switch", "dump_level"}
    if set(dump) - allowed:
        raise ValueError("unsupported dump option (no exception/watch/profiling config)")
    if dump.get("dump_mode") not in ("all", "input", "output"):
        raise ValueError("explicit dump_mode all/input/output is required")
    if dump.get("dump_data", "tensor") not in ("tensor", "stats"):
        raise ValueError("dump_data must be tensor or stats")
    if dump.get("dump_op_switch", "off") != "off":
        raise ValueError("this runner uses model dump, not single-op dump")
    if dump.get("dump_level", "op") != "op":
        raise ValueError("diagnostic dump_level must be op")
    entries = dump.get("dump_list")
    if not isinstance(entries, list) or not entries or len(entries) > 32:
        raise ValueError("dump_list must contain 1..32 selected-node entries")
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) - {"model_name", "layer"}:
            raise ValueError("dump_list entries allow only model_name and layer")
        layers = entry.get("layer")
        if (not isinstance(layers, list) or not layers or len(layers) > 256
                or any(not isinstance(x, str) or not x.strip() for x in layers)):
            raise ValueError("explicit nonempty layer names required; full-model dump is disabled")
        if "model_name" in entry and (not isinstance(entry["model_name"], str)
                                      or not entry["model_name"].strip()):
            raise ValueError("model_name must be nonempty when supplied")
    destination = dump.get("dump_path")
    if not isinstance(destination, str) or not Path(destination).is_absolute():
        raise ValueError("dump_path must be absolute (no cwd/environment expansion)")
    destination = require_run_output(destination)
    if not destination.is_dir() or any(destination.iterdir()):
        raise ValueError("dump_path must be an existing empty directory in AI_RUN_DIR")
    # Newer runtimes may redirect the configured destination through these.
    if any(os.environ.get(k) for k in ("ASCEND_DUMP_PATH", "NPU_COLLECT_PATH")):
        raise ValueError("unset ASCEND_DUMP_PATH/NPU_COLLECT_PATH to preserve dump provenance")
    return source


def _graph_nodes(value: Any):
    """Read ATC --mode=1 graph.op entries, not arbitrary name/type dicts."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key == "op" and isinstance(item, list):
                for node in item:
                    if isinstance(node, dict) and isinstance(node.get("name"), str):
                        op_type = node.get("type", node.get("op"))
                        if isinstance(op_type, str):
                            yield {"name": node["name"], "type": op_type}
            else:
                yield from _graph_nodes(item)
    elif isinstance(value, list):
        for item in value:
            yield from _graph_nodes(item)


def prepare_dump_config(
    graphs: Sequence[str | Path], *, op_types: Sequence[str],
    output: str | Path, dump_dir: str | Path, limit: int = 1,
    data: str = "tensor",
) -> dict[str, Any]:
    if not graphs or not op_types or not 1 <= limit <= 256:
        raise ValueError("supply OM JSON(s), operator type(s), and limit in 1..256")
    if data not in ("tensor", "stats"):
        raise ValueError("data must be tensor or stats")
    config = _new_output(output)
    plan = _new_output(str(config) + ".plan.json")
    destination = _new_output(dump_dir)
    records, names = [], []
    for graph in graphs:
        graph = Path(graph).expanduser().resolve()
        selected = [node for node in _graph_nodes(load_json_object(graph))
                    if node["type"] in op_types][:limit]
        if not selected:
            raise ValueError(f"no selected operator types in {graph}; inspect actual compiled names")
        records.append({"graph": _identity(graph), "selected_nodes": selected})
        for node in selected:
            if node["name"] not in names:
                names.append(node["name"])
    if len(names) > 256:
        raise ValueError("selected node names exceed 256; split into bounded captures")
    destination.mkdir(parents=True)
    payload = {"dump": {"dump_path": str(destination),
                        "dump_list": [{"layer": names}],
                        "dump_mode": "all", "dump_data": data,
                        "dump_op_switch": "off", "dump_level": "op"}}
    atomic_write_json(config, payload)
    validate_acl_dump_config(config)
    result = {"schema_version": 1, "status": "CONFIGURED_NOT_CAPTURED",
              "config": _identity(config), "graphs": records,
              "selection": "first N matching nodes per JSON in graph order; not decoder layer IDs",
              "model_name_policy": "omitted: selected names apply to every loaded model",
              "requires_new_om": False, "formal_latency_evidence": False}
    atomic_write_json(plan, result)
    return result


def _npy(path: Path) -> np.ndarray:
    # memmap + allow_pickle=False prevents arbitrary pickle execution and avoids
    # loading the complete state bank just to compare it.
    value = np.load(path, mmap_mode="r", allow_pickle=False)
    if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
        raise ValueError(f"only real numeric/bool .npy tensors are supported: {path}")
    if value.dtype.kind == "f" and value.itemsize not in (2, 4, 8):
        raise ValueError(f"only float16/32/64 NPY floats are supported: {path}")
    if value.nbytes > MAX_TENSOR_BYTES:
        raise ValueError(f"tensor exceeds {MAX_TENSOR_BYTES} byte diagnostic limit: {path}")
    if not value.flags.c_contiguous or not value.dtype.isnative:
        raise ValueError(f"convert dump to native-endian logical C-contiguous layout first: {path}")
    return value


def _scalar(value: Any) -> Any:
    value = value.item()
    return None if isinstance(value, float) and not math.isfinite(value) else value


def compare_tensor(reference: Path, actual: Path, *, atol: float = 0., rtol: float = 0.) -> dict[str, Any]:
    _validate_tolerance(atol, rtol)
    ref, got = _npy(reference), _npy(actual)
    result: dict[str, Any] = {
        "reference": _identity(reference), "actual": _identity(actual),
        "reference_shape": list(ref.shape), "actual_shape": list(got.shape),
        "reference_dtype": str(ref.dtype), "actual_dtype": str(got.dtype),
        "atol": atol, "rtol": rtol, "status": "FAIL",
    }
    if ref.shape != got.shape or ref.dtype != got.dtype:
        result["reason"] = "shape-or-dtype-mismatch; no implicit reshape/cast/layout conversion"
        return result
    numeric = bits = nonfinite_ref = nonfinite_got = outside = 0
    max_error: int | float = 0
    first = None
    a, b = ref.reshape(-1), got.reshape(-1)
    for start in range(0, a.size, 65536):
        x, y = a[start:start + 65536], b[start:start + 65536]
        unequal = x != y
        numeric += int(np.count_nonzero(unequal))
        bits += int(np.count_nonzero(np.any(
            x.view(np.uint8).reshape(-1, ref.itemsize) !=
            y.view(np.uint8).reshape(-1, ref.itemsize), axis=1)))
        if ref.dtype.kind == "f":
            finite_x, finite_y = np.isfinite(x), np.isfinite(y)
            nonfinite_ref += int(np.count_nonzero(~finite_x))
            nonfinite_got += int(np.count_nonzero(~finite_y))
            finite = finite_x & finite_y
            # Calculate only on finite pairs; do not hide NaN/Inf as equality.
            xd, yd = x[finite].astype(np.longdouble), y[finite].astype(np.longdouble)
            error = np.abs(xd - yd)
            if error.size:
                max_error = max(max_error, error.max())
            bad = ~finite
            bad[finite] = error > (atol + rtol * np.abs(xd))
        else:
            # Integer IDs/selectors must remain exact, including above 2**53.
            bad = unequal
            if np.any(bad):
                integer_error = np.abs(x[bad].astype(object) - y[bad].astype(object))
                max_error = max(max_error, int(integer_error.max()))
        outside += int(np.count_nonzero(bad))
        if first is None and np.any(bad):
            index = start + int(np.flatnonzero(bad)[0])
            first = {"flat_index": index, "coordinates": list(np.unravel_index(index, ref.shape)),
                     "reference": _scalar(a[index]), "actual": _scalar(b[index])}
            first["coordinates"] = [int(v) for v in first["coordinates"]]
    finite_error_overflow = ref.dtype.kind == "f" and max_error > np.finfo(np.float64).max
    serial_error = None if finite_error_overflow else float(max_error) if ref.dtype.kind == "f" else int(max_error)
    result.update(status="PASS" if outside == 0 else "FAIL",
                  compared_elements=int(a.size), numeric_different_elements=numeric,
                  bitwise_different_elements=bits, outside_tolerance_elements=outside,
                  reference_nonfinite=nonfinite_ref, actual_nonfinite=nonfinite_got,
                  max_finite_abs_error=serial_error,
                  finite_error_overflowed_float64=bool(finite_error_overflow),
                  first_difference=first)
    return result


def compare_operators(mapping: str | Path, output: str | Path, *,
                      atol: float = 0., rtol: float = 0.) -> dict[str, Any]:
    """Compare explicitly mapped calls in declared causal order, never by name sorting."""
    mapping = Path(mapping).expanduser().resolve()
    payload = load_json_object(mapping)
    operators = payload.get("operators")
    if payload.get("schema_version") != 1 or not isinstance(operators, list) or not operators:
        raise ValueError("mapping requires schema_version=1 and nonempty operators")
    output = _new_output(output)
    result: dict[str, Any] = {"schema_version": 1, "report_kind": "operator-tensor-comparison",
        "formal_latency_evidence": False, "mapping": _identity(mapping),
        "order_scope": "caller-declared causal order; not inferred from filenames",
        "operators": [], "first_failing_operator": None, "status": "PASS",
        "claim_boundary": "mapped tensors only; neither kernel culpability nor whole-model parity"}
    ids = set()
    for item in operators:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) or item["id"] in ids:
            raise ValueError("each operator must have a unique string id")
        ids.add(item["id"])
        record = {"id": item["id"], "inputs": [], "outputs": []}
        for side in ("inputs", "outputs"):
            entries = item.get(side, [])
            if not isinstance(entries, list) or (side == "outputs" and not entries):
                raise ValueError("operator outputs must contain explicit reference/actual pairs")
            for pair in entries:
                check = compare_tensor((mapping.parent / pair["reference"]).resolve(),
                                       (mapping.parent / pair["actual"]).resolve(), atol=atol, rtol=rtol)
                record[side].append({"name": pair["name"], **check})
        inputs_exact = bool(record["inputs"]) and all(
            x.get("bitwise_different_elements") == 0 and x["status"] == "PASS"
            for x in record["inputs"])
        failure = any(x["status"] == "FAIL" for side in ("inputs", "outputs") for x in record[side])
        record["status"] = "FAIL" if failure else "PASS"
        record["mapped_inputs_bitwise_equal"] = inputs_exact
        record["localization"] = (
            "upstream-or-input-mapping" if any(x["status"] == "FAIL" for x in record["inputs"])
            else "output-divergence; verify ALL inputs/attrs/layout/version before blaming operator"
            if failure else "mapped-pairs-pass")
        result["operators"].append(record)
        if failure:
            result["status"] = "FAIL"
            if result["first_failing_operator"] is None:
                result["first_failing_operator"] = item["id"]
    atomic_write_json(output, result)
    return result


def _mtp_case(case: Path):
    payload = load_json_object(case)
    if payload.get("schema_version") != 1 or payload.get("op_type") != "GatedDeltaRuleMTP":
        raise ValueError("expected schema_version=1 op_type=GatedDeltaRuleMTP")
    if payload.get("layout") != "logical-ND":
        raise ValueError("convert CANN physical dump to logical-ND first; layout cannot be guessed")
    if not isinstance(payload.get("provenance"), dict) or not payload["provenance"]:
        raise ValueError("record OM/node/transaction dump provenance explicitly")
    if set(payload.get("inputs", {})) != set(MTP_INPUTS) or set(payload.get("outputs", {})) != set(MTP_OUTPUTS):
        raise ValueError("MTP requires all seven named inputs and both OM outputs")
    attrs = payload.get("attrs", {})
    if set(attrs) != set(MTP_ATTRS) or type(attrs["chunk_size"]) is not int:
        raise ValueError("copy all three attrs from the actual compiled node; no inferred defaults")
    if (attrs["chunk_size"] != 64 or attrs["output_final_state"] is not True
            or attrs["use_qk_l2norm_in_kernel"] is not True):
        raise ValueError("case is outside locked Qwen GDR-MTP attributes (64,true,true)")
    arrays = {k: _npy((case.parent / p).resolve()) for k, p in payload["inputs"].items()}
    if sum(value.nbytes for value in arrays.values()) > MAX_TENSOR_BYTES:
        raise ValueError("total MTP inputs exceed diagnostic memory budget")
    q, k, v = (arrays[x] for x in ("query", "key", "value"))
    if q.ndim != 4 or q.shape != k.shape or v.ndim != 4 or q.shape[:3] != v.shape[:3]:
        raise ValueError("expected Q/K [B,T,H,Dk], V [B,T,H,Dv]")
    batch, rows, heads, width = q.shape
    if not 1 <= rows <= 16 or min(batch, heads, width, v.shape[-1]) <= 0:
        raise ValueError("case outside positive B/H/D and fixed 1..16 row contract")
    expected = {"query": (q.shape, "float16"), "key": (q.shape, "float16"),
                "value": (v.shape, "float16"), "g": ((batch, rows, heads), "float32"),
                "beta": ((batch, rows, heads), "float16"),
                "initial_state": ((batch, rows, heads, width, v.shape[-1]), "float32"),
                "accepted_tokens": ((batch,), "int8")}
    for name, (shape, dtype) in expected.items():
        if arrays[name].shape != shape or str(arrays[name].dtype) != dtype:
            raise ValueError(f"{name} shape/dtype differs from locked logical MTP ABI")
    selector = arrays["accepted_tokens"]
    if np.any(selector < 0) or np.any(selector >= rows):
        raise ValueError("accepted_tokens is a previous-state BANK SLOT, outside [0,T)")
    for name, shape, dtype in (("core_attn", v.shape, "float16"),
                               ("last_recurrent_state", expected["initial_state"][0], "float32")):
        value = _npy((case.parent / payload["outputs"][name]).resolve())
        if value.shape != shape or str(value.dtype) != dtype:
            raise ValueError(f"{name} OM output shape/dtype differs from locked MTP ABI")
    return payload, arrays


def _bind_argument(value: Any, parent: Path, device: str, identities: list):
    """A data-only grammar: tensors, scalar values and nested tensor lists."""
    import torch
    if isinstance(value, list):
        return [_bind_argument(v, parent, device, identities) for v in value]
    if not isinstance(value, dict) or len(value) != 1:
        raise ValueError("argument must be {tensor: path}, {dtype: name}, {value: literal}, or a list of these")
    if "dtype" in value:
        # ScalarType arguments cannot be inferred from a dump's output dtype.
        allowed = {"bool", "uint8", "int8", "int16", "int32", "int64",
                   "float16", "bfloat16", "float32", "float64"}
        if not isinstance(value["dtype"], str) or value["dtype"] not in allowed:
            raise ValueError("unsupported explicit dispatcher dtype")
        return getattr(torch, value["dtype"])
    if "tensor" in value:
        path = (parent / value["tensor"]).resolve()
        array = _npy(path)
        identities.append(_identity(path))
        if sum(item["bytes"] for item in identities) > MAX_TENSOR_BYTES:
            raise ValueError("total captured inputs exceed diagnostic memory budget")
        return torch.from_numpy(np.array(array, copy=True)).to(device)
    if "value" not in value:
        raise ValueError("unknown argument encoding")
    literal = value["value"]
    def valid(item):
        return (item is None or isinstance(item, (str, bool, int))
                or isinstance(item, float) and math.isfinite(item)
                or isinstance(item, list) and all(valid(v) for v in item))
    if not valid(literal):
        raise ValueError("argument literal must be finite JSON scalar/list")
    return literal


def _require_npu_tensor(tensor: Any, device_id: int) -> None:
    import torch
    if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "npu"
            or tensor.device.index != device_id):
        raise RuntimeError("native replay returned a non-NPU/wrong-device Tensor; no CPU fallback")


def replay_operator(case: str | Path, output_dir: str | Path, *, device_id: int = 0,
                    atol: float = 0., rtol: float = 0.) -> dict[str, Any]:
    """Replay any explicitly mapped registered aten/npu Tensor operator on NPU.

    No eval, imports named by the case, approximate formula, dtype coercion or
    automatic GE-to-PyTorch argument guessing. Unsupported schemas fail closed.
    """
    _validate_tolerance(atol, rtol)
    case = Path(case).expanduser().resolve()
    payload = load_json_object(case)
    if payload.get("schema_version") != 1 or payload.get("layout") != "logical-ND":
        raise ValueError("operator case requires schema_version=1 and layout=logical-ND")
    if not isinstance(payload.get("provenance"), dict) or not payload["provenance"]:
        raise ValueError("operator case requires explicit dump provenance")
    name, overload = payload.get("torch_op", ""), payload.get("overload", "default")
    if (not isinstance(name, str) or not re.fullmatch(r"(aten|npu)::[A-Za-z_][A-Za-z_0-9]*", name)
            or not isinstance(overload, str) or not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", overload)):
        raise ValueError("torch_op/overload must name a registered aten or npu operator")
    arguments, outputs = payload.get("arguments"), payload.get("outputs")
    if not isinstance(arguments, dict) or not isinstance(outputs, dict) or not outputs:
        raise ValueError("operator case requires named arguments and nonempty output mapping")
    if type(device_id) is not int or device_id < 0:
        raise ValueError("device_id must be nonnegative")
    destination = _new_output(output_dir)
    import torch
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError("operator replay requires a real NPU; no CPU fallback")
    torch.npu.set_device(device_id)
    namespace, op_name = name.split("::")
    operation = getattr(getattr(getattr(torch.ops, namespace), op_name), overload)
    schema = operation._schema
    if set(arguments) - {a.name for a in schema.arguments}:
        raise ValueError(f"argument names do not match installed dispatcher schema: {schema}")
    if any(a.name not in arguments and not a.has_default_value() for a in schema.arguments):
        raise ValueError(f"required argument missing for dispatcher schema: {schema}")
    identities: list[dict[str, Any]] = []
    bound = {k: _bind_argument(v, case.parent, f"npu:{device_id}", identities)
             for k, v in arguments.items()}
    if not identities:
        raise ValueError("same-input replay requires captured tensor inputs")
    # Validate all expected outputs before expensive execution, and reject
    # duplicate selectors (which could conceal a missing output).
    selectors = []
    expected_bytes = 0
    for output_name, entry in outputs.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", output_name):
            raise ValueError("output labels must be simple identifiers")
        if not isinstance(entry, dict) or set(entry) != {"index", "tensor"}:
            raise ValueError("each output needs an index and tensor path")
        index = entry.get("index")
        if (not isinstance(index, list) or any(type(i) is not int or i < 0 for i in index)
                or index in selectors):
            raise ValueError("each output requires a unique list index; [] selects a single Tensor")
        selectors.append(index)
        expected_bytes += _npy((case.parent / entry["tensor"]).resolve()).nbytes
    if expected_bytes > MAX_TENSOR_BYTES:
        raise ValueError("total expected outputs exceed diagnostic memory budget")
    destination.mkdir(parents=True)
    native = {"torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown"),
              "device": str(torch.npu.get_device_name(device_id)), "device_id": device_id,
              "dispatcher_schema": str(schema)}
    atomic_write_json(destination / "invocation.json", {
        "schema_version": 1, "case": _identity(case), "native": native,
        "inputs": identities, "provenance": payload["provenance"],
        "execution_status": "not_recorded", "cpu_fallback": False})
    with torch.inference_mode():
        result = operation(**bound)
        torch.npu.synchronize()
    def tensor_leaves(value, prefix=()):
        if isinstance(value, torch.Tensor):
            return {prefix}
        if isinstance(value, (tuple, list)):
            return set().union(*(tensor_leaves(v, (*prefix, i)) for i, v in enumerate(value)))
        return set()
    leaves = tensor_leaves(result)
    if leaves != {tuple(i) for i in selectors}:
        raise ValueError("output mapping must cover every native Tensor output exactly once")
    pairs = []
    result_bytes = 0
    for output_name, entry in outputs.items():
        tensor = result
        for index in entry["index"]:
            tensor = tensor[index]
        _require_npu_tensor(tensor, device_id)
        result_bytes += tensor.numel() * tensor.element_size()
        if result_bytes > MAX_TENSOR_BYTES:
            raise ValueError("native results exceed diagnostic tensor budget")
        path = destination / (output_name + ".npy")
        np.save(path, tensor.detach().contiguous().cpu().numpy(), allow_pickle=False)
        pairs.append({"name": output_name, "reference": str(path),
                      "actual": str((case.parent / entry["tensor"]).resolve())})
    mapping = destination / "pairs.json"
    atomic_write_json(mapping, {"schema_version": 1,
        "operators": [{"id": name + "." + overload, "outputs": pairs}]})
    comparison = compare_operators(mapping, destination / "comparison.json", atol=atol, rtol=rtol)
    return {"status": comparison["status"], "report": str(destination / "comparison.json"),
            "scope": "same-input mapped native operator vs OM; not full-model or kernel-build equivalence",
            "formal_latency_evidence": False, "cpu_fallback": False}


def replay_gdr_mtp(case: str | Path, output_dir: str | Path, *, device_id: int = 0,
                   atol: float = 0., rtol: float = 0.) -> dict[str, Any]:
    """Run native MTP on captured OM inputs, with no CPU/small-model fallback."""
    _validate_tolerance(atol, rtol)
    case = Path(case).expanduser().resolve()
    payload, arrays = _mtp_case(case)
    output = _new_output(output_dir)
    if type(device_id) is not int or device_id < 0:
        raise ValueError("device_id must be nonnegative")
    import torch
    import torch_npu
    if not torch.npu.is_available():
        raise RuntimeError("native MTP replay requires a real NPU; no CPU fallback")
    torch.npu.set_device(device_id)
    operation = getattr(torch_npu, "npu_gated_delta_rule_mtp", None)
    if not callable(operation):
        operation = torch.ops.npu.npu_gated_delta_rule_mtp.default
    tensors = [torch.from_numpy(np.array(arrays[k], copy=True)).to(f"npu:{device_id}")
               for k in MTP_INPUTS]
    output.mkdir(parents=True)
    attrs = payload["attrs"]
    # Persist provenance before the call, even if native execution throws.
    native = {"torch": torch.__version__, "torch_npu": getattr(torch_npu, "__version__", "unknown"),
              "torch_npu_module": str(getattr(torch_npu, "__file__", "unknown")),
              "device": str(torch.npu.get_device_name(device_id)), "device_id": device_id,
              "dispatcher_schema": str(torch.ops.npu.npu_gated_delta_rule_mtp.default._schema)}
    atomic_write_json(output / "invocation.json", {
        "schema_version": 1, "case": _identity(case), "native": native,
        "inputs": {k: _identity((case.parent / payload["inputs"][k]).resolve()) for k in MTP_INPUTS},
        "attrs": attrs, "provenance": payload["provenance"],
        "execution_status": "not_recorded", "cpu_fallback": False,
        "note": "package versions do not establish native/OM kernel build equality"})
    with torch.inference_mode():
        answer = operation(*tensors, *(attrs[k] for k in MTP_ATTRS))
        torch.npu.synchronize()
    if not isinstance(answer, (tuple, list)) or len(answer) != 2:
        raise RuntimeError("native MTP did not return both output and full state bank")
    pairs = []
    for name, tensor in zip(MTP_OUTPUTS, answer):
        _require_npu_tensor(tensor, device_id)
        if tensor.numel() * tensor.element_size() > MAX_TENSOR_BYTES:
            raise ValueError("native MTP result exceeds diagnostic tensor budget")
        path = output / (name + ".npy")
        np.save(path, tensor.detach().contiguous().cpu().numpy(), allow_pickle=False)
        pairs.append({"name": name, "reference": str(path),
                      "actual": str((case.parent / payload["outputs"][name]).resolve())})
    mapping = output / "pairs.json"
    atomic_write_json(mapping, {"schema_version": 1,
        "operators": [{"id": "same-input-native-mtp-vs-om", "outputs": pairs}]})
    result = compare_operators(mapping, output / "comparison.json", atol=atol, rtol=rtol)
    return {"status": result["status"], "report": str(output / "comparison.json"),
            "scope": "same-input native MTP vs captured OM outputs; not whole-model parity",
            "formal_latency_evidence": False, "cpu_fallback": False}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare-dump")
    prepare.add_argument("--om-json", type=Path, action="append", required=True)
    prepare.add_argument("--op-type", action="append", required=True)
    prepare.add_argument("--limit", type=int, default=1)
    prepare.add_argument("--data", choices=("tensor", "stats"), default="tensor")
    prepare.add_argument("--dump-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--mapping", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    replay = commands.add_parser("replay-gdr-mtp")
    replay.add_argument("--case", type=Path, required=True)
    replay.add_argument("--output-dir", type=Path, required=True)
    replay.add_argument("--device-id", type=int, default=0)
    generic = commands.add_parser("replay-operator")
    generic.add_argument("--case", type=Path, required=True)
    generic.add_argument("--output-dir", type=Path, required=True)
    generic.add_argument("--device-id", type=int, default=0)
    for command in (compare, replay, generic):
        command.add_argument("--atol", type=float, default=0.)
        command.add_argument("--rtol", type=float, default=0.)
    args = parser.parse_args(argv)
    if args.command == "prepare-dump":
        result = prepare_dump_config(args.om_json, op_types=args.op_type, output=args.output,
                                     dump_dir=args.dump_dir, limit=args.limit, data=args.data)
    elif args.command == "compare":
        result = compare_operators(args.mapping, args.output, atol=args.atol, rtol=args.rtol)
    elif args.command == "replay-operator":
        result = replay_operator(args.case, args.output_dir, device_id=args.device_id,
                                 atol=args.atol, rtol=args.rtol)
    else:
        result = replay_gdr_mtp(args.case, args.output_dir, device_id=args.device_id,
                               atol=args.atol, rtol=args.rtol)
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    return int(result["status"] == "FAIL")


if __name__ == "__main__":
    raise SystemExit(main())
