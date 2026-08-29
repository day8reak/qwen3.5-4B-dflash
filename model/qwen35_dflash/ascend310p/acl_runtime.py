"""Strict static-shape pyACL executor for hash-locked OM graph bundles.

This is intentionally a low-level building block.  The built-in recompute
backend maps one integrated target+DFlash graph onto ``run_graph``; an external
incremental backend may still own its cache/state ABI.  Manifest-declared names
are ordered aliases, so execution does not depend on TorchAir/ATC internal
tensor names.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .utils import contained_path, load_json_object, sha256_file


class AclRuntimeError(RuntimeError):
    pass


def _check(ret: Any, operation: str) -> None:
    if int(ret) != 0:
        raise AclRuntimeError(f"{operation} failed with ACL error {ret}")


def _value_and_ret(value: Any, operation: str) -> Any:
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], int):
        result, ret = value
        _check(ret, operation)
        return result
    return value


def _dtype_table(acl: Any) -> dict[int, np.dtype[Any]]:
    fallback = {
        0: np.dtype(np.float32),
        1: np.dtype(np.float16),
        2: np.dtype(np.int8),
        3: np.dtype(np.int32),
        4: np.dtype(np.uint8),
        6: np.dtype(np.int16),
        7: np.dtype(np.uint16),
        8: np.dtype(np.uint32),
        9: np.dtype(np.int64),
        10: np.dtype(np.uint64),
        11: np.dtype(np.float64),
        12: np.dtype(np.bool_),
    }
    named = {
        "ACL_FLOAT": np.float32,
        "ACL_FLOAT16": np.float16,
        "ACL_INT8": np.int8,
        "ACL_INT32": np.int32,
        "ACL_UINT8": np.uint8,
        "ACL_INT16": np.int16,
        "ACL_UINT16": np.uint16,
        "ACL_UINT32": np.uint32,
        "ACL_INT64": np.int64,
        "ACL_UINT64": np.uint64,
        "ACL_DOUBLE": np.float64,
        "ACL_BOOL": np.bool_,
    }
    for name, dtype in named.items():
        if hasattr(acl, name):
            fallback[int(getattr(acl, name))] = np.dtype(dtype)
    return fallback


@dataclass
class _Binding:
    index: int
    name: str
    runtime_name: str
    shape: tuple[int, ...]
    dtype: np.dtype[Any]
    size: int
    pointer: int
    data_buffer: Any


class _AclModel:
    def __init__(
        self,
        acl: Any,
        name: str,
        path: Path,
        *,
        input_names: tuple[str, ...] = (),
        output_names: tuple[str, ...] = (),
    ) -> None:
        self.acl = acl
        self.name = name
        self.path = path
        self.model_id: Any | None = None
        self.desc: Any | None = None
        self.input_dataset: Any | None = None
        self.output_dataset: Any | None = None
        self.inputs: list[_Binding] = []
        self.outputs: list[_Binding] = []
        try:
            self.model_id = _value_and_ret(
                acl.mdl.load_from_file(str(path)), "acl.mdl.load_from_file"
            )
            self.desc = acl.mdl.create_desc()
            if self.desc is None:
                raise AclRuntimeError("acl.mdl.create_desc returned null")
            _check(acl.mdl.get_desc(self.desc, self.model_id), "acl.mdl.get_desc")
            self.input_dataset, self.inputs = self._create_dataset(
                "input", input_names
            )
            self.output_dataset, self.outputs = self._create_dataset(
                "output", output_names
            )
        except BaseException:
            self.close()
            raise

    def _name(self, io_type: str, index: int) -> str:
        function = getattr(self.acl.mdl, f"get_{io_type}_name_by_index")
        value = _value_and_ret(function(self.desc, index), f"get_{io_type}_name_by_index")
        if isinstance(value, bytes):
            value = value.decode("utf-8")
        return str(value)

    def _shape(self, io_type: str, index: int) -> tuple[int, ...]:
        function = getattr(self.acl.mdl, f"get_{io_type}_dims")
        value = _value_and_ret(function(self.desc, index), f"get_{io_type}_dims")
        dims = value.get("dims") if isinstance(value, Mapping) else value
        shape = tuple(int(item) for item in dims)
        if not shape or any(item <= 0 for item in shape):
            raise AclRuntimeError(
                f"{self.name} {io_type} {index} is not a fixed positive shape: {shape}"
            )
        return shape

    def _dtype(self, io_type: str, index: int) -> np.dtype[Any]:
        function = getattr(self.acl.mdl, f"get_{io_type}_data_type")
        value = _value_and_ret(function(self.desc, index), f"get_{io_type}_data_type")
        table = _dtype_table(self.acl)
        if int(value) not in table:
            raise AclRuntimeError(
                f"{self.name} {io_type} {index} uses unsupported ACL dtype {value}"
            )
        return table[int(value)]

    def _create_dataset(
        self,
        io_type: str,
        aliases: tuple[str, ...],
    ) -> tuple[Any, list[_Binding]]:
        dataset = self.acl.mdl.create_dataset()
        if dataset is None:
            raise AclRuntimeError(f"acl.mdl.create_dataset failed for {io_type}")
        count = int(getattr(self.acl.mdl, f"get_num_{io_type}s")(self.desc))
        if aliases and len(aliases) != count:
            self.acl.mdl.destroy_dataset(dataset)
            raise AclRuntimeError(
                f"{self.name} declares {len(aliases)} {io_type} names but OM has {count}"
            )
        if len(set(aliases)) != len(aliases):
            self.acl.mdl.destroy_dataset(dataset)
            raise AclRuntimeError(f"{self.name} declares duplicate {io_type} names")
        size_function = getattr(self.acl.mdl, f"get_{io_type}_size_by_index")
        bindings: list[_Binding] = []
        try:
            for index in range(count):
                runtime_name = self._name(io_type, index)
                size = int(size_function(self.desc, index))
                shape = self._shape(io_type, index)
                dtype = self._dtype(io_type, index)
                dense_size = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
                if dense_size != size:
                    raise AclRuntimeError(
                        f"{self.name} {io_type} {index} buffer is {size} bytes but "
                        f"dense {shape}/{dtype} needs {dense_size}; provide a graph-specific backend"
                    )
                policy = int(getattr(self.acl, "ACL_MEM_MALLOC_HUGE_FIRST", 0))
                pointer = _value_and_ret(
                    self.acl.rt.malloc(size, policy), "acl.rt.malloc"
                )
                data_buffer = self.acl.create_data_buffer(pointer, size)
                if data_buffer is None:
                    self.acl.rt.free(pointer)
                    raise AclRuntimeError("acl.create_data_buffer returned null")
                try:
                    _value_and_ret(
                        self.acl.mdl.add_dataset_buffer(dataset, data_buffer),
                        "acl.mdl.add_dataset_buffer",
                    )
                except BaseException:
                    self.acl.destroy_data_buffer(data_buffer)
                    self.acl.rt.free(pointer)
                    raise
                bindings.append(
                    _Binding(
                        index=index,
                        name=aliases[index] if aliases else runtime_name,
                        runtime_name=runtime_name,
                        shape=shape,
                        dtype=dtype,
                        size=size,
                        pointer=int(pointer),
                        data_buffer=data_buffer,
                    )
                )
        except BaseException:
            for binding in reversed(bindings):
                self.acl.destroy_data_buffer(binding.data_buffer)
                self.acl.rt.free(binding.pointer)
            self.acl.mdl.destroy_dataset(dataset)
            raise
        return dataset, bindings

    def run(self, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        expected = {binding.name for binding in self.inputs}
        actual = set(inputs)
        if actual != expected:
            raise ValueError(
                f"{self.name} input names differ: missing={sorted(expected-actual)}, "
                f"extra={sorted(actual-expected)}"
            )
        host_inputs: list[np.ndarray] = []
        for binding in self.inputs:
            value = np.asarray(inputs[binding.name])
            if value.dtype != binding.dtype:
                raise TypeError(
                    f"{self.name}:{binding.name} dtype is {value.dtype}, expected {binding.dtype}"
                )
            if tuple(value.shape) != binding.shape:
                raise ValueError(
                    f"{self.name}:{binding.name} shape is {tuple(value.shape)}, "
                    f"expected {binding.shape}"
                )
            value = np.ascontiguousarray(value)
            host_inputs.append(value)
            source = self.acl.util.numpy_to_ptr(value)
            kind = int(getattr(self.acl, "ACL_MEMCPY_HOST_TO_DEVICE", 1))
            _check(
                self.acl.rt.memcpy(
                    binding.pointer,
                    binding.size,
                    source,
                    value.nbytes,
                    kind,
                ),
                "acl.rt.memcpy(host_to_device)",
            )
        _check(
            self.acl.mdl.execute(
                self.model_id,
                self.input_dataset,
                self.output_dataset,
            ),
            "acl.mdl.execute",
        )
        outputs: dict[str, np.ndarray] = {}
        for binding in self.outputs:
            value = np.empty(binding.shape, dtype=binding.dtype)
            destination = self.acl.util.numpy_to_ptr(value)
            kind = int(getattr(self.acl, "ACL_MEMCPY_DEVICE_TO_HOST", 2))
            _check(
                self.acl.rt.memcpy(
                    destination,
                    value.nbytes,
                    binding.pointer,
                    binding.size,
                    kind,
                ),
                "acl.rt.memcpy(device_to_host)",
            )
            outputs[binding.name] = value
        return outputs

    def close(self) -> None:
        for bindings, dataset in (
            (self.outputs, self.output_dataset),
            (self.inputs, self.input_dataset),
        ):
            for binding in reversed(bindings):
                try:
                    self.acl.destroy_data_buffer(binding.data_buffer)
                finally:
                    self.acl.rt.free(binding.pointer)
            bindings.clear()
            if dataset is not None:
                self.acl.mdl.destroy_dataset(dataset)
        self.input_dataset = None
        self.output_dataset = None
        if self.desc is not None:
            self.acl.mdl.destroy_desc(self.desc)
            self.desc = None
        if self.model_id is not None:
            self.acl.mdl.unload(self.model_id)
            self.model_id = None


class AclOmRuntime:
    """Load and execute all static OM graphs in one pyACL device context."""

    def __init__(
        self,
        deployment_manifest_path: str | Path,
        *,
        device_id: int = 0,
        acl_module: Any | None = None,
    ) -> None:
        self.manifest_path = Path(deployment_manifest_path).expanduser().resolve()
        self.manifest = load_json_object(self.manifest_path)
        if self.manifest.get("artifact_kind") != "qwen35-dflash-ascend310p-om-bundle":
            raise ValueError("pyACL runtime requires a DFlash Ascend 310P OM bundle")
        if self.manifest.get("status") != "PASS":
            raise ValueError("deployment manifest is not passing")
        self.root = self.manifest_path.parent
        self.device_id = int(device_id)
        self.acl = acl_module or importlib.import_module("acl")
        self.models: dict[str, _AclModel] = {}
        self._initialized = False
        try:
            _check(self.acl.init(), "acl.init")
            self._initialized = True
            _check(self.acl.rt.set_device(self.device_id), "acl.rt.set_device")
            graphs = self.manifest.get("graphs", [])
            if not isinstance(graphs, list) or not graphs:
                raise ValueError("deployment manifest contains no OM graphs")
            for graph in graphs:
                name = str(graph["name"])
                if name in self.models:
                    raise ValueError(f"deployment manifest repeats OM graph name: {name}")
                om = graph["om"]
                path = contained_path(self.root, str(om["path"]))
                if not path.is_file() or sha256_file(path) != om["sha256"]:
                    raise ValueError(f"OM hash check failed before load: {name}")
                self.models[name] = _AclModel(
                    self.acl,
                    name,
                    path,
                    input_names=tuple(str(item) for item in graph.get("input_names", [])),
                    output_names=tuple(str(item) for item in graph.get("output_names", [])),
                )
        except BaseException:
            self.close()
            raise

    @property
    def graph_names(self) -> tuple[str, ...]:
        return tuple(self.models)

    def graph_inputs(self, name: str) -> tuple[dict[str, Any], ...]:
        model = self.models[name]
        return tuple(
            {
                "name": item.name,
                "runtime_name": item.runtime_name,
                "shape": list(item.shape),
                "dtype": str(item.dtype),
            }
            for item in model.inputs
        )

    def graph_outputs(self, name: str) -> tuple[dict[str, Any], ...]:
        model = self.models[name]
        return tuple(
            {
                "name": item.name,
                "runtime_name": item.runtime_name,
                "shape": list(item.shape),
                "dtype": str(item.dtype),
            }
            for item in model.outputs
        )

    def run_graph(self, name: str, inputs: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        if name not in self.models:
            raise KeyError(f"OM graph is not loaded: {name!r}")
        return self.models[name].run(inputs)

    def synchronize(self) -> None:
        operation = getattr(self.acl.rt, "synchronize_device", None)
        if operation is not None:
            _check(operation(), "acl.rt.synchronize_device")

    def artifact_hashes(self) -> dict[str, str]:
        return {
            str(graph["name"]): str(graph["om"]["sha256"])
            for graph in self.manifest.get("graphs", [])
        }

    def close(self) -> None:
        for model in reversed(tuple(self.models.values())):
            model.close()
        self.models.clear()
        if self._initialized:
            try:
                self.acl.rt.reset_device(self.device_id)
            finally:
                self.acl.finalize()
                self._initialized = False

    def __enter__(self) -> "AclOmRuntime":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()
