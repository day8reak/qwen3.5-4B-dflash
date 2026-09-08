"""Export a factory-provided DFlash graph suite to standard TorchAir AIR."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
import importlib
import os
from pathlib import Path
import platform
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .contracts import AirGraphSpec
from .custom_op_export import audit_custom_op_export, prepare_custom_op_export
from .standard_op_export import prepare_aten_softplus_export, audit_aten_softplus_export
from .runtime_input_export import canonical_runtime_input_abi
from .utils import atomic_write_json, file_record, require_run_output, resolve_callable


@contextmanager
def _working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _tensor_record(value: Any) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        return {
            "kind": "tensor",
            "shape": list(value.shape),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "requires_grad": bool(value.requires_grad),
        }
    return {"kind": type(value).__name__}


def _module_version(module_name: str) -> str | None:
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    value = getattr(module, "__version__", None)
    return None if value is None else str(value)


def _normalize_specs(value: Any) -> tuple[AirGraphSpec, ...]:
    if isinstance(value, AirGraphSpec):
        specs = (value,)
    elif isinstance(value, Iterable):
        specs = tuple(value)
    else:
        raise TypeError("AIR factory must return AirGraphSpec or an iterable of them")
    if not specs:
        raise ValueError("AIR factory returned no graphs")
    if not all(isinstance(item, AirGraphSpec) for item in specs):
        raise TypeError("AIR factory returned a non-AirGraphSpec item")
    names = [item.name for item in specs]
    if len(set(names)) != len(names):
        raise ValueError("AIR graph names must be unique")
    return specs


def export_air_bundle(
    factory: str | Callable[[Mapping[str, Any]], Sequence[AirGraphSpec]],
    factory_config: Mapping[str, Any],
    bundle_dir: str | Path,
    *,
    torchair_module: Any | None = None,
) -> dict[str, Any]:
    """Export every graph from ``factory`` and retain a hash-complete manifest."""

    root = require_run_output(bundle_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"AIR bundle directory is not empty: {root}")
    torchair = torchair_module
    if torchair is None:
        try:
            torchair = importlib.import_module("torchair")
        except ImportError as error:
            raise RuntimeError(
                "TorchAir is required for AIR export; activate the declared CANN/TorchAir environment"
            ) from error

    # Import TorchAir before invoking the factory. The production factory loads
    # both 4B checkpoints, so a missing export runtime must fail before that
    # expensive and memory-heavy operation starts.
    factory_callable = resolve_callable(factory)
    prepare = getattr(factory_callable, "prepare_export", None)
    preflight = prepare(dict(factory_config), torchair) if callable(prepare) else None
    specs = _normalize_specs(factory_callable(dict(factory_config)))
    from .incremental_plan import validate_incremental_bundle
    validate_incremental_bundle([
        {"name": spec.name, "role": spec.role, "input_names": list(spec.input_names),
         "output_names": list(spec.output_names), "metadata": dict(spec.metadata)}
        for spec in specs
    ])
    root.mkdir(parents=True, exist_ok=True)
    air_root = root / "air"
    air_root.mkdir()

    graphs: list[dict[str, Any]] = []
    for spec in specs:
        graph_dir = air_root / spec.name
        graph_dir.mkdir()
        custom_op_sessions = [
            prepare_custom_op_export(item, torchair) for item in spec.custom_ops
        ]
        softplus_session = (
            prepare_aten_softplus_export(torchair)
            if spec.metadata.get("standard_op_export_contracts") else None
        )
        call_kwargs = {
            "model": spec.model.eval(),
            "export_path": str(graph_dir),
            "export_name": spec.name,
            "dynamic": bool(spec.dynamic),
        }
        if spec.compiler_config is not None:
            call_kwargs["config"] = spec.compiler_config
        call_kwargs.update(dict(spec.example_kwargs))
        input_abi_context = (
            canonical_runtime_input_abi(
                torchair, public_inputs=spec.example_args,
                public_names=spec.input_names,
                explicit_test_double=torchair_module is not None,
                require_static_shapes=True,
            )
            if spec.metadata.get("incremental_contract") else nullcontext(None)
        )
        with (
            torch.inference_mode(), _working_directory(graph_dir),
            input_abi_context as runtime_input_abi,
        ):
            torchair.dynamo_export(*spec.example_args, **call_kwargs)

        custom_op_audit = audit_custom_op_export(
            custom_op_sessions,
            graph_dir,
            relative_to=root,
        )

        standard_op_audit = [] if softplus_session is None else [
            audit_aten_softplus_export(softplus_session, graph_dir,
                                      calls_before=0, relative_to=root)
        ]

        air_files = sorted(graph_dir.glob("*.air"))
        if len(air_files) != 1:
            raise RuntimeError(
                f"TorchAir export for {spec.name!r} produced {len(air_files)} AIR files"
            )
        payload_files = sorted(path for path in graph_dir.rglob("*") if path.is_file())
        records = [file_record(path, relative_to=root) for path in payload_files]
        air_record = next(
            item for item in records if item["path"] == air_files[0].relative_to(root).as_posix()
        )
        graphs.append(
            {
                "name": spec.name,
                "role": spec.role,
                "dynamic": bool(spec.dynamic),
                "model_class": f"{type(spec.model).__module__}.{type(spec.model).__qualname__}",
                "input_names": list(spec.input_names),
                "output_names": list(spec.output_names),
                "example_args": [_tensor_record(item) for item in spec.example_args],
                "example_kwargs": {
                    name: _tensor_record(item)
                    for name, item in spec.example_kwargs.items()
                },
                "metadata": dict(spec.metadata),
                **({"runtime_input_abi": runtime_input_abi}
                   if runtime_input_abi is not None else {}),
                "custom_op_audit": custom_op_audit,
                "standard_op_overrides": standard_op_audit,
                "air": air_record,
                "payload_files": records,
            }
        )

    factory_name = (
        factory
        if isinstance(factory, str)
        else f"{factory_callable.__module__}:{factory_callable.__qualname__}"
    )
    manifest = {
        "schema_version": 2,
        "artifact_kind": "qwen35-dflash-torchair-bundle",
        "status": "PASS",
        "factory": factory_name,
        "factory_config": dict(factory_config),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_npu": _module_version("torch_npu"),
            "torchair": str(getattr(torchair, "__version__", "unknown")),
        },
        "operator_preflight": preflight,
        "graphs": graphs,
    }
    manifest_path = atomic_write_json(root / "air-manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
