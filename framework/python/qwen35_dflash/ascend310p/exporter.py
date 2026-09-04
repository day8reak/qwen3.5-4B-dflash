"""Export a factory-provided DFlash graph suite to standard TorchAir AIR."""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import platform
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch

from .contracts import AirGraphSpec
from .custom_op_export import (
    NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
    NPU_GATED_DELTA_RULE_MTP_DEFAULT_GE_OP_TYPE,
    NPU_GATED_DELTA_RULE_MTP_TORCH_OP,
    audit_custom_op_export,
    prepare_custom_op_export,
    validate_adn_attention_ge_prototype_environment,
    validate_gdr_ge_prototype_environment,
    validate_gdr_mtp_ge_prototype_environment,
)
from .standard_op_export import (
    audit_aten_softplus_export,
    prepare_aten_softplus_export,
)
from .torchair_compat import index_safe_external_weight_conversion
from .utils import atomic_write_json, file_record, require_run_output, resolve_callable


_SOFTPLUS_CONSUMER_TORCH_OPS = frozenset(
    {
        NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
        NPU_GATED_DELTA_RULE_MTP_TORCH_OP,
    }
)


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


def _set_input_dim_gears(spec: AirGraphSpec, torchair: Any) -> None:
    if not spec.input_dim_gears:
        return
    inference = getattr(torchair, "inference", None)
    setter = getattr(inference, "set_dim_gears", None)
    if setter is None:
        try:
            inference = importlib.import_module("torchair.inference")
        except ImportError as error:
            raise RuntimeError(
                "TorchAir dimension gears require torchair.inference.set_dim_gears"
            ) from error
        setter = getattr(inference, "set_dim_gears", None)
    if not callable(setter):
        raise RuntimeError(
            "TorchAir dimension gears require torchair.inference.set_dim_gears"
        )
    for input_index, dimensions in spec.input_dim_gears.items():
        setter(
            spec.example_args[input_index],
            {axis: list(gears) for axis, gears in dimensions.items()},
        )


def _softplus_minimum_occurrences(spec: AirGraphSpec) -> int:
    """Return whether this graph executes a Target Gated DeltaNet path."""

    return int(
        any(
            item.minimum_occurrences > 0
            and item.torch_op in _SOFTPLUS_CONSUMER_TORCH_OPS
            for item in spec.custom_ops
        )
    )


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
    explicit_test_double = bool(
        torchair_module is not None
        and getattr(
            torchair_module,
            "_qwen35_dflash_explicit_test_double",
            False,
        )
    )
    if torchair is None:
        try:
            torchair = importlib.import_module("torchair")
        except ImportError as error:
            raise RuntimeError(
                "TorchAir is required for AIR export; activate the declared CANN/TorchAir environment"
            ) from error

    gdr_ge_prototype = validate_gdr_ge_prototype_environment()
    adn_attention_ge_prototype = (
        validate_adn_attention_ge_prototype_environment()
    )

    # Receiver TorchAir releases either leave aten.softplus.default
    # unimplemented or lower it themselves. Register the exact override before
    # tracing, then audit the serialized GE outcome rather than relying only on
    # which converter won the release-specific registration order.
    softplus_export = prepare_aten_softplus_export(torchair)

    # Import TorchAir before invoking the factory. The production factory loads
    # both 4B checkpoints, so a missing export runtime must fail before that
    # expensive and memory-heavy operation starts.
    factory_callable = resolve_callable(factory)
    specs = _normalize_specs(factory_callable(dict(factory_config)))
    requires_gdr_mtp = any(
        custom_op.torch_op == NPU_GATED_DELTA_RULE_MTP_TORCH_OP
        for spec in specs
        for custom_op in spec.custom_ops
    )
    gdr_mtp_ge_prototype = (
        validate_gdr_mtp_ge_prototype_environment()
        if requires_gdr_mtp
        else {
            "status": "NOT_REQUIRED",
            "ge_op_type": NPU_GATED_DELTA_RULE_MTP_DEFAULT_GE_OP_TYPE,
            "reason": "no graph declares npu::npu_gated_delta_rule_mtp",
        }
    )
    root.mkdir(parents=True, exist_ok=True)
    air_root = root / "air"
    air_root.mkdir()

    graphs: list[dict[str, Any]] = []
    for spec in specs:
        _set_input_dim_gears(spec, torchair)
        softplus_calls_before = softplus_export.converter_calls
        graph_dir = air_root / spec.name
        graph_dir.mkdir()
        custom_op_sessions = [
            prepare_custom_op_export(item, torchair) for item in spec.custom_ops
        ]
        call_kwargs = {
            "model": spec.model.eval(),
            "export_path": str(graph_dir),
            "export_name": spec.name,
            "dynamic": bool(spec.dynamic),
        }
        if spec.compiler_config is not None:
            call_kwargs["config"] = spec.compiler_config
        call_kwargs.update(dict(spec.example_kwargs))
        with (
            torch.inference_mode(),
            _working_directory(graph_dir),
            index_safe_external_weight_conversion(
                torchair,
                required=bool(spec.dynamic),
                explicit_test_double=explicit_test_double,
            ) as external_weight_mapping,
        ):
            torchair.dynamo_export(*spec.example_args, **call_kwargs)

        custom_op_audit = audit_custom_op_export(
            custom_op_sessions,
            graph_dir,
            relative_to=root,
        )
        softplus_audit = audit_aten_softplus_export(
            softplus_export,
            graph_dir,
            calls_before=softplus_calls_before,
            relative_to=root,
            minimum_occurrences=_softplus_minimum_occurrences(spec),
        )

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
                "input_dim_gears": {
                    str(input_index): {
                        str(axis): list(gears)
                        for axis, gears in dimensions.items()
                    }
                    for input_index, dimensions in spec.input_dim_gears.items()
                },
                "model_class": f"{type(spec.model).__module__}.{type(spec.model).__qualname__}",
                "input_names": list(spec.input_names),
                "output_names": list(spec.output_names),
                "example_args": [_tensor_record(item) for item in spec.example_args],
                "example_kwargs": {
                    name: _tensor_record(item)
                    for name, item in spec.example_kwargs.items()
                },
                "metadata": dict(spec.metadata),
                "torchair_external_weight_mapping": (
                    external_weight_mapping.as_manifest_record()
                ),
                "standard_op_overrides": [softplus_audit],
                "custom_op_audit": custom_op_audit,
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
        "schema_version": 3,
        "artifact_kind": "qwen35-dflash-torchair-bundle",
        "status": "PASS",
        "factory": factory_name,
        "factory_config": dict(factory_config),
        "environment": {
            "python": platform.python_version(),
            "torch": str(torch.__version__),
            "torch_npu": _module_version("torch_npu"),
            "torchair": str(getattr(torchair, "__version__", "unknown")),
            "gdr_ge_prototype": gdr_ge_prototype,
            "gdr_mtp_ge_prototype": gdr_mtp_ge_prototype,
            "adn_attention_ge_prototype": adn_attention_ge_prototype,
        },
        "graphs": graphs,
    }
    manifest_path = atomic_write_json(root / "air-manifest.json", manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
