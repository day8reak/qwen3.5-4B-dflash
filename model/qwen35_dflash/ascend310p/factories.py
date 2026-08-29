"""Built-in TorchAir factory for the locked cache-free DFlash draft core."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import nn

from qwen35_dflash.config import Qwen35DFlashConfig
from qwen35_dflash.model import DFlashDraftModel

from .contracts import AirGraphSpec
from .integrated import integrated_recompute_graph_spec
from .resources import resolve_locked_data
from .target_adapter import TransformersDFlashTargetAdapter
from .utils import load_json_object, resolve_callable


_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}


def _boolean(config: Mapping[str, Any], name: str, default: bool) -> bool:
    value = config.get(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a JSON boolean")
    return value


def _checkpoint_resource(
    config: Mapping[str, Any],
    *,
    role: str,
    default_asset_id: str,
) -> tuple[Path, Any | None]:
    directory_key = f"{role}_dir"
    asset_key = f"{role}_asset_id"
    if directory_key in config and asset_key in config:
        raise ValueError(f"provide {directory_key} or {asset_key}, not both")
    if directory_key in config:
        root = Path(str(config[directory_key])).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"{role} checkpoint directory is missing: {root}")
        return root, None
    resource = resolve_locked_data(str(config.get(asset_key, default_asset_id)))
    resource.file("config.json", verify=True)
    general_verify = _boolean(config, "verify_checkpoint", True)
    verify = _boolean(config, f"verify_{role}_checkpoint", general_verify)
    if verify:
        checkpoint_files = [
            str(item["path"])
            for item in resource.manifest.get("files", [])
            if str(item["path"]).startswith("model.safetensors")
        ]
        if not checkpoint_files:
            raise ValueError(f"locked {role} resource declares no safetensors files")
        for relative_path in checkpoint_files:
            resource.file(relative_path, verify=True)
    return resource.root, resource


def _require_device_support(device: str) -> None:
    if not device.startswith("npu"):
        return
    try:
        importlib.import_module("torch_npu")
    except ImportError as error:
        raise RuntimeError("torch_npu is required for an NPU TorchAir export") from error


def _validate_base_pair(
    target_dir: Path,
    draft_dir: Path,
    *,
    target_resource: Any | None,
    draft_resource: Any | None,
) -> tuple[Qwen35DFlashConfig, Mapping[str, Any]]:
    draft_config = Qwen35DFlashConfig.from_pretrained(draft_dir)
    target_config = load_json_object(target_dir / "config.json")
    text_config = target_config.get("text_config", target_config)
    expected = {
        "hidden_size": draft_config.hidden_size,
        "num_hidden_layers": draft_config.num_target_layers,
        "vocab_size": draft_config.vocab_size,
    }
    mismatches = {
        name: {"target": text_config.get(name), "draft": value}
        for name, value in expected.items()
        if int(text_config.get(name, -1)) != int(value)
    }
    if mismatches:
        raise ValueError(f"target/DFlash configuration mismatch: {mismatches}")
    if target_resource is not None and draft_resource is not None:
        target_revision = target_resource.manifest.get("source", {}).get("revision")
        draft_base_revision = draft_resource.manifest.get("source", {}).get(
            "base_model_revision"
        )
        if draft_base_revision != target_revision:
            raise ValueError(
                "DFlash resource was trained for a different target revision: "
                f"{draft_base_revision!r} != {target_revision!r}"
            )
    return draft_config, target_config


def create_integrated_recompute_graph(
    config: Mapping[str, Any],
) -> tuple[AirGraphSpec, ...]:
    """Build the locked text target + official DFlash fixed-gear graph.

    The graph follows the official linear-block convention: row zero is the
    already committed anchor token and rows ``1:`` are draft proposals.  Every
    call recomputes the committed prefix, avoiding any unproven target or draft
    cache commit/rollback semantics.
    """

    if "max_sequence_length" not in config:
        raise ValueError("integrated export requires an explicit max_sequence_length")
    max_sequence_length = int(config["max_sequence_length"])
    if max_sequence_length <= 1:
        raise ValueError("max_sequence_length must exceed one token")
    example_sequence_length = int(config.get("example_sequence_length", 2))
    if not 1 <= example_sequence_length <= max_sequence_length:
        raise ValueError("example_sequence_length is outside the fixed gear")
    dtype_name = str(config.get("dtype", "float16"))
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported integrated AIR dtype: {dtype_name!r}")
    dtype = _DTYPES[dtype_name]
    device = str(config.get("device", "npu:0"))
    _require_device_support(device)
    target_dir, target_resource = _checkpoint_resource(
        config, role="target", default_asset_id="qwen3.5-4b"
    )
    draft_dir, draft_resource = _checkpoint_resource(
        config, role="draft", default_asset_id="qwen3.5-4b-dflash"
    )
    draft_config, target_config = _validate_base_pair(
        target_dir,
        draft_dir,
        target_resource=target_resource,
        draft_resource=draft_resource,
    )

    custom_target_factory = config.get("target_factory")
    if custom_target_factory:
        payload = dict(config.get("target_factory_config", {}))
        payload.update(
            {
                "target_dir": str(target_dir),
                "device": device,
                "dtype": dtype_name,
                "target_layer_ids": list(draft_config.target_layer_ids),
                "target_hidden_size": draft_config.hidden_size,
                "target_num_hidden_layers": draft_config.num_target_layers,
                "vocab_size": draft_config.vocab_size,
            }
        )
        target_model = resolve_callable(str(custom_target_factory))(payload)
        if not isinstance(target_model, nn.Module):
            raise TypeError("custom target factory must return torch.nn.Module")
    else:
        target_model = TransformersDFlashTargetAdapter.from_pretrained(
            target_dir,
            layer_ids=draft_config.target_layer_ids,
            target_hidden_size=draft_config.hidden_size,
            target_num_hidden_layers=draft_config.num_target_layers,
            vocab_size=draft_config.vocab_size,
            device=device,
            dtype=dtype,
            attn_implementation=str(config.get("attn_implementation", "eager")),
        )
    draft_model = DFlashDraftModel.from_pretrained(
        draft_dir,
        device=device,
        dtype=dtype,
    ).eval()
    pad_token_id = int(config.get("pad_token_id", 0))
    if not 0 <= pad_token_id < draft_config.vocab_size:
        raise ValueError("pad_token_id is outside the locked vocabulary")
    target_source = None if target_resource is None else target_resource.manifest["source"]
    draft_source = None if draft_resource is None else draft_resource.manifest["source"]
    metadata = {
        "dtype": dtype_name,
        "target_checkpoint": str(target_dir),
        "target_checkpoint_asset_id": (
            None if target_resource is None else target_resource.asset_id
        ),
        "target_checkpoint_manifest_sha256": (
            None if target_resource is None else target_resource.manifest_sha256
        ),
        "target_source": target_source,
        "draft_checkpoint": str(draft_dir),
        "draft_checkpoint_asset_id": (
            None if draft_resource is None else draft_resource.asset_id
        ),
        "draft_checkpoint_manifest_sha256": (
            None if draft_resource is None else draft_resource.manifest_sha256
        ),
        "draft_source": draft_source,
        "target_architectures": list(target_config.get("architectures", [])),
        "target_feature_layer_ids": list(draft_config.target_layer_ids),
        "target_feature_width": draft_config.feature_size,
        "proposal_tokens": draft_config.block_size - 1,
        "target_adapter": (
            str(custom_target_factory)
            if custom_target_factory
            else "transformers-all-hidden-states-recompute"
        ),
        "semantic_reference": "z-lab/dflash greedy linear-block verification",
    }
    return (
        integrated_recompute_graph_spec(
            target_model,
            draft_model,
            max_sequence_length=max_sequence_length,
            example_sequence_length=example_sequence_length,
            pad_token_id=pad_token_id,
            device=device,
            name=str(config.get("name", "dflash_recompute")),
            metadata=metadata,
        ),
    )


def compose_graph_factories(config: Mapping[str, Any]) -> tuple[AirGraphSpec, ...]:
    """Compose graph specs from the user's existing target and this drafter.

    Example configuration::

        {"components": [
          {"factory": "existing_target.export:create_graphs", "config": {...}},
          {"factory": "qwen35_dflash.ascend310p.factories:create_cache_free_draft_graph",
           "config": {...}}
        ]}
    """

    components = config.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("composed AIR factory needs a non-empty components list")
    graphs: list[AirGraphSpec] = []
    for index, component in enumerate(components):
        if not isinstance(component, Mapping):
            raise TypeError(f"AIR factory component {index} must be an object")
        factory = resolve_callable(str(component["factory"]))
        result = factory(dict(component.get("config", {})))
        if isinstance(result, AirGraphSpec):
            graphs.append(result)
        else:
            graphs.extend(result)
    names = [graph.name for graph in graphs]
    if len(set(names)) != len(names):
        raise ValueError("composed AIR graph names must be unique")
    return tuple(graphs)


def create_cache_free_draft_graph(config: Mapping[str, Any]) -> tuple[AirGraphSpec, ...]:
    """Load the official 69-tensor drafter and one frozen NPZ export case.

    This graph is one component of the end-to-end bundle, not a standalone
    prompt generator.  Existing target projects should return it alongside
    their main prefill and verify/decode graph specs from a project factory.
    """

    if "draft_dir" in config and "draft_asset_id" in config:
        raise ValueError("provide draft_dir or draft_asset_id, not both")
    resource = None
    if "draft_dir" in config:
        draft_dir = Path(str(config["draft_dir"])).expanduser().resolve()
    else:
        resource = resolve_locked_data(
            str(config.get("draft_asset_id", "qwen3.5-4b-dflash"))
        )
        resource.file("config.json", verify=True)
        resource.file(
            "model.safetensors",
            verify=bool(config.get("verify_checkpoint", True)),
        )
        draft_dir = resource.root
    case_path = Path(str(config["case"])).expanduser().resolve()
    dtype_name = str(config.get("dtype", "float16"))
    if dtype_name not in _DTYPES:
        raise ValueError(f"unsupported DFlash AIR dtype: {dtype_name!r}")
    device = str(config.get("device", "npu:0"))
    _require_device_support(device)
    dtype = _DTYPES[dtype_name]
    model = DFlashDraftModel.from_pretrained(
        draft_dir,
        device=device,
        dtype=dtype,
    ).eval()
    with np.load(case_path, allow_pickle=False) as archive:
        required = {"target_hidden", "noise_embedding", "position_ids"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ValueError(f"DFlash AIR case is missing arrays: {missing}")
        target_hidden = torch.from_numpy(archive["target_hidden"]).to(
            device=device, dtype=dtype
        )
        noise_embedding = torch.from_numpy(archive["noise_embedding"]).to(
            device=device, dtype=dtype
        )
        position_ids = torch.from_numpy(archive["position_ids"]).to(
            device=device, dtype=torch.long
        )
    return (
        AirGraphSpec(
            name=str(config.get("name", "dflash_draft_core")),
            role="draft",
            model=model,
            example_args=(target_hidden, noise_embedding, position_ids),
            input_names=("target_hidden", "noise_embedding", "position_ids"),
            output_names=("hidden_states",),
            dynamic=bool(config.get("dynamic", False)),
            metadata={
                "checkpoint": str(draft_dir),
                "checkpoint_asset_id": None if resource is None else resource.asset_id,
                "checkpoint_manifest_sha256": (
                    None if resource is None else resource.manifest_sha256
                ),
                "case": str(case_path),
                "dtype": dtype_name,
                "scope": "cache-free draft core; target prefill/verify graphs still required",
            },
        ),
    )
