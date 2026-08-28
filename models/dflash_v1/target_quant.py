"""Original ``inference.py --quant_mode enable`` contracts for DFlash.

The deployment route is fixed rather than user-pluggable: the original
``utils.quant_model(model, quanted_pth)`` converts Target linears, while the
same YAML supplies raw INT8 embedding weights and FP32 row scales.  DFlash
loads those artifacts, applies the original ``INT8 * FP32 -> FP16`` embedding
boundary to each current token block, and audits the resulting ``QLinear``
topology.  No quantizer or input-provider callback is exposed on the CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import importlib
import inspect
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch import Tensor, nn


TARGET_QUANT_MODE_ENV = "DFLASH_TARGET_QUANT_MODE"
TARGET_QUANT_CONFIG_ENV = "DFLASH_TARGET_QUANT_CONFIG"
TARGET_QUANT_WEIGHT_PATH_ENV = "DFLASH_TARGET_QUANT_WEIGHT_PATH"
TARGET_EMBEDDING_WEIGHT_PATH_ENV = "DFLASH_TARGET_EMBEDDING_WEIGHT_PATH"
TARGET_EMBEDDING_SCALE_PATH_ENV = "DFLASH_TARGET_EMBEDDING_SCALE_PATH"
ORIGINAL_QUANTIZER_SPEC = "models.dflash_v1.original_quant:quant_model"

QUANT_MODE_DISABLED = "disabled"
QUANT_MODE_W8A8_DYNAMIC = "w8a8_dynamic"
SUPPORTED_TARGET_QUANT_MODES = (
    QUANT_MODE_DISABLED,
    QUANT_MODE_W8A8_DYNAMIC,
)


@dataclass(frozen=True)
class TargetQuantizationRequest:
    """Normalized process-local request consumed by the target factory."""

    mode: str
    config_path: Path | None
    quant_weight_path: Path | None
    embedding_weight_path: Path | None
    embedding_scale_path: Path | None

    @property
    def enabled(self) -> bool:
        return self.mode != QUANT_MODE_DISABLED

    @classmethod
    def from_environment(cls) -> "TargetQuantizationRequest":
        mode = os.environ.get(TARGET_QUANT_MODE_ENV, QUANT_MODE_DISABLED).strip()
        if mode not in SUPPORTED_TARGET_QUANT_MODES:
            raise ValueError(
                f"{TARGET_QUANT_MODE_ENV} must be one of "
                f"{SUPPORTED_TARGET_QUANT_MODES}; got {mode!r}"
            )
        config_text = _optional_text(os.environ.get(TARGET_QUANT_CONFIG_ENV))
        quant_weight_text = _optional_text(
            os.environ.get(TARGET_QUANT_WEIGHT_PATH_ENV)
        )
        embedding_weight_text = _optional_text(
            os.environ.get(TARGET_EMBEDDING_WEIGHT_PATH_ENV)
        )
        embedding_scale_text = _optional_text(
            os.environ.get(TARGET_EMBEDDING_SCALE_PATH_ENV)
        )
        if mode == QUANT_MODE_DISABLED:
            stale = [
                name
                for name, value in (
                    (TARGET_QUANT_CONFIG_ENV, config_text),
                    (TARGET_QUANT_WEIGHT_PATH_ENV, quant_weight_text),
                    (TARGET_EMBEDDING_WEIGHT_PATH_ENV, embedding_weight_text),
                    (TARGET_EMBEDDING_SCALE_PATH_ENV, embedding_scale_text),
                )
                if value is not None
            ]
            if stale:
                raise ValueError(
                    "target quantization is disabled but quantization settings "
                    "remain configured: " + ", ".join(stale)
                )
            return cls(mode, None, None, None, None)

        missing = [
            name
            for name, value in (
                (TARGET_QUANT_CONFIG_ENV, config_text),
                (TARGET_QUANT_WEIGHT_PATH_ENV, quant_weight_text),
                (TARGET_EMBEDDING_WEIGHT_PATH_ENV, embedding_weight_text),
                (TARGET_EMBEDDING_SCALE_PATH_ENV, embedding_scale_text),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"{mode} requires " + ", ".join(missing)
            )
        assert config_text is not None
        assert quant_weight_text is not None
        assert embedding_weight_text is not None
        assert embedding_scale_text is not None
        return cls(
            mode=mode,
            config_path=_validated_regular_file(
                config_text,
                label="original inference YAML",
            ),
            quant_weight_path=_validated_directory(
                quant_weight_text,
                label="target quantization weight path",
            ),
            embedding_weight_path=_validated_regular_file(
                embedding_weight_text,
                label="target embedding weight path",
            ),
            embedding_scale_path=_validated_regular_file(
                embedding_scale_text,
                label="target embedding scale path",
            ),
        )


@dataclass(frozen=True)
class OriginalQuantConfig:
    """Three artifact paths read from the original inference YAML."""

    config_path: Path
    quant_weight_path: Path
    embedding_weight_path: Path
    embedding_scale_path: Path


@dataclass(frozen=True)
class TargetQuantizationResult:
    """Result returned by the deployment-owned quantizer callback.

    ``expected_qlinear_paths`` is the complete set of modules the artifact is
    intended to replace.  The bridge compares it with the actual model rather
    than accepting a partially converted target merely because one QLinear is
    present.

    The optional draft modules let a converter retain or separately load the
    FP16 tied embedding/LM-head view required by the current six-layer Draft.
    When omitted, the bridge uses the modules captured immediately before
    quantization and verifies that the callback did not mutate them.
    """

    execution_model: nn.Module
    expected_qlinear_paths: Sequence[str]
    profile: Mapping[str, object]
    draft_input_embeddings: nn.Module | None = None
    draft_output_embeddings: nn.Module | None = None


@dataclass(frozen=True)
class LinearTopologyEntry:
    """Shape contract captured before one ``nn.Linear`` is converted.

    The NPU ``QLinear`` stores its INT8 weight as ``[in_features,
    out_features]`` while ``nn.Linear.weight`` uses
    ``[out_features, in_features]``.  Capturing this information before the
    deployment callback runs lets the bridge detect a transposed or unrelated
    artifact without interpreting the artifact itself.
    """

    in_features: int
    out_features: int
    has_bias: bool


def _optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _validated_data_path(value: str, *, label: str) -> Path:
    """Resolve one deployment-owned file or directory without following a leaf link."""

    path = Path(value).expanduser()
    if path.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file() and not path.is_dir():
        raise ValueError(f"{label} must be a regular file or directory")
    return path.resolve()


def _validated_regular_file(value: str, *, label: str) -> Path:
    path = _validated_data_path(value, label=label)
    if not path.is_file():
        raise ValueError(f"{label} must be a regular file")
    return path


def _validated_directory(value: str, *, label: str) -> Path:
    path = _validated_data_path(value, label=label)
    if not path.is_dir():
        raise ValueError(f"{label} must be a directory")
    return path


def _resolve_yaml_data_path(
    value: object,
    *,
    config_path: Path,
    label: str,
    regular_file: bool,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{label} in {config_path} must be a non-empty path")
    raw = Path(value.strip()).expanduser()
    candidates = (
        [raw]
        if raw.is_absolute()
        else [Path.cwd() / raw, config_path.parent / raw]
    )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    for candidate in unique:
        if candidate.exists():
            validator = (
                _validated_regular_file
                if regular_file
                else _validated_data_path
            )
            return validator(str(candidate), label=label)
    searched = ", ".join(str(candidate) for candidate in unique)
    raise FileNotFoundError(f"{label} does not exist; searched: {searched}")


def load_original_quant_config(value: str | Path) -> OriginalQuantConfig:
    """Load the exact three quant fields consumed by the original inference."""

    config_path = _validated_regular_file(
        str(value),
        label="original inference YAML",
    )
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("PyYAML is required to read --config") from error
    with config_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    if not isinstance(payload, Mapping):
        raise TypeError("original inference YAML must contain a mapping")
    required = ("quanted_pth", "embedding_weight_path", "embedding_scale_path")
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(
            "original inference YAML is missing " + ", ".join(missing)
        )
    quant_weight_path = _resolve_yaml_data_path(
        payload["quanted_pth"],
        config_path=config_path,
        label="config.quanted_pth",
        regular_file=False,
    )
    if not quant_weight_path.is_dir():
        raise ValueError("config.quanted_pth must be a directory")
    return OriginalQuantConfig(
        config_path=config_path,
        quant_weight_path=quant_weight_path,
        embedding_weight_path=_resolve_yaml_data_path(
            payload["embedding_weight_path"],
            config_path=config_path,
            label="config.embedding_weight_path",
            regular_file=True,
        ),
        embedding_scale_path=_resolve_yaml_data_path(
            payload["embedding_scale_path"],
            config_path=config_path,
            label="config.embedding_scale_path",
            regular_file=True,
        ),
    )


def load_callback(
    specification: str,
    *,
    label: str,
) -> tuple[Callable[..., Any], dict[str, object]]:
    """Load one reviewed ``MODULE:FUNCTION`` callback and report its identity."""

    if not isinstance(specification, str) or specification.count(":") != 1:
        raise ValueError(f"{label} must use MODULE:FUNCTION syntax")
    module_name, function_name = specification.split(":", 1)
    if not module_name or not function_name or "." in function_name:
        raise ValueError(f"{label} must use MODULE:FUNCTION syntax")
    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"{label} is not callable: {specification}")
    source = inspect.getsourcefile(function) or inspect.getfile(function)
    return function, {
        "specification": specification,
        "module": getattr(function, "__module__", module_name),
        "qualname": getattr(function, "__qualname__", function_name),
        "source_file": str(Path(source).resolve()) if source else None,
    }


def _select_callback_abi(
    function: Callable[..., Any],
    *,
    positional: tuple[object, ...],
    extended_keywords: Mapping[str, object],
    label: str,
    expected: str,
) -> str:
    """Select a supported callback ABI without executing the callback."""

    signature = inspect.signature(function)
    try:
        signature.bind(*positional, **extended_keywords)
    except TypeError:
        try:
            signature.bind(*positional)
        except TypeError as error:
            raise TypeError(f"{label} must accept {expected}") from error
        return "simple"
    return "extended"


def quantizer_callback_abi(function: Callable[..., Any]) -> str:
    """Validate the fixed original ``quant_model`` callable ABI."""

    return _select_callback_abi(
        function,
        positional=(object(), "quant-weight-path"),
        extended_keywords={
            "device": torch.device("cpu"),
            "output_dtype": torch.float16,
        },
        label="target quantizer",
        expected=(
            "(model, quant_weight_path) or (model, quant_weight_path, *, "
            "device, output_dtype)"
        ),
    )


def preconversion_linear_topology(
    execution_model: nn.Module,
) -> dict[str, LinearTopologyEntry]:
    """Freeze every text target Linear path, shape, and bias contract."""

    topology: dict[str, LinearTopologyEntry] = {}
    for name, module in execution_model.named_modules():
        if not name or not isinstance(module, nn.Linear):
            continue
        weight = _module_weight(module, name=f"pre-conversion Linear {name}")
        expected_shape = (int(module.out_features), int(module.in_features))
        if weight.ndim != 2 or tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"pre-conversion Linear {name}.weight must have shape "
                f"{expected_shape}; got {tuple(weight.shape)}"
            )
        topology[name] = LinearTopologyEntry(
            in_features=int(module.in_features),
            out_features=int(module.out_features),
            has_bias=module.bias is not None,
        )
    if not topology:
        raise RuntimeError("unquantized target exposes no nn.Linear modules")
    return topology


def preconversion_linear_paths(execution_model: nn.Module) -> tuple[str, ...]:
    """Return the exact pre-conversion Linear paths (compatibility helper)."""

    return tuple(preconversion_linear_topology(execution_model))


def invoke_quantizer(
    function: Callable[..., Any],
    execution_model: nn.Module,
    quant_weight_path: Path,
    *,
    device: torch.device,
    output_dtype: torch.dtype,
) -> object:
    """Call either the existing two-argument quantizer or the extended ABI.

    Signature binding selects the ABI before execution, so a ``TypeError``
    raised inside the quantizer is never mistaken for an argument mismatch.
    """

    positional = (execution_model, str(quant_weight_path))
    if quantizer_callback_abi(function) == "simple":
        return function(*positional)
    return function(
        *positional,
        device=device,
        output_dtype=output_dtype,
    )


def normalize_quantizer_result(
    value: object,
    *,
    original_execution_model: nn.Module,
    default_expected_qlinear_paths: Sequence[str],
) -> TargetQuantizationResult:
    """Normalize the explicit contract or an existing all-linear converter."""

    if value is None:
        value = original_execution_model
    if isinstance(value, nn.Module):
        value = TargetQuantizationResult(
            execution_model=value,
            expected_qlinear_paths=tuple(default_expected_qlinear_paths),
            profile={
                "conversion_scope": "all_preexisting_nn_linear",
                "quantizer_return": "nn.Module_or_in_place",
            },
        )
    if not isinstance(value, TargetQuantizationResult):
        raise TypeError(
            "target quantizer must return nn.Module, None for in-place conversion, "
            "or TargetQuantizationResult"
        )
    if not isinstance(value.execution_model, nn.Module):
        raise TypeError("quantizer result execution_model must be nn.Module")
    return value


def _module_weight(module: nn.Module, *, name: str) -> Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError(f"{name} must expose Tensor weight")
    return weight


def _json_profile(profile: Mapping[str, object]) -> dict[str, object]:
    if not isinstance(profile, Mapping):
        raise TypeError("target quantization profile must be a mapping")
    result: dict[str, object] = {}
    for key, value in profile.items():
        if not isinstance(key, str) or not key:
            raise TypeError("target quantization profile keys must be non-empty strings")
        if value is not None and not isinstance(value, (str, int, float, bool)):
            raise TypeError(
                "target quantization profile values must be JSON scalar values"
            )
        result[key] = value
    return result


def audit_quantized_target(
    result: TargetQuantizationResult,
    *,
    qlinear_type: type[nn.Module],
    original_linear_topology: Mapping[str, LinearTopologyEntry],
    draft_input_embeddings: nn.Module,
    draft_output_embeddings: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    vocab_size: int,
    hidden_size: int,
) -> dict[str, object]:
    """Fail closed on partial conversion or unusable Draft shared weights."""

    if not isinstance(qlinear_type, type) or not issubclass(qlinear_type, nn.Module):
        raise TypeError("qlinear_type must be an nn.Module class")
    expected_paths = tuple(result.expected_qlinear_paths)
    if not expected_paths or any(
        not isinstance(path, str) or not path for path in expected_paths
    ):
        raise ValueError("expected_qlinear_paths must contain non-empty module paths")
    if len(set(expected_paths)) != len(expected_paths):
        raise ValueError("expected_qlinear_paths contains duplicates")

    if not isinstance(original_linear_topology, Mapping) or not original_linear_topology:
        raise ValueError("original_linear_topology must be a non-empty mapping")
    topology = dict(original_linear_topology)
    if any(
        not isinstance(path, str)
        or not path
        or not isinstance(entry, LinearTopologyEntry)
        for path, entry in topology.items()
    ):
        raise TypeError(
            "original_linear_topology must map module paths to "
            "LinearTopologyEntry values"
        )

    topology_set = set(topology)
    expected_set = set(expected_paths)
    unknown_expected = sorted(expected_set - topology_set)
    if unknown_expected:
        raise RuntimeError(
            "quantizer manifest contains paths that were not pre-conversion "
            f"nn.Linear modules: {unknown_expected}"
        )

    modules = dict(result.execution_model.named_modules())
    observed = {
        name: module
        for name, module in modules.items()
        if isinstance(module, qlinear_type)
    }
    observed_set = set(observed)
    if observed_set != expected_set:
        missing = sorted(expected_set - observed_set)
        unexpected = sorted(observed_set - expected_set)
        raise RuntimeError(
            "quantized target QLinear coverage differs from the converter "
            f"manifest: missing={missing}, unexpected={unexpected}"
        )

    expected_passthrough = topology_set - expected_set
    observed_passthrough = {
        name
        for name, module in modules.items()
        if name and isinstance(module, nn.Linear)
    }
    if observed_passthrough != expected_passthrough:
        missing = sorted(expected_passthrough - observed_passthrough)
        unexpected = sorted(observed_passthrough - expected_passthrough)
        raise RuntimeError(
            "post-conversion nn.Linear topology differs from the frozen target: "
            f"missing={missing}, unexpected={unexpected}"
        )

    scale_dtypes: set[str] = set()
    scale_layouts: set[str] = set()
    for path, module in sorted(observed.items()):
        entry = topology[path]
        if entry.has_bias:
            raise RuntimeError(
                f"{path} had a bias before conversion, but the current QLinear "
                "ABI has no bias input"
            )
        weight = getattr(module, "W_q", None)
        scale = getattr(module, "scale", None)
        if not isinstance(weight, Tensor) or weight.dtype is not torch.int8:
            raise TypeError(f"{path}.W_q must be an INT8 Tensor")
        if not isinstance(scale, Tensor) or not torch.is_floating_point(scale):
            raise TypeError(f"{path}.scale must be a floating-point Tensor")
        expected_weight_shape = (entry.in_features, entry.out_features)
        if weight.ndim != 2 or tuple(weight.shape) != expected_weight_shape:
            raise ValueError(
                f"{path}.W_q must use [in_features,out_features] layout "
                f"{expected_weight_shape}; got {tuple(weight.shape)}"
            )
        if scale.ndim != 1 or scale.numel() not in {1, entry.out_features}:
            raise ValueError(
                f"{path}.scale must be one-dimensional with 1 or "
                f"{entry.out_features} elements; got shape {tuple(scale.shape)}"
            )
        if not bool(torch.isfinite(scale).all()):
            raise FloatingPointError(f"{path}.scale contains non-finite values")
        if weight.device != device or scale.device != device:
            raise ValueError(
                f"{path} quantization buffers must be on {device}; got "
                f"W_q={weight.device}, scale={scale.device}"
            )
        scale_dtypes.add(str(scale.dtype))
        scale_layouts.add(
            "per_tensor" if scale.numel() == 1 else "per_output_channel"
        )

    for path in sorted(expected_passthrough):
        entry = topology[path]
        module = modules[path]
        assert isinstance(module, nn.Linear)
        weight = _module_weight(module, name=f"passthrough Linear {path}")
        expected_shape = (entry.out_features, entry.in_features)
        if tuple(weight.shape) != expected_shape:
            raise ValueError(
                f"passthrough Linear {path}.weight must retain shape "
                f"{expected_shape}; got {tuple(weight.shape)}"
            )
        if (module.bias is not None) != entry.has_bias:
            raise RuntimeError(
                f"passthrough Linear {path} changed its bias contract"
            )
        if weight.device != device or weight.dtype != dtype:
            raise ValueError(
                f"passthrough Linear {path}.weight must remain {dtype} on "
                f"{device}; got {weight.dtype}/{weight.device}"
            )

    expected_weight_shape = (int(vocab_size), int(hidden_size))
    draft_weights = {
        "input_embedding": _module_weight(
            draft_input_embeddings,
            name="Draft input embedding",
        ),
        "lm_head": _module_weight(
            draft_output_embeddings,
            name="Draft LM head",
        ),
    }
    for name, weight in draft_weights.items():
        if tuple(weight.shape) != expected_weight_shape:
            raise ValueError(
                f"Draft {name} weight must have shape {expected_weight_shape}; "
                f"got {tuple(weight.shape)}"
            )
        if not torch.is_floating_point(weight) or weight.dtype != dtype:
            raise TypeError(f"Draft {name} weight must remain {dtype}")
        if weight.device != device:
            raise ValueError(
                f"Draft {name} weight is on {weight.device}, expected {device}"
            )

    return {
        "status": "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM",
        "scheme": QUANT_MODE_W8A8_DYNAMIC,
        "qlinear_count": len(observed),
        "qlinear_paths": sorted(observed),
        "preconversion_linear_count": len(topology),
        "passthrough_linear_count": len(expected_passthrough),
        "passthrough_linear_paths": sorted(expected_passthrough),
        "linear_topology_validation": "PASS_EXACT_PATH_SHAPE_BIAS",
        "quantized_weight_layout": "K_by_N",
        "quantized_weight_dtype": "torch.int8",
        "quantized_scale_dtypes": sorted(scale_dtypes),
        "quantized_scale_layouts": sorted(scale_layouts),
        "linear_output_dtype": str(dtype),
        "draft_embedding_dtype": str(draft_weights["input_embedding"].dtype),
        "draft_lm_head_dtype": str(draft_weights["lm_head"].dtype),
        "profile": _json_profile(result.profile),
    }


class OriginalQuantizedEmbedding(nn.Module):
    """Raw embedding artifacts used by the original ``quant_mode=enable``."""

    def __init__(
        self,
        quant_weight: Tensor,
        quant_scale: Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if quant_weight.ndim != 2 or quant_weight.dtype is not torch.int8:
            raise TypeError("quantized embedding weight must be INT8 [V,H]")
        if quant_scale.ndim != 2 or quant_scale.dtype is not torch.float32:
            raise TypeError("quantized embedding scale must be FP32 [V,S]")
        if quant_scale.shape[0] != quant_weight.shape[0]:
            raise ValueError("embedding weight and scale vocabulary sizes differ")
        if quant_scale.shape[1] not in {1, quant_weight.shape[1]}:
            raise ValueError("embedding scale width must be 1 or hidden_size")
        if output_dtype is not torch.float16:
            raise TypeError("original HIAI quant embedding output must be FP16")
        if not bool(torch.isfinite(quant_scale).all()):
            raise FloatingPointError("quantized embedding scale is non-finite")
        self.register_buffer("quant_weight", quant_weight.contiguous())
        self.register_buffer("quant_scale", quant_scale.contiguous())
        self.output_dtype = output_dtype

    @property
    def audit(self) -> Mapping[str, object]:
        return {
            "embedding_input_backend": "builtin_original_inference",
            "embedding_weight_shape": list(self.quant_weight.shape),
            "embedding_weight_dtype": str(self.quant_weight.dtype),
            "embedding_scale_shape": list(self.quant_scale.shape),
            "embedding_scale_dtype": str(self.quant_scale.dtype),
            "embedding_dequantization": "int8_mul_fp32_scale_then_fp16",
        }

    def forward(self, input_ids: Tensor) -> Tensor:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("quantized embedding input_ids must have shape [1,T]")
        if input_ids.dtype != torch.long:
            raise TypeError("quantized embedding input_ids must use torch.long")
        if input_ids.device != self.quant_weight.device:
            raise ValueError("input_ids and quantized embedding must share a device")
        quant_rows = self.quant_weight[input_ids]
        scale_rows = self.quant_scale[input_ids]
        hidden = quant_rows.to(torch.float32) * scale_rows
        return hidden.to(self.output_dtype)


def load_original_quantized_embedding(
    weight_path: Path,
    scale_path: Path,
    *,
    vocab_size: int,
    hidden_size: int,
    device: torch.device,
    output_dtype: torch.dtype,
) -> OriginalQuantizedEmbedding:
    """Load the raw ``np.tofile`` artifacts produced by ``inference.py``."""

    weight_path = _validated_regular_file(
        str(weight_path),
        label="config.embedding_weight_path",
    )
    scale_path = _validated_regular_file(
        str(scale_path),
        label="config.embedding_scale_path",
    )
    expected_weight_values = int(vocab_size) * int(hidden_size)
    if weight_path.stat().st_size != expected_weight_values:
        raise ValueError(
            "INT8 embedding file size does not match "
            f"[{vocab_size},{hidden_size}]"
        )
    if scale_path.stat().st_size % (int(vocab_size) * 4):
        raise ValueError("FP32 embedding scale file cannot reshape to [vocab,-1]")
    scale_width = scale_path.stat().st_size // (int(vocab_size) * 4)
    if scale_width not in {1, int(hidden_size)}:
        raise ValueError("embedding scale width must be 1 or hidden_size")
    weight_array = np.fromfile(weight_path, dtype=np.int8)
    scale_array = np.fromfile(scale_path, dtype=np.float32)
    weight = torch.from_numpy(weight_array).reshape(vocab_size, hidden_size)
    scale = torch.from_numpy(scale_array).reshape(vocab_size, scale_width)
    module = OriginalQuantizedEmbedding(
        weight,
        scale,
        output_dtype=output_dtype,
    )
    return module.to(device).eval()


__all__ = [
    "QUANT_MODE_DISABLED",
    "QUANT_MODE_W8A8_DYNAMIC",
    "ORIGINAL_QUANTIZER_SPEC",
    "SUPPORTED_TARGET_QUANT_MODES",
    "TARGET_EMBEDDING_SCALE_PATH_ENV",
    "TARGET_EMBEDDING_WEIGHT_PATH_ENV",
    "TARGET_QUANT_CONFIG_ENV",
    "TARGET_QUANT_MODE_ENV",
    "TARGET_QUANT_WEIGHT_PATH_ENV",
    "OriginalQuantConfig",
    "OriginalQuantizedEmbedding",
    "TargetQuantizationRequest",
    "TargetQuantizationResult",
    "LinearTopologyEntry",
    "audit_quantized_target",
    "invoke_quantizer",
    "load_callback",
    "load_original_quant_config",
    "load_original_quantized_embedding",
    "normalize_quantizer_result",
    "preconversion_linear_paths",
    "preconversion_linear_topology",
    "quantizer_callback_abi",
]
