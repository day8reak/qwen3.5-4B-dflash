"""Correctness-first CPU/CUDA emulation of the NPU W8A8 ``QLinear`` path.

The emulator intentionally reuses exported ``W_q`` and weight scales from an
already converted target.  It does not quantize floating-point weights and it
does not claim bitwise parity with an NPU until the same activation has been
compared on that device.

For each input row the implemented reference is::

    activation_scale = max(abs(x)) / 127
    x_q = clamp(round(x / activation_scale), -127, 127).to(int8)
    accumulator = int32(x_q) @ int32(W_q)
    y = accumulator * weight_scale * activation_scale

The output is FP16, matching the target ``QLinear`` implementation.  CPU uses
PyTorch's exact INT8-to-INT32 matrix multiply when available and retains an
INT32-conversion fallback for older builds.  CUDA uses chunked FP64 GEMMs for
the integer accumulator: every product and Qwen-sized sum is exactly
representable in FP64, while avoiding any dependency on CUDA integer-GEMM
support.  This is a diagnostic path, not a performance backend.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path

import torch
from torch import Tensor, nn


ARTIFACT_FORMAT = "qwen3.5-w8a8-linear-emulation-v1"
ARTIFACT_SCHEMA_VERSION = 1
ACTIVATION_QMAX = 127
CUDA_OUTPUT_CHUNK = 2048
EMULATION_STATUS = "PASS_FORMULA_ASSEMBLY_NO_REAL_NPU_PARITY"


@dataclass(frozen=True)
class W8A8ArtifactEntry:
    source_path: str
    framework_path: str
    filename: str
    in_features: int
    out_features: int
    weight_scale_elements: int


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TypeError(f"{name} must be a positive integer")
    return value


def _load_json_without_duplicates(path: Path) -> object:
    def pairs_hook(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs_hook)


def _framework_path(source_path: str) -> str:
    if source_path == "lm_head":
        return source_path
    if source_path.startswith("language_model."):
        return "model." + source_path
    raise ValueError(
        "portable W8A8 export supports the text target only; unexpected "
        f"QLinear path {source_path!r}"
    )


def _is_framework_text_linear_path(path: str) -> bool:
    return path == "lm_head" or path.startswith("model.language_model.")


def _module_at(root: nn.Module, path: str) -> nn.Module:
    try:
        module = root.get_submodule(path)
    except (AttributeError, KeyError) as error:
        raise KeyError(f"model has no module at {path!r}") from error
    if not isinstance(module, nn.Module):
        raise TypeError(f"{path!r} did not resolve to torch.nn.Module")
    return module


def _replace_module(root: nn.Module, path: str, replacement: nn.Module) -> None:
    parent_path, separator, child_name = path.rpartition(".")
    if not child_name:
        raise ValueError(f"cannot replace empty module path {path!r}")
    parent = root if not separator else _module_at(root, parent_path)
    if child_name not in parent._modules:
        raise KeyError(f"{path!r} is not a registered child module")
    setattr(parent, child_name, replacement)


def _resolve_quantized_execution_model(target: nn.Module) -> nn.Module:
    """Unwrap the shipped facade/bridge without depending on private classes."""

    current = target
    visited: set[int] = set()
    while id(current) not in visited:
        visited.add(id(current))
        nested = getattr(current, "target", None)
        if isinstance(nested, nn.Module) and nested is not current:
            current = nested
            continue
        execution_model = getattr(current, "dflash_execution_model", None)
        if isinstance(execution_model, nn.Module) and execution_model is not current:
            current = execution_model
            continue
        break
    return current


def dynamic_quantize_per_token(x: Tensor) -> tuple[Tensor, Tensor]:
    """Apply the documented symmetric per-token INT8 dynamic quantization."""

    if not isinstance(x, Tensor) or x.ndim < 2:
        raise ValueError("dynamic quantization input must be a Tensor with rank >= 2")
    if not torch.is_floating_point(x) or x.dtype not in {
        torch.float16,
        torch.bfloat16,
        torch.float32,
    }:
        raise TypeError("dynamic quantization input must be FP16, BF16, or FP32")
    if not bool(torch.isfinite(x).all()):
        raise FloatingPointError("dynamic quantization input contains non-finite values")

    source = x.to(torch.float32)
    maximum = source.abs().amax(dim=-1)
    scale = maximum / float(ACTIVATION_QMAX)
    # A zero row has the exact result q=0, scale=0.  Using one only as the
    # denominator avoids 0/0 without changing either returned tensor.
    denominator = torch.where(scale == 0, torch.ones_like(scale), scale)
    quantized = torch.round(source / denominator.unsqueeze(-1))
    quantized = quantized.clamp(-ACTIVATION_QMAX, ACTIVATION_QMAX).to(torch.int8)
    return quantized, scale.to(torch.float32)


def _validate_w8a8_tensors(weight: Tensor, scale: Tensor) -> tuple[int, int]:
    if not isinstance(weight, Tensor) or weight.dtype is not torch.int8:
        raise TypeError("W_q must be an INT8 Tensor")
    if weight.ndim != 2 or 0 in weight.shape:
        raise ValueError("W_q must have non-empty [in_features,out_features] shape")
    if not isinstance(scale, Tensor) or scale.dtype is not torch.float32:
        raise TypeError("weight scale must be a float32 Tensor")
    if scale.ndim != 1 or scale.numel() not in {1, int(weight.shape[1])}:
        raise ValueError(
            "weight scale must be one-dimensional with 1 or out_features elements"
        )
    if not bool(torch.isfinite(scale).all()):
        raise FloatingPointError("weight scale contains non-finite values")
    in_features, out_features = (int(item) for item in weight.shape)
    maximum_accumulator = in_features * ACTIVATION_QMAX * ACTIVATION_QMAX
    if maximum_accumulator > torch.iinfo(torch.int32).max:
        raise OverflowError(
            "the exact QLinear accumulator could overflow int32 for this K"
        )
    return in_features, out_features


def _exact_accumulator(
    quantized: Tensor,
    weight: Tensor,
    *,
    cuda_output_chunk: int,
) -> Tensor:
    rows = quantized.reshape(-1, quantized.shape[-1])
    if rows.device != weight.device:
        raise ValueError("quantized activation and W_q must use the same device")
    if rows.device.type == "cpu":
        if int(rows.shape[0]) == 0:
            return torch.empty(
                (0, int(weight.shape[1])),
                dtype=torch.int32,
                device=rows.device,
            )
        int_mm = getattr(torch, "_int_mm", None)
        if callable(int_mm):
            # This kernel consumes INT8 operands and returns the exact INT32
            # accumulator.  Besides avoiding a 4x temporary weight copy, it is
            # materially faster for the [2560,248320] Qwen LM head.  The
            # independent CPU validation compares it bitwise with an INT64
            # oracle; this remains a correctness path rather than a speed claim.
            return int_mm(rows.contiguous(), weight.contiguous())
        return torch.matmul(rows.to(torch.int32), weight.to(torch.int32))
    if rows.device.type != "cuda":
        raise ValueError("W8A8 formula emulation supports only CPU and CUDA")

    chunk = _positive_int(cuda_output_chunk, name="cuda_output_chunk")
    # FP64 represents all products and Qwen-sized integer sums exactly.  Keep
    # W_q conversion chunked so the 248320-column LM head does not create a
    # multi-gigabyte temporary double tensor.
    rows_fp64 = rows.to(torch.float64)
    output = torch.empty(
        (int(rows.shape[0]), int(weight.shape[1])),
        dtype=torch.float32,
        device=rows.device,
    )
    for start in range(0, int(weight.shape[1]), chunk):
        end = min(start + chunk, int(weight.shape[1]))
        exact = torch.matmul(rows_fp64, weight[:, start:end].to(torch.float64))
        output[:, start:end] = exact.to(torch.float32)
    return output


def emulate_w8a8_linear(
    x: Tensor,
    weight: Tensor,
    scale: Tensor,
    *,
    output_dtype: torch.dtype = torch.float16,
    cuda_output_chunk: int = CUDA_OUTPUT_CHUNK,
) -> Tensor:
    """Execute the portable formula with the same ``[K,N]`` weight layout."""

    in_features, out_features = _validate_w8a8_tensors(weight, scale)
    if x.shape[-1] != in_features:
        raise ValueError(
            f"activation K={x.shape[-1]} does not match W_q K={in_features}"
        )
    if x.device != weight.device or scale.device != weight.device:
        raise ValueError("activation, W_q, and weight scale must share one device")
    if output_dtype not in {torch.float16, torch.bfloat16}:
        raise TypeError("QLinear emulation output must be FP16 or BF16")

    quantized, pertoken_scale = dynamic_quantize_per_token(x)
    accumulator = _exact_accumulator(
        quantized,
        weight,
        cuda_output_chunk=cuda_output_chunk,
    ).to(torch.float32)
    rows = accumulator.reshape(-1, out_features)
    # Match the public reference order: accumulator * weight scale, followed
    # by the per-token activation scale.
    rows = rows * scale.reshape(1, -1)
    rows = rows * pertoken_scale.reshape(-1, 1)
    return rows.to(output_dtype).reshape(*x.shape[:-1], out_features)


class EmulatedW8A8Linear(nn.Module):
    """Drop-in diagnostic replacement for one bias-free ``nn.Linear``."""

    def __init__(
        self,
        weight: Tensor,
        scale: Tensor,
        *,
        output_dtype: torch.dtype = torch.float16,
        draft_weight: Tensor | None = None,
        cuda_output_chunk: int = CUDA_OUTPUT_CHUNK,
    ) -> None:
        super().__init__()
        in_features, out_features = _validate_w8a8_tensors(weight, scale)
        if weight.device != scale.device:
            raise ValueError("W_q and weight scale must use the same device")
        if draft_weight is not None:
            expected = (out_features, in_features)
            if tuple(draft_weight.shape) != expected:
                raise ValueError(
                    f"Draft-facing FP weight must have shape {expected}; "
                    f"got {tuple(draft_weight.shape)}"
                )
            if not torch.is_floating_point(draft_weight):
                raise TypeError("Draft-facing weight must be floating point")
        self.in_features = in_features
        self.out_features = out_features
        self.output_dtype = output_dtype
        self.cuda_output_chunk = _positive_int(
            cuda_output_chunk,
            name="cuda_output_chunk",
        )
        self.register_buffer("W_q", weight.contiguous())
        self.register_buffer("scale", scale.contiguous())
        # Only the LM head retains this view.  The Draft requires the original
        # floating-point shared head while Target execution uses W_q/scale.
        self._draft_weight = draft_weight

    @property
    def weight(self) -> Tensor | None:
        return self._draft_weight

    @property
    def bias(self) -> None:
        return None

    def forward(self, x: Tensor) -> Tensor:
        return emulate_w8a8_linear(
            x,
            self.W_q,
            self.scale,
            output_dtype=self.output_dtype,
            cuda_output_chunk=self.cuda_output_chunk,
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"output_dtype={self.output_dtype}, bias=False"
        )


def _entry_from_json(value: object, *, index: int) -> W8A8ArtifactEntry:
    if not isinstance(value, Mapping):
        raise TypeError(f"artifact entry {index} must be an object")
    expected = {
        "source_path",
        "framework_path",
        "filename",
        "in_features",
        "out_features",
        "weight_scale_elements",
    }
    if set(value) != expected:
        raise ValueError(
            f"artifact entry {index} fields differ: "
            f"missing={sorted(expected - set(value))}, "
            f"unexpected={sorted(set(value) - expected)}"
        )
    source_path = value["source_path"]
    framework_path = value["framework_path"]
    filename = value["filename"]
    if not all(isinstance(item, str) and item for item in (source_path, framework_path, filename)):
        raise TypeError(f"artifact entry {index} paths must be non-empty strings")
    if Path(filename).name != filename or not filename.endswith(".safetensors"):
        raise ValueError(f"artifact entry {index} has an unsafe filename")
    if framework_path != _framework_path(source_path):
        raise ValueError(f"artifact entry {index} has an invalid framework mapping")
    return W8A8ArtifactEntry(
        source_path=source_path,
        framework_path=framework_path,
        filename=filename,
        in_features=_positive_int(value["in_features"], name="in_features"),
        out_features=_positive_int(value["out_features"], name="out_features"),
        weight_scale_elements=_positive_int(
            value["weight_scale_elements"],
            name="weight_scale_elements",
        ),
    )


def _artifact_manifest(root: Path) -> tuple[dict[str, object], tuple[W8A8ArtifactEntry, ...]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError("W8A8 emulation artifact must be a real directory")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise FileNotFoundError("W8A8 emulation artifact lacks manifest.json")
    raw = _load_json_without_duplicates(manifest_path)
    if not isinstance(raw, Mapping):
        raise TypeError("W8A8 emulation manifest must be a JSON object")
    manifest = dict(raw)
    if manifest.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unsupported W8A8 emulation artifact schema_version")
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError("unsupported W8A8 emulation artifact format")
    required_contract = {
        "scope": "target_linear_only",
        "activation_quantization": "symmetric_dynamic_per_token_int8",
        "activation_qmax": ACTIVATION_QMAX,
        "weight_layout": "K_by_N",
        "weight_dtype": "int8",
        "weight_scale_dtype": "float32",
        "linear_output_dtype": "float16",
        "accumulator": "int32",
    }
    for field, expected in required_contract.items():
        if manifest.get(field) != expected:
            raise ValueError(
                f"W8A8 emulation artifact {field} must be {expected!r}"
            )
    raw_entries = manifest.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("W8A8 emulation manifest has no linear entries")
    entries = tuple(
        _entry_from_json(item, index=index)
        for index, item in enumerate(raw_entries)
    )
    for field, values in (
        ("source_path", [item.source_path for item in entries]),
        ("framework_path", [item.framework_path for item in entries]),
        ("filename", [item.filename for item in entries]),
    ):
        if len(values) != len(set(values)):
            raise ValueError(f"W8A8 artifact contains duplicate {field}")
    passthrough = manifest.get("passthrough_linear_paths")
    if not isinstance(passthrough, list) or any(
        not isinstance(item, str) or not item for item in passthrough
    ):
        raise TypeError("passthrough_linear_paths must be a string list")
    if len(passthrough) != len(set(passthrough)):
        raise ValueError("passthrough_linear_paths contains duplicates")
    if any(item != _framework_path(item.removeprefix("model.")) for item in passthrough):
        raise ValueError("passthrough_linear_paths contains an invalid text mapping")
    expected_files = {"manifest.json", *(item.filename for item in entries)}
    observed_files = {item.name for item in root.iterdir()}
    if observed_files != expected_files:
        raise ValueError(
            "W8A8 artifact file set differs from manifest: "
            f"missing={sorted(expected_files - observed_files)}, "
            f"unexpected={sorted(observed_files - expected_files)}"
        )
    return manifest, entries


def _load_entry(root: Path, entry: W8A8ArtifactEntry) -> tuple[Tensor, Tensor]:
    from safetensors import safe_open

    path = root / entry.filename
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact tensor file is not a real file: {entry.filename}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        if set(handle.keys()) != {"W_q", "scale"}:
            raise ValueError(f"{entry.filename} must contain only W_q and scale")
        metadata = handle.metadata() or {}
        if metadata.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"{entry.filename} artifact format metadata mismatch")
        if metadata.get("source_path") != entry.source_path:
            raise ValueError(f"{entry.filename} source_path metadata mismatch")
        if metadata.get("framework_path") != entry.framework_path:
            raise ValueError(f"{entry.filename} framework_path metadata mismatch")
        weight = handle.get_tensor("W_q")
        scale = handle.get_tensor("scale")
    in_features, out_features = _validate_w8a8_tensors(weight, scale)
    if (in_features, out_features, int(scale.numel())) != (
        entry.in_features,
        entry.out_features,
        entry.weight_scale_elements,
    ):
        raise ValueError(f"{entry.filename} tensor shapes differ from manifest")
    return weight, scale


def export_w8a8_emulation_artifact(
    target: nn.Module,
    destination: str | Path,
    *,
    expected_qlinear_paths: Sequence[str],
) -> dict[str, object]:
    """Export audited QLinear buffers as a bounded-memory artifact directory."""

    from safetensors.torch import save_file

    if not isinstance(target, nn.Module):
        raise TypeError("target must be torch.nn.Module")
    expected = tuple(expected_qlinear_paths)
    if not expected or any(not isinstance(path, str) or not path for path in expected):
        raise ValueError("expected_qlinear_paths must contain non-empty strings")
    if len(expected) != len(set(expected)):
        raise ValueError("expected_qlinear_paths contains duplicates")

    root = Path(destination).expanduser()
    if root.exists() or root.is_symlink():
        raise FileExistsError("W8A8 emulation artifact destination already exists")
    root = root.resolve()
    if not root.parent.is_dir():
        raise FileNotFoundError("parent directory for W8A8 export does not exist")
    temporary = root.with_name(f".{root.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError("temporary W8A8 export directory already exists")

    execution_model = _resolve_quantized_execution_model(target)
    modules = dict(execution_model.named_modules())
    observed_qlinear = {
        name
        for name, module in modules.items()
        if name
        and type(module).__name__ == "QLinear"
        and isinstance(getattr(module, "W_q", None), Tensor)
        and isinstance(getattr(module, "scale", None), Tensor)
    }
    if observed_qlinear != set(expected):
        raise RuntimeError(
            "export QLinear paths differ from the target audit: "
            f"missing={sorted(set(expected) - observed_qlinear)}, "
            f"unexpected={sorted(observed_qlinear - set(expected))}"
        )
    passthrough_source = sorted(
        name
        for name, module in modules.items()
        if name and isinstance(module, nn.Linear)
    )
    passthrough_framework = [_framework_path(path) for path in passthrough_source]

    entries: list[dict[str, object]] = []
    temporary.mkdir(parents=False, mode=0o700)
    try:
        for index, source_path in enumerate(sorted(expected)):
            module = modules[source_path]
            weight = getattr(module, "W_q")
            scale = getattr(module, "scale")
            assert isinstance(weight, Tensor) and isinstance(scale, Tensor)
            in_features, out_features = _validate_w8a8_tensors(weight, scale)
            weight_cpu = weight.detach().to(device="cpu").contiguous()
            scale_cpu = scale.detach().to(device="cpu").contiguous()
            filename = f"linear-{index:04d}.safetensors"
            framework_path = _framework_path(source_path)
            save_file(
                {"W_q": weight_cpu, "scale": scale_cpu},
                str(temporary / filename),
                metadata={
                    "format": ARTIFACT_FORMAT,
                    "source_path": source_path,
                    "framework_path": framework_path,
                },
            )
            entries.append(
                {
                    "source_path": source_path,
                    "framework_path": framework_path,
                    "filename": filename,
                    "in_features": in_features,
                    "out_features": out_features,
                    "weight_scale_elements": int(scale_cpu.numel()),
                }
            )
            del weight_cpu, scale_cpu

        manifest: dict[str, object] = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "format": ARTIFACT_FORMAT,
            "scope": "target_linear_only",
            "activation_quantization": "symmetric_dynamic_per_token_int8",
            "activation_qmax": ACTIVATION_QMAX,
            "weight_layout": "K_by_N",
            "weight_dtype": "int8",
            "weight_scale_dtype": "float32",
            "linear_output_dtype": "float16",
            "accumulator": "int32",
            "entries": entries,
            "passthrough_linear_paths": passthrough_framework,
            "real_npu_numerical_parity": "PENDING_SAME_ACTIVATION_DEVICE_COMPARISON",
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, root)
    finally:
        if temporary.exists():
            for path in temporary.iterdir():
                path.unlink()
            temporary.rmdir()

    return {
        "status": "PASS_EXPORTED_Q_LINEAR_BUFFERS_NO_NUMERICAL_CLAIM",
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_path": str(root),
        "qlinear_count": len(entries),
        "passthrough_linear_count": len(passthrough_framework),
        "real_npu_numerical_parity": "PENDING_SAME_ACTIVATION_DEVICE_COMPARISON",
    }


def apply_w8a8_emulation(
    target: nn.Module,
    artifact: str | Path,
    *,
    device: str | torch.device,
    dtype: torch.dtype,
) -> dict[str, object]:
    """Replace the framework text Target's linear modules using one artifact."""

    if not isinstance(target, nn.Module):
        raise TypeError("target must be torch.nn.Module")
    requested_device = torch.device(device)
    if requested_device.type not in {"cpu", "cuda"}:
        raise ValueError("W8A8 formula emulation is available only on CPU/CUDA")
    if requested_device.type == "cuda" and requested_device.index is None:
        requested_device = torch.device("cuda", torch.cuda.current_device())
    if dtype is not torch.float16:
        raise ValueError(
            "strict NPU QLinear emulation requires --dtype float16 because the "
            "current NPU QLinear output is fixed to FP16"
        )

    root = Path(artifact).expanduser().resolve()
    manifest, entries = _artifact_manifest(root)
    quantized_paths = {item.framework_path for item in entries}
    passthrough = manifest["passthrough_linear_paths"]
    assert isinstance(passthrough, list)
    passthrough_paths = set(passthrough)
    expected_text_paths = quantized_paths | passthrough_paths
    if quantized_paths & passthrough_paths:
        raise ValueError("quantized and passthrough linear paths overlap")

    observed_text_linear_paths = {
        name
        for name, module in target.named_modules()
        if name
        and _is_framework_text_linear_path(name)
        and isinstance(module, nn.Linear)
    }
    if observed_text_linear_paths != expected_text_paths:
        raise RuntimeError(
            "framework text Linear topology differs from the exported NPU target: "
            f"missing={sorted(expected_text_paths - observed_text_linear_paths)}, "
            f"unexpected={sorted(observed_text_linear_paths - expected_text_paths)}"
        )
    output_getter = getattr(target, "get_output_embeddings", None)
    output_module = output_getter() if callable(output_getter) else None
    if not isinstance(output_module, nn.Module):
        raise TypeError("framework target does not expose its LM head")

    replaced: list[str] = []
    for entry in entries:
        original = _module_at(target, entry.framework_path)
        if not isinstance(original, nn.Linear):
            raise TypeError(f"{entry.framework_path} is no longer nn.Linear")
        expected_weight_shape = (entry.out_features, entry.in_features)
        if tuple(original.weight.shape) != expected_weight_shape:
            raise ValueError(
                f"{entry.framework_path} framework weight must have shape "
                f"{expected_weight_shape}; got {tuple(original.weight.shape)}"
            )
        if original.bias is not None:
            raise ValueError(f"{entry.framework_path} has a bias unsupported by QLinear")
        if original.weight.device != requested_device or original.weight.dtype != dtype:
            raise ValueError(
                f"{entry.framework_path} original weight must be {dtype} on "
                f"{requested_device}; got {original.weight.dtype}/{original.weight.device}"
            )
        weight, scale = _load_entry(root, entry)
        draft_weight = original.weight if original is output_module else None
        replacement = EmulatedW8A8Linear(
            weight.to(requested_device),
            scale.to(requested_device),
            output_dtype=torch.float16,
            draft_weight=draft_weight,
        ).eval()
        _replace_module(target, entry.framework_path, replacement)
        replaced.append(entry.framework_path)

    final_emulated = {
        name
        for name, module in target.named_modules()
        if isinstance(module, EmulatedW8A8Linear)
    }
    if final_emulated != quantized_paths:
        raise RuntimeError("framework W8A8 replacement coverage changed unexpectedly")
    for path in passthrough_paths:
        if not isinstance(_module_at(target, path), nn.Linear):
            raise RuntimeError(f"passthrough Linear {path!r} was unexpectedly replaced")

    input_getter = getattr(target, "get_input_embeddings", None)
    input_module = input_getter() if callable(input_getter) else None
    output_module_after = output_getter() if callable(output_getter) else None
    for label, module in (
        ("input embedding", input_module),
        ("Draft-facing LM head", output_module_after),
    ):
        weight = getattr(module, "weight", None)
        if not isinstance(weight, Tensor) or weight.dtype != dtype:
            raise TypeError(f"{label} must retain a {dtype} weight for the Draft")
        if weight.device != requested_device:
            raise ValueError(f"{label} weight must remain on {requested_device}")

    visual_linears = sorted(
        name
        for name, module in target.named_modules()
        if isinstance(module, nn.Linear) and name.startswith("model.visual.")
    )
    audit: dict[str, object] = {
        "status": EMULATION_STATUS,
        "scheme": "w8a8_dynamic_per_token_formula_emulation",
        "scope": "target_text_linear_only",
        "artifact_format": ARTIFACT_FORMAT,
        "artifact_path": str(root),
        "device": str(requested_device),
        "activation_quantization": "scale=max(abs(row))/127; q=round(x/scale)",
        "activation_zero_row_policy": "q_zero_scale_zero",
        "activation_int8_range": [-ACTIVATION_QMAX, ACTIVATION_QMAX],
        "weight_layout": "K_by_N",
        "accumulator": (
            "torch_int8_int_mm_exact_int32"
            if requested_device.type == "cpu" and callable(getattr(torch, "_int_mm", None))
            else "torch_int32_conversion_matmul_fallback"
            if requested_device.type == "cpu"
            else "exact_chunked_fp64_integer_sum"
        ),
        "dequantization_order": "accumulator_times_weight_scale_times_pertoken_scale",
        "linear_output_dtype": str(torch.float16),
        "qlinear_count": len(replaced),
        "qlinear_paths": sorted(replaced),
        "passthrough_linear_paths": sorted(passthrough_paths),
        "unused_visual_linear_count": len(visual_linears),
        "draft_embedding_dtype": str(getattr(input_module, "weight").dtype),
        "draft_lm_head_dtype": str(getattr(output_module_after, "weight").dtype),
        "performance_claim": "NONE_CORRECTNESS_ONLY",
        "real_npu_numerical_parity": "PENDING_SAME_ACTIVATION_DEVICE_COMPARISON",
        "embedding_quantization": "NOT_EMULATED_USE_FRAMEWORK_FP16_EMBEDDING",
        "draft_quantization": "DISABLED_FP16",
    }
    setattr(target, "_dflash_w8a8_emulation_audit", audit)
    return dict(audit)


def target_w8a8_emulation_audit(target: nn.Module) -> dict[str, object]:
    raw = getattr(target, "_dflash_w8a8_emulation_audit", None)
    if raw is None:
        return {
            "status": "DISABLED",
            "scheme": "disabled",
            "scope": "framework_target",
        }
    if not isinstance(raw, Mapping):
        raise TypeError("target W8A8 emulation audit must be a mapping")
    return dict(raw)


def compare_formula_output(reference: Tensor, candidate: Tensor) -> dict[str, object]:
    """Compare one real QLinear output with an emulated same-input output."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate output shapes differ")
    reference_fp32 = reference.detach().to(device="cpu", dtype=torch.float32)
    candidate_fp32 = candidate.detach().to(device="cpu", dtype=torch.float32)
    difference = (reference_fp32 - candidate_fp32).abs()
    denominator = reference_fp32.norm() * candidate_fp32.norm()
    cosine = (
        1.0
        if float(denominator) == 0.0 and torch.equal(reference_fp32, candidate_fp32)
        else float((reference_fp32.flatten() @ candidate_fp32.flatten()) / denominator)
        if float(denominator) != 0.0
        else 0.0
    )
    return {
        "bitwise_equal": bool(torch.equal(reference.detach().cpu(), candidate.detach().cpu())),
        "max_abs_error": float(difference.max()) if difference.numel() else 0.0,
        "mean_abs_error": float(difference.mean()) if difference.numel() else 0.0,
        "cosine_similarity": cosine,
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
        "shape": list(reference.shape),
    }


__all__ = [
    "ACTIVATION_QMAX",
    "ARTIFACT_FORMAT",
    "EMULATION_STATUS",
    "EmulatedW8A8Linear",
    "apply_w8a8_emulation",
    "compare_formula_output",
    "dynamic_quantize_per_token",
    "emulate_w8a8_linear",
    "export_w8a8_emulation_artifact",
    "target_w8a8_emulation_audit",
]
