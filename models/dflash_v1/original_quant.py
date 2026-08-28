"""Packaged copy of the receiver's original ``utils.py`` quant conversion.

The public entry point deliberately keeps the original two-argument ABI:
``quant_model(model, quanted_pth)``.  Weight-key mapping, ND-to-blocked-ZN
conversion, QLinear indices, and the expert exclusion match the supplied
receiver implementation.  The QLinear class is resolved from the execution
model's modeling module so the ordinary and rollback files share the same
primitive without changing either modeling source.
"""

from __future__ import annotations

import glob
import importlib

from safetensors import safe_open
import torch
from torch import Tensor, nn


def nd_to_zn_int8(weight_nd: Tensor) -> Tensor:
    """Convert a 2-D or batched 3-D INT8 ND weight to flattened ZN order."""

    if weight_nd.ndim == 2:
        k_size, n_size = weight_nd.shape
        batch = False
    elif weight_nd.ndim == 3:
        batch_size, k_size, n_size = weight_nd.shape
        batch = True
    else:
        raise ValueError(f"Expected 2D or 3D, got {weight_nd.ndim}D")
    if k_size % 32:
        raise ValueError(f"K={k_size} must be a multiple of 32")
    if n_size % 16:
        raise ValueError(f"N={n_size} must be a multiple of 16")

    if batch:
        weight = weight_nd.reshape(
            batch_size,
            k_size // 32,
            32,
            n_size // 16,
            16,
        )
        weight_zn = weight.permute(0, 1, 3, 4, 2)
    else:
        weight = weight_nd.reshape(k_size // 32, 32, n_size // 16, 16)
        weight_zn = weight.permute(0, 2, 3, 1)
    return weight_zn.reshape(weight_nd.shape)


def nd_to_blocked_zn_int8(weight_nd: Tensor) -> Tensor:
    """Apply the original 65,280-column blocking before ZN conversion."""

    max_n_per_block = 65_280
    if weight_nd.ndim == 2:
        _, n_size = weight_nd.shape
    elif weight_nd.ndim == 3:
        _, _, n_size = weight_nd.shape
    else:
        raise ValueError(f"Expected 2D or 3D, got {weight_nd.ndim}D")
    if n_size <= max_n_per_block:
        return nd_to_zn_int8(weight_nd)

    blocks: list[Tensor] = []
    for column in range(0, n_size, max_n_per_block):
        end = min(column + max_n_per_block, n_size)
        block = (
            weight_nd[:, column:end]
            if weight_nd.ndim == 2
            else weight_nd[:, :, column:end]
        )
        blocks.append(nd_to_zn_int8(block).flatten())
    return torch.cat(blocks).reshape(weight_nd.shape)


def pack_int4_to_int32_simple(int8_tensor: Tensor) -> Tensor:
    """Retain the original helper used by compatible quant artifacts."""

    uint4_tensor = int8_tensor.to(torch.uint8) & 0x0F
    reshaped = uint4_tensor.view(int8_tensor.shape[0], -1, 8)
    shift_mask = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28],
        dtype=torch.int32,
        device=int8_tensor.device,
    )
    return torch.sum(
        reshaped.to(torch.int32) << shift_mask,
        dim=2,
    ).to(torch.int32)


def replace_module_by_names(
    model: nn.Module,
    modules_to_replace: dict[str, nn.Module],
) -> nn.Module:
    """Replace modules by exact names using the original identity walk."""

    def helper(child: nn.Module) -> None:
        for child_name, candidate in child.named_children():
            replaced = False
            for full_name, module in model.named_modules():
                if full_name not in modules_to_replace:
                    continue
                if candidate is module:
                    child.add_module(
                        child_name,
                        modules_to_replace.pop(full_name),
                    )
                    replaced = True
                    break
            if not replaced:
                helper(candidate)

    helper(model)
    return model


def _qlinear_class(model: nn.Module) -> type[nn.Module]:
    modeling = importlib.import_module(type(model).__module__)
    qlinear = getattr(modeling, "QLinear", None)
    if not isinstance(qlinear, type) or not issubclass(qlinear, nn.Module):
        from models.modeling_qwen3_5_hiai_nd import QLinear

        qlinear = QLinear
    return qlinear


def replace_linear_to_qlinear(
    model: nn.Module,
    quant_state_dict: dict[str, Tensor],
) -> nn.Module:
    """Apply the supplied ``replace_linear_to_QLinear`` conversion."""

    qlinear = _qlinear_class(model)
    replacements: dict[str, nn.Module] = {}
    index = 15
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        alternate = (
            name.replace("language_model.", "model.", 1)
            if name.startswith("language_model.")
            else name
        )
        weight_key = f"{alternate}_quant_weight"
        scale_key = f"{alternate}_quant_scale"
        if weight_key not in quant_state_dict:
            continue
        if scale_key not in quant_state_dict:
            raise KeyError(f"missing quant scale for {name}: {scale_key}")
        weight = nd_to_blocked_zn_int8(quant_state_dict.pop(weight_key).t())
        scale = quant_state_dict.pop(scale_key).to(torch.float32)
        if scale.dim() == 0:
            scale = scale.unsqueeze(0)
        if "mlp.experts" in name:
            continue
        replacements[name] = qlinear(W_q=weight, scale=scale, idx=index)
        index += 1
    if replacements:
        replace_module_by_names(model, replacements)
    return model


def quant_model(model: nn.Module, quant_weight_path: str) -> nn.Module:
    """Load ``data*.safetensors`` and invoke the original QLinear conversion."""

    files = sorted(glob.glob(f"{quant_weight_path}/data*.safetensors"))
    if not files:
        raise FileNotFoundError(
            f"no data*.safetensors found under {quant_weight_path}"
        )
    quant_state_dict: dict[str, Tensor] = {}
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                if key in quant_state_dict:
                    raise RuntimeError(f"duplicate quant tensor key: {key}")
                quant_state_dict[key] = handle.get_tensor(key)
    return replace_linear_to_qlinear(model, quant_state_dict)


__all__ = [
    "nd_to_blocked_zn_int8",
    "nd_to_zn_int8",
    "pack_int4_to_int32_simple",
    "quant_model",
    "replace_linear_to_qlinear",
    "replace_module_by_names",
]
