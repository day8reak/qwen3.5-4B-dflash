"""Correctness-first DFlash V1 primitives for an Ascend 310P overlay.

The public functions in this module intentionally match :class:`DFlashOps`.
They are a decomposed PyTorch candidate for replacing the six internal
operators while bringing up the complete V1 flow.  In particular, attention
does not call SDPA or a flash-attention implementation: it executes QK matmul,
boolean masking, FP32 softmax, and probability/value matmul explicitly.

All casts stay on the input device.  This module never moves a tensor to CPU
and has no fallback dispatcher.  Running it on CPU is simulation evidence;
running it on an NPU requires the caller to place every tensor on that NPU and
to disable fallback in ``ModuleDFlashOps``.
"""

from __future__ import annotations

import math
from numbers import Integral, Real
import os

import torch
from torch import Tensor
import torch.nn.functional as F


_SUPPORTED_DTYPES = frozenset((torch.float16, torch.bfloat16, torch.float32))
EXHAUSTIVE_CHECKS_ENV = "DFLASH_ASCEND310P_EXHAUSTIVE_CHECKS"
_TRUE_ENV_VALUES = frozenset(("1", "true", "yes", "on"))
_FALSE_ENV_VALUES = frozenset(("0", "false", "no", "off"))


def exhaustive_value_checks_enabled(device: torch.device | str) -> bool:
    """Return whether every intermediate tensor should be scanned for finiteness.

    CPU execution is the reduced-shape diagnostic oracle and keeps the original
    fail-closed checks.  Accelerator execution defaults to boundary-only checks:
    shape/device/dtype contracts still run for every primitive, while the final
    Draft logits remain finite-checked before Top1.  Set
    ``DFLASH_ASCEND310P_EXHAUSTIVE_CHECKS=1`` only for numerical diagnosis; it
    intentionally adds a device synchronization for every checked tensor.
    """

    device_type = (
        device.type
        if isinstance(device, torch.device)
        else str(device).split(":", 1)[0].lower()
    )
    if device_type == "cpu":
        return True
    raw = os.environ.get(EXHAUSTIVE_CHECKS_ENV, "0").strip().lower()
    if raw in _TRUE_ENV_VALUES:
        return True
    if raw in _FALSE_ENV_VALUES:
        return False
    raise ValueError(
        f"{EXHAUSTIVE_CHECKS_ENV} must be one of "
        "0/1, false/true, no/yes, or off/on"
    )


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_float_tensor(name: str, value: object) -> Tensor:
    tensor = _require_tensor(name, value)
    if tensor.dtype not in _SUPPORTED_DTYPES:
        supported = "float16, bfloat16, or float32"
        raise TypeError(f"{name} must use {supported}; got {tensor.dtype}")
    return tensor


def _require_same_device_dtype(operation: str, *named_tensors: tuple[str, Tensor]) -> None:
    reference_name, reference = named_tensors[0]
    for name, tensor in named_tensors[1:]:
        if tensor.device != reference.device:
            raise ValueError(
                f"{operation} tensors must share one device; "
                f"{reference_name} is on {reference.device}, {name} is on {tensor.device}"
            )
        if tensor.dtype != reference.dtype:
            raise TypeError(
                f"{operation} tensors must share one dtype; "
                f"{reference_name} is {reference.dtype}, {name} is {tensor.dtype}"
            )


def _require_finite(name: str, tensor: Tensor, *, boundary: bool = False) -> None:
    if not boundary and not exhaustive_value_checks_enabled(tensor.device):
        return
    # ``item`` synchronizes the current device.  Keep it at model boundaries
    # during normal accelerator execution, and at every primitive only in the
    # explicit exhaustive diagnostic mode.
    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"{name} contains a non-finite value")


def _checked_output(operation: str, output: Tensor, dtype: torch.dtype) -> Tensor:
    if output.dtype != dtype:
        raise RuntimeError(
            f"{operation} changed output dtype from {dtype} to {output.dtype}"
        )
    _require_finite(f"{operation} output", output)
    return output


def _rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def _repeat_kv(states: Tensor, repetitions: int) -> Tensor:
    if repetitions == 1:
        return states
    batch, heads, sequence, head_dim = states.shape
    expanded = states[:, :, None, :, :].expand(
        batch, heads, repetitions, sequence, head_dim
    )
    return expanded.reshape(batch, heads * repetitions, sequence, head_dim)


def rms_norm(x: Tensor, weight: Tensor, eps: float) -> Tensor:
    """Apply the checkpoint's RMSNorm precision boundary.

    Reduction and reciprocal square root run in FP32.  The normalized value is
    cast back before multiplication by the stored effective scale, matching
    ``TorchDFlashOps``.
    """

    x = _require_float_tensor("x", x)
    weight = _require_float_tensor("weight", weight)
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("rms_norm x must have a non-empty last dimension")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(
            "rms_norm weight must be rank-1 and match the last dimension of x"
        )
    _require_same_device_dtype("rms_norm", ("x", x), ("weight", weight))
    if isinstance(eps, bool) or not isinstance(eps, Real):
        raise TypeError("rms_norm eps must be a real scalar")
    eps_value = float(eps)
    if not math.isfinite(eps_value) or eps_value <= 0.0:
        raise ValueError("rms_norm eps must be finite and positive")
    _require_finite("rms_norm x", x)
    _require_finite("rms_norm weight", weight)

    x_fp32 = x.float()
    normalized = x_fp32 * torch.rsqrt(
        x_fp32.square().mean(dim=-1, keepdim=True) + eps_value
    )
    output = weight * normalized.to(dtype=x.dtype)
    return _checked_output("rms_norm", output, x.dtype)


def linear(x: Tensor, weight: Tensor) -> Tensor:
    """Bias-free linear projection with ``weight[out, in]``."""

    x = _require_float_tensor("x", x)
    weight = _require_float_tensor("weight", weight)
    if x.ndim < 1 or x.shape[-1] == 0:
        raise ValueError("linear x must have a non-empty last dimension")
    if weight.ndim != 2 or weight.shape[0] == 0:
        raise ValueError("linear weight must have shape [out_features, in_features]")
    if weight.shape[1] != x.shape[-1]:
        raise ValueError(
            f"linear input width {x.shape[-1]} does not match weight width {weight.shape[1]}"
        )
    _require_same_device_dtype("linear", ("x", x), ("weight", weight))
    _require_finite("linear x", x)
    _require_finite("linear weight", weight)

    output = F.linear(x, weight)
    return _checked_output("linear", output, x.dtype)


def rotary(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    """Apply half-rotation RoPE to ``[batch, heads, sequence, head_dim]``."""

    query = _require_float_tensor("query", query)
    key = _require_float_tensor("key", key)
    cosine = _require_float_tensor("cosine", cosine)
    sine = _require_float_tensor("sine", sine)
    _require_same_device_dtype(
        "rotary",
        ("query", query),
        ("key", key),
        ("cosine", cosine),
        ("sine", sine),
    )
    if query.ndim != 4 or key.ndim != 4:
        raise ValueError("rotary query and key must be rank-4 [batch, heads, sequence, dim]")
    if cosine.ndim != 3 or sine.ndim != 3 or cosine.shape != sine.shape:
        raise ValueError("rotary cosine and sine must have the same rank-3 shape")
    if query.shape[0] != key.shape[0] or query.shape[0] != cosine.shape[0]:
        raise ValueError("rotary batch dimensions must match")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != cosine.shape[-1]:
        raise ValueError("rotary head dimensions must match")
    if query.shape[-1] == 0 or query.shape[-1] % 2:
        raise ValueError("rotary head dimension must be positive and even")
    if key.shape[-2] != cosine.shape[-2]:
        raise ValueError("rotary cosine/sine sequence must cover the complete key sequence")
    if query.shape[-2] == 0 or query.shape[-2] > key.shape[-2]:
        raise ValueError("rotary query sequence must be non-empty and no longer than key")
    for name, tensor in (
        ("rotary query", query),
        ("rotary key", key),
        ("rotary cosine", cosine),
        ("rotary sine", sine),
    ):
        _require_finite(name, tensor)

    cosine_heads = cosine.unsqueeze(1)
    sine_heads = sine.unsqueeze(1)
    query_length = query.shape[-2]
    query_cosine = cosine_heads[..., -query_length:, :]
    query_sine = sine_heads[..., -query_length:, :]
    rotated_query = query * query_cosine + _rotate_half(query) * query_sine
    rotated_key = key * cosine_heads + _rotate_half(key) * sine_heads
    return (
        _checked_output("rotary query", rotated_query, query.dtype),
        _checked_output("rotary key", rotated_key, key.dtype),
    )


def attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attention_mask: Tensor | None,
    scale: float,
    key_value_groups: int,
) -> Tensor:
    """Decomposed, cache-free GQA attention.

    ``attention_mask`` is either ``None`` or a rank-4 boolean tensor
    broadcastable to ``[batch, query_heads, query_length, key_length]``.  True
    means visible.  Additive masks are rejected so mask polarity cannot be
    silently misinterpreted.
    """

    query = _require_float_tensor("query", query)
    key = _require_float_tensor("key", key)
    value = _require_float_tensor("value", value)
    _require_same_device_dtype(
        "attention", ("query", query), ("key", key), ("value", value)
    )
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("attention tensors must be rank-4 [batch, heads, sequence, dim]")
    if query.shape[0] != key.shape[0] or key.shape[0] != value.shape[0]:
        raise ValueError("attention batch dimensions must match")
    if key.shape[1] != value.shape[1]:
        raise ValueError("attention key and value head counts must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("attention key and value sequence lengths must match")
    if query.shape[-1] != key.shape[-1] or query.shape[-1] != value.shape[-1]:
        raise ValueError("attention query/key/value head dimensions must match")
    if min(query.shape[-2], key.shape[-2], query.shape[-1]) <= 0:
        raise ValueError("attention sequence lengths and head dimension must be positive")
    if isinstance(key_value_groups, bool) or not isinstance(key_value_groups, Integral):
        raise TypeError("attention key_value_groups must be an integer")
    group_count = int(key_value_groups)
    if group_count <= 0:
        raise ValueError("attention key_value_groups must be positive")
    if query.shape[1] != key.shape[1] * group_count:
        raise ValueError(
            "attention query heads must equal key/value heads times key_value_groups"
        )
    if isinstance(scale, bool) or not isinstance(scale, Real):
        raise TypeError("attention scale must be a real scalar")
    scale_value = float(scale)
    if not math.isfinite(scale_value) or scale_value <= 0.0:
        raise ValueError("attention scale must be finite and positive")
    for name, tensor in (
        ("attention query", query),
        ("attention key", key),
        ("attention value", value),
    ):
        _require_finite(name, tensor)

    key = _repeat_kv(key, group_count)
    value = _repeat_kv(value, group_count)
    scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
    scores = scores * scale_value
    _require_finite("attention scores", scores)

    if attention_mask is not None:
        attention_mask = _require_tensor("attention_mask", attention_mask)
        if attention_mask.device != query.device:
            raise ValueError("attention_mask must be on the same device as query")
        if attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must be boolean (True means visible)")
        if attention_mask.ndim != 4:
            raise ValueError("attention_mask must be rank-4")
        score_shape = scores.shape
        if any(
            mask_dim not in (1, score_dim)
            for mask_dim, score_dim in zip(attention_mask.shape, score_shape)
        ):
            raise ValueError(
                f"attention_mask shape {tuple(attention_mask.shape)} is not broadcastable "
                f"to {tuple(score_shape)}"
            )
        visible = attention_mask.expand(score_shape)
        if (
            exhaustive_value_checks_enabled(query.device)
            and not bool(visible.any(dim=-1).all().item())
        ):
            raise ValueError("attention_mask contains a fully masked query row")
        scores = scores.masked_fill(~visible, float("-inf"))

    probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
    _require_finite("attention probabilities", probabilities)
    output = torch.matmul(probabilities, value.float()).to(dtype=query.dtype)
    return _checked_output("attention", output, query.dtype)


def swiglu(gate: Tensor, up: Tensor) -> Tensor:
    """Apply SiLU(gate) * up without changing dtype or device."""

    gate = _require_float_tensor("gate", gate)
    up = _require_float_tensor("up", up)
    _require_same_device_dtype("swiglu", ("gate", gate), ("up", up))
    if gate.shape != up.shape or gate.ndim < 1:
        raise ValueError("swiglu gate and up must have the same non-scalar shape")
    _require_finite("swiglu gate", gate)
    _require_finite("swiglu up", up)
    output = F.silu(gate) * up
    return _checked_output("swiglu", output, gate.dtype)


def top1(hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
    """Return greedy token IDs; exact ties select the lowest vocabulary ID."""

    hidden = _require_float_tensor("hidden", hidden)
    lm_head_weight = _require_float_tensor("lm_head_weight", lm_head_weight)
    if hidden.ndim != 3:
        raise ValueError("top1 hidden must have shape [batch, draft_length, hidden_size]")
    if lm_head_weight.ndim != 2 or lm_head_weight.shape[0] == 0:
        raise ValueError("top1 lm_head_weight must have shape [vocab_size, hidden_size]")
    if hidden.shape[-1] == 0 or hidden.shape[-1] != lm_head_weight.shape[-1]:
        raise ValueError("top1 hidden size must match lm_head_weight")
    _require_same_device_dtype(
        "top1", ("hidden", hidden), ("lm_head_weight", lm_head_weight)
    )
    _require_finite("top1 hidden", hidden)
    _require_finite("top1 lm_head_weight", lm_head_weight)
    logits = F.linear(hidden, lm_head_weight)
    # Preserve one fail-closed numerical boundary without scanning the shared
    # 248320x2560 LM-head weight (and every intermediate weight) each round.
    _require_finite("top1 logits", logits, boundary=True)
    token_ids = torch.argmax(logits, dim=-1)
    if token_ids.dtype != torch.int64:
        raise RuntimeError(f"top1 must return int64 IDs; got {token_ids.dtype}")
    return token_ids


__all__ = (
    "EXHAUSTIVE_CHECKS_ENV",
    "attention",
    "exhaustive_value_checks_enabled",
    "linear",
    "rms_norm",
    "rotary",
    "swiglu",
    "top1",
)
