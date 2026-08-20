"""Replaceable primitive operations for the DFlash draft golden."""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
import torch.nn.functional as F


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


@runtime_checkable
class DFlashOps(Protocol):
    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor: ...

    def linear(self, x: Tensor, weight: Tensor) -> Tensor: ...

    def rotary(
        self,
        query: Tensor,
        key: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> tuple[Tensor, Tensor]: ...

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scale: float,
        key_value_groups: int,
    ) -> Tensor: ...

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor: ...

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor: ...


class TorchDFlashOps:
    """Eager PyTorch oracle matching the public Z-Lab Qwen3 DFlash math."""

    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + eps
        )
        # Unlike Qwen3.5 MTP, Qwen3 DFlash stores the effective scale itself.
        return weight * normalized.to(dtype=x.dtype)

    def linear(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.linear(x, weight)

    def rotary(
        self,
        query: Tensor,
        key: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cosine = cosine.unsqueeze(1)
        sine = sine.unsqueeze(1)
        query_length = query.shape[-2]
        query_cosine = cosine[..., -query_length:, :]
        query_sine = sine[..., -query_length:, :]
        return (
            query * query_cosine + _rotate_half(query) * query_sine,
            key * cosine + _rotate_half(key) * sine,
        )

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scale: float,
        key_value_groups: int,
    ) -> Tensor:
        use_native_gqa = (
            key_value_groups > 1
            and attention_mask is None
            and key.shape[-1] == value.shape[-1] <= 256
        )
        if not use_native_gqa:
            key = _repeat_kv(key, key_value_groups)
            value = _repeat_kv(value, key_value_groups)
        # The public DFlash Transformers path explicitly selects SDPA. Keeping
        # that backend here removes an otherwise measurable eager-vs-SDPA
        # rounding difference while retaining one replaceable attention ABI.
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=scale,
            enable_gqa=use_native_gqa,
        )

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor:
        return F.silu(gate) * up

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("non-finite DFlash hidden value before LM head")
        logits = F.linear(hidden, lm_head_weight)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite DFlash logits before Top1")
        return torch.argmax(logits, dim=-1)


class ModuleDFlashOps:
    """Dispatch every primitive to an internal module or explicit simulation fallback."""

    required_operations = (
        "rms_norm",
        "linear",
        "rotary",
        "attention",
        "swiglu",
        "top1",
    )

    def __init__(self, module: ModuleType, *, strict: bool = True) -> None:
        self.module = module
        self.strict = strict
        self.fallback = TorchDFlashOps()
        missing = [name for name in self.required_operations if not hasattr(module, name)]
        if missing and strict:
            raise ValueError(f"custom DFlash op module is missing: {', '.join(missing)}")

    @classmethod
    def from_name(cls, module_name: str, *, strict: bool = True) -> "ModuleDFlashOps":
        return cls(importlib.import_module(module_name), strict=strict)

    def _call(self, name: str, *args):
        operation = getattr(self.module, name, None)
        if operation is not None:
            return operation(*args)
        if self.strict:
            raise RuntimeError(f"custom DFlash operation is unavailable: {name}")
        return getattr(self.fallback, name)(*args)

    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        return self._call("rms_norm", x, weight, eps)

    def linear(self, x: Tensor, weight: Tensor) -> Tensor:
        return self._call("linear", x, weight)

    def rotary(
        self,
        query: Tensor,
        key: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self._call("rotary", query, key, cosine, sine)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scale: float,
        key_value_groups: int,
    ) -> Tensor:
        return self._call(
            "attention",
            query,
            key,
            value,
            attention_mask,
            scale,
            key_value_groups,
        )

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor:
        return self._call("swiglu", gate, up)

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        return self._call("top1", hidden, lm_head_weight)
