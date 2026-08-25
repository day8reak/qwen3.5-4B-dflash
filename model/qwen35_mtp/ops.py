"""Fine-grained replaceable operations used by the portable MTP block.

An internal Ascend adapter can provide the same five calls with existing
operators or registered custom operators.  The default implementation is pure
PyTorch and is the numerical CPU oracle.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Protocol, runtime_checkable

import torch
from torch import Tensor
import torch.nn.functional as F


@runtime_checkable
class MtpOps(Protocol):
    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor: ...

    def linear(self, x: Tensor, weight: Tensor) -> Tensor: ...

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        additive_mask: Tensor | None,
        scale: float,
    ) -> Tensor: ...

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor: ...

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor: ...


class TorchMtpOps:
    """Reference math matching the Transformers Qwen3.5 eager path."""

    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        normalized = x.float() * torch.rsqrt(
            x.float().pow(2).mean(dim=-1, keepdim=True) + eps
        )
        # Qwen3.5 stores the RMSNorm delta; effective scale is 1 + weight.
        return (normalized * (1.0 + weight.float())).to(dtype=x.dtype)

    def linear(self, x: Tensor, weight: Tensor) -> Tensor:
        return F.linear(x, weight)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        additive_mask: Tensor | None,
        scale: float,
    ) -> Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) * scale
        if additive_mask is not None:
            scores = scores + additive_mask
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        return torch.matmul(probabilities, value)

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor:
        return F.silu(gate) * up

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        if not torch.isfinite(hidden).all():
            raise FloatingPointError("non-finite hidden value before LM head")
        logits = F.linear(hidden, lm_head_weight)
        if not torch.isfinite(logits).all():
            raise FloatingPointError("non-finite logit before Top1")
        # torch.argmax returns the first index on ties, which is the locked rule.
        return torch.argmax(logits, dim=-1)


class ModuleMtpOps:
    """Dispatch operations to an internal module, with an explicit fallback mode.

    The module may contain Python functions, ``torch.library`` custom ops, or
    wrappers around an internal Ascend runtime.  Use ``strict=True`` on target
    so an accidental CPU fallback cannot be reported as a device result.
    """

    required_operations = ("rms_norm", "linear", "attention", "swiglu", "top1")

    def __init__(self, module: ModuleType, *, strict: bool = True) -> None:
        self.module = module
        self.strict = strict
        self.fallback = TorchMtpOps()
        missing = [name for name in self.required_operations if not hasattr(module, name)]
        if missing and strict:
            raise ValueError(f"custom MTP op module is missing: {', '.join(missing)}")

    @classmethod
    def from_name(cls, module_name: str, *, strict: bool = True) -> "ModuleMtpOps":
        return cls(importlib.import_module(module_name), strict=strict)

    def _call(self, name: str, *args):
        operation = getattr(self.module, name, None)
        if operation is not None:
            return operation(*args)
        if self.strict:
            raise RuntimeError(f"custom MTP operation is unavailable: {name}")
        return getattr(self.fallback, name)(*args)

    def rms_norm(self, x: Tensor, weight: Tensor, eps: float) -> Tensor:
        return self._call("rms_norm", x, weight, eps)

    def linear(self, x: Tensor, weight: Tensor) -> Tensor:
        return self._call("linear", x, weight)

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        additive_mask: Tensor | None,
        scale: float,
    ) -> Tensor:
        return self._call("attention", query, key, value, additive_mask, scale)

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor:
        return self._call("swiglu", gate, up)

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        return self._call("top1", hidden, lm_head_weight)
