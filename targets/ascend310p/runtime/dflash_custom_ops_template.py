"""Bind the DFlash golden ABI to internally registered Ascend operators.

Copy this module, rename the namespace/functions to the internal registration
names, and pass its import name through ``--ops-backend``.  A target result
must use strict dispatch; never combine it with ``--allow-op-fallback``.
"""

from __future__ import annotations

import torch


def rms_norm(x, weight, eps):
    """Qwen3 RMSNorm whose stored weight is the effective scale."""

    return torch.ops.qwen35_ascend_dflash.rms_norm(x, weight, eps)


def linear(x, weight):
    return torch.ops.qwen35_ascend_dflash.linear(x, weight)


def rotary(query, key, cosine, sine):
    """Apply full-head RoPE and return ``(rotated_query, rotated_key)``."""

    return torch.ops.qwen35_ascend_dflash.rotary(query, key, cosine, sine)


def attention(query, key, value, attention_mask, scale, key_value_groups):
    """Run GQA SDPA; a boolean mask uses True for visible positions."""

    return torch.ops.qwen35_ascend_dflash.attention(
        query,
        key,
        value,
        attention_mask,
        scale,
        key_value_groups,
    )


def swiglu(gate, up):
    return torch.ops.qwen35_ascend_dflash.swiglu(gate, up)


def top1(hidden, lm_head_weight):
    """Return INT64 argmax IDs; the lowest token ID wins exact ties."""

    return torch.ops.qwen35_ascend_dflash.top1(hidden, lm_head_weight)
