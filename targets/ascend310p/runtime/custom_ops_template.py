"""Template binding the portable MTP module to registered Ascend custom ops.

Rename the namespace/functions to the internal registration names.  Importing
this module is harmless; calls fail if the target library has not registered
the operators.  Never use this module with ``--allow-op-fallback`` for a board
result.
"""

from __future__ import annotations

import torch


def rms_norm(x, weight, eps):
    return torch.ops.qwen35_ascend.rms_norm(x, weight, eps)


def linear(x, weight):
    return torch.ops.qwen35_ascend.linear(x, weight)


def attention(query, key, value, additive_mask, scale):
    return torch.ops.qwen35_ascend.attention(
        query, key, value, additive_mask, scale
    )


def swiglu(gate, up):
    return torch.ops.qwen35_ascend.swiglu(gate, up)


def top1(hidden, lm_head_weight):
    return torch.ops.qwen35_ascend.top1(hidden, lm_head_weight)
