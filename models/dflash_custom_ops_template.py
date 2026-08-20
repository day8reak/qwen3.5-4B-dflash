"""Template binding the V1 PyTorch golden to existing registered operators.

Copy this file, replace only the namespace/function calls, and pass the module
name through ``--ops-backend``.  Device validation must use strict dispatch;
silent fallback is allowed only for explicit CPU simulation.
"""

from __future__ import annotations

import torch


def rms_norm(x, weight, eps):
    return torch.ops.replace_with_internal_namespace.rms_norm(x, weight, eps)


def linear(x, weight):
    return torch.ops.replace_with_internal_namespace.linear(x, weight)


def rotary(query, key, cosine, sine):
    return torch.ops.replace_with_internal_namespace.rotary(
        query, key, cosine, sine
    )


def attention(query, key, value, attention_mask, scale, key_value_groups):
    # Golden boolean mask semantics: True means the key position is visible.
    return torch.ops.replace_with_internal_namespace.attention(
        query,
        key,
        value,
        attention_mask,
        scale,
        key_value_groups,
    )


def swiglu(gate, up):
    return torch.ops.replace_with_internal_namespace.swiglu(gate, up)


def top1(hidden, lm_head_weight):
    # Must return INT64 IDs; the lowest vocabulary ID wins exact ties.
    return torch.ops.replace_with_internal_namespace.top1(
        hidden, lm_head_weight
    )
