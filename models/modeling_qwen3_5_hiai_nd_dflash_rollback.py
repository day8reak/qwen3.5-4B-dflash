# coding=utf-8
"""Qwen3.5 HIAI target with an explicit DFlash rollback integration boundary.

The ordinary HIAI path keeps its original math and state ownership: when
``accepted_tokens`` is ``None`` it calls the current
``npu_chunk_gated_delta_rule`` ABI with a call-local ``INT16[B]`` effective
length and uses the receiver's original in-place convolution/cache updates.  A
vectorized DFlash verification call passes ``accepted_tokens: int8[B]`` and
exactly ``K + 1`` input rows (``anchor + K proposals``).  In that mode:

* the completed ``npu_gated_delta_rule_mtp`` operator selects the previously
  accepted recurrent-state slot and returns one provisional state per row;
* a correctness-first tensor decomposition does the same for causal-conv
  state, providing a precise replacement boundary for a future fused operator;
* full-attention K/V writes are issued one row at a time so a verification
  block crossing a 64-token cache boundary is correct with the existing
  ``npu_cache_update_`` ABI.

This file is model-side integration code, not a complete scheduler.  The owner
of the 32-layer cache must keep one shared accepted count and logical KV cursor,
and must discard the whole provisional call on failure.  The correction/bonus
token is not part of the committed cache until it is used as the next anchor.
"""

from typing import Callable, Optional, Tuple

import math
import torch
import torch.nn.functional as F
from torch import nn
import torch_npu

from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.activations import ACT2FN
from transformers.generation import GenerationMixin
from transformers.initialization import copy_ as init_copy_
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, logging
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils.import_utils import (
    is_causal_conv1d_available,
    is_flash_linear_attention_available,
)

from .configuration_qwen3_5 import Qwen3_5Config, Qwen3_5TextConfig
from .dflash_v1.dflash_target_features import (
    DFlashFeatureCollector,
    QWEN35_4B_DFLASH_TARGET_FEATURES,
)
from .modeling_qwen3_5_hiai_nd import QLinear, _cache_update_for_export

if is_causal_conv1d_available():
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
else:
    causal_conv1d_update, causal_conv1d_fn = None, None

if is_flash_linear_attention_available():
    from fla.modules import FusedRMSNormGated
    from fla.ops.gated_delta_rule import (
        chunk_gated_delta_rule,
        fused_recurrent_gated_delta_rule,
    )
else:
    chunk_gated_delta_rule, fused_recurrent_gated_delta_rule = None, None
    FusedRMSNormGated = None

logger = logging.get_logger(__name__)


DFLASH_BLOCK_SIZE = 16
DFLASH_MAX_PROPOSALS = DFLASH_BLOCK_SIZE - 1
DFLASH_MAX_VERIFY_TOKENS = DFLASH_BLOCK_SIZE


def _normalize_gdr_effective_length(
    effective_length: Optional[torch.Tensor],
    *,
    batch_size: int,
    physical_sequence_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the current GDR valid-row count as one reusable INT16 tensor."""

    if physical_sequence_length <= 0:
        raise ValueError("GDR physical sequence length must be positive")
    if effective_length is None:
        return torch.full(
            (batch_size,),
            physical_sequence_length,
            dtype=torch.int16,
            device=device,
        )
    if not isinstance(effective_length, torch.Tensor):
        raise TypeError("gdr_effective_length must be a Tensor")
    if effective_length.dtype != torch.int16:
        raise TypeError("gdr_effective_length must use torch.int16")
    if tuple(effective_length.shape) != (batch_size,):
        raise ValueError(
            f"gdr_effective_length must have shape [{batch_size}], "
            f"got {tuple(effective_length.shape)}"
        )
    if effective_length.device != device:
        raise ValueError("gdr_effective_length and hidden states must share one device")
    if effective_length.device.type == "cpu" and effective_length.numel():
        minimum = int(effective_length.min().item())
        maximum = int(effective_length.max().item())
        if minimum < 1 or maximum > physical_sequence_length:
            raise ValueError(
                "gdr_effective_length values must be in "
                f"[1,{physical_sequence_length}]"
            )
    return effective_length


def _require_dflash_accepted_tokens(
    accepted_tokens: torch.Tensor,
    *,
    batch_size: int,
    state_slots: int,
    device: torch.device,
) -> None:
    """Validate the device ABI without introducing an NPU-to-host sync."""

    if not isinstance(accepted_tokens, torch.Tensor):
        raise TypeError("accepted_tokens must be a Tensor")
    if accepted_tokens.dtype != torch.int8:
        raise TypeError("accepted_tokens must use torch.int8")
    if accepted_tokens.ndim != 1 or tuple(accepted_tokens.shape) != (batch_size,):
        raise ValueError(
            f"accepted_tokens must have shape [{batch_size}], "
            f"got {tuple(accepted_tokens.shape)}"
        )
    if accepted_tokens.device != device:
        raise ValueError("accepted_tokens and state banks must share one device")
    if not 1 <= state_slots <= DFLASH_MAX_VERIFY_TOKENS:
        raise ValueError(
            f"DFlash state_slots must be in [1,{DFLASH_MAX_VERIFY_TOKENS}]"
        )
    # A device-side range check belongs in the custom operator/graph.  Checking
    # NPU values here with .item() would force a synchronization on every round.
    if accepted_tokens.device.type == "cpu" and accepted_tokens.numel():
        minimum = int(accepted_tokens.min().item())
        maximum = int(accepted_tokens.max().item())
        if minimum < 0 or maximum >= state_slots:
            raise ValueError(
                f"accepted_tokens values must be in [0,{state_slots - 1}]"
            )


def _select_dflash_state_slot(
    state_bank: torch.Tensor,
    accepted_tokens: torch.Tensor,
) -> torch.Tensor:
    """Select ``state_bank[b, accepted_tokens[b]]`` for every batch row."""

    if state_bank.ndim < 3:
        raise ValueError("a DFlash state bank must have rank at least 3")
    batch_size, state_slots = state_bank.shape[:2]
    _require_dflash_accepted_tokens(
        accepted_tokens,
        batch_size=batch_size,
        state_slots=state_slots,
        device=state_bank.device,
    )
    index_shape = (batch_size, 1, *((1,) * (state_bank.ndim - 2)))
    gather_index = accepted_tokens.to(torch.long).view(index_shape)
    gather_index = gather_index.expand(batch_size, 1, *state_bank.shape[2:])
    return torch.gather(state_bank, 1, gather_index).squeeze(1)


def seed_dflash_gdn_state_banks(
    conv_state: torch.Tensor,
    recurrent_state: torch.Tensor,
    verify_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Expand committed scalar GDN states into first-round DFlash banks.

    ``verify_tokens`` is ``K + 1``.  All slots initially contain the same
    committed state and the first verification call therefore uses an all-zero
    ``accepted_tokens`` selector.  Recurrent state is deliberately promoted to
    FP32 because that is the completed GDR MTP operator's locked input ABI.
    """

    if isinstance(verify_tokens, bool) or not isinstance(verify_tokens, int):
        raise TypeError("verify_tokens must be an integer")
    if not 1 <= verify_tokens <= DFLASH_MAX_VERIFY_TOKENS:
        raise ValueError(
            f"verify_tokens must be in [1,{DFLASH_MAX_VERIFY_TOKENS}]"
        )
    if conv_state.ndim != 3:
        raise ValueError("conv_state must have shape [B,C,Kc]")
    if recurrent_state.ndim != 4:
        raise ValueError("recurrent_state must have shape [B,H,Dk,Dv]")
    if conv_state.shape[0] != recurrent_state.shape[0]:
        raise ValueError("conv_state and recurrent_state batches differ")
    if conv_state.device != recurrent_state.device:
        raise ValueError("conv_state and recurrent_state devices differ")
    conv_bank = conv_state.unsqueeze(1).expand(
        conv_state.shape[0], verify_tokens, *conv_state.shape[1:]
    ).clone()
    recurrent_state = recurrent_state.to(torch.float32)
    recurrent_bank = recurrent_state.unsqueeze(1).expand(
        recurrent_state.shape[0], verify_tokens, *recurrent_state.shape[1:]
    ).clone()
    return conv_bank, recurrent_bank


def rebase_dflash_gdn_state_banks(
    conv_state_bank: torch.Tensor,
    recurrent_state_bank: torch.Tensor,
    accepted_tokens: torch.Tensor,
    next_verify_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the committed slot and resize banks when the next K changes."""

    if conv_state_bank.ndim != 4:
        raise ValueError("conv_state_bank must have shape [B,T,C,Kc]")
    if recurrent_state_bank.ndim != 5:
        raise ValueError("recurrent_state_bank must have shape [B,T,H,Dk,Dv]")
    if conv_state_bank.shape[:2] != recurrent_state_bank.shape[:2]:
        raise ValueError("conv and recurrent state-bank batch/slot shapes differ")
    committed_conv = _select_dflash_state_slot(conv_state_bank, accepted_tokens)
    committed_recurrent = _select_dflash_state_slot(
        recurrent_state_bank, accepted_tokens
    )
    return seed_dflash_gdn_state_banks(
        committed_conv,
        committed_recurrent,
        next_verify_tokens,
    )


def torch_dflash_causal_conv1d_mtp(
    hidden_states: torch.Tensor,
    conv_state_bank: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    accepted_tokens: torch.Tensor,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correctness-first causal-conv state bank on the input tensor's device.

    Args:
        hidden_states: ``[B,C,T]`` projected Q/K/V rows.
        conv_state_bank: ``[B,T,C,Kc]`` previous provisional windows.
        weight: ``[C,Kc]`` depthwise convolution weight.
        bias: optional ``[C]`` bias.
        accepted_tokens: ``int8[B]`` slot selected from the previous bank.

    Returns:
        Activated convolution rows ``[B,C,T]`` and provisional state windows
        ``[B,T,C,Kc]``.  The decomposition executes on NPU when its inputs are
        NPU tensors; it is not a CPU fallback.
    """

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape [B,C,T]")
    if conv_state_bank.ndim != 4:
        raise ValueError("conv_state_bank must have shape [B,T,C,Kc]")
    batch_size, channels, sequence_length = hidden_states.shape
    if sequence_length < 1 or sequence_length > DFLASH_MAX_VERIFY_TOKENS:
        raise ValueError(
            f"DFlash verify sequence length must be in "
            f"[1,{DFLASH_MAX_VERIFY_TOKENS}]"
        )
    expected_bank_prefix = (batch_size, sequence_length, channels)
    if tuple(conv_state_bank.shape[:3]) != expected_bank_prefix:
        raise ValueError(
            "conv_state_bank must use the same [B,T,C] dimensions as input"
        )
    state_length = conv_state_bank.shape[-1]
    if tuple(weight.shape) != (channels, state_length):
        raise ValueError(
            f"conv weight must have shape {(channels, state_length)}, "
            f"got {tuple(weight.shape)}"
        )
    if bias is not None and tuple(bias.shape) != (channels,):
        raise ValueError("conv bias must have shape [C]")
    if hidden_states.device != conv_state_bank.device:
        raise ValueError("hidden_states and conv_state_bank devices differ")
    if hidden_states.dtype != conv_state_bank.dtype:
        raise ValueError("hidden_states and conv_state_bank dtypes differ")
    if activation not in {"silu", "swish"}:
        raise ValueError("the locked Qwen3.5 GDN path requires SiLU")

    base_state = _select_dflash_state_slot(conv_state_bank, accepted_tokens)
    history = torch.cat((base_state, hidden_states), dim=-1).to(weight.dtype)
    convolution = F.conv1d(
        history,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=channels,
    )
    output = F.silu(convolution[:, :, -sequence_length:])
    # ``unfold`` creates every rolling Kc window in one tensor operation.  Drop
    # window zero (the round-start state) so slot i remains the state after
    # consuming input rows 0..i.  This replaces T Python slices plus stack.
    next_state_bank = (
        history.unfold(-1, state_length, 1)[..., 1:, :]
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    return output.to(hidden_states.dtype), next_state_bank.to(hidden_states.dtype)


def _npu_gated_delta_rule_mtp(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    accepted_tokens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve and call the completed GDR MTP bridge; never fall back silently."""

    operation = getattr(torch_npu, "npu_gated_delta_rule_mtp", None)
    if not callable(operation):
        npu_namespace = getattr(torch.ops, "npu", None)
        operation = (
            getattr(npu_namespace, "npu_gated_delta_rule_mtp", None)
            if npu_namespace is not None
            else None
        )
    if not callable(operation):
        raise RuntimeError(
            "DFlash rollback requires the registered "
            "npu_gated_delta_rule_mtp custom operator"
        )
    result = operation(
        query,
        key,
        value,
        g,
        beta,
        initial_state,
        accepted_tokens,
        64,
        True,
        True,
    )
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise TypeError("npu_gated_delta_rule_mtp must return (out, state_bank)")
    return result[0], result[1]


class Qwen3_5RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = torch_npu.adn_rms_norm(
            x.float(), 1.0 + self.weight.float(), self.eps
        )[0]
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class Qwen3_5RMSNormGated(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = torch_npu.adn_rms_norm(
            hidden_states.to(torch.float32),
            self.weight.float(),
            self.variance_epsilon,
        )[0]
        out = hidden_states * F.silu(gate.to(torch.float32))
        return out.to(input_dtype)


class Qwen3_5RotaryEmbedding1(nn.Module):
    """Cached text-only MRoPE used by the HIAI path."""

    inv_freq: torch.Tensor
    cos_cached: torch.Tensor
    sin_cached: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.register_buffer("temp", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])
        self._set_cos_sin_cache(seq_len=self.max_seq_len_cached, device=device)

    @staticmethod
    def compute_default_rope_parameters(config, device=None, seq_len=None):
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get(
            "partial_rotary_factor", 1.0
        )
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    def _set_cos_sin_cache(self, seq_len, device=None):
        position_ids = torch.arange(seq_len, device=device, dtype=torch.float)
        position_ids_expanded = position_ids[None, None, :].expand(3, 1, -1)
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(
            3, 1, -1, 1
        )
        freqs = (
            inv_freq_expanded.float()
            @ position_ids_expanded.unsqueeze(2).float()
        ).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer(
            "cos_cached", emb.cos() * self.attention_scaling, persistent=False
        )
        self.register_buffer(
            "sin_cached", emb.sin() * self.attention_scaling, persistent=False
        )

    @torch.no_grad()
    def forward(self, x, position_ids):
        pos = position_ids[0] if position_ids.ndim == 3 else position_ids
        cos = self.cos_cached[:, pos, :].squeeze(0)
        sin = self.sin_cached[:, pos, :].squeeze(0)
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    @staticmethod
    def apply_interleaved_mrope(freqs, mrope_section):
        freqs_t = freqs[0]
        for dim_idx, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim_idx] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim_idx, ..., idx]
        return freqs_t


class Qwen3_5RotaryEmbedding(nn.Module):
    inv_freq: torch.Tensor

    def __init__(self, config: Qwen3_5TextConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings
        self.config = config
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: Callable = self.compute_default_rope_parameters
        if self.rope_type != "default":
            rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)
        self.mrope_section = config.rope_parameters.get("mrope_section", [11, 11, 10])

    @staticmethod
    def compute_default_rope_parameters(
        config: Qwen3_5TextConfig | None = None,
        device: Optional["torch.device"] = None,
        seq_len: int | None = None,
    ) -> tuple["torch.Tensor", float]:
        base = config.rope_parameters["rope_theta"]
        partial_rotary_factor = config.rope_parameters.get(
            "partial_rotary_factor", 1.0
        )
        head_dim = (
            getattr(config, "head_dim", None)
            or config.hidden_size // config.num_attention_heads
        )
        dim = int(head_dim * partial_rotary_factor)
        attention_factor = 1.0
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, dim, 2, dtype=torch.int64).to(
                    device=device, dtype=torch.float
                )
                / dim
            )
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    @dynamic_rope_update
    def forward(self, x, position_ids):
        if position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(
                3, position_ids.shape[0], -1
            )
        inv_freq_expanded = (
            self.inv_freq[None, None, :, None]
            .float()
            .expand(3, position_ids.shape[1], -1, 1)
            .to(x.device)
        )
        position_ids_expanded = position_ids[:, :, None, :].float()
        freqs = (
            inv_freq_expanded.float() @ position_ids_expanded.float()
        ).transpose(2, 3)
        freqs = self.apply_interleaved_mrope(freqs, self.mrope_section)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

    def apply_interleaved_mrope(self, freqs, mrope_section):
        freqs_t = freqs[0]
        for dim, offset in enumerate((1, 2), start=1):
            length = mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            freqs_t[..., idx] = freqs[dim, ..., idx]
        return freqs_t


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
    return (
        torch.cat([q_embed, q_pass], dim=-1),
        torch.cat([k_embed, k_pass], dim=-1),
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(
        batch, num_key_value_heads * n_rep, slen, head_dim
    )


def eager_attention_forward(
    module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask
    attn_weights = nn.functional.softmax(
        attn_weights, dim=-1, dtype=torch.float32
    ).to(query.dtype)
    attn_weights = nn.functional.dropout(
        attn_weights, p=dropout, training=module.training
    )
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()
    return attn_output, attn_weights


class Qwen3_5Attention(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(
            config,
            "head_dim",
            config.hidden_size // config.num_attention_heads,
        )
        self.num_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = (
            config.num_attention_heads // config.num_key_value_heads
        )
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.hidden_size = config.hidden_size
        self.q_proj = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * self.head_dim * 2,
            bias=config.attention_bias,
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            config.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim,
            config.hidden_size,
            bias=config.attention_bias,
        )
        self.q_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3_5RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5RotaryEmbedding1(config=config)
        self.block_size = 64
        self.kv_max_len = getattr(self.config, "kv_cache_max_len", 2048)
        self.expandLen = self.num_key_value_heads * self.head_dim // 16
        self.kv_block_table = self._generate_blocktable(
            [self.kv_max_len], self.block_size
        )
        self.register_buffer(
            "block_table",
            torch.tensor(self.kv_block_table, dtype=torch.int32),
            persistent=False,
        )

    def _rebuild_block_table(self):
        self.kv_max_len = getattr(
            self.config, "kv_cache_max_len", self.kv_max_len
        )
        self.kv_block_table = self._generate_blocktable(
            [self.kv_max_len], self.block_size
        )
        self.block_table = torch.tensor(
            self.kv_block_table,
            dtype=torch.int32,
            device=self.block_table.device,
        )

    def _generate_blocktable(self, kv_seq_len_list, kv_block_size):
        batch_size = len(kv_seq_len_list)
        max_context_len = max(kv_seq_len_list)
        max_blk_num = math.ceil(max_context_len / kv_block_size)
        blk_table_result = []
        acc_cur_blk_num = 0
        for b_idx in range(batch_size):
            blk_num_cur_batch = math.ceil(
                kv_seq_len_list[b_idx] / kv_block_size
            )
            blk_table_cur_batch = list(
                range(acc_cur_blk_num, blk_num_cur_batch + acc_cur_blk_num)
            )
            blk_table_cur_batch += [0] * (max_blk_num - blk_num_cur_batch)
            blk_table_result.append(blk_table_cur_batch)
            acc_cur_blk_num += blk_num_cur_batch
        return blk_table_result

    def transform_nz_2_nd(self, input_matrix):
        b, n, s, d = input_matrix.shape
        output_matrix = (
            input_matrix.reshape(b, n, d // 16, s, 16)
            .transpose(2, 3)
            .contiguous()
        )
        return output_matrix.reshape(b, n, s, d)

    def transform_nd_2_nz(self, input_matrix):
        b, n, s, d = input_matrix.shape
        output_matrix = (
            input_matrix.reshape(b, n, s, d // 16, 16)
            .transpose(2, 3)
            .contiguous()
        )
        return output_matrix.reshape(b, n, s, d)

    def update(
        self, new_k, cache_position, past_key_value, *, export_flag=False
    ):
        b, s, n, d = new_k.shape
        block_idx = cache_position[0] // self.block_size
        offset_in_block = (cache_position[0] % self.block_size).to(torch.int32)
        target_blocks = block_idx.reshape(1).to(torch.int32)
        k_flattened = new_k.reshape(b, s, -1, 16)
        return _cache_update_for_export(
            past_key_value.to(new_k.device),
            k_flattened[0, :, :, :].to(torch.float16),
            target_blocks,
            offset_in_block,
            export_flag=export_flag,
        )

    def update_dflash(
        self, new_k, cache_position, past_key_value, *, export_flag=False
    ):
        """Correctness fallback for a K+1 write that may cross cache blocks.

        The receiver's current CacheUpdate call supplies one target block and
        one offset.  Until its multi-row/cross-block ABI is proven, issue the
        already-supported single-row form for every verification row.  This is
        deliberately an optimization boundary, not a claim that 16 launches
        per K/V tensor are production-efficient.
        """

        batch_size, sequence_length, _, _ = new_k.shape
        if batch_size != 1:
            raise ValueError("the current block-table CacheUpdate path requires B=1")
        if cache_position is None or cache_position.ndim != 1:
            raise ValueError("DFlash cache_position must have shape [T]")
        if cache_position.numel() != sequence_length:
            raise ValueError("DFlash cache_position length must equal K+1")
        if sequence_length > DFLASH_MAX_VERIFY_TOKENS:
            raise ValueError(
                "DFlash CacheUpdate supports at most "
                f"{DFLASH_MAX_VERIFY_TOKENS} rows"
            )

        flattened = new_k.reshape(batch_size, sequence_length, -1, 16)
        updated_cache = past_key_value.to(new_k.device)
        for token_index in range(sequence_length):
            position = cache_position[token_index]
            target_block = (position // self.block_size).reshape(1).to(torch.int32)
            offset_in_block = (position % self.block_size).to(torch.int32)
            updated_cache = _cache_update_for_export(
                updated_cache,
                flattened[0, token_index : token_index + 1].to(torch.float16),
                target_block,
                offset_in_block,
                export_flag=export_flag,
            )
        return updated_cache

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        accepted_tokens: Optional[torch.Tensor] = None,
        allQLen=0,
        export_flag=False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            cos,
            sin,
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        if accepted_tokens is None:
            key_states = self.update(
                key_states,
                cache_position,
                past_key_values[0],
                export_flag=export_flag,
            ).to(query_states.device)
            value_states = self.update(
                value_states,
                cache_position,
                past_key_values[1],
                export_flag=export_flag,
            ).to(query_states.device)
        else:
            _require_dflash_accepted_tokens(
                accepted_tokens,
                batch_size=input_shape[0],
                state_slots=input_shape[1],
                device=query_states.device,
            )
            key_states = self.update_dflash(
                key_states,
                cache_position,
                past_key_values[0],
                export_flag=export_flag,
            ).to(query_states.device)
            value_states = self.update_dflash(
                value_states,
                cache_position,
                past_key_values[1],
                export_flag=export_flag,
            ).to(query_states.device)
        past_key_values = (key_states, value_states)
        attention_mask = attention_mask.to(torch.float16)
        query_states = query_states.transpose(1, 2).contiguous()
        q_origin_shape = query_states.shape
        q_seq_len = query_states.shape[2]
        q1 = self.transform_nd_2_nz(
            query_states.reshape(
                -1, self.num_heads, q_seq_len, self.head_dim
            )
        )
        q2 = q1.reshape(
            -1, self.num_heads * self.head_dim // 16, q_seq_len, 16
        )
        attn_params = {
            "query": q2,
            "key": [key_states],
            "value": [value_states],
            "actual_seq_lengths_q": [q_seq_len],
            "actual_seq_lengths_kv": [self.kv_max_len],
            "block_table": self.block_table,
            "num_heads": self.num_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "block_size": self.block_size,
            "input_layout": "BNSD",
            "scale_value": self.scaling,
            "inner_precise": 2,
            "atten_mask": attention_mask,
        }
        # allQLen is a sequence-length list (SymInt[]), not a PSE Tensor.
        # Keep the receiver frontend ABI identical in eager and AIR export so
        # dispatcher validation reaches the registered Fake/GE converter.
        attn_params["all_seq_lengths_q"] = allQLen
        attn_output = torch_npu.adn_fused_infer_attention(**attn_params)
        attn_output = attn_output.reshape(q_origin_shape)
        attn_output = (
            self.transform_nz_2_nd(attn_output)
            .transpose(1, 2)
            .contiguous()
        )
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_values

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward1(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        allQLen=0,
        export_flag=False,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Retain the receiver's alternate attention implementation unchanged."""

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, gate = torch.chunk(
            self.q_proj(hidden_states).view(
                *input_shape, -1, self.head_dim * 2
            ),
            2,
            dim=-1,
        )
        gate = gate.reshape(*input_shape, -1)
        query_states = self.q_norm(query_states.view(hidden_shape))
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape))
        value_states = self.v_proj(hidden_states).view(hidden_shape)
        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            cos,
            sin,
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        if past_key_values is not None and past_key_values[0] is not None:
            key_states = torch_npu.npu_scatter_nd_update_(
                past_key_values[0].to(key_states.device),
                cache_position.to(key_states.device),
                key_states.transpose(0, 1).to(torch.float16),
            )
            value_states = torch_npu.npu_scatter_nd_update_(
                past_key_values[1].to(key_states.device),
                cache_position.to(key_states.device),
                value_states.transpose(0, 1).to(torch.float16),
            )
            past_key_values = (key_states, value_states)
            key_states = key_states.transpose(1, 2).transpose(0, 1)
            value_states = value_states.transpose(1, 2).transpose(0, 1)

        attention_mask = attention_mask.to(torch.float16)
        if False:
            query_states = query_states.transpose(1, 2).contiguous()
            key_states = key_states.transpose(1, 2).contiguous()
            value_states = value_states.transpose(1, 2).contiguous()
            attention_interface: Callable = eager_attention_forward
            attn_output, attn_weights = attention_interface(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                dropout=0.0 if not self.training else self.attention_dropout,
                scaling=self.scaling,
                **kwargs,
            )
        else:
            query_states = query_states.transpose(1, 2).contiguous()
            q_origin_shape = query_states.shape
            q_seq_len = query_states.shape[2]
            actual_seq_len_q = [q_seq_len]
            actual_seq_len_kv = [key_states.shape[1]]
            key_states = key_states.reshape(
                -1,
                self.block_size,
                self.num_key_value_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
            value_states = value_states.reshape(
                -1,
                self.block_size,
                self.num_key_value_heads,
                self.head_dim,
            ).permute(0, 2, 1, 3)
            q1 = self.transform_nd_2_nz(
                query_states.reshape(
                    -1, self.num_heads, q_seq_len, self.head_dim
                )
            )
            k1 = self.transform_nd_2_nz(key_states)
            v1 = self.transform_nd_2_nz(value_states)
            q2 = q1.reshape(
                -1,
                self.num_heads * self.head_dim // 16,
                q_seq_len,
                16,
            )
            k2 = k1.reshape(
                -1,
                self.num_key_value_heads * self.head_dim // 16,
                self.block_size,
                16,
            )
            v2 = v1.reshape(
                -1,
                self.num_key_value_heads * self.head_dim // 16,
                self.block_size,
                16,
            )
            block_table = torch.tensor(
                self.kv_block_table,
                dtype=torch.int32,
                device=q2.device,
            )
            attn_params = {
                "query": q2,
                "key": [k2],
                "value": [v2],
                "actual_seq_lengths_q": actual_seq_len_q,
                "actual_seq_lengths_kv": actual_seq_len_kv,
                "block_table": block_table,
                "num_heads": self.num_heads,
                "num_key_value_heads": self.num_key_value_heads,
                "block_size": self.block_size,
                "input_layout": "BNSD",
                "scale_value": self.scaling,
                "inner_precise": 2,
                "atten_mask": attention_mask,
            }
            # allQLen is a sequence-length list (SymInt[]), not a PSE Tensor.
            attn_params["all_seq_lengths_q"] = allQLen
            attn_output = torch_npu.adn_fused_infer_attention(**attn_params)
            attn_output = attn_output.reshape(q_origin_shape)
            attn_output = (
                self.transform_nz_2_nd(attn_output)
                .transpose(1, 2)
                .contiguous()
            )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = attn_output * torch.sigmoid(gate)
        attn_output = self.o_proj(attn_output)
        return attn_output, None, past_key_values


class Qwen3_5MLP(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.up_proj = nn.Linear(
            self.hidden_size, self.intermediate_size, bias=False
        )
        self.down_proj = nn.Linear(
            self.intermediate_size, self.hidden_size, bias=False
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(
            self.act_fn(self.gate_proj(x)) * self.up_proj(x)
        )


def apply_mask_to_padding_states(hidden_states, attention_mask):
    if (
        attention_mask is not None
        and attention_mask.dim() == 2
        and attention_mask.shape[1] == hidden_states.shape[1]
    ):
        mask = attention_mask[:, None, :, None]
        mask = mask.expand(
            hidden_states.shape[0],
            hidden_states.shape[1],
            hidden_states.shape[2],
            hidden_states.shape[3],
        )
        return hidden_states.mul_(mask)
    return hidden_states


def torch_causal_conv1d_update(
    hidden_states, conv_state, weight, bias, activation
):
    _, hidden_size, seq_len = hidden_states.shape
    state_len = conv_state.shape[-1]
    hidden_states_new = torch.cat([conv_state, hidden_states], dim=-1).to(
        weight.dtype
    )
    conv_state.copy_(hidden_states_new[:, :, -state_len:])
    out = F.conv1d(
        hidden_states_new,
        weight.unsqueeze(1),
        bias,
        padding=0,
        groups=hidden_size,
    )
    out = F.silu(out[:, :, -seq_len:])
    return out.to(hidden_states.dtype)


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_v_heads = config.linear_num_value_heads
        self.num_k_heads = config.linear_num_key_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_idx = layer_idx
        self.activation = config.hidden_act
        self.act = ACT2FN[config.hidden_act]
        self.layer_norm_epsilon = config.rms_norm_eps
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            bias=False,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            padding=self.conv_kernel_size - 1,
        )
        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        A = torch.empty(self.num_v_heads).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.norm = Qwen3_5RMSNormGated(
            self.head_v_dim, eps=self.layer_norm_epsilon
        )
        self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
        self.causal_conv1d_fn = causal_conv1d_fn
        self.causal_conv1d_update = (
            causal_conv1d_update or torch_causal_conv1d_update
        )
        self.in_proj_qkv = nn.Linear(
            self.hidden_size,
            self.key_dim * 2 + self.value_dim,
            bias=False,
        )
        self.in_proj_z = nn.Linear(
            self.hidden_size, self.value_dim, bias=False
        )
        self.in_proj_b = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.in_proj_a = nn.Linear(
            self.hidden_size, self.num_v_heads, bias=False
        )
        self.layer_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        cache_params: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        accepted_tokens: Optional[torch.Tensor] = None,
        gdr_effective_length: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ):
        if cache_params is None or len(cache_params) != 2:
            raise ValueError("GDN requires (conv_state, recurrent_state)")
        batch_size, seq_len, _ = hidden_states.shape
        conv_state = cache_params[0]
        recurrent_state = cache_params[1]
        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)
        z = self.in_proj_z(hidden_states).reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        b = self.in_proj_b(hidden_states)
        a = self.in_proj_a(hidden_states)
        if accepted_tokens is None:
            mixed_qkv = self.causal_conv1d_update(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                self.activation,
            ).transpose(1, 2)
        else:
            if conv_state.ndim != 4:
                raise ValueError(
                    "DFlash conv_state must have shape [B,K+1,C,Kc]"
                )
            if recurrent_state.ndim != 5:
                raise ValueError(
                    "DFlash recurrent_state must have shape [B,K+1,H,Dk,Dv]"
                )
            expected_conv = (
                batch_size,
                seq_len,
                self.conv_dim,
                self.conv_kernel_size,
            )
            expected_recurrent = (
                batch_size,
                seq_len,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
            )
            if tuple(conv_state.shape) != expected_conv:
                raise ValueError(
                    f"DFlash conv_state must have shape {expected_conv}, "
                    f"got {tuple(conv_state.shape)}"
                )
            if tuple(recurrent_state.shape) != expected_recurrent:
                raise ValueError(
                    f"DFlash recurrent_state must have shape {expected_recurrent}, "
                    f"got {tuple(recurrent_state.shape)}"
                )
            if recurrent_state.dtype != torch.float32:
                raise TypeError("DFlash recurrent_state bank must use FP32")
            mixed_qkv, next_conv_state = torch_dflash_causal_conv1d_mtp(
                mixed_qkv,
                conv_state,
                self.conv1d.weight.squeeze(1),
                self.conv1d.bias,
                accepted_tokens,
                self.activation,
            )
            conv_state = next_conv_state
            mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv,
            [self.key_dim, self.key_dim, self.value_dim],
            dim=-1,
        )
        query = query.reshape(
            batch_size, seq_len, -1, self.head_k_dim
        )
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(
            batch_size, seq_len, -1, self.head_v_dim
        )
        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(
            a.float() + self.dt_bias
        )
        if self.num_v_heads // self.num_k_heads > 1:
            repeat = self.num_v_heads // self.num_k_heads
            query = query.repeat_interleave(repeat, dim=2)
            key = key.repeat_interleave(repeat, dim=2)
        if accepted_tokens is None:
            if gdr_effective_length is None:
                raise ValueError("ordinary GDN requires gdr_effective_length")
            core_attn_out, last_recurrent_state = (
                torch_npu.npu_chunk_gated_delta_rule(
                    query,
                    key,
                    value.contiguous(),
                    g=g,
                    beta=beta,
                    effective_length=gdr_effective_length,
                    chunk_size=1 if seq_len == 1 else 64,
                    initial_state=recurrent_state.to(torch.float32),
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            )
            recurrent_state = last_recurrent_state.to(torch.float16)
        else:
            core_attn_out, next_recurrent_state = _npu_gated_delta_rule_mtp(
                query,
                key,
                value.contiguous(),
                g,
                beta,
                recurrent_state.contiguous(),
                accepted_tokens,
            )
            if tuple(next_recurrent_state.shape) != tuple(recurrent_state.shape):
                raise ValueError(
                    "npu_gated_delta_rule_mtp returned an invalid state-bank shape"
                )
            if next_recurrent_state.dtype != torch.float32:
                raise TypeError(
                    "npu_gated_delta_rule_mtp state output must use FP32"
                )
            recurrent_state = next_recurrent_state
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        output = self.out_proj(core_attn_out)
        return output, (conv_state, recurrent_state)


class Qwen3_5DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.block_type = config.layer_types[layer_idx]
        if self.block_type == "linear_attention":
            self.linear_attn = Qwen3_5GatedDeltaNet(config, layer_idx)
        elif self.block_type == "full_attention":
            self.self_attn = Qwen3_5Attention(config, layer_idx)
        self.mlp = Qwen3_5MLP(config, config.intermediate_size)
        self.input_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        torch.npu.empty_cache()

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        past_residual: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        new_kv_cache_pos=None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx=None,
        allQLen=0,
        token_count=0,
        export_flag=False,
        accepted_tokens: Optional[torch.Tensor] = None,
        gdr_effective_length: Optional[torch.Tensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Tuple]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if self.block_type == "linear_attention":
            hidden_states, present_key_value = self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                cache_position=new_kv_cache_pos,
                attention_mask=attention_mask,
                accepted_tokens=accepted_tokens,
                gdr_effective_length=gdr_effective_length,
            )
        elif self.block_type == "full_attention":
            hidden_states, _, present_key_value = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=new_kv_cache_pos,
                accepted_tokens=accepted_tokens,
                allQLen=allQLen,
                export_flag=export_flag,
                **kwargs,
            )
        else:
            raise ValueError(f"unsupported layer type {self.block_type!r}")
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, present_key_value


class Qwen3_5PreTrainedModel(PreTrainedModel):
    config: Qwen3_5Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["Qwen3_5DecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
    _is_stateful = True
    _can_compile_fullgraph = True

    @torch.no_grad()
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, Qwen3_5GatedDeltaNet):
            nn.init.ones_(module.dt_bias)
            init_copy_(
                module.A_log,
                torch.empty_like(module.A_log).uniform_(0, 16).log_(),
            )
        elif isinstance(module, Qwen3_5RMSNorm):
            nn.init.zeros_(module.weight)
        elif "RotaryEmbedding" in module.__class__.__name__ and hasattr(
            module, "_set_cos_sin_cache"
        ):
            module._set_cos_sin_cache(
                seq_len=module.max_seq_len_cached,
                device=module.inv_freq.device,
            )
        elif hasattr(module, "_rebuild_block_table") and hasattr(
            module, "block_table"
        ):
            module._rebuild_block_table()


class Qwen3_5TextModel(Qwen3_5PreTrainedModel):
    """HIAI text body with feature capture and opt-in DFlash state banks."""

    config: Qwen3_5TextConfig
    dflash_feature_contract_id = "qwen3.5-4b-dflash-hiai-feature-source-v1"
    dflash_feature_source = (
        "package_local:modeling_qwen3_5_hiai_nd_dflash_rollback.py"
    )
    dflash_feature_capture_point = "decoder_post_layer_pre_final_norm"
    dflash_state_contract_id = "qwen3.5-4b-dflash-target-state-bank-v1"

    def __init__(self, config: Qwen3_5TextConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(
            config.vocab_size, config.hidden_size, config.pad_token_id
        )
        self.layers = nn.ModuleList(
            [
                Qwen3_5DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = Qwen3_5RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.gradient_checkpointing = False
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        new_kv_cache_pos=None,
        allQLen=0,
        token_count=0,
        export_flag=False,
        output_dflash_features: bool = False,
        accepted_tokens: Optional[torch.Tensor] = None,
        gdr_effective_length: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds.to(torch.float16)
        gdr_effective_length = _normalize_gdr_effective_length(
            gdr_effective_length,
            batch_size=int(hidden_states.shape[0]),
            physical_sequence_length=int(hidden_states.shape[1]),
            device=hidden_states.device,
        )
        if accepted_tokens is not None:
            if past_key_values is None:
                raise ValueError(
                    "DFlash rollback mode requires persistent 32-layer state"
                )
            _require_dflash_accepted_tokens(
                accepted_tokens,
                batch_size=hidden_states.shape[0],
                state_slots=hidden_states.shape[1],
                device=hidden_states.device,
            )
        dflash_collector = None
        if output_dflash_features:
            dflash_collector = DFlashFeatureCollector(
                QWEN35_4B_DFLASH_TARGET_FEATURES,
                enabled=True,
                detach=True,
                clone=True,
            )

        for idx, decoder_layer in enumerate(self.layers):
            if past_key_values is not None and idx >= len(past_key_values):
                break
            past_key_value = (
                past_key_values[idx] if past_key_values is not None else None
            )
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_value,
                new_kv_cache_pos=new_kv_cache_pos,
                use_cache=use_cache,
                layer_idx=idx,
                allQLen=allQLen,
                token_count=token_count,
                export_flag=export_flag,
                accepted_tokens=accepted_tokens,
                gdr_effective_length=gdr_effective_length,
                **kwargs,
            )
            hidden_states = layer_outputs[0]
            if past_key_values is not None:
                past_key_values[idx] = layer_outputs[1]
            if dflash_collector is not None:
                dflash_collector.capture(idx, hidden_states)

        hidden_states = self.norm(hidden_states)
        if dflash_collector is None:
            return hidden_states
        dflash_features = dflash_collector.finalize()
        if dflash_features is None:
            raise RuntimeError("enabled DFlash collector returned no features")
        return hidden_states, dflash_features


class KwargsForCausalLM(FlashAttentionKwargs, TransformersKwargs):
    ...


class Qwen3_5ForCausalLM(Qwen3_5PreTrainedModel, GenerationMixin):
    """Tensor-returning HIAI LM with opt-in feature/state-bank execution."""

    _tied_weights_keys = {"lm_head.weight": "language_model.embed_tokens.weight"}
    config: Qwen3_5TextConfig
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^visual.*"]
    dflash_feature_contract_id = "qwen3.5-4b-dflash-hiai-feature-source-v1"
    dflash_feature_source = (
        "package_local:modeling_qwen3_5_hiai_nd_dflash_rollback.py"
    )
    dflash_feature_capture_point = "decoder_post_layer_pre_final_norm"
    dflash_state_contract_id = "qwen3.5-4b-dflash-target-state-bank-v1"

    def __init__(self, config):
        super().__init__(config)
        self.language_model = Qwen3_5TextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        new_kv_cache_pos=None,
        allQLen=0,
        token_count=0,
        export_flag=False,
        output_dflash_features: bool = False,
        dflash_skip_lm_head: bool = False,
        dflash_last_token_only: bool = False,
        accepted_tokens: Optional[torch.Tensor] = None,
        gdr_effective_length: Optional[torch.Tensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        text_output = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            new_kv_cache_pos=new_kv_cache_pos,
            allQLen=allQLen,
            token_count=token_count,
            export_flag=export_flag,
            output_dflash_features=output_dflash_features,
            accepted_tokens=accepted_tokens,
            gdr_effective_length=gdr_effective_length,
            **kwargs,
        )
        if output_dflash_features:
            if (
                not isinstance(text_output, tuple)
                or len(text_output) != 2
                or not all(isinstance(item, torch.Tensor) for item in text_output)
            ):
                raise TypeError(
                    "feature-enabled text model must return "
                    "(hidden_states, dflash_features)"
                )
            hidden_states, dflash_features = text_output
        else:
            if not isinstance(text_output, torch.Tensor):
                raise TypeError(
                    "ordinary HIAI text model forward must return a Tensor"
                )
            hidden_states = text_output
            dflash_features = None

        if not isinstance(dflash_skip_lm_head, bool):
            raise TypeError("dflash_skip_lm_head must be a bool")
        if not isinstance(dflash_last_token_only, bool):
            raise TypeError("dflash_last_token_only must be a bool")
        if dflash_skip_lm_head and dflash_last_token_only:
            raise ValueError(
                "dflash_skip_lm_head and dflash_last_token_only are mutually exclusive"
            )
        if dflash_skip_lm_head:
            # Earlier prompt chunks still need every decoder/state/feature row,
            # but no logits. Preserve the output ABI with a zero-row Tensor.
            logits = hidden_states.new_empty(
                (hidden_states.shape[0], 0, self.vocab_size)
            )
        elif dflash_last_token_only:
            # The final prompt chunk keeps all real rows for persistent state
            # and DFlash feature capture, while only its last real row is sampled.
            logits = self.lm_head(hidden_states[:, -1:, :])
        else:
            logits = self.lm_head(hidden_states)
        if dflash_features is None:
            return logits
        return logits, dflash_features
