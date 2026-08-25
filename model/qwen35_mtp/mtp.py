"""Portable PyTorch implementation of the official Qwen3.5 one-layer MTP head."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable

import torch
from torch import Tensor, nn

from .config import Qwen35MTPConfig
from .ops import MtpOps, TorchMtpOps
from .weights import EMBEDDING_WEIGHT, SafeTensorRepository


class WeightOnlyLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, ops: MtpOps) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        return self.ops.linear(x, self.weight)


class Qwen35RMSNorm(nn.Module):
    """Qwen3.5 RMSNorm, whose checkpoint parameter is a delta from one."""

    def __init__(self, dim: int, eps: float, ops: MtpOps) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim))
        self.eps = eps
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        return self.ops.rms_norm(x, self.weight, self.eps)


class TextRotaryEmbedding(nn.Module):
    """Text-only Qwen3.5 RoPE.

    Qwen3.5 uses three MRoPE position streams.  For text-only inference the
    streams are identical, so this reduces exactly to the scalar position path.
    """

    def __init__(self, config: Qwen35MTPConfig) -> None:
        super().__init__()
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.rotary_dim, 2, dtype=torch.float32)
                / config.rotary_dim
            )
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, position_ids: Tensor, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def apply_partial_rope(
    query: Tensor,
    key: Tensor,
    cosine: Tensor,
    sine: Tensor,
) -> tuple[Tensor, Tensor]:
    cosine = cosine.unsqueeze(1)
    sine = sine.unsqueeze(1)
    rotary_dim = cosine.shape[-1]
    query_rotary, query_pass = query[..., :rotary_dim], query[..., rotary_dim:]
    key_rotary, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
    query_rotary = query_rotary * cosine + rotate_half(query_rotary) * sine
    key_rotary = key_rotary * cosine + rotate_half(key_rotary) * sine
    return (
        torch.cat((query_rotary, query_pass), dim=-1),
        torch.cat((key_rotary, key_pass), dim=-1),
    )


@dataclass(frozen=True)
class MTPKVCache:
    key: Tensor
    value: Tensor

    @property
    def sequence_length(self) -> int:
        return int(self.key.shape[-2])


@dataclass(frozen=True)
class MTPOutput:
    hidden_states: Tensor
    cache: MTPKVCache
    top1_token_ids: Tensor | None = None


class Qwen35MTPAttention(nn.Module):
    def __init__(self, config: Qwen35MTPConfig, ops: MtpOps) -> None:
        super().__init__()
        self.config = config
        self.ops = ops
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = WeightOnlyLinear(config.hidden_size, q_width * 2, ops)
        self.k_proj = WeightOnlyLinear(config.hidden_size, kv_width, ops)
        self.v_proj = WeightOnlyLinear(config.hidden_size, kv_width, ops)
        self.o_proj = WeightOnlyLinear(q_width, config.hidden_size, ops)
        self.q_norm = Qwen35RMSNorm(config.head_dim, config.rms_norm_eps, ops)
        self.k_norm = Qwen35RMSNorm(config.head_dim, config.rms_norm_eps, ops)
        self.rotary = TextRotaryEmbedding(config)
        self.scale = config.head_dim**-0.5

    @staticmethod
    def _repeat_kv(states: Tensor, repetitions: int) -> Tensor:
        if repetitions == 1:
            return states
        batch, heads, sequence, head_dim = states.shape
        expanded = states[:, :, None, :, :].expand(
            batch, heads, repetitions, sequence, head_dim
        )
        return expanded.reshape(batch, heads * repetitions, sequence, head_dim)

    @staticmethod
    def _causal_mask(
        query_length: int,
        past_length: int,
        *,
        device: torch.device,
    ) -> Tensor:
        key_length = past_length + query_length
        query_positions = torch.arange(
            past_length, key_length, device=device
        ).view(query_length, 1)
        key_positions = torch.arange(key_length, device=device).view(1, key_length)
        allowed = key_positions <= query_positions
        mask = torch.zeros((query_length, key_length), dtype=torch.float32, device=device)
        return mask.masked_fill(~allowed, float("-inf")).view(
            1, 1, query_length, key_length
        )

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        past_key_values: MTPKVCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, MTPKVCache]:
        batch, sequence, _ = hidden_states.shape
        c = self.config
        q_and_gate = self.q_proj(hidden_states).reshape(
            batch, sequence, c.num_attention_heads, c.head_dim * 2
        )
        query, gate = q_and_gate.chunk(2, dim=-1)
        gate = gate.reshape(batch, sequence, c.num_attention_heads * c.head_dim)
        query = self.q_norm(query).transpose(1, 2)
        key = self.k_norm(
            self.k_proj(hidden_states).reshape(
                batch, sequence, c.num_key_value_heads, c.head_dim
            )
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).reshape(
            batch, sequence, c.num_key_value_heads, c.head_dim
        ).transpose(1, 2)

        cosine, sine = self.rotary(position_ids, query.dtype)
        query, key = apply_partial_rope(query, key, cosine, sine)

        past_length = 0
        if past_key_values is not None:
            if past_key_values.key.shape[:2] != key.shape[:2]:
                raise ValueError("MTP cache batch/head shape does not match the current input")
            past_length = past_key_values.sequence_length
            key = torch.cat((past_key_values.key, key), dim=-2)
            value = torch.cat((past_key_values.value, value), dim=-2)
        new_cache = MTPKVCache(key=key, value=value)

        repeated_key = self._repeat_kv(key, c.num_key_value_groups)
        repeated_value = self._repeat_kv(value, c.num_key_value_groups)
        causal = self._causal_mask(
            sequence, past_length, device=hidden_states.device
        )
        if attention_mask is not None:
            if tuple(attention_mask.shape[-2:]) != tuple(causal.shape[-2:]):
                raise ValueError("attention_mask has an incompatible query/key shape")
            causal = causal + attention_mask.float()
        mixed = self.ops.attention(
            query, repeated_key, repeated_value, causal, self.scale
        )
        mixed = mixed.transpose(1, 2).contiguous().reshape(
            batch, sequence, c.num_attention_heads * c.head_dim
        )
        mixed = mixed * torch.sigmoid(gate)
        return self.o_proj(mixed), new_cache


class Qwen35MTPMLP(nn.Module):
    def __init__(self, config: Qwen35MTPConfig, ops: MtpOps) -> None:
        super().__init__()
        self.ops = ops
        self.gate_proj = WeightOnlyLinear(
            config.hidden_size, config.intermediate_size, ops
        )
        self.up_proj = WeightOnlyLinear(
            config.hidden_size, config.intermediate_size, ops
        )
        self.down_proj = WeightOnlyLinear(
            config.intermediate_size, config.hidden_size, ops
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.ops.swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen35MTPDecoderLayer(nn.Module):
    def __init__(self, config: Qwen35MTPConfig, ops: MtpOps) -> None:
        super().__init__()
        self.input_layernorm = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps, ops
        )
        self.self_attn = Qwen35MTPAttention(config, ops)
        self.post_attention_layernorm = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps, ops
        )
        self.mlp = Qwen35MTPMLP(config, ops)

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Tensor,
        past_key_values: MTPKVCache | None,
        attention_mask: Tensor | None,
    ) -> tuple[Tensor, MTPKVCache]:
        residual = hidden_states
        mixed, cache = self.self_attn(
            self.input_layernorm(hidden_states),
            position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        hidden_states = residual + mixed
        residual = hidden_states
        hidden_states = residual + self.mlp(
            self.post_attention_layernorm(hidden_states)
        )
        return hidden_states, cache


class Qwen35MTPModule(nn.Module):
    def __init__(self, config: Qwen35MTPConfig, ops: MtpOps) -> None:
        super().__init__()
        self.pre_fc_norm_embedding = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps, ops
        )
        self.pre_fc_norm_hidden = Qwen35RMSNorm(
            config.hidden_size, config.rms_norm_eps, ops
        )
        self.fc = WeightOnlyLinear(config.hidden_size * 2, config.hidden_size, ops)
        self.layers = nn.ModuleList([Qwen35MTPDecoderLayer(config, ops)])
        self.norm = Qwen35RMSNorm(config.hidden_size, config.rms_norm_eps, ops)

    def forward(
        self,
        inputs_embeds: Tensor,
        hidden_sources: Tensor,
        position_ids: Tensor,
        past_key_values: MTPKVCache | None = None,
        attention_mask: Tensor | None = None,
    ) -> tuple[Tensor, MTPKVCache]:
        if inputs_embeds.shape != hidden_sources.shape:
            raise ValueError("MTP embeddings and hidden sources must have identical shapes")
        combined = torch.cat(
            (
                self.pre_fc_norm_embedding(inputs_embeds),
                self.pre_fc_norm_hidden(hidden_sources),
            ),
            dim=-1,
        )
        hidden_states = self.fc(combined)
        hidden_states, cache = self.layers[0](
            hidden_states,
            position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        return self.norm(hidden_states), cache


@dataclass(frozen=True)
class DraftState:
    cache: MTPKVCache
    last_hidden_state: Tensor
    next_token_ids: Tensor


class Qwen35MTPDrafter(nn.Module):
    """Official Qwen3.5 MTP block with tied main embedding/LM-head weight."""

    def __init__(
        self,
        config: Qwen35MTPConfig,
        embedding: nn.Embedding,
        *,
        ops: MtpOps | None = None,
    ) -> None:
        super().__init__()
        if embedding.num_embeddings != config.vocab_size:
            raise ValueError("embedding vocabulary does not match the MTP config")
        if embedding.embedding_dim != config.hidden_size:
            raise ValueError("embedding width does not match the MTP config")
        self.config = config
        self.ops = ops or TorchMtpOps()
        self.embed_tokens = embedding
        self.mtp = Qwen35MTPModule(config, self.ops)

    @property
    def lm_head_weight(self) -> Tensor:
        return self.embed_tokens.weight

    def forward(
        self,
        input_ids: Tensor,
        hidden_sources: Tensor,
        position_ids: Tensor,
        *,
        past_key_values: MTPKVCache | None = None,
        attention_mask: Tensor | None = None,
        project_top1: bool = False,
    ) -> MTPOutput:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must be [batch, sequence]")
        if hidden_sources.shape[:2] != input_ids.shape:
            raise ValueError("hidden_sources must align with input_ids")
        if position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must align with input_ids")
        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states, cache = self.mtp(
            inputs_embeds,
            hidden_sources,
            position_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
        )
        top1 = (
            self.ops.top1(hidden_states[:, -1:, :], self.lm_head_weight)
            if project_top1
            else None
        )
        return MTPOutput(hidden_states=hidden_states, cache=cache, top1_token_ids=top1)

    @torch.inference_mode()
    def prefill(self, prefix_ids: Tensor, main_hidden_states: Tensor) -> DraftState:
        """Rebuild draft cache from committed tokens with the official one-token shift."""

        if prefix_ids.ndim != 2 or prefix_ids.shape[1] < 2:
            raise ValueError("MTP prefill needs at least two committed tokens")
        if main_hidden_states.shape[:2] != prefix_ids.shape:
            raise ValueError("main hidden states must cover the committed prefix")
        input_ids = prefix_ids[:, 1:]
        hidden_sources = main_hidden_states[:, :-1, :]
        positions = torch.arange(
            1, prefix_ids.shape[1], device=prefix_ids.device, dtype=torch.long
        ).view(1, -1).expand(prefix_ids.shape[0], -1)
        output = self.forward(
            input_ids,
            hidden_sources,
            positions,
            project_top1=True,
        )
        assert output.top1_token_ids is not None
        return DraftState(
            cache=output.cache,
            last_hidden_state=output.hidden_states[:, -1:, :],
            next_token_ids=output.top1_token_ids[:, -1],
        )

    @torch.inference_mode()
    def propose(
        self,
        prefix_ids: Tensor,
        main_hidden_states: Tensor,
        max_draft_tokens: int,
        *,
        eos_token_ids: Iterable[int] = (),
    ) -> list[int]:
        """Generate a serial NEXTN proposal from a recomputed, committed cache."""

        if prefix_ids.shape[0] != 1:
            raise ValueError("the accuracy-first scheduler currently requires batch size 1")
        if max_draft_tokens <= 0:
            return []
        eos = set(int(token) for token in eos_token_ids)
        state = self.prefill(prefix_ids, main_hidden_states)
        proposals: list[int] = []
        prefix_length = int(prefix_ids.shape[1])
        for index in range(max_draft_tokens):
            token = int(state.next_token_ids.item())
            proposals.append(token)
            if token in eos or index + 1 == max_draft_tokens:
                break
            token_input = torch.tensor(
                [[token]], dtype=torch.long, device=prefix_ids.device
            )
            position = torch.tensor(
                [[prefix_length + index]],
                dtype=torch.long,
                device=prefix_ids.device,
            )
            output = self.forward(
                token_input,
                state.last_hidden_state,
                position,
                past_key_values=state.cache,
                project_top1=True,
            )
            assert output.top1_token_ids is not None
            state = DraftState(
                cache=output.cache,
                last_hidden_state=output.hidden_states[:, -1:, :],
                next_token_ids=output.top1_token_ids[:, -1],
            )
        return proposals

    def load_official_mtp_state(
        self,
        tensors: dict[str, Tensor],
    ) -> None:
        required = set(self.config.required_tensor_shapes())
        if set(tensors) != required:
            missing = sorted(required - set(tensors))
            extra = sorted(set(tensors) - required)
            raise ValueError(f"invalid official MTP state: missing={missing}, extra={extra}")
        incompatible = self.load_state_dict(tensors, strict=False, assign=True)
        allowed_missing = {"embed_tokens.weight"}
        if set(incompatible.missing_keys) != allowed_missing or incompatible.unexpected_keys:
            raise ValueError(
                "MTP state did not map cleanly: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        self.requires_grad_(False)
        self.eval()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        embedding: nn.Embedding | None = None,
        ops: MtpOps | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
    ) -> "Qwen35MTPDrafter":
        config = Qwen35MTPConfig.from_pretrained(model_dir)
        repository = SafeTensorRepository(model_dir)
        target_device = torch.device(device)
        if embedding is None:
            weight = repository.load(
                [EMBEDDING_WEIGHT], device=target_device, dtype=dtype
            )[EMBEDDING_WEIGHT]
            embedding = nn.Embedding.from_pretrained(weight, freeze=True)
        state = repository.load(
            config.required_tensor_shapes(), device=target_device, dtype=dtype
        )
        drafter = cls(config, embedding, ops=ops)
        drafter.to(
            device=target_device,
            dtype=dtype if dtype is not None else embedding.weight.dtype,
        )
        drafter.load_official_mtp_state(state)
        return drafter
