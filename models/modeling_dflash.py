"""Cache-free PyTorch golden for the official Qwen3.5-4B DFlash draft core."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .dflash_config import Qwen35DFlashConfig, audit_official_4b_dflash_config
from .dflash_ops import DFlashOps, TorchDFlashOps


def extract_context_feature(
    hidden_states: Sequence[Tensor],
    layer_ids: Sequence[int],
) -> Tensor:
    """Concatenate target decoder outputs selected by DFlash layer IDs.

    Transformers includes the embedding output at index zero, hence the +1.
    """

    if not layer_ids:
        raise ValueError("layer_ids cannot be empty")
    required = max(layer_ids) + 2
    if len(hidden_states) < required:
        raise ValueError(
            f"target returned {len(hidden_states)} hidden states; {required} are required"
        )
    selected = [hidden_states[layer_id + 1] for layer_id in layer_ids]
    reference_shape = selected[0].shape
    if any(item.shape != reference_shape for item in selected[1:]):
        raise ValueError("selected target hidden states have inconsistent shapes")
    return torch.cat(selected, dim=-1)


class WeightOnlyLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ops: DFlashOps,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device=device, dtype=dtype)
        )
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        return self.ops.linear(x, self.weight)


class DFlashRMSNorm(nn.Module):
    def __init__(
        self,
        dim: int,
        eps: float,
        ops: DFlashOps,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(dim, device=device, dtype=dtype))
        self.eps = eps
        self.ops = ops

    def forward(self, x: Tensor) -> Tensor:
        return self.ops.rms_norm(x, self.weight, self.eps)


class DFlashRotaryEmbedding(nn.Module):
    def __init__(
        self,
        config: Qwen35DFlashConfig,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__()
        # Preserve the CPU oracle's FP32 constant construction, then place the
        # small immutable buffer beside the directly constructed parameters.
        # Without this transfer an NPU draft fails at its first RoPE multiply.
        inv_freq = 1.0 / (
            config.rope_theta
            ** (
                torch.arange(0, config.head_dim, 2, dtype=torch.float32)
                / config.head_dim
            )
        )
        if device is not None:
            inv_freq = inv_freq.to(device=device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self,
        position_ids: Tensor,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor]:
        if position_ids.device != self.inv_freq.device:
            raise ValueError(
                "position_ids and rotary inv_freq must share one device: "
                f"got {position_ids.device} and {self.inv_freq.device}"
            )
        frequencies = position_ids.float().unsqueeze(-1) * self.inv_freq.view(1, 1, -1)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        return embedding.cos().to(dtype=dtype), embedding.sin().to(dtype=dtype)


class DFlashAttention(nn.Module):
    def __init__(
        self,
        config: Qwen35DFlashConfig,
        layer_index: int,
        ops: DFlashOps,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.ops = ops
        self.q_proj = WeightOnlyLinear(
            config.hidden_size, config.query_width, ops, device=device, dtype=dtype
        )
        self.k_proj = WeightOnlyLinear(
            config.hidden_size, config.key_value_width, ops, device=device, dtype=dtype
        )
        self.v_proj = WeightOnlyLinear(
            config.hidden_size, config.key_value_width, ops, device=device, dtype=dtype
        )
        self.o_proj = WeightOnlyLinear(
            config.query_width, config.hidden_size, ops, device=device, dtype=dtype
        )
        self.q_norm = DFlashRMSNorm(
            config.head_dim, config.rms_norm_eps, ops, device=device, dtype=dtype
        )
        self.k_norm = DFlashRMSNorm(
            config.head_dim, config.rms_norm_eps, ops, device=device, dtype=dtype
        )
        layer_type = config.layer_types[layer_index]
        # This intentionally matches public DFlash: sliding layers are causal;
        # the final full-attention layer is bidirectional over the draft block.
        self.is_causal = layer_type == "sliding_attention"
        self.sliding_window = (
            config.sliding_window
            if layer_type == "sliding_attention" and config.use_sliding_window
            else None
        )
        self.scale = config.head_dim**-0.5

    def _attention_mask(
        self,
        query_length: int,
        context_length: int,
        *,
        device: torch.device,
    ) -> Tensor | None:
        if not self.is_causal and self.sliding_window is None:
            return None
        key_length = context_length + query_length
        query_positions = context_length + torch.arange(
            query_length, device=device
        ).view(query_length, 1)
        key_positions = torch.arange(key_length, device=device).view(1, key_length)
        visible = torch.ones(
            (query_length, key_length), dtype=torch.bool, device=device
        )
        if self.is_causal:
            visible &= key_positions <= query_positions
        if self.sliding_window is not None:
            visible &= query_positions - key_positions < self.sliding_window
            if not self.is_causal:
                visible &= key_positions - query_positions < self.sliding_window
        return visible.view(1, 1, query_length, key_length)

    def forward(
        self,
        hidden_states: Tensor,
        target_hidden: Tensor,
        cosine: Tensor,
        sine: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        batch, query_length, _ = hidden_states.shape
        context_length = target_hidden.shape[1]
        config = self.config
        query = self.q_proj(hidden_states).reshape(
            batch, query_length, config.num_attention_heads, config.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)

        key_context = self.k_proj(target_hidden)
        key_noise = self.k_proj(hidden_states)
        value_context = self.v_proj(target_hidden)
        value_noise = self.v_proj(hidden_states)
        key = torch.cat((key_context, key_noise), dim=1).reshape(
            batch,
            context_length + query_length,
            config.num_key_value_heads,
            config.head_dim,
        )
        value = torch.cat((value_context, value_noise), dim=1).reshape(
            batch,
            context_length + query_length,
            config.num_key_value_heads,
            config.head_dim,
        )
        key = self.k_norm(key).transpose(1, 2)
        value = value.transpose(1, 2)
        query, key = self.ops.rotary(query, key, cosine, sine)

        if attention_mask is None:
            attention_mask = self._attention_mask(
                query_length,
                context_length,
                device=hidden_states.device,
            )
        elif tuple(attention_mask.shape[-2:]) != (
            query_length,
            context_length + query_length,
        ):
            raise ValueError("attention_mask has an incompatible query/key shape")

        mixed = self.ops.attention(
            query,
            key,
            value,
            attention_mask,
            self.scale,
            config.num_key_value_groups,
        )
        mixed = mixed.transpose(1, 2).contiguous().reshape(
            batch, query_length, config.query_width
        )
        return self.o_proj(mixed)


class DFlashMLP(nn.Module):
    def __init__(
        self,
        config: Qwen35DFlashConfig,
        ops: DFlashOps,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.ops = ops
        self.gate_proj = WeightOnlyLinear(
            config.hidden_size,
            config.intermediate_size,
            ops,
            device=device,
            dtype=dtype,
        )
        self.up_proj = WeightOnlyLinear(
            config.hidden_size,
            config.intermediate_size,
            ops,
            device=device,
            dtype=dtype,
        )
        self.down_proj = WeightOnlyLinear(
            config.intermediate_size,
            config.hidden_size,
            ops,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(self.ops.swiglu(self.gate_proj(x), self.up_proj(x)))


class DFlashDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen35DFlashConfig,
        layer_index: int,
        ops: DFlashOps,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.input_layernorm = DFlashRMSNorm(
            config.hidden_size, config.rms_norm_eps, ops, device=device, dtype=dtype
        )
        self.self_attn = DFlashAttention(
            config, layer_index, ops, device=device, dtype=dtype
        )
        self.post_attention_layernorm = DFlashRMSNorm(
            config.hidden_size, config.rms_norm_eps, ops, device=device, dtype=dtype
        )
        self.mlp = DFlashMLP(config, ops, device=device, dtype=dtype)

    def forward(
        self,
        hidden_states: Tensor,
        target_hidden: Tensor,
        cosine: Tensor,
        sine: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn(
            self.input_layernorm(hidden_states),
            target_hidden,
            cosine,
            sine,
            attention_mask,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class DFlashDraftModel(nn.Module):
    """Official six-layer drafter, without cache or speculative scheduling.

    ``target_hidden`` contains concatenated hidden features from the eight
    configured target layers. ``noise_embedding`` contains one clean anchor
    embedding followed by mask-token embeddings. The returned row zero belongs
    to the anchor; rows ``1:`` are the parallel draft predictions.
    """

    def __init__(
        self,
        config: Qwen35DFlashConfig,
        ops: DFlashOps | None = None,
        *,
        device: str | torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.ops = ops or TorchDFlashOps()
        self.layers = nn.ModuleList(
            [
                DFlashDecoderLayer(
                    config, index, self.ops, device=device, dtype=dtype
                )
                for index in range(config.num_hidden_layers)
            ]
        )
        self.norm = DFlashRMSNorm(
            config.hidden_size, config.rms_norm_eps, self.ops, device=device, dtype=dtype
        )
        self.rotary = DFlashRotaryEmbedding(config, device=device)
        self.fc = WeightOnlyLinear(
            config.feature_size,
            config.hidden_size,
            self.ops,
            device=device,
            dtype=dtype,
        )
        self.hidden_norm = DFlashRMSNorm(
            config.hidden_size, config.rms_norm_eps, self.ops, device=device, dtype=dtype
        )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        ops: DFlashOps | None = None,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "DFlashDraftModel":
        from .dflash_weights import load_dflash_weights

        config = Qwen35DFlashConfig.from_pretrained(model_dir)
        mismatches = audit_official_4b_dflash_config(config)
        if mismatches:
            raise RuntimeError(
                "draft config is not the locked official Qwen3.5-4B-DFlash "
                "contract: " + "; ".join(mismatches)
            )
        model = cls(config, ops=ops, device=device, dtype=dtype)
        load_dflash_weights(model, model_dir)
        model.eval()
        return model

    def set_ops(self, ops: DFlashOps) -> None:
        """Replace all primitive dispatchers without rebuilding or copying weights."""

        self.ops = ops
        for module in self.modules():
            if module is not self and hasattr(module, "ops"):
                module.ops = ops

    def forward(
        self,
        target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if target_hidden.ndim != 3 or noise_embedding.ndim != 3:
            raise ValueError("target_hidden and noise_embedding must be rank-3")
        if target_hidden.shape[0] != noise_embedding.shape[0]:
            raise ValueError("target and noise batches differ")
        if target_hidden.shape[-1] != self.config.feature_size:
            raise ValueError(
                f"target feature width must be {self.config.feature_size}, "
                f"got {target_hidden.shape[-1]}"
            )
        if noise_embedding.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"noise embedding width must be {self.config.hidden_size}, "
                f"got {noise_embedding.shape[-1]}"
            )
        if not 1 <= noise_embedding.shape[1] <= self.config.block_size:
            raise ValueError(
                f"noise length must be in [1, {self.config.block_size}]"
            )
        expected_positions = target_hidden.shape[1] + noise_embedding.shape[1]
        if position_ids.ndim != 2 or position_ids.shape != (
            noise_embedding.shape[0],
            expected_positions,
        ):
            raise ValueError(
                "position_ids must cover target context followed by the draft block"
            )
        if target_hidden.dtype != noise_embedding.dtype:
            raise ValueError("target_hidden and noise_embedding dtypes differ")

        target_hidden = self.hidden_norm(self.fc(target_hidden))
        hidden_states = noise_embedding
        cosine, sine = self.rotary(position_ids, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                target_hidden,
                cosine,
                sine,
                attention_mask,
            )
        return self.norm(hidden_states)

    def draft_hidden(
        self,
        target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if noise_embedding.shape[1] < 2:
            raise ValueError("a DFlash draft needs one anchor and at least one mask token")
        return self(
            target_hidden,
            noise_embedding,
            position_ids,
            attention_mask,
        )[:, 1:, :]

    def compute_logits(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        logits = self.ops.linear(hidden, lm_head_weight)
        logits = logits * self.config.output_multiplier
        softcap = self.config.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def draft_top1(
        self,
        target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        lm_head_weight: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.draft_hidden(
            target_hidden,
            noise_embedding,
            position_ids,
            attention_mask,
        )
        # Positive scaling and tanh softcapping are monotonic, so they cannot
        # change Top1. This boundary permits a fused LM-head + argmax operator.
        return self.ops.top1(hidden, lm_head_weight)

    def embed_block(self, block_ids: Tensor, embedding_weight: Tensor) -> Tensor:
        if block_ids.ndim != 2:
            raise ValueError("block_ids must have shape [batch, block]")
        return F.embedding(block_ids, embedding_weight) * self.config.input_embedding_scale
