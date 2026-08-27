"""Official Qwen3.5-4B DFlash draft core with an exact incremental KV cache.

The cache-free entry points remain the numerical golden.  Cached entry points
mirror the upstream ``DynamicCache`` lifecycle: every round appends only newly
committed Target features, exposes the current anchor/mask block transiently to
attention, and retains no rejected block rows after the forward completes.
"""

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


class DFlashDraftKVCache:
    """Per-layer committed Draft KV with transactional round staging.

    Cached tensors use ``[B,Hkv,S,D]``.  ``begin_round`` freezes the committed
    boundary, each layer stages ``new Target context + transient Draft block``,
    and ``finish_round`` retains only the context prefix.  A failed forward
    discards staged tensors without modifying the previous committed cache.
    """

    def __init__(self, *, num_layers: int, max_length: int) -> None:
        if isinstance(num_layers, bool) or not isinstance(num_layers, int):
            raise TypeError("num_layers must be an integer")
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer")
        if num_layers <= 0 or max_length <= 0:
            raise ValueError("num_layers and max_length must be positive")
        self.num_layers = num_layers
        self.max_length = max_length
        self._keys: list[Tensor | None] = [None] * num_layers
        self._values: list[Tensor | None] = [None] * num_layers
        self._staged: list[tuple[Tensor, Tensor] | None] | None = None
        self._committed_length = 0
        self._round_base_length: int | None = None
        self._round_context_length: int | None = None
        self._round_block_length: int | None = None
        self._rounds = 0
        self._aborted_rounds = 0
        self._crop_calls = 0
        self._tokens_appended = 0
        self._tokens_reused = 0
        self._peak_committed_length = 0

    @property
    def committed_length(self) -> int:
        return self._committed_length

    def get_seq_length(self) -> int:
        return self._committed_length

    @property
    def round_context_length(self) -> int:
        if self._round_context_length is None:
            raise RuntimeError("no Draft cache round is active")
        return self._round_context_length

    @property
    def round_base_length(self) -> int:
        if self._round_base_length is None:
            raise RuntimeError("no Draft cache round is active")
        return self._round_base_length

    def begin_round(self, *, new_context_length: int, block_length: int) -> None:
        for name, value in (
            ("new_context_length", new_context_length),
            ("block_length", block_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
        if new_context_length < 0 or block_length <= 0:
            raise ValueError(
                "new_context_length must be non-negative and block_length positive"
            )
        if self._staged is not None:
            raise RuntimeError("a Draft cache round is already active")
        context_length = self._committed_length + new_context_length
        if context_length + block_length > self.max_length:
            raise ValueError(
                "Draft cache round exceeds max_length: "
                f"context={context_length}, block={block_length}, "
                f"max_length={self.max_length}"
            )
        self._round_base_length = self._committed_length
        self._round_context_length = context_length
        self._round_block_length = block_length
        self._staged = [None] * self.num_layers

    def update(
        self,
        layer_index: int,
        key_states: Tensor,
        value_states: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Stage one layer and return committed+new+transient KV for attention."""

        if self._staged is None:
            raise RuntimeError("begin_round() must run before Draft cache update")
        if isinstance(layer_index, bool) or not isinstance(layer_index, int):
            raise TypeError("layer_index must be an integer")
        if not 0 <= layer_index < self.num_layers:
            raise IndexError("Draft cache layer_index is out of range")
        if self._staged[layer_index] is not None:
            raise RuntimeError(f"Draft cache layer {layer_index} was updated twice")
        if key_states.ndim != 4 or value_states.ndim != 4:
            raise ValueError("Draft KV tensors must have shape [B,H,S,D]")
        if key_states.shape != value_states.shape:
            raise ValueError("Draft key/value shapes differ")
        if key_states.dtype != value_states.dtype:
            raise ValueError("Draft key/value dtypes differ")
        if key_states.device != value_states.device:
            raise ValueError("Draft key/value devices differ")

        assert self._round_base_length is not None
        assert self._round_context_length is not None
        assert self._round_block_length is not None
        new_context_length = (
            self._round_context_length - self._round_base_length
        )
        expected_rows = new_context_length + self._round_block_length
        if int(key_states.shape[-2]) != expected_rows:
            raise ValueError(
                "Draft cache update rows must contain new context plus block: "
                f"expected {expected_rows}, got {key_states.shape[-2]}"
            )

        old_key = self._keys[layer_index]
        old_value = self._values[layer_index]
        if self._round_base_length == 0:
            if old_key is not None or old_value is not None:
                raise RuntimeError("zero-length Draft cache retained layer tensors")
            combined_key = key_states
            combined_value = value_states
        else:
            if old_key is None or old_value is None:
                raise RuntimeError("committed Draft cache is missing a layer")
            if int(old_key.shape[-2]) != self._round_base_length:
                raise RuntimeError("Draft cache layer length is inconsistent")
            if (
                old_key.shape[:-2] != key_states.shape[:-2]
                or old_key.shape[-1] != key_states.shape[-1]
            ):
                raise ValueError("Draft cache key geometry changed across rounds")
            if old_value.shape != old_key.shape:
                raise RuntimeError("committed Draft key/value geometry differs")
            if old_key.dtype != key_states.dtype or old_key.device != key_states.device:
                raise ValueError("Draft cache dtype/device changed across rounds")
            combined_key = torch.cat((old_key, key_states), dim=-2)
            combined_value = torch.cat((old_value, value_states), dim=-2)

        expected_total = self._round_context_length + self._round_block_length
        if int(combined_key.shape[-2]) != expected_total:
            raise RuntimeError("Draft cache staged an invalid total sequence length")
        self._staged[layer_index] = (combined_key, combined_value)
        return combined_key, combined_value

    def finish_round(self) -> None:
        if self._staged is None:
            raise RuntimeError("no Draft cache round is active")
        if any(item is None for item in self._staged):
            missing = [
                index for index, item in enumerate(self._staged) if item is None
            ]
            raise RuntimeError(f"Draft cache layers were not updated: {missing}")
        assert self._round_base_length is not None
        assert self._round_context_length is not None
        context_length = self._round_context_length
        staged = self._staged
        next_keys: list[Tensor | None] = []
        next_values: list[Tensor | None] = []
        for item in staged:
            assert item is not None
            key_states, value_states = item
            next_keys.append(key_states[..., :context_length, :])
            next_values.append(value_states[..., :context_length, :])
        appended = context_length - self._round_base_length
        self._keys = next_keys
        self._values = next_values
        self._committed_length = context_length
        self._rounds += 1
        self._tokens_appended += appended
        self._tokens_reused += self._round_base_length
        self._peak_committed_length = max(
            self._peak_committed_length,
            self._committed_length,
        )
        self._clear_round()

    def abort_round(self) -> None:
        if self._staged is not None:
            self._aborted_rounds += 1
        self._clear_round()

    def _clear_round(self) -> None:
        self._staged = None
        self._round_base_length = None
        self._round_context_length = None
        self._round_block_length = None

    def crop(self, length: int) -> None:
        if isinstance(length, bool) or not isinstance(length, int):
            raise TypeError("Draft cache crop length must be an integer")
        if self._staged is not None:
            raise RuntimeError("cannot crop Draft cache during an active round")
        if not 0 <= length <= self._committed_length:
            raise ValueError("Draft cache crop length is outside committed state")
        if length == self._committed_length:
            return
        if length == 0:
            self._keys = [None] * self.num_layers
            self._values = [None] * self.num_layers
        else:
            self._keys = [
                None if item is None else item[..., :length, :]
                for item in self._keys
            ]
            self._values = [
                None if item is None else item[..., :length, :]
                for item in self._values
            ]
        self._committed_length = length
        self._crop_calls += 1

    def clear(self) -> None:
        self._keys = [None] * self.num_layers
        self._values = [None] * self.num_layers
        self._committed_length = 0
        self._clear_round()

    @property
    def audit(self) -> dict[str, object]:
        logical_elements = 0
        for key_states, value_states in zip(self._keys, self._values):
            if key_states is not None:
                logical_elements += key_states.numel()
            if value_states is not None:
                logical_elements += value_states.numel()
        element_size = next(
            (
                item.element_size()
                for item in (*self._keys, *self._values)
                if item is not None
            ),
            0,
        )
        return {
            "enabled": True,
            "mode": "upstream_equivalent_append_then_crop",
            "num_layers": self.num_layers,
            "max_length": self.max_length,
            "committed_length": self._committed_length,
            "active_round": self._staged is not None,
            "rounds": self._rounds,
            "aborted_rounds": self._aborted_rounds,
            "crop_calls": self._crop_calls,
            "tokens_appended": self._tokens_appended,
            "tokens_reused": self._tokens_reused,
            "peak_committed_length": self._peak_committed_length,
            "logical_bytes": logical_elements * element_size,
        }


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
        # Match vLLM 0.27.1's per-layer DFlash resolver: sliding-attention
        # layers are causal unless dflash_config.causal overrides them, while
        # the full-attention layer is non-causal over the parallel draft block.
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

    def forward_cached(
        self,
        hidden_states: Tensor,
        new_target_hidden: Tensor,
        cosine: Tensor,
        sine: Tensor,
        cache: DFlashDraftKVCache,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Attend with committed context KV plus this round's transient block."""

        batch, query_length, _ = hidden_states.shape
        new_context_length = int(new_target_hidden.shape[1])
        config = self.config
        query = self.q_proj(hidden_states).reshape(
            batch, query_length, config.num_attention_heads, config.head_dim
        )
        query = self.q_norm(query).transpose(1, 2)

        key_context = self.k_proj(new_target_hidden)
        key_noise = self.k_proj(hidden_states)
        value_context = self.v_proj(new_target_hidden)
        value_noise = self.v_proj(hidden_states)
        key_new = torch.cat((key_context, key_noise), dim=1).reshape(
            batch,
            new_context_length + query_length,
            config.num_key_value_heads,
            config.head_dim,
        )
        value_new = torch.cat((value_context, value_noise), dim=1).reshape(
            batch,
            new_context_length + query_length,
            config.num_key_value_heads,
            config.head_dim,
        )
        key_new = self.k_norm(key_new).transpose(1, 2)
        value_new = value_new.transpose(1, 2)
        query, key_new = self.ops.rotary(query, key_new, cosine, sine)
        key, value = cache.update(self.layer_index, key_new, value_new)
        context_length = cache.round_context_length

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
            raise ValueError("attention_mask has an incompatible cached query/key shape")

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

    def forward_cached(
        self,
        hidden_states: Tensor,
        new_target_hidden: Tensor,
        cosine: Tensor,
        sine: Tensor,
        cache: DFlashDraftKVCache,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        residual = hidden_states
        hidden_states = residual + self.self_attn.forward_cached(
            self.input_layernorm(hidden_states),
            new_target_hidden,
            cosine,
            sine,
            cache,
            attention_mask,
        )
        residual = hidden_states
        return residual + self.mlp(self.post_attention_layernorm(hidden_states))


class DFlashDraftModel(nn.Module):
    """Official six-layer drafter with cache-free and incremental entry points.

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
        projected = self.project_target_hidden(target_hidden)
        return self.forward_projected(
            projected,
            noise_embedding,
            position_ids,
            attention_mask,
        )

    def project_target_hidden(self, target_hidden: Tensor) -> Tensor:
        """Project raw eight-layer Target features exactly once per token."""

        if target_hidden.ndim != 3:
            raise ValueError("target_hidden must be rank-3")
        if target_hidden.shape[-1] != self.config.feature_size:
            raise ValueError(
                f"target feature width must be {self.config.feature_size}, "
                f"got {target_hidden.shape[-1]}"
            )
        return self.hidden_norm(self.fc(target_hidden))

    def new_kv_cache(
        self,
        *,
        max_length: int | None = None,
    ) -> DFlashDraftKVCache:
        """Create request-local Draft KV state without allocating model weights."""

        effective_maximum = (
            int(self.config.max_position_embeddings)
            if max_length is None
            else int(max_length)
        )
        return DFlashDraftKVCache(
            num_layers=len(self.layers),
            max_length=effective_maximum,
        )

    def forward_projected(
        self,
        projected_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run the Draft block from cached ``[B,C,hidden_size]`` features."""

        if projected_target_hidden.ndim != 3 or noise_embedding.ndim != 3:
            raise ValueError(
                "projected_target_hidden and noise_embedding must be rank-3"
            )
        if projected_target_hidden.shape[0] != noise_embedding.shape[0]:
            raise ValueError("target and noise batches differ")
        if projected_target_hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"projected target width must be {self.config.hidden_size}, "
                f"got {projected_target_hidden.shape[-1]}"
            )
        if noise_embedding.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"noise embedding width must be {self.config.hidden_size}, "
                f"got {noise_embedding.shape[-1]}"
            )
        # Match upstream DFlash: block_size is the total query-row count,
        # including the clean anchor.  block_size=16 therefore permits at
        # most 15 proposal/mask rows.
        maximum_query_rows = self.config.block_size
        if not 1 <= noise_embedding.shape[1] <= maximum_query_rows:
            raise ValueError(
                f"noise length must be in [1, {maximum_query_rows}]"
            )
        expected_positions = (
            projected_target_hidden.shape[1] + noise_embedding.shape[1]
        )
        if position_ids.ndim != 2 or position_ids.shape != (
            noise_embedding.shape[0],
            expected_positions,
        ):
            raise ValueError(
                "position_ids must cover target context followed by the draft block"
            )
        if projected_target_hidden.dtype != noise_embedding.dtype:
            raise ValueError("target_hidden and noise_embedding dtypes differ")
        if projected_target_hidden.device != noise_embedding.device:
            raise ValueError("target_hidden and noise_embedding devices differ")

        hidden_states = noise_embedding
        cosine, sine = self.rotary(position_ids, hidden_states.dtype)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                projected_target_hidden,
                cosine,
                sine,
                attention_mask,
            )
        return self.norm(hidden_states)

    def forward_cached_projected(
        self,
        new_projected_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        cache: DFlashDraftKVCache,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        """Run one block while appending only newly committed Target features.

        ``new_projected_target_hidden`` is the prompt on the first round and
        only ``accepted + correction/bonus input`` rows on later rounds.  The
        cache already owns every older committed context row.
        """

        if not isinstance(cache, DFlashDraftKVCache):
            raise TypeError("cache must be DFlashDraftKVCache")
        if cache.num_layers != len(self.layers):
            raise ValueError("Draft cache layer count differs from the model")
        if new_projected_target_hidden.ndim != 3 or noise_embedding.ndim != 3:
            raise ValueError(
                "new_projected_target_hidden and noise_embedding must be rank-3"
            )
        if new_projected_target_hidden.shape[0] != noise_embedding.shape[0]:
            raise ValueError("target and noise batches differ")
        if new_projected_target_hidden.shape[-1] != self.config.hidden_size:
            raise ValueError(
                f"projected target width must be {self.config.hidden_size}, "
                f"got {new_projected_target_hidden.shape[-1]}"
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
        if new_projected_target_hidden.dtype != noise_embedding.dtype:
            raise ValueError("target_hidden and noise_embedding dtypes differ")
        if new_projected_target_hidden.device != noise_embedding.device:
            raise ValueError("target_hidden and noise_embedding devices differ")

        base_length = cache.committed_length
        new_context_length = int(new_projected_target_hidden.shape[1])
        block_length = int(noise_embedding.shape[1])
        context_length = base_length + new_context_length
        complete_position_rows = context_length + block_length
        incremental_position_rows = new_context_length + block_length
        if position_ids.ndim != 2 or position_ids.shape[0] != noise_embedding.shape[0]:
            raise ValueError(
                "position_ids must use the Draft batch and cover either the "
                "incremental tail or the complete cached sequence"
            )
        if int(position_ids.shape[1]) == incremental_position_rows:
            tail_positions = position_ids
        elif int(position_ids.shape[1]) == complete_position_rows:
            # Backwards-compatible cache-free-shaped input. Production callers
            # pass only the incremental tail to avoid allocating O(C) position
            # metadata on every round.
            tail_positions = position_ids[:, base_length:]
        else:
            raise ValueError(
                "position_ids must cover either new context plus block or "
                "cached context plus new context plus block"
            )

        cache.begin_round(
            new_context_length=new_context_length,
            block_length=block_length,
        )
        try:
            # Old context keys are already rotary-encoded.  Encode only the
            # newly committed context and the transient block at absolute
            # positions starting from the committed cache boundary.
            cosine, sine = self.rotary(tail_positions, noise_embedding.dtype)
            hidden_states = noise_embedding
            for layer in self.layers:
                hidden_states = layer.forward_cached(
                    hidden_states,
                    new_projected_target_hidden,
                    cosine,
                    sine,
                    cache,
                    attention_mask,
                )
            hidden_states = self.norm(hidden_states)
            cache.finish_round()
        except Exception:
            cache.abort_round()
            raise
        return hidden_states

    def forward_cached(
        self,
        new_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        cache: DFlashDraftKVCache,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        projected = self.project_target_hidden(new_target_hidden)
        return self.forward_cached_projected(
            projected,
            noise_embedding,
            position_ids,
            cache,
            attention_mask,
        )

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

    def draft_hidden_projected(
        self,
        projected_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if noise_embedding.shape[1] < 2:
            raise ValueError("a DFlash draft needs one anchor and at least one mask token")
        return self.forward_projected(
            projected_target_hidden,
            noise_embedding,
            position_ids,
            attention_mask,
        )[:, 1:, :]

    def draft_hidden_cached_projected(
        self,
        new_projected_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        cache: DFlashDraftKVCache,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        if noise_embedding.shape[1] < 2:
            raise ValueError("a DFlash draft needs one anchor and at least one mask token")
        return self.forward_cached_projected(
            new_projected_target_hidden,
            noise_embedding,
            position_ids,
            cache,
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

    def draft_top1_projected(
        self,
        projected_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        lm_head_weight: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.draft_hidden_projected(
            projected_target_hidden,
            noise_embedding,
            position_ids,
            attention_mask,
        )
        return self.ops.top1(hidden, lm_head_weight)

    def draft_top1_cached_projected(
        self,
        new_projected_target_hidden: Tensor,
        noise_embedding: Tensor,
        position_ids: Tensor,
        cache: DFlashDraftKVCache,
        lm_head_weight: Tensor,
        attention_mask: Tensor | None = None,
    ) -> Tensor:
        hidden = self.draft_hidden_cached_projected(
            new_projected_target_hidden,
            noise_embedding,
            position_ids,
            cache,
            attention_mask,
        )
        return self.ops.top1(hidden, lm_head_weight)

    def embed_block(self, block_ids: Tensor, embedding_weight: Tensor) -> Tensor:
        if block_ids.ndim != 2:
            raise ValueError("block_ids must have shape [batch, block]")
        return F.embedding(block_ids, embedding_weight) * self.config.input_embedding_scale
