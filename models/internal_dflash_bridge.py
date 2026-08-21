"""Bridge a Qwen3.5 HIAI wrapper to DFlash V1.

The HIAI target owns model loading and its custom
operators.  DFlash V1, however, asks the target to evaluate a complete prefix
on every call and does not carry target cache state across calls.  This module
adapts those two interfaces without replacing attention, GDN, CacheUpdate, or
any other target operator.

Each call builds a fresh hybrid cache from the model configuration:

* linear-attention layers receive ``(conv_state, recurrent_state)``;
* full-attention layers receive block-table ``(key_cache, value_cache)``;
* a multi-token prefix is right-padded to a complete 64-token prefill chunk;
* the complete padded prefix is executed as one fresh prefill and sliced back
  to the real token rows;
* only ``logits`` and optional ``dflash_features`` cross back to DFlash.

The explicit chunk alignment is important for the target GDN kernel: its
multi-token path uses ``chunk_size=64``.  It is also important that the device
is synchronized before the call-local KV/GDN tensors leave scope; otherwise an
asynchronous custom operator may still be consuming storage which the caching
allocator is free to reuse for the next full-prefix call.

The public factory is :func:`load_qwen35_target`.  It is consumed by
``models.dflash_v1.internal_target_loader`` and needs no hand-written reset
hook.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import os
from typing import Any

import torch
from torch import Tensor, nn


KV_CACHE_MAX_LEN_ENV = "DFLASH_HIAI_KV_CACHE_MAX_LEN"
BLOCK_SIZE = 64
FEATURE_WIDTH = 20_480
VOCAB_SIZE = 248_320

_FEATURE_SOURCE = "package_local:modeling_qwen3_5_hiai_nd.py"
_CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
_FEATURE_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a positive integer")
    try:
        normalized = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a positive integer") from error
    if normalized <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _round_up_prefill_length(sequence_length: int) -> int:
    """Return the execution length required by the target GDN chunk path."""

    length = _positive_int(sequence_length, name="sequence_length")
    if length == 1:
        # The target intentionally selects its recurrent chunk_size=1 path for
        # a single-token request.
        return 1
    return ((length + BLOCK_SIZE - 1) // BLOCK_SIZE) * BLOCK_SIZE


def _tensor_field(output: object, name: str) -> Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    return value if isinstance(value, Tensor) else None


def _unwrap_logits_and_features(
    output: object,
    *,
    feature_enabled: bool,
) -> tuple[Tensor, Tensor | None]:
    """Normalize the receiver's Tensor/tuple/sidecar output ABI."""

    features = _tensor_field(output, "dflash_features")
    base = getattr(output, "base_output", output)
    logits = _tensor_field(base, "logits")
    if logits is None and isinstance(base, Tensor):
        logits = base
    if isinstance(base, (tuple, list)):
        if not base or not isinstance(base[0], Tensor):
            raise TypeError("HIAI target tuple output must start with Tensor logits")
        logits = base[0]
        if feature_enabled and features is None and len(base) > 1:
            candidate = base[1]
            if isinstance(candidate, Tensor):
                features = candidate
    if logits is None and isinstance(output, (tuple, list)):
        if not output or not isinstance(output[0], Tensor):
            raise TypeError("HIAI target tuple output must start with Tensor logits")
        logits = output[0]
        if feature_enabled and features is None and len(output) > 1:
            candidate = output[1]
            if isinstance(candidate, Tensor):
                features = candidate
    if logits is None:
        raise TypeError("HIAI target output does not expose Tensor logits")
    if feature_enabled and features is None:
        raise TypeError(
            "feature-enabled HIAI target did not return dflash_features; "
            "check the direct collector in modeling_qwen3_5_hiai_nd.py"
        )
    if not feature_enabled and features is not None:
        raise RuntimeError(
            "HIAI target returned dflash_features while the feature route was disabled"
        )
    return logits, features


class InternalDFlashTarget(nn.Module):
    """Stateless DFlash-facing facade over the receiver's loaded HIAI model."""

    dflash_feature_source = _FEATURE_SOURCE
    dflash_feature_capture_point = _CAPTURE_POINT
    dflash_feature_contract_id = _FEATURE_CONTRACT_ID
    dflash_full_prefix_isolation_mode = "receiver_reset_hook"
    dflash_full_prefix_isolation_evidence = (
        "bridge allocates fresh external hybrid KV/GDN state, aligns every "
        "multi-token prefill to 64 rows, and synchronizes before releasing "
        "call-local state"
    )
    dflash_full_prefix_execution_mode = "fresh_prefill"
    dflash_prefill_chunk_size = 64
    dflash_decode_chunk_size = 1

    def __init__(
        self,
        model_wrapper: nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
        kv_cache_max_len: int,
    ) -> None:
        super().__init__()
        if not isinstance(model_wrapper, nn.Module):
            raise TypeError("Qwen3_5ForCausalLMWrapper must be torch.nn.Module")
        execution_model = getattr(model_wrapper, "model", None)
        if not isinstance(execution_model, nn.Module):
            raise TypeError(
                "Qwen3_5ForCausalLMWrapper must expose its HIAI model as .model"
            )
        if dtype is not torch.float16:
            raise ValueError("the current HIAI DFlash bridge supports FP16 only")
        self.model_wrapper = model_wrapper
        self.requested_device = torch.device(device)
        self.requested_dtype = dtype
        self.kv_cache_max_len = _positive_int(
            kv_cache_max_len,
            name="kv_cache_max_len",
        )
        if self.kv_cache_max_len % BLOCK_SIZE != 0:
            raise ValueError("kv_cache_max_len must be divisible by block_size=64")
        self.config.kv_cache_max_len = self.kv_cache_max_len
        self.dflash_full_attention_block_tables_rebuilt = (
            self._rebuild_full_attention_block_tables()
        )
        self._prepared_call: tuple[int, bool, int, int] | None = None
        self._full_prefix_calls = 0
        self._full_prefix_completions = 0
        self._full_prefix_failures = 0
        self._device_synchronizations = 0
        self._total_padding_tokens = 0
        self._last_requested_sequence_length: int | None = None
        self._last_execution_sequence_length: int | None = None

    @property
    def dflash_full_prefix_bridge_audit(self) -> Mapping[str, object]:
        """Expose bounded runtime facts without returning mutable model state."""

        return {
            "prefill_alignment": "right_pad_s_gt_1_to_multiple_of_64",
            "call_local_state_release_barrier": True,
            "full_prefix_calls": self._full_prefix_calls,
            "full_prefix_completions": self._full_prefix_completions,
            "full_prefix_failures": self._full_prefix_failures,
            "device_synchronizations": self._device_synchronizations,
            "total_padding_tokens": self._total_padding_tokens,
            "last_requested_sequence_length": self._last_requested_sequence_length,
            "last_execution_sequence_length": self._last_execution_sequence_length,
        }

    @property
    def dflash_execution_model(self) -> nn.Module:
        """The package-local HIAI class that actually executes target math."""

        model = getattr(self.model_wrapper, "model", None)
        if not isinstance(model, nn.Module):
            raise RuntimeError("receiver wrapper lost its .model execution module")
        return model

    @property
    def config(self) -> object:
        config = getattr(self.dflash_execution_model, "config", None)
        if config is None:
            raise TypeError("HIAI execution model must expose config")
        return config

    def get_input_embeddings(self) -> nn.Module:
        getter = getattr(self.dflash_execution_model, "get_input_embeddings", None)
        if not callable(getter):
            raise TypeError("HIAI execution model lacks get_input_embeddings()")
        module = getter()
        if not isinstance(module, nn.Module):
            raise TypeError("get_input_embeddings() must return torch.nn.Module")
        return module

    def get_output_embeddings(self) -> nn.Module:
        getter = getattr(self.dflash_execution_model, "get_output_embeddings", None)
        module = getter() if callable(getter) else None
        if module is None:
            module = getattr(self.dflash_execution_model, "lm_head", None)
        if not isinstance(module, nn.Module):
            raise TypeError("HIAI execution model must expose its LM head")
        return module

    def _rebuild_full_attention_block_tables(self) -> int:
        """Keep per-layer block tables aligned with the bridge cache length."""

        layer_types = tuple(getattr(self.config, "layer_types", ()))
        expected = sum(item == "full_attention" for item in layer_types)
        rebuilt = 0
        expected_blocks = self.kv_cache_max_len // BLOCK_SIZE
        for module in self.dflash_execution_model.modules():
            rebuild = getattr(module, "_rebuild_block_table", None)
            if not callable(rebuild):
                continue
            rebuild()
            rebuilt += 1
            if int(getattr(module, "kv_max_len", -1)) != self.kv_cache_max_len:
                raise RuntimeError(
                    "full-attention layer kept a stale kv_cache_max_len"
                )
            block_table = getattr(module, "block_table", None)
            if not isinstance(block_table, Tensor) or tuple(block_table.shape) != (
                1,
                expected_blocks,
            ):
                raise RuntimeError(
                    "full-attention layer rebuilt an invalid block_table shape"
                )
        if rebuilt != expected:
            raise RuntimeError(
                "HIAI full-attention block-table rebuild count differs from "
                f"config.layer_types: expected {expected}, got {rebuilt}"
            )
        return rebuilt

    def prepare_dflash_full_prefix_call(
        self,
        *,
        input_ids: Tensor,
        sequence_length: int,
        output_dflash_features: bool,
        logits_to_keep: int,
        call_index: int,
    ) -> None:
        """Arm exactly one call; all mutable state is allocated in ``forward``."""

        if self._prepared_call is not None:
            raise RuntimeError("a previous DFlash target call was prepared but not run")
        if sequence_length != int(input_ids.shape[1]):
            raise ValueError("prepared sequence_length does not match input_ids")
        self._prepared_call = (
            int(sequence_length),
            bool(output_dflash_features),
            int(logits_to_keep),
            int(call_index),
        )
        return None

    def _fresh_attention_mask(self, sequence_length: int) -> Tensor:
        rows = torch.arange(
            sequence_length,
            device=self.requested_device,
        ).view(sequence_length, 1)
        columns = torch.arange(
            self.kv_cache_max_len,
            device=self.requested_device,
        ).view(1, self.kv_cache_max_len)
        visible = columns <= rows
        zero = torch.zeros((), device=self.requested_device, dtype=torch.float32)
        negative_infinity = torch.full(
            (),
            float("-inf"),
            device=self.requested_device,
            dtype=torch.float32,
        )
        return torch.where(visible, zero, negative_infinity).unsqueeze(0).unsqueeze(0)

    def _execution_input_ids(self, input_ids: Tensor) -> tuple[Tensor, int]:
        """Right-pad a real prefix so every multi-token GDN chunk is complete."""

        real_length = int(input_ids.shape[1])
        execution_length = _round_up_prefill_length(real_length)
        if execution_length > self.kv_cache_max_len:
            raise ValueError(
                "64-token prefill alignment exceeds kv_cache_max_len: "
                f"requested={real_length}, aligned={execution_length}, "
                f"kv_cache_max_len={self.kv_cache_max_len}"
            )
        if execution_length == real_length:
            return input_ids, execution_length

        configured_pad = getattr(self.config, "pad_token_id", None)
        pad_token_id = 0 if configured_pad is None else int(configured_pad)
        vocab_size = _positive_int(
            getattr(self.config, "vocab_size", None),
            name="config.vocab_size",
        )
        if pad_token_id < 0 or pad_token_id >= vocab_size:
            raise ValueError("config.pad_token_id is outside the target vocabulary")
        padding = torch.full(
            (1, execution_length - real_length),
            pad_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        return torch.cat((input_ids, padding), dim=1), execution_length

    def _synchronize_call_local_state(self) -> None:
        """Finish opaque target kernels before fresh state tensors leave scope."""

        if self.requested_device.type == "cpu":
            # CPU is used only by the reduced-shape bridge contract tests; its
            # operations are already synchronous.
            return
        backend = getattr(torch, self.requested_device.type, None)
        synchronize = getattr(backend, "synchronize", None)
        if not callable(synchronize):
            raise RuntimeError(
                f"{self.requested_device.type} backend lacks synchronize(); "
                "cannot safely release call-local KV/GDN state"
            )
        try:
            synchronize(self.requested_device)
        except TypeError:
            synchronize()
        self._device_synchronizations += 1

    def _fresh_hybrid_cache(self, *, batch_size: int) -> list[tuple[Tensor, Tensor]]:
        config = self.config
        layer_types = tuple(getattr(config, "layer_types", ()))
        num_layers = _positive_int(
            getattr(config, "num_hidden_layers", None),
            name="config.num_hidden_layers",
        )
        if len(layer_types) != num_layers:
            raise ValueError("config.layer_types must describe every decoder layer")

        linear_num_value_heads = _positive_int(
            getattr(config, "linear_num_value_heads", None),
            name="config.linear_num_value_heads",
        )
        linear_num_key_heads = _positive_int(
            getattr(config, "linear_num_key_heads", None),
            name="config.linear_num_key_heads",
        )
        linear_key_head_dim = _positive_int(
            getattr(config, "linear_key_head_dim", None),
            name="config.linear_key_head_dim",
        )
        linear_value_head_dim = _positive_int(
            getattr(config, "linear_value_head_dim", None),
            name="config.linear_value_head_dim",
        )
        linear_conv_kernel_dim = _positive_int(
            getattr(config, "linear_conv_kernel_dim", None),
            name="config.linear_conv_kernel_dim",
        )
        linear_conv_dim = (
            (linear_key_head_dim * linear_num_key_heads) * 2
            + (linear_value_head_dim * linear_num_value_heads)
        )

        num_key_value_heads = _positive_int(
            getattr(config, "num_key_value_heads", None),
            name="config.num_key_value_heads",
        )
        head_dim = _positive_int(
            getattr(config, "head_dim", None),
            name="config.head_dim",
        )
        packed_width = num_key_value_heads * head_dim
        if packed_width % 16 != 0:
            raise ValueError("num_key_value_heads * head_dim must be divisible by 16")
        kv_shape = (
            self.kv_cache_max_len // BLOCK_SIZE,
            packed_width // 16,
            BLOCK_SIZE,
            16,
        )

        result: list[tuple[Tensor, Tensor]] = []
        for layer_index, layer_type in enumerate(layer_types):
            if layer_type == "linear_attention":
                conv_state = torch.zeros(
                    (batch_size, linear_conv_dim, linear_conv_kernel_dim),
                    dtype=self.requested_dtype,
                    device=self.requested_device,
                )
                recurrent_state = torch.zeros(
                    (
                        batch_size,
                        linear_num_value_heads,
                        linear_key_head_dim,
                        linear_value_head_dim,
                    ),
                    dtype=self.requested_dtype,
                    device=self.requested_device,
                )
                result.append((conv_state, recurrent_state))
            elif layer_type == "full_attention":
                key_cache = torch.zeros(
                    kv_shape,
                    dtype=self.requested_dtype,
                    device=self.requested_device,
                )
                value_cache = torch.zeros_like(key_cache)
                result.append((key_cache, value_cache))
            else:
                raise ValueError(
                    f"unsupported config.layer_types[{layer_index}]={layer_type!r}"
                )
        return result

    def forward(
        self,
        input_ids: Tensor,
        *,
        use_cache: bool = False,
        return_dict: bool = True,
        output_hidden_states: bool = False,
        output_dflash_features: bool = False,
        logits_to_keep: int = 0,
        **kwargs: Any,
    ) -> Mapping[str, Tensor]:
        """Run one complete prefix using fresh call-local HIAI state."""

        if kwargs:
            raise ValueError(
                "unsupported DFlash target kwargs: " + ", ".join(sorted(kwargs))
            )
        if use_cache is not False or return_dict is not True:
            raise ValueError("outer DFlash target ABI requires use_cache=False/return_dict=True")
        if output_hidden_states is not False:
            raise ValueError("DFlash captures only the selected eight target layers")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("DFlash HIAI bridge supports input_ids shape [1,S]")
        if input_ids.device != self.requested_device:
            raise ValueError(
                f"input_ids are on {input_ids.device}, expected {self.requested_device}"
            )
        sequence_length = int(input_ids.shape[1])
        if sequence_length <= 0 or sequence_length > self.kv_cache_max_len:
            raise ValueError(
                f"prefix length must be in [1,{self.kv_cache_max_len}]"
            )
        expected_prepared = (
            sequence_length,
            bool(output_dflash_features),
            int(logits_to_keep),
        )
        prepared = self._prepared_call
        self._prepared_call = None
        if prepared is None or prepared[:3] != expected_prepared:
            raise RuntimeError(
                "forward must be preceded by a matching "
                "prepare_dflash_full_prefix_call"
            )

        self._full_prefix_calls += 1
        self._last_requested_sequence_length = sequence_length
        execution_input_ids, execution_length = self._execution_input_ids(input_ids)
        self._last_execution_sequence_length = execution_length
        self._total_padding_tokens += execution_length - sequence_length

        embeddings = self.get_input_embeddings()
        inputs_embeds = embeddings(
            execution_input_ids.to(embeddings.weight.device)
        ).to(self.requested_device)
        if inputs_embeds.dtype != self.requested_dtype:
            raise ValueError(
                f"input embeddings use {inputs_embeds.dtype}, expected {self.requested_dtype}"
            )
        attention_mask = self._fresh_attention_mask(execution_length)
        position_ids = torch.arange(
            execution_length,
            device=self.requested_device,
        ).unsqueeze(0)
        new_kv_cache_pos = torch.arange(
            execution_length,
            device=self.requested_device,
        )
        past_key_values = self._fresh_hybrid_cache(batch_size=1)

        try:
            with torch.inference_mode():
                raw_output = self.dflash_execution_model(
                    input_ids=execution_input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    new_kv_cache_pos=new_kv_cache_pos,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    inputs_embeds=inputs_embeds,
                    embed_scale=None,
                    output_pos=None,
                    # Keep the receiver's logical-length input identical to
                    # its normal padded-prefill convention.  Only the tensor
                    # rows are aligned to the physical 64-token GDN chunk.
                    allQLen=[sequence_length],
                    output_dflash_features=bool(output_dflash_features),
                )
            logits, features = _unwrap_logits_and_features(
                raw_output,
                feature_enabled=bool(output_dflash_features),
            )
            expected_logits = (1, execution_length, VOCAB_SIZE)
            if tuple(logits.shape) != expected_logits:
                raise ValueError(
                    "HIAI target logits shape must be "
                    f"{expected_logits}, got {tuple(logits.shape)}"
                )
            if logits.device != self.requested_device:
                raise ValueError("HIAI target logits left the requested NPU device")
            real_logits = logits[:, :sequence_length, :]
            result: dict[str, Tensor] = {
                "logits": (
                    real_logits[:, -1:, :]
                    if int(logits_to_keep) == 1
                    else real_logits
                ),
            }
            if output_dflash_features:
                assert features is not None
                expected_features = (1, execution_length, FEATURE_WIDTH)
                if tuple(features.shape) != expected_features:
                    raise ValueError(
                        "HIAI dflash_features shape must be "
                        f"{expected_features}, got {tuple(features.shape)}"
                    )
                if features.device != self.requested_device:
                    raise ValueError(
                        "HIAI dflash_features left the requested NPU device"
                    )
                if features.dtype != self.requested_dtype:
                    raise ValueError(
                        "HIAI dflash_features use "
                        f"{features.dtype}, expected {self.requested_dtype}"
                    )
                result["dflash_features"] = features[:, :sequence_length, :]

            # Opaque NPU kernels may execute asynchronously.  The local cache
            # tensors must stay alive until every target kernel is complete.
            self._synchronize_call_local_state()
        except Exception:
            self._full_prefix_failures += 1
            raise
        self._full_prefix_completions += 1
        return result


def load_qwen35_target(
    target_dir: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the existing receiver wrapper and return the DFlash target bridge."""

    raw_max_len = os.environ.get(KV_CACHE_MAX_LEN_ENV)
    if raw_max_len is None:
        raise RuntimeError(
            f"{KV_CACHE_MAX_LEN_ENV} is unset; pass --kv-cache-max-len to run_npu"
        )
    kv_cache_max_len = _positive_int(raw_max_len, name=KV_CACHE_MAX_LEN_ENV)
    module = importlib.import_module("models.export_model_wrapper_qwen3_5")
    wrapper_class = getattr(module, "Qwen3_5ForCausalLMWrapper", None)
    if not isinstance(wrapper_class, type):
        raise TypeError(
            "models.export_model_wrapper_qwen3_5 must export "
            "Qwen3_5ForCausalLMWrapper"
        )
    hiai_module = importlib.import_module("models.modeling_qwen3_5_hiai_nd")
    expected_model_class = getattr(hiai_module, "Qwen3_5ForCausalLM", None)
    if not isinstance(expected_model_class, type):
        raise TypeError(
            "models.modeling_qwen3_5_hiai_nd must export Qwen3_5ForCausalLM"
        )
    wrapper_model_class = getattr(module, "Qwen3_5ForCausalLM", None)
    if (
        wrapper_model_class is not None
        and wrapper_model_class is not expected_model_class
    ):
        raise RuntimeError(
            "export_model_wrapper_qwen3_5 imports a different "
            "Qwen3_5ForCausalLM; it must use "
            "models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM"
        )
    wrapper = wrapper_class(
        model_path=target_dir,
        device=torch.device(device).type,
        dtype=dtype,
        embedding_config={"embedding_in_omc": False, "mul_twice": False},
    )
    if not isinstance(wrapper, nn.Module):
        raise TypeError("Qwen3_5ForCausalLMWrapper must inherit torch.nn.Module")
    if type(getattr(wrapper, "model", None)) is not expected_model_class:
        raise RuntimeError(
            "Qwen3_5ForCausalLMWrapper.model is not the package-local "
            "models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM"
        )
    wrapper.eval()
    target = InternalDFlashTarget(
        wrapper,
        device=torch.device(device),
        dtype=dtype,
        kv_cache_max_len=kv_cache_max_len,
    )
    return target.eval()


__all__ = [
    "BLOCK_SIZE",
    "InternalDFlashTarget",
    "KV_CACHE_MAX_LEN_ENV",
    "load_qwen35_target",
]
