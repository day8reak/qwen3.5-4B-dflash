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

from .dflash_v1.target_quant import (
    QUANT_MODE_DISABLED,
    TargetQuantizationRequest,
    audit_quantized_target,
    invoke_input_provider,
    invoke_quantizer,
    load_callback,
    normalize_quantizer_result,
    preconversion_linear_topology,
    validate_input_provider_output,
)


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
        draft_input_embeddings: nn.Module | None = None,
        draft_output_embeddings: nn.Module | None = None,
        quantization_request: TargetQuantizationRequest | None = None,
        target_input_provider: Any = None,
        target_input_provider_identity: Mapping[str, object] | None = None,
        target_quantization_audit: Mapping[str, object] | None = None,
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
        self.quantization_request = (
            quantization_request or TargetQuantizationRequest.from_environment()
        )
        if self.quantization_request.enabled and not callable(target_input_provider):
            raise TypeError(
                "quantized target requires a callable target input provider"
            )
        if not self.quantization_request.enabled and target_input_provider is not None:
            raise ValueError(
                "disabled target quantization must not install an input provider"
            )
        self._target_input_provider = target_input_provider
        self._target_input_provider_identity = dict(
            target_input_provider_identity or {}
        )
        self._target_quantization_static_audit = dict(
            target_quantization_audit
            or {
                "status": "DISABLED",
                "scheme": QUANT_MODE_DISABLED,
            }
        )
        self._draft_input_embeddings = (
            draft_input_embeddings
            if draft_input_embeddings is not None
            else self._execution_input_embeddings()
        )
        self._draft_output_embeddings = (
            draft_output_embeddings
            if draft_output_embeddings is not None
            else self._execution_output_embeddings()
        )
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
        self._target_input_provider_calls = 0
        self._target_input_provider_successes = 0
        self._target_input_provider_failures = 0

    def _execution_input_embeddings(self) -> nn.Module:
        getter = getattr(self.dflash_execution_model, "get_input_embeddings", None)
        module = getter() if callable(getter) else None
        if not isinstance(module, nn.Module):
            raise TypeError("HIAI execution model lacks input embeddings")
        return module

    def _execution_output_embeddings(self) -> nn.Module:
        getter = getattr(self.dflash_execution_model, "get_output_embeddings", None)
        module = getter() if callable(getter) else None
        if module is None:
            module = getattr(self.dflash_execution_model, "lm_head", None)
        if not isinstance(module, nn.Module):
            raise TypeError("HIAI execution model lacks its LM head")
        return module

    @property
    def dflash_target_quantization_audit(self) -> Mapping[str, object]:
        return {
            **self._target_quantization_static_audit,
            "input_provider_identity": dict(self._target_input_provider_identity),
            "input_provider_calls": self._target_input_provider_calls,
            "input_provider_successes": self._target_input_provider_successes,
            "input_provider_failures": self._target_input_provider_failures,
            "input_provider_output_contract": (
                "final_fp16_layer0_hidden"
                if self.quantization_request.enabled
                else "ordinary_fp16_embedding"
            ),
        }

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
            "target_quantization": dict(self.dflash_target_quantization_audit),
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
        return self._draft_input_embeddings

    def get_output_embeddings(self) -> nn.Module:
        return self._draft_output_embeddings

    def _target_inputs(self, input_ids: Tensor) -> Tensor:
        if not self.quantization_request.enabled:
            embeddings = self.get_input_embeddings()
            inputs_embeds = embeddings(
                input_ids.to(embeddings.weight.device)
            ).to(self.requested_device)
            if inputs_embeds.dtype != self.requested_dtype:
                raise ValueError(
                    f"input embeddings use {inputs_embeds.dtype}, expected "
                    f"{self.requested_dtype}"
                )
            return inputs_embeds

        provider = self._target_input_provider
        assert callable(provider)
        embedding_weight_path = self.quantization_request.embedding_weight_path
        embedding_scale_path = self.quantization_request.embedding_scale_path
        assert embedding_weight_path is not None
        assert embedding_scale_path is not None
        self._target_input_provider_calls += 1
        try:
            value = invoke_input_provider(
                provider,
                self.model_wrapper,
                input_ids,
                embedding_weight_path,
                embedding_scale_path,
                device=self.requested_device,
                output_dtype=self.requested_dtype,
            )
            inputs_embeds = validate_input_provider_output(
                value,
                sequence_length=int(input_ids.shape[1]),
                hidden_size=_positive_int(
                    getattr(self.config, "hidden_size", None),
                    name="config.hidden_size",
                ),
                device=self.requested_device,
                dtype=self.requested_dtype,
            )
        except Exception:
            self._target_input_provider_failures += 1
            raise
        self._target_input_provider_successes += 1
        return inputs_embeds

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

        inputs_embeds = self._target_inputs(execution_input_ids)
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

    quantization_request = TargetQuantizationRequest.from_environment()
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
    execution_model = wrapper.model
    draft_input_getter = getattr(execution_model, "get_input_embeddings", None)
    draft_input_embeddings = (
        draft_input_getter() if callable(draft_input_getter) else None
    )
    draft_output_getter = getattr(execution_model, "get_output_embeddings", None)
    draft_output_embeddings = (
        draft_output_getter() if callable(draft_output_getter) else None
    )
    if draft_output_embeddings is None:
        draft_output_embeddings = getattr(execution_model, "lm_head", None)
    if not isinstance(draft_input_embeddings, nn.Module) or not isinstance(
        draft_output_embeddings,
        nn.Module,
    ):
        raise TypeError(
            "unquantized target must expose FP16 embedding and LM-head modules "
            "for the DFlash Draft"
        )

    input_provider = None
    input_provider_identity: Mapping[str, object] | None = None
    quantization_audit: Mapping[str, object] = {
        "status": "DISABLED",
        "scheme": QUANT_MODE_DISABLED,
    }
    if quantization_request.enabled:
        assert quantization_request.quantizer_spec is not None
        assert quantization_request.input_provider_spec is not None
        assert quantization_request.quant_weight_path is not None
        assert quantization_request.embedding_weight_path is not None
        assert quantization_request.embedding_scale_path is not None
        quantizer, quantizer_identity = load_callback(
            quantization_request.quantizer_spec,
            label="target quantizer",
        )
        input_provider, input_provider_identity = load_callback(
            quantization_request.input_provider_spec,
            label="target input provider",
        )
        original_linear_topology = preconversion_linear_topology(
            execution_model,
        )
        default_expected_qlinear_paths = tuple(original_linear_topology)
        raw_result = invoke_quantizer(
            quantizer,
            execution_model,
            quantization_request.quant_weight_path,
            device=torch.device(device),
            output_dtype=dtype,
        )
        quantized = normalize_quantizer_result(
            raw_result,
            original_execution_model=execution_model,
            default_expected_qlinear_paths=default_expected_qlinear_paths,
        )
        if type(quantized.execution_model) is not expected_model_class:
            raise RuntimeError(
                "target quantizer must preserve the package-local "
                "Qwen3_5ForCausalLM execution-model class"
            )
        # The existing converter may create new QLinear buffers after the
        # wrapper was initially placed on NPU.  Move the converted module once
        # here so QLinear.forward does not copy W_q/scale on every call.
        wrapper.model = quantized.execution_model.to(torch.device(device)).eval()
        if quantized.draft_input_embeddings is not None:
            draft_input_embeddings = quantized.draft_input_embeddings
        if quantized.draft_output_embeddings is not None:
            draft_output_embeddings = quantized.draft_output_embeddings
        qlinear_type = getattr(hiai_module, "QLinear", None)
        if not isinstance(qlinear_type, type):
            raise TypeError("HIAI target source must export QLinear")
        assembly = audit_quantized_target(
            quantized,
            qlinear_type=qlinear_type,
            original_linear_topology=original_linear_topology,
            draft_input_embeddings=draft_input_embeddings,
            draft_output_embeddings=draft_output_embeddings,
            device=torch.device(device),
            dtype=dtype,
            vocab_size=_positive_int(
                getattr(wrapper.model.config, "vocab_size", None),
                name="config.vocab_size",
            ),
            hidden_size=_positive_int(
                getattr(wrapper.model.config, "hidden_size", None),
                name="config.hidden_size",
            ),
        )
        quantization_audit = {
            **assembly,
            "quantizer_identity": quantizer_identity,
            "quant_weight_path": str(quantization_request.quant_weight_path),
            "quant_weight_path_kind": (
                "directory"
                if quantization_request.quant_weight_path.is_dir()
                else "file"
            ),
            "embedding_weight_path": str(
                quantization_request.embedding_weight_path
            ),
            "embedding_weight_path_kind": (
                "directory"
                if quantization_request.embedding_weight_path.is_dir()
                else "file"
            ),
            "embedding_scale_path": str(
                quantization_request.embedding_scale_path
            ),
            "embedding_scale_path_kind": (
                "directory"
                if quantization_request.embedding_scale_path.is_dir()
                else "file"
            ),
            "numerical_validation": "PENDING_REAL_NPU_PARITY",
        }
    else:
        qlinear_type = getattr(hiai_module, "QLinear", None)
        if isinstance(qlinear_type, type) and any(
            isinstance(item, qlinear_type) for item in execution_model.modules()
        ):
            raise RuntimeError(
                "target quantization is disabled but the loaded target already "
                "contains QLinear modules"
            )

    target = InternalDFlashTarget(
        wrapper,
        device=torch.device(device),
        dtype=dtype,
        kv_cache_max_len=kv_cache_max_len,
        draft_input_embeddings=draft_input_embeddings,
        draft_output_embeddings=draft_output_embeddings,
        quantization_request=quantization_request,
        target_input_provider=input_provider,
        target_input_provider_identity=input_provider_identity,
        target_quantization_audit=quantization_audit,
    )
    return target.eval()


__all__ = [
    "BLOCK_SIZE",
    "InternalDFlashTarget",
    "KV_CACHE_MAX_LEN_ENV",
    "load_qwen35_target",
]
