"""Bridge a Qwen3.5 HIAI wrapper to DFlash full-prefix and rollback routes.

The HIAI target owns model loading and its custom operators.  This module
keeps the old complete-prefix adapter as an oracle and adds a persistent
rollback transaction for the separate two-pass chunk-GDR modeling.

The legacy full-prefix route builds a fresh hybrid cache per call:

* linear-attention layers receive ``(conv_state, recurrent_state)``;
* full-attention layers receive block-table ``(key_cache, value_cache)``;
* a multi-token prefix is right-padded to a complete 64-token prefill chunk;
* the complete padded prefix is executed as one fresh prefill and sliced back
  to the real token rows;
* only ``logits`` and optional ``dflash_features`` cross back to DFlash.

The rollback route bootstraps the prompt in block-aligned chunks of at most 64
real rows, keeps one scalar hybrid state across rounds, and executes one
``K + 1`` verification block with the original chunk GDR.  After Target
acceptance it calls the original chunk GDR again with ``effective_length=a+1``
to publish the committed recurrent state.  Full-attention KV uses provisional
physical writes plus a logical commit cursor.  Causal-conv prefix states remain
a Tensor golden on the input device.

Both routes can reuse the original ``inference.py --quant_mode enable`` W8A8
Target conversion.  The packaged copy of ``utils.quant_model`` replaces
Target ``nn.Linear`` modules with the existing ``QLinear`` primitive; the
three artifact paths come from the same YAML.  Quantized embedding rows are
dequantized at the bridge boundary for only the current token block.  Draft
embedding and LM-head modules stay FP16 and are audited independently.

The legacy chunk alignment is important for the target GDN kernel: its
multi-token path uses ``chunk_size=64``.  It is also important that the device
is synchronized before the call-local KV/GDN tensors leave scope; otherwise an
asynchronous custom operator may still be consuming storage which the caching
allocator is free to reuse for the next full-prefix call.

The legacy public factory :func:`load_qwen35_target` preserves the V1
full-prefix oracle.  :func:`load_qwen35_rollback_target` binds the separate
rollback modeling/wrapper and exposes a persistent transaction used by the
incremental CPU/CUDA/NPU scheduler.
"""

from __future__ import annotations

from collections.abc import Mapping
import importlib
import os

import torch
from torch import Tensor, nn

from .dflash_v1.target_quant import (
    ORIGINAL_QUANTIZER_SPEC,
    QUANT_MODE_DISABLED,
    OriginalQuantizedEmbedding,
    TargetQuantizationRequest,
    audit_quantized_target,
    invoke_quantizer,
    load_callback,
    load_original_quantized_embedding,
    normalize_quantizer_result,
    preconversion_linear_topology,
)


KV_CACHE_MAX_LEN_ENV = "DFLASH_HIAI_KV_CACHE_MAX_LEN"
BLOCK_SIZE = 64
FEATURE_WIDTH = 20_480
VOCAB_SIZE = 248_320
DFLASH_MAX_VERIFY_TOKENS = 16

_FEATURE_SOURCE = "package_local:modeling_qwen3_5_hiai_nd.py"
_ROLLBACK_FEATURE_SOURCE = (
    "package_local:modeling_qwen3_5_hiai_nd_dflash_rollback.py"
)
_CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
_FEATURE_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"
_ROLLBACK_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-chunk-gdr-rollback-v1"


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


def _clone_scalar_gdn_state(
    conv_state: Tensor,
    recurrent_state: Tensor,
) -> tuple[Tensor, Tensor]:
    if conv_state.ndim != 3 or recurrent_state.ndim != 4:
        raise ValueError("scalar GDN state must use conv rank 3/recurrent rank 4")
    if conv_state.shape[0] != recurrent_state.shape[0]:
        raise ValueError("conv and recurrent GDN state batches differ")
    if conv_state.device != recurrent_state.device:
        raise ValueError("conv and recurrent GDN state devices differ")
    return conv_state.clone(), recurrent_state.clone()


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
    """Full-prefix oracle plus opt-in persistent HIAI rollback facade."""

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
        rollback_enabled: bool = False,
        draft_input_embeddings: nn.Module | None = None,
        draft_output_embeddings: nn.Module | None = None,
        quantization_request: TargetQuantizationRequest | None = None,
        quantized_embedding: OriginalQuantizedEmbedding | None = None,
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
        if self.quantization_request.enabled and not isinstance(
            quantized_embedding,
            OriginalQuantizedEmbedding,
        ):
            raise TypeError(
                "quantized target requires the original INT8 embedding artifacts"
            )
        if not self.quantization_request.enabled and quantized_embedding is not None:
            raise ValueError(
                "disabled target quantization must not install quantized embedding"
            )
        self._target_quantized_embedding = quantized_embedding
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
        self.rollback_enabled = bool(rollback_enabled)
        if self.rollback_enabled:
            self.dflash_feature_source = _ROLLBACK_FEATURE_SOURCE
            self.dflash_full_prefix_execution_mode = "persistent_incremental_rollback"
            self.dflash_rollback_contract_id = _ROLLBACK_CONTRACT_ID
            self.dflash_rollback_mode = (
                "original-gdr-two-pass-chunk-torch-conv-prefix-logical-paged-kv"
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
        self._persistent_state: list[tuple[Tensor, Tensor]] | None = None
        self._persistent_cursor = 0
        self._persistent_mode: str | None = None
        self._last_committed_rows = 0
        self._pending_verify_rows: int | None = None
        self._pending_verify_output: Mapping[str, Tensor] | None = None
        self._rollback_invalid = False
        self._ordinary_prefill_token_calls = 0
        self._ordinary_prefill_lm_head_skips = 0
        self._ordinary_decode_calls = 0
        self._rollback_prefill_token_calls = 0
        self._rollback_prefill_lm_head_skips = 0
        self._rollback_verify_calls = 0
        self._rollback_commit_calls = 0
        self._rollback_gdr_verify_layer_calls = 0
        self._rollback_gdr_commit_layer_calls = 0
        self._rollback_aborts = 0
        self._target_embedding_calls = 0
        self._target_embedding_successes = 0
        self._target_embedding_failures = 0

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
            "embedding_lookup_calls": self._target_embedding_calls,
            "embedding_lookup_successes": self._target_embedding_successes,
            "embedding_lookup_failures": self._target_embedding_failures,
            "embedding_output_contract": (
                "original_int8_mul_fp32_scale_then_fp16"
                if self.quantization_request.enabled
                else "ordinary_fp16_embedding"
            ),
        }

    @property
    def dflash_full_prefix_bridge_audit(self) -> Mapping[str, object]:
        """Expose bounded runtime facts without returning mutable model state."""

        return {
            "prefill_alignment": "right_pad_s_gt_1_to_multiple_of_64",
            "gdr_effective_length_contract": "int16_batch_call_local_valid_rows",
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
    def dflash_rollback_audit(self) -> Mapping[str, object]:
        return {
            "contract_id": getattr(self, "dflash_rollback_contract_id", None),
            "mode": getattr(self, "dflash_rollback_mode", None),
            "enabled": self.rollback_enabled,
            "historical_prefix_replay_during_verify": False,
            "conv_bank_backend": "torch_tensor_golden_on_input_device",
            "gdr_backend": "npu_chunk_gated_delta_rule_two_pass",
            "ordinary_gdr_effective_length_contract": (
                "int16_batch_call_local_valid_rows"
            ),
            "verify_gdr_policy": (
                "original_chunk_gdr_effective_length_equals_verify_rows"
            ),
            "commit_gdr_policy": (
                "same_round_start_state_second_chunk_effective_length_a_plus_1"
            ),
            "custom_gdr_mtp_required": False,
            "persistent_gdn_state": (
                "scalar_conv_fp16_recurrent_original_receiver_dtype"
            ),
            "persistent_call_synchronization_policy": (
                "same_device_stream_dependencies_no_per_call_host_barrier"
            ),
            "prefill_execution_mode": (
                "block_aligned_real_token_chunks_original_gdr"
            ),
            "prefill_chunk_size": BLOCK_SIZE,
            "prefill_lm_head_policy": "last_real_prompt_row_only",
            "prefill_call_counter_semantics": (
                "legacy *_prefill_token_calls fields count target forwards; "
                "each forward now contains 1..64 real prompt rows"
            ),
            "kv_policy": "physical_provisional_writes_logical_cursor_commit",
            "persistent_mode": self._persistent_mode,
            "persistent_cursor": self._persistent_cursor,
            "last_committed_rows": self._last_committed_rows,
            "pending_verify_rows": self._pending_verify_rows,
            "session_invalid": self._rollback_invalid,
            "cumulative_counter_fields": (
                "ordinary_prefill_token_calls",
                "ordinary_prefill_lm_head_skips",
                "ordinary_decode_calls",
                "rollback_prefill_token_calls",
                "rollback_prefill_lm_head_skips",
                "rollback_verify_calls",
                "rollback_commit_calls",
                "rollback_gdr_verify_layer_calls",
                "rollback_gdr_commit_layer_calls",
                "rollback_aborts",
            ),
            "ordinary_prefill_token_calls": self._ordinary_prefill_token_calls,
            "ordinary_prefill_lm_head_skips": (
                self._ordinary_prefill_lm_head_skips
            ),
            "ordinary_decode_calls": self._ordinary_decode_calls,
            "rollback_prefill_token_calls": self._rollback_prefill_token_calls,
            "rollback_prefill_lm_head_skips": (
                self._rollback_prefill_lm_head_skips
            ),
            "rollback_verify_calls": self._rollback_verify_calls,
            "rollback_commit_calls": self._rollback_commit_calls,
            "rollback_gdr_verify_layer_calls": (
                self._rollback_gdr_verify_layer_calls
            ),
            "rollback_gdr_commit_layer_calls": (
                self._rollback_gdr_commit_layer_calls
            ),
            "rollback_aborts": self._rollback_aborts,
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
            weight = getattr(embeddings, "weight", None)
            if not isinstance(weight, Tensor):
                raise TypeError("Draft input embedding must expose Tensor weight")
            inputs_embeds = embeddings(input_ids.to(weight.device)).to(
                self.requested_device
            )
            if inputs_embeds.dtype != self.requested_dtype:
                raise ValueError(
                    f"input embeddings use {inputs_embeds.dtype}, expected "
                    f"{self.requested_dtype}"
                )
            return inputs_embeds

        embedding = self._target_quantized_embedding
        assert isinstance(embedding, OriginalQuantizedEmbedding)
        self._target_embedding_calls += 1
        try:
            inputs_embeds = embedding(input_ids)
            expected = (
                1,
                int(input_ids.shape[1]),
                _positive_int(
                    getattr(self.config, "hidden_size", None),
                    name="config.hidden_size",
                ),
            )
            if tuple(inputs_embeds.shape) != expected:
                raise ValueError(
                    f"quantized embedding output must have shape {expected}"
                )
            if (
                inputs_embeds.device != self.requested_device
                or inputs_embeds.dtype != self.requested_dtype
            ):
                raise ValueError(
                    "quantized embedding output has the wrong device or dtype"
                )
        except Exception:
            self._target_embedding_failures += 1
            raise
        self._target_embedding_successes += 1
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

    def _incremental_attention_mask(
        self,
        *,
        start_position: int,
        sequence_length: int,
    ) -> Tensor:
        positions = start_position + torch.arange(
            sequence_length,
            device=self.requested_device,
        )
        columns = torch.arange(
            self.kv_cache_max_len,
            device=self.requested_device,
        )
        visible = columns.view(1, -1) <= positions.view(-1, 1)
        zero = torch.zeros((), device=self.requested_device, dtype=torch.float32)
        negative_infinity = torch.full(
            (),
            float("-inf"),
            device=self.requested_device,
            dtype=torch.float32,
        )
        return torch.where(visible, zero, negative_infinity).unsqueeze(0).unsqueeze(0)

    def _reset_persistent_session(self, mode: str) -> None:
        if mode not in {"ordinary", "rollback"}:
            raise ValueError("persistent mode must be ordinary or rollback")
        discard = getattr(
            self.model_wrapper,
            "discard_dflash_chunk_state",
            None,
        )
        if not callable(discard):
            raise TypeError(
                "rollback Target must expose discard_dflash_chunk_state()"
            )
        discard()
        self._persistent_state = self._fresh_hybrid_cache(batch_size=1)
        self._persistent_cursor = 0
        self._persistent_mode = mode
        self._last_committed_rows = 0
        self._pending_verify_rows = None
        self._pending_verify_output = None
        self._rollback_invalid = False

    def _execute_incremental_rows(
        self,
        input_ids: Tensor,
        *,
        start_position: int,
        output_dflash_features: bool,
        dflash_chunk_verify: bool = False,
        state_override: list[tuple[Tensor, Tensor]] | None = None,
        require_logits: bool = True,
        last_token_logits_only: bool = False,
    ) -> Mapping[str, Tensor]:
        if not self.rollback_enabled:
            raise RuntimeError(
                "persistent execution requires load_qwen35_rollback_target()"
            )
        persistent_state = self._persistent_state
        if persistent_state is None:
            raise RuntimeError("persistent HIAI state has not been initialized")
        if not isinstance(dflash_chunk_verify, bool):
            raise TypeError("dflash_chunk_verify must be a bool")
        if state_override is not None and not dflash_chunk_verify:
            raise ValueError("state_override is reserved for chunk verification")
        state = persistent_state if state_override is None else state_override
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("persistent HIAI input_ids must have shape [1,T]")
        if input_ids.dtype != torch.long:
            raise TypeError("persistent HIAI input_ids must use torch.long")
        if input_ids.device != self.requested_device:
            raise ValueError(
                "persistent HIAI input_ids must stay on the requested device"
            )
        sequence_length = int(input_ids.shape[1])
        if sequence_length <= 0:
            raise ValueError("persistent HIAI execution requires at least one row")
        if not isinstance(last_token_logits_only, bool):
            raise TypeError("last_token_logits_only must be a bool")
        if last_token_logits_only and not require_logits:
            raise ValueError(
                "last_token_logits_only requires the LM head to be enabled"
            )
        logical_end = start_position + sequence_length
        if start_position < 0 or logical_end > self.kv_cache_max_len:
            raise ValueError("persistent HIAI rows exceed kv_cache_max_len")

        inputs_embeds = self._target_inputs(input_ids)
        gdr_effective_length = torch.full(
            (int(input_ids.shape[0]),),
            sequence_length,
            dtype=torch.int16,
            device=self.requested_device,
        )
        positions = torch.arange(
            start_position,
            logical_end,
            device=self.requested_device,
            dtype=torch.long,
        )
        try:
            with torch.inference_mode():
                raw_output = self.dflash_execution_model(
                    input_ids=input_ids,
                    attention_mask=self._incremental_attention_mask(
                        start_position=start_position,
                        sequence_length=sequence_length,
                    ),
                    position_ids=positions.unsqueeze(0),
                    past_key_values=state,
                    new_kv_cache_pos=positions,
                    use_cache=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    inputs_embeds=inputs_embeds,
                    embed_scale=None,
                    output_pos=None,
                    allQLen=[logical_end],
                    output_dflash_features=bool(output_dflash_features),
                    dflash_skip_lm_head=not bool(require_logits),
                    dflash_last_token_only=last_token_logits_only,
                    dflash_chunk_verify=dflash_chunk_verify,
                    gdr_effective_length=gdr_effective_length,
                )
            logits, features = _unwrap_logits_and_features(
                raw_output,
                feature_enabled=bool(output_dflash_features),
            )
            expected_logits = (
                1,
                1 if last_token_logits_only else (
                    sequence_length if require_logits else 0
                ),
                VOCAB_SIZE,
            )
            if tuple(logits.shape) != expected_logits:
                raise ValueError(
                    f"persistent HIAI logits must have shape {expected_logits}"
                )
            result: dict[str, Tensor] = {"logits": logits}
            if output_dflash_features:
                assert features is not None
                expected_features = (1, sequence_length, FEATURE_WIDTH)
                if tuple(features.shape) != expected_features:
                    raise ValueError(
                        "persistent HIAI features must have shape "
                        f"{expected_features}"
                    )
                result["dflash_features"] = features
            # Persistent state remains owned by this bridge and the next call
            # consumes it on the same device stream.  Stream dependencies keep
            # that ordering without a host-visible device barrier after every
            # prompt/decode row.  Runner and benchmark stage boundaries still
            # synchronize explicitly; the full-prefix oracle still barriers
            # before releasing its genuinely call-local state.
            return result
        except Exception:
            if dflash_chunk_verify:
                self._rollback_invalid = True
            raise

    def _prepare_rollback_state(
        self,
        verify_tokens: int,
    ) -> list[tuple[Tensor, Tensor]]:
        """Clone scalar GDN state while sharing provisional paged-KV storage."""

        if not self.rollback_enabled:
            raise RuntimeError("this HIAI bridge was not loaded in rollback mode")
        if not 1 <= verify_tokens <= DFLASH_MAX_VERIFY_TOKENS:
            raise ValueError(
                "rollback verify block must contain 1.."
                f"{DFLASH_MAX_VERIFY_TOKENS} rows"
            )
        state = self._persistent_state
        if state is None:
            raise RuntimeError("rollback state has not been initialized")
        layer_types = tuple(getattr(self.config, "layer_types", ()))
        if len(layer_types) != len(state):
            raise RuntimeError("persistent state does not cover every Target layer")
        provisional: list[tuple[Tensor, Tensor]] = []
        linear_layers = 0
        for layer_type, layer_state in zip(layer_types, state):
            if layer_type == "linear_attention":
                provisional.append(_clone_scalar_gdn_state(*layer_state))
                linear_layers += 1
            elif layer_type == "full_attention":
                # Physical KV writes are intentionally shared.  The logical
                # cursor, not a memory copy, commits only accepted rows.
                provisional.append(layer_state)
            else:
                raise ValueError(f"unsupported rollback layer type {layer_type!r}")
        if linear_layers == 0:
            raise RuntimeError("rollback target has no linear-attention layers")
        return provisional

    def begin_ordinary(self, prompt_ids: Tensor) -> Mapping[str, Tensor]:
        """Prefill with block-aligned real-token chunks on the original GDR."""

        self._reset_persistent_session("ordinary")
        last_output: Mapping[str, Tensor] | None = None
        prompt_length = int(prompt_ids.shape[1])
        for start in range(0, prompt_length, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, prompt_length)
            final_chunk = end == prompt_length
            chunk_length = end - start
            last_output = self._execute_incremental_rows(
                prompt_ids[:, start:end],
                start_position=start,
                output_dflash_features=False,
                require_logits=final_chunk,
                last_token_logits_only=final_chunk,
            )
            self._persistent_cursor += chunk_length
            self._ordinary_prefill_token_calls += 1
            self._ordinary_prefill_lm_head_skips += chunk_length - int(final_chunk)
        if last_output is None:
            raise ValueError("ordinary prefill requires a non-empty prompt")
        return last_output

    def advance_ordinary(self, input_ids: Tensor) -> Mapping[str, Tensor]:
        if self._persistent_mode != "ordinary" or self._persistent_state is None:
            raise RuntimeError("ordinary persistent session is not active")
        if tuple(input_ids.shape) != (1, 1):
            raise ValueError("ordinary persistent decode requires [1,1] input")
        output = self._execute_incremental_rows(
            input_ids,
            start_position=self._persistent_cursor,
            output_dflash_features=False,
            require_logits=True,
        )
        self._persistent_cursor += 1
        self._ordinary_decode_calls += 1
        return output

    def begin_rollback(self, prompt_ids: Tensor) -> Mapping[str, Tensor]:
        """Chunk-prefill scalar GDN state and retain real prompt features."""

        if not self.rollback_enabled:
            raise RuntimeError("rollback factory must be used for this route")
        self._reset_persistent_session("rollback")
        last_logits: Tensor | None = None
        features: list[Tensor] = []
        prompt_length = int(prompt_ids.shape[1])
        for start in range(0, prompt_length, BLOCK_SIZE):
            end = min(start + BLOCK_SIZE, prompt_length)
            final_chunk = end == prompt_length
            chunk_length = end - start
            output = self._execute_incremental_rows(
                prompt_ids[:, start:end],
                start_position=start,
                output_dflash_features=True,
                require_logits=final_chunk,
                last_token_logits_only=final_chunk,
            )
            if final_chunk:
                last_logits = output["logits"]
            features.append(output["dflash_features"])
            self._persistent_cursor += chunk_length
            self._rollback_prefill_token_calls += 1
            self._rollback_prefill_lm_head_skips += chunk_length - int(final_chunk)
        if last_logits is None:
            raise ValueError("rollback prefill requires a non-empty prompt")
        return {
            "logits": last_logits,
            "dflash_features": torch.cat(features, dim=1),
        }

    def verify_rollback(self, block_ids: Tensor) -> Mapping[str, Tensor]:
        if self._persistent_mode != "rollback" or self._persistent_state is None:
            raise RuntimeError("rollback persistent session is not active")
        if self._rollback_invalid:
            raise RuntimeError("rollback session is invalid after a failed verify")
        if self._pending_verify_rows is not None:
            raise RuntimeError("a rollback verification is already pending")
        verify_tokens = int(block_ids.shape[1])
        provisional_state = self._prepare_rollback_state(verify_tokens)
        output = self._execute_incremental_rows(
            block_ids,
            start_position=self._persistent_cursor,
            output_dflash_features=True,
            dflash_chunk_verify=True,
            state_override=provisional_state,
            require_logits=True,
        )
        self._pending_verify_rows = verify_tokens
        self._pending_verify_output = output
        self._rollback_verify_calls += 1
        self._rollback_gdr_verify_layer_calls += sum(
            layer_type == "linear_attention"
            for layer_type in tuple(getattr(self.config, "layer_types", ()))
        )
        return output

    def commit_rollback(self, accepted_draft_tokens: int) -> Mapping[str, Tensor]:
        rows = self._pending_verify_rows
        output = self._pending_verify_output
        if rows is None or output is None:
            raise RuntimeError("no rollback verification is pending")
        if isinstance(accepted_draft_tokens, bool) or not isinstance(
            accepted_draft_tokens,
            int,
        ):
            raise TypeError("accepted_draft_tokens must be an integer")
        if not 0 <= accepted_draft_tokens < rows:
            raise ValueError("accepted count is outside the pending verify block")
        committed_rows = accepted_draft_tokens + 1
        commit = getattr(
            self.model_wrapper,
            "commit_dflash_chunk_state",
            None,
        )
        if not callable(commit):
            self._rollback_invalid = True
            raise TypeError(
                "rollback Target must expose commit_dflash_chunk_state()"
            )
        try:
            with torch.inference_mode():
                committed_gdn = commit(committed_rows)
            if not isinstance(committed_gdn, Mapping):
                raise TypeError("chunk commit must return a layer-index mapping")
            state = self._persistent_state
            if state is None:
                raise RuntimeError("rollback state disappeared before commit")
            layer_types = tuple(getattr(self.config, "layer_types", ()))
            linear_indices = {
                index
                for index, layer_type in enumerate(layer_types)
                if layer_type == "linear_attention"
            }
            if set(committed_gdn) != linear_indices:
                raise RuntimeError(
                    "chunk commit did not return every linear-attention layer"
                )
            updated = list(state)
            for index in sorted(linear_indices):
                pair = committed_gdn[index]
                if not isinstance(pair, (tuple, list)) or len(pair) != 2:
                    raise TypeError("each chunk commit state must be (conv, recurrent)")
                conv_state, recurrent_state = pair
                old_conv, old_recurrent = state[index]
                if not isinstance(conv_state, Tensor) or not isinstance(
                    recurrent_state,
                    Tensor,
                ):
                    raise TypeError("chunk commit states must be Tensors")
                if tuple(conv_state.shape) != tuple(old_conv.shape):
                    raise ValueError("chunk commit returned an invalid conv shape")
                if tuple(recurrent_state.shape) != tuple(old_recurrent.shape):
                    raise ValueError(
                        "chunk commit returned an invalid recurrent shape"
                    )
                if conv_state.device != old_conv.device:
                    raise ValueError("chunk commit conv state changed device")
                if recurrent_state.device != old_recurrent.device:
                    raise ValueError("chunk commit recurrent state changed device")
                if conv_state.dtype != old_conv.dtype:
                    raise TypeError("chunk commit conv state changed dtype")
                if recurrent_state.dtype != torch.float32:
                    raise TypeError("chunk commit recurrent state must use FP32")
                # Match the receiver's ordinary GDR cache boundary exactly:
                # the operator returns FP32, then persistent storage uses the
                # dtype of the round-start state (normally FP16).
                updated[index] = (
                    conv_state,
                    recurrent_state.to(old_recurrent.dtype),
                )
        except Exception:
            self._rollback_invalid = True
            raise

        self._persistent_state = updated
        self._persistent_cursor += committed_rows
        self._last_committed_rows = committed_rows
        self._rollback_gdr_commit_layer_calls += len(linear_indices)
        committed = {
            "logits": output["logits"][:, :committed_rows, :],
            "dflash_features": output["dflash_features"][:, :committed_rows, :],
        }
        self._pending_verify_rows = None
        self._pending_verify_output = None
        self._rollback_commit_calls += 1
        return committed

    def abort_rollback(self) -> None:
        if self._pending_verify_rows is not None or self._rollback_invalid:
            self._rollback_aborts += 1
        discard = getattr(
            self.model_wrapper,
            "discard_dflash_chunk_state",
            None,
        )
        try:
            if callable(discard):
                discard()
        finally:
            # A failed asynchronous verify/commit may have partially updated
            # provisional buffers.  Fail closed instead of publishing them.
            self._persistent_state = None
            self._persistent_mode = None
            self._pending_verify_rows = None
            self._pending_verify_output = None
            self._rollback_invalid = True

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
        gdr_effective_length = torch.full(
            (int(execution_input_ids.shape[0]),),
            sequence_length,
            dtype=torch.int16,
            device=self.requested_device,
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
                    gdr_effective_length=gdr_effective_length,
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


def _load_qwen35_target_impl(
    target_dir: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
    wrapper_module_name: str,
    hiai_module_name: str,
    rollback_enabled: bool,
) -> nn.Module:
    quantization_request = TargetQuantizationRequest.from_environment()
    raw_max_len = os.environ.get(KV_CACHE_MAX_LEN_ENV)
    if raw_max_len is None:
        raise RuntimeError(
            f"{KV_CACHE_MAX_LEN_ENV} is unset; pass --kv-cache-max-len to run_npu"
        )
    kv_cache_max_len = _positive_int(raw_max_len, name=KV_CACHE_MAX_LEN_ENV)
    module = importlib.import_module(wrapper_module_name)
    wrapper_class = getattr(module, "Qwen3_5ForCausalLMWrapper", None)
    if not isinstance(wrapper_class, type):
        raise TypeError(
            f"{wrapper_module_name} must export Qwen3_5ForCausalLMWrapper"
        )
    hiai_module = importlib.import_module(hiai_module_name)
    expected_model_class = getattr(hiai_module, "Qwen3_5ForCausalLM", None)
    if not isinstance(expected_model_class, type):
        raise TypeError(
            f"{hiai_module_name} must export Qwen3_5ForCausalLM"
        )
    wrapper_model_class = getattr(module, "Qwen3_5ForCausalLM", None)
    if (
        wrapper_model_class is not None
        and wrapper_model_class is not expected_model_class
    ):
        raise RuntimeError(
            f"{wrapper_module_name} imports a different Qwen3_5ForCausalLM; "
            f"it must use {hiai_module_name}.Qwen3_5ForCausalLM"
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
            "Qwen3_5ForCausalLMWrapper.model is not the requested package-local "
            f"{hiai_module_name}.Qwen3_5ForCausalLM"
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

    quantized_embedding: OriginalQuantizedEmbedding | None = None
    quantization_audit: Mapping[str, object] = {
        "status": "DISABLED",
        "scheme": QUANT_MODE_DISABLED,
    }
    if quantization_request.enabled:
        assert quantization_request.config_path is not None
        assert quantization_request.quant_weight_path is not None
        assert quantization_request.embedding_weight_path is not None
        assert quantization_request.embedding_scale_path is not None
        quantizer, quantizer_identity = load_callback(
            ORIGINAL_QUANTIZER_SPEC,
            label="original utils.quant_model",
        )
        original_linear_topology = preconversion_linear_topology(execution_model)
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
            default_expected_qlinear_paths=tuple(original_linear_topology),
        )
        if type(quantized.execution_model) is not expected_model_class:
            raise RuntimeError(
                "target quantizer must preserve the requested package-local "
                "Qwen3_5ForCausalLM execution-model class"
            )
        converted_model = quantized.execution_model.to(torch.device(device)).eval()
        replace_model = getattr(wrapper, "replace_dflash_execution_model", None)
        if callable(replace_model):
            replace_model(converted_model)
        else:
            wrapper.model = converted_model
        if getattr(wrapper, "model", None) is not converted_model:
            raise RuntimeError(
                "target wrapper did not retain the quantized execution model"
            )
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
        quantized_embedding = load_original_quantized_embedding(
            quantization_request.embedding_weight_path,
            quantization_request.embedding_scale_path,
            vocab_size=_positive_int(
                getattr(wrapper.model.config, "vocab_size", None),
                name="config.vocab_size",
            ),
            hidden_size=_positive_int(
                getattr(wrapper.model.config, "hidden_size", None),
                name="config.hidden_size",
            ),
            device=torch.device(device),
            output_dtype=dtype,
        )
        quantization_audit = {
            **assembly,
            **dict(quantized_embedding.audit),
            "route": "rollback" if rollback_enabled else "full_prefix",
            "config_path": str(quantization_request.config_path),
            "quantizer_identity": quantizer_identity,
            "quantizer_contract": "utils.quant_model(model, quanted_pth)",
            "quant_weight_path": str(quantization_request.quant_weight_path),
            "quant_weight_path_kind": (
                "directory"
                if quantization_request.quant_weight_path.is_dir()
                else "file"
            ),
            "embedding_weight_path": str(
                quantization_request.embedding_weight_path
            ),
            "embedding_weight_path_kind": "file",
            "embedding_scale_path": str(
                quantization_request.embedding_scale_path
            ),
            "embedding_scale_path_kind": "file",
            "numerical_validation": "PENDING_REAL_NPU_ROLLBACK_PARITY",
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

    if rollback_enabled:
        for method_name in (
            "commit_dflash_chunk_state",
            "discard_dflash_chunk_state",
        ):
            if not callable(getattr(wrapper, method_name, None)):
                raise TypeError(
                    "rollback HIAI wrapper must expose " + method_name + "()"
                )

    target = InternalDFlashTarget(
        wrapper,
        device=torch.device(device),
        dtype=dtype,
        kv_cache_max_len=kv_cache_max_len,
        rollback_enabled=rollback_enabled,
        draft_input_embeddings=draft_input_embeddings,
        draft_output_embeddings=draft_output_embeddings,
        quantization_request=quantization_request,
        quantized_embedding=quantized_embedding,
        target_quantization_audit=quantization_audit,
    )
    return target.eval()


def load_qwen35_target(
    target_dir: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the unchanged receiver model for the full-prefix oracle."""

    return _load_qwen35_target_impl(
        target_dir,
        device=device,
        dtype=dtype,
        wrapper_module_name="models.export_model_wrapper_qwen3_5",
        hiai_module_name="models.modeling_qwen3_5_hiai_nd",
        rollback_enabled=False,
    )


def load_qwen35_rollback_target(
    target_dir: str,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the separate rollback modeling through the wrapper adapter."""

    return _load_qwen35_target_impl(
        target_dir,
        device=device,
        dtype=dtype,
        wrapper_module_name=(
            "models.export_model_wrapper_qwen3_5_dflash_rollback"
        ),
        hiai_module_name="models.modeling_qwen3_5_hiai_nd_dflash_rollback",
        rollback_enabled=True,
    )


__all__ = [
    "BLOCK_SIZE",
    "InternalDFlashTarget",
    "KV_CACHE_MAX_LEN_ENV",
    "load_qwen35_rollback_target",
    "load_qwen35_target",
]
