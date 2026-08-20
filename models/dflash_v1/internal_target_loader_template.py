"""Copyable loader for a receiver-owned Qwen3.5-4B target implementation.

The DFlash V1 CLI loads a target through ``MODULE:FUNCTION`` and requires a
``torch.nn.Module``.  This template keeps the receiver-specific construction
in exactly one function, :func:`create_internal_target`, and wraps the result
in a fail-closed facade matching the cache-free V1 target ABI.  ``use_cache``
alone cannot make a receiver model stateless when its installed KV/GDN
operators mutate private buffers, so every accepted forward is preceded by an
explicit receiver-owned full-prefix isolation hook.

This module is *not* an ACLNN binding.  Raw ``aclnnXxxGetWorkspaceSize`` and
``aclnnXxx`` symbols are C/C++ host APIs and need a receiver-owned extension or
framework integration.  The factory below must call that existing integration
rather than trying to invoke an ACLNN shared library from this Python file.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import importlib
import inspect
import operator
from pathlib import Path
from threading import RLock
from typing import Any
import weakref

import torch
from torch import Tensor, nn


OFFICIAL_VOCAB_SIZE = 248_320
OFFICIAL_HIDDEN_SIZE = 2_560
OFFICIAL_FEATURE_SIZE = 20_480

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}
_FORBIDDEN_STATE_KWARGS = frozenset(
    {
        "past_key_values",
        "cache_params",
        "cache_state",
        "kv_cache",
        "conv_state",
        "recurrent_state",
        "initial_state",
        "new_kv_cache_pos",
        "allQLen",
        "token_count",
        "export_flag",
    }
)

ISOLATION_MODE_ATTRIBUTE = "dflash_full_prefix_isolation_mode"
ISOLATION_HOOK_ATTRIBUTE = "prepare_dflash_full_prefix_call"
ISOLATION_EVIDENCE_ATTRIBUTE = "dflash_full_prefix_isolation_evidence"
FEATURE_SOURCE_ATTRIBUTE = "dflash_feature_source"
FEATURE_CAPTURE_POINT_ATTRIBUTE = "dflash_feature_capture_point"
FEATURE_SOURCE_SHA256_ATTRIBUTE = "dflash_feature_source_sha256"
FEATURE_CONTRACT_ID_ATTRIBUTE = "dflash_feature_contract_id"
FULL_PREFIX_EXECUTION_MODE_ATTRIBUTE = "dflash_full_prefix_execution_mode"
PREFILL_CHUNK_SIZE_ATTRIBUTE = "dflash_prefill_chunk_size"
DECODE_CHUNK_SIZE_ATTRIBUTE = "dflash_decode_chunk_size"
FACADE_CONTRACT_ID = "qwen3.5-4b-dflash-v1-full-prefix-isolation-r6"

_FORMAL_HIAI_FEATURE_SOURCE = "package_local:modeling_qwen3_5_hiai_nd.py"
_FORMAL_HIAI_CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
_FORMAL_HIAI_FEATURE_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"
_FORMAL_FULL_PREFIX_EXECUTION_MODE = "fresh_prefill"
_DECLARED_PREFILL_CHUNK_SIZE = 64
_DECLARED_DECODE_CHUNK_SIZE = 1

_FORMAL_ISOLATION_MODES = frozenset(
    {
        "receiver_reset_hook",
        "fresh_instance",
    }
)
_SIMULATION_ONLY_ISOLATION_MODE = "assumed"
_NONFORMAL_ISOLATION_MODES = frozenset(
    {_SIMULATION_ONLY_ISOLATION_MODE, "proven_stateless"}
)
_KNOWN_RECEIVER_STATE_SCOPE = (
    "kv_cache",
    "conv_state",
    "recurrent_state",
    "new_kv_cache_pos",
    "allQLen",
    "token_count",
    "export_flag",
)


def _tensor_field(output: object, name: str) -> Tensor | None:
    if isinstance(output, Mapping):
        value = output.get(name)
    else:
        value = getattr(output, name, None)
    return value if isinstance(value, Tensor) else None


def _output_field(output: object, name: str) -> object | None:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _module_weight(module: object, *, name: str) -> Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise TypeError(f"internal target {name} must expose a Tensor weight")
    return weight


def _validate_isolation_mode(mode: object, *, formal_npu: bool) -> str:
    """Normalize one receiver declaration without inferring state semantics."""

    if not isinstance(mode, str) or not mode.strip():
        raise RuntimeError(
            f"internal target must declare non-empty {ISOLATION_MODE_ATTRIBUTE!r}"
        )
    normalized = mode.strip().lower()
    allowed = _FORMAL_ISOLATION_MODES | _NONFORMAL_ISOLATION_MODES
    if normalized not in allowed:
        raise RuntimeError(
            "unsupported DFlash full-prefix isolation mode "
            f"{mode!r}; expected one of {sorted(allowed)}"
        )
    if formal_npu and normalized not in _FORMAL_ISOLATION_MODES:
        raise RuntimeError(
            "formal HIAI NPU execution permits only receiver_reset_hook or "
            "fresh_instance because this route has known in-place KV/GDN state"
        )
    return normalized


def _normalize_optional_sha256(value: object, *, attribute: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or len(value) != 64:
        raise TypeError(f"internal target {attribute!r} must be 64 hex characters")
    try:
        int(value, 16)
    except ValueError as error:
        raise TypeError(
            f"internal target {attribute!r} must be 64 hex characters"
        ) from error
    return value.lower()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _callable_identity(function: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "module": getattr(function, "__module__", None),
        "qualname": getattr(function, "__qualname__", None),
        "source_file": None,
        "source_sha256": None,
    }
    try:
        source = inspect.getsourcefile(function)
    except (OSError, TypeError):
        source = None
    if source is None:
        return identity
    path = Path(source).expanduser().resolve()
    identity["source_file"] = str(path)
    if path.is_file():
        identity["source_sha256"] = _sha256_file(path)
    return identity


def _target_type_identity(target: nn.Module) -> dict[str, object]:
    target_type = type(target)
    identity: dict[str, object] = {
        "fqcn": f"{target_type.__module__}.{target_type.__qualname__}",
        "module": target_type.__module__,
        "source_file": None,
        "source_sha256": None,
        "weight_identity": "PENDING_RECEIVER_EVIDENCE",
    }
    try:
        source = inspect.getfile(target_type)
    except (OSError, TypeError):
        source = None
    if source is not None:
        path = Path(source).expanduser().resolve()
        identity["source_file"] = str(path)
        if path.is_file():
            identity["source_sha256"] = _sha256_file(path)
    return identity


def _execution_model(target: nn.Module) -> nn.Module:
    """Resolve the HIAI module that executes target math behind a bridge."""

    execution_model = getattr(target, "dflash_execution_model", target)
    if not isinstance(execution_model, nn.Module):
        raise TypeError("dflash_execution_model must be torch.nn.Module")
    return execution_model


def _prepare_device_backend(device: str | torch.device) -> None:
    """Import ``torch_npu`` and select the card before target construction."""

    device_text = str(device)
    if device_text.split(":", 1)[0].lower() != "npu":
        return
    try:
        importlib.import_module("torch_npu")
    except ImportError as error:
        raise RuntimeError(
            "device=npu requires torch_npu to be importable before the "
            "internal target is constructed"
        ) from error

    npu = getattr(torch, "npu", None)
    if npu is None:
        raise RuntimeError("torch_npu imported but torch.npu is unavailable")
    is_available = getattr(npu, "is_available", None)
    if not callable(is_available):
        raise RuntimeError("torch.npu.is_available is unavailable")
    if not bool(is_available()):
        raise RuntimeError("torch_npu is installed but no NPU device is available")
    set_device = getattr(npu, "set_device", None)
    if not callable(set_device):
        raise RuntimeError("torch.npu.set_device is unavailable")
    set_device(device_text)


def create_internal_target(
    target_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """REPLACE ONLY THIS FUNCTION with the receiver's real target factory.

    The implementation should:

    1. construct/load the existing internal Qwen3.5-4B target on ``device``;
       for the formal HIAI source route this must be the exact package-local
       directly integrated ``modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM``
       class (the CPU/HF golden may still use the top-level conditional
       generation wrapper);
    2. keep its five installed target operators inside that model runtime;
    3. expose DFlash features either through the packaged hook bridge, with an
       explicit decoder-layer sequence/resolver, or through source-level
       collector instrumentation when Python forward hooks cannot observe the
       compiled graph;
    4. return an ``nn.Module`` whose forward accepts the arguments enforced by
       :class:`InternalTargetFacade`;
    5. declare ``dflash_full_prefix_isolation_mode`` and expose a callable
       ``prepare_dflash_full_prefix_call`` on the returned module;
    6. for formal NPU execution, declare ``dflash_feature_source`` as
       ``package_local:modeling_qwen3_5_hiai_nd.py``,
       ``dflash_feature_capture_point`` as
       ``decoder_post_layer_pre_final_norm``, and
       ``dflash_feature_source_sha256`` as the integrated source-file hash.
    7. declare ``dflash_full_prefix_execution_mode='fresh_prefill'`` plus the
       observed receiver settings ``dflash_prefill_chunk_size=64`` and
       ``dflash_decode_chunk_size=1``.  The prepare hook must select a fresh
       prefill invocation, not merely clear buffers while leaving the receiver
       state machine in decode mode.

    The isolation hook is invoked with keyword-only call metadata immediately
    before *every* target forward::

        prepare_dflash_full_prefix_call(
            input_ids=input_ids,
            sequence_length=S,
            output_dflash_features=bool,
            logits_to_keep=0_or_1,
            call_index=one_based_index,
        )

    Use mode ``receiver_reset_hook`` when the hook resets all receiver-owned
    mutable KV/GDN/external state.  The hook must return ``None``.  Use
    ``fresh_instance`` when it returns a newly constructed ``nn.Module`` for
    this call.  Use ``proven_stateless`` only for CPU/non-HIAI diagnostics,
    with a non-empty
    ``dflash_full_prefix_isolation_evidence`` string and a no-op audit hook
    returning ``None``.  That string remains a receiver declaration, not
    independent proof.  Both ``proven_stateless`` and ``assumed`` are rejected
    on formal HIAI NPU execution.  The receiver owns the concrete
    types and reset values of fields such as ``new_kv_cache_pos``, ``allQLen``,
    ``token_count``, and ``export_flag``; this portable template deliberately
    does not assign or infer them.

    Do not place direct ``ctypes``/ACLNN calls here.  Use the internal
    framework's already tested Python/C++ binding.  A schematic replacement is
    intentionally left as comments because receiver import paths and the hook
    bridge's concrete decoder location is integration-owned::

        raw = internal_framework.load_qwen35(target_dir, device=device, dtype=dtype)
        from .dflash_target_hook_bridge import HookedDFlashTarget
        wrapped = HookedDFlashTarget(
            raw,
            layer_path="model.language_model.layers",  # replace with real path
            detach=True,
            clone=True,
        )
        wrapped.dflash_full_prefix_isolation_mode = "receiver_reset_hook"
        wrapped.prepare_dflash_full_prefix_call = receiver_reset_full_prefix
        # Eager hooks are CPU/debug fallback only.  Formal NPU execution must
        # instead return the directly integrated HIAI model with the three feature
        # provenance attributes described above.
        return wrapped

    ``layer_path`` may be omitted only when the bridge's resolver recognizes
    the receiver model.  Prefer an explicit path to the ordered 32-layer
    ``nn.ModuleList``.  The hook bridge is eager-only; if the internal compiler
    bypasses Python module hooks, add the collector in the receiver's model
    source instead and return that feature-enabled module here.
    """

    raise NotImplementedError(
        "replace create_internal_target() in internal_target_loader_template.py "
        "with the receiver-owned Qwen3.5-4B factory; this template does not "
        "call raw ACLNN APIs"
    )


class InternalTargetFacade(nn.Module):
    """Validate and preserve the target interface consumed by DFlash V1."""

    def __init__(
        self,
        target: nn.Module,
        *,
        device: torch.device,
        dtype: torch.dtype,
        expected_vocab_size: int = OFFICIAL_VOCAB_SIZE,
        expected_hidden_size: int = OFFICIAL_HIDDEN_SIZE,
        expected_feature_size: int = OFFICIAL_FEATURE_SIZE,
    ) -> None:
        super().__init__()
        if not isinstance(target, nn.Module):
            raise TypeError("internal target factory must return torch.nn.Module")
        if not isinstance(dtype, torch.dtype):
            raise TypeError("dtype must be a torch.dtype")
        if not torch.empty((), dtype=dtype).is_floating_point():
            raise TypeError("internal target dtype must be floating point")
        self.target = target
        self._raw_target_identity = _target_type_identity(_execution_model(target))
        self.requested_device = torch.device(device)
        self.requested_dtype = dtype
        self.expected_vocab_size = int(expected_vocab_size)
        self.expected_hidden_size = int(expected_hidden_size)
        self.expected_feature_size = int(expected_feature_size)
        if min(
            self.expected_vocab_size,
            self.expected_hidden_size,
            self.expected_feature_size,
        ) <= 0:
            raise ValueError("expected target dimensions must be positive")
        self._formal_npu = (
            str(device).split(":", 1)[0].lower() == "npu"
            or self.requested_device.type == "npu"
        )
        self._isolation_mode = _validate_isolation_mode(
            getattr(target, ISOLATION_MODE_ATTRIBUTE, None),
            formal_npu=self._formal_npu,
        )
        isolation_hook = getattr(target, ISOLATION_HOOK_ATTRIBUTE, None)
        if not callable(isolation_hook):
            raise RuntimeError(
                "internal target must expose callable "
                f"{ISOLATION_HOOK_ATTRIBUTE!r}; use_cache=False does not reset "
                "receiver-owned KV/GDN state"
            )
        self._isolation_hook = isolation_hook
        self._isolation_hook_identity = _callable_identity(isolation_hook)
        evidence = getattr(target, ISOLATION_EVIDENCE_ATTRIBUTE, None)
        if evidence is not None and not isinstance(evidence, str):
            raise TypeError(
                f"internal target {ISOLATION_EVIDENCE_ATTRIBUTE!r} must be str"
            )
        self._isolation_evidence = evidence.strip() if evidence else None
        if (
            self._isolation_mode == "proven_stateless"
            and self._isolation_evidence is None
        ):
            raise RuntimeError(
                "proven_stateless requires non-empty "
                f"{ISOLATION_EVIDENCE_ATTRIBUTE!r}"
            )
        feature_source = getattr(target, FEATURE_SOURCE_ATTRIBUTE, None)
        capture_point = getattr(target, FEATURE_CAPTURE_POINT_ATTRIBUTE, None)
        if feature_source is not None and not isinstance(feature_source, str):
            raise TypeError(
                f"internal target {FEATURE_SOURCE_ATTRIBUTE!r} must be str"
            )
        if capture_point is not None and not isinstance(capture_point, str):
            raise TypeError(
                f"internal target {FEATURE_CAPTURE_POINT_ATTRIBUTE!r} must be str"
            )
        self._feature_source = feature_source
        self._feature_capture_point = capture_point
        feature_contract_id = getattr(target, FEATURE_CONTRACT_ID_ATTRIBUTE, None)
        if feature_contract_id is not None and not isinstance(feature_contract_id, str):
            raise TypeError(
                f"internal target {FEATURE_CONTRACT_ID_ATTRIBUTE!r} must be str"
            )
        self._feature_contract_id = feature_contract_id
        self._feature_source_sha256 = _normalize_optional_sha256(
            getattr(target, FEATURE_SOURCE_SHA256_ATTRIBUTE, None),
            attribute=FEATURE_SOURCE_SHA256_ATTRIBUTE,
        )
        self._validate_feature_provenance(formal_npu=self._formal_npu)
        execution_mode = getattr(
            target,
            FULL_PREFIX_EXECUTION_MODE_ATTRIBUTE,
            None,
        )
        prefill_chunk_size = getattr(target, PREFILL_CHUNK_SIZE_ATTRIBUTE, None)
        decode_chunk_size = getattr(target, DECODE_CHUNK_SIZE_ATTRIBUTE, None)
        if execution_mode is not None and not isinstance(execution_mode, str):
            raise TypeError(
                f"internal target {FULL_PREFIX_EXECUTION_MODE_ATTRIBUTE!r} must be str"
            )
        for name, value in (
            (PREFILL_CHUNK_SIZE_ATTRIBUTE, prefill_chunk_size),
            (DECODE_CHUNK_SIZE_ATTRIBUTE, decode_chunk_size),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
            ):
                raise TypeError(f"internal target {name!r} must be a positive int")
        self._full_prefix_execution_mode = execution_mode
        self._prefill_chunk_size = prefill_chunk_size
        self._decode_chunk_size = decode_chunk_size
        if self._formal_npu:
            if self._full_prefix_execution_mode != _FORMAL_FULL_PREFIX_EXECUTION_MODE:
                raise RuntimeError(
                    "formal NPU target must declare fresh_prefill full-prefix execution"
                )
            if self._prefill_chunk_size != _DECLARED_PREFILL_CHUNK_SIZE:
                raise RuntimeError("formal NPU target must declare prefill chunk_size=64")
            if self._decode_chunk_size != _DECLARED_DECODE_CHUNK_SIZE:
                raise RuntimeError("formal NPU target must declare decode chunk_size=1")
        self._isolation_hook_calls = 0
        self._isolation_hook_successes = 0
        self._isolation_hook_failures = 0
        self._target_forward_calls = 0
        self._target_forward_completions = 0
        self._target_forward_failures = 0
        self._output_validation_failures = 0
        self._last_sequence_length: int | None = None
        self._last_output_dflash_features: bool | None = None
        # Weak references reject reuse of the same live secondary module
        # without retaining one full target per call or misclassifying a later
        # Python object that legitimately reuses a released numeric id.
        self._used_fresh_targets: weakref.WeakSet[nn.Module] = weakref.WeakSet()
        # Isolation and the corresponding target forward are one critical
        # section.  Otherwise two callers could reset the same in-place
        # receiver state and then execute against each other's state.
        self._full_prefix_call_lock = RLock()
        self._validate_static_contract()

    @property
    def config(self) -> object | None:
        return getattr(self.target, "config", None)

    @property
    def dflash_full_prefix_isolation_mode(self) -> str:
        """Return the normalized receiver isolation mode."""

        return self._isolation_mode

    @property
    def dflash_full_prefix_facade_contract_id(self) -> str:
        return FACADE_CONTRACT_ID

    @property
    def dflash_full_prefix_isolation_audit(self) -> dict[str, object]:
        """Return a copy of counters suitable for run evidence/reports."""

        with self._full_prefix_call_lock:
            all_calls_prepared = (
                self._isolation_hook_failures == 0
                and self._isolation_hook_calls == self._isolation_hook_successes
                and self._isolation_hook_successes == self._target_forward_calls
            )
            return {
                "facade_contract_id": FACADE_CONTRACT_ID,
                "mode": self._isolation_mode,
                "formal_npu": self._formal_npu,
                "evidence": self._isolation_evidence,
                "evidence_authority": "receiver_declared",
                "isolation_hook_identity": dict(self._isolation_hook_identity),
                "receiver_owned_state_scope": _KNOWN_RECEIVER_STATE_SCOPE,
                "feature_contract_id": self._feature_contract_id,
                "raw_target_identity": dict(self._raw_target_identity),
                "full_prefix_execution_mode": self._full_prefix_execution_mode,
                "declared_chunk_modes": {
                    "prefill_chunk_size": self._prefill_chunk_size,
                    "decode_chunk_size": self._decode_chunk_size,
                },
                "actual_chunk_mode_trace": "PENDING_RECEIVER_TRACE",
                "prepare_forward_serialized": True,
                "prepare_calls": self._isolation_hook_calls,
                "prepare_successes": self._isolation_hook_successes,
                "prepare_failures": self._isolation_hook_failures,
                "all_calls_prepared": all_calls_prepared,
                "isolation_hook_calls": self._isolation_hook_calls,
                "isolation_hook_successes": self._isolation_hook_successes,
                "isolation_hook_failures": self._isolation_hook_failures,
                "target_forward_calls": self._target_forward_calls,
                "target_forward_completions": self._target_forward_completions,
                "target_forward_failures": self._target_forward_failures,
                "output_validation_failures": self._output_validation_failures,
                "last_sequence_length": self._last_sequence_length,
                "last_output_dflash_features": self._last_output_dflash_features,
            }

    @property
    def dflash_feature_source(self) -> str | None:
        return self._feature_source

    @property
    def dflash_feature_capture_point(self) -> str | None:
        return self._feature_capture_point

    @property
    def dflash_feature_source_sha256(self) -> str | None:
        return self._feature_source_sha256

    @property
    def dflash_feature_contract_id(self) -> str | None:
        return self._feature_contract_id

    def _validate_feature_provenance(self, *, formal_npu: bool) -> None:
        if not formal_npu:
            return
        if self._feature_source != _FORMAL_HIAI_FEATURE_SOURCE:
            raise RuntimeError(
                "formal NPU target must use the directly integrated "
                "modeling_qwen3_5_hiai_nd.py feature route"
            )
        if self._feature_capture_point != _FORMAL_HIAI_CAPTURE_POINT:
            raise RuntimeError(
                "formal NPU target has the wrong DFlash feature capture point"
            )
        if self._feature_contract_id != _FORMAL_HIAI_FEATURE_CONTRACT_ID:
            raise RuntimeError(
                "formal NPU target has the wrong DFlash feature contract id"
            )
        if self._feature_source_sha256 is None:
            raise RuntimeError(
                "formal NPU target must declare dflash_feature_source_sha256"
            )

    def _validate_feature_provenance_for(self, target: nn.Module) -> None:
        """Require a fresh target to carry the same direct-source identity."""

        if not self._formal_npu:
            return
        source = getattr(target, FEATURE_SOURCE_ATTRIBUTE, None)
        point = getattr(target, FEATURE_CAPTURE_POINT_ATTRIBUTE, None)
        contract_id = getattr(target, FEATURE_CONTRACT_ID_ATTRIBUTE, None)
        digest = _normalize_optional_sha256(
            getattr(target, FEATURE_SOURCE_SHA256_ATTRIBUTE, None),
            attribute=FEATURE_SOURCE_SHA256_ATTRIBUTE,
        )
        if (
            source != self._feature_source
            or point != self._feature_capture_point
            or contract_id != self._feature_contract_id
            or digest != self._feature_source_sha256
        ):
            raise RuntimeError(
                "fresh internal target differs from the controller's HIAI "
                "direct feature-source identity"
            )

    def get_input_embeddings(self) -> nn.Module:
        getter = getattr(self.target, "get_input_embeddings", None)
        if not callable(getter):
            raise TypeError("internal target must provide get_input_embeddings()")
        module = getter()
        if not isinstance(module, nn.Module):
            raise TypeError(
                "internal target get_input_embeddings() must return nn.Module"
            )
        return module

    def get_output_embeddings(self) -> nn.Module:
        getter = getattr(self.target, "get_output_embeddings", None)
        if not callable(getter):
            raise TypeError("internal target must provide get_output_embeddings()")
        module = getter()
        if not isinstance(module, nn.Module):
            raise TypeError(
                "internal target get_output_embeddings() must return nn.Module"
            )
        return module

    def _validate_static_contract_for(
        self,
        target: nn.Module,
        *,
        label: str,
    ) -> None:
        input_getter = getattr(target, "get_input_embeddings", None)
        output_getter = getattr(target, "get_output_embeddings", None)
        if not callable(input_getter) or not callable(output_getter):
            raise TypeError(
                f"{label} must provide get_input_embeddings() and "
                "get_output_embeddings()"
            )
        input_module = input_getter()
        output_module = output_getter()
        if not isinstance(input_module, nn.Module):
            raise TypeError(f"{label} input embedding must be nn.Module")
        if not isinstance(output_module, nn.Module):
            raise TypeError(f"{label} output embedding must be nn.Module")
        input_weight = _module_weight(input_module, name="input embedding")
        output_weight = _module_weight(output_module, name="output embedding")
        expected = (self.expected_vocab_size, self.expected_hidden_size)
        if tuple(input_weight.shape) != expected:
            raise ValueError(
                f"{label} input embedding shape mismatch: "
                f"expected {expected}, got {tuple(input_weight.shape)}"
            )
        if tuple(output_weight.shape) != expected:
            raise ValueError(
                f"{label} output embedding shape mismatch: "
                f"expected {expected}, got {tuple(output_weight.shape)}"
            )
        for name, weight in (
            ("input embedding", input_weight),
            ("output embedding", output_weight),
        ):
            if not torch.is_floating_point(weight):
                raise TypeError(f"{label} {name} weight must be floating point")
            if weight.device != self.requested_device:
                raise ValueError(
                    f"{label} {name} is on {weight.device}, expected "
                    f"{self.requested_device}; the factory must place weights"
                )
            if weight.dtype != self.requested_dtype:
                raise ValueError(
                    f"{label} {name} uses {weight.dtype}, expected "
                    f"{self.requested_dtype}; the factory must apply dtype"
                )

    def _validate_static_contract(self) -> None:
        self._validate_static_contract_for(self.target, label="internal target")

    def _prepare_isolated_target(
        self,
        *,
        input_ids: Tensor,
        output_dflash_features: bool,
        logits_to_keep: int,
    ) -> nn.Module:
        """Run the mandatory receiver hook and select this call's target."""

        call_index = self._isolation_hook_calls + 1
        sequence_length = int(input_ids.shape[1])
        self._isolation_hook_calls += 1
        self._last_sequence_length = sequence_length
        self._last_output_dflash_features = output_dflash_features
        input_ids_before_hook = input_ids.detach().clone()
        try:
            prepared = self._isolation_hook(
                input_ids=input_ids,
                sequence_length=sequence_length,
                output_dflash_features=output_dflash_features,
                logits_to_keep=logits_to_keep,
                call_index=call_index,
            )
            if not torch.equal(input_ids, input_ids_before_hook):
                raise RuntimeError(
                    "receiver isolation hook modified live input_ids in place"
                )
            if self._isolation_mode == "fresh_instance":
                if not isinstance(prepared, nn.Module):
                    raise TypeError(
                        "fresh_instance isolation hook must return torch.nn.Module"
                    )
                if prepared is self.target:
                    raise RuntimeError(
                        "fresh_instance isolation hook returned the controller "
                        "target instead of a fresh module"
                    )
                if prepared in self._used_fresh_targets:
                    raise RuntimeError(
                        "fresh_instance isolation hook reused a previously "
                        "executed target module"
                    )
                prepared.eval()
                self._validate_static_contract_for(
                    prepared,
                    label="fresh internal target",
                )
                self._validate_feature_provenance_for(prepared)
                prepared_identity = _target_type_identity(_execution_model(prepared))
                if (
                    prepared_identity.get("fqcn")
                    != self._raw_target_identity.get("fqcn")
                    or prepared_identity.get("source_sha256")
                    != self._raw_target_identity.get("source_sha256")
                ):
                    raise RuntimeError(
                        "fresh internal target type/artifact identity differs "
                        "from the controller target"
                    )
                self._used_fresh_targets.add(prepared)
                call_target = prepared
            else:
                if prepared is not None:
                    raise TypeError(
                        f"{self._isolation_mode} isolation hook must return None"
                    )
                call_target = self.target
        except Exception as error:
            self._isolation_hook_failures += 1
            raise RuntimeError(
                "DFlash full-prefix isolation hook failed closed before target "
                f"forward (mode={self._isolation_mode}, call_index={call_index}); "
                f"{type(error).__name__}: {error}"
            ) from error
        self._isolation_hook_successes += 1
        return call_target

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        if not isinstance(input_ids, Tensor):
            raise TypeError("internal target input_ids must be a Tensor")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("DFlash V1 target input_ids must have shape [1, S], S > 0")
        if input_ids.dtype not in _INTEGER_DTYPES:
            raise TypeError("DFlash V1 target input_ids must use an integer dtype")
        if input_ids.device != self.requested_device:
            raise ValueError(
                f"target input_ids are on {input_ids.device}, expected "
                f"{self.requested_device}"
            )

    def _validate_output(
        self,
        output: object,
        *,
        sequence_length: int,
        output_dflash_features: bool,
        logits_to_keep: int,
    ) -> None:
        returned_state = [
            name
            for name in _FORBIDDEN_STATE_KWARGS
            if _output_field(output, name) is not None
        ]
        if returned_state:
            raise RuntimeError(
                "internal target returned receiver-owned cache/state across the "
                "portable boundary: " + ", ".join(sorted(returned_state))
            )

        logits = _tensor_field(output, "logits")
        if logits is None:
            raise TypeError("internal target output must expose Tensor logits")
        allowed_rows = {sequence_length} if logits_to_keep == 0 else {1, sequence_length}
        if (
            logits.ndim != 3
            or logits.shape[0] != 1
            or logits.shape[1] not in allowed_rows
            or logits.shape[2] != self.expected_vocab_size
        ):
            rows = sorted(allowed_rows)
            raise ValueError(
                "internal target logits shape mismatch: expected batch=1, "
                f"rows in {rows}, vocab={self.expected_vocab_size}; got "
                f"{tuple(logits.shape)}"
            )
        if not torch.is_floating_point(logits):
            raise TypeError("internal target logits must be floating point")
        if logits.device != self.requested_device:
            raise ValueError(
                f"internal target logits are on {logits.device}, expected "
                f"{self.requested_device}"
            )

        features = _tensor_field(output, "dflash_features")
        if not output_dflash_features:
            if features is not None:
                raise ValueError(
                    "internal target returned dflash_features while feature "
                    "capture was disabled"
                )
            return
        expected_features = (1, sequence_length, self.expected_feature_size)
        if features is None:
            raise TypeError(
                "feature-enabled internal target output must expose Tensor "
                "dflash_features; attach the hook bridge or source collector"
            )
        if tuple(features.shape) != expected_features:
            raise ValueError(
                "internal target dflash_features shape mismatch: "
                f"expected {expected_features}, got {tuple(features.shape)}"
            )
        if not torch.is_floating_point(features):
            raise TypeError("internal target dflash_features must be floating point")
        if features.device != self.requested_device:
            raise ValueError(
                f"internal target dflash_features are on {features.device}, "
                f"expected {self.requested_device}"
            )
        if features.dtype != self.requested_dtype:
            raise ValueError(
                f"internal target dflash_features use {features.dtype}, "
                f"expected {self.requested_dtype}"
            )

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
    ) -> object:
        """Forward one prefix with no cache/recurrent state crossing this ABI."""

        self._validate_input_ids(input_ids)
        if use_cache is not False:
            raise ValueError("DFlash V1 internal target requires use_cache=False")
        if return_dict is not True:
            raise ValueError("DFlash V1 internal target requires return_dict=True")
        if output_hidden_states is not False:
            raise ValueError(
                "DFlash V1 captures only selected features; "
                "output_hidden_states must remain False"
            )
        if isinstance(logits_to_keep, bool):
            raise TypeError("logits_to_keep must be integer 0 or 1, not bool")
        try:
            keep = int(operator.index(logits_to_keep))
        except TypeError as error:
            raise TypeError("logits_to_keep must be integer 0 or 1") from error
        if keep not in (0, 1):
            raise ValueError("DFlash V1 internal target supports logits_to_keep=0 or 1")
        forbidden = sorted(
            key
            for key in _FORBIDDEN_STATE_KWARGS.intersection(kwargs)
            if kwargs[key] is not None
        )
        if forbidden:
            raise ValueError(
                "DFlash V1 forbids caller-provided cache/recurrent state: "
                + ", ".join(forbidden)
            )

        capture_features = bool(output_dflash_features)
        with self._full_prefix_call_lock:
            call_target = self._prepare_isolated_target(
                input_ids=input_ids,
                output_dflash_features=capture_features,
                logits_to_keep=keep,
            )
            self._target_forward_calls += 1
            try:
                with torch.inference_mode():
                    output = call_target(
                        input_ids=input_ids,
                        use_cache=False,
                        return_dict=True,
                        output_hidden_states=False,
                        output_dflash_features=capture_features,
                        logits_to_keep=keep,
                        **kwargs,
                    )
            except Exception:
                self._target_forward_failures += 1
                raise
            self._target_forward_completions += 1
            try:
                self._validate_output(
                    output,
                    sequence_length=int(input_ids.shape[1]),
                    output_dflash_features=capture_features,
                    logits_to_keep=keep,
                )
            except Exception:
                self._output_validation_failures += 1
                raise
        return output


def load_target(
    target_dir: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """CLI loader: prepare device, call one receiver factory, validate facade."""

    # This must precede torch.device("npu:...") and every receiver model load.
    _prepare_device_backend(device)
    requested_device = torch.device(device)
    if not isinstance(dtype, torch.dtype):
        raise TypeError("target loader dtype must be a torch.dtype")
    if not torch.empty((), dtype=dtype).is_floating_point():
        raise TypeError("target loader dtype must be floating point")

    root = Path(target_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"internal target directory does not exist: {root}")

    target = create_internal_target(str(root), requested_device, dtype)
    if not isinstance(target, nn.Module):
        raise TypeError("create_internal_target() must return torch.nn.Module")
    target.eval()
    facade = InternalTargetFacade(
        target,
        device=requested_device,
        dtype=dtype,
        expected_vocab_size=OFFICIAL_VOCAB_SIZE,
        expected_hidden_size=OFFICIAL_HIDDEN_SIZE,
        expected_feature_size=OFFICIAL_FEATURE_SIZE,
    )
    return facade.eval()
