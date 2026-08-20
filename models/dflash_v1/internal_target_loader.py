"""Embedded loader for an existing internal HIAI Qwen3.5 target.

The internal server keeps ``modeling_qwen3_5_hiai_nd.py`` in the parent
``models`` package and installs this DFlash package as ``models.dflash_v1``.
The default receiver bridge reuses ``Qwen3_5ForCausalLMWrapper`` and allocates
fresh external hybrid state for every complete-prefix target call.  The NPU
runner supplies its reviewed factory through one callable:

``DFLASH_HIAI_TARGET_FACTORY``
    ``MODULE:FUNCTION`` returning either the packaged ``InternalDFlashTarget``
    bridge or another reviewed target facade.

``DFLASH_HIAI_RESET_HOOK``
    Optional advanced ``MODULE:FUNCTION`` used when a custom target does not
    expose ``prepare_dflash_full_prefix_call``.  It resets the receiver-owned
    KV/GDN/request state before every complete-prefix target call.

The public :func:`load_target` ABI is consumed by the DFlash adapter.  Custom
operators remain owned by the original HIAI model; this module never replaces
or calls raw ACLNN symbols.
"""

from __future__ import annotations

import hashlib
import importlib
import os
from pathlib import Path
from typing import Any, Callable

import torch
from torch import nn

from .internal_target_loader_template import (
    FEATURE_CAPTURE_POINT_ATTRIBUTE,
    FEATURE_CONTRACT_ID_ATTRIBUTE,
    FEATURE_SOURCE_SHA256_ATTRIBUTE,
    FEATURE_SOURCE_ATTRIBUTE,
    FULL_PREFIX_EXECUTION_MODE_ATTRIBUTE,
    InternalTargetFacade,
    ISOLATION_EVIDENCE_ATTRIBUTE,
    ISOLATION_HOOK_ATTRIBUTE,
    ISOLATION_MODE_ATTRIBUTE,
    PREFILL_CHUNK_SIZE_ATTRIBUTE,
    DECODE_CHUNK_SIZE_ATTRIBUTE,
    OFFICIAL_FEATURE_SIZE,
    OFFICIAL_HIDDEN_SIZE,
    OFFICIAL_VOCAB_SIZE,
    _prepare_device_backend,
)


TARGET_FACTORY_ENV = "DFLASH_HIAI_TARGET_FACTORY"
RESET_HOOK_ENV = "DFLASH_HIAI_RESET_HOOK"
RESET_EVIDENCE_ENV = "DFLASH_HIAI_RESET_EVIDENCE"
PREFILL_CHUNK_SIZE_ENV = "DFLASH_HIAI_PREFILL_CHUNK_SIZE"
DECODE_CHUNK_SIZE_ENV = "DFLASH_HIAI_DECODE_CHUNK_SIZE"

_FEATURE_SOURCE = "receiver_owned:modeling_qwen3_5_hiai_nd.py"
_CAPTURE_POINT = "decoder_post_layer_pre_final_norm"
_FEATURE_CONTRACT_ID = "qwen3.5-4b-dflash-hiai-feature-source-v1"


def _load_callable(specification: str, *, label: str) -> Callable[..., Any]:
    if not isinstance(specification, str) or ":" not in specification:
        raise ValueError(f"{label} must use MODULE:FUNCTION syntax")
    module_name, function_name = specification.rsplit(":", 1)
    function = getattr(importlib.import_module(module_name), function_name, None)
    if not callable(function):
        raise TypeError(f"{label} is not callable: {specification}")
    return function


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _prepare_target_contract(target: nn.Module) -> nn.Module:
    """Attach only portable metadata and an explicitly supplied reset hook."""

    source = Path(__file__).resolve().parent.parent / "modeling_qwen3_5_hiai_nd.py"
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(
            "embedded NPU layout requires models/modeling_qwen3_5_hiai_nd.py"
        )

    declared = {
        FEATURE_SOURCE_ATTRIBUTE: _FEATURE_SOURCE,
        FEATURE_CAPTURE_POINT_ATTRIBUTE: _CAPTURE_POINT,
        FEATURE_CONTRACT_ID_ATTRIBUTE: _FEATURE_CONTRACT_ID,
    }
    for name, expected in declared.items():
        actual = getattr(target, name, None)
        if actual != expected:
            raise RuntimeError(
                f"direct HIAI target must declare {name}={expected!r}; got {actual!r}"
            )
    setattr(target, FEATURE_SOURCE_SHA256_ATTRIBUTE, _sha256_file(source))

    existing_hook = getattr(target, ISOLATION_HOOK_ATTRIBUTE, None)
    existing_mode = getattr(target, ISOLATION_MODE_ATTRIBUTE, None)
    reset_specification = os.environ.get(RESET_HOOK_ENV)
    if callable(existing_hook):
        if existing_mode not in {"receiver_reset_hook", "fresh_instance"}:
            raise RuntimeError(
                "an existing prepare_dflash_full_prefix_call requires "
                "dflash_full_prefix_isolation_mode receiver_reset_hook or fresh_instance"
            )
    else:
        if not reset_specification:
            raise RuntimeError(
                "the raw HIAI target has no prepare_dflash_full_prefix_call; "
                "pass --reset-hook MODULE:FUNCTION to the NPU runner"
            )
        reset_hook = _load_callable(reset_specification, label="reset hook")

        def prepare_dflash_full_prefix_call(
            *,
            input_ids: torch.Tensor,
            sequence_length: int,
            output_dflash_features: bool,
            logits_to_keep: int,
            call_index: int,
        ) -> None:
            result = reset_hook(
                target,
                input_ids=input_ids,
                sequence_length=sequence_length,
                output_dflash_features=output_dflash_features,
                logits_to_keep=logits_to_keep,
                call_index=call_index,
            )
            if result is not None:
                raise TypeError("receiver reset hook must return None")
            return None

        setattr(target, ISOLATION_MODE_ATTRIBUTE, "receiver_reset_hook")
        setattr(target, ISOLATION_HOOK_ATTRIBUTE, prepare_dflash_full_prefix_call)
        setattr(
            target,
            ISOLATION_EVIDENCE_ATTRIBUTE,
            os.environ.get(
                RESET_EVIDENCE_ENV,
                f"receiver reset hook {reset_specification}",
            ),
        )

    setattr(target, FULL_PREFIX_EXECUTION_MODE_ATTRIBUTE, "fresh_prefill")
    setattr(
        target,
        PREFILL_CHUNK_SIZE_ATTRIBUTE,
        _positive_env_int(PREFILL_CHUNK_SIZE_ENV, 64),
    )
    setattr(
        target,
        DECODE_CHUNK_SIZE_ATTRIBUTE,
        _positive_env_int(DECODE_CHUNK_SIZE_ENV, 1),
    )
    return target


def create_internal_target(
    target_dir: str,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Call the existing inference/bridge factory and prepare its contract."""

    specification = os.environ.get(TARGET_FACTORY_ENV)
    if not specification:
        raise RuntimeError(
            "embedded NPU execution requires --target-factory MODULE:FUNCTION"
        )
    factory = _load_callable(specification, label="target factory")
    target = factory(target_dir, device=device, dtype=dtype)
    if not isinstance(target, nn.Module):
        raise TypeError("target factory must return torch.nn.Module")
    return _prepare_target_contract(target)


def load_target(
    target_dir: str,
    device: str | torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    """Load the embedded receiver target and return the checked V1 facade."""

    _prepare_device_backend(device)
    requested_device = torch.device(device)
    root = Path(target_dir).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"internal target directory does not exist: {root}")
    target = create_internal_target(str(root), requested_device, dtype).eval()
    return InternalTargetFacade(
        target,
        device=requested_device,
        dtype=dtype,
        expected_vocab_size=OFFICIAL_VOCAB_SIZE,
        expected_hidden_size=OFFICIAL_HIDDEN_SIZE,
        expected_feature_size=OFFICIAL_FEATURE_SIZE,
    ).eval()


__all__ = ["InternalTargetFacade", "create_internal_target", "load_target"]
