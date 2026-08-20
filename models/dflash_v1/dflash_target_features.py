"""Selective target hidden-state capture for Qwen3.5 DFlash integration.

This module is deliberately independent of the target's attention, GDN and
cache implementations.  A target decoder loop creates one collector per
forward, calls :meth:`capture` after each decoder layer, and exposes
:meth:`finalize` only when DFlash features were explicitly requested.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)

@dataclass(frozen=True)
class DFlashTargetFeatureSpec:
    """Contract for post-decoder hidden states consumed by a DFlash draft."""

    layer_ids: tuple[int, ...]
    hidden_size: int
    num_hidden_layers: int

    def __post_init__(self) -> None:
        layer_ids = tuple(int(layer_id) for layer_id in self.layer_ids)
        object.__setattr__(self, "layer_ids", layer_ids)
        if self.hidden_size <= 0 or self.num_hidden_layers <= 0:
            raise ValueError("hidden size and target layer count must be positive")
        if not layer_ids:
            raise ValueError("at least one DFlash target layer is required")
        if tuple(sorted(set(layer_ids))) != layer_ids:
            raise ValueError("DFlash target layer IDs must be sorted and unique")
        if layer_ids[0] < 0 or layer_ids[-1] >= self.num_hidden_layers:
            raise ValueError("a DFlash target layer ID is outside the decoder")

    @property
    def feature_size(self) -> int:
        return len(self.layer_ids) * self.hidden_size

    @classmethod
    def from_draft_config(cls, config: Any) -> "DFlashTargetFeatureSpec":
        return cls(
            layer_ids=tuple(config.target_layer_ids),
            hidden_size=int(config.hidden_size),
            num_hidden_layers=int(config.num_target_layers),
        )


# Keep the target-side feature helper independent of the draft implementation.
# These values are the frozen Qwen3.5-4B DFlash feature contract and are checked
# against ``models.dflash_v1.dflash_config`` by downstream checks.  A receiving
# target package therefore needs only this helper plus the patched modeling
# sibling; it must not need the draft-only ``dflash_config.py`` module.
QWEN35_4B_DFLASH_TARGET_FEATURES = DFlashTargetFeatureSpec(
    layer_ids=(1, 5, 9, 13, 17, 21, 25, 29),
    hidden_size=2560,
    num_hidden_layers=32,
)


class DFlashFeatureCollector:
    """Collect only selected post-layer outputs without changing target math.

    ``detach=True`` removes the auxiliary output from autograd while leaving
    the target's original hidden tensor untouched.  ``clone=True`` additionally
    protects captured values when a custom target mutates decoder outputs in
    place.  It defaults on for the accuracy-first integration phase.
    """

    def __init__(
        self,
        spec: DFlashTargetFeatureSpec = QWEN35_4B_DFLASH_TARGET_FEATURES,
        *,
        enabled: bool,
        detach: bool = True,
        clone: bool = True,
    ) -> None:
        self.spec = spec
        self.enabled = bool(enabled)
        self.detach = bool(detach)
        self.clone = bool(clone)
        self._selected = frozenset(spec.layer_ids)
        self._captured: dict[int, Tensor] = {}
        self._slots = {
            layer_index: slot
            for slot, layer_index in enumerate(self.spec.layer_ids)
        }
        self._feature_buffer: Tensor | None = None
        self._prefix_shape: tuple[int, int] | None = None
        self._dtype: torch.dtype | None = None
        self._device: torch.device | None = None

    @property
    def captured_layer_ids(self) -> tuple[int, ...]:
        return tuple(sorted(self._captured))

    def capture(self, layer_index: int, hidden_states: Tensor) -> None:
        """Capture a decoder output after layer ``layer_index`` has run."""

        layer_index = int(layer_index)
        if not self.enabled or layer_index not in self._selected:
            return
        if layer_index in self._captured:
            raise RuntimeError(f"DFlash target layer {layer_index} was captured twice")
        if hidden_states.ndim != 3:
            raise ValueError("a DFlash target hidden state must have shape [B,S,H]")
        if hidden_states.shape[-1] != self.spec.hidden_size:
            raise ValueError(
                f"target hidden width must be {self.spec.hidden_size}, "
                f"got {hidden_states.shape[-1]} at layer {layer_index}"
            )

        prefix_shape = tuple(hidden_states.shape[:2])
        if self._prefix_shape is None:
            self._prefix_shape = prefix_shape
            self._dtype = hidden_states.dtype
            self._device = hidden_states.device
        elif prefix_shape != self._prefix_shape:
            raise ValueError("captured DFlash target layers have different [B,S] shapes")
        elif hidden_states.dtype != self._dtype or hidden_states.device != self._device:
            raise ValueError("captured DFlash target layers differ in dtype or device")

        captured = hidden_states.detach() if self.detach else hidden_states
        if self.clone and self.detach:
            if self._feature_buffer is None:
                self._feature_buffer = hidden_states.new_empty(
                    (*prefix_shape, self.spec.feature_size)
                )
            slot = self._slots[layer_index]
            start = slot * self.spec.hidden_size
            captured = self._feature_buffer[
                ..., start : start + self.spec.hidden_size
            ]
            captured.copy_(hidden_states.detach())
        elif self.clone:
            captured = captured.clone()
        self._captured[layer_index] = captured

    def finalize(self) -> Tensor | None:
        """Concatenate selected layers in checkpoint order as ``[B,S,20480]``."""

        if not self.enabled:
            return None
        missing = [
            layer_index
            for layer_index in self.spec.layer_ids
            if layer_index not in self._captured
        ]
        if missing:
            raise RuntimeError(f"DFlash target layers were not captured: {missing}")
        features = self._feature_buffer
        if features is None:
            features = torch.cat(
                [self._captured[layer_index] for layer_index in self.spec.layer_ids],
                dim=-1,
            )
        if features.shape[-1] != self.spec.feature_size:
            raise RuntimeError("DFlash feature concatenation produced an invalid width")
        return features


@dataclass
class DFlashBaseModelOutputWithPast(BaseModelOutputWithPast):
    """Opt-in target base-model output; ordinary forwards keep their old type."""

    dflash_features: Tensor | None = None


@dataclass
class DFlashCausalLMOutputWithPast(CausalLMOutputWithPast):
    """Opt-in causal-LM output carrying selected target decoder features."""

    dflash_features: Tensor | None = None
