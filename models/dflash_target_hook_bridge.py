"""Eager target-feature bridge that leaves an internal Qwen3.5 target intact.

This bridge is for the correctness-first V1 route.  It registers temporary
forward hooks on the selected decoder layers, consumes
``output_dflash_features`` itself, and delegates every other argument to the
unchanged target.  The internal attention, GDN, cache and custom-operator
calls are therefore not replaced by this module.

The bridge is intentionally eager and serializes forwards while hooks are
installed.  Compiled/static-graph runtimes that do not execute PyTorch module
hooks must use the source-level collector insertion instead.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from threading import RLock
from typing import Any

import torch
from torch import Tensor, nn

from .dflash_target_features import (
    DFlashFeatureCollector,
    DFlashTargetFeatureSpec,
    QWEN35_4B_DFLASH_TARGET_FEATURES,
)


def _output_field(output: object, name: str) -> object | None:
    if isinstance(output, Mapping):
        return output.get(name)
    return getattr(output, name, None)


def _layer_hidden(output: object, *, layer_index: int) -> Tensor:
    if isinstance(output, Tensor):
        hidden = output
    elif isinstance(output, (tuple, list)) and output and isinstance(output[0], Tensor):
        hidden = output[0]
    elif isinstance(output, Mapping):
        candidate = output.get("last_hidden_state", output.get("hidden_states"))
        hidden = candidate if isinstance(candidate, Tensor) else None
    else:
        candidate = getattr(output, "last_hidden_state", None)
        hidden = candidate if isinstance(candidate, Tensor) else None
    if not isinstance(hidden, Tensor):
        raise TypeError(
            f"decoder layer {layer_index} must return a Tensor or a Tensor-first output"
        )
    return hidden


def _resolve_attribute(root: object, dotted_path: str) -> object:
    current = root
    for component in dotted_path.split("."):
        if not component:
            raise ValueError("decoder layer path contains an empty component")
        if not hasattr(current, component):
            raise AttributeError(
                f"decoder layer path {dotted_path!r} is missing component {component!r}"
            )
        current = getattr(current, component)
    return current


def resolve_qwen35_decoder_layers(
    target: nn.Module,
    *,
    dotted_path: str | None = None,
    num_hidden_layers: int = 32,
) -> tuple[nn.Module, ...]:
    """Resolve one explicit or well-known eager Qwen3.5 decoder-layer path."""

    if dotted_path is not None:
        candidates = ((dotted_path, _resolve_attribute(target, dotted_path)),)
    else:
        candidates = []
        for path in (
            "model.language_model.layers",
            "model.layers",
            "language_model.layers",
            "layers",
        ):
            try:
                value = _resolve_attribute(target, path)
            except AttributeError:
                continue
            candidates.append((path, value))

    valid: list[tuple[str, tuple[nn.Module, ...]]] = []
    for path, value in candidates:
        # ``nn.ModuleList`` deliberately behaves like a sequence but is not
        # registered as ``collections.abc.Sequence`` on every supported
        # PyTorch version.
        if not isinstance(value, (Sequence, nn.ModuleList)):
            continue
        layers = tuple(value)
        if len(layers) != num_hidden_layers:
            continue
        if not all(isinstance(layer, nn.Module) for layer in layers):
            continue
        valid.append((path, layers))

    if not valid:
        requested = dotted_path if dotted_path is not None else "known Qwen3.5 paths"
        raise ValueError(
            f"could not resolve {num_hidden_layers} decoder layers from {requested}; "
            "pass the internal layer path explicitly"
        )
    if len(valid) > 1:
        paths = ", ".join(path for path, _ in valid)
        raise ValueError(
            f"multiple decoder layer paths are valid ({paths}); choose one explicitly"
        )
    return valid[0][1]


@dataclass(frozen=True)
class DFlashHookedTargetOutput:
    """Minimal feature-enabled output consumed by the V1 adapter."""

    base_output: object
    logits: Tensor
    past_key_values: object | None
    dflash_features: Tensor

    def __getattr__(self, name: str) -> object:
        if isinstance(self.base_output, Mapping) and name in self.base_output:
            return self.base_output[name]
        return getattr(self.base_output, name)


class HookedDFlashTarget(nn.Module):
    """Wrap an eager target without changing its custom-operator implementation."""

    def __init__(
        self,
        target: nn.Module,
        decoder_layers: Sequence[nn.Module] | None = None,
        *,
        layer_path: str | None = None,
        feature_spec: DFlashTargetFeatureSpec = QWEN35_4B_DFLASH_TARGET_FEATURES,
        detach: bool = True,
        clone: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(target, nn.Module):
            raise TypeError("target must be a torch.nn.Module")
        if decoder_layers is not None and layer_path is not None:
            raise ValueError("pass decoder_layers or layer_path, not both")
        if decoder_layers is None:
            decoder_layers = resolve_qwen35_decoder_layers(
                target,
                dotted_path=layer_path,
                num_hidden_layers=feature_spec.num_hidden_layers,
            )
        layers = tuple(decoder_layers)
        if len(layers) != feature_spec.num_hidden_layers:
            raise ValueError(
                f"expected {feature_spec.num_hidden_layers} decoder layers, got {len(layers)}"
            )
        if not all(isinstance(layer, nn.Module) for layer in layers):
            raise TypeError("every decoder layer must be a torch.nn.Module")
        selected = [layers[index] for index in feature_spec.layer_ids]
        if len({id(layer) for layer in selected}) != len(selected):
            raise ValueError("selected decoder layers must be distinct modules")
        self.target = target
        self.decoder_layers = layers
        self.feature_spec = feature_spec
        self.detach = bool(detach)
        self.clone = bool(clone)
        self._forward_lock = RLock()

    @property
    def config(self):
        if not hasattr(self.target, "config"):
            raise AttributeError("internal target does not expose config")
        return self.target.config

    def get_input_embeddings(self):
        getter = getattr(self.target, "get_input_embeddings", None)
        if not callable(getter):
            raise TypeError("internal target does not expose get_input_embeddings()")
        return getter()

    def get_output_embeddings(self):
        getter = getattr(self.target, "get_output_embeddings", None)
        if not callable(getter):
            raise TypeError("internal target does not expose get_output_embeddings()")
        return getter()

    def forward(
        self,
        *args: Any,
        output_dflash_features: bool = False,
        **kwargs: Any,
    ) -> object:
        # Serialize ordinary calls too: an ordinary call entering while a
        # feature call has hooks installed would otherwise trigger those hooks.
        with self._forward_lock:
            if not output_dflash_features:
                return self.target(*args, **kwargs)

            collector = DFlashFeatureCollector(
                self.feature_spec,
                enabled=True,
                detach=self.detach,
                clone=self.clone,
            )
            handles = []
            try:
                for layer_index in self.feature_spec.layer_ids:
                    layer = self.decoder_layers[layer_index]

                    def capture(_module, _inputs, output, index=layer_index):
                        collector.capture(
                            index,
                            _layer_hidden(output, layer_index=index),
                        )

                    handles.append(layer.register_forward_hook(capture))
                base_output = self.target(*args, **kwargs)
            finally:
                for handle in handles:
                    handle.remove()

            logits = _output_field(base_output, "logits")
            if not isinstance(logits, Tensor):
                raise TypeError("internal target output must expose Tensor logits")
            features = collector.finalize()
            if not isinstance(features, Tensor):
                raise RuntimeError("feature collector returned no Tensor")
            return DFlashHookedTargetOutput(
                base_output=base_output,
                logits=logits,
                past_key_values=_output_field(base_output, "past_key_values"),
                dflash_features=features,
            )
