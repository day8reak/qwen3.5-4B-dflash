"""Bind the deployed Qwen3.5 wrapper to the rollback HIAI modeling.

The original deployment wrapper is receiver-owned and is intentionally not
copied into this repository.  This adapter reuses its weight-loading and
device setup exactly, while replacing only the module-global
``Qwen3_5ForCausalLM`` constructor for the duration of wrapper construction.
The replacement is process-local, protected by a lock, and restored before
``__init__`` returns.  A fail-closed identity check prevents a wrapper that
hard-codes a different modeling class from entering the rollback route.
"""

from __future__ import annotations

import importlib
from threading import RLock
from typing import Any

from torch import nn

from .modeling_qwen3_5_hiai_nd_dflash_rollback import Qwen3_5ForCausalLM


_BASE_WRAPPER_MODULE = "models.export_model_wrapper_qwen3_5"
_CONSTRUCTION_LOCK = RLock()


class Qwen3_5ForCausalLMWrapper(nn.Module):
    """Composition wrapper that preserves the deployed loader implementation."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        base_module = importlib.import_module(_BASE_WRAPPER_MODULE)
        base_wrapper_class = getattr(
            base_module,
            "Qwen3_5ForCausalLMWrapper",
            None,
        )
        if not isinstance(base_wrapper_class, type):
            raise TypeError(
                f"{_BASE_WRAPPER_MODULE} must export "
                "Qwen3_5ForCausalLMWrapper"
            )
        if not hasattr(base_module, "Qwen3_5ForCausalLM"):
            raise RuntimeError(
                f"{_BASE_WRAPPER_MODULE} does not expose its model constructor; "
                "the rollback adapter cannot safely bind it"
            )

        with _CONSTRUCTION_LOCK:
            original_model_class = base_module.Qwen3_5ForCausalLM
            base_module.Qwen3_5ForCausalLM = Qwen3_5ForCausalLM
            try:
                delegate = base_wrapper_class(*args, **kwargs)
            finally:
                base_module.Qwen3_5ForCausalLM = original_model_class

        if not isinstance(delegate, nn.Module):
            raise TypeError("the deployed Qwen3.5 wrapper must inherit nn.Module")
        if type(getattr(delegate, "model", None)) is not Qwen3_5ForCausalLM:
            raise RuntimeError(
                "the deployed wrapper did not construct the rollback "
                "Qwen3_5ForCausalLM; its loading implementation needs an "
                "explicit rollback adapter"
            )
        self.delegate = delegate

    @property
    def model(self) -> Qwen3_5ForCausalLM:
        model = getattr(self.delegate, "model", None)
        if type(model) is not Qwen3_5ForCausalLM:
            raise RuntimeError("the deployed wrapper replaced its rollback model")
        return model

    def replace_dflash_execution_model(self, model: nn.Module) -> None:
        """Install an in-place or replacement result from the Target quantizer.

        ``torch.nn.Module.__setattr__`` does not reliably dispatch a property
        setter for child modules, so the bridge uses this explicit method when
        the rollback composition wrapper is active.
        """

        if type(model) is not Qwen3_5ForCausalLM:
            raise TypeError(
                "quantized rollback Target must preserve Qwen3_5ForCausalLM"
            )
        self.delegate.model = model
        if getattr(self.delegate, "model", None) is not model:
            raise RuntimeError("deployed wrapper rejected the quantized Target")

    def dflash_target_input_provider_wrapper(self) -> nn.Module:
        """Expose the deployed wrapper expected by existing input providers."""

        return self.delegate

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.delegate(*args, **kwargs)


__all__ = ["Qwen3_5ForCausalLM", "Qwen3_5ForCausalLMWrapper"]
