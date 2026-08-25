"""Whole-model and draft backend boundaries shared by CPU and Ascend adapters."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
from pathlib import Path
from typing import Any, Iterable, Protocol, Sequence

import torch
from torch import Tensor, nn

from .mtp import Qwen35MTPDrafter
from .ops import MtpOps, TorchMtpOps


@dataclass(frozen=True)
class MainEvaluation:
    """Main-model rows required by ordinary or speculative generation."""

    hidden_states: Tensor
    top1_token_ids: Tensor


class MainBackend(Protocol):
    backend_id: str

    def evaluate(
        self,
        input_ids: Tensor,
        top1_positions: Sequence[int],
    ) -> MainEvaluation: ...


class DraftBackend(Protocol):
    backend_id: str

    def propose(
        self,
        prefix_ids: Tensor,
        main_hidden_states: Tensor,
        max_draft_tokens: int,
        *,
        eos_token_ids: Iterable[int] = (),
    ) -> list[int]: ...


class TransformersMainBackend:
    """Text-only CPU/NPU reference backed by the official Transformers model.

    It deliberately recomputes the committed prefix.  This is slow but avoids
    making any unverified cache-rollback claim while the target runtime is
    unavailable.
    """

    backend_id = "transformers-qwen3.5-recompute"

    def __init__(
        self,
        owner: nn.Module,
        text_model: nn.Module,
        lm_head: nn.Module,
        *,
        device: torch.device,
        ops: MtpOps | None = None,
    ) -> None:
        self.owner = owner
        self.text_model = text_model
        self.lm_head = lm_head
        self.device = device
        self.ops = ops or TorchMtpOps()
        self._hidden_cache: dict[tuple[int, ...], Tensor] = {}

    @property
    def embedding(self) -> nn.Embedding:
        embedding = getattr(self.text_model, "embed_tokens", None)
        if not isinstance(embedding, nn.Embedding):
            raise TypeError("Qwen3.5 text model does not expose an embedding table")
        return embedding

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.bfloat16,
        ops: MtpOps | None = None,
    ) -> "TransformersMainBackend":
        from transformers import Qwen3_5ForConditionalGeneration

        target_device = torch.device(device)
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(Path(model_dir).expanduser().resolve()),
            dtype=dtype,
            local_files_only=True,
            attn_implementation="eager",
        )
        model.eval()
        if target_device.type != "cpu":
            model.to(target_device)
        text_model = model.model.language_model
        # The proof is text-only.  Release the unused vision tower after its
        # checkpoint has loaded; no multimodal correctness claim is made.
        model.model.visual = None
        return cls(
            model,
            text_model,
            model.lm_head,
            device=target_device,
            ops=ops,
        )

    @torch.inference_mode()
    def evaluate(
        self,
        input_ids: Tensor,
        top1_positions: Sequence[int],
    ) -> MainEvaluation:
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError("the reference main backend requires [1, sequence] input")
        cache_key = tuple(int(token) for token in input_ids[0].tolist())
        hidden_states = self._hidden_cache.get(cache_key)
        if hidden_states is None:
            input_ids = input_ids.to(device=self.device, dtype=torch.long)
            outputs = self.text_model(
                input_ids=input_ids, use_cache=False, return_dict=True
            )
            hidden_states = outputs.last_hidden_state
            self._hidden_cache[cache_key] = hidden_states
        if top1_positions:
            normalized = [
                position if position >= 0 else hidden_states.shape[1] + position
                for position in top1_positions
            ]
            if min(normalized) < 0 or max(normalized) >= hidden_states.shape[1]:
                raise IndexError("a requested LM-head row is outside the input sequence")
            rows = hidden_states[:, normalized, :]
            top1 = self.ops.top1(rows, self.lm_head.weight)
        else:
            top1 = torch.empty(
                (1, 0), dtype=torch.long, device=hidden_states.device
            )
        return MainEvaluation(hidden_states=hidden_states, top1_token_ids=top1)

    def clear_cache(self) -> None:
        self._hidden_cache.clear()


class TorchMTPDraftBackend:
    backend_id = "official-qwen3.5-mtp-pytorch"

    def __init__(self, drafter: Qwen35MTPDrafter) -> None:
        self.drafter = drafter

    def propose(
        self,
        prefix_ids: Tensor,
        main_hidden_states: Tensor,
        max_draft_tokens: int,
        *,
        eos_token_ids: Iterable[int] = (),
    ) -> list[int]:
        return self.drafter.propose(
            prefix_ids,
            main_hidden_states,
            max_draft_tokens,
            eos_token_ids=eos_token_ids,
        )


def _load_factory(specification: str):
    module_name, separator, attribute = specification.partition(":")
    module = importlib.import_module(module_name)
    if not separator:
        attribute = "create_backend"
    factory = getattr(module, attribute, None)
    if factory is None or not callable(factory):
        raise ValueError(
            f"backend factory {attribute!r} is missing from module {module_name!r}"
        )
    return factory


def load_external_main_backend(
    specification: str,
    *,
    model_dir: str | Path,
    options: dict[str, Any] | None = None,
) -> MainBackend:
    backend = _load_factory(specification)(
        role="main",
        model_dir=str(Path(model_dir).expanduser().resolve()),
        options=options or {},
    )
    if not hasattr(backend, "backend_id") or not callable(
        getattr(backend, "evaluate", None)
    ):
        raise TypeError("external main backend does not implement the required protocol")
    return backend


def load_external_draft_backend(
    specification: str,
    *,
    model_dir: str | Path,
    options: dict[str, Any] | None = None,
) -> DraftBackend:
    backend = _load_factory(specification)(
        role="draft",
        model_dir=str(Path(model_dir).expanduser().resolve()),
        options=options or {},
    )
    if not hasattr(backend, "backend_id") or not callable(
        getattr(backend, "propose", None)
    ):
        raise TypeError("external draft backend does not implement the required protocol")
    return backend


def build_torch_backends(
    model_dir: str | Path,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.bfloat16,
    ops: MtpOps | None = None,
) -> tuple[TransformersMainBackend, TorchMTPDraftBackend]:
    operations = ops or TorchMtpOps()
    main = TransformersMainBackend.from_pretrained(
        model_dir, device=device, dtype=dtype, ops=operations
    )
    drafter = Qwen35MTPDrafter.from_pretrained(
        model_dir,
        embedding=main.embedding,
        ops=operations,
        device=device,
        dtype=dtype,
    )
    return main, TorchMTPDraftBackend(drafter)
