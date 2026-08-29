"""Text-only Qwen3.5 target adapter for the integrated DFlash graph."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from qwen35_dflash.model import extract_context_feature


class TransformersDFlashTargetAdapter(nn.Module):
    """Expose logits and official DFlash features from a Transformers target.

    This portable route intentionally recomputes the whole committed prefix and
    asks Transformers for diagnostic hidden states.  A production target can
    replace it with the selective ``output_dflash_features`` interface while
    keeping the exact same integrated graph ABI.
    """

    def __init__(
        self,
        language_model: nn.Module,
        lm_head: nn.Module,
        *,
        layer_ids: Sequence[int],
        target_hidden_size: int,
        target_num_hidden_layers: int,
        vocab_size: int,
    ) -> None:
        super().__init__()
        self.language_model = language_model
        self.lm_head = lm_head
        self.layer_ids = tuple(int(item) for item in layer_ids)
        if not self.layer_ids:
            raise ValueError("DFlash target adapter needs at least one layer")
        if tuple(sorted(set(self.layer_ids))) != self.layer_ids:
            raise ValueError("DFlash target layer IDs must be sorted and unique")
        if self.layer_ids[0] < 0 or self.layer_ids[-1] >= int(
            target_num_hidden_layers
        ):
            raise ValueError("a DFlash target layer ID is outside the target model")
        embedding = self.get_input_embeddings()
        output_embedding = self.get_output_embeddings()
        expected_shape = (int(vocab_size), int(target_hidden_size))
        if tuple(embedding.weight.shape) != expected_shape:
            raise ValueError(
                "target input embedding shape differs from the locked DFlash base: "
                f"{tuple(embedding.weight.shape)} != {expected_shape}"
            )
        if tuple(output_embedding.weight.shape) != expected_shape:
            raise ValueError(
                "target LM-head shape differs from the locked DFlash base: "
                f"{tuple(output_embedding.weight.shape)} != {expected_shape}"
            )

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        layer_ids: Sequence[int],
        target_hidden_size: int,
        target_num_hidden_layers: int,
        vocab_size: int,
        device: str | torch.device,
        dtype: torch.dtype,
        attn_implementation: str = "eager",
    ) -> "TransformersDFlashTargetAdapter":
        from transformers import Qwen3_5ForConditionalGeneration

        owner = Qwen3_5ForConditionalGeneration.from_pretrained(
            str(Path(model_dir).expanduser().resolve()),
            dtype=dtype,
            local_files_only=True,
            attn_implementation=str(attn_implementation),
            low_cpu_mem_usage=True,
        ).eval()
        language_model = owner.model.language_model
        lm_head = owner.lm_head
        # The deployment contract is text-only.  Do not move the unused vision
        # tower to the NPU or retain it in the exported module hierarchy.
        owner.model.visual = None
        adapter = cls(
            language_model,
            lm_head,
            layer_ids=layer_ids,
            target_hidden_size=target_hidden_size,
            target_num_hidden_layers=target_num_hidden_layers,
            vocab_size=vocab_size,
        ).eval()
        target_device = torch.device(device)
        if target_device.type != "cpu":
            adapter.to(target_device)
        return adapter

    def get_input_embeddings(self) -> nn.Module:
        embedding = getattr(self.language_model, "embed_tokens", None)
        if embedding is None or not hasattr(embedding, "weight"):
            raise TypeError("Qwen3.5 text target has no input embedding weight")
        return embedding

    def get_output_embeddings(self) -> nn.Module:
        if not hasattr(self.lm_head, "weight"):
            raise TypeError("Qwen3.5 text target has no LM-head weight")
        return self.lm_head

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        use_cache: bool = False,
        return_dict: bool = True,
        output_dflash_features: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if use_cache:
            raise ValueError("the recompute target adapter requires use_cache=False")
        if not return_dict:
            raise ValueError("the recompute target adapter requires return_dict=True")
        if not output_dflash_features:
            raise ValueError("the integrated graph requires DFlash target features")
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_hidden_states=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        last_hidden_state = getattr(outputs, "last_hidden_state", None)
        if hidden_states is None or last_hidden_state is None:
            raise RuntimeError("Transformers target did not return hidden states")
        features = extract_context_feature(hidden_states, self.layer_ids)
        logits = self.lm_head(last_hidden_state)
        return logits, features
