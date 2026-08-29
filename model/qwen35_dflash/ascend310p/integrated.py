"""One fixed-gear target+DFlash graph for exact recompute deployment."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .contracts import AirGraphSpec


class IntegratedDFlashRecomputeGraph(nn.Module):
    """Combine an existing target model and the official DFlash drafter.

    The target must implement the opt-in ``output_dflash_features`` contract.
    Inputs are right padded, so causal target outputs for the valid committed
    prefix are unaffected by padding.  DFlash receives the same validity mask
    and logical positions, preventing padded target rows from entering its
    sliding/full attention.

    Outputs are intentionally small: target Top1 for every fixed-gear row and
    DFlash proposal Top1.  A host scheduler can therefore run exact ordinary
    verification without transferring full-vocabulary logits or hidden states.
    """

    def __init__(self, target_model: nn.Module, draft_model: nn.Module) -> None:
        super().__init__()
        self.target_model = target_model
        self.draft_model = draft_model
        config = getattr(draft_model, "config", None)
        if config is None:
            raise TypeError("DFlash draft model must expose config")
        self.block_size = int(config.block_size)
        self.mask_token_id = int(config.mask_token_id)
        if self.block_size < 2:
            raise ValueError("integrated DFlash graph needs at least one proposal row")
        embedding = self._embedding()
        output_embedding = self._output_embedding()
        if embedding.weight.ndim != 2 or output_embedding.weight.ndim != 2:
            raise ValueError("target embedding and LM head must be matrices")
        if embedding.weight.shape != output_embedding.weight.shape:
            raise ValueError("target embedding and LM head shapes differ")
        if embedding.weight.shape[0] != int(config.vocab_size):
            raise ValueError("target vocabulary differs from the DFlash checkpoint")
        if embedding.weight.shape[1] != int(config.hidden_size):
            raise ValueError("target hidden width differs from the DFlash checkpoint")

    def _embedding(self) -> nn.Module:
        getter = getattr(self.target_model, "get_input_embeddings", None)
        if not callable(getter):
            raise TypeError("target model does not expose get_input_embeddings()")
        embedding = getter()
        if embedding is None or not hasattr(embedding, "weight"):
            raise TypeError("target input embedding has no weight")
        return embedding

    def _output_embedding(self) -> nn.Module:
        getter = getattr(self.target_model, "get_output_embeddings", None)
        if not callable(getter):
            raise TypeError("target model does not expose get_output_embeddings()")
        embedding = getter()
        if embedding is None or not hasattr(embedding, "weight"):
            raise TypeError("target output embedding has no weight")
        return embedding

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("integrated graph inputs must be matching [B,S] tensors")
        if input_ids.shape[0] != 1:
            raise ValueError("integrated DFlash graph currently requires batch size 1")
        valid = attention_mask.to(dtype=torch.bool)
        target_outputs = self.target_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            output_dflash_features=True,
        )
        if isinstance(target_outputs, tuple) and len(target_outputs) == 2:
            logits, target_hidden = target_outputs
        else:
            logits = getattr(target_outputs, "logits", None)
            target_hidden = getattr(target_outputs, "dflash_features", None)
        if logits is None or target_hidden is None:
            raise RuntimeError("target model did not return logits and dflash_features")
        if logits.shape[:2] != input_ids.shape:
            raise ValueError("target logits do not cover the fixed sequence gear")
        if target_hidden.shape[:2] != input_ids.shape:
            raise ValueError("target DFlash features do not cover the fixed sequence gear")
        target_top1 = torch.argmax(logits, dim=-1)

        sequence_lengths = valid.to(dtype=torch.long).sum(dim=-1, keepdim=True)
        anchor_indices = (sequence_lengths - 1).clamp_min(0)
        anchor_ids = torch.gather(input_ids, dim=1, index=anchor_indices)
        mask_ids = torch.full(
            (input_ids.shape[0], self.block_size - 1),
            self.mask_token_id,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        block_ids = torch.cat((anchor_ids, mask_ids), dim=-1)
        noise_embedding = self.draft_model.embed_block(
            block_ids,
            self._embedding().weight,
        )

        # The official DFlash block starts with the already committed current
        # token.  Its target feature belongs to the block, not to the context;
        # including it in both places shifts every proposal by one token.
        physical_positions = torch.arange(
            input_ids.shape[1], dtype=torch.long, device=input_ids.device
        ).view(1, -1)
        context_valid = valid & (physical_positions < anchor_indices)
        context_lengths = context_valid.to(dtype=torch.long).sum(dim=-1, keepdim=True)
        context_positions = context_valid.to(dtype=torch.long).cumsum(dim=-1) - 1
        context_positions = context_positions.clamp_min(0)
        draft_offsets = torch.arange(
            self.block_size,
            dtype=torch.long,
            device=input_ids.device,
        ).view(1, -1)
        draft_positions = context_lengths + draft_offsets
        position_ids = torch.cat((context_positions, draft_positions), dim=-1)
        draft_top1 = self.draft_model.draft_top1(
            target_hidden,
            noise_embedding,
            position_ids,
            self._output_embedding().weight,
            context_attention_mask=context_valid,
        )
        return target_top1, draft_top1


def integrated_recompute_graph_spec(
    target_model: nn.Module,
    draft_model: nn.Module,
    *,
    max_sequence_length: int,
    example_sequence_length: int = 2,
    pad_token_id: int = 0,
    device: str | torch.device = "npu:0",
    name: str = "dflash_recompute",
    metadata: dict[str, Any] | None = None,
) -> AirGraphSpec:
    """Create the standard fixed-gear AIR spec used by the built-in backend."""

    if max_sequence_length <= 1:
        raise ValueError("max_sequence_length must exceed one token")
    if not 1 <= example_sequence_length <= max_sequence_length:
        raise ValueError("example_sequence_length is outside the fixed gear")
    reserved_metadata = {
        "max_sequence_length",
        "block_size",
        "padding",
        "state_policy",
    }
    conflicts = reserved_metadata.intersection(metadata or {})
    if conflicts:
        raise ValueError(
            "integrated graph metadata cannot override reserved fields: "
            f"{sorted(conflicts)}"
        )
    graph = IntegratedDFlashRecomputeGraph(target_model, draft_model).eval()
    target_device = torch.device(device)
    input_ids = torch.full(
        (1, max_sequence_length),
        int(pad_token_id),
        dtype=torch.long,
        device=target_device,
    )
    attention_mask = torch.zeros_like(input_ids)
    attention_mask[:, :example_sequence_length] = 1
    return AirGraphSpec(
        name=name,
        role="generation-recompute",
        model=graph,
        example_args=(input_ids, attention_mask),
        input_names=("input_ids", "attention_mask"),
        output_names=("target_top1", "draft_top1"),
        dynamic=False,
        metadata={
            **(metadata or {}),
            "max_sequence_length": max_sequence_length,
            "block_size": graph.block_size,
            "padding": "right",
            "state_policy": "recompute committed prefixes",
        },
    )
