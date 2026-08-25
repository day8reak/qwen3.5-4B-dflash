"""Fixed-gear ONNX export for the MTP core without duplicating tied weights."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from .mtp import MTPKVCache, Qwen35MTPModule


class MTPCoreExportWrapper(nn.Module):
    """Export boundary using materialized embeddings and external tied LM head.

    Keeping embedding lookup and full-vocabulary Top1 outside this graph lets a
    target reuse the ordinary model's existing operators and avoids a duplicate
    248320x2560 table in the ONNX artifact.
    """

    def __init__(self, mtp: Qwen35MTPModule) -> None:
        super().__init__()
        self.mtp = mtp

    def forward(
        self,
        inputs_embeds: Tensor,
        hidden_sources: Tensor,
        position_ids: Tensor,
        past_key: Tensor,
        past_value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        hidden, cache = self.mtp(
            inputs_embeds,
            hidden_sources,
            position_ids,
            past_key_values=MTPKVCache(past_key, past_value),
        )
        return hidden, cache.key, cache.value


def export_mtp_core_onnx(
    wrapper: MTPCoreExportWrapper,
    output: str | Path,
    *,
    sequence_length: int,
    past_length: int,
    hidden_size: int,
    kv_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    opset_version: int = 18,
) -> Path:
    if sequence_length <= 0 or past_length < 0:
        raise ValueError("sequence_length must be positive and past_length non-negative")
    output_path = Path(output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    device = next(wrapper.parameters()).device
    inputs = (
        torch.zeros((1, sequence_length, hidden_size), dtype=dtype, device=device),
        torch.zeros((1, sequence_length, hidden_size), dtype=dtype, device=device),
        torch.arange(
            past_length, past_length + sequence_length, device=device
        ).view(1, sequence_length),
        torch.zeros((1, kv_heads, past_length, head_dim), dtype=dtype, device=device),
        torch.zeros((1, kv_heads, past_length, head_dim), dtype=dtype, device=device),
    )
    wrapper.eval()
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            inputs,
            str(output_path),
            input_names=[
                "inputs_embeds",
                "hidden_sources",
                "position_ids",
                "past_key",
                "past_value",
            ],
            output_names=["mtp_hidden", "present_key", "present_value"],
            opset_version=opset_version,
            do_constant_folding=True,
            dynamo=False,
            external_data=True,
        )
    return output_path
