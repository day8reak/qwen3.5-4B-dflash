from __future__ import annotations

import unittest

import torch
from torch import nn

from qwen35_mtp.config import Qwen35MTPConfig
from qwen35_mtp.mtp import Qwen35MTPDrafter


def tiny_config() -> Qwen35MTPConfig:
    return Qwen35MTPConfig(
        hidden_size=16,
        intermediate_size=32,
        vocab_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        partial_rotary_factor=0.5,
        mtp_num_hidden_layers=1,
        mtp_use_dedicated_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
        attn_output_gate=True,
        hidden_act="silu",
        dtype="float32",
    )


def initialized_drafter() -> Qwen35MTPDrafter:
    torch.manual_seed(1234)
    config = tiny_config()
    embedding = nn.Embedding(config.vocab_size, config.hidden_size)
    model = Qwen35MTPDrafter(config, embedding)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    model.eval()
    return model


class MTPMathTest(unittest.TestCase):
    def test_incremental_last_row_matches_full_causal_forward(self):
        model = initialized_drafter()
        input_ids = torch.tensor([[3, 4, 5, 6]], dtype=torch.long)
        hidden_sources = torch.randn(1, 4, model.config.hidden_size) * 0.1
        positions = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        full = model(input_ids, hidden_sources, positions)
        prefix = model(input_ids[:, :3], hidden_sources[:, :3], positions[:, :3])
        step = model(
            input_ids[:, 3:],
            hidden_sources[:, 3:],
            positions[:, 3:],
            past_key_values=prefix.cache,
        )
        torch.testing.assert_close(
            step.hidden_states[:, -1], full.hidden_states[:, -1], rtol=1e-5, atol=1e-5
        )
        torch.testing.assert_close(
            step.cache.key, full.cache.key, rtol=1e-4, atol=1e-6
        )
        torch.testing.assert_close(
            step.cache.value, full.cache.value, rtol=1e-4, atol=1e-6
        )

    def test_portable_block_matches_transformers_reference_components(self):
        try:
            from transformers import Qwen3_5TextConfig
            from transformers.models.qwen3_5.modeling_qwen3_5 import (
                Qwen3_5DecoderLayer,
                Qwen3_5RMSNorm,
            )
        except ImportError as error:
            self.skipTest(str(error))

        model = initialized_drafter()
        c = model.config
        reference_config = Qwen3_5TextConfig(
            vocab_size=c.vocab_size,
            hidden_size=c.hidden_size,
            intermediate_size=c.intermediate_size,
            num_hidden_layers=1,
            num_attention_heads=c.num_attention_heads,
            num_key_value_heads=c.num_key_value_heads,
            head_dim=c.head_dim,
            layer_types=["full_attention"],
            rms_norm_eps=c.rms_norm_eps,
            attention_bias=False,
            attention_dropout=0.0,
            attn_output_gate=True,
            hidden_act="silu",
            rope_parameters={
                "rope_type": "default",
                "rope_theta": c.rope_theta,
                "partial_rotary_factor": c.partial_rotary_factor,
                "mrope_section": [1, 1, 0],
            },
        )
        pre_embedding = Qwen3_5RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        pre_hidden = Qwen3_5RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        fusion = nn.Linear(c.hidden_size * 2, c.hidden_size, bias=False)
        layer = Qwen3_5DecoderLayer(reference_config, 0)
        final_norm = Qwen3_5RMSNorm(c.hidden_size, eps=c.rms_norm_eps)
        pre_embedding.load_state_dict(model.mtp.pre_fc_norm_embedding.state_dict())
        pre_hidden.load_state_dict(model.mtp.pre_fc_norm_hidden.state_dict())
        fusion.load_state_dict(model.mtp.fc.state_dict())
        layer.load_state_dict(model.mtp.layers[0].state_dict())
        final_norm.load_state_dict(model.mtp.norm.state_dict())

        input_ids = torch.tensor([[9, 10, 11]], dtype=torch.long)
        sources = torch.randn(1, 3, c.hidden_size) * 0.1
        positions = torch.tensor([[1, 2, 3]], dtype=torch.long)
        portable = model(input_ids, sources, positions)
        embeds = model.embed_tokens(input_ids)
        fused = fusion(torch.cat((pre_embedding(embeds), pre_hidden(sources)), dim=-1))
        cosine, sine = model.mtp.layers[0].self_attn.rotary(positions, fused.dtype)
        mask = model.mtp.layers[0].self_attn._causal_mask(
            3, 0, device=fused.device
        )
        reference = layer(
            fused,
            position_embeddings=(cosine, sine),
            attention_mask=mask,
            position_ids=positions,
        )
        reference = final_norm(reference)
        torch.testing.assert_close(
            portable.hidden_states, reference, rtol=1e-5, atol=1e-5
        )

    def test_prefill_uses_official_one_token_shift(self):
        model = initialized_drafter()
        prefix = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        hidden = torch.randn(1, 4, model.config.hidden_size)
        state = model.prefill(prefix, hidden)
        direct = model(
            prefix[:, 1:],
            hidden[:, :-1],
            torch.tensor([[1, 2, 3]], dtype=torch.long),
            project_top1=True,
        )
        torch.testing.assert_close(
            state.last_hidden_state, direct.hidden_states[:, -1:]
        )
        self.assertEqual(state.cache.sequence_length, prefix.shape[1] - 1)


if __name__ == "__main__":
    unittest.main()
