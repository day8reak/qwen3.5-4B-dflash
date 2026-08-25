"""Locked text/MTP configuration extracted from a Qwen3.5 checkpoint."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen35MTPConfig:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    partial_rotary_factor: float
    mtp_num_hidden_layers: int
    mtp_use_dedicated_embeddings: bool
    attention_bias: bool
    attention_dropout: float
    attn_output_gate: bool
    hidden_act: str
    dtype: str

    @property
    def rotary_dim(self) -> int:
        return int(self.head_dim * self.partial_rotary_factor)

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, raw_config: dict[str, Any]) -> "Qwen35MTPConfig":
        text = raw_config.get("text_config", raw_config)
        rope = text.get("rope_parameters", {})
        config = cls(
            hidden_size=int(text["hidden_size"]),
            intermediate_size=int(text["intermediate_size"]),
            vocab_size=int(text["vocab_size"]),
            num_attention_heads=int(text["num_attention_heads"]),
            num_key_value_heads=int(text["num_key_value_heads"]),
            head_dim=int(text["head_dim"]),
            rms_norm_eps=float(text.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
            partial_rotary_factor=float(rope.get("partial_rotary_factor", 1.0)),
            mtp_num_hidden_layers=int(text.get("mtp_num_hidden_layers", 0)),
            mtp_use_dedicated_embeddings=bool(
                text.get("mtp_use_dedicated_embeddings", False)
            ),
            attention_bias=bool(text.get("attention_bias", False)),
            attention_dropout=float(text.get("attention_dropout", 0.0)),
            attn_output_gate=bool(text.get("attn_output_gate", False)),
            hidden_act=str(text.get("hidden_act", "silu")),
            dtype=str(text.get("dtype", raw_config.get("dtype", "bfloat16"))),
        )
        config.validate()
        return config

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen35MTPConfig":
        path = Path(model_dir) / "config.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "mtp_num_hidden_layers": self.mtp_num_hidden_layers,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"configuration values must be positive: {invalid}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.rotary_dim <= 0 or self.rotary_dim > self.head_dim:
            raise ValueError("rotary_dim must be in (0, head_dim]")
        if self.rotary_dim % 2:
            raise ValueError("rotary_dim must be even")
        if self.mtp_num_hidden_layers != 1:
            raise ValueError(
                "this reference implements the official one-layer Qwen3.5 MTP checkpoint"
            )
        if self.mtp_use_dedicated_embeddings:
            raise ValueError("dedicated MTP embeddings are outside this checkpoint contract")
        if self.attention_bias:
            raise ValueError("the locked Qwen3.5 MTP projections are bias-free")
        if not self.attn_output_gate:
            raise ValueError("the locked Qwen3.5 full-attention block requires a Q output gate")
        if self.hidden_act != "silu":
            raise ValueError("the locked Qwen3.5 MTP MLP requires SiLU/SwiGLU")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["rotary_dim"] = self.rotary_dim
        result["num_key_value_groups"] = self.num_key_value_groups
        return result

    def required_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        h = self.hidden_size
        i = self.intermediate_size
        q = self.num_attention_heads * self.head_dim
        kv = self.num_key_value_heads * self.head_dim
        d = self.head_dim
        return {
            "mtp.fc.weight": (h, h * 2),
            "mtp.layers.0.input_layernorm.weight": (h,),
            "mtp.layers.0.mlp.down_proj.weight": (h, i),
            "mtp.layers.0.mlp.gate_proj.weight": (i, h),
            "mtp.layers.0.mlp.up_proj.weight": (i, h),
            "mtp.layers.0.post_attention_layernorm.weight": (h,),
            "mtp.layers.0.self_attn.k_norm.weight": (d,),
            "mtp.layers.0.self_attn.k_proj.weight": (kv, h),
            "mtp.layers.0.self_attn.o_proj.weight": (h, q),
            "mtp.layers.0.self_attn.q_norm.weight": (d,),
            "mtp.layers.0.self_attn.q_proj.weight": (q * 2, h),
            "mtp.layers.0.self_attn.v_proj.weight": (kv, h),
            "mtp.norm.weight": (h,),
            "mtp.pre_fc_norm_embedding.weight": (h,),
            "mtp.pre_fc_norm_hidden.weight": (h,),
        }


OFFICIAL_QWEN35_4B = {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "vocab_size": 248320,
    "num_attention_heads": 16,
    "num_key_value_heads": 4,
    "head_dim": 256,
    "rotary_dim": 64,
    "mtp_num_hidden_layers": 1,
    "mtp_use_dedicated_embeddings": False,
}


def audit_official_4b_config(config: Qwen35MTPConfig) -> list[str]:
    """Return human-readable mismatches against the locked official 4B shape."""

    actual = config.to_dict()
    return [
        f"{key}: expected {expected!r}, got {actual.get(key)!r}"
        for key, expected in OFFICIAL_QWEN35_4B.items()
        if actual.get(key) != expected
    ]
