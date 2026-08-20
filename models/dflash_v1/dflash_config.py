"""Locked configuration and tensor contract for Qwen3.5-4B-DFlash."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Qwen35DFlashConfig:
    hidden_size: int
    intermediate_size: int
    vocab_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_target_layers: int
    target_layer_ids: tuple[int, ...]
    layer_types: tuple[str, ...]
    block_size: int
    mask_token_id: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    sliding_window: int
    use_sliding_window: bool
    attention_bias: bool
    attention_dropout: float
    hidden_act: str
    dtype: str
    input_embedding_scale: float = 1.0
    output_multiplier: float = 1.0
    final_logit_softcapping: float | None = None

    @property
    def feature_size(self) -> int:
        return len(self.target_layer_ids) * self.hidden_size

    @property
    def query_width(self) -> int:
        return self.num_attention_heads * self.head_dim

    @property
    def key_value_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Qwen35DFlashConfig":
        dflash = raw.get("dflash_config", {})
        rope = raw.get("rope_parameters", {})
        config = cls(
            hidden_size=int(raw["hidden_size"]),
            intermediate_size=int(raw["intermediate_size"]),
            vocab_size=int(raw["vocab_size"]),
            num_hidden_layers=int(raw["num_hidden_layers"]),
            num_attention_heads=int(raw["num_attention_heads"]),
            num_key_value_heads=int(raw["num_key_value_heads"]),
            head_dim=int(raw["head_dim"]),
            num_target_layers=int(raw["num_target_layers"]),
            target_layer_ids=tuple(int(item) for item in dflash["target_layer_ids"]),
            layer_types=tuple(str(item) for item in raw["layer_types"]),
            block_size=int(dflash.get("block_size", 16)),
            mask_token_id=int(dflash["mask_token_id"]),
            rms_norm_eps=float(raw.get("rms_norm_eps", 1e-6)),
            rope_theta=float(rope.get("rope_theta", 10_000.0)),
            max_position_embeddings=int(raw.get("max_position_embeddings", 262_144)),
            sliding_window=int(raw.get("sliding_window", 4096)),
            use_sliding_window=bool(raw.get("use_sliding_window", False)),
            attention_bias=bool(raw.get("attention_bias", False)),
            attention_dropout=float(raw.get("attention_dropout", 0.0)),
            hidden_act=str(raw.get("hidden_act", "silu")),
            dtype=str(raw.get("dtype", raw.get("torch_dtype", "bfloat16"))),
            input_embedding_scale=float(dflash.get("input_embedding_scale", 1.0)),
            output_multiplier=float(dflash.get("output_multiplier", 1.0)),
            final_logit_softcapping=(
                None
                if dflash.get("final_logit_softcapping") is None
                else float(dflash["final_logit_softcapping"])
            ),
        )
        config.validate()
        return config

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "Qwen35DFlashConfig":
        path = Path(model_dir).expanduser().resolve() / "config.json"
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def validate(self) -> None:
        positive = {
            "hidden_size": self.hidden_size,
            "intermediate_size": self.intermediate_size,
            "vocab_size": self.vocab_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "num_target_layers": self.num_target_layers,
            "block_size": self.block_size,
            "max_position_embeddings": self.max_position_embeddings,
            "sliding_window": self.sliding_window,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"DFlash configuration values must be positive: {invalid}")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by KV heads")
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must contain one entry per draft layer")
        unsupported = sorted(set(self.layer_types) - {"sliding_attention", "full_attention"})
        if unsupported:
            raise ValueError(f"unsupported DFlash layer types: {unsupported}")
        if not self.target_layer_ids:
            raise ValueError("at least one target hidden layer is required")
        if tuple(sorted(set(self.target_layer_ids))) != self.target_layer_ids:
            raise ValueError("target_layer_ids must be sorted and unique")
        if self.target_layer_ids[0] < 0 or self.target_layer_ids[-1] >= self.num_target_layers:
            raise ValueError("a target layer ID is outside the target decoder")
        if not 0 <= self.mask_token_id < self.vocab_size:
            raise ValueError("mask_token_id is outside the vocabulary")
        if self.attention_bias:
            raise ValueError("the locked checkpoint uses bias-free projections")
        if self.attention_dropout != 0.0:
            raise ValueError("the inference golden requires zero attention dropout")
        if self.hidden_act != "silu":
            raise ValueError("the locked checkpoint requires SiLU/SwiGLU")
        if self.output_multiplier <= 0:
            raise ValueError("output_multiplier must be positive for stable Top1 semantics")
        if self.final_logit_softcapping is not None and self.final_logit_softcapping <= 0:
            raise ValueError("final_logit_softcapping must be positive when present")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["target_layer_ids"] = list(self.target_layer_ids)
        result["layer_types"] = list(self.layer_types)
        result["feature_size"] = self.feature_size
        result["query_width"] = self.query_width
        result["key_value_width"] = self.key_value_width
        result["num_key_value_groups"] = self.num_key_value_groups
        return result

    def required_tensor_shapes(self) -> dict[str, tuple[int, ...]]:
        hidden = self.hidden_size
        intermediate = self.intermediate_size
        query = self.query_width
        key_value = self.key_value_width
        shapes: dict[str, tuple[int, ...]] = {
            "fc.weight": (hidden, self.feature_size),
            "hidden_norm.weight": (hidden,),
            "norm.weight": (hidden,),
        }
        for layer in range(self.num_hidden_layers):
            prefix = f"layers.{layer}"
            shapes.update(
                {
                    f"{prefix}.input_layernorm.weight": (hidden,),
                    f"{prefix}.post_attention_layernorm.weight": (hidden,),
                    f"{prefix}.self_attn.q_proj.weight": (query, hidden),
                    f"{prefix}.self_attn.k_proj.weight": (key_value, hidden),
                    f"{prefix}.self_attn.v_proj.weight": (key_value, hidden),
                    f"{prefix}.self_attn.o_proj.weight": (hidden, query),
                    f"{prefix}.self_attn.q_norm.weight": (self.head_dim,),
                    f"{prefix}.self_attn.k_norm.weight": (self.head_dim,),
                    f"{prefix}.mlp.gate_proj.weight": (intermediate, hidden),
                    f"{prefix}.mlp.up_proj.weight": (intermediate, hidden),
                    f"{prefix}.mlp.down_proj.weight": (hidden, intermediate),
                }
            )
        return shapes

    @property
    def parameter_count(self) -> int:
        return sum(math.prod(shape) for shape in self.required_tensor_shapes().values())


OFFICIAL_QWEN35_4B_DFLASH = {
    "hidden_size": 2560,
    "intermediate_size": 9216,
    "vocab_size": 248320,
    "num_hidden_layers": 6,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "num_target_layers": 32,
    "target_layer_ids": [1, 5, 9, 13, 17, 21, 25, 29],
    "layer_types": [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ],
    "block_size": 16,
    "mask_token_id": 248077,
    "rms_norm_eps": 1e-6,
    "rope_theta": 10_000_000.0,
    "sliding_window": 4096,
    "use_sliding_window": True,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "hidden_act": "silu",
    "feature_size": 20480,
    "parameter_count": 634425856,
    "tensor_count": 69,
}


# Phase-1 target feature interface.  Keep these values in the configuration
# module so the target integration never duplicates layer IDs in its forward.
DFLASH_TARGET_LAYER_IDS = tuple(
    int(layer_id) for layer_id in OFFICIAL_QWEN35_4B_DFLASH["target_layer_ids"]
)
DFLASH_TARGET_HIDDEN_SIZE = int(OFFICIAL_QWEN35_4B_DFLASH["hidden_size"])
DFLASH_TARGET_NUM_HIDDEN_LAYERS = int(
    OFFICIAL_QWEN35_4B_DFLASH["num_target_layers"]
)
DFLASH_TARGET_FEATURE_SIZE = int(OFFICIAL_QWEN35_4B_DFLASH["feature_size"])
DFLASH_CONFIG = {
    "feature_layers": list(DFLASH_TARGET_LAYER_IDS),
    "feature_dim": DFLASH_TARGET_FEATURE_SIZE,
    "target_hidden_size": DFLASH_TARGET_HIDDEN_SIZE,
    "target_num_hidden_layers": DFLASH_TARGET_NUM_HIDDEN_LAYERS,
}


def audit_official_4b_dflash_config(config: Qwen35DFlashConfig) -> list[str]:
    actual = config.to_dict()
    actual["parameter_count"] = config.parameter_count
    actual["tensor_count"] = len(config.required_tensor_shapes())
    return [
        f"{name}: expected {expected!r}, got {actual.get(name)!r}"
        for name, expected in OFFICIAL_QWEN35_4B_DFLASH.items()
        if actual.get(name) != expected
    ]
