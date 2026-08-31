"""TorchAir graph factory for the repository's ``quant`` branch.

This module deliberately reuses the quant branch instead of constructing a
second target implementation:

* ``models.internal_dflash_bridge.load_qwen35_target`` owns Target loading;
* ``models.dflash_v1.original_quant.quant_model`` installs the original W8A8
  ``QLinear`` modules;
* the YAML-declared INT8 embedding and FP32 scales feed the Target;
* ``models.dflash_v1.modeling_dflash.DFlashDraftModel`` remains FP16.

The first OM route is a fixed-gear, full-prefix recompute graph.  It is the
smallest state-safe ABI that a C/C++ AscendCL host can call directly.  The
existing quant rollback implementation remains the semantic/performance
reference for a later explicit-state incremental OM suite.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib
import json
import os
from pathlib import Path
from typing import Any, Iterator, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .contracts import AirGraphSpec, CustomOpExportSpec
from .custom_op_export import (
    ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
    ADN_FUSED_INFER_ATTENTION_TORCH_OP,
    ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
    ADN_RMS_NORM_TORCH_OP,
    FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP,
    NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
    NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
    NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
    NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
    NPU_DYNAMIC_QUANT_TORCH_OP,
    NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
    NPU_QUANT_MATMUL_TORCH_OP,
    NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE,
    NPU_SCATTER_ND_UPDATE_TORCH_OP,
)
from .integrated import (
    enable_padded_draft_context,
    integrated_recompute_graph_spec,
)


QUANT_BASE_REVISION = "28f93e784a2beed87020a80bd93c8788754eab1c"
QUANT_GRAPH_FACTORY_ID = "qwen3.5-4b-quant-w8a8-dflash-recompute-v4"
_TARGET_GDN_CHUNK = 64
_GDR_EFFECTIVE_LENGTH_MAX = torch.iinfo(torch.int16).max
_DTYPES = {"float16": torch.float16}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_quant_source_lock() -> dict[str, Any]:
    """Verify every declared source-file/hash pair before loading weights."""

    repository_root = Path(__file__).resolve().parents[4]
    lock_path = repository_root / "SOURCE_LOCK.json"
    if lock_path.is_symlink() or not lock_path.is_file():
        raise FileNotFoundError(f"quant SOURCE_LOCK.json is missing: {lock_path}")
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("quant SOURCE_LOCK.json must contain an object")
    verified: list[dict[str, Any]] = []

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                if isinstance(key, str) and key.endswith("_file"):
                    hash_key = key[:-5] + "_sha256"
                    expected = value.get(hash_key)
                    if isinstance(item, str) and isinstance(expected, str):
                        candidate = repository_root / item
                        if candidate.is_symlink():
                            raise ValueError(
                                f"SOURCE_LOCK payload must not be a symlink: {item}"
                            )
                        path = candidate.resolve()
                        if (
                            path == repository_root
                            or repository_root not in path.parents
                            or not path.is_file()
                        ):
                            raise FileNotFoundError(
                                f"SOURCE_LOCK payload is invalid: {item}"
                            )
                        actual = _sha256(path)
                        if actual != expected:
                            raise ValueError(
                                f"quant source differs from SOURCE_LOCK: {item}"
                            )
                        verified.append(
                            {"path": item, "sha256": actual}
                        )
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    if len(verified) < 10:
        raise ValueError("quant SOURCE_LOCK contains too few verified source pairs")
    return {
        "path": str(lock_path),
        "sha256": _sha256(lock_path),
        "verified_file_count": len(verified),
    }


def _required_directory(config: Mapping[str, Any], name: str) -> Path:
    if name not in config:
        raise ValueError(f"quant AIR factory requires {name}")
    raw = Path(str(config[name])).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {raw}")
    path = raw.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name} is not a regular directory: {path}")
    return path


def _required_file(config: Mapping[str, Any], name: str) -> Path:
    if name not in config:
        raise ValueError(f"quant AIR factory requires {name}")
    raw = Path(str(config[name])).expanduser()
    if raw.is_symlink():
        raise ValueError(f"{name} must not be a symlink: {raw}")
    path = raw.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{name} is not a regular file: {path}")
    return path


def _append_receiver_models(receiver_models_dir: Path | None) -> None:
    """Extend the already selected ``models`` package with receiver files."""

    if receiver_models_dir is None:
        return
    wrapper = receiver_models_dir / "export_model_wrapper_qwen3_5.py"
    if wrapper.is_symlink() or not wrapper.is_file():
        raise FileNotFoundError(
            "receiver_models_dir must contain export_model_wrapper_qwen3_5.py"
        )
    package = importlib.import_module("models")
    search_path = getattr(package, "__path__", None)
    if search_path is None:
        raise TypeError("the selected models module is not a package")
    value = str(receiver_models_dir)
    if value not in search_path:
        search_path.append(value)


@contextmanager
def _quant_environment(
    *,
    quant_config: Path,
    kv_cache_max_len: int,
) -> Iterator[Mapping[str, Any]]:
    """Install the exact environment consumed by the quant branch loader."""

    from models.dflash_v1.target_quant import (
        QUANT_MODE_W8A8_DYNAMIC,
        TARGET_EMBEDDING_SCALE_PATH_ENV,
        TARGET_EMBEDDING_WEIGHT_PATH_ENV,
        TARGET_QUANT_CONFIG_ENV,
        TARGET_QUANT_MODE_ENV,
        TARGET_QUANT_WEIGHT_PATH_ENV,
        load_original_quant_config,
    )

    resolved = load_original_quant_config(quant_config)
    values = {
        TARGET_QUANT_MODE_ENV: QUANT_MODE_W8A8_DYNAMIC,
        TARGET_QUANT_CONFIG_ENV: str(resolved.config_path),
        TARGET_QUANT_WEIGHT_PATH_ENV: str(resolved.quant_weight_path),
        TARGET_EMBEDDING_WEIGHT_PATH_ENV: str(resolved.embedding_weight_path),
        TARGET_EMBEDDING_SCALE_PATH_ENV: str(resolved.embedding_scale_path),
        "DFLASH_HIAI_KV_CACHE_MAX_LEN": str(kv_cache_max_len),
    }
    previous = {name: os.environ.get(name) for name in values}
    os.environ.update(values)
    try:
        yield {
            "config_path": str(resolved.config_path),
            "config_sha256": _sha256(resolved.config_path),
            "quant_weight_path": str(resolved.quant_weight_path),
            "embedding_weight_path": str(resolved.embedding_weight_path),
            "embedding_scale_path": str(resolved.embedding_scale_path),
        }
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _repeat_kv(states: Tensor, repetitions: int) -> Tensor:
    if repetitions == 1:
        return states
    batch, heads, sequence, head_dim = states.shape
    expanded = states[:, :, None, :, :].expand(
        batch, heads, repetitions, sequence, head_dim
    )
    return expanded.reshape(batch, heads * repetitions, sequence, head_dim)


def _rotate_half(value: Tensor) -> Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class AirDFlashOps:
    """Pure Tensor form of the quant branch's strict Ascend draft operations.

    Runtime-only finite checks intentionally stay outside the exported graph;
    they would introduce host synchronization and graph breaks.  The formulas,
    FP32 reduction/softmax boundaries, and argmax tie behavior are unchanged.
    """

    def rms_norm(self, value: Tensor, weight: Tensor, eps: float) -> Tensor:
        value_fp32 = value.float()
        normalized = value_fp32 * torch.rsqrt(
            value_fp32.square().mean(dim=-1, keepdim=True) + float(eps)
        )
        return weight * normalized.to(value.dtype)

    def linear(self, value: Tensor, weight: Tensor) -> Tensor:
        return F.linear(value, weight)

    def rotary(
        self,
        query: Tensor,
        key: Tensor,
        cosine: Tensor,
        sine: Tensor,
    ) -> tuple[Tensor, Tensor]:
        cosine_heads = cosine.unsqueeze(1)
        sine_heads = sine.unsqueeze(1)
        query_length = query.shape[-2]
        return (
            query * cosine_heads[..., -query_length:, :]
            + _rotate_half(query) * sine_heads[..., -query_length:, :],
            key * cosine_heads + _rotate_half(key) * sine_heads,
        )

    def attention(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        attention_mask: Tensor | None,
        scale: float,
        key_value_groups: int,
    ) -> Tensor:
        key = _repeat_kv(key, int(key_value_groups))
        value = _repeat_kv(value, int(key_value_groups))
        scores = torch.matmul(query.float(), key.float().transpose(-2, -1))
        scores = scores * float(scale)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32)
        return torch.matmul(probabilities, value.float()).to(query.dtype)

    def swiglu(self, gate: Tensor, up: Tensor) -> Tensor:
        return F.silu(gate) * up

    def top1(self, hidden: Tensor, lm_head_weight: Tensor) -> Tensor:
        return torch.argmax(F.linear(hidden, lm_head_weight), dim=-1)


class QuantFullPrefixExportTarget(nn.Module):
    """Adapt the quant Target to a single, capture-safe fixed AIR graph.

    The public ``InternalDFlashTarget.forward`` deliberately owns Python-side
    call guards, counters and a device synchronization used by the eager
    full-prefix oracle.  Those operations must not enter a TorchAir graph.
    This adapter therefore calls the hash-locked receiver model directly and
    keeps its fresh state local to the exported graph.  The private bridge ABI
    is safe here only because ``SOURCE_LOCK.json`` is verified before this
    module is constructed.
    """

    def __init__(self, target: nn.Module) -> None:
        super().__init__()
        required = (
            "_fresh_attention_mask",
            "_fresh_hybrid_cache",
            "dflash_execution_model",
        )
        missing = [name for name in required if not hasattr(target, name)]
        if missing:
            raise TypeError(
                "quant Target lacks the locked AIR bridge ABI: "
                + ", ".join(missing)
            )
        quantized_embedding = getattr(target, "_target_quantized_embedding", None)
        if not isinstance(quantized_embedding, nn.Module):
            raise TypeError("quant AIR export requires the W8A8 target embedding")
        self.target = target

    def get_input_embeddings(self) -> nn.Module:
        return self.target.get_input_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.target.get_output_embeddings()

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        **kwargs: Any,
    ) -> tuple[Tensor, Tensor]:
        # The physical gear is always fully executed. Right-padding comes
        # after every valid token, so causal Target rows inside the valid
        # prefix cannot depend on padding. The receiver's original GDR ABI
        # nevertheless needs the call-local logical row count as INT16[B].
        # Derive it inside the graph to keep the external two-input OM ABI.
        del kwargs
        sequence_length = int(input_ids.shape[1])
        gdr_effective_length = (
            attention_mask.to(dtype=torch.bool)
            .to(dtype=torch.long)
            .sum(dim=1)
            .to(dtype=torch.int16)
        )
        inputs_embeds = self.target._target_quantized_embedding(input_ids)
        positions = torch.arange(
            sequence_length,
            dtype=torch.long,
            device=input_ids.device,
        )
        state = self.target._fresh_hybrid_cache(batch_size=1)
        outputs = self.target.dflash_execution_model(
            input_ids=input_ids,
            attention_mask=self.target._fresh_attention_mask(sequence_length),
            position_ids=positions.unsqueeze(0),
            past_key_values=state,
            new_kv_cache_pos=positions,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            inputs_embeds=inputs_embeds,
            embed_scale=None,
            output_pos=None,
            allQLen=[sequence_length],
            output_dflash_features=True,
            gdr_effective_length=gdr_effective_length,
            export_flag=True,
        )
        # The locked rollback/non-rollback receiver both expose the same
        # feature-enabled tensor tuple.  Avoid the eager bridge's generic
        # mapping inspection because TorchAir must see one tensor-only graph.
        logits, features = outputs
        return logits, features


def create_quant_recompute_graph(
    config: Mapping[str, Any],
) -> tuple[AirGraphSpec, ...]:
    """Load the locked quant Target/Draft pair and return one AIR graph spec."""

    max_sequence_length = int(config.get("max_sequence_length", 0))
    if max_sequence_length <= 0 or max_sequence_length % _TARGET_GDN_CHUNK:
        raise ValueError(
            "max_sequence_length must be positive and divisible by the "
            "quant Target's 64-token GDN chunk"
        )
    if max_sequence_length > _GDR_EFFECTIVE_LENGTH_MAX:
        raise ValueError(
            "max_sequence_length exceeds the original GDR INT16 "
            "effective_length ABI"
        )
    example_sequence_length = int(config.get("example_sequence_length", 2))
    if not 1 <= example_sequence_length <= max_sequence_length:
        raise ValueError("example_sequence_length is outside the fixed gear")
    dtype_name = str(config.get("dtype", "float16"))
    if dtype_name not in _DTYPES:
        raise ValueError("quant AIR export supports Target/Draft float16 only")
    dtype = _DTYPES[dtype_name]
    device = str(config.get("device", "npu:0"))
    if not device.startswith("npu"):
        raise ValueError("formal quant AIR export requires an explicit NPU device")
    custom_op_exports = (
        CustomOpExportSpec(
            torch_op=NPU_DYNAMIC_QUANT_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "npu_dynamic_quant_ge_op_type",
                    NPU_DYNAMIC_QUANT_DEFAULT_GE_OP_TYPE,
                )
            ),
        ),
        CustomOpExportSpec(
            torch_op=NPU_QUANT_MATMUL_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "npu_quant_matmul_ge_op_type",
                    NPU_QUANT_MATMUL_DEFAULT_GE_OP_TYPE,
                )
            ),
        ),
        CustomOpExportSpec(
            torch_op=ADN_RMS_NORM_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "adn_rms_norm_ge_op_type",
                    ADN_RMS_NORM_DEFAULT_GE_OP_TYPE,
                )
            ),
        ),
        CustomOpExportSpec(
            torch_op=NPU_CHUNK_GATED_DELTA_RULE_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "npu_chunk_gated_delta_rule_ge_op_type",
                    NPU_CHUNK_GATED_DELTA_RULE_DEFAULT_GE_OP_TYPE,
                )
            ),
        ),
        CustomOpExportSpec(
            torch_op=FUNCTIONAL_NPU_CACHE_UPDATE_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "npu_cache_update_ge_op_type",
                    NPU_CACHE_UPDATE_DEFAULT_GE_OP_TYPE,
                )
            ),
        ),
        CustomOpExportSpec(
            torch_op=ADN_FUSED_INFER_ATTENTION_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "adn_fused_infer_attention_ge_op_type",
                    ADN_FUSED_INFER_ATTENTION_DEFAULT_GE_OP_TYPE,
                )
            ),
        ),
        # forward1 is not selected by the fixed recompute graph, but its
        # schema/Meta contract must still be valid if that receiver path is
        # enabled later. Zero means preflight it without inventing a graph hit.
        CustomOpExportSpec(
            torch_op=NPU_SCATTER_ND_UPDATE_TORCH_OP,
            ge_op_type=str(
                config.get(
                    "npu_scatter_nd_update_ge_op_type",
                    NPU_SCATTER_ND_UPDATE_DEFAULT_GE_OP_TYPE,
                )
            ),
            minimum_occurrences=0,
        ),
    )
    target_dir = _required_directory(config, "target_dir")
    draft_dir = _required_directory(config, "draft_dir")
    quant_config = _required_file(config, "quant_config")
    input_manifest = _required_file(config, "input_manifest")
    receiver_models_dir = _required_directory(config, "receiver_models_dir")
    if (
        receiver_models_dir.is_symlink() or not receiver_models_dir.is_dir()
    ):
        raise FileNotFoundError(
            f"receiver_models_dir is not a regular directory: {receiver_models_dir}"
        )
    source_lock_identity = _verify_quant_source_lock()
    from .input_manifest import verify_quant_input_manifest

    locked_inputs = verify_quant_input_manifest(input_manifest)
    locked_roots = locked_inputs["roots"]
    locked_files = locked_inputs["files"]
    expected_paths = {
        "target_dir": (
            target_dir,
            locked_roots["target_checkpoint"],
        ),
        "draft_dir": (
            draft_dir,
            locked_roots["draft_checkpoint"],
        ),
        "quant_config": (
            quant_config,
            locked_files["quant_config"][0],
        ),
        "receiver_wrapper": (
            receiver_models_dir / "export_model_wrapper_qwen3_5.py",
            locked_files["receiver_wrapper"][0],
        ),
    }
    mismatches = {
        name: {"configured": str(configured), "locked": str(locked)}
        for name, (configured, locked) in expected_paths.items()
        if configured.resolve() != locked.resolve()
    }
    if mismatches:
        raise ValueError(f"factory paths differ from input_manifest: {mismatches}")
    _append_receiver_models(receiver_models_dir)

    # Fail before loading 4B weights if the NPU extension is unavailable.
    try:
        importlib.import_module("torch_npu")
    except ImportError as error:
        raise RuntimeError("torch_npu is required for quant AIR export") from error

    from models.dflash_v1.modeling_dflash import DFlashDraftModel
    from models.internal_dflash_bridge import load_qwen35_target

    with _quant_environment(
        quant_config=quant_config,
        kv_cache_max_len=max_sequence_length,
    ) as quant_identity:
        quant_paths = {
            Path(str(quant_identity["quant_weight_path"])).resolve(),
            Path(str(quant_identity["embedding_weight_path"])).resolve(),
            Path(str(quant_identity["embedding_scale_path"])).resolve(),
        }
        locked_quant_paths = {
            locked_roots["quant_linear_weights"].resolve(),
            *(item.resolve() for item in locked_files["quant_embedding"]),
        }
        if quant_paths != locked_quant_paths:
            raise ValueError(
                "quant YAML artifact paths differ from the input_manifest"
            )
        target = load_qwen35_target(
            str(target_dir),
            device=torch.device(device),
            dtype=dtype,
        )
    draft = DFlashDraftModel.from_pretrained(
        draft_dir,
        ops=AirDFlashOps(),
        device=device,
        dtype=dtype,
    ).eval()
    enable_padded_draft_context(draft)
    target_adapter = QuantFullPrefixExportTarget(target).eval()
    pad_token_id = int(config.get("pad_token_id", 0))
    metadata = {
        "factory_id": QUANT_GRAPH_FACTORY_ID,
        "quant_branch_base_revision": QUANT_BASE_REVISION,
        "quant_source_lock": source_lock_identity,
        "target_precision": "W8A8 dynamic QLinear with FP16 outputs",
        "target_quant_mode": "w8a8_dynamic",
        "target_embedding": "INT8 weight * FP32 row scale -> FP16",
        "draft_precision": "FP16",
        "draft_dtype": dtype_name,
        "dtype": dtype_name,
        "target_checkpoint_manifest_sha256": locked_inputs["group_sha256"][
            "target_checkpoint"
        ],
        "draft_checkpoint_manifest_sha256": locked_inputs["group_sha256"][
            "draft_checkpoint"
        ],
        "quant_input_manifest_sha256": locked_inputs["manifest_sha256"],
        "quant_linear_manifest_sha256": locked_inputs["group_sha256"][
            "quant_linear_weights"
        ],
        "quant_embedding_manifest_sha256": locked_inputs["group_sha256"][
            "quant_embedding"
        ],
        "target_dir": str(target_dir),
        "draft_dir": str(draft_dir),
        "receiver_models_dir": str(receiver_models_dir),
        "quant_config": dict(quant_identity),
        "target_quantization_audit": dict(
            getattr(target, "dflash_target_quantization_audit", {})
        ),
        "physical_target_gear": max_sequence_length,
        "valid_prefix_policy": "right-padded causal rows only",
        "gdr_effective_length_contract": (
            "INT16[B] call-local valid rows derived from attention_mask"
        ),
        "custom_op_export_contracts": [
            {
                "torch_target": item.torch_target,
                "ge_op_type": item.ge_op_type,
                "minimum_occurrences": item.minimum_occurrences,
                "preservation": (
                    "one registered GE operator; no Tensor decomposition"
                    if item.minimum_occurrences
                    else "optional path: validate metadata without requiring a graph node"
                ),
            }
            for item in custom_op_exports
        ],
        "claim_boundary": (
            "fixed-gear recompute ABI; persistent rollback OM state is a "
            "separate later optimization"
        ),
    }
    return (
        integrated_recompute_graph_spec(
            target_adapter,
            draft,
            max_sequence_length=max_sequence_length,
            example_sequence_length=example_sequence_length,
            pad_token_id=pad_token_id,
            device=device,
            name=str(config.get("name", "quant_dflash_recompute")),
            metadata=metadata,
            custom_ops=custom_op_exports,
        ),
    )


__all__ = [
    "AirDFlashOps",
    "QUANT_BASE_REVISION",
    "QUANT_GRAPH_FACTORY_ID",
    "QuantFullPrefixExportTarget",
    "create_quant_recompute_graph",
]
