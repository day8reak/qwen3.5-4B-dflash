from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
FRAMEWORK_PYTHON = ROOT / "framework" / "python"
if str(FRAMEWORK_PYTHON) not in sys.path:
    sys.path.insert(0, str(FRAMEWORK_PYTHON))

from qwen35_dflash.ascend310p.compiler import compile_air_bundle
from qwen35_dflash.ascend310p.input_manifest import (
    build_quant_input_manifest,
    verify_quant_input_manifest,
)
from qwen35_dflash.ascend310p.quant_factory import (
    AirDFlashOps,
    QUANT_BASE_REVISION,
    QuantFullPrefixExportTarget,
    create_quant_recompute_graph,
)
from qwen35_dflash.ascend310p.utils import sha256_file
from qwen35_dflash.ascend310p.workflow import DEFAULT_GRAPH_FACTORY
from models.dflash_v1 import dflash_ascend310p_ops as golden_ops


class _FakeExecutionModel(nn.Module):
    def __init__(self, embedding: nn.Module, lm_head: nn.Module) -> None:
        super().__init__()
        # Keep references outside Module registration: the owning fake target
        # already registers both weights.
        object.__setattr__(self, "embedding", embedding)
        object.__setattr__(self, "lm_head", lm_head)
        self.last_gdr_effective_length: torch.Tensor | None = None

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        inputs_embeds: torch.Tensor,
        output_dflash_features: bool,
        gdr_effective_length: torch.Tensor,
        **kwargs: object,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del input_ids, kwargs
        assert output_dflash_features is True
        self.last_gdr_effective_length = gdr_effective_length.detach().clone()
        return self.lm_head(inputs_embeds), torch.cat(
            (inputs_embeds, inputs_embeds), dim=-1
        )


class _FakeQuantTarget(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(32, 4)
        self.lm_head = nn.Linear(4, 32, bias=False)
        self.execution = _FakeExecutionModel(self.embedding, self.lm_head)
        self.public_forward_calls = 0
        self._target_quantized_embedding = self.embedding
        self.dflash_target_quantization_audit = {
            "status": "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM",
            "scheme": "w8a8_dynamic",
        }

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    @property
    def dflash_execution_model(self) -> nn.Module:
        return self.execution

    def _fresh_attention_mask(self, sequence_length: int) -> torch.Tensor:
        return torch.zeros(1, 1, sequence_length, sequence_length)

    def _fresh_hybrid_cache(
        self, *, batch_size: int
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        return [
            (
                torch.zeros(batch_size, 1, 1),
                torch.zeros(batch_size, 1, 1, 1),
            )
        ]

    def forward(self, input_ids: torch.Tensor, **kwargs: object):
        del input_ids, kwargs
        self.public_forward_calls += 1
        raise AssertionError("AIR adapter must bypass stateful public forward")


class _FakeDraft(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            block_size=16,
            mask_token_id=31,
            vocab_size=32,
            hidden_size=4,
        )

    def embed_block(
        self, block_ids: torch.Tensor, embedding_weight: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.embedding(block_ids, embedding_weight)

    def draft_top1(
        self,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        lm_head_weight: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del target_hidden, position_ids, lm_head_weight, attention_mask
        return torch.arange(
            1,
            noise_embedding.shape[1],
            dtype=torch.long,
            device=noise_embedding.device,
        ).unsqueeze(0)


def test_default_factory_is_quant_branch_factory() -> None:
    assert DEFAULT_GRAPH_FACTORY.endswith("quant_factory:create_quant_recompute_graph")
    assert QUANT_BASE_REVISION == "28f93e784a2beed87020a80bd93c8788754eab1c"


def test_quant_graph_rejects_non_chunk_aligned_gear_before_weight_load() -> None:
    with pytest.raises(ValueError, match="divisible"):
        create_quant_recompute_graph({"max_sequence_length": 65})


def test_quant_graph_rejects_gear_outside_gdr_int16_abi() -> None:
    with pytest.raises(ValueError, match="INT16"):
        create_quant_recompute_graph({"max_sequence_length": 32768})


def test_quant_target_adapter_bypasses_eager_guards_and_sync() -> None:
    target = _FakeQuantTarget().eval()
    adapter = QuantFullPrefixExportTarget(target).eval()
    input_ids = torch.tensor([[1, 2, 0, 0]], dtype=torch.long)
    mask = torch.tensor([[1, 1, 0, 0]], dtype=torch.long)
    logits, features = adapter(
        input_ids,
        mask,
        use_cache=False,
        return_dict=True,
        output_dflash_features=True,
    )
    assert logits.shape == (1, 4, 32)
    assert features.shape == (1, 4, 8)
    assert target.public_forward_calls == 0
    assert target.execution.last_gdr_effective_length is not None
    assert target.execution.last_gdr_effective_length.dtype == torch.int16
    assert target.execution.last_gdr_effective_length.tolist() == [2]


def test_air_ops_match_quant_branch_decomposed_golden() -> None:
    torch.manual_seed(7)
    ops = AirDFlashOps()
    value = torch.randn(1, 3, 8, dtype=torch.float32)
    weight = torch.randn(8, dtype=torch.float32)
    torch.testing.assert_close(
        ops.rms_norm(value, weight, 1e-6),
        golden_ops.rms_norm(value, weight, 1e-6),
        rtol=0,
        atol=0,
    )
    linear_weight = torch.randn(5, 8, dtype=torch.float32)
    torch.testing.assert_close(
        ops.linear(value, linear_weight),
        golden_ops.linear(value, linear_weight),
        rtol=0,
        atol=0,
    )
    assert torch.equal(
        ops.top1(value, linear_weight),
        golden_ops.top1(value, linear_weight),
    )

    query = torch.randn(1, 4, 2, 8, dtype=torch.float32)
    key = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    val = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    visible = torch.ones(1, 1, 2, 5, dtype=torch.bool)
    visible[..., 0, 4] = False
    torch.testing.assert_close(
        ops.attention(query, key, val, visible, 0.125, 2),
        golden_ops.attention(query, key, val, visible, 0.125, 2),
        rtol=0,
        atol=0,
    )


def test_padded_draft_context_matches_compact_quant_draft() -> None:
    from qwen35_dflash.ascend310p.integrated import enable_padded_draft_context
    from models.dflash_v1.dflash_config import Qwen35DFlashConfig
    from models.dflash_v1.modeling_dflash import DFlashDraftModel

    config = Qwen35DFlashConfig(
        hidden_size=8,
        intermediate_size=16,
        vocab_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=4,
        num_target_layers=2,
        target_layer_ids=(0,),
        layer_types=("sliding_attention", "full_attention"),
        block_size=4,
        mask_token_id=31,
        rms_norm_eps=1e-6,
        rope_theta=10_000.0,
        max_position_embeddings=32,
        sliding_window=4,
        use_sliding_window=True,
        attention_bias=False,
        attention_dropout=0.0,
        hidden_act="silu",
        dtype="float32",
    )
    torch.manual_seed(19)
    compact = DFlashDraftModel(
        config,
        ops=AirDFlashOps(),
        device="cpu",
        dtype=torch.float32,
    ).eval()
    with torch.no_grad():
        for parameter in compact.parameters():
            parameter.normal_(mean=0.0, std=0.02)
    padded = copy.deepcopy(compact).eval()
    enable_padded_draft_context(padded)

    context = torch.randn(1, 3, config.hidden_size)
    padding = torch.randn(1, 3, config.hidden_size)
    block = torch.randn(1, config.block_size, config.hidden_size)
    compact_positions = torch.arange(7, dtype=torch.long).unsqueeze(0)
    padded_positions = torch.tensor(
        [[0, 1, 2, 0, 0, 0, 3, 4, 5, 6]], dtype=torch.long
    )
    context_valid = torch.tensor(
        [[True, True, True, False, False, False]], dtype=torch.bool
    )
    expected = compact.forward_projected(
        context,
        block,
        compact_positions,
    )
    actual = padded.forward_projected(
        torch.cat((context, padding), dim=1),
        block,
        padded_positions,
        context_valid,
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    exported = torch.export.export(
        padded,
        (
            torch.randn(1, 6, config.feature_size),
            block,
            padded_positions,
            context_valid,
        ),
        strict=True,
    )
    assert len(list(exported.graph.nodes)) > 100


def test_compile_uses_air_framework_and_hash_locks_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    bundle = run_dir / "bundle"
    graph_dir = bundle / "air" / "quant_dflash_recompute"
    graph_dir.mkdir(parents=True)
    air = graph_dir / "quant_dflash_recompute.air"
    air.write_bytes(b"quant-air")
    manifest = {
        "schema_version": 1,
        "artifact_kind": "qwen35-dflash-torchair-bundle",
        "status": "PASS",
        "graphs": [
            {
                "name": "quant_dflash_recompute",
                "role": "generation-recompute",
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["target_top1", "draft_top1"],
                "air": {
                    "path": "air/quant_dflash_recompute/quant_dflash_recompute.air",
                    "bytes": air.stat().st_size,
                    "sha256": sha256_file(air),
                },
                "payload_files": [
                    {
                        "path": "air/quant_dflash_recompute/quant_dflash_recompute.air",
                        "bytes": air.stat().st_size,
                        "sha256": sha256_file(air),
                    }
                ],
            }
        ],
    }
    manifest_path = bundle / "air-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    atc = tmp_path / "atc"
    atc.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    atc.chmod(0o755)
    commands: list[list[str]] = []

    def runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        commands.append(list(command))
        output = next(item.split("=", 1)[1] for item in command if item.startswith("--output="))
        Path(output + ".om").write_bytes(b"quant-om")
        return subprocess.CompletedProcess(command, 0, stdout="ok")

    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    result = compile_air_bundle(
        manifest_path,
        soc_version="Ascend310P3",
        atc_bin=atc,
        runner=runner,
        atc_identity="fake-atc",
    )
    assert result["status"] == "PASS"
    assert commands[0][1:3] == ["--mode=0", "--framework=1"]
    assert commands[0][-1] == "--soc_version=Ascend310P3"


def test_quant_input_manifest_hashes_and_rechecks_external_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    quant = tmp_path / "quant"
    receiver = tmp_path / "receiver" / "models"
    for directory in (target, draft, quant, receiver):
        directory.mkdir(parents=True)
    (target / "config.json").write_text("{}", encoding="utf-8")
    (target / "model.safetensors").write_bytes(b"target")
    (draft / "config.json").write_text("{}", encoding="utf-8")
    (draft / "model.safetensors").write_bytes(b"draft")
    (quant / "data-00001.safetensors").write_bytes(b"quant")
    embedding_weight = tmp_path / "embedding.bin"
    embedding_scale = tmp_path / "embedding-scale.bin"
    embedding_weight.write_bytes(b"weight")
    embedding_scale.write_bytes(b"scale")
    (receiver / "export_model_wrapper_qwen3_5.py").write_text(
        "# receiver\n", encoding="utf-8"
    )
    quant_config = tmp_path / "quant.yaml"
    quant_config.write_text(
        "\n".join(
            (
                f"quanted_pth: {quant}",
                f"embedding_weight_path: {embedding_weight}",
                f"embedding_scale_path: {embedding_scale}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    output = run_dir / "quant-input-manifest.json"
    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    built = build_quant_input_manifest(
        target_dir=target,
        draft_dir=draft,
        quant_config=quant_config,
        receiver_models_dir=receiver,
        output=output,
    )
    verified = verify_quant_input_manifest(output)
    assert verified["manifest_sha256"] == built["manifest_sha256"]
    assert verified["roots"]["target_checkpoint"] == target.resolve()
    (target / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="payload changed"):
        verify_quant_input_manifest(output)


def test_quant_factory_builds_graph_from_quant_branch_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    target_dir = tmp_path / "target"
    draft_dir = tmp_path / "draft"
    quant_dir = tmp_path / "quant"
    receiver = tmp_path / "receiver" / "models"
    for directory in (target_dir, draft_dir, quant_dir, receiver):
        directory.mkdir(parents=True)
    (target_dir / "config.json").write_text("{}", encoding="utf-8")
    (draft_dir / "config.json").write_text("{}", encoding="utf-8")
    (quant_dir / "data.safetensors").write_bytes(b"q")
    embedding_weight = tmp_path / "embedding.bin"
    embedding_scale = tmp_path / "embedding-scale.bin"
    embedding_weight.write_bytes(b"w")
    embedding_scale.write_bytes(b"s")
    (receiver / "export_model_wrapper_qwen3_5.py").write_text(
        "# receiver\n", encoding="utf-8"
    )
    quant_config = tmp_path / "quant.yaml"
    quant_config.write_text(
        f"quanted_pth: {quant_dir}\n"
        f"embedding_weight_path: {embedding_weight}\n"
        f"embedding_scale_path: {embedding_scale}\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AI_RUN_DIR", str(run_dir))
    manifest = run_dir / "inputs.json"
    build_quant_input_manifest(
        target_dir=target_dir,
        draft_dir=draft_dir,
        quant_config=quant_config,
        receiver_models_dir=receiver,
        output=manifest,
    )

    import models
    from qwen35_dflash.ascend310p import quant_factory
    from qwen35_dflash.ascend310p.contracts import AirGraphSpec
    from qwen35_dflash.ascend310p.integrated import IntegratedDFlashRecomputeGraph
    from models.dflash_v1 import modeling_dflash
    from models import internal_dflash_bridge

    monkeypatch.setattr(models, "__path__", list(models.__path__))
    fake_target = _FakeQuantTarget().eval()
    fake_draft = _FakeDraft().eval()
    monkeypatch.setattr(
        internal_dflash_bridge,
        "load_qwen35_target",
        lambda *args, **kwargs: fake_target,
    )
    monkeypatch.setattr(
        modeling_dflash.DFlashDraftModel,
        "from_pretrained",
        lambda *args, **kwargs: fake_draft,
    )
    def cpu_graph_spec(
        target_model: nn.Module,
        draft_model: nn.Module,
        **kwargs: object,
    ) -> AirGraphSpec:
        graph = IntegratedDFlashRecomputeGraph(target_model, draft_model).eval()
        length = int(kwargs["max_sequence_length"])
        ids = torch.zeros((1, length), dtype=torch.long)
        mask = torch.zeros_like(ids)
        mask[:, : int(kwargs["example_sequence_length"])] = 1
        return AirGraphSpec(
            name=str(kwargs["name"]),
            role="generation-recompute",
            model=graph,
            example_args=(ids, mask),
            input_names=("input_ids", "attention_mask"),
            output_names=("target_top1", "draft_top1"),
            metadata=dict(kwargs["metadata"]),
        )

    monkeypatch.setattr(
        quant_factory,
        "integrated_recompute_graph_spec",
        cpu_graph_spec,
    )
    monkeypatch.setattr(
        quant_factory,
        "enable_padded_draft_context",
        lambda model: model,
    )
    real_torch_device = torch.device
    monkeypatch.setattr(
        quant_factory.torch,
        "device",
        lambda value: (
            real_torch_device("cpu")
            if str(value).startswith("npu")
            else real_torch_device(value)
        ),
    )
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    specs = create_quant_recompute_graph(
        {
            "target_dir": str(target_dir),
            "draft_dir": str(draft_dir),
            "quant_config": str(quant_config),
            "input_manifest": str(manifest),
            "receiver_models_dir": str(receiver),
            "max_sequence_length": 64,
            "example_sequence_length": 2,
            "dtype": "float16",
            "device": "npu:0",
        }
    )
    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "quant_dflash_recompute"
    assert spec.metadata["target_quant_mode"] == "w8a8_dynamic"
    assert spec.metadata["gdr_effective_length_contract"] == (
        "INT16[B] call-local valid rows derived from attention_mask"
    )
    assert spec.metadata["quant_source_lock"]["verified_file_count"] >= 10
    assert spec.metadata["target_checkpoint_manifest_sha256"]
    assert spec.input_names == ("input_ids", "attention_mask")
    assert spec.output_names == ("target_top1", "draft_top1")


def test_cpp_runtime_is_bundled_under_quant_branch() -> None:
    source = ROOT / "framework" / "runtime" / "cpp"
    assert (source / "CMakeLists.txt").is_file()
    assert (source / "src" / "acl_executor.cpp").is_file()
    text = (source / "src" / "acl_executor.cpp").read_text(encoding="utf-8")
    assert "aclmdlExecuteAsync" in text
    assert "aclrtSynchronizeStream" in text
