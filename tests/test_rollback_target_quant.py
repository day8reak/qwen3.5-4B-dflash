from __future__ import annotations

import argparse
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models import internal_dflash_bridge as bridge_module
from models.dflash_v1 import benchmark_npu, original_quant, run_npu
from models.dflash_v1.dflash_hiai_feature_check import (
    FEATURE_SOURCE,
    ROLLBACK_FEATURE_SOURCE,
    verify_direct_source_file,
)
from models.dflash_v1.target_quant import (
    QUANT_MODE_W8A8_DYNAMIC,
    TARGET_EMBEDDING_SCALE_PATH_ENV,
    TARGET_EMBEDDING_WEIGHT_PATH_ENV,
    TARGET_QUANT_CONFIG_ENV,
    TARGET_QUANT_MODE_ENV,
    TARGET_QUANT_WEIGHT_PATH_ENV,
)


class FakeQLinear(nn.Module):
    def __init__(
        self,
        W_q: torch.Tensor,
        scale: torch.Tensor,
        idx: int,
    ) -> None:
        super().__init__()
        self.register_buffer("W_q", W_q)
        self.register_buffer("scale", scale)
        self.idx = idx

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*value.shape[:-1], int(self.W_q.shape[1])),
            device=value.device,
            dtype=torch.float16,
        )


class FakeRollbackModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            layer_types=[],
            num_hidden_layers=0,
            vocab_size=8,
            hidden_size=2,
            kv_cache_max_len=64,
        )
        self.embedding = nn.Embedding(8, 2, dtype=torch.float16)
        self.lm_head = nn.Linear(2, 8, bias=False, dtype=torch.float16)

    def get_input_embeddings(self) -> nn.Module:
        return self.embedding

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head


class FakeRollbackWrapper(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        del args, kwargs
        self.model = FakeRollbackModel().eval()
        self.replace_calls = 0

    def replace_dflash_execution_model(self, model: nn.Module) -> None:
        if type(model) is not FakeRollbackModel:
            raise TypeError("wrong execution model type")
        self.model = model
        self.replace_calls += 1


def fake_quant_model(model: nn.Module, quant_weight_path: str) -> nn.Module:
    if type(model) is not FakeRollbackModel:
        raise TypeError("wrong model supplied to quantizer")
    if not Path(quant_weight_path).is_dir():
        raise ValueError("wrong quant artifact path")
    model.lm_head = FakeQLinear(
        W_q=torch.zeros((2, 8), dtype=torch.int8),
        scale=torch.ones((8,), dtype=torch.float32),
        idx=15,
    )
    return model


class RollbackTargetQuantTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> dict[str, str]:
        quant = root / "linear"
        quant.mkdir()
        weight = root / "embedding_weight.bin"
        weight.write_bytes(bytes(range(16)))
        scale = root / "embedding_scale.bin"
        scale.write_bytes(struct.pack("<8f", *([0.5] * 8)))
        config = root / "qwen3.5.yaml"
        config.write_text(
            "\n".join(
                (
                    f"quanted_pth: {quant}",
                    f"embedding_weight_path: {weight}",
                    f"embedding_scale_path: {scale}",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "config": str(config),
            "linear": str(quant),
            "embedding": str(weight),
            "embedding-scale": str(scale),
        }

    def test_loader_uses_original_yaml_and_builtin_embedding_route(self) -> None:
        wrapper_module_name = "tests_fake_rollback_wrapper"
        hiai_module_name = "tests_fake_rollback_hiai"
        wrapper_module = ModuleType(wrapper_module_name)
        wrapper_module.Qwen3_5ForCausalLMWrapper = FakeRollbackWrapper
        wrapper_module.Qwen3_5ForCausalLM = FakeRollbackModel
        hiai_module = ModuleType(hiai_module_name)
        hiai_module.Qwen3_5ForCausalLM = FakeRollbackModel
        hiai_module.QLinear = FakeQLinear

        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            environment = {
                bridge_module.KV_CACHE_MAX_LEN_ENV: "64",
                TARGET_QUANT_MODE_ENV: QUANT_MODE_W8A8_DYNAMIC,
                TARGET_QUANT_CONFIG_ENV: artifacts["config"],
                TARGET_QUANT_WEIGHT_PATH_ENV: artifacts["linear"],
                TARGET_EMBEDDING_WEIGHT_PATH_ENV: artifacts["embedding"],
                TARGET_EMBEDDING_SCALE_PATH_ENV: artifacts["embedding-scale"],
            }
            with patch.dict(
                sys.modules,
                {
                    wrapper_module_name: wrapper_module,
                    hiai_module_name: hiai_module,
                },
            ), patch.object(
                original_quant,
                "quant_model",
                fake_quant_model,
            ), patch.dict(os.environ, environment, clear=True):
                target = bridge_module._load_qwen35_target_impl(
                    temporary,
                    device=torch.device("cpu"),
                    dtype=torch.float16,
                    wrapper_module_name=wrapper_module_name,
                    hiai_module_name=hiai_module_name,
                    rollback_enabled=True,
                )
                inputs = target._target_inputs(
                    torch.tensor([[1, 2]], dtype=torch.long)
                )

        self.assertTrue(target.rollback_enabled)
        torch.testing.assert_close(
            inputs,
            torch.tensor([[[1.0, 1.5], [2.0, 2.5]]], dtype=torch.float16),
            rtol=0,
            atol=0,
        )
        self.assertEqual(target.model_wrapper.replace_calls, 1)
        self.assertIsInstance(target.dflash_execution_model.lm_head, FakeQLinear)
        self.assertIsInstance(target.get_output_embeddings(), nn.Linear)
        audit = target.dflash_target_quantization_audit
        self.assertEqual(audit["scheme"], QUANT_MODE_W8A8_DYNAMIC)
        self.assertEqual(audit["route"], "rollback")
        self.assertEqual(audit["qlinear_paths"], ["lm_head"])
        self.assertEqual(
            audit["linear_topology_validation"],
            "PASS_EXACT_PATH_SHAPE_BIAS",
        )
        self.assertEqual(audit["embedding_lookup_calls"], 1)
        self.assertEqual(audit["embedding_lookup_failures"], 0)

    def test_builtin_converter_keeps_original_key_and_zn_contract(self) -> None:
        model = nn.Module()
        model.add_module("language_model", nn.Module())
        model.language_model.add_module(
            "proj",
            nn.Linear(32, 16, bias=False),
        )
        state = {
            "model.proj_quant_weight": torch.arange(
                16 * 32,
                dtype=torch.int16,
            ).to(torch.int8).reshape(16, 32),
            "model.proj_quant_scale": torch.ones(16, dtype=torch.float32),
        }
        with patch.object(
            original_quant,
            "_qlinear_class",
            return_value=FakeQLinear,
        ):
            converted = original_quant.replace_linear_to_qlinear(model, state)
        self.assertIs(converted, model)
        self.assertIsInstance(model.language_model.proj, FakeQLinear)
        self.assertEqual(tuple(model.language_model.proj.W_q.shape), (32, 16))
        self.assertEqual(model.language_model.proj.idx, 15)
        self.assertEqual(state, {})

    def test_run_npu_quant_configuration_uses_only_original_config(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            args = argparse.Namespace(
                quant_mode=run_npu.ORIGINAL_QUANT_ENABLE,
                target_factory=run_npu.DEFAULT_TARGET_FACTORY,
                config=artifacts["config"],
                report=str(Path(temporary) / "report.json"),
            )
            with patch.dict(os.environ, {}, clear=True):
                run_npu._configure_target_quantization(args)
                self.assertEqual(
                    os.environ[TARGET_QUANT_MODE_ENV],
                    QUANT_MODE_W8A8_DYNAMIC,
                )
                self.assertEqual(
                    os.environ[TARGET_QUANT_CONFIG_ENV],
                    str(Path(artifacts["config"]).resolve()),
                )
                args.target_factory = "custom:factory"
                with self.assertRaisesRegex(ValueError, "packaged factory"):
                    run_npu._configure_target_quantization(args)

    def test_benchmark_parser_accepts_original_inference_switches(self) -> None:
        args = benchmark_npu._parser().parse_args(
            [
                "--mode",
                "dflash",
                "--target-dir",
                "/target",
                "--draft-dir",
                "/draft",
                "--prompt-ids",
                "1,2",
                "--max-new-tokens",
                "32",
                "--kv-cache-max-len",
                "2048",
                "--report",
                "/tmp/report.json",
                "--config",
                "/data/qwen3.5.yaml",
                "--quant_mode",
                "enable",
            ]
        )
        self.assertEqual(args.quant_mode, run_npu.ORIGINAL_QUANT_ENABLE)
        self.assertEqual(args.config, "/data/qwen3.5.yaml")

    def test_feature_source_checker_accepts_original_and_rollback_files(self) -> None:
        repository = Path(__file__).resolve().parents[1]
        original = verify_direct_source_file(
            repository / "models" / "modeling_qwen3_5_hiai_nd.py"
        )
        rollback = verify_direct_source_file(
            repository
            / "models"
            / "modeling_qwen3_5_hiai_nd_dflash_rollback.py"
        )
        self.assertEqual(original["status"], "PASS_DIRECT_SOURCE_CONTRACT")
        self.assertEqual(original["feature_source"], FEATURE_SOURCE)
        self.assertEqual(rollback["status"], "PASS_DIRECT_SOURCE_CONTRACT")
        self.assertEqual(rollback["feature_source"], ROLLBACK_FEATURE_SOURCE)


if __name__ == "__main__":
    unittest.main()
