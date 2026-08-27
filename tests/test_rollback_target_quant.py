from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from models import internal_dflash_bridge as bridge_module
from models.dflash_v1 import benchmark_npu, run_npu
from models.dflash_v1.dflash_hiai_feature_check import (
    FEATURE_SOURCE,
    ROLLBACK_FEATURE_SOURCE,
    verify_direct_source_file,
)
from models.dflash_v1.target_quant import (
    QUANT_MODE_W8A8_DYNAMIC,
    TARGET_EMBEDDING_SCALE_PATH_ENV,
    TARGET_EMBEDDING_WEIGHT_PATH_ENV,
    TARGET_INPUT_PROVIDER_ENV,
    TARGET_QUANT_MODE_ENV,
    TARGET_QUANT_WEIGHT_PATH_ENV,
    TARGET_QUANTIZER_ENV,
    TargetQuantizationResult,
)


class FakeQLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, scale: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("W_q", weight)
        self.register_buffer("scale", scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros(
            (*value.shape[:-1], int(self.W_q.shape[1])),
            device=value.device,
            dtype=torch.float16,
        )


PROVIDER_WRAPPERS: list[nn.Module] = []


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
        self.provider_owner = nn.Module()
        self.replace_calls = 0

    def replace_dflash_execution_model(self, model: nn.Module) -> None:
        if type(model) is not FakeRollbackModel:
            raise TypeError("wrong execution model type")
        self.model = model
        self.replace_calls += 1

    def dflash_target_input_provider_wrapper(self) -> nn.Module:
        return self.provider_owner


def fake_quantizer(
    model: nn.Module,
    quant_weight_path: str,
    *,
    device: torch.device,
    output_dtype: torch.dtype,
) -> TargetQuantizationResult:
    del quant_weight_path
    if type(model) is not FakeRollbackModel:
        raise TypeError("wrong model supplied to quantizer")
    original_head = model.lm_head
    model.lm_head = FakeQLinear(
        torch.zeros((2, 8), dtype=torch.int8, device=device),
        torch.ones((8,), dtype=output_dtype, device=device),
    )
    return TargetQuantizationResult(
        execution_model=model,
        expected_qlinear_paths=("lm_head",),
        profile={"test": True},
        draft_input_embeddings=model.embedding,
        draft_output_embeddings=original_head,
    )


def fake_input_provider(
    model_wrapper: nn.Module,
    input_ids: torch.Tensor,
    *,
    embedding_weight_path: str,
    embedding_scale_path: str,
    device: torch.device,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    PROVIDER_WRAPPERS.append(model_wrapper)
    del embedding_weight_path, embedding_scale_path
    return torch.zeros(
        (1, int(input_ids.shape[1]), 2),
        device=device,
        dtype=output_dtype,
    )


class RollbackTargetQuantTests(unittest.TestCase):
    def _artifacts(self, root: Path) -> dict[str, str]:
        result: dict[str, str] = {}
        for name in ("linear", "embedding", "embedding-scale"):
            path = root / f"{name}.bin"
            path.write_bytes(name.encode("utf-8"))
            result[name] = str(path)
        return result

    def test_loader_quantizes_rollback_model_and_retains_fp16_draft_views(self) -> None:
        PROVIDER_WRAPPERS.clear()
        wrapper_module_name = "tests_fake_rollback_wrapper"
        hiai_module_name = "tests_fake_rollback_hiai"
        wrapper_module = ModuleType(wrapper_module_name)
        wrapper_module.Qwen3_5ForCausalLMWrapper = FakeRollbackWrapper
        wrapper_module.Qwen3_5ForCausalLM = FakeRollbackModel
        hiai_module = ModuleType(hiai_module_name)
        hiai_module.Qwen3_5ForCausalLM = FakeRollbackModel
        hiai_module.QLinear = FakeQLinear
        callback_module = sys.modules[__name__]
        callback_prefix = callback_module.__name__

        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            environment = {
                bridge_module.KV_CACHE_MAX_LEN_ENV: "64",
                TARGET_QUANT_MODE_ENV: QUANT_MODE_W8A8_DYNAMIC,
                TARGET_QUANTIZER_ENV: f"{callback_prefix}:fake_quantizer",
                TARGET_QUANT_WEIGHT_PATH_ENV: artifacts["linear"],
                TARGET_INPUT_PROVIDER_ENV: f"{callback_prefix}:fake_input_provider",
                TARGET_EMBEDDING_WEIGHT_PATH_ENV: artifacts["embedding"],
                TARGET_EMBEDDING_SCALE_PATH_ENV: artifacts["embedding-scale"],
            }
            with patch.dict(
                sys.modules,
                {
                    wrapper_module_name: wrapper_module,
                    hiai_module_name: hiai_module,
                },
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
        self.assertEqual(tuple(inputs.shape), (1, 2, 2))
        self.assertEqual(PROVIDER_WRAPPERS, [target.model_wrapper.provider_owner])
        self.assertEqual(target.model_wrapper.replace_calls, 1)
        self.assertIsInstance(target.dflash_execution_model.lm_head, FakeQLinear)
        self.assertIsInstance(target.get_output_embeddings(), nn.Linear)
        audit = target.dflash_target_quantization_audit
        self.assertEqual(audit["scheme"], QUANT_MODE_W8A8_DYNAMIC)
        self.assertEqual(audit["route"], "rollback")
        self.assertEqual(audit["qlinear_paths"], ["lm_head"])
        self.assertEqual(audit["linear_topology_validation"], "PASS_EXACT_PATH_SHAPE_BIAS")

    def test_run_npu_quant_configuration_is_fail_closed(self) -> None:
        callback_prefix = sys.modules[__name__].__name__
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = self._artifacts(Path(temporary))
            args = argparse.Namespace(
                target_quant_mode=QUANT_MODE_W8A8_DYNAMIC,
                target_factory=run_npu.DEFAULT_TARGET_FACTORY,
                target_quantizer=f"{callback_prefix}:fake_quantizer",
                target_quant_weight_path=artifacts["linear"],
                target_input_provider=f"{callback_prefix}:fake_input_provider",
                target_embedding_weight_path=artifacts["embedding"],
                target_embedding_scale_path=artifacts["embedding-scale"],
                report=str(Path(temporary) / "report.json"),
            )
            with patch.dict(os.environ, {}, clear=True):
                run_npu._configure_target_quantization(args)
                self.assertEqual(
                    os.environ[TARGET_QUANT_MODE_ENV],
                    QUANT_MODE_W8A8_DYNAMIC,
                )
                args.target_factory = "custom:factory"
                with self.assertRaisesRegex(ValueError, "packaged factory"):
                    run_npu._configure_target_quantization(args)

    def test_benchmark_parser_accepts_same_quant_contract(self) -> None:
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
                "--target-quant-mode",
                "w8a8_dynamic",
                "--target-quantizer",
                "callbacks:quantize",
                "--target-quant-weight-path",
                "/quant",
                "--target-input-provider",
                "callbacks:inputs",
                "--target-embedding-weight-path",
                "/embedding",
                "--target-embedding-scale-path",
                "/embedding-scale",
            ]
        )
        self.assertEqual(args.target_quant_mode, QUANT_MODE_W8A8_DYNAMIC)
        self.assertEqual(args.target_quantizer, "callbacks:quantize")

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
