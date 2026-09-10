"""Preserve Draft rounding while retaining the AdnRmsNorm graph boundary.

CPU receiver arithmetic is a fixture; these checks do not validate NPU kernels.
"""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "framework/python"))
from qwen35_dflash.ascend310p.quant_factory import AirDFlashOps
from models.dflash_v1 import dflash_ascend310p_ops as native
from models.dflash_v1.dflash_ops import TorchDFlashOps
from rms_norm_test_support import adn_rms_norm_cpu  # noqa: F401


@pytest.mark.parametrize("shape", [(1, 64, 2560), (1, 16, 32, 128), (1, 64, 8, 128)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_unit_gamma_preserves_cast_before_effective_weight(adn_rms_norm_cpu, shape, dtype):
    generator = torch.Generator().manual_seed(739)
    x = torch.randn(shape, generator=generator).to(dtype)
    weight = torch.randn(shape[-1], generator=generator).to(dtype)
    weight[:2] = torch.tensor([0, -1], dtype=dtype)
    eps = 1e-5
    expected = TorchDFlashOps().rms_norm(x, weight, eps)
    result = native._adn_rms_norm(x, weight, eps)
    torch.testing.assert_close(result, expected, rtol=0, atol=0)
    torch.testing.assert_close(AirDFlashOps().rms_norm(x, weight, eps), expected, rtol=0, atol=0)
    assert len(adn_rms_norm_cpu) == 2
    for call in adn_rms_norm_cpu:
        assert call["input_dtype"] == call["gamma"].dtype == torch.float32
        assert call["input_shape"] == shape and call["epsilon"] == eps
        assert torch.equal(call["gamma"], torch.ones(shape[-1], dtype=torch.float32))
    assert not result[..., 0].count_nonzero()
    # The CPU reference does not invoke the NPU dispatcher.
    torch.testing.assert_close(native.rms_norm(x, weight, eps), expected, rtol=0, atol=0)
    assert len(adn_rms_norm_cpu) == 2
    if dtype == torch.float16:
        wrongly_fused = torch.ops.npu.adn_rms_norm.default(x.float(), weight.float(), eps)[0].half()
        assert not torch.equal(wrongly_fused, expected), "fixture must expose the shifted rounding boundary"


def test_export_retains_custom_norm_without_reduction_decomposition(adn_rms_norm_cpu):
    class Norm(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.linspace(-1, 2, 128).half())

        def forward(self, x):
            return AirDFlashOps().rms_norm(x, self.weight, 1e-6)

    exported = torch.export.export(Norm(), (torch.randn(1, 16, 32, 128).half(),), strict=True)
    exported = exported.run_decompositions({})
    nodes = list(exported.graph.nodes)
    calls = [n for n in nodes if str(n.target) == "npu.adn_rms_norm.default"]
    assert len(calls) == 1
    assert calls[0].args[0].meta["val"].dtype == torch.float32
    assert calls[0].args[1].meta["val"].dtype == torch.float32
    assert not any(str(n.target) in {"aten.rsqrt.default", "aten.mean.dim"} for n in nodes)
    output = next(n for n in nodes if n.op == "output").args[0][0]
    assert str(output.target) == "aten.mul.Tensor"
    assert all(arg.meta["val"].dtype == torch.float16 for arg in output.args)


def test_missing_receiver_is_an_error_not_a_decomposed_export(monkeypatch, adn_rms_norm_cpu):
    monkeypatch.setattr(torch.ops.npu, "adn_rms_norm", SimpleNamespace())
    with pytest.raises(RuntimeError, match="Draft requires the receiver"):
        AirDFlashOps().rms_norm(torch.ones(1, 128).half(), torch.ones(128).half(), 1e-6)
