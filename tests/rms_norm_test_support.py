"""CPU arithmetic/dispatcher fixture only; never receiver/device evidence."""
import pytest
import torch

_SCHEMA_LIBRARIES = []


@pytest.fixture
def adn_rms_norm_cpu():
    try:
        torch.ops.npu.adn_rms_norm.default
    except AttributeError:
        schema = torch.library.Library("npu", "FRAGMENT")
        schema.define("adn_rms_norm(Tensor input, Tensor gamma, float epsilon=1e-6) -> (Tensor, Tensor)")
        _SCHEMA_LIBRARIES.append(schema)
    if not torch._C._dispatch_has_kernel_for_dispatch_key("npu::adn_rms_norm", "Meta"):
        meta = torch.library.Library("npu", "IMPL", "Meta")
        meta.impl("adn_rms_norm", lambda input, gamma, epsilon=1e-6: (
            torch.empty_like(input),
            input.new_empty((*input.shape[:-1], 1), dtype=torch.float32),
        ))
        _SCHEMA_LIBRARIES.append(meta)
    calls = []

    def implementation(input, gamma, epsilon=1e-6):
        calls.append({"input_dtype": input.dtype, "input_shape": tuple(input.shape),
                      "gamma": gamma.detach().clone(), "epsilon": epsilon})
        value = input.float()
        rstd = torch.rsqrt(value.square().mean(dim=-1, keepdim=True) + epsilon)
        return (value * rstd * gamma.float()).to(input.dtype), rstd

    cpu = torch.library.Library("npu", "IMPL", "CPU")
    cpu.impl("adn_rms_norm", implementation)
    try:
        yield calls
    finally:
        cpu._destroy()
