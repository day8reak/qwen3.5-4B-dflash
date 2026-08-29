# DFlash rollback 源码索引

默认路线是 persistent incremental rollback；完整历史前缀重算只保留为诊断 oracle。

## 入口与调度

| 文件 | 职责 |
| --- | --- |
| `run_rollback.py` | CPU/CUDA/NPU 共用入口、validate/dflash 模式和报告 |
| `run_npu.py` | HIAI 固定参数入口；默认 FP16，可选 Target W8A8 |
| `benchmark_npu.py` | 正确性门禁后的 independent-process NPU benchmark |
| `dflash_rollback_decode.py` | ordinary incremental、Draft/verify、longest-prefix accept、EOS |
| `dflash_rollback_adapter.py` | framework transaction、feature 生命周期和 Draft KV adapter |
| `diagnose_acceptance.py` | 固定 workload 的 proposal/Target/接受率诊断 |
| `dflash_reference_decode_v1.py` | 旧 full-prefix oracle，不是默认 DFlash |

## Draft

| 文件 | 职责 |
| --- | --- |
| `modeling_dflash.py` | 官方 6 层 Draft 和 request-local committed/transient KV cache |
| `dflash_config.py` | `block_size`、6 层/69 tensor 和官方 shape 合同 |
| `dflash_weights.py` | checkpoint revision/hash/tensor 审计与流式加载 |
| `dflash_ops.py` | CPU/CUDA Torch primitive interface |
| `dflash_ascend310p_ops.py` | NPU 上无 CPU fallback 的 Tensor 分解 backend |

`block_size` 包含 anchor：B=16 对应 K=15 proposals、T=16 Target rows。Draft cache 每轮让
attention 看见 old committed + new committed + transient block，返回前只保留 committed 部分。

## Target rollback

| 文件 | 职责 |
| --- | --- |
| `modeling_qwen3_5_dflash.py` | CPU/CUDA feature-enabled Target |
| `../modeling_qwen3_5_hiai_nd_dflash_rollback.py` | 独立 HIAI rollback Target modeling |
| `../internal_dflash_bridge.py` | HIAI state bank、prompt chunk 和 paged-KV logical cursor |
| `../export_model_wrapper_qwen3_5_dflash_rollback.py` | 原部署 wrapper 的 rollback adapter |
| `dflash_target_features.py` | 八层 Target feature 合同 |

原 `../modeling_qwen3_5_hiai_nd.py` 保持不变。Prompt 多 token 走原 GDR chunk，verify 才走
`npu_gated_delta_rule_mtp`。Causal-conv 当前是输入 NPU 上的 Tensor golden；完整算子计划见文档。

## Target W8A8

| 文件 | 职责 |
| --- | --- |
| `original_quant.py` | 原 `utils.py` 的 key 映射、blocked-ZN 与 QLinear 替换 |
| `target_quant.py` | YAML、INT8 embedding/scale、完整 QLinear topology 审计 |
| `preflight_target_quant.py` | 不加载 Draft 的装配、公式和 transaction 预检 |
| `w8a8_emulation.py` / `validate_w8a8_cpu.py` | CPU/CUDA 公式诊断，不是 NPU 整网结论 |

量化默认关闭。启用只需：

```bash
--config ./config/qwen3.5.yaml --quant_mode enable
```

量化只替换 Target Linear 和 Target 输入 embedding；Draft-facing embedding、LM head 和 6 层主体
保持 FP16。关闭量化时省略参数，或传 `--quant_mode disable`。

## 文档与检查

- [当前架构与完整 DFlash 差异](../../docs/DFLASH_ARCHITECTURE.md)
- [自定义算子及 I/O](../../docs/DFLASH_OPERATORS.md)
- [运行、benchmark 与报告门禁](../../docs/DFLASH_RUN_AND_VALIDATE.md)

```bash
python -m models.dflash_v1.run_rollback --help
python -m models.dflash_v1.run_npu --help
python -m models.dflash_v1.benchmark_npu --help
python -m models.dflash_v1.preflight_target_quant --help
python -m pytest -q tests/test_dflash_rollback_scheduler.py \
  tests/test_dflash_runtime_optimizations.py tests/test_rollback_target_quant.py
```
