# DFlash rollback 源码索引

当前默认路线是 persistent incremental rollback；完整历史前缀重算只保留为 oracle。

## 运行与调度

- `run_rollback.py`：CPU/CUDA/NPU 统一入口；
- `run_npu.py`：HIAI rollback 简化入口；
- `dflash_rollback_decode.py`：T=K+1 verify、最长连续接受、correction/bonus 调度；
- `dflash_rollback_adapter.py`：CPU/CUDA `DynamicCache` 事务和 Qwen3.5 Draft 接线；
- `dflash_reference_decode_v1.py`：旧 full-prefix sequential oracle；
- `dflash_qwen_adapter_v1.py`：旧 oracle 的完整 adapter/CLI 和共享加载、审计工具。

兼容命令 `python -m models.dflash_qwen_adapter_v1` 默认进入 rollback；若需要显式运行旧 oracle，
使用 `python -m models.dflash_v1.dflash_qwen_adapter_v1`。

## Target

- `modeling_qwen3_5_dflash.py`：CPU/CUDA feature-enabled framework Target；
- `../modeling_qwen3_5_hiai_nd.py`：保留不变的普通/full-prefix HIAI 文件；
- `../modeling_qwen3_5_hiai_nd_dflash_rollback.py`：独立 HIAI state-bank 文件；
- `../internal_dflash_bridge.py`：persistent HIAI transaction、GDN bank 和 logical KV cursor；
- `../export_model_wrapper_qwen3_5_dflash_rollback.py`：复用部署 wrapper 加载逻辑并绑定 rollback
  modeling。

## Draft 与算子 backend

- `modeling_dflash.py`：官方六层 Draft；
- `dflash_config.py` / `dflash_weights.py`：锁定结构、69 tensors 和 checkpoint hash；
- `dflash_ops.py`：Torch CPU/CUDA backend；
- `dflash_ascend310p_ops.py`：NPU Tensor 分解 backend；
- `dflash_custom_ops_template.py`：未来 fused/custom Draft op 接线模板。

统一使用 proposal-count K：anchor 不计入 K，Target verify T=`K+1`，官方最大 K=16/T=17。

## 文档

- [Scheduler 与 token 验证](../../docs/DFLASH_V1_SCHEDULER.md)
- [验证流程与报告](../../docs/DFLASH_V1_VALIDATION.md)
- [rollback 自定义算子分析](../../docs/DFLASH_ROLLBACK_OPERATOR_ANALYSIS.md)
- [NPU 部署](../../docs/NPU_DEPLOYMENT.md)

## 快速检查

```bash
python -m models.dflash_v1.run_rollback --help
python -m models.dflash_v1.run_npu --help
python tests/test_dflash_rollback_scheduler.py
python tests/test_dflash_framework_rollback.py
python tests/test_internal_dflash_bridge_rollback.py
python tests/test_dflash_rollback_helpers.py
```
