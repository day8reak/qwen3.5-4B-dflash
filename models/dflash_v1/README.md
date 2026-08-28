# DFlash rollback 源码索引

当前默认路线是 persistent incremental rollback；完整历史前缀重算只作为 oracle 保留。

## 调度与运行

| 文件 | 职责 |
| --- | --- |
| run_rollback.py | CPU、CUDA、NPU 共用入口和报告生成 |
| run_npu.py | HIAI 固定参数入口 |
| benchmark_npu.py | rollback 正确性门禁后的 NPU 3 warmup + 10 次同步性能测试 |
| preflight_target_quant.py | Target-only W8A8 装配、QLinear 数值和 rollback transaction 预检 |
| dflash_rollback_decode.py | ordinary incremental、T=K+1 verify、accept 和 EOS 调度 |
| dflash_rollback_adapter.py | framework Target transaction、新增 feature 生命周期和 Draft KV cache adapter |
| dflash_reference_decode_v1.py | 旧 full-prefix sequential oracle，不是默认执行路径 |
| diagnose_acceptance.py | 固定 workload 的 proposal、verifier 和接受率诊断 |

## Target 与 Draft

| 文件 | 职责 |
| --- | --- |
| modeling_qwen3_5_dflash.py | CPU/CUDA feature-enabled Target |
| ../modeling_qwen3_5_hiai_nd_dflash_rollback.py | 独立 HIAI rollback Target |
| ../internal_dflash_bridge.py | HIAI state bank 与 logical KV cursor |
| ../export_model_wrapper_qwen3_5_dflash_rollback.py | 部署 wrapper adapter |
| modeling_dflash.py | 官方 6 层 Draft、无 cache golden 与 request-local committed KV cache |
| dflash_config.py、dflash_weights.py | 结构、69 tensor 和 checkpoint identity 门禁 |
| dflash_ops.py | CPU/CUDA Torch backend |
| dflash_ascend310p_ops.py | NPU Tensor 分解 backend |
| original_quant.py | 用户原 `utils.py` 的 key 映射、blocked-ZN 与 QLinear 替换实现 |
| target_quant.py | 原 inference YAML、量化 embedding 与完整 QLinear 拓扑审计 |
| w8a8_emulation.py、validate_w8a8_cpu.py | QLinear CPU/CUDA 公式仿真和确定性验证；不是 NPU 整网结论 |

量化路线只替换 Target Linear，并按原 YAML 索引 INT8 embedding weight 与 FP32 scale。
rollback modeling 导入原 HIAI 文件的同一个 `QLinear` 类型，因此没有第二套量化算子；Draft
共享的 embedding/LM head 仍由独立 FP16 module 提供。CLI 只需 `--config ... --quant_mode
enable`，prefill、decode 和 verify 都走内置 embedding 路径；GDR/GDR-MTP、state bank 和
logical KV 事务保持不变。

本包统一使用官方 DFlash 口径：`block_size` 是包含 clean anchor 的 Draft query/Target verify
总行数。官方配置 `block_size=16` 因此对应 1 个 anchor 加最多 15 个 proposal，即
`K=block_size-1=15`。接受率诊断仍显式记录 proposal count K，避免把 K 与 block_size 混用。

## 文档与检查

- [完整框架流程](../../docs/DFLASH_ARCHITECTURE.md)
- [与官方完整 DFlash 的差异](../../docs/DFLASH_UPSTREAM_COMPARISON.md)
- [自定义算子表](../../docs/DFLASH_OPERATORS.md)
- [运行和验证](../../docs/DFLASH_RUN_AND_VALIDATE.md)

~~~bash
python -m models.dflash_v1.run_rollback --help
python -m models.dflash_v1.run_npu --help
python -m models.dflash_v1.benchmark_npu --help
python -m models.dflash_v1.preflight_target_quant --help
python tests/test_dflash_rollback_scheduler.py
python tests/test_dflash_framework_rollback.py
python tests/test_internal_dflash_bridge_rollback.py
python tests/test_dflash_rollback_helpers.py
python -m pytest -q tests/test_run_npu_modes.py tests/test_dflash_runtime_optimizations.py
python -m pytest -q tests/test_benchmark_npu.py tests/test_msprof_script.py \
  tests/test_source_lock_benchmark.py tests/test_rollback_target_quant.py
~~~
