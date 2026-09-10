# AIR/OM/C++ 推理框架

将 Qwen3.5-4B W8A8 Target 和 FP16 DFlash 导出为静态 AIR，经 ATC 编译成 OM，
由 C++17/AscendCL 完成 ordinary greedy 或 strict-greedy DFlash 生成。

先了解模型结构、四张 OM 如何配合以及 DFlash 为什么能加速，见
[结构与运行流程](../docs/DFLASH_ARCHITECTURE.md)。

## 1. 准备环境和输入

打开 [AIR/OM/C++ 从零部署手册](../docs/GDR_CHUNK_AIR_OM.md)，从第 1 步开始准备：

- 匹配设备的 CANN、PyTorch/NPU、TorchAir、ATC、AscendCL 和自定义算子包；
- Target、官方 Draft、W8A8 Linear/embedding、外部 receiver 加载器；
- 源码之外的运行目录、固定 prompt 和输入 manifest。

## 2. 导出、转换并构建

按手册继续执行 `export-air`、`compile-om` 和 `build-cpp`。增量 factory 为：

```text
qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs
```

| 图 | 功能 |
|---|---|
| `target_prefill` | 64 行物理 gear，处理 prompt 并输出特征和状态 |
| `target_decode` | 可选的一行普通 decode；已加载时也用于关闭 Draft 后的生成 |
| `target_verify` | 16 行 verify，融合接受判断和第二次 GDR commit |
| `draft` | 64 行特征 gear、16 行 Draft block，一次生成 15 个 proposal |

DFlash 部署需要 3 个 OM；加入普通模式对照后共 4 个。C++ 按模式加载所需模型，
持久保存状态 buffer，并核对 manifest、OM hash 和有序 tensor ABI。

Target 和 Draft 的 RMSNorm 默认导出为自定义 `AdnRmsNorm`；标准 4B 的三张
Target AIR 各检查至少 105 个节点，Draft 检查 32 个。已有 factory 配置若显式
选择 `RmsNorm`，需改为 `AdnRmsNorm` 后重新导出 AIR、编译 OM，才能使用这个 GE 类型。

OM Draft 的 QK/PV 矩阵乘默认使用 FP16 输入，缩放、Mask、Softmax 保持 FP32。
`factory.json` 的 `draft_attention_matmul_dtype` 可设为 `float32` 导出对照组；
配置变更需要重新导出 AIR 和编译 OM。逐轮 proposal 和接受率允许随精度变化，
最终 token 仍按 ordinary 严格验证。Cube 路径和 FP32 累加/直接输出须在目标机确认。

## 3. 验证并采集

1. 使用 `infer-cpp` 进行普通/DFlash 各 3 次预热和 10 次测量。
2. 比较 NPU ordinary/DFlash、OM ordinary/DFlash 的 token IDs、EOS 和停止原因。
3. 使用 `prepare-chunk-plan` 生成所需模式的加载计划。
4. 使用 `tools/profile_om.py --profile-mode ordinary|dflash --profile-stage all`，
   指定运行目录、runner 和 deployment manifest，自动准备加载计划并分别采集各阶段。
   单阶段将 `all` 替换为阶段名；内部统一调用 `tools/run_msprof.sh --profile-backend cpp`。

普通模式支持 `prefill|decode|all`；DFlash 支持 `prefill|draft|verify|all`。
`all` 在一个 C++ 进程中为各阶段分别创建一次采集窗口。全部命令、参数和报告路径均在
[完整手册](../docs/GDR_CHUNK_AIR_OM.md)中。
每阶段自动生成算子类型汇总、单任务明细和慢算子排序，FP16/FP32 输入分开统计，
用于检查 AdnRmsNorm、CacheUpdate、GDR、矩阵乘耗时；Python NPU 的 wrapper 同样提供这些报告。

C++ 采集会比较各次预热和采集的输入 token、有效输出，并保存
`profile/msprof/<label>.iterations.jsonl`。verify 的填充行只记录，不参与判定；
有效输出或输入变化会返回失败并写出首个差异。记录在失败后仍保留，
路径也写入 invocation manifest；中间设备状态未逐项比较。

`infer-cpp --trace-rounds` 记录每轮 proposal、Target 验证、接受前缀和实际输出，
`--eos-token-id` 可对齐 NPU 的 EOS 策略。
`infer-cpp --low-memory` 先测试普通模式、卸载模型后再测试 DFlash，
最多同时驻留三张 OM；两组均执行 3+10，报告保留严格 token/EOS 对照及分组顺序。
`run-e2e-cpp` 和直接 C++ paired 入口同样支持。chunk 模式还会复用串行工作内存。
`python -m qwen35_dflash.ascend310p.compare_rounds` 按相同的已提交前缀比较两份报告；
最终 token 一致与每轮一致分别检查。逐轮记录用于诊断，性能基线不启用该参数。

代码已通过主机模拟检查；真实 TorchAir/ATC、自定义 GE 算子、AscendCL 和设备精度/性能
需要在目标机验证。

## 4. 查找实现和合同

| 路径 | 内容 |
|---|---|
| `python/qwen35_dflash/ascend310p/` | 图工厂、AIR 导出、ATC 编译、manifest、tokenizer 和 runner 控制面 |
| `runtime/cpp/` | AscendCL 执行、增量状态、生成调度和阶段采集 |
| `abi/dflash-chunk-v3.json` | 增量图、状态与设备验证合同 |
| `abi/performance-v1.json` | 性能计时和测量合同 |
| `scripts/lock_quant_inputs.py` | 外部输入锁定 |
| `scripts/compare_cpp_closed_runtime.py` | 相同设备、token 和计时范围的性能报告比较 |
| `FRAMEWORK_LOCK.json` | 图、量化、运行时和 profiling 合同 |

字段定义和调用边界见 [AIR/OM/C++ 接口参考](../docs/QUANT_AIR_OM_FRAMEWORK.md)。
