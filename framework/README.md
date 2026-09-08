# AIR/OM/C++ 推理框架

将 Qwen3.5-4B W8A8 Target 和 FP16 DFlash 导出为静态 AIR，经 ATC 编译成 OM，
由 C++17/AscendCL 完成 ordinary greedy 或 strict-greedy DFlash 生成。

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
| `target_decode` | 可选的一行普通 decode |
| `target_verify` | 16 行 verify，融合接受判断和第二次 GDR commit |
| `draft` | 64 行特征 gear、16 行 Draft block，一次生成 15 个 proposal |

DFlash 部署需要 3 个 OM；加入普通模式对照后共 4 个。C++ 按模式加载所需模型，
持久保存状态 buffer，并核对 manifest、OM hash 和有序 tensor ABI。

## 3. 验证并采集

1. 使用 `infer-cpp` 进行普通/DFlash 各 3 次预热和 10 次测量。
2. 比较 NPU ordinary、OM ordinary、OM DFlash 的 token IDs、EOS 和停止原因。
3. 使用 `prepare-chunk-plan` 生成所需模式的加载计划。
4. 使用 `tools/run_msprof.sh --profile-backend cpp` 采集单个阶段或 `all`。

普通模式支持 `prefill|decode|all`；DFlash 支持 `prefill|draft|verify|all`。
`all` 在一个 C++ 进程中为各阶段分别创建一次采集窗口。全部命令、参数和报告路径均在
[完整手册](../docs/GDR_CHUNK_AIR_OM.md)中。

代码已通过主机模拟检查；真实 TorchAir/ATC、自定义 GE 算子、AscendCL 和设备精度/性能
需要在目标机验证。

## 4. 查找实现和合同

| 路径 | 内容 |
|---|---|
| `python/qwen35_dflash/ascend310p/` | 图工厂、AIR 导出、ATC 编译、manifest、tokenizer 和 runner 控制面 |
| `runtime/cpp/` | AscendCL 执行、增量状态、生成调度和阶段采集 |
| `abi/dflash-chunk-v1.json` | 增量图、状态与设备验证合同 |
| `abi/performance-v1.json` | 性能计时和测量合同 |
| `scripts/lock_quant_inputs.py` | 外部输入锁定 |
| `scripts/compare_cpp_closed_runtime.py` | 相同设备、token 和计时范围的性能报告比较 |
| `FRAMEWORK_LOCK.json` | 图、量化、运行时和 profiling 合同 |

字段定义和调用边界见 [AIR/OM/C++ 接口参考](../docs/QUANT_AIR_OM_FRAMEWORK.md)。
