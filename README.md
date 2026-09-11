# Qwen3.5-4B DFlash

在 Ascend 310P 上运行 Qwen3.5-4B ordinary greedy 和 DFlash speculative decoding，
支持 batch=1、greedy、FP16 Target 或 W8A8 Target。Draft 使用官方
Qwen3.5-4B-DFlash checkpoint，以 FP16 执行。

已有 OM 时，从 [当前版本运行命令与结果](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md)开始：
包含多 prompt 一次测试、允许输出差异、离线汇总、decode/Draft/Verify 计时提取，
以及 deterministic 开关和 FC 漂移的定位结果。
当前 8 条、每条 128 token 的设备报告显示整体 1.50075× 加速、20.69% 候选接受率；
7 条更快，1 条变慢，各模式重复稳定，跨模式输出不同，任务质量未评估。
上述结果属于默认 Chunk 路线。增量 OM 也可通过 `--verify-gdr mtp` 选择 GDR MTP；
导出、清单选择、状态差异和对照命令见 [两条验证路径](docs/GDR_VERIFY_ROUTES.md)。

再看 [DFlash 结构与生成流程](docs/DFLASH_ARCHITECTURE.md)：从整体流程和逐轮例子，
理解 Target/Draft、三张 DFlash OM、状态提交，以及获得加速的条件。

## 1. 选择运行方式

| 方式 | 适用场景 | 完整操作手册 |
|---|---|---|
| Python NPU | FP16/W8A8 验证、生成、benchmark、细分阶段 msprof | [Python NPU 从零运行](docs/DFLASH_RUN_AND_VALIDATE.md) |
| AIR → OM → C++ | W8A8 模型导出、转换、AscendCL 执行、OM 阶段 msprof | [AIR/OM/C++ 从零部署](docs/GDR_CHUNK_AIR_OM.md) |

两份手册均从环境和模型准备开始，按顺序提供所需文件、命令、输出位置及检查方法。
源码仓库不包含 checkpoint、量化权重、CANN、自定义算子包或 NPU receiver 加载器；
这些依赖的路径和要求在各手册前置步骤中说明。

## 2. 按顺序完成运行

1. 准备 Linux/NPU 软件栈、源码和独立运行目录。
2. 准备 Target、官方 Draft、外部 receiver 加载器；W8A8 模式再准备量化输入。
3. 执行 Python NPU ordinary/DFlash token、EOS 和停止原因一致性检查。
4. 使用 C++ 时，继续锁定输入、导出 AIR、编译 OM、构建 runner，并与 NPU ordinary 比较。
5. 正确性通过后，执行不带 profiler 的时延测量，再用 msprof 定位阶段和算子耗时。

## 3. OM 划分

| OM | 功能 | 单独普通模式 | 单独 DFlash 模式 |
|---|---|---|---|
| `target_prefill.om` | 64 行物理 gear 的 prompt 分块，输出特征和状态 | 加载 | 加载 |
| `target_decode.om` | 真正的一行 Target decode | 加载 | 不加载 |
| `target_verify.om` | 16 行 verify、Top1、接受判断；Chunk 重算或 MTP bank 选择提交状态 | 不加载 | 加载 |
| `draft.om` | 特征投影、Draft KV 追加和一次并行 proposal | 不加载 | 加载 |

DFlash 使用 3 个 OM，普通模式使用 2 个；对照部署共 4 个。
两种模式共用 prefill，verify 内部完成 commit，无需独立 commit OM。
显式 GDN、conv、KV 状态由 C++ 持续保存在设备 buffer 中。

## 4. 单次 msprof

统一使用 `tools/run_msprof.sh`，以下参数放在 `--` 之前。采集由 msprof 动态 PID CLI
控制，无需 pyACL；窗口外完成预热和输入状态准备。

| 后端和模式 | `--profile-stage` 可选值 | `all` 的行为 |
|---|---|---|
| `--profile-backend python --profile-mode ordinary` | `prefill`、`decode`、`all` | 同一进程分别采一次 prefill 和 decode |
| `--profile-backend cpp --profile-mode ordinary` | `prefill`、`decode`、`all` | 同一进程分别采一次 prefill 和 decode |
| `--profile-backend python --profile-mode dflash` | `prefill`、`feature-project`、`draft`、`verify-input`、`verify`、`target-top1`、`accept-commit`、`draft-verify`、`decode-round`、`all` | 9 个阶段分别采一次 |
| `--profile-backend cpp --profile-mode dflash` | `prefill`、`draft`、`verify`、`all` | 3 个阶段分别采一次 |

每个窗口保存独立算子数据、阶段报告和汇总 CSV。C++ 的 verify 已融合接受判断和提交；
Python 的 verify、accept-commit 可以分别采集。完整命令和精确采集范围见对应操作手册。

## 5. 检查结果

ordinary Target 是 strict-greedy 正确性对照；token IDs、EOS 和停止原因必须一致。
性能比较使用相同 prompt token、精度、输出 token 和同步边界，保留 3 次预热与 10 次测量。
msprof 算子累计时间用于定位热点，不能直接替代整段生成时延。

上述为默认严格模式。当前允许输出差异的速度实验使用
`benchmark_prompts.py --allow-output-differences`：
输出对照仍如实记录，满足各自 3+10 和执行检查时标为 `PASS_WITH_DIFFERENCES`。
该结果比较各模式自己的输出，不代表质量等价或严格正确性通过。
当前文档中的设备数据来自用户提供的报告；主机模拟测试不提供设备加速结论。

## 6. 参考资料

| 文档 | 内容 |
|---|---|
| [当前版本运行命令与结果](docs/DFLASH_CURRENT_USAGE_AND_RESULTS.md) | 运行/重编起点、8 条 prompt 增益、时延口径、分项提取、确定性漂移与待定位问题 |
| [DFlash 结构与生成流程](docs/DFLASH_ARCHITECTURE.md) | 整体流程、逐轮 token 与缓存、两遍 GDR、加速条件、算子精度、显存和采集范围 |
| [自定义算子](docs/DFLASH_OPERATORS.md) | 必需 ABI、Tensor 实现与性能候选 |
| [AIR/OM/C++ 接口](docs/QUANT_AIR_OM_FRAMEWORK.md) | factory、manifest、tensor ABI、CLI 与计时范围 |
| [DFlash 源码索引](models/dflash_v1/README.md) | 命令、调度、Target、Draft、量化与 profiling 文件 |
| [框架目录](framework/README.md) | 导出、编译、C++ runtime 和合同文件 |
