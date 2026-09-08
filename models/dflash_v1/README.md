# DFlash 源码索引

本目录实现 Qwen3.5-4B 的 persistent incremental DFlash、ordinary 对照、量化和阶段采集。
运行时先阅读 [Python NPU 从零运行手册](../../docs/DFLASH_RUN_AND_VALIDATE.md)，
按环境、checkpoint、receiver、验证、benchmark、msprof 的顺序执行。
AIR/OM/C++ 部署见 [部署手册](../../docs/GDR_CHUNK_AIR_OM.md)。

## 1. 运行、调度和采集

| 文件 | 职责 |
|---|---|
| `run_npu.py` | NPU 命令，FP16/Target W8A8，validate/dflash 和阶段采集 |
| `run_rollback.py` | CPU/CUDA/NPU 共用执行与报告 |
| `benchmark_npu.py` | ordinary/DFlash 独立进程、同步整段生成测量 |
| `stage_profile.py` | 单个阶段和 all 的状态准备、预热、采集与报告 |
| `msprof_cli.py` | msprof 动态 PID CLI 的 start/stop/quit 控制 |
| `dflash_rollback_decode.py` | ordinary incremental、Draft/verify、连续前缀接受和 EOS |
| `dflash_rollback_adapter.py` | Target 事务、feature 生命周期和 Draft KV |
| `diagnose_acceptance.py` | proposal、Target Top1 和接受率诊断 |
| `dflash_reference_decode_v1.py` | 完整前缀重算诊断 oracle |

## 2. Draft

| 文件 | 职责 |
|---|---|
| `modeling_dflash.py` | 官方 6 层 Draft 与 request-local committed/transient KV |
| `dflash_config.py` | block、6 层/69 tensor 和 checkpoint shape 合同 |
| `dflash_weights.py` | revision、hash、tensor 审计和流式加载 |
| `dflash_ops.py` | CPU/CUDA Torch primitives |
| `dflash_ascend310p_ops.py` | NPU Tensor backend，禁用 CPU fallback |

`block_size` 包含 anchor；B=16 对应最多 15 个 proposal、16 个 Target verify 行。
Draft attention 读取已提交 KV、本轮追加 KV 和 transient block；成功后只保存 committed KV。

## 3. Target 与状态提交

| 文件 | 职责 |
|---|---|
| `modeling_qwen3_5_dflash.py` | CPU/CUDA feature-enabled Target |
| `../modeling_qwen3_5_hiai_nd.py` | NPU ordinary Target |
| `../modeling_qwen3_5_hiai_nd_dflash_rollback.py` | NPU rollback Target |
| `../internal_dflash_bridge.py` | GDN state、两遍 chunk GDR commit 和 paged-KV cursor |
| `../export_model_wrapper_qwen3_5_dflash_rollback.py` | receiver wrapper 的 chunk transaction adapter |
| `dflash_target_features.py` | 八层 Target feature 合同 |

Prompt、verify 和 accepted-prefix commit 使用接受 `INT16[B] effective_length` 的
`npu_chunk_gated_delta_rule`。verify 的第二遍 GDR 从本轮初始 state 按 `accepted+1` 提交。
causal-conv 使用 NPU Tensor 公式实现，接口与优化候选见[算子清单](../../docs/DFLASH_OPERATORS.md)。

## 4. Target W8A8

| 文件 | 职责 |
|---|---|
| `original_quant.py` | 量化 key 映射、blocked-ZN 与 QLinear 替换 |
| `target_quant.py` | YAML、INT8 embedding/scale 和 QLinear topology 审计 |
| `preflight_target_quant.py` | Target 量化装配、公式和事务预检 |
| `w8a8_emulation.py` / `validate_w8a8_cpu.py` | CPU/CUDA 量化公式诊断 |

量化默认关闭。启用时传 `--config <量化 YAML> --quant_mode enable`，仅替换 Target Linear
和 Target 输入 embedding；Draft-facing embedding、LM head 和 Draft 主体保持 FP16。
YAML 的生成及完整调用步骤见[运行手册](../../docs/DFLASH_RUN_AND_VALIDATE.md)。

## 5. 进一步查阅

| 文档 | 内容 |
|---|---|
| [DFlash 架构](../../docs/DFLASH_ARCHITECTURE.md) | token、feature、cache、state 与功能范围 |
| [算子清单](../../docs/DFLASH_OPERATORS.md) | 算子职责、dtype、shape 与性能候选 |
| [框架接口](../../docs/QUANT_AIR_OM_FRAMEWORK.md) | AIR/OM/C++ 配置、ABI 和产物 |
