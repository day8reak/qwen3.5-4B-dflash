# Ascend NPU benchmark 与 msprof 采集

本页用于测量仓库当前 Qwen3.5-4B DFlash V1 实现的真实 NPU 性能，并用
`msprof` 定位算子、Runtime 和 AI Core 流水瓶颈。benchmark 不修改模型数学、草稿
结构、token 验证规则或 HIAI ABI。

## 1. 测量边界

新增入口：

```bash
python -B -m models.dflash_v1.benchmark_npu --help
```

固定机器可读合同为 `config/npu_benchmark_v1.json`。

每个进程只测一个模式：

- `ordinary`：当前 authoritative ordinary full-prefix greedy replay；
- `dflash`：当前 target bootstrap + sequential full-prefix DFlash V1 replay。

这两个模式反映仓库当前真实代码，不代表路线图中的增量 cache、单次整块 verify 或
state transaction。当前 V1 可能比 ordinary 更慢；benchmark 的目的首先是得到可信基线，
不能把未来路线的预期收益写成当前结果。

计时前会完成以下工作，它们不进入 host latency：

1. 校验 HIAI 源码、loader、checkpoint 和本地运行时身份；
2. 加载 Target 与官方 DFlash draft；
3. 执行 full-prefix state isolation、feature zero-impact 以及 ordinary/DFlash strict-greedy
   零 token 差异门禁；
4. 重置 NPU peak-memory 统计。

每个 warmup/measurement 的计时区间为：

```text
device synchronize
    -> 一次完整 ordinary 或 DFlash generation
    -> device synchronize
```

报告的 latency 和 output tokens/s 包含 Python/runtime 调度以及完整设备执行；不包含模型
加载、prompt tokenization 和 correctness gate。阶段级、算子级耗时必须以 `msprof` timeline
为准。

## 2. 前置条件

先完成 [NPU 部署流程](NPU_DEPLOYMENT.md)，确保部署根目录中至少存在：

```text
models/modeling_qwen3_5_hiai_nd.py
models/internal_dflash_bridge.py
models/dflash_v1/benchmark_npu.py
models/dflash_v1/run_npu.py
```

还需要：

- 与设备匹配的 PyTorch、`torch_npu` 和 CANN；
- 可用的 `npu-smi`；
- 使用 msprof 时，PATH 中有可执行的 `msprof`；
- 本地 Qwen3.5-4B Target 与官方 `z-lab/Qwen3.5-4B-DFlash` 权重；
- benchmark/report/profile 路径位于源码、Target 权重和 Draft 权重目录之外。

benchmark 强制使用 FP16、EOS `248044`、package-local Ascend 310P ops，且不提供 CPU
fallback 选项。CPU/CUDA 运行不能作为 NPU 性能证据。

## 3. 未 profiling 的正式基线

同一对比必须固定以下项目：Git revision、Target/Draft checkpoint、设备、CANN/driver/
firmware、prompt token、thinking 模式、`max_new_tokens`、K、cache 长度和 chunk 配置。

示例变量：

```bash
export DFLASH_REPO=/path/to/qwen3.5-4B-dflash
export DEPLOY_ROOT=/path/to/qwen35-runtime
export MODEL_PYTHON=/path/to/python
export TARGET_DIR=/path/to/Qwen3.5-4B
export DRAFT_DIR=/path/to/Qwen3.5-4B-DFlash
export RUN_ROOT=/path/to/npu-benchmark-run

cd "$DEPLOY_ROOT"
export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"
```

ordinary 与 DFlash 必须分别启动进程，避免 allocator、状态或先后顺序污染：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
  --mode ordinary \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 32 \
  --block-size 16 \
  --warmup 3 \
  --repetitions 10 \
  --device npu:0 \
  --report "$RUN_ROOT/ordinary.json"

"$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
  --mode dflash \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 32 \
  --block-size 16 \
  --warmup 3 \
  --repetitions 10 \
  --device npu:0 \
  --report "$RUN_ROOT/dflash.json"
```

`--block-size` 使用官方总行数口径：包含 1 个 anchor，因此值 16 对应 K=15。它在 ordinary
的计时路径中不参与生成，但仍用于进程内的 DFlash correctness gate 和 case identity，
所以两边必须保持一致。

每份报告至少核对：

- `strict_greedy_exact_match == true`；
- `correctness_gate.dflash_execution.status == "PASS"`；
- `benchmark.status == "PASS"`；
- 所有 repetition 的 `generated_token_ids_sha256` 相同；
- `target_integration.*_call_reconciliation.status == "PASS"`；
- `operator_fallback_enabled == false`；
- `runtime_identity.device_name`、torch/torch_npu 版本和 source hash 符合本次 case。

性能比较使用 `benchmark.summary.latency_ms` 和
`benchmark.summary.aggregate_output_tokens_per_second`。保留全部 10 条原始 measurement，
不要只抄一个最小值。EOS 提前结束时，必须同时报告每次实际生成 token 数。

## 4. 用 msprof 做诊断

`tools/run_msprof.sh` 会执行真实 NPU preflight，采集 `npu-smi info`，固定输出归属，记录
Git/source identity，并默认启用：

```text
ascendcl=on
runtime-api=on
task-time=on
aicpu=on
ai-core=on
aic-mode=task-based
aic-metrics=PipeUtilization
msproftx=on
```

msprof 会给运行增加开销，所以 profile 数据不能替代上一节的未 profiling 基线。诊断时
建议只保留 1 个 warmup 和 1 个 measurement：

```bash
"$DFLASH_REPO/tools/run_msprof.sh" \
  --label dflash-pipe \
  --output-dir "$RUN_ROOT/msprof-dflash-pipe" \
  --python "$MODEL_PYTHON" \
  --aic-metrics PipeUtilization \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode dflash \
    --target-dir "$TARGET_DIR" \
    --draft-dir "$DRAFT_DIR" \
    --kv-cache-max-len 4096 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat \
    --enable-thinking \
    --max-new-tokens 32 \
    --block-size 16 \
    --warmup 1 \
    --repetitions 1 \
    --device npu:0 \
    --report "$RUN_ROOT/msprof-dflash-pipe/dflash.json"
```

wrapper 会设置 `DFLASH_BENCHMARK_MSTX=1`，因此 timeline 中可见：

```text
qwen35/dflash/warmup/0
qwen35/dflash/measure/0
```

correctness gate 仍会在同一 profile 进程执行，但不在上述 benchmark range 内。分析正式
measurement 时按 `qwen35/<mode>/measure/<index>` 定位，避免把 checkpoint 校验、模型加载
或 correctness gate 计入一次生成。

建议分开采集以下 metric，不能在不同 metric pass 之间直接比较 msprof 下的 host latency：

```bash
--aic-metrics PipeUtilization
--aic-metrics Memory
--aic-metrics MemoryUB
```

每个 `--label` 和 `--output-dir` 组合只能使用一次，防止覆盖旧 evidence。输出结构：

```text
<output-dir>/
├── profile/msprof/<label>/
├── log/msprof-<label>.log
├── log/preflight-<label>.log
├── log/device-<label>.log
└── manifest/<label>.json
```

## 5. 结果解释

当前实现是 full-prefix replay，重点先看：

1. `target_forward_calls` 和 `target_input_tokens_recomputed` 是否随输出长度快速增加；
2. DFlash 的 feature replay、Draft 六层和 sequential verify 各占多少设备时间；
3. `fc.weight [2560,20480]` projection、LM head/Top-1、attention、数据搬运和同步是否是热点；
4. Pipeline 利用率、Memory/MemoryUB 指标是否支持相同的瓶颈判断；
5. ordinary/DFlash 的 prompt、输出 token、设备和 timing scope 是否完全一致。

只有 integrated strict-greedy accuracy 通过、相同 case 的未 profiling 10/10 稳定分布完成后，
才能讨论 speedup。单个 msprof trace、CPU 结果或一次最小 latency 都不能作为性能晋级结论。

## 6. 官方工具参考

- [CANN msprof 命令行采集说明](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/profiling/atlasprofiling_16_0012.html)
- [MindStudio MSTX API 说明](https://www.hiascend.com/document/detail/en/mindstudio/830/API/mstxAPIReference/msprof_tx_0001.html)
