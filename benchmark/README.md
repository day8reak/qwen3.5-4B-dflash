# Ascend NPU benchmark 快速上手

本页给出 Qwen3.5-4B DFlash V1 在真实 Ascend NPU 上的最短性能测试流程：先分别采集
未 profiling 的 ordinary/DFlash 基线，再用 `msprof` 做一次诊断。完整的计时边界、报告字段
和结果解释见 [NPU benchmark 详细说明](../docs/NPU_BENCHMARK.md)。

当前两种模式测量的是仓库已有实现：

- `ordinary`：普通 full-prefix greedy replay；
- `dflash`：target bootstrap + sequential full-prefix DFlash V1 replay。

当前 V1 可能比 ordinary 更慢。这里的结果不能代表路线图中的增量 cache 或单次整块 verify。

## 1. 准备环境

先取得 `benchmark` 分支：

```bash
git clone --branch benchmark --single-branch \
  https://github.com/day8reak/qwen3.5-4B-dflash.git
cd qwen3.5-4B-dflash
export DFLASH_REPO="$PWD"
```

已有 clone 时，只需切到 `benchmark` 分支并把 `DFLASH_REPO` 指向仓库根目录。然后按
[Ascend NPU 部署说明](../docs/NPU_DEPLOYMENT.md) 把源码部署到运行工程。需要：

- Python 3.10，以及与设备匹配的 PyTorch、`torch_npu` 和 CANN；
- 可用的 `npu-smi`；使用 profiling 时还需要 PATH 中有 `msprof`；
- 本地 Qwen3.5-4B Target 和官方 Qwen3.5-4B-DFlash checkpoint；
- 一个位于源码和 checkpoint 目录之外的可写结果目录。

设置实际路径：

```bash
set -euo pipefail

export DEPLOY_ROOT=/path/to/qwen35-runtime
export MODEL_PYTHON=/path/to/python
export TARGET_DIR=/path/to/Qwen3.5-4B
export DRAFT_DIR=/path/to/Qwen3.5-4B-DFlash
export RUN_ROOT=/path/to/npu-benchmark-run
export KV_CACHE_MAX_LEN=4096

cd "$DEPLOY_ROOT"
export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN_ROOT"

test -x "$MODEL_PYTHON"
test -f "$DEPLOY_ROOT/models/dflash_v1/benchmark_npu.py"
test -d "$TARGET_DIR"
test -d "$DRAFT_DIR"
npu-smi info
"$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu --help
```

`KV_CACHE_MAX_LEN` 必须替换为部署配置中的真实值，并且能被 64 整除。

## 2. 跑未 profiling 基线

ordinary 和 DFlash 必须用两个独立进程；以下命令使用相同 prompt、生成长度、`block_size`、cache
长度、3 次 warmup 和 10 次正式测量：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
  --mode ordinary \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --kv-cache-max-len "$KV_CACHE_MAX_LEN" \
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
  --kv-cache-max-len "$KV_CACHE_MAX_LEN" \
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

不要用 msprof 下的耗时替代这组基线；profiling 本身会增加开销。

## 3. 跑一次 msprof 诊断

基线通过后，用 1 次 warmup + 1 次 measurement 采集 DFlash timeline 和 AI Core 指标：

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
    --kv-cache-max-len "$KV_CACHE_MAX_LEN" \
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

timeline 中正式测量区间是 `qwen35/dflash/measure/0`。需要 Memory 或 MemoryUB 指标时，
更换 `--aic-metrics`，同时使用新的 `--label` 和 `--output-dir`，不要覆盖已有证据。

## 4. 判断是否成功

先检查 `ordinary.json` 和 `dflash.json`：

- `strict_greedy_exact_match` 为 `true`；
- `correctness_gate.dflash_execution.status` 为 `PASS`；
- `benchmark.status` 为 `PASS`；
- 10 次 measurement 的 `generated_token_ids_sha256` 一致；
- `target_integration.*_call_reconciliation.status` 为 `PASS`；
- `operator_fallback_enabled` 为 `false`。

性能比较读取 `benchmark.summary.latency_ms` 和
`benchmark.summary.aggregate_output_tokens_per_second`，并保留全部 10 次原始结果。msprof 输出在：

```text
$RUN_ROOT/msprof-dflash-pipe/
├── profile/msprof/dflash-pipe/
├── log/
└── manifest/dflash-pipe.json
```

常见问题：

- 找不到 `benchmark_npu.py`：先重新部署 `models/dflash_v1/`；
- NPU preflight 失败：检查 `npu-smi`、`torch_npu`、CANN 与当前设备环境；
- 找不到 `msprof`：激活与设备匹配的 CANN 工具环境；
- 结果目录被拒绝：把 `RUN_ROOT` 移到源码、Target 和 Draft checkpoint 之外；
- DFlash 比 ordinary 慢：当前 V1 是 sequential full-prefix replay，这本身不代表测量失败。

更多细节见 [NPU benchmark 与 msprof](../docs/NPU_BENCHMARK.md) 和
[DFlash 性能路线图](../docs/DFLASH_FULL_AND_PERFORMANCE_ROADMAP.md)。
