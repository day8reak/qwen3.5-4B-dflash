# DFlash rollback 运行与验证

本文只保留可执行命令、报告门禁和性能口径。算法见[当前架构](DFLASH_ARCHITECTURE.md)，算子
见[自定义算子清单](DFLASH_OPERATORS.md)。

## 1. 证据分级

| 结论 | 最低证据 |
| --- | --- |
| scheduler 正确 | 同一 Target 下 ordinary/DFlash token ID、EOS、stop reason 完全一致 |
| rollback 正确 | verify 不含历史前缀；state/KV/feature/position 同时提交 `1+a`；拒绝后下一 token 仍一致 |
| NPU 路线通过 | 真实 Ascend 310P、无 fallback、记录 runtime/device/source/op 身份和 kernel trace |
| 端到端加速 | 相同 workload、精度、输出、同步边界的独立进程 3+10 配对测量 |

CPU reduced-shape 只能证明辅助逻辑；CUDA 只能证明 framework 路线。算子累计时间、单次 msprof
结果和 correctness PASS 都不能替代端到端加速证据。

## 2. 准备

- Python 3.10、`transformers==5.14.1`、`safetensors`；
- 匹配 CPU/CUDA/NPU 的 PyTorch；
- 本地 Qwen3.5-4B Target 与完整 tokenizer；
- 本地锁定的 `z-lab/Qwen3.5-4B-DFlash` checkpoint；
- 报告、日志、profile 和 cache 放在源码仓库外。

```bash
python -m pip install "transformers==5.14.1" safetensors huggingface-hub
export PYTHONDONTWRITEBYTECODE=1
```

入口会检查 Draft config、6 层/69 tensor、shape/dtype/hash，以及 Target/Draft 共享权重、device 和
dtype。`block_size` 包含 anchor，范围 2..16；B=16 对应 K=15、T=16。

## 3. CPU/CUDA

```bash
export PYTHONPATH="$PWD"

python -B -m models.dflash_v1.run_rollback \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat --enable-thinking \
  --max-new-tokens 32 \
  --execution-mode validate \
  --block-size 16 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cuda:0 \
  --report /path/to/run/dflash-cuda.json
```

CPU 改为 `--device cpu`，并按环境选择 dtype。CPU/CUDA 使用
`FrameworkDFlashRollbackTarget + TorchDFlashOps`：Target DynamicCache/GDN 在 verify 前保存，
commit 时只重放 anchor 与 accepted proposal，不重放 prompt 或更早历史。

CUDA 不可用时必须直接失败，不能用 CPU 结果伪装。不要传 HIAI factory、source、reset hook 或
NPU ops backend。

## 4. HIAI/NPU 部署

保留原 ordinary 文件，新增 rollback 文件：

```text
models/
├── modeling_qwen3_5_hiai_nd.py                         # ordinary；适配新 GDR effective_length ABI
├── export_model_wrapper_qwen3_5.py                     # 原文件，不覆盖
├── modeling_qwen3_5_hiai_nd_dflash_rollback.py
├── export_model_wrapper_qwen3_5_dflash_rollback.py
├── internal_dflash_bridge.py
└── dflash_v1/
```

复制源码后使用部署环境声明的 Python：

```bash
export DEPLOY_ROOT=/path/to/copied-runtime
export MODEL_PYTHON=/path/to/deployment/python
export PYTHONPATH="$DEPLOY_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1

"$MODEL_PYTHON" -B -m py_compile \
  "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/internal_dflash_bridge.py" \
  "$DEPLOY_ROOT/models/dflash_v1/run_npu.py"
```

先确认部署的原 GDR 算子包已经升级到带 `effective_length` 的新 ABI：

```text
npu_chunk_gated_delta_rule(
  query, key, value, g, beta, effective_length,
  chunk_size=64, initial_state=None,
  output_final_state=False, use_qk_l2norm_in_kernel=False
)
```

其中 `effective_length` 必须接受 `[B] INT16`。普通 modeling、rollback ordinary、verify 和
accepted-prefix commit 都会传这个输入；旧注册包会在第一次 Target 调用时直接接口不匹配。
只检查原 GDR 注册：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import torch
import torch_npu

op = getattr(torch_npu, "npu_chunk_gated_delta_rule", None)
if not callable(op):
    op = getattr(getattr(torch.ops, "npu", None), "npu_chunk_gated_delta_rule", None)
assert callable(op), "npu_chunk_gated_delta_rule is not registered"
print("CHUNK_GDR_REGISTERED")
PY
```

语法和符号可见不证明 shape、数值或整网通过。

## 5. NPU FP16

先用 B=2 跑至少一轮 Draft/verify：

```bash
export RUN_DIR=/path/to/dflash-run
mkdir -p "$RUN_DIR"

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 2048 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat --enable-thinking \
  --max-new-tokens 2 \
  --execution-mode validate \
  --block-size 2 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-fp16-smoke.json"
```

`run_npu` 固定 FP16、EOS 248044、prefill chunk 64、decode chunk 1 和 package-local NPU Draft
backend。`kv-cache-max-len` 必须覆盖 prompt+output，并能被 64 整除。Prompt、verify 和 commit
都使用原 GDR chunk 路线：verify 传当前 `T`，commit 从同一个 round-start state 出发传
`a+1`。本分支不要求注册 GDR-MTP。

如果 anchor 立即 EOS，报告会是 `INCONCLUSIVE_NO_DRAFT_ROUND`；换固定非立即结束 prompt。B=2
通过后再跑 `--max-new-tokens 32 --block-size 16`。

离线 validate 通过后，日常单跑改为：

```bash
  --execution-mode dflash
```

`dflash` 模式不额外跑 ordinary；它仍检查 checkpoint/source/device，但
`strict_greedy_exact_match=null`，正确性证据必须来自同 revision 的 validate 报告。

默认就是非量化模式。可以省略量化参数，也可显式传 `--quant_mode disable`。

## 6. Target W8A8 dynamic

量化 artifact 由原部署工程提供。分支不重新量化，也不复制权重。

先跑不加载 Draft 的预检：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.preflight_target_quant \
  --target-dir /path/to/Qwen3.5-4B \
  --prompt-ids 1,2,3,4 \
  --device npu:0 \
  --kv-cache-max-len 2048 \
  --config ./config/qwen3.5.yaml \
  --compare-first-qlinear \
  --report "$RUN_DIR/target-quant-preflight.json"
```

YAML：

```yaml
quanted_pth: /data/qwen35-w8a8/linear
embedding_weight_path: /data/qwen35-w8a8/embedding_weight.bin
embedding_scale_path: /data/qwen35-w8a8/embedding_scale.bin
```

`quanted_pth` 是包含 `data*.safetensors` 的目录；两个 embedding 文件是原 `numpy.tofile` raw
artifact。预检覆盖 Linear->QLinear 拓扑、同 activation 公式诊断、真实 multi-token prefill、
ordinary S=1、rollback S=1 和 embedding lookup，但不替代整网 DFlash 门禁。

预检通过后，在第 5 节命令追加：

```bash
  --config ./config/qwen3.5.yaml \
  --quant_mode enable
```

Target Linear/输入 embedding 走 W8A8，Draft embedding、LM head 和 6 层主体保持 FP16。整网
validate 比较的是“同一个 W8A8 Target 的 ordinary 与 DFlash”；若还要求 W8A8 与 FP16 token
一致，需要另做 ordinary 精度对照。

## 7. NPU 性能基准与 msprof

先区分三条路径：

| 对象 | 入口 | 用途 |
| --- | --- | --- |
| 原 main 非 DFlash | 原工程 `inference.py` | 旧部署权威基线 |
| rollback ordinary | `benchmark_npu --mode ordinary` | 同 receiver、同门禁的 scheduler 内部控制组 |
| rollback DFlash | `benchmark_npu --mode dflash` | 当前 Draft/verify/commit 路线 |

未 profiling 的正式基线使用独立进程、3 次 warmup、10 次 measurement。每次 measurement 包含
一轮完整 generation 和末尾设备同步，不包含 checkpoint hash、模型加载、tokenizer 或前置
correctness gate。

```bash
export BENCH_DIR=/path/to/dflash-benchmark
mkdir -p "$BENCH_DIR"

for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode "$MODE" \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking \
    --max-new-tokens 32 --block-size 16 \
    --warmup 3 --repetitions 10 \
    --device npu:0 \
    --report "$BENCH_DIR/$MODE.json"
done
```

W8A8 对比时两个进程必须追加完全相同的参数：

```bash
QUANT_ARGS=(--config ./config/qwen3.5.yaml --quant_mode enable)
```

再把 `"${QUANT_ARGS[@]}"` 放到 ordinary 和 dflash 两条命令中。不能用 W8A8 DFlash 与 FP16
ordinary 计算 scheduler speedup。

正式比较前检查：

- copied source-tree hash、Target/Draft checkpoint、device/runtime 相同；
- prompt token hash、chat template、thinking、max tokens、实际输出 token 和输出 hash 相同；
- `block_size`、KV 长度、chunk、quant mode 相同；
- acceptance、Draft/verify calls、Target rows、fallback rounds 工作量可解释。

量化或低接受率场景不应假设 B=16 最快；保持上述身份不变，追加 B=`2/4/6/8/16` sweep。

### 7.1 原 main 非 DFlash 模型

原模型从原部署工程根目录运行 `inference.py` 和 main 的
`models/modeling_qwen3_5_hiai_nd.py`。`inference.py` 与配置不在本仓库中。下面命令故意保持
非量化：

~~~bash
cd /path/to/qwen3.5-main-runtime
export DFLASH_SOURCE=/path/to/copied-qwen3.5-4B-dflash
export BENCH_DIR=/path/to/dflash-benchmark
mkdir -p "$BENCH_DIR"

"$DFLASH_SOURCE/tools/run_msprof.sh" \
  --label main-original-unquantized \
  --output-dir "$BENCH_DIR/msprof-main-original-unquantized" \
  --python python3 \
  --aic-metrics PipeUtilization \
  --no-msproftx \
  -- \
  python3 inference.py \
    --config ./config/qwen3.5.ymal \
    --max_token 32
~~~

这条命令与下面的 rollback ordinary 并不是同一入口。若要计算“整体方案相对旧部署”的
speedup，还要对齐 prompt/input IDs、thinking、实际输出 token、cache 初始化和计时同步边界；
参数名字相似不自动代表边界相同。

### 7.2 rollback 内部 ordinary 控制组

`--mode ordinary` 并不是原 main 非 DFlash 模型。msprof 建议 1+1，只用于诊断：

```bash
tools/run_msprof.sh \
  --label rollback-ordinary-pipe \
  --output-dir "$BENCH_DIR/msprof-rollback-ordinary-pipe" \
  --python "$MODEL_PYTHON" \
  --aic-metrics PipeUtilization \
  --no-msproftx \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode ordinary \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking \
    --max-new-tokens 32 --block-size 16 \
    --warmup 1 --repetitions 1 --device npu:0 \
    --report "$BENCH_DIR/msprof-rollback-ordinary-pipe/ordinary.json"
```

### 7.3 rollback DFlash

```bash
tools/run_msprof.sh \
  --label dflash-pipe \
  --output-dir "$BENCH_DIR/msprof-dflash-pipe" \
  --python "$MODEL_PYTHON" \
  --aic-metrics PipeUtilization \
  --no-msproftx \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode dflash \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking \
    --max-new-tokens 32 --block-size 16 \
    --warmup 1 --repetitions 1 --device npu:0 \
    --report "$BENCH_DIR/msprof-dflash-pipe/dflash.json"
```

`run_msprof.sh` 不读取 Git/branch/dirty 信息，复制代码到目标机后可直接使用；manifest 通过源码
内容 hash 标识运行。wrapper 默认关闭 MSTX，并采 AI Core、task-time、runtime-api 和 AscendCL。
只有目标环境确认支持时才显式传 `--msproftx`；若出现 `mstx.range_start failed`，移除该选项后
重采。默认进程级 profile 还包含模型加载、correctness gate 和 warmup，不能把算子累计值或整个
profile 时长当作单次 measurement latency。

### 7.4 只采一次 prefill 或 Draft 生成 + Target verify

**普通模式也支持单次采集**，Python NPU 和新增 C++ OM 入口均可使用。
保留下面默认 DFlash 命令不变；普通模式在 wrapper 的 `--` 前加
`--profile-mode ordinary --profile-stage prefill|decode|all`（实际填写其中一个阶段名）。
`all` 在普通模式下只有两个窗口：完整 prefill 一次、一行 decode 一次。

```bash
tools/run_msprof.sh \
  --label ordinary-all --output-dir "$BENCH_DIR/ordinary-all" \
  --python "$MODEL_PYTHON" \
  --profile-mode ordinary --profile-stage all --profile-warmup 1 \
  --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 --device npu:0 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking
```

只采一次 prefill 或 decode，替换上面的 `all` 并换新输出路径即可。
W8A8 在应用参数中照常加 `--config /path/to/qwen3.5.ymal --quant_mode enable`。
此 ordinary 调用本分支 Target 的 `begin_ordinary` / `advance_ordinary`，与 rollback
benchmark ordinary 对照一致；不运行原 main `inference.py`。沿用现有 `--draft-dir` 配置和
checkpoint 审计，但不加载 Draft 模型，也不调用 Draft 投影/生成或 Target verify/commit。

- `prefill` 窗口包含 cache reset、完整 prompt 的分块 Target forward 和最后一行 LM head；
  不采集 prompt tensor 上传、anchor Top1、Draft 特征收集/投影。
- `decode` 先在窗口外执行真实 ordinary prefill，取首 token 并上传 `[1,1]` 输入；只采一次
  `advance_ordinary` 的 Target forward、LM head 和状态更新。下一 token 的 Top1 在窗口外。
- 所有 warmup 都从相同 prompt 新建状态，采集前也重新准备；`--max-new-tokens` 不控制窗口。
  KV 要容纳 prompt 加一行；anchor 为 EOS 时明确退出，不生成空 decode 采集。

`all` 的两份算子数据在 `profile/msprof/ordinary-all/prefill/` 和 `decode/`，
同步耗时见 `ordinary-all-stage-summary.csv`。`ordinary-all-stage-report.json` 记录
`captured_ordinary_calls`，其中 decode 窗口必须是 prefill=0、decode=1，Draft/verify/commit=0。
prefill 的 `ordinary_prefill_token_calls` 历史字段名实际计数 64 行分块，不能当作 token 数。

C++ 同样用此 wrapper，新增 `--profile-backend cpp`，应用换成
`qwen35_dflash_acl_runner --model-kind chunk`，模型输入是 hash-locked 加载计划。
完整命令、四图/三图准备、融合 verify 的范围与输出说明见
[增量 OM 文档的单次 msprof 部分](GDR_CHUNK_AIR_OM.md#c-和-python-的单次-msprof)。
C++ 采集控制只需要 Python 标准库，不依赖 torch_npu 或 pyACL；普通只加载 prefill/decode
两个 OM。C++ 的 Top1 融合在 OM 内，计时范围与 Python 有差异，不能直接比较窗口时长。

以下继续说明默认 `--profile-mode dflash` 的 Python 9 阶段诊断。

本分支的 verify/commit 使用两次原 chunk GDR：第一次生成全部 verify rows 的输出，
第二次按 `accepted + 1` 重算 committed GDN state。单独测第一次使用 `verify`，
单独测第二次及提交使用 `accept-commit`；测 Draft 到提交的首轮事务使用 `decode-round`。
`draft-verify` 的窗口在第一次 verify 后结束，第二次 GDR 在该窗口之外。

Python NPU 路线可以传 `--profile-stage STAGE`。参数放在 wrapper 的 `--` **之前**，
应用使用 `models.dflash_v1.run_npu`，不经过 `benchmark_npu` 的 correctness/warmup/measurement 循环。
例如只采一次完整 prefill：

```bash
tools/run_msprof.sh \
  --label single-prefill \
  --output-dir "$BENCH_DIR/single-prefill" \
  --python "$MODEL_PYTHON" \
  --profile-stage prefill --profile-warmup 1 \
  --aic-metrics PipeUtilization \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 --block-size 16 --device npu:0 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking
```

只采一次 Draft 生成和紧接着的 Target verify：

```bash
tools/run_msprof.sh \
  --label single-draft-verify \
  --output-dir "$BENCH_DIR/single-draft-verify" \
  --python "$MODEL_PYTHON" \
  --profile-stage draft-verify --profile-warmup 1 \
  --aic-metrics PipeUtilization \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 --block-size 16 --device npu:0 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking
```

单独测一次 Draft 和一次 Target verify，可以分别运行下面两种模式（每种模式一个进程、一个采集窗口）：

```bash
for stage in draft verify; do
  tools/run_msprof.sh \
    --label "single-$stage" \
    --output-dir "$BENCH_DIR/single-$stage" \
    --python "$MODEL_PYTHON" \
    --profile-stage "$stage" --profile-warmup 1 \
    --aic-metrics PipeUtilization \
    -- \
    "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
      --target-dir /path/to/Qwen3.5-4B \
      --draft-dir /path/to/Qwen3.5-4B-DFlash \
      --kv-cache-max-len 2048 --block-size 16 --device npu:0 \
      --prompt "请用一句话解释为什么天空是蓝色的。" \
      --prompt-mode chat --enable-thinking
done
```

读取各自 `single-draft-stage-report.json` / `single-verify-stage-report.json` 中的
`profiled_elapsed_ms`，即对应阶段的同步耗时（毫秒）。该值排除模型加载、前置阶段和
msprof 控制握手时间，包含 profiling 开销；具体算子时间读取各自输出目录的 `op_summary*.csv`。
`verify` 会先在窗口外执行真实 prefill 和 Draft，使用其状态与 token 构建完整 verify 输入。
`draft` 完成后清理请求，不执行 verify。两次运行应保持 prompt、B、预热次数和量化配置一致。
单独测出的两段耗时不能视为联合窗口的精确拆分：`draft-verify` 还包含 proposal 整理和
block 创建，且单独采集引入了各自的边界同步与 profiling 开销。联合窗口仍不在 Draft 与 verify
之间额外插入同步。

一条命令依次单独采集所有阶段，使用 `--profile-stage all`：

```bash
tools/run_msprof.sh \
  --label all-stages \
  --output-dir "$BENCH_DIR/all-stages" \
  --python "$MODEL_PYTHON" \
  --profile-stage all --profile-warmup 1 \
  --aic-metrics PipeUtilization \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 --block-size 16 --device npu:0 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking
```

`all` 只启动一个应用进程、加载一份 Target/Draft，按下表所列阶段各采一次。
每个阶段先从相同 prompt 重建首轮状态并执行指定次数的预热，准备和预热都不采集；
这些重复准备会计入命令的总运行时间。控制器为每个阶段重新 attach 同一个应用 PID，
分别执行 start/stop/quit，退出前一个 msprof 后再进入下一阶段，无需 pyACL。
每次 start/stop/quit 都须有成功回执，各阶段的独立输出目录直接传给 msprof，
不会依靠 PROF 目录生成时间猜测阶段归属。

`all` 的输出路径：

- `profile/msprof/<label>/<stage>/`：该阶段独立的原始数据和算子 CSV。
- `<label>-stage-report.json`：共同运行身份，以及 `captures` 中逐阶段的范围、同步耗时和结果。
- `<label>-stage-summary.csv`：每阶段 `profiled_elapsed_ms`、算子行数、GDR verify/commit 层调用次数和数据路径。
- `manifest/<label>-control.json`：共同应用 PID，以及每阶段的 msprof PID、命令、回执、退出状态。

日志中的 `preparing stage=...` / `captured stage=...` 标明当前阶段。
任一阶段的执行、握手、导出或结果检查失败，整体返回 FAIL；已有原始数据和控制日志保留。
所有阶段都成功后才输出最终汇总 CSV。已有输出目录/报告不会被覆盖。
在原命令中切换单个阶段或 `all` 时，其余模型、量化和 prompt 参数保持一致。

W8A8 沿用应用参数 `--config /path/to/qwen3.5.ymal --quant_mode enable`。每次使用新的输出目录；
需分别查看 Memory/MemoryUB 时，保持 prompt、B、量化配置一致，换 `--aic-metrics` 单独采集。

| 选项 | 窗口内执行 | 窗口外执行 |
| --- | --- | --- |
| `prefill` | 一次 Target `begin_rollback`：新建 cache、所有真实 prompt 分块、feature 收集、最后真实行 LM head | 模型加载、预热、anchor Top1；本模式不执行 Draft feature projection 或 Draft/verify |
| `feature-project` | 一次真实 prompt features 的 Target→Draft `fc + hidden_norm` 投影 | Target prefill、shape 检查、投影结果指纹回传；本模式不执行 Draft/verify |
| `draft` | 一次首轮 Draft proposal：首次 Draft KV 构建、Draft 计算及 Top1 | 模型加载、预热、prefill、prompt feature projection、anchor Top1、proposal 整理和请求清理；不执行 Target verify |
| `verify-input` | 一次 Draft token ID 回传、EOS 截断及 proposal 整理、verify block tensor 创建/上传 | Draft、Target verify 及后处理 |
| `verify` | 一次首轮 Target verify，含标量 GDN state 克隆、第一次原 chunk GDR（`effective_length=T`）、Target LM head | 模型加载、预热、prefill、Draft、proposal 整理、verify 输入 tensor 创建，以及 Target Top1/accept/第二次 GDR commit |
| `target-top1` | 一次 verify logits 的有限值检查、argmax 和 token ID 回传 | Target LM head 已包含在 verify 中；prefill anchor Top1 在本窗口外 |
| `accept-commit` | 接受判断、零接受时关闭推测、从保留的 round-start state 执行第二次原 chunk GDR（`effective_length=accepted+1`）、conv state 选择、持久状态 dtype 转换、逻辑 KV 游标提交；继续推测时包含下一轮 feature projection | Target Top1、第一次 verify 的 Target 前向 |
| `draft-verify` | 首轮 Draft proposal（含首次 Draft KV 构建和 Draft Top1）、proposal 整理与 block 构建、一次 Target verify（含状态克隆及第一次 GDR） | 模型加载、预热、prefill、prompt feature projection、anchor Top1、Target verify 后的 Top1/accept/第二次 GDR commit |
| `decode-round` | 首轮事务：prefix tensor、Draft、verify 输入准备、Target verify、Target Top1、接受判断与提交，包含两次 GDR；内部没有额外采集同步 | prefill、prompt feature projection、anchor Top1、外层生成循环的 token 输出/EOS 终止判断、回调和文本解码 |

上述九项也都可以作为单独的 `--profile-stage` 值。`draft-verify` 和 `decode-round`
是有重叠范围的联合窗口，因此不要把九项耗时相加当作请求耗时。`all` 中各窗口同样各自带有
边界同步和 profiling 开销，不是同一轮时间线的无扰动拆分。
`accept-commit` 若真实接受数量为 0，会按生产规则关闭推测、跳过下一轮 feature projection；
此时仍须用 `effective_length=1` 执行第二次 GDR 并提交 anchor state，不能作为纯主机阶段。
本分支所有阶段均要求导出非空算子 CSV，包括零接受时的 `accept-commit`。
阶段 JSON 的 `captured_gdr_layer_calls.verify/commit` 和汇总 CSV 的
`gdr_verify_layer_calls/gdr_commit_layer_calls` 来自窗口前后累计计数器的差值，排除准备、
预热和窗口外提交。单独 `verify` 应只有 verify 层调用，单独 `accept-commit` 应只有 commit
层调用，`decode-round` 应有两者；任一不符均报 FAIL。这些是代码执行计数，具体 kernel
耗时仍取自 msprof 算子明细。
预处理/tokenizer、模型加载、prefill anchor Top1、文本输出及后续 decode 轮次不属于 `all` 的独立阶段。

`--profile-warmup` 默认 1，预热不启动采集。每次预热和正式采集都从同一个 prompt 重建请求状态，
不会沿用预热后推进的 KV/cursor。设为 0 可观察当前进程中未做该阶段显式预热的调用；
`all` 中前面的阶段仍可能已经触发相同内核，不能把后面的窗口当作进程冷启动。
Draft/verify 相关模式仍须先完成 prefill，`verify` 还须先完成 Draft。
这里采的是 **prefill 后的首轮**，不代表后续已有 Draft KV 的稳定 decode 轮次。

一个 prefill 可以包含多个 64-token 分块，不等于只执行一个模型 chunk。Draft/verify 相关模式都按完整 B
请求 proposal：B=16 请求 15 个 draft token，通常 verify T=16；遇到 proposal EOS 会按生产规则
缩短 T，实际值见报告 `result.verify_rows`；`draft` 不执行 verify，该值为 0，proposal 结果仍按 EOS
规则截断。若 prefill anchor 已是 EOS，则报错退出而不生成空采集。
此诊断分支忽略 `max-new-tokens` 的生成预算，也不运行 `execution-mode=validate` 的 ordinary 对照。

内部使用 **msprof 原生动态采集 CLI**，无需 Python `acl` 模块，也不调用 profiling API。
wrapper 在启动应用前自动设置 `PROFILING_MODE=dynamic`；模型加载和预热完成后，应用同步 NPU，
通过本地 socket 等待。控制程序执行以下命令并接管其交互输入：

```bash
msprof --dynamic=on --pid=<应用PID> --output=<raw目录> \
  --ascendcl=on --runtime-api=on --task-time=on --aicpu=on \
  --ai-core=on --aic-mode=task-based --aic-metrics=PipeUtilization
# 自动发送：start → 指定阶段执行完毕并同步 NPU → stop → quit
```

只有收到明确的 start 成功回执才放行指定阶段。兼容 `dynamic profiling start success......`
和目标机返回的 `dynamic profiling for pid <应用PID> start success`，带 PID 时必须与当前应用
一致；stop、quit 使用相同规则。实际匹配的回执保存在控制报告的 `acknowledgements` 字段。
执行完毕、同步 NPU 后，
等待 stop、quit 的成功回执及 msprof 正常退出，才继续应用的后处理。Draft 与 verify 之间不额外
插入同步。流程使用回执握手控制边界，不使用固定 delay/duration，也不暂停整个应用进程。
`--profile-timeout 600` 是默认的每步控制超时，覆盖等候模型加载/预热、阶段执行及 CLI 命令完成；
较慢机器可增大该值。超时、未识别的成功回执或进程异常退出均报 FAIL，并清理本次创建的进程。
如果停在 `msprof_start_sent`，工具已打印带 PID 的 `start success`，但没有
`msprof_start_acknowledged` / `application_started`，这是旧版 wrapper 未识别回执的现象。
按 Ctrl+C 结束该次运行，更新到包含此兼容修复的代码后，用原命令和新的输出目录重跑；
增大超时不能解决回执格式不匹配。正常执行结束还应看到 `msprof_stop_acknowledged` 和
`msprof_quit_acknowledged`，最终以 wrapper 的完整报告与导出检查为准。
目标环境需要支持 `--dynamic=on --pid` 的 msprof 和配套 CANN runtime；仅看到提示符或
“Start profiling” 日志不算采集已启动。如果安装版本的交互协议不同，请保留日志核对。

本模式通过 wrapper 启动；直接给 `run_npu` 传阶段参数会提示使用 wrapper。不要再外套进程级
msprof；不接受 `--msproftx` 或额外 `--msprof-arg`。动态采集方式见
[CANN msprof 交互式采集文档](https://www.hiascend.com/document/detail/zh/canncommercial/800/devaids/devtools/profiling/atlasprofiling_16_0016.html)，
回执语义见 [CANN runtime 的 DynProfClient 实现](https://gitcode.com/cann/runtime/blob/68752f679cfb68365e472eec38855ec4fd4721f6/src/dfx/msprof/collector/dvvp/msprof/dynamic_profiling/src/dyn_prof_client.cpp)。
同一应用允许退出交互模式后再次执行 msprof 连接，见
[CANN 动态采集的 quit 说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/Profiling/atlasprofiling_16_0016.html)。
关闭采集后 wrapper 对每个阶段目录执行 `msprof --export=on --output=... --summary-format=csv`，参数见
[msprof 离线导出文档](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0021.html)。

输出位于 `--output-dir` 下：

- `profile/msprof/<label>/`：raw 数据及导出的 `op_summary*.csv`。
- `<label>-stage-report.json`：阶段次数/范围和 token 结果。
- `<label>-stage-summary.csv`：阶段同步耗时和算子行数。
- `manifest/<label>-control.json`：应用 PID、实际 attach 命令、start/stop/quit 回执及进程退出状态。
- `manifest/<label>.json`：运行身份和最终状态；`log/msprof-<label>.log` 保留完整交互输出。

wrapper 仅在控制握手、应用、导出、窗口内 GDR 调用次数和各阶段非空算子 CSV 检查
都成功后返回 PASS。阶段报告的
`PASS_CAPTURE` 只表示阶段流程完成，预热对照仅检查该阶段 token 结果稳定，不替代
strict-greedy 正确性门禁。`profiled_elapsed_ms` 是带 profiling 开销的同步窗口时间，排除了
attach/start/stop/quit 及等待控制回执的时间；具体算子时长看 CSV，正式时延仍按 3+10 测量。
该 CLI 集成已通过主机上的交互协议与异常路径测试；真实算子采集效果须在目标 NPU/CANN 上验证。

当前参数适用于 Python NPU 的 two-pass chunk-GDR rollback 路线。
OM/C++ runner 的现有流程见 [AIR/OM 框架](QUANT_AIR_OM_FRAMEWORK.md)。

## 8. 报告门禁

`validate`：

```python
assert report["route"] == "qwen3.5-dflash-incremental-rollback"
assert report["execution_mode"] == "validate"
assert report["correctness_gate"]["status"] == "PASS"
assert report["strict_greedy_exact_match"] is True
assert report["historical_prefix_replay_during_verify"] is False
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["dflash_execution_gate"]["target_verify_calls"] > 0
assert report["draft_kv_cache_audit"]["mode"] == "upstream_equivalent_append_then_crop"
assert report["operator_fallback_enabled"] is False
```

`dflash` 单跑：

```python
assert report["execution_mode"] == "dflash"
assert report["ordinary"] is None
assert report["strict_greedy_exact_match"] is None
assert report["correctness_gate"]["status"] == "NOT_RUN_DFLASH_ONLY"
```

NPU rollback：

```python
audit = report["target_rollback_audit"]
assert audit["gdr_backend"] == "npu_chunk_gated_delta_rule_two_pass"
assert audit["custom_gdr_mtp_required"] is False
assert audit["conv_bank_backend"] == "torch_tensor_golden_on_input_device"
assert audit["kv_policy"] == "physical_provisional_writes_logical_cursor_commit"
assert audit["prefill_execution_mode"] == "block_aligned_real_token_chunks_original_gdr"
assert audit["prefill_chunk_size"] == 64
assert audit["session_invalid"] is False
```

W8A8：

```python
quant = report["target_quantization"]
assert quant["status"] == "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM"
assert quant["scheme"] == "w8a8_dynamic"
assert quant["route"] == "rollback"
assert quant["linear_topology_validation"] == "PASS_EXACT_PATH_SHAPE_BIAS"
assert quant["qlinear_count"] > 0
assert quant["embedding_lookup_failures"] == 0
```

assembly PASS 只证明装配成功；数值结论仍来自同 activation 对照、整网 token/state 门禁和真实
NPU trace。

benchmark：

```python
assert report["benchmark"]["status"] == "PASS"
assert report["benchmark"]["summary"]["count"] == 10
assert report["strict_greedy_exact_match"] is True
assert report["operator_fallback_enabled"] is False
```

同时比较 `latency_ms`、aggregate tokens/s、peak memory、全部 10 条 measurement、acceptance、
Draft/verify calls 和 Target rows。`persistent_cursor` 等是 session state，不是累计 counter。

## 9. 自动化检查

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_scheduler.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_framework_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_internal_dflash_bridge_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_helpers.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_run_npu_modes.py \
  tests/test_dflash_runtime_optimizations.py \
  tests/test_benchmark_npu.py \
  tests/test_msprof_script.py \
  tests/test_msprof_stage_script.py \
  tests/test_msprof_acknowledgements.py \
  tests/test_stage_profile.py \
  tests/test_source_lock_benchmark.py \
  tests/test_rollback_target_quant.py
```

这些仍是 CPU/reduced-shape 证据。

## 10. 真机门禁顺序

1. B=2，连续多轮，accepted 0/1，ordinary token 零差异。
2. B=`2/4/6/8/16`，accepted `0/1/K-1/K`，最后一轮动态 T。
3. cursor `62/63/64/65`，rejected KV tail 不可见并被覆写。
4. 24 层 GDR/conv、8 层 KV、feature、position 共用同一个 `a`。
5. rejection 后继续至少一个 token，比较完整 state tuple。
6. 多 prompt、多进程重复，无状态泄漏、越界或持续内存增长。
7. 无 CPU fallback；记录 device/runtime/source/op/kernel identity。
8. 正确性闭合后再做 unprofiled 3+10 和单独 msprof 归因。

若发布 W8A8，还需分别通过 Target quant preflight、W8A8 ordinary/DFlash、W8A8 与要求的 FP16
ordinary 精度门禁。FP16 rollback PASS 不能替代量化 rollback PASS。
