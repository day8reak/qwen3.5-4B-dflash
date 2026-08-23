# Ascend NPU 部署与运行

本流程采用直接源码集成，不运行 patch，也不要求手工修改模型代码。

量化 Target 实验请先阅读
[量化版运行与排错指南](DFLASH_V1_QUANT_RUNBOOK.md)，再按需查阅
[DFlash V1 NPU Quant Target 适配](DFLASH_V1_NPU_QUANT_DESIGN.md)。`quant` 分支复用现有
`QLinear`、量化 artifact 和 Target 自定义算子，不新增量化 kernel；不传量化参数时仍走
`v1-r1` 的 FP16 路径。

量化首次接入应先运行 `models.dflash_v1.preflight_target_quant`。它只加载 Target，不读取
Draft checkpoint，并在完整 DFlash 前检查转换覆盖、input provider、feature 零影响和有界
full-prefix 状态隔离；具体命令见量化适配文档。

## 1. 路径变量

```bash
export DFLASH_REPO=/path/to/qwen3.5-4B-dflash
export DEPLOY_ROOT=/path/to/qwen35-runtime
export MODEL_PYTHON=/path/to/python
```

- `DFLASH_REPO`：本仓库根目录。
- `DEPLOY_ROOT`：运行工程根目录，下面已有 `models/` 和模型 wrapper。
- `MODEL_PYTHON`：该工程实际使用的 Python 3.10。

先核对：

```bash
set -euo pipefail
test -f "$DFLASH_REPO/models/modeling_qwen3_5_hiai_nd.py"
test -f "$DFLASH_REPO/models/dflash_v1/run_npu.py"
test -f "$DEPLOY_ROOT/models/configuration_qwen3_5.py"
test -f "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5.py"
test -x "$MODEL_PYTHON"
"$MODEL_PYTHON" -V
```

## 2. 目标目录

```text
qwen35-runtime/
└── models/
    ├── __init__.py
    ├── configuration_qwen3_5.py
    ├── export_model_wrapper_qwen3_5.py
    ├── modeling_qwen3_5_hiai_nd.py
    ├── internal_dflash_bridge.py
    ├── 其余运行文件
    └── dflash_v1/
```

`modeling_qwen3_5_hiai_nd.py` 保持原 HIAI 算子和默认 Tensor ABI，只增加可选 feature
返回。CPU/CUDA 的 `modeling_qwen3_5_dflash.py` 不可覆盖这个文件。

停止推理进程并按部署规范保留回退副本后，部署三个交付项：

```bash
set -euo pipefail

test ! -e "$DEPLOY_ROOT/models/dflash_v1.r1.new"
cp -a "$DFLASH_REPO/models/dflash_v1" \
  "$DEPLOY_ROOT/models/dflash_v1.r1.new"

# 首次部署时，目标 dflash_v1 不应存在；升级时先按部署规范将旧目录改名留存。
test ! -e "$DEPLOY_ROOT/models/dflash_v1"
mv "$DEPLOY_ROOT/models/dflash_v1.r1.new" \
  "$DEPLOY_ROOT/models/dflash_v1"

install -m 0644 "$DFLASH_REPO/models/internal_dflash_bridge.py" \
  "$DEPLOY_ROOT/models/internal_dflash_bridge.py"
install -m 0644 "$DFLASH_REPO/models/modeling_qwen3_5_hiai_nd.py" \
  "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd.py"
```

不要覆盖 `models/__init__.py`、`configuration_qwen3_5.py`、
`export_model_wrapper_qwen3_5.py` 或其他运行文件。

`-B` 不能忽略旧字节码。部署后先只读检查：

```bash
find "$DEPLOY_ROOT/models" -maxdepth 2 -type f \
  -name 'modeling_qwen3_5_hiai_nd*.pyc' -print
```

若有输出，只按部署规范处理列出的精确文件，或改用干净副本；不要递归清理整个工程。

## 3. 源码合同检查

```bash
set -euo pipefail
export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DEPLOYED_HIAI_SOURCE="$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd.py"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B -m py_compile "$DEPLOYED_HIAI_SOURCE"
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_hiai_feature_check \
  --source "$DEPLOYED_HIAI_SOURCE"
```

通过时状态为 `PASS_DIRECT_SOURCE_CONTRACT`。检查内容包括：

- `output_dflash_features` 默认关闭；
- 在层 `1,5,9,13,17,21,25,29` 的层后、最终 norm 前采集；
- 默认返回 logits Tensor；开启时返回 `(logits, dflash_features)`；
- feature shape 为 `[B,S,20480]`；
- 不依赖 patch 或 ModelOutput sidecar。

## 4. Wrapper 绑定检查

`models/export_model_wrapper_qwen3_5.py` 必须让
`Qwen3_5ForCausalLMWrapper.model` 使用同包的
`models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM`。运行入口会校验实际类型；若不符会在
加载后立即失败，不会静默混用两份 target。

`--kv-cache-max-len` 必须使用部署配置中的实际值，并且能被 64 整除。Bridge 的 Target/Draft
激活、logits 和 feature 边界固定为 FP16；`quant` 分支可在该边界内选择原 FP16 Target，或让
Target 的既有 `QLinear` 执行 W8A8 dynamic linear。加载后会用该值重建所有 full-attention
层的 block table；
重建层数或 shape 不一致会在 draft 加载前失败。

`v1-r1` 的 bridge 对 `S=1` 保持单 token 路线；对 `S>1` 则把输入张量右补齐到下一个 64-token
边界，使 GDN 的物理输入与 `chunk_size=64` 对齐。`allQLen` 仍传真实长度，返回值也在 bridge
内截回真实 token 行，因此补齐 token 不进入 DFlash 前缀。每次 target forward 后还会同步
NPU，等本次异步 kernel 全部完成后才释放调用级 KV/GDN state。该同步是 V1 correctness 路线
的生命周期门禁，不是性能实现。

## 5. 最小 NPU smoke

先确认普通文本推理仍能正常生成，再运行：

```bash
set -euo pipefail

export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RUN_DIR=/path/to/dflash-run
mkdir -p "$RUN_DIR"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-v1-npu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-npu-smoke.log"
```

将 `4096` 替换为部署配置的真实值。入口自动固定 FP16、EOS `248044`、package-local NPU
backend、无 CPU fallback，并在运行前后复核源码。chat prompt 默认启用 thinking；需要复现
非 thinking 输入时显式传 `--no-enable-thinking`。

target 加载后、draft 构造前会检查当前设备可用内存。草稿参数 FP16 约需 1.18 GiB，另预留
512 MiB 安全空间；不足时会直接报告 free/required，而不是等分配失败。这个门禁只能发现明显
不足，正式长序列仍需为 target、feature、logits 和 workspace 留出更多空间，并使用没有其他
进程占用的设备。

## 6. 通过条件

```text
strict_greedy_exact_match = true
verification_mode = sequential_isolated_prefix
feature_capture_zero_impact = true
bounded_full_prefix_repeatability = true
draft_calls > 0
target_feature_calls > 0
target verify calls > 0
operator_fallback_enabled = false
device = npu:0
dtype = torch.float16
runtime_preflight.source_integration = direct
runtime_preflight.source_modified_by_runtime = false
target_integration.isolation.bridge_runtime.prefill_alignment = right_pad_s_gt_1_to_multiple_of_64
target_integration.isolation.bridge_runtime.call_local_state_release_barrier = true
```

门禁先执行 `P → P` 对照，再执行异长 `P → Q → P`。若前者失败，不能只因 Top-1 相同就
放宽阈值：先确认报告中的 bridge physical length 已对齐到 64、每次完成调用都有一次 device
synchronization。若仍失败，再依据新增的 max/mean error、RMSE、relative RMSE 和 cosine 判断
是稀疏数值异常还是整段漂移；只有后者失败时，再检查 fresh state shape、
`kv_cache_max_len`、block table 和 full-prefix prefill。

最小 smoke 通过后再使用：

```text
--max-new-tokens 32 --max-draft-tokens 16
```

真实 NPU 接受率、无 fallback、显存和性能只能由目标设备运行确认。

## 7. 接受率低时的分层诊断

不要先根据 `accepted / proposed` 百分比修改草稿模型。第一步应确认两条 Target 路径对同一
前缀是否等价：

- 正常增量路径：prompt prefill 一次，随后复用 KV/GDN state 单 token decode；
- DFlash V1 路径：每次都用 fresh state 重算完整前缀。

仓库提供只读诊断入口。它不会修改权重或部署源码，默认也不会在终端或 JSON 中写出
prompt、生成 token ID：

```bash
set -euo pipefail

export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RUN_DIR=/path/to/dflash-run
export PROMPT_FILE=/path/to/prompt.txt
mkdir -p "$RUN_DIR"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.diagnose_acceptance \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt-file "$PROMPT_FILE" \
  --prompt-mode chat \
  --enable-thinking \
  --device npu:0 \
  --dtype float16 \
  --eos-token-id 248044 \
  --target-parity-decode-steps 4 \
  --acceptance-rounds 16 \
  --verification-mode sequential \
  --proposal-counts 1,4,8,16 \
  --trace-draft-layers \
  --report "$RUN_DIR/dflash-v1-acceptance-diagnosis.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-acceptance-diagnosis.log"
```

把 `4096` 替换为正常推理配置的真实 `kv_cache_max_len`。诊断默认校验完整草稿权重
SHA-256；只有临时缩短排查时间时才使用 `--no-verify-draft-sha256`，正式结论仍应打开校验。
`PROMPT_FILE` 按 UTF-8 读取整个文件；`chat` 会套本地 Qwen chat template。文件若已经包含完整
模板则用 `--prompt-mode raw`。默认 thinking 与公开 DFlash benchmark 一致；非 thinking A/B
显式加 `--no-enable-thinking`。终端会直接打印 Target 续写文本、最大 K 的逐轮接受长度以及
early / middle / late 三段均值；JSON 默认仍不保存明文 token ID。

### 为什么是 K=1、4、8、16

这里的 `K` 是草稿 proposal/mask token 数。每个 draft query 还包含 1 个已经由 Target
确认的 anchor：

```text
query rows = 1 个 anchor + K 个 proposal
K=1 / 4 / 8 / 16 对应 query rows=2 / 5 / 9 / 17
```

本包统一采用 vLLM `num_speculative_tokens` 口径：K 只计算 proposal，anchor 是额外一行。
因此官方配置值 16 对应最大 K=16，而不是 15。选择 1/4/8/16 可以观察单 token、小 block、
中 block 和最大 proposal 档位，同时保持全部档位语义一致。K=8 已足以排查低接受率，K
口径本身不能解释“不同设备/精度都只接受相近 token 数”的现象。

### 结果怎么读

先看 `Target 增量 vs full-prefix`：

- `FAIL_TOP1_DIVERGENCE`：先停止解释接受率。说明 fresh full-prefix bridge 和正常增量推理
  已经产生不同的 Target token；优先核对 fresh KV/GDN state、position、cache position、
  `allQLen` 和 causal mask。
- `PASS_TOP1_WITH_NUMERIC_DIFFERENCE`：token 相同但数值不完全相同。查看 JSON 中最早异常
  的 `feature_layers`；它按层 `1,5,9,13,17,21,25,29` 分开给出 max/mean error、RMSE 和
  cosine。
- `PASS_BITWISE_EQUAL`：两条 Target 路径在本次有限前缀上逐 bit 相等，可以继续解释
  draft 接受率。

CPU/CUDA 现在也执行同一类探针：先 `use_cache=True` prefill，再只喂单 token 并复用
DynamicCache，随后逐前缀和 `use_cache=False` 重算结果比较。这样 GPU token 一致不再被误当成
feature 一致；报告同时给出 `max_feature_relative_rmse` 与最小 cosine。

再看每个 K：

- `first_proposal_accuracy`：每轮第一个草稿 token 的命中率；K=1 也很低时，优先查
  feature、草稿权重和 FP16 draft backend，而不是长 block 调度。
- `mean_accepted_draft_tokens`：每轮平均接受的草稿 token 数。
- `mean_theoretical_emitted_per_verify`：接受 token 加一个 Target correction/bonus，较接近
  每次 verify 能推进多少 token 的口径，也是应与公开 accept length 对照的字段。
- `full_block_accept_rate`：整块全部接受的轮次比例。
- `phase_metrics_by_proposal_count`：把实际完成轮次等分为 early / middle / late；配合终端的
  `逐轮接受长度` 判断后段回升究竟是否稳定存在。

需要进一步区分草稿 backend 时，加 `--shadow-torch-ops`。它在同一 NPU、同一组输入和
同一份权重上，把当前分解 backend 与 `TorchDFlashOps` 做一次 shadow 比较；如果该环境的
NPU SDPA 不支持，会明确显示 `SKIPPED_TORCH_OPS_UNSUPPORTED`，不会伪装成通过。

`--trace-draft-layers` 会为每个 round/K 记录无明文 SHA-256 和数值健康信息，边界依次包括：

```text
8 层 target feature
→ fc / hidden_norm
→ noise embedding / position / rotary
→ draft layer 0..5
→ final norm / draft hidden
→ proposal / verifier Top-1
```

这会增加设备同步和 D2H 诊断开销，只用于定位，不用于性能计时。报告默认仍不含 prompt 或
生成 token 明文。

### 与 GPU FP16 报告逐轮比较

必须使用同一 prompt、同一 `proposal-counts` 和同一轮数。先按 GPU 文档生成带
`--trace-draft-layers` 的 FP16 报告，再在 NPU 命令中增加：

```text
--compare-report "$RUN_DIR/gpu-fp16-diagnosis.json"
```

结果解释：

- `MATCH_ON_ALL_RECORDED_ROUNDS`：相同 dtype 下，公共 round 的输入、各层和 token 指纹均
  一致；NPU 独有草稿算子不再是当前首要嫌疑。
- `DIVERGED_IN_FIRST_ROUND`：首轮没有历史 draft cache，优先看报告中的
  `first_divergence.boundary`，排查输入、feature、position、dtype 或对应草稿层。
- `DIVERGED_AFTER_FIRST_ROUND`：首轮一致而后续分叉，优先查 committed prefix、Target
  replay 与状态/缓存演进。
- `*_TOKEN_LEVEL_ONLY`：至少一份报告没开层级追踪，结论只覆盖 proposal/verifier token，
  不能定位内部首个浮点边界。

DFlash V1 每轮只有一次并行 draft forward；不要为提高接受率增加逐个 mask 替换和重复
draft 的循环，那会改变算法。

`v1-r1` 的正确性决策默认是 sequential：proposal `i` 只用
`committed_prefix + 已接受的 proposal[:i]` 调一次 fresh target。工具仍额外执行一次
vectorized target 以记录 `vectorized_prefix_invariance`，但它不参与接受决策。若该字段为
`FAIL_PREFIX_ROW_DIVERGENCE`，说明更长输入改变了较早行的 Target Top-1；这能解释旧版
BF16 中途 strict-greedy mismatch，不能据此判定草稿权重错误。

### 导出单轮 oracle 输入

需要把当前实现与独立官方实现逐层对照时，可额外传：

```text
--oracle-bundle "$RUN_DIR/first-round-k16.safetensors"
```

文件包含首轮最大 K 的 `target_hidden`、noise embedding、position、projection、rotary、六层
输出、final hidden 和 Top-1，不含模型权重。它含有原始中间 tensor，可能承载输入相关信息，
只放在受控 `RUN_DIR`，不要提交 Git。独立 oracle 必须加载同一 checkpoint 和 target
embedding/LM head；这个 bundle 本身不是官方一致性 PASS。

诊断只覆盖有限轮次，不能单独证明整网性能或所有长度下的状态正确性。建议先用 16 轮定位，
再把 `--acceptance-rounds` 提高到 64 或更多确认趋势。
