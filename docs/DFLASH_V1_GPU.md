# DFlash V1 GPU 运行说明

该路线用于在 NVIDIA CUDA GPU 上验证完整 V1 全前缀流程。它使用交付的
`modeling_qwen3_5_dflash.py` 目标模型旁路和 `TorchDFlashOps` 草稿原语；草稿 attention 由
PyTorch CUDA SDPA 执行。它不加载 HIAI receiver、不调用 310P 自定义算子，也不能作为
Ascend 310P 通过证据。

## 前提

- Python 环境使用 `transformers==5.14.1`；
- PyTorch 必须是 CUDA build，且 `torch.cuda.is_available()` 为真；
- target 和官方 `z-lab/Qwen3.5-4B-DFlash` 草稿权重均在本地；
- 推荐先用 FP16。BF16 只有在 `torch.cuda.is_bf16_supported()` 为真时允许；
- 纯权重约 10 GiB，实际还需要 logits、激活和 CUDA workspace。建议至少 16 GiB 显存，
  24 GiB 更稳。该数值是容量规划估计，不是本包实测峰值。

开发环境可能没有 NVIDIA GPU，因此真实权重 GPU 结果以运行服务器上的报告为准。

## 最小 `2 + 1` smoke

从解压后的 `qwen3_5` 根目录运行：

```bash
set -euo pipefail
: "${TARGET_DIR:?}" "${DRAFT_DIR:?}" "${RUN_DIR:?}"
MODEL_PYTHON="${MODEL_PYTHON:-python}"
mkdir -p "$RUN_DIR"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
export PYTHONPATH="$PWD"

"$MODEL_PYTHON" -B -m models.dflash_v1.dflash_qwen_adapter_v1 \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cuda:0 \
  --report "$RUN_DIR/dflash-v1-gpu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-gpu-smoke.log"
```

GPU 路线不要传 `--target-loader`、`--hiai-source`、`--target-factory`、`--reset-hook`、
`--ops-backend` 或 `--allow-op-fallback`。默认 target 使用包内 feature-enabled HF 实现，默认
draft backend 报告为 `torch_cuda`。CUDA 不可用时会在 1.27 GB 草稿权重哈希之前失败。

`max_new_tokens=2` 若第一 token 就遇到 EOS，会因没有执行完整 draft/feature/verify round 而
判为不充分；换一个固定、不会立即 EOS 的 prompt 重试，不能跳过该门禁。

## 报告核对

至少检查：

```bash
python - "$RUN_DIR/dflash-v1-gpu-smoke.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["classification"] == "CUDA/framework full-prefix validation"
assert report["runtime_identity"]["device_type"] == "cuda"
assert report["device"].startswith("cuda")
assert report["dtype"] == "torch.float16"
assert report["ops_backend"] == "torch_cuda"
assert report["operator_fallback_enabled"] is False
assert report["strict_greedy_exact_match"] is True
assert report["verification_mode"] == "sequential_isolated_prefix"
assert report["feature_capture_zero_impact"] is True
assert report["bounded_full_prefix_repeatability"] is True
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["ordinary"]["reached_eos"] == report["dflash"]["reached_eos"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]
assert report["dflash_execution_gate"]["status"] == "PASS"
assert report["dflash_execution_gate"]["draft_round_executed"] is True
assert report["dflash_execution_gate"]["draft_calls"] > 0
assert report["dflash_execution_gate"]["target_feature_calls"] > 0
assert report["dflash_execution_gate"]["target_verify_calls"] > 0
print("DFLASH_V1_CUDA_FRAMEWORK_REPORT_GATE_PASS")
PY
```

GPU 报告验证的是 HF/PyTorch V1 流程、特征旁路和严格 greedy token 等价。r13 默认对每个
proposal 单独执行完整前缀 target 校验；这条 correctness 路线不把“同一个更长 target 输入中
较早 logit 行不变”当作前提。它不证明 HIAI
直接源码集成、receiver 状态隔离、310P 自定义算子、310P 无 fallback 或性能收益。

## CUDA 环境快速检查

```bash
"$MODEL_PYTHON" - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda build:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
PY
```

只有 `cuda available: True` 后，前面的真实权重 smoke 才能用于判断 CUDA 路线是否跑通。

## 接受率诊断：先做 FP16/BF16 A/B

GPU 和 NPU 都使用 FP16 时结果相同，只能降低设备独有问题的优先级，不能排除 FP16 本身。
在 `torch.cuda.is_bf16_supported()` 为真时，对同一 prompt、同一 K 和同一轮数分别运行：

```bash
set -euo pipefail
: "${TARGET_DIR:?}" "${DRAFT_DIR:?}" "${RUN_DIR:?}" "${PROMPT_FILE:?}"
MODEL_PYTHON="${MODEL_PYTHON:-python}"
mkdir -p "$RUN_DIR"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
export PYTHONPATH="$PWD"

for DTYPE in float16 bfloat16; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.diagnose_acceptance \
    --target-dir "$TARGET_DIR" \
    --draft-dir "$DRAFT_DIR" \
    --prompt-file "$PROMPT_FILE" \
    --prompt-mode chat \
    --enable-thinking \
    --device cuda:0 \
    --dtype "$DTYPE" \
    --eos-token-id 248044 \
    --acceptance-rounds 16 \
    --verification-mode sequential \
    --proposal-counts 1,4,8,16 \
    --trace-draft-layers \
    --report "$RUN_DIR/gpu-$DTYPE-diagnosis.json" \
    2>&1 | tee "$RUN_DIR/gpu-$DTYPE-diagnosis.log"
done
```

CUDA 路线不需要 `--kv-cache-max-len`，也不传 NPU loader/factory/backend 参数。BF16 不支持
时命令会在加载大权重之前拒绝，不能把跳过 BF16 写成数值等价。`PROMPT_FILE` 是 UTF-8
纯文本；`chat` 会套用本地 Qwen chat template，若文件本身已经是完整模板文本则改成
`--prompt-mode raw`。终端会打印 Target 续写文本、最大 K 的逐轮接受长度，以及 early / middle /
late 三段均值。

然后直接比较两个已有报告，不再重复加载权重：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.diagnose_acceptance \
  --compare-reports \
    "$RUN_DIR/gpu-float16-diagnosis.json" \
    "$RUN_DIR/gpu-bfloat16-diagnosis.json" \
  --report "$RUN_DIR/gpu-bfloat16-vs-float16.json"
```

此前 BF16 在生成中途出现 ordinary/DFlash token mismatch 时，应先看报告里的
`vectorized_prefix_invariance`：若它失败而 sequential 决策正常，问题是整块 target 验证对
序列长度/kernel 选择过敏，不是 BF16 draft 本身。跨 dtype 时浮点 tensor 的原始 SHA 必然
不同，因此工具只比较 prefix、position、proposal、
verifier token 和接受率指标，不会把 BF16/FP16 的浮点 hash 差异误报成某层实现错误。重点看：

- `K=1 first_proposal_accuracy`；
- `mean_theoretical_emitted_per_verify`；
- `metric_deltas_by_proposal_count`；
- proposal 首次发生变化的 round。

如果 BF16 明显恢复，优先定位低精度边界；如果同 dtype GPU/NPU 的逐轮层级指纹全部相同，
则先查共享的草稿实现、调度或评测 workload，而不是 NPU 独有算子。

一条短 prompt 不能代表官方多数据集平均接受长度。官方公开口径是
`completion_tokens / spec_verify_ct`；本包对应优先看 `mean_theoretical_emitted_per_verify` 或
主运行报告的 `mean_emitted_tokens_per_draft_round`，而不是 `accepted / proposed`。定位时先
用固定 prompt 找首个分叉，之后
再用多条代表性 prompt 汇总 `emitted/verify` 分布。若 early 低而 late 高，换至少三类 prompt：
若上升总绑定相同绝对 round，优先查首轮状态/feature；若上升跟随文本进入稳定句式，通常包含
workload 难度因素。DFlash V1 是每轮一次并行 block 预测，不要加入逐 mask 迭代替换。

最小 smoke 也可把 `--prompt` 换成 `--prompt-file "$PROMPT_FILE"`。两种文本输入默认都在
本地套用 Qwen chat template，默认启用 thinking，并在终端及 JSON 报告中输出 ordinary
Target 与 DFlash 的解码续写；非 thinking A/B 显式加 `--no-enable-thinking`。报告不会保存
prompt 明文。本包统一使用 vLLM proposal-count 口径，anchor 不计入
K，所以 proposal K 最大为 16，K=16 时 draft query 为 17 行。
