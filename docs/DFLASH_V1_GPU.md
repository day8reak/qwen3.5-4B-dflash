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

"$MODEL_PYTHON" -B -m models.dflash_qwen_adapter_v1 \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cuda:0 \
  --report "$RUN_DIR/dflash-v1-gpu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-gpu-smoke.log"
```

GPU 路线不要传 `--target-loader`、`--hiai-source`、`--overlay-preflight-report`、
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
print("DFLASH_V1_R6_CUDA_FRAMEWORK_REPORT_GATE_PASS")
PY
```

GPU 报告验证的是 HF/PyTorch V1 流程、特征旁路和严格 greedy token 等价。它不证明 HIAI
源码 patch、receiver 状态隔离、310P 自定义算子、310P 无 fallback 或性能收益。

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
