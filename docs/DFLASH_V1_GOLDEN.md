# Qwen3.5-4B DFlash V1 Golden

V1 是 correctness-first 的完整前缀重算路线。本包按 vLLM 口径把 K 定义为 proposal token
数，clean anchor 不计入 K，因此支持 K=16；此时 draft query 为 1 个 anchor 加 16 个 mask，
共 17 行。r13 默认让同一个普通 target 按 proposal 逐次验证独立完整前缀，接受最长连续匹配
前缀，并产生 correction/bonus；一次 target 调用验证整块仅作为 prefix-invariance 诊断。

## 不可放宽的正确性条件

同一 target、同一 dtype 下：

- ordinary 与 DFlash 最终 token ID 完全一致；
- EOS 和停止原因完全一致；
- 开启 feature capture 前后 target logits 不变；
- 加速设备至少实际执行一个 draft、feature 和 target-verify round。

V1 不实现 KV/GDN 投机状态提交或回退；每次 target 调用都从干净状态重算完整前缀。

## Golden 分层

| 层级 | 作用 |
|---|---|
| CPU framework | 调度、权重、shape、feature 和严格 greedy 参考 |
| CUDA framework | 相同 PyTorch 路线的 GPU 分派验证 |
| NPU ordinary | HIAI target 的设备本地权威 baseline |
| NPU DFlash | 必须逐 token 匹配同一个 NPU ordinary baseline |

CPU/GPU 接受率可以诊断 draft，但不能替代真实 NPU 接受率。NPU target features、FP16、自定义
算子和状态隔离都会影响 proposal 与接受长度。

## CPU 快速运行

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B \
  -m models.dflash_v1.dflash_qwen_adapter_v1 \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cpu \
  --report /path/to/run/dflash-v1-cpu.json
```

CPU 使用 `models.dflash_v1.modeling_qwen3_5_dflash`，不需要 HIAI 文件。启动时会验证
官方草稿 config、69 个 BF16 tensor、shape、文件大小和完整 safetensors SHA-256。

## NPU 内嵌布局

```text
models/
├── modeling_qwen3_5_hiai_nd.py
├── 原 HIAI inference 文件
└── dflash_v1/
    └── 本仓库 DFlash V1 实现
```

完整直接源码检查、已实现 bridge 及命令见
[NPU_DEPLOYMENT.md](NPU_DEPLOYMENT.md)。NPU 日常运行不需要 overlay JSON：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report /path/to/run/dflash-v1-npu-smoke.json
```

## NPU 报告检查

```bash
python - /path/to/run/dflash-v1-npu-smoke.json <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["strict_greedy_exact_match"] is True
assert report["verification_mode"] == "sequential_isolated_prefix"
assert report["feature_capture_zero_impact"] is True
assert report["bounded_full_prefix_repeatability"] is True
assert report["operator_fallback_enabled"] is False
assert report["device"].startswith("npu")
assert report["dtype"] == "torch.float16"
assert report["npu_layout"] == "embedded"
assert report["request"]["eos_token_ids"] == [248044]
assert report["request"]["formal_locked_eos_token_id"] == 248044
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["ordinary"]["reached_eos"] == report["dflash"]["reached_eos"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]

preflight = report["runtime_preflight"]
assert preflight["status"] == "PASS_EMBEDDED_RUNTIME_PREFLIGHT"
assert preflight["layout"] == "embedded"
assert preflight["feature_contract_id"] == "qwen3.5-4b-dflash-hiai-feature-source-v1"
assert preflight["source_integration"] == "direct"
assert preflight["source_modified_by_runtime"] is False

isolation = report["target_integration"]["isolation"]
assert isolation["formal_npu"] is True
assert isolation["mode"] in {"receiver_reset_hook", "fresh_instance"}
assert isolation["prepare_forward_serialized"] is True
assert isolation["all_calls_prepared"] is True
assert isolation["prepare_failures"] == 0
assert isolation["full_prefix_execution_mode"] == "fresh_prefill"
bridge = isolation["bridge_runtime"]
assert bridge["prefill_alignment"] == "right_pad_s_gt_1_to_multiple_of_64"
assert bridge["call_local_state_release_barrier"] is True
assert bridge["full_prefix_failures"] == 0
assert bridge["full_prefix_completions"] == isolation["target_forward_calls"]
assert bridge["device_synchronizations"] == isolation["target_forward_calls"]

round_gate = report["dflash_execution_gate"]
assert round_gate["status"] == "PASS"
assert round_gate["draft_round_executed"] is True
assert round_gate["draft_calls"] > 0
assert round_gate["target_feature_calls"] > 0
assert round_gate["target_verify_calls"] > 0
print("DFLASH_V1_NPU_FRAMEWORK_REPORT_GATE_PASS")
PY
```

该检查仍不能替代具体 310P 型号、kernel trace、无 fallback 和性能证据。

## 扩大验证

最小 smoke 通过后：

1. 增加 prompt 长度并覆盖 64-token prefill 分块边界；
2. 改为 `max_new_tokens=32`、`max_draft_tokens=8`，确认稳定后再测 16；
3. 记录每轮 proposal、接受长度和 correction；
4. 用 profiler 确认 target、draft 和所有中间 tensor 留在 NPU；
5. 最后才测延迟、吞吐和加速比。

为了让每次运行固定输入，推荐把问题保存为 UTF-8 文本并使用
`--prompt-file /path/to/prompt.txt --prompt-mode chat`。入口会在本地加载 Target tokenizer，
套用 Qwen chat template，默认启用 thinking，并打印 ordinary Target 与 DFlash 的解码文本。
需要非 thinking 对照时显式加 `--no-enable-thinking`。不要用来源不明的裸 token ID 作为
接受率 workload；它可能并不对应正常对话前缀。
