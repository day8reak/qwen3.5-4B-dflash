# DFlash rollback：Ascend NPU 部署与运行

本路线保留普通 HIAI modeling 和部署 wrapper，只新增独立 rollback modeling、wrapper adapter、
persistent bridge 与 DFlash package。当前需要部署侧已经安装并注册用户完成的
`npu_gated_delta_rule_mtp`。

## 1. 目标目录

```text
runtime/
└── models/
    ├── __init__.py
    ├── configuration_qwen3_5.py
    ├── export_model_wrapper_qwen3_5.py                 # 部署原文件，保留
    ├── export_model_wrapper_qwen3_5_dflash_rollback.py # 新增 adapter
    ├── modeling_qwen3_5_hiai_nd.py                     # 原文件，保留
    ├── modeling_qwen3_5_hiai_nd_dflash_rollback.py     # 新增
    ├── internal_dflash_bridge.py                       # 更新
    └── dflash_v1/                                      # 更新
```

rollback wrapper adapter 只在构造部署 wrapper 时临时把其 module-global
`Qwen3_5ForCausalLM` 指向独立 rollback 类，构造后立即恢复，并校验最终 `.model` 的准确类型。
如果部署 wrapper 在函数内部硬编码另一模型类，adapter 会 fail closed，此时必须显式修改该
wrapper 的构造器，不能静默混用。

## 2. 部署文件

下面只展示应复制的交付项；覆盖/备份动作按实际部署系统的发布规范执行：

```bash
export DFLASH_REPO=/path/to/qwen3.5-4B-dflash
export DEPLOY_ROOT=/path/to/runtime

test -f "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5.py"
test -f "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd.py"
test -f "$DFLASH_REPO/models/modeling_qwen3_5_hiai_nd_dflash_rollback.py"
test -f "$DFLASH_REPO/models/export_model_wrapper_qwen3_5_dflash_rollback.py"
test -f "$DFLASH_REPO/models/internal_dflash_bridge.py"
test -d "$DFLASH_REPO/models/dflash_v1"
```

需要交付：

```text
models/modeling_qwen3_5_hiai_nd_dflash_rollback.py
models/export_model_wrapper_qwen3_5_dflash_rollback.py
models/internal_dflash_bridge.py
models/dflash_v1/
```

不要用 rollback 文件覆盖 `modeling_qwen3_5_hiai_nd.py`，也不要覆盖部署原
`export_model_wrapper_qwen3_5.py`。checkpoint、cache、OM/ONNX、日志和报告都应位于仓库外。

## 3. 静态检查与 GDR 注册

```bash
export MODEL_PYTHON=/path/to/deployment/python
export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

"$MODEL_PYTHON" -B -m py_compile \
  "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/internal_dflash_bridge.py" \
  "$DEPLOY_ROOT/models/dflash_v1/run_rollback.py"

"$MODEL_PYTHON" -B - <<'PY'
import torch
import torch_npu

op = getattr(torch_npu, "npu_gated_delta_rule_mtp", None)
if not callable(op):
    namespace = getattr(torch.ops, "npu", None)
    op = getattr(namespace, "npu_gated_delta_rule_mtp", None)
assert callable(op), "npu_gated_delta_rule_mtp is not registered"
print("GDR_MTP_REGISTERED")
PY
```

`py_compile` 只证明语法；注册检查只证明符号可见，都不是 shape/数值/整网通过。

## 4. 当前 HIAI 状态路线

Prompt 为避免原 64-row prefill padding 写入 persistent GDN 状态，使用普通单 token path 逐行
bootstrap。每轮 verify：

```text
[anchor, K proposals]，T=K+1
GDN recurrent → npu_gated_delta_rule_mtp state bank (FP32)
GDN conv      → Torch Tensor conv bank golden（仍在输入 NPU device）
attention KV  → 逐 row npu_cache_update_ 写 provisional K/V
attention     → 现有 adn_fused_infer_attention + T 行 causal mask
commit        → GDN 记录 accepted slot；KV 只推进 logical cursor 1+a
```

下一轮从新 cursor 覆写拒绝 KV 尾部。K 在最后一轮变小时，bridge 先选择已接受槽，再把 bank
rebase 到新的 T。任一 verify 失败后 HIAI session 直接失效，防止提交部分更新的 32 层状态。

## 5. 最小 smoke

```bash
export RUN_DIR=/path/to/dflash-run
mkdir -p "$RUN_DIR"

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-rollback-npu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-rollback-npu-smoke.log"
```

将 `4096` 替换为部署配置真实值；它必须为正且能被 64 整除。入口固定 FP16、EOS 248044、
package-local NPU Draft backend和默认 factory
`models.internal_dflash_bridge:load_qwen35_rollback_target`。不要传 full-prefix `reset-hook`。

`max_new_tokens=2` 可能因 bootstrap 立即 EOS 而没有 Draft round；这时 token 结果可以正确，但
报告为 `INCONCLUSIVE_NO_DRAFT_ROUND`，需要换固定非立即结束的 prompt 才能作为 rollback smoke。

## 6. 报告门禁

```bash
"$MODEL_PYTHON" -B - "$RUN_DIR/dflash-rollback-npu-smoke.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["route"] == "qwen3.5-dflash-incremental-rollback"
assert report["classification"].startswith("HIAI/NPU rollback")
assert report["strict_greedy_exact_match"] is True
assert report["verification_mode"] == "incremental_transactional_rollback"
assert report["historical_prefix_replay_during_verify"] is False
assert report["operator_fallback_enabled"] is False
assert report["runtime_identity"]["device_type"] == "npu"
assert report["dflash_execution_gate"]["status"] == "PASS"
assert report["dflash_execution_gate"]["target_verify_calls"] > 0
audit = report["target_rollback_audit"]
assert audit["enabled"] is True
assert audit["gdr_backend"] == "npu_gated_delta_rule_mtp"
assert audit["conv_bank_backend"] == "torch_tensor_golden_on_input_device"
assert audit["historical_prefix_replay_during_verify"] is False
assert audit["session_invalid"] is False
assert audit["pending_verify_rows"] is None
print("DFLASH_ROLLBACK_NPU_REPORT_GATE_PASS")
PY
```

完整 logits 当前会被 scheduler 搬到 host 做有限 T 行 Top-1；这不是算子 fallback，但可能是明显
性能瓶颈。正确性闭合后可实现 `TargetLmHeadTop1Accept`，直接输出 Top-1、accepted 和
correction/bonus，避免 `[T,248320]` D2H。

## 7. 310P 分阶段验证

| 阶段 | 至少覆盖 |
|---|---|
| 小块接线 | K=1、连续多轮、accepted 0/1、ordinary token 零差异 |
| bank | K=1/4/8/16，accepted `0/1/K-1/K`，最后一轮动态 T |
| KV boundary | cursor `62/63/64/65`，拒绝尾部下一轮不可见且被覆写 |
| attention | T=2/5/9/17、多档历史长度、每个有效 logits row 对齐 oracle |
| feature | 八个锁定层、只追加 `a+1` 行、开关不改变 Target Top-1 |
| 故障 | 在不同 decoder 层失败，session 必须整体失效而非部分提交 |
| 稳定性 | 多 prompt、多次重复、无越界/状态泄漏/持续内存增长 |
| 身份 | 记录 device/runtime/operator package/source hashes，无 CPU fallback |

先跑 K=1，再跑 K=16；先证明 strict greedy，再测性能。现有 fused attention 或 CacheUpdate 若
能力测试通过就复用，不能仅凭接口名先重写 kernel。

## 8. 常见失败定位

- wrapper 类型失败：部署 wrapper 没有通过 module-global 类构造模型，需要显式 adapter；
- `npu_gated_delta_rule_mtp` 不可见：先修 process-local 注册，不允许静默退回 ordinary GDR；
- 第一轮就 mismatch：查 prompt 单 token bootstrap、feature、anchor 和 selector=0；
- 第二轮开始 mismatch：优先查 `1+a` cursor、上一轮 bank slot a、correction 作为下一 anchor；
- K 改变后 mismatch：查 select 后 rebase 和新 T；
- 63/64 附近失败：查逐 row CacheUpdate 的 target block/offset 与 mask；
- token 正确但很慢：profile 完整 logits D2H、逐 row CacheUpdate、conv Tensor 分解和 Draft 热点。

当前仓库内的自动测试是 CPU/reduced-shape 模拟证据；必须由上述目标设备报告才能声明 310P
rollback 路线通过。
