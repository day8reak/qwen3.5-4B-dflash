# DFlash V1：验证流程与报告解读

本文用通俗语言解释“程序到底验证了什么”。验证不是一次 ordinary/DFlash 文本比较，而是一串
按顺序执行的门禁。前一层失败，后一层不会继续。

先读：

- [整体架构](DFLASH_V1_ARCHITECTURE.md)
- [调度与 token 验证](DFLASH_V1_SCHEDULER.md)

## 1. 先区分三类问题

| 问题 | 判断标准 |
|---|---|
| 生成正确性 | ordinary 与 DFlash 的 token IDs、EOS、stop reason 是否完全相同 |
| Draft 提议质量 | proposal 被连续接受的比例和每轮实际推进 token 数 |
| 性能 | 在正确性通过后，真实设备的延迟、吞吐、内存和稳定性 |

这三类问题不能混为一谈：

```text
token 不一致              → 正确性失败
token 一致但接受率低      → 正确，但可能没有加速收益
token 一致且接受率合理    → 才值得测性能
```

## 2. 验证的实际调用顺序

总入口是 `dflash_qwen_adapter_v1.main()`，核心验证函数是
`validate_qwen35_dflash_strict_greedy()`：

```text
启动参数与设备检查
  ↓
Target config / Draft checkpoint / device / dtype 检查
  ↓
加载 Target 和 Draft
  ↓
[Gate 1] Target 状态隔离：8 次 Target 调用
  ↓
[Gate 2] Feature 零影响：2 次 Target 调用
  ↓
[Gate 3] 独立 ordinary full-prefix greedy
  ↓
[Gate 4] DFlash Target bootstrap
  ↓
[Gate 5] feature → Draft proposal → sequential Target verify
  ↓
[Gate 6] ordinary 与 DFlash token/EOS/stop 精确比较
  ↓
[Gate 7] 确认真实执行过完整 Draft round
  ↓
[Gate 8] NPU prepare/forward/synchronize 调用数对账
  ↓
再次检查运行时身份，原子写入报告
```

下面逐层解释。

## 3. Gate 0：模型和运行参数是否合法

在生成前会检查：

- Target 是 Qwen3.5-4B 对应的 32 层 hybrid text config；
- Target embedding 和 LM head shape 为 `[248320,2560]`；
- Draft config 与官方 6 层 DFlash 合同一致；
- Draft checkpoint 的 69 个 tensor 名称和 shape 正确；
- Target、Draft、embedding、LM head 的 device/dtype 一致；
- `block_size` 位于 `2..16`，对应 K 位于 `1..15`；
- NPU 只允许 FP16 和 EOS `248044`；
- CUDA/NPU 在大权重加载前先检查设备可用性。

这一层失败说明装配或参数不对，还没有进入 DFlash 算法。

## 4. Gate 1：Target 状态隔离

实现：

```text
Qwen35DFlashFullPrefixAdapter.validate_full_prefix_state_isolation()
```

它构造原前缀 P 和不同长度前缀 Q，共运行 8 次 Target：

```text
普通模式： P → P → Q(feature)  → P
feature 模式：P → P → Q(ordinary) → P
```

为什么不是只跑 `P→Q→P`？

- 第一个 `P→P` 先测设备本身的立即重复性。
- 如果立即 P→P 都不稳定，不能把差异归咎于 Q 的状态污染。
- P→P 稳定、P→Q→P 失败，才更像跨调用 KV/GDN 残留。

比较对象包括：

- ordinary P 的 logits；
- feature-mode P 的 logits；
- feature-mode P 的完整 `[1,S,20480]` features。

logits 要求 Top-1 完全一致，同时允许 dtype 对应的小浮点误差：

| dtype | rtol | atol |
|---|---:|---:|
| FP16 | `5e-3` | `5e-3` |
| BF16 | `2e-2` | `2e-2` |
| FP32 | `1e-5` | `1e-6` |

报告：

```text
bounded_full_prefix_repeatability = true
full_prefix_repeatability_audit.status = PASS_BOUNDED_P_Q_P
```

注意：这是有界行为门禁，不是每个内部 state 的真实设备 trace；后者仍可能是 `PENDING`。

## 5. Gate 2：Feature 开关零影响

实现：

```text
Qwen35DFlashFullPrefixAdapter.validate_feature_capture_zero_impact()
```

同一前缀分别运行：

```text
features=False → ordinary_logits（立即 clone）
features=True  → feature_logits（立即 clone）+ dflash_features
```

要求：

- logits shape/device/dtype 不变；
- 所有位置 Top-1 相同；
- logits 浮点差异在上述容差内；
- feature shape 为 `[1,S,20480]`。

先 clone 再运行第二次是为了防止闭源/设备运行时复用同一输出 buffer，避免第二次 forward 覆盖
第一次结果后产生假相等。

报告：

```text
feature_capture_zero_impact = true
feature_capture_audit.status = PASS_BOUNDED_ZERO_IMPACT
```

## 6. Gate 3：独立 ordinary greedy

实现：

```text
ordinary_full_prefix_greedy()
```

它从原 prompt 开始，每个 token 都执行：

```text
Target(完整 committed prefix)
→ 读取最后一行 logits Top-1
→ 提交 token
```

这一路不调用 Draft，得到：

```text
ordinary.generated_token_ids
ordinary.reached_eos
ordinary.stop_reason
```

它是本次请求的最终正确性答案。

## 7. Gate 4：DFlash bootstrap

DFlash 路线从相同 prompt 重新开始，先用 Target 生成一个 clean anchor：

```text
bootstrap = ordinary_full_prefix_greedy(max_new_tokens=1)
```

如果这个 token 已经是 EOS，后续没有 Draft round。CPU 框架可将这种短请求标为 inconclusive；
CUDA/NPU 正式验证要求至少执行一轮 Draft，所以通常需要 `max_new_tokens >= 2` 且 prompt 不应立即
结束。

## 8. Gate 5：真实执行 Draft 与 Target verify

bootstrap 后调用：

```text
dflash_full_prefix_greedy(..., verification_mode="sequential")
```

每轮数据闭环为：

```text
committed prefix
→ Target features
→ Draft K proposals
→ proposals 返回 Scheduler
→ Scheduler 逐个调用 Target 验证
→ 提交 accepted + correction/bonus
```

程序记录每轮：

```text
proposed_token_ids
target_token_ids
accepted_draft_token_ids
fallback_token_id
emitted_token_ids
```

其中 `fallback_token_id` 在这里表示 correction/bonus/普通 Target token，不是设备算子回退。

## 9. Gate 6：最终精确一致

实现：

```text
assert_exact_greedy_match(ordinary, dflash)
```

它严格比较：

1. 两路 prompt IDs 相同；
2. `generated_token_ids` 整个 tuple 完全相同；
3. `reached_eos` 相同；
4. `stop_reason` 相同。

这里没有 FP16/BF16 容差。一个 token 不同就会抛异常，并给出第一处 generated offset：

```text
ordinary/DFlash mismatch at generated offset N:
ordinary=..., dflash=...
```

只有该函数返回后，最终报告才会写：

```text
strict_greedy_exact_match = true
```

因此这个字段不是“先写 true 再希望它正确”，而是前面的硬断言成功后才会出现。

## 10. Gate 7：不能只跑 bootstrap 就假装 DFlash 通过

实现：

```text
_dflash_execution_gate()
```

它要求：

```text
draft_calls > 0
target_feature_calls > 0
target_verify_calls > 0
```

三项同时大于 0 才说明真正跑过：

```text
Target feature → Draft → Scheduler → Target verify
```

报告：

```text
dflash_execution_gate.status = PASS
dflash_execution_gate.draft_round_executed = true
```

这正是为什么加速设备 smoke 不应只生成一个 token。

## 11. Gate 8：NPU 调用计数和状态生命周期

CPU/CUDA 的默认 Transformers Target 没有 receiver state 计数；NPU facade 会记录：

```text
prepare_calls
target_forward_calls
target_forward_completions
prepare_failures
target_forward_failures
output_validation_failures
```

Bridge 还记录：

```text
full_prefix_calls
full_prefix_completions
full_prefix_failures
device_synchronizations
```

最终 `_target_forward_reconciliation()` 计算本次验证应执行的 Target 次数：

```text
10 次预解码门禁
+ ordinary Target logits 调用
+ DFlash Target logits 调用
+ DFlash Target feature 调用
```

正式 NPU 要求：

```text
prepare delta
= target forward delta
= Scheduler/Adapter 预期调用数
```

同时要求每个完成的 Bridge forward 都有一次 synchronize。这样可以发现“报告自称 reset，但某些
Target 调用实际没 prepare”或“异步 kernel 尚未结束就释放 state”的问题。

## 12. NPU 额外的配置门禁

`run_npu` 还固定：

- package-local Target loader、Bridge 和 HIAI Target；
- package-local `dflash_ascend310p_ops`；
- FP16；
- EOS `248044`；
- 禁止显式外部 Draft ops backend；
- 禁止 `allow_op_fallback`；
- `kv_cache_max_len` 为 64 的倍数；
- 运行前后源码/运行时身份不变。

这些能证明 Python 装配没有切到允许的 fallback 路线，但具体 310P 型号、底层 kernel trace 和
性能仍要用设备工具另行证明。

## 13. 最小报告应该怎么看

先看正确性，不要先看接受率：

```python
import json

with open("dflash-report.json", encoding="utf-8") as stream:
    report = json.load(stream)

assert report["strict_greedy_exact_match"] is True
assert report["feature_capture_zero_impact"] is True
assert report["bounded_full_prefix_repeatability"] is True

assert (
    report["ordinary"]["generated_token_ids"]
    == report["dflash"]["generated_token_ids"]
)
assert report["ordinary"]["reached_eos"] == report["dflash"]["reached_eos"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]

gate = report["dflash_execution_gate"]
assert gate["status"] == "PASS"
assert gate["draft_calls"] > 0
assert gate["target_feature_calls"] > 0
assert gate["target_verify_calls"] > 0
```

NPU 再加：

```python
assert report["device"].startswith("npu")
assert report["dtype"] == "torch.float16"
assert report["operator_fallback_enabled"] is False

target_integration = report["target_integration"]
isolation = target_integration["isolation"]
assert isolation["all_calls_prepared"] is True
assert isolation["prepare_failures"] == 0

reconciliation = target_integration["validation_call_reconciliation"]
assert reconciliation["matches"] is True

bridge = isolation["bridge_runtime"]
assert bridge["full_prefix_failures"] == 0
assert bridge["device_synchronizations"] == isolation["target_forward_calls"]
```

完整可复制脚本以 [CPU/Golden](DFLASH_V1_GOLDEN.md)中的当前命令为准。

## 14. 接受率字段怎么读

```text
acceptance_rate = accepted_draft_tokens / drafted_tokens
```

它只统计 Draft proposal 命中比例。每轮还会提交一个 Target correction 或 bonus，所以更接近
推进效率的字段是：

```text
mean_emitted_tokens_per_draft_round
```

举例：K=3，一轮接受 2 个 proposal 后由 Target correction 1 个 token：

```text
accepted = 2
emitted  = 3
```

接受率低但 strict greedy exact match 为 true，表示调度正确但 Draft 提议质量需要排查。

## 15. 失败时从哪里开始查

| 首个失败门禁 | 更可能的问题 |
|---|---|
| config/checkpoint/device | 路径、版本、shape、dtype、设备装配 |
| 立即 P→P | 不确定性、未初始化内存、异步同步 |
| P→P 通过、P→Q→P 失败 | KV/GDN 状态残留、fresh state 或生命周期 |
| feature zero-impact | feature collector 改变了 Target 或输出 buffer 被覆盖 |
| Draft round 没执行 | `max_new_tokens` 太小、bootstrap 立即 EOS、入口接线错误 |
| exact token gate | sequential verify、Target logits、EOS 或前缀提交逻辑 |
| token 一致但接受率低 | anchor、feature、position/mask、Draft 权重或设备数值 |
| NPU call reconciliation | 某次 forward 未 prepare、计数漏记或 Bridge 绕过 facade |

推荐原则：先修最早失败的门禁。后面的结果建立在前面的假设上，跳过早期失败去调接受率通常会
掩盖真正原因。

## 16. 各设备的证据边界

| 路线 | 能证明 | 不能替代 |
|---|---|---|
| CPU | 公共 Scheduler、Draft 结构、feature contract、strict greedy | CUDA/NPU 设备执行 |
| CUDA | 相同 PyTorch 结构的 GPU dtype/device 和完整 round | NPU HIAI Target 与状态 |
| NPU | 当前 NPU Target + Draft + Scheduler 的行为门禁 | 未采集的底层 kernel trace、性能和具体设备身份 |

CPU/GPU 是重要 golden，但 NPU 最终必须与同一个 NPU ordinary Target 比较，不能用另一个设备的
接受率替代。

当这些 correctness 门禁稳定后，下一阶段才是
[完整 DFlash 与提速路线](DFLASH_FULL_AND_PERFORMANCE_ROADMAP.md)：一次 Target 整块验证、
KV/GDN 状态事务、Draft cache 与 NPU 融合。
