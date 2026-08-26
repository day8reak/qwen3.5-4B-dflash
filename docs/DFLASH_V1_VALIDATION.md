# DFlash rollback：验证流程与报告

当前 CPU、CUDA、NPU 共同入口是 `models/dflash_v1/run_rollback.py`；兼容入口
`python -m models.dflash_qwen_adapter_v1` 也指向它。NPU 的简化入口是
`python -m models.dflash_v1.run_npu`。

## 1. 三类结论必须分开

| 结论 | 门禁 |
|---|---|
| 生成正确性 | ordinary 与 DFlash 的 token ID、EOS、stop reason 完全相同 |
| Rollback 接线 | verify 不含历史前缀；accepted 状态、feature、KV cursor 对齐 |
| 性能/设备交付 | 目标设备无 fallback、kernel trace、延迟、显存和稳定性 |

CPU 模拟通过不能声明 310P 通过；token 一致也不能自动声明有加速收益。

## 2. 启动前检查

入口先检查：

- Target config 是锁定的 Qwen3.5-4B 32 层 hybrid text config；
- Draft 是官方 6 层、69 tensor checkpoint，并校验模型 SHA-256；
- embedding、LM head、Draft 的 device/dtype/shape 一致；
- proposal K 在 `1..16`，verify T 为 `K+1`；
- NPU 固定 FP16、EOS `248044`，并选择 package-local draft backend；
- rollback scheduler、adapter、bridge、wrapper 和 HIAI modeling 的源文件身份在运行前后不变；
- 报告不能写进 runtime package、Target 或 Draft 目录。

## 3. Ordinary 权威基线

`ordinary_incremental_greedy()` 不是完整前缀重算：

```text
prompt → 一次 persistent prefill
上一个 generated token → 一次单 token advance
重复到 EOS/max_new_tokens
```

CPU/CUDA 使用 `DynamicCache`；HIAI rollback bridge 使用自己的 32 层 persistent hybrid state。
这一路不调用 Draft，生成结果是本次请求的权威 token stream。

## 4. DFlash rollback 路线

从相同 prompt 重建一个独立 persistent session：

```text
prompt prefill → clean anchor
Target feature history + anchor/mask block → Draft K proposals
[anchor, proposals...] → 一次 Target T=K+1 verify
最长连续匹配 → accepted=a
提交 anchor + accepted proposals
correction/bonus 成为下一轮 anchor
```

CPU/CUDA verify 后恢复 KV/GDN 快照，只逐 token 重放最多 `a+1` 行。NPU 用 GDR/conv state
bank 和 logical KV cursor 提交，不重放历史 prefix。详细 off-by-one 规则见
[Scheduler 与 Token 验证](DFLASH_V1_SCHEDULER.md)。

## 5. 最终零差异门禁

`assert_exact_greedy_match()` 比较：

```text
ordinary.generated_token_ids == dflash.generated_token_ids
ordinary.reached_eos == dflash.reached_eos
ordinary.stop_reason == dflash.stop_reason
```

不存在浮点容差。只要一个 token ID 不同，报告不会写出
`strict_greedy_exact_match=true`，而是直接给出第一处分叉。

至少有一次 Draft round 才能证明真实走过 `feature → draft → T=K+1 verify → commit`。prompt
若在 bootstrap 立即 EOS，结果可保持正确，但 `dflash_execution_gate.status` 会是
`INCONCLUSIVE_NO_DRAFT_ROUND`。

## 6. 报告关键字段

正式 rollback 报告至少检查：

```text
route = qwen3.5-dflash-incremental-rollback
strict_greedy_exact_match = true
verification_mode = incremental_transactional_rollback
historical_prefix_replay_during_verify = false
operator_fallback_enabled = false       # CUDA/NPU
dflash_execution_gate.draft_round_executed = true
```

CPU/CUDA 还应检查：

```text
target_rollback_audit.mode =
  dynamic-cache-crop-linear-state-restore-bounded-token-replay
target_rollback_audit.commit_replay_scope =
  anchor_plus_accepted_prefix_only_one_token_per_call
target_rollback_audit.pending_transaction = false
```

NPU 还应检查：

```text
target_rollback_audit.enabled = true
target_rollback_audit.gdr_backend = npu_gated_delta_rule_mtp
target_rollback_audit.conv_bank_backend = torch_tensor_golden_on_input_device
target_rollback_audit.kv_policy =
  physical_provisional_writes_logical_cursor_commit
target_rollback_audit.session_invalid = false
```

`torch_tensor_golden_on_input_device` 表示 conv bank 由 Torch Tensor 分解，但输入在 NPU 时计算仍在
NPU；它不是 `allow_op_fallback` 意义上的 CPU fallback。

## 7. 统计字段

| 字段 | 含义 |
|---|---|
| `target_verify_calls` | T=K+1 Target verify 次数 |
| `target_input_tokens_recomputed` | 本路径实际送入 Target 的 prompt/增量/block 行数；rollback 中不含重复历史 prefix |
| `drafted_tokens` | Draft proposal 总数 |
| `accepted_draft_tokens` | 最长连续匹配中接受的 proposal 总数 |
| `rejected_draft_tokens` | 未提交 proposal 总数 |
| `fallback_tokens` | bootstrap、correction 或 bonus 的 Target token 数 |
| `acceptance_rate` | `accepted_draft_tokens / drafted_tokens` |
| `rollback_commit_replay_calls` | CPU/CUDA 为提交状态执行的 bounded single-token 调用数 |

接受率只描述 Draft 质量。更接近每轮推进能力的是
`mean_emitted_tokens_per_draft_round`，但两者都不能代替真实延迟测量。

## 8. 当前自动化证据

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_scheduler.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_framework_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_internal_dflash_bridge_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_helpers.py
```

覆盖接受长度 `0..K`、correction/bonus、KV crop、GDN state restore、短提交重放、NPU bank
select/rebase 和 logical cursor。它们是 reduced-shape/CPU 证据，不是 310P 全模型证据。

## 9. 310P 仍需补齐

- 24 个 GDN 层多轮执行，`accepted=0/1/K-1/K`；
- K `1/4/8/16`，以及 round 尾部 K 变化后的 bank rebase；
- committed cursor 位于 `62/63/64/65` 时拒绝尾部不可见且被正确覆写；
- 现有 `adn_fused_infer_attention` 对 T `2/5/9/17`、历史 KV 和因果 mask 的能力；
- feature 开关对 Target Top-1 零影响；
- operator 注册、runtime/device identity、无 CPU fallback、重复稳定性；
- 正确性通过后才测端到端延迟、接受长度和峰值内存。

旧 `dflash_reference_decode_v1.py` 的 sequential full-prefix 路线继续作为独立 oracle，可用于定位
整块 prefix-invariance 问题，但不应把它的报告当成当前 rollback 路径已运行。
