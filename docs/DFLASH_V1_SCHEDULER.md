# DFlash rollback：调度与 Token 验证

当前默认实现位于：

- `models/dflash_v1/dflash_rollback_decode.py`：后端无关的调度器；
- `models/dflash_v1/dflash_rollback_adapter.py`：CPU/CUDA 事务缓存和 Qwen/Draft 接线；
- `models/internal_dflash_bridge.py`：HIAI/NPU state bank 与逻辑 KV cursor；
- `models/dflash_v1/dflash_reference_decode_v1.py`：旧的完整前缀 correctness oracle，不再是
  CPU/GPU/NPU 默认验证路径。

## 1. 当前验证不再重算历史前缀

普通基线和 DFlash 都维护一个持久 Target 状态。每轮 Target 只接收：

```text
[anchor, proposal_1, ..., proposal_K]
T = K + 1，1 <= T <= 17
```

历史 prompt/committed prefix 不会再次进入正式 verify 调用。报告中的：

```text
verification_mode = incremental_transactional_rollback
historical_prefix_replay_during_verify = false
```

是这条合同的显式记录。

## 2. Bootstrap 与状态对齐

先对 prompt 做一次 prefill，读取 prompt 最后一行 logits 的 Top-1，得到 clean anchor：

```text
Target state/features 已处理：prompt
已输出但尚未作为 Target 输入处理：anchor
```

Draft 的 context feature 正好覆盖 `prompt`，Draft block 第 0 行放 anchor，后面 K 行放 mask。
这与 checkpoint 的训练数据流一致，不会把 anchor 同时放进 context feature 和 block。

## 3. 一次 T=K+1 验证怎样读 logits

Target 对整块做因果计算：

```text
input row 0 = anchor       → logits row 0 验证 proposal_1
input row 1 = proposal_1   → logits row 1 验证 proposal_2
...
input row K-1              → logits row K-1 验证 proposal_K
input row K                → logits row K 给出 all-match bonus
```

Scheduler 从左到右比较 proposal 与对应 Target Top-1，只接受最长连续匹配前缀。若首次错误位置是
`a`，则接受 `proposal[:a]`，并使用 `target_top1[a]` 作为 correction；若 K 个全部匹配，
`target_top1[K]` 是 bonus。

错误 proposal 后面的 logits 不参与决定。因果 mask 保证错误位置之前的行不会读取右侧拒绝尾部。

## 4. Commit 的 off-by-one 规则

若本轮接受了 `a` 个 proposal，Target 输入状态只提交：

```text
block[:, :a+1] = anchor + a 个已接受 proposal
```

因此状态 cursor 增加 `1+a`。本轮输出的 correction/bonus 不在这次提交中；它是下一轮尚未处理的
anchor：

```text
Target state/features 已处理：... + anchor + accepted proposals
已输出但尚未处理：correction/bonus
```

这条规则必须同时用于 GDN recurrent、causal-conv、attention KV cursor、position 和 feature
history。把 correction 提前写入状态，或只推进 `a` 行，都会产生一轮后的错位。

## 5. CPU/CUDA 如何 rollback

`FrameworkDFlashRollbackTarget` 使用一个持久 `DynamicCache`：

1. verify 前记录 full-attention KV 长度；
2. clone 每个 linear-attention 层的 conv/recurrent state 和初始化标志；
3. 一次执行完整 T 行 verify；
4. 得到 `a` 后，用 `DynamicCache.crop()` 恢复 attention KV，并恢复 GDN 快照；
5. 逐 token 重放 `anchor + accepted proposals`，最多 `K+1` 行，提交精确的普通增量状态；
6. 丢弃拒绝尾部，绝不重放历史 prefix。

提交阶段选择逐 token 而不是一次短 chunk，是 correctness-first 决定：它让提交后的 GDN 数值路径
尽量与 ordinary incremental decode 一致。代价是每轮多 `1+a` 个小调用；后续可以在设备证据证明
短 chunk 与逐 token 状态等价后再优化。

## 6. NPU 如何 rollback

HIAI route 不做 CPU/GPU 式重放：

- `npu_gated_delta_rule_mtp` 为每个 verify row 产生 recurrent state bank；
- `torch_dflash_causal_conv1d_mtp` 在输入所在 NPU device 上产生 conv state bank；它是 Torch Tensor
  golden，不是 CPU fallback；
- 第一次 verify 从 prefill 后的 scalar state 扩成 T 个槽；
- 下一轮通过 `accepted_tokens=a` 选择上一轮第 a 槽；若下一轮 T 改变，先 select 再 rebase；
- full-attention K/V 可以物理写入全部 provisional rows，但只推进 logical cursor `1+a`；下一轮从
  新 cursor 覆写拒绝尾部，并由 mask/length 保证尾部不可见。

注意 modeling 收到的 `accepted_tokens` 是“上一轮已提交槽的 selector”。当前轮的接受长度只有在
Target logits 返回后才能得到，bridge 把它保存给下一轮使用。

## 7. 两个例子

中途错误：

```text
anchor = 101
proposal = [202, 999, 888]
Target rows = [202, 303, ... , ...]

a = 1
状态提交输入 = [101, 202]
本轮输出 = [202, 303]
下一轮 anchor = 303
```

全部命中：

```text
anchor = 101
proposal = [202, 303, 404]
Target rows = [202, 303, 404, 505]

a = 3
状态提交输入 = [101, 202, 303, 404]
本轮输出 = [202, 303, 404, 505]
下一轮 anchor = 505
```

## 8. EOS 与生成上限

Draft 若在固定宽度槽中提出 EOS，EOS 后的 proposal 不再验证。提交 accepted token 或
correction/bonus 时遇到 EOS 立即停止。若 `max_new_tokens` 已用完，已计算的 bonus 不输出；状态
可以已经处理最后一个 accepted input，但不会把未输出 bonus 当作 committed token。

每轮必须推进至少一个输出 token。空 proposal 会退化成只验证 `[anchor]`，提交 anchor 输入并输出
Target correction；官方 Draft route 正常返回至少一个 proposal。

## 9. Ordinary 基线与最终门禁

`ordinary_incremental_greedy()` 使用同一个后端的普通持久缓存：prompt prefill 一次，之后每次只
输入上一个已生成 token。它不使用 Draft，也不做完整前缀重算。

`assert_exact_greedy_match()` 最终严格比较：

```text
generated_token_ids
reached_eos
stop_reason
```

没有浮点容差或文本级宽松比较。任意一个 token ID 不同即失败。旧 full-prefix sequential 实现只
用于额外 oracle/定位，不参与默认 CPU、CUDA 或 NPU rollback 报告。

## 10. 当前验证覆盖

- `tests/test_dflash_rollback_scheduler.py`：`accepted=0..K`、correction/bonus 和状态提交边界；
- `tests/test_dflash_framework_rollback.py`：CPU `DynamicCache` KV crop、GDN restore 和有界重放；
- `tests/test_internal_dflash_bridge_rollback.py`：HIAI bank select/rebase 与逻辑 KV cursor；
- `tests/test_dflash_rollback_helpers.py`：conv bank 与逐 token reference。

这些是 CPU/模拟证据。310P 上仍须验证 24 个 GDN 层、多轮、KV block boundary、现有 fused
attention 的 T=2/5/9/17 能力以及无 fallback/device identity。
