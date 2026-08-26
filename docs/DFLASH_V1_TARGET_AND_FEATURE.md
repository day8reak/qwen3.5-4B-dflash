# DFlash rollback：Target、Feature 与状态所有权

## 1. Target 的职责

Target 同时提供：

1. ordinary incremental 的权威 logits；
2. DFlash T=K+1 verify 的每行 logits；
3. 八个 decoder 层输出拼接成的 Draft feature；
4. 与 verify 配套的 KV/GDN provisional state transaction。

接受规则仍在 Scheduler：Target 不直接决定哪些 proposal 输出。

## 2. CPU/CUDA Target

`models/dflash_v1/modeling_qwen3_5_dflash.py` 在 Transformers Qwen3.5 text decoder 上增加 opt-in
feature collector。`FrameworkDFlashRollbackTarget` 持有一个 `DynamicCache`：

- ordinary：prompt prefill 一次，之后单 token advance；
- rollback verify：保存 round-start KV/GDN，执行 T 行；
- commit：恢复 round-start，逐 token 重放 `anchor + accepted`；
- abort：恢复后销毁 transaction，禁止继续使用半提交状态。

`DynamicCache.crop()` 只处理 attention KV，因此 conv/recurrent tensor、初始化标志和
`has_previous_state` 必须另行 snapshot/restore。恢复 inference tensor 时也必须处于
`torch.inference_mode()`。

## 3. HIAI/NPU Target

普通文件保持：

```text
models/modeling_qwen3_5_hiai_nd.py
```

rollback 独立文件是：

```text
models/modeling_qwen3_5_hiai_nd_dflash_rollback.py
```

`accepted_tokens=None` 时仍走普通单 token/chunk GDN。传 `accepted_tokens: INT8[B]` 时，输入必须
为 `[anchor, proposals...]`，T<=17，并进入：

```text
GDN recurrent bank → npu_gated_delta_rule_mtp
GDN conv bank      → torch_dflash_causal_conv1d_mtp
full-attention KV  → provisional paged-cache rows
```

Bridge 持有跨轮 state。prompt 为避免 64-row padding 污染状态，逐 token bootstrap；verify
从 logical cursor 开始构造 position、cache position、mask 和 `allQLen`。

## 4. Feature 捕获合同

固定层号：

```text
1, 5, 9, 13, 17, 21, 25, 29（0-based）
```

捕获点是每层 decoder 输出后、final norm 前：

```text
8 × [B,S,2560] → [B,S,20480]
```

collector 检查层数、顺序、batch/sequence、dtype/device 和 hidden width，并 detach+clone，防止
后续原地计算覆盖 feature。

## 5. Feature 生命周期

Bootstrap 后 feature history 覆盖 prompt，而 anchor 尚未作为 Target 输入。每轮 verify 得到 T 行
feature；接受 a 个 proposal 后只追加：

```text
verify_features[:, :a+1]
```

也就是 anchor 和 a 个 accepted proposal。Correction/bonus 是下一轮 anchor，不在本轮追加。
因此进入下一次 Draft 前始终满足：

```text
feature_history_length == committed_token_length - 1
```

## 6. Full-attention KV

CPU/CUDA 将 speculative KV append 到 `DynamicCache`，之后 crop 回 round start，再 replay accepted
输入。NPU 可以保留物理 provisional K/V：commit 只推进 logical cursor `1+a`，mask 和有效长度
不得读取拒绝尾部，下一轮从新 cursor 覆写。

必须验证 cursor 位于 62/63/64/65 时的 block crossing。当前 NPU correctness fallback 对每个
verify row 分别调用已有 `npu_cache_update_`，性能版是否需要 `CacheUpdateMTP` 由真机能力和
profiling 决定。

## 7. GDN state

CPU/CUDA snapshot scalar conv/recurrent state，commit 使用普通单 token 路径重建。NPU 为每个
linear-attention 层保存：

```text
conv_state_bank      [B,T,8192,4]       FP16
recurrent_state_bank [B,T,32,128,128]   FP32
```

bank slot i 表示执行 verify input rows `0..i` 后的状态。当前轮接受 a 个 proposal时，第 a 槽就是
`anchor + a proposals` 后的已提交状态；下一轮用 accepted selector a。

## 8. 正确性门禁

- ordinary incremental 与 rollback 最终 token/EOS/stop 零差异；
- feature-enabled verify 的 Target Top-1 不得偏离 ordinary 权威流；
- accepted `0/1/K-1/K` 后再跑下一 token，与普通增量状态对比；
- correction/bonus 不得提前进入 cache 或 feature；
- 任一层失败不得留下可继续使用的部分 state；
- NPU 必须记录真实 device/runtime/operator identity 并禁用 CPU fallback。

自动化测试覆盖 framework restore 和 bridge state machine，但当前没有 310P 全模型证据。剩余
算子 ABI 见[rollback 算子分析](DFLASH_ROLLBACK_OPERATOR_ANALYSIS.md)。
