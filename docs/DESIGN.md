# 设计：官方 Qwen3.5-4B MTP 的可移植实现

## 1. 语义边界

本 PoC 先支持 text-only、batch 1、greedy。主模型始终是唯一的 verifier；draft
只提供 proposal。严格 greedy 的不可放宽门禁是：

```text
ordinary token IDs == MTP token IDs
ordinary EOS/stop reason == MTP EOS/stop reason
```

采样模式需要 proposal/target 概率、随机数和残差分布验收，不在本版范围内。

## 2. 官方 drafter 数据流

对于已提交 token `x[0..n]`，MTP 的输入有一个固定 shift：

```text
input token:       x[1], x[2], ..., x[n]
hidden source:     h[0], h[1], ..., h[n-1]
```

其中 `h[i]` 是主模型最后一层、最终 RMSNorm 后的 hidden。最后一行 MTP 输出预测
`x[n+1]`。继续产生第二个 draft 时，输入是第一个 draft token，hidden source 是
上一 MTP step 的输出 hidden。

每一行执行：

```text
RMSNorm(token embedding) --+
                              concat -> mtp.fc -> full attention -> SwiGLU MLP
RMSNorm(hidden source) ----+                              -> mtp.norm
                                                           -> tied LM head Top1
```

RMSNorm 必须使用 Qwen3.5 的 delta 参数语义：有效 scale 是 `1 + weight`。

## 3. 严格 speculative acceptance

给定 proposals `d1..dK`，主模型一次计算 `prefix + d1..dK`，得到 K+1 个
target Top1：

- 从头接受连续相同的 draft；
- 第一个不同时丢弃该位置及之后所有 draft；
- 输出 target 在该位置的 correction；
- 若 K 个全部接受，再输出 target 的第 K+1 个 token。

CPU 实现按完整 token 序列计算 hidden；相同序列允许 memoize，任何 token 不同都会
使用不同 cache key。因而拒绝候选不会进入下一轮 state。目标 runtime 后续可把这种
全序列语义替换为事务式增量 cache，但必须覆盖 `accepted=0..K` 的 state 测试。

## 4. 两级替换接口

粗粒度接口：

- `MainBackend.evaluate(input_ids, top1_positions)`；
- `DraftBackend.propose(prefix_ids, main_hidden, max_draft_tokens)`。

细粒度 MTP ops：

- `rms_norm`；
- `linear`；
- `attention`；
- `swiglu`；
- `top1`。

内部框架可实现粗粒度 backend；PyTorch/NPU extension 可实现细粒度 ops。target 模式
必须 strict，缺少任一 op 时应失败，不能自动退回 CPU。
