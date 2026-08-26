# DFlash V1：Draft 模型详细实现

本文只讲 6 层 DFlash Draft：输入从哪里来、为什么有 anchor 和 mask、6 层怎样产生 K 个
proposal，以及 CPU/GPU/NPU 后端怎样复用同一模型结构。

先读整体关系：[DFlash V1 整体架构](DFLASH_V1_ARCHITECTURE.md)。

## 1. Draft 不是另一个完整语言模型

Draft 不直接接收 prompt 文本，也不单独决定最终 token。它需要两类输入：

```text
Target 的 8 层 context feature
当前 clean anchor + K 个 mask token 的 embedding
```

输出只是候选：

```text
proposal IDs [1,K]
```

proposal 随后必须回到 Scheduler，由 Target 逐个验证。

## 2. 固定结构

实现位于 `models/dflash_v1/modeling_dflash.py`，配置合同位于 `dflash_config.py`。

| 项目 | 固定值 |
|---|---:|
| hidden size | 2560 |
| intermediate size | 9216 |
| Draft decoder 层数 | 6 |
| attention heads / KV heads | 32 / 8 |
| head dim | 128 |
| Target feature 层数 | 8 |
| Target feature width | 20480 |
| 官方 block_size | 16（含 anchor） |
| 最大 proposal 数 K | 15 |
| mask token ID | 248077 |
| vocab size | 248320 |
| checkpoint tensor 数 | 69 |

Draft 不是临时造的小模型。`dflash_weights.py` 会检查官方 checkpoint 的 config、69 个 tensor
名称、shape、dtype 和文件身份；不符合合同时在运行前失败。

## 3. 为什么需要 clean anchor

DFlash 路线开始时先让 Target 对 prompt 生成一个权威 token，这个 token 是 clean anchor。

假设：

```text
prompt          = [p0, p1, p2]
Target anchor   = a0
committed       = [p0, p1, p2, a0]
```

Adapter 的 `propose()` 将它拆成：

```text
context_ids = committed[:-1] = [p0, p1, p2]
block_ids   = [a0, MASK, MASK, ..., MASK]
```

Target feature 只覆盖 `context_ids`；anchor 位于 Draft block 第 0 行。这样不会把 anchor 同时放进
Target context 和 Draft block 两次。

## 4. block_size 与 K 的关系

官方 `block_size` 表示包含 anchor 的 query 总行数，K 表示 mask/proposal 数，因此
`K=block_size-1`：

```text
K=1  → [anchor, mask]                  共 2 行
K=3  → 1 个 anchor + 3 个 mask          共 4 行
K=5  → 1 个 anchor + 5 个 mask          共 6 行
K=7  → 1 个 anchor + 7 个 mask          共 8 行
K=15 → 1 个 anchor + 15 个 mask        共 16 行
```

Draft 最终丢弃 anchor 对应的 row 0，只输出 K 个 proposal。官方配置 `block_size=16` 的
最大 K 是 15，不能把配置值直接当 proposal 数。

## 5. 输入准备

### 5.1 Target feature

8 层 feature 先经过：

```text
[B,C,20480]
    ↓ fc.weight [2560,20480]
[B,C,2560]
    ↓ hidden_norm
projected_target_hidden
```

### 5.2 Draft block

`[anchor, MASK × K]` 使用 Target input embedding 权重：

```text
block_ids [B,K+1]
    ↓ shared Target embedding
noise_embedding [B,K+1,2560]
```

Draft 还会构造覆盖 `context + block` 的 position IDs。Target feature、block embedding、Draft
参数、Target embedding 和 LM head 必须在同一 device 和 dtype。

## 6. 每个 Draft layer 做什么

每层先把 Target context 和 Draft block 映射成 attention K/V，而 query 来自 Draft block：

```text
Q = q_proj(draft_hidden)

K = concat(
      k_proj(projected_target_hidden),
      k_proj(draft_hidden)
    )

V = concat(
      v_proj(projected_target_hidden),
      v_proj(draft_hidden)
    )
```

随后是标准残差结构：

```text
input RMSNorm
→ attention
→ residual add
→ post-attention RMSNorm
→ SwiGLU MLP
→ residual add
```

6 层类型为：

```text
layer 0..4: sliding_attention，causal，window=4096
layer 5:    full_attention，Draft block 内 non-causal
```

最后一层 non-causal 的含义是多个 mask row 可以并行去噪，不是说这些 token 可以绕过 Target
验证。

## 7. 从 hidden 到 proposal

6 层结束后：

```text
final RMSNorm
→ 丢弃 row 0 的 anchor hidden
→ 使用共享 Target lm_head.weight
→ argmax Top-1
→ proposal IDs [B,K]
```

`output_multiplier` 和可选 tanh softcap 是单调变换，不改变 Top-1。因此当前实现可以在统一
`DFlashOps.top1()` 边界得到 proposal。

## 8. 为什么 Draft 没有 cache

当前 V1 每轮重新计算：

- 当前 committed prefix 的 Target feature；
- `[anchor, MASK × K]` Draft block；
- 全部 6 层 Draft。

这样 CPU/GPU/NPU 更容易对齐，也不需要处理 Draft cache 在 proposal 被拒绝时的回滚。代价是
计算量大，所以它是 correctness-first 实现，不是最终性能结构。

## 9. Draft ops 如何切换设备

模型层不直接判断 CPU/CUDA/NPU，而是调用一个 `ops` 策略对象：

| 原语 | CPU/CUDA | NPU |
|---|---|---|
| RMSNorm | `TorchDFlashOps.rms_norm` | `dflash_ascend310p_ops.rms_norm` |
| Linear | `TorchDFlashOps.linear` | NPU backend 同 ABI |
| RoPE | `TorchDFlashOps.rotary` | NPU backend 同 ABI |
| Attention | PyTorch SDPA | NPU 分解 attention |
| SwiGLU | `F.silu(gate) * up` | NPU Tensor 上同公式 |
| Top-1 | LM head + argmax | NPU Tensor 上同公式 |

CPU 和 CUDA 使用同一个 `TorchDFlashOps`；PyTorch 根据 Tensor 的 device 分派。NPU runner 使用
package-local `dflash_ascend310p_ops`，禁止缺算子时静默回退到 CPU。

这不是全局 hook。`DFlashDraftModel.from_pretrained(..., ops=selected_ops)` 将策略对象显式传入
各个 Draft 子模块。

### 9.1 `quant` 分支不会量化 Draft

`quant` 分支当前只量化 NPU Target 的约定 Linear。Draft 仍加载同一个官方 6 层 checkpoint，
使用 FP16 Target feature、FP16 Draft-facing embedding/LM head 和原 NPU Draft backend。
因此量化命令不会改变本页的 Draft 结构、K 定义、mask 或 Scheduler 接口。若以后量化 Draft，
必须作为独立近似边界重新比较逐层 hidden、proposal Top-1、接受长度和最终 token，不能与 Target
量化同时静默开启。

## 10. Adapter.propose() 的真实顺序

`Qwen35DFlashFullPrefixAdapter.propose()` 做以下事情：

1. 检查 committed prefix、device、token 范围。
2. 取 `context_ids = prefix_ids[:, :-1]`。
3. 让 Target 对 context 重新计算 `[1,C,20480]` feature。
4. 构造 `[anchor, MASK × K]`。
5. 用 Target embedding 得到 Draft block embedding。
6. 构造 context + block 的 position IDs。
7. 调用 `draft.draft_top1()`。
8. 检查输出恰好是 `[1,K]` 整数 token，且在 Target 词表范围内。
9. 把 proposal 返回 Scheduler。

第 9 步之后不是直接输出文本，而是进入
[调度与 token 验证](DFLASH_V1_SCHEDULER.md)。

## 11. Draft 排错顺序

| 现象 | 优先检查 |
|---|---|
| K 个 proposal shape 不对 | K/anchor 定义、是否错误丢行、mask block 长度 |
| 第一个 proposal 就经常错误 | anchor、Target feature、feature 投影、position IDs、checkpoint |
| CPU/GPU 都低接受率 | 公共 Draft 结构、mask、权重加载或 prompt workload |
| GPU 正常但 NPU 低 | NPU Target feature 或 NPU Draft ops 数值差异 |
| 最终 token 与 ordinary 不同 | 先查 Scheduler/Target verify，不应通过降低接受标准解决 |
| K 越大后半段越差 | 正常可能性较高；查看每轮连续接受长度，而非只看总命中数 |

当前 Draft 为什么还不代表已经提速，以及后续的 Draft cache、融合算子和
Target 整块验证怎样配合，见
[完整 DFlash 与提速路线](DFLASH_FULL_AND_PERFORMANCE_ROADMAP.md)。
