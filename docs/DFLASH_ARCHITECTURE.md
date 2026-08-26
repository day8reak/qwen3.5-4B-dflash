# DFlash rollback 框架与数据流

本文是当前 rollback 分支的权威框架说明。它描述 strict-greedy 模式下 Prompt、Target feature、
Draft proposal、Target verify 和状态提交如何对齐。旧 full-prefix sequential 路线只用于定位，
不属于默认 CPU、CUDA 或 NPU 流程。

当前实现与官方锁定 Transformers/MLX runner 的 Draft cache、block-size 口径、sampling 和
Target rollback 差异，见[官方完整 DFlash 对照](DFLASH_UPSTREAM_COMPARISON.md)。

## 1. 固定术语与不变量

- K 是 Draft proposal 数，不包含 anchor；范围为 1 到 16。
- T 是 Target verify 的输入行数，T=K+1；范围为 2 到 17。
- 当前调度只支持 batch 1、strict greedy。
- Target 是唯一裁判。Draft token 在通过 Target 验证前不能成为最终结果。
- Target 状态、position、feature 和 cache 必须用同一个 accepted count 原子提交。

Prompt prefill 后以及每轮结束后都保持同一不变量：

~~~text
已处理并写入 Target state/feature：current anchor 之前的全部 token
已输出但尚未作为 Target 输入处理：current anchor
~~~

这个边界决定了所有 off-by-one 规则：如果本轮接受 a 个 proposal，提交的是 anchor 加 a 个
proposal，共 1+a 行；correction 或 bonus 是下一轮 anchor，不能在本轮提前写入状态。

## 2. 组件和所有权

| 组件 | 主要职责 | 不拥有的职责 |
| --- | --- | --- |
| Scheduler | 组织 bootstrap、Draft、verify、连续接受、EOS 和长度停止 | 不直接修改 Target 内部 cache |
| Draft adapter | 用已提交 feature history 构造 anchor 加 K 个 mask，返回 K 个 proposal | 不决定 token 是否接受 |
| Target transaction | 维护 committed state，执行 provisional verify，并提交或放弃一轮状态 | 不修改接受规则 |
| Feature collector | 收集八个锁定层的 Target hidden，按 token 与 committed boundary 对齐 | 不保存拒绝尾部 |
| Validator CLI | 分别运行 ordinary 与 DFlash session，比较最终 token、EOS、stop reason | ordinary 对照不属于生产热路径 |

各设备实现：

| 角色 | CPU/CUDA | HIAI/NPU |
| --- | --- | --- |
| Target modeling | models/dflash_v1/modeling_qwen3_5_dflash.py | models/modeling_qwen3_5_hiai_nd_dflash_rollback.py |
| Target transaction | FrameworkDFlashRollbackTarget | InternalDFlashTarget |
| Draft backend | TorchDFlashOps | package-local dflash_ascend310p_ops |
| 状态策略 | DynamicCache 快照、恢复和有界重放 | GDN/conv bank 与 paged-KV logical cursor |
| CLI | models.dflash_v1.run_rollback | models.dflash_v1.run_npu |

原 models/modeling_qwen3_5_hiai_nd.py 不覆盖；rollback modeling 和 wrapper 都使用独立文件。

## 3. 一次完整请求

当前验证 CLI 先运行一个独立 ordinary incremental session，再从相同 prompt 新建 DFlash
session。两条流最终必须逐 token 完全相同。

~~~mermaid
flowchart TD
    P[相同 prompt] --> O[Ordinary persistent session]
    P --> R[DFlash persistent session]
    O --> OG[Incremental greedy tokens]
    R --> PF[Prompt bootstrap]
    PF --> A[Clean anchor]
    A --> D[Draft proposals]
    D --> V[Target transactional verify]
    V --> C[Commit accepted prefix]
    C --> D
    C --> DG[DFlash tokens]
    OG --> X[Exact token EOS stop comparison]
    DG --> X
~~~

Ordinary session 的行为是：

1. prompt 只 prefill 一次；
2. 读取最后一行 logits 得到第一个 generated token；
3. 之后每次只把上一个 generated token 送入 Target；
4. 不调用 Draft，也不重算完整前缀。

DFlash session 的行为是：

1. prompt prefill，得到 clean anchor 和 prompt 对应的八层 feature；
2. 输出 anchor，但暂不把 anchor 写入 Target state；
3. Draft 提出 K 个 token；
4. Target 一次验证 anchor 加 K 个 proposal；
5. Scheduler 求最长连续接受长度 a；
6. Target 只提交 anchor 加前 a 个 proposal；
7. correction 或 bonus 成为下一轮尚未处理的 anchor。

## 4. Draft 如何提出 K 个 token

Draft 不是另一个独立语言模型。它使用两类输入：

~~~text
Target 已提交 feature history
[current anchor, MASK × K] 的共享 embedding
~~~

Target feature 固定来自 decoder 层 1、5、9、13、17、21、25、29 的层输出，捕获点在 final
norm 之前：

~~~text
8 × [B,C,2560] → [B,C,20480]
~~~

其中 C 只覆盖 current anchor 之前的已处理 token。Draft block 的第 0 行才放 anchor，后面 K
行放 mask token 248077，因此不会把 anchor 同时放进 feature context 和 Draft block。

官方 Draft 合同为：

| 项目 | 固定值 |
| --- | ---: |
| Draft 层数 | 6 |
| hidden / intermediate | 2560 / 9216 |
| attention heads / KV heads | 32 / 8 |
| head dim | 128 |
| feature width | 20480 |
| vocab size | 248320 |
| checkpoint tensor 数 | 69 |
| 最大 proposal K | 16 |

前五层是 causal sliding attention，最后一层允许 Draft block 内 non-causal 可见。最后经过共享
Target LM head，只取 mask 对应的 K 行 Top-1。

当前 adapter 保存 Target feature history，但 Draft 自身还没有跨轮 KV cache；每轮仍会重新处理
已保存的 feature context。这不影响 rollback 正确性，但属于后续性能优化点。

## 5. Target 如何验证 token

一次 Target 输入和 logits 的位置关系如下：

| Target 输入行 | 该行 Top-1 的用途 |
| --- | --- |
| row 0 = anchor | 验证 proposal d1 |
| row i = di，1 ≤ i < K | 验证 proposal d(i+1) |
| row K = dK | 全部 proposal 命中时提供 bonus |

等价表示：

~~~text
input  = [anchor, d1, d2, ..., dK]
top1   = [t1,     t2, t3, ..., bonus]
compare d1==t1, d2==t2, ...
~~~

Scheduler 从左到右比较，只接受最长连续匹配前缀。设首次不匹配的位置为 a：

~~~text
accepted proposals = d[:a]
correction          = top1[a]
committed inputs    = input[:a+1]
next anchor         = correction
~~~

如果 K 个 proposal 全部命中，则 a=K，top1[K] 是 bonus。错误 proposal 右侧的 logits 不参与
决策；块内 causal mask 必须保证较早行看不到右侧 proposal。

### 示例：中途拒绝

~~~text
anchor    = 101
proposal  = [202, 999, 888]
Target    = [202, 303, ... , ...]
accepted  = 1

本轮输出     = [202, 303]
状态提交输入 = [101, 202]
下一轮 anchor = 303
~~~

### 示例：全部接受

~~~text
anchor    = 101
proposal  = [202, 303, 404]
Target    = [202, 303, 404, 505]
accepted  = 3

本轮输出     = [202, 303, 404, 505]
状态提交输入 = [101, 202, 303, 404]
下一轮 anchor = 505
~~~

## 6. 状态事务

~~~mermaid
sequenceDiagram
    participant S as Scheduler
    participant D as Draft
    participant T as Target
    participant X as State transaction

    S->>T: prefill prompt
    T->>X: commit prompt state and feature
    T-->>S: clean anchor

    loop 未到 EOS 或长度上限
        S->>D: feature history + anchor + K masks
        D-->>S: d1 ... dK
        S->>X: begin round at committed cursor
        S->>T: verify anchor + d1 ... dK
        T->>X: produce provisional KV GDN conv states
        T-->>S: T rows logits and features
        S->>S: calculate longest accepted prefix a
        S->>X: commit input rows 0 ... a
        X-->>S: advance cursor by 1+a
        S->>S: correction or bonus becomes next anchor
    end
~~~

三类状态必须一起处理：

| 状态 | CPU/CUDA | HIAI/NPU |
| --- | --- | --- |
| Full-attention KV | verify 前记住长度；commit 前 crop，再有界重放 | provisional 物理写入；logical cursor 只推进 1+a |
| GDN recurrent | clone round-start state；恢复后逐 token 重放 | npu_gated_delta_rule_mtp 输出 T 槽 FP32 state bank |
| GDN causal-conv | clone round-start window；恢复后逐 token 重放 | Torch Tensor golden 输出 T 槽 conv bank |
| Target feature | 使用 commit replay 的 1+a 行 | 只截取 verify feature 的前 1+a 行 |

CPU/CUDA commit replay 最多 K+1 个单 token 调用，只重放当前 anchor 和已接受 proposal，不包含
prompt 或更早历史。使用逐 token 路线是为了让提交后的 GDN 数值路径贴近 ordinary incremental。

NPU 的 accepted_tokens 表示上一轮接受长度，用来选择上一轮 state bank 的槽。第一轮从 prompt
后的 scalar state 建立 T 个槽；下一轮 K 改变时先选择已提交槽，再 rebase 到新的 T。Paged KV
可以保留拒绝尾部的物理内容，但 mask 和有效长度不能读取它，下一轮从 logical cursor 覆写。

Prompt 在 NPU bridge 中逐 token bootstrap，避免把 64-row padding 写入 persistent GDN state。
任一 verify 失败后 session 整体失效，不能带着部分更新的 32 层状态继续。

## 7. Feature、EOS 和长度边界

每轮接受 a 个 proposal 后，feature history 只追加 anchor 和已接受 proposal 对应的 1+a 行。
Correction 或 bonus 不追加，因为它还没有作为 Target 输入执行。下一轮 Draft 前必须满足：

~~~text
feature_history_length = committed_token_length - 1
~~~

Draft proposal 中出现 EOS 时，EOS 后面的固定宽度槽不再验证。输出 accepted token 或
correction/bonus 时遇到 EOS 立即停止。达到 max_new_tokens 时，已计算但未输出的 bonus 不能进入
结果，也不能被当成已提交 token。

## 8. 正确性和运行身份

最终门禁是：

~~~text
ordinary.generated_token_ids == dflash.generated_token_ids
ordinary.reached_eos          == dflash.reached_eos
ordinary.stop_reason          == dflash.stop_reason
~~~

没有文本级宽松比较或浮点容差。整块 verify 若因 kernel、position、mask 或状态选择产生 Top-1
分叉，必须定位原因，不能通过放宽接受规则掩盖。

报告用下面三个字段确认已走新框架：

~~~text
route = qwen3.5-dflash-incremental-rollback
verification_mode = incremental_transactional_rollback
historical_prefix_replay_during_verify = false
~~~

CPU 是模拟证据；CUDA 是 framework 设备证据。Ascend 310P 还必须禁用 fallback，并记录真实
runtime、device、算子包和 kernel trace，才能声明 Target rollback 通过。

## 9. 当前性能边界

已经消除的是 Target verify 的历史前缀重算。仍可能限制性能的部分包括：

- CPU/CUDA commit 的最多 K+1 次单 token replay；
- NPU causal-conv Tensor 分解和逐 row CacheUpdate；
- Target 与 Draft 的完整词表 logits/Top-1；
- Draft 每轮重算 feature context，尚无跨轮 Draft KV cache；
- state bank 的峰值内存和 host/device 同步。

这些优化必须在 strict-greedy 和拒绝后下一 token 状态都完全对齐后逐项引入。详细算子边界见
[自定义算子文档](DFLASH_OPERATORS.md)。
