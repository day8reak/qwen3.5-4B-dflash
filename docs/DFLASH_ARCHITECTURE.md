# DFlash rollback 框架与数据流

本文是当前 quant 分支的权威框架说明。它描述 strict-greedy 模式下 Prompt、Target feature、
Draft proposal、Target verify、W8A8 Target 和状态提交如何对齐。旧 full-prefix sequential 路线
只用于定位，不属于默认 CPU、CUDA 或 NPU 流程。

当前实现的 `block_size` 和 Draft KV cache 生命周期已与官方锁定 Transformers/MLX runner
对齐；sampling 和 Target rollback 的实现差异见
[官方完整 DFlash 对照](DFLASH_UPSTREAM_COMPARISON.md)。

## 1. 固定术语与不变量

- `block_size` 是包含 clean anchor 的 Draft query/Target verify 最大总行数，范围为 2 到 16。
- K 是本轮 Draft proposal 数，不包含 anchor，`K≤block_size-1`；最大范围为 1 到 15。
- T 是本轮 Target verify 输入行数，`T=K+1≤block_size`；范围为 2 到 16。
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
| Draft adapter | 维护已提交的逐层 KV，只投影本轮新增 feature，构造 anchor 加 K 个 mask并返回 proposal | 不决定 token 是否接受 |
| Target transaction | 维护 committed state，执行 provisional verify，并提交或放弃一轮状态 | 不修改接受规则 |
| Feature collector | 收集八个锁定层的 Target hidden，按 token 与 committed boundary 对齐 | 不保存拒绝尾部 |
| Runner / validator | `dflash` 只跑生产路径；`validate` 分别运行 ordinary 与 DFlash 并比较 token、EOS、stop reason | 单跑模式不能声称本次已完成 ordinary 对照 |

各设备实现：

| 角色 | CPU/CUDA | HIAI/NPU |
| --- | --- | --- |
| Target modeling | models/dflash_v1/modeling_qwen3_5_dflash.py | models/modeling_qwen3_5_hiai_nd_dflash_rollback.py |
| Target transaction | FrameworkDFlashRollbackTarget | InternalDFlashTarget |
| Draft backend | TorchDFlashOps | package-local dflash_ascend310p_ops |
| 状态策略 | DynamicCache 快照、恢复和有界重放 | GDN/conv bank 与 paged-KV logical cursor |
| CLI | models.dflash_v1.run_rollback | models.dflash_v1.run_npu |

原 models/modeling_qwen3_5_hiai_nd.py 不覆盖；rollback modeling 和 wrapper 都使用独立文件。

### 2.1 Target W8A8 路线

量化是 rollback Target 的可选执行后端，不是另一套 scheduler。默认 `disabled` 保留 FP16
对照；显式启用 `w8a8_dynamic` 时，装配顺序是：

~~~text
rollback HIAI FP16 model
  → 冻结全部 nn.Linear path/shape/bias
  → 部署方 quantizer 读取已有 artifact
  → Target Linear 替换为原 HIAI QLinear
  → 审计 QLinear 完整覆盖、W_q/scale layout/device
  → 保留独立 FP16 Draft embedding 与 LM head
~~~

rollback modeling 直接导入原 `modeling_qwen3_5_hiai_nd.py` 中的 `QLinear`。因此 quant 分支没有
新增或修改量化 matmul 算子，也没有改原 GDR；用户已有 converter 仍面对同一个 QLinear 类型。

量化 embedding 由单独 input-provider 返回最终 FP16 layer-0 hidden：

~~~text
真实 prompt chunk [1,1..64] ─┐
decode token       [1,1]      ├→ input-provider → FP16 [1,T,2560]
verify block       [1,1..16] ─┘                    → rollback Target
~~~

provider 只处理当前真实输入行，不接收完整历史前缀，也不补写 padding。后续 position、GDR/GDR-MTP、
feature、paged KV 和 state-bank commit 与 FP16 rollback 完全共用。Target 量化不会改变 Draft
checkpoint：Draft 六层、embedding 和完整词表 LM head 仍为 FP16。
rollback 外层 wrapper 会把底层原部署 wrapper 传给已有 provider，因此 provider 的第一个参数 ABI
与非 rollback 量化路线一致；它不需要适配组合 wrapper。

## 3. 一次完整请求

CLI 有两个显式执行模式：

- `--execution-mode validate`：先运行独立 ordinary incremental session，再从相同 prompt 新建
  DFlash session；两条流最终必须逐 token 完全相同。
- `--execution-mode dflash`：只运行一次 DFlash session，用于已经通过离线门禁后的正常生成；
  报告把 correctness gate 标成 `NOT_RUN_DFLASH_ONLY`，不伪造 exact-match PASS。

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
Draft 已提交的逐层 K/V + 本轮新增 Target feature
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
| 官方 block_size / proposal capacity | 16 / 15 |

前五层是 causal sliding attention，最后一层允许 Draft block 内 non-causal 可见。最后经过共享
Target LM head，只取 mask 对应的 K 行 Top-1。

当前 adapter 会在 feature 首次提交时执行一次 `20480→2560 + RMSNorm`。首轮只投影 prompt，
后续只投影新提交的 `1+a` 行；投影结果在下一次 Draft 消费后立即释放，不再累积完整
`[B,C,2560]` history。

Draft 为 6 层分别维护 committed K/V。设上一轮 cache 长度为 `C_old`、本轮新增 Target
feature 为 `Δ=1+a`（首轮为 prompt 长度）、当前 Draft block 为 T 行：

~~~text
每层只投影 K/V(new feature Δ + transient block T)
attention 读取 [cached K/V C_old, new K/V Δ, transient K/V T]
proposal 完成后只保留前 C_old+Δ 行
block 的 T 行无论接受多少都不会进入下一轮 Draft cache
RoPE 只构造 Δ+T 行绝对 position；旧 C_old 行沿用 cache 中已旋转的 K
~~~

这与官方 `DynamicCache.update()` 后 crop、MLX draft cache 后 trim 的边界相同。异常发生在任意
Draft 层或 final norm 时，本轮 staged K/V 整体放弃，旧 committed cache 不变。无 cache 的
`forward_projected` 路线继续作为数值 golden。

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

如果某轮 `a=0`，当前请求后续关闭 Draft，继续在同一个 rollback transaction 中用单行 Target
verify。这个 exact fallback 不重放历史前缀，也不改变输出 token；它避免低匹配 prompt 连续执行
昂贵的 16-row Draft/verify。

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
        S->>D: committed KV + new feature + anchor + K masks
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
| GDN recurrent | clone round-start state；恢复后逐 token 重放 | npu_gated_delta_rule_mtp 输出 T 槽 FP32 state bank，model 直接接管返回 bank |
| GDN causal-conv | clone round-start window；恢复后逐 token 重放 | Torch Tensor golden 输出 T 槽 conv bank |
| Target feature | 使用 commit replay 的 1+a 行 | 只截取 verify feature 的前 1+a 行 |

CPU/CUDA commit replay 最多 K+1 个单 token 调用，只重放当前 anchor 和已接受 proposal，不包含
prompt 或更早历史。使用逐 token 路线是为了让提交后的 GDN 数值路径贴近 ordinary incremental。

NPU 的 accepted_tokens 表示上一轮接受长度，用来选择上一轮 state bank 的槽。第一轮从 prompt
后的 scalar state 建立 T 个槽；下一轮 K 改变时先选择已提交槽，再 rebase 到新的 T。Paged KV
可以保留拒绝尾部的物理内容，但 mask 和有效长度不能读取它，下一轮从 logical cursor 覆写。

Prompt 在 NPU bridge 中按 KV block 边界拆成最多 64 个真实 token 的 chunk，不把 padding 写入
persistent GDN/conv/KV state。多 token chunk 继续调用原版
`npu_chunk_gated_delta_rule`，由原 modeling 的 `seq_len > 1` 分支选择 `chunk_size=64`；只有真实的
单 token 尾块才走 `chunk_size=1`。所有真实 prompt 行仍执行 decoder/state/feature 计算，ordinary
与 rollback 都只对最后一个真实 prompt 行计算一次 248320 词表 logits。任一 verify 失败后
session 整体失效，不能带着部分更新的 32 层状态继续。

## 7. Feature、EOS 和长度边界

每轮接受 a 个 proposal 后，只生成 anchor 和已接受 proposal 对应的 `1+a` 行 pending feature。
Correction 或 bonus 不加入，因为它还没有作为 Target 输入执行。下一轮 Draft 前必须满足：

~~~text
draft_kv_cache_length + pending_feature_length = committed_token_length - 1
~~~

Draft forward 成功后，pending feature 被转换为各层 K/V 并清空；因此稳态 host/model tensor 不再
保存完整 feature history。FP16 下 6 层 cache 的逻辑大小约为每个 committed token 24 KiB，
上下文 2048 时约 48 MiB，不包含当轮最多 16 行 transient K/V 和 allocator workspace。

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

报告用下面字段确认已走新框架和 cache：

~~~text
route = qwen3.5-dflash-incremental-rollback
verification_mode = incremental_transactional_rollback
historical_prefix_replay_during_verify = false
draft_kv_cache_audit.mode = upstream_equivalent_append_then_crop
target_quantization.scheme = disabled 或 w8a8_dynamic
~~~

启用 W8A8 时还要求 `target_quantization.route=rollback`、完整 Linear 拓扑审计 PASS、input-provider
调用全部成功，并在同一量化 Target 上完成 ordinary 与 DFlash strict-greedy 零 token 差异。

只有 `execution_mode=validate` 且 `correctness_gate.status=PASS` 时，
`strict_greedy_exact_match` 才为 true；`execution_mode=dflash` 的该字段为 null。

CPU 是模拟证据；CUDA 是 framework 设备证据。Ascend 310P 还必须禁用 fallback，并记录真实
runtime、device、算子包和 kernel trace，才能声明 Target rollback 通过。

## 9. 当前性能边界

已经消除的是 Target verify 的历史前缀重算。仍可能限制性能的部分包括：

- CPU/CUDA commit 的最多 K+1 次单 token replay；
- NPU causal-conv Tensor 分解和逐 row CacheUpdate；
- Target 与 Draft 仍计算完整词表 logits；当前只把设备侧 Top-1 ID 搬回 host；
- Draft attention 仍读取长度 C+T 的 K/V，当前 cache 拼接和 attention 仍随上下文增长；但 6 层
  历史 context K/V projection 已消除；
- state bank 的峰值内存；stage/benchmark 边界仍需要 host/device 同步。

GDR/conv 返回的新 state bank 会直接替换 persistent state 引用，不再额外 `copy_` 回旧 bank；这减少
B16 每轮约 768 MiB recurrent bank 加 24 MiB conv bank 的目的端复制，但不改变 bank 本身的峰值
容量。

Persistent rollback 调用通过同一设备流依赖串联，不再在每个 prompt/decode row 后插入全设备
host barrier；full-prefix oracle 的 state 是 call-local，释放前仍保留同步。

NPU Draft backend 默认只在最终 logits 边界做 finite 检查；shape/device/dtype 检查始终启用。
`DFLASH_ASCEND310P_EXHAUSTIVE_CHECKS=1` 会恢复每个 primitive 的中间 tensor/weight 扫描，仅用于
数值诊断，因为每次 `.item()` 都会形成设备同步。

这些优化必须在 strict-greedy 和拒绝后下一 token 状态都完全对齐后逐项引入。详细算子边界见
[自定义算子文档](DFLASH_OPERATORS.md)。
