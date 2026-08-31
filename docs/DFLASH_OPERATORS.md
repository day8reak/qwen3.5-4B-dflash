# DFlash 自定义算子清单

本文区分三件事：当前 strict-greedy rollback 能否运行、去掉生产 golden 还缺什么、性能优化可能
需要什么。只有实测热点才进入优化算子开发；sampling、streaming、batch 和 transaction 所有权
首先是软件能力，不应被包装成一个巨型算子。

## 1. 当前结论

- 已完成并接入：原 `ChunkGatedDeltaRule` 的 `effective_length` ABI；prompt、decode、verify 和
  accepted-prefix commit 都复用它，不调用 `GatedDeltaRuleMTP`。
- 当前可运行：causal-conv 使用输入 NPU 上的 Tensor golden；KV 使用现有
  `npu_cache_update_` 逐 row 写；attention 使用现有 `adn_fused_infer_attention`；Draft 使用
  package-local Torch-NPU 分解 primitives。
- 去掉 production golden 的首要新增算子：`CausalConv1dChunkCommit`。
- `GDRChunkStateCommit` 是跳过第二次 output 计算的性能候选，不是当前 correctness 必需算子。
- `CacheUpdateMTP` 和 `FusedInferAttentionMTP` 不是默认必做；先证明现有算子的多行、跨块和数值
  能力，再根据 profile 决定。
- 当前高价值性能候选：Draft/Target full-vocab Top-1、Draft GQA、W8A8 dynamic-quant+matmul
  dispatch 融合。

因此，“完整官方 generation 功能”不能仅靠新增算子完成；“当前 NPU 路线不再依赖 conv
golden”则明确需要 `CausalConv1dChunkCommit`。

## 2. 总表

表中 P0 表示已有 correctness 依赖，P1 表示生产或实测热点优先，P2/P3 表示 profile 后选择。

| 算子 | 当前状态 | 功能 | 主要输入 | 主要输出 | 需要程度 |
| --- | --- | --- | --- | --- | --- |
| `ChunkGatedDeltaRule` | 复用 receiver 新 ABI、已接线 | prompt/decode；verify `T` 行；从同一 S0 按 `a+1` 二次计算 committed state | Q/K/V、g、beta、`effective_length`、标量初始 state | attention output、最终 FP32 state | P0，已有 |
| `GDRChunkStateCommit` | 未实现 | 消费第一次的 capsule，只计算接受前缀最终 state，跳过 query/output 路径 | K/V/g/beta 或紧凑中间量、S0、`commit_length` | 单个 FP32 state | 性能 P1，profile 后 |
| `CausalConv1dChunkCommit` | Torch-NPU golden | 从标量 window 计算 T 行输出和 prefix windows，按 `a+1` 提交一个 window | mixed QKV、标量 conv state、weight/bias、`commit_length` | activated rows、单个 committed window | P1，去 golden 必需 |
| `CacheUpdateMTP` | 现有 op 逐 row | 一次写入 T 行 paged K 或 V，支持跨 64-token block | cache、updates、positions、block table | 原位 cache | 条件 P1 |
| `FusedInferAttentionMTP` | 复用现有 attention | 历史 paged KV + 当前 T 行 block-causal attention | Q、K/V cache、mask、length/table | T 行 attention | 条件 P1 |
| `DFlashBlockGQA` | Draft Tensor 分解 | 直接读取 committed/new KV，避免 repeat/concat 和小算子链 | Draft Q、committed/new K/V、mask | 6 层 attention rows | 性能 P1 |
| `DFlashDraftLmHeadTop1` | 普通 LM head + argmax | 分块完整词表 matmul，只落地 K 个 Top-1 ID | Draft hidden、完整 FP16 LM head | `[B,K]` IDs | 性能 P1 |
| `TargetLmHeadTop1Accept` | 普通 LM head + argmax + host scan | 完整词表 Top-1，并计算连续接受数和 correction/bonus | Target hidden、LM head、proposal | Top-1、accepted、next token | 性能 P1 |
| `FusedDynamicQuantLinear` | 每次 QLinear 分别 dynamic-quant + quant-matmul | 在一个设备任务内完成激活量化和 W8A8 matmul | FP16 activation、INT8 weight、FP32 scale | FP16 output | W8A8 性能 P1，先 profile |
| `DFlashFeatureProjectNorm` | Linear + RMSNorm | 只处理本轮新增 `1+a` feature | `[B,Δ,20480]`、projection/norm weight | `[B,Δ,2560]` | 性能 P2 |
| `DraftKVAppendCrop` | request-local Tensor cache | attention 可见 old+new+block，只提交 old+new | 6 层 cache、新 context、transient block | committed cache | 性能/内存 P2 |
| Draft small-op fusion | 标准 Tensor ops | 融合 RMSNorm/RoPE/SwiGLU 等短链 | hidden、norm/rope/MLP 参数 | 同数学输出 | 性能 P2 |

## 3. 形状符号

| 符号 | 当前 Qwen3.5-4B 值 |
| --- | ---: |
| batch `B` | 1 |
| verify rows `T` | 1..16；正常 speculative round 为 2..16 |
| hidden `H` | 2560 |
| vocab `V` | 248320 |
| Target GDN/full-attention layers | 24 / 8 |
| GDN mixed channels `Cg` | 8192 |
| GDN value heads/dim | 32 / 128 |
| GDN conv window `Kc` | 4 |
| Draft Q/KV heads/dim | 32 / 8 / 128 |
| paged-KV block | 64 tokens |

物理 layout 必须以当前 receiver/exporter 为准；下面同时给出稳定的逻辑 ABI，不能只靠 shape
相同就假设 layout 等价。

## 4. 状态算子

### 4.1 原 `ChunkGatedDeltaRule`：新增 `effective_length` ABI 已适配

```text
npu_chunk_gated_delta_rule(
  query, key, value, g, beta, effective_length,
  chunk_size=64,
  initial_state=None,
  output_final_state=False,
  use_qk_l2norm_in_kernel=False
) -> (core_attn_out, final_state)
```

| Tensor | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| query/key/value | `[B,S,32,128]` | FP16 | 本次普通 GDR 的物理输入行 |
| g | `[B,S,32]` | FP32 | decay/gate |
| beta | `[B,S,32]` | FP16 | update coefficient |
| `effective_length` | `[B]` | INT16 | 每个 batch 在本次 S 行中的有效前缀长度 |
| initial/final state | `[B,32,128,128]` | FP32 | 本次调用前/后的 recurrent state |
| core output | `[B,S,32,128]` | FP16 | 本次物理 S 行输出 |

`effective_length` 是 call-local valid rows，不是累计 KV 长度 `allQLen`。例如 full-prefix
oracle 把真实 37 行右补齐到物理 S=64 时传 `[37]`；
persistent prompt chunk 为 64+1 时两次分别传 `[64]` 和 `[1]`；decode 传 `[1]`。同一个
`INT16[B]` Tensor 在一次 Target forward 的 24 个 GDN 层间复用，避免每层重复构造。

普通 modeling 和 rollback modeling 的 ordinary 分支都调用这个新 ABI。部署侧若仍注册旧签名，
必须先更新原 GDR 算子包；Python 侧不能通过删掉该参数兼容，否则 padding 会污染 final state。

### 4.2 Verify/commit：复用原 GDR

```text
S0 = scalar committed recurrent state

verify_out, _ = ChunkGatedDeltaRule(
  q, k, v, g, beta, effective_length=T, initial_state=S0
)
a = longest_prefix_accept(verify_out, proposals)
_, S1 = ChunkGatedDeltaRule(
  cached_q, cached_k, cached_v, cached_g, cached_beta,
  effective_length=a+1, initial_state=S0
)
```

framework 为 24 个 GDN 层各保存一个 request-local capsule：Q/K/V/g/beta、S0 和 conv prefix
windows。第二次调用的 `commit_length` 必须是 `a+1`，因为 anchor 也作为本轮 Target 输入提交。
所有层成功后才发布 S1；失败则整轮/session fail-closed。正确性版不需要修改自定义算子。

若 msprof 证明第二次调用的 output/query 路径是热点，可新增 `GDRChunkStateCommit`：输入 cached
K/V/g/beta（或原 GDR 已生成的紧凑中间量）、S0 和 `INT16[B] commit_length`，只输出
`[B,32,128,128]` FP32 state。该优化必须与上述第二次原 GDR 的 state 数值对齐，并重新执行整网
零 token-ID mismatch 门禁。

### 4.3 `CausalConv1dChunkCommit`：生产优先

```text
causal_conv1d_chunk_commit(
  hidden_states, initial_conv_state, weight, bias,
  commit_length, activation="silu"
) -> (output, committed_conv_state)
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden states | `[B,Cg,T]` | FP16 |
| initial conv state | `[B,Cg,Kc]` | FP16 |
| weight | `[Cg,Kc]` | FP16 |
| bias | `[Cg]` 或无 | FP16 |
| `commit_length` | `[B]` INT16 或 host scalar | 本轮 `a+1` |
| output | `[B,Cg,T]` | FP16 |
| committed conv state | `[B,Cg,Kc]` | FP16 |

当前 `torch_dflash_causal_conv1d_chunk` 暂存 `[B,T,Cg,Kc]` prefix windows，再选第
`commit_length-1` 个窗口；输入在 NPU 时没有 CPU fallback，但会形成 concat/unfold、depthwise
conv、activation 和中间 tensor。新算子必须对齐每个 commit length，并与 GDR、KV、feature
使用同一个 `a+1`。

验收档位：`K=1/3/5/7/15`，`a=0/1/K-1/K`，连续多轮及动态 T；拒绝后继续执行至少一个 token，
确认 rejected window 未污染 committed state。

## 5. Target 条件算子

### 5.1 `CacheUpdateMTP`

当前 K/V 分别对 T 行调用现有 `npu_cache_update_`，正确但一轮最多形成 `2*T` 次 launch。只有
现有 ABI 不能多行/跨块，或 msprof 证明它是热点时才新增：

| 输入/输出 | Shape / dtype |
| --- | --- |
| packed cache | `[num_blocks,64,64,16]` FP16 |
| logical updates | `[B,T,4,256]` FP16；或物理 `[T,64,16]` |
| positions | `[B,T]` INT32/INT64 |
| block table | `[B,max_blocks]` INT32 |
| cache out | 原位更新，与输入 cache 同 layout |

必须覆盖 round start `62/63/64/65`、T=`2/4/6/8/16`、prefix/suffix sentinel 不变，以及 rejected
tail 下一轮不可见并被覆写。算子只做物理写入；是否提交 `1+a` 仍由 runtime cursor 决定。

### 5.2 `FusedInferAttentionMTP`

先验证现有 `adn_fused_infer_attention`：

| 输入/输出 | 逻辑 shape / dtype |
| --- | --- |
| current Q | `[B,T,16,256]` FP16；当前 packed 形式为 `[B,256,T,16]` |
| paged K/V | 8 层各自的 receiver layout |
| mask | `[B,1,T,kv_max_len]` FP16 |
| block table / lengths | INT32/INT64 runtime values |
| output | `[B,T,16,256]` FP16，随后投影回 hidden |

只有它不能同时处理历史 KV、当前块内 causal mask、动态真实长度和跨块位置，或性能明显不足时，
才扩展/新增 MTP 版本。

## 6. 性能算子

### 6.1 Draft GQA 与 Top-1

`DFlashBlockGQA` 建议直接消费：

```text
Q                 [B,32,T,128]
committed K/V     [B,8,C,128]
new context K/V   [B,8,Δ,128]
transient K/V     [B,8,T,128]
mask              block/sliding visibility
-> output          [B,32,T,128]
```

前五层是 sliding causal，最后一层允许当前 block 内 non-causal 可见，不能把六层统一成一种 mask。
cache commit 只包含 old+Δ，不包含 transient T。

`DFlashDraftLmHeadTop1` 输入 hidden `[B,K,2560]` 和完整 FP16 weight `[248320,2560]`，输出
`[B,K]` token ID。必须遍历完整词表，精确 tie 时选择最小 vocab ID；未经批准不能换成 shortlist。

### 6.2 Target Top-1 与接受

`TargetLmHeadTop1Accept`：

| Tensor | Shape / dtype |
| --- | --- |
| Target hidden | `[B,T,2560]` FP16 |
| LM head | `[248320,2560]` FP16 或已批准目标 layout |
| proposals | `[B,T-1]` INT32/INT64 |
| Top-1 IDs | `[B,T]` INT32/INT64 |
| accepted | `[B]` INT8 |
| correction/bonus | `[B]` INT32/INT64 |

当前已经只把 T 个 argmax ID 搬回 host，不再搬完整 logits；新算子的收益来自避免完整 logits
落地、减少 kernel/host 边界。普通 LM head + argmax 必须保留为 golden。

### 6.3 W8A8 `FusedDynamicQuantLinear`

原 QLinear 每次调用分别执行 `npu_dynamic_quant(x)` 和 `npu_quant_matmul(...)`：

```text
x            [B,T,K] FP16
W_q          [K,N] INT8 blocked-ZN carrier
weight scale [N] 或导出约定 shape FP32
-> output    [B,T,N] FP16
```

如果 Runtime timeline 显示大量 DynamicQuant/QuantMatmul 下发间隙或隐式同步，可以融合激活量化、
per-token scale 和 quant matmul。不能只凭 QuantMatmul kernel 时间判断；先锁定相同 token hash、
T 分布和调用次数，再比较端到端 latency。融合后必须逐 T=`1..64`、每个 Linear path 与原
QLinear 对齐。

### 6.4 其余候选

- `DFlashFeatureProjectNorm`：只对本轮新增 Δ=`1+a` 行执行 `20480->2560 + RMSNorm`；当前已经
  每个 committed token 只算一次，profile 热时才融合。
- `DraftKVAppendCrop`：减少 cache concat/allocator 峰值；异常时旧 committed cache 必须原样可用。
- RMSNorm/RoPE/SwiGLU fusion：只在小算子 launch 成为热点时做，保持 FP32 reduction 和 FP16
  rounding boundary。
- `GDRChunkStateCommit`：只有第二次原 GDR 的 query/output 计算成为实测热点时才开发。

## 7. 不应做成自定义算子的内容

以下内容默认留在 scheduler/runtime：

- longest-prefix accept 的请求级控制流；
- EOS、max-new-tokens、zero-accept Target-only fallback；
- 24 层 recurrent、24 层 conv、8 层 KV、feature、position 的原子 commit/abort；
- sampling 的随机流、概率比 rejection 和 residual correction；
- ordinary/DFlash correctness gate 和报告。

可以在 profile 证明 host round trip 是瓶颈后，把“Top-1 + accepted reduction”做成小型设备算子，
但不要让一个 kernel 隐式拥有整个 request transaction。

## 8. 开发顺序

1. 用原 GDR 两次 chunk、conv golden、逐 row KV 和现有 attention 跑通 B=2，再扩到 B=16。
2. 覆盖 accepted `0/1/K-1/K`、动态 T、cursor `62/63/64/65` 和 rejection 后下一 token。
3. 开发 `CausalConv1dChunkCommit`，逐 commit length 对齐 golden，去掉 production conv golden。
4. 在相同 token/hash/调用次数下采集无 profiler 3+10 latency，再用 msprof 定位热点。
5. 按收益依次评估 Draft GQA/Top-1、Target Top-1、CacheUpdate、W8A8 fused linear。
6. 每替换一个算子，重新跑完整 state 门禁和 ordinary/DFlash strict-greedy 零差异。

当前不再持久化 `[B,T,32,128,128]` recurrent bank。原 GDR final state 是 FP32；bridge 在发布
时复用 ordinary receiver 已有的 persistent cache dtype 边界（当前为 FP16），确保本分支测到的
chunk/recurrent 差异不混入新的状态存储口径。T=16 的 conv prefix windows 约 24 MiB/24 层，
Q/K/V/g/beta capsule 约 9 MiB/24 层，另有 FP32 round-start state 快照。实际峰值必须用设备
profile 统计；若要实验 FP32 persistent state，应作为单独精度分支并重跑 ordinary 对照。
