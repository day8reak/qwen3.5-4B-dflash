# DFlash 自定义算子清单

本文区分三件事：当前 strict-greedy rollback 能否运行、去掉生产 golden 还缺什么、性能优化可能
需要什么。只有实测热点才进入优化算子开发；sampling、streaming、batch 和 transaction 所有权
首先是软件能力，不应被包装成一个巨型算子。

## 1. 当前结论

- 已完成并接入：`GatedDeltaRuleMTP`；原 `ChunkGatedDeltaRule` 已适配新增的
  `effective_length` ABI。
- 当前可运行：causal-conv 使用输入 NPU 上的 Tensor golden；KV 使用现有
  `npu_cache_update_` 逐 row 写；attention 使用现有 `adn_fused_infer_attention`；Draft 使用
  package-local Torch-NPU 分解 primitives。
- 去掉 production golden 的首要新增算子：`CausalConv1dMTP`。
- `CacheUpdateMTP` 和 `FusedInferAttentionMTP` 不是默认必做；先证明现有算子的多行、跨块和数值
  能力，再根据 profile 决定。
- 当前高价值性能候选：Draft/Target full-vocab Top-1、Draft GQA、W8A8 dynamic-quant+matmul
  dispatch 融合。
- `framework/quant-air-om` 的导出预检覆盖当前 Target modeling 出现的七个前端算子：四个缺
  Meta 的 receiver 算子补精确 Fake，三个已有 Meta 的 torch-npu 算子校验后复用；AIR 中仍需
  逐 type 审计 GE 节点。这只解决 FakeTensor/图保留，不替代这里要求的真实算子数值与性能证据。

因此，“完整官方 generation 功能”不能仅靠新增算子完成；“当前 NPU 路线不再依赖 conv
golden”则明确需要 `CausalConv1dMTP`。

## 2. 总表

表中 P0 表示已有 correctness 依赖，P1 表示生产或实测热点优先，P2/P3 表示 profile 后选择。

| 算子 | 当前状态 | 功能 | 主要输入 | 主要输出 | 需要程度 |
| --- | --- | --- | --- | --- | --- |
| `ChunkGatedDeltaRule` | 复用 receiver 新 ABI、已接线 | 普通 prompt/decode GDR；固定物理行数时忽略无效尾部 | Q/K/V、g、beta、`effective_length`、初始 state | attention output、最终 FP32 state | P0，已有 |
| `GatedDeltaRuleMTP` | 已完成、已接线 | 选择上一轮 committed recurrent slot，计算 T 行并保存逐行 provisional state | Q/K/V、g、beta、state bank、`accepted_tokens` | attention output、FP32 state bank | P0，已有 |
| `CausalConv1dMTP` | Torch-NPU golden | 从 committed scalar 或同一 accepted slot 执行 depthwise causal conv，保存逐行 conv window | mixed QKV、scalar/banked conv state、weight/bias、`accepted_tokens` | activated rows、FP16 conv bank | P1，去 golden 必需 |
| `CacheUpdateMTP` | 现有 op 逐 row | 一次写入 T 行 paged K 或 V，支持跨 64-token block | cache、updates、positions、block table | 原位 cache | 条件 P1 |
| `FusedInferAttentionMTP` | 复用现有 attention | 历史 paged KV + 当前 T 行 block-causal attention | Q、K/V cache、mask、length/table | T 行 attention | 条件 P1 |
| `DFlashBlockGQA` | Draft Tensor 分解 | 直接读取 committed/new KV，避免 repeat/concat 和小算子链 | Draft Q、committed/new K/V、mask | 6 层 attention rows | 性能 P1 |
| `DFlashDraftLmHeadTop1` | 普通 LM head + argmax | 分块完整词表 matmul，只落地 K 个 Top-1 ID | Draft hidden、完整 FP16 LM head | `[B,K]` IDs | 性能 P1 |
| `TargetLmHeadTop1Accept` | 普通 LM head + argmax + host scan | 完整词表 Top-1，并计算连续接受数和 correction/bonus | Target hidden、LM head、proposal | Top-1、accepted、next token | 性能 P1 |
| `FusedDynamicQuantLinear` | 每次 QLinear 分别 dynamic-quant + quant-matmul | 在一个设备任务内完成激活量化和 W8A8 matmul | FP16 activation、INT8 weight、FP32 scale | FP16 output | W8A8 性能 P1，先 profile |
| `DFlashFeatureProjectNorm` | Linear + RMSNorm | 只处理本轮新增 `1+a` feature | `[B,Δ,20480]`、projection/norm weight | `[B,Δ,2560]` | 性能 P2 |
| `DraftKVAppendCrop` | request-local Tensor cache | attention 可见 old+new+block，只提交 old+new | 6 层 cache、新 context、transient block | committed cache | 性能/内存 P2 |
| Draft small-op fusion | 标准 Tensor ops | 融合 RMSNorm/RoPE/SwiGLU 等短链 | hidden、norm/rope/MLP 参数 | 同数学输出 | 性能 P2 |
| `StateBankSelectRebase` | Tensor gather/expand | T 改变时选择 committed slot 并建立新 bank | state bank、accepted、新 T | rebased bank | 性能 P3 |

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

`effective_length` 是 call-local valid rows，不是累计 KV 长度 `allQLen`，也不是
`accepted_tokens`。例如 full-prefix oracle 把真实 37 行右补齐到物理 S=64 时传 `[37]`；
persistent prompt chunk 为 64+1 时两次分别传 `[64]` 和 `[1]`；decode 传 `[1]`。同一个
`INT16[B]` Tensor 在一次 Target forward 的 24 个 GDN 层间复用，避免每层重复构造。

普通 modeling 和 rollback modeling 的 ordinary 分支都调用这个新 ABI。部署侧若仍注册旧签名，
必须先更新原 GDR 算子包；Python 侧不能通过删掉该参数兼容，否则 padding 会污染 final state。

### 4.2 `GatedDeltaRuleMTP`：已完成

```text
npu_gated_delta_rule_mtp(
  query, key, value, g, beta,
  initial_state, accepted_tokens,
  chunk_size=64,
  output_final_state=True,
  use_qk_l2norm_in_kernel=True
) -> (core_attn_out, state_bank)
```

| Tensor | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| query/key/value | `[B,T,32,128]` | FP16 | 当前 verify rows |
| g | `[B,T,32]` | FP32 | decay/gate |
| beta | `[B,T,32]` | FP16 | update coefficient |
| initial state bank | `[B,T,32,128,128]` | FP32 | 上一轮 provisional slots |
| `accepted_tokens` | `[B]` | INT8 | 选择上一轮 committed slot |
| core output | `[B,T,32,128]` | FP16 | 当前 T 行输出 |
| next state bank | `[B,T,32,128,128]` | FP32 | slot i 为处理 row 0..i 后状态 |

还需补齐真实设备证据：24 层、多轮、`a=0/1/K-1/K`、K 改变和 rejection 后至少一个 token。
`accepted_tokens` 是上一轮接受数；当前轮接受数在算子执行后才由 Target Top-1 决定。
当前 GDR-MTP 是精确 T=1..16 的 recurrent/state-bank 语义，没有 padding tail；其
`chunk_size=64` 仅保留现有调用 ABI，不代表执行原 GDR 的 chunk 路线。本次改动不向 GDR-MTP
增加 `effective_length`。

### 4.3 `CausalConv1dMTP`：生产优先

```text
causal_conv1d_mtp(
  hidden_states, conv_state_bank, weight, bias,
  accepted_tokens, activation="silu"
) -> (output, next_conv_state_bank)
```

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden states | `[B,Cg,T]` | FP16 |
| committed conv / previous conv bank | `[B,Cg,Kc]` 或 `[B,T,Cg,Kc]` | FP16 |
| weight | `[Cg,Kc]` | FP16 |
| bias | `[Cg]` 或无 | FP16 |
| `accepted_tokens` | `[B]` | INT8 |
| output | `[B,Cg,T]` | FP16 |
| next conv bank | `[B,T,Cg,Kc]` | FP16 |

当前 `torch_dflash_causal_conv1d_mtp` 已实现相同语义，输入在 NPU 时没有 CPU fallback，但会形成
gather、concat/unfold、depthwise conv、activation 和中间 tensor。新算子必须逐 row、逐 state
slot 对齐该 golden，并与 GDR、KV、feature 使用同一个 accepted count。

五 OM incremental graph 已经持久化选中的 scalar conv state，因此该路径直接消费
`[B,Cg,Kc]`，不再先复制 24 份 T=16 input bank，也不执行 24 次 previous-slot gather；原始
torch_npu rollback 路径仍可传 `[B,T,Cg,Kc]` bank。两种输入必须生成完全相同的逐 row output 与
next bank，CPU exact test 已冻结这一点；真实 AIR/OM 的算子数和时延仍需 msprof 确认。

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
- `StateBankSelectRebase`：固定 B/T 时通常无收益；动态 T gather/expand 明显时再开发。

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

1. 用现有 GDR-MTP、conv golden、逐 row KV 和现有 attention 跑通 B=2，再扩到 B=16。
2. 覆盖 accepted `0/1/K-1/K`、动态 T、cursor `62/63/64/65` 和 rejection 后下一 token。
3. 开发 `CausalConv1dMTP`，逐 row/slot 对齐 golden，去掉 production conv golden。
4. 在相同 token/hash/调用次数下采集无 profiler 3+10 latency，再用 msprof 定位热点。
5. 按收益依次评估 Draft GQA/Top-1、Target Top-1、CacheUpdate、W8A8 fused linear。
6. 每替换一个算子，重新跑完整 state 门禁和 ordinary/DFlash strict-greedy 零差异。

T=16 时单层 recurrent bank 约 32 MiB，24 层约 768 MiB；conv bank 24 层约 24 MiB。若 profile
显示 bank 峰值比短重放更差，应比较“完整 provisional bank”与“round-start state + accepted 短重放”，
但不能未经精度批准把 FP32 recurrent state 改成 FP16。
