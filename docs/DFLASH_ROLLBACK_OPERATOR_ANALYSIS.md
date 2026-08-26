# Qwen3.5-4B DFlash 回退版改造与自定义算子分析

## 1. 结论

本次交付了一份独立的 HIAI modeling 修改版，原文件没有覆盖。修改版把已经完成的
`npu_gated_delta_rule_mtp` 接到了 Qwen3.5 Target 的 24 个 GDN 层，并为另外两类状态给出
了 correctness-first 路径：

1. GDN recurrent state：使用现有 GDR MTP state bank。
2. GDN causal-conv state：使用同样的逐位置 state bank；当前由 NPU Tensor 分解实现。
3. Full-attention KV：保留物理 cache，用逻辑位置回退；跨 64-token block 时暂时逐 token
   调用现有 `npu_cache_update_`。

因此，**除 GDR 外，唯一确定存在的 Target 状态语义缺口是 causal-conv state 回退**。
它可以先用当前文件中的分解实现验证正确性，但生产性能版建议增加
`CausalConv1dMTP`。`CacheUpdateMTP` 和 `FusedInferAttentionMTP` 是否必须新增，要先测现有
算子的多 token、跨 block 和 mask 能力，不能仅凭算子名判断。

当前分支已经补上 scheduler、CPU/CUDA transaction、HIAI bridge 和部署 wrapper adapter：
正式路径执行一次 `T=K+1` 整块 verify，并跨轮保存 state，不再用增长的完整前缀验证。
它仍是 correctness bring-up，而不是高性能完成态：NPU causal-conv 还是 Tensor golden，
CacheUpdate 仍逐 row，完整 logits 仍回 host 做 Top-1，而且当前环境没有 310P 全模型证据。

## 2. 来源与边界

| 项目 | 锁定值 |
|---|---|
| 用户指定仓库 | `day8reak/qwen3.5-4B-dflash` |
| 仓库提交 | `01121dc154f9f0f83e1c10a3c9978ec2495fcf1e` |
| 原 HIAI 文件 SHA-256 | `e02e9818ce4a896e2bd1882f37ba20d7afb3712df69058a6a4f6c0973b6b6851` |
| 官方 DFlash 行为参考 | `z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a` |
| Target config | `Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` |
| Target config SHA-256 | `ddc63e1c717afa86c865bb5e01313d89d72bb53b97ad4a8a03ba8510c0621670` |
| 本次验证设备 | CPU 静态/辅助逻辑验证；没有 310P 真机证据 |

用户分享记录锁定的 GDR MTP ABI 为：

```text
npu_gated_delta_rule_mtp(
  query, key, value, g, beta,
  initial_state, accepted_tokens,
  chunk_size=64,
  output_final_state=True,
  use_qk_l2norm_in_kernel=True
) -> (core_attn_out, state_bank)
```

其硬约束是 `T == MTP + 1`。已知输入/输出合同：

```text
query/key/value : [B,T,H,D]       FP16
g               : [B,T,H]         FP32
beta            : [B,T,H]         FP16
initial_state   : [B,T,H,Dk,Dv]   FP32
accepted_tokens : [B]             INT8
core_attn_out   : [B,T,H,Dv]      FP16
state_bank      : [B,T,H,Dk,Dv]   FP32
```

## 3. 修改版做了什么

### 3.1 默认普通推理不变

`accepted_tokens=None` 时仍走原来的：

```text
causal_conv1d_update
→ npu_chunk_gated_delta_rule
→ recurrent_state 原位更新
→ 原 CacheUpdate 路线
```

这样修改版不会把普通 prefill/decode 强制切到只支持 `T=MTP+1` 的新算子。

### 3.2 DFlash verify 路径

当调用方传入 `accepted_tokens: INT8[B]` 时，输入必须是固定的：

```text
[anchor, proposal_1, ..., proposal_K]
T = K + 1，K <= 16
```

GDN state ABI 改为：

```text
conv_state_bank      [B,T,8192,4]        FP16
recurrent_state_bank [B,T,32,128,128]    FP32
```

修改版提供三个辅助边界：

- `seed_dflash_gdn_state_banks`：prefill 后把普通单状态扩成第一轮 state bank；
- `rebase_dflash_gdn_state_banks`：下一轮 K 改变时先选中已提交状态，再改 bank 槽数；
- `torch_dflash_causal_conv1d_mtp`：卷积 state bank 的 NPU Tensor 分解 golden。

### 3.3 `accepted_tokens` 的准确含义

它不是“当前这次 verify 已经接受多少”，而是**上一次 verify 的接受长度，用于选择本次
计算的起始状态**。

假设上一轮输入为：

```text
slot 0 = 执行 anchor 后的状态
slot 1 = 执行 anchor + d1 后的状态
...
slot a = 执行 anchor + d1 ... da 后的状态
```

若连续接受了 `a` 个 proposal，则下一轮所有 24 个 GDN 层都传同一个：

```text
accepted_tokens = a
```

下一轮 KV 写入位置同时移动到：

```text
round_start + 1 + a
```

`correction/bonus` 虽然本轮已经输出，但还没有作为 Target 输入执行；它是下一轮的 anchor。
这条 off-by-one 规则必须在 scheduler、GDN、conv、KV cursor 和 feature slice 中完全一致。

### 3.4 Full-attention KV 回退

KV 不需要像 GDN 一样保存每个 token 的完整 state bank。候选 K/V 可以写进物理 cache；接受
长度出来后只回退逻辑 cursor，并保证 mask/`actual_seq_lengths` 不读取拒绝尾部。下一轮会从
新 cursor 覆写旧尾部。

当前源文件的 CacheUpdate 只显式传一个 `target_block + offset`。修改版在 DFlash 路径中逐行
调用它，因此位置 `63 → 64` 也能落到两个物理 block；这是正确性 fallback，性能上不应长期
保留 17 次 K 写入加 17 次 V 写入。

## 4. Qwen3.5-4B 的准确尺寸

| 项目 | 值 |
|---|---:|
| Decoder 层 | 32 |
| GDN / Full-attention 层 | 24 / 8 |
| Hidden / MLP | 2560 / 9216 |
| Target attention Q / KV heads | 16 / 4 |
| Target attention head dim | 256 |
| GDN K / V heads | 16 / 32 |
| GDN K / V head dim | 128 / 128 |
| GDN mixed QKV width | `2×(16×128) + 32×128 = 8192` |
| GDN conv kernel | 4 |
| Vocab | 248320 |
| 最大 proposal K / verify T | 16 / 17 |
| KV block size | 64 |
| 单层 packed KV cache | `[num_blocks,64,64,16]` FP16 |
| Target packed Q | `[B,256,T,16]` FP16 |

注意：当前 HIAI 源码和 bridge 都把 conv state 长度定义成 `linear_conv_kernel_dim=4`，而有些
framework reference 使用 `kernel_size-1`。新算子必须复现当前接收端的真实 ABI，不能擅自把
4 改成 3；最终以原算子 trace/golden 为准。

## 5. Target 侧算子表

“必须性”分为三类：

- **确定需要**：没有该语义就不能正确回退；
- **条件需要**：当前算子通过能力测试即可复用，否则才新增/扩展；
- **性能可选**：普通算子能表达正确数学，只在 profiling 后融合。

| 优先级 | 位置/建议算子 | 功能 | 主要输入 | 输出/副作用 | 必须性与当前状态 |
|---|---|---|---|---|---|
| P0 | `GatedDeltaRuleMTP` | 从上一轮 state bank 选 `accepted_tokens` 槽，执行 T 行 GDR，保存每行 provisional recurrent state | `q/k/v [B,T,32,128]` FP16；`g [B,T,32]` FP32；`beta [B,T,32]` FP16；`state [B,T,32,128,128]` FP32；`accepted [B]` INT8 | `out [B,T,32,128]` FP16；`state_bank` 同输入 state shape/FP32 | **已由用户完成**；修改版已接入，仍需整网真机验证 |
| P0 | `CausalConv1dMTP` | 与 GDR 使用同一接受槽，得到每行卷积输出和每行 conv window | 建议 `x [B,T,8192]` 或 `[B,8192,T]` FP16；`weight [8192,4]` FP16；`state [B,T,8192,4]` FP16；`accepted [B]` INT8 | `y` 与 x 对应；`state_bank [B,T,8192,4]` FP16 | **语义确定需要**；修改版已有 Tensor 分解 golden，生产版建议新增融合算子 |
| P0 | 扩展 `CacheUpdate` 或新增 `CacheUpdateMTP` | 一次写 T 个连续 K/V，正确跨越 64-token block | 单个 cache `[Nblock,64,64,16]` FP16；update 原始 `[B,T,4,256]` 或 packed `[T,64,16]`；`start_pos [B]`/`positions [B,T]`；block table INT32 | 原位 cache 或显式 `cache_out`；必须覆盖位置 62/63/64/65 | **条件需要**；修改版逐 token 复用现有 op 可保正确性。若现有 op 本身支持跨块多行，只需改调用，不要另造 kernel |
| P0 | 扩展 `adn_fused_infer_attention` 或 `FusedInferAttentionMTP` | 历史 paged KV + 当前 T 行块内 causal attention | packed Q `[B,256,T,16]`；K/V cache list；mask `[B,1,T,kv_max_len]` FP16；block table；真实 Q/KV length | packed output `[B,256,T,16]`，随后还原 `[B,T,4096]` | **条件需要**；先验证现有 op 的 `T=2/5/9/17`、历史长度和跨块行为，失败才改 kernel |
| P0-runtime | `DFlashStateTransaction`（runtime，不建议做成算子） | 让 24 层 GDR、24 层 conv、8 层 KV cursor、position 和 feature 使用同一 accepted count；失败时整轮丢弃 | 32 层 state handle、`accepted [B]`、`round_start`、T | 新 logical cursor；下一轮 state selector；失败不部分提交 | **已在 scheduler/bridge 实现**；CPU/CUDA 用快照恢复，NPU 失败时销毁 session |
| P0/P1 | `StateBankRebase` | K 动态变化时先 gather 已接受槽，再复制为下一轮 T' 个初始槽 | conv/recurrent bank、`accepted [B]`、`next_T` | 新的 T' 槽 state bank | 固定 K=16 时不需要；修改版已有 Tensor helper，只有图内动态 K/性能受限时才值得自定义 |
| P1 | `TargetLmHeadTop1Accept` | 避免落地完整 `[B,T,248320]` logits，同时完成 Top-1 和最长连续匹配 | hidden `[B,T,2560]`；LM head `[248320,2560]`；proposal `[B,T-1]` INT；可选 EOS | `top1 [B,T]` INT；`accepted [B]` INT8；correction/bonus token | **性能可选**；先保留普通 LM head + argmax 作为 oracle |
| P1 | 现有 DynamicQuant/QuantMatmul | 支持 Target 的 T 行量化线性层 | 激活首维包含 `B×T`，权重/scale 沿用当前 ABI | FP16 投影结果 | 不先新增算子；必须验证现有实现不是只支持 decode `T=1` 或 prefill `T=64` |

### 最小 Target 算子结论

如果现有 `CacheUpdate` 和 fused attention 的能力测试通过，则最小新增集合是：

```text
已完成：GatedDeltaRuleMTP
还建议完成：CausalConv1dMTP
```

`CacheUpdateMTP` 是强性能候选，但逐 token existing-op fallback 可以先闭合正确性。
`FusedInferAttentionMTP` 必须由真机结果决定，不能现在就判定一定要重写。

## 6. Draft 侧算子表

原仓库的 `dflash_ascend310p_ops.py` 已用 NPU Tensor 分解实现六个原语：`rms_norm`、
`linear`、`rotary`、`attention`、`swiglu`、`top1`。这些可以做功能 bring-up，不等于高性能
kernel。Draft 与 Target 回退是两组不同的算子接线。

| 优先级 | 建议算子/处理 | 功能 | 输入 | 输出 | 结论 |
|---|---|---|---|---|---|
| P1 | `DFlashFeatureProjection` 或高效 Linear | 8 层 Target feature 投影 | feature `[B,C,20480]`；weight `[2560,20480]` | `[B,C,2560]` | Draft 最大单个 GEMM 热点候选；先 profile，再决定 FP16/量化融合 |
| P1 | `DFlashBlockGQA` | 5 层 causal sliding + 最后 1 层 block non-causal 的小块 GQA | Q `[B,32,T,128]`；K/V `[B,8,C+T,128]`；bool mask `[B,1,T,C+T]`；scale | `[B,32,T,128]` | 分解 attention 可保证数学；真机性能通常值得融合，但必须保留两种 mask 语义 |
| P1 | `DFlashDraftLmHeadTop1` | LM head + argmax，避免 K 行全 vocab logits 落地 | hidden `[B,K,2560]`；weight `[248320,2560]` | token IDs `[B,K]` INT64；同值时最小 vocab ID 胜出 | 高价值性能算子；与 Target accept 算子可共享底层分块 Top-1，但 ABI 不应混为一个 |
| P2 | RMSNorm/RoPE/SwiGLU 融合 | 减少 6 层 Draft 的小算子 launch | hidden/weight/cos/sin，shape 见 Draft 模型 | 同数学输出 | 不是正确性必需；RMSNorm 必须用 checkpoint 的直接 scale，不能套 Target 的 `1+weight` |
| P2 | Draft KV cache/crop | 每轮只处理新 context/block，并在 accepted 后裁剪 | 6 层 Draft K/V、logical cursor、accepted | 更新后的 cache | 属于后续性能阶段；当前无 cache 重算版可作为正确性 oracle |

## 7. 非算子代码的当前实现

| 组件 | 当前 rollback 行为 | 仍需设备验证/优化 |
|---|---|---|
| `dflash_rollback_decode.py` | 一次 T=K+1 verify；最长连续接受；提交 `1+a` 个输入状态；correction/bonus 留作下一 anchor | 多 prompt 全模型零 token mismatch |
| `dflash_rollback_adapter.py` | CPU/CUDA 持久 `DynamicCache`；KV crop；GDN state restore；仅逐 token 重放 `anchor+accepted` | 证明可安全合并成一次短 chunk 后再优化调用数 |
| `internal_dflash_bridge.py` | prompt 单 token bootstrap；GDN scalar→bank；K 变化 rebase；logical paged-KV cursor | 32 层真实 kernel trace、block boundary 和故障注入 |
| rollback wrapper adapter | 构造期临时绑定独立 rollback modeling，复用部署 wrapper 的权重加载，随后恢复原全局类并校验实际类型 | 在目标部署 wrapper 版本上实装检查；若其构造器硬编码类则需显式 wrapper 修改 |
| mask/position | verify 从 committed cursor 开始；每行只看历史与块内左侧；position/cache position/allQLen 同步推进 | 现有 fused attention 的 T=2/5/9/17 真机能力 |
| feature 生命周期 | prefill feature 保存一次；每轮只追加 `features[:,:a+1]`；不追加 correction/bonus | 长序列内存和 Draft cache 后续优化 |
| 失败处理 | CPU/CUDA 恢复后销毁 transaction；HIAI bank 可能部分覆写时销毁整个 session，fail closed | 真机 kernel 失败注入 |

旧 `dflash_reference_decode_v1.py` 不删除，作为完整前缀 sequential oracle；默认 CPU/GPU/NPU CLI
已切到 `run_rollback.py`。

## 8. 内存与实现取舍

`T=17, B=1` 时，单个 GDN recurrent bank 大约为：

```text
17 × 32 × 128 × 128 × 4 bytes = 34 MiB / layer
24 个 GDN 层 = 816 MiB
```

conv bank 约为：

```text
17 × 8192 × 4 × 2 bytes = 1.0625 MiB / layer
24 层 = 25.5 MiB
```

所以两类 bank 合计约 `841.5 MiB × B`，还不含 KV、权重和临时 workspace。若 runtime 为失败
原子性做完整双 buffer，内存还会显著增加。建议比较两条路线：

1. 保留当前 GDR state-bank 路线：一次算完所有 provisional state，速度好但占内存；
2. 只保存 round-start state，得到 accepted 后对 `anchor + accepted proposals` 重放短 GDN/conv：
   内存低，但每轮多一次短重放。

不能为了省内存直接把 FP32 recurrent bank 改成 FP16；这会改变用户已验证的 GDR ABI，必须
另做精度实验和审批。

## 9. 验证门禁

### 已完成

- 原文件按 SHA-256 原样保留；
- 修改版 `py_compile` 通过；
- CPU 小尺寸测试通过：state-bank seed/select/rebase；
- CPU 小尺寸测试通过：卷积 bank 与逐 token reference 完全对齐；
- 默认普通 GDR 调用仍保留在 `accepted_tokens is None` 分支；
- Scheduler 边界测试通过：`accepted=0..K`、mismatch、all-match、correction/bonus；
- CPU/CUDA framework transaction 测试通过：KV crop、GDN restore、bounded token replay；
- HIAI bridge 模拟测试通过：bank selector、动态 T rebase、拒绝 KV 尾部的 logical cursor 覆写；
- 兼容 CPU/GPU CLI 与 NPU `run_npu` 已切到 persistent rollback 报告合同。

### 310P 上必须补齐

| 门禁 | 至少覆盖 |
|---|---|
| GDR 整网接线 | 24 层、多个连续 round、accepted `0/1/K-1/K` |
| Conv bank | 与逐 token ordinary GDN 的 conv 输出/state 比较；K `1/4/8/16` |
| KV boundary | committed position `62/63/64/65`，拒绝尾部下一轮不可见 |
| Attention | T `2/5/9/17`；每一行 Top-1 对齐独立前缀 oracle；历史长度多档 |
| 状态事务 | mismatch 第 0/中间/最后、all-match；32 层使用同一 a；故障注入不部分提交 |
| Feature | 仍为层 `1,5,9,13,17,21,25,29`，只保留 `a+1` 行，开关不改变 logits |
| 整网 | ordinary incremental 与 DFlash strict-greedy token ID、EOS、stop reason 零差异 |
| 运行身份 | 禁止 CPU fallback，记录设备/运行时/算子包哈希和实际 kernel trace |
| 稳定性 | 多 prompt、多轮重复，无偶发越界、状态泄漏或内存增长 |

当前环境没有 `torch_npu`、OMC 和 310P 设备，因此本文不声明 target 运行通过、无 fallback
通过或获得任何加速比。

## 10. 后续开发顺序

1. 在 310P 上先跑 K=1/T=2 的 ordinary incremental 对 rollback，确认 wrapper、GDR 注册、mask、
   feature 和 cursor；再扩到 K=16/T=17。
2. 用当前分解 conv + 逐 row CacheUpdate 做 correctness oracle，覆盖 accepted `0/1/K-1/K` 与
   cursor `62/63/64/65`。
3. 验证现有 fused attention 的 T=2/5/9/17；只有能力失败才新增/扩展 attention kernel。
4. 完成全模型、多 prompt、多轮 strict-greedy 零 token mismatch 和无 fallback 证据。
5. 开发 `CausalConv1dMTP`；若 CacheUpdate launch 成为热点，再开发 `CacheUpdateMTP`。
6. Profile 完整 logits D2H、Draft feature projection、block GQA、LM-head Top-1；按实测选择融合。
