# DFlash rollback 算子清单

本文回答两个问题：当前 rollback 为什么已经能走 correctness 路线，以及除用户已完成的
GDR MTP 外，哪些位置还需要或可能需要自定义算子。

## 1. 结论

当前版本要先跑通 rollback，不再缺少硬性的数学功能：

- GDN recurrent state 已接入 npu_gated_delta_rule_mtp；
- GDN causal-conv state 由输入 NPU device 上的 Torch Tensor golden 实现；
- full-attention KV 逐 row 复用 npu_cache_update_；
- attention 先复用 adn_fused_infer_attention；
- Top-1 先用普通 LM head 和设备侧 argmax，只把 T 个 ID 搬回 host 做连续 accept scan。
- Draft 6 层 committed K/V 已由 request-local Torch cache 实现，当前 block 成功后裁掉、异常时
  abort；它不再是 correctness 缺口。

因此当前 correctness bring-up 的最小外部依赖仍只有已经完成并注册的
npu_gated_delta_rule_mtp。面向生产性能，最先建议开发 CausalConv1dMTP。CacheUpdateMTP 和
FusedInferAttentionMTP 必须先做现有算子能力测试；TargetLmHeadTop1Accept 是高价值性能候选。

| 分类 | 算子 | 当前决定 |
| --- | --- | --- |
| 已完成 | GatedDeltaRuleMTP | 已接入，补 24 层、多轮真机证据 |
| 已完成 golden | Draft KV cache/crop | 语义与官方 append-then-crop 对齐；真机 profile 后决定是否做 cache-aware GQA |
| 生产优先 | CausalConv1dMTP | 建议新增，替换 Tensor 分解 golden |
| 条件新增 | CacheUpdateMTP | 现有多行/跨块能力或性能不足时新增 |
| 条件新增 | FusedInferAttentionMTP | 现有 T=2/4/6/8/16 能力失败时扩展或新增 |
| 性能优先 | TargetLmHeadTop1Accept | correctness 不依赖，但可消除完整 logits 落地和 D2H |
| Profiling 后 | Draft GQA、Draft Top-1、projection | Draft 热点确认后逐项融合；projection 已按 token 缓存 |

## 2. Target 尺寸和状态

| 项目 | 值 |
| --- | ---: |
| Decoder 层数 | 32 |
| GDN / full-attention 层数 | 24 / 8 |
| Hidden / MLP | 2560 / 9216 |
| Target attention Q / KV heads | 16 / 4 |
| Target attention head dim | 256 |
| GDN K / V heads | 16 / 32 |
| GDN K / V head dim | 128 / 128 |
| GDN mixed QKV channels | 8192 |
| GDN conv kernel/state length | 4 |
| Vocab size | 248320 |
| 官方 block_size / 最大 proposal K / verify T | 16 / 15 / 16 |
| KV block size | 64 |
| 单层 packed KV cache | [num_blocks,64,64,16] FP16 |

当前 HIAI receiver 的 conv state 最后一维是 4。自定义算子必须复现这个实际 ABI，不能按其他
framework 中常见的 kernel_size-1 擅自改成 3。

## 3. 已完成：GatedDeltaRuleMTP

功能：从上一轮 state bank 选择 accepted_tokens 对应槽，以该状态执行当前 T 行 GDR，并为每个
输入行保存 provisional recurrent state。

~~~text
npu_gated_delta_rule_mtp(
  query,
  key,
  value,
  g,
  beta,
  initial_state,
  accepted_tokens,
  chunk_size=64,
  output_final_state=True,
  use_qk_l2norm_in_kernel=True
) -> (core_attn_out, state_bank)
~~~

输入输出：

| Tensor | Shape | Dtype | 含义 |
| --- | --- | --- | --- |
| query / key / value | [B,T,32,128] | FP16 | 当前 verify block 的 GDR 输入 |
| g | [B,T,32] | FP32 | decay/gate |
| beta | [B,T,32] | FP16 | update 系数 |
| initial_state | [B,T,32,128,128] | FP32 | 上一轮 provisional state bank |
| accepted_tokens | [B] | INT8 | 选择上一轮已提交槽 |
| core_attn_out | [B,T,32,128] | FP16 | 当前 T 行输出 |
| state_bank | [B,T,32,128,128] | FP32 | 每行执行后的 provisional state |

accepted_tokens 是上一轮接受长度，不是当前 verify 尚未得出的接受长度。slot i 表示执行本轮输入
row 0 到 row i 后的状态。当前还需要的工作是目标设备上覆盖 24 层、连续多轮、accepted
为 0、1、K-1、K，以及 K 改变时的 select/rebase。

当前 modeling 直接接管算子返回的 `state_bank` 并更新 persistent state list，不再把完整 FP32
bank `copy_` 回旧 tensor；B16 下可省去约 768 MiB/轮的额外目的端写入。

## 4. 生产优先：CausalConv1dMTP

### 功能

GDN 的 causal-conv window 也包含前缀历史。算子必须与 GDR 使用同一个 accepted_tokens 选择
起始窗口，计算 T 行 depthwise causal convolution，并输出每一行之后的 conv state bank。

### 建议 ABI

| Tensor | Shape | Dtype |
| --- | --- | --- |
| hidden_states | [B,8192,T] | FP16 |
| conv_state_bank | [B,T,8192,4] | FP16 |
| weight | [8192,4] | FP16 |
| bias | [8192] 或无 | FP16 |
| accepted_tokens | [B] | INT8 |
| activation | scalar attribute | SiLU |
| output | [B,8192,T] | FP16 |
| next_state_bank | [B,T,8192,4] | FP16 |

状态槽的含义与 GDR 完全相同：next_state_bank[:,i] 是输入 row 0 到 row i 执行后的窗口。

### 当前替代实现

models/modeling_qwen3_5_hiai_nd_dflash_rollback.py 中的
torch_dflash_causal_conv1d_mtp 已实现同一数学语义。输入是 NPU tensor 时，分解运算仍在 NPU
上执行，因此它不是 CPU fallback；rolling state 已改为一次 unfold/permute，但仍包含 concat、
grouped conv 和中间 tensor，不适合作为最终性能实现。

### 验收

- K 为 1、3、5、7、15；
- accepted 为 0、1、K-1、K；
- 每个输出 row 和 state slot 对齐逐 token ordinary reference；
- 下一轮至少再执行一个 token，确认拒绝尾部没有污染；
- 与 GDR、KV 和 feature 使用完全相同的 accepted count。

## 5. 条件新增：CacheUpdateMTP

### 什么时候需要

当前实现对 K 和 V 的 T 行分别逐 row 调用现有 npu_cache_update_，可以跨越 64-token block，
但最多产生 2T 次 update launch。先验证现有算子是否已有多行或向量 position ABI；满足功能和
性能时只改调用，不另造 kernel。

以下任一情况成立时再新增或扩展：

- 现有 ABI 不能一次写入 T 行；
- 不能正确跨越位置 63 到 64；
- 逐 row launch 在整网 profile 中成为明显热点。

### 建议 ABI

| Tensor | Shape / dtype | 含义 |
| --- | --- | --- |
| cache | [Nblock,64,64,16] FP16 | 单个 packed K 或 V cache |
| update | [B,T,4,256] FP16 或 [T,64,16] FP16 | 当前 T 行 |
| positions | [B,T] INT32/INT64 | 每一行逻辑位置 |
| block_table | [B,max_blocks] INT32 | 逻辑块到物理块映射 |
| cache_out | 与 cache 相同，或原位副作用 | 写入全部 provisional rows |

这个算子只负责物理写入。接受后推进 logical cursor 1+a 是 runtime 事务，不应让 CacheUpdate
自己决定接受长度。

### 验收

- T 为 2、4、6、8、16；
- round start 位于 62、63、64、65；
- 写入位置与逐 row oracle 完全相同；
- prefix 和未触及 suffix sentinel 不变；
- 拒绝尾部在下一轮不可见，并从新 cursor 被覆盖。

## 6. 条件新增：FusedInferAttentionMTP

现有 adn_fused_infer_attention 如果能够正确处理历史 paged KV、当前 T 行块内 causal mask 和
真实长度，就继续复用。先测下面的逻辑 ABI：

| Tensor | Shape / dtype |
| --- | --- |
| packed query | [B,256,T,16] FP16 |
| K/V cache | 8 层各自的 packed paged cache |
| attention mask | [B,1,T,kv_max_len] FP16 |
| block table | [B,max_blocks] INT32 |
| actual query / KV length | runtime scalar 或 vector |
| output | [B,256,T,16] FP16，恢复后为 [B,T,4096] |

只有现有 op 在 T=2、4、6、8、16、不同历史长度或跨块场景中出现能力限制或数值不等价时，才扩展
它或新增 FusedInferAttentionMTP。每个有效 row 的 Top-1 都要与独立前缀 oracle 对齐。

## 7. 高价值性能算子：TargetLmHeadTop1Accept

当前 Target 仍生成 `[B,T,248320]` 完整 logits，但 argmax 已在设备上执行，只把 T 个 token ID
搬到 host 做连续匹配。大 logits 的计算/落地和边界同步仍存在，但不再有完整 logits D2H。

建议功能：

1. 分块执行 Target LM head；
2. 每行只保留 Top-1 token，严格相等时选择最小 vocab ID；
3. 将前 T-1 行与 proposal 比较；
4. 输出最长连续 accepted count、correction 或 bonus。

建议 ABI：

| Tensor | Shape / dtype |
| --- | --- |
| hidden | [B,T,2560] FP16 |
| lm_head_weight | [248320,2560] FP16 或已批准的目标格式 |
| proposal | [B,T-1] INT32/INT64 |
| optional EOS | 标量或小 vector |
| top1_ids | [B,T] INT32/INT64 |
| accepted_tokens | [B] INT8 |
| next_token | [B] INT32/INT64 |

这个算子不是 bring-up 必需项。实现后仍应保留普通 LM head 加 argmax 作为 golden，并比较所有
Top-1、tie、accepted 和 correction/bonus。

## 8. Runtime 逻辑不要做成算子

DFlashStateTransaction 负责让 24 层 recurrent bank、24 层 conv bank、8 层 KV、position、
feature 和 logical cursor 使用同一个 a，并在失败时整轮失效。它是 scheduler/bridge 的状态
所有权协议，不建议包装成一个巨大的自定义算子。

StateBankRebase 当前也已有 Tensor helper。固定 K 时不需要 rebase；只有图内动态 K 或 profiling
证明 gather/copy 明显受限时，才考虑独立 StateBankRebase 算子。

DynamicQuant 和 quant matmul 先做 T 行能力检查。如果现有实现支持激活首维 B×T，就不新增
算子；不能假定它们只支持 decode T=1 或 prefill T=64。

当前 quant 分支复用原 HIAI modeling 中同一个 `QLinear`：`npu_dynamic_quant` 接收当前
`[B,T,K]` 激活，`npu_quant_matmul` 输出 FP16 `[B,T,N]`。rollback modeling 不复制第二套
QLinear，也不改变其公式。量化 embedding/input-provider 是部署数据准备与第 0 层输入 ABI，
不是新的自定义算子；它必须分别覆盖真实 prompt chunk `T=1..64`、decode `T=1` 和 verify
`T=1..16`。若其中任一 T 不支持，应先修现有量化路径的 shape contract，不能回退到完整前缀
或逐 token prefill 来掩盖。

## 9. Draft 侧候选

这些算子不影响 Target rollback 是否正确，应在 NPU correctness 闭合后按 profile 选择。

| 优先级 | 候选 | 输入 | 输出 | 目的 |
| --- | --- | --- | --- | --- |
| P2 | DFlashFeatureProjection | 新增 feature [B,1+a,20480]；weight [2560,20480] | [B,1+a,2560] | 当前每个 token 只算一次，输出被 KV cache 消费后释放；profile 显示仍热时再融合 |
| P1 | DFlashBlockGQA | Q [B,32,T,128]；committed K/V [B,8,C,128]；new K/V [B,8,1+a+T,128]；mask | [B,32,T,128] | 避免显式 cache concat，并融合 5 层 causal sliding 与末层 block non-causal attention |
| P1 | DFlashDraftLmHeadTop1 | hidden [B,K,2560]；weight [248320,2560] | IDs [B,K] | 避免 Draft 完整 vocab logits 落地 |
| P2 | RMSNorm、RoPE、SwiGLU 融合 | 各层 hidden 与参数 | 同数学输出 | 减少小算子 launch |
| P2 | DraftKVAppendCrop | 6 层 committed K/V、new context K/V、transient block K/V | 更新后 committed cache | 当前 Torch golden 已避免历史 K/V projection；仅在 concat/分配成为热点时开发 |

Draft RMSNorm 使用 checkpoint 的直接 scale，不能套用 Target 的 1+weight 语义。最后一层
attention 的 block non-causal mask 也不能被统一改成前五层的 causal mask。

`DraftKVAppendCrop` 不接收 Target accepted count：进入下一轮 Draft 的 feature 本身已经只包含
Target 提交的 `1+a` 行。算子应在一次 Draft forward 内让 attention 看见 old+new+block，并在
返回前只提交 old+new。失败时旧 cache 必须保持可复用；无 cache `forward_projected` 是 golden。

## 10. 内存取舍

T=16、B=1 时，单层 recurrent bank 为 32 MiB，24 层为 768 MiB；单层 conv bank 为
1 MiB，24 层为 24 MiB。两者合计约 792 MiB，还未包含 KV、权重和 workspace。

Draft committed KV 的 FP16 逻辑大小为每 token 24 KiB（6 层、K/V、8 KV heads、head dim
128），C=2048 时约 48 MiB；当前轮 transient block 最多再增加 16 行。它远小于 Target state
bank，但 benchmark 仍应同时记录 allocator peak，防止 cache 拼接造成隐藏的双份峰值。

需要在真机比较：

1. 保留完整 T 槽 bank，一次计算所有 provisional state；
2. 只保留 round-start state，得到 a 后短重放 anchor 加 accepted proposals。

不能为了省内存直接把 FP32 recurrent bank 改为 FP16；这会改变已锁定的 GDR ABI，需要单独
精度实验和批准。

## 11. 建议开发顺序

1. 用当前 conv golden、逐 row CacheUpdate 和现有 attention 跑通 K=1/T=2。
2. 扩到 `block_size=16`（K=15/T=16），覆盖 accepted 0、1、K-1、K 和 cursor 62、63、64、65。
3. 证明 ordinary 与 DFlash 多 prompt、多轮 strict-greedy 零 token mismatch，且无 fallback。
4. 开发 CausalConv1dMTP，并逐 row、逐 state slot 对齐 golden。
5. Profile CacheUpdate、attention、完整词表 LM head/Top-1 和 Draft 热点；当前完整 logits D2H
   已消除，只剩 T 个 Top-1 ID 回传。
6. 只对实测热点开发 CacheUpdateMTP、TargetLmHeadTop1Accept 或 Draft 融合算子。
7. 每替换一个算子，重新跑拒绝后下一 token 状态门禁和端到端 token 门禁。
