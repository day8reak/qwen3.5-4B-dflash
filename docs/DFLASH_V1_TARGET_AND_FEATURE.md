# DFlash V1：Target 与 Feature 详细实现

本文只讲完整 Qwen3.5-4B Target：它为什么是权威答案、为了 DFlash 增加了什么、NPU 为什么还
需要 Bridge，以及相关门禁怎样发现状态污染。

先读整体关系：[DFlash V1 整体架构](DFLASH_V1_ARCHITECTURE.md)。

## 1. Target 的两个职责

在 DFlash V1 中，Target 同时负责：

1. 对任意完整前缀返回每一行 logits，Scheduler 用它验证 proposal。
2. 在显式打开 feature 时，返回 8 个 decoder layer 的 hidden states 拼接结果。

公共 Adapter 看到的接口可以简化为：

```text
features=False:
input_ids [1,S] → logits [1,S,248320]

features=True:
input_ids [1,S] → logits + dflash_features [1,S,20480]
```

Target 不负责接受 proposal，也不运行 6 层 Draft。接受规则属于 Scheduler。

## 2. CPU/CUDA Target

CPU 和 CUDA 使用：

```text
models/dflash_v1/modeling_qwen3_5_dflash.py
```

它保持 Transformers Qwen3.5 的 text decoder 数学，只在 decoder loop 增加可选 collector。
同一份 Python 模型放到 CPU 或 CUDA，PyTorch 根据 Tensor device 选择设备 kernel。

普通路径默认关闭 feature，所以普通 Target 返回结构和数学不应改变。

## 3. NPU Target

NPU 使用：

```text
models/modeling_qwen3_5_hiai_nd.py
```

这份 Target 自己显式调用 HIAI/NPU attention、GDN、CacheUpdate 等路径。DFlash 没有把
CPU Target 动态搬到 NPU，也没有在运行时 patch 它。

DFlash 只增加 opt-in feature 输出：

```text
output_dflash_features=False → 保持普通 logits Tensor
output_dflash_features=True  → (logits, dflash_features)
```

因此 Target-only inference 仍可关闭 feature，完全不进入 Draft 或 Scheduler。

### 3.1 `quant` 分支的 Target

量化分支仍使用同一份 NPU Target 和 feature collector，只在 Bridge 装配阶段增加两件事：

1. quantizer 用 Linear 量化权重把约定的 `nn.Linear` 替换为已有 `QLinear`；
2. input-provider 用 embedding 权重和 embedding scale 复现普通量化推理的第 0 层 FP16 hidden。

Draft 不随 Target 一起量化，仍读取 FP16 feature，并保留 FP16 embedding/LM-head view。Scheduler
和逐 proposal 验证规则也不改变。这样量化误差首先只来自 Target，可以分别比较 ordinary
量化 Target、fresh full-prefix Target、feature 和最终 DFlash token。

三个数据路径、两个 callback 和 Target-only 预检命令见
[量化版运行与排错指南](DFLASH_V1_QUANT_RUNBOOK.md)。

## 4. Feature 捕获位置

固定层号为：

```text
1, 5, 9, 13, 17, 21, 25, 29
```

它们从 0 开始计数。每层捕获点是 decoder layer 已经完成 residual/MLP 之后、最终模型 norm
之前：

```python
for layer_id, decoder_layer in enumerate(layers):
    hidden_states = decoder_layer(...)
    collector.capture(layer_id, hidden_states)

hidden_states = final_norm(hidden_states)
```

每层 hidden shape 是 `[B,S,2560]`。collector 按固定层号顺序写入预分配 buffer：

```text
8 × 2560 = 20480
最终 feature = [B,S,20480]
```

实现位于 `models/dflash_v1/dflash_target_features.py`。它会检查：

- 8 层全部出现且没有重复；
- batch 和 sequence length 一致；
- dtype 和 device 一致；
- hidden width 恰好为 2560；
- 输出拼接顺序与 checkpoint 合同一致。

默认使用 `detach + clone`，避免后续原地操作覆盖已经捕获的层输出。

## 5. Adapter 为什么要求完整 logits

`Qwen35DFlashFullPrefixAdapter.forward_logits()` 调用 Target 时设置：

```text
use_cache=False
output_dflash_features=False
logits_to_keep=0
```

`logits_to_keep=0` 表示验证需要 `[1,S,vocab]` 全部行，而不是只保留最后一行。Scheduler 在
sequential 模式主要读取最后一行，但完整 shape 合同能阻止 Target loader 静默改变公共 ABI。

`_replay_target_features()` 则打开 feature，并只要求一个 LM-head row，因为 Draft 真正需要的是
全部 decoder feature，而不是全部 vocab logits。

## 6. 为什么 NPU 需要 Bridge

公共 Scheduler 只传 `input_ids`，但 NPU Target 还需要：

- linear-attention 的 `conv_state + recurrent_state`；
- full-attention 的 block-table K/V cache；
- attention mask、position IDs、cache positions 和逻辑长度；
- prefill/decode chunk 语义。

`models/internal_dflash_bridge.py` 每次 Target 调用都会：

1. 根据 `layer_types` 创建全新的 32 层 hybrid state。
2. 为 linear-attention 层创建新的 conv/recurrent state。
3. 为 full-attention 层创建新的 block-table K/V。
4. 重建与 `kv_cache_max_len` 一致的 block table。
5. 构造 mask、position 和 cache position。
6. 执行一次 fresh full-prefix prefill。
7. 截取真实逻辑长度的 logits/features。
8. 等 NPU 异步执行完成后再释放本次 state。

V1 不把这些状态交给 Scheduler，也不跨 Target 调用复用它们。

## 7. 为什么 NPU 前缀会补到 64

当前 Target GDN 多 token 路径使用 `chunk_size=64`：

```text
S = 1 → 物理长度 1
S > 1 → 物理长度 ceil(S/64) × 64
```

例如逻辑前缀长度 17，Bridge 会右补齐到 64 行执行，但：

- `allQLen` 仍是 17；
- 返回结果只保留前 17 行；
- padding token 不会进入 committed prefix；
- Scheduler 仍认为序列长度是 17。

补齐是 Target 物理执行细节，不改变 DFlash token 语义。

## 8. Feature 零影响门禁

`validate_feature_capture_zero_impact()` 对同一前缀运行两次：

```text
A: features=False → clone logits_A
B: features=True  → clone logits_B + 检查 feature shape
```

然后比较：

- logits device/dtype 相同；
- 每个位置 Top-1 相同；
- 浮点差异不超过该 dtype 的有界容差；
- feature shape 恰好是 `[1,S,20480]`。

先 clone A 再运行 B 很重要：某些运行时会复用输出 buffer，如果不立即 clone，B 可能覆盖 A，
导致错误地“自己和自己相等”。

报告字段：

```text
feature_capture_zero_impact = true
feature_capture_audit.status = PASS_BOUNDED_ZERO_IMPACT
```

## 9. 状态隔离门禁

`validate_full_prefix_state_isolation()` 不仅运行一次 `P→Q→P`，还先做立即重复对照：

```text
ordinary 模式：P → P → Q → P
feature 模式： P → P → Q → P
```

其中 Q 与 P 长度不同。检查内容包括：

- 立即 `P→P` 是否稳定；
- 经过 Q 后，P 的 logits Top-1 和数值是否仍稳定；
- feature 模式的 logits 是否稳定；
- P 的完整 feature 是否稳定。

立即 `P→P` 是设备本身重复性基线，`P→Q→P` 才用于发现跨调用 KV/GDN 残留。

报告状态是：

```text
full_prefix_repeatability_audit.status = PASS_BOUNDED_P_Q_P
```

它是有界行为检查，不等于逐个内部 state 的设备 trace；报告会把后者保留为 `PENDING`。

## 10. 常见问题怎么定位

| 现象 | 优先检查 |
|---|---|
| feature shape 不对 | 捕获层号、捕获点、拼接顺序、hidden size |
| feature 开关后 logits 变化 | collector 是否原地修改 hidden、输出 buffer 是否复用、分支是否改变 Target forward |
| 立即 P→P 都不稳定 | 设备算子确定性、异步同步、未初始化内存 |
| P→P 稳定但 P→Q→P 失败 | KV/GDN state 没有重建或清理、call-local tensor 生命周期 |
| CPU/GPU 正常但 NPU feature 不同 | NPU Target 实现、padding/chunk、位置/mask 或自定义算子数值 |
| ordinary 文本正常但接受率低 | 先确认 feature 与 Draft 输入；不要先改 verifier |

下一步阅读：[Draft 模型](DFLASH_V1_DRAFT.md)。
