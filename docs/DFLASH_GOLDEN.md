# Qwen3.5-4B DFlash PyTorch golden

## 边界

这是独立于官方 Qwen3.5 一层 MTP 的第二条候选路线。它不会替换
`qwen35_mtp`，也不复用那 15 个 MTP tensor。实现锁定到：

- `z-lab/dflash@07ebd93db9f472af339b644bb70221ad8428328a`；
- `z-lab/Qwen3.5-4B-DFlash@9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf`；
- 69 个 tensor、634425856 个 BF16 参数。

第一版只实现无 cache 的 draft core，允许用完整已提交前缀重新构建 target feature。
它不包含 target verify、接受率统计或 Qwen3.5 Gated DeltaNet 状态回滚，因此只能作为
CPU/custom-op 数值 golden，不能标记为 310P 性能或端到端通过。

## 对共享方案的两个修正

公开 DFlash 不是把 `[target context, mask embedding]` 拼成一个序列后共同经过六层
self-attention。真实数据流是：

```text
8 x target hidden [B,C,2560]
          │ concat
          v
Linear(20480,2560) -> RMSNorm -> target_context [B,C,2560]
                                      │
mask/anchor embedding [B,D,2560]       │ 每层重新做 K/V projection
          │                           │
          └── Q only ── 6 x decoder layer
                         │
                         └── K/V = [target_context, current draft hidden]
```

另外，只有 target verifier 才能计算 acceptance rate。仅运行 draft forward 可以检查
hidden、logits 和候选 token，但不能单独得出接受率。

## 官方 shape

- target layer IDs：`1,5,9,13,17,21,25,29`；Transformers 输出包含 embedding
  row，因此实际读取 `hidden_states[2,6,10,14,18,22,26,30]`；
- `target_hidden`：`[B,C,20480]`；
- `noise_embedding`：`[B,D,2560]`，第 0 行是 target 已生成的 clean anchor，后面是
  mask token `248077` 的 embedding；
- `position_ids`：`[B,C+D]`；
- 输出：`[B,D,2560]`，真正 draft rows 是 `output[:,1:,:]`；
- checkpoint 的 `use_sliding_window=true`，因此前五层是 sliding causal attention，
  最后一层是 full bidirectional attention；
- Q heads/KV heads/head dim：`32/8/128`，block 上限 `16`。

## 纯 PyTorch 使用

```python
import torch
from qwen35_dflash import DFlashDraftModel, extract_context_feature

draft = DFlashDraftModel.from_pretrained(
    "/path/to/z-lab-Qwen3.5-4B-DFlash",
    dtype=torch.float16,
)

target_hidden = extract_context_feature(
    target_outputs.hidden_states,
    draft.config.target_layer_ids,
)

block_ids = torch.full((1, 16), draft.config.mask_token_id, dtype=torch.long)
block_ids[:, 0] = clean_anchor_token
noise_embedding = draft.embed_block(block_ids, target_embedding_weight)
position_ids = torch.arange(
    target_hidden.shape[1] + block_ids.shape[1]
).unsqueeze(0)

draft_hidden = draft.draft_hidden(
    target_hidden,
    noise_embedding,
    position_ids,
)
draft_top1 = draft.ops.top1(draft_hidden, target_lm_head_weight)
```

上面的 `output_hidden_states` 路径只用于未改 target 时的诊断。按现有 torch_npu 工程接入
时，应使用选择性 `output_dflash_features=True` 接口，避免保留全部 32 层；具体三个
插入点见 [DFLASH_TARGET_INTEGRATION.md](DFLASH_TARGET_INTEGRATION.md)。

无 cache 模式要求 `target_hidden` 覆盖完整 draft context。若只传最近一次 verify 的
hidden，必须同时实现并恢复此前的 draft KV cache，否则语义不完整。

## 自定义算子替换

内部 Python 模块需要导出六个同名函数：

```python
def rms_norm(x, weight, eps): ...
def linear(x, weight): ...
def rotary(query, key, cosine, sine): ...
def attention(query, key, value, attention_mask, scale, key_value_groups): ...
def swiglu(gate, up): ...
def top1(hidden, lm_head_weight): ...
```

golden 自动生成的 `attention_mask` 是 boolean visible mask（`True` 表示可见）；
`key_value_groups=4` 允许内部 fused attention 直接使用 GQA，不要求调用方先物理复制 KV。

然后：

```python
from qwen35_dflash import DFlashDraftModel, ModuleDFlashOps

ops = ModuleDFlashOps.from_name("internal_dflash_ops", strict=True)
draft = DFlashDraftModel.from_pretrained(
    "/path/to/dflash",
    ops=ops,
    device="npu:0",
    dtype=torch.float16,
)
```

`strict=True` 时缺少任一算子立即失败。CPU 模拟才允许 `strict=False`；目标结果禁止
静默回退。

310P 侧可从
`targets/ascend310p/runtime/dflash_custom_ops_template.py` 复制绑定模块；固定输入输出和
mask 语义见 `targets/ascend310p/abi/dflash-draft-core-v1.json`。模板中的
`qwen35_ascend_dflash` 只是占位 namespace，需要改成内部实际注册名。

## checkpoint 审计和固定 case

```bash
PYTHONPATH=model python -m qwen35_dflash audit \
  --draft-dir /path/to/z-lab-Qwen3.5-4B-DFlash \
  --verify-model-hash --output /path/in/current/run/dflash-audit.json

PYTHONPATH=model python -m qwen35_dflash run-case \
  --draft-dir /path/to/z-lab-Qwen3.5-4B-DFlash \
  --case /path/to/case.npz \
  --dtype float16 \
  --output /path/in/current/run/draft-hidden.npy \
  --report /path/in/current/run/draft-report.json
```

`case.npz` 必须包含 `target_hidden`、`noise_embedding`、`position_ids`。用内部算子时
增加 `--ops-backend internal_dflash_ops`，不要增加 `--allow-op-fallback`。

下一阶段才接 strict greedy target verifier，并覆盖 block `4/8/16`、拒绝位置
`accepted=0..D-1`、full-attention KV 和 Gated DeltaNet recurrent/conv state。
