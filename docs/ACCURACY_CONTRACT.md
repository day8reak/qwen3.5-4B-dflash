# 准确率契约

## 固定范围

- checkpoint：`Qwen/Qwen3.5-4B`，revision 见 `specs/upstream.lock.json`；
- text-only，batch 1；
- greedy，默认最多 2 个 draft token；
- 官方 BF16 MTP 权重，完整 248320 词表；
- tokenizer/chat template 必须来自同一 checkpoint。

## 门禁顺序

1. config、15 个 MTP tensor、shape、dtype、共享 embedding 审计通过；
2. portable MTP 对齐 Transformers 官方 full-attention/MLP reference；
3. full 与 incremental draft hidden/cache 在冻结容差内；
4. framework ordinary 可重复；
5. CPU ordinary/MTP 最终 token ID 零 mismatch；
6. target ordinary 对齐 framework ordinary；
7. target MTP 对齐 target ordinary，且 MTP counters 非零；
8. 强制覆盖每个接受长度 `r=0..K`；
9. 真机每个冻结 case 连续 10/10 数值成功后才测性能。

## 数值规则

- token ID、position、长度、counter：完全相同；
- greedy 最终 token stream：零 mismatch；
- FP32 单元测试 hidden：`rtol=1e-5, atol=1e-5`；
- full/incremental FP32 cache：`rtol=1e-4, atol=1e-6`；
- BF16/FP16 target 中间值阈值须在内部运行前冻结，并报告 max abs、relative L2、
  cosine、NaN/Inf；不能只报 cosine；
- Top1 需记录 top1/top2 margin；相同分数选择最小 token ID。

任何量化、shortlist、近似 attention、图边界变化或阈值放宽都是新的审批边界，本版
没有授权。

## 310P 精度审批边界

官方 checkpoint 是 BF16，但华为文档明确说明 310P 的 SelfAttention q/k/v/mask 与
SwiGLU 路径不支持 BF16。建议的第一项 target 实验是：

```text
原公式/权重：BF16 checkpoint + Qwen3.5 eager 语义
候选：BF16 权重显式转换 FP16；MatMul/attention/SwiGLU 用 FP16，
      RMSNorm/softmax/Top1 比较按现有算子允许的最高精度累加
范围：ordinary 与 MTP 均使用同一候选精度，不改模型结构、词表或 acceptance
风险：溢出/舍入改变 hidden、logit 排名和最终 token
门禁：framework BF16 vs target ordinary；target ordinary vs target MTP；零 MTP 额外 token mismatch
回滚：保留 BF16 CPU reference 和未转换 checkpoint，不推广 target artifact
```

用户已于 `2026-08-18T10:51:53Z` 明确批准该实验，批准记录位于
`.work/qwen3.5-4b/20260818T105058Z-8154-e71669/out/approvals/`。冻结阈值见
`specs/fp16-experiment.json`；批准不允许放宽阈值或直接推广候选。

首个真实文本 CPU admission case 的结果如下：

| 比较项 | relative L2 | cosine | Top1 |
|---|---:|---:|---|
| ordinary hidden | 0.009342 | 0.999957 | 一致 |
| ordinary full-vocab logits | 0.008984 | 0.999946 | 一致 |
| isolated FP16 MTP hidden | 0.009213 | 0.999958 | 一致 |
| end-to-end FP16 MTP hidden | 0.011953 | 0.999929 | 一致 |
| end-to-end FP16 MTP logits | 0.010184 | 0.999933 | 一致 |

全模型权重无 NaN/Inf 或 FP16 overflow；4205751296 个主模型元素中 10056 个极小值
转零，120599552 个 MTP 元素中 331 个转零。该指数范围损失属于已批准风险，必须
完整报告，并由上述冻结 hidden/logit/Top1 门禁约束；它不是“权重应精确往返”的新门禁。
当前结论仅为 `ELIGIBLE_FOR_ASCEND310P_TESTING_NOT_PROMOTED`。
