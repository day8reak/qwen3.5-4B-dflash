# v42：修复固定载体 Draft 与当前 torch_npu 的 attention 语义差异

## 结论与范围

OM 确实复用当前分支的 `DFlashDraftModel`、权重和算子数学，但不是把整个
`run_npu` 生成循环原样序列化。AIR 的 `DraftProposeStateGraph` 额外实现了固定容量
KV、逻辑长度屏蔽、RoPE 位置和 proposal 载体，因此 wrapper 也必须独立对照 eager。
`run_npu.py` 调用本分支的 `run_rollback.py`，这里的 rollback 是模块名，不是另一个分支。

源码基线 `534d0ecbbcf737479f3792048c17643e60491724` 存在确定的 mask 差异：

| 项目 | 当前分支 eager | 修复前的显式状态 Draft wrapper | v42 |
| --- | --- | --- | --- |
| 前五层 sliding attention | causal，使用各层 sliding window | causal，但遗漏 window 距离限制 | 保留实际层策略与距离限制 |
| 最后一层 full attention | 非 causal，可以看同一 block 内后面的有效 proposal | 被统一 mask 强制成 causal | 恢复 full attention |
| 尾部 K 小于 15 | 实际只构造 anchor + K 个 mask | 总是物理 16 行 | 保留 16 行，但所有层只允许 block 列 0..K 可见 |

默认 KV capacity=2048 小于官方 sliding window=4096，所以遗漏 window 不一定在该配置下
改变结果；最后一层的 causal/full 差异在 K=15 时也会出现，不依赖短输出预算或 OM 切换。
不能因此把 19.59% 与约 60% 的全部差距归因于这一个错误，修复后的真机接受率尚待测量。

修复作用于 `draft-propose`，以及复用此类的旧 `fused-speculative-step`。
不要把这个结论推广到所有历史“单 OM”：独立 recompute wrapper 有自己的 mask 路径。
本次不改变四静态 OM 默认拓扑、Target 验证/提交、权重、精度、KV 容量、调度、EOS 或驻留策略。

## 实现与 review

wrapper 在构造时读取每层 `self_attn.is_causal` 和 `self_attn.sliding_window`，缺失或非法
策略直接拒绝，不假定“所有层 causal”或“只有最后一层 full”。mask 按逻辑位置计算：

- cache 列只允许 `< logical_draft_cursor + committed_input_count`；物理 cache 空洞不可见。
- block 列只允许 `<= logical_proposal_count`；full attention 也不能看到 padding。
- causal/window 限制按该层 eager 语义施加；block 的逻辑位置紧接有效 context，不能用
  固定 KV capacity 代替其起点。
- 无效 query 行沿用最后一条有效 query 的可见范围，避免短 window、短 K 时出现全屏蔽
  softmax/NaN；它们在所有层都不能成为可见 key，也不会写进已提交 context。

检查了 context feature 投影、anchor 选择、RoPE、cache update、有效 KV 前缀与下一轮 cursor；
仍沿用原先静态 Tile/Scatter 路径，不重引入动态 INT64 BroadcastTo。
Target 的 final-token parity 不是 Draft 质量证明：Target 可以纠正错误 proposals，最终输出
仍与 ordinary 完全相同，但接受率低。回归必须直接比较 Draft，而非只比较最终生成文本，
也不能只比较两次调用同一个错误 wrapper。

## 如何重跑

**需要从 `export-air` 开始重新导出，再 `compile-om`；仅更新 runner 不能改变旧 OM 中的 mask。**
沿用 [静态四图指南](STATIC_SPLIT_OM_DEFAULT.md#从哪里重新运行) 的 factory、量化输入与命令，
把新 bundle 放进活动 run 的新目录，保留旧产物用于对照。无需重新量化。
本修复不改 C++ ABI，已匹配四图 ABI 的 runner 1.25.0 可继续使用。
当前 CLI 按整个 bundle 导出/编译，按该流程重建四图；不要手动混装旧 AIR/OM 或修改哈希。
旧 fused 候选需要重新生成其 fused AIR/OM。

新 **air-manifest.json** 中应有下面的构图声明；deployment manifest 持有对应 AIR manifest
的哈希。声明用于确认导出源码语义，不等于已完成 ATC/NPU 数值审计：

```bash
jq -e '
  [.graphs[] | select(.role == "draft-propose") | .metadata |
    (.draft_attention_mask_policy == "per-layer-logical-prefix-v1" and
     (.draft_attention_layers | length) == 6 and
     ([.draft_attention_layers[0:5][] |
       (.is_causal == true and .sliding_window == 4096)] | all) and
     .draft_attention_layers[5].is_causal == false and
     .draft_attention_layers[5].sliding_window == null)] == [true]
' "$STATIC_SPLIT_BUNDLE/air-manifest.json"
```

没有该字段的旧 AIR/OM 不会因拉取 v42 代码自动得到修复；不要补写字段伪装新产物。

## 对照门禁和仍待验证的差异

`tests/test_draft_attention_parity.py` 使用真实 Draft 类、六层 5+1 策略、小尺寸随机权重，
分别覆盖 FP32/FP16、K=1/3/7/15、window=1/8/4096、连续多轮 KV、无效载体污染、缺失策略拒绝，
以及固定尺寸的 strict `torch.export` 在不同 count/K 下仍对齐 compact eager。
另直接调用本分支 `dflash_ascend310p_ops` 的 strict dispatcher 做 CPU 对照，不启用 fallback。
这些都是源码/CPU 模拟证据，不是原始 checkpoint 接受率或真实 torch_npu/OM 运行证据。

本次主机验证：Python 全量 727 passed（另 43 subtests），C++/fake ACL 32/32，
源码锁 36 个文件全部匹配。固定 seed=4、FP32、K=15、window=4096 的同权重对照中，
上述修复前提交有 3/15 个 proposal ID 不同，修复后为 0/15；这是缩小模型的反例，不能
解读成真实 checkpoint 的接受率。所声明本地 profile 仍为 simulation-only，严格设备
preflight 没有发现可用 310P，ATC/真机门禁保留待执行。

接下来在同一物理 310P 上用同一 checkpoint、量化输入、prompt token IDs、chat template、EOS，
先比较相同逐轮 K 下的 proposal IDs、首次不一致位置、Target verify IDs 和 committed count，
再测接受率；仍须保持 ordinary/DFlash 的 token ID、EOS、stop reason 零差异。
不要用提高接受率替代正确性门禁，也不保证修复后一定达到 60%。

`max_new_tokens=32` 相同只固定生成预算，还不代表两个生成循环完全相同：

- `run_npu --block-size` 包含 anchor；K=15 对应 `--block-size 16`，不能把默认值当作 16。
- 当前 eager 按 `min(Kmax, remaining)` 选 K，C++ 为 correction/bonus 预留一个位置，按
  `min(Kmax, remaining-1)` 选 K，最后一个 token 走 Decode1。这次没有修改该调度差异。
- 当前 eager 首次零接受后转 target-only；用户 OM 报告为 fallback disabled，仍持续 Draft。
  后续是否继续提议会改变统计分母。本次没有自动改这个设置来提高数字。
- 对齐统计口径：报告的 token acceptance 是 `accepted_draft_tokens / drafted_tokens`，
  不等于“至少接受一个 token 的事务比例”。用户该用例为 19/97=19.59%，但非零接受事务
  为 7/11=63.64%；没有 eager 原始计数，不能断定其 60% 使用了哪一种口径。

重复同一短 prompt 十次用于稳定性/时延，不等于十个独立质量样本。只有冻结逐轮输入与
统计规则后，才能继续把残余差异定位到特征、精度、算子转换或调度，而不是仅凭总接受率猜测。
