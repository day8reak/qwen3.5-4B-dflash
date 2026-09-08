# 默认：合并 Prefill + Decode1 + 静态 Draft + Verify16

从 v41 / C++ runner 1.25.0 起，默认构图采用四个静态 OM。先在此路径跑通并完成严格
greedy 对齐，再另建动态候选；本次不缩减模型、量化精度、KV 容量或请求长度。
这是用户选择的源码默认值，不代表已经取得 Ascend310P 真机正确性或性能结论。

**v42 修复 Draft mask 语义，四图默认拓扑仍保持 v41。** 旧显式状态 Draft 把最后一层
full attention 错误覆盖成 causal；新实现保留逐层策略，并在 K<15 时屏蔽物理 block padding。
已经跑通 v41 的用户也需从 `export-air → compile-om` 重建新 bundle；只更新 C++ 无效。
详见 [Draft/eager 对齐与重跑说明](DRAFT_OM_ATTENTION_PARITY.md)。这次没有重新量化或修改
C++ ABI，匹配四图 ABI 的 runner 1.25.0 可继续使用；接受率改善仍需真机测量。

## 文档入口与版本范围

本页是当前 AIR/OM 部署的主操作入口。以下旧文档中的显式 factory、配置和报告检查
不会随着 CLI 默认值变更自动迁移；不能仅凭“都是四个 OM”混用。

| 需要做什么 | 使用的文档 | 适用范围 |
| --- | --- | --- |
| 新默认构图、重跑、静态尺寸与分组显存 | 本页及所链接的 static-split factory/runner 模板 | v41 / runner 1.25.0 起 |
| 环境、自定义算子 ABI、历史 AIR/ATC 排错 | [框架参考](QUANT_AIR_OM_FRAMEWORK.md) | 第 4–11 节保留旧 fused 对照；不是默认运行命令 |
| 复用旧正确静态 fused OM，仅修改生命周期 | [v40 显存说明](STATIC_OM_MEMORY.md) | 仅旧静态 fused，不适用新 Prefill ABI / Draft shape |
| 旧 fused 静态导出或动态排错 | [静态 fused 基线](STATIC_FUSED_OM_BASELINE.md)、[性能候选参考](INCREMENTAL_OM_PERFORMANCE.md) | 显式回退 / 独立候选 |
| PyTorch eager rollback 对照 | [rollback 运行与验证](DFLASH_RUN_AND_VALIDATE.md) | 不是 OM 生成或 OM 验收 |

四图拓扑默认值最初对应源码提交 `acb43d5b3d975e3204e75d915e7a41375ed0f6b0`。
该源码的主机验证为 Python 676 passed（另 43 subtests）、C++/fake ACL 32/32、
ASan/UBSan 生命周期专项 4/4，以及 31 个 C++ 报告正例 / 186 个损坏报告拒绝用例。
这些不是新 OM 的 ATC 编译、真机显存、零 token 差异或加速证据；设备门禁仍待执行。

## 构图和常驻范围

| 物理 OM | 输入 / 输出数 | 固定物理行数 | 生命周期 |
| --- | ---: | --- | --- |
| `target-prefill` | 9 / 11 | prompt chunk 64 | body + Top1/EOS head 合为一个 OM，连续 prompt 块之间常驻，阶段结束后卸载 |
| `target-decode1` | 8 / 8 | 1 | ordinary 热循环常驻；DFlash 尾部或显式 target-only 回退使用 |
| `draft-propose` | 8 / 4 | Target feature N=64，proposal block 16 | 与 Verify 一起常驻，不在每轮交替时重载 |
| `target-verify-commit` | 9 / 13 | 16 | 最多 15 proposals，按实际接受数提交，与 Draft 一起常驻 |

不再生成单独的 `target-prefill-head` 或 `fused-speculative-step`。
Prefill 复用原 body 与精确 head：原 7 个输入追加 EOS 表及 count，原 8 个输出追加
`committed_token_ids`、`commit_count`、`finished`。小的 `last_hidden` 输出仍保留在 ABI 中。
N 大于 64 时，prompt 仍分块处理，每块都会计算 head，但只观察最终块的 compact 结果；
中间块的 head 不推进逻辑 cursor。Target/Draft 的权重、计算精度和 accept/commit 数学不变。

默认 `phase-resident` 的活动组是：

```text
ordinary：{Prefill} → {Decode1，循环}
DFlash：  {Prefill} → {Draft + Verify，循环} → {Decode1，仅尾部/回退需要时}
```

启动时逐个加载、复制并校验 metadata、卸载，避免启动即分配所有权重。
真正换组前先完成 stream 上的待执行工作，检查卸载成功，再复用权重区。
Draft/Verify 的两份活权重占据互不重叠的区间；两者交替执行不换组。
KV、GDR/conv state、feature 和 compact-result 缓冲区由应用持有，不随 OM 卸载。
每次重载都重新校验物理 I/O ABI；失败停止，不生成 PASS 报告。

以 `aclmdlQuerySize` 的 weight/work 值为准，显式分配预算为：

```text
phase weights = max(Prefill, Decode1, ALIGN_UP(Draft, 512) + Verify)
workspace     = max(四个 OM 的 work_bytes)
显式总量       = phase weights + workspace + state/carriers 等应用缓冲区
```

GE、driver 内部分配和其他进程占用另计，需要真机测量。这不是压缩 OM 文件，也不假定不同
OM 自动共享权重。它避免四图权重同时常驻，但不能保证在所有设备/空闲量下都不再 OOM。
显式 `all-resident` 可用于对照，预算恢复为四份权重之和。
短到不进入 Draft 的请求允许仅 Prefill 组常驻，并不要求人为装载两张热图。

Decode1 不能在这次修改中删除：ordinary 依赖它；DFlash 剩余生成预算为 1，或启用
`request-target-only` 后触发零接受回退，也使用它。正常的 Draft→Verify 事务不调用 Decode1。

## 静态尺寸与逻辑长度

四图均要求 `dynamic=false`、`input_dim_gears={}`，不设置动态 shape/gear。
`draft_static_feature_rows` 默认 64，可另导出 128 等 64 的正整数倍，且不得超过 KV capacity。
`example_sequence_length` 不是这个开关。W8A8 的 DynamicQuant 量化算子也不等于动态形状。

每次 Draft 都绑定完整 N 行；运行前在同一 stream 清零源载体后的区域。首轮源为真实 prompt，
后续由 feature 策略决定源载体范围，设备 count 继续屏蔽无效行。`fixed-16` 指后续源载体
通常为 16 行，不是把静态 Draft OM 改成 16 行。`committed-prefix` 也不能减少固定 N 的物理计算。
Verify 输出只有 16 行，不能假定其余 feature 区域已被覆写。

请求在 tokenize/chat template **之后**必须满足：

- `prompt_tokens <= N`，超出报错，不截断。
- `prompt_tokens + max_new_tokens + N <= KV capacity`，给整段静态 cache write 留出空间。

增大 N 需要新 AIR/OM，且增加补齐行的计算。动态方案在静态真机门禁通过后另行验证，
不会通过手改 manifest 或 pbtxt 把动态 OM 标成静态。

## 从哪里重新运行

**重建 C++ runner，并从 export-air → compile-om 重新生成四图。**
这次改变了 Prefill ABI 和 Draft 物理 shape：旧的 body/head/fused OM 不能靠改名、重排
manifest 或只重建 runner 转成新默认。v40 文档中“不用重新导出”仅适用于旧静态 fused
拓扑的生命周期修复，不适用于此次 v41 默认构图。
沿用已锁定的 checkpoint、量化输入和 CANN 环境，不需要重新量化；新产物放活动 run 的新目录，
保留旧 bundle 作为回退。

1. 填写 [factory 模板](../config/quant_air_om_static_split_factory.example.json)，保存为
   `$AI_RUN_DIR/factory-static-split64.json`。保留真实 checkpoint、量化 manifest 和 receiver
   路径，使用 `merged_prefill: true`、`draft_static_feature_rows: 64`。从旧 fused 配置迁移时移除
   `fused_speculative_step`、`unified_target_step`、`fused_static_feature_rows`，不要混用选择器。
2. 填写 [runner 模板](../config/quant_air_om_static_split_runner.example.json)，保存为
   `$AI_RUN_DIR/runner-static-split.json`。必须填真实设备、CANN、driver、firmware 身份。
   首轮保持 `phase-resident`、`async-memset`、`fixed-16`、window 1、`separate`、fallback disabled。
3. 在已经激活的模型/CANN 环境执行；每步成功后再进行下一步：

命令须使用本分支的 `framework/python`，不要误用 workspace 旧参考实现或旧安装包。
从当前部署源码根目录设置模块路径（不新建 Python 环境）：

```bash
export DFLASH_SOURCE="$PWD"
export PYTHONPATH="$DFLASH_SOURCE/framework/python:$DFLASH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
```

```bash
export STATIC_SPLIT_BUNDLE="$AI_RUN_DIR/artifacts/quant-dflash-static-split64"

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-static-split" \
  --output "$AI_RUN_DIR/reports/cpp-build-static-split.json" \
  --ascendcl-root "$ASCEND_HOME_PATH" --device-memory-policy normal-only

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_state_graphs \
  --factory-config "$AI_RUN_DIR/factory-static-split64.json" \
  --bundle-dir "$STATIC_SPLIT_BUNDLE"

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$STATIC_SPLIT_BUNDLE/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version Ascend310P3

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$STATIC_SPLIT_BUNDLE/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/cpp-static-split/qwen35_dflash_incremental_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner-static-split.json" \
  --model-dir "$TARGET_MODEL_DIR" \
  --prompt "请用一句话解释为什么天空是蓝色的。" --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --output "$AI_RUN_DIR/reports/cpp-infer-static-split.json"
```

`MODEL_PYTHON`、`ATC_BIN`、`ASCEND_HOME_PATH`、`TARGET_MODEL_DIR` 使用当前已声明、
已验证的环境/资源路径。为复现问题，优先保持原失败用例的 prompt/chat 参数。
`export-air`、`build-om`、`run-e2e-cpp` 省略 `--factory` 时现在也选择此 factory；上面显式
写出便于审计。Python `run-e2e` 和单图 probe 仍是重算诊断入口，不应混淆。
Python 控制面会传 `--draft-static-feature-rows N`；直接调 C++ 的新四图模式默认为 N=64，
其他 N 必须显式传参。任何 manifest/物理 OM 不一致都应停止。

## 核验与回传

编译后的 manifest 应只有这四个角色，所有图均静态，Draft 保留 `draft_static_shape`。
下列检查不能代替 runner 对实际 OM 的 dtype/shape/bytes 检查：

```bash
jq -e '
  ([.graphs[].role] == ["target-prefill", "target-decode1", "draft-propose", "target-verify-commit"]) and
  ([.graphs[] | (.dynamic == false and .input_dim_gears == {})] | all) and
  ([.graphs[] | select(.role == "target-prefill") |
    ((.input_names | length) == 9 and (.output_names | length) == 11)] == [true]) and
  ([.graphs[] | select(.role == "draft-propose") |
    (.draft_static_shape.feature_rows == 64 and .draft_kv_repeat_audit.status == "PASS")] == [true])
' "$STATIC_SPLIT_BUNDLE/deployment-manifest.json"
```

AIR 输入审计会验证所有四图的固定正整数 shape；Draft 必须有 8 个绑定。
已有自定义算子、Scatter index Tile、GQA K/V Tile 导出审计继续执行，不能删门禁来通过编译。
报告的 `abi.physical_topology` 为 `merged-prefill-four-static-split-v1`；检查：

- 独立 `target_prefill_head_executions=0`，`fused_speculative_step_executions=0`；
- `draft_static_feature_rows=64`，`draft_dynamic_shape=false`，Draft 动态计划与 OM gear 数为 0；
- `draft_static_physical_feature_rows = 64 * draft_propose_executions`，source + padding = physical；
- phase 的 `allocated_weight_bytes` 符合上述分组公式；`peak_resident_models` 最多 2；
  `split_group_loads` 等于初始 prefill Draft 调用数，不随每轮 Draft/Verify 交替增长；
- 换组同步、卸载、重载、ABI 检查都计入生成 wall time，`model_load_excluded_from_latency=false`。

`models[].model_id` 在 phase 模式仅是启动检查 ID，不是永久角色 ID。
当前 `analyze-msprof` 尚不支持 phase 模式跨加载周期的归因，会明确拒绝；可保留原始 profile
及带 role 的执行 trace，不要用启动 ModelId 关联整个运行期间的 CSV。

本地覆盖静态导出语义、fake ACL 分组生命周期/受限预算/故障清理及 C++→Python 报告门禁；
均属主机模拟证据。还须在物理 310P 禁用 fallback 后取得 ordinary/DFlash token ID、EOS、
stop reason 零差异，并完成 3 warmup + 10 次未 profiling 测量，才能声称设备跑通/性能改善。
若失败，回传新 manifest、model-query/分配诊断、第一条 GE/OP ERROR 上下文及完整 runner 日志；
只看到末尾 ACL 500002 不能认定仍是同一个算子根因。

## 导出前的源码锁错误

`ValueError: quant source differs from SOURCE_LOCK: docs/DFLASH_RUN_AND_VALIDATE.md`
发生在加载权重、实际 AIR 构图之前。`SOURCE_LOCK.json` 也锁定部分文档：`798e2c2`
更新了上述文档，但漏同步 `npu_benchmark.documentation_sha256`，因此干净 Git 提交也会失败。
后续修复仅同步这个已审阅文档的哈希，保留全部源码锁检查，不改变模型、runner 或 OM ABI。

使用修复提交后，在本文配置的模块路径和模型环境中可以先执行不加载权重的真实门禁：

```bash
"$MODEL_PYTHON" -c 'from qwen35_dflash.ascend310p.quant_factory import _verify_quant_source_lock; print(_verify_quant_source_lock())'
```

门禁通过后，失败的导出从 `export-air` 重跑；已生成且有效的 v41 OM 不因文档哈希修复而失效。
这句话仅适用于文档哈希修复，不适用于 v42 的 Draft mask 修复；后者必须重新导出/编译。
不要关闭检查、删除锁条目、批量重算未知脏工作树的哈希或清理无关本地文件。
以后包括纯文档提交在内，也应运行 `tests/test_source_lock_benchmark.py` 与静态四图回归中的
真实源码锁正例 / 文档变更拒绝用例；最终提交前跑完整测试。

## 显式回退

- 旧静态 fused 四图：使用 `create_quant_fused_speculative_step_graphs` 和
  [原静态 fused 配置](../config/quant_air_om_static_fused_factory.example.json)，runner 可显式选
  phase-resident 或 all-resident；旧 bundle 只作为对照，修复 Draft mask 需重导 fused AIR/OM。
- 旧 split-head 五图：仍用 `create_quant_incremental_state_graphs`，但设
  `merged_prefill: false`、`draft_static_feature_rows: 0`，Draft 回到动态契约；默认 all-resident。
- 原 unified Target-step 或 recompute：显式选择对应旧 factory 与匹配配置，不与新四图混装。

回退用独立 bundle；不能把新 `draft_static_shape` 填进旧动态图来绕过 ABI 门禁。
