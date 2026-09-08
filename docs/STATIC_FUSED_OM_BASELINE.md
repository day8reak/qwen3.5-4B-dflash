# 先跑通静态 fused OM，再恢复动态

如果已生成正确的 v39 静态 OM，但启动时报显存不足，先看
[v40 按阶段常驻修复](STATIC_OM_MEMORY.md)。它只需重建 runner / 改配置，
不要求再导出同一份 OM；下文的重新导出步骤适用于尚未生成正确静态图的情况。

v39 / C++ runner 1.23.0 提供显式静态候选和 Draft KV 头复制修复。原动态配置保留供回滚；
**本轮先使用固定 N=64。** Host/Fake 测试不是 CANN/310P 数值证据，
必须在设备上重新导出、编译、运行。

## v39：静态输入不会自动移除 KV 头复制的 BroadcastTo

2026-09-08 接收端日志中的 FP16
`[1,8,1,2064,128] -> [1,8,4,2064,128]` 在 `MatchConstShape` 报
`input shape and const shape not match`，随后 auto-tiling 失败并返回 ACL 500002。
这组尺寸符合广播规则；仅凭日志不能断言常量错误或 CANN 不支持五维广播。
它与 v37 的 **INT64 Scatter 索引**不是同一节点语义，不能仅按 `BroadcastTo_1` 名称定位。
2064 是 KV capacity 2048 加 block 16，不是 feature 行数 N，也不能证明整图已切静态。

v39 把 `AirDFlashOps.attention` 的 K、V 共用 `_repeat_kv` 改为
`unsqueeze(2) -> repeat(1,1,G,1,1) -> reshape`，保持每个 KV 头相邻复制的顺序。
不能直接在原 head 轴上 `repeat(1,G,1,1)`，那会改变 GQA 的头对应关系。
FP32 attention/softmax、mask、原量化权重和计数/state ABI 均不改变。

新 cached Draft 图声明 `draft_kv_repeat_policy: gqa-head-repeat-tile-v1`。
导出沿每个 K/V cache write 的 `ScatterElements -> Concat -> Unsqueeze -> Tile -> Reshape`
检查连接（允许中间 Identity），要求 group 轴为 2，并从 Const **实际二进制数据**核对
`[1,1,4,1,1]`，不靠可读字符串或 Tile 总数断言通过。
锁定的六层 Draft 必须得到 12 条不同的 K/V 绑定；group=1 不需要复制。
缺少 pbtxt、仍走相关 BroadcastTo、错误 repeats/连接或缺失绑定时停止并保留 AIR 诊断。
无关的 mask/state BroadcastTo 不在本门禁禁止范围内。
`compile-om` 在调用 ATC 前验证 `draft_kv_repeat_audit`，并将其保留到 deployment manifest。
旧 bundle 可回滚，但不自动声明含有本修复。

本修复的 CPU/FX/模拟 GE 审计不是实际 TorchAir、ATC 或 Ascend 310P 推理证据。
还需要新静态 AIR/OM 的真实导出检查，以及下文的严格 greedy 和重复运行门槛。

## 固定什么，不固定什么

| OM | 物理输入行数 |
| --- | --- |
| target-prefill | 64 |
| target-prefill-head | 1 |
| target-decode1 | 1 |
| fused-speculative-step | feature 输入 N=64；内部 Target verify T=16 |

四个图均以 `dynamic=False` 导出，不声明 input_dim_gears。KV 容量保持 2048，
仅驻留一份 fused OM，不为 prefill/decode 各复制一份融合权重。
committed_input_count、previous_commit_count、proposal_count、EOS 和两个 logical cursor
仍是运行时 Tensor，不能冻结为导出示例值。补零不会增加逻辑提交数。
W8A8 的 DynamicQuant 是量化算子，与本次固定 tensor shape 是两回事，不应替换它。

Runner 从 OM 读取实际 shape，并要求其与 manifest 的静态 N 一致；静态 fused dataset
不调用 aclmdlSetInputDynamicDims 或 aclmdlSetDatasetTensorDesc，也不分配动态控制输入。
每次执行前，在同一 stream 上清零 feature 源载体之后的区域；首次以真实 prompt 长度为界，
后续以 16 行（或已知 committed-prefix/待同步上界）为界，设备 count 继续屏蔽无效行。
Target verify 只输出 16 行，不能假定 64 行输入缓冲区的其余部分已被覆写。

静态初版有明确边界：

- tokenize/chat template **之后** prompt_tokens <= N；超出则拒绝，不截断。
- prompt_tokens + max_new_tokens + N <= KV capacity。当前 cache scatter 会写整个
  物理载体；预留 N 行避免补齐行 clamp 到最后一个有效 KV 槽位。这是保守的基线门禁，
  不是缩小 KV 分配，也不是已经验证了末尾回绕写入。
- N 必须是 64 的正整数倍，且不超过 KV capacity。更大 prompt 可另导出 N=128 等档位。
  增大 N 会增加无效 Draft 计算，不能假设比动态快。

当前 prompt=17、max_new_tokens=32、N=64、KV=2048 满足上述门禁。

## 从哪里重跑

复用已锁定的 checkpoint、量化输入、环境和 runner 身份配置；无需重新量化。
**从 export-air 开始重跑 AIR → OM，并重建 C++ runner**。重新编译旧动态 AIR、
只改 manifest 的 dynamic 字段或只重编 C++ 都不能得到静态模型。
v39 的 KV 修复同样需要从新源码导出 AIR，不能只重编 v38 的 AIR。
C++ ABI 仍是 1.23.0；如果已确认用上该 runner，可复用它，新 Python 控制面会打印
`stage=fused-manifest mode=static feature_rows=64 om=... sha256=...`，便于核对实际运行产物。
这一行在 `infer-cpp` 控制面的 stderr，而非 C++ 子进程日志中；它说明 manifest 选择，
实际 OM 的输入 shape 仍由 C++ 校验。
所有新产物使用活动 run 下的新路径；保留旧 bundle。

1. 将已成功导出的 `factory-fused.json` 另存为
   `$AI_RUN_DIR/factory-fused-static64.json`，保留真实路径，仅新增
   `"fused_static_feature_rows": 64`。也可填写
   [静态配置模板](../config/quant_air_om_static_fused_factory.example.json)。
   `example_sequence_length: 64` 不是这个开关，不能代替它。
2. 使用当前声明的模型 Python、CANN 环境和同一 fused factory：

```bash
export STATIC_FUSED_BUNDLE="$AI_RUN_DIR/artifacts/quant-dflash-fused-static64"

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_fused_speculative_step_graphs \
  --factory-config "$AI_RUN_DIR/factory-fused-static64.json" \
  --bundle-dir "$STATIC_FUSED_BUNDLE"

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$STATIC_FUSED_BUNDLE/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version Ascend310P3

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-static64" \
  --output "$AI_RUN_DIR/reports/cpp-build-static64.json" \
  --ascendcl-root "$ASCEND_HOME_PATH" --device-memory-policy normal-only
```

`ATC_BIN` 和 `ASCEND_HOME_PATH` 使用已验证的当前 CANN 路径；
不要填另一个 toolkit 的路径。`MODEL_PYTHON` 使用已有模型环境解释器。
每条命令成功后再运行下一条；首次失败保留完整日志，不继续使用残缺 bundle。

3. 编译后检查 manifest，不按固定文件名拼接 OM：

```bash
jq -e '
  (.graphs | length == 4) and
  ([.graphs[] | (.dynamic == false and .input_dim_gears == {})] | all) and
  ([.graphs[] | select(.role == "fused-speculative-step") |
    .fused_static_shape.feature_rows] == [64]) and
  ([.graphs[] | select(.role == "fused-speculative-step") |
    (.draft_kv_repeat_audit.status == "PASS" and
     .draft_kv_repeat_audit.policy == "gqa-head-repeat-tile-v1" and
     .draft_kv_repeat_audit.layers == 6 and .draft_kv_repeat_audit.groups == 4 and
     .draft_kv_repeat_audit.repeats == [1,1,4,1,1] and
     (.draft_kv_repeat_audit.files | length > 0) and
     ([.draft_kv_repeat_audit.files[] | (.bindings | length == 12)] | all))] == [true])
' "$STATIC_FUSED_BUNDLE/deployment-manifest.json"
```

导出还会核查 GE Data 的 15 个输入均为正整数 shape，与静态 example 一致，
并保留已有 public-input、custom-op、ScatterElements/Tile 及新的 KV 头复制审计。
不要手改 pbtxt、删审计或给 ATC 追加动态参数绕过门禁。

4. 先用 `draft_feature_policy: "fixed-16"`、`dflash_sync_window: 1`、
   `prefill_completion_policy: "separate"`、`zero_accept_fallback_policy: "disabled"`
   运行原来失败的用例。复制已填写真实硬件身份的 runner 配置为
   `runner-fused-static64.json`，保持 `state_policy: "incremental-explicit-state-v2"`。

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$STATIC_FUSED_BUNDLE/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/cpp-static64/qwen35_dflash_incremental_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner-fused-static64.json" \
  --model-dir "$TARGET_MODEL_DIR" \
  --prompt "请用一句话解释为什么天空是蓝色的。" --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --output "$AI_RUN_DIR/reports/cpp-infer-static64.json"
```

`TARGET_MODEL_DIR` 指向已锁定的 Target checkpoint。为复现本次问题，应优先使用
原运行命令的同一个 prompt/chat 参数；上面的短 prompt 只是示例。
控制面自动向 runner 传入 `--fused-static-feature-rows 64`，直接调用 C++ 时也必须提供它。
原动态 OM 配这个参数，或静态 OM 未提供该参数，都应拒绝执行。

## 通过门槛与回传

首次 fused decode 必须出现 decode-done，最终 ordinary/DFlash token ID 与 EOS 必须零差异。
正式报告保留 3 次 warmup、10 次测量；不能把 fake ACL PASS 或仅能 load OM 当作跑通。
检查 `model_memory_query` 与 `execution_io_counters`：

- `fused_static_feature_rows == 64`，`draft_dynamic_shape == false`；
- Draft 的实际 OM gear 数及动态 shape-plan 数均为 0；
- `fused_static_physical_feature_rows == 64 * fused_speculative_step_executions`；
- source rows + padding rows == physical rows；padding memset 单独计数。
  旧 `draft_verify_feature_input_rows/rows_elided` 描述源载体，不代表静态图少计算了这些行。

若失败，回传本次 runner 日志的第一条 GE/OP ERROR 附近上下文、静态 manifest、
AIR input 审计和 OM metadata；不要只截清理阶段的 500002。
静态真机结果与重复执行稳定后，再以保留的动态配置另起 bundle 验证，
不要覆盖已跑通的静态回滚基线。本改动不宣称解决了所有 CANN 动态算子问题。
