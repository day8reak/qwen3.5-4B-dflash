# v45：可选 Target 状态快照与 Decode1 重放

runner 1.28.0 / Python 控制面提供默认关闭的 `--diagnose-target-parity`。
runner 1.29.0 进一步提供可选 `--acl-dump-config`，捕获原 OM 的选定中间节点，
配合[通用逐算子比较与原生回放](OM_OPERATOR_DIAGNOSTICS.md)；无需重编 AIR/OM。
用于定位已报告的 ordinary/DFlash 首个 token 分歧，不修正模型数值、不替换普通
Target 权威，也不把局部诊断当成整模型正确性或性能 PASS。

当前用户轨迹的首个分歧在第 2 个 Verify 事务：接受 5 个草稿后返回的**纠正 token**
是 2014，ordinary 在生成下标 8 / 含 prompt 下标 25 返回 27382。零接受回退尚未发生。
本功能默认抓前 2 个 decode 事务，覆盖这个边界；无需重新量化或重导已有 v43+ OM。

## 开启方法

更新 Python 源码并重建 runner；1.27.0 不认识新开关。沿用声明的模型、CANN 环境、
当前失败 bundle 和 runner 配置。在活动 run 下使用新的 build/output 路径：

```bash
"$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-diag128" \
  --output "$AI_RUN_DIR/reports/cpp-build-diag128.json" \
  --ascendcl-root "$ASCEND_HOME_PATH" --device-memory-policy normal-only

"$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$STATIC_SPLIT_BUNDLE/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/cpp-diag128/qwen35_dflash_incremental_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner-static-split.json" \
  --model-dir "$TARGET_MODEL_DIR" \
  --prompt "世界上最高的山是哪座山？ " --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --eos-token-id 248044 \
  --diagnose-target-parity --diagnostic-max-transactions 2 \
  --output "$AI_RUN_DIR/reports/target-parity-diag128.json"
```

这些变量指向现有声明资源；不要复制文档路径覆盖旧报告。若原失败请求有不同的
prompt/chat、EOS、K、max_new_tokens 或 fallback，以该请求为准。保持 max_new_tokens=32，
**不是改成 2**；`diagnostic-max-transactions` 只限制抓取轮数，不改生成预算和 Draft K。

`infer-cpp` 的开关没有 `true` 参数；直接调用 C++ 时使用
`--diagnose-target-parity true --diagnostic-max-transactions 2`。
C++ 定位模式使用 `--measurement-protocol profile`（此模式默认），不能和 evidence 混用。
Python 自动选择 profile；无需 msprof。内部没有 3+10 配对 benchmark，原 warmup/repetition
参数不用于定位重放。关闭定位开关后，原正式协议和默认同步/下载行为不变。

支持默认四个静态 OM：Prefill、Decode1、Draft、Verify16；要求
`dflash_sync_window=1`、`prefill_completion_policy=separate`。
不支持 fused/unified/dynamic 或合并窗口；不兼容时明确拒绝，不偷偷改变调度策略。
抓取数可选 1–4，默认 2。每个事务是一次 speculative Verify 或 target-only Verify K=0，
Prefill 不占此数量。遇到 EOS/长度结束则提前结束；Prefill 即结束时标为 `NOT_CHECKED`。

## 如何定位

定位分两段，额外读取/同步的时间不作为性能数据：

1. 复用现有 DFlash 调度器，保留真实 proposal、EOS、回退和终端 bonus 规则。
   抓取每个事务前后的 Target conv/GDR/KV/cursor、实际 Verify 输入 IDs 和 compact 输出。
   达到上限后，在下一次调用前停止抓取。
2. 复用同一个执行器和已分配的 Target 状态区，恢复主机快照，用 Decode1 逐 token 重放。
   不同时保留额外一份 HBM 状态，也不在每轮捕获中切换到 Decode1。

报告包含两组重放，不能混读：

| 字段 | 起点与用途 |
| --- | --- |
| `incoming_state_vs_chained_decode1` | 从共同 Prefill 状态连续 Decode1 到本轮入口，对比捕获的 Verify 入口状态；观察此前是否已积累状态差异 |
| `chained_decode1_tokens` / `chained_decode1_state_vs_verify` | 连续携带 Decode1 自己的状态，比较本轮 token 与提交状态 |
| `same_input_state_decode1_tokens` / `same_input_state_decode1_state_vs_verify` | 恢复该轮捕获的 Verify 入口状态后再执行 Decode1；隔离此前状态差异，观察本轮分块/提交/结果选择差异 |

两种重放都强制输入 DFlash **实际消费的已提交 token**，不把 Decode1 预测结果自动作为
下一输入。首个分歧后这属于 teacher-forced 比较，不是自由生成的 ordinary 序列。
共同起点是 DFlash Prefill 后的 Target 状态，不单独检验普通/DFlash Prefill 是否不同。

若入口状态已经不同而相同入口重放一致，应重点检查此前状态提交/累计数值差异；若相同
入口仍输出不同，则分歧在本轮计算、分块、输出选择等范围。二者都不是仅凭日志就证明某个
kernel 有 bug。当前怀疑的 ordinary GDR 与 Verify GDR-MTP 的精度差异仍需设备证据验证。

## 输出和退出码

上例会生成：

```text
reports/target-parity-diag128-runner-raw.json
reports/target-parity-diag128-runner-raw.json.invocation.json
log/target-parity-diag128-cpp-runner.log
```

raw 的 `report_kind=cpp-ascendcl-target-parity-diagnostic`、`status=DIAGNOSTIC`、
`formal_latency_evidence=false`，是独立 schema 1，不交给 paired PASS 验证器。
`diagnostic.token_parity` / `cursor_parity` 仅描述本次有限重放；有 token/cursor 分歧时，
**先保存 raw，再非零退出并写 `.failure.json`**。这时 Python 顶层 final JSON 不生成，
异常会给出 `target_parity_report` 路径。抓取/ACL/存储阶段失败可能没有完整 raw，需一并
保留 failure/invocation/log。无分歧则另写用户指定的 final JSON，仍不是模型 PASS。

快速查看首个 token 分歧（报告是合法 JSON，包括布尔值和非有限数）：

```bash
"$AI_MODEL_PYTHON" - "$AI_RUN_DIR/reports/target-parity-diag128-runner-raw.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    diagnostic = json.load(stream)["diagnostic"]
keys = ("captured_transactions", "token_parity", "cursor_parity", "first_token_mismatch")
print(json.dumps({key: diagnostic.get(key) for key in keys}, indent=2, ensure_ascii=False))
PY
```

`first_token_mismatch` 直接给出 1-based transaction、重放种类、0-based Verify 行号、
生成/绝对 token 下标、输入 token、Decode1 expected 和 Verify actual。
同事务优先列出 chained 重放，再列 same-input-state；完整列表在对应 `mismatches`。
`token_role` 区分 `accepted-proposal`、`target-correction`、`target-bonus`、`target-only`。
终端超预算 bonus 也参与图输出检查，但 `within_generation_budget=false`，不能误称最终
生成序列在该处不一致。`device_prediction_index` 与 host 绝对下标分别报告，避免 cursor
本身错误时误标首个生成位置。

状态摘要按状态数组 axis 0 分层，**不是 decoder 的原始层编号**。每层有 shape/dtype、
比较/排除元素数、bitwise/numeric 差异数、最大有限绝对误差、首个差异坐标和值、
NaN/Inf 计数。非有限值输出 null，并保留独立计数；不会写成非法 JSON `NaN`。
conv/GDR 全量对比；paged KV 只比较两个 cursor 的共同有效前缀。被拒绝推测的尾部 KV
不要求清零，不因这些无效位置不同就报告已提交状态错误。状态位级不同是观测，不单独
构成 token 不一致的因果证据；诊断退出门禁仍检查 token 和 cursor。

额外快照在主机内存；最多 2 GiB 保守快照预算，不包含已有 runner/模型的全部 RSS。
估算超额在 Prefill 前拒绝，可减少捕获轮数。额外 HBM 状态分配为 0，已有模型按原驻留
策略切换。报告保存摘要而非完整张量；快照在进程结束释放。

## 验证边界

已有 OM 不暴露原始 logits：这里的 Verify Top1 来自 accepted prefix 和纠正/bonus 的
compact 结果，不能分析 Top1–Top2 margin，也不是直接 OM/torch_npu logits 对照。
本地 fake ACL / CPU 回归验证参数开关、状态恢复、定位字段、EOS/K/回退边界、非零退出及
报告保护；不冒充 Ascend310P 上的实际数值。用户已报告的真机 FAIL 仍未闭合。
默认模式的首对即停、严格 zero mismatch 校验和当前权威模型均保持不变。

v47 raw 另含 `model_execution_trace`（实际 model ID/role/物理行数）及每事务的
`capture_execution_trace_range`、`chained_execution_trace_range`、
`same_input_execution_trace_range`。区间是该数组的 0-based `[begin,end)`，**不是**
CANN 的 task ID 或 dump data_index。配合 dump 子目录中的模型加载实例辨别调用，
不能把第 N 个 dump 文件直接当第 N 个 decode 事务。

## v46 可选模型对照说明

上述状态重放可复用 runner 1.28.0；要启用 dump 必须重编为 1.29.0。
仅增加诊断时不需要重编 AIR/OM；
启用 [v46 GDR 递推对照](OM_VERIFY_GDR_REFERENCE.md) 时则必须重新导出、编译模型。
默认仍用 GDR-MTP；请勿把新增对照理解为 MTP 已经修复。
