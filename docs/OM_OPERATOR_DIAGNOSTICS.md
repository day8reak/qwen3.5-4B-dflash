# v47：通用 OM 逐算子诊断与同输入原生回放

本功能不限于 GDR。可选节点 dump、张量比较和原生回放是三个独立步骤：

| 功能 | 范围 | 不做什么 |
| --- | --- | --- |
| `prepare-dump` + `infer-cpp --acl-dump-config` | 原 OM 中按编译后节点名选择输入/输出 | 不改 OM，不默认 dump 全模型，不假定源算子与融合节点一一对应 |
| `compare` | 任意算子显式配对的逻辑 NPY 输入/输出，按声明的因果顺序报告首差 | 不猜布局、调用配对，不隐式 reshape/cast |
| `replay-operator` | 将捕获的同一组输入交给已安装的 `aten` / `npu` dispatcher | 不执行 case 中任意 Python，不用近似公式代替原生算子，不提供 CPU fallback |
| `replay-gdr-mtp` | 同输入 GDR-MTP 回放，额外验证本项目的七输入、两输出及三个属性 | 不宣称不同算子构建等价，不证明整个模型正确 |

适用示例包括投影/MatMul、DynamicQuant、Norm、激活/门控、cast、索引/状态提交和
注意力算子。是否能直接回放取决于该环境有无对应 dispatcher 和能否取得完整逻辑输入。
私有融合算子没有直接对应接口时，先比较它的边界，另行定义等价子图；不能自动拆成
若干基础算子后声称是同一个内核。其他算子目前没有被证明有问题。

## 1. 在原失败 OM 上采集

只需更新 Python 控制面并重建 runner 为 **1.29.0**；**不重导 AIR，不重编 OM**。
按工作区流程创建活动 run、激活已声明环境，沿用原失败 bundle、prompt/chat、EOS、
max_new_tokens=32、K=15、fallback 策略。所有新文件放在 `AI_RUN_DIR`，旧证据保留。
构建方法见 [Target 定点诊断](OM_TARGET_PARITY_DIAGNOSTICS.md)，使用新 build/report 路径。

CANN 支持通过 `aclInit(config)` 配置模型 dump。本实现将配置传入 `aclInit`，位于任何
模型加载之前，覆盖 phase-resident 后续加载实例；默认仍是 `aclInit(nullptr)`。
配置加载时机及按节点选择规则参见 [CANN dump API](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_0315.html)。
这是公开 API 路线，本地没有 CANN 9.0.0/NPU，实际部署版本的 dump 可用性仍待验证。

先用与 OM 匹配的 ATC 查看编译后结构。`ATC`、`VERIFY_OM` 由声明资源和原 bundle 解析；
在活动 run 下运行，不在源码目录生成编译器临时文件：

```bash
cd "$AI_RUN_DIR"
"$ATC" --mode=1 --om="$VERIFY_OM" --json="$AI_RUN_DIR/verify-structure.json"

"$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p.operator_diagnostics prepare-dump \
  --om-json "$AI_RUN_DIR/verify-structure.json" \
  --op-type GatedDeltaRuleMTP --limit 1 \
  --dump-dir "$AI_RUN_DIR/om-dump" --output "$AI_RUN_DIR/acl-dump.json"
```

`--mode=1 --om --json` 只生成结构 JSON，不编译新模型。参见 [ATC JSON 参数](https://www.hiascend.com/doc_center/source/zh/canncommercial/63RC1/inferapplicationdev/atctool/atctool_000062.html)。
若只诊断 Norm/MatMul，换成**结构 JSON 中实际出现的 type**；`--op-type` 可重复。
`--limit` 是每份 JSON 中所有匹配类型合计的前 N 个节点，不是每种类型各 N 个或 decoder 层号。
无匹配会报错，不回退到全模型 dump。精确指定非首个节点时，可修改新生成的 ACL 配置中的
`dump_list[].layer` 为所需编译后名称；`.plan.json` 描述初始选择，实际执行以冻结的配置为准。

随后在原 `infer-cpp` 命令上增加：

```bash
--diagnose-target-parity --diagnostic-max-transactions 2 \
--acl-dump-config "$AI_RUN_DIR/acl-dump.json"
```

这些是附加参数，不是独立 shell 命令。输出报告也必须使用新路径。
Python 会校验配置并保存 `<raw>.acl-dump.json` 快照；invocation 记录快照哈希。
只允许活动 run 内空目录、显式选定节点、`dump_op_switch=off`、`dump_level=op`，
拒绝全量 dump、异常/watch/profiling 混合配置及重定向 dump 路径的环境变量。
直接调用 C++ 时为 `--diagnose-target-parity true --acl-dump-config ...`；C++ 仅校验
模式和文件存在性，建议通过 Python 入口使用完整范围校验。

预填充、实际捕获和 Decode1 重放都可能产生 dump；指定节点名可能出现在多个 OM 中。
默认省略 `model_name`，避免把内存加载的 OM 文件名误当内部模型名。需限定模型时，必须
填写编译模型内部名称。节点名属于编译后 graph，不一定是 `dynamo.pbtxt` 名称。

诊断遇到 token 不一致仍保存 raw 并非零退出；这不是 dump 失败，检查是否已生成数据。
保留完整目录结构、raw/invocation/failure/log。CANN 的模型加载实例、stream/task ID、
算子执行 data_index 与 C++ 的 `model_execution_trace` 不是同一套编号；利用事务的
capture/chained/same-input trace 区间和节点身份配对，不能按文件名排序或单个数字猜调用。
原位更新的算子可能分别产生更新前输入和更新后输出文件，需要保留二者。

默认抓前两个 decode 事务，最多四个；仍可能有大量 Prefill/重放调用，先选少量节点。
可用 `prepare-dump --data stats` 看统计，但 stats 不是张量，不能用于同输入回放。
dump 有同步、内存和磁盘成本，所有相关运行均非正式性能证据；不要和无 dump 延迟混用。
更多路径和文件规则见 [离线模型 dump](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/devaids/ModelAccuracyAnalyzer/atlasaccuracy_16_0028.html)。

## 2. 转换为逻辑 NPY，再显式比较

使用本机匹配 CANN 版本的 `tools/operator_cmp/compare/msaccucmp.py`，路径记为
`MSACCUCMP`，解释器记为 `CANN_PYTHON`。先运行 `convert --help` 核验版本参数。
`DUMP_FILE` 指定一个原始 dump 文件（或该版本支持的直接父目录，不假定递归扫描）：

```bash
"$CANN_PYTHON" "$MSACCUCMP" convert \
  -d "$DUMP_FILE" -out "$AI_RUN_DIR/converted"
```

转换参数见 [msaccucmp convert](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/modelaccuracy/atlasaccuracy_16_0054.html)。
FRACTAL/NZ/paged cache 等物理布局需按真实 origin shape/layout 转换；不要将 raw 内存
直接 reshape 成 Q/K/V。工具不猜转换参数。Tensor 比较器只接受 native-endian、逻辑
C-contiguous 的实数/整数/bool NPY，单张量最多 512 MiB，不加载 pickle；BF16 等不能
无损表示为当前 NPY dtype 的格式需另行适配，不能偷换成 FP16。

活动 run 中创建 `pairs.json`，相对路径以该 JSON 所在目录为基准，例如：

```json
{
  "schema_version": 1,
  "operators": [{
    "id": "transaction1.layer0.projection",
    "inputs": [{"name": "x", "reference": "native/x.npy", "actual": "om/x.npy"}],
    "outputs": [{"name": "y", "reference": "native/y.npy", "actual": "om/y.npy"}]
  }]
}
```

```bash
"$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p.operator_diagnostics compare \
  --mapping "$AI_RUN_DIR/pairs.json" --output "$AI_RUN_DIR/operator-comparison.json"
```

`operators` 按人工确认的依赖顺序排列，报告首个失败节点、输入/输出各自的首差坐标、
最大有限绝对误差、numeric/bitwise 差异数和 NaN/Inf。输入已不同标为上游/输入映射问题；
输入对照一致而输出不同，只是缩小范围，还要核对**全部**输入、属性、布局、实现版本。
只填一部分 inputs 时，`mapped_inputs_bitwise_equal` 也只表示这部分相同。
默认 atol=rtol=0；可显式设置浮点诊断容差，报告保留原始差异数。整数 token/selector
永远精确比较，非有限值永远失败。空映射、缺文件、shape/dtype 不符不能变成 PASS。
该容差不改变模型的 zero token-ID mismatch 验收标准。

## 3. 任意已注册算子的同输入原生回放

用原 OM 捕获的输入执行当前环境的原生 dispatcher，检查同输入时输出是否一致。
例如已确认对应 `aten::mm.default` 的矩阵乘法，创建 `mm-case.json`：

```json
{
  "schema_version": 1,
  "layout": "logical-ND",
  "torch_op": "aten::mm",
  "overload": "default",
  "provenance": {"om_sha256": "实际哈希", "node": "实际编译节点名", "call": "实际调用身份"},
  "arguments": {"self": {"tensor": "x.npy"}, "mat2": {"tensor": "weight.npy"}},
  "outputs": {"result": {"index": [], "tensor": "om-output.npy"}}
}
```

```bash
"$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p.operator_diagnostics replay-operator \
  --case "$AI_RUN_DIR/mm-case.json" --output-dir "$AI_RUN_DIR/mm-native" --device-id 0
```

实际参数名必须符合已安装算子的 schema；工具记录 schema、版本、设备及 case/输入哈希。
参数编码可为 `{"tensor":"file.npy"}`、`{"value":64}`、`{"value":null}`、
`{"dtype":"float16"}`，也支持由这些项构成的嵌套 list。标量/list 属性从实际调用抄录，
不能仅根据输出形状反推。`index:[]` 选择单 Tensor 返回值，`index:[0]` 选择 tuple 第一个；
必须一对一覆盖所有 Tensor 输出，不能漏掉辅助 state/rstd 等输出再报告成功。
总输入和总期望输出各限 512 MiB。写输出前检查新目录，不覆盖已有证据。
没有真实 NPU、没有注册算子、参数不匹配或格式不支持时失败，不退回 CPU。

输出包括 `invocation.json`、各原生输出 NPY、`pairs.json` 和 `comparison.json`。
invocation 是执行前身份记录，`execution_status=not_recorded` 不代表执行成功；实际结果
看 comparison/命令退出码。只有输入输出比较通过才返回 PASS，且范围仅为该次映射。
CPU 单元测试只模拟 NPU transport，不能算真实 NPU 原生算子验证。

GDR-MTP 可用专用 `replay-gdr-mtp` 子命令，其 case 形式为：

```json
{
  "schema_version": 1, "layout": "logical-ND", "op_type": "GatedDeltaRuleMTP",
  "provenance": {"om_sha256": "实际哈希", "node": "实际节点名", "transaction": 1},
  "attrs": {"chunk_size": 64, "output_final_state": true, "use_qk_l2norm_in_kernel": true},
  "inputs": {
    "query": "query.npy", "key": "key.npy", "value": "value.npy", "g": "g.npy",
    "beta": "beta.npy", "initial_state": "initial_state.npy", "accepted_tokens": "accepted_tokens.npy"
  },
  "outputs": {"core_attn": "core_attn.npy", "last_recurrent_state": "last_recurrent_state.npy"}
}
```

本项目 ABI：Q/K/V FP16 `[B,T,H,D]`，g FP32、beta FP16 `[B,T,H]`，state bank FP32
`[B,T,H,Dk,Dv]`，accepted_tokens INT8 `[B]`，是**入口 bank 槽号**而非本轮长度。
输出比较包含所有物理行和全部状态槽，不切成仅 accepted prefix。
保留原物理 T=16 及 padding 输入，不能自行裁到 T=2 后称为相同输入。

## 结论边界

当前实现提供可执行的采集/配对/回放入口，不是全模型自动逐节点对齐系统；没有自行
采集当前分支 torch_npu 整模型的每个中间值，也未实现自动匹配融合节点/反解物理布局。
完整路径可先对照投影/conv/门控 → GDR → Norm/输出投影 → cache/commit，随后按首次
分叉扩展 full attention、量化和 Draft。不能把后续传播的差异都视作多个算子 bug。

相同输入的原生调用与 OM 输出仍不同，会缩小到该编译/执行区域，但也可能是注册版本、
属性、布局或编译融合差异；相同输入回放通过也不能排除原图的内存生命周期问题。
本功能不改默认 MTP、权重、图、精度、scheduler 或严格 greedy 权威；原真机 FAIL 尚未闭合。
dump 可能含权重和请求数据，留在当前 run，不提交到源码库或公开上传。
