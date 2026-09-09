# v48：强制 ATC 保留原图 dtype

本次只修改 AIR→OM 编译策略，**继续使用 GDR-MTP，不改成两次 chunk GDR**。
四图拓扑、量化权重、Draft mask、accept/commit、FP32 MTP state bank 和 C++ 调度均不变。
这不是把整个模型改为 FP32，也不是把 recurrent state 强转 FP16。

## 1. 默认行为与拒绝条件

所有通过本框架编译的 AIR 图统一默认追加：

```text
--precision_mode=must_keep_origin_dtype
```

覆盖默认 Prefill / Decode1 / Draft / Verify 四图，以及显式 fused、动态和 recompute
诊断图；不能让 ordinary 和 Verify/Draft 因漏传参数而使用不同策略。
`compile-om`、`build-om`、`run-e2e`、`run-e2e-cpp` 共用此保护，
build/e2e 在加载 checkpoint 前校验，直接调用 Python 编译 API 也不能绕过。

若目标 ATC 使用 v2 参数，可以显式传：

```bash
--atc-arg=--precision_mode_v2=origin
```

它替代默认参数，不再追加旧参数。每次只能设置一个精度选项，包括重复同一个选项也会拒绝。
CLI 的 `--atc-arg` 使用 `--key=value` 形式；精度选项名的中划线别名会规范化并校验。

拒绝 `force_fp16`、`allow_fp32_to_fp16`、混合精度以及其他非 origin 模式，包括
`force_fp32`（强制全 FP32 也不是保留原图 dtype）。同时拒绝 `customize_dtypes`、
`input_fp16_nodes` 等可改变节点 dtype 的覆盖，以及已有 ABI 核心选项覆盖。
其他诊断参数，例如 `--atc-arg=--log=info`，仍可使用。
ATC 在 origin 模式失败时直接保留日志、非零退出，**不会自动重试降精度**。

为什么不能只看 Python 的 `.float()`：显式 FP32 子图仍需匹配的编译精度策略。
CANN 的默认精度策略、origin 约束及选项互斥见
[ATC 参数概览](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/atctool/atlasatc_16_0039.html)
和 [precision_mode_v2](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/atctool/atlasatcparam_16_0069.html)。
节点级 `customize_dtypes` 的高优先级见
[官方说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910/devaids/atctool/atlasatcparam_16_0075.html)；
因此本保护不接受额外 dtype 配置文件。

这是编译器策略保护，不能证明内核内部每一步精确等价；某些原始 dtype 不受支持时可能
编译失败或改变节点执行位置。不要静默降级或关闭原有导出/运行门禁，应保留首条 OP/GE
错误、ATC 日志及实际图结构，再按 [逐算子诊断](OM_OPERATOR_DIAGNOSTICS.md) 定位。

## 2. 本次需要重建什么

- **Python 控制面：更新**，确保 PYTHONPATH 指向拉取新提交后的 framework/python。
- **C++：不需要重建**，原 runner 1.29.0 可继续使用。
- **AIR：本次不需要重新导出**，可复用原本已通过审计的完整 AIR + 外置权重 payload。
- **OM：必须重新编译整个 bundle**。只重新执行 infer-cpp 或修改旧 manifest 标签无效。

上面只适用于 v48 的编译策略改动。若旧 AIR 尚未包含 v42/v43 的图内语义修复，
或还要更改 factory / shape / GDR policy，仍须按 [静态四图指南](STATIC_SPLIT_OM_DEFAULT.md)
重新导出 AIR。不要重新量化或为本次试验修改模型精度、输入、K、EOS、fallback 策略。

保留原失败 bundle 作为对照。按工作区规则启动**新的活动 run**、激活既有环境，
设置源码 PYTHONPATH，并将 `OLD_BUNDLE` 指向原 bundle。编译器拒绝写入非空 om/，
因此复制原 AIR 与 manifest 到新 bundle；不复制旧 om/ 或旧 deployment-manifest。
这些变量与解释器/ATC 均使用已声明资源，不新建环境：

```bash
: "${AI_RUN_DIR:?请先创建新的活动 run}"
: "${OLD_BUNDLE:?请设置原完整 bundle 路径}"
export PRECISION_BUNDLE="$AI_RUN_DIR/artifacts/quant-dflash-static-split64-origin"

test -f "$OLD_BUNDLE/air-manifest.json" &&
test -d "$OLD_BUNDLE/air" &&
test ! -e "$PRECISION_BUNDLE" &&
mkdir -p "$PRECISION_BUNDLE" &&
cp -a --reflink=auto "$OLD_BUNDLE/air" "$OLD_BUNDLE/air-manifest.json" "$PRECISION_BUNDLE/" &&
(
  cd "$AI_RUN_DIR" &&
  "$MODEL_PYTHON" -m qwen35_dflash.ascend310p compile-om \
    --air-manifest "$PRECISION_BUNDLE/air-manifest.json" \
    --atc "$ATC_BIN" --soc-version Ascend310P3
)
```

`Ascend310P3` 应与原产物/实际目标一致。原 AIR 下的所有 payload 文件会重新核对
大小和 SHA256，不更改 AIR manifest；reflink 不支持时会实际复制，请预留存储空间。
新 OM 因图/编译策略变化可能有不同的 weight/work 内存需求，应以实际查询为准。

接着沿用原 infer-cpp / Target parity diagnostic 命令，**只将 deployment manifest
替换为新 bundle 的文件，并给 report 使用新路径**：

```bash
--deployment-manifest "$PRECISION_BUNDLE/deployment-manifest.json"
```

上面是原命令的替换参数，不是独立命令。需要快速复现现有首差时，继续使用
`--diagnose-target-parity --diagnostic-max-transactions 2`，保持同一 prompt/chat、
EOS=248044、max_new_tokens=32、K=15 及原 runner 配置。诊断不属于正式性能证据。
重新编译后节点名可能变化，若同时 dump，必须从**新 OM** 重建结构映射和节点选择。

## 3. 不用 jq 检查是否生效

```bash
"$MODEL_PYTHON" - "$PRECISION_BUNDLE/deployment-manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    manifest = json.load(stream)
compiler = manifest["compiler"]
allowed = {
    "--precision_mode=must_keep_origin_dtype",
    "--precision_mode_v2=origin",
}
assert compiler["precision_policy"] == "preserve_graph_dtypes"
selected = [arg for arg in compiler["extra_args"] if arg.startswith("--precision")]
assert len(selected) == 1 and selected[0] in allowed, selected
for graph in manifest["graphs"]:
    actual = [arg for arg in graph["atc_command"] if arg.startswith("--precision")]
    assert actual == selected, (graph["role"], actual)
    print(graph["role"], actual[0], graph["om"]["sha256"])
print("compiler policy: preserve_graph_dtypes")
PY
```

manifest 的 compiler.extra_args 记录**实际生效**参数，包括自动追加的 origin；
graphs[].atc_command 记录每次真实命令。原 C++ 报告的 control_plane.compiler 也会携带
该字段。旧 manifest 缺字段只说明它没有记录新策略，不允许手工补字段冒充重新编译。
以上仅核验编译配置和产物身份，不代替实际执行后的 tensor / token 对照。

## 4. 证据边界

主机回归覆盖默认注入、两种 origin 形式、参数别名、冲突/重复/覆盖拒绝、
checkpoint 加载前失败、全部静态四图使用同一策略、动态/重算调用兼容，以及编译失败
不降精度重试。MTP custom-op 审计随 manifest 保留，不切换算子或放宽接受标准。

目前没有本次新 OM 的真机结果，不能宣称接受率恢复或 MTP 根因已修复。
先通过相同前缀下的 Target parity 诊断，再取得 OM ordinary/DFlash 与当前 torch_npu
参考的零 token/EOS/stop 差异；最后才比较无 dump 的同机 3+10 性能。
