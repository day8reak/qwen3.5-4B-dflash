# v46：可选 Verify GDR 数值对照，默认保留 MTP

## 为什么增加对照

用户真机诊断：相同 Target 入口状态、输入
`[25,271,16,13,220,2972]` 下，Decode1 逐步输出最后一个 token 为
`27382`，Verify 返回 `2014`。逻辑 cursor 一致；更早的事务已经出现 recurrent
state 差异。这证明两条路径目前不等价，但不足以证明是哪条内核指令出错。

当前 ordinary Decode1 使用 `npu_chunk_gated_delta_rule(chunk_size=1)`，
每次返回的 recurrent state 转成 FP16，再由 OM 状态接口转回 FP32。
Verify 的 MTP 算子返回 FP32 状态库，没有相同的显式逐 token 舍入。
这只是模型侧可见的差异，不证明 MTP 内部没有舍入，也不证明它是本次失败根因。
只在最终 bank 上补 cast 一般不能保证复现逐步状态反馈；需要真实同输入对照确认。

本仓库只有 MTP 调用桥和导出契约，没有该自定义算子的 kernel/host 实现。
如果后续同输入证据将分歧定位到 MTP 内部，再检查实际部署版本对应的算子源码、
构建配置和精度约定。现阶段应先排查模型包装和导出边界，不能仅凭状态差异
归因于 MTP 内核，更不能弱化普通 Target 权威。

## 下一步：定位首次分歧，而非替换 MTP

v47 已增加[选定节点 dump、通用张量比较及同输入原生回放](OM_OPERATOR_DIAGNOSTICS.md)。
它在原失败 OM 上通过 CANN 获取中间数据，不要求更换图；支持其他算子，不仅是 GDR。
原有 `--diagnose-target-parity` 本身仍只是两个 OM 的最终状态/compact token 回放，
仍报告 `raw_logits_available=false`。需要显式开 dump、转换和映射后再做原生回放。
完整模型的三入口自动逐节点对齐尚未实现，融合/布局/调用配对不得猜测。

冻结事务 1、2 的完整物理 16 行输入（包括有效区外的行）、入口状态、cursor、
selector、权重和算子构建身份。对照三个入口：当前分支的 torch_npu Verify、
导出 wrapper 在 torch_npu 中的 eager 执行、该 wrapper 编译的 OM。

| 首次不一致的边界 | 排查范围 |
| --- | --- |
| 原 torch_npu Verify 与导出 wrapper eager | 模型包装、状态选择、padding/mask、dtype/layout 适配 |
| wrapper eager 与 OM | converter、编译精度/融合、后端算子实现及注册版本 |
| GDR 前 Q/K/V/g/beta 或选定状态已经不同 | 投影、conv、门控、状态输入等上游；不能先归责 GDR |
| GDR 全部输入及属性相同，但 core_attn/state_bank 不同 | GDR 执行/编译区域；再区分版本、布局处理和内核数值 |
| GDR 输出相同，而提交 scalar state 不同 | bank 槽选择、cast、commit 路径 |

先观察事务 1 的首个线性注意力层：现有报告已显示其 recurrent state 分歧，
无需等到事务 2 的第 9 个生成 token 才抓取。conv history 相等不等于 conv 输出或
Q/K/V/g/beta 相等。每个比较需记录实际 decoder layer ID（而非只给分组轴）、
token 行、首差坐标、dtype/shape/stride、非有限值数量和误差统计；门禁仍保留
ordinary 零 token-ID mismatch，不以临时浮点容差掩盖失败。

将抓到的同一组 GDR 输入分别送入原生 MTP 与单算子 OM，比较全部输出行及每个
bank 槽；再与逐步原生 GDR 对照。只有实际输入、属性和所用实现身份已锁定，
才能把“torch_npu 能运行”和“OM 数值正确”变成可检验的结论。
独立子图会改变编译上下文，若独立子图通过而完整图失败，仍须回到原图检查
融合、输入生命周期和实际中间值，不能据此宣称原图正确。

可先扩展 C++ 为同入口的 Verify K=0 逐步回放，辅助区分行零路径与块内后续行。
它复用现有 Verify16 OM，只需重编诊断 C++，但物理形状仍为 16，不能称为 T=1
算子对照。真正暴露中间张量的诊断子图需要另行导出 AIR/OM，保留原 bundle；
普通 C++ 日志无法读取 OM 未暴露的内部 tensor；v47 的 CANN runtime dump 是另一条
不改 AIR/OM 的采集通路，dump 支持范围需以部署的 CANN/自定义算子版本验证。

## 两条路径

| factory 配置 | 行为 | 用途 |
| --- | --- | --- |
| `target_verify_gdr_policy="mtp-block-v1"`，也是省略时的默认值 | 原 GDR-MTP，每层一次多 token 调用 | 保留当前路线及失败复现 |
| `target_verify_gdr_policy="decode1-recurrence-v1"` | 每个物理 token 一次原生单步 GDR，每步 FP16→FP32 状态反馈 | 可选数值对照，非 MTP 内核修复 |

对照复用 Decode1 的 GDR 调用参数：物理 T=1、`chunk_size=1`、
`effective_length=INT16[B]` 全 1、开启 final state 和 Q/K 归一化。
输入是本次已选定的 FP32 scalar state，不在第一个 token 前额外舍入。
每个 bank 槽 i 保存消费第 i 行之后、经过 FP16 舍入的 FP32 状态；接受/回滚仍选
`accepted_count` 槽、推进 `accepted_count+1` 个输入。

只改变 GDR 子路径。QKV 投影、conv、完整注意力、Draft、量化文件、权重、EOS、
proposal 预算、接受率分母、四个 OM 及其张量 ABI、C++ 调度和驻留方式不变。
ordinary 和当前分支 eager torch_npu 的默认行为也不改变。
因此即使对照通过，也不意味着已证明两种 DFlash 的 proposal/接受率一致。

代价：Qwen3.5 的 24 个线性注意力层、T=16 时，共 384 次原生单步 GDR，
取代 24 次 MTP。不重算历史前缀，也不逐 token 重跑整个 Target，但可能增加
算子调度、编译及运行开销。没有真机性能或显存改善承诺。

## 如何使用

只有要运行单步对照时，在活动 run 下新建 factory JSON，保留现有所有资源路径，
设置：

```json
{
  "merged_prefill": true,
  "draft_static_feature_rows": 64,
  "target_verify_gdr_policy": "decode1-recurrence-v1"
}
```

这是需要合并到完整 factory 配置的三个字段，不是完整配置。
动态 TargetStep 不支持该对照并会明确拒绝，不能静默固定某个动态 gear。

更新实际加载的 receiver modeling 和 Python 导出代码。导出会检查 receiver 支持
此 policy，并验证 SOURCE_LOCK；不能只伪造 capability 标记或跳过哈希验证。
若 receiver 另有源码副本，需同步实际文件并按现有流程重新锁定其输入身份。

按 [静态 split 指南](STATIC_SPLIT_OM_DEFAULT.md#从哪里重新运行) 的
`export-air`、`compile-om` 步骤生成**全新 bundle**，保留原 MTP bundle。
对照改变图内算子，必须生成新 AIR/OM；runner 1.28.0 兼容，无需仅为此重编 C++，
也无需重新量化。默认 MTP bundle 不需要为了新增对照而重新编译。

在新 AIR manifest 上使用 Python 核验，无需 jq：

```bash
"$MODEL_PYTHON" - "$REFERENCE_BUNDLE/air-manifest.json" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)
verify = next(g for g in report["graphs"] if g["role"] == "target-verify-commit")
metadata = verify["metadata"]
assert not verify["dynamic"]
assert metadata["target_verify_gdr_policy"] == "decode1-recurrence-v1"
print(metadata["target_verify_gdr_policy"])
print(metadata["target_verify_gdr_state_rounding"])
print(json.dumps(metadata["custom_op_export_contracts"], indent=2))
PY
```

`REFERENCE_BUNDLE` 指向本次新生成的 bundle。导出门禁要求实际保留至少
`24*16=384` 个 `ChunkGatedDeltaRule` 节点，不能只把 metadata 标成新 policy。

随后沿用原失败 prompt/chat、32-token 预算、K=15、EOS=248044 和原 runner 配置，
先运行 `infer-cpp --diagnose-target-parity --diagnostic-max-transactions 2`，
把 `--deployment-manifest` 指向新 bundle，报告写到新路径。
对照结果仍然不能替代完整 ordinary/DFlash 零 token-ID mismatch 门禁。
不要改 ordinary 参考为 Verify，也不要关闭失败检查。

## 证据边界

`tests/test_verify_gdr_parity.py` 使用真实模型侧 helper/GDN forward，配合明确标识的
CPU 单步 GDR fixture，覆盖 T=1/2/6/16、batch=1/2、非零且不在 FP16 网格的初始状态、
标量/状态库入口、不同接受槽、拒绝后续跑、逐步输出与每个状态槽，以及 strict
torch.export 的固定形状和状态 cast。反例还验证“只在最终 bank 上 cast”并不等价。

这些测试验证调用和反馈规则，不验证真实自定义算子或 ATC。当前本地 profile
为 simulation-only，没有可用 ATC/310P。原真机失败尚未修复或闭合。
下一步应采集并进行同输入原生/OM 对照；只有进一步确认内核范围后才进入 MTP
算子实现修复。单步对照保留为可选证据，不是默认替代方案。
