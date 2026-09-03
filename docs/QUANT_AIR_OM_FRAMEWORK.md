# 基于 `quant` 分支的 AIR → OM → C++ token 推理框架

## 1. 目标和边界

本框架直接建立在远端 `quant` 分支提交
`28f93e784a2beed87020a80bd93c8788754eab1c` 上。它不会把另一套模型冒充成量化模型：

- Target 由 `models.internal_dflash_bridge.load_qwen35_target` 加载；
- Target Linear 使用 `models.dflash_v1.original_quant.quant_model` 转成原有 W8A8
  `QLinear`；
- Target embedding 使用量化 YAML 指向的 INT8 weight 与 FP32 row scale；
- Target 输出边界、Draft、共享 embedding 和 LM head 保持 FP16；
- Draft 使用本分支现有 `models.dflash_v1.modeling_dflash.DFlashDraftModel`；
- greedy 接受规则保持 ordinary Target 权威，token ID、EOS 和 stop reason 必须零差异。

交付链路如下：

```text
Qwen3.5-4B Target checkpoint
W8A8 Linear / embedding artifacts
Qwen3.5-4B-DFlash checkpoint
quant 分支源码
        │
        │ TorchAir dynamo_export
        ▼
AIR graph + 外置权重文件 + air-manifest.json
        │
        │ atc --mode=0 --framework=1 --soc_version=<精确型号>
        ▼
target-prefill.om + target-prefill-head.om + target-decode1.om
                 + fused-speculative-step.om + deployment-manifest.json
        │
        │ C++17 / AscendCL
        ▼
四个 OM 各加载一次 → device-resident 状态机 → 生成 token → EOS/长度停止
```

当前 C++ 主部署路线是四物理 OM 的精确融合投机拓扑：prefill body/head 分离、ordinary decode
独立，Draft proposal 与固定 `T=16` Target verify 合并为一个物理 OM。Target/Draft KV、GDR、
conv state 和 compact carrier 都是显式、常驻 device 的状态。静态完整前缀重算图仍保留，但只用作
最小 correctness/单图诊断基线，不再是 `run-e2e-cpp` 的默认产物。因此：

- “是否真的调用 OM 生成完整 token 序列”可以验证；
- “是否达到闭源增量推理框架时延”必须在真实设备测量，当前不能预先声称；
- fused 4-OM 必须先通过 ordinary greedy 的 token/EOS 零差异、完整常驻集合显存以及未开启
  profiling 的同机 3+10 时延门禁，才能从候选提升为正式性能结论。

这里必须区分“当前分支实际产物”和“最终低时延拓扑”：

| 阶段 | 逻辑图 | 物理 OM 数 | 当前状态 |
| --- | --- | ---: | --- |
| C++ 主部署候选 | prefill body/head、ordinary decode、Draft+固定 T16 verify supergraph | 4 个 | 默认生成；代码/Fake-ACL 已验证，真机门禁待执行 |
| 可选对照候选 | prefill body/head、decode、draft、verify 分离 | 5 个 | 显式 factory 可生成，用于定位融合边界收益 |
| 单图诊断基线 | Target 全前缀 + Draft proposal 整图 | 1 个 | 显式 factory 可生成，不用于最终低时延结论 |

`verify` 不能与 ordinary `decode` 共用一个含糊的状态合同：前者一次计算最多 16 个 provisional
row，并只按接受数提交一个 GDN/conv state-bank 槽；后者固定提交单个 row。用户口头所说的
“prefill、decode、draft 三类”在逻辑上因此落为上述四个角色。为满足“非末 prompt chunk
不执行完整 LM head”的热路径约束，`target-prefill` body 输出 device-resident `last_hidden`，只在
最后一个 chunk 调用 `target-prefill-head`；再把 Draft 与固定 T16 verify 合并后，主路线最终是
四个物理 OM。

## 2. 冻结 ABI

本节首先记录仍保留的单图重算诊断 ABI。fused 4-OM 的每个 role、状态 tensor、动态 gear 和
commit/rollback 合同见 [增量 OM 与 C++ 高性能路线](INCREMENTAL_OM_PERFORMANCE.md)；主流程
生成与运行命令从第 4 节开始全部使用该 4-OM ABI。

图名为 `quant_dflash_recompute`，batch 固定为 1：

| 顺序 | 名称 | dtype | shape | 含义 |
| --- | --- | --- | --- | --- |
| input 0 | `input_ids` | INT64 | `[1,S]` | 已提交前缀，右侧补 pad |
| input 1 | `attention_mask` | INT64 | `[1,S]` | 有效前缀为 1，右侧 padding 为 0 |
| output 0 | `target_top1` | INT64 | `[1,S]` | 每个物理 Target row 的 greedy Top1 |
| output 1 | `draft_top1` | INT64 | `[1,15]` | anchor 后最多 15 个 proposal |

`S` 是编译期静态 gear，必须是 64 的倍数。C++ runner 每次只读取
`target_top1[prefix_length-1]`，Draft 只读取 attention mask 声明的有效前缀。pad 位位于未来，
因果 Target 的有效 row 不依赖这些 pad row。

原 GDR 的新 ABI 还要求一个 `[B] INT16 effective_length`。它不是新增的 OM 外部输入：AIR 图
在内部统计 `attention_mask` 的非零行数并传给普通 GDR。比如 `S=64`、有效前缀为 37 时，
GDR 的物理序列仍为 64 行，但 `effective_length=[37]`；MTP verify 算子 ABI 不受这次改动影响。
由于该字段是 INT16，`S` 不能超过 32767。

### 2.1 全部自定义算子的 Fake/Meta 与 AIR 保留合同

两个 Target modeling 文件里的 `torch_npu.*` 调用保持原样，没有加入 export-only Tensor 公式
分支。导出适配发生在模型加载之后、`dynamo_export` 之前：

```text
modeling 中的 torch_npu.<op> 或 AIR 专用 qwen35_dflash::<op>
        │ 校验源 npu::<op> dispatcher schema/alias
        │ 校验已有 Meta，缺失时注册精确 Fake（只声明 shape/dtype/alias）
        ▼
源 npu.<op>.default 或无冲突的 qwen35_dflash.<op>.default
        │ receiver-private converter 或已校验的 TorchAir builtin converter
        ▼
对应 GE/AIR 节点（dynamo.pbtxt 必须可审计）
```

当前锁定的七个算子如下。`required` 表示当前完整前缀重算图必须实际出现，`optional` 表示仍做
schema/Meta 预检，但不能虚构一次图命中。

| 前端 FX target | Fake/Meta 输出合同 | 默认 GE type | converter | 当前图 |
| --- | --- | --- | --- | --- |
| `npu.npu_dynamic_quant.default` | INT8 输出与 input 同 shape；FP32 scale 为 `input.shape[:-1]` | `DynamicQuant` | TorchAir builtin | required |
| `qwen35_dflash.npu_quant_matmul_v4444.default` | broadcast batch + `[M,N]`，当前调用输出 FP16 | `QuantBatchMatmulV4444` | 框架注册 | required |
| `npu.adn_rms_norm.default` | 输出 0 与 input 同 shape/dtype；输出 1 为 `[*input.shape[:-1],1]` FP32 | `AdnRmsNorm` | 框架注册 | required |
| `npu.npu_chunk_gated_delta_rule.default` | output 为 value shape/query dtype；final state 为 initial-state shape/FP32 | `ChunkGatedDeltaRule` | 框架注册 | required |
| `qwen35_dflash.npu_cache_update.default` | 返回同 shape/dtype/device 的非 alias 更新值 | `CacheUpdate` | 框架注册 | required |
| `npu.adn_fused_infer_attention.default` | 按 layout 推导；当前 packed `BNSD` 路径保持 query shape/FP16 | `AdnFusedInferAttention` | 框架映射 | required |
| `npu.npu_scatter_nd_update_.default` | 返回同一个 `Tensor(a!)`，不能丢失写 alias | `ScatterNdUpdate` | TorchAir builtin | optional（仅 `forward1`） |

这里的 `AdnRmsNorm` 不是通用 CANN `RmsNorm`。receiver 原导出器明确调用
`custom_op("AdnRmsNorm", inputs={"self": ..., "gamma": ...},
outputs=["y", "rstd"], attrs={"epsilon": ...})`，所以 factory、converter 和图审计必须使用同一
名称及 named ABI，不能以一个自洽但错误的 `RmsNorm` 测试替代 receiver 合同。其余名称分别由
receiver converter（`ChunkGatedDeltaRule`、`CacheUpdate`、`AdnFusedInferAttention`）、310P
receiver 量化包（`QuantBatchMatmulV4444`）和 TorchAir builtin converter（`DynamicQuant`、
`ScatterNdUpdate`）核对。增量 verify 用到的第八个前端不属于重算图七项，其 GE type 单独锁定为
`GatedDeltaRuleMTP`。

这些 `*_ge_op_type` 字段是显式可审计的锁值，不是自由重命名入口；任一值偏离上表都会在加载
4B checkpoint 前失败。需要支持另一套 receiver 时，应连同其 GE prototype、converter named
ABI、图节点审计和真机 ATC 证据一起版本化，不能只改 JSON 字符串。

GDR 的模型合同是 Q/K/V `[B,S,32,128]` FP16、g `[B,S,32]` FP32、beta
`[B,S,32]` FP16、`effective_length [B]` INT16、initial/final state
`[B,32,128,128]` FP32、输出 `[B,S,32,128]` FP16；当前导出必须设置
`output_final_state=True`。PyTorch 前端参数顺序是 `effective_length, chunk_size,
initial_state, ...`，但当前 GE v2 原型的输入顺序是 `initial_state,
effective_length`，三个标量是 ATTR。框架因此必须使用 named inputs/outputs/attrs：

```text
inputs: query, key, value, g, beta, initial_state, effective_length
outputs: core_attn, last_recurrent_state
attrs: chunk_size, output_final_state, use_qk_l2norm_in_kernel
```

不能把 PyTorch schema 的十个参数按 positional 顺序直接传给 GE，否则
`effective_length` 会落到 `initial_state` 位置，`chunk_size` 会落到 Tensor 输入位置。
receiver 的 310P attention 包注册的是 `AdnFusedInferAttention`，不是 CANN A2 路径的
`FusedInferAttentionScore`。它使用 dynamic `key/value`、15 个 optional inputs、单一
`attention_out` 输出以及 `scale_value` attr。converter 按该原型绑定 named inputs，并将前端
三个 `SymInt[]` 长度参数分别构造成 INT64 GE Tensor：`all_seq_lengths_q`、
`actual_seq_lengths_q`、`actual_seq_lengths_kv`；不能把三者折叠为一个输入。当前前端没有
`actual_seq_lengths_q_back`，所以该 optional input 保持 `None`。两个 HIAI modeling 文件在
eager 和 AIR 路径都把 `allQLen` 传给 `all_seq_lengths_q`；当前路线没有 PSE tensor，因此
`pse_shift` 也保持 `None`。

Fake/Meta 不执行任何算子数值，正式 eager、AIR 和 OM 也不会执行 Fake。若 torch-npu 已有 Meta，
框架先运行 shape/dtype/alias 探针并复用；只有缺失时才调用 `torch.library.register_fake`。任何 schema
参数、kw-only 标志、返回个数或可变 alias 漂移都会提前失败。

W8A8 matmul 不再把 `npu.npu_quant_matmul.default` 直接交给项目 converter。receiver
自带的 TorchAir 已经为这个同名 target 注册了 `QuantBatchMatmulV3` converter，重复注册的
项目 converter 不一定能替换 registry 实际使用的旧项。AIR 导出因此改用项目私有、无冲突的
`qwen35_dflash.npu_quant_matmul_v4444.default` 前端。普通 `torch_npu` eager 与 AIR 不再根据
“私有 op 是否已经注册”隐式选路：每个 `QLinear` 默认固定走 eager，AIR factory 加载量化 Target
后才显式打开 export mode，并核对打开的 QLinear 数量与量化审计中的 `qlinear_count` 完全一致；
该 mode 还必须同时处于 active Dynamo capture 才能选中私有前端，因此一次导出后同一对象再做
普通 eager 调用也不会误走项目私有 export frontend。

量化文件中的权重 scale 继续保留为 FP32。普通 eager 严格复用 `quant` 分支已经验证过的调用
合同：`INT8 activation + INT8 weight + FP32 weight scale + FP32 per-token scale`。这里不能先调用
`npu_trans_quant_param`；该接口在 eager 返回 INT64 carrier，而 receiver 的 V4444 注册表没有
`INT8 + INT8 + INT64` 这一列，预编码会在 `GetWorkspace` 阶段形成找不到 binary 的 integral
key。AIR 则只在 active Dynamo capture 中使用项目私有 frontend，仍消费同一份 FP32 scale。
因此直接 `torch_npu` 推理和 AIR/OM 导出保持相同量化参数语义，同时不会因私有 op 注册状态或
残留 factory flag 串路。310P3 receiver 安装的是
`QuantBatchMatmulV4444`，其 GE 输入顺序
`x1,x2,scale,offset,bias,pertoken_scale` 与 PyTorch 前端的
`x1,x2,scale,offset,pertoken_scale,bias` 不同，因此框架按名称绑定六个输入，并显式写入：

```text
outputs: y
attrs: dtype=DT_FLOAT16(1), transpose_x1=false, transpose_x2=false, group_size=0
```

当前模型调用合同锁定 `output_dtype=torch.float16`、`group_sizes=None`；其他组合会在 AIR
导出时直接失败，不能静默生成一个可能选错 kernel 的 OM。已有运行目录中的
`factory-fused.json` 也必须把 `npu_quant_matmul_ge_op_type` 从
`QuantBatchMatmulV3` 改为 `QuantBatchMatmulV4444`。

`npu_quant_matmul` 的预检使用实际 W8A8 路径代表形状：`x1=[1,64,2560]`、
`x2=[2560,8192]`、`scale=[8192]`、`pertoken_scale=[64]`。torch-npu 要求
`pertoken_scale.shape[0] == x1.shape[-2]`（M 维）；不能把 batch 与 M 相乘后写成 `[B*M]`。

receiver 的原始 `npu.npu_cache_update_` 使用 `Tensor(a!) -> Tensor(a!)` 原地 ABI。模型公开
forward 现在不再逐层携带 `export_flag`；一个内部 `_npu_cache_update` helper 在普通 eager 中严格调用
原始算子，在 active Dynamo capture 中自动选择无 alias 前端
`qwen35_dflash.npu_cache_update.default`（“旧 cache 输入 -> 新 cache 输出”），再精确 lowering 为一个
`CacheUpdate` GE 节点。这样 eager 不增加 clone 或额外 NPU launch，AIR/AOT 图也不产生整个 KV
cache 的 `aten.copy`。rollback 多行写显式串接每一行的更新输出。旧 export driver 的
Functionalize+`copy_` 方案虽然能绕过 alias 报错，但每个 CacheUpdate 都会在 AOT 图留下一个大
Tensor copy；本框架因此只借鉴它的无 alias frontend，不复制这部分实现。仅设置
`keep_inference_input_mutations=True` 也不能解除非 ATen 算子“返回值带 alias”的限制。

`CacheUpdate` 的 GE 原型不是按前端 Python 参数名解析，而是固定为：

```text
inputs: x, updates, targetBlock, offsetInBlock
outputs: x
attrs: none
```

因此 converter 使用 named inputs/output，并显式完成 `input -> x`、`target_block -> targetBlock`、
`offset_in_block -> offsetInBlock` 映射；不能再把四个前端参数交给 positional 自动解析。

预期 GE type 都在 `factory-fused.json` 中显式锁定。builtin converter 和当前 fused-attention 精确
映射必须使用表中的 type；GDR 当前只允许上述 `ChunkGatedDeltaRule` v2 named ABI。该值不会
自动回退：指定 IR 未注册、converter 没有命中，或 `dynamo.pbtxt` 中 required type 数为 0，
导出都会失败，不会生成伪 PASS manifest。

### 2.2 为什么原模型看起来没有导出适配也能生成 AIR

receiver 的 `models/export_model_wrapper_qwen3_5.py` 只负责逐层执行并展平输出 state，因此单看
wrapper 确实看不到导出适配。但与它配套的 `export_glm_model_qwen3_5.py` 在调用
`torchair.dynamo_export` 前已经注册了 RMSNorm、attention、CacheUpdate、GDR 的 Meta 和 GE
converter，并为原地 CacheUpdate 定义了 functional frontend 与 Functionalize 实现。也就是说，
原路线同样依赖导出适配，只是适配位于 export driver，而不是 modeling 文件。

本框架沿用“导出准备统一注册 Fake/converter”的分层方式并补强 schema、alias、GE prototype 与
图节点审计；CacheUpdate 由模型内部 helper 在捕获期间无参数地自动选路，以避免旧 Functionalize
方案生成的大 Tensor copy。不能直接复制旧脚本的全部定义：旧 GDR schema 没有 quant 分支新增的
`effective_length`，旧 attention converter 通过 `pse_shift` 临时转运 `allQLen`，且旧脚本没有解决
当前环境把 QuantMatmul 降为 V3 的冲突。
配套 `inference.py` 只是 eager runner（`compiled_model = model_wrapper`），也不能用来证明 AIR
无需 Fake、Functionalize 或 converter。

### 2.3 TorchAir 标准算子精确补丁

目标环境的 TorchAir 虽然注册了 `torch.ops.aten.softplus.default`，对应 converter 却会主动抛出
`NotImplementedError`。模型的 Gated DeltaNet 在公共 `forward()` 中通过 `F.softplus` 计算
FP32 gate，因此框架在 `dynamo_export` 前覆盖该 converter，并精确生成一个 GE
`SoftplusV2` 节点。`beta` 和 `threshold` 必须是有限的编译期数值，并原样写入 GE Float 属性；
当前模型使用 PyTorch 默认值 `1.0` 和 `20.0`。

这不是近似替换，也不会改两个 modeling 文件的 eager 数学。不要把它展开为
`maximum + log1p + exp + abs`：展开会增加图节点，并且不再直接保留 PyTorch Softplus 的
threshold 分支。导出后可检查：

```bash
rg -n 'type: "SoftplusV2"' \
  "$AI_RUN_DIR/artifacts/quant-dflash/air/quant_dflash_recompute/dynamo.pbtxt"
```

至少应命中一次；`air-manifest.json` 的 `standard_op_overrides` 同时记录本图 converter 调用数。
七个 NPU 自定义算子的 Fake、converter 和保留门禁仍由上一节独立执行。

如果 `S=64`，则 `prompt_tokens + generated_tokens` 不能超过 64。需要更长上下文时重新编译
`S=128/256/...`。当前重算路径的延迟随 `S` 增大，因此先使用能够覆盖验证 workload 的最小
gear。

完整合同见 `framework/FRAMEWORK_LOCK.json`。

## 3. 真实环境要求

正式验证必须满足：

1. 一块可访问的真实 Ascend 310P 设备；
2. 匹配驱动、固件的 CANN、Torch NPU、TorchAir 和 ATC；
3. AscendCL 头文件与 `libascendcl.so`；
4. `npu-smi info` 可用，且存在 `/dev/davinciN`；
5. Python 3.10、PyTorch、`torch_npu`、`torchair`、Transformers 5.14.1、PyYAML、
   safetensors；
6. 外置 Qwen3.5-4B、官方 Qwen3.5-4B-DFlash checkpoint；
7. 原量化 YAML 及其 `quanted_pth`、`embedding_weight_path`、
   `embedding_scale_path`；
8. receiver 的 `models/export_model_wrapper_qwen3_5.py`。
9. `ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 中只有一套
   `ChunkGatedDeltaRule`，且其 `op_proto.h` 包含 `effective_length` v2 输入。
10. `ASCEND_CUSTOM_OPP_PATH` 中包含一套完整 ADN vendor，其 prototype 注册
    `AdnFusedInferAttention`，并带有 Ascend310P `.o/.json` 预编译 kernel。
11. `ASCEND_CUSTOM_OPP_PATH` 中包含完整 `QuantBatchMatmulV4444` vendor，且它的
    `op_api/lib/libcust_opapi.so` 在 `LD_LIBRARY_PATH` 中优先于冲突实现；其 ops-info 必须包含
    `INT8 + INT8 + FP32 scale -> FP16`。修改环境变量后必须启动新的 Python 进程。

禁止用 CPU fallback 代替设备结论。仓库不保存 checkpoint、量化权重、AIR、OM、编译缓存、
日志或性能报告。

先确认代码分支确实基于 `quant`：

```bash
git switch framework/quant-air-om
git merge-base --is-ancestor \
  28f93e784a2beed87020a80bd93c8788754eab1c HEAD
```

第二条命令退出码必须为 0。

设置源码和外置运行目录。下面所有路径都必须替换成真实绝对路径：

```bash
export REPO_ROOT=/ABSOLUTE/PATH/qwen3.5-4B-dflash
export MODEL_PYTHON=/ABSOLUTE/PATH/python3
export AI_RUN_DIR=/ABSOLUTE/PATH/quant-air-om-run
export PYTHONPATH="$REPO_ROOT/framework/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export ADN_OPP=/ABSOLUTE/PATH/adn_fa_and_norm/packages/vendors/customize
export QMM_OPP=/ABSOLUTE/PATH/vendors/customize_quantMatmul
export QMM_OPAPI="$QMM_OPP/op_api/lib"
export LD_LIBRARY_PATH="$QMM_OPAPI${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export ASCEND_CUSTOM_OPP_PATH="$QMM_OPP:$ADN_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
mkdir -p "$AI_RUN_DIR"
cd "$REPO_ROOT"
```

确认软件栈：

```bash
"$MODEL_PYTHON" - <<'PY'
import torch
import torch_npu
import torchair
print("torch", torch.__version__)
print("torch_npu", getattr(torch_npu, "__version__", "unknown"))
print("torchair", getattr(torchair, "__version__", "unknown"))
print("npu_available", torch.npu.is_available())
print("npu_quant_matmul.schema", torch.ops.npu.npu_quant_matmul.default._schema)
PY

npu-smi info
atc --version
```

任何 import、设备或 ATC 失败都应先修环境；不要让导出流程自动降级到 CPU。

在加载权重前单独检查 GDR 和 ADN GE 原型：

```bash
"$MODEL_PYTHON" - <<'PY'
from qwen35_dflash.ascend310p.custom_op_export import (
    validate_adn_attention_ge_prototype_environment,
    validate_gdr_ge_prototype_environment,
)

gdr = validate_gdr_ge_prototype_environment()
adn = validate_adn_attention_ge_prototype_environment()
print("GDR", gdr)
print("ADN", adn)
assert gdr["status"] == "PASS"
assert gdr["abi"] == "effective-length-v2-named-inputs"
assert adn["status"] == "PASS"
assert adn["ge_op_type"] == "AdnFusedInferAttention"
assert adn["abi"] == "receiver-adn-attention-v1-named-inputs"
PY
```

导出器也会自动执行同一检查。若两个环境变量指向两套同名
`ChunkGatedDeltaRule`，会在加载 checkpoint 前失败并列出两个 `op_proto.h`。必须同时从
`ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 移除旧 vendor 根；不要只依赖路径先后顺序，
也不要删除不属于当前用户的安装目录。
ADN 检查还会确认 prototype 的完整输入/输出/attr 顺序以及
`op_impl/ai_core/tbe/kernel/ascend310p/adn_fused_infer_attention` 下至少一对
`AdnFusedInferAttention_*.o/.json`。若当前 `ASCEND_CUSTOM_OPP_PATH` 只有 GDR、QuantMatmul
等 vendor 而没有 ADN，导出器会在加载 4B 权重前给出明确错误。

## 4. 锁定外部模型和量化输入

先生成一次 hash-complete manifest。它会读取并哈希 Target、Draft、量化 Linear、量化
embedding、量化 YAML 和 receiver wrapper。大权重首次哈希需要时间，这是防止生成 AIR 前后
权重漂移的必要步骤。

```bash
"$MODEL_PYTHON" framework/scripts/lock_quant_inputs.py \
  --target-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --draft-dir /ABSOLUTE/PATH/Qwen3.5-4B-DFlash \
  --quant-config /ABSOLUTE/PATH/qwen3.5.yaml \
  --receiver-models-dir /ABSOLUTE/PATH/qwen35-runtime/models \
  --output "$AI_RUN_DIR/quant-input-manifest.json"
```

输入目录不能是 symlink，也不能在 manifest 生成后增加、删除或修改文件。AIR factory 会在加载
4B 权重前重新核验 manifest，并逐项核对仓库 `SOURCE_LOCK.json` 中的量化、Target、Draft、
bridge 和 rollback 源码 hash；任意字节变化都会失败。

复制并编辑 factory 配置：

```bash
cp config/quant_air_om_fused_factory.example.json \
  "$AI_RUN_DIR/factory-fused.json"
```

`factory-fused.json` 示例：

```json
{
  "target_dir": "/ABSOLUTE/PATH/Qwen3.5-4B",
  "draft_dir": "/ABSOLUTE/PATH/Qwen3.5-4B-DFlash",
  "quant_config": "/ABSOLUTE/PATH/qwen3.5.yaml",
  "input_manifest": "/ABSOLUTE/PATH/quant-air-om-run/quant-input-manifest.json",
  "receiver_models_dir": "/ABSOLUTE/PATH/qwen35-runtime/models",
  "max_sequence_length": 2048,
  "example_sequence_length": 64,
  "eos_table_width": 4,
  "pad_token_id": 0,
  "dtype": "float16",
  "device": "npu:0",
  "npu_dynamic_quant_ge_op_type": "DynamicQuant",
  "npu_quant_matmul_ge_op_type": "QuantBatchMatmulV4444",
  "adn_rms_norm_ge_op_type": "AdnRmsNorm",
  "npu_chunk_gated_delta_rule_ge_op_type": "ChunkGatedDeltaRule",
  "npu_gated_delta_rule_mtp_ge_op_type": "GatedDeltaRuleMTP",
  "npu_cache_update_ge_op_type": "CacheUpdate",
  "adn_fused_infer_attention_ge_op_type": "AdnFusedInferAttention",
  "npu_scatter_nd_update_ge_op_type": "ScatterNdUpdate"
}
```

`max_sequence_length` 可按实际容量调整，但必须为 64 的正整数倍且不超过 GDR INT16
`effective_length` 上限。`eos_table_width` 必须能容纳 tokenizer 的全部 EOS ID。若只需要复现单图
诊断基线，才复制 `config/quant_air_om_factory.example.json` 并显式选择
`create_quant_recompute_graph`。

`receiver_models_dir` 用于补齐 `models.export_model_wrapper_qwen3_5`。量化分支本身仍提供
HIAI modeling、QLinear、DFlash 和 bridge；receiver wrapper 只复用原工程的加载和设备初始化。

## 5. 分层验证

### 5.1 Host 回归

这一步验证 Python 合同、量化 factory、manifest、ATC 命令生成、C++ scheduler 和 fake ACL。
它不是设备结论。

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$MODEL_PYTHON" -m pytest -q

# 只重跑 AIR/Fake/Meta/manifest/ATC/C++ 合同测试
PYTHONDONTWRITEBYTECODE=1 \
  "$MODEL_PYTHON" -m pytest -q tests/test_quant_air_om_framework.py

cmake -S framework/runtime/cpp \
  -B "$AI_RUN_DIR/build/cpp-host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQWEN35_DFLASH_BUILD_ACL_RUNNER=OFF \
  -DQWEN35_DFLASH_BUILD_TESTS=ON
cmake --build "$AI_RUN_DIR/build/cpp-host" --parallel
ctest --test-dir "$AI_RUN_DIR/build/cpp-host" --output-on-failure
```

通过标准：所有 Python 测试和两个 CTest 都通过。框架专项测试会检查源算子的 schema/Fake/alias，
并以 strict `torch.export` 和 AOTAutograd 确认 CacheUpdate 连续写直接捕获为相同数量的
`qwen35_dflash.npu_cache_update.default`，图中不存在源 alias op 或 `aten.copy`；普通 eager 仍必须
保持原地更新。其余前端 target 也必须保留。这仍然只验证 FakeTensor/图捕获元数据，不执行 NPU
算子数值。fake ACL 测试会编译生产 `acl_executor.cpp`，但只证明 host 侧 buffer、调用顺序、
scheduler 和 JSON 门禁。

### 5.2 量化 PyTorch 图探针

在导出前用真实权重执行一次图：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p probe-pytorch \
  --factory-config "$AI_RUN_DIR/factory-fused.json" \
  --input-token-ids 1,2,3,4 \
  --output "$AI_RUN_DIR/reports/pytorch-probe.json"
```

必须检查：

- `status == PASS`；
- `cpu_fallback == false`；
- `graph_metadata.target_quant_mode == "w8a8_dynamic"`；
- `graph_metadata.target_quantization_audit.status` 是量化 assembly PASS；
- `target_top1` shape 为 `[1,S]`；
- `draft_top1` shape 为 `[1,15]`；
- 输出 token ID 非负且小于词表大小。

`probe-pytorch` 是集成重算语义探针，即使传入 fused 配置也只执行诊断图，不决定后续会生成几个
OM。它仍不是 OM 证据，但能在耗时导出之前发现 checkpoint、量化 topology、embedding、Draft
或 receiver ABI 错误。

### 5.3 从拷贝源码目录采集完整 AIR 诊断

NPU 机器上即使只有直接拷贝的源码、没有 `.git`，也可以执行：

```bash
"$MODEL_PYTHON" framework/scripts/collect_air_debug.py \
  --factory \
    qwen35_dflash.ascend310p.quant_factory:create_quant_fused_speculative_step_graphs \
  --factory-config "$AI_RUN_DIR/factory-fused.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts/air-debug-fused" \
  --output-dir "$AI_RUN_DIR/reports"
```

采集器不调用 Git，也不复制 checkpoint、量化权重、AIR 或 OM。它会记录关键源码 SHA256 并附带
这些小型源码文件的快照，同时收集 Python/package/CANN/NPU 身份、七个前端算子的真实 schema
与 dispatch table、GDR/ADN prototype 及 kernel 预检结果 `ge-prototypes.json`、脱敏后的 factory
配置、完整 Dynamo 导出日志和已有的 JSON/log/pbtxt 诊断。
即使 AIR 导出失败，也会先在 `--output-dir` 生成 `air-debug-*.tar.gz`，随后返回原导出退出码；
把该压缩包回传即可。若日志过大，可增加 `--no-dynamo-logs`，但第一次失败建议保留默认完整日志。

## 6. 生成 AIR

```bash
export FUSED_BUNDLE="$AI_RUN_DIR/artifacts/quant-dflash-fused"

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p export-air \
  --factory \
    qwen35_dflash.ascend310p.quant_factory:create_quant_fused_speculative_step_graphs \
  --factory-config "$AI_RUN_DIR/factory-fused.json" \
  --bundle-dir "$FUSED_BUNDLE"
```

预期生成：

```text
$FUSED_BUNDLE/
├── air-manifest.json
└── air/
    ├── target-prefill/target-prefill.air
    ├── target-prefill-head/target-prefill-head.air
    ├── target-decode1/target-decode1.air
    └── fused-speculative-step/fused-speculative-step.air
```

每个目录还会包含 `dynamo.pbtxt`、外置权重和辅助文件。导出器要求每个 role 恰好一个 `.air`，
并把所有 payload 的大小和 SHA-256 写入 `air-manifest.json`。下面的门禁会同时检查：恰好四图、
动态 Draft 输入 gear、显式 `effective_length`、融合边界和所有声明的自定义算子节点：

```bash
"$MODEL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["FUSED_BUNDLE"])
data = json.loads((root / "air-manifest.json").read_text())
assert data["status"] == "PASS"
assert data["schema_version"] == 3
gdr_proto = data["environment"]["gdr_ge_prototype"]
assert gdr_proto["status"] == "PASS"
assert gdr_proto["abi"] == "effective-length-v2-named-inputs"
expected_names = [
    "target-prefill",
    "target-prefill-head",
    "target-decode1",
    "fused-speculative-step",
]
assert [graph["name"] for graph in data["graphs"]] == expected_names
graphs = {graph["name"]: graph for graph in data["graphs"]}
assert graphs["target-prefill"]["input_names"][1] == "effective_length"
fused = graphs["fused-speculative-step"]
capacity = fused["metadata"]["kv_cache_max_len"]
assert fused["dynamic"] is True
assert fused["input_dim_gears"]["0"]["1"] == \
    list(range(1, 17)) + list(range(64, capacity + 1, 64))
assert fused["metadata"]["verify_input_ids_externalized"] is False
assert fused["metadata"]["target_verify_rows"] == 16
assert all(
    graph["metadata"]["quant_branch_base_revision"] ==
    "28f93e784a2beed87020a80bd93c8788754eab1c"
    for graph in data["graphs"]
)
audits = [item for graph in data["graphs"] for item in graph["custom_op_audit"]]
assert audits
for item in audits:
    assert item["status"] == "PASS"
    assert item["ge_node_occurrences"] >= item["minimum_occurrences"]
ge_types = {item["ge_op_type"] for item in audits}
assert {
    "DynamicQuant", "QuantBatchMatmulV4444", "AdnRmsNorm",
    "ChunkGatedDeltaRule", "GatedDeltaRuleMTP", "CacheUpdate",
    "AdnFusedInferAttention", "ScatterNdUpdate",
} <= ge_types
print("fused four-OM AIR manifest gate: PASS")
PY
```

还可以直接检查 TorchAir 的可读 GE 图：

```bash
rg -n '(type|op): "(SoftplusV2|DynamicQuant|QuantBatchMatmulV4444|AdnRmsNorm|ChunkGatedDeltaRule|GatedDeltaRuleMTP|CacheUpdate|AdnFusedInferAttention|ScatterNdUpdate)"' \
  "$FUSED_BUNDLE"/air/*/dynamo.pbtxt
```

并确认旧类型没有残留：

```bash
if rg -n '(type|op): "RmsNorm"' "$FUSED_BUNDLE"/air/*/dynamo.pbtxt; then
  echo "FAIL: receiver requires AdnRmsNorm, not RmsNorm" >&2
  exit 1
fi

if rg -n '(type|op): "QuantBatchMatmulV3"' "$FUSED_BUNDLE"/air/*/dynamo.pbtxt; then
  echo "FAIL: stale QuantBatchMatmulV3 lowering" >&2
  exit 1
fi

if rg -n '(type|op): "FusedInferAttentionScore"' \
  "$FUSED_BUNDLE"/air/*/dynamo.pbtxt; then
  echo "FAIL: stale A2 FusedInferAttentionScore lowering" >&2
  exit 1
fi
```

这里检查的是 AIR 内部 GE type，`npu.*.default` 前端 FX 名称不会原样作为 GE type 出现。
`air-manifest.json` 对框架 converter 同时记录调用数和 GE 节点数；TorchAir builtin 没有被框架
包装，所以其 `converter_calls` 为 `null`，只以 schema/Meta 探针和最终 GE 节点作为门禁。
不同 TorchAir 版本会把 `dynamo.pbtxt` 的节点类型写成 `type: "..."` 或 `op: "..."`。审计器
同时识别两种字段；同一文件、同一 GE type 取两者较大的计数，不能相加后把一个物理节点算两次。

如果 `dynamo_export` 报 graph break，应保留完整 TorchAir 日志并定位首个不支持节点；不能用空
AIR、伪文件或 CPU export 替代。若仍出现 `does not support running with fake tensors`，先看错误中
的完整 `npu.<op>.default`：本分支会预检上述七个 target，出现第八个算子表示 receiver 源码/算子
包已经漂移，需要先锁定它的真实 schema、输出元数据和 GE IR，不能套用已有 Fake 合同。

## 7. AIR 编译成 OM

使用设备对应的精确 SoC 名称，例如真实环境确认是 `Ascend310P3` 时：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$FUSED_BUNDLE/air-manifest.json" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3
```

框架固定调用：

```text
atc --mode=0 --framework=1 \
    --model=<每个 role 对应的 .air> \
    --output=<.../om/role-name> \
    --soc_version=<精确 310P variant>
```

禁止使用模糊的 `Ascend310P`。成功后得到：

```text
$FUSED_BUNDLE/
├── deployment-manifest.json
└── om/
    ├── target-prefill.om
    ├── target-prefill-head.om
    ├── target-decode1.om
    └── fused-speculative-step.om
```

编译器在调用 ATC 前重新核验 AIR 与所有外置 payload 的 hash；ATC 成功后记录 OM hash、ATC
版本、完整命令和日志。它还会重新校验并把 `custom_op_audit` 传入 deployment manifest；缺失
审计或 GE 节点数少于 converter 命中数时不会调用 ATC。退出码为 0 但 OM 缺失或为空也判定
失败。

也可以一次执行 AIR + OM：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-om \
  --factory \
    qwen35_dflash.ascend310p.quant_factory:create_quant_fused_speculative_step_graphs \
  --factory-config "$AI_RUN_DIR/factory-fused.json" \
  --bundle-dir "$FUSED_BUNDLE" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3
```

确认不是误走单图 factory：

```bash
jq -e '
  (.graphs | length == 4) and
  ([.graphs[].name] == [
    "target-prefill", "target-prefill-head", "target-decode1",
    "fused-speculative-step"
  ]) and
  ([.graphs[].om.path] | all(endswith(".om")))
' "$FUSED_BUNDLE/deployment-manifest.json"
```

如果这里只得到 `quant_dflash_recompute.om`，说明命令仍显式使用了
`create_quant_recompute_graph`，或复用了旧 bundle；换一个空的 `$FUSED_BUNDLE` 并按上面的
factory 重跑。不能把单图 manifest 交给多 OM runner。

## 8. 构建 C++ AscendCL runner

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-release" \
  --output "$AI_RUN_DIR/reports/cpp-build.json" \
  --ascendcl-root /ABSOLUTE/PATH/CANN \
  --device-memory-policy normal-only
```

生产 runner 使用：

- `aclInit`、`aclrtSetDevice`、显式 context/stream；
- `aclmdlLoadFromFileWithMem`，四个 role 在进程启动时各加载一次，共享串行 workspace；
- `aclrtMallocHost` pinned host buffer；
- 持久化 device buffer 和 dataset；
- 默认把所有显式 device 分配编译为 `normal-only`；可在独立目录构建 `huge-first` 精确候选，
  两种 runner 都把策略写入报告；
- Target/Draft KV、GDR/conv state、feature 和 proposal carrier 保持 device-resident；
- prompt 以 64-row body 分块，仅末 chunk 执行一次 head；ordinary 后续只执行 decode1；
- DFlash 每个物理 fused 调用依次完成 Draft proposal、固定 T16 Target verify 和精确 commit；
- 每个同步窗口只下载 compact commit 结果，不把完整 state 搬回 host；
- token 调度、DFlash 接受/correction/bonus、EOS 都在 C++17 中完成。

融合只删除 Draft→verify 的一次物理模型 launch 和跨模型 carrier 边界，不改变 Target/Draft
数学、greedy 接受规则或 fixed-T16 verify 工作量。

生产二进制通常位于：

```text
$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner
$AI_RUN_DIR/build/cpp-release/qwen35_dflash_incremental_acl_runner
```

主流程必须使用第二个 `qwen35_dflash_incremental_acl_runner`。第一个二进制只运行单一重计算
OM 基线；第二个二进制运行五图、统一 Target-step 四图或 fused speculative-step 四图常驻 OM
候选。`build-cpp` 会同时构建并 host-test 两者，控制面会按 `state_policy` 严格核对二进制的
`--help` 合同，runner 选错会在加载 checkpoint/AIR/OM 前失败。各拓扑的导出 factory、runner
配置、直接运行、report 门禁、同二进制
`one-token-h2d`/`last-token-d2d` A/B 和 msprof 命令见
`docs/INCREMENTAL_OM_PERFORMANCE.md` 第 5 节。runner 配置中的 `decode_carrier_policy` 只改变
C++ buffer/copy 路由，不要求重新生成 AIR 或 OM；`dflash_sync_window=1..8` 同样不改 AIR/OM
ABI。候选 2 直接合并两槽 compact result，候选 3..8 通过 4 KiB staging arena 把多个完整
speculative transaction 合并到一个 host-visible barrier，并逐轮按剩余 token budget 自适应缩小
K；默认仍为 1，必须完成同机正反顺序 1/2/4/8 A/B。独立的
`prefill_completion_policy=separate|coalesce-first-verify`
也不改 AIR/OM ABI；候选策略把最后 prefill 与第一次 verify 合为一次 D2H/同步，但会延迟首 token
host 可见时间，必须分别比较 TTFT 与总时延。prefill control 按 base/count/proposal/full 四档
live prefix 复制：五图 slot 为 896 bytes；统一四图在 EOS count 后追加一个 64-byte 对齐的常驻
INT32 零值，slot 为 960 bytes，ordinary T=1 直接绑定该零值，不再把正 proposal carrier 写成 0。
`zero_accept_fallback_policy=disabled|request-target-only` 也只改变 C++ 调度；候选在首次零接受并
消费完整同步窗口后关闭本请求后续 Draft，低接受率 A/B、计数门禁和正反序 3+10 命令见增量性能
文档第 5.5.2 节。
前三档仍为 578/644/708 bytes；这些内部 carrier 变化不改 tensor 名、shape 或 AIR/OM ABI。
`normal-only`/`huge-first` 必须构建为两个同源码二进制，并做正反顺序真机 A/B；完整命令与
选择门槛见增量性能文档第 5.5.5 节，未完成证据前默认仍是 `normal-only`。

不要把 build 目录或二进制提交进源码仓库。

## 9. 用 C++ 调用 OM 完整生成 token

复制并填写真实运行时身份：

```bash
cp config/quant_air_om_incremental_runner.example.json \
  "$AI_RUN_DIR/runner-fused.json"
```

`device_model` 必须写具体产品和 310P variant；`cann`、`driver`、`firmware` 必须来自当前
设备，不能保留 `REPLACE_*`。还必须保留
`"state_policy": "incremental-explicit-state-v2"`；不要使用单图示例里的
`recompute-committed-prefixes`。

运行：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest \
    "$FUSED_BUNDLE/deployment-manifest.json" \
  --runner \
    "$AI_RUN_DIR/build/cpp-release/qwen35_dflash_incremental_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner-fused.json" \
  --model-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --chat \
  --max-new-tokens 32 \
  --max-draft-tokens 15 \
  --device-id 0 \
  --output "$AI_RUN_DIR/reports/cpp-infer.json"
```

这条命令中 Python 只负责一次 tokenizer、启动 runner 和最后 detokenize。进入 runner 后：

1. 控制面校验四个 role、输入/输出 ABI 和 OM SHA-256；
2. C++ 再校验 hash 并常驻加载四个 OM；
3. C++ 运行 ordinary greedy 3 次 warmup + 10 次测量；
4. C++ 运行 DFlash strict-greedy 3 次 warmup + 10 次测量；
5. ordinary 调 prefill/head/decode1，DFlash 调 prefill/head/fused step，直到 EOS 或长度上限；
6. C++ 比较 ordinary 与 DFlash token IDs、EOS/stop reason；
7. 任一 token 不一致立即失败。

所以它不是“Python 算完 token、C++ 只读结果”，而是 C++ 的 token 循环实际调用四个 OM 完成
Target/Draft 推理。`qwen35_dflash_acl_runner` 只认识 `--model` 单图 ABI；把它与 fused manifest
混用会被新的 preflight 直接拒绝。

该命令默认把控制面和 C++ runner 进度实时输出到终端，同时逐行刷新到
`$AI_RUN_DIR/log/<报告名>-cpp-runner.log`。典型输出如下：

```text
[infer-cpp] stage=runner-start live child output follows
[qwen35-dflash] stage=load-om-start
[qwen35-dflash] phase=warmup run=1/3 mode=ordinary-greedy stage=prefill-start generated=0/32 ...
[qwen35-dflash] phase=measurement run=1/10 mode=dflash-strict-greedy stage=decode-done generated=16/32 ...
```

每个 prefill/decode 计时区间的日志都在计时开始前或结束后输出，不计入
`prefill_ms`、`decode_ms` 和 `model_total_ms`。正式无人值守采样如需静默，可加
`--no-progress`；日志文件仍会完整保留子进程输出。

## 10. 一键端到端

先构建 C++ runner，然后执行：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e-cpp \
  --factory \
    qwen35_dflash.ascend310p.quant_factory:create_quant_fused_speculative_step_graphs \
  --factory-config "$AI_RUN_DIR/factory-fused.json" \
  --bundle-dir "$AI_RUN_DIR/e2e/artifacts" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3 \
  --runner \
    "$AI_RUN_DIR/build/cpp-release/qwen35_dflash_incremental_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner-fused.json" \
  --model-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --chat \
  --max-new-tokens 32 \
  --max-draft-tokens 15 \
  --device-id 0 \
  --report-dir "$AI_RUN_DIR/e2e/reports"
```

它依次执行真实设备 preflight、AIR、ATC、OM hash gate、C++ paired generation 和最终
summary。任何阶段失败都不会伪造后续 PASS。

当前代码中 `run-e2e-cpp` 即使省略 `--factory` 也默认选择上述 fused 4-OM factory；文档仍显式写出
该参数，便于从日志直接确认拓扑。Python `run-e2e` 的默认值仍是单图重算 factory，两者不要混淆。

## 11. 如何判定框架“能用”

### 11.1 功能 PASS

最终 `cpp-infer.json` 或 `e2e/reports/summary.json` 必须同时满足：

```python
assert report["status"] == "PASS"
assert report["runner_id"] == "qwen35-dflash-ascendcl-cpp-incremental-v3"
assert report["cpu_fallback"] is False
assert report["backend_metadata"]["state_policy"] == \
       "incremental-explicit-state-v2"
assert report["abi"]["physical_topology"] == \
       "split-prefill-head-four-resident-fused-speculative-step-v1"
assert {item["role"] for item in report["models"]} == {
    "target-prefill", "target-prefill-head", "target-decode1",
    "fused-speculative-step",
}
assert report["ordinary_parity"]["status"] == "PASS"
assert report["ordinary_parity"]["token_id_mismatches"] == 0
assert report["ordinary_parity"]["eos_mismatches"] == 0
assert report["ordinary"]["warmup"] == 3
assert report["ordinary"]["repetitions"] == 10
assert report["dflash"]["warmup"] == 3
assert report["dflash"]["repetitions"] == 10
assert report["ordinary"]["stable_generated_token_ids"] == \
       report["dflash"]["stable_generated_token_ids"]
io = report["execution_io_counters"]
assert io["model_executions"] == sum(
    io[name] for name in (
        "target_prefill_executions", "target_prefill_head_executions",
        "target_decode1_executions", "fused_speculative_step_executions",
    )
)
assert io["draft_to_verify_model_launches_elided"] == \
       io["fused_speculative_step_executions"]
assert io["stream_synchronizations"] <= io["model_executions"]
```

此外检查：

- 报告中的 OM SHA-256 与 `deployment-manifest.json` 一致；
- device/cann/driver/firmware 与 `npu-smi` 和当前软件栈一致；
- 10 次输出 token 和 stop reason 完全稳定；
- prompt token 数不超过 `abi.sequence_capacity`，每轮 proposal 不超过 15；
- 没有 CPU fallback、mock、fake ACL 或 simulation 标记。
- 四个 `models[].sha256` 与同一份 deployment manifest 一致；
- `execution_io_counters` 中 prefill、decode、fused execution 之和与物理 model execution 闭合；
  如果真机 msprof 显示每轮完整 state H2D/D2H，说明实际 runner/source 与报告身份不一致。

直接查看本次传输缩减量：

```bash
jq '{
  runner_id,
  topology: .abi.physical_topology,
  models: [.models[] | {role, model_id, sha256}],
  executions: .execution_io_counters | {
    model_executions, target_prefill_executions,
    target_prefill_head_executions, target_decode1_executions,
    fused_speculative_step_executions,
    draft_to_verify_model_launches_elided, stream_synchronizations
  },
  ordinary_parity
}' "$AI_RUN_DIR/reports/cpp-infer.json"
```

### 11.2 多 workload 正确性

至少覆盖：

1. 中文短 prompt；
2. 英文短 prompt；
3. code prompt；
4. 接近 gear 上限的长 prompt；
5. 会提前 EOS 的 prompt；
6. 生成到长度上限的 prompt；
7. Draft 全接受、首 token 拒绝和中间拒绝三类轮次；
8. 同一 prompt 重复进程执行，检查无状态泄漏。

每个 workload 都必须零 token/EOS 差异。不能只用一条容易全部接受的短文本证明框架正确。

### 11.3 性能 PASS

功能通过后才看性能。报告中使用 `latency_ms.model_total`，排除模型加载和 tokenizer；保留
完整 10 个原始值、median 和 p90。

需要与闭源框架比较时，闭源报告按
`framework/abi/closed-runtime-baseline-v1.json` 填写，必须相同：

- 物理设备和 device ID；
- CANN、driver、firmware；
- Target/Draft/量化输入 manifest；
- Target W8A8、Draft FP16；
- 静态 gear；
- prompt token IDs、输出 token IDs、EOS；
- concurrency=1；
- 模型加载排除；
- 显式设备同步的 model-loop 计时范围。

然后先给定可接受比率，例如 median/p90 均不得超过闭源的 1.10 倍：

```bash
"$MODEL_PYTHON" framework/scripts/compare_cpp_closed_runtime.py \
  --cpp-report "$AI_RUN_DIR/reports/cpp-infer.json" \
  --closed-report "$AI_RUN_DIR/reports/closed-runtime.json" \
  --max-median-ratio 1.10 \
  --max-p90-ratio 1.10 \
  --output "$AI_RUN_DIR/reports/cpp-vs-closed.json"
```

阈值必须在看结果前确定。即使 C++ 更快，只要 token/EOS 或运行身份不一致仍然 FAIL。

### 11.4 用 msprof 单独分析当前 OM

主路线必须 profile 同一次 fused bundle 中的四个常驻 OM：`target-prefill`、
`target-prefill-head`、`target-decode1`、`fused-speculative-step`。使用
[增量 OM 与 C++ 高性能路线](INCREMENTAL_OM_PERFORMANCE.md) 第 5.7 节的完整命令；它以
`qwen35_dflash_incremental_acl_runner --measurement-protocol profile` 运行真实状态机，并把运行时
model ID、每次物理执行 role/gear、AscendCL API 次数与 msprof CSV 关联。不能分别用随机 state
直接调用四个 OM 后把结果当作端到端性能。

分支还提供五图基线 `create_quant_incremental_state_graphs`、四物理 OM 动态候选
`create_quant_unified_target_step_graphs`、四物理 OM Draft+verify 精确融合候选
`create_quant_fused_speculative_step_graphs` 和同一个常驻 OM C++ runner。生成后，应使用
`docs/INCREMENTAL_OM_PERFORMANCE.md` 第 5.7 节对应拓扑的完整状态机 msprof 命令，并按
model ID/role/gear 分组；不同拓扑的 profile 不得混成同一份时延基线。统一候选省略独立
`target-decode1`，由 `target-verify-commit` 的 T=1 gear 执行 ordinary decode；T=K+1 只执行
本轮有效 verify 行。fused 候选保留静态 `target-decode1`，由一个
`fused-speculative-step` model ID 同时运行逻辑 Draft 和固定 T16 Target verify；msprof 只能报告
该 supergraph 的物理总耗时，再按 operator 行分析内部热点，不能伪造两个 OM 时延。该 runner 会把长 prompt 的中间
64-row chunk 留在同一 stream，仅最后一个 chunk 下载 compact 结果并同步；报告中的 elided
prefill 计数必须与 `ceil(prompt_tokens/64)-1` 按请求数闭合。每个 chunk 的 ID、有效长度、累计
token 数、proposal count 和 EOS 表仍合并为一次 H2D，但复制长度按消费活性收窄为
base/count/proposal/full 四档；默认 `eos_table_width=4` 时，五图为 578/644/708/896 bytes；统一
四图追加常驻对齐零值后为 578/644/708/960 bytes。EOS tail 和常驻零值只在首次使用或 Reset
改变 EOS 身份时刷新，ordinary T=1 不产生独立 4-byte `K=0` H2D。
`prefill_control_upload_operations` 必须等于
`target_prefill_executions`，四条路径 operation 之和必须等于该总数，实际 byte 加 elided byte
必须等于全量 carrier 等价 byte。普通连续 decode 可由
同一二进制选择两种精确策略：`one-token-h2d` 仅把一行 compact 结果直接绑定，多 token commit
走 8-byte H2D；`last-token-d2d` 把最后提交的 compact token 留在 device，第 0 行直接绑定，多
token 末行做一次 8-byte D2D 到对齐 scalar。report 必须满足
`decode_id_device_carrier_hits + decode_id_upload_operations == target_decode1_executions`，且
`decode_id_device_compaction_operations == decode_id_multi_token_carrier_hits`；真机必须用相同
runner/OM/input 做未 profile 的 3+10 A/B，msprof API timeline 仅用于解释新增 D2D 与被替换 H2D。
同步窗口 1/2 也必须使用相同 runner/OM/input 做正反顺序 3+10；实际 D2H 加
`speculative_d2h_operations_elided` 与 `prefill_verify_d2h_operations_elided` 后按 transaction
闭合，stream synchronization 按
`speculative_sync_windows` 闭合，完整命令见增量性能文档第 5.5.1 节。

多 OM profile 必须使用报告 schema 12 中的运行时 model ID、逐次执行 trace、Draft feature
策略、编译期设备内存分配策略与逐次物理行数做严格归因，
不能靠文件顺序猜测 OM 角色。采集完成后运行
`python -m qwen35_dflash.ascend310p analyze-msprof`；完整命令、输入约束和判定规则见
`docs/INCREMENTAL_OM_PERFORMANCE.md` 第 5.7 节。分析器会拒绝只导出单个 model/iteration 的
不完整结果、重复执行记录以及 ACL API 次数不闭合的报告。

下面的手工流程仅用于可选的单图 `quant_dflash_recompute.om` 诊断基线：Target 全前缀和 Draft
proposal 在同一张图里，只能回答整图算子/device task/API 耗时，不能拆出四个模型级 role，也
不能替代上面的 fused 4-OM profile。

正式时延基线仍然使用 11.3 中未开 profiling 的 3 次 warmup + 10 次
measurement 报告。下面的 msprof 命令只做瓶颈定位，采集器引入的开销不能算入
闭源框架对比值。

在真实 310P 环境先准备路径。`PROFILE_ROOT` 必须在拷贝的源码树之外；
`MODEL_PYTHON` 必须能导入当前 CANN 匹配的 `torch_npu`。`TOKEN_IDS` 不是随机数，
必须换成正式 workload 的真实 token ID：

```bash
export DFLASH_SOURCE=/ABSOLUTE/PATH/qwen3.5-4B-dflash
export PROFILE_ROOT=/ABSOLUTE/PATH/quant-air-om-msprof
export MODEL_PYTHON=/ABSOLUTE/PATH/python
export RUNNER=/ABSOLUTE/PATH/qwen35_dflash_acl_runner
export OM=/ABSOLUTE/PATH/quant_dflash_recompute.om
export OM_SHA256="$(sha256sum "$OM" | awk '{print $1}')"
export PAD_TOKEN_ID=0
export TOKEN_IDS='151644,8948,198,2610,525'
export CASE_KIND=prefill

mkdir -p "$PROFILE_ROOT"
```

上面的 token 只是格式示例，不得作为正确性或性能证据。`PAD_TOKEN_ID` 也要与
`runner.json` 和 tokenizer 一致。先确认 OM hash 和 runner 接口：

```bash
sha256sum "$OM"
"$RUNNER" --help
```

下面对同一个 OM 分别采集 `PipeUtilization`、`Memory` 和 `MemoryUB`。一个
profile 只收集一组 AI Core metrics，每组使用独立 label 和输出目录：

```bash
for AIC_METRIC in PipeUtilization Memory MemoryUB; do
  LABEL="quant-recompute-${CASE_KIND}-${AIC_METRIC}"
  CASE_ROOT="$PROFILE_ROOT/$LABEL"

  "$DFLASH_SOURCE/tools/run_msprof.sh" \
    --label "$LABEL" \
    --output-dir "$CASE_ROOT" \
    --python "$MODEL_PYTHON" \
    --aic-metrics "$AIC_METRIC" \
    --no-msproftx \
    -- \
    "$RUNNER" \
      --model "$OM" \
      --model-sha256 "$OM_SHA256" \
      --output "$CASE_ROOT/cpp-report.json" \
      --prompt-token-ids "$TOKEN_IDS" \
      --pad-token-id "$PAD_TOKEN_ID" \
      --max-new-tokens 1 \
      --max-draft-tokens 15 \
      --warmup 3 \
      --repetitions 10 \
      --device-id 0 \
      --progress true
done
```

runner 的真机证据合同固定为 paired 3+10，不允许改成 1+1。这里把
`max-new-tokens` 设为 1，所以 ordinary 和 DFlash 每轮都只执行一次相同的集成
OM，合计为 `2 × (3 + 10) = 26` 次单 OM 重复采样。这个 profile 不包含
decode 循环，不用于计算 acceptance rate；它的目的是让同一输入的 OM 任务重复出现，
便于排除单次抖动。

要比较当前集成 OM 在三种逻辑位置上的成本，只修改输入和 `CASE_KIND`，
然后重跑上面的循环：

| `CASE_KIND` | `TOKEN_IDS` | 采样含义 |
| --- | --- | --- |
| `prefill` | 完整真实 prompt | 首次 Target + Draft 集成图调用 |
| `proposal` | 某轮开始时的已提交前缀 | 生成 proposal 时的集成图输入 |
| `verify` | 同一已提交前缀 + 该轮真实 Draft proposals | Target verify 时的扩展前缀 |

例如，从一轮真实 DFlash trace 中拿到前缀和 proposal 后：

```bash
export CASE_KIND=proposal
export TOKEN_IDS='REAL_COMMITTED_PREFIX_TOKEN_IDS'
# 重跑上面三个 AIC_METRIC 的 for 循环

export CASE_KIND=verify
export TOKEN_IDS='REAL_COMMITTED_PREFIX_PLUS_REAL_PROPOSAL_TOKEN_IDS'
# 再重跑上面三个 AIC_METRIC 的 for 循环
```

这种方法只改变同一静态 OM 的有效前缀和 mask。如果 proposal/verify 的逻辑长度
增加，但 device task 耗时基本不变，这是静态全前缀重算成本主导的直接证据之一。
因为 Target 和 Draft 仍在一张图中，不能把两者的高层 wall time 从这个数字中
强行分开。

每次采集完后，先找到该 label 下 msprof 生成的 `PROF_*` 目录，再执行
query 和 CSV export。下面以 prefill/PipeUtilization 为例：

```bash
export LABEL=quant-recompute-prefill-PipeUtilization
export CASE_ROOT="$PROFILE_ROOT/$LABEL"

find "$CASE_ROOT/profile/msprof/$LABEL" \
  -maxdepth 2 -type d -name 'PROF_*' -print

export PROF_DIR=/ABSOLUTE/PATH/TO/PROF_DIRECTORY
msprof --query=on --output="$PROF_DIR"
msprof --export=on --output="$PROF_DIR" --summary-format=csv

rg --files "$PROF_DIR" | \
  rg '/(op_summary|op_statistic|api_statistic|task_time)_[^/]*\.csv$'
```

不同 CANN 版本的子目录层级可能不同，所以先用 `find` 确认唯一的
`PROF_*` 目录，不要猜路径。优先保留和分析：

| 文件 | 用途 |
| --- | --- |
| `op_summary_*.csv` | 每个算子/task 的执行时间、输入 shape 和核心利用率 |
| `op_statistic_*.csv` | 按算子类型聚合的调用次数和总时间，适合找 Top 瓶颈 |
| `api_statistic_*.csv` | H2D/D2H、`aclmdlExecuteAsync` 和 stream synchronize 等 host API 成本 |
| `task_time_*.csv` | device task 调度与执行时间，用于区分 host 等待和 NPU 计算 |
| `manifest/<label>.json` | 本次 msprof 命令、framework/C++ 与模型适配源码内容 hash、设备和采集参数 |
| `cpp-report.json` | 输入/输出稳定性、OM 调用数和 runner 计时边界 |

发回问题时，上述四类 CSV、manifest、`cpp-report.json` 以及
`log/msprof-<label>.log` 通常就是第一轮分析最需要的小文件，无需先上传完整时间线。

分析时遵循以下规则：

1. 先用 `cpp-report.json` 确认确实是 26 次单图调用且 token 稳定；
2. 只对执行次数与采样协议匹配的算子做单次均值，不要把模型加载任务除以 26；
3. `aclmdlExecuteAsync` 是 host 侧下发时间，不是完整 device 时延；需要结合
   `aclrtSynchronizeStream`、`task_time` 和时间线；
4. 不要把并行或重叠的 AI Core task 耗时直接相加当作 OM wall time；
5. 若少数自定义算子占主导，再根据精确节点、shape 和 metrics 做有界优化；
6. 若整张静态图的物理工作量才是主导，则进入第 13 节的增量状态 ABI，继续优化
   C++ 控制面不会消除全前缀重算。

多 OM 候选应使用 `qwen35_dflash_incremental_acl_runner` 运行完整状态机。主 fused 路线按
`target-prefill`、`target-prefill-head`、`target-decode1`、`fused-speculative-step` 的 model ID
分组；五图对照再使用独立 `draft-propose` 与 `target-verify-commit`。完整命令见
`docs/INCREMENTAL_OM_PERFORMANCE.md` 第 5.7 节。不要拿只接受集成图 2 input/2 output ABI 的
`qwen35_dflash_acl_runner` 改文件名运行这些状态 OM，也不要用随机零 state 作为性能或正确性
证据。

## 12. 常见失败定位

| 失败 | 含义 | 处理 |
| --- | --- | --- |
| input manifest hash mismatch | 权重、量化文件或 wrapper 已变化 | 先确认变化是否预期；重新冻结，不要跳过校验 |
| 找不到 `export_model_wrapper_qwen3_5.py` | receiver 路径错误 | 修正 `receiver_models_dir` |
| QLinear coverage mismatch | 量化权重与 Target topology 不同 | 核对 YAML、checkpoint revision 和 quant artifact |
| `torch_npu`/TorchAir import 失败 | 环境不匹配 | 使用与 CANN/驱动匹配的声明环境 |
| 模型加载阶段 `BatchMatMul4` / `507014 AICore timeout`，同步 traceback 落在 `_set_cos_sin_cache` | tied `lm_head.weight` 触发 Transformers 补缺初始化，旧 modeling 又在已迁移到 NPU 后重复构建 `max_position_embeddings` 长度的 RoPE cache；与 prompt 长度、DFlash decode 和 block table 无关 | 更新本分支并新起 Python 进程；设置 `ASCEND_LAUNCH_BLOCKING=1` 复验，日志不得再出现 `_initialize_missing_keys -> _init_weights -> _set_cos_sin_cache`。超时后的旧 NPU context 不可继续复用 |
| `unsupported operator: npu.<op>.default` | 七算子预检未运行、实际代码不是本分支，或 receiver 新增了第八个算子 | 核对远端提交、`PYTHONPATH`；已覆盖清单见 2.1，不能用 modeling Tensor fallback 掩盖 |
| `schema drifted from the locked export contract` | torch-npu/receiver 算子签名与当前锁不一致 | 记录 dispatcher schema；按真实版本更新 schema、Fake、converter 和测试，不能跳过校验 |
| `Meta contract mismatch` / `lost input alias` | 上游 Meta 或本地 Fake 与真实 shape/dtype/原位语义不一致 | 停止导出，先以算子包实现和实机输出重新冻结合同 |
| `RmsNorm` unsupported、或图审计找不到 `AdnRmsNorm` | factory/旧 converter 使用了通用 GE 名称，但 receiver 注册的是 `AdnRmsNorm(self,gamma)->(y,rstd)` | 把 `adn_rms_norm_ge_op_type` 改为 `AdnRmsNorm`，更新本分支后重新生成空 AIR bundle；不要复用旧 AIR |
| `the pertoken_scale 1st dim value must be x1 m dim value` | 旧版框架的 QuantMatmul Meta 探针误用了 `[B*M]` scale；不是 NPU kernel 失败 | 更新本分支；确认 `x1=[1,M,K]` 时探针和模型都传 `pertoken_scale=[M]` |
| `aclnnQuantMatmulV4 failed, error code is 161002`，并提示 `Scale dtype should be UINT64 or INT64, actual dtype is DT_FLOAT` | 当前进程解析到了官方或另一套同名 `aclnnQuantMatmulV4`，没有进入支持 FP32 weight scale 和 per-token scale 的自定义 V4444 wrapper；同一环境下原 `quant` 分支也会失败 | 不要修改模型 scale。确认 `customize_quantMatmul/op_api/lib` 位于 `LD_LIBRARY_PATH` 最前、完整 vendor 位于 `ASCEND_CUSTOM_OPP_PATH`，排除同名旧包后新起 Python 进程；用 `/proc/<PID>/maps` 或 `LD_DEBUG=bindings` 核对实际加载的 `libcust_opapi.so` |
| `aclnnQuantMatmulV4 failed, error code is 561103`，并提示 `Cannot find bin ... int8/ND/int8/ND/int64/ND/float16/ND` | 旧框架对 `quant` 分支的 FP32 scale 错误调用了 `npu_trans_quant_param`，形成 V4444 注册表不存在的 `INT8 + INT8 + INT64` 组合 | 更新本分支；普通 eager 应与 `quant@28f93e7` 一样直接传 FP32 weight scale 和 FP32 per-token scale，AIR 继续走私有 FP32-scale frontend。不要用 `.to(torch.uint64)` 或 `.to(torch.int64)` 数值强转 |
| `Found a custom (non-ATen) operator whose output has alias annotations`，随后 `Original traceback` 指向 `npu_cache_update_` | 导出前没有准备私有 CacheUpdate frontend，或实际 modeling 仍显式进入了源 alias op | 更新本分支并在新 Python 进程导出；FX/AOT target 应为 `qwen35_dflash.npu_cache_update.default`，不应含 `npu.npu_cache_update_.default` 或 `aten.copy`，且 `dynamo.pbtxt` 仍包含 `CacheUpdate` |
| `ERR03005 GRAPH internal error`，`Original traceback` 指向 `qwen35_dflash.npu_cache_update.default`，Meta 单测通过 | 旧 converter 把前端 snake_case 参数按 positional 传入，但 GE `CacheUpdate` 原型要求 `x/updates/targetBlock/offsetInBlock -> x` | 更新本分支；确认 AIR manifest 中该算子的 `converter_mode` 为 `named-cache-update-x-v1` |
| `TorchAir IR contains 0 DynamicQuant nodes`，但 `dynamo.pbtxt` 明确含 `op: "DynamicQuant"` | 旧审计器只识别 `type:` 字段；AIR 和权重保存实际上已经完成 | 更新本分支；不要把 DynamicQuant 改为 optional，确认 manifest 中 `ge_node_occurrences >= 1` |
| `TorchAir IR contains 0 QuantBatchMatmulV4444 nodes for npu.npu_quant_matmul.default`，同时图中仍有 `QuantBatchMatmulV3` | receiver TorchAir 的内置 V3 converter 与旧项目 converter 注册在同一个 FX target 上，内置项仍被采用 | 更新本分支并重新导出到空目录；确认 audit target 已变为 `qwen35_dflash.npu_quant_matmul_v4444.default`，图中有 V4444 且无 V3 |
| ATC 报 `QuantBatchMatmulV3` unsupported 或 FP32 scale 不匹配 | 旧 factory/builtin converter 生成了 CANN V3，而 receiver 实际安装 V4444 | 把现有 `factory.json` 改为 `QuantBatchMatmulV4444`，重新生成 AIR；确认图中有 V4444 且没有 V3 |
| `No supported Ops kernel and engine ... FusedInferAttentionScore`（Ascend310P3） | 旧 AIR 把 receiver 的 ADN 前端错误 lower 成 A2 GE type；真实 310P 包注册的是 `AdnFusedInferAttention` | 更新本分支和现有 `factory.json`，把完整 ADN vendor 加入 `ASCEND_CUSTOM_OPP_PATH`，重新导出到空目录；确认图中只有 `AdnFusedInferAttention` 再跑 ATC |
| `AdnFusedInferAttention GE prototype is absent from the active ASCEND_CUSTOM_OPP_PATH` | PyTorch/LD 能加载 ADN op_api，但 ATC 的 OPP 搜索路径里没有对应 prototype/kernel vendor | 把 ADN 安装包的 `packages/vendors/<vendor>` 根加入 `ASCEND_CUSTOM_OPP_PATH`；不要只加入 `op_api/lib` |
| `pse_shift` 期望 `Optional[Tensor]` 但收到 `[64]` / `immutable_list` | 旧版 modeling 在 export 路径把 `allQLen` 长度列表误接到了 PSE 输入，尚未进入 Fake/converter | 更新本分支；确认两个 modeling 文件均传 `all_seq_lengths_q=allQLen` 且不构造伪 PSE Tensor |
| `GE IR ... is not registered` | factory 中某个 `*_ge_op_type` 与目标 CANN/自定义包不一致 | 使用已正式注册且与算子实现一致的 GE type；不能用同名伪节点 |
| custom-op converter/GE-node count 为 0 | 算子被绕开、converter 未调用或 GE 图丢失节点 | 导出按 FAIL 处理，保留 `dynamo.pbtxt` 和完整 TorchAir 日志 |
| TorchAir graph break | 某个 Python/自定义 op 未被捕获 | 定位首个 graph break，补正式 converter；不要伪造 AIR |
| ATC unsupported op | TorchAir 图中存在 ATC 不支持节点 | 保留算子名和编译日志，决定分解或正式自定义算子 |
| generic `Ascend310P` rejected | SoC 身份不精确 | 从设备/ATC 支持列表填写真实 variant |
| OM input/output count mismatch | 导出 ABI 漂移 | 必须恢复 2 input/2 output INT64 合同或版本化新 ABI |
| C++ OM hash mismatch | OM 被替换或 manifest 错配 | 使用同一次 build 的 OM 和 deployment manifest |
| `qwen35_dflash_acl_runner: unknown option --measurement-protocol` | 旧控制面把仅多 OM runner 支持的参数传给了单 OM runner，或选择了错误二进制 | 更新本分支；fused/五图/统一四图必须使用 `qwen35_dflash_incremental_acl_runner`。新 preflight 会在权重导出前拒绝这种组合 |
| 期望多 OM 但只生成 `quant_dflash_recompute.om` | 命令省略/写错 factory，仍选择单图重算路线，或查看的是旧 bundle | 使用 `create_quant_fused_speculative_step_graphs` 和新的空 bundle；按第 7 节确认 manifest 恰好四个 role |
| `ValueError: invalid literal for int() with base 10: 'input_ids'` | Python tokenizer 返回 `BatchEncoding`/Mapping，旧控制面误把字段名当 token ID 遍历；此时尚未启动 C++ runner | 更新本分支后重跑同一 `infer-cpp`；无需重新生成 AIR/OM，命令中的重复 `--chat`、`--max-new-tokens` 和 `--max-draft-tokens` 各保留一次 |
| ordinary/DFlash token mismatch | 接受、correction、pad 或图语义错误 | 停止性能测试，定位首个 token 分叉 |
| 延迟明显慢于闭源 | 完整前缀重算成为主瓶颈 | profile 后进入增量 OM state ABI，不要只优化 Python |

## 13. 性能验证与后续候选

当前 C++ 已消除 Python token 热循环、重复 OM load、重复 host/device buffer 分配和多余 stream
同步，并实现了五图基线、四物理 OM 统一 Target-step 候选和作为 C++ 默认 factory 的 fused
四物理 OM 候选。若真实 profile 显示 OM 计算主导，
应基于 `quant` 已有 rollback 语义比较这些**逻辑角色**：

1. `target-prefill.om`：分块 prompt body，不含 LM head；
2. `target-prefill-head.om`：只在最后一个 prompt chunk 后运行量化 LM head/Top1/EOS；
3. `target-decode1.om`：ordinary 单 token decode；
4. `target-verify-commit.om`：anchor + 最多 15 proposals；
5. `draft-propose.om`：增量 Draft KV；
6. 统一候选删除第 3 项独立文件，由第 4 项的动态 T=1..16 同时承担 decode/verify；
7. C++ request context 持有 Target/Draft state 的 persistent device buffer；
8. proposal 直接在 device 上从 Draft 传给 Target verify；
9. verify graph 尾部精确计算 accepted count 并选择 state slot，只把 compact commit result 搬回
   host；
10. 对每一个 state 分支做 ordinary token/EOS 零差异门禁。

多 OM 可能更快的原因不是“文件数量更多”，而是它们允许把已经计算过的状态留在
NPU，后续调用只处理新增 token：

| 路径 | 每次物理工作 | 性能含义 |
| --- | --- | --- |
| 单图诊断重算 OM | 对静态 `S` 重算 Target 全前缀，并同时计算 Draft | 生成 1 个 token 也会重做历史行；不需要 Draft 时也付出 Draft 代价 |
| `target_prefill_64.om` | prompt 分块只执行一次，产生 Target KV/GDR/conv state | 历史 prompt 不再在每个 decode 轮次重算 |
| `target-prefill-head.om` | 只消费最后一个 `[1,1,H]` hidden | 非末 64-row chunk 不再计算完整词表投影 |
| `target_decode_1.om` | 处理 1 个新 token，读写常驻 Target state | ordinary/correction 路径的 query 长度从 `S` 降为 1 |
| `draft_16.om` | 复用 Draft KV/feature state 生成一块 proposals | 不再为每轮 proposal 重算 Draft 全前缀 |
| `target_verify_16.om` | 一次验证 anchor + 最多 15 个 proposals | 接受多个 token 时，一次 Target 调用被多个输出 token 分摊 |
| `fused-speculative-step.om`（独立候选） | 在一张精确 supergraph 内顺序执行上述 Draft 与固定 T16 verify | 每个 speculative transaction 删除一次 Draft→verify 物理 OM launch；不减少 Target 数学工作 |

拆分后 ATC 还可以针对 `S=1`、`S=16` 和 `S=64` 分别选择内核、tiling 和工作区，避免用一个
大的静态 shape 覆盖所有阶段。但物理文件不一定正好是四个：若 verify 能证明 `T=1` gear，decode
可与 verify 合并；若多 gear、自定义算子和分支均通过，可测试 2 个动态 OM；若 launch 边界占主导，
可比较保留静态 decode 的 fused speculative-step 四图候选。要得到端到端
收益，必须同时满足：

- 五个当前物理 OM 只加载一次，device buffer 复用；
- KV/GDR/conv/Draft state 常驻 device，不在每轮整体 H2D/D2H；
- verify 的 candidate bank 留在 graph workspace，尾部只持久化 accepted slot，拒绝时不重建全部
  历史；
- C++ 尽量使用同一 stream 上的异步执行，不在每个小操作后同步；
- Draft 接受率和每轮接受 token 数足以摊薄 `draft + verify` 两次调度开销。

如果巨大 state 每轮搬回 CPU、新增了过多 stream synchronize、小 shape 的启动开销占主导，
或 Draft 几乎全被拒绝，多 OM 路径可能反而更慢。不同 Target OM 还可能各自携带一份 Target
权重，不能假设自动共享。新增的 `qwen35_dflash_om_inspect` 会记录每个 OM 的
`aclmdlQuerySize(workSize, weightSize)`，按 `sum(weights) + max(serial workspace) + state + margin`
计算候选集；当前 C++ runner 的 JSON 也会记录单 OM 的查询值。

完整状态字节公式、不同物理拓扑候选、当前五图 inspector 构建/执行命令、一次同步热循环和审批门禁见
[增量 OM 与 C++ 高性能路线](INCREMENTAL_OM_PERFORMANCE.md)。门禁顺序必须是：先做单 OM
msprof、候选集合内存/load 和状态搬运审计，再做未 profiling 的端到端 3+10，最后与同身份闭源
基线比较。

代码默认选择 fused factory 只代表交付拓扑已明确，不代表性能候选已经通过真机提升门禁。在没有
真实 baseline、state-branch、完整显存和零差异证据前，报告中的候选状态仍必须保持
`APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE`。

## 14. 官方接口依据

- [TorchAir `dynamo_export`](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0052.html)：接口参数、静态/动态图、单图约束和 AIR 大小约束；
- [TorchAir 自定义算子 converter](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00045.html)：自定义 ATen IR 需要注册 converter 后转换为 GE IR；
- [TorchAir AIR 产物与外置权重](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0019.html)：`export.air`、`dynamo.pbtxt` 和 `weight_*` 的关系；
- [ATC `--framework`](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/atc/atlasatcparam_16_0014.html)：TorchAir 标准 AIR 使用 `--framework=1`；
- [AscendCL `aclmdlExecuteAsync`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0299.html)：异步模型执行接口；
- [AscendCL `aclrtMemcpyAsync`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0106.html)：stream 上的异步输入输出复制；
- [AscendCL `aclmdlLoadFromFile`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0283.html)：从 OM 文件加载模型。
- [msprof 采集命令](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0011.html)：进程级 profiling 的采集参数；
- [msprof query/export](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/devaids/Profiling/atlasprofiling_16_0021.html)：查询和导出采集数据；
- [msprof 数据文件](https://www.hiascend.com/document/detail/zh/canncommercial/900/devaids/Profiling/atlasprofiling_16_0035.html)：各类 summary CSV 的字段和含义。
