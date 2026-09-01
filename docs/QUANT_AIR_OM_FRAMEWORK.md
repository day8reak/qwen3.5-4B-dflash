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
quant_dflash_recompute.om + deployment-manifest.json
        │
        │ C++17 / AscendCL
        ▼
加载一次 OM → ordinary/DFlash 多轮调用 → 生成 token → EOS/长度停止
```

第一版冻结的是静态完整前缀重算 ABI。它能够由 OM 完成完整的 logits/Top1 推理，并由 C++
循环连续生成 token；但它尚未把 `quant` 分支的 persistent rollback cache、GDR state bank 和
Draft KV cache 变成显式 OM 输入/输出。因此：

- “是否真的调用 OM 生成完整 token 序列”可以验证；
- “是否达到闭源增量推理框架时延”必须在真实设备测量，当前不能预先声称；
- 如果 OM 执行时间主要消耗在完整前缀重算，下一阶段应拆成 prefill/decode/verify/draft
  增量 OM，而不是继续微调 Python。

这里必须区分“当前分支实际产物”和“最终低时延拓扑”：

| 阶段 | 逻辑图 | 物理 OM 数 | 当前状态 |
| --- | --- | ---: | --- |
| 当前可验证基线 | Target 全前缀 + Draft proposal 整图 | 每个静态 `S` gear 1 个 | 已接入 AIR/ATC/C++ |
| 低时延目标 | `target_prefill_64`、`target_decode_1`、`target_verify_16`、`draft_16` | 至少 4 个；多 cache/shape gear 时更多 | 需要显式状态 ABI 与真机门禁 |

`verify` 不能与 ordinary `decode` 共用一个含糊的状态合同：前者一次计算最多 16 个 provisional
row，并只按接受数提交一个 GDN/conv state-bank 槽；后者固定提交单个 row。用户口头所说的
“prefill、decode、draft 三类”在物理部署中因此通常落为上述四个 OM 角色。

## 2. 冻结 ABI

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
modeling 中的 torch_npu.<op>
        │ 校验 npu::<op> dispatcher schema/alias
        │ 校验已有 Meta，缺失时注册精确 Fake（只声明 shape/dtype/alias）
        ▼
npu.<op>.default（FX 节点仍保留）
        │ receiver-private converter 或已校验的 TorchAir builtin converter
        ▼
对应 GE/AIR 节点（dynamo.pbtxt 必须可审计）
```

当前锁定的七个算子如下。`required` 表示当前完整前缀重算图必须实际出现，`optional` 表示仍做
schema/Meta 预检，但不能虚构一次图命中。

| 前端 FX target | Fake/Meta 输出合同 | 默认 GE type | converter | 当前图 |
| --- | --- | --- | --- | --- |
| `npu.npu_dynamic_quant.default` | INT8 输出与 input 同 shape；FP32 scale 为 `input.shape[:-1]` | `DynamicQuant` | TorchAir builtin | required |
| `npu.npu_quant_matmul.default` | broadcast batch + `[M,N]`，当前调用输出 FP16 | `QuantBatchMatmulV3` | TorchAir builtin | required |
| `npu.adn_rms_norm.default` | 输出 0 与 input 同 shape/dtype；输出 1 为 `[*input.shape[:-1],1]` FP32 | `RmsNorm` | 框架注册 | required |
| `npu.npu_chunk_gated_delta_rule.default` | output 为 value shape/query dtype；final state 为 initial-state shape/FP32 | `ChunkGatedDeltaRule` | 框架注册 | required |
| `qwen35_dflash.npu_cache_update.default` | 返回同 shape/dtype/device 的非 alias 更新值 | `CacheUpdate` | 框架注册 | required |
| `npu.adn_fused_infer_attention.default` | 按 layout 推导；当前 packed `BNSD` 路径保持 query shape/FP16 | `FusedInferAttentionScore` | 框架映射 | required |
| `npu.npu_scatter_nd_update_.default` | 返回同一个 `Tensor(a!)`，不能丢失写 alias | `ScatterNdUpdate` | TorchAir builtin | optional（仅 `forward1`） |

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
`adn_fused_infer_attention` 的当前重算 lowering 还要求
`all_seq_lengths_q == actual_seq_lengths_q`，不满足时在生成错误 AIR 前直接失败。
`allQLen` 是 `SymInt[]` 序列长度，两个 HIAI modeling 文件在 eager 和 AIR 路径都把它传给
`all_seq_lengths_q`。当前路线没有 PSE tensor，因此 `pse_shift` 保持 `None`；不能为了通过类型
检查把长度列表转换成 Tensor 后塞进 `pse_shift`。

Fake/Meta 不执行任何算子数值，正式 eager、AIR 和 OM 也不会执行 Fake。若 torch-npu 已有 Meta，
框架先运行 shape/dtype/alias 探针并复用；只有缺失时才调用 `torch.library.register_fake`。任何 schema
参数、kw-only 标志、返回个数或可变 alias 漂移都会提前失败。

`npu_quant_matmul` 的预检使用实际 W8A8 路径代表形状：`x1=[1,64,2560]`、
`x2=[2560,8192]`、`scale=[8192]`、`pertoken_scale=[64]`。torch-npu 要求
`pertoken_scale.shape[0] == x1.shape[-2]`（M 维）；不能把 batch 与 M 相乘后写成 `[B*M]`。

receiver 的原始 `npu.npu_cache_update_` 使用 `Tensor(a!) -> Tensor(a!)` 原地 ABI。PyTorch
AOTAutograd 不能 functionalize 带返回 alias 的非 ATen 算子，所以 AIR 专用 modeling 路径通过
无 alias 前端 `qwen35_dflash.npu_cache_update.default` 表达“旧 cache 输入 -> 新 cache 输出”，
再精确 lowering 为一个 `CacheUpdate` GE 节点。普通 eager 路径仍直接调用原始原地算子，不增加
clone 或额外 NPU launch；rollback 多行写在 AIR 路径中显式串接每一行的更新输出。设置
`keep_inference_input_mutations=True` 不能解除非 ATen 算子“返回值带 alias”的限制。

`CacheUpdate` 的 GE 原型不是按前端 Python 参数名解析，而是固定为：

```text
inputs: x, updates, targetBlock, offsetInBlock
outputs: x
attrs: none
```

因此 converter 使用 named inputs/output，并显式完成 `input -> x`、`target_block -> targetBlock`、
`offset_in_block -> offsetInBlock` 映射；不能再把四个前端参数交给 positional 自动解析。

七个预期 GE type 都在 `factory.json` 中显式锁定。builtin converter 和当前 fused-attention 精确
映射必须使用表中的 type；GDR 当前只允许上述 `ChunkGatedDeltaRule` v2 named ABI。该值不会
自动回退：指定 IR 未注册、converter 没有命中，或 `dynamo.pbtxt` 中 required type 数为 0，
导出都会失败，不会生成伪 PASS manifest。

### 2.2 TorchAir 标准算子精确补丁

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
PY

npu-smi info
atc --version
```

任何 import、设备或 ATC 失败都应先修环境；不要让导出流程自动降级到 CPU。

在加载权重前单独检查 GDR GE 原型：

```bash
"$MODEL_PYTHON" - <<'PY'
from qwen35_dflash.ascend310p.custom_op_export import (
    validate_gdr_ge_prototype_environment,
)

result = validate_gdr_ge_prototype_environment()
print(result)
assert result["status"] == "PASS"
assert result["abi"] == "effective-length-v2-named-inputs"
PY
```

导出器也会自动执行同一检查。若两个环境变量指向两套同名
`ChunkGatedDeltaRule`，会在加载 checkpoint 前失败并列出两个 `op_proto.h`。必须同时从
`ASCEND_CUSTOM_OPP_PATH` 和 `LD_LIBRARY_PATH` 移除旧 vendor 根；不要只依赖路径先后顺序，
也不要删除不属于当前用户的安装目录。

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
cp config/quant_air_om_factory.example.json "$AI_RUN_DIR/factory.json"
```

`factory.json` 示例：

```json
{
  "target_dir": "/ABSOLUTE/PATH/Qwen3.5-4B",
  "draft_dir": "/ABSOLUTE/PATH/Qwen3.5-4B-DFlash",
  "quant_config": "/ABSOLUTE/PATH/qwen3.5.yaml",
  "input_manifest": "/ABSOLUTE/PATH/quant-air-om-run/quant-input-manifest.json",
  "receiver_models_dir": "/ABSOLUTE/PATH/qwen35-runtime/models",
  "max_sequence_length": 64,
  "example_sequence_length": 2,
  "pad_token_id": 0,
  "dtype": "float16",
  "device": "npu:0",
  "npu_dynamic_quant_ge_op_type": "DynamicQuant",
  "npu_quant_matmul_ge_op_type": "QuantBatchMatmulV3",
  "adn_rms_norm_ge_op_type": "RmsNorm",
  "npu_chunk_gated_delta_rule_ge_op_type": "ChunkGatedDeltaRule",
  "npu_cache_update_ge_op_type": "CacheUpdate",
  "adn_fused_infer_attention_ge_op_type": "FusedInferAttentionScore",
  "npu_scatter_nd_update_ge_op_type": "ScatterNdUpdate",
  "name": "quant_dflash_recompute"
}
```

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

通过标准：所有 Python 测试和两个 CTest 都通过。框架专项测试会以 strict `torch.export`
确认七个前端 target 可保留，其中 cache/scatter 还会检查 writable alias；这仍然只验证
FakeTensor/图捕获元数据，不执行算子数值。fake ACL 测试会编译生产 `acl_executor.cpp`，但只证明
host 侧 buffer、调用顺序、scheduler 和 JSON 门禁。

### 5.2 量化 PyTorch 图探针

在导出前用真实权重执行一次图：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p probe-pytorch \
  --factory-config "$AI_RUN_DIR/factory.json" \
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

这个探针仍不是 OM 证据，但能在耗时导出之前发现 checkpoint、量化 topology、embedding、
Draft 或 receiver ABI 错误。

### 5.3 从拷贝源码目录采集完整 AIR 诊断

NPU 机器上即使只有直接拷贝的源码、没有 `.git`，也可以执行：

```bash
"$MODEL_PYTHON" framework/scripts/collect_air_debug.py \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --output-dir "$AI_RUN_DIR/reports"
```

采集器不调用 Git，也不复制 checkpoint、量化权重、AIR 或 OM。它会记录关键源码 SHA256 并附带
这些小型源码文件的快照，同时收集 Python/package/CANN/NPU 身份、七个前端算子的真实 schema
与 dispatch table、脱敏后的 factory 配置、完整 Dynamo 导出日志和已有的 JSON/log/pbtxt 诊断。
即使 AIR 导出失败，也会先在 `--output-dir` 生成 `air-debug-*.tar.gz`，随后返回原导出退出码；
把该压缩包回传即可。若日志过大，可增加 `--no-dynamo-logs`，但第一次失败建议保留默认完整日志。

## 6. 生成 AIR

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts/quant-dflash"
```

预期生成：

```text
$AI_RUN_DIR/artifacts/quant-dflash/
├── air-manifest.json
└── air/
    └── quant_dflash_recompute/
        ├── quant_dflash_recompute.air
        ├── dynamo.pbtxt
        └── TorchAir 生成的外置权重/辅助文件
```

导出器要求每个图恰好一个 `.air`，并把所有 payload 的大小和 SHA-256 写入
`air-manifest.json`。检查：

```bash
"$MODEL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["AI_RUN_DIR"]) / "artifacts" / "quant-dflash"
data = json.loads((root / "air-manifest.json").read_text())
assert data["status"] == "PASS"
assert data["schema_version"] == 3
gdr_proto = data["environment"]["gdr_ge_prototype"]
assert gdr_proto["status"] == "PASS"
assert gdr_proto["abi"] == "effective-length-v2-named-inputs"
assert len(data["graphs"]) == 1
graph = data["graphs"][0]
assert graph["name"] == "quant_dflash_recompute"
assert graph["input_names"] == ["input_ids", "attention_mask"]
assert graph["output_names"] == ["target_top1", "draft_top1"]
assert graph["metadata"]["quant_branch_base_revision"] == \
    "28f93e784a2beed87020a80bd93c8788754eab1c"
assert graph["metadata"]["gdr_effective_length_contract"] == \
    "INT16[B] call-local valid rows derived from attention_mask"
assert graph["metadata"]["target_quant_mode"] == "w8a8_dynamic"
audit = graph["custom_op_audit"]
assert len(audit) == 7
expected = {
    "npu.npu_dynamic_quant.default": ("DynamicQuant", 1, "torchair-builtin"),
    "npu.npu_quant_matmul.default": ("QuantBatchMatmulV3", 1, "torchair-builtin"),
    "npu.adn_rms_norm.default": ("RmsNorm", 1, "framework-registered-ge-ir"),
    "npu.npu_chunk_gated_delta_rule.default": (
        "ChunkGatedDeltaRule", 1, "framework-registered-ge-ir"
    ),
    "qwen35_dflash.npu_cache_update.default": (
        "CacheUpdate", 1, "framework-registered-ge-ir"
    ),
    "npu.adn_fused_infer_attention.default": (
        "FusedInferAttentionScore", 1, "framework-registered-ge-ir"
    ),
    "npu.npu_scatter_nd_update_.default": (
        "ScatterNdUpdate", 0, "torchair-builtin"
    ),
}
for item in audit:
    ge_type, minimum, policy = expected[item["torch_target"]]
    assert item["status"] == "PASS"
    assert item["ge_op_type"] == ge_type
    assert item["minimum_occurrences"] == minimum
    assert item["converter_policy"] == policy
    assert item["ge_node_occurrences"] >= minimum
    if policy == "torchair-builtin":
        assert item["converter_calls"] is None
    else:
        assert item["converter_calls"] >= minimum
print("AIR manifest gate: PASS")
PY
```

还可以直接检查 TorchAir 的可读 GE 图：

```bash
rg -n '(type|op): "(SoftplusV2|DynamicQuant|QuantBatchMatmulV3|RmsNorm|ChunkGatedDeltaRule|CacheUpdate|FusedInferAttentionScore|ScatterNdUpdate)"' \
  "$AI_RUN_DIR/artifacts/quant-dflash/air/quant_dflash_recompute/dynamo.pbtxt"
```

前六个 required type 至少应各命中一次；固定重算图不经过 `forward1` 时，`ScatterNdUpdate` 可以
不出现。这里检查的是 AIR 内部 GE type，`npu.*.default` 前端 FX 名称不会原样作为 GE type
出现。`air-manifest.json` 对框架 converter 同时记录调用数和 GE 节点数；TorchAir builtin 没有被
框架包装，所以其 `converter_calls` 为 `null`，只以 schema/Meta 探针和最终 GE 节点作为门禁。
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
  --air-manifest "$AI_RUN_DIR/artifacts/quant-dflash/air-manifest.json" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3
```

框架固定调用：

```text
atc --mode=0 --framework=1 \
    --model=<quant_dflash_recompute.air> \
    --output=<.../quant_dflash_recompute> \
    --soc_version=<精确 310P variant>
```

禁止使用模糊的 `Ascend310P`。成功后得到：

```text
$AI_RUN_DIR/artifacts/quant-dflash/
├── deployment-manifest.json
└── om/
    └── quant_dflash_recompute.om
```

编译器在调用 ATC 前重新核验 AIR 与所有外置 payload 的 hash；ATC 成功后记录 OM hash、ATC
版本、完整命令和日志。它还会重新校验并把 `custom_op_audit` 传入 deployment manifest；缺失
审计或 GE 节点数少于 converter 命中数时不会调用 ATC。退出码为 0 但 OM 缺失或为空也判定
失败。

也可以一次执行 AIR + OM：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-om \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts/quant-dflash" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3
```

## 8. 构建 C++ AscendCL runner

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-release" \
  --output "$AI_RUN_DIR/reports/cpp-build.json" \
  --ascendcl-root /ABSOLUTE/PATH/CANN
```

生产 runner 使用：

- `aclInit`、`aclrtSetDevice`、显式 context/stream；
- `aclmdlLoadFromFile`，进程内只加载一次 OM；
- `aclrtMallocHost` pinned host buffer；
- 持久化 device buffer 和 dataset；
- 每轮排队 H2D、`aclmdlExecuteAsync`、D2H；
- 每次 OM 调用只做一次 `aclrtSynchronizeStream`；
- token 调度、DFlash 接受/correction/bonus、EOS 都在 C++17 中完成。

生产二进制通常位于：

```text
$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner
```

不要把 build 目录或二进制提交进源码仓库。

## 9. 用 C++ 调用 OM 完整生成 token

复制并填写真实运行时身份：

```bash
cp config/quant_air_om_runner.example.json "$AI_RUN_DIR/runner.json"
```

`device_model` 必须写具体产品和 310P variant；`cann`、`driver`、`firmware` 必须来自当前
设备，不能保留 `REPLACE_*`。

运行：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest \
    "$AI_RUN_DIR/artifacts/quant-dflash/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --chat \
  --max-new-tokens 32 \
  --max-draft-tokens 15 \
  --device-id 0 \
  --output "$AI_RUN_DIR/reports/cpp-infer.json"
```

这条命令中 Python 只负责一次 tokenizer、启动 runner 和最后 detokenize。进入 runner 后：

1. C++ 校验 OM SHA-256；
2. C++ 加载 OM；
3. C++ 运行 ordinary greedy 3 次 warmup + 10 次测量；
4. C++ 运行 DFlash strict-greedy 3 次 warmup + 10 次测量；
5. 每轮生成都反复调用 OM，直到 EOS 或 `max_new_tokens`；
6. C++ 比较 ordinary 与 DFlash token IDs、EOS/stop reason；
7. 任一 token 不一致立即失败。

所以它不是“Python 算完 token、C++ 只读结果”，而是 C++ 的 token 循环实际调用 OM 完成
Target/Draft 推理。

## 10. 一键端到端

先构建 C++ runner，然后执行：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e-cpp \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/e2e/artifacts" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3 \
  --runner "$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner.json" \
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

## 11. 如何判定框架“能用”

### 11.1 功能 PASS

最终 `cpp-infer.json` 或 `e2e/reports/summary.json` 必须同时满足：

```python
assert report["status"] == "PASS"
assert report["cpu_fallback"] is False
assert report["ordinary_parity"]["status"] == "PASS"
assert report["ordinary_parity"]["token_id_mismatches"] == 0
assert report["ordinary_parity"]["eos_mismatches"] == 0
assert report["ordinary"]["warmup"] == 3
assert report["ordinary"]["repetitions"] == 10
assert report["dflash"]["warmup"] == 3
assert report["dflash"]["repetitions"] == 10
assert report["ordinary"]["stable_generated_token_ids"] == \
       report["dflash"]["stable_generated_token_ids"]
```

此外检查：

- 报告中的 OM SHA-256 与 `deployment-manifest.json` 一致；
- device/cann/driver/firmware 与 `npu-smi` 和当前软件栈一致；
- 10 次输出 token 和 stop reason 完全稳定；
- prompt token 数加最大输出数不超过静态 gear；
- 没有 CPU fallback、mock、fake ACL 或 simulation 标记。

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

## 12. 常见失败定位

| 失败 | 含义 | 处理 |
| --- | --- | --- |
| input manifest hash mismatch | 权重、量化文件或 wrapper 已变化 | 先确认变化是否预期；重新冻结，不要跳过校验 |
| 找不到 `export_model_wrapper_qwen3_5.py` | receiver 路径错误 | 修正 `receiver_models_dir` |
| QLinear coverage mismatch | 量化权重与 Target topology 不同 | 核对 YAML、checkpoint revision 和 quant artifact |
| `torch_npu`/TorchAir import 失败 | 环境不匹配 | 使用与 CANN/驱动匹配的声明环境 |
| `unsupported operator: npu.<op>.default` | 七算子预检未运行、实际代码不是本分支，或 receiver 新增了第八个算子 | 核对远端提交、`PYTHONPATH`；已覆盖清单见 2.1，不能用 modeling Tensor fallback 掩盖 |
| `schema drifted from the locked export contract` | torch-npu/receiver 算子签名与当前锁不一致 | 记录 dispatcher schema；按真实版本更新 schema、Fake、converter 和测试，不能跳过校验 |
| `Meta contract mismatch` / `lost input alias` | 上游 Meta 或本地 Fake 与真实 shape/dtype/原位语义不一致 | 停止导出，先以算子包实现和实机输出重新冻结合同 |
| `the pertoken_scale 1st dim value must be x1 m dim value` | 旧版框架的 QuantMatmul Meta 探针误用了 `[B*M]` scale；不是 NPU kernel 失败 | 更新本分支；确认 `x1=[1,M,K]` 时探针和模型都传 `pertoken_scale=[M]` |
| `Found a custom (non-ATen) operator whose output has alias annotations`，随后 `Original traceback` 指向 `npu_cache_update_` | 旧版 AIR 路径把 `Tensor(a!) -> Tensor(a!)` 直接交给 AOTAutograd；Fake 正确也无法 functionalize | 更新本分支；确认 FX target 为 `qwen35_dflash.npu_cache_update.default`，且 `dynamo.pbtxt` 仍包含 `CacheUpdate` |
| `ERR03005 GRAPH internal error`，`Original traceback` 指向 `qwen35_dflash.npu_cache_update.default`，Meta 单测通过 | 旧 converter 把前端 snake_case 参数按 positional 传入，但 GE `CacheUpdate` 原型要求 `x/updates/targetBlock/offsetInBlock -> x` | 更新本分支；确认 AIR manifest 中该算子的 `converter_mode` 为 `named-cache-update-x-v1` |
| `TorchAir IR contains 0 DynamicQuant nodes`，但 `dynamo.pbtxt` 明确含 `op: "DynamicQuant"` | 旧审计器只识别 `type:` 字段；AIR 和权重保存实际上已经完成 | 更新本分支；不要把 DynamicQuant 改为 optional，确认 manifest 中 `ge_node_occurrences >= 1` |
| `pse_shift` 期望 `Optional[Tensor]` 但收到 `[64]` / `immutable_list` | 旧版 modeling 在 export 路径把 `allQLen` 长度列表误接到了 PSE 输入，尚未进入 Fake/converter | 更新本分支；确认两个 modeling 文件均传 `all_seq_lengths_q=allQLen` 且不构造伪 PSE Tensor |
| `GE IR ... is not registered` | factory 中某个 `*_ge_op_type` 与目标 CANN/自定义包不一致 | 使用已正式注册且与算子实现一致的 GE type；不能用同名伪节点 |
| custom-op converter/GE-node count 为 0 | 算子被绕开、converter 未调用或 GE 图丢失节点 | 导出按 FAIL 处理，保留 `dynamo.pbtxt` 和完整 TorchAir 日志 |
| TorchAir graph break | 某个 Python/自定义 op 未被捕获 | 定位首个 graph break，补正式 converter；不要伪造 AIR |
| ATC unsupported op | TorchAir 图中存在 ATC 不支持节点 | 保留算子名和编译日志，决定分解或正式自定义算子 |
| generic `Ascend310P` rejected | SoC 身份不精确 | 从设备/ATC 支持列表填写真实 variant |
| OM input/output count mismatch | 导出 ABI 漂移 | 必须恢复 2 input/2 output INT64 合同或版本化新 ABI |
| C++ OM hash mismatch | OM 被替换或 manifest 错配 | 使用同一次 build 的 OM 和 deployment manifest |
| ordinary/DFlash token mismatch | 接受、correction、pad 或图语义错误 | 停止性能测试，定位首个 token 分叉 |
| 延迟明显慢于闭源 | 完整前缀重算成为主瓶颈 | profile 后进入增量 OM state ABI，不要只优化 Python |

## 13. 下一性能阶段

当前 C++ 已消除 Python token 热循环、重复 OM load、重复 host/device buffer 分配和多余 stream
同步。若真实 profile 显示 OM 计算主导，下一步应基于 `quant` 已有 rollback 语义拆分：

1. `target_prefill_64.om`：分块 prompt prefill；
2. `target_decode_1.om`：ordinary 单 token decode；
3. `target_verify_16.om`：anchor + 最多 15 proposals；
4. `draft_16.om`：增量 Draft KV；
5. 显式输入/输出 32 层 paged KV、24 层 GDR/conv state bank、8 层 feature 和 Draft KV；
6. C++ 负责同一 accepted count 下的原子 commit/abort；
7. 对每一个 state 分支做 ordinary token/EOS 零差异门禁。

这一步是新的状态 ABI，不能在没有真实 baseline 和 state-branch 测试时悄悄替换当前功能基线。

## 14. 官方接口依据

- [TorchAir `dynamo_export`](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0052.html)：接口参数、静态/动态图、单图约束和 AIR 大小约束；
- [TorchAir 自定义算子 converter](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00045.html)：自定义 ATen IR 需要注册 converter 后转换为 GE IR；
- [TorchAir AIR 产物与外置权重](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0019.html)：`export.air`、`dynamo.pbtxt` 和 `weight_*` 的关系；
- [ATC `--framework`](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/atc/atlasatcparam_16_0014.html)：TorchAir 标准 AIR 使用 `--framework=1`；
- [AscendCL `aclmdlExecuteAsync`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0299.html)：异步模型执行接口；
- [AscendCL `aclrtMemcpyAsync`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0106.html)：stream 上的异步输入输出复制；
- [AscendCL `aclmdlLoadFromFile`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0283.html)：从 OM 文件加载模型。
