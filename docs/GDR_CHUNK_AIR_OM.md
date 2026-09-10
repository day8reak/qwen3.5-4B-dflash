# Ascend 310P：从模型到 AIR、OM、C++ 运行和 msprof

按本文顺序完成环境准备、输入检查、模型导出、转换、C++ 执行和性能采集。
支持 batch=1、strict greedy、W8A8 Target＋FP16 Draft。

| OM | 物理输入 | 职责 |
|---|---:|---|
| `target_prefill.om` | 64 行，有效 1..64 | prompt 分块、末行 Top1、Target 特征和状态 |
| `target_decode.om` | 1 行 | 普通 greedy decode；已加载时也供 DFlash 关闭草稿后使用 |
| `target_verify.om` | 16 行，有效 1..16 | verify、Top1、接受判断、第二次 GDR 和 committed state |
| `draft.om` | 64 行特征＋16 行 block | 特征投影、Draft KV 追加、一次 1..15 token proposal |

下面配置导出四个 OM，便于普通/DFlash 对照。默认 paired 运行同时加载四个；
低显存 paired 模式分组测试，最多同时加载三个。单模式 DFlash 运行只加载
prefill、verify、draft 三个，普通运行只加载 prefill、decode 两个。
两种模式共用同一个 prefill OM。
verify 包含状态提交计算，没有独立 commit OM。

第一遍 GDR 保持双输出：core 用于后续 Target 计算，原始 FP32 state 作为
`verify_discard_t<layer>_recurrent` 输出到独立设备缓冲区。
第二遍 GDR 仍从本轮初始 state 计算接受前缀，只有它的结果进入缓存。
24 份 discard 输出各为 FP32 `[1,32,128,128]`，合计 48 MiB，
C++ 不将其拷回 CPU，也不在下一轮读取。OM 数量保持不变。

设备适配状态：代码和主机模拟测试已具备；真实 TorchAir/ATC、AscendCL、token 精度和性能
必须在目标设备完成验证，不能把模拟测试当作设备结果。

## 1. 准备源码和目录

需要 Linux、可访问的 Ascend 310P、与驱动/固件匹配的 CANN 和 NPU Python 环境。
先把下面的绝对路径改成实际路径。`AI_RUN_DIR` 使用源码和权重目录之外的全新目录；
使用模型 workspace 时，由 `ws session start` 分配该目录，不要覆盖已有 session 的路径。

```bash
export REPO_ROOT=/absolute/path/qwen3.5-4B-dflash
export AI_RUN_DIR=/absolute/path/qwen35-run
export MODEL_PYTHON=/absolute/path/npu-env/bin/python
export CANN_ROOT=/absolute/path/ascend-toolkit
export TARGET_DIR=/absolute/path/Qwen3.5-4B
export DRAFT_DIR=/absolute/path/Qwen3.5-4B-DFlash
export RECEIVER_ROOT=/absolute/path/qwen35-receiver
export RECEIVER_MODELS_DIR="$RECEIVER_ROOT/models"

# REPO_ROOT 必须是尚未创建的目录。
git clone --single-branch --branch feature/gdr-chunk-verify \
  https://github.com/day8reak/qwen3.5-4B-dflash.git "$REPO_ROOT"
mkdir -p "$AI_RUN_DIR/log" "$AI_RUN_DIR/reports" "$AI_RUN_DIR/cache" "$AI_RUN_DIR/tmp"
cd "$REPO_ROOT"

source "$CANN_ROOT/set_env.sh"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$AI_RUN_DIR/tmp"
export HF_HOME="$AI_RUN_DIR/cache/huggingface"
export TORCH_HOME="$AI_RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$AI_RUN_DIR/cache"
export PYTHONPATH="$REPO_ROOT/framework/python:$REPO_ROOT:$RECEIVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

权重、AIR、OM、编译目录、日志和报告均放在源码仓库外。后续命令在同一个 Bash 会话执行，
使用这里定义的变量。源码已经位于 `REPO_ROOT` 时，从 `mkdir` 开始执行，无需再次 clone。

## 2. 安装依赖并检查 NPU

`MODEL_PYTHON` 使用 Python 3.10，并已安装与 CANN 配套的 PyTorch 和 `torch_npu`。
CANN 驱动、固件、PyTorch/NPU 扩展及设备自定义算子包需要由设备软件栈提供；本仓库不包含
这些安装包。不要用普通 CPU PyTorch 覆盖 NPU 环境中的 PyTorch。

安装模型侧依赖：

```bash
"$MODEL_PYTHON" -m pip install numpy PyYAML safetensors huggingface-hub "transformers==5.14.1"

npu-smi info | tee "$AI_RUN_DIR/log/npu-smi.txt"
"$MODEL_PYTHON" -B - <<'PY'
import torch
import torch_npu
assert torch.npu.is_available(), "没有可用 NPU"
torch.npu.set_device("npu:0")
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("device:", torch.npu.get_device_name(0))
PY
```

按自定义算子包的安装说明完成注册后，检查 Target 所需接口：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import torch_npu
required = (
    "npu_chunk_gated_delta_rule", "adn_fused_infer_attention",
    "adn_rms_norm", "npu_dynamic_quant", "npu_quant_matmul",
)
missing = [name for name in required if not callable(getattr(torch_npu, name, None))]
assert not missing, f"缺少 NPU 算子: {missing}"
print("Target operator symbols: PASS")
PY
```

`npu_chunk_gated_delta_rule` 必须接受 `effective_length: INT16[B]`，含义是本次调用的有效行数。
接口形式为：

```text
npu_chunk_gated_delta_rule(query, key, value, g, beta, effective_length,
                         chunk_size=64, initial_state=None,
                         output_final_state=False, use_qk_l2norm_in_kernel=False)
```

这里检查增量 AIR 图实际使用的五类接口。导出器会在加载权重前进一步检查 dispatcher schema、
Fake/Meta 和 GE converter；Fake/Meta 只描述输出 shape/dtype，不执行算子或提供 CPU 数值替代。
符号检查通过后，仍需通过后面的真实模型执行验证 shape、dtype 和数值。

自定义算子包的环境脚本需要正确设置 `ASCEND_CUSTOM_OPP_PATH` 和对应动态库路径。
导出器检查已配置的 GDR/attention GE prototype，拒绝重复的 GDR 注册、不含
`effective_length` 的 GDR ABI，以及缺少 310P attention kernel 的已配置包。

## 3. 准备模型和外部加载器

需要完整的 Target checkpoint、tokenizer，以及官方 Draft checkpoint。已有匹配文件时，直接
设置好 `TARGET_DIR` 和 `DRAFT_DIR`；需要下载时，以下命令从 `SOURCE_LOCK.json` 读取锁定版本：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
lock = json.loads((Path(os.environ["REPO_ROOT"]) / "SOURCE_LOCK.json").read_text())
for section, directory in (("target_checkpoint", "TARGET_DIR"), ("dflash_checkpoint", "DRAFT_DIR")):
    item = lock[section]
    snapshot_download(repo_id=item["repository"], revision=item["revision"],
                      local_dir=os.environ[directory])
PY
```

Draft 文件保留 checkpoint 中的 BF16 数据，运行时加载为 FP16；不要预先改写 checkpoint。
加载器会检查 Draft 的配置、6 层/69 tensor、shape、dtype 和文件 hash。

还需要外部文件 `$RECEIVER_MODELS_DIR/export_model_wrapper_qwen3_5.py` 及其依赖。
该文件由 Qwen3.5 NPU receiver 软件包提供，本仓库不附带。它必须导出
`Qwen3_5ForCausalLMWrapper` 类和模块级 `Qwen3_5ForCausalLM`，负责权重加载及设备初始化。
将完整 receiver 软件包放到 `RECEIVER_ROOT`，然后建立本次运行专用的模块搜索配置：

```bash
mkdir -p "$AI_RUN_DIR/python-bootstrap"
cat > "$AI_RUN_DIR/python-bootstrap/sitecustomize.py" <<'PY'
import os
import models
receiver_models = os.environ.get("RECEIVER_MODELS_DIR")
if receiver_models and receiver_models not in models.__path__:
    models.__path__.append(receiver_models)
PY
export PYTHONPATH="$AI_RUN_DIR/python-bootstrap:$REPO_ROOT/framework/python:$REPO_ROOT:$RECEIVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$MODEL_PYTHON" -B - <<'PY'
import importlib.util, os
from pathlib import Path
expected = Path(os.environ["RECEIVER_MODELS_DIR"]) / "export_model_wrapper_qwen3_5.py"
assert expected.is_file(), f"缺少外部加载器: {expected}"
spec = importlib.util.find_spec("models.export_model_wrapper_qwen3_5")
assert spec is not None and Path(spec.origin).resolve() == expected.resolve()
print("receiver wrapper:", spec.origin)
PY
```

配置放在运行目录中，不修改模型源码。模型文件优先从本仓库查找，外部目录提供缺少的加载器
及辅助模块。若 wrapper 的依赖无法导入，补齐 receiver 软件包和对应 Python 依赖后再执行。

## 4. 准备 W8A8 输入

需要以下三个量化输入；仓库消费这些文件，不提供 W8A8 权重生成工具：

| 输入 | 内容 |
|---|---|
| `QUANT_LINEAR_DIR` | 含 `data*.safetensors` 的 Target W8A8 Linear 权重目录 |
| `QUANT_EMBED_WEIGHT` | Target embedding 的 INT8 raw binary |
| `QUANT_EMBED_SCALE` | 每词表行的 FP32 scale raw binary |

量化数据必须与 `TARGET_DIR` 的结构和权重匹配。设置路径并生成 YAML：

```bash
export QUANT_LINEAR_DIR=/absolute/path/qwen35-w8a8/linear
export QUANT_EMBED_WEIGHT=/absolute/path/qwen35-w8a8/embedding_weight.bin
export QUANT_EMBED_SCALE=/absolute/path/qwen35-w8a8/embedding_scale.bin
export QUANT_CONFIG="$AI_RUN_DIR/qwen35-w8a8.yaml"

"$MODEL_PYTHON" -B - <<'PY'
import os, yaml
from pathlib import Path
values = {
    "quanted_pth": os.environ["QUANT_LINEAR_DIR"],
    "embedding_weight_path": os.environ["QUANT_EMBED_WEIGHT"],
    "embedding_scale_path": os.environ["QUANT_EMBED_SCALE"],
}
Path(os.environ["QUANT_CONFIG"]).write_text(yaml.safe_dump(values, sort_keys=False))
PY
```

YAML 中就是这三个绝对路径字段。Target 的全部 Linear（包括 LM head）使用 W8A8，
输入 embedding 使用 INT8 weight 和 FP32 scale。Draft embedding、Draft 主体和
Draft LM head 使用 FP16。Target 图必须使用 `dflash_execution_model.lm_head`；
bridge 的公开 `get_output_embeddings()` 保留的是供 Draft 使用的 FP16 checkpoint head。

## 5. 检查导出和 C++ 工具链

安装与当前 CANN/PyTorch 配套的 TorchAir，准备 ATC、CMake、C++17 编译器及 AscendCL。
`CANN_ROOT` 应包含 `include/acl/acl.h` 和 AscendCL 库。指定真实 ATC 路径及精确 SoC：

```bash
export ATC_BIN=/absolute/path/atc
export SOC_VERSION=Ascend310P3
export MAX_SEQUENCE_LENGTH=512
export MAX_NEW_TOKENS=32

"$ATC_BIN" --version | tee "$AI_RUN_DIR/log/atc-version.txt"
cmake --version
c++ --version
"$MODEL_PYTHON" -B - <<'PY'
import torchair
print("torchair:", getattr(torchair, "__version__", "unknown"))
PY
```

`Ascend310P3` 是示例，必须换成设备支持的精确型号，不能填写泛称 `Ascend310P`。
`MAX_SEQUENCE_LENGTH=C` 是逻辑 KV 容量，须为 64 的倍数，范围 64..32704。
物理 KV 额外保留 64 行 scratch；请求须满足 `prompt_tokens + max_new_tokens <= C`。

## 6. 固定 prompt 并执行 NPU 正确性检查

写入要测试的文本，生成一份与 C++ 文本入口相同的 chat token IDs：

```bash
cat > "$AI_RUN_DIR/prompt.txt" <<'TEXT'
请用一句话解释为什么天空是蓝色的。
TEXT

"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
from transformers import AutoTokenizer
from qwen35_dflash.ascend310p.generation import tokenize_prompt
run = Path(os.environ["AI_RUN_DIR"])
prompt = (run / "prompt.txt").read_text().strip()
tokenizer = AutoTokenizer.from_pretrained(os.environ["TARGET_DIR"], local_files_only=True)
ids = tokenize_prompt(tokenizer, prompt, chat=True)
assert len(ids) + int(os.environ["MAX_NEW_TOKENS"]) <= int(os.environ["MAX_SEQUENCE_LENGTH"])
(run / "prompt-ids.json").write_text(json.dumps(ids))
(run / "prompt-ids.csv").write_text(",".join(map(str, ids)) + "\n")
print("prompt tokens:", len(ids))
PY

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --config "$QUANT_CONFIG" --quant_mode enable \
  --kv-cache-max-len "$MAX_SEQUENCE_LENGTH" --device npu:0 \
  --prompt-json "$AI_RUN_DIR/prompt-ids.json" \
  --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
  --execution-mode validate --report "$AI_RUN_DIR/reports/npu-validate.json"
```

报告要求 `correctness_gate.status=PASS`、`strict_greedy_exact_match=true`，且
`operator_fallback_enabled=false`。该命令比较同一个 W8A8 Target 的 ordinary 和 DFlash；
后面还会把 ordinary token 与 OM 结果比较。anchor 立即为 EOS 时没有 Draft/verify 轮次，
应选择一条能够实际生成 token 的固定 prompt。

## 7. 锁定输入并生成导出配置

Target、Draft、量化输入及 receiver 路径使用普通目录和文件，不使用 symlink。
以下命令哈希完整输入；大权重读取可能需要较长时间：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/framework/scripts/lock_quant_inputs.py" \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --quant-config "$QUANT_CONFIG" --receiver-models-dir "$RECEIVER_MODELS_DIR" \
  --output "$AI_RUN_DIR/quant-input-manifest.json"

"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
run = Path(os.environ["AI_RUN_DIR"])
config = {
    "target_dir": os.environ["TARGET_DIR"],
    "draft_dir": os.environ["DRAFT_DIR"],
    "quant_config": os.environ["QUANT_CONFIG"],
    "input_manifest": str(run / "quant-input-manifest.json"),
    "receiver_models_dir": os.environ["RECEIVER_MODELS_DIR"],
    "max_sequence_length": int(os.environ["MAX_SEQUENCE_LENGTH"]),
    "include_ordinary_decode": True,
    "dtype": "float16", "device": "npu:0", "adn_rms_norm_ge_op_type": "RmsNorm",
}
(run / "factory.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

导出前会再次检查外部输入和 `SOURCE_LOCK.json`。冻结输入后不要增加、删除或修改文件。
`RmsNorm` 必须是目标环境注册的 GE type；自定义包注册的是 `AdnRmsNorm` 时，填写
`"adn_rms_norm_ge_op_type": "AdnRmsNorm"`。两者的 GE 输入名分别为 `x` 和 `self`，
导出器按所选类型处理，不接受任意 GE 名称。

## 8. 导出 AIR

从运行目录启动转换，保存完整日志：

```bash
cd "$AI_RUN_DIR"
set -o pipefail
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts" \
  2>&1 | tee "$AI_RUN_DIR/log/export-air.log"
```

成功后得到 `artifacts/air-manifest.json`，以及 `artifacts/air/<graph>/` 下的 AIR、
`dynamo.pbtxt` 和外置权重。四个 graph 名称与本文开头的 OM 名称一致。
manifest 保存有序输入/输出的 dtype、shape、文件 hash、算子预检和每张图的节点审计。
导出器按原始输入张量的存储身份确定 GE `Data.index` 和 Data 节点顺序，并记录
`runtime_input_abi` 审计。不能仅靠 Python 参数名或 manifest 的 `input_names` 控制
TorchDynamo 的捕获顺序；形状相同的多个 KV 状态也必须按张量身份区分。
Draft 的公开输入顺序固定为 `features, start_position, valid_rows, anchor, proposal_count, ...KV states`。
每个图必须只完成一次输入规范化，并保持公开输入的静态 shape；未知的运行时输入或
捕获中丢失的公开输入会使导出失败。
三个 Target 图均检查 `RmsNorm/AdnRmsNorm`、`DynamicQuant`、
`QuantBatchMatmulV4444`、`ChunkGatedDeltaRule`、`AdnFusedInferAttention`，
以及 `SoftplusV2`。Draft 使用 Tensor 算子，不要求出现 Target 自定义节点。
量化 matmul 在 AIR 捕获时使用专用前端，保留 FP32 weight/per-token scale 和 FP16 输出；
普通 NPU 推理仍调用同一套 receiver 量化接口。
每张 Target 图的 QuantBatchMatmulV4444 节点数不能少于加载器记录的
QLinear 数量；标准 4B Target 为 249，包含词表输出 head。该检查用于发现启用了量化前端、
但部分量化模块没有进入导出图的情况。
保留整个 AIR 目录，不要只复制 `.air` 文件。

卷积历史窗口通过切片加 `stack` 导出，保持每个有效前缀的状态。
verify 接受长度使用 INT32 `Cumsum → Equal → ReduceSum`，最后输出 INT64 接受数；
短块的 padding 不参与接受判断，第二次 GDR 提交 `accepted_count+1` 行。
缓存写入索引与 Draft 的 KV head 复制使用静态 `repeat/Tile`。
Target attention 使用 `all_seq_lengths_q=[C+64]`、
`actual_seq_lengths_q=[当前物理行数]` 和 `actual_seq_lengths_kv=[C+64]`；
这三个前端长度列表分别映射为 INT64 GE 输入。运行时因果 mask 同时排除
`start_position+valid_rows` 以后的缓存位置，`pse_shift` 留空。
这些路径不调用 `unfold`、`index_copy`、`amin`、`min(dim=...)` 或 `cumprod`。
若日志以 `ERR03007 GRAPH feature not supported` 结束，
查看完整日志中第一条 `NotImplementedError` 或 converter 异常及其对应的 `Original traceback`，
不要只截取末尾的 FX 图代码。FakeTensor 检查通过不代表所有标准算子都能转为 GE。

导出失败后，使用新的空 bundle 目录重试，例如将 `--bundle-dir` 改为
`"$AI_RUN_DIR/artifacts-custom-ops"`；后续 `--air-manifest` 和
`--deployment-manifest` 的路径也须指向该目录。导出器不会覆盖非空目录。
如果 ATC 报 `pse_shift DT_INT64`，需要更新源码并重新导出 AIR；只重新编译已有
AIR 无法改变错误的输入映射。编译器会提前拒绝使用整数 PSE 策略的增量 bundle。

含 GDR 的图在交给 GE 保存前会生成 `air/<图名>/gdr-output-dtypes.json`，
逐节点核对两个物理输出：`core_attn` 为 `DT_FLOAT16`，
`last_recurrent_state` 为 `DT_FLOAT`（FP32）。Verify 第二次 GDR 的 `core_attn`
即使没有被后续节点使用，也必须满足这个接口。类型不符会停止导出并保留这份报告。
同一份检查结果写入 AIR manifest 的 `graphs[].runtime_input_abi.gdr_output_dtypes`。
该检查的范围是 TorchAir 交给 GE 的描述，不代表 ATC 类型推导后的结果。

verify 还生成 `air/target_verify/verify-discard-outputs.json`，
逐项核对最后 24 个输出连接到第一遍 GDR 的第二个输出，并保持
`output_final_state=True`。允许透明的 Identity 输出节点，不接受 Cast、算术替代
或第二遍 commit state。检查失败会停止导出，报告保留在该目录。
第一遍和第二遍的 GDR 参数与计算公式保持一致，仅有效长度分别为
`valid_rows` 和 `accepted_count+1`；两遍都读取本轮开始时的 initial state。

## 9. 将 AIR 转为 OM

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$AI_RUN_DIR/artifacts/air-manifest.json" \
  --atc "$ATC_BIN" --soc-version "$SOC_VERSION" \
  2>&1 | tee "$AI_RUN_DIR/log/compile-om.log"
```

编译流程使用 `atc --mode=0 --framework=1`，验证 AIR payload 后逐图编译。
增量套件自动添加 `--precision_mode=must_keep_origin_dtype`，保留图中的 FP32
归一化、RoPE、Softmax、attention matmul 和 GDR 累加。不能因为权重为 FP16，
就把这些中间计算也降为 FP16。也支持显式传入 `--precision_mode_v2=origin`；
编译器拒绝同时指定两种精度参数或使用降精度模式。
启动 ATC 前还会检查整组图的 `runtime_input_abi`：要求 `status=PASS`，实际 Data 绑定
与公开输入的顺序、dtype、静态 shape 一致。缺少或不通过这项审计时，须重新导出到空的
bundle 目录；只编辑 manifest 无法修正 AIR 中的输入顺序。

编译失败时，终端异常会附带 ATC 的首个结构化错误摘要，完整日志保存在
`log/dflash-atc/<图名>.log`。如果出现
`ChunkGatedDeltaRule ... DT_FLOAT of output [core_attn] is not supported`，
说明 ATC 看到的 `core_attn` 为 FP32，而所列实现要求 FP16。
同一条错误中其他 op store 的 `not found` 不能单独作为 kernel 缺失的依据。
先查看对应的 `gdr-output-dtypes.json`：若交给 GE 前已经为 FP32，检查导出描述；
若该检查通过，继续核对保存后的 AIR 以及接收端算子包的 InferShape/InferDataType
实现和 ATC 变换。两个输出具有不同类型，不能一起改成 FP16，也不能仅凭
`REG_OP` 使用 `TensorType::ALL()` 就认定实际类型推导正确或错误。

成功后得到：

```text
artifacts/
  air-manifest.json
  deployment-manifest.json
  air/...
  om/
    target_prefill.om
    target_decode.om
    target_verify.om
    draft.om
```

`deployment-manifest.json` 保存每个 OM 的 SHA-256、大小、ATC 参数及 tensor ABI。
检查 graph 数量与产物非空：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["AI_RUN_DIR"]) / "artifacts"
report = json.loads((root / "deployment-manifest.json").read_text())
assert report["status"] == "PASS"
assert report["compiler"]["precision_policy"] == "preserve_graph_dtypes"
assert {g["name"] for g in report["graphs"]} == {"target_prefill", "target_decode", "target_verify", "draft"}
for graph in report["graphs"]:
    om = root / graph["om"]["path"]
    assert om.is_file() and om.stat().st_size > 0
    audit = graph["runtime_input_abi"]
    assert audit["status"] == "PASS" and audit["calls"] == 1
    assert [b["logical_name"] for b in audit["bindings"]] == graph["input_names"]
    if graph["name"] == "target_verify":
        assert audit["verify_discard_outputs"]["status"] == "PASS"
        discard = graph["metadata"]["incremental_contract"]["verify_discard_states"]
        assert len(discard) == 24
        assert graph["metadata"]["tensor_abi"]["outputs"][-24:] == discard
        assert all(t["dtype"] == "float32" and t["shape"] == [1,32,128,128] for t in discard)
print("OM files: PASS")
PY
```

## 10. 编译 C++ AscendCL runner

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp" \
  --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build.json"

export CPP_RUNNER="$AI_RUN_DIR/build/cpp/qwen35_dflash_acl_runner"
"$CPP_RUNNER" --help
```

需要生成 `qwen35_dflash_acl_runner`；名字带 `_fake` 的程序只用于主机测试。
构建日志在 `$AI_RUN_DIR/log/dflash-cpp-build/`，编译身份及 runner hash 在 `cpp-build.json`。
C++ 在每组测试前加载所需模型，持久保存 device buffer，循环使用 AscendCL 执行 OM。
加载 paired/DFlash 模式时，日志应包含
`verify_discard_buffer_bytes=50331648`，表示第一遍 GDR 的 24 份独立 FP32
输出缓冲区；普通模式该值为 0。discard state 不分配 host buffer、不做 D2H，
不会进入缓存交换或提交逻辑。
更新 C++ 源码后再次执行 `build-cpp`，使用新的 `--build-dir` 和 `--output` 路径，
并将 `CPP_RUNNER` 指向本次生成的可执行文件。构建器不覆盖非空目录或已有报告。

## 11. 运行 C++ 普通/DFlash 对照

从设备信息和软件包安装记录填写运行身份：

```bash
export DEVICE_MODEL='填写具体310P产品和型号'
export CANN_VERSION='填写CANN版本'
export DRIVER_VERSION='填写驱动版本'
export FIRMWARE_VERSION='填写固件版本'

"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
config = {
    "device_model": os.environ["DEVICE_MODEL"], "cann": os.environ["CANN_VERSION"],
    "driver": os.environ["DRIVER_VERSION"], "firmware": os.environ["FIRMWARE_VERSION"],
    "runtime": "AscendCL C++", "pad_token_id": 0,
}
(Path(os.environ["AI_RUN_DIR"]) / "runner.json").write_text(json.dumps(config, indent=2) + "\n")
PY

"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$AI_RUN_DIR/artifacts/deployment-manifest.json" \
  --runner "$CPP_RUNNER" --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir "$TARGET_DIR" --prompt "$(cat "$AI_RUN_DIR/prompt.txt")" --chat \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens 15 --device-id 0 \
  --eos-token-id 248044 \
  --output "$AI_RUN_DIR/reports/cpp-paired.json"
```

这里 Python 负责 tokenizer 和文本解码；生成循环、OM 执行、接受判断复核、EOS 处理在 C++。
同一进程交错运行普通/DFlash，各 3 次 warmup＋10 次测量。
报告的 `output.text` 是生成文本，`ordinary`、`dflash` 保存完整 token 和时延分布。
`--eos-token-id` 与第 6 步 NPU 报告的 `request.eos_token_ids` 保持一致；多个 EOS 可以
重复传入该参数。不传时使用 tokenizer 的 EOS，它可能与 Draft checkpoint 的 EOS 不同。

显存紧张时使用 `--low-memory`：

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$AI_RUN_DIR/artifacts/deployment-manifest.json" \
  --runner "$CPP_RUNNER" --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir "$TARGET_DIR" --prompt "$(cat "$AI_RUN_DIR/prompt.txt")" --chat \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens 15 --device-id 0 \
  --eos-token-id 248044 --low-memory \
  --output "$AI_RUN_DIR/reports/cpp-paired-low-memory.json"
```

普通模式先用 prefill/decode 完成 3 次预热、10 次测量；随后卸载这两张 OM，
释放对应缓冲区，再加载 prefill/draft/verify 完成 DFlash 的 3+10。
切换保留同一个 ACL runtime、context 和 stream，生成循环内不加载或卸载模型。
报告仍检查两种模式的 token、EOS 和停止原因，`protocol.low_memory=true`、
`max_resident_models=3`；`protocol.order` 明确记录分组顺序。
加载时间计入 `startup_ms.acl_and_model_load`，组间卸载时间单独记录在
`startup_ms.mode_switch_unload`，均不计入模型循环时延。
分组测量的设备温度、频率和其他任务负载可能与交错测量不同，比较性能时保留该协议区别。

DFlash 组不加载 decode OM；零接受关闭 Draft 后，使用 verify 的 `valid_rows=1`
继续生成。这会影响低接受率请求的时延，应查看实际 `stage_ms`。
`run-e2e-cpp` 同样支持此参数；直接 C++ 使用 `--model-kind chunk --mode paired --low-memory`。
单模式和 msprof 已按模式选择所需 OM，不接收此配对测量参数。
切换低显存模式不改变 AIR/OM 或 tensor ABI，已有匹配的四图 bundle 可以直接使用。

所有 chunk 模式默认复用串行执行的工作内存：查询各 OM 的 `work_bytes` 后，
分配一块大小为最大值的设备缓冲区，由所有已加载 OM 共用，权重仍各自独立。
该方式使用 AscendCL 的
[aclmdlLoadFromFileWithMem](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_03_0092.html)，
每次执行都在同一 stream 同步后才调用下一张图。全部模型卸载后才释放共享工作内存。
日志 `workspace policy=shared_serial` 给出 `shared_bytes`、`separate_sum_bytes`
及两者差额 `saved_work_bytes`；这是工作内存申请量的差额，实际峰值仍需设备测量。
查询不可用时记录 `policy=per_model`，按每张 OM 独立管理工作内存。

C++ 会在执行推理前逐项比较 OM 描述与 chunk plan。若失败，Python 异常会附带第一项
差异，例如 `graph=draft input[1] expected={...} actual={...}`；其中显示张量名、dtype、
字节数、rank 和 shape。完整日志还包含该 OM 的全部输入/输出描述：

```bash
cat "$AI_RUN_DIR/log/cpp-paired-cpp-runner.log"
```

若 `start_position INT64[1]` 对应到实际的 `valid_rows INT16[1]`，说明输入顺序不一致，
应使用带 `runtime_input_abi` 审计的源码重新导出 AIR、转换 OM，再运行 C++。
不要更改控制张量 dtype 或取消校验来绕过错误。新 bundle 路径要同时用于
`compile-om --air-manifest` 和 `infer-cpp --deployment-manifest`；重试报告使用新的
`--output` 文件名，例如 `cpp-paired-io.json`，避免覆盖失败日志。

## 12. 检查 token 一致性和时延范围

将 NPU ordinary、NPU DFlash、OM ordinary 和 OM DFlash 四者作精确比较：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ["AI_RUN_DIR"]) / "reports"
native = json.loads((root / "npu-validate.json").read_text())
om = json.loads((root / "cpp-paired.json").read_text())
assert native["correctness_gate"]["status"] == "PASS"
assert native["operator_fallback_enabled"] is False
assert om["status"] == "PASS" and om["cpu_fallback"] is False
assert om["ordinary_parity"]["token_id_mismatches"] == 0
assert om["ordinary_parity"]["eos_mismatches"] == 0
ordinary = native["ordinary"]
assert native["dflash"]["prompt_token_ids"] == om["prompt_token_ids"]
assert set(native["request"]["eos_token_ids"]) == set(om["eos_token_ids"])
assert ordinary["generated_token_ids"] == native["dflash"]["generated_token_ids"]
assert ordinary["generated_token_ids"] == om["ordinary"]["stable_generated_token_ids"]
assert ordinary["generated_token_ids"] == om["dflash"]["stable_generated_token_ids"]
assert ordinary["stop_reason"] == om["ordinary"]["stable_stop_reason"] == om["dflash"]["stable_stop_reason"]
print("Native ordinary/DFlash / OM ordinary/DFlash tokens: PASS")
print(om["output"]["text"])
PY
```

普通/DFlash 两个 OM 路径一致，仍可能共有导出误差，所以不能省略 NPU ordinary 对照。
继续覆盖长 prompt、跨 64 行边界、EOS、零/部分/全接受和重复请求。

要比较每轮的草稿、验证结果和实际输出，启用逐轮记录：

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$AI_RUN_DIR/artifacts/deployment-manifest.json" \
  --runner "$CPP_RUNNER" --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir "$TARGET_DIR" --prompt "$(cat "$AI_RUN_DIR/prompt.txt")" --chat \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens 15 --device-id 0 \
  --eos-token-id 248044 --trace-rounds \
  --output "$AI_RUN_DIR/reports/cpp-rounds.json"

"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p.compare_rounds \
  --native "$AI_RUN_DIR/reports/npu-validate.json" \
  --cpp "$AI_RUN_DIR/reports/cpp-rounds.json" \
  --output "$AI_RUN_DIR/reports/round-comparison.json"
```

`dflash.measurements[i].rounds` 包含 prefill 和之后每轮的
`proposed_token_ids`、`target_token_ids`、`accepted_draft_token_ids`、
`emitted_token_ids`、`fallback_token_id`，并记录所用 `stage`。
`committed_prefix_length` 是本轮开始时的 prompt＋已输出 token 长度，包含当前 anchor。
verify 记录只保留有效行，不包含物理图的 padding。

比较工具检查所有测量轮次，按相同的已提交 token 前缀对齐；
`first_difference` 给出首个可比较前缀的具体差异，
`native_only_prefix_lengths/cpp_only_prefix_lengths` 表示两条执行路径的分轮边界不同。
最终 token 相同不能证明每轮相同。缺少逐轮数据会显示 `NOT_AVAILABLE`，不会判为通过；
EOS 策略、最终 token 或逐轮记录没有全部匹配时，比较命令返回 1 并保留报告。

Draft OM 的物理 block 为 16 行。`proposal_count INT16[1]` 指定实际草稿数
K=`min(max_draft_tokens, 剩余输出预算, 15)`，所有层的注意力都只允许
anchor＋K 个 mask 作为有效 block key，包括非因果层。输出只读取前 K 项。
该输入由 C++ 自动填写，推理命令无须增加参数。AIR、OM 和 C++ runner 使用同一
`qwen35-dflash-chunk-v3` ABI；接口不一致时应从空 bundle 目录导出、编译并构建 runner。
逐轮差异应按相同已提交 token 前缀对齐，再检查 Target 特征和 Draft 的输入、计算与状态。
工具比较 token 记录，不代表中间张量已逐项对齐。

`--trace-rounds` 用于诊断，会增加主机记录开销。性能基线使用不带该参数的命令。
`latency_ms.model_total` 排除模型加载和 tokenizer；请求清零单独记录在
`latency_ms.request_reset`，比较包含请求初始化的时延时需要加回。
报告的 `stage_ms` 按图保留每次同步 OM 调用时间，包含必要的输入/输出复制。具体算子时间
使用下一步的 msprof。不要把不同计时范围、精度、prompt 或输出长度的结果直接计算加速比。

## 13. 普通模式：只采一次 prefill 和一次 decode

先生成加载计划，再由 wrapper 启动 C++。需要支持动态 PID 采集的 msprof：

```bash
export MSPROF_BIN=/absolute/path/msprof
"$MSPROF_BIN" --version
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p prepare-chunk-plan \
  --deployment-manifest "$AI_RUN_DIR/artifacts/deployment-manifest.json" \
  --mode ordinary --output "$AI_RUN_DIR/ordinary-plan.txt"
read -r ORDINARY_PLAN_SHA _ < <(sha256sum "$AI_RUN_DIR/ordinary-plan.txt")
read -r PROMPT_TOKEN_IDS < "$AI_RUN_DIR/prompt-ids.csv"

"$REPO_ROOT/tools/run_msprof.sh" \
  --label om-ordinary-all --output-dir "$AI_RUN_DIR/msprof/ordinary-all" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend cpp --profile-mode ordinary \
  --profile-stage all --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$CPP_RUNNER" --model-kind chunk \
    --model "$AI_RUN_DIR/ordinary-plan.txt" --model-sha256 "$ORDINARY_PLAN_SHA" \
    --prompt-token-ids "$PROMPT_TOKEN_IDS" --eos-token-ids 248044 --device-id 0
```

`all` 分别开启 prefill 和 decode 两个窗口，每个只采一次。
每次 warmup 和测量都从相同 prompt 重建状态；decode 的 prefill/KV 准备在窗口外。
普通 C++ 只加载两个 OM。控制器只需要 Python 标准库；C++ 推理和采集不调用 pyACL。

只测单个阶段时，执行以下完整命令；`PROFILE_STAGE` 可填 `prefill` 或 `decode`：

```bash
PROFILE_STAGE=decode
"$REPO_ROOT/tools/run_msprof.sh" \
  --label "om-ordinary-$PROFILE_STAGE" --output-dir "$AI_RUN_DIR/msprof/ordinary-$PROFILE_STAGE" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend cpp --profile-mode ordinary \
  --profile-stage "$PROFILE_STAGE" --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$CPP_RUNNER" --model-kind chunk \
    --model "$AI_RUN_DIR/ordinary-plan.txt" --model-sha256 "$ORDINARY_PLAN_SHA" \
    --prompt-token-ids "$PROMPT_TOKEN_IDS" --eos-token-ids 248044 --device-id 0
```

## 14. DFlash 模式：分别采一次 prefill、draft、verify

只采一次 verify 并按耗时排序算子，在第 1 步配置的同一终端执行：

```bash
cd "$REPO_ROOT"
"$MODEL_PYTHON" -B tools/profile_verify_om.py \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" --device-id 0
```

脚本读取 `artifacts/deployment-manifest.json`，生成与现有 OM 配套的三图计划。
prompt 和 EOS 优先取自 `reports/cpp-paired.json`；该文件不存在时，读取
`prompt-ids.csv` 并使用 EOS `248044`。也可用 `--prompt-report /path/to/report.json`
指定报告，或用 `--prompt-token-ids "..." --eos-token-ids 248044` 指定 token IDs。
`--deployment-manifest` 可指定其他运行目录中的 OM 清单。

默认预热一次，然后只采一次完整的 `target_verify.om` 调用；prefill、draft、模型加载、
状态准备和预热都在采集窗口外。加载三个 OM 可能耗时数分钟，默认每次控制转换的等待上限
为 600 秒。采集使用 msprof 动态 CLI，无需 pyACL。`MSPROF_BIN` 未设置时从 PATH 查找
`msprof`。默认 `--max-new-tokens 32 --max-draft-tokens 15 --profile-warmup 1`。

每次创建新的 `msprof/verify-<随机后缀>/`，开始时打印完整路径，完成时输出：

| 文件 | 内容 |
|---|---|
| `hotspots.txt` | 耗时最高的 20 类算子和 20 个任务，保留 AI_CORE/AI_CPU、静态/动态和输入形状 |
| `operator-types.csv` | 按算子类型、任务类型和 OP State 分组的次数、总时长、平均值、最大值，单位 ms |
| `operator-tasks.csv` | 全部任务按耗时排序，保留名称、形状和输入 dtype |
| `capture/verify-stage-summary.csv` | 同步的 verify 阶段时间 `profiled_elapsed_ms` |
| `capture/profile/msprof/verify/` | 原始 PROF 数据、时间线和 `op_summary*.csv` |
| `capture/log/` | 设备检查、msprof 和执行日志 |

排序使用 msprof 的 `Task Duration(us)` 并换算成 ms。不同 stream 的任务可能重叠，
算子时长之和不等于阶段耗时；多个 CSV 分别统计，不合并累加。阶段耗时包含 profiling
开销。判断慢算子时，同时查看排序结果与时间线中的空隙。

需要分别采 prefill、draft、verify 时，使用以下命令：

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p prepare-chunk-plan \
  --deployment-manifest "$AI_RUN_DIR/artifacts/deployment-manifest.json" \
  --mode dflash --output "$AI_RUN_DIR/dflash-plan.txt"
read -r DFLASH_PLAN_SHA _ < <(sha256sum "$AI_RUN_DIR/dflash-plan.txt")

PROFILE_STAGE=all
"$REPO_ROOT/tools/run_msprof.sh" \
  --label "om-dflash-$PROFILE_STAGE" --output-dir "$AI_RUN_DIR/msprof/dflash-$PROFILE_STAGE" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend cpp --profile-mode dflash \
  --profile-stage "$PROFILE_STAGE" --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$CPP_RUNNER" --model-kind chunk \
    --model "$AI_RUN_DIR/dflash-plan.txt" --model-sha256 "$DFLASH_PLAN_SHA" \
    --prompt-token-ids "$PROMPT_TOKEN_IDS" --eos-token-ids 248044 --device-id 0
```

`PROFILE_STAGE` 可填 `prefill`、`draft`、`verify`、`all`。`all` 复用一个 C++ 进程，
依次为三个阶段创建独立窗口；DFlash 不加载 decode OM。

| 阶段 | 窗口内 | 窗口外的准备 |
|---|---|---|
| `prefill` | 全部 Target prompt 分块，包含各块 Top1 | 状态清零；不执行 Draft |
| `decode` | 一次一行 Target decode、LM head、Top1 和状态更新 | prefill、KV 初始化 |
| `draft` | 一次 draft OM：末块特征投影、Draft KV 追加、K token proposal | Target prefill、长 prompt 前面分块的 Draft KV 初始化 |
| `verify` | 一次 verify OM：两次 GDR、Target Top1、接受判断和 committed state | prefill、Draft、EOS 截断、block 整理；之后的 C++ 状态指针发布 |

一次 prefill 窗口可以有多次 64 行 OM 调用。verify 内部已经融合 commit，C++ 不提供
独立 `accept-commit` 窗口；内部算子可从 verify 的算子 CSV 查看。
采集时 K=`min(max_draft_tokens, max_new_tokens-1, 15)`，减去的一项是 prefill
已经输出的 anchor。可在 C++ runner 参数中设置 `--max-draft-tokens` 和
`--max-new-tokens`；默认 K=15。各窗口的报告记录 `proposal_count`。
普通 decode 需要 prompt 后至少留一行 KV，DFlash verify/all 需要 K+1 行空间。
anchor 为 EOS 时退出，不伪造空的 decode/verify 采集。

## 15. 读取 msprof 输出

第 13 步 `all` 的输出为：

```text
msprof/ordinary-all/
  profile/msprof/om-ordinary-all/prefill/   # 原始 PROF_* 和 op_summary*.csv
  profile/msprof/om-ordinary-all/decode/
  om-ordinary-all-stage-report.json
  om-ordinary-all-stage-summary.csv
  manifest/om-ordinary-all.json
  manifest/om-ordinary-all-control.json
  log/msprof-om-ordinary-all.log
```

单阶段的 raw 目录是 `profile/msprof/<label>/`，没有阶段子目录。
`op_summary*.csv` 提供算子执行时间；`stage-summary.csv` 提供同步阶段时间、算子行数和路径。
`profiled_elapsed_ms` 包含 profiling 开销，排除 msprof attach/start/stop/quit 等待。

控制器等待 start 回执后放行应用，同步设备并收到 stop/quit 回执后完成一个窗口。
所有窗口都要求调用次数正确、进程成功退出且导出非空算子 CSV，任一失败则整体 FAIL。
`--profile-timeout` 默认 600 秒；应按模型加载和阶段执行时间设置。
每次采集使用新的 label/输出路径，已有 raw 数据及报告不会被覆盖。

## 16. 单模式运行和部署容量

使用第 14 步生成的三图计划，可直接启动 DFlash 生成：

```bash
"$CPP_RUNNER" --model-kind chunk --mode dflash \
  --model "$AI_RUN_DIR/dflash-plan.txt" --model-sha256 "$DFLASH_PLAN_SHA" \
  --prompt-token-ids "$PROMPT_TOKEN_IDS" --eos-token-ids 248044 \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens 15 --device-id 0 \
  --output "$AI_RUN_DIR/reports/cpp-dflash.json"
```

单模式 runner 仍执行 3+10 测量，报告中 `ordinary_parity=NOT_RUN`。
只部署 DFlash 且已完成普通对照时，可以将 `factory.json` 的 `include_ordinary_decode` 改为
`false`，在一个全新 bundle 目录重新执行第 8、9 步，导出三个 OM；这种 bundle 不能运行普通
decode 或 paired。不同 OM 不会自动共享权重，设备显存要覆盖驻留模型和状态 buffer。

状态语义与开销：Target verify 用本轮初始 recurrent state 做两次 GDR，第二次
`effective_length=accepted+1`；C++ 核对接受数后统一发布第二遍状态，
第一遍的 raw FP32 state 仅保留设备输出缓冲区。零接受后关闭 Draft，以
已加载的 `target_decode` 执行后续单 token 生成。单模式 DFlash 和低显存 paired 的 DFlash 组只加载三个 OM，
继续使用 verify 的 `valid_rows=1`，其物理图仍为 16 行。
`speculation_disable_events` 和 `target_only_fallback_rounds` 记录关闭 Draft 及后续轮数；
`stage_ms` 显示实际调用了 decode 还是 verify。两种后备路径都需要与 ordinary 检查
token 等价。固定 64 行 Draft gear、非末尾 prompt 块的 Draft KV 初始化、每块 prefill
的 LM head，以及 functional KV 更新，都可能增加开销，应按实际 msprof 数据评估。

## 17. 失败定位

| 报错位置 | 检查内容 |
|---|---|
| NPU/ATC preflight | 设备可见性、驱动/CANN、精确 SoC、解释器与环境变量 |
| wrapper 导入 | receiver 文件、依赖及第 3 步的模块搜索配置 |
| SOURCE_LOCK/input manifest | 源码、模型和量化数据是否与锁定内容一致 |
| TorchAir graph break/unsupported op | 首个失败算子的 schema、Fake/Meta、converter 和 GE 注册 |
| fused attention 导出 | 长度专用 INT64 输入、空 `pse_shift`、运行时因果及有效前缀 mask |
| ATC 编译 | `log/compile-om.log`、具体不支持节点及算子包 |
| C++ 加载 OM 时 out of memory | 最后一条 `load graph`、各图的 `om-memory`、加载前后的 `device-memory` 和 `npu-smi info` 中的其他进程 |
| C++ I/O 不匹配 | OM 和计划是否配套，实际 dtype/shape/字节数是否匹配 |
| token 不一致 | 保留首个差异轮次、接受数、有效行数、GDN/conv/KV 状态；停止性能比较 |
| msprof 等待或空 CSV | control JSON 的 ready/start/stop/quit 回执、目标 PID、runtime 日志及导出日志 |

runner 在加载前通过 AscendCL 的
[aclmdlQuerySize](https://www.hiascend.com/document/detail/zh/canncommercial/601/inferapplicationdev/aclcppdevg/aclcppdevg_03_0099.html)
记录全部待加载 OM 的 `weight_bytes` 和 `work_bytes`，通过
[aclrtGetMemInfo](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0107.html)
记录加载前后的可用设备内存。HBM、DDR 查询可能指向同一物理内存，不能相加；
不支持或失败的查询显示 `unavailable`，不会阻止正常加载。查询结果不包含全部运行时开销，
不能作为峰值内存保证。这些诊断在模型加载和释放时执行，从模型循环计时中排除。

若底层日志为 `MallocWeightsMem` / `InitWeightMem` 失败，申请的是该 OM 的权重内存，
不能把它解释为 GDR state 输出的字节数。verify 的 24 个 discard FP32 state 输出共
48 MiB，属于另行分配的 I/O buffer；加载该 OM 成功后才分配这些 buffer。
用 `npu-smi info` 检查其他进程的占用，确认任务身份并正常退出不再需要的任务，
释放内存后重跑推理即可。只测 verify 时使用第 14 步的 DFlash 采集命令，会加载三张 OM，
无需重新导出即可省去普通 decode OM；`infer-cpp --low-memory` 配对测量最多同时驻留三张。

runner 启动时还记录本进程 `pid`、`ppid`、可见的 `NSpid` 和 PID namespace。
若 `npu-smi` 的 PID 在当前 shell 查不到，先确认是否位于容器中。特权容器内的
`npu-smi` 显示宿主机 PID，可在宿主机用 `ps -fp <PID>` 和
`rg '^NSpid:' /proc/<PID>/status` 找到任务及容器内 PID；详见
[Ascend PID 对应关系](https://www.hiascend.com/document/detail/zh/mindstudio/830/T%26ITools/Profiling/atlasprofiling_16_0013.html)。
不要仅凭容器内找不到进程判定为显存泄漏。其他用户的任务由其所有者处理。

正常退出及 C++ 捕获异常时都会执行清理：同步、卸载已加载模型、释放共享工作内存和 I/O buffer、
销毁 stream/context、ResetDevice、Finalize。日志中的 `cleanup-error` 给出失败接口
与返回码；`cleanup_end` 汇总成功卸载数量、已申请/已释放的 device buffer 字节数
和错误数。`device-memory phase=after_release` 在卸载模型与释放 buffer 后采样，
此时 context 尚未销毁。buffer 计数包含 runner 分配的共享工作内存，不包含 GE 内部申请，
也不证明驱动已完成回收。低显存组间卸载失败会阻止下一组加载；
正常完成推理后如清理接口报错，runner 返回失败且不写入 PASS 报告。
确认宿主机 PID 消失后仍持续占用时，应保留退出日志和驱动日志继续定位。

命令参数和 tensor ABI 的集中说明见 [AIR/OM/C++ 接口参考](QUANT_AIR_OM_FRAMEWORK.md)。
