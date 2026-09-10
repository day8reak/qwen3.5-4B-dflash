# Ascend 310P：从模型到 AIR、OM、C++ 运行和 msprof

按本文顺序完成环境准备、输入检查、模型导出、转换、C++ 执行和性能采集。
支持 batch=1、strict greedy、W8A8 Target＋FP16 Draft。

模型结构、逐轮接受示例、状态提交和加速条件见
[DFlash 结构与生成流程](DFLASH_ARCHITECTURE.md)。

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
    "npu_cache_update_",
)
missing = [name for name in required if not callable(getattr(torch_npu, name, None))]
assert not missing, f"缺少 NPU 算子: {missing}"
print("Target operator symbols: PASS")
PY
```

三个 Target OM 使用 `CacheUpdate` 写入 paged KV：prefill 按对齐的 64 行块写入，
decode 写一行，verify 逐行写入 16 行。块号和块内偏移均为 INT32，
verify 跨块时逐行重算位置。Draft 的 dense KV 使用 `ScatterElements`。
导出检查同时核对 dispatcher/Meta 和 GE 节点，不能只凭 Python 接口存在判断注册完整。
四张图的接口和状态提交规则见[模型结构与运行流程](DFLASH_ARCHITECTURE.md)。

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
    "draft_attention_matmul_dtype": "float16",
    "dtype": "float16", "device": "npu:0", "adn_rms_norm_ge_op_type": "AdnRmsNorm",
}
(run / "factory.json").write_text(json.dumps(config, indent=2) + "\n")
PY
```

导出前会再次检查外部输入和 `SOURCE_LOCK.json`。冻结输入后不要增加、删除或修改文件。
`adn_rms_norm_ge_op_type` 默认填写 `AdnRmsNorm`，对应自定义 GE 算子，输入名为
`self` 和 `gamma`。普通 Target、DFlash Target 和 Draft 共用这个选择。
只有环境明确要求 GE `RmsNorm` 时才显式填写 `RmsNorm`，其输入名为 `x` 和 `gamma`。
PyTorch 调用了 `adn_rms_norm` 并不表示 OM 的 GE 节点一定叫 `AdnRmsNorm`；节点类型
由这个配置决定。已有 `factory.json` 中的显式值不会被默认值覆盖。
更改此值后，使用独立、空的 bundle 执行第 8、9 步，再按第 11、12 步验证；
C++ runner 和输入权重可复用。msprof 的 OP Type 应与所选 GE 类型一致。

`draft_attention_matmul_dtype` 默认 `float16`，控制 OM Draft attention 的 QK 和 PV
两次矩阵乘：Q/K/V 和 Softmax 概率在进入 MatMul 前转为 FP16，结果转为 FP32；
缩放、Mask 和 Softmax 在 FP32 中执行，attention 最终返回模型的 FP16 dtype。
该精度选择会改变舍入，可能改变 proposal 和接受率。设置 `float32` 可导出 FP32
矩阵乘基线；两种配置都保留在 AIR/deployment manifest 的 graph metadata 中。
`torch_npu` Draft 路径使用 FP32 矩阵乘，可继续作为对照。

精度配置在导出时生效。每种配置使用独立、空的 `--bundle-dir`，分别执行第 8、9 步，
并让第 11 步的 `infer-cpp --deployment-manifest` 指向对应 bundle。使用相同 prompt、
EOS、生成长度和 `--trace-rounds` 比较 proposal、接受率及最终 token；允许 proposal
变化，最终 token/EOS 仍须通过 ordinary 对照。C++ runner 和外部权重可复用。

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
三个 Target 图均检查 `AdnRmsNorm`（或显式选择的 `RmsNorm`）、`DynamicQuant`、
`QuantBatchMatmulV4444`、`ChunkGatedDeltaRule`、`AdnFusedInferAttention`、`CacheUpdate`，
以及 `SoftplusV2`。Draft 的 RMSNorm 也使用并审计 `AdnRmsNorm`，其余计算使用
Tensor 算子，不要求出现 GDR 等 Target 专用节点。
标准 4B Target 的 prefill、decode、verify 图各至少保留 105 个 RMSNorm 节点，
6 层 Draft 图至少保留 32 个；数量按加载的模型层数计算并写入导出合同。
Draft 用 FP32 输入和全 1 的 gamma 调用自定义 RMSNorm，将归一化结果转回 FP16，
再乘 checkpoint 中的有效权重；保留其归一化后先舍入、再缩放的顺序。
量化 matmul 在 AIR 捕获时使用专用前端，保留 FP32 weight/per-token scale 和 FP16 输出；
普通 NPU 推理仍调用同一套 receiver 量化接口。
每张 Target 图的 QuantBatchMatmulV4444 节点数不能少于加载器记录的
QLinear 数量；标准 4B Target 为 249，包含词表输出 head。该检查用于发现启用了量化前端、
但部分量化模块没有进入导出图的情况。
`CacheUpdate` 的最低节点数分别为 prefill 16、decode 16、verify 256；
verify 的计数包含每层 K/V 的 16 次单行写入。这些是 AIR 节点检查，
目标机的实际任务耗时通过本文后面的分阶段 msprof 测量。
保留整个 AIR 目录，不要只复制 `.air` 文件。

卷积历史窗口通过切片加 `stack` 导出，保持每个有效前缀的状态。
verify 接受长度使用 INT32 `Cumsum → Equal → ReduceSum`，最后输出 INT64 接受数；
短块的 padding 不参与接受判断，第二次 GDR 提交 `accepted_count+1` 行。
Draft 缓存写入索引与 KV head 复制使用静态 `repeat/Tile`。
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
归一化、RoPE、Softmax 和 GDR 状态边界，以及配置选定的 Draft MatMul 输入 dtype。
Draft 的 FP16 选择通过图中显式 Cast 实现。也支持显式传入 `--precision_mode_v2=origin`；
编译器拒绝同时指定两种精度参数或使用降精度模式。

增量套件的 `draft` 图默认额外使用 `--deterministic=1`，三个 Target 图的默认参数不变。
这是根据同一份 FC AIR 的设备对照加入的：关闭时 native/OM 均有 19/20 次输出变化，
开启后两条路径均为 0/20，且有效输出逐位相同。完整 Draft 仍需重放、普通生成对照和重新测速。
各图实际命令保存在 `graphs[].atc_command`，额外参数保存在 `compiler.graph_extra_args`。
显式 `--atc-arg=--deterministic=0` 可用于对照实验；该公共参数会传给所有待编译图。

FP16 MatMul 后的 FP32 Cast 为 ATC 的 `MatmulCastFusionPass` 提供融合机会，
但源码中的 Cast 不保证直接 FP32 输出；没有融合时会先得到经过 FP16 舍入的结果。
在目标机检查编译后 MatMul/BatchMatMul 的输入 dtype 与实现，并用第 14 步的 Draft
单步 msprof 采集核对 Cube 活动和时延。不能只凭 FP32 输出 dtype 判断累加精度。
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

### 只重编已有套件的 Draft，启用确定性计算

已有正确的 AIR 和四图 deployment 时，从 ATC 这一步开始即可。复用原 Draft AIR 及其
精度参数，只替换确定性选项为 1；三个 Target OM 保持原文件和 SHA-256。

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p recompile-draft-om \
  --deployment-manifest "$AI_RUN_DIR/artifacts/deployment-manifest.json" \
  --output "$AI_RUN_DIR/artifacts/deployment-manifest-deterministic.json" \
  --atc "$ATC_BIN"

export DEPLOYMENT_MANIFEST="$AI_RUN_DIR/artifacts/deployment-manifest-deterministic.json"
```

新 manifest 必须位于原 manifest 所在目录，并使用尚不存在的文件名。新 Draft OM 和日志
写到同级的 `draft-deterministic-*`，旧 manifest/OM 保留。AIR、权重 payload、旧 OM 的
hash 或 ABI 审计不一致时直接停止。新 manifest 编译成功不等于完整模型验证通过。
后续命令的 `--deployment-manifest` 要使用上述新变量。
如果运行 Draft 冻结输入重放，请重新生成快照，不要传旧的 `--inputs`：快照绑定旧 Draft OM hash。

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

### 一次测试多个 prompt 和接受率

更新源码并按第 10 步构建新的 runner 后，运行：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/benchmark_prompts.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --runner-config "$AI_RUN_DIR/runner.json" --model-dir "$TARGET_DIR" \
  --device-id 0 --max-new-tokens 128 --max-draft-tokens 15 \
  --eos-token-id 248044 --low-memory
```

默认有 8 条：中文解释、学习计划、数学、代码、翻译、摘要、英文解释和故事创作。
每条均使用 chat 模板，普通与 DFlash 各做 3 次预热、10 次正式测量，结束时检查输出
token、EOS 和停止原因一致。每次生成前重置请求缓存；一条失败会记录原因并继续其余条目。
普通生成本身的稳定性和两种模式的精确一致性检查沿用原 runner，不放宽阈值。

整批只启动一个 C++ 进程：去掉 `--low-memory` 时四图加载一次，逐 prompt 交错跑两种模式；
加上时先跑完全部普通 prompt，再统一卸载、加载 DFlash 所需三图，跑完全部 DFlash prompt。
两种协议都复用加载的模型。低显存模式的分组顺序可能受温度和外部负载变化影响，报告会明确记录。
模型加载、模式切换和请求重置均排除在模型循环时延之外。

脚本打印本次 `Output` 目录 `$AI_RUN_DIR/prompt-suite-*`，其中：

| 文件 | 内容 |
|---|---|
| `summary.md` / `summary.json` | 每条接受率、每轮产出、DFlash tok/s、相对普通速度、生成长度、停止原因和失败原因（完整字段见 JSON） |
| `request.json` | prompt 原文及 token IDs、tokenizer 来源、OM/runner hash、Draft ATC 命令和测试参数 |
| `runner-batch.json.cases/*.json` | 每条完整普通/DFlash 报告，包含每次测量和逐轮 trace |
| `runner-batch.json` / `runner.log` | 整批模型复用协议、加载时间和执行日志 |

接受率按正式测量的 `accepted_draft_tokens / drafted_tokens` 计算。总接受率是总接受数
除以总候选数，不是简单平均每条的百分比；没有候选时显示 N/A。每轮产出包括该轮实际
发出的已接受 token 和补充 token。速度比为普通与 DFlash 的模型总时延中位数之比，
大于 1 才说明该条更快。失败条目保留在表中，总结只对通过的条目统计，整批仍标为失败。
这组样本用来比较任务差异，不能证明所有 prompt 都有高接受率。

可通过 `--prompts "$AI_RUN_DIR/prompts.json"` 使用自己的测试集（最多 64 条）。文件格式：

```json
[
  {"id": "math_01", "category": "数学", "prompt": "计算 37 × 28，并解释步骤。"},
  {"id": "code_01", "category": "代码", "prompt": "写一个 Python 函数统计单词频率。"}
]
```

也可直接使用字符串数组。`id` 只能包含字母、数字、下划线或连字符且不能重复。
默认最多生成 128 个新 token，遇到 EOS 可提前结束；JSON 同时记录实际长度和停止原因。
prompt 加输出预算超出 OM 的固定上下文容量时提前报错，可减小 `--max-new-tokens` 或缩短 prompt。
这项测试直接测模型循环，不套 msprof；算子分析仍使用第 14 步的独立采集入口。

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

## 13. 普通模式：分别采一次 prefill 和 decode

配置 msprof 和要测量的 OM 清单，然后运行统一采集入口：

```bash
export MSPROF_BIN=/absolute/path/msprof
export DEPLOYMENT_MANIFEST="$AI_RUN_DIR/artifacts/deployment-manifest.json"

"$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --profile-mode ordinary --profile-stage all --device-id 0
```

`DEPLOYMENT_MANIFEST` 指向本次要测量的模型产物；如果使用其他 bundle 目录，在这里设置对应路径。
脚本根据清单重新生成加载计划并记录清单、runner 的 SHA-256，不复用手工保存的计划。
准备计划使用上述模型 Python 环境；实际采集由 C++ runner 执行，普通模式只加载 prefill 和 decode 两张 OM。

`all` 在一个 C++ 进程中分别采一次 prefill、一次 decode，每个窗口都从相同 prompt
重建状态并在窗口外预热。只测其中一项时，将 `--profile-stage all` 改为
`--profile-stage prefill` 或 `--profile-stage decode`。

prompt 和 EOS 优先取自 `reports/cpp-paired.json`；不存在时读取 `prompt-ids.csv`，
EOS 默认 `248044`。可用 `--prompt-report /path/to/report.json` 指定推理报告，
或用 `--prompt-token-ids "..." --eos-token-ids 248044` 指定相同的输入。
每次生成新的输出目录，开始时打印路径和所用 prompt 来源。

## 14. DFlash 模式：分别采一次 prefill、draft、verify

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --profile-mode dflash --profile-stage all --device-id 0 \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens 15
```

`--profile-stage` 可选 `prefill`、`draft`、`verify`、`all`。
例如只测一次验证，将上面的 `all` 改为 `verify`。
DFlash 只加载 prefill、draft、verify 三张 OM，`all` 复用同一个进程，
按顺序开启三个独立窗口，每个阶段只采一次。内部统一调用 `tools/run_msprof.sh`，
由同一套动态 msprof 控制器完成 start/stop/quit；无需 pyACL，也不用再套外层 msprof。

默认 `--profile-warmup 1 --profile-timeout 600 --aic-metrics PipeUtilization`。
每次 warmup 和采集前都清零请求缓存，从同一 prompt 重新执行准备步骤。
所有 warmup 和采集结果都与第一次 warmup 比较输入 token 和有效输出；
verify 只比较实际 block 的有效行，物理 16 行中的填充输出保留在日志中，不参与通过判定。
有效输出或输入 token 不一致仍返回失败；中间缓存和特征张量尚未逐项比较。
模型只加载一次；加载可能耗时数分钟，
必要时提高控制转换的 `--profile-timeout`。指标也可选 `Memory` 或 `MemoryUB`。

| 阶段 | 采集窗口内 | 窗口外 |
|---|---|---|
| `prefill` | 完整 Target prompt，包含全部 64 行分块、CacheUpdate 和各块 Top1 | 状态清零；不执行 Draft |
| `decode` | 一次一行 Target decode、CacheUpdate、LM head、Top1 和状态更新 | prefill、KV 初始化 |
| `draft` | 一次 draft OM：末块特征投影、dense KV 追加、候选生成和 Top1 | Target prefill、长 prompt 前面分块的 Draft KV 初始化 |
| `verify` | 一次 verify OM：CacheUpdate、两遍 GDR、Target Top1、接受判断和 committed state | prefill、Draft、EOS 截断、block 整理；之后的 C++ 状态指针发布 |

一次 prefill 窗口可能包含多次 OM 调用。verify 已融合 commit，第一遍 GDR 的 state
只保留设备缓冲区并丢弃，第二遍 state 才提交。无需为内部 commit 另建 OM 或采集窗口。
AIR 的 CacheUpdate 节点约束是每次 prefill 16 个、decode 16 个、verify 256 个；
实际 OM 任务以 msprof 导出为准。Draft 的 dense KV 仍可出现 ScatterElements。

采集时 K=`min(max_draft_tokens, max_new_tokens-1, 15)`；EOS 截断可能进一步缩短
verify 的有效输入。普通 decode 需要 prompt 后至少留一行 KV，DFlash verify/all 需要
K+1 行空间。anchor 为 EOS 时退出，不伪造空的 decode/verify 采集。

Python NPU 的 `verify` 只采第一遍 GDR，`accept-commit` 单独采第二遍及提交；
C++ OM 的 `verify` 采整个融合图。两者同名窗口范围不同，不能直接比较其时间。
Python 的细分和联合阶段见 [Python NPU 手册](DFLASH_RUN_AND_VALIDATE.md)。

## 15. 读取每阶段算子耗时

统一入口为每次运行创建目录：

```text
msprof/<mode>-<stage>-<随机后缀>/
  profile-request.json                 # 模型清单、runner hash、输入和采集参数
  plan.txt                             # 从指定清单生成的加载计划
  capture/
    all-stage-report.json
    all-stage-summary.csv              # 各窗口同步时延和 stage_scope
    all-operator-types.csv             # 各阶段、各精度的算子次数/耗时
    all-operator-tasks.csv              # 每个任务及输入输出形状、dtype
    all-hotspots.txt                    # 各阶段的慢算子排序
    profile/msprof/all.iterations.jsonl # 各次预热/采集的输入、输出与比较结果
    profile/msprof/all/prefill/         # 原始 PROF_*、op_summary*.csv
    profile/msprof/all/draft/           # 普通模式对应 decode
    profile/msprof/all/verify/
    manifest/all.json
    manifest/all-control.json
    log/msprof-all.log
```

示例为 `--profile-stage all`；单阶段时文件前缀改为阶段名，
raw 目录直接是 `profile/msprof/<stage>/`。直接使用 `run_msprof.sh` 时，
同样生成这些汇总，文件前缀为指定的 `--label`。

若 start/stop/quit 均成功，随后出现 `profile output differs from warmup`，
表示采集后的输出一致性检查失败。正常的模型卸载和 `cleanup_end errors=0`
是退出清理记录。原始采集目录和 `*.iterations.jsonl` 会保留；此时不生成通过报告，
也可能尚未导出 CSV。

读取 `iterations.jsonl` 中最后一条 `event="completed"`、`check.status="FAIL"`：
`check.first_difference` 给出字段、从 0 开始的下标、参考值和实际值。
`input_token_ids` 不同表示准备阶段已产生不同输入；verify 的第一项是 anchor，
其后是实际候选。输入相同但 `output_token_ids` 不同，表示有效输出发生变化，
需要继续核对设备上的缓存、特征和算子结果。`input_state_comparison="NOT_RUN"`
明确表示未比较中间张量，不能仅凭 token 相同认定全部设备输入相同。
`verify_valid_rows` 给出实际比较行数；仅 `padding_token_ids_match=false`
不会报错。若有效行数为 16，则填充行不能解释差异。

日志中的 `iteration_trace=` 和 `manifest/<label>.json` 的
`artifacts.iteration_trace` 都指向该记录。预热之间的差异也会保留记录并立即失败；
关闭预热会失去此项检查，不能用于确认问题已修复。

若 `all` 在 draft 的 stop/quit 成功后报
`application control disconnected before ready`，表示应用在下一次 ready 前退出。
控制器日志里的 `stage=verify` 是正在等待的阶段；只有出现 verify 的
`application_ready`、`application_started`，才表示已进入它的采集窗口。
例如 JSONL 最后一条是 draft 的 `check.status="FAIL"`，就应先查 draft 的差异。
`application_done` 只表示窗口内执行完成，输出比较在 stop/quit 后进行。

C++ 会在卸载模型前打印 `application_error stage=... message=...`。
控制器收到断连后继续接收应用日志，等待其自然退出，等待时间受
`--profile-timeout` 限制；超时才终止仍未退出的应用。
`manifest/<label>-control.json` 中的 `application_failure` 记录
`waiting_stage`、`waiting_event`、`last_stopped_stage` 和 `exit_wait_timed_out`；
`application_error` 保留已识别的 C++ 异常，`application_output_tail` 保留日志尾部。
所有要求的阶段未完成时，即使应用退出码为 0，整体也会 FAIL。

`infer-cpp` 会丢弃预热结果，正式重复的稳定性检查比较最终生成 token 和停止原因，
没有要求每轮全部 draft 候选一致。被 Target 拒绝的候选即使变化，最终结果也可能不变。
stage profiling 则以第一次预热为参考，逐项比较后续预热和采集的有效候选。
因此，正常生成通过、stage profiling 失败，并不能单凭这一点证明 msprof 改变了计算结果。

需要判断 draft 候选变化是否来自输入时，开启额外检查。
若差异报在 verify 的 `input_token_ids`，可直接采集 verify：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/profile_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --profile-mode dflash --profile-stage verify --device-id 0 \
  --max-new-tokens "$MAX_NEW_TOKENS" --max-draft-tokens 15 \
  --profile-warmup 3 --profile-audit-draft-inputs
```

也可选择 `--profile-stage draft`，或保持 `--profile-stage all`，同时检查
独立 draft 阶段和 verify 准备阶段的 Draft 输入。
每个阶段都会重新清零并准备请求；verify 会重新调用 Draft 生成候选，
不复用独立 draft 窗口的输出。因此，draft 阶段全部 PASS 不能证明 verify
准备时的候选或 Draft 输入相同。
该开关默认关闭，普通模式和单独的 prefill 不使用。C++ runner 直接接收的参数形式为
`--profile-audit-draft-inputs true`，可放在 `run_msprof.sh` 的 `--` 后。
这里使用 3 次预热，且每次都检查差异。若 `measured=false` 的预热记录已 FAIL，
当前阶段的采集窗口还未开始，说明差异在未启用该次采样时也存在；
若预热全 PASS、只有 `measured=true` 失败，再结合下面的输入 hash 缩小范围。

检查在每次独立 draft 阶段的执行前，或 verify 准备时的 Draft 调用前，
读取 features、当前 Draft KV 的完整设备字节，
并对即将上传的 `anchor`、`start_position`、`valid_rows`、`proposal_count`
计算 SHA-256，写入同一 JSONL 的 `draft_input_sha256`。
`draft_input_scope="draft_stage"` 表示独立 draft 阶段；
`draft_input_scope="verify_preparation"` 表示为 verify 生成候选的那次 Draft 调用。
关闭检查或处于 prefill/decode 阶段时，这个字段为 `null`。
verify 记录中的 `MATCH_SHA256` 比较的是准备阶段的 **Draft 输入**，
不表示 Target verify 的全部缓存或图内张量已经比较过。
读回、计算 hash 和写 `prepared` 记录都在 msprof start 前；
`completed` 复用本次执行前的 hash，与参考迭代比较，并非执行后重新读回。
检查不增加 OM 调用，也不改变缓存提交。
这是定位开关，额外读回会影响准备耗时和缓存热度，性能测量时保持关闭。

| JSONL 结果 | 下一步检查 |
|---|---|
| `DIFFERENT_SHA256`，只有 `features` hash 不同 | 两次 prefill 交给 draft 的特征不同，即使 prefill top1 相同也不能视为相同输入 |
| `DIFFERENT_SHA256`，`d0_key` 等 KV 或控制参数不同 | 核对缓存复位、长 prompt 的上下文追加和参数准备 |
| `MATCH_SHA256`，有效候选仍不同 | 所检查的输入字节相同，继续检查 draft 图内部算子、工作内存及执行稳定性 |

hash 检查覆盖图边界输入，包括物理填充区；它不读取权重、共享工作内存或层内张量。
hash 差异仅作为诊断记录，不单独放宽或替代有效 token 的一致性检查。
若 verify 的输入在下标 8 变化，对应的是第 8 个候选（下标 7），因为下标 0 是 anchor。
候选不同后，Target 在这些候选之后算出的预测也可能不同；应先定位准备阶段的差异。
当变化的候选原本就被拒绝时，接受数、fallback 和最终生成文本仍可能一致。
检查后暂时稳定，也不能单凭这一点认定问题修复：额外读回和同步会改变执行条件。
更新采集脚本和 C++ runner 即可使用上述诊断，已有 AIR/OM 无需重新生成；
C++ 构建步骤见第 10 节。

`operator-types.csv` 按 stage、原始 CSV、device/model、算子类型、任务类型、OP State、
输入和输出 dtype 分组，包含 count、total_ms、mean_ms、max_ms。
筛选 `op_type=CacheUpdate` 可查看缓存更新；筛选矩阵乘时，
`FLOAT;FLOAT` 与 `FLOAT16;FLOAT16` 分开统计。`operator-tasks.csv` 保留
op_name、stream/task ID、输入输出形状，可定位具体 GDR 或矩阵乘节点。

排序将 `Task Duration(us)` 换算为 ms。不同 stream 的任务可能重叠，算子时长之和
不等于阶段时间；多个 CSV、不同阶段分别统计，不合并累加。`stage-summary.csv` 中的
`profiled_elapsed_ms` 包含 profiling 开销，排除 msprof 控制等待。
判断端到端是否提速仍使用不带 profiler 的 3+10 测量。

控制器收到 start 回执才放行应用，设备同步后收到 stop/quit 回执才完成窗口。
调用次数错误、进程失败、空导出或没有有效任务时长时整体 FAIL，不生成成功的算子汇总。
`PASS_CAPTURE` 表示采集流程完成，不能替代完整 token/接受率验证。


### 15.1 输入 hash 相同但候选变化：固定输入回放

若 Draft 的 17 个边界输入 hash 全部相同，而未采集的预热仍出现候选差异，
先用临时工具 [debug_draft_om.py](../tools/debug_draft_om.py)。
它不启动 msprof，也不重新导出或编译 OM；只需编译带回放入口的新 C++ runner：

```bash
# 使用新的构建路径；已有构建目录/报告不会被覆盖。
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-draft-replay" \
  --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build-draft-replay.json"
export CPP_RUNNER="$AI_RUN_DIR/build/cpp-draft-replay/qwen35_dflash_acl_runner"

"$MODEL_PYTHON" -B "$REPO_ROOT/tools/debug_draft_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --device-id 0 --max-draft-tokens 15 --repetitions 20 --workspace both
```

prompt 默认读取本 run 的 `reports/cpp-paired.json`，其次为 `prompt-ids.csv`；
也可传 `--prompt-report` 或 `--prompt-token-ids`。`--repetitions` 是每个实验阶段的
Draft 次数，不是正式 benchmark 的 3+10 协议。

工具在 run 下新建 `debug-draft-*`。`both` 顺序运行两个进程：
先共享工作内存，再由 GE 为各 OM 单独分配工作内存。两边仍只加载 DFlash 的三张图，
权重始终各自独立；独立工作内存可能提高峰值占用，实际需求见对应 `.log` 的
`om-memory` 查询。两个进程共享同一份锁定的 Draft 输入快照，不重新取样作为 A/B 输入。
快照绑定 Draft OM hash、tensor ABI、prompt、proposal_count；复用时逐文件校验。
既有正式生成和 profiling 的内存策略不变。

每个进程依次执行：

| 阶段 | 操作 |
|---|---|
| `isolated` | 一次 Prefill 准备之后，只重复 Draft。每次恢复同一份完整输入，输入/输出地址固定，不交换或提交回放输出 KV |
| `interleaved_prefill` | 每次 Reset + Prefill，然后覆盖全部 Draft 输入为原快照，再执行 Draft；用于检查前序图执行和工作内存历史的影响 |

长 prompt 的 Prefill 准备仍按原路径初始化较早分块的 Draft KV，这些调用不计入回放次数。
两个阶段都不执行 Target verify，不推进解码。每次记录实际设备输入的执行前/后 hash、
所有输出 hash、有效候选、设备地址及第一个不同位置。额外同步与读回会改变执行条件；
回放全通过不能排除原故障，也不能用这里的耗时作性能结论。

结果在 `comparison.json`、`shared.json`、`private.json`，逐次记录路径由报告的
`trace` 给出。`input_directory` 下保留原始输入二进制，供后续逐层/单算子复现。
即使发现候选变化或输入被改写，也继续完成本组次数、保存 FAIL 报告并返回非零。
ACL 执行错误则立即退出并保留已有 trace 和日志，不发布成功报告。
正常生成与 msprof 原来的严格检查没有放宽。

| 观察 | 可以缩小的范围 |
|---|---|
| `inputs_unchanged=false` | 图执行后改写了只读边界输入；按不同 hash 的名字查 alias、越界写或输入算子契约 |
| `isolated` 中输入完全相同但候选不同 | 无需重新生成 Prefill 特征或跨图交替就能复现；继续定位 Draft 内部中间张量、算子与运行时 |
| 仅共享工作内存且交替 Prefill 时复现 | 工作内存复用/前序图执行是优先对照方向，尚不能仅凭此认定具体 kernel |
| token 相同、`full_output_hash_mismatch_iterations` 非零 | 检查 KV 输出；完整 hash 包含物理填充区，不能直接等同有效缓存错误 |
| 全部通过 | 本次扰动下未复现，保留原故障为待定位，不能宣布修复 |

`PASS_REPLAY_CHECKS` 表示有效 token 和逻辑 KV 重复一致、恢复后的输入正确且调用未改写输入；
物理填充区的变化另外报告。`distinct_workspace_policies_exercised=false` 表示没有完成
真正的共享/独立对照，例如工作内存查询失败导致共享侧回退。主机 fake ACL 报告明确标记
`fake_acl=true`，不能作为 NPU 证据。

对照还检查两个进程的参考候选（`reference_tokens_match_across_processes`）和有效 KV
（`reference_valid_kv_match_across_processes`）是否一致；即使各自重复稳定，
只要这两项有一项不同，对照仍返回失败。完整输出 hash 另外保留，不把填充字节当作有效缓存。

只跑一个内存策略可用 `--workspace shared` 或 `private`；复用已有快照加
`--inputs "<上一份报告的 input_directory>"`。直接调用 runner 的临时参数为
`--debug-draft-replay N --debug-draft-workspace shared|private`，可另带
`--debug-draft-inputs PATH`；仅允许 `--model-kind chunk --mode dflash`，不能与 profiling 混用。

同 stream 串行执行时共享工作内存本身符合
[AscendCL 的接口约束](https://www.hiascend.com/document/detail/zh/canncommercial/5046/inferapplicationdev/aclcppdevg/aclcppdevg_03_0079.html)。
如果固定输入仍变，可继续对照 ATC 的
[确定性计算选项](https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/devaids/atctool/atlasatcparam_16_0090.html)
`--deterministic=1`；默认是 0。这需要重新编译受测 OM，并记录新 hash，
不能靠 C++ 环境变量宣称旧 OM 已确定。该选项是否覆盖实际出问题的算子仍需设备实测。
仅降低到 FP16 不足以解释同输入为什么变化，不能在没有中间结果证据时认定精度是根因。

### 15.2 共享和独立工作内存都复现：按 KV 行定位

2026-09-10 用户返回的真实设备回放中，固定输入的检查结果如下。本地未执行该 NPU 实验。

| 工作内存 | 只重复 Draft | Prefill 与 Draft 交替 | 输入恢复/执行后变化 |
|---|---|---|---|
| 共享 | 20 次中 11 次候选不同 | 20 次中 5 次候选不同 | 均为 0 |
| 各 OM 独立 | 20 次中 10 次候选不同 | 20 次中 10 次候选不同 | 均为 0 |

这说明 msprof、跨图交替以及跨模型共享工作内存都不是复现所必需的条件，
不能据此排除 OM 内部中间张量、workspace、权重或短暂写入等问题。
两个进程的参考候选相同，但 `d0_value` 等输出的完整 hash 不同。
`full_output_hash_mismatch_iterations=19` 表示 19 次调用中至少一个输出不同，
不表示每一个 KV tensor 都变化 19 次。

源码中的 context KV 路径为：

```text
features → fc → hidden_norm → 各层 K/V 投影 → K norm/RoPE（仅 K）→ ScatterElements → dN_key/value
```

六层 context K/V 投影使用同一份 projected features；不是逐层 attention 的输出。
提议阶段的 QK/PV BatchMatMul 不在这些 context KV 的上游数据依赖上。
因此，若有效 KV 重复变化，应先检查这条投影和写出路径，同时保留内存破坏的可能；
单凭完整 hash 不能认定 V 投影、RMSNorm、ScatterElements 或某个计算精度就是根因。

现在回放会把每个 `[B,H,S,D]` FP16 KV 输出拆成三个行区间。对于当前
`start_position=0, valid_rows=17` 的案例，源码为 512 的逻辑容量多留 64 行，
Draft KV 实际 `S=576`（用户的全零输入 hash 与 `[1,8,576,128]` FP16 对应）：

| 报告区间 | 行范围，左闭右开 | 含义 |
|---|---|---|
| `valid_prefix` | `[0,17)` | 应供后续 Draft 读取的逻辑缓存；变化会使回放失败 |
| `written_padding` | `[17,64)` | 固定 64 行图实际写出、但不在本轮逻辑前缀内的部分 |
| `untouched_tail` | `[64,576)` | Scatter 写入区以外，应该保留输入字节的部分 |

物理长度始终读取 OM tensor ABI，不能用逻辑容量代替。所有区间都逐个 batch/head 取 `S` 维切片。不能把整个 buffer 的前
`valid_rows * heads * head_dim` 个元素当作有效区。长 prompt 的 `valid_prefix`
还包含已提交的旧前缀；分析另外检查 `[0,start_position)` 和未写尾部是否保持原输入。

只需重新编译 C++ runner，继续复用当前 OM 和旧输入快照。构建使用新目录，
然后将 `DRAFT_INPUT_SNAPSHOT` 指向上一份报告的 `input_directory`：

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-draft-kv-audit" \
  --ascendcl-root "$CANN_ROOT" \
  --output "$AI_RUN_DIR/reports/cpp-build-draft-kv-audit.json"
export CPP_RUNNER="$AI_RUN_DIR/build/cpp-draft-kv-audit/qwen35_dflash_acl_runner"

"$MODEL_PYTHON" -B "$REPO_ROOT/tools/debug_draft_om.py" \
  --run-dir "$AI_RUN_DIR" --runner "$CPP_RUNNER" \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --inputs "$DRAFT_INPUT_SNAPSHOT" --workspace private \
  --repetitions 20 --device-id 0 --max-draft-tokens 15
```

已经确认独立工作内存也复现，本轮可以只跑 `private`。工具自动生成
`private-kv-analysis.json`，无需再跑 msprof。重点查看：

- `phase_counts`：各 tensor/区间的变化次数，以及有效 KV 变化次数。
- `examples[].comparison`：第一个差异的 `[batch,head,row,channel]` 坐标、
  FP16 原始位值、变化元素数、最大/平均绝对误差和最大 FP16 ULP 距离。
- `reference_outside_write_vs_input` 和 `examples[].outside_write_vs_input`：
  原输入的旧前缀和未写尾部是否被破坏，包括第 0 次参考输出。

`<report>.replay/outputs/reference/` 保存参考输出，`outputs/differences/` 保存
每个 KV tensor/区间的第一个变化样本，同一次调用的样本文件复用。
最多保留每个 KV tensor 三份变化样本及一份参考，而不是每轮保存整个输出。
数值误差是这些样本的误差，不是所有调用的最大值；非有限数单独计数，不参与误差平均。
读取二进制前验证 SHA256、形状和区间，结果可以在主机上重新分析：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/analyze_draft_replay.py" "$DRAFT_REPLAY_REPORT"
```

旧报告没有输出二进制，无法从 hash 还原数值，必须用新 runner 再回放一次。
此改动只增加诊断，不修改现有 OM、模型精度、正常生成或 msprof 的检查门槛，
也不宣称已修复当前设备上的不稳定问题。

## 16. 单模式运行和部署容量

生成单模式加载计划，再启动 DFlash 生成：

```bash
"$MODEL_PYTHON" -B -m qwen35_dflash.ascend310p prepare-chunk-plan \
  --deployment-manifest "$DEPLOYMENT_MANIFEST" \
  --mode dflash --output "$AI_RUN_DIR/dflash-plan.txt"
read -r DFLASH_PLAN_SHA _ < <(sha256sum "$AI_RUN_DIR/dflash-plan.txt")
read -r PROMPT_TOKEN_IDS < "$AI_RUN_DIR/prompt-ids.csv"

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
的 LM head，以及 Draft dense KV 和图边界状态复制，都可能增加开销，应按实际 msprof 数据评估。

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
