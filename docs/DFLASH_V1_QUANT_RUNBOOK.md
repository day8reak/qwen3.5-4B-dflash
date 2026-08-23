# DFlash V1 量化版：运行与排错指南

这是一份面向第一次接入者的操作手册。目标是让你先看懂数据怎么流动，再按固定顺序运行，
最后能根据报错阶段快速定位问题。

当前 `quant` 分支实现的是：

> **NPU Target 的 Linear 使用已有 W8A8 `QLinear`，DFlash Draft 保持 FP16。**

它不会重新生成量化权重，也不会修改已有 NPU 自定义算子。真实量化 artifact 的解释方式仍由
现有量化工具负责；DFlash 只负责调用、校验、提取 feature、提议 token 和让同一个量化 Target
逐个验证。

## 1. 先看懂整体框架

```text
prompt / prompt.txt
        │
        ▼
本地 tokenizer + chat template
        │
        ├──────────────── ordinary greedy ────────────────┐
        │                                                  │
        ▼                                                  ▼
量化 NPU Target                                      权威 token 序列
  ├─ input provider → 第 0 层 FP16 hidden
  ├─ Linear → QLinear(W8A8 dynamic-per-token)
  ├─ attention / GDN / cache → 原 NPU 路径
  ├─ logits
  └─ 8 层 feature [1,S,20480] FP16
        │
        ▼
FP16 DFlash Draft（6 层，checkpoint 不变）
        │
        ▼
一次并行提出 K 个 proposal
        │
        ▼
同一个量化 NPU Target 逐 proposal 做 fresh full-prefix 验证
        │
        ├─ 连续相同：接受
        └─ 第一个不同：使用 Target token 纠正
        │
        ▼
要求最终 token / EOS / stop reason 与 ordinary greedy 完全一致
```

这里有三个容易混淆的边界：

1. **Target 被量化，Draft 没有量化。** Draft 仍读取 FP16 feature，并使用 FP16
   embedding/LM head。
2. **量化只替换 Target 的 Linear 模块。** RMSNorm、RoPE、attention、GDN、KV/GDN state 和
   feature collector 继续走已有 Target 实现。
3. **V1 每次 Target 调用都从新状态重算完整前缀。** 它先保证正确性，不做投机 KV/GDN
   commit/rollback。

## 2. 两种 artifact 不要混用

本流程会出现两个不同目录：

| 名称 | 谁生成 | 谁读取 | 用途 |
|---|---|---|---|
| 部署量化 artifact | 现有量化工具 | `--target-quantizer` | 在 NPU Target 中创建 `QLinear` |
| W8A8 仿真 artifact | `preflight_target_quant` | CPU/CUDA framework Target | 导出真实 `W_q/scale`，复现 Linear 公式 |

不要把第二个目录传给 `--target-quant-artifact`。它只用于 CPU/CUDA correctness 诊断，不能替代
部署量化 artifact。

## 3. 代码各自负责什么

- `models/dflash_v1/run_npu.py`：完整量化 NPU DFlash 入口。
- `models/dflash_v1/preflight_target_quant.py`：只加载量化 Target 的低成本预检，不读 Draft
  checkpoint。
- `models/dflash_v1/target_quant.py`：量化 callback ABI、Linear 拓扑和 Draft FP16 共享权重合同。
- `models/internal_dflash_bridge.py`：调用量化器和 input provider；每次 Target forward 新建
  KV/GDN state。
- `models/dflash_v1/w8a8_emulation.py`：CPU/CUDA 上复现 `QLinear` 的 W8A8 数学公式。
- `models/dflash_v1/dflash_qwen_adapter_v1.py`：加载 Target/Draft、执行严格 greedy 对照并生成
  报告。
- `models/dflash_v1/diagnose_acceptance.py`：接受率、首个 proposal 分叉和分层诊断。

## 4. 你只需要接两个已有函数

DFlash 不知道部署量化 artifact 的私有格式，因此需要两个 `MODULE:FUNCTION` callback。

### 4.1 Quantizer

如果已有量化函数就是：

```python
def quant_model(model, artifact_path):
    ...
    return model
```

命令可以直接写：

```text
--target-quantizer your_module:quant_model
```

也支持扩展签名：

```python
def quantize_target(model, artifact_path, *, device, output_dtype):
    ...
    return model
```

转换完成后会逐项校验：

- 转换前每个 `nn.Linear` 的路径、输入维、输出维和 bias；
- 转换后的完整 `QLinear` 路径集合；
- `W_q` 必须是 INT8 `[in_features,out_features]`；
- `scale` 必须是一维、有限值，长度为 `1` 或 `out_features`；
- 未量化的 Linear 必须仍在原路径，shape/bias 不变；
- 所有量化 buffer 在请求的 NPU 上；
- Draft-facing embedding 和 LM head 仍是 `[248320,2560]` FP16。

NPU 装配接受当前 `QLinear` 支持的浮点 scale；导出本分支的 CPU/CUDA 仿真 artifact 时还会
进一步要求 scale 为 float32。若导出在这里失败，不要静默转换 dtype，应先确认普通量化推理给
`npu_quant_matmul` 的实际 scale 合同，再决定是否需要升级 artifact schema。

默认返回 `model` 或 `None` 表示“转换全部原 Linear”。如果 artifact 只量化一部分 Linear，必须
返回 `TargetQuantizationResult` 并列出准确路径；不能让 DFlash 猜测范围。

### 4.2 Target input provider

Bridge 直接调用 Target execution model，不绕回普通推理 wrapper。因此还需要一个函数返回普通
量化推理在 decoder 第 0 层真正消费的最终 hidden：

```python
def build_target_inputs(
    model_wrapper,
    input_ids,
    *,
    artifact_path,
    device,
    output_dtype,
):
    # 复用普通量化推理已有的 embedding / scale / 反量化步骤。
    inputs_embeds = existing_input_path(...)
    return inputs_embeds
```

固定输出合同：

```text
shape  = [1,S,2560]
dtype  = torch.float16
device = 请求的 npu:N
value  = 全部 finite
```

`S` 是 Bridge 右补齐后的物理长度；当逻辑长度大于 1 时会补到 64 的整数倍。provider 不能只处理
原始逻辑长度。若普通量化推理使用 INT8 embedding + scale，provider 必须复用相同的 scale 语义；
不要在这里自行猜乘法、除法、offset 或 layout。

## 5. 路径变量

以下命令都从仓库或部署工程根目录执行。先准备：

```bash
set -euo pipefail

export MODEL_PYTHON=/path/to/model/python
export TARGET_DIR=/path/to/Qwen3.5-4B
export DRAFT_DIR=/path/to/Qwen3.5-4B-DFlash
export QUANT_ARTIFACT=/path/to/deployment-quant-artifact
export PROMPT_FILE=/path/to/prompt.txt
export RUN_DIR=/path/to/new-run-directory
export KV_CACHE_MAX_LEN=4096

# 放置 your_quant_bridge.py 的目录；若 callback 已在部署包内，可省略这一项。
export QUANT_CALLBACK_ROOT=/path/to/callback-parent

mkdir -p "$RUN_DIR"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
export PYTHONPATH="$PWD:$QUANT_CALLBACK_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

含义：

- `MODEL_PYTHON`：能够导入 PyTorch、torch_npu、Transformers 和当前模型的 Python 3.10。
- `TARGET_DIR`：本地 Qwen3.5-4B target checkpoint。
- `DRAFT_DIR`：官方 Qwen3.5-4B-DFlash checkpoint。
- `QUANT_ARTIFACT`：普通量化 Target 已使用的部署 artifact。
- `PROMPT_FILE`：固定 UTF-8 文本；报告不保存 prompt 明文。
- `RUN_DIR`：本次报告和导出文件的独立目录，不能放进源码或权重目录。
- `KV_CACHE_MAX_LEN`：与普通 NPU 推理配置相同，且能被 64 整除。

先做便宜检查：

```bash
test -x "$MODEL_PYTHON"
test -d "$TARGET_DIR"
test -d "$DRAFT_DIR"
test -e "$QUANT_ARTIFACT"
test -f "$PROMPT_FILE"

"$MODEL_PYTHON" -B - <<'PY'
import importlib
import torch
import torch_npu  # noqa: F401 - registers torch.npu

for spec in (
    "your_quant_bridge:quantize_target",
    "your_quant_bridge:build_target_inputs",
):
    module_name, function_name = spec.split(":", 1)
    function = getattr(importlib.import_module(module_name), function_name)
    print(spec, "callable=", callable(function))
print("torch=", torch.__version__)
print("npu_available=", torch.npu.is_available())
PY
```

## 6. 第一步：只跑量化 Target 预检

先把固定 prompt 转成一小段合法 token ID，或者直接从普通推理日志中选择同一段。假设：

```bash
export PROMPT_IDS=123,456,789
```

运行：

```bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.preflight_target_quant \
  --target-dir "$TARGET_DIR" \
  --prompt-ids "$PROMPT_IDS" \
  --device npu:0 \
  --kv-cache-max-len "$KV_CACHE_MAX_LEN" \
  --target-quantizer your_quant_bridge:quantize_target \
  --target-quant-artifact "$QUANT_ARTIFACT" \
  --target-input-provider your_quant_bridge:build_target_inputs \
  --export-w8a8-emulation-artifact "$RUN_DIR/w8a8-linear-artifact" \
  --report "$RUN_DIR/target-quant-preflight.json"
```

它执行 8 次有界 Target 调用，检查：

1. 同一前缀连续两次 ordinary logits 可重复；
2. 打开 feature 不改变 logits；
3. feature 模式的 logits/features 可重复；
4. 插入一个不同长度前缀后，再跑原前缀仍相同；
5. input provider 每次恰好成功一次；
6. QLinear 路径、shape、scale、device 和未替换 Linear 拓扑完整；
7. 可选导出真实 `W_q/scale` 给 CPU/CUDA 公式仿真。

核对报告：

```bash
"$MODEL_PYTHON" -B - "$RUN_DIR/target-quant-preflight.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["status"] == "PASS_TARGET_QUANT_ASSEMBLY_AND_BOUNDED_PREFIX_PROBES"
assert report["draft_checkpoint_read"] is False
quant = report["target_quantization_final"]
assert quant["status"] == "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM"
assert quant["scheme"] == "w8a8_dynamic"
assert quant["qlinear_count"] > 0
assert quant["linear_topology_validation"] == "PASS_EXACT_PATH_SHAPE_BIAS"
assert quant["quantized_weight_layout"] == "K_by_N"
assert quant["input_provider_failures"] == 0
assert report["bounded_probes"]["status"] == "PASS_BOUNDED_TARGET_PROBES"
export = report["w8a8_emulation_export"]
assert export["status"] == "PASS_EXPORTED_Q_LINEAR_BUFFERS_NO_NUMERICAL_CLAIM"
print("TARGET_QUANT_PREFLIGHT_REPORT_PASS")
PY
```

这个 PASS **不证明**：普通增量量化推理等价、真实 NPU QLinear 数值与 CPU 公式相同、完整
DFlash token 零差异、接受率或性能。

## 7. 第二步：CPU 复现同一份 W8A8 Linear

CPU 路线读取上一步导出的 `w8a8-linear-artifact`，在 framework Target 中把对应文本 Linear
替换成公式仿真模块。它不会量化 Draft，也不会仿真量化 embedding/input provider。

```bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_qwen_adapter_v1 \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-mode chat \
  --enable-thinking \
  --device cpu \
  --dtype float16 \
  --max-new-tokens 16 \
  --max-draft-tokens 4 \
  --eos-token-id 248044 \
  --target-w8a8-emulation-artifact "$RUN_DIR/w8a8-linear-artifact" \
  --report "$RUN_DIR/cpu-w8a8-dflash.json"
```

CPU 公式是：

```text
S_x = max(abs(X), dim=-1) / 127
X_q = clamp(round(X / S_x), -127, 127).to(int8)
A   = int32(X_q) @ int32(W_q)
Y   = (A.float() * S_w) * S_x
Y   = Y.to(float16)
```

CPU 优先使用 exact INT8→INT32 `torch._int_mm`；旧 PyTorch 没有该 kernel 时才回退显式 INT32
matmul。它是 correctness 路线，不是量化性能实现。完整 4B CPU 推理本身仍可能很慢。

报告至少检查：

```bash
"$MODEL_PYTHON" -B - "$RUN_DIR/cpu-w8a8-dflash.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

emulation = report["target_w8a8_emulation"]
assert emulation["status"] == "PASS_FORMULA_ASSEMBLY_NO_REAL_NPU_PARITY"
assert emulation["scope"] == "target_text_linear_only"
assert emulation["linear_output_dtype"] == "torch.float16"
assert emulation["draft_quantization"] == "DISABLED_FP16"
assert report["strict_greedy_exact_match"] is True
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["dflash_execution_gate"]["draft_round_executed"] is True
print("CPU_W8A8_DFLASH_FRAMEWORK_PASS")
PY
```

这个 PASS 证明 CPU 上同一仿真 Target 的 ordinary 与 DFlash 调度自洽；它仍不能证明真实 NPU
`npu_dynamic_quant/npu_quant_matmul` 的 rounding、scale 解释和 accumulation 与公式完全一致。

## 8. 第三步：同 activation 的单层 NPU/CPU 对照

这是区分“Linear 公式不一致”和“整网其他部分不一致”的最有效步骤。

选择第一个 QLinear，在一次固定 Target 调用中保存它的输入 `x_npu` 和真实输出 `npu_output`。
然后用同一次调用的 `W_q/scale` 在 CPU 计算：

```python
import torch
from models.dflash_v1.w8a8_emulation import (
    compare_formula_output,
    emulate_w8a8_linear,
)

formula_output = emulate_w8a8_linear(
    x_npu.detach().cpu(),
    qlinear.W_q.detach().cpu(),
    qlinear.scale.detach().cpu(),
    output_dtype=torch.float16,
)
comparison = compare_formula_output(
    npu_output.detach().cpu(),
    formula_output,
)
print(comparison)
```

必须使用**同一 activation**，不能分别跑两次整网后拿不同输入比较。建议按实际 forward 顺序：

1. 第 0 层第一个 Linear；
2. 第一个 GDN/attention 输出投影；
3. 第一个 MLP；
4. 中间层；
5. LM head。

判断方法：

- 第一层已明显分叉：查 activation dynamic-quant rounding、scale dtype/layout、input provider。
- 所有单层 same-input 都一致，但整网分叉：查 embedding、RMSNorm、attention/GDN、state 或
  64-token padding。
- 单层只在特定 K/N shape 分叉：查对应 QLinear artifact 的转置和 scale 长度。

## 9. 第四步：完整量化 NPU DFlash

前面三步通过后，再加载 1.27 GB Draft checkpoint：

```bash
set -euo pipefail

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.run_npu \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-mode chat \
  --enable-thinking \
  --device npu:0 \
  --kv-cache-max-len "$KV_CACHE_MAX_LEN" \
  --max-new-tokens 64 \
  --max-draft-tokens 16 \
  --target-quant-mode w8a8_dynamic \
  --target-quantizer your_quant_bridge:quantize_target \
  --target-quant-artifact "$QUANT_ARTIFACT" \
  --target-input-provider your_quant_bridge:build_target_inputs \
  --report "$RUN_DIR/npu-quant-dflash.json"
```

`run_npu` 固定 FP16、EOS `248044`、proposal K 范围 `1..16`、fresh full-prefix 和 sequential
verifier。量化模式不会切换 Draft backend 或量化 Draft。

报告核心检查：

```bash
"$MODEL_PYTHON" -B - "$RUN_DIR/npu-quant-dflash.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["strict_greedy_exact_match"] is True
assert report["verification_mode"] == "sequential_isolated_prefix"
assert report["feature_capture_zero_impact"] is True
assert report["bounded_full_prefix_repeatability"] is True
assert report["operator_fallback_enabled"] is False
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["ordinary"]["reached_eos"] == report["dflash"]["reached_eos"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]

gate = report["dflash_execution_gate"]
assert gate["draft_round_executed"] is True
assert gate["draft_calls"] > 0
assert gate["target_feature_calls"] > 0
assert gate["target_verify_calls"] > 0

quant = report["target_integration"]["isolation"]["bridge_runtime"]["target_quantization"]
assert quant["scheme"] == "w8a8_dynamic"
assert quant["status"] == "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM"
assert quant["linear_topology_validation"] == "PASS_EXACT_PATH_SHAPE_BIAS"
assert quant["input_provider_calls"] == quant["input_provider_successes"]
assert quant["input_provider_failures"] == 0
print("NPU_QUANT_DFLASH_FRAMEWORK_GATE_PASS")
PY
```

注意：最后这行只代表 framework/调度门禁。还需要设备 trace 证明没有 CPU fallback，才可以称为
真实 NPU 路线通过；接受率和性能也必须单独测量。

## 10. 四类比较分别回答什么

| 比较 | 回答的问题 | 不能回答的问题 |
|---|---|---|
| CPU 公式 vs INT64 oracle | 仿真里的整数累加是否正确 | NPU kernel 是否同公式 |
| 同 activation：NPU QLinear vs CPU 公式 | dynamic quant、scale、matmul 数值是否一致 | 整网 state/attention 是否一致 |
| 普通增量量化 Target vs fresh full-prefix Target | Bridge 是否等价于正常量化推理 | Draft 接受率是否正常 |
| 量化 ordinary vs 量化 DFlash | speculative 调度是否保持严格 greedy | 量化相对 FP16 的业务精度 |

不要用后一个 PASS 替代前一个。例如 ordinary 与 DFlash 使用同一条错误 full-prefix Target 路线时，
两者仍可能互相一致，所以必须单独对照普通增量量化推理。

## 11. 按报错阶段定位

| 报错/现象 | 最可能原因 | 先做什么 |
|---|---|---|
| callback import/ABI 失败 | `PYTHONPATH` 或 `MODULE:FUNCTION` 写错 | 单独运行第 5 节 import 检查 |
| `QLinear coverage differs` | converter 范围与声明不一致 | 打印转换前后模块路径；部分量化返回显式 result |
| `W_q must use [in_features,out_features]` | 权重转置或 artifact 对错模型 | 核对原 Linear `[out,in]` 与 `W_q [in,out]` |
| `scale must ... 1 or out_features` | scale 粒度/layout 不匹配 | 查普通量化 converter 如何传 `npu_quant_matmul` |
| `post-conversion nn.Linear topology differs` | 漏替换、误删或新增了 Linear | 对照报告的 missing/unexpected 路径 |
| Draft embedding/LM head 不是 FP16 | converter 原地覆盖了共享权重 | 在 result 中显式提供独立 FP16 Draft modules |
| provider shape/dtype/device 失败 | 返回了 INT8 embedding、scale tuple 或逻辑长度 | 返回 decoder 第 0 层最终 FP16 hidden，处理物理 padding |
| P→P 或 P→Q→P 不一致 | KV/GDN/异步状态残留 | 检查 fresh cache、同步和普通量化 wrapper 的状态字段 |
| feature 开关改变 logits | collector 插入点或返回 ABI 有副作用 | 先停 Draft，只比较 feature off/on Target |
| CPU artifact topology mismatch | 导出 artifact 与当前 framework checkpoint 不同 | 用同一 target revision 重新导出，不按名字强行套用 |
| same-activation 第一层分叉 | rounding、scale 或 weight layout | 保存第一层 x/W_q/scale/output 做最小对照 |
| strict greedy mismatch | verifier/状态/Target 路径不一致 | 找第一轮、第一 proposal 的 Target Top-1 分叉 |
| strict greedy 通过但接受率低 | Draft proposal 质量或 workload 难度 | 用 `diagnose_acceptance` 扫 K=1/4/8/16 |
| CPU 很慢 | correctness-only 的 4B full-prefix 路线 | 先短 prompt、`max-new-tokens=2`；不要据此评性能 |

## 12. 接受率低时怎么查

正确性通过后再运行：

```bash
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.diagnose_acceptance \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --prompt-file "$PROMPT_FILE" \
  --prompt-mode chat \
  --enable-thinking \
  --device cpu \
  --dtype float16 \
  --eos-token-id 248044 \
  --proposal-counts 1,4,8,16 \
  --acceptance-rounds 16 \
  --trace-draft-layers \
  --target-w8a8-emulation-artifact "$RUN_DIR/w8a8-linear-artifact" \
  --report "$RUN_DIR/cpu-w8a8-acceptance.json"
```

优先看：

- `K=1 first_proposal_accuracy`：第一 proposal 都不准时，先查 feature/Draft 输入；
- 第一个分叉 round 和 Draft layer；
- early/middle/late 三段接受长度；
- `mean_theoretical_emitted_per_verify`，而不是只看 `accepted/proposed`；
- 同一 prompt 的 FP16 Target 与 W8A8 Target 报告差异。

如果最终 token 完全一致，只是接受率下降，这是性能/提议质量问题，不是生成正确性失败。此时应
先比较量化前后 8 层 feature 和 Draft 每层输出，不要先修改 verifier。

## 13. 提交 bug 时最小信息

为避免泄露 prompt、路径和权重，只需要提供脱敏后的：

1. 完整 traceback（可以替换绝对路径前缀）；
2. `torch/torch_npu/transformers` 版本和设备型号；
3. `target_quantization_final` 或完整报告中的 `target_quantization` 子对象；
4. `qlinear_count`、missing/unexpected 路径；
5. input-provider calls/successes/failures；
6. 第一处分叉的层名、shape、dtype、max/mean absolute error；
7. ordinary/DFlash 的 token 是否零差异、首个分叉 round；
8. 是否启用 fallback，以及实际 device trace 结论。

不要上传模型权重、量化 artifact、prompt 明文、tokenizer 私有文件或包含凭据的环境变量。

## 14. 什么情况下算完成

完整闭环至少需要：

- Target-only 预检 PASS；
- QLinear 精确拓扑/shape/scale 门禁 PASS；
- same-activation NPU/CPU Linear 对照达到冻结阈值；
- 普通增量量化 Target 与 fresh full-prefix Target token 对齐；
- 量化 ordinary 与量化 DFlash token/EOS/stop reason 零差异；
- 至少一个真实 Draft/feature/verify round；
- NPU trace 证明无 CPU fallback；
- 多 prompt 接受率和稳定性结果；
- 单独的性能测量。

当前仓库能够提供代码和前置门禁，但没有随仓库分发真实量化 artifact 或 NPU 运行结果。因此，
CPU 公式与合成拓扑验证可以本地完成；真实 NPU numerical parity、无 fallback、接受率和性能必须
在部署设备上执行后才能提升状态。
