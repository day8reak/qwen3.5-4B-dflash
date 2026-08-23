# DFlash V1 NPU Quant Target 适配分析与实施方案

状态：`quant` 分支已经加入量化装配合同、Bridge 接线和 CLI；尚未取得真实 NPU 量化
Target parity/DFlash 零差异结果，因此不得视为量化运行通过。

## 1. 目标与非目标

本实验希望让 DFlash V1 使用已经量化的 NPU Target，同时继续满足：

- 普通量化 Target greedy 是本实验的权威 token 序列；
- 量化 Target + DFlash 的 token、EOS、stop reason 与普通量化 Target 完全一致；
- feature 开关不改变量化 Target logits；
- 每次 full-prefix Target 调用仍使用 fresh KV/GDN state；
- Draft checkpoint、Draft 网络结构和 sequential verifier 不因 Target 量化而改变。

本轮不同时量化 Draft。Target 量化和 Draft 量化是两个独立的近似边界；同时修改会让误差来源
无法定位。

## 2. 当前源码已经具备什么

`models/modeling_qwen3_5_hiai_nd.py` 已包含 `QLinear`：

```text
FP16 activation
  → npu_dynamic_quant
  → INT8 activation + per-token scale
  → npu_quant_matmul(INT8 activation, quantized weight, weight scale,
                    pertoken_scale, output_dtype=FP16)
  → FP16 layer output
```

`QLinear` 保存：

- `W_q`：部署量化工具准备的量化权重；
- `scale`：与该权重匹配的 scale；
- `idx`：量化工具的层索引；
- 每次 forward 动态量化 activation；
- `npu_quant_matmul` 输出 FP16。

因此 Target 的 decoder hidden 和 DFlash feature 仍可保持 FP16，feature ABI
`[B,S,20480]` 不需要改成 INT8。

## 3. `v1-r1` 的量化断点与 `quant` 分支修复

### 3.1 Bridge 没有应用量化转换

`v1-r1` 的 `models/internal_dflash_bridge.py` 只负责：

1. 构造 `Qwen3_5ForCausalLMWrapper`；
2. 取得 `.model`；
3. 新建 fresh hybrid state；
4. 执行 full-prefix Target。

`quant` 分支现在通过显式 `--target-quantizer` 和 `--target-quant-artifact` 调用已有量化器。
转换前会冻结全部 `nn.Linear` 路径；已有两参数量化器若直接返回 `nn.Module`（或原地转换后
返回 `None`），转换后必须一一对应为 `QLinear`。有意排除部分 Linear 的量化器则必须返回
`TargetQuantizationResult`，明确给出完整预期路径，不能靠“检测到一个 QLinear”通过。

### 3.2 Bridge 绕过了正常 wrapper 的输入预处理

Bridge 为了精确控制 KV/GDN state，直接调用 `.model` execution model。`v1-r1` 自行执行：

```python
inputs_embeds = get_input_embeddings()(input_ids)
```

如果量化 Target 的正常推理还使用量化 embedding table、embedding scale 或 wrapper 内的输入
预处理，只替换 linear 仍不等价。`quant` 分支因此强制 `--target-input-provider`；它必须返回
量化普通推理在第 0 层真正消费的最终 FP16 hidden，shape 为 `[1,S,2560]`。返回原始 INT8
embedding、未消费的 scale 或含义不明的 tuple 都会失败。

### 3.3 Draft 仍依赖浮点 embedding 和 LM head

`Qwen35DFlashFullPrefixAdapter` 会复用 Target 的：

- `embed_tokens.weight`：生成 `[anchor, MASK × K]` 的 Draft block embedding；
- `lm_head.weight`：把 Draft hidden 转成 proposal Top-1。

当前合同要求两者都是 `[248320,2560]` 的浮点 Tensor，并与 Draft 的 device/dtype 一致。

所以不能把 embedding 或 LM head 原地替换成只暴露 `W_q/scale` 的 `QLinear`，否则 Draft 失去
共享权重。量化 Target 路线必须保留或单独提供 FP16 Draft-side embedding/LM-head 权重。

### 3.4 当前 formal NPU 合同明确是非量化

`run_npu` 默认仍锁定 FP16、非量化 Target。量化模式必须显式选择；装配结果、完整 QLinear
路径、callback identity 和 input-provider 调用计数写入最终 report 的
`target_integration.isolation.bridge_runtime.target_quantization`。

## 4. 当前量化范围

从模块替换的角度看，当前转换器只把约定范围内的 `nn.Linear` 换成 `QLinear`。默认的
“全部 Linear”合同覆盖 attention 的 Q/K/V/O projection、GDN 的输入/门控/输出 projection、
MLP 的 gate/up/down projection，以及转换器实际遍历到的 LM head。RMSNorm、RoPE、激活、
residual、GDN 核心、fused attention、CacheUpdate 和 KV/GDN state 不做量化替换，继续使用
原有 NPU 路径。

从完整推理链看则不应简称为“只有 Linear”：量化部署若还使用 INT8 embedding table/scale，
它由独立 input-provider 复用，最终交给 decoder 第 0 层的仍是 FP16 hidden。这个输入步骤不是
`QLinear` 替换，但属于量化 Target 路线的一部分。

### 4.1 原始与替换公式

原 FP16 linear：

```text
Y_fp16 = X_fp16 · W_fp16^T
```

实验替换为 Target W8A8 dynamic-per-token linear：

```text
X_q, S_x = DynamicQuantInt8(X_fp16)
Y_quant  = QuantMatMul(X_q, W_q, S_w, pertoken_scale=S_x,
                      output_dtype=FP16)
Y_quant ≈ Y_fp16
```

`W_q/S_w` 的布局和 scale 组合由现有量化 artifact 及其配套转换器负责；DFlash 包不重新解释、
转置或生成这些数据。

### 4.2 本实验保留 FP16 的部分

- Target input embedding 对 Draft 可见的权重；
- Target LM head 对 Draft 可见的权重；
- Target RMSNorm、attention、GDN 输出和 feature ABI；
- 整个 6 层 Draft；
- sequential verifier；
- full-prefix fresh state、64-token 对齐和同步逻辑。

量化转换器可替换 Target 内符合 artifact 合同的 linear，包括量化部署实际会替换的 LM head。
Bridge 在转换前保留 Draft-facing FP16 embedding/LM-head view；如果量化器会原地改写这些
Tensor，则必须在 `TargetQuantizationResult` 中另外提供独立 FP16 Draft-side 模块。

### 4.3 为什么 Draft 暂时保持 FP16

量化 Target 已经会改变 feature 和 Target logits。Draft 也量化会同时引入第二组 proposal 误差，
无法区分接受率变化来自 Target、feature 还是 Draft。本分支先只验证 “quant Target + FP16 Draft”。

## 5. 运行接口

### 5.1 CLI

`run_npu` 的显式参数：

```text
--target-quant-mode disabled|w8a8_dynamic
--target-quantizer MODULE:FUNCTION
--target-quant-artifact /path/to/quantized-target-artifact
--target-input-provider MODULE:FUNCTION   # 量化 embedding/input 路线使用
```

规则：

- 默认 `disabled`，保持 `v1-r1` 行为不变；
- `w8a8_dynamic` 必须同时提供 quantizer、artifact 和 input provider；
- CLI 会在读取 Draft 大权重前导入两个 callback 并校验其支持的调用签名；
- 即使量化 Target 使用普通 FP16 embedding，也必须通过 provider 显式复用该路线；
- 量化参数不能对 CPU/CUDA 路线生效；
- 报告不能只写 `dtype=float16`，还要写 Target quantization profile。

### 5.2 Quantizer callback

建议 ABI：

```python
def apply_target_quantization(
    execution_model: torch.nn.Module,
    artifact_path: str,
    *,
    device: torch.device,
    output_dtype: torch.dtype,
) -> torch.nn.Module | None | TargetQuantizationResult:
    ...
```

职责：

- 可直接复用已有 `quant_model(model, artifact_path)` 两参数函数；
- 也可接收扩展的 keyword-only `device/output_dtype`；
- 使用和正常量化 Target 完全相同的 artifact 解释方式；
- 替换约定范围内的 `nn.Linear` 为当前 HIAI source 的 `QLinear`；
- 把量化 buffers 放到请求 NPU；
- 保留 FP16 embedding 和 LM head；
- 返回实际 execution model；原地转换可返回 `None`；
- 默认合同要求转换前的全部 `nn.Linear` 变成 `QLinear`；若实际范围不同，返回显式
  `TargetQuantizationResult(expected_qlinear_paths=...)`。

DFlash Bridge 只负责加载 callback、调用一次并审计结果，不复制量化工具的 checkpoint 解析代码。

### 5.3 Target input provider

建议 ABI：

```python
def build_quant_target_inputs(
    model_wrapper: torch.nn.Module,
    input_ids: torch.Tensor,
    *,
    artifact_path: str,
    device: torch.device,
    output_dtype: torch.dtype,
) -> torch.Tensor | dict[str, torch.Tensor]:
    ...
```

返回值至少包含：

```text
inputs_embeds: [1,S,2560] FP16 NPU Tensor
```

简单的已有 provider 可以只接收 `(model_wrapper, input_ids)`；扩展 ABI 还会收到
`artifact_path/device/output_dtype`。它必须产生正常量化 Target 在第 0 层真正消费的同一
FP16 hidden。若部署量化 embedding 使用
INT8 table + scale，scale 的语义和反量化/融合操作由 provider 复用现有实现；Bridge 不猜
`multiply`、`divide`、offset、group 或 layout 规则。

量化模式始终要求显式 provider；如果普通量化 Target 使用 FP16 embedding，provider 可以直接
调用同一个 embedding，但仍需显式声明，避免因误配而静默走回非量化输入路径。

## 6. Bridge 侧设计

`InternalDFlashTarget` 增加：

- `target_quant_mode`；
- 可选 `target_input_provider`；
- `dflash_target_quantization_audit`；
- input provider calls/successes/failures 计数；
- QLinear count、quantized buffer dtype/device 审计；
- float embedding/LM-head preservation 审计。

每个 full-prefix 调用的顺序：

```text
logical input_ids
  → 64-token physical padding（S>1）
  → quant input provider 或普通 FP16 embedding
  → fresh KV/GDN state
  → quantized HIAI Target forward
  → slice 回 logical S
  → NPU synchronize
  → 只返回 logits/features
```

input provider 必须处理物理 padding 后的 IDs；Bridge 仍在输出侧裁回真实长度。

## 7. 必须 fail-closed 的静态/运行时检查

启用量化时至少检查：

1. quant artifact 是明确的常规文件或目录，但不能是 symlink；
2. quantizer callback 的 module/function identity 被写入报告；
3. 转换后 `QLinear` 数量大于 0；
4. 每个 `QLinear.W_q` 是量化整数 Tensor；
5. 每个 `QLinear.scale` 是浮点 Tensor；
6. `W_q/scale` 位于请求 NPU，或由转换器明确声明合法的 lazy placement；
7. 每个 `QLinear` 输出仍是 FP16；
8. Target embedding/LM head 仍是 `[248320,2560]` FP16；
9. input provider 输出是 `[1,S,2560]` FP16 NPU Tensor；
10. feature 仍是 `[1,S,20480]` FP16；
11. quant 模式下不允许缺失 callback 后退回非量化 Target；
12. 报告中的 quant mode、QLinear count、provider calls 与 Target forward calls 对账。

实际 QLinear 名单/数量应由配套 artifact 的转换规则冻结；在拿到该规则前不能仅凭 “检测到一个
QLinear” 就把整个 Target 标成已量化。

## 8. 数值风险

### 8.1 Target logits 改变

W8A8 是近似计算。即使 `output_dtype=FP16`，输出也不是原 FP16 linear 的逐 bit 等价结果。
量化误差会逐层传播到最终 logits，可能改变普通 Target greedy token。

### 8.2 DFlash feature 改变

8 层 feature 来自量化 decoder，所以 Draft 的 K/V context 也会变化。即使普通量化 Target token
保持不变，proposal 和接受率仍可能变化。

### 8.3 embedding 路线不一致

如果 DFlash Bridge 使用 FP16 embedding，而正常量化推理使用另一套量化 embedding/input
预处理，两条 Target 路径从第 0 层就不同。此时后续 strict-greedy 自洽也不能证明它等价于正常
量化推理。因此 input-provider parity 是硬门禁，不是可选性能优化。

### 8.4 共享 LM head

当前 Draft Top-1 使用 FP16 Target LM head。如果量化部署把 LM head 也替换成量化模块，必须
另外保留 FP16 head，或单独设计并验证 Quant Top-1 原语；本实验不隐式选择其中一种。

## 9. 分层验证计划

### 9.1 非量化回归

在 `quant` 分支不传量化参数时，必须保持当前 `v1-r1`：

- CLI 参数和默认行为不变；
- NPU Bridge 仍使用普通 embedding；
- ordinary/DFlash strict-greedy 门禁不变；
- quant audit 明确为 `disabled`。

### 9.2 量化装配合同测试

使用小型 fake Target/callback 验证：

- quantizer 恰好调用一次；
- artifact/callback 缺失时在 Target forward 前失败；
- input provider 每次 full-prefix forward 调用一次；
- physical padding length 正确传给 provider；
- provider 失败时不执行 Target；
- QLinear/embedding/LM-head audit 正确；
- quant metadata/counters 能进入最终报告。

这些是 CPU/meta 合同测试，不是 NPU 数值证据。

### 9.3 真实 NPU Target parity

使用同一 prompt、同一量化 artifact 和同一输入预处理，先比较：

```text
正常量化增量 Target
vs
DFlash Bridge 量化 full-prefix Target
```

逐步检查：

- bootstrap token；
- 每个 decode step 的 Target Top-1；
- feature layer 1/5/9/13/17/21/25/29；
- final feature `[B,S,20480]`；
- P→P 和 P→Q→P；
- padding 到 64 前后的逻辑行。

这一步通过前不运行接受率结论。

### 9.4 DFlash correctness

以普通量化 Target 为 oracle，要求：

- quant ordinary 与 quant DFlash generated token IDs 零差异；
- EOS/stop reason 完全相同；
- feature 开关 Top-1 零差异；
- 至少执行一个 Draft/feature/sequential verify round；
- 无 CPU fallback；
- quantizer/provider/prepare/forward/synchronize 计数全部对账。

### 9.5 量化相对 FP16 的精度

量化 Target 是近似实验，不能用 “quant DFlash 与 quant ordinary 自洽” 替代量化精度评估。
还要在冻结 prompt 集上比较 FP16 ordinary 与 quant ordinary：

- 首个 token 分叉位置；
- EOS/stop reason；
- 生成文本；
- 8 层 feature 误差；
- logits Top-1 和 margin；
- DFlash acceptance/emitted-per-verify 变化。

初始 admission 建议使用零 token mismatch 的严格 smoke gate；如果真实量化 Target 本身与 FP16
存在业务允许的差异，需要另行冻结任务级质量阈值，不能事后放宽 DFlash 门禁。

## 10. 回滚与分支边界

- 所有实验只在 `quant` 分支；
- `main` 和 `v1-r1` 不修改；
- 默认 `target_quant_mode=disabled`；
- 删除量化参数即可回到当前 FP16 Target 路线；
- 量化 Target 未通过真实设备 parity 前，不创建新的正式 release tag。

## 11. 已确认并实施的第一阶段边界

第一项实现是：

> 在 `quant` 分支加入显式 `w8a8_dynamic` Target 模式。使用部署方已有 quantizer 解释现有
> quant artifact，把 artifact 约定范围内的 linear 替换为 `QLinear`；保持 Draft-facing
> embedding、LM head、所有 Target 输出和整个 DFlash Draft 为 FP16。若正常量化 Target 使用
> 独立 embedding/input 预处理，则通过必需的 input-provider callback 复用该路径。保持 V1
> full-prefix fresh state、64-token 对齐、sequential verifier 和所有零 token 差异门禁不变。

该实验会修改 Target 数学，是近似边界。预期收益是减少 Target linear 的权重带宽/计算成本；
风险是 Target token、feature 和接受率变化。回滚是关闭量化参数或删除 `quant` 分支，`main` 不受
影响。

## 12. 运行入口

量化器和 input provider 必须来自同一套已跑通的普通量化 Target，实现后运行：

```bash
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --prompt-file "$PROMPT_TXT" \
  --prompt-mode chat \
  --device npu:0 \
  --kv-cache-max-len "$KV_CACHE_MAX_LEN" \
  --max-new-tokens 64 \
  --max-draft-tokens 16 \
  --target-quant-mode w8a8_dynamic \
  --target-quantizer your_quant_bridge:quantize_target \
  --target-quant-artifact "$QUANT_ARTIFACT" \
  --target-input-provider your_quant_bridge:build_target_inputs \
  --report "$RUN_DIR/dflash-quant-report.json"
```

`quantize_target` 可以直接指向现有两参数量化函数。如果它量化全部 Linear，可以直接返回模型；
若范围不同，使用：

```python
from models.dflash_v1.target_quant import TargetQuantizationResult

def quantize_target(model, artifact_path):
    converted = existing_quant_model(model, artifact_path)
    return TargetQuantizationResult(
        execution_model=converted,
        expected_qlinear_paths=EXACT_ARTIFACT_LINEAR_PATHS,
        profile={"scheme": "existing_w8a8_dynamic"},
    )
```

`build_target_inputs` 必须复用普通量化推理已有的 embedding/scale 语义，并返回完成反量化或
等价预处理后的 FP16 `[1,S,2560]`；DFlash Bridge 不会猜 scale 是乘、除或其他布局规则。
