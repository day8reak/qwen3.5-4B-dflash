# 内部服务器 NPU 直接运行流程

本流程假定内部 inference 已能用 `modeling_qwen3_5_hiai_nd.py` 在 NPU 正常生成文本，
并且该文件已经直接集成 DFlash feature route。本版本不再 patch、生成或覆盖 modeling 源码。

## 1. 三个路径变量

```bash
export DFLASH_REPO=/path/to/qwen3.5-4B-dflash
export INTERNAL_ROOT=/path/to/internal-inference
export MODEL_PYTHON=/path/to/internal-inference环境/bin/python
```

- `DFLASH_REPO`：克隆本 GitHub 仓库后的根目录，下面应有 `models/dflash_v1/`。
- `INTERNAL_ROOT`：原 inference 工程根目录，下面应有 `models/modeling_qwen3_5_hiai_nd.py`。
- `MODEL_PYTHON`：原 inference 实际启动时使用的 Python。激活环境后可用
  `command -v python` 查看；不要另建一套 Python。

先核对，不要凭路径名猜：

```bash
set -euo pipefail
test -f "$DFLASH_REPO/models/dflash_v1/run_npu.py"
test -f "$INTERNAL_ROOT/models/modeling_qwen3_5_hiai_nd.py"
test -x "$MODEL_PYTHON"
"$MODEL_PYTHON" -V
```

## 2. 目录放法

原 HIAI 模型留在 `models/` 根目录，DFlash 放到它的子包中：

```text
internal-inference/
└── models/
    ├── __init__.py
    ├── modeling_qwen3_5_hiai_nd.py   # 已直接集成 feature route
    ├── configuration_qwen3_5.py
    ├── internal_dflash_bridge.py     # 本仓库已实现，直接复用现有 wrapper
    ├── 原工程其他 HIAI 文件
    └── dflash_v1/                    # 复制本仓库的整个目录
```

首次部署：

```bash
set -euo pipefail
test ! -e "$INTERNAL_ROOT/models/dflash_v1"
test ! -e "$INTERNAL_ROOT/models/internal_dflash_bridge.py"
cp -a "$DFLASH_REPO/models/dflash_v1" "$INTERNAL_ROOT/models/dflash_v1"
cp "$DFLASH_REPO/models/internal_dflash_bridge.py" \
  "$INTERNAL_ROOT/models/internal_dflash_bridge.py"
```

更新时不要把两版文件逐个混合覆盖；按内部策略移走旧 `dflash_v1` 后，再整体复制新目录。
不要覆盖内部工程自己的 `models/__init__.py`、`configuration_qwen3_5.py` 或 HIAI modeling。

## 3. 只读检查直接集成的 modeling

以下命令不会修改任何文件：

```bash
set -euo pipefail
export PYTHONPATH="$INTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HIAI_SOURCE="$INTERNAL_ROOT/models/modeling_qwen3_5_hiai_nd.py"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B -m py_compile "$HIAI_SOURCE"
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_hiai_feature_check \
  --source "$HIAI_SOURCE"
```

通过时状态为 `PASS_DIRECT_SOURCE_CONTRACT`。检查项包括：

- `Qwen3_5TextModel.forward` 和 `Qwen3_5ForCausalLM.forward` 显式接收
  `output_dflash_features=False`；
- 层 `1,5,9,13,17,21,25,29` 的输出由 collector 聚合；
- feature 宽度为 `20480`；
- CausalLM 显式透传 feature 开关并返回 feature sidecar；
- modeling 不导入已删除的 patch 工具。

## 4. Bridge 已直接实现

根据现有 inference，仓库中的 `models/internal_dflash_bridge.py` 已经固定复用：

```python
from models.export_model_wrapper_qwen3_5 import Qwen3_5ForCausalLMWrapper
```

它采用与原 `qwen3_5_inference_loop()` 相同的状态布局：

- linear-attention：`(past_conv_state, past_recurrent_state)`；
- full-attention：`(past_key_in, past_value_in)`；
- KV block size：`64`；
- embedding 从 `model_wrapper.model.get_input_embeddings()` 获取；
- 每次 DFlash target 调用都新建完整 state，并从位置 0 做 full-prefix prefill。

因此不需要手写 factory，也不需要 reset hook。Bridge 不改 attention/GDN/CacheUpdate，
这些仍由 `model_wrapper.model` 中原来的 HIAI 自定义算子执行。

Bridge 会在加载后核对 `model_wrapper.model` 的真实类型必须是
`models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM`。若这里报错，说明
`export_model_wrapper_qwen3_5.py` 仍导入了另一份 modeling；先让原 wrapper 指向已经能在
NPU 吐字的这份 HIAI 类，不能同时加载两份 target 实现。

运行前只需从原 inference YAML 读取：

```text
config_data['kv_cache_max_len']
```

并作为 `--kv-cache-max-len` 传入。当前 bridge 只支持原 inference 的
`quant_mode=disable` / FP16 路线。

## 5. 自定义算子环境检查

先用原 inference 跑一次普通文本生成，确认 `torch_npu`、内部算子注册和权重路径仍正常。
DFlash 不替换 target 内已有的 ChunkGatedDeltaRule、CacheUpdate、attention 或其他 HIAI 算子。
它只在草稿侧选择 package-local NPU backend；其余受支持的 PyTorch 运算按 tensor device
分派到 NPU。

如果要运行仓库提供的静态接口审计：

```bash
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  "$DFLASH_REPO/tools/audit_internal_custom_ops.py" \
  --vendors-root /path/to/opp/vendors \
  --nm "$(command -v nm)" \
  --pretty
```

静态审计通过不等于真机执行通过；最终仍以运行 trace 和输出对齐为准。

## 6. 最小 NPU smoke

```bash
set -euo pipefail

export PYTHONPATH="$INTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RUN_DIR=/path/to/dflash-run
mkdir -p "$RUN_DIR"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-v1-npu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-npu-smoke.log"
```

将 `4096` 替换为原 inference YAML 的实际值。`run_npu` 自动固定：

- `dtype=float16`；
- EOS token `248044`；
- package-local `dflash_ascend310p_ops`；
- `max_draft_tokens` 范围 `1..15`；
- 禁止 CPU fallback 和外部 ops backend；
- modeling 与 DFlash 源码只读检查，并在运行前后复核文件哈希。

## 7. Smoke 通过条件

报告至少应满足：

```text
strict_greedy_exact_match = true
feature_capture_zero_impact = true
bounded_full_prefix_repeatability = true
ordinary token IDs = DFlash token IDs
ordinary EOS/stop reason = DFlash EOS/stop reason
draft_calls > 0
target_feature_calls > 0
target verify calls > 0
operator_fallback_enabled = false
device = npu:0
dtype = torch.float16
runtime_preflight.source_integration = direct
runtime_preflight.source_modified_by_runtime = false
```

如果 `P → Q → P` 重复前缀门禁失败，先检查 bridge 的 fresh state shape、
`kv_cache_max_len` 和 full-prefix prefill。不要继续统计接受率。

## 8. 扩大到正式 V1 block

最小 smoke 通过后，改为：

```text
--max-new-tokens 32 --max-draft-tokens 15
```

再记录真实 NPU 接受率、无 CPU fallback、显存和性能。CPU/GPU 结果只验证 framework/draft；
NPU 最终输出必须与同一个 NPU target 的普通 greedy 完全一致。
