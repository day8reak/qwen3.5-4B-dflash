# Ascend NPU 部署与运行

本流程采用直接源码集成，不运行 patch，也不要求手工修改模型代码。

## 1. 路径变量

```bash
export DFLASH_REPO=/path/to/qwen3.5-4B-dflash
export DEPLOY_ROOT=/path/to/qwen35-runtime
export MODEL_PYTHON=/path/to/python
```

- `DFLASH_REPO`：本仓库根目录。
- `DEPLOY_ROOT`：运行工程根目录，下面已有 `models/` 和模型 wrapper。
- `MODEL_PYTHON`：该工程实际使用的 Python 3.10。

先核对：

```bash
set -euo pipefail
test -f "$DFLASH_REPO/models/modeling_qwen3_5_hiai_nd.py"
test -f "$DFLASH_REPO/models/dflash_v1/run_npu.py"
test -f "$DEPLOY_ROOT/models/configuration_qwen3_5.py"
test -f "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5.py"
test -x "$MODEL_PYTHON"
"$MODEL_PYTHON" -V
```

## 2. 目标目录

```text
qwen35-runtime/
└── models/
    ├── __init__.py
    ├── configuration_qwen3_5.py
    ├── export_model_wrapper_qwen3_5.py
    ├── modeling_qwen3_5_hiai_nd.py
    ├── internal_dflash_bridge.py
    ├── 其余运行文件
    └── dflash_v1/
```

`modeling_qwen3_5_hiai_nd.py` 保持原 HIAI 算子和默认 Tensor ABI，只增加可选 feature
返回。CPU/CUDA 的 `modeling_qwen3_5_dflash.py` 不可覆盖这个文件。

停止推理进程并按部署规范保留回退副本后，部署三个交付项：

```bash
set -euo pipefail

test ! -e "$DEPLOY_ROOT/models/dflash_v1.r10.new"
cp -a "$DFLASH_REPO/models/dflash_v1" \
  "$DEPLOY_ROOT/models/dflash_v1.r10.new"

# 首次部署时，目标 dflash_v1 不应存在；升级时先按部署规范将旧目录改名留存。
test ! -e "$DEPLOY_ROOT/models/dflash_v1"
mv "$DEPLOY_ROOT/models/dflash_v1.r10.new" \
  "$DEPLOY_ROOT/models/dflash_v1"

install -m 0644 "$DFLASH_REPO/models/internal_dflash_bridge.py" \
  "$DEPLOY_ROOT/models/internal_dflash_bridge.py"
install -m 0644 "$DFLASH_REPO/models/modeling_qwen3_5_hiai_nd.py" \
  "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd.py"
```

不要覆盖 `models/__init__.py`、`configuration_qwen3_5.py`、
`export_model_wrapper_qwen3_5.py` 或其他运行文件。

`-B` 不能忽略旧字节码。部署后先只读检查：

```bash
find "$DEPLOY_ROOT/models" -maxdepth 2 -type f \
  -name 'modeling_qwen3_5_hiai_nd*.pyc' -print
```

若有输出，只按部署规范处理列出的精确文件，或改用干净副本；不要递归清理整个工程。

## 3. 源码合同检查

```bash
set -euo pipefail
export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export DEPLOYED_HIAI_SOURCE="$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd.py"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B -m py_compile "$DEPLOYED_HIAI_SOURCE"
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_hiai_feature_check \
  --source "$DEPLOYED_HIAI_SOURCE"
```

通过时状态为 `PASS_DIRECT_SOURCE_CONTRACT`。检查内容包括：

- `output_dflash_features` 默认关闭；
- 在层 `1,5,9,13,17,21,25,29` 的层后、最终 norm 前采集；
- 默认返回 logits Tensor；开启时返回 `(logits, dflash_features)`；
- feature shape 为 `[B,S,20480]`；
- 不依赖 patch 或 ModelOutput sidecar。

## 4. Wrapper 绑定检查

`models/export_model_wrapper_qwen3_5.py` 必须让
`Qwen3_5ForCausalLMWrapper.model` 使用同包的
`models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM`。运行入口会校验实际类型；若不符会在
加载后立即失败，不会静默混用两份 target。

`--kv-cache-max-len` 必须使用部署配置中的实际值，并且能被 64 整除。当前 bridge 仅支持
FP16、非量化 target 路线。

## 5. 最小 NPU smoke

先确认普通文本推理仍能正常生成，再运行：

```bash
set -euo pipefail

export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"
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

将 `4096` 替换为部署配置的真实值。入口自动固定 FP16、EOS `248044`、package-local NPU
backend、无 CPU fallback，并在运行前后复核源码。

## 6. 通过条件

```text
strict_greedy_exact_match = true
feature_capture_zero_impact = true
bounded_full_prefix_repeatability = true
draft_calls > 0
target_feature_calls > 0
target verify calls > 0
operator_fallback_enabled = false
device = npu:0
dtype = torch.float16
runtime_preflight.source_integration = direct
runtime_preflight.source_modified_by_runtime = false
```

如果 `P → Q → P` 重复前缀门禁失败，先检查 fresh state shape、`kv_cache_max_len` 和
full-prefix prefill，不要继续统计接受率。

最小 smoke 通过后再使用：

```text
--max-new-tokens 32 --max-draft-tokens 15
```

真实 NPU 接受率、无 fallback、显存和性能只能由目标设备运行确认。
