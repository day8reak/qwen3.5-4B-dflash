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
    ├── internal_dflash_bridge.py     # 仅在需要适配现有加载/重置函数时使用
    ├── 原工程其他 HIAI 文件
    └── dflash_v1/                    # 复制本仓库的整个目录
```

首次部署：

```bash
set -euo pipefail
test ! -e "$INTERNAL_ROOT/models/dflash_v1"
cp -a "$DFLASH_REPO/models/dflash_v1" "$INTERNAL_ROOT/models/dflash_v1"
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

## 4. 复用现有 target 加载和状态重置

`run_npu` 需要一个已经存在的 target factory：

```python
def load_qwen35_target(target_dir: str, *, device, dtype):
    # 直接复用原 inference 已跑通的加载函数。
    # 返回 models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM。
    ...
```

V1 每次 target 调用都重算完整前缀。内部 HIAI target 即使收到 `use_cache=False`，仍可能更新
KV、GDN conv/recurrent state 和请求计数，因此每次调用前还必须进入一个全新 prefill 请求。

如果 target 本身已经提供：

```python
prepare_dflash_full_prefix_call(
    *, input_ids, sequence_length, output_dflash_features, logits_to_keep, call_index
)
```

则不传 `--reset-hook`。否则在原工程中提供一个薄适配：

```python
def reset_qwen35_full_prefix(
    target,
    *,
    input_ids,
    sequence_length: int,
    output_dflash_features: bool,
    logits_to_keep: int,
    call_index: int,
):
    existing_inference_start_fresh_prefill(target, sequence_length)
    return None
```

这两个函数可以放在 `models/internal_dflash_bridge.py`。只需要把其中调用替换为内部 inference
已有的真实加载/新请求入口；不要在桥里重写 attention/GDN，也不要猜内部 state 的 shape、
dtype 或初始值。

例如实际 import 名为：

```text
--target-factory models.internal_dflash_bridge:load_qwen35_target
--reset-hook models.internal_dflash_bridge:reset_qwen35_full_prefix
```

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
  --target-factory models.internal_dflash_bridge:load_qwen35_target \
  --reset-hook models.internal_dflash_bridge:reset_qwen35_full_prefix \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-v1-npu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-npu-smoke.log"
```

如果 target 自己实现了标准 prepare 方法，删除 `--reset-hook` 那一行。`run_npu` 自动固定：

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

如果 `P → Q → P` 重复前缀门禁失败，先修 reset hook。不要继续统计接受率。

## 8. 扩大到正式 V1 block

最小 smoke 通过后，改为：

```text
--max-new-tokens 32 --max-draft-tokens 15
```

再记录真实 NPU 接受率、无 CPU fallback、显存和性能。CPU/GPU 结果只验证 framework/draft；
NPU 最终输出必须与同一个 NPU target 的普通 greedy 完全一致。
