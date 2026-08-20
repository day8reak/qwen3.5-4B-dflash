# 内部服务器目录与 NPU 运行流程

本流程适用于以下前提：内部 inference 已能使用
`modeling_qwen3_5_hiai_nd.py` 在 NPU 上正常生成文本。DFlash 不替换这份 target，也不改动
其中已有的 ChunkGatedDeltaRule、CacheUpdate 或其他自定义算子；只增加 feature 旁路，并在
每次 V1 完整前缀调用前复用接收工程自己的“新请求”状态初始化。

## 1. 推荐目录

把本仓库的 `models/dflash_v1` 整个目录放入原 inference 的 `models` 包中。原模型继续留在
根目录：

```text
internal-inference/
├── models/
│   ├── __init__.py
│   ├── modeling_qwen3_5_hiai_nd.py
│   ├── configuration_qwen3_5.py
│   ├── internal_dflash_bridge.py       # 接收工程自有的加载/重置薄适配
│   ├── 原工程的其他模型和自定义算子文件
│   └── dflash_v1/
│       ├── __init__.py
│       ├── run_npu.py
│       ├── dflash_qwen_adapter_v1.py
│       ├── dflash_reference_decode_v1.py
│       ├── modeling_dflash.py
│       ├── dflash_weights.py
│       ├── dflash_ops.py
│       ├── dflash_ascend310p_ops.py
│       ├── dflash_target_features.py
│       ├── dflash_hiai_feature_runtime.py
│       ├── dflash_hiai_feature_patch.py
│       ├── internal_target_loader.py
│       └── 其余同目录运行文件
└── 原 inference 启动文件
```

运行时只把 `internal-inference/` 加入 `PYTHONPATH`。不要把 DFlash 文件扁平复制到
`models/`，也不要用 `modeling_qwen3_5_dflash.py` 覆盖 HIAI target。

第一次部署可以直接复制完整子目录；不要覆盖内部工程自己的 `models/__init__.py`：

```bash
set -euo pipefail

export DFLASH_REPO=/path/to/qwen3.5-4B-dflash
export INTERNAL_ROOT=/path/to/internal-inference

test -f "$INTERNAL_ROOT/models/modeling_qwen3_5_hiai_nd.py"
test -d "$DFLASH_REPO/models/dflash_v1"
test ! -e "$INTERNAL_ROOT/models/dflash_v1"
cp -a "$DFLASH_REPO/models/dflash_v1" "$INTERNAL_ROOT/models/dflash_v1"
```

更新版本时，先按内部策略保存或移走旧的 `models/dflash_v1`，再整体复制新目录；不要把两版
文件逐个混合覆盖。

## 2. 自动增加 HIAI feature 旁路

以下命令先只读检查真实源码是否满足已知 decoder-loop 锚点：

```bash
set -euo pipefail

export INTERNAL_ROOT=/path/to/internal-inference
export MODEL_PYTHON=/path/to/internal/python
export HIAI_SOURCE="$INTERNAL_ROOT/models/modeling_qwen3_5_hiai_nd.py"
export PYTHONPATH="$INTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" \
  --dry-run
```

`status` 正常后再执行自动修改。工具会额外留下一个非 Python 后缀的本地回退副本：

```bash
PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" \
  --in-place \
  --backup-suffix .pre-dflash-v1

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" \
  --check
```

修改后的 HIAI 文件会显式导入：

```python
from .dflash_v1.dflash_target_features import (
    DFlashFeatureCollector,
    QWEN35_4B_DFLASH_TARGET_FEATURES,
)
from .dflash_v1.dflash_hiai_feature_runtime import attach_dflash_features
```

feature 关闭时保持原返回 ABI；开启时捕获 decoder 层
`1,5,9,13,17,21,25,29` 的 post-layer/pre-final-norm hidden，并返回
`dflash_features: [1,S,20480]`。

## 3. 复用现有 inference 的两个接点

### Target factory

推荐把接收工程自有的两个薄适配放在
`models/internal_dflash_bridge.py`。`--target-factory` 指向已经跑通的模型加载函数，接口必须是：

```python
def load_qwen35_target(target_dir: str, *, device, dtype):
    # 复用现有内部 inference 的加载逻辑。
    # 返回原始 models.modeling_qwen3_5_hiai_nd.Qwen3_5ForCausalLM。
    ...
```

返回对象必须已经把权重放在请求的 NPU 和 dtype 上。不要返回 tokenizer、pipeline 或另一层
业务 wrapper。

### 完整前缀状态重置

如果原 target 已经暴露
`prepare_dflash_full_prefix_call(...)`，可以省略 `--reset-hook`。否则提供一个薄适配函数：

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
    # 调用现有 inference “开始一个全新请求/重新 prefill”的代码。
    # 必须处理其 KV、GDN conv/recurrent state 以及请求计数状态。
    # 不要在这里猜内部状态的 dtype、shape 或初始值。
    existing_inference_reset_for_new_request(target, sequence_length)
    return None
```

V1 会在普通 baseline、feature gate、bootstrap 和每轮 verify 前调用该函数。它不是性能优化；
它用于保证每次完整前缀计算不会继承上一次 HIAI 请求的可变状态。

## 4. 最小 NPU smoke

先使用两个新 token、一个 draft proposal：

```bash
set -euo pipefail

export INTERNAL_ROOT=/path/to/internal-inference
export MODEL_PYTHON=/path/to/internal/python
export RUN_DIR=/path/to/run
export PYTHONPATH="$INTERNAL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

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

`run_npu` 会自动设置：

- `dtype=float16`；
- `eos_token_id=248044`；
- `npu_layout=embedded`；
- package-local `internal_target_loader`；
- 根目录中的 `modeling_qwen3_5_hiai_nd.py`；
- package-local `dflash_ascend310p_ops`；
- 禁止 CPU fallback 和外部 ops backend。

它还会在推理前后重新检查当前 HIAI feature source 和 DFlash 运行文件；不需要生成或传入
overlay JSON。

## 5. Smoke 通过条件

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
```

如果 `P → Q → P` 重复前缀门禁失败，先修接收工程的 reset hook，不要继续调接受率。

## 6. 扩大到正式 V1 block

最小 smoke 通过后，把参数改为：

```bash
--max-new-tokens 32 --max-draft-tokens 15
```

此时再统计真实 NPU 接受率、无 CPU fallback 和性能。CPU/GPU 接受率只能作为参考；NPU
最终输出必须与同一个 NPU target 的普通 greedy 完全一致。
