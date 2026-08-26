# DFlash V1 Ascend NPU 接入说明

本页说明代码边界和真机验证要求；实际目录、直接源码检查和命令以
[NPU_DEPLOYMENT.md](NPU_DEPLOYMENT.md) 为准。
后续将当前全前缀 V1 升级为单次整块 verify 时所需的 KV/GDN 状态与算子改造，见
[完整 DFlash 与提速路线](DFLASH_FULL_AND_PERFORMANCE_ROADMAP.md)。

## 实现结构

```text
共享 DFlash V1 scheduler
├── Target
│   ├── CPU/CUDA：models.dflash_v1.modeling_qwen3_5_dflash
│   └── NPU：models.modeling_qwen3_5_hiai_nd
└── Draft
    ├── CPU/CUDA：TorchDFlashOps
    └── NPU：models.dflash_v1.dflash_ascend310p_ops
```

NPU target 继续执行 HIAI modeling 中已有的自定义算子。DFlash 不用 PyTorch hook
全局替换它们，也不从 Python 直接调用原始 ACLNN C API。直接集成的 HIAI feature route
只增加：

- `output_dflash_features=False` 显式参数；
- decoder 层 `1,5,9,13,17,21,25,29` 的层后捕获；
- `[B,S,20480]` feature 输出；
- feature flag 在 TextModel/ForCausalLM 之间的显式透传。

默认路径仍返回原 `logits: Tensor`；只有 `output_dflash_features=True` 时返回
`(logits, dflash_features)`。这条 HIAI 主路线不使用 HF ModelOutput sidecar。

## NPU 与 CPU/CUDA 不同的地方

CPU/CUDA framework target 在 `use_cache=False` 下进行完整前缀计算。HIAI target 即使
收到同样参数，底层算子仍可能原位维护：

- block-table KV；
- GDN `conv_state`；
- GDN `recurrent_state`；
- `new_kv_cache_pos`；
- `allQLen`；
- `token_count`；
- `export_flag`。

本仓库直接包含 `models/internal_dflash_bridge.py`。它使用
`Qwen3_5ForCausalLMWrapper`，并按模型配置的 shape，在每次
完整前缀调用时重新创建上述 state；不再需要手写 `--reset-hook`。

## 已实现的 HIAI bridge

`models/internal_dflash_bridge.py` 已实现：

- 用 `Qwen3_5ForCausalLMWrapper(model_path=..., device="npu", dtype=float16)` 加载权重；
- 从 `.model.config` 读取 32 层 hybrid 结构；
- linear-attention state 使用固定的 conv/recurrent shape 合同；
- full-attention state 使用 `[max_len/64, kv_heads*head_dim/16, 64, 16]`；
- 把 `config.kv_cache_max_len` 写入模型后重建并校验每个 full-attention block table；
- 每次调用从位置 0 对完整前缀执行 fresh prefill；`S>1` 的物理输入右补齐到 64 的倍数，
  逻辑 `allQLen` 和返回的 logits/features 仍只覆盖真实 token；
- 返回前同步 NPU，禁止调用级 KV/GDN state 在异步 kernel 完成前被释放或复用；
- 将 HIAI Tensor/tuple 输出统一为 `logits + dflash_features`。

无需修改这个文件，也无需实现 reset 函数。唯一新增的运行参数
`--kv-cache-max-len` 必须与部署配置完全相同。

## 最小运行

```bash
set -euo pipefail

export DEPLOY_ROOT=/path/to/qwen35-runtime
export MODEL_PYTHON=/path/to/python
export RUN_DIR=/path/to/run
export PYTHONPATH="$DEPLOY_ROOT${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$RUN_DIR"

PYTHONDONTWRITEBYTECODE=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --block-size 2 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-v1-npu-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-npu-smoke.log"
```

把 `4096` 替换为部署配置的 `kv_cache_max_len`。这个入口自动固定：
FP16、EOS `248044`、内嵌目录、根 HIAI source、package-local loader 和
NPU backend。它会直接检查当前内嵌源码树，不需要额外生成 overlay 预检文件。

## 正确性门禁

1. 原始 HIAI 普通生成在修改前后 token 不变。
2. 同一个 prefix 的 feature=False/True logits Top-1 一致且在 dtype 容差内。
3. feature shape 为 `[1,S,20480]`，dtype 为 FP16，device 为请求的 NPU。
4. `P → P` 对照和异长 `P → Q → P` 均通过 bounded repeatability；报告保留误差指标。
5. NPU DFlash 与同一 NPU target ordinary greedy 的 token、EOS、stop reason 完全一致。
6. 至少执行一个 draft、一个 feature forward 和一个 target verify。
7. 无 CPU fallback。

第 4 项若连 `P → P` 都失败，先确认 `bridge_runtime` 报告了 64-token alignment、每个完成
forward 都执行了 synchronization，再看 max/mean/RMSE/relative-RMSE/cosine；不要因为
Top-1 暂时相同就直接放宽阈值。只有 `P → Q → P` 失败时，再查 fresh hybrid state、block
table 或完整前缀 prefill。门禁通过前不要继续解释接受率。

## 接受率和性能

最小 smoke 通过后，改为：

```text
max_new_tokens=32
block_size=8（K=7；稳定后再测 block_size=16，即 K=15）
```

CPU/GPU 接受率只作为诊断参考。最终 NPU 接受率依赖 NPU target features、draft backend、
FP16 数值和 target logits，必须在真实设备上测量。没有真机报告前不得声明 310P 加速比或无
fallback 已通过。

固定 workload 推荐保存为 UTF-8 文件，并使用
`--prompt-file /path/to/prompt.txt --prompt-mode chat`。入口会在本地套用 Qwen chat
template，默认启用 thinking，运行结束直接打印 ordinary Target 与 DFlash 两份解码文本。
非 thinking 对照显式加 `--no-enable-thinking`。本包统一使用官方 DFlash `block_size`
口径：它包含 clean anchor，所以官方最大 `block_size=16` 对应 K=15，draft query 共 16 行。

## 自定义算子说明

- `ChunkGatedDeltaRule`、`CacheUpdate` 等 target 算子由原 HIAI modeling/runtime 调用；
- `dflash_ascend310p_ops.py` 是 draft 六原语的 package-local NPU backend；
- 普通受支持 PyTorch 算子根据 NPU tensor 自动分派；
- 私有自定义算子不会仅凭 `device=npu:0` 自动替换，必须由原模型或显式 backend 调用。
