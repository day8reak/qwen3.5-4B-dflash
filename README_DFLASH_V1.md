# DFlash V1 r6 — 实现与设备接入说明

当前实现版本为 r6，依赖 `transformers==5.14.1`。正式 310P 路线是：对接收方自有的
`modeling_qwen3_5_hiai_nd.py` 做可审计源码 patch，并在同一 Python 包内提供 runtime
sidecar。`dflash_target_hook_bridge.py` 只保留为 CPU/eager 调试备用，不能作为正式 HIAI
NPU 证据。

先阅读 [310P 全流程接入说明](docs/DFLASH_V1_ASCEND310P.md)、
[中文 Golden 指南](docs/DFLASH_V1_GOLDEN.md) 和
[覆盖文件说明](docs/TARGET_OVERLAY_ZH.md)。

## r6 的准确率路线

- 普通 target greedy 始终是权威结果；DFlash 的 token ID、EOS 和停止原因必须与其完全一致。
- 每个 target 调用都重算完整已提交前缀。V1 不做投机 KV/GDN state 的提交、分支或回退。
- `use_cache=False` 只是 portable forward ABI 的请求，不能证明 HIAI 内部没有 block-table KV、
  `CacheUpdate`、卷积 state 或 GDN recurrent state。
- 正式 HIAI 调用前必须由接收方以 `receiver_reset_hook` 清空全部可变状态，或返回一个真正的
  `fresh_instance`；两种方式都必须进入 fresh prefill。声明的 prefill `chunk_size=64`、decode
  `chunk_size=1` 仍须由真机 trace 核实。
- r6 会做异长 `P -> Q -> P` 的 logits/feature 可重复性检查。它是有界行为门禁，不是对每个
  KV/GDN/外部状态字段已经清零的逐项证明。
- 现有自定义算子的实现、ABI 和二进制完全不改。`CacheUpdate` 与
  `ChunkGatedDeltaRule` 是 receiver HIAI 路线依赖；DFlash controller/草稿不直接调用它们。

## 正式 HIAI 特征路径

patcher 在 target decoder 层 `1,5,9,13,17,21,25,29` 的层后、最终 norm 前捕获输出，并按
此顺序拼成 `[B,S,20480]`。`output_dflash_features=False` 时保留原有返回表达式；开启时由
sidecar 在不丢失接收方私有输出字段的前提下附加 `dflash_features`。

接收包同级至少需要：

```text
modeling_qwen3_5_hiai_nd.py          # 接收方自有；由 patcher 修改，不在交付哈希闭包内
dflash_target_features.py            # r6 交付 sidecar
dflash_hiai_feature_runtime.py        # r6 交付 sidecar
internal_target_loader.py             # 从模板定制；接收方自有，不在交付哈希闭包内
configuration_qwen3_5.py              # 接收方已有依赖
```

安全执行 patcher 的三步命令：

```bash
export GOLDEN_ROOT=/path/to/extracted/qwen3_5
export TARGET_QWEN_DIR=/path/to/transformer/model/qwen3_5
export HIAI_SOURCE="$TARGET_QWEN_DIR/modeling_qwen3_5_hiai_nd.py"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --dry-run --show-diff
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --in-place
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --check
```

`--in-place` 默认生成 `.pre-dflash-v1` 备份；已有备份时工具会 fail closed。正式运行前还要在
receiver factory 中记录 patched source 的实际 SHA-256，并声明 feature contract、fresh
prefill、64/1 chunk 以及所选状态隔离方式，详见 310P 指南。

## 完整 singular CLI 闭包

仓库源码集中在 `models/dflash_v1/`；其中的
[README](models/dflash_v1/README.md) 按运行调度、草稿模型、target、设备 backend 和 HIAI
接入解释各文件。旧的 `python -m models.dflash_qwen_adapter_v1` 命令仍可用，但新代码推荐：

```bash
python -m models.dflash_v1.dflash_qwen_adapter_v1 --help
```

`TARGET_OVERLAY_FULL.json` 定义 **13 个运行文件**，供 overlay 检查工具从上述目录读取：

```text
dflash_ascend310p_ops.py
dflash_config.py
dflash_hiai_feature_patch.py
dflash_hiai_feature_runtime.py
dflash_ops.py
dflash_qwen_adapter_v1.py
dflash_reference_decode_v1.py
dflash_target_features.py
dflash_target_hook_bridge.py
dflash_weights.py
internal_target_loader_template.py
modeling_dflash.py
modeling_qwen3_5_dflash.py
```

此外必须保留接收方自有 `configuration_qwen3_5.py`；正式 HIAI 路线另有接收方自有、已 patch
的 `modeling_qwen3_5_hiai_nd.py` 和由模板定制的 `internal_target_loader.py`。

完整复制、patch HIAI 源码并完成 `internal_target_loader.py` 的 receiver factory 后，再运行
无权重检查；未修改的 placeholder 会被拒绝：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B "$GOLDEN_ROOT/tools/validate_target_overlay.py" \
  --scope v1-cli \
  --source-models-dir "$GOLDEN_ROOT/models/dflash_v1" \
  --package-dir /path/to/transformer/model/qwen3_5 \
  --package-name transformer.model.qwen3_5 \
  --hiai-source /path/to/transformer/model/qwen3_5/modeling_qwen3_5_hiai_nd.py \
  | tee /path/to/run/overlay-preflight.json
```

`--scope v1-cli` 会把 `--hiai-source` 锁到上述 receiver package 内的同名真实文件，并调用
交付 patcher 的语义级 `--check`；外部同名文件、缺失文件、symlink 或被篡改的旁路都会失败。
正式 NPU CLI 必须通过 `--overlay-preflight-report` 传入这份 JSON，并会在推理前后重新核对
13 个运行文件、HIAI source 和 loader 哈希。

预检也会拒绝 receiver package 中已有的 `.pyc`/`__pycache__`；`-B` 只防止新增缓存。若已存在，
请依内部工程策略清理，或改用干净的 overlay 副本，不要对未知目录做宽泛递归删除。

该门禁防止误复制、未实现 factory、源码漂移和普通状态泄漏，不是针对恶意 receiver Python
代码的安全沙箱。`create_internal_target()` 属于接收方可信集成代码；真实权重身份和设备算子
trace 仍必须由接收方证据补齐。

`transformers`（复数）是本包锁定的 Hugging Face `5.14.1` 依赖；
`transformer.model.qwen3_5`（单数）只是内部接收包名示例，部署时应替换为真实包名。

## CUDA GPU 路线

CUDA 已增加默认 `TorchDFlashOps`/SDPA 路线、设备可用性检查和可选真 GPU tiny test，命令见
[DFLASH_V1_GPU.md](docs/DFLASH_V1_GPU.md)。本发布机是 CPU-only PyTorch，故真实 GPU 权重
运行仍为 `PENDING`；GPU 结果只验证 HF/PyTorch V1 流程，不能替代 HIAI/310P 证据。

## 结论边界

当前真实 4B/草稿权重、内部闭源 receiver、FP16 严格 greedy、Ascend 310P 无 fallback、
逐 state/chunk trace、跨 64 token 边界和性能均为 `PENDING`。CPU 结果只是模拟证据；本包
不能声明 310P 已跑通或已有加速。DFlash2 与 V2 state commit/rollback 不在本包范围内。
