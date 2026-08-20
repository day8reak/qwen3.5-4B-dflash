# DFlash V1 源码索引

本目录是完整的 DFlash V1 Python 包。文件保持同级是有意设计：内部 HIAI 接收工程会把
`TARGET_OVERLAY_FULL.json` 中的文件按原文件名扁平复制到
`transformer.model.qwen3_5`，同级相对 import 因而可以直接工作。

## 运行与调度

- `dflash_qwen_adapter_v1.py`：CPU/CUDA/NPU 命令行入口和整条验证流程。
- `dflash_reference_decode_v1.py`：无 cache 的完整前缀 DFlash 调度 golden。

## 草稿模型

- `modeling_dflash.py`：六层 DFlash 草稿模型。
- `dflash_config.py`：草稿结构与 shape 合同。
- `dflash_weights.py`：官方草稿 checkpoint 校验和加载。

## Target 主模型与 feature

- `modeling_qwen3_5_dflash.py`：CPU/CUDA 使用的 Transformers 5.14.1 target。
- `configuration_qwen3_5.py`：对应的 Qwen3.5 配置。
- `dflash_target_features.py`：八层 feature collector 和输出类型。
- `dflash_target_hook_bridge.py`：仅供 eager/CPU 调试的 hook 方案。

## 设备算子 backend

- `dflash_ops.py`：六个草稿原语的统一 Python ABI。
- `dflash_ascend310p_ops.py`：Ascend/NPU 的分解 PyTorch backend。
- `dflash_custom_ops_template.py`：接入内部 fused/custom op 的模板。

## 内部 HIAI 接入

- `dflash_hiai_feature_patch.py`：私有 HIAI target 的 AST patch/check 工具。
- `dflash_hiai_feature_runtime.py`：保留私有输出字段的 feature sidecar。
- `internal_target_loader_template.py`：接收工程 target loader/facade 模板。

仓库内推荐入口：

```bash
python -m models.dflash_v1.dflash_qwen_adapter_v1 --help
```

旧入口 `python -m models.dflash_qwen_adapter_v1` 仍由上层兼容文件转发；其他历史
`models.dflash_*` import 则由 `models/__init__.py` 的兼容搜索路径按需解析，不会提前加载模型。
