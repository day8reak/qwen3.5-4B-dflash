# DFlash V1 源码索引

本目录整体放在内部 inference 的 `models/dflash_v1/`。内部已经跑通的
`models/modeling_qwen3_5_hiai_nd.py` 保持在父目录；不再把 DFlash 文件扁平复制到原模型
包，也不覆盖 HIAI target。

## 运行与调度

- `run_npu.py`：内嵌目录的一键 NPU 入口，自动派生 HIAI source、loader、FP16 和 EOS。
- `dflash_qwen_adapter_v1.py`：CPU/CUDA/NPU 完整入口和严格 greedy 验证流程。
- `dflash_reference_decode_v1.py`：无 cache 的完整前缀 DFlash 调度 golden。

## 草稿模型

- `modeling_dflash.py`：六层 DFlash 草稿模型。
- `dflash_config.py`：草稿结构与 shape 合同。
- `dflash_weights.py`：官方草稿 checkpoint 校验和加载。

## Target 主模型与 feature

- `modeling_qwen3_5_dflash.py`：CPU/CUDA 使用的 Transformers 5.14.1 target。
- `configuration_qwen3_5.py`：CPU/CUDA target 配置。
- `dflash_target_features.py`：八层 feature collector 和输出类型。
- `dflash_hiai_feature_check.py`：只读检查父目录 HIAI target 已直接集成 feature route。
- `dflash_hiai_feature_runtime.py`：保留内部输出字段的 feature sidecar。
- `../internal_dflash_bridge.py`：复用现有 wrapper，并为每次调用新建 hybrid state。
- `internal_target_loader.py`：把已实现的 bridge 包装成 DFlash target facade。
- `internal_target_loader_template.py`：facade 合同及自定义 loader 参考。
- `dflash_target_hook_bridge.py`：仅供 eager/CPU 调试的 hook 方案。

## 设备算子 backend

- `dflash_ops.py`：六个草稿原语的统一 Python ABI。
- `dflash_ascend310p_ops.py`：Ascend/NPU 的分解 PyTorch backend。
- `dflash_custom_ops_template.py`：接入内部 fused/custom op 的模板。

## 入口

```bash
python -m models.dflash_v1.run_npu --help
python -m models.dflash_v1.dflash_qwen_adapter_v1 --help
```

完整内网部署流程见 [NPU_INTERNAL_LAYOUT.md](../../docs/NPU_INTERNAL_LAYOUT.md)。
