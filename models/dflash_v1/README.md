# DFlash V1 源码索引

本目录整体放在目标工程的 `models/dflash_v1/`。仓库根目录直接提供
`models/modeling_qwen3_5_hiai_nd.py`；部署时将它放到父目录同名位置。不要用 CPU/CUDA
target 覆盖它。

## 运行与调度

- `run_npu.py`：内嵌目录的一键 NPU 入口，自动派生 HIAI source、loader、FP16 和 EOS。
- `diagnose_acceptance.py`：CPU/CUDA/NPU 都对比 cached-incremental 与 fresh-full-prefix
  Target，并可在相同 greedy 前缀上扫描 K=1/4/8/16；支持直接传 UTF-8 prompt/txt、
  FP16/BF16 A/B、早中后段接受率、逐轮层级指纹、跨报告首个分叉和单轮 oracle tensor
  bundle，默认不输出 token ID。
- `dflash_qwen_adapter_v1.py`：CPU/CUDA/NPU 完整入口和严格 greedy 验证流程。
- `dflash_reference_decode_v1.py`：无 cache 的完整前缀 DFlash 调度 golden。

## 草稿模型

- `modeling_dflash.py`：六层 DFlash 草稿模型。
- `dflash_config.py`：草稿结构与 shape 合同。
- `dflash_weights.py`：官方草稿 checkpoint 校验和加载。

本包统一使用 vLLM 的 proposal-count 口径：`max_draft_tokens=K` 就是 proposal/mask 数，
clean anchor 不计入 K。官方配置值为 16，因此 K 的合法范围是 1 到 16；K=16 时 draft
query 是 1 个 anchor 加 16 个 mask，共 17 行。诊断报告始终明确记录 proposal count K。

## Target 主模型与 feature

- `modeling_qwen3_5_dflash.py`：CPU/CUDA 使用的 Transformers 5.14.1 target。
- `configuration_qwen3_5.py`：CPU/CUDA target 配置。
- `dflash_target_features.py`：八层 feature collector 和输出类型。
- `dflash_hiai_feature_check.py`：只读检查父目录 HIAI target 已直接集成 feature route。
- `dflash_hiai_feature_runtime.py`：旧 ModelOutput sidecar 兼容代码；本次 HIAI Tensor/tuple
  主路线不导入它。
- `../internal_dflash_bridge.py`：复用现有 wrapper，并为每次调用新建 hybrid state。
- `internal_target_loader.py`：把已实现的 bridge 包装成 DFlash target facade。
- `internal_target_loader_template.py`：facade 合同及自定义 loader 参考。
- `dflash_target_hook_bridge.py`：仅供 eager/CPU 调试的 hook 方案。

## 设备算子 backend

- `dflash_ops.py`：六个草稿原语的统一 Python ABI。
- `dflash_ascend310p_ops.py`：Ascend/NPU 的分解 PyTorch backend。
- `dflash_custom_ops_template.py`：接入 fused/custom op 的模板。

## 入口

```bash
python -m models.dflash_v1.run_npu --help
python -m models.dflash_v1.diagnose_acceptance --help
python -m models.dflash_v1.dflash_qwen_adapter_v1 --help
```

三个入口都接受 `--prompt "文本"` 或 `--prompt-file /path/to/prompt.txt`。默认
`--prompt-mode chat` 使用本地主模型 tokenizer 的 chat template，并输出解码后的 ordinary
Target 与 DFlash 文本；`raw` 模式只做普通 tokenizer 编码。

完整 NPU 部署流程见 [NPU_DEPLOYMENT.md](../../docs/NPU_DEPLOYMENT.md)。
