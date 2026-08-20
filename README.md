# Qwen3.5-4B DFlash V1

这是 Qwen3.5-4B 的 DFlash V1 PyTorch 实现，包含完整前缀重算调度、六层草稿模型、
CPU/CUDA 后端，以及接入内部 Ascend 310P/HIAI 主模型所需的源码 patch 和 loader 模板。

仓库只保留可运行源码、部署工具、许可证和中文使用说明。测试日志、验证报告、发布清单和
模型权重不放在 GitHub 仓库中。

## 目录

- `models/`：主模型 feature 旁路、DFlash 草稿模型、解码调度和设备 backend。
- `tools/`：310P overlay 与内部自定义算子静态预检工具。
- `config/`：预检工具运行时读取的内部算子接口合同。
- `docs/`：CPU、CUDA 和 Ascend 310P 使用说明。
- `SOURCE_LOCK.json`：CPU/CUDA framework 启动和 checkpoint 身份检查所需的精简运行合同。
- `TARGET_OVERLAY*.json`：NPU 部署工具读取的文件闭包定义，不是测试 evidence。

## 环境

使用 Python 3.10 和 `transformers==5.14.1`。请先安装与设备匹配的 PyTorch：

- CPU：官方 CPU PyTorch；
- CUDA：对应 CUDA 版本的 PyTorch；
- NPU：内部服务器声明的 PyTorch、`torch_npu` 和自定义算子环境。

然后安装通用依赖：

```bash
python -m pip install "transformers==5.14.1" safetensors huggingface-hub
```

## 最小运行

准备本地主模型和官方 `z-lab/Qwen3.5-4B-DFlash` 草稿权重：

```bash
export PYTHONPATH="$PWD"
python -B -m models.dflash_qwen_adapter_v1 \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cpu \
  --report /path/to/run/dflash-v1-cpu.json
```

完整说明：

- [实现和文件说明](README_DFLASH_V1.md)
- [CPU/Golden 使用说明](docs/DFLASH_V1_GOLDEN.md)
- [CUDA GPU 使用说明](docs/DFLASH_V1_GPU.md)
- [Ascend 310P 接入说明](docs/DFLASH_V1_ASCEND310P.md)
- [内部工程文件覆盖说明](docs/TARGET_OVERLAY_ZH.md)

仓库不包含 Qwen3.5-4B 或 DFlash 权重。真实 CUDA 和 Ascend 310P 结果需要在对应服务器上
执行上述流程确认，CPU 结果不能替代设备验证。
