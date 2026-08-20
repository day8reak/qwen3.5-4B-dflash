# Qwen3.5-4B DFlash V1

这是 Qwen3.5-4B 的 DFlash V1 PyTorch 实现，包含完整前缀重算调度、六层草稿模型、
CPU/CUDA 后端，以及直接接入内部 Ascend 310P/HIAI 主模型所需的只读检查和 loader。

仓库只保留可运行源码、部署工具、许可证和中文使用说明。测试日志、验证报告、发布清单和
模型权重不放在 GitHub 仓库中。

## 目录

- `models/dflash_v1/`：DFlash 调度器、草稿模型、CPU/CUDA/NPU backend 和运行入口。
- `models/internal_dflash_bridge.py`：复用现有 HIAI wrapper，并为每次完整前缀调用创建
  全新的 hybrid KV/GDN state。
- `models/dflash_qwen_adapter_v1.py`：旧命令兼容入口。
- `tools/`：内部自定义算子静态预检工具。
- `config/`：预检工具使用的内部算子接口合同。
- `docs/`：CPU、CUDA 和 Ascend NPU 使用说明。
- `SOURCE_LOCK.json`：framework 启动和草稿 checkpoint 身份检查所需的精简运行合同。

内部 NPU 服务器采用“原模型不搬家、DFlash 放子目录”的结构：

```text
内部 inference 工程/
└── models/
    ├── modeling_qwen3_5_hiai_nd.py   # 已能在 NPU 吐字的原 HIAI target
    ├── configuration_qwen3_5.py      # 原工程文件
    ├── internal_dflash_bridge.py     # 本仓库已实现，无需手写
    ├── 其他原 HIAI 文件
    └── dflash_v1/                    # 本仓库的 models/dflash_v1 整目录
```

不要用 CPU/GPU 的 `modeling_qwen3_5_dflash.py` 覆盖 HIAI modeling。NPU 继续执行原
HIAI target 和其中已有的自定义算子；内部 modeling 应已直接包含可选的八层 feature 输出。
这份内部文件不在本仓库中，部署时保留服务器上已经改好的版本；DFlash runner 只读校验，
不会再次修改它。

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
python -B -m models.dflash_v1.dflash_qwen_adapter_v1 \
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
- [内部服务器目录与 NPU 运行流程](docs/NPU_INTERNAL_LAYOUT.md)

## 内部 NPU 快速入口

先按 [内部服务器目录与 NPU 运行流程](docs/NPU_INTERNAL_LAYOUT.md) 只读确认根目录中的
`modeling_qwen3_5_hiai_nd.py` 已直接集成 feature 旁路。bridge 会复用现有
`Qwen3_5ForCausalLMWrapper`，并按照原 inference 的 hybrid-cache shape 在每次 target 调用时
新建状态，因此不再需要手写 factory/reset：

```bash
export PYTHONPATH=/path/to/internal/inference

python -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report /path/to/run/dflash-v1-npu-smoke.json
```

把 `4096` 替换成原 inference YAML 中 `config_data['kv_cache_max_len']` 的真实值。
不需要修改 bridge 源码。`run_npu` 会自动固定 FP16、EOS `248044`、内嵌目录、package-local NPU
backend 和 HIAI source，不再要求 overlay JSON。

仓库不包含 Qwen3.5-4B 或 DFlash 权重。真实 CUDA 和 Ascend 310P 结果需要在对应服务器上
执行上述流程确认，CPU 结果不能替代设备验证。
