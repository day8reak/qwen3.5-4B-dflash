# Qwen3.5-4B DFlash V1（v1-r1）

这是 Qwen3.5-4B 的 DFlash V1 PyTorch 实现，包含完整前缀重算调度、六层草稿模型、
CPU/CUDA 后端，以及 Ascend NPU/HIAI target 所需的检查和 loader。

仓库只保留可运行源码、部署工具、许可证和中文使用说明。测试日志、验证报告、发布清单和
模型权重不放在 GitHub 仓库中。

## 目录

- `models/dflash_v1/`：DFlash 调度器、草稿模型、CPU/CUDA/NPU backend 和运行入口。
- `models/modeling_qwen3_5_hiai_nd.py`：NPU target 的直接集成版本，包含八层 feature
  collector，并保持默认 forward 返回不变。
- `models/internal_dflash_bridge.py`：复用现有 HIAI wrapper，并为每次完整前缀调用创建
  全新的 hybrid KV/GDN state。
- `models/dflash_qwen_adapter_v1.py`：旧命令兼容入口。
- `tools/`：自定义算子静态预检工具。
- `config/`：预检工具使用的算子接口合同。
- `docs/`：CPU、CUDA 和 Ascend NPU 使用说明。
- `SOURCE_LOCK.json`：framework 启动和草稿 checkpoint 身份检查所需的精简运行合同。

NPU 部署采用“target 在父包、DFlash 放子目录”的结构：

```text
qwen35-runtime/
└── models/
    ├── modeling_qwen3_5_hiai_nd.py   # 本仓库直接提供
    ├── configuration_qwen3_5.py      # 已有配置文件
    ├── internal_dflash_bridge.py     # 本仓库已实现，无需手写
    ├── 其他运行文件
    └── dflash_v1/                    # 本仓库的 models/dflash_v1 整目录
```

不要用 CPU/GPU 的 `modeling_qwen3_5_dflash.py` 覆盖 NPU modeling。部署时将本仓库的
`models/modeling_qwen3_5_hiai_nd.py` 整体复制到目标工程同名位置。runner 只读校验它的
feature ABI，不会在运行时 patch 或修改它。

## 环境

使用 Python 3.10 和 `transformers==5.14.1`。请先安装与设备匹配的 PyTorch：

- CPU：官方 CPU PyTorch；
- CUDA：对应 CUDA 版本的 PyTorch；
- NPU：与目标设备匹配的 PyTorch、`torch_npu` 和自定义算子环境。

然后安装通用依赖：

```bash
python -m pip install "transformers==5.14.1" safetensors huggingface-hub
```

## 最小运行

准备本地主模型和官方 `z-lab/Qwen3.5-4B-DFlash` 草稿权重：
主模型目录还必须包含完整 tokenizer 文件；文本输入全程使用本地文件，不联网下载。

```bash
export PYTHONPATH="$PWD"
python -B -m models.dflash_v1.dflash_qwen_adapter_v1 \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cpu \
  --report /path/to/run/dflash-v1-cpu.json
```

建议先读主文档，再按问题进入子文档：

- [整体架构与完整数据流](docs/DFLASH_V1_ARCHITECTURE.md)
  - [Target 与 Feature](docs/DFLASH_V1_TARGET_AND_FEATURE.md)
  - [Draft 模型](docs/DFLASH_V1_DRAFT.md)
  - [Scheduler 与 token 验证](docs/DFLASH_V1_SCHEDULER.md)
  - [验证流程与报告解读](docs/DFLASH_V1_VALIDATION.md)
- [从 V1 到完整 DFlash 与真正提速](docs/DFLASH_FULL_AND_PERFORMANCE_ROADMAP.md)
- [实现和文件索引](README_DFLASH_V1.md)
- 运行文档：
  - [CPU/Golden](docs/DFLASH_V1_GOLDEN.md)
  - [CUDA GPU](docs/DFLASH_V1_GPU.md)
  - [Ascend NPU 部署与运行](docs/NPU_DEPLOYMENT.md)
  - [Ascend 310P 接口与边界](docs/DFLASH_V1_ASCEND310P.md)

## NPU 快速入口

先按 [Ascend NPU 部署与运行](docs/NPU_DEPLOYMENT.md) 部署仓库中的
`modeling_qwen3_5_hiai_nd.py`。bridge 会复用现有
`Qwen3_5ForCausalLMWrapper`，并按模型配置的 hybrid-cache shape 在每次 target 调用时
新建状态。`v1-r1` 还会把 `S>1` 的完整前缀在 bridge 内右补齐到 64-token GDN chunk，执行后只
截回真实 token 行，并在释放本次临时 KV/GDN state 前同步 NPU；同时会按
`--kv-cache-max-len` 重建所有 full-attention block table，因此不再需要手写 factory/reset：

```bash
export PYTHONPATH=/path/to/qwen35-runtime

python -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report /path/to/run/dflash-v1-npu-smoke.json
```

把 `4096` 替换成部署配置中 `kv_cache_max_len` 的真实值。
不需要修改 bridge 源码。`run_npu` 会自动固定 FP16、EOS `248044`、内嵌目录、package-local NPU
backend 和 HIAI source，不再要求 overlay JSON。

如果已经能生成但接受率偏低，按
[NPU 接受率分层诊断](docs/NPU_DEPLOYMENT.md#7-接受率低时的分层诊断) 运行
`models.dflash_v1.diagnose_acceptance`。它会先判定正常增量 Target 与 DFlash fresh
full-prefix Target 是否等价，再以逐 proposal 的独立前缀验证统计 K=1/4/8/16；旧的一次
向量化 target 验证只保留为 prefix-invariance 诊断。新版也支持 CUDA FP16/BF16 A/B、逐轮
无明文层级指纹和两份报告的首个分叉定位，避免把 kernel 随序列长度产生的舍入差异误报为
BF16 调度错误。
GPU 的 FP16/BF16 对照命令见 [DFlash V1 GPU 运行说明](docs/DFLASH_V1_GPU.md)。

固定文本也可以放进 UTF-8 文件，然后把 `--prompt "..."` 换成
`--prompt-file /path/to/prompt.txt`。默认 `--prompt-mode chat` 会套用本地主模型 tokenizer 的
chat template，且默认开启 thinking；只有文件已经包含完整模板文本时才使用
`--prompt-mode raw`。需要复现非 thinking workload 时显式传 `--no-enable-thinking`。入口会
直接打印 ordinary Target 和 DFlash 的解码结果。

报告中的 `accepted_draft_tokens / drafted_tokens` 不是官方 accept length。更接近官方口径的
字段是 `mean_emitted_tokens_per_draft_round`（接受 proposal 加 correction/bonus）。接受率仍然
强依赖 prompt、thinking 模式和生成阶段，不能用一条短文本替代多 workload 评测。

仓库不包含 Qwen3.5-4B 或 DFlash 权重。真实 CUDA 和 Ascend 310P 结果需要在对应服务器上
执行上述流程确认，CPU 结果不能替代设备验证。
