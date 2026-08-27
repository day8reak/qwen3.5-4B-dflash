# Qwen3.5-4B DFlash rollback

这个分支实现了 Qwen3.5-4B 的 strict-greedy DFlash rollback。当前默认路线维护持久 Target
状态，每轮只验证 anchor 与 Draft proposal 组成的小块，不再把不断增长的历史前缀重新送入
Target。

当前实现目标是先闭合正确性和状态事务。仓库中的 CPU/reduced-shape 测试不是 Ascend 310P
真机证据，也不代表已经获得端到端加速。

## 当前框架

| 环节 | 当前实现 |
| --- | --- |
| Draft | 官方 6 层、69 tensor；逐层 committed KV cache；`block_size=16` 含 anchor，最多提出 K=15 个 token |
| Target verify | 一次输入 [anchor, d1, ..., dK]，本轮 T=K+1≤`block_size`，最大 16 |
| 接受规则 | 只接受从 d1 开始的最长连续匹配前缀 |
| CPU/CUDA 状态 | 持久 DynamicCache；verify 后恢复 KV/GDN，再只重放 anchor 和已接受 proposal |
| HIAI/NPU 状态 | GDR MTP recurrent bank、输入 NPU 上的 Torch conv golden、paged-KV logical cursor |
| 正确性 | `validate` 与独立 ordinary incremental greedy 做零差异门禁；`dflash` 只跑生产路径 |

~~~mermaid
flowchart LR
    P[Prompt prefill] --> A[Target clean anchor]
    A --> D[Draft: anchor + K masks]
    D --> Q[K proposals]
    Q --> V[Target once: T = K + 1]
    V --> M[Longest contiguous match a]
    M --> C[Commit anchor + accepted a]
    M --> N[Correction or bonus becomes next anchor]
    C --> D
    N --> D
~~~

一轮结束后的核心不变量是：

~~~text
Target state 与 feature 已处理到 current anchor 之前
current anchor 已输出，但还没有作为 Target 输入处理
~~~

因此本轮状态提交长度是 1+a，而 correction 或 all-match bonus 要留作下一轮 anchor。

## 代码入口

| 文件 | 作用 |
| --- | --- |
| models/dflash_v1/run_rollback.py | CPU、CUDA、NPU 共用的 rollback CLI |
| models/dflash_v1/run_npu.py | 固定 HIAI 参数的 NPU 简化入口 |
| models/dflash_v1/benchmark_npu.py | ordinary/DFlash 独立进程的同步 NPU 性能基准 |
| models/dflash_v1/dflash_rollback_decode.py | proposal 验证、最长连续接受和输出调度 |
| models/dflash_v1/dflash_rollback_adapter.py | CPU/CUDA Target transaction 与 Draft 接线 |
| models/internal_dflash_bridge.py | HIAI persistent state、bank selector 和 logical KV cursor |
| models/modeling_qwen3_5_hiai_nd_dflash_rollback.py | 独立 rollback HIAI modeling |
| models/export_model_wrapper_qwen3_5_dflash_rollback.py | 复用部署 wrapper 的 rollback adapter |

原 models/modeling_qwen3_5_hiai_nd.py 保持不变。代码目录名 dflash_v1 是兼容路径，不代表
默认调度仍是旧的 full-prefix V1；旧实现只保留为诊断 oracle。

## 最小运行

使用 Python 3.10、transformers 5.14.1，并先安装与设备匹配的 PyTorch。主模型和
z-lab/Qwen3.5-4B-DFlash 权重必须位于本地，仓库不包含权重、cache、ONNX/OM、日志或报告。

CPU 或 CUDA：

~~~bash
export PYTHONPATH="$PWD"
python -B -m models.dflash_v1.run_rollback \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 32 \
  --execution-mode validate \
  --block-size 16 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cuda:0 \
  --report /path/to/run/dflash-rollback.json
~~~

CPU 将 device 改为 cpu，并按环境选择 float32、float16 或 bfloat16。

HIAI/NPU：

~~~bash
export PYTHONPATH=/path/to/runtime
python -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 32 \
  --execution-mode validate \
  --block-size 16 \
  --device npu:0 \
  --report /path/to/run/dflash-rollback-npu.json
~~~

NPU 部署必须已经注册用户完成的 npu_gated_delta_rule_mtp。kv-cache-max-len 要与部署配置一致、
为正且能被 64 整除。

离线门禁通过后可把 `--execution-mode validate` 改为 `dflash`，从而不再额外运行 ordinary；该
单跑报告不会把未执行的 ordinary 对照标成 exact-match PASS。

## 文档

- [DFlash 框架与 token/state 流程](docs/DFLASH_ARCHITECTURE.md)
- [当前 rollback 与官方完整 DFlash 的差异](docs/DFLASH_UPSTREAM_COMPARISON.md)
- [现有算子与待开发自定义算子](docs/DFLASH_OPERATORS.md)
- [CPU、CUDA、NPU 运行和验证](docs/DFLASH_RUN_AND_VALIDATE.md)
- [源码索引](models/dflash_v1/README.md)

正式 rollback 报告至少应满足：

~~~text
route = qwen3.5-dflash-incremental-rollback
verification_mode = incremental_transactional_rollback
historical_prefix_replay_during_verify = false
strict_greedy_exact_match = true
draft_kv_cache_audit.mode = upstream_equivalent_append_then_crop
~~~

只有在 Ascend 310P 上禁用 fallback，并记录 runtime、device、算子包身份、kernel trace 和多轮
严格 token 对齐后，才能声明目标路线通过；性能结论还需要独立的端到端配对测量。

性能测试使用 `python -B -m models.dflash_v1.benchmark_npu`，ordinary 与 dflash 分别启动进程，
默认各 3 次 warmup、10 次 measurement。完整命令、报告门禁和 `tools/run_msprof.sh` 用法见
[运行和验证第 7 节](docs/DFLASH_RUN_AND_VALIDATE.md#7-npu-性能基准与-msprof)。
