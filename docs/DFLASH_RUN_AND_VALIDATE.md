# DFlash rollback 运行与验证

本文合并 CPU golden、CUDA、HIAI/NPU 部署和报告门禁。框架算法见
[DFlash rollback 框架](DFLASH_ARCHITECTURE.md)，算子 ABI 见
[DFlash rollback 算子清单](DFLASH_OPERATORS.md)。

## 1. 先区分三类结论

| 结论 | 最低门禁 |
| --- | --- |
| 生成正确性 | ordinary 与 DFlash 的 token ID、EOS、stop reason 完全相同 |
| Rollback 接线 | verify 不含历史前缀；state、feature、position 和 KV cursor 提交同一个 1+a |
| 目标设备交付 | 真实设备无 fallback，记录 runtime、device、算子包、kernel trace、稳定性和延迟 |

CPU reduced-shape 测试只能证明辅助逻辑。CUDA 报告只能证明 framework 路线。两者都不能替代
Ascend 310P 的完整 Target、真实自定义算子和无 fallback 证据。

## 2. 通用准备

- Python 3.10；
- transformers 5.14.1；
- 与 CPU、CUDA 或 NPU 匹配的 PyTorch；
- 本地 Qwen3.5-4B Target checkpoint 和完整 tokenizer；
- 本地 z-lab/Qwen3.5-4B-DFlash 官方 Draft checkpoint；
- `block_size` 在 2 到 16 且包含 anchor；proposal capacity=`block_size-1`，本轮 T≤`block_size`；
- 报告、日志和 cache 写到仓库外的运行目录。

安装通用依赖：

~~~bash
python -m pip install "transformers==5.14.1" safetensors huggingface-hub
~~~

入口会检查 Target config、Draft 6 层结构、69 tensor 名称/shape/dtype、checkpoint hash，以及
Target embedding、LM head、Draft 的 device 和 dtype。

## 3. CPU 和 CUDA

从仓库根目录运行：

~~~bash
export PYTHONPATH="$PWD"
export PYTHONDONTWRITEBYTECODE=1

python -B -m models.dflash_v1.run_rollback \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 32 \
  --block-size 16 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cuda:0 \
  --report /path/to/run/dflash-rollback-cuda.json
~~~

CPU 将 device 改为 cpu，并按环境选择 dtype。真实 4B Target 和 Draft 在 CPU 上开销很大，日常
辅助逻辑验证优先运行第 9 节的 reduced-shape 测试。

CUDA 路线不要传 HIAI factory、source、reset hook 或 NPU ops backend。它使用
FrameworkDFlashRollbackTarget、DynamicCache 和 TorchDFlashOps。CUDA 不可用时必须直接失败，
不能用 CPU 结果伪装 CUDA。

如果要比较 FP16 和 BF16，必须保持 prompt、chat template、thinking、block_size、生成长度和 checkpoint
完全相同，并先确认 torch.cuda.is_bf16_supported 为真。跨 dtype 的浮点 tensor hash 不会相同，
判断重点应是 proposal、Target Top-1、accepted length 和最终 token。

## 4. HIAI/NPU 部署结构

目标工程保留原 ordinary modeling 和原部署 wrapper，只新增 rollback 文件：

~~~text
runtime/
└── models/
    ├── configuration_qwen3_5.py
    ├── export_model_wrapper_qwen3_5.py
    ├── export_model_wrapper_qwen3_5_dflash_rollback.py
    ├── modeling_qwen3_5_hiai_nd.py
    ├── modeling_qwen3_5_hiai_nd_dflash_rollback.py
    ├── internal_dflash_bridge.py
    └── dflash_v1/
~~~

需要部署的交付项：

~~~text
models/modeling_qwen3_5_hiai_nd_dflash_rollback.py
models/export_model_wrapper_qwen3_5_dflash_rollback.py
models/internal_dflash_bridge.py
models/dflash_v1/
~~~

不要用 rollback 文件覆盖 models/modeling_qwen3_5_hiai_nd.py，也不要覆盖部署原
models/export_model_wrapper_qwen3_5.py。Rollback wrapper adapter 在构造时绑定独立 modeling，
构造后恢复原全局类并检查实际模型类型；部署 wrapper 若硬编码其他类，adapter 会 fail closed。

## 5. NPU 静态检查

使用部署环境声明的 model Python：

~~~bash
export DEPLOY_ROOT=/path/to/runtime
export MODEL_PYTHON=/path/to/deployment/python
export PYTHONPATH="$DEPLOY_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1

"$MODEL_PYTHON" -B -m py_compile \
  "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/internal_dflash_bridge.py" \
  "$DEPLOY_ROOT/models/dflash_v1/run_rollback.py"
~~~

检查 GDR MTP 在当前进程可见：

~~~bash
"$MODEL_PYTHON" -B - <<'PY'
import torch
import torch_npu

op = getattr(torch_npu, "npu_gated_delta_rule_mtp", None)
if not callable(op):
    namespace = getattr(torch.ops, "npu", None)
    op = getattr(namespace, "npu_gated_delta_rule_mtp", None)
assert callable(op), "npu_gated_delta_rule_mtp is not registered"
print("GDR_MTP_REGISTERED")
PY
~~~

py_compile 只证明语法；符号可见只证明注册成功，都不是 shape、数值或整网通过。

## 6. NPU 最小 smoke

~~~bash
export RUN_DIR=/path/to/dflash-run
mkdir -p "$RUN_DIR"

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --block-size 2 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-rollback-npu-smoke.json"
~~~

kv-cache-max-len 必须与部署配置一致、为正且能被 64 整除。run_npu 固定 FP16、EOS 248044、
prefill chunk 64、decode chunk 1 和 package-local NPU Draft backend。不要传旧 full-prefix
reset hook。

Prompt 在 bridge 中逐 token bootstrap；verify 才进入 T=K+1 的 GDR MTP 路线。max-new-tokens
为 2 时，如果 bootstrap anchor 立即命中 EOS，就没有 Draft round。此时 token 可以正确，但
报告状态是 INCONCLUSIVE_NO_DRAFT_ROUND，需要换一个固定非立即结束 prompt。

### 6.1 长运行诊断与当前优化

`run_npu` 是 correctness validator：同一进程先跑 ordinary，再跑独立 DFlash session。因此请求
总时长不能直接当成纯 DFlash latency。进度日志现在会输出 `ordinary_decode_end`、每个
`dflash_round_end` 和 `dflash_decode_end`，报告的 `timings_seconds` 进一步拆分 checkpoint hash、
Target load、Draft load、ordinary decode 与 DFlash decode。

当前热路径已经做了五项 exact 优化：

- prompt 仍按 S=1 更新 Target state，但中间行跳过完整 LM Head，两条 session 都只在最后一行算 logits；
- Target 在设备侧完成 argmax，只回传 T 个 token ID，不再把完整 logits 转 FP32 后搬到 CPU；
- Target feature 的 `20480→2560 + RMSNorm` 每个 committed token 只执行一次并缓存投影结果；
- 任一 Draft round 接受数为 0 后，本请求切到同一 transaction 的 S=1 Target-only continuation。
- persistent Target 调用依赖同一设备流排序，不再在每个 prompt/decode row 后执行全设备 host
  barrier；runner 和 benchmark 仍在阶段/计时边界显式同步，full-prefix oracle 释放 call-local
  state 前仍同步。

NPU Draft primitive 默认采用 boundary-only finite policy：shape/device/dtype 始终检查，最终 Draft
logits 仍做 finite gate，但不再为每个 linear 重复扫描整块 weight 和中间 tensor。只有诊断 NaN/Inf
时才临时启用下面的同步密集模式：

~~~bash
DFLASH_ASCEND310P_EXHAUSTIVE_CHECKS=1 "$MODEL_PYTHON" -B \
  -m models.dflash_v1.run_npu ...
~~~

`block_size=16` 只在接受率足以摊薄 Draft/verify 时有收益。若报告显示首轮零接受，后续会自动
S=1 fallback；若持续低但非零接受，应使用相同 prompt 对比 B=2/4/8/16，而不是只看 B=16。

## 7. NPU 性能基准与 msprof

`benchmark` 分支原有的独立进程、同步计时、稳定输出检查、peak-memory 和 msprof 能力已经接到
当前 rollback 路线：

~~~text
models/dflash_v1/benchmark_npu.py
config/npu_benchmark_v1.json
tools/run_msprof.sh
~~~

benchmark 在计时前加载 Target/Draft，并执行一次 ordinary 与 DFlash 的 strict-greedy 零差异
门禁。每个计时迭代只包含一次完整 generation 和末尾设备同步，不包含 checkpoint hash、模型
加载、tokenizer 或 correctness gate。ordinary 与 DFlash 必须分别启动进程，避免 allocator、
持久状态和执行先后污染对比。

~~~bash
export BENCH_DIR=/path/to/dflash-benchmark
mkdir -p "$BENCH_DIR"

for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode "$MODE" \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat \
    --enable-thinking \
    --max-new-tokens 32 \
    --block-size 16 \
    --warmup 3 \
    --repetitions 10 \
    --device npu:0 \
    --report "$BENCH_DIR/$MODE.json"
done
~~~

两份报告必须锁定同一 Git revision、checkpoint、设备、prompt token hash、thinking、生成长度、
`block_size`、KV 长度和 chunk 配置。`block_size=16` 是 B=16 总行数，proposal 数 K=15；即使
ordinary 不使用 proposal，它仍属于 case identity 和进程内 correctness gate。

正式比较至少检查：

~~~python
assert report["status"] == "PASS"
assert report["strict_greedy_exact_match"] is True
assert report["historical_prefix_replay_during_verify"] is False
assert report["operator_fallback_enabled"] is False
assert report["benchmark"]["status"] == "PASS"
assert report["benchmark"]["summary"]["count"] == 10
~~~

比较 `benchmark.summary.latency_ms`、`aggregate_output_tokens_per_second` 和
`accelerator_memory`，同时保留全部 10 条 measurement；不要只报告最小值。每条 measurement
还记录 replay stats、adapter stats 和 receiver audit delta，可直接判断是否首轮零接受后进入
Target-only、实际调用了多少 Draft/verify，以及投影了多少 feature token。

msprof 会改变 latency，只用于热点归因。建议用 1 次 warmup、1 次 measurement，并通过 MSTX
范围 `qwen35/<mode>/measure/0` 定位正式迭代：

~~~bash
tools/run_msprof.sh \
  --label dflash-pipe \
  --output-dir "$BENCH_DIR/msprof-dflash-pipe" \
  --python "$MODEL_PYTHON" \
  --aic-metrics PipeUtilization \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode dflash \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking \
    --max-new-tokens 32 --block-size 16 \
    --warmup 1 --repetitions 1 --device npu:0 \
    --report "$BENCH_DIR/msprof-dflash-pipe/dflash.json"
~~~

建议把 `PipeUtilization`、`Memory`、`MemoryUB` 分开采集。wrapper 会拒绝 simulation-only、CPU
device 和 operator fallback，且要求输出目录在源码仓库外；profile 下的 host latency 不能替代
上面的未 profiling 3+10 基线。

## 8. 报告门禁

通用字段：

~~~python
import json

with open("/path/to/run/dflash-rollback.json", encoding="utf-8") as stream:
    report = json.load(stream)

assert report["route"] == "qwen3.5-dflash-incremental-rollback"
assert report["strict_greedy_exact_match"] is True
assert report["verification_mode"] == "incremental_transactional_rollback"
assert report["historical_prefix_replay_during_verify"] is False
assert 2 <= report["block_size"] <= 16
assert report["request"]["proposal_capacity"] == report["block_size"] - 1
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["ordinary"]["reached_eos"] == report["dflash"]["reached_eos"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]
assert report["dflash_execution_gate"]["status"] == "PASS"
assert report["dflash_execution_gate"]["target_verify_calls"] > 0
~~~

CUDA/NPU 还应满足：

~~~python
assert report["operator_fallback_enabled"] is False
assert report["runtime_identity"]["device_type"] in {"cuda", "npu"}
~~~

CPU/CUDA transaction：

~~~python
audit = report["target_rollback_audit"]
assert audit["historical_prefix_replay_during_verify"] is False
assert audit["pending_transaction"] is False
assert audit["commit_replay_scope"] == (
    "anchor_plus_accepted_prefix_only_one_token_per_call"
)
~~~

NPU transaction：

~~~python
audit = report["target_rollback_audit"]
assert audit["enabled"] is True
assert audit["gdr_backend"] == "npu_gated_delta_rule_mtp"
assert audit["conv_bank_backend"] == "torch_tensor_golden_on_input_device"
assert audit["kv_policy"] == (
    "physical_provisional_writes_logical_cursor_commit"
)
assert audit["persistent_call_synchronization_policy"] == (
    "same_device_stream_dependencies_no_per_call_host_barrier"
)
assert audit["session_invalid"] is False
assert audit["pending_verify_rows"] is None
~~~

torch_tensor_golden_on_input_device 表示 conv 分解运算跟随输入 tensor 的 NPU device，不表示
允许 CPU operator fallback。

主要统计字段：

| 字段 | 含义 |
| --- | --- |
| target_verify_calls | T=K+1 verify 次数 |
| target_input_tokens_recomputed | 实际送入 Target 的 prompt、增量和 block 行数，不含 verify 历史重放 |
| drafted_tokens | Draft proposal 总数 |
| accepted_draft_tokens | 最长连续前缀中接受的 proposal 总数 |
| rejected_draft_tokens | 未提交 proposal 总数 |
| fallback_tokens | bootstrap、correction 或 bonus token 数 |
| acceptance_rate | accepted_draft_tokens 除以 drafted_tokens |
| speculation_disable_events | 首次零接受后关闭本请求 Draft 的次数，最多为 1 |
| target_only_fallback_rounds | 关闭 Draft 后执行的 S=1 Target-only round 数 |
| draft_feature_tokens_projected | 实际做过 20480→2560 投影的 committed token 数 |
| rollback_commit_replay_calls | CPU/CUDA 为提交状态执行的有界单 token 调用数 |

接受率只描述 Draft 质量。mean_emitted_tokens_per_draft_round 更接近每轮推进量，但两者都不能
替代端到端 latency。

## 9. 自动化检查

~~~bash
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_scheduler.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_framework_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_internal_dflash_bridge_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_helpers.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_dflash_runtime_optimizations.py \
  tests/test_benchmark_npu.py \
  tests/test_msprof_script.py \
  tests/test_source_lock_benchmark.py
~~~

| 测试 | 覆盖 |
| --- | --- |
| test_dflash_rollback_scheduler.py | accepted 0 到 K、correction、bonus、EOS 和长度 |
| test_dflash_framework_rollback.py | DynamicCache crop、GDN restore、有界 commit replay |
| test_internal_dflash_bridge_rollback.py | bank select/rebase、logical KV cursor、失败状态 |
| test_dflash_rollback_helpers.py | causal-conv bank 与逐 token reference |
| test_dflash_runtime_optimizations.py | feature projection cache 等价性与 NPU finite policy |
| test_benchmark_npu.py | 同步区间、3+10 合同、稳定输出和 rollback invocation |
| test_msprof_script.py | shell 语法、simulation/CPU/fallback fail-closed |
| test_source_lock_benchmark.py | benchmark 与 rollback runtime 文件 hash 身份 |

这些测试是 CPU/reduced-shape 证据。

## 10. Ascend 310P 分阶段门禁

| 阶段 | 至少覆盖 |
| --- | --- |
| 小块接线 | K=1、连续多轮、accepted 0/1、ordinary token 零差异 |
| State bank | K=1/3/5/7/15，accepted 0/1/K-1/K，最后一轮动态 T |
| KV boundary | cursor 62/63/64/65，拒绝尾部不可见且被覆盖 |
| Attention | T=2/4/6/8/16、多档历史长度、每个有效 row 对齐独立 prefix oracle |
| Feature | 八个固定层、只提交 1+a 行、开关不改变 Target Top-1 |
| 故障 | 不同 decoder 层失败后 session 整体失效 |
| 稳定性 | 多 prompt、多进程重复，无越界、状态泄漏或持续内存增长 |
| 身份 | 记录 device、runtime、算子包/source hash 和 kernel trace，无 CPU fallback |
| 整网 | token ID、EOS、stop reason 对 ordinary incremental 全部零差异 |

先跑 K=1，再扩 K=15（`block_size=16`）；先证明正确性，再测性能。现有 CacheUpdate 或
fused attention 能力通过时应直接复用，不因名称不同而提前重写 kernel。

## 11. 接受率和分叉诊断

diagnose_acceptance 保留 sequential full-prefix oracle，用来定位 proposal、Target prefix
invariance 和 dtype 分叉；它不是默认 rollback 运行入口。

~~~bash
python -B -m models.dflash_v1.diagnose_acceptance \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt-file /path/to/fixed-prompt.txt \
  --prompt-mode chat \
  --enable-thinking \
  --device cuda:0 \
  --dtype float16 \
  --eos-token-id 248044 \
  --acceptance-rounds 16 \
  --verification-mode sequential \
  --proposal-counts 1,3,5,7,15 \
  --trace-draft-layers \
  --report /path/to/run/acceptance-diagnosis.json
~~~

排错顺序：

1. 第一轮 mismatch：检查 prompt bootstrap、feature、anchor、position 和 selector 0。
2. 第二轮开始 mismatch：检查 1+a cursor、上一轮 bank slot a 和 correction 是否仍为未处理 anchor。
3. K 改变后 mismatch：检查先 select 已提交槽，再 rebase 到新 T。
4. 63/64 附近失败：检查 CacheUpdate block/offset、mask 和实际 KV length。
5. token 正确但慢：profile 完整 logits D2H、逐 row CacheUpdate、conv Tensor 分解、Draft context
   重算和同步。

一条短 prompt 不能代表平均接受长度。比较接受率时必须锁定 tokenizer、chat template、thinking、
dtype、K、checkpoint 和生成阶段，并使用多条代表性 workload。

## 12. 完成标准

目标路线通过需要同时满足：

- 至少一轮 Draft、T=K+1 verify 和 commit 真正执行；
- ordinary 与 DFlash strict-greedy 全 token、EOS、stop reason 零差异；
- 拒绝后继续至少一个 token，完整状态仍与 ordinary 对齐；
- 24 层 GDR/conv、8 层 KV、position 和 feature 共用同一个 accepted count；
- 310P 无 fallback，运行身份和 kernel trace 可审计；
- 多 prompt、多轮和 block boundary 重复稳定。

端到端提速是独立门禁：使用相同设备、prompt、输出长度和 warmup，分别报告 prefill、Draft、
verify、state/rollback、host、总 latency、accepted tokens 和峰值内存。正确性通过本身不构成
性能结论。
