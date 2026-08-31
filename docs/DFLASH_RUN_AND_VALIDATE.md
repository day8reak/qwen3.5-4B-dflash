# DFlash rollback 运行与验证

本文只保留可执行命令、报告门禁和性能口径。算法见[当前架构](DFLASH_ARCHITECTURE.md)，算子
见[自定义算子清单](DFLASH_OPERATORS.md)。

## 1. 证据分级

| 结论 | 最低证据 |
| --- | --- |
| scheduler 正确 | 同一 Target 下 ordinary/DFlash token ID、EOS、stop reason 完全一致 |
| rollback 正确 | verify 不含历史前缀；state/KV/feature/position 同时提交 `1+a`；拒绝后下一 token 仍一致 |
| NPU 路线通过 | 真实 Ascend 310P、无 fallback、记录 runtime/device/source/op 身份和 kernel trace |
| 端到端加速 | 相同 workload、精度、输出、同步边界的独立进程 3+10 配对测量 |

CPU reduced-shape 只能证明辅助逻辑；CUDA 只能证明 framework 路线。算子累计时间、单次 msprof
结果和 correctness PASS 都不能替代端到端加速证据。

## 2. 准备

- Python 3.10、`transformers==5.14.1`、`safetensors`；
- 匹配 CPU/CUDA/NPU 的 PyTorch；
- 本地 Qwen3.5-4B Target 与完整 tokenizer；
- 本地锁定的 `z-lab/Qwen3.5-4B-DFlash` checkpoint；
- 报告、日志、profile 和 cache 放在源码仓库外。

```bash
python -m pip install "transformers==5.14.1" safetensors huggingface-hub
export PYTHONDONTWRITEBYTECODE=1
```

入口会检查 Draft config、6 层/69 tensor、shape/dtype/hash，以及 Target/Draft 共享权重、device 和
dtype。`block_size` 包含 anchor，范围 2..16；B=16 对应 K=15、T=16。

## 3. CPU/CUDA

```bash
export PYTHONPATH="$PWD"

python -B -m models.dflash_v1.run_rollback \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat --enable-thinking \
  --max-new-tokens 32 \
  --execution-mode validate \
  --block-size 16 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cuda:0 \
  --report /path/to/run/dflash-cuda.json
```

CPU 改为 `--device cpu`，并按环境选择 dtype。CPU/CUDA 使用
`FrameworkDFlashRollbackTarget + TorchDFlashOps`：Target DynamicCache/GDN 在 verify 前保存，
commit 时只重放 anchor 与 accepted proposal，不重放 prompt 或更早历史。

CUDA 不可用时必须直接失败，不能用 CPU 结果伪装。不要传 HIAI factory、source、reset hook 或
NPU ops backend。

## 4. HIAI/NPU 部署

保留原 ordinary 文件，新增 rollback 文件：

```text
models/
├── modeling_qwen3_5_hiai_nd.py                         # ordinary；适配新 GDR effective_length ABI
├── export_model_wrapper_qwen3_5.py                     # 原文件，不覆盖
├── modeling_qwen3_5_hiai_nd_dflash_rollback.py
├── export_model_wrapper_qwen3_5_dflash_rollback.py
├── internal_dflash_bridge.py
└── dflash_v1/
```

复制源码后使用部署环境声明的 Python：

```bash
export DEPLOY_ROOT=/path/to/copied-runtime
export MODEL_PYTHON=/path/to/deployment/python
export PYTHONPATH="$DEPLOY_ROOT:$PYTHONPATH"
export PYTHONDONTWRITEBYTECODE=1

"$MODEL_PYTHON" -B -m py_compile \
  "$DEPLOY_ROOT/models/modeling_qwen3_5_hiai_nd_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/export_model_wrapper_qwen3_5_dflash_rollback.py" \
  "$DEPLOY_ROOT/models/internal_dflash_bridge.py" \
  "$DEPLOY_ROOT/models/dflash_v1/run_npu.py"
```

先确认部署的原 GDR 算子包已经升级到带 `effective_length` 的新 ABI：

```text
npu_chunk_gated_delta_rule(
  query, key, value, g, beta, effective_length,
  chunk_size=64, initial_state=None,
  output_final_state=False, use_qk_l2norm_in_kernel=False
)
```

其中 `effective_length` 必须接受 `[B] INT16`。普通 modeling 和 rollback ordinary 路径都会传
这个输入；旧注册包会在第一次 Target 调用时直接接口不匹配。GDR-MTP 的 ABI 不变，再检查其
注册：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import torch
import torch_npu

op = getattr(torch_npu, "npu_gated_delta_rule_mtp", None)
if not callable(op):
    op = getattr(getattr(torch.ops, "npu", None), "npu_gated_delta_rule_mtp", None)
assert callable(op), "npu_gated_delta_rule_mtp is not registered"
print("GDR_MTP_REGISTERED")
PY
```

语法和符号可见不证明 shape、数值或整网通过。

## 5. NPU FP16

先用 B=2 跑至少一轮 Draft/verify：

```bash
export RUN_DIR=/path/to/dflash-run
mkdir -p "$RUN_DIR"

"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 2048 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat --enable-thinking \
  --max-new-tokens 2 \
  --execution-mode validate \
  --block-size 2 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-fp16-smoke.json"
```

`run_npu` 固定 FP16、EOS 248044、prefill chunk 64、decode chunk 1 和 package-local NPU Draft
backend。`kv-cache-max-len` 必须覆盖 prompt+output，并能被 64 整除。Prompt 多 token 使用原 GDR
chunk 路线，并传本次 chunk 的 `INT16[B] effective_length`；只有 verify 使用 ABI 不变的
GDR-MTP。

如果 anchor 立即 EOS，报告会是 `INCONCLUSIVE_NO_DRAFT_ROUND`；换固定非立即结束 prompt。B=2
通过后再跑 `--max-new-tokens 32 --block-size 16`。

离线 validate 通过后，日常单跑改为：

```bash
  --execution-mode dflash
```

`dflash` 模式不额外跑 ordinary；它仍检查 checkpoint/source/device，但
`strict_greedy_exact_match=null`，正确性证据必须来自同 revision 的 validate 报告。

默认就是非量化模式。可以省略量化参数，也可显式传 `--quant_mode disable`。

## 6. Target W8A8 dynamic

量化 artifact 由原部署工程提供。分支不重新量化，也不复制权重。

先跑不加载 Draft 的预检：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.preflight_target_quant \
  --target-dir /path/to/Qwen3.5-4B \
  --prompt-ids 1,2,3,4 \
  --device npu:0 \
  --kv-cache-max-len 2048 \
  --config ./config/qwen3.5.yaml \
  --compare-first-qlinear \
  --report "$RUN_DIR/target-quant-preflight.json"
```

YAML：

```yaml
quanted_pth: /data/qwen35-w8a8/linear
embedding_weight_path: /data/qwen35-w8a8/embedding_weight.bin
embedding_scale_path: /data/qwen35-w8a8/embedding_scale.bin
```

`quanted_pth` 是包含 `data*.safetensors` 的目录；两个 embedding 文件是原 `numpy.tofile` raw
artifact。预检覆盖 Linear->QLinear 拓扑、同 activation 公式诊断、真实 multi-token prefill、
ordinary S=1、rollback S=1 和 embedding lookup，但不替代整网 DFlash 门禁。

预检通过后，在第 5 节命令追加：

```bash
  --config ./config/qwen3.5.yaml \
  --quant_mode enable
```

Target Linear/输入 embedding 走 W8A8，Draft embedding、LM head 和 6 层主体保持 FP16。整网
validate 比较的是“同一个 W8A8 Target 的 ordinary 与 DFlash”；若还要求 W8A8 与 FP16 token
一致，需要另做 ordinary 精度对照。

## 7. NPU 性能基准与 msprof

先区分三条路径：

| 对象 | 入口 | 用途 |
| --- | --- | --- |
| 原 main 非 DFlash | 原工程 `inference.py` | 旧部署权威基线 |
| rollback ordinary | `benchmark_npu --mode ordinary` | 同 receiver、同门禁的 scheduler 内部控制组 |
| rollback DFlash | `benchmark_npu --mode dflash` | 当前 Draft/verify/commit 路线 |

未 profiling 的正式基线使用独立进程、3 次 warmup、10 次 measurement。每次 measurement 包含
一轮完整 generation 和末尾设备同步，不包含 checkpoint hash、模型加载、tokenizer 或前置
correctness gate。

```bash
export BENCH_DIR=/path/to/dflash-benchmark
mkdir -p "$BENCH_DIR"

for MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode "$MODE" \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking \
    --max-new-tokens 32 --block-size 16 \
    --warmup 3 --repetitions 10 \
    --device npu:0 \
    --report "$BENCH_DIR/$MODE.json"
done
```

W8A8 对比时两个进程必须追加完全相同的参数：

```bash
QUANT_ARGS=(--config ./config/qwen3.5.yaml --quant_mode enable)
```

再把 `"${QUANT_ARGS[@]}"` 放到 ordinary 和 dflash 两条命令中。不能用 W8A8 DFlash 与 FP16
ordinary 计算 scheduler speedup。

正式比较前检查：

- copied source-tree hash、Target/Draft checkpoint、device/runtime 相同；
- prompt token hash、chat template、thinking、max tokens、实际输出 token 和输出 hash 相同；
- `block_size`、KV 长度、chunk、quant mode 相同；
- acceptance、Draft/verify calls、Target rows、fallback rounds 工作量可解释。

量化或低接受率场景不应假设 B=16 最快；保持上述身份不变，追加 B=`2/4/6/8/16` sweep。

### 7.1 原 main 非 DFlash 模型

原模型从原部署工程根目录运行 `inference.py` 和 main 的
`models/modeling_qwen3_5_hiai_nd.py`。`inference.py` 与配置不在本仓库中。下面命令故意保持
非量化：

~~~bash
cd /path/to/qwen3.5-main-runtime
export DFLASH_SOURCE=/path/to/copied-qwen3.5-4B-dflash
export BENCH_DIR=/path/to/dflash-benchmark
mkdir -p "$BENCH_DIR"

"$DFLASH_SOURCE/tools/run_msprof.sh" \
  --label main-original-unquantized \
  --output-dir "$BENCH_DIR/msprof-main-original-unquantized" \
  --python python3 \
  --aic-metrics PipeUtilization \
  --no-msproftx \
  -- \
  python3 inference.py \
    --config ./config/qwen3.5.ymal \
    --max_token 32
~~~

这条命令与下面的 rollback ordinary 并不是同一入口。若要计算“整体方案相对旧部署”的
speedup，还要对齐 prompt/input IDs、thinking、实际输出 token、cache 初始化和计时同步边界；
参数名字相似不自动代表边界相同。

### 7.2 rollback 内部 ordinary 控制组

`--mode ordinary` 并不是原 main 非 DFlash 模型。msprof 建议 1+1，只用于诊断：

```bash
tools/run_msprof.sh \
  --label rollback-ordinary-pipe \
  --output-dir "$BENCH_DIR/msprof-rollback-ordinary-pipe" \
  --python "$MODEL_PYTHON" \
  --aic-metrics PipeUtilization \
  --no-msproftx \
  -- \
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    --mode ordinary \
    --target-dir /path/to/Qwen3.5-4B \
    --draft-dir /path/to/Qwen3.5-4B-DFlash \
    --kv-cache-max-len 2048 \
    --prompt "请用一句话解释为什么天空是蓝色的。" \
    --prompt-mode chat --enable-thinking \
    --max-new-tokens 32 --block-size 16 \
    --warmup 1 --repetitions 1 --device npu:0 \
    --report "$BENCH_DIR/msprof-rollback-ordinary-pipe/ordinary.json"
```

### 7.3 rollback DFlash

```bash
tools/run_msprof.sh \
  --label dflash-pipe \
  --output-dir "$BENCH_DIR/msprof-dflash-pipe" \
  --python "$MODEL_PYTHON" \
  --aic-metrics PipeUtilization \
  --no-msproftx \
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
```

`run_msprof.sh` 不读取 Git/branch/dirty 信息，复制代码到目标机后可直接使用；manifest 通过源码
内容 hash 标识运行。wrapper 默认关闭 MSTX，并采 AI Core、task-time、runtime-api 和 AscendCL。
只有目标环境确认支持时才显式传 `--msproftx`；若出现 `mstx.range_start failed`，移除该选项后
重采。默认进程级 profile 还包含模型加载、correctness gate 和 warmup，不能把算子累计值或整个
profile 时长当作单次 measurement latency。

## 8. 报告门禁

`validate`：

```python
assert report["route"] == "qwen3.5-dflash-incremental-rollback"
assert report["execution_mode"] == "validate"
assert report["correctness_gate"]["status"] == "PASS"
assert report["strict_greedy_exact_match"] is True
assert report["historical_prefix_replay_during_verify"] is False
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["dflash_execution_gate"]["target_verify_calls"] > 0
assert report["draft_kv_cache_audit"]["mode"] == "upstream_equivalent_append_then_crop"
assert report["operator_fallback_enabled"] is False
```

`dflash` 单跑：

```python
assert report["execution_mode"] == "dflash"
assert report["ordinary"] is None
assert report["strict_greedy_exact_match"] is None
assert report["correctness_gate"]["status"] == "NOT_RUN_DFLASH_ONLY"
```

NPU rollback：

```python
audit = report["target_rollback_audit"]
assert audit["gdr_backend"] == "npu_gated_delta_rule_mtp"
assert audit["conv_bank_backend"] == "torch_tensor_golden_on_input_device"
assert audit["kv_policy"] == "physical_provisional_writes_logical_cursor_commit"
assert audit["prefill_execution_mode"] == "block_aligned_real_token_chunks_original_gdr"
assert audit["prefill_chunk_size"] == 64
assert audit["session_invalid"] is False
```

W8A8：

```python
quant = report["target_quantization"]
assert quant["status"] == "PASS_ASSEMBLY_CONTRACT_NO_NUMERICAL_CLAIM"
assert quant["scheme"] == "w8a8_dynamic"
assert quant["route"] == "rollback"
assert quant["linear_topology_validation"] == "PASS_EXACT_PATH_SHAPE_BIAS"
assert quant["qlinear_count"] > 0
assert quant["embedding_lookup_failures"] == 0
```

assembly PASS 只证明装配成功；数值结论仍来自同 activation 对照、整网 token/state 门禁和真实
NPU trace。

benchmark：

```python
assert report["benchmark"]["status"] == "PASS"
assert report["benchmark"]["summary"]["count"] == 10
assert report["strict_greedy_exact_match"] is True
assert report["operator_fallback_enabled"] is False
```

同时比较 `latency_ms`、aggregate tokens/s、peak memory、全部 10 条 measurement、acceptance、
Draft/verify calls 和 Target rows。`persistent_cursor` 等是 session state，不是累计 counter。

## 9. 自动化检查

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_scheduler.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_framework_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_internal_dflash_bridge_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_helpers.py
PYTHONDONTWRITEBYTECODE=1 python -m pytest -q \
  tests/test_run_npu_modes.py \
  tests/test_dflash_runtime_optimizations.py \
  tests/test_benchmark_npu.py \
  tests/test_msprof_script.py \
  tests/test_source_lock_benchmark.py \
  tests/test_rollback_target_quant.py
```

这些仍是 CPU/reduced-shape 证据。

## 10. 真机门禁顺序

1. B=2，连续多轮，accepted 0/1，ordinary token 零差异。
2. B=`2/4/6/8/16`，accepted `0/1/K-1/K`，最后一轮动态 T。
3. cursor `62/63/64/65`，rejected KV tail 不可见并被覆写。
4. 24 层 GDR/conv、8 层 KV、feature、position 共用同一个 `a`。
5. rejection 后继续至少一个 token，比较完整 state tuple。
6. 多 prompt、多进程重复，无状态泄漏、越界或持续内存增长。
7. 无 CPU fallback；记录 device/runtime/source/op/kernel identity。
8. 正确性闭合后再做 unprofiled 3+10 和单独 msprof 归因。

若发布 W8A8，还需分别通过 Target quant preflight、W8A8 ordinary/DFlash、W8A8 与要求的 FP16
ordinary 精度门禁。FP16 rollback PASS 不能替代量化 rollback PASS。
