# Ascend 310P 内部接入

## 1. 不需要拿到框架源码的接法

在内部网络新建一个 Python 模块，例如 `internal_qwen35_310p_backend.py`，导出：

```python
def create_backend(*, role, model_dir, options):
    if role == "main":
        return InternalMainBackend(...)
    if role == "draft":
        return InternalDraftBackend(...)
```

`InternalMainBackend` 提供 `backend_id` 和：

```python
evaluate(input_ids: Tensor, top1_positions: Sequence[int]) -> MainEvaluation
```

输出必须包含所有输入行的最终 normalized hidden `[1,S,2560]`，以及指定行经共享
LM head 得到的 Top1 `[1,R]`。`InternalDraftBackend.propose` 接收相同 hidden，返回
不超过请求数量的 token ID。

运行时指定：

```bash
python -m qwen35_mtp compare \
  --model-dir /internal/path/Qwen3.5-4B \
  --main-backend internal_qwen35_310p_backend:create_backend \
  --draft-backend internal_qwen35_310p_backend:create_backend \
  --prompt '测试文本' --chat --max-new-tokens 8
```

## 2. 第一阶段：普通模型

先只实现 main backend：

1. 使用现有算子构造官方 32 层 text graph；
2. 暴露最终 normalized hidden；
3. 对请求的行执行 tied LM head + 稳定 Top1；
4. 可先整段前缀重算，不要求增量 cache；
5. 与本仓库 PyTorch ordinary 按 token 和中间 hidden 对比。

这是目标模型基线。它不过，不能开始归因 MTP。

## 3. 第二阶段：官方 MTP

可选择：

- 实现整个 draft backend；或
- 提供 `rms_norm/linear/attention/swiglu/top1` 五个函数，通过 `--ops-backend`
  替换纯 PyTorch函数。

shape 契约在 `targets/ascend310p/abi/runtime-v1.json`。初版同样允许从已提交前缀
重建 draft KV。默认 K=2，与 Qwen 官方 vLLM 推荐配置一致。

## 4. strict target 规则

- target backend/op 缺失立即失败；
- 禁止 target run 使用 `--allow-op-fallback`；
- 报告必须包含 backend ID、CANN/驱动/固件、设备型号、artifact SHA-256；
- CPU/ONNX 输出只能标 `simulation`；
- Atlas 300I Duo 与 Atlas 200I Pro 的容器、驱动挂载和内存能力需分别记录，不能只写
  “310P”。

[SelfAttentionOperation](https://www.hiascend.com/document/detail/zh/mindie/1.0.RC1/mindiert/rtdev/ascendtb_01_0076.html)
和 [ActivationOperation](https://www.hiascend.com/doc_center/source/zh/mindie/1.0.RC1/mindiert/rtdev/ascendtb_01_0049.html)
均列出 310P 的 BF16 限制。因此，FP16 不是隐式默认值。该实验已获明确批准，并已
通过单个真实文本 CPU admission case，但仍必须在内部先关闭 ordinary 精度，再检查
MTP 是否引入额外偏差；CPU PASS 不能替代真机结论。

## 5. 固定 S1/P1 MTP core 板端冒烟

候选 ONNX 的输入顺序是：

```text
inputs_embeds, hidden_sources, position_ids, past_key, past_value
```

它不包含共享 embedding、tied LM head 或 Top1；这三项由普通主模型现有算子复用。
固定 case 同时提供 `.npy` 和无头 `.bin`，shape/dtype/hash 以 case 的
`manifest.json` 为准。先用内部转换器生成目标 artifact，然后严格按上述顺序送入，
取回 `mtp_hidden, present_key, present_value`。禁止运行时回退 CPU。

本地可复核导出图：

```bash
PYTHONPATH=model python targets/ascend310p/scripts/validate_mtp_onnx.py \
  --input /path/to/qwen35-4b-mtp-core-s1-p1-fp16.onnx \
  --output /tmp/onnx-validation.json

PYTHONPATH=model python targets/ascend310p/scripts/run_onnx_case.py \
  --model /path/to/qwen35-4b-mtp-core-s1-p1-fp16.onnx \
  --case-dir /path/to/real-text-nihao-comma-s1-p1 \
  --output /tmp/onnxruntime-compare.json
```

## 6. 路线跑通后的增量优化

再实现 target transaction：批量 verify `[base,d1,d2]`，只提交正确前缀 state。
Qwen3.5 main 同时含 full-attention KV 和 Gated DeltaNet recurrent/conv state；每种
`accepted=0,1,2` 都必须与 ordinary replay 后的 state 及下一个 token 对齐。增量实现
只是性能替换，不改变 scheduler acceptance 公式。

## 7. 真机 benchmark 与 msprof

性能测量必须使用真实 target profile，禁止 `--allow-op-fallback`。普通生成和 MTP
分别启动进程，不能用 `compare` 的先 ordinary 后 MTP 顺序作为性能样本。统一契约见
`targets/ascend310p/abi/performance-v1.json`。

先在无 profiler 条件下保留 3 次 warmup 后的 10 次原始真机结果：

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" "$AI_MODEL_PYTHON" -m qwen35_mtp benchmark \
  --mode ordinary \
  --model-dir "$MODEL_DIR" \
  --main-backend "$MAIN_BACKEND" \
  --prompt-token-ids "$PROMPT_TOKEN_IDS" \
  --device npu:0 --dtype float16 \
  --max-new-tokens 32 --warmup 3 --repetitions 10 \
  --output "$AI_RUN_DIR/out/performance/ordinary.json"

PYTHONPATH="$AI_MODEL_ROOT/model" "$AI_MODEL_PYTHON" -m qwen35_mtp benchmark \
  --mode mtp \
  --model-dir "$MODEL_DIR" \
  --main-backend "$MAIN_BACKEND" \
  --draft-backend "$DRAFT_BACKEND" \
  --prompt-token-ids "$PROMPT_TOKEN_IDS" \
  --device npu:0 --dtype float16 \
  --max-new-tokens 32 --max-draft-tokens 2 \
  --warmup 3 --repetitions 10 \
  --output "$AI_RUN_DIR/out/performance/mtp.json"
```

backend 应实现无参数 `synchronize()`；benchmark 会在每次计时前后调用它。每轮还会按
`reset_benchmark_state()`、`reset_state()`、`clear_cache()` 的优先级调用第一个可用的
状态清理 hook，并检查所有实测轮次的 token IDs 完全一致。

msprof 只跑 1 至 3 个代表性 iteration，用于诊断而不是替代上述无 profiler 基线：

```bash
targets/ascend310p/scripts/run_msprof.sh --label mtp-pipe -- \
  env PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_mtp benchmark \
  --mode mtp \
  --model-dir "$MODEL_DIR" \
  --main-backend "$MAIN_BACKEND" \
  --draft-backend "$DRAFT_BACKEND" \
  --prompt-token-ids "$PROMPT_TOKEN_IDS" \
  --device npu:0 --dtype float16 \
  --max-new-tokens 32 --max-draft-tokens 2 \
  --warmup 3 --repetitions 2 --enable-mstx \
  --output "$AI_RUN_DIR/out/performance/mtp-msprof.json"
```

包装脚本默认采集 AscendCL、Runtime API、task time、AICPU、AI Core、
`PipeUtilization` 和 MSTX，产物只写入 `$AI_RUN_DIR/profile/msprof`、`out/performance`
及 `log`。分析内存时应另跑 `--aic-metrics Memory` 或 `MemoryUB`，不要把不同 PMU
指标混入同一次基线。
