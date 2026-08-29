# Qwen3.5-4B DFlash：PyTorch → AIR → OM → Ascend 310P

## 当前实现

默认路径不再要求外部工程自行拼出 target/verify backend。仓库内已有一条完整的首版
重计算路线：

1. 从 `project.yaml`、`specs/data.lock.json` 和 shared manifest 解析并校验
   `qwen3.5-4b` 与 `qwen3.5-4b-dflash`；
2. 加载 text-only Qwen3.5 target 与官方 69-tensor DFlash checkpoint；
3. 构造一个固定档位、batch-1 的 `generation-recompute` 图；
4. 用 TorchAir `dynamo_export` 生成 AIR 和外置权重，并记录逐文件 SHA-256；
5. 用 ATC `mode=0/framework=1` 生成 OM；
6. 低时延路径用原生 AscendCL C++ 加载 OM 并在一个进程内完成两种调度；pyACL 路径保留
   为功能参考；
7. 对每次 prefill/decode 前后同步设备，输出 3 次 warmup + 10 次实测的原始值与汇总；
8. DFlash 报告必须与同一 OM 生成的 ordinary 报告保持零 token-ID/EOS mismatch。
9. `run-e2e-cpp` 把上述构建、C++ 普通基线、DFlash 对照和最终汇总作为首选 fail-fast
   target 工作流；`run-e2e` 是等价的 Python/pyACL 功能参考。

当前 workspace 的 `ascend310p-local` profile 仍是 `simulation-only`，没有 TorchAir、ATC、
AscendCL/pyACL 或 310P。本轮已有锁定真实权重的 FP16 CPU 一体图 PASS 和 C++ host/fake-ACL
PASS，但它们不能替代 AIR、OM、310P 精度或性能证据。

需要实际判断“能否生成 OM、pyACL 是否真的调用 OM、能否从 prompt 完整生成 token/text”时，
按 [DFlash Ascend 310P 验收手册](DFLASH_ASCEND310P_VALIDATION.md) 逐级执行；其中明确给出
真机 PASS 门槛、独立 hash/报告检查和失败定位顺序。

正式低时延路径推荐使用 [AscendCL C++ runner](DFLASH_ASCEND310P_CPP_RUNTIME.md)：OM 加载、
预分配 buffer、ordinary/DFlash 调度和 3+10 均留在一个 C++ 进程中。本文后面的 pyACL 路线
仍保留为功能参考和交叉验证，不应被当作最终 host 性能上限。

## 一体图 ABI 与严格贪心语义

图输入是右补齐的静态张量：

```text
input_ids      int64 [1,S]
attention_mask int64 [1,S]
```

图输出是：

```text
target_top1 int64 [1,S]
draft_top1  int64 [1,15]
```

锁定 checkpoint 的 `block_size=16`。与 DFlash 官方实现一致，block 第 0 行是已经提交的
当前 token；它不再进入 target feature context。第 1–15 行才是 proposal。校验时，从
当前 token 对应的 target row 开始比较 proposal；遇到首个不匹配即提交 target correction，
全部匹配则提交 target bonus。最终 token 永远由 ordinary target 的 Top1 决定。

首版不保存 target KV、Gated DeltaNet state 或 draft KV。每次调用都按已提交前缀重算，
因此没有未验证的 state commit/rollback。容量条件是：

```text
prompt_token_count + max_new_tokens - 1 <= S
```

## 1. 建立 workspace run 和真机环境

先执行 workspace 规定的上下文、session 和一次性环境激活：

```bash
kit/bin/ws context qwen3.5-4b --task model --target ascend310p
eval "$(kit/bin/ws session start --model qwen3.5-4b --target ascend310p --task model --format shell)"
eval "$(kit/bin/ws env qwen3.5-4b --target ascend310p --format shell --activate)"
"$AI_TARGET_PREFLIGHT" "$AI_MODEL_ROOT" \
  --require-model-python --require-atc --require-device
```

preflight 必须确认 TorchAir、`torch_npu`、ATC、pyACL 与具体 310P 设备；simulation profile
只能得到 PENDING，不能继续形成 target 结论。

## 2. 固定导出档位

在 `$AI_RUN_DIR/input/` 中准备例如 `dflash-fp16-s256.json`：

```json
{
  "max_sequence_length": 256,
  "example_sequence_length": 8,
  "device": "npu:0",
  "dtype": "float16",
  "pad_token_id": 0,
  "verify_checkpoint": true
}
```

默认 factory 是：

```text
qwen35_dflash.ascend310p.factories:create_integrated_recompute_graph
```

它会验证两个 manifest、revision、base revision、tensor count/bytes/shards 以及 checkpoint
文件 hash。若现有 torch_npu target 已实现选择性 `output_dflash_features`，可在配置中设置
`target_factory` 和 `target_factory_config` 替换默认 Transformers all-hidden-state adapter；
输入输出 ABI 与 host scheduler 不变。

## 3. 导出 AIR 并编译 OM

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p build-om \
  --factory qwen35_dflash.ascend310p.factories:create_integrated_recompute_graph \
  --factory-config "$AI_RUN_DIR/input/dflash-fp16-s256.json" \
  --bundle-dir "$AI_RUN_DIR/out/dflash-ascend310p" \
  --atc "$ASCEND310P_ATC_BIN" \
  --soc-version "$EXACT_SOC_VERSION" \
  --atc-arg=--precision_mode=force_fp16
```

`EXACT_SOC_VERSION` 必须来自实际设备/profile，不能根据“310P”猜测。ATC 的
`mode/framework/model/output/soc_version` 是锁定参数，extra args 不能覆盖。
泛化的 `Ascend310P` 会被拒绝，必须使用实际 ATC 变体（例如设备确认为该变体时使用
`Ascend310P3`）。`build-om` 会在加载两份 4B checkpoint 前先检查 SoC 和 ATC executable；
`export-air` 也会在调用 graph factory 前先导入 TorchAir，避免缺工具时先消耗大量内存和
时间。

输出包括：

```text
air-manifest.json
deployment-manifest.json
air/dflash_recompute/dflash_recompute.air
air/dflash_recompute/<external weights>
om/dflash_recompute.om
```

TorchAir 没有独立的输出命名参数。runtime 因此按 deployment manifest 的有序 ABI 名给
OM binding 做索引别名，同时记录真实 OM tensor 名；不依赖内部生成名称恰好等于 Python
变量名。

## 4. 准备 backend 身份

ordinary 与 DFlash 使用同一 OM，只切换主机调度。两个 JSON 除 `ordinary_only` 外必须相同。
例如 `$AI_RUN_DIR/input/backend-ordinary.json`：

```json
{
  "graph_name": "dflash_recompute",
  "pad_token_id": 0,
  "device_model": "<concrete Atlas product and 310P variant>",
  "cann": "<exact CANN version>",
  "driver": "<exact driver version>",
  "firmware": "<exact firmware version>",
  "runtime": "<exact pyACL/runtime identity>",
  "ordinary_only": true
}
```

`backend-dflash.json` 把最后一项改为 `false`。不能把 `device_model` 写成泛化的
`Ascend310P`；报告必须能追溯到具体产品和 device ID。

## 5. 推荐：一键 target 流程

准备好上述 factory 和两个 backend JSON 后，推荐直接运行：

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e \
  --factory-config "$AI_RUN_DIR/input/dflash-fp16-s256.json" \
  --bundle-dir "$AI_RUN_DIR/out/dflash-ascend310p" \
  --atc "$ASCEND310P_ATC_BIN" \
  --soc-version "$EXACT_SOC_VERSION" \
  --atc-arg=--precision_mode=force_fp16 \
  --ordinary-backend-config "$AI_RUN_DIR/input/backend-ordinary.json" \
  --dflash-backend-config "$AI_RUN_DIR/input/backend-dflash.json" \
  --prompt '请简要介绍昇腾310P。' --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --report-dir "$AI_RUN_DIR/out/performance"
```

该命令没有可覆盖的 warmup/repetition 参数，target 协议固定为 3+10。它会先验证：

- 具体 ATC SoC 变体和可执行 ATC；
- workspace 声明的 `--require-model-python --require-atc --require-device` strict preflight，
  并把原始输出写入 run log；
- TorchAir，以及内置 factory 所需的 `torch_npu`；
- 内置 runtime 所需的 pyACL；
- factory 的显式 NPU device；
- 两个 backend JSON 除 `ordinary_only=true/false` 外逐字段完全相同；
- bundle 与 report 目录互不重叠且没有旧输出。

只有 AIR、OM、ordinary 3+10、DFlash 3+10 和零 mismatch 全部通过，才写入：

```text
out/performance/ordinary.json
out/performance/dflash.json
out/performance/summary.json
```

`summary.json` 直接包含 prompt 输出文本/token IDs、设备和 runtime 身份、AIR/OM/report
hash、strict preflight log hash、ordinary/DFlash 的 prefill、decode、model total、
end-to-end 分布，以及两者 median 比值。任一步失败都不会生成 PASS summary。

以下两节保留为需要分步诊断时的等价手工流程。

## 6. 先生成 ordinary 权威报告

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-om \
  --deployment-manifest "$AI_RUN_DIR/out/dflash-ascend310p/deployment-manifest.json" \
  --backend qwen35_dflash.ascend310p.recompute_backend:create_backend \
  --backend-config "$AI_RUN_DIR/input/backend-ordinary.json" \
  --prompt '请简要介绍昇腾310P。' --chat \
  --max-new-tokens 32 --max-draft-tokens 15 \
  --warmup 3 --repetitions 10 --device-id 0 \
  --output "$AI_RUN_DIR/out/performance/ordinary.json"
```

tokenizer 默认从 project 声明的锁定 `qwen3.5-4b` asset 解析并校验；也可显式传
`--model-asset-id qwen3.5-4b`。ordinary 模式每次只提交最后一个有效 row 的
`target_top1`，报告标记为 `ordinary-greedy-reference`。

## 7. 运行 DFlash 并强制零 mismatch

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-om \
  --deployment-manifest "$AI_RUN_DIR/out/dflash-ascend310p/deployment-manifest.json" \
  --backend qwen35_dflash.ascend310p.recompute_backend:create_backend \
  --backend-config "$AI_RUN_DIR/input/backend-dflash.json" \
  --ordinary-reference "$AI_RUN_DIR/out/performance/ordinary.json" \
  --prompt '请简要介绍昇腾310P。' --chat \
  --max-new-tokens 32 --max-draft-tokens 15 \
  --warmup 3 --repetitions 10 --device-id 0 \
  --output "$AI_RUN_DIR/out/performance/dflash.json"
```

target 模式没有 `--ordinary-reference` 会在计时前失败。门禁会比较：

- OM SHA-256；
- device、CANN、driver、firmware、runtime identity；
- tokenizer manifest、prompt、chat 模式和 `max_new_tokens`；
- 两边各自 10 轮的 token 稳定性和 stop reason；
- ordinary 与 DFlash 的完整生成 token IDs 和 EOS。

只有 `token_id_mismatches=0` 且 `eos_mismatches=0` 才写入 PASS DFlash 报告。

## 8. 延迟口径

每个 prefill/decode 调用前后均执行 device synchronization。报告包含 10 轮原始值及
min/max/mean/median/p90/population stdev：

- `prefill`：一次 integrated OM 调用并提交首个 target token；
- `decode`：后续所有 proposal/verify 调用之和；
- `model_total`：`prefill + decode`；
- `tokenize`、`detokenize`：纯主机 tokenizer 时间；
- `end_to_end`：tokenize 开始到 detokenize 完成；
- `time_to_first_token`：tokenize + synchronized prefill。

当前单 OM 方案避免重复驻留两份 4B target 权重，但 verification 调用仍会执行图内 DFlash，
ordinary 模式也会计算未使用的 DFlash 输出。报告衡量的是这条首版可验证路线的真实延迟，
不能解释成已完成 cache/pipeline 优化后的上限。

## 9. CPU 实权重 smoke（仅模拟）

`probe-pytorch` 可以在没有 CANN 的机器上验证锁定 target、8 层 feature、DFlash block 与
Top1 接线。把 factory 配置中的 `device` 改为 `cpu` 后运行：

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p probe-pytorch \
  --factory-config "$AI_RUN_DIR/input/dflash-fp16-s2.json" \
  --input-token-ids 1,2 --threads 16 \
  --output "$AI_RUN_DIR/out/reference/integrated-fp16-s2.json"
```

该报告显式记录 `cpu_fallback=true`。它只能证明 PyTorch 接线，不能用于 AIR/OM、310P
精度、性能或 acceptance-rate 结论。

## 失败即停止

以下任一情况不得形成 target PASS：checkpoint/manifest hash 不符、target/draft revision
不匹配、TorchAir graph break、AIR/OM 为空、ATC 非零退出、OM ABI shape/dtype/count 不符、
CPU fallback、设备身份泛化、非有限值、负 token、10 轮不稳定、ordinary/DFlash token 或
EOS 不一致。
