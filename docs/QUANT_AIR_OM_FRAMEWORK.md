# 基于 `quant` 分支的 AIR → OM → C++ token 推理框架

## 1. 目标和边界

本框架直接建立在远端 `quant` 分支提交
`28f93e784a2beed87020a80bd93c8788754eab1c` 上。它不会把另一套模型冒充成量化模型：

- Target 由 `models.internal_dflash_bridge.load_qwen35_target` 加载；
- Target Linear 使用 `models.dflash_v1.original_quant.quant_model` 转成原有 W8A8
  `QLinear`；
- Target embedding 使用量化 YAML 指向的 INT8 weight 与 FP32 row scale；
- Target 输出边界、Draft、共享 embedding 和 LM head 保持 FP16；
- Draft 使用本分支现有 `models.dflash_v1.modeling_dflash.DFlashDraftModel`；
- greedy 接受规则保持 ordinary Target 权威，token ID、EOS 和 stop reason 必须零差异。

交付链路如下：

```text
Qwen3.5-4B Target checkpoint
W8A8 Linear / embedding artifacts
Qwen3.5-4B-DFlash checkpoint
quant 分支源码
        │
        │ TorchAir dynamo_export
        ▼
AIR graph + 外置权重文件 + air-manifest.json
        │
        │ atc --mode=0 --framework=1 --soc_version=<精确型号>
        ▼
quant_dflash_recompute.om + deployment-manifest.json
        │
        │ C++17 / AscendCL
        ▼
加载一次 OM → ordinary/DFlash 多轮调用 → 生成 token → EOS/长度停止
```

第一版冻结的是静态完整前缀重算 ABI。它能够由 OM 完成完整的 logits/Top1 推理，并由 C++
循环连续生成 token；但它尚未把 chunk-GDR 分支的 persistent rollback cache、标量 GDN state、
verify/commit capsule 和 Draft KV cache 变成显式 OM 输入/输出。因此：

- “是否真的调用 OM 生成完整 token 序列”可以验证；
- “是否达到闭源增量推理框架时延”必须在真实设备测量，当前不能预先声称；
- 如果 OM 执行时间主要消耗在完整前缀重算，下一阶段应拆成 prefill/decode/verify/draft
  增量 OM，而不是继续微调 Python。

这里必须区分“当前分支实际产物”和“最终低时延拓扑”：

| 阶段 | 逻辑图 | 物理 OM 数 | 当前状态 |
| --- | --- | ---: | --- |
| 当前可验证基线 | Target 全前缀 + Draft proposal 整图 | 每个静态 `S` gear 1 个 | 已接入 AIR/ATC/C++ |
| 低时延目标 | `target_prefill_64`、`target_decode_1`、`target_verify_16`、`target_state_commit_16`、`draft_16` | 至少 5 个；多 cache/shape gear 时更多 | 需要显式状态 ABI 与真机门禁 |

`verify` 不能与 ordinary `decode` 共用一个含糊的状态合同：前者一次计算最多 16 个 provisional
row，接受数产生后还要从同一个 round-start state 以 `a+1` 执行 state commit；后者固定提交
单个 row。用户口头所说的“prefill、decode、draft 三类”在物理部署中因此通常落为上述五个
OM 角色，除非 verify/accept/state-only commit 被证明可以安全融合。

## 2. 冻结 ABI

图名为 `quant_dflash_recompute`，batch 固定为 1：

| 顺序 | 名称 | dtype | shape | 含义 |
| --- | --- | --- | --- | --- |
| input 0 | `input_ids` | INT64 | `[1,S]` | 已提交前缀，右侧补 pad |
| input 1 | `attention_mask` | INT64 | `[1,S]` | 有效前缀为 1，右侧 padding 为 0 |
| output 0 | `target_top1` | INT64 | `[1,S]` | 每个物理 Target row 的 greedy Top1 |
| output 1 | `draft_top1` | INT64 | `[1,15]` | anchor 后最多 15 个 proposal |

`S` 是编译期静态 gear，必须是 64 的倍数。C++ runner 每次只读取
`target_top1[prefix_length-1]`，Draft 只读取 attention mask 声明的有效前缀。pad 位位于未来，
因果 Target 的有效 row 不依赖这些 pad row。

原 GDR 的新 ABI 还要求一个 `[B] INT16 effective_length`。它不是新增的 OM 外部输入：AIR 图
在内部统计 `attention_mask` 的非零行数并传给普通 GDR。比如 `S=64`、有效前缀为 37 时，
GDR 的物理序列仍为 64 行，但 `effective_length=[37]`；MTP verify 算子 ABI 不受这次改动影响。
由于该字段是 INT16，`S` 不能超过 32767。

### 2.1 `adn_rms_norm` 自定义算子保留合同

Target 的两个 modeling 文件继续直接调用 `torch_npu.adn_rms_norm`，没有加入 export-only Tensor
公式分支。PyTorch FX 与 GE/AIR 使用不同的算子名称层级：

```text
torch_npu.adn_rms_norm
        │ dispatcher schema: npu::adn_rms_norm(...)->(Tensor, Tensor)
        │ Fake/Meta 只声明 shape/dtype，不计算 RMSNorm
        ▼
npu.adn_rms_norm.default（FX 节点）
        │ TorchAir converter，一对一 lowering
        ▼
RmsNorm（GE/AIR 单节点，默认）
```

接收方实机已确认第一个输出与 input 同 shape/dtype，第二个输出为
`[*input.shape[:-1],1] FP32`。框架按这个合同注册 Fake/Meta；Fake 代码只创建 meta tensor，正式
NPU eager、AIR 和 OM 都不会执行它。converter 默认发出 CANN 已注册的 `RmsNorm` GE 节点，
不会拆成 `square/mean/rsqrt/mul` 链。

如果目标环境的私有算子包把同一 IR 注册成其他 GE type，例如 `AdnRmsNorm`，可以在
`factory.json` 中显式设置 `adn_rms_norm_ge_op_type`。该值不会自动回退：指定的 GE IR 未注册、
converter 没有命中或 `dynamo.pbtxt` 中没有该 type，导出都会失败，不会生成伪 PASS manifest。

如果 `S=64`，则 `prompt_tokens + generated_tokens` 不能超过 64。需要更长上下文时重新编译
`S=128/256/...`。当前重算路径的延迟随 `S` 增大，因此先使用能够覆盖验证 workload 的最小
gear。

完整合同见 `framework/FRAMEWORK_LOCK.json`。

## 3. 真实环境要求

正式验证必须满足：

1. 一块可访问的真实 Ascend 310P 设备；
2. 匹配驱动、固件的 CANN、Torch NPU、TorchAir 和 ATC；
3. AscendCL 头文件与 `libascendcl.so`；
4. `npu-smi info` 可用，且存在 `/dev/davinciN`；
5. Python 3.10、PyTorch、`torch_npu`、`torchair`、Transformers 5.14.1、PyYAML、
   safetensors；
6. 外置 Qwen3.5-4B、官方 Qwen3.5-4B-DFlash checkpoint；
7. 原量化 YAML 及其 `quanted_pth`、`embedding_weight_path`、
   `embedding_scale_path`；
8. receiver 的 `models/export_model_wrapper_qwen3_5.py`。

禁止用 CPU fallback 代替设备结论。仓库不保存 checkpoint、量化权重、AIR、OM、编译缓存、
日志或性能报告。

先确认代码分支确实基于 `quant`：

```bash
git switch framework/quant-air-om
git merge-base --is-ancestor \
  28f93e784a2beed87020a80bd93c8788754eab1c HEAD
```

第二条命令退出码必须为 0。

设置源码和外置运行目录。下面所有路径都必须替换成真实绝对路径：

```bash
export REPO_ROOT=/ABSOLUTE/PATH/qwen3.5-4B-dflash
export MODEL_PYTHON=/ABSOLUTE/PATH/python3
export AI_RUN_DIR=/ABSOLUTE/PATH/quant-air-om-run
export PYTHONPATH="$REPO_ROOT/framework/python:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$AI_RUN_DIR"
cd "$REPO_ROOT"
```

确认软件栈：

```bash
"$MODEL_PYTHON" - <<'PY'
import torch
import torch_npu
import torchair
print("torch", torch.__version__)
print("torch_npu", getattr(torch_npu, "__version__", "unknown"))
print("torchair", getattr(torchair, "__version__", "unknown"))
print("npu_available", torch.npu.is_available())
PY

npu-smi info
atc --version
```

任何 import、设备或 ATC 失败都应先修环境；不要让导出流程自动降级到 CPU。

## 4. 锁定外部模型和量化输入

先生成一次 hash-complete manifest。它会读取并哈希 Target、Draft、量化 Linear、量化
embedding、量化 YAML 和 receiver wrapper。大权重首次哈希需要时间，这是防止生成 AIR 前后
权重漂移的必要步骤。

```bash
"$MODEL_PYTHON" framework/scripts/lock_quant_inputs.py \
  --target-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --draft-dir /ABSOLUTE/PATH/Qwen3.5-4B-DFlash \
  --quant-config /ABSOLUTE/PATH/qwen3.5.yaml \
  --receiver-models-dir /ABSOLUTE/PATH/qwen35-runtime/models \
  --output "$AI_RUN_DIR/quant-input-manifest.json"
```

输入目录不能是 symlink，也不能在 manifest 生成后增加、删除或修改文件。AIR factory 会在加载
4B 权重前重新核验 manifest，并逐项核对仓库 `SOURCE_LOCK.json` 中的量化、Target、Draft、
bridge 和 rollback 源码 hash；任意字节变化都会失败。

复制并编辑 factory 配置：

```bash
cp config/quant_air_om_factory.example.json "$AI_RUN_DIR/factory.json"
```

`factory.json` 示例：

```json
{
  "target_dir": "/ABSOLUTE/PATH/Qwen3.5-4B",
  "draft_dir": "/ABSOLUTE/PATH/Qwen3.5-4B-DFlash",
  "quant_config": "/ABSOLUTE/PATH/qwen3.5.yaml",
  "input_manifest": "/ABSOLUTE/PATH/quant-air-om-run/quant-input-manifest.json",
  "receiver_models_dir": "/ABSOLUTE/PATH/qwen35-runtime/models",
  "max_sequence_length": 64,
  "example_sequence_length": 2,
  "pad_token_id": 0,
  "dtype": "float16",
  "device": "npu:0",
  "adn_rms_norm_ge_op_type": "RmsNorm",
  "name": "quant_dflash_recompute"
}
```

`receiver_models_dir` 用于补齐 `models.export_model_wrapper_qwen3_5`。量化分支本身仍提供
HIAI modeling、QLinear、DFlash 和 bridge；receiver wrapper 只复用原工程的加载和设备初始化。

## 5. 分层验证

### 5.1 Host 回归

这一步验证 Python 合同、量化 factory、manifest、ATC 命令生成、C++ scheduler 和 fake ACL。
它不是设备结论。

```bash
PYTHONDONTWRITEBYTECODE=1 \
  "$MODEL_PYTHON" -m pytest -q

cmake -S framework/runtime/cpp \
  -B "$AI_RUN_DIR/build/cpp-host" \
  -DCMAKE_BUILD_TYPE=Release \
  -DQWEN35_DFLASH_BUILD_ACL_RUNNER=OFF \
  -DQWEN35_DFLASH_BUILD_TESTS=ON
cmake --build "$AI_RUN_DIR/build/cpp-host" --parallel
ctest --test-dir "$AI_RUN_DIR/build/cpp-host" --output-on-failure
```

通过标准：所有 Python 测试和两个 CTest 都通过。fake ACL 测试会编译生产
`acl_executor.cpp`，但只证明 host 侧 buffer、调用顺序、scheduler 和 JSON 门禁。

### 5.2 量化 PyTorch 图探针

在导出前用真实权重执行一次图：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p probe-pytorch \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --input-token-ids 1,2,3,4 \
  --output "$AI_RUN_DIR/reports/pytorch-probe.json"
```

必须检查：

- `status == PASS`；
- `cpu_fallback == false`；
- `graph_metadata.target_quant_mode == "w8a8_dynamic"`；
- `graph_metadata.target_quantization_audit.status` 是量化 assembly PASS；
- `target_top1` shape 为 `[1,S]`；
- `draft_top1` shape 为 `[1,15]`；
- 输出 token ID 非负且小于词表大小。

这个探针仍不是 OM 证据，但能在耗时导出之前发现 checkpoint、量化 topology、embedding、
Draft 或 receiver ABI 错误。

## 6. 生成 AIR

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p export-air \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts/quant-dflash"
```

预期生成：

```text
$AI_RUN_DIR/artifacts/quant-dflash/
├── air-manifest.json
└── air/
    └── quant_dflash_recompute/
        ├── quant_dflash_recompute.air
        ├── dynamo.pbtxt
        └── TorchAir 生成的外置权重/辅助文件
```

导出器要求每个图恰好一个 `.air`，并把所有 payload 的大小和 SHA-256 写入
`air-manifest.json`。检查：

```bash
"$MODEL_PYTHON" - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["AI_RUN_DIR"]) / "artifacts" / "quant-dflash"
data = json.loads((root / "air-manifest.json").read_text())
assert data["status"] == "PASS"
assert data["schema_version"] == 2
assert len(data["graphs"]) == 1
graph = data["graphs"][0]
assert graph["name"] == "quant_dflash_recompute"
assert graph["input_names"] == ["input_ids", "attention_mask"]
assert graph["output_names"] == ["target_top1", "draft_top1"]
assert graph["metadata"]["quant_branch_base_revision"] == \
    "28f93e784a2beed87020a80bd93c8788754eab1c"
assert graph["metadata"]["gdr_effective_length_contract"] == \
    "INT16[B] call-local valid rows derived from attention_mask"
assert graph["metadata"]["target_quant_mode"] == "w8a8_dynamic"
audit = graph["custom_op_audit"]
assert len(audit) == 1
assert audit[0]["status"] == "PASS"
assert audit[0]["torch_target"] == "npu.adn_rms_norm.default"
assert audit[0]["ge_op_type"] == "RmsNorm"
assert audit[0]["converter_calls"] > 0
assert audit[0]["ge_node_occurrences"] > 0
print("AIR manifest gate: PASS")
PY
```

还可以直接检查 TorchAir 的可读 GE 图：

```bash
rg -n 'type: "RmsNorm"' \
  "$AI_RUN_DIR/artifacts/quant-dflash/air/quant_dflash_recompute/dynamo.pbtxt"
```

至少应命中一次。这里检查的是 AIR 内部 GE type；`npu.adn_rms_norm.default` 是前端 FX 名称，
不会原样作为 GE type 出现在 AIR。`air-manifest.json` 同时记录前端 converter 命中数和 GE 节点
数，并要求 GE 节点数覆盖全部 converter 调用，共同证明这些调用没有被 export-only Tensor
实现替换。

如果 `dynamo_export` 报 graph break，应保留完整 TorchAir 日志并定位首个不支持节点；不能用空
AIR、伪文件或 CPU export 替代。若错误已从 FakeTensor 阶段推进到另一个自定义算子，应为那个
算子单独确认 schema、真实输出元数据和 GE IR，不能套用 RMSNorm 的 Fake 合同。

## 7. AIR 编译成 OM

使用设备对应的精确 SoC 名称，例如真实环境确认是 `Ascend310P3` 时：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p compile-om \
  --air-manifest "$AI_RUN_DIR/artifacts/quant-dflash/air-manifest.json" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3
```

框架固定调用：

```text
atc --mode=0 --framework=1 \
    --model=<quant_dflash_recompute.air> \
    --output=<.../quant_dflash_recompute> \
    --soc_version=<精确 310P variant>
```

禁止使用模糊的 `Ascend310P`。成功后得到：

```text
$AI_RUN_DIR/artifacts/quant-dflash/
├── deployment-manifest.json
└── om/
    └── quant_dflash_recompute.om
```

编译器在调用 ATC 前重新核验 AIR 与所有外置 payload 的 hash；ATC 成功后记录 OM hash、ATC
版本、完整命令和日志。它还会重新校验并把 `custom_op_audit` 传入 deployment manifest；缺失
审计或 GE 节点数少于 converter 命中数时不会调用 ATC。退出码为 0 但 OM 缺失或为空也判定
失败。

也可以一次执行 AIR + OM：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-om \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_recompute_graph \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/artifacts/quant-dflash" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3
```

## 8. 构建 C++ AscendCL runner

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-release" \
  --output "$AI_RUN_DIR/reports/cpp-build.json" \
  --ascendcl-root /ABSOLUTE/PATH/CANN
```

生产 runner 使用：

- `aclInit`、`aclrtSetDevice`、显式 context/stream；
- `aclmdlLoadFromFile`，进程内只加载一次 OM；
- `aclrtMallocHost` pinned host buffer；
- 持久化 device buffer 和 dataset；
- 每轮排队 H2D、`aclmdlExecuteAsync`、D2H；
- 每次 OM 调用只做一次 `aclrtSynchronizeStream`；
- token 调度、DFlash 接受/correction/bonus、EOS 都在 C++17 中完成。

生产二进制通常位于：

```text
$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner
```

不要把 build 目录或二进制提交进源码仓库。

## 9. 用 C++ 调用 OM 完整生成 token

复制并填写真实运行时身份：

```bash
cp config/quant_air_om_runner.example.json "$AI_RUN_DIR/runner.json"
```

`device_model` 必须写具体产品和 310P variant；`cann`、`driver`、`firmware` 必须来自当前
设备，不能保留 `REPLACE_*`。

运行：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest \
    "$AI_RUN_DIR/artifacts/quant-dflash/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --chat \
  --max-new-tokens 32 \
  --max-draft-tokens 15 \
  --device-id 0 \
  --output "$AI_RUN_DIR/reports/cpp-infer.json"
```

这条命令中 Python 只负责一次 tokenizer、启动 runner 和最后 detokenize。进入 runner 后：

1. C++ 校验 OM SHA-256；
2. C++ 加载 OM；
3. C++ 运行 ordinary greedy 3 次 warmup + 10 次测量；
4. C++ 运行 DFlash strict-greedy 3 次 warmup + 10 次测量；
5. 每轮生成都反复调用 OM，直到 EOS 或 `max_new_tokens`；
6. C++ 比较 ordinary 与 DFlash token IDs、EOS/stop reason；
7. 任一 token 不一致立即失败。

所以它不是“Python 算完 token、C++ 只读结果”，而是 C++ 的 token 循环实际调用 OM 完成
Target/Draft 推理。

## 10. 一键端到端

先构建 C++ runner，然后执行：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e-cpp \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --bundle-dir "$AI_RUN_DIR/e2e/artifacts" \
  --atc /ABSOLUTE/PATH/atc \
  --soc-version Ascend310P3 \
  --runner "$AI_RUN_DIR/build/cpp-release/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir /ABSOLUTE/PATH/Qwen3.5-4B \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --chat \
  --max-new-tokens 32 \
  --max-draft-tokens 15 \
  --device-id 0 \
  --report-dir "$AI_RUN_DIR/e2e/reports"
```

它依次执行真实设备 preflight、AIR、ATC、OM hash gate、C++ paired generation 和最终
summary。任何阶段失败都不会伪造后续 PASS。

## 11. 如何判定框架“能用”

### 11.1 功能 PASS

最终 `cpp-infer.json` 或 `e2e/reports/summary.json` 必须同时满足：

```python
assert report["status"] == "PASS"
assert report["cpu_fallback"] is False
assert report["ordinary_parity"]["status"] == "PASS"
assert report["ordinary_parity"]["token_id_mismatches"] == 0
assert report["ordinary_parity"]["eos_mismatches"] == 0
assert report["ordinary"]["warmup"] == 3
assert report["ordinary"]["repetitions"] == 10
assert report["dflash"]["warmup"] == 3
assert report["dflash"]["repetitions"] == 10
assert report["ordinary"]["stable_generated_token_ids"] == \
       report["dflash"]["stable_generated_token_ids"]
```

此外检查：

- 报告中的 OM SHA-256 与 `deployment-manifest.json` 一致；
- device/cann/driver/firmware 与 `npu-smi` 和当前软件栈一致；
- 10 次输出 token 和 stop reason 完全稳定；
- prompt token 数加最大输出数不超过静态 gear；
- 没有 CPU fallback、mock、fake ACL 或 simulation 标记。

### 11.2 多 workload 正确性

至少覆盖：

1. 中文短 prompt；
2. 英文短 prompt；
3. code prompt；
4. 接近 gear 上限的长 prompt；
5. 会提前 EOS 的 prompt；
6. 生成到长度上限的 prompt；
7. Draft 全接受、首 token 拒绝和中间拒绝三类轮次；
8. 同一 prompt 重复进程执行，检查无状态泄漏。

每个 workload 都必须零 token/EOS 差异。不能只用一条容易全部接受的短文本证明框架正确。

### 11.3 性能 PASS

功能通过后才看性能。报告中使用 `latency_ms.model_total`，排除模型加载和 tokenizer；保留
完整 10 个原始值、median 和 p90。

需要与闭源框架比较时，闭源报告按
`framework/abi/closed-runtime-baseline-v1.json` 填写，必须相同：

- 物理设备和 device ID；
- CANN、driver、firmware；
- Target/Draft/量化输入 manifest；
- Target W8A8、Draft FP16；
- 静态 gear；
- prompt token IDs、输出 token IDs、EOS；
- concurrency=1；
- 模型加载排除；
- 显式设备同步的 model-loop 计时范围。

然后先给定可接受比率，例如 median/p90 均不得超过闭源的 1.10 倍：

```bash
"$MODEL_PYTHON" framework/scripts/compare_cpp_closed_runtime.py \
  --cpp-report "$AI_RUN_DIR/reports/cpp-infer.json" \
  --closed-report "$AI_RUN_DIR/reports/closed-runtime.json" \
  --max-median-ratio 1.10 \
  --max-p90-ratio 1.10 \
  --output "$AI_RUN_DIR/reports/cpp-vs-closed.json"
```

阈值必须在看结果前确定。即使 C++ 更快，只要 token/EOS 或运行身份不一致仍然 FAIL。

## 12. 常见失败定位

| 失败 | 含义 | 处理 |
| --- | --- | --- |
| input manifest hash mismatch | 权重、量化文件或 wrapper 已变化 | 先确认变化是否预期；重新冻结，不要跳过校验 |
| 找不到 `export_model_wrapper_qwen3_5.py` | receiver 路径错误 | 修正 `receiver_models_dir` |
| QLinear coverage mismatch | 量化权重与 Target topology 不同 | 核对 YAML、checkpoint revision 和 quant artifact |
| `torch_npu`/TorchAir import 失败 | 环境不匹配 | 使用与 CANN/驱动匹配的声明环境 |
| `unsupported operator: npu.adn_rms_norm.default` | 当前代码未加载 Fake/Meta 注册，或实际运行的不是本分支 | 核对远端提交、`PYTHONPATH` 和 `air-manifest` schema；不要改 modeling 为 Tensor fallback |
| `GE IR ... is not registered` | `adn_rms_norm_ge_op_type` 与目标 CANN/自定义包不一致 | 默认使用 `RmsNorm`；私有 type 必须先由同一环境正式注册 |
| custom-op converter/GE-node count 为 0 | 算子被绕开、converter 未调用或 GE 图丢失节点 | 导出按 FAIL 处理，保留 `dynamo.pbtxt` 和完整 TorchAir 日志 |
| TorchAir graph break | 某个 Python/自定义 op 未被捕获 | 定位首个 graph break，补正式 converter；不要伪造 AIR |
| ATC unsupported op | TorchAir 图中存在 ATC 不支持节点 | 保留算子名和编译日志，决定分解或正式自定义算子 |
| generic `Ascend310P` rejected | SoC 身份不精确 | 从设备/ATC 支持列表填写真实 variant |
| OM input/output count mismatch | 导出 ABI 漂移 | 必须恢复 2 input/2 output INT64 合同或版本化新 ABI |
| C++ OM hash mismatch | OM 被替换或 manifest 错配 | 使用同一次 build 的 OM 和 deployment manifest |
| ordinary/DFlash token mismatch | 接受、correction、pad 或图语义错误 | 停止性能测试，定位首个 token 分叉 |
| 延迟明显慢于闭源 | 完整前缀重算成为主瓶颈 | profile 后进入增量 OM state ABI，不要只优化 Python |

## 13. 下一性能阶段

当前 C++ 已消除 Python token 热循环、重复 OM load、重复 host/device buffer 分配和多余 stream
同步。若真实 profile 显示 OM 计算主导，下一步应基于 chunk-GDR rollback 语义拆分：

1. `target_prefill_64.om`：分块 prompt prefill；
2. `target_decode_1.om`：ordinary 单 token decode；
3. `target_verify_16.om`：anchor + 最多 15 proposals；
4. `target_state_commit_16.om`：消费 verify capsule、round-start state 和 `a+1`；
5. `draft_16.om`：增量 Draft KV；
6. 显式输入/输出 8 层 paged KV、24 层标量 GDR/conv state、8 层 feature、verify capsule 和 Draft KV；
7. C++ 负责同一 `a+1` 下的原子 commit/abort；
8. 对每一个 state 分支做 ordinary token/EOS 零差异门禁。

这一步是新的状态 ABI，不能在没有真实 baseline 和 state-branch 测试时悄悄替换当前功能基线。

## 14. 官方接口依据

- [TorchAir `dynamo_export`](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0052.html)：接口参数、静态/动态图、单图约束和 AIR 大小约束；
- [TorchAir 自定义算子 converter](https://www.hiascend.com/document/detail/zh/Pytorch/710/modthirdparty/torchairuseguide/torchair_00045.html)：自定义 ATen IR 需要注册 converter 后转换为 GE IR；
- [TorchAir AIR 产物与外置权重](https://www.hiascend.com/document/detail/zh/Pytorch/600/modthirdparty/torchairuseguide/torchair_0019.html)：`export.air`、`dynamo.pbtxt` 和 `weight_*` 的关系；
- [ATC `--framework`](https://www.hiascend.com/document/detail/en/canncommercial/800/devaids/atc/atlasatcparam_16_0014.html)：TorchAir 标准 AIR 使用 `--framework=1`；
- [AscendCL `aclmdlExecuteAsync`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0299.html)：异步模型执行接口；
- [AscendCL `aclrtMemcpyAsync`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0106.html)：stream 上的异步输入输出复制；
- [AscendCL `aclmdlLoadFromFile`](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0283.html)：从 OM 文件加载模型。
