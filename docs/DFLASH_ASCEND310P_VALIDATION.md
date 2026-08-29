# Qwen3.5-4B DFlash / Ascend 310P 验收手册

本文回答两个问题：

1. 框架代码是否具备从锁定 checkpoint 生成 OM 的完整路径；
2. 生成的 OM 是否真的被 AscendCL C++（推荐）或 pyACL（功能参考）调用，并完成
   prompt → token/text 的严格贪心推理。

代码路径已经实现，但“实现了路径”和“已在真机验证可用”是两个不同结论。当前仓库的
CPU/模拟测试已通过；当前声明的 `ascend310p-local` profile 是 `simulation-only`，没有
TorchAir、ATC、AscendCL/pyACL 或物理 310P，因此目前不能声称真实 OM 已生成或真机推理
已通过。
只有本文第 6～8 节全部通过，才能把框架标记为真实 310P 可用。

## 1. 框架到底做什么

完整链路如下：

```text
锁定 target + DFlash checkpoint
             │
             ▼
PyTorch 一体重计算图
  input_ids int64 [1,S]
  attention_mask int64 [1,S]
             │ TorchAir dynamo_export
             ▼
AIR + 外置权重 + SHA-256 manifest
             │ ATC mode=0/framework=1
             ▼
dflash_recompute.om
             │ AscendCL C++（低时延）或 pyACL（交叉验证）
             ▼
target_top1 int64 [1,S] + draft_top1 int64 [1,15]
             │ Python host scheduler
             ▼
ordinary greedy 或 DFlash 接受/纠错/bonus
             │ 循环至 EOS 或长度上限
             ▼
token IDs + 文本 + 分阶段延迟报告
```

OM 不是只跑一个孤立 drafter。默认 factory 把锁定的普通 Qwen3.5 target 和官方
DFlash drafter 组合成 `generation-recompute` 一体图。host 端每轮调用同一 OM：

- ordinary 模式只提交最后一个有效位置的 `target_top1`；
- DFlash 模式先取得 proposal，再用普通 target 输出逐 token 校验；首个不匹配时提交
  target correction，全部匹配时提交 target bonus；
- 最终生成序列必须与 ordinary 模式逐 token 完全一致，包括 EOS 位置。

首版是静态 batch-1、右补齐、完整前缀重算，尚未实现 KV/Gated DeltaNet state 的增量
提交与回滚。固定档位必须满足：

```text
prompt_token_count + max_new_tokens - 1 <= S
```

## 2. 什么结果才算“能用”

| 层级 | 验证内容 | 能证明什么 | 能否声称 310P 可用 |
| --- | --- | --- | --- |
| L0 | 契约、JSON/ABI、CLI 静态检查 | 仓库结构和参数自洽 | 否 |
| L1 | 77 项完整回归和专项控制流测试 | 导出/编译/runtime/scheduler 的 Python 逻辑可复现 | 否 |
| L2 | 锁定真实权重的 CPU `probe-pytorch` | target、8 层 feature、DFlash、Top1 接线可执行 | 否，仅模拟证据 |
| L3 | TorchAir 生成 AIR、ATC 生成非空 OM | 构图和编译链在目标工具链通过 | 否，还没证明 OM 可执行 |
| L4 | AscendCL C++ 在物理 310P 上跑 ordinary 与 DFlash 3+10 | OM 可加载、可执行、可持续生成 token | 是，功能可用 |
| L5 | 零 token/EOS mismatch、稳定性与延迟报告 | 严格贪心正确，且性能数据可审计 | 是，可形成准确率/性能证据 |

最低的真机功能验收是 L4；对本项目的正式验收要求是 L5。只有 L0～L2 通过时，应写
“框架实现和模拟验证通过，真实目标待验”，不能写“OM 推理通过”。

## 3. 每次验证先建立独立 workspace run

从 workspace 根目录执行：

```bash
kit/bin/ws context qwen3.5-4b --task model --target ascend310p
eval "$(kit/bin/ws session start \
  --model qwen3.5-4b --target ascend310p --task model --format shell)"
eval "$(kit/bin/ws env qwen3.5-4b \
  --target ascend310p --format shell --activate)"
```

先记录本次源码和环境身份：

```bash
git branch --show-current
git rev-parse HEAD
printf 'run=%s\nroot=%s\npython=%s\n' \
  "$AI_RUN_ID" "$AI_RUN_DIR" "$AI_MODEL_PYTHON"
```

所有配置、日志、AIR、OM 和报告只能放在 `$AI_RUN_DIR`。一次失败后不要覆盖旧目录；建立
新 session 重跑，这样 hash、日志和判定能一一对应。

## 4. L0/L1：任何机器都应先通过的验证

### 4.1 运行完整测试

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -B -m pytest -q -p no:cacheprovider \
  "$AI_MODEL_ROOT/tests" \
  2>&1 | tee "$AI_RUN_DIR/log/unit-tests.log"
```

本文编写时的预期结果是：

```text
82 passed, 1 skipped, 7 warnings, 3 subtests passed
```

其中专项测试覆盖：

- AIR 输出必须位于当前 run，且 AIR/外置权重全部记录 hash；
- TorchAir 缺失时必须在加载两份 4B checkpoint 前失败；
- ATC 固定使用 `framework=1`，禁止覆盖核心参数；
- 通用 `Ascend310P` SoC 名称会被拒绝，必须给出具体变体；
- AIR 或外置权重被篡改后，ATC 调用前即失败；
- OM hash、ABI 输入输出顺序和 int64 dtype 会被检查；
- CPU fallback、非目标设备元数据和非 3+10 计时会被拒绝；
- ordinary/DFlash backend 除 `ordinary_only` 外必须相同；
- 全接受、bonus、部分拒绝、target correction、EOS 和静态档位边界；
- ordinary 与 DFlash 的最终 token/EOS mismatch 必须为零。

如只想快速定位本框架，可先运行：

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -B -m pytest -q -p no:cacheprovider \
  "$AI_MODEL_ROOT/tests/test_dflash.py" \
  "$AI_MODEL_ROOT/tests/test_dflash_ascend310p.py"
```

### 4.2 检查 CLI 是否可加载

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p --help

PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e --help

PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e-cpp --help
```

必须能看到 `export-air`、`compile-om`、`build-om`、`build-cpp`、`probe-pytorch`、
`infer-om`、`infer-cpp`、`run-e2e` 和 `run-e2e-cpp`。CLI 可加载只证明 Python 安装和
模块路径正确。

## 5. L2：锁定真实权重 CPU smoke（可选、耗时）

这一步会加载真实 target 与 DFlash 权重，适合在没有 CANN 的机器上确认一体图接线。它
可能占用较多内存并运行数分钟，不适合作为每次快速回归。

在 `$AI_RUN_DIR/input/dflash-fp16-s2.json` 保存：

```json
{
  "max_sequence_length": 2,
  "example_sequence_length": 2,
  "device": "cpu",
  "dtype": "float16",
  "pad_token_id": 0,
  "verify_checkpoint": true
}
```

执行：

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p probe-pytorch \
  --factory-config "$AI_RUN_DIR/input/dflash-fp16-s2.json" \
  --input-token-ids 1,2 --threads 16 \
  --output "$AI_RUN_DIR/out/reference/integrated-fp16-s2.json"
```

报告应满足：

```text
status == PASS
scope == real-weight PyTorch integrated-graph probe
cpu_fallback == true
output_shapes.target_top1 == [1,2]
output_shapes.draft_top1 == [1,15]
所有输出 token ID 非负
```

`cpu_fallback=true` 是这里的预期值，也正是它不能替代真机结论的原因。这一步的耗时不能
当作 prefill、decode 或 310P latency。

## 6. L3：真实 CANN/TorchAir/ATC 环境预检与 OM 构建

### 6.1 切换到真实 target profile

当前 `ascend310p-local` 是 simulation profile。需要先在 `workspace.yaml`/目标 manifest 中
选择一个锁定的真实 profile，使一次环境激活同时暴露：

- `AI_MODEL_PYTHON` 和可执行的 `AI_TARGET_PREFLIGHT`；
- TorchAir、`torch_npu`、AscendCL C/C++ headers/动态库；pyACL 的 `acl` 模块只在 Python
  交叉验证路径中必需；
- 可执行 ATC，以及实际设备对应的具体 `soc_version`；
- 物理 Ascend 310P device node、驱动、固件和 CANN runtime。

严格预检：

```bash
"$AI_TARGET_PREFLIGHT" "$AI_MODEL_ROOT" \
  --require-model-python --require-atc --require-device \
  2>&1 | tee "$AI_RUN_DIR/log/strict-target-preflight.log"
```

随后检查关键模块和设备是否真的可见：

```bash
"$AI_MODEL_PYTHON" - <<'PY'
import acl
import torch
import torch_npu
import torchair

assert hasattr(torch, "npu"), "torch.npu is unavailable"
print("acl:", getattr(acl, "__file__", "built-in"))
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("torchair:", getattr(torchair, "__version__", "unknown"))
print("npu_count:", torch.npu.device_count())
assert torch.npu.device_count() > 0
PY

"$ASCEND310P_ATC_BIN" --version
npu-smi info
```

预检非零退出、任一 import 失败、`npu_count == 0` 或看不到物理设备时立即停止。simulation
profile 在这里失败是正确行为，不能用 `--allow-simulation` 绕过正式验收。

### 6.2 准备固定图档位

在 `$AI_RUN_DIR/input/dflash-fp16-s256.json` 保存：

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

`S=256` 只是示例。档位越大，一体图重算开销和内存压力越高。先用能容纳测试 prompt 的
最小档位跑通，再验证业务档位。`EXACT_SOC_VERSION` 必须来自实际 profile/设备，例如只有
设备确认为对应变体时才可填写 `Ascend310P3`，不得凭“310P”猜测。

### 6.3 只构建 AIR/OM（用于分步诊断）

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p build-om \
  --factory qwen35_dflash.ascend310p.factories:create_integrated_recompute_graph \
  --factory-config "$AI_RUN_DIR/input/dflash-fp16-s256.json" \
  --bundle-dir "$AI_RUN_DIR/out/dflash-ascend310p" \
  --atc "$ASCEND310P_ATC_BIN" \
  --soc-version "$EXACT_SOC_VERSION" \
  --atc-arg=--precision_mode=force_fp16 \
  2>&1 | tee "$AI_RUN_DIR/log/build-om.log"
```

至少应生成：

```text
$AI_RUN_DIR/out/dflash-ascend310p/air-manifest.json
$AI_RUN_DIR/out/dflash-ascend310p/air/dflash_recompute/dflash_recompute.air
$AI_RUN_DIR/out/dflash-ascend310p/om/dflash_recompute.om
$AI_RUN_DIR/out/dflash-ascend310p/deployment-manifest.json
$AI_RUN_DIR/log/dflash-atc/dflash_recompute.log
```

文件存在还不等于构建通过。`air-manifest.json`、`deployment-manifest.json` 必须都是
`status=PASS`，AIR 的所有 payload、OM 和 manifest hash 必须吻合，OM 必须非空。

## 7. L4/L5：真实 OM 完整 prompt → token 验收

正式低时延验收优先使用 `build-cpp` + `run-e2e-cpp`。它在单个 C++ 进程中一次加载 OM、
复用 pinned host/device buffer 和 dataset，并交错执行 ordinary/DFlash 3+10；完整命令、
报告字段和闭源框架 A/B 见
[AscendCL C++ 低时延推理](DFLASH_ASCEND310P_CPP_RUNTIME.md)。

下面 7.1～8 节保留 Python/pyACL 功能参考路径，便于交叉定位 ABI 与 runtime 问题；它不是
最终 host 性能上限。

### 7.1 准备成对 pyACL backend 配置

在 `$AI_RUN_DIR/input/backend-ordinary.json` 保存：

```json
{
  "graph_name": "dflash_recompute",
  "pad_token_id": 0,
  "device_model": "<具体 Atlas 产品与 310P 变体>",
  "cann": "<精确 CANN 版本>",
  "driver": "<精确驱动版本>",
  "firmware": "<精确固件版本>",
  "runtime": "<精确 pyACL/runtime 身份>",
  "ordinary_only": true
}
```

复制为 `$AI_RUN_DIR/input/backend-dflash.json`，只把最后一项改为 `false`。两个文件除
`ordinary_only` 外任何字段不同都会失败。身份字段必须来自设备和安装记录，不能写
`unknown` 或泛化的 `Ascend310P`。

### 7.2 pyACL 一键功能参考

对一个全新、空的 bundle/report 目录执行：

```bash
set -o pipefail
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
  --report-dir "$AI_RUN_DIR/out/performance" \
  2>&1 | tee "$AI_RUN_DIR/log/run-e2e.log"
```

这条命令按固定顺序执行：严格设备预检 → AIR → OM → ordinary 3 warmup + 10 measured →
DFlash 3+10 → token/EOS 零 mismatch → 最终 summary。任何一步异常都会非零退出；只有全部
通过才会写出：

```text
$AI_RUN_DIR/out/performance/ordinary.json
$AI_RUN_DIR/out/performance/dflash.json
$AI_RUN_DIR/out/performance/summary.json
```

### 7.3 证明调用的是 OM，而不是 CPU fallback

正式验收同时要求以下证据，缺一项都不要下真机结论：

1. strict preflight 原始日志显示真实 device 和 ATC 均通过；
2. `deployment-manifest.json` 记录非空 OM、SHA-256、ATC 路径/版本和具体 SoC；
3. C++ 路径实际执行 `aclInit`、`aclrtSetDevice`、`aclmdlLoadFromFile` 和
   `aclmdlExecuteAsync`；pyACL 参考路径对应执行 `acl.init`、`acl.rt.set_device`、
   `acl.mdl.load_from_file` 和 `acl.mdl.execute`；两种 runtime 都没有 CPU 执行后备路径；
4. ordinary 和 DFlash 报告的 `backend_metadata.cpu_fallback` 都必须为 `false`；
5. 两份报告记录相同 OM hash、device ID、CANN、driver、firmware 和 runtime；
6. 运行期间 `npu-smi info` 能独立观察到对应设备，必要时再采集目标 profile 声明的
   profiler/ACL runtime 证据；
7. `summary.json` 的所有 artifact hash 重新计算后仍一致。

仅在 backend/runner JSON 里手填设备名称不是硬件证据；它必须与 preflight、AscendCL
实际加载、OM hash 和独立设备观测共同成立。

## 8. 独立检查 pyACL 参考报告与所有 hash

C++ 路径使用 `performance-cpp/summary.json`，其独立验证和闭源 A/B 命令见
[C++ 文档第 7 节](DFLASH_ASCEND310P_CPP_RUNTIME.md#7-与闭源框架做同口径-ab)。本节脚本
针对 `run-e2e` 产生的 `ordinary.json` / `dflash.json` / `summary.json`。

下面的只读检查会重新计算 AIR payload、OM、报告和 summary 引用的 hash，并验证真机、
3+10、稳定性和严格贪心门禁：

```bash
"$AI_MODEL_PYTHON" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run = Path(os.environ["AI_RUN_DIR"]).resolve()
bundle = run / "out" / "dflash-ascend310p"
reports = run / "out" / "performance"

def load(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)

def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()

def check_record(root, record):
    path = Path(record["path"])
    path = path if path.is_absolute() else root / path
    path = path.resolve()
    assert path.is_file(), f"missing artifact: {path}"
    assert digest(path) == record["sha256"], f"hash mismatch: {path}"
    if "bytes" in record:
        assert path.stat().st_size == int(record["bytes"]), f"size mismatch: {path}"

def check_nested_records(value):
    if isinstance(value, dict):
        if "path" in value and "sha256" in value:
            check_record(run, value)
        else:
            for child in value.values():
                check_nested_records(child)
    elif isinstance(value, list):
        for child in value:
            check_nested_records(child)

air = load(bundle / "air-manifest.json")
deployment = load(bundle / "deployment-manifest.json")
ordinary = load(reports / "ordinary.json")
dflash = load(reports / "dflash.json")
summary = load(reports / "summary.json")

assert air["status"] == "PASS"
assert air["artifact_kind"] == "qwen35-dflash-torchair-bundle"
assert len(air["graphs"]) == 1
assert air["graphs"][0]["name"] == "dflash_recompute"
for graph in air["graphs"]:
    for record in graph["payload_files"]:
        check_record(bundle, record)

assert deployment["status"] == "PASS"
assert deployment["artifact_kind"] == "qwen35-dflash-ascend310p-om-bundle"
assert deployment["target"]["target_id"] == "ascend310p"
assert deployment["target"]["soc_version"].lower() not in {
    "310p", "ascend310p", "atlas310p"
}
check_record(bundle, deployment["air_manifest"])
for graph in deployment["graphs"]:
    check_record(bundle, graph["om"])
    assert graph["input_names"] == ["input_ids", "attention_mask"]
    assert graph["output_names"] == ["target_top1", "draft_top1"]

for name, report in (("ordinary", ordinary), ("dflash", dflash)):
    assert report["status"] == "PASS", name
    assert report["warmup"] == 3 and report["repetitions"] == 10, name
    assert len(report["measurements"]) == 10, name
    assert report["backend_metadata"]["cpu_fallback"] is False, name
    assert report["backend_metadata"]["device"]["target_id"] == "ascend310p", name
    assert all(
        item["generated_token_ids"] == report["stable_generated_token_ids"]
        and item["stop_reason"] == report["stable_stop_reason"]
        for item in report["measurements"]
    ), f"unstable {name} repetitions"

assert ordinary["report_kind"] == "ordinary-greedy-reference"
assert dflash["report_kind"] == "dflash-strict-greedy-target"
assert ordinary["stable_generated_token_ids"] == dflash["stable_generated_token_ids"]
assert ordinary["stable_stop_reason"] == dflash["stable_stop_reason"]
assert ordinary["backend_metadata"]["artifacts"] == dflash["backend_metadata"]["artifacts"]
assert dflash["ordinary_parity"]["token_id_mismatches"] == 0
assert dflash["ordinary_parity"]["eos_mismatches"] == 0

assert summary["status"] == "PASS"
assert summary["ordinary_parity"]["token_id_mismatches"] == 0
assert summary["ordinary_parity"]["eos_mismatches"] == 0
assert summary["output"]["token_ids"] == dflash["stable_generated_token_ids"]
check_nested_records(summary["artifacts"])

print("PASS: AIR/OM hashes, pyACL target metadata, 3+10 stability and token/EOS parity")
print("tokens:", summary["output"]["token_ids"])
print("text:", summary["output"]["text"])
print("median speedup:", summary["dflash_speedup_over_ordinary_median"])
PY
```

脚本打印 `PASS` 才表示报告内部自洽。它不能替代第 7.3 节的独立设备观测，但能阻止旧
OM、篡改文件、假 3+10、CPU fallback 或 token mismatch 被误当成通过。

## 9. 不要只测一个 prompt

一次 `run-e2e` 或 `run-e2e-cpp` 只形成一个 prompt 的配对证据。正式功能验收建议至少覆盖：

- 短中文问答；
- 短英文问答；
- 代码或结构化输出；
- 接近静态档位上限的长 prompt；
- 可能较早产生 EOS 的短回答。

每个 prompt 使用新 workspace session 和新输出目录，ordinary/DFlash 都保持同一 tokenizer、
同一 OM、同一设备身份、同一 `max_new_tokens`。所有 case 都必须零 token-ID/EOS mismatch。
性能对比则必须在同一档位、设备状态和测量协议下解读，不要把不同 session 的裸 median
直接拼成一个加速比。

## 10. 常见失败与定位顺序

### 10.1 checkpoint 加载前失败

- `soc_version must identify the concrete...`：使用了泛化 SoC 名，回到设备/profile 查具体变体；
- `ATC is unavailable...`：激活的 profile 没有声明可执行 ATC；
- strict preflight 非零：先看 `$AI_RUN_DIR/log/dflash-run-e2e-preflight.log`；
- `torchair` / `torch_npu` / `acl is required`：模型 Python 与目标 CANN 环境未正确组合；
- factory device 不是 `npu:*`：正式 `run-e2e` 禁止 CPU factory。

这些失败发生在加载 4B 权重前，是 fail-fast 设计，不要通过删除检查继续。

### 10.2 AIR 导出失败

- TorchAir graph break：从异常中的首个不支持 op/动态 Python 分支定位；
- NPU OOM：先减小 `S`，确认没有重复 target 实例，再检查 profile 的设备内存；
- checkpoint hash/revision 不符：不要跳过验证或替换小模型，修复 shared asset/manifest；
- 输出目录非空：新建 session，不覆盖旧 AIR。

### 10.3 ATC 编译失败

查看：

```bash
sed -n '1,240p' "$AI_RUN_DIR/log/dflash-atc/dflash_recompute.log"
```

优先核对具体 SoC、CANN/ATC 版本、AIR payload 是否齐全、外置权重 hash 和不支持算子。
不要手工改 AIR 后继续编译；manifest 会把这种情况判为篡改。

### 10.4 AscendCL C++ / pyACL 加载或执行失败

- OM hash 不匹配：确认 runtime 使用本 run 的 `deployment-manifest.json`；
- 输入/输出数目、shape 或 dtype 不符：核对 ABI 必须是两个 int64 输入和两个 int64 输出；
- `aclmdlLoadFromFile`/`aclmdlExecuteAsync` 或 pyACL 对应接口错误：保留 ACL 错误码、
  runner hash、驱动/CANN/固件身份和设备日志；
- 静态档位超限：增大 `S` 或减少 prompt/`max_new_tokens`，不要截断有效 token。

### 10.5 ordinary/DFlash 不一致

先检查两份 backend JSON 是否只差 `ordinary_only`，再按以下顺序排查：

1. tokenizer manifest、prompt、chat template 与 `max_new_tokens`；
2. `input_ids` 是否右补齐，`attention_mask` 是否为连续有效前缀；
3. DFlash row 0 是否只作为已提交 anchor，没有再次进入 target feature context；
4. logical position IDs 是否忽略 padding；
5. proposal 首个不匹配是否提交 target correction；
6. 全接受时是否提交 target bonus；
7. EOS 是否在两条路径的同一位置停止。

任何 mismatch 都以 ordinary 输出为权威；不能用“文本看起来一样”代替 token-ID 相等。

### 10.6 结果不稳定或性能很差

- 先确保 10 次 token IDs 和 stop reason 完全稳定；
- 确认每次 prefill/decode 前后都有 device synchronization；
- 记录温度、频率、并发负载和功耗策略；
- 分开观察 `prefill`、`decode`、`model_total`、`end_to_end`；
- 首版每轮重算完整前缀，且 ordinary 模式也执行图内未使用的 DFlash 计算，性能不代表未来
  增量 cache/pipeline 优化的上限。

## 11. 最终验收清单

只有下面全部为“是”，才能报告“框架已在真实 Ascend 310P 上生成并调用 OM 完成 token
推理”：

- [ ] 使用锁定的 `qwen3.5-4b` target 和官方 `qwen3.5-4b-dflash`，manifest/hash 全部通过；
- [ ] 完整测试通过；
- [ ] strict target preflight 退出码为 0；
- [ ] 真实 TorchAir、`torch_npu`、ATC、AscendCL C++ runtime 和物理设备都可见；
- [ ] `soc_version` 是具体 310P 变体；
- [ ] AIR、外置权重、OM 非空且 hash 完整；
- [ ] C++ runner 成功加载并执行 `dflash_recompute.om`，无 CPU fallback；pyACL 交叉验证按需通过；
- [ ] ordinary 3+10 和 DFlash 3+10 都稳定；
- [ ] 两条路径 token-ID mismatch 为 0，EOS mismatch 为 0；
- [ ] 输出 token IDs 能被锁定 tokenizer 解码为文本；
- [ ] 报告包含具体 device/CANN/driver/firmware/runtime 身份和所有原始延迟；
- [ ] 独立 hash/报告检查脚本通过；
- [ ] 至少一个代表性多 prompt 集合全部通过。
- [ ] 若声称“接近闭源框架”，同设备/同 token 的 median 与 p90 显式阈值比较也通过。

若任一项未完成，报告中应明确写 `PENDING` 或 `simulation evidence only`，并附上失败阶段、
原始日志和下一条可执行命令。
