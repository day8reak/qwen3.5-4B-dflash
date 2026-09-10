# Python NPU：从环境准备到运行、验证和 msprof

本文按顺序运行 Qwen3.5-4B ordinary/DFlash，并采集单次 prefill、decode 或 Draft/verify。
支持 batch=1、strict greedy，Target 可选 FP16/W8A8，Draft 运行时为 FP16。
C++ OM 的完整部署步骤见 [AIR → OM → C++ 操作手册](GDR_CHUNK_AIR_OM.md)。

## 1. 准备源码和目录

需要 Linux、可访问的 Ascend 310P、与驱动/固件匹配的 CANN 和 NPU Python 环境。
先把下面的绝对路径改成实际路径。`AI_RUN_DIR` 使用源码和权重目录之外的全新目录；
使用模型 workspace 时，由 `ws session start` 分配该目录，不要覆盖已有 session 的路径。

```bash
export REPO_ROOT=/absolute/path/qwen3.5-4B-dflash
export AI_RUN_DIR=/absolute/path/qwen35-run
export MODEL_PYTHON=/absolute/path/npu-env/bin/python
export CANN_ROOT=/absolute/path/ascend-toolkit
export TARGET_DIR=/absolute/path/Qwen3.5-4B
export DRAFT_DIR=/absolute/path/Qwen3.5-4B-DFlash
export RECEIVER_ROOT=/absolute/path/qwen35-receiver
export RECEIVER_MODELS_DIR="$RECEIVER_ROOT/models"

# REPO_ROOT 必须是尚未创建的目录。
git clone --single-branch --branch feature/gdr-chunk-verify \
  https://github.com/day8reak/qwen3.5-4B-dflash.git "$REPO_ROOT"
mkdir -p "$AI_RUN_DIR/log" "$AI_RUN_DIR/reports" "$AI_RUN_DIR/cache" "$AI_RUN_DIR/tmp"
cd "$REPO_ROOT"

source "$CANN_ROOT/set_env.sh"
export PYTHONDONTWRITEBYTECODE=1
export TMPDIR="$AI_RUN_DIR/tmp"
export HF_HOME="$AI_RUN_DIR/cache/huggingface"
export TORCH_HOME="$AI_RUN_DIR/cache/torch"
export XDG_CACHE_HOME="$AI_RUN_DIR/cache"
export PYTHONPATH="$REPO_ROOT/framework/python:$REPO_ROOT:$RECEIVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
```

权重、AIR、OM、编译目录、日志和报告均放在源码仓库外。后续命令在同一个 Bash 会话执行，
使用这里定义的变量。源码已经位于 `REPO_ROOT` 时，从 `mkdir` 开始执行，无需再次 clone。

## 2. 安装依赖并检查 NPU

`MODEL_PYTHON` 使用 Python 3.10，并已安装与 CANN 配套的 PyTorch 和 `torch_npu`。
CANN 驱动、固件、PyTorch/NPU 扩展及设备自定义算子包需要由设备软件栈提供；本仓库不包含
这些安装包。不要用普通 CPU PyTorch 覆盖 NPU 环境中的 PyTorch。

安装模型侧依赖：

```bash
"$MODEL_PYTHON" -m pip install numpy PyYAML safetensors huggingface-hub "transformers==5.14.1"

npu-smi info | tee "$AI_RUN_DIR/log/npu-smi.txt"
"$MODEL_PYTHON" -B - <<'PY'
import torch
import torch_npu
assert torch.npu.is_available(), "没有可用 NPU"
torch.npu.set_device("npu:0")
print("torch:", torch.__version__)
print("torch_npu:", getattr(torch_npu, "__version__", "unknown"))
print("device:", torch.npu.get_device_name(0))
PY
```

按自定义算子包的安装说明完成注册后，检查 Target 所需接口：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import torch_npu
required = (
    "npu_chunk_gated_delta_rule", "adn_fused_infer_attention",
    "adn_rms_norm", "npu_cache_update_",
)
missing = [name for name in required if not callable(getattr(torch_npu, name, None))]
assert not missing, f"缺少 NPU 算子: {missing}"
print("Target operator symbols: PASS")
PY
```

`npu_chunk_gated_delta_rule` 必须接受 `effective_length: INT16[B]`，含义是本次调用的有效行数。
接口形式为：

```text
npu_chunk_gated_delta_rule(query, key, value, g, beta, effective_length,
                         chunk_size=64, initial_state=None,
                         output_final_state=False, use_qk_l2norm_in_kernel=False)
```

符号检查通过后，仍需通过后面的真实模型执行验证 shape、dtype 和数值。

## 3. 准备模型和外部加载器

需要完整的 Target checkpoint、tokenizer，以及官方 Draft checkpoint。已有匹配文件时，直接
设置好 `TARGET_DIR` 和 `DRAFT_DIR`；需要下载时，以下命令从 `SOURCE_LOCK.json` 读取锁定版本：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import snapshot_download
lock = json.loads((Path(os.environ["REPO_ROOT"]) / "SOURCE_LOCK.json").read_text())
for section, directory in (("target_checkpoint", "TARGET_DIR"), ("dflash_checkpoint", "DRAFT_DIR")):
    item = lock[section]
    snapshot_download(repo_id=item["repository"], revision=item["revision"],
                      local_dir=os.environ[directory])
PY
```

Draft 文件保留 checkpoint 中的 BF16 数据，运行时加载为 FP16；不要预先改写 checkpoint。
加载器会检查 Draft 的配置、6 层/69 tensor、shape、dtype 和文件 hash。

还需要外部文件 `$RECEIVER_MODELS_DIR/export_model_wrapper_qwen3_5.py` 及其依赖。
该文件由 Qwen3.5 NPU receiver 软件包提供，本仓库不附带。它必须导出
`Qwen3_5ForCausalLMWrapper` 类和模块级 `Qwen3_5ForCausalLM`，负责权重加载及设备初始化。
将完整 receiver 软件包放到 `RECEIVER_ROOT`，然后建立本次运行专用的模块搜索配置：

```bash
mkdir -p "$AI_RUN_DIR/python-bootstrap"
cat > "$AI_RUN_DIR/python-bootstrap/sitecustomize.py" <<'PY'
import os
import models
receiver_models = os.environ.get("RECEIVER_MODELS_DIR")
if receiver_models and receiver_models not in models.__path__:
    models.__path__.append(receiver_models)
PY
export PYTHONPATH="$AI_RUN_DIR/python-bootstrap:$REPO_ROOT/framework/python:$REPO_ROOT:$RECEIVER_ROOT${PYTHONPATH:+:$PYTHONPATH}"

"$MODEL_PYTHON" -B - <<'PY'
import importlib.util, os
from pathlib import Path
expected = Path(os.environ["RECEIVER_MODELS_DIR"]) / "export_model_wrapper_qwen3_5.py"
assert expected.is_file(), f"缺少外部加载器: {expected}"
spec = importlib.util.find_spec("models.export_model_wrapper_qwen3_5")
assert spec is not None and Path(spec.origin).resolve() == expected.resolve()
print("receiver wrapper:", spec.origin)
PY
```

配置放在运行目录中，不修改模型源码。模型文件优先从本仓库查找，外部目录提供缺少的加载器
及辅助模块。若 wrapper 的依赖无法导入，补齐 receiver 软件包和对应 Python 依赖后再执行。

## 4. 选择精度并固定运行参数

FP16 不需要量化文件，在当前 Bash 会话设置：

```bash
QUANT_ARGS=()
export KV_CAPACITY=2048
export MAX_NEW_TOKENS=32
cat > "$AI_RUN_DIR/prompt.txt" <<'TEXT'
请用一句话解释为什么天空是蓝色的。
TEXT

NPU_ARGS=(
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR"
  --kv-cache-max-len "$KV_CAPACITY" --device npu:0
  --prompt-file "$AI_RUN_DIR/prompt.txt" --prompt-mode chat --enable-thinking
)
```

`KV_CAPACITY` 必须是 64 的倍数，覆盖 prompt 和输出 token。
`--block-size` 包含一个 anchor，取值 2..16；16 表示最多 15 个 proposal。
NPU prefill 每块最多 64 个真实 token，ordinary decode 每次一行。

要使用 W8A8，先完成下面的量化输入准备，然后设置 `QUANT_ARGS`。只运行 FP16 可直接进入第 5 步。

需要以下三个量化输入；仓库消费这些文件，不提供 W8A8 权重生成工具：

| 输入 | 内容 |
|---|---|
| `QUANT_LINEAR_DIR` | 含 `data*.safetensors` 的 Target W8A8 Linear 权重目录 |
| `QUANT_EMBED_WEIGHT` | Target embedding 的 INT8 raw binary |
| `QUANT_EMBED_SCALE` | 每词表行的 FP32 scale raw binary |

量化数据必须与 `TARGET_DIR` 的结构和权重匹配。设置路径并生成 YAML：

```bash
export QUANT_LINEAR_DIR=/absolute/path/qwen35-w8a8/linear
export QUANT_EMBED_WEIGHT=/absolute/path/qwen35-w8a8/embedding_weight.bin
export QUANT_EMBED_SCALE=/absolute/path/qwen35-w8a8/embedding_scale.bin
export QUANT_CONFIG="$AI_RUN_DIR/qwen35-w8a8.yaml"

"$MODEL_PYTHON" -B - <<'PY'
import os, yaml
from pathlib import Path
values = {
    "quanted_pth": os.environ["QUANT_LINEAR_DIR"],
    "embedding_weight_path": os.environ["QUANT_EMBED_WEIGHT"],
    "embedding_scale_path": os.environ["QUANT_EMBED_SCALE"],
}
Path(os.environ["QUANT_CONFIG"]).write_text(yaml.safe_dump(values, sort_keys=False))
PY
```

YAML 中就是这三个绝对路径字段。Target Linear 和输入 embedding 使用量化数据，
Draft embedding、Draft 主体和 LM head 使用 FP16。

```bash
QUANT_ARGS=(--config "$QUANT_CONFIG" --quant_mode enable)

"$MODEL_PYTHON" -B -m models.dflash_v1.preflight_target_quant \
  --target-dir "$TARGET_DIR" --prompt-ids 1,2,3,4 \
  --kv-cache-max-len "$KV_CAPACITY" --device npu:0 \
  --config "$QUANT_CONFIG" --compare-first-qlinear \
  --report "$AI_RUN_DIR/reports/quant-preflight.json"
```

量化预检验证装配、部分公式和状态调用，随后仍须完成整网 ordinary/DFlash 对照。
FP16 和 W8A8 的输出精度比较需要独立数据，不能由 W8A8 内部的 ordinary/DFlash 一致性推出。

## 5. 执行一次正确性验证

先用小 block 和两个输出 token 检查基本调用：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode validate --block-size 2 --max-new-tokens 2 \
  --report "$AI_RUN_DIR/reports/smoke.json"
```

随后执行完整 block：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode validate --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
  --report "$AI_RUN_DIR/reports/validate.json"
```

`validate` 分别创建 ordinary 和 DFlash 的独立状态，比较 token IDs、EOS 和停止原因。
读取报告检查：

```bash
"$MODEL_PYTHON" -B - <<'PY'
import json, os
from pathlib import Path
report = json.loads((Path(os.environ["AI_RUN_DIR"]) / "reports/validate.json").read_text())
assert report["execution_mode"] == "validate"
assert report["correctness_gate"]["status"] == "PASS"
assert report["strict_greedy_exact_match"] is True
assert report["operator_fallback_enabled"] is False
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]
assert report["dflash_execution_gate"]["target_verify_calls"] > 0
print(report["dflash"]["generated_text"])
PY
```

出现 `INCONCLUSIVE_NO_DRAFT_ROUND` 时，prompt 的 anchor 已结束生成，应换一条能进入
Draft/verify 的固定 prompt。完成短文本后，还应覆盖长 prompt、EOS、零/部分/全接受及重复请求。

## 6. 单独运行 DFlash 生成

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
  "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
  --execution-mode dflash --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
  --report "$AI_RUN_DIR/reports/dflash.json"
```

该模式只执行 DFlash；`ordinary=null`，`strict_greedy_exact_match=null`，
`correctness_gate.status=NOT_RUN_DFLASH_ONLY`。生成文本在报告的 `dflash.generated_text`。

## 7. 测量不带 msprof 的普通/DFlash 时延

使用独立进程分别测量普通和 DFlash，每种模式 3 次预热、10 次测量：

```bash
for GENERATION_MODE in ordinary dflash; do
  "$MODEL_PYTHON" -B -m models.dflash_v1.benchmark_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" \
    --mode "$GENERATION_MODE" --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
    --warmup 3 --repetitions 10 \
    --report "$AI_RUN_DIR/reports/benchmark-$GENERATION_MODE.json"
done
```

两次运行使用同一份权重、精度、prompt、thinking、KV 容量和 token 预算。
普通模式调用 Target 的 `begin_ordinary`/`advance_ordinary`；DFlash 调用 Draft/verify/commit。
测量包含完整 generation 及末尾设备同步，排除加载、tokenizer 和前置正确性检查。
读取原始 10 个值、median、p90，并确认实际输出 token 一致后再计算加速比。

## 8. 普通模式：分别只采一次 prefill 和 decode

设置 msprof 路径，运行下面的一条采集命令：

```bash
export MSPROF_BIN=/absolute/path/msprof
"$MSPROF_BIN" --version

"$REPO_ROOT/tools/run_msprof.sh" \
  --label python-ordinary-all --output-dir "$AI_RUN_DIR/msprof/ordinary-all" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend python --profile-mode ordinary \
  --profile-stage all --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}"
```

`all` 依次创建 prefill 和 decode 两个窗口，每个只采一次。
普通阶段仍检查 `--draft-dir` 的 checkpoint 配置，但不加载 Draft 模型，不执行特征投影、
Draft 生成、Target verify 或 commit。

| 阶段 | 窗口内 | 窗口外 |
|---|---|---|
| `prefill` | cache reset、全部 prompt 分块的 Target forward、末行 LM head | prompt 上传、anchor Top1；不收集 Draft 特征 |
| `decode` | 一次 `[1,1]` 的 Target forward、LM head 和状态更新 | prefill/KV 准备、首 token Top1、decode 输入上传、下一 token Top1 |

只测一段时执行：

```bash
PROFILE_STAGE=decode
"$REPO_ROOT/tools/run_msprof.sh" \
  --label "python-ordinary-$PROFILE_STAGE" --output-dir "$AI_RUN_DIR/msprof/ordinary-$PROFILE_STAGE" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend python --profile-mode ordinary \
  --profile-stage "$PROFILE_STAGE" --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}"
```

`PROFILE_STAGE` 填 `prefill` 或 `decode`。decode 需要 prompt 后留一行 KV；
anchor 已是 EOS 时不会产生 decode 窗口。

## 9. DFlash 模式：单阶段或 all 采集

```bash
PROFILE_STAGE=all
"$REPO_ROOT/tools/run_msprof.sh" \
  --label "python-dflash-$PROFILE_STAGE" --output-dir "$AI_RUN_DIR/msprof/dflash-$PROFILE_STAGE" \
  --python "$MODEL_PYTHON" --msprof-bin "$MSPROF_BIN" \
  --profile-backend python --profile-mode dflash \
  --profile-stage "$PROFILE_STAGE" --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$MODEL_PYTHON" -B -m models.dflash_v1.run_npu \
    "${NPU_ARGS[@]}" "${QUANT_ARGS[@]}" --block-size 16
```

`PROFILE_STAGE` 可以选下表任意一个阶段。`all` 在同一个应用进程中按表的顺序各采一次，
每个窗口使用相同 prompt 重建状态。每次命令使用新 label/输出路径。

| 阶段 | 窗口内 |
|---|---|
| `prefill` | begin 调用内的缓存清零、完整 Target prefill、特征收集、末行 LM head；不含 Draft 投影和 anchor Top1 |
| `feature-project` | prompt 特征的 Draft `fc + hidden_norm` |
| `draft` | 一次 Draft 生成，含首轮 Draft KV 构建和 Draft Top1 |
| `verify-input` | proposal token 回读、EOS 截断、verify block 创建与上传 |
| `verify` | 一次 Target verify，含第一遍 chunk GDR；不含 Target Top1 和第二遍 GDR |
| `target-top1` | Target 有限值检查、argmax 和 token 回读；LM head 已在 verify 中 |
| `accept-commit` | 接受判断、第二遍 GDR、conv state 选择、KV/cursor 提交；继续推测时含下一轮特征投影 |
| `draft-verify` | 一次 Draft、输入整理和 Target verify；不含 Target Top1 及状态提交 |
| `decode-round` | 一次 Draft → verify → Top1 → 接受/提交事务，含两遍 GDR |

verify 使用窗口外真实 prefill 和 Draft 产生的输入。commit 的第二遍 GDR 从本轮初始状态
执行 `effective_length=accepted+1`，零接受时也会执行，不能当作没有算子的纯主机阶段。
Target 的 CacheUpdate 位于 prefill、普通 decode 和 verify 窗口中；Draft 的 KV 操作属于
Draft 窗口。C++ OM 将两遍 GDR 和接受/提交放在一张 verify 图内，其 `verify` 计时范围更大，
不能直接与 Python `verify` 的时间比较。
`draft-verify` 和 `decode-round` 的内部不添加采集边界同步；单独阶段的时长不能直接相加
当作联合窗口时长。

## 10. 查看采集结果

普通 `all` 示例对应的输出：

```text
msprof/ordinary-all/
  profile/msprof/python-ordinary-all/prefill/
  profile/msprof/python-ordinary-all/decode/
  python-ordinary-all-stage-report.json
  python-ordinary-all-stage-summary.csv
  python-ordinary-all-operator-types.csv
  python-ordinary-all-operator-tasks.csv
  python-ordinary-all-hotspots.txt
  manifest/python-ordinary-all.json
  manifest/python-ordinary-all-control.json
  log/msprof-python-ordinary-all.log
```

每个阶段目录包含原始 `PROF_*` 数据及导出的 `op_summary*.csv`。
单阶段 raw 目录直接是 `profile/msprof/<label>/`。

| 文件/字段 | 用途 |
|---|---|
| `op_summary*.csv` | 具体算子的执行时间 |
| `<label>-stage-summary.csv` | 每阶段同步时长、`stage_scope`、算子行数、数据目录 |
| `<label>-operator-types.csv` | 每阶段各算子类型的次数、总时长、平均及最大时长，单位 ms |
| `<label>-operator-tasks.csv` | 每个任务的算子名、时长、stream/task ID、输入输出形状及 dtype |
| `<label>-hotspots.txt` | 各阶段慢算子类型和单任务排序 |
| `profiled_elapsed_ms` | 带 profiling 开销的同步窗口时间，排除控制器等待 |
| `captured_ordinary_calls` | decode 窗口中 `ordinary_prefill_token_calls=0`、`ordinary_decode_calls=1`；prefill 字段按最多 64 行的块计数 |
| `captured_gdr_layer_calls` | 检查 verify/commit 的 GDR 调用是否属于所选窗口 |
| `<label>-control.json` | 应用/采集 PID、start/stop/quit 回执、退出状态 |

算子汇总由 wrapper 自动生成，按阶段、原始 CSV、device/model、算子类型、任务类型、
OP State 及输入/输出 dtype 分组。`FLOAT;FLOAT` 与 `FLOAT16;FLOAT16` 矩阵乘分开统计；
筛选 `op_type=CacheUpdate` 或 `ChunkGatedDeltaRule` 可查看对应算子。
不同 stream 可能重叠，任务时长之和不等于窗口时长。多个导出 CSV 和联合窗口分别统计，
不会把 `draft`、`verify`、`draft-verify` 的重叠范围累加成请求时延。

`--profile-warmup 1` 表示先在窗口外预热一次，测量时再次从相同 prompt 准备状态。
阶段诊断忽略 `--max-new-tokens` 的生成预算，block 不按剩余输出预算截短。
`PASS_CAPTURE` 证明窗口流程完成，不能代替整段 token 正确性和第 7 步的时延测量。

采集使用 msprof 动态 PID CLI，无需 pyACL。阶段参数放在 wrapper 的 `--` 前；
不要直接给应用添加这些参数，也不要再套一层进程级 msprof。
支持 `PipeUtilization`、`Memory`、`MemoryUB`，每组指标单独运行并使用新输出路径。
不带 `--profile-stage` 的 wrapper 会采集整个应用进程，包含加载和预热。

## 11. 处理失败

| 现象 | 检查方法 |
|---|---|
| NPU 不可见或 import 失败 | 第 2 步的软件栈、设备权限、自定义算子注册 |
| 找不到 receiver wrapper | 第 3 步路径、模块搜索配置和 receiver 依赖 |
| 量化装配失败 | 三个 YAML 路径、checkpoint、QLinear 结构和 embedding dtype |
| anchor 为 EOS | 选择可以进入 decode/verify 的固定 prompt |
| KV 容量不足 | prompt＋输出预算、完整 verify block 与 64 行对齐要求 |
| token 不一致 | 保存报告和首个差异轮次，定位状态/接受判断后再测性能 |
| 停在 `msprof_start_sent` | 查看 controller 日志是否收到匹配应用 PID 的完整 start 成功回执 |
| 等待超时 | `--profile-timeout` 默认为 600 秒；根据加载、预热和阶段耗时设置 |
| 没有算子 CSV | 检查 start/stop/quit 回执、导出退出码及原始 PROF 数据 |

控制器要求启停回执、应用成功退出及每窗口非空算子数据；任一失败都返回 FAIL，保留日志。
CPU/fake ACL 测试只验证控制流程，真实 NPU 采集以设备导出结果为准。

## 12. CPU/CUDA 功能检查

CPU/CUDA 使用对应的模型 Python、PyTorch 和第 3 步准备的 checkpoint，不加载 NPU receiver。
在源码目录执行：

```bash
"$MODEL_PYTHON" -B -m models.dflash_v1.run_rollback \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --prompt-file "$AI_RUN_DIR/prompt.txt" --prompt-mode chat --enable-thinking \
  --execution-mode validate --block-size 16 --max-new-tokens "$MAX_NEW_TOKENS" \
  --device cuda:0 --dtype float16 --eos-token-id 248044 \
  --report "$AI_RUN_DIR/reports/cuda-validate.json"
```

CPU 使用 `--device cpu` 并选择其支持的 dtype。这里检查调度和模型功能，不生成 NPU 性能证据。
