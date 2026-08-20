# Qwen3.5 DFlash V1 r6 目标覆盖说明

发布 ID：`qwen3.5-4b-dflash-v1-ascend310p-iface-tf5.14.1-20260820-r6`。

## 先选清楚路线

正式 Ascend 310P 路线不再以官方 `modeling_qwen3_5_dflash.py` 或 eager forward hook 作为
target 实现。它要求：

1. 用 r6 patcher 修改接收方自有的 `modeling_qwen3_5_hiai_nd.py`；
2. 将 `dflash_target_features.py`、`dflash_hiai_feature_runtime.py` 放到该文件同一包目录；
3. 从 `internal_target_loader_template.py` 定制接收方自己的 `internal_target_loader.py`；
4. 保留接收方已有的 `configuration_qwen3_5.py`。

`modeling_qwen3_5_dflash.py` 是官方 Transformers/CPU golden 路线；
`dflash_target_hook_bridge.py` 只供 CPU 或 eager debug 使用。二者都不能替代正式 HIAI 源码
patch 的真机证据。

## 完整交付闭包

若要在内部 singular 包中直接运行 V1 CLI，应严格复制 `TARGET_OVERLAY_FULL.json` 中定义的
13 个运行文件：

```text
dflash_ascend310p_ops.py
dflash_config.py
dflash_hiai_feature_patch.py
dflash_hiai_feature_runtime.py
dflash_ops.py
dflash_qwen_adapter_v1.py
dflash_reference_decode_v1.py
dflash_target_features.py
dflash_target_hook_bridge.py
dflash_weights.py
internal_target_loader_template.py
modeling_dflash.py
modeling_qwen3_5_dflash.py
```

以下三个文件由接收方维护，不在 13 文件复制集合中：

```text
configuration_qwen3_5.py             # 接收方已有依赖
modeling_qwen3_5_hiai_nd.py          # 接收方自有，patch 后单独记录哈希
internal_target_loader.py            # 接收方从模板定制，单独记录哈希
```

不要覆盖接收方原有 `configuration_qwen3_5.py`。13 文件的拷贝必须是原子操作；不要让 Python
进程在半拷贝状态中 import。

按清单一次性复制：

```bash
set -euo pipefail
: "${GOLDEN_ROOT:?请设置仓库根目录}" \
  "${TARGET_QWEN_DIR:?请设置接收方 qwen3_5 包目录}"
python - "$GOLDEN_ROOT" "$TARGET_QWEN_DIR" <<'PY'
import json
import os
import shutil
import sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
target = Path(sys.argv[2]).resolve()
contract = json.loads((root / "TARGET_OVERLAY_FULL.json").read_text(encoding="utf-8"))
target.mkdir(parents=True, exist_ok=True)

for relative in contract["required_copy_files"]:
    source = root / relative
    destination = target / source.name
    temporary = destination.with_name(destination.name + ".dflash-tmp")
    if not source.is_file() or source.is_symlink():
        raise SystemExit(f"invalid source file: {source}")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)
    print(f"installed: {destination}")
PY
```

## HIAI patcher：dry-run、apply、check

上一步已经复制 runtime sidecar。现在审阅 patch diff：

```bash
set -euo pipefail
: "${GOLDEN_ROOT:?请设置解压包根目录}" \
  "${TARGET_QWEN_DIR:?请设置接收方 qwen3_5 包目录}"
HIAI_SOURCE="$TARGET_QWEN_DIR/modeling_qwen3_5_hiai_nd.py"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --dry-run --show-diff
```

dry-run JSON 必须报告 `status=verified`，并报告唯一 decoder loop、最终 norm、causal model 调用和输出
return 锚点。确认 diff 后再 apply；`--in-place` 会先保存
`modeling_qwen3_5_hiai_nd.py.pre-dflash-v1`：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --in-place
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --check
sha256sum "$HIAI_SOURCE"
```

若不希望直接改接收目录，可把第二条改成：

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$GOLDEN_ROOT" python -B -m models.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" \
  --output /path/to/review/modeling_qwen3_5_hiai_nd.py
```

patcher 遇到歧义锚点、外来/局部 marker 或已有输出文件会 fail closed，不会猜测改写。重复
`--check` 是幂等的。它只增加 feature flag、八层 collector 和输出 sidecar，不修改
attention、GDN、cache 或任何自定义算子。

## 接收方 loader 不是普通复制文件

复制模板后只实现其中 `create_internal_target()`：

```bash
cp "$GOLDEN_ROOT/models/internal_target_loader_template.py" \
   "$TARGET_QWEN_DIR/internal_target_loader.py"
```

factory 返回的 target 必须使用与普通 greedy 相同的真实权重，并声明：

```text
dflash_feature_source = receiver_owned:modeling_qwen3_5_hiai_nd.py
dflash_feature_capture_point = decoder_post_layer_pre_final_norm
dflash_feature_contract_id = qwen3.5-4b-dflash-hiai-feature-source-v1
dflash_feature_patch_sha256 = <本次 patched HIAI 文件的真实 64 位十六进制 SHA-256>
dflash_full_prefix_execution_mode = fresh_prefill
dflash_prefill_chunk_size = 64
dflash_decode_chunk_size = 1
```

状态隔离只能选：

- `receiver_reset_hook`：每次 forward 前重置全部 receiver-owned KV、conv、recurrent 和外部
  状态，并强制本次调用走 fresh prefill；hook 返回 `None`。
- `fresh_instance`：每次调用返回一个新的、权重与 provenance 相同的 `nn.Module`，不能重复
  返回同一对象。

两种模式都要暴露 `prepare_dflash_full_prefix_call(...)`。`proven_stateless` 和 `assumed` 仅
允许 CPU/非 HIAI 诊断，正式 NPU 会拒绝。`use_cache=False` 本身不满足隔离合同；
`new_kv_cache_pos`、`allQLen`、`token_count`、`export_flag` 等具体类型和 reset 值只能由接收方
实现确定，本包不会猜。

## 无权重闭包自检

完整复制、patch HIAI 源码并实现 receiver factory 后执行；原样 placeholder 会失败：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B "$GOLDEN_ROOT/tools/validate_target_overlay.py" \
  --scope v1-cli \
  --source-models-dir "$GOLDEN_ROOT/models" \
  --package-dir "$TARGET_QWEN_DIR" \
  --package-name transformer.model.qwen3_5 \
  --hiai-source "$HIAI_SOURCE" \
  | tee /path/to/run/overlay-preflight.json
```

检查器会验证 13 个交付文件逐字节一致、相对 import 闭包、真实
`__package__`/`__file__`、`transformers==5.14.1`，并在 singular 命名空间运行 CLI
`--help`；还会要求 `--hiai-source` 恰好位于该 package 内，调用交付 patcher 复验源码语义
和 SHA-256。它不会加载权重，也不能证明 receiver-owned loader 的业务实现或真机 state
trace；这些仍须通过正式报告门禁和设备证据。
正式 NPU CLI 还必须用 `--overlay-preflight-report` 绑定这份 JSON；运行前后都会重新哈希
13 个运行文件、HIAI source 和 loader，发现 preflight 后漂移即停止。

检查器是防误集成和漂移的完整性门禁，不是用来隔离恶意 receiver Python 的安全沙箱；
receiver factory 是接收方可信代码，但其真实权重和真机执行仍需独立证据。

若实际包名不是 `transformer.model.qwen3_5`，请同时修改 `--package-name`；
`--package-dir` 必须与包名后缀严格一致。

## 旧包兼容提示

V1 r3/V2 r2 若报
`No module named 'transformer.model.qwen3_5.dflash_config'`，把相应旧包的
`dflash_config.py` 补到同级目录只能临时修复旧包 import；它不会升级为 r6，也不能替代
上述 13 文件闭包和 HIAI source patch。

## 仍为 PENDING

源码 patch/check 和 overlay import 成功只证明静态接入准备完成。真实权重、receiver 状态
隔离、64/1 chunk 实际路径、跨 64 token 边界、310P 无 fallback、token-ID 严格一致和性能
结果，在拿到真机 trace/report 前全部保持 `PENDING`。

本仓库只提供接入代码和工具；真实 receiver source、设备状态隔离及性能结果需要在内部服务器
按上述命令验证。
