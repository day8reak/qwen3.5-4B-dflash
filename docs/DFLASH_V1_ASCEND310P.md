# Qwen3.5-4B DFlash V1 r6：Ascend 310P 全流程接入

发布 ID：`qwen3.5-4b-dflash-v1-ascend310p-iface-tf5.14.1-20260820-r6`。

## 结论

可以在**完全不修改现有自定义算子实现**的前提下验证 DFlash V1。r6 的正式 310P 主路线是：

```text
receiver-owned modeling_qwen3_5_hiai_nd.py
        │ r6 fail-closed source patcher
        │ decoder 层 1/5/9/13/17/21/25/29 层后、最终 norm 前
        ▼
[B,S,20480] dflash_features + 原 target 输出
        │
        ▼
官方结构的 6 层 Qwen3.5-4B-DFlash 草稿
        │ 最多 K 个 Top-1 proposal
        ▼
同一 receiver target 完整前缀验证
        │
        └─ 仅提交连续匹配前缀；否则提交 target 纠正 token
```

普通 target greedy 始终是权威结果；通过条件是最终 token ID、EOS 和停止原因零差异。V1
接受速度慢，以完整前缀重算先证明路线。它没有 KV/GDN 投机 state 的 commit/rollback，也
没有 DFlash2/V2 优化。

当前包只有 CPU/framework 和静态接口证据。真实权重、内部 receiver、310P FP16、无
fallback、逐 state/chunk trace 和性能都仍为 `PENDING`。

## 为什么 `use_cache=False` 还不够

portable facade 对 target 传 `use_cache=False`，但这只约束外部 forward ABI。已知 HIAI
receiver 内部仍可能执行：

- full-attention block-table KV 和原地 `CacheUpdate`；
- GDN 的原地 conv state、recurrent state 和 `ChunkGatedDeltaRule`；
- `new_kv_cache_pos`、`allQLen`、`token_count`、`export_flag` 等外部状态机字段。

因此每次完整前缀 forward 前，接收方必须选一种隔离方式：

1. `receiver_reset_hook`：重置全部 receiver-owned 可变状态，并强制当前调用回到 fresh
   prefill；hook 返回 `None`。
2. `fresh_instance`：返回一个从未用于前一调用的新 `nn.Module`，使用与普通 target 相同的
   权重和 source provenance。

正式 HIAI 只接受这两种方式；`proven_stateless`、`assumed` 只允许 CPU/非 HIAI 诊断。
prepare 和对应 target forward 在 facade 内串行执行，避免并发 reset 互相污染。

receiver 还必须声明 `fresh_prefill`、prefill `chunk_size=64`、decode `chunk_size=1`。这是
接口声明，实际路径必须由真机 trace 验证；报告会保留
`actual_chunk_mode_trace=PENDING_RECEIVER_TRACE`。不要猜测上述状态字段的类型或 reset 值。

## r6 source patch 的边界

正式 NPU 不使用 eager hook。`dflash_target_hook_bridge.py` 只供 CPU/eager debug；静态图或
内部编译器可能绕过 Python hook。

patcher 只做以下变化：

- 为 `Qwen3_5TextModel` 和 `Qwen3_5ForCausalLM` 增加默认关闭的
  `output_dflash_features`；
- 在八个锁定层的 decoder 输出处采集、最终 norm 前完成拼接；
- 通过 `dflash_hiai_feature_runtime.py` 给原始输出附加 `dflash_features`；
- 默认关闭时保留 receiver 原返回表达式、私有字段和 cache ABI；
- 锚点缺失、多义、局部 patch 或外来 marker 时 fail closed。

它不修改 attention、GDN、cache、图中的 custom-op 节点，也不修改任何算子源码、注册或
二进制。

## 第一步：准备完整接收包

设置变量：

```bash
export GOLDEN_ROOT=/path/to/extracted/qwen3_5
export PACKAGE_PYTHON_ROOT=/path/containing/transformer
export TARGET_QWEN_DIR="$PACKAGE_PYTHON_ROOT/transformer/model/qwen3_5"
export TARGET_DIR=/path/to/Qwen3.5-4B
export DRAFT_DIR=/path/to/Qwen3.5-4B-DFlash
export CUSTOM_OP_ROOT=/path/to/custom-op-root
export RUN_DIR=/path/to/this-run
export MODEL_PYTHON="${MODEL_PYTHON:-python}"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX
mkdir -p "$RUN_DIR"
```

`RUN_DIR` 必须是独立输出目录，不能位于 target 权重目录、draft 权重目录或 receiver Python
package 内；CPU 和 NPU CLI 都会拒绝让 `--report` 覆盖 prompt、权重或运行源码，NPU 还会
额外保护 preflight、HIAI source 和 loader。

按 `TARGET_OVERLAY_FULL.json` 从 `models/dflash_v1/` 逐文件原子复制 13 个运行文件；目标文件
仍按 basename 放在 receiver 的平级包目录。它们是：

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

保留 receiver 自有的 `configuration_qwen3_5.py`。正式路线另外有两个 receiver-owned 文件：
已 patch 的 `modeling_qwen3_5_hiai_nd.py` 和从模板定制的 `internal_target_loader.py`；分别
保存 SHA-256，但不要用仓库文件覆盖它们。

可以用下面的命令按 JSON 清单复制，避免漏掉同级依赖：

```bash
set -euo pipefail
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

## 第二步：patch receiver HIAI 源码

上一步已经把两个 sidecar 放到 HIAI source 同包目录。现在 dry-run 并审阅 diff：

```bash
set -euo pipefail
HIAI_SOURCE="$TARGET_QWEN_DIR/modeling_qwen3_5_hiai_nd.py"

PYTHONPATH="$GOLDEN_ROOT" "$MODEL_PYTHON" -B -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --dry-run --show-diff \
  > "$RUN_DIR/hiai-patch-dry-run.json"
```

确认后 apply 和 check：

```bash
set -euo pipefail
PYTHONPATH="$GOLDEN_ROOT" "$MODEL_PYTHON" -B -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --in-place \
  | tee "$RUN_DIR/hiai-patch-apply.json"

PYTHONPATH="$GOLDEN_ROOT" "$MODEL_PYTHON" -B -m models.dflash_v1.dflash_hiai_feature_patch \
  --source "$HIAI_SOURCE" --check \
  | tee "$RUN_DIR/hiai-patch-check.json"

sha256sum "$HIAI_SOURCE" | tee "$RUN_DIR/hiai-patched-source.sha256"
```

`--in-place` 默认创建 `.pre-dflash-v1` 备份，已有备份会拒绝覆盖。若接收方源码不满足唯一
语义锚点，应停止并人工适配，不能使用字符串替换猜改。

## 第三步：实现 receiver loader 和隔离 hook

```bash
cp "$GOLDEN_ROOT/models/dflash_v1/internal_target_loader_template.py" \
   "$TARGET_QWEN_DIR/internal_target_loader.py"
```

只替换 `create_internal_target(target_dir, device, dtype)`。它必须通过内部已有 Python/C++
binding 加载真实 target，不能用 `ctypes` 猜 ACLNN ABI。返回模块必须保留
`get_input_embeddings()`、`get_output_embeddings()` 和完整 logits ABI，并声明：

正式 HIAI factory 必须返回已 patch 文件导出的 package-local
`Qwen3_5ForCausalLM` 原始模型；CPU/HF golden 才使用
`Qwen3_5ForConditionalGeneration`。如果内部执行对象还有额外 wrapper，先提供可审计的真实
执行模型 identity 链，不要用另一个同名假类绕过门禁。

```text
dflash_feature_source = receiver_owned:modeling_qwen3_5_hiai_nd.py
dflash_feature_capture_point = decoder_post_layer_pre_final_norm
dflash_feature_contract_id = qwen3.5-4b-dflash-hiai-feature-source-v1
dflash_feature_patch_sha256 = <patched HIAI source 的实际 SHA-256>
dflash_full_prefix_isolation_mode = receiver_reset_hook | fresh_instance
prepare_dflash_full_prefix_call = <接收方实现的 callable>
dflash_full_prefix_execution_mode = fresh_prefill
dflash_prefill_chunk_size = 64
dflash_decode_chunk_size = 1
```

facade 会在**每次** target forward 前调用：

```text
prepare_dflash_full_prefix_call(
    input_ids=input_ids,
    sequence_length=S,
    output_dflash_features=bool,
    logits_to_keep=0_or_1,
    call_index=one_based_index,
)
```

普通验证要返回 `[1,S,248320]` 完整 logits；feature 调用可用 `logits_to_keep=1`，但必须返回
`[1,S,20480]` features。facade 会拒绝跨边界传入或返回
`past_key_values/cache_params/kv_cache/conv_state/recurrent_state/initial_state` 及已知外部状态字段。

定制后保存哈希并做导入检查：

```bash
sha256sum "$TARGET_QWEN_DIR/internal_target_loader.py" \
  | tee "$RUN_DIR/internal-target-loader.sha256"
PYTHONPATH="$PACKAGE_PYTHON_ROOT" "$MODEL_PYTHON" -B -c \
  'from transformer.model.qwen3_5.internal_target_loader import load_target; assert callable(load_target)'
```

现在运行完整无权重闭包检查。它会拒绝未修改的 `NotImplementedError` placeholder，核对
13 个交付文件、package-local HIAI source 和实际 loader，并生成正式运行必须绑定的报告：

检查器会拒绝接收包中的既有 `.pyc`/`__pycache__`。先只读检查：

```bash
find "$TARGET_QWEN_DIR" -maxdepth 2 \
  \( -type f -name '*.pyc' -o -type d -name '__pycache__' \) -print
```

若有输出，请按内部工程的缓存清理策略处理，或在一份干净的 overlay 副本中验证；不要对未知
目录执行宽泛递归删除。`-B` 和 `PYTHONDONTWRITEBYTECODE=1` 只阻止生成新缓存，不会清除旧缓存。

```bash
set -euo pipefail
PYTHONPATH="$PACKAGE_PYTHON_ROOT" "$MODEL_PYTHON" -B \
  "$GOLDEN_ROOT/tools/validate_target_overlay.py" \
  --scope v1-cli \
  --source-models-dir "$GOLDEN_ROOT/models/dflash_v1" \
  --package-dir "$TARGET_QWEN_DIR" \
  --package-name transformer.model.qwen3_5 \
  --hiai-source "$HIAI_SOURCE" \
  | tee "$RUN_DIR/overlay-preflight.json"
```

如果内部包名不同，替换 `--package-name` 和后续 `-m`/`--target-loader` 的模块名。

## 第四步：静态审计现有算子包

按服务器实际环境逐条 source，不要把多个 vendor 路径拼错：

```bash
set -euo pipefail
source "$CUSTOM_OP_ROOT/vendors/customize_dynamic/bin/set_env.bash"
source "$CUSTOM_OP_ROOT/vendors/customize_linearAttention/bin/set_env.bash"
source "$CUSTOM_OP_ROOT/vendors/customize_grouped/bin/set_env.bash"
source "$CUSTOM_OP_ROOT/vendors/customize_quantMatmul/bin/set_env.bash"
source "$CUSTOM_OP_ROOT/vendors/customize_scatter/bin/set_env.bash"

printf '%s\n' "$ASCEND_CUSTOM_OPP_PATH" | tr ':' '\n' | nl -ba \
  | tee "$RUN_DIR/custom-opp-order.txt"
PYTHONPATH="$GOLDEN_ROOT" "$MODEL_PYTHON" -B \
  "$GOLDEN_ROOT/tools/audit_internal_custom_ops.py" \
  --vendors-root "$CUSTOM_OP_ROOT/vendors" --nm --pretty \
  | tee "$RUN_DIR/internal-custom-ops-audit.json"
```

`CacheUpdate` 与 `ChunkGatedDeltaRule` 是 receiver V1 route 的静态依赖，且
`CacheUpdate` 有同名 provider/ABI 风险；头文件、ops-info 和运行 `.so` 必须来自 selected
provider。审计成功仍不证明 target 本次实际调用和 state reset，需真机 trace。

## 第五步：运行 310P `2 + 1` smoke

```bash
set -euo pipefail
export PYTHONPATH="$PACKAGE_PYTHON_ROOT"

"$MODEL_PYTHON" -B -m transformer.model.qwen3_5.dflash_qwen_adapter_v1 \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --target-loader transformer.model.qwen3_5.internal_target_loader:load_target \
  --hiai-source "$HIAI_SOURCE" \
  --overlay-preflight-report "$RUN_DIR/overlay-preflight.json" \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-v1-smoke.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-smoke.log"
```

正式 NPU 强制自定义 `--target-loader`、FP16、无 CPU fallback。不要传
`--allow-op-fallback`。省略 `--ops-backend` 会严格选择当前 package-local
`dflash_ascend310p_ops`；不要把 `libcust_opapi.so` 路径直接填成 Python backend。
正式 r6 NPU 也会拒绝任何显式外部 `--ops-backend`；外部实现需另做 provenance 和逐原语
oracle 验证，不能只依靠 target 最终纠正后的 token 一致性。
正式 NPU 还要求 EOS 参数恰好为单个 `248044`；省略、重复、额外或错误 EOS 都会在草稿
权重哈希前失败。

启动时会顺序哈希 1.27 GB 官方草稿权重；慢盘长时间停在
`draft_checkpoint_audit_begin` 可以是正常现象。只有 config、69 个 tensor、shape、大小和
完整 SHA-256 全部命中锁定 revision 后才继续加载 target。

`max_new_tokens=2` 仍可能因 prompt 后立即 EOS 而没有 draft round；正式 NPU 会把这种结果
判为不充分并 fail closed。此时换一个不会立即 EOS 的固定 prompt，不能跳过 round gate。

## 第六步：核对报告门禁

完整断言脚本见 [DFLASH_V1_GOLDEN.md](DFLASH_V1_GOLDEN.md)。至少同时满足：

- `strict_greedy_exact_match=true`、`feature_capture_zero_impact=true`；
- `bounded_full_prefix_repeatability=true`；这是异长 P-Q-P 有界行为证据，不是逐 state 证明；
- `target_integration.isolation.facade_contract_id` 为 r6 contract；
- isolation mode 为 `receiver_reset_hook` 或 `fresh_instance`，fresh prefill、声明 64/1 chunk；
- feature source/capture/contract ID/真实 patched source SHA-256 全部匹配；
- prepare/forward 全部成功且 `validation_call_reconciliation.status=PASS`；
- `dflash_execution_gate.status=PASS` 且确有 draft、feature、target verify 调用；
- ordinary/DFlash token、EOS、stop reason 完全相同；
- `operator_fallback_enabled=false`、运行时身份和设备指向本次 310P。

`actual_chunk_mode_trace=PENDING_RECEIVER_TRACE` 是诚实边界，不会因声明 64/1 自动变 PASS。

## 第七步：扩大用例和采集 trace

smoke 通过后至少补两类用例：

1. prompt/已提交前缀跨过 64 token 边界，验证 chunk 切换和 reset；
2. `--max-new-tokens 32 --max-draft-tokens 15`，覆盖最大 proposal 长度。

示例：

```bash
set -euo pipefail
"$MODEL_PYTHON" -B -m transformer.model.qwen3_5.dflash_qwen_adapter_v1 \
  --target-dir "$TARGET_DIR" \
  --draft-dir "$DRAFT_DIR" \
  --target-loader transformer.model.qwen3_5.internal_target_loader:load_target \
  --hiai-source "$HIAI_SOURCE" \
  --overlay-preflight-report "$RUN_DIR/overlay-preflight.json" \
  --prompt-json "$RUN_DIR/prompt-over-64.json" \
  --max-new-tokens 32 \
  --max-draft-tokens 15 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device npu:0 \
  --report "$RUN_DIR/dflash-v1-k15-over64.json" \
  2>&1 | tee "$RUN_DIR/dflash-v1-k15-over64.log"
```

内部 trace 应能关联每个 facade call index 与 fresh prefill、实际 64/1 chunk、call-local block
table、CacheUpdate、GDN conv/recurrent state 的初始化和销毁。仅看到 kernel 名称不等于状态
隔离正确。

## 回传证据和失败处理

请回传：

1. overlay、patcher、算子静态审计 JSON 和全部相关 SHA-256；
2. receiver loader/HIAI source 哈希，实际 package/module 路径；
3. smoke、跨 64、K=15 报告和完整日志；
4. NPU/CANN/PyTorch/`torch_npu`/内部框架/设备身份；
5. target/draft revision 与权重身份；
6. prepare/forward counters、每轮 proposal/接受/纠正 token；
7. state/chunk/operator profiler trace 与无 CPU fallback 证据；
8. 失败时第一个异常的完整 stack trace。

常见 fail-closed 错误：

- `replace create_internal_target()`：receiver factory 尚未实现；
- formal NPU target 缺 feature provenance：没有走 HIAI source patch，或未记录真实 source hash；
- isolation mode 被拒绝：正式 HIAI 使用了 `assumed/proven_stateless`；
- `all_calls_prepared=false` 或 reconciliation 失败：部分 forward 绕过 facade prepare；
- 返回 state 字段：receiver 把 KV/GDN/外部状态泄漏到 portable 边界；
- feature shape 错误：层号、捕获点、顺序或隐藏维不符合合同；
- DFlash round inconclusive：只执行了 bootstrap，未完成 draft/feature/verify；
- custom-op provider 冲突：编译头和动态链接 `.so` 不是同一 selected provider。

## 决策优先级与未证明事项

当前 r6 的正式 NPU 路线使用 receiver source patch。CPU framework、静态源码检查或文档本身
都不会自动证明内部 receiver、状态 reset 或 310P 设备执行已经通过。

在收到上述真机证据前，以下均不得声明通过：真实 4B/草稿权重、receiver 状态 reset、实际
chunk 路径、跨 64 边界、310P 严格 token 一致、全程无 fallback、OM/ATC、延迟、吞吐或
加速比。CPU fallback 只算模拟证据。
