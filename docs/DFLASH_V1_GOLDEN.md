# Qwen3.5-4B DFlash V1 r6 Golden（Transformers 5.14.1 / Ascend 310P 接口版）

发布 ID：`qwen3.5-4b-dflash-v1-ascend310p-iface-tf5.14.1-20260820-r6`。

## 本包验证的目标

V1 是 correctness-first 的完整前缀重算路线。目标模型先生成一个干净 anchor；官方结构的
六层 DFlash 草稿基于 `[clean_anchor, mask, ...]` 提议最多 15 个 token；同一个普通 target
验证整个候选块，只提交从块首开始连续匹配的最长前缀，并在不匹配时提交 target 纠正
token、全匹配时提交 bonus token。

硬性门槛是同一 target、同一 FP16 路径下：

- 最终 token ID 序列逐个完全一致；
- EOS 行为和停止原因完全一致；
- 开启 feature capture 前后 target logits 位级不变；
- 正式 NPU 至少实际执行一个 draft/feature/target-verify round。

完整前缀重算会比较慢，这正是本版本允许的取舍。V1 不实现 KV/GDN 投机 state 的提交、分支
或回退，也不用于证明性能。

## 三种 target 特征实现不要混用

| 路线 | 用途 | 正式 310P 是否接受 |
|---|---|---|
| patch 接收方 `modeling_qwen3_5_hiai_nd.py` + runtime sidecar | 内部 HIAI receiver | 是；r6 主路线 |
| `modeling_qwen3_5_dflash.py` | 官方 Transformers 5.14.1/CPU framework golden | 否；仅 framework 参考 |
| `dflash_target_hook_bridge.py` | eager CPU/debug | 否；静态图可能绕过 hook |

正式 HIAI patch 在 decoder 层 `1,5,9,13,17,21,25,29` 层后、最终 norm 前捕获，并按顺序
拼为 `[B,S,20480]`。默认 `output_dflash_features=False` 保持原 receiver 返回 ABI；开启后
sidecar 只附加 `dflash_features`，不替换私有 output 字段。

详细 patch/copy 命令见 [TARGET_OVERLAY_ZH.md](TARGET_OVERLAY_ZH.md)。

## 状态隔离是正式 NPU 的硬门禁

`use_cache=False` 只表示 portable forward 不要求返回/复用 cache。已知 HIAI target 内部仍
有 full-attention block-table KV 与原地 `CacheUpdate`，GDN 路径也有原地 conv/recurrent
state。因此，每个完整前缀 forward 前必须做到以下二选一：

- `receiver_reset_hook`：接收方清空全部 KV、conv、recurrent 和外部状态，并把状态机切回
  fresh prefill；
- `fresh_instance`：每次返回真正的新 target 实例，且使用与普通 target 相同的权重和
  source provenance。

正式 HIAI 不接受 `proven_stateless` 或 `assumed`。receiver 必须声明
`fresh_prefill`、prefill `chunk_size=64`、decode `chunk_size=1`。这些是接收方声明，不是
实际 trace；报告会故意保留
`actual_chunk_mode_trace=PENDING_RECEIVER_TRACE`，直到内部 profiler/运行时证据补齐。

r6 在 decode 前运行异长 `P -> Q -> P`：比较第一次和第三次 P 的 logits；feature 路径还
比较 `dflash_features`。该门禁只能发现会传播到可见输出的部分状态泄漏，不能逐项证明
`new_kv_cache_pos`、`allQLen`、`token_count`、`export_flag`、KV、conv 或 recurrent state
都已正确 reset。

## 现有自定义算子保持不变

r6 不修改任何现有自定义算子的源码、ABI、注册或二进制。DFlash controller/草稿不会直接
调用 receiver 的 ACLNN target 接口。对正式 HIAI 路线：

- receiver 接口声明 `CacheUpdate` 应在本次 full-prefix call 内完成 block-table KV 更新；
- receiver 接口声明 `ChunkGatedDeltaRule` 应使用本次 call 隔离的 GDN 状态；
- `DynamicQuant`、`QuantBatchMatmulV4444`、`GroupedMatmul` 是否由普通 target 调用仍以 trace
  为准，不能直接替代草稿侧 FP16 dense primitive。

静态接口审计命令：

```bash
PYTHONPATH=. python tools/audit_internal_custom_ops.py \
  --vendors-root /path/to/custom-op-root/vendors \
  --nm \
  --pretty > /path/to/run/internal-custom-ops-audit.json
```

该审计只验证 selected provider 的 ops-info、头文件、`.so` 导出和同名冲突，不执行算子，
也不证明本轮模型实际走到对应 kernel。

## 包和命名空间

`transformers`（复数）是 Hugging Face 依赖，本包锁定 `5.14.1`；
`transformer.model.qwen3_5`（单数）只是内部接收包名示例。目标文件使用相对 import，实际
部署时必须使用服务器真实完整包名。

完整 singular CLI overlay 是 13 个交付文件，加接收方已有
`configuration_qwen3_5.py`。正式路线另外需要接收方自有、已 patch 的
`modeling_qwen3_5_hiai_nd.py` 和由模板定制的 `internal_target_loader.py`；后两者分别留存
SHA-256，但不由仓库文件覆盖。

## CPU/framework 快速运行

CPU 只验证 framework 流程，不替代 CUDA/NPU 设备验证。进入仓库根目录后运行小参数流程：

```bash
PYTHONPATH=. python -m models.dflash_v1.dflash_qwen_adapter_v1 \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --eos-token-id 248044 \
  --dtype float16 \
  --device cpu \
  --report /path/to/run/dflash-v1-cpu.json
```

启动时会先对官方草稿 checkpoint 做 fail-closed 审计：锁定 config、69 个 BF16 tensor、
shape、文件大小及完整 `model.safetensors` SHA-256。1.27 GB 顺序哈希在慢盘上耗时较长是
正常现象，stderr 应先后显示 `draft_checkpoint_audit_begin/end`。CPU 报告即使 token 一致，
也不能升级为 310P 证据。

## 310P 最小真实流程

完成 13 文件 overlay、HIAI patcher `--check`、receiver loader 定制和五个 vendor 静态审计
后，运行 `2 + 1` smoke：

```bash
set -euo pipefail
: "${PACKAGE_PYTHON_ROOT:?}" "${TARGET_DIR:?}" "${DRAFT_DIR:?}" \
  "${RUN_DIR:?}" "${HIAI_SOURCE:?}"
MODEL_PYTHON="${MODEL_PYTHON:-python}"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$PACKAGE_PYTHON_ROOT"
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPYCACHEPREFIX

# RUN_DIR 必须独立于 receiver package、target 和 draft 目录；正式 CLI 会拒绝覆盖输入。

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

不要传 `--allow-op-fallback`。省略 `--ops-backend` 时，NPU 路线使用当前 package-local
`dflash_ascend310p_ops` 的六原语分解。正式 r6 NPU 路线禁止显式 `--ops-backend`；外部 backend
必须先有独立 provenance 和逐原语 oracle 门禁，当前仅能用于 CPU 开发实验。

## r6 报告门禁

对真实 NPU 报告运行：

```bash
python - "$RUN_DIR/dflash-v1-smoke.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    report = json.load(stream)

assert report["strict_greedy_exact_match"] is True
assert report["feature_capture_zero_impact"] is True
assert report["bounded_full_prefix_repeatability"] is True
assert report["operator_fallback_enabled"] is False
assert report["dtype"] == "torch.float16"
assert report["device"].startswith("npu")
assert report["runtime_identity"]["device_type"] == "npu"
assert report["draft_checkpoint"]["status"] == "PASS"
assert report["request"]["eos_token_ids"] == [248044]
assert report["request"]["formal_locked_eos_token_id"] == 248044
assert report["overlay_preflight"]["status"] == "PASS_CURRENT_MATCH"
assert report["overlay_preflight"]["package"] == "transformer.model.qwen3_5"
assert report["overlay_preflight"]["transformers_version"] == "5.14.1"
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
assert report["ordinary"]["reached_eos"] == report["dflash"]["reached_eos"]
assert report["ordinary"]["stop_reason"] == report["dflash"]["stop_reason"]

integration = report["target_integration"]
isolation = integration["isolation"]
assert isolation["facade_contract_id"] == "qwen3.5-4b-dflash-v1-full-prefix-isolation-r6"
assert isolation["formal_npu"] is True
assert isolation["mode"] in {"receiver_reset_hook", "fresh_instance"}
assert isolation["prepare_forward_serialized"] is True
assert isolation["all_calls_prepared"] is True
assert isolation["prepare_failures"] == 0
assert isolation["full_prefix_execution_mode"] == "fresh_prefill"
assert isolation["declared_chunk_modes"] == {
    "prefill_chunk_size": 64,
    "decode_chunk_size": 1,
}
assert isolation["actual_chunk_mode_trace"] == "PENDING_RECEIVER_TRACE"

feature = integration["feature_capture"]
assert feature["status"] == "PASS_DECLARED_SOURCE_PATCH"
assert feature["source"] == "receiver_owned:modeling_qwen3_5_hiai_nd.py"
assert feature["capture_point"] == "decoder_post_layer_pre_final_norm"
assert feature["contract_id"] == "qwen3.5-4b-dflash-hiai-feature-source-v1"
actual_source = feature["actual_package_source"]
assert actual_source["status"] == "PASS_ACTUAL_PACKAGE_SOURCE"
assert actual_source["patch_contract_id"] == feature["contract_id"]
assert actual_source["source_sha256"] == feature["patched_source_sha256"]
assert report["overlay_preflight"]["hiai_source_sha256"] == feature[
    "patched_source_sha256"
]

reconciliation = integration["validation_call_reconciliation"]
assert reconciliation["status"] == "PASS"
assert reconciliation["matches"] is True

round_gate = report["dflash_execution_gate"]
assert round_gate["status"] == "PASS"
assert round_gate["draft_round_executed"] is True
assert round_gate["draft_calls"] > 0
assert round_gate["target_feature_calls"] > 0
assert round_gate["target_verify_calls"] > 0
print("DFLASH_V1_R6_NPU_FRAMEWORK_REPORT_GATE_PASS")
PY
```

这里的 source/capture/contract/hash 是 receiver 声明并由 facade 校验的 provenance；调用计数
reconciliation 证明 scheduler 观察到的 target 调用均经过 prepare；异长 P-Q-P 证明有界输出
可重复。三者合起来仍不能替代逐 state trace。`actual_chunk_mode_trace` 保留 `PENDING` 是
有意边界；上述标记只代表通用 NPU/framework 报告门禁，具体是否为 310P 还要核对
`runtime_identity.device_name`、`npu-smi` 和内部设备清单。该 `PENDING` 是预期行为，不应
人为改成 PASS。

smoke 通过后再用更长 prompt 覆盖 64 token 分块边界，并扩大到
`--max-new-tokens 32 --max-draft-tokens 15`。跨 64 边界、实际 CacheUpdate/GDN 输入输出及
state reset 必须用 receiver profiler/trace 证明。

## 回传证据

请回传：

1. overlay preflight、patcher dry-run/apply/check JSON，以及 patched HIAI source 和 loader 哈希；
2. smoke/长上下文报告和完整日志；
3. `npu-smi info`、Python、PyTorch、`torch_npu`、CANN、内部框架及设备身份；
4. target/draft checkpoint revision、config 和权重哈希；
5. ordinary/DFlash token、EOS、stop reason、每轮 proposal/接受/纠正信息；
6. facade prepare/forward 计数、异长 P-Q-P 结果、跨 64 边界 state/chunk trace；
7. 五个 vendor 的顺序、ops-info/头文件/实际加载 `.so` 哈希；
8. profiler 的 target/draft NPU 驻留与无 CPU fallback 证据；
9. 失败时第一个异常的完整 stack trace。

## 决策覆盖和 PENDING 边界

当前 r6 的正式 NPU 路线使用 receiver HIAI source patch；这不会因为文档或 CPU framework
运行成功而自动证明内部 receiver、状态隔离或 310P 设备路径已经通过。

真实 4B target/草稿权重、内部 receiver、状态和 chunk trace、310P FP16 严格一致、全程无
fallback、内存、延迟、吞吐和加速比目前都保持 `PENDING`。V1 不支持 sampling、batch>1、
OM/ATC 已验证声明，也不包含 V2/DFlash2 的 state rollback 优化。
