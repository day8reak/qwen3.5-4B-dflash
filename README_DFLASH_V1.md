# Qwen3.5-4B DFlash rollback

`rollback` 分支已经把 CPU、CUDA 和 HIAI/NPU 的默认调度切到持久状态回退，不再用增长的完整
前缀验证 proposal。原文件 `models/modeling_qwen3_5_hiai_nd.py` 保持不变；NPU rollback 使用独立
文件 `models/modeling_qwen3_5_hiai_nd_dflash_rollback.py`。

先阅读：

- [调度、token 验证与 commit 规则](docs/DFLASH_V1_SCHEDULER.md)
- [验证流程与报告字段](docs/DFLASH_V1_VALIDATION.md)
- [Target rollback 与自定义算子表](docs/DFLASH_ROLLBACK_OPERATOR_ANALYSIS.md)
- [NPU 部署](docs/NPU_DEPLOYMENT.md)
- [CUDA 运行](docs/DFLASH_V1_GPU.md)

## 当前实现

```text
prompt persistent prefill → clean anchor
          ↓
feature history + [anchor, K masks] → Draft K proposals
          ↓
Target([anchor, proposals...])      # 一次 T=K+1 verify
          ↓
longest contiguous match a
          ↓
commit Target input state [anchor, accepted[:a]]
correction/bonus → next anchor
```

CPU/CUDA 使用 `DynamicCache`：verify 前保存 GDN conv/recurrent state，verify 后 crop attention KV、
恢复 GDN，并逐 token 重放最多 `a+1` 行。NPU 使用已接入的
`npu_gated_delta_rule_mtp` recurrent bank、Torch Tensor conv bank golden 和 paged-KV logical cursor。

普通基线同样是 persistent incremental greedy。最终 token ID、EOS 和 stop reason 必须零差异。
旧的 full-prefix sequential 实现仍在 `models/dflash_v1/dflash_reference_decode_v1.py`，只作为
correctness oracle/诊断工具。

## 代码入口

- `models/dflash_v1/run_rollback.py`：CPU/CUDA/NPU 统一 rollback CLI；
- `models/dflash_qwen_adapter_v1.py`：兼容 CLI，默认转到 rollback；
- `models/dflash_v1/run_npu.py`：NPU 简化入口；
- `models/dflash_v1/dflash_rollback_decode.py`：调度器；
- `models/dflash_v1/dflash_rollback_adapter.py`：framework 事务与 Draft adapter；
- `models/internal_dflash_bridge.py`：HIAI persistent state；
- `models/export_model_wrapper_qwen3_5_dflash_rollback.py`：复用部署 wrapper 的权重加载逻辑，只绑定
  独立 rollback modeling。

## CPU/CUDA 命令

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m models.dflash_v1.run_rollback \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --max-new-tokens 32 \
  --max-draft-tokens 16 \
  --eos-token-id 248044 \
  --device cuda:0 \
  --dtype float16 \
  --report /path/to/run/dflash-rollback-cuda.json
```

CPU 将 `--device` 改为 `cpu`，并按环境选择 dtype。CPU/CUDA 不传 HIAI factory/source/reset 参数。

## NPU 命令

部署工程必须已有原始 `models/export_model_wrapper_qwen3_5.py`，并注册用户完成的
`npu_gated_delta_rule_mtp`：

```bash
PYTHONDONTWRITEBYTECODE=1 python -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --max-new-tokens 32 \
  --max-draft-tokens 16 \
  --device npu:0 \
  --report /path/to/run/dflash-rollback-npu.json
```

`kv_cache_max_len` 必须与部署配置一致且能被 64 整除。当前 NPU bridge 为避免 64-row padding
污染 persistent GDN state，prompt 使用普通单 token 路线逐行 bootstrap；verify 才进入 T=K+1
MTP 路线。

## 自定义算子结论

| 算子/区域 | 当前 correctness 路线 | 后续结论 |
|---|---|---|
| GDR MTP | `npu_gated_delta_rule_mtp`，已接入 | 需 24 层、多轮真机验证 |
| causal-conv bank | 输入 device 上的 Torch Tensor 分解 | 建议生产版 `CausalConv1dMTP` |
| paged KV update | 逐 row 复用现有 `npu_cache_update_` | 先测；性能不足再做 `CacheUpdateMTP` |
| fused attention | 复用 `adn_fused_infer_attention` | T/历史长度/跨块能力失败才扩展 |
| LM-head Top-1/accept | 完整 logits 后在 host 做 Top-1 | 高价值性能候选，不是 bring-up 必需 |

完整输入输出 ABI 见[算子分析表](docs/DFLASH_ROLLBACK_OPERATOR_ANALYSIS.md)。

## 当前证据边界

已通过 reduced-shape CPU 测试：接受长度 `0..K`、CPU/GPU cache rollback、GDN state restore、
NPU bridge bank select/rebase、logical KV cursor、conv bank golden。当前 workspace 的
`ascend310p` target 是 simulation-only，因此这里不声明 310P 全模型通过、无 fallback 或任何
加速比。
