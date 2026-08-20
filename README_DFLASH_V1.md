# DFlash V1 — 实现与设备接入说明

本实现依赖 `transformers==5.14.1`。CPU/CUDA 使用完整的
`models/dflash_v1/modeling_qwen3_5_dflash.py`；NPU 使用
本仓库直接提供的 `models/modeling_qwen3_5_hiai_nd.py`，DFlash 代码整体放在它旁边的
`models/dflash_v1/`。

先阅读：

- [Ascend NPU 部署与运行](docs/NPU_DEPLOYMENT.md)
- [CPU/Golden 指南](docs/DFLASH_V1_GOLDEN.md)
- [CUDA GPU 指南](docs/DFLASH_V1_GPU.md)
- [Ascend 接口与验证边界](docs/DFLASH_V1_ASCEND310P.md)

## 共同算法

- 普通 target greedy 始终是权威结果；DFlash 的 token ID、EOS 和停止原因必须完全一致。
- 每个 target 调用都重算完整已提交前缀。V1 不提交、分支或回退投机 KV/GDN state。
- target 先生成 anchor，draft 最多产生 15 个 proposal，target 验证最长连续匹配前缀并给出
  correction/bonus。
- CPU、CUDA 和 NPU 共用 `dflash_reference_decode_v1.py` 与
  `Qwen35DFlashFullPrefixAdapter`。

## Target 路由

```text
DFlash V1 scheduler
├── CPU/CUDA target：models.dflash_v1.modeling_qwen3_5_dflash
└── NPU target：models.modeling_qwen3_5_hiai_nd
```

源码文件可以不同，模型语义不能不同。NPU 必须使用相同权重、tokenizer 和文本网络，并在
decoder 层 `1,5,9,13,17,21,25,29` 的层后、最终 norm 前输出
`dflash_features: [B,S,20480]`。

`models/modeling_qwen3_5_hiai_nd.py` 已直接集成 feature route；运行时不再 patch。
该 route 不替换 attention、GDN、CacheUpdate 或其他自定义算子，只增加
`output_dflash_features=False` 的显式开关。默认仍返回 logits Tensor；仅开启时返回
`(logits, dflash_features)`。

## NPU 状态边界

`use_cache=False` 不能自动清除 HIAI target 的 block-table KV、GDN conv/recurrent state。
本仓库的 `models/internal_dflash_bridge.py` 按模型配置构造调用级状态：

1. 每次调用都按 `layer_types` 新建 32 层 hybrid state；
2. linear 层新建 `(conv_state, recurrent_state)`，full-attention 层新建 block-table `(K,V)`；
3. 用完整前缀执行一次 fresh prefill；
4. 只把 logits 和可选 features 返回 DFlash，不跨调用返回 state；
5. 保证 `P → Q → P` 中两次 `P` 的 logits/features 一致。

shape/dtype 来自模型配置和固定 ABI。`run_npu` 默认使用已经实现好的 bridge；
只需传部署配置中的 `kv_cache_max_len`。

## Draft backend

```text
CPU/CUDA：TorchDFlashOps
NPU：dflash_ascend310p_ops
```

普通受支持 PyTorch 算子由 tensor device 分派；target 自定义算子仍由 HIAI modeling
显式调用。这里没有全局 monkey patch。

## NPU 最小命令

```bash
export PYTHONPATH=/path/to/qwen35-runtime

python -B -m models.dflash_v1.run_npu \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --kv-cache-max-len 4096 \
  --prompt-ids 151644,872,198 \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report /path/to/run/dflash-v1-npu-smoke.json
```

把 `4096` 换成部署配置的实际值。默认 bridge 固定导入
`models.export_model_wrapper_qwen3_5.Qwen3_5ForCausalLMWrapper`。

## 验证边界

CPU/CUDA 可以验证 framework 调度与 draft 数学，但不能替代 NPU target、状态隔离、接受率和
性能验证。NPU 上必须分别证明：

- 普通 NPU greedy 与 NPU DFlash token/EOS/stop reason 零差异；
- feature 开关不改变 target logits；
- 至少实际执行一次 draft/feature/verify round；
- 没有 CPU fallback；
- 小参数通过后再测 `max_draft_tokens=15` 的接受率和性能。
