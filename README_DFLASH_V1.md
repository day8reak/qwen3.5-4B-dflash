# DFlash V1 — 实现与设备接入说明

本实现依赖 `transformers==5.14.1`。CPU/CUDA 使用完整的
`models/dflash_v1/modeling_qwen3_5_dflash.py`；NPU 使用
本仓库直接提供的 `models/modeling_qwen3_5_hiai_nd.py`，DFlash 代码整体放在它旁边的
`models/dflash_v1/`。

先阅读：

- [整体架构与完整数据流](docs/DFLASH_V1_ARCHITECTURE.md)
- [Target 与 Feature](docs/DFLASH_V1_TARGET_AND_FEATURE.md)
- [Draft 模型](docs/DFLASH_V1_DRAFT.md)
- [Scheduler 与 token 验证](docs/DFLASH_V1_SCHEDULER.md)
- [验证流程与报告解读](docs/DFLASH_V1_VALIDATION.md)
- [从 V1 到完整 DFlash 与真正提速](docs/DFLASH_FULL_AND_PERFORMANCE_ROADMAP.md)
- [Ascend NPU 部署与运行](docs/NPU_DEPLOYMENT.md)
- [CPU/Golden 指南](docs/DFLASH_V1_GOLDEN.md)
- [CUDA GPU 指南](docs/DFLASH_V1_GPU.md)
- [Ascend 接口与验证边界](docs/DFLASH_V1_ASCEND310P.md)
- [Target 状态回退版与自定义算子分析](docs/DFLASH_ROLLBACK_OPERATOR_ANALYSIS.md)

`rollback` 分支额外保留
`models/modeling_qwen3_5_hiai_nd_dflash_rollback.py`。原
`models/modeling_qwen3_5_hiai_nd.py` 不变；只有调用方显式传入
`accepted_tokens` 时，新文件才进入 GDR/conv state-bank 与跨 block KV 写入路径。
现有 V1 scheduler/bridge 仍是完整前缀重算，不会自动启用该路径，接入要求和剩余算子边界
见上述分析文档。

## 共同算法

- 普通 target greedy 始终是权威结果；DFlash 的 token ID、EOS 和停止原因必须完全一致。
- 每个 target 调用都重算完整已提交前缀。V1 不提交、分支或回退投机 KV/GDN state。
- target 先生成 anchor，draft 最多产生 16 个 proposal，target 验证最长连续匹配前缀并给出
  correction/bonus。
- `v1-r1` 默认逐个 proposal 使用独立完整前缀校验；一次调用验证整块的 vectorized 路径只用于
  诊断，因为不同输入长度可能选择不同 kernel，不能假定更长输入里较早 logit 行逐 bit 不变。
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
3. `S=1` 走单 token 路线；`S>1` 在 bridge 内右补齐到 64 的倍数后执行一次 fresh prefill，
   只返回真实 token 对应的 logits/features；逻辑 `allQLen` 仍使用未补齐的真实长度；
4. 按实际 `kv_cache_max_len` 重建每个 full-attention 层的 block table；
5. 返回前同步目标设备，确认异步算子已经结束后才释放本次临时 state；只把 logits 和可选
   features 返回 DFlash，不跨调用返回 state；
6. 先做 `P → P` 重复性对照，再做异长 `P → Q → P`；要求 Top-1 相同且数值在 dtype 对应
   容差内，并把误差写入报告。

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
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 2 \
  --max-draft-tokens 1 \
  --device npu:0 \
  --report /path/to/run/dflash-v1-npu-smoke.json
```

把 `4096` 换成部署配置的实际值。默认 bridge 固定导入
`models.export_model_wrapper_qwen3_5.Qwen3_5ForCausalLMWrapper`。

若要让每次运行读取同一段固定文本，把 `--prompt` 换成
`--prompt-file /path/to/prompt.txt`；文件按 UTF-8 读取。默认 `chat` 模式会套用本地主模型
tokenizer 的 chat template，并默认启用 thinking；`raw` 只适用于已经自行构造好模板的文本。
非 thinking 测试显式传 `--no-enable-thinking`。运行结束会打印 ordinary Target 与 DFlash
两份解码文本。

## 验证边界

CPU/CUDA 可以验证 framework 调度与 draft 数学，但不能替代 NPU target、状态隔离、接受率和
性能验证。NPU 上必须分别证明：

- 普通 NPU greedy 与 NPU DFlash token/EOS/stop reason 零差异；
- feature 开关不改变 target logits；
- 至少实际执行一次 draft/feature/verify round；
- 没有 CPU fallback；
- 小参数通过后再测 `max_draft_tokens=16` 的接受率和性能。
