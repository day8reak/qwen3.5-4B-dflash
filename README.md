# Qwen3.5-4B MTP / DFlash / Ascend 310P 准确率优先 PoC

这个仓库现在包含三条边界明确的路径：

1. 普通 Qwen3.5-4B greedy PyTorch 基线；
2. 官方单层 MTP/NEXTN drafter + 主模型严格校验。
3. 官方 Z-Lab Qwen3.5-4B-DFlash 的无 cache draft-core golden。

本项目不依赖拿不到的内部框架源码。内部框架只需实现
`MainBackend.evaluate` 和 `DraftBackend.propose`，或者替换 MTP 的 5 个细粒度
算子。默认实现全部使用原生 PyTorch，可在 CPU 上模拟。

DFlash 路径已复现官方 feature projection、六层 drafter、逐层 target-context
K/V 注入、RoPE/GQA、前五层 sliding-causal attention 和末层 bidirectional
attention，并把六类 primitive 暴露为可替换接口。它目前不是完整端到端 DFlash：
尚未包含 target verifier、接受/拒绝、draft/target cache 与 Qwen3.5 Gated
DeltaNet 状态提交/回滚、循环调度。

## 已锁定的官方结构

- `hidden_size=2560`，`intermediate_size=9216`；
- 16 个 Q head、4 个 KV head、`head_dim=256`；
- 1 层 full-attention MTP block；
- embedding 与 LM head 共享，MTP 无专用 embedding；
- `RMSNorm(embedding)` 与 `RMSNorm(main hidden)` 拼接后经
  `mtp.fc: 5120 -> 2560`；
- Q 投影含输出 gate，shape 为 `8192 x 2560`；
- checkpoint 中恰好 15 个 `mtp.*` tensor，均为 BF16。

来源是 [Qwen/Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) 的
`config.json`/SafeTensors，以及
[vLLM Qwen3.5 MTP 实现](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_5_mtp.py)。

## 快速验证

不安装到共享环境，直接设置源码路径：

```bash
export PYTHONPATH="$PWD/model"
export QWEN35_MODEL_DIR=/path/to/Qwen3.5-4B

python -m qwen35_mtp audit --model-dir "$QWEN35_MODEL_DIR"
python -m unittest discover -s tests -v
```

审计 DFlash checkpoint 或运行固定 draft-core case：

```bash
python -m qwen35_dflash audit \
  --draft-dir /path/to/z-lab-Qwen3.5-4B-DFlash

python -m qwen35_dflash run-case \
  --draft-dir /path/to/z-lab-Qwen3.5-4B-DFlash \
  --case /path/to/dflash-case.npz \
  --dtype float16 --output /tmp/dflash-hidden.npy
```

DFlash 的输入、算子 ABI、官方 revision 和精确边界见
[DFlash golden 说明](docs/DFLASH_GOLDEN.md)。

现有 torch_npu target 的第一阶段只接选择性 hidden feature 输出，不改 GDN、attention、
cache 或 `hiai_nd`。目录映射和 `modeling_qwen3_5.py` 的三个稳定插入点见
[DFlash target 接入说明](docs/DFLASH_TARGET_INTEGRATION.md)。

运行普通基线：

```bash
python -m qwen35_mtp ordinary \
  --model-dir "$QWEN35_MODEL_DIR" \
  --prompt '介绍一下杭州。' --chat --max-new-tokens 8
```

运行 MTP，并与普通路径逐 token 比较：

```bash
python -m qwen35_mtp compare \
  --model-dir "$QWEN35_MODEL_DIR" \
  --prompt '介绍一下杭州。' --chat \
  --max-new-tokens 8 --max-draft-tokens 2
```

`compare` 只有在 ordinary/MTP 的 token ID、EOS 结果完全一致时才返回
`PASS`。MTP counters 还必须显示真实 drafted/accepted token，不能静默回退。

## 为什么第一版会很慢

CPU reference 和最初的 310P 适配允许每轮重算已提交前缀。这样不需要猜测未知
runtime 的 KV/GDN cache 所有权，也没有错误候选污染 state 的风险。路线跑通后再在
同一 backend ABI 内实现增量 cache、verify transaction 和 rollback。

详细接口见 [Ascend 310P 接入说明](docs/ASCEND310P_INTEGRATION.md)，数值门禁见
[准确率契约](docs/ACCURACY_CONTRACT.md)。

## 当前证据边界

当前工作区目标 profile 是 `simulation-only`。CPU、ONNX 和 checkpoint 结构结果不能
称为 ATC、CAModel 或 Ascend 310P 真机通过。内部运行后请按
[反馈格式](docs/BOARD_FEEDBACK.md) 回传原始 JSON、日志和 artifact hash。

另外，官方 310P SelfAttention/SwiGLU 路径不支持 BF16，而原 checkpoint 是 BF16。
FP16 转换实验已获明确批准并通过单个真实文本 CPU admission case；它仍只是
`ELIGIBLE_FOR_ASCEND310P_TESTING_NOT_PROMOTED`，不会静默转换、降低验收阈值或替代
内部 ordinary 图与 310P 真机验证。冻结阈值见 `specs/fp16-experiment.json`。
