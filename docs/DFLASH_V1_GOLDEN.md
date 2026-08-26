# DFlash rollback CPU/Golden

CPU golden 验证 scheduler、Draft 数学、framework cache transaction 和 strict-greedy token 等价；
它不能替代 CUDA 或 310P 设备证据。

## 1. CPU 运行

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. python -B \
  -m models.dflash_v1.run_rollback \
  --target-dir /path/to/Qwen3.5-4B \
  --draft-dir /path/to/Qwen3.5-4B-DFlash \
  --prompt "请用一句话解释为什么天空是蓝色的。" \
  --prompt-mode chat \
  --enable-thinking \
  --max-new-tokens 8 \
  --max-draft-tokens 4 \
  --eos-token-id 248044 \
  --dtype float32 \
  --device cpu \
  --report /path/to/run/dflash-rollback-cpu.json
```

真实 4B Target + Draft 的 CPU 内存和时间成本很高；自动化逻辑检查优先使用下面的 reduced-shape
测试。

## 2. 报告检查

```python
import json

with open("/path/to/run/dflash-rollback-cpu.json", encoding="utf-8") as stream:
    report = json.load(stream)

assert report["route"] == "qwen3.5-dflash-incremental-rollback"
assert report["classification"] == "CPU/framework rollback simulation"
assert report["strict_greedy_exact_match"] is True
assert report["verification_mode"] == "incremental_transactional_rollback"
assert report["historical_prefix_replay_during_verify"] is False
assert report["ordinary"]["generated_token_ids"] == report["dflash"]["generated_token_ids"]
audit = report["target_rollback_audit"]
assert audit["pending_transaction"] is False
assert audit["historical_prefix_replay_during_verify"] is False
```

如果 prompt 在 bootstrap 立即 EOS，`dflash_execution_gate` 会标为 inconclusive；换一个固定 prompt
来证明至少执行一轮 Draft/verify/commit。

## 3. 快速自动化检查

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_scheduler.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_framework_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_internal_dflash_bridge_rollback.py
PYTHONDONTWRITEBYTECODE=1 python tests/test_dflash_rollback_helpers.py
```

分别覆盖 accepted `0..K`、correction/bonus、KV/GDN restore、bounded commit replay、HIAI bank
rebase/logical cursor 与 causal-conv golden。

## 4. Full-prefix oracle

旧 correctness-first sequential full-prefix CLI 仍可显式运行：

```bash
python -B -m models.dflash_v1.dflash_qwen_adapter_v1 ...
```

它逐 proposal 重算独立完整前缀，用于诊断整块 causal/prefix-invariance 差异。兼容入口
`python -m models.dflash_qwen_adapter_v1` 已经指向 rollback，不再指向旧 oracle。
