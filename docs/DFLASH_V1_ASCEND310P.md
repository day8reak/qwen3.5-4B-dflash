# DFlash rollback：Ascend 310P 接口边界

实际文件部署和命令见 [NPU_DEPLOYMENT.md](NPU_DEPLOYMENT.md)，完整算子 ABI 见
[rollback 算子分析](DFLASH_ROLLBACK_OPERATOR_ANALYSIS.md)。

## 实现结构

```text
shared rollback scheduler / official Draft
└── HIAI Target transaction
    ├── ordinary prompt/decode: accepted_tokens=None
    └── verify T=K+1
        ├── npu_gated_delta_rule_mtp recurrent bank
        ├── Torch Tensor causal-conv bank golden
        ├── existing npu_cache_update_ per row
        └── existing adn_fused_infer_attention
```

NPU 使用独立 `modeling_qwen3_5_hiai_nd_dflash_rollback.py`，原
`modeling_qwen3_5_hiai_nd.py` 不覆盖。Feature 仍固定捕获层
`1,5,9,13,17,21,25,29`，shape `[B,S,20480]`。

## Rollback ABI

```text
input_ids       [B,T] INT64, T=K+1<=17
accepted_tokens [B]   INT8
```

`accepted_tokens` 选择上一轮 state bank；第一轮为 0。GDR bank：

```text
initial/state [B,T,32,128,128] FP32
out           [B,T,32,128]     FP16
```

Conv bank：

```text
state [B,T,8192,4] FP16
out   [B,8192,T]   FP16
```

Full-attention cache 为 block size 64 的 packed paged KV。Verify 可物理写入全部 T 行，但 commit
只推进 logical cursor `1+a`；拒绝尾部不可见，并由下一轮覆盖。

## Bridge 状态所有权

- prompt 逐 token bootstrap，不用 padded full-prefix 初始化 persistent state；
- 32 层 state 跨轮持有；
- scalar GDN state 在第一次 verify 扩成 T 槽；
- T 变化时 select 已提交槽后 rebase；
- position、cache position、mask、allQLen 与 logical cursor 一致；
- verify 失败后整个 session 失效；
- 只返回 logits 和 feature，不把 state 所有权交给 Scheduler。

## 条件新增算子

确定的生产候选是 `CausalConv1dMTP`。`CacheUpdateMTP` 和
`FusedInferAttentionMTP` 是条件候选：必须先在 310P 上验证现有算子的 T=2/5/9/17、跨块、历史
KV 和 causal mask 能力。完整 logits D2H 明显时，可增加 `TargetLmHeadTop1Accept`。

## 真机门禁

1. ordinary incremental 与 rollback token/EOS/stop 零差异；
2. accepted `0/1/K-1/K`，K `1/4/8/16`，连续多轮；
3. cursor `62/63/64/65` 的拒绝尾部不可见；
4. 24 层 GDR/conv 和 8 层 KV 使用同一个接受长度；
5. feature 开关不改变 Target Top-1；
6. 无 CPU fallback，记录 device/runtime/operator source/package identity；
7. 故障注入后无部分提交，重复运行无状态泄漏和持续内存增长。

当前 workspace target profile 是 simulation-only，所以 reduced-shape CPU 测试不能被表述为 310P
通过或性能证据。
