# 增量 OM 与 C++ 高性能路线

当前 `quant_dflash_recompute.om` 是 correctness 基线，不是最终性能形态。它的每次调用都重算完整
前缀，而且一个图同时运行 Target 和 Draft。C++ speculative round 又会调用这个大图两次：先取
proposal，再把 `prefix + proposals` 重算一次进行 verify。普通生成也会无条件支付 Draft 成本。

因此，拆成多个逻辑角色可能显著提速；**不是因为 OM 文件数从 1 变成 4**，而是因为它允许：

- prompt 只 prefill 一次；
- ordinary decode 每次只处理 1 行且不运行 Draft；
- Draft 复用自己的 KV 和 Target 新增 feature，只生成最多 15 个 proposal；
- Target verify 固定处理 16 个因果行，但只提交运行时 `logical_proposal_count` 指定的前缀，
  不再重算历史前缀；
- proposal、KV、GDR/conv state 和 feature 全部留在 device；
- 一个 speculative round 只在 accept/commit 后同步一次，并只回传少量 token ID 和计数。

完整机器可读合同见 `framework/abi/incremental-performance-v2.json`。用户已明确批准该状态图，
批准记录位于 `framework/abi/approvals/incremental-performance-v2.json`，当前状态是
`APPROVED_IN_IMPLEMENTATION_NOT_ACTIVE`：可以实现，但还不能冒充已经生成、真机验证或达到性能
目标的 OM。

## 1. “4 个 OM”只是候选，不是结论

四个逻辑角色是：

| 角色 | 主要工作 | 不应再做的工作 |
| --- | --- | --- |
| `target-prefill` | 1..64 行 prompt chunk，原 ChunkGatedDeltaRule | 非末 chunk 的完整 LM head、Draft proposal |
| `target-decode1` | 1 行普通 decode / zero-accept fallback | Draft |
| `draft-propose` | 复用 Draft KV，输出固定 `[anchor,p0..p14]` device carrier | Target |
| `target-verify-commit` | 固定 T=16 verify、逻辑 K=1..15、精确 accept/状态选择 | 历史前缀重算、host state 搬运 |

物理文件不一定恰好是四个：

- 如果 verify OM 能可靠包含 `T=1` gear，`target-decode1` 可以与它合并，成为 3 个常驻 OM；
- 如果 receiver 自定义算子和 ATC 能证明多 gear/分支完全可用，可进一步测试 2 个动态 OM；
- 如果静态小图明显更快且显存足够，才选 4 个静态 OM；
- 如果 launch/host 边界是主要热点，才测试 Draft→verify 的 supergraph。

真正的选择规则是：所有候选都必须零 token/EOS 差异，然后选真机端到端 median latency 最低者。

## 2. 为什么不能直接认定 4 OM 更快

不同 Target OM 可能各自携带一份 Target 权重。CANN 文档允许串行模型共享一块最大 workspace，
但不能据此假设不同 OM 文件会自动共享权重。四静态图候选中有三个 Target 角色，若权重重复，
可能多占数 GiB，甚至因为内存压力、装载失败或带宽竞争而更慢。

参考 CANN API：[`aclmdlQuerySize`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0304.html)
返回模型 work/weight 大小；[`aclmdlLoadFromFileWithMem`](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_0285.html)
允许调用方提供内存，并明确描述串行模型共享 workspace 的场景。后者对“同一模型多线程共享
weightPtr”的说明不能外推为“不同 OM 自动共享权重”。

每个候选必须先记录所有 OM 的：

```text
sum(weightSize)
max(workSize)                 # 仅在模型严格串行并使用显式共享 workspace 时
persistent state bytes
peak transient state/workspace
I/O buffer + runtime margin
```

然后在完整候选集同时 load 的条件下测试，不能逐个 OM 单独 load 成功就宣称组合可用。

### 2.1 查询完整候选集

构建后，使用新增的 `qwen35_dflash_om_inspect`。下面的 `974651392` 是本页表格中
`persistent state + T16 transient banks` 的小计；1 GiB margin 只是示例，真机应按 I/O、feature、
logit、allocator 和并存服务重新声明。`DEVICE_BUDGET_BYTES` 必须来自本次目标机预算，不能照抄：

```bash
CPP_BUILD="$AI_RUN_DIR/build/cpp-performance"
STATE_BYTES=999817216
IO_RUNTIME_MARGIN_BYTES=1073741824
DEVICE_BUDGET_BYTES=<本次进程可使用的设备字节预算>

cmake -S framework/runtime/cpp -B "$CPP_BUILD" \
  -DQWEN35_DFLASH_BUILD_ACL_RUNNER=ON \
  -DQWEN35_DFLASH_BUILD_TESTS=ON \
  -DASCENDCL_ROOT="$ASCEND_HOME_PATH"
cmake --build "$CPP_BUILD" -j

"$CPP_BUILD/qwen35_dflash_om_inspect" \
  --model target-prefill="$AI_RUN_DIR/om/target-prefill.om" \
  --model target-decode1="$AI_RUN_DIR/om/target-decode1.om" \
  --model draft-propose="$AI_RUN_DIR/om/draft-propose.om" \
  --model target-verify-commit="$AI_RUN_DIR/om/target-verify-commit.om" \
  --state-bytes "$STATE_BYTES" \
  --io-runtime-margin-bytes "$IO_RUNTIME_MARGIN_BYTES" \
  --device-budget-bytes "$DEVICE_BUDGET_BYTES" \
  --output "$AI_RUN_DIR/out/performance/four-resident-memory.json"
```

三 OM 候选删除 `target-decode1` 行，并把输出名改为
`three-resident-memory.json`。重点查看：

```bash
jq '{models, budget, assumptions, claim_boundary}' \
  "$AI_RUN_DIR/out/performance/four-resident-memory.json"
```

`fits_declared_budget=true` 只表示静态估算未超预算。它不替代完整集合同时 load、运行时显存峰值、
数值正确性或性能测试。

## 3. 已知状态预算

以下按 batch 1、Target KV 最大长度 2048、verify `T=16` 计算：

| 项目 | 字节 | MiB | 生命周期 |
| --- | ---: | ---: | --- |
| Target scalar conv FP16 | 1,572,864 | 1.5 | persistent |
| Target scalar recurrent FP32 | 50,331,648 | 48 | persistent |
| 8 层 Target K/V FP16 | 67,108,864 | 64 | persistent |
| 6 层 Draft K/V FP16 | 50,331,648 | 48 | persistent |
| persistent state 小计 | 169,345,024 | 161.5 | request lifetime |
| Target conv bank FP16 | 25,165,824 | 24 | verify transient |
| Target recurrent bank FP32 | 805,306,368 | 768 | verify transient |
| verify bank 小计 | 830,472,192 | 792 | graph workspace/transient |

这里尚未包含模型权重、ATC workspace、I/O、feature、logit 中间量和 allocator/runtime overhead。
尤其是 792 MiB state bank：它不应该每轮搬回 host，也不应该作为下一轮的持久输出。推荐在
verify graph 尾部根据接受数 `a` gather slot `a`，只持久化选中的 scalar state；bank 只作为当轮
workspace。

持久 recurrent 统一用 FP32：普通 GDR 仍保留 receiver 现有的 FP16 输出边界，再把该结果无损
扩宽到 FP32；GDR-MTP 选中的 FP32 state 则无需每轮降回 FP16。这样 prefill/decode/verify 的
外部 binding dtype 固定，同时不丢掉 rollback 路线已经保留的 FP32 state。

## 4. 精确 accept/commit 合同

Target 输入与输出对齐为：

```text
input = [anchor, p0, p1, ..., p(K-1)]
top1  = [t0,     t1, t2, ..., tK]
```

`a` 是从开头连续满足 `p[j] == t[j]` 的 proposal 数。首次不匹配时输出
`p[0:a] + t[a]`；全部匹配时输出 `p[0:K] + t[K]`。如果已接受 proposal 中出现 EOS，必须在
第一个 EOS 截断，不能再输出 bonus。

Target 真正提交的输入行永远是 `anchor + a 个 accepted proposal`，即 `1+a` 行；对应 GDR/conv
bank slot 正好是 `a`。correction/bonus 虽然已经作为生成 token 输出，但仍是下一轮尚未处理的
anchor。这一规则与当前 rollback scheduler 一致，不能为了融合而改变。

生产 verify 固定物理 `T=16`，另传 `[1] INT32 logical_proposal_count`。如果本轮只需要 K 个
proposal，`pK..p14` 是 scratch suffix。Target 的 attention、GDR、conv、MLP 和 LM head 都是
因果计算，因此 suffix 不能影响前面的 logit、feature 或 state slot；graph tail 只比较逻辑前缀、
只 gather slot `a`、只推进 `1+a`。这不是 padding 近似：promote 前仍必须在逻辑
`K=1/3/5/7/15`、跨 62/63/64/65 位置以及拒绝后续跑 token 的测试中证明该因果 suffix 约束。

设备侧实现位于 `ExactAcceptCommitStateGraph`。它还会在第一个 proposal EOS 截断 drafted/accepted
范围、抑制 EOS 后 bonus，并把未提交 feature 行清零。主机可先运行捕获与穷举测试：

```bash
PYTHONPATH="$PWD/framework/python:$PWD" "$MODEL_PYTHON" -m pytest -q \
  tests/test_incremental_om_transaction.py \
  tests/test_incremental_performance_contract.py
```

## 5. C++ 热循环目标

推荐的 C++ request context 一次建立并复用：

```text
ACL device/context/stream
所有候选 OM model ID/description
一块共享 serial workspace（仅在 query/load 验证后）
每个 role 的 dataset
两组可 ping-pong 的 scalar state / feature buffer
Target paged KV、Draft KV、logical cursor
proposal/verify/commit 小 buffer
```

高性能 speculative round 应是：

```text
enqueue draft-propose
  [anchor,p0..p14] device buffer ───────┐
enqueue target-verify-commit <──────────┘
enqueue compact commit result D2H
aclrtSynchronizeStream                  # 本轮唯一 host-visible barrier
host 只处理 EOS/长度/输出文本
```

禁止把 proposal 先 D2H、同步、再 H2D 给 Target；禁止把 state bank、KV、feature 或完整 logits
每轮搬到 CPU。ordinary route 则只调用 prefill/decode，不得调用 Draft。

当前单一基线 runner 的 JSON 也会新增：

```text
model_memory_query.work_bytes
model_memory_query.weight_bytes
model_memory_query.source = aclmdlQuerySize
execution_io_counters.host_to_device_bytes
execution_io_counters.full_host_to_device_bytes
execution_io_counters.device_to_host_bytes
execution_io_counters.full_device_to_host_bytes
execution_io_counters.maximum_target_elements_per_call
```

当前 runner 已在不改变 OM ABI 的前提下使用 changed-range H2D，并把 Target D2H 限制为尾部
`K+1` 行；上述计数用于让 msprof API timeline 与 runner 自报字节互相校验。这只能减少传输和
host API 开销，不能消除 OM 内部的完整前缀重算。只有上述多模型 inspector 才会按
`sum(weights) + max(serial workspace) + state + margin` 计算候选集合。

## 6. 真机选择顺序

1. 对每个候选 OM 记录 `aclmdlQuerySize` 的 `workSize/weightSize` 和 SHA-256。
2. 计算 `sum(weightSize) + max(workSize) + state + I/O + margin`；不能假定不同 OM 权重共享。
3. 一次性 load 完整候选集，记录真实成功/失败和设备身份。
4. 分别以 1..3 次短跑用 msprof 定位 kernel、launch、memcpy、同步和空洞；profile 数据不能作
   正式 latency 基线。
5. 关闭 msprof，以同 prompt/token 上限运行 3 warmup + 10 measurement。
6. 先要求 ordinary/DFlash token ID、EOS、stop reason 全部零差异，再比较 median、p90、TTFT、
   TPOT、每轮同步数和接受率。
7. 只提升通过上述门禁且端到端最快的拓扑。

单 OM 的 msprof 命令已经写在 `docs/QUANT_AIR_OM_FRAMEWORK.md` 的“单独 profile 每个 OM”章节。
角色拆分完成后，对 `target-prefill`、`target-decode1`、`draft-propose` 和
`target-verify-commit` 分别套用同一命令，并使用不同 `--label`，避免把 profile 目录混在一起。

## 7. 当前证据边界

本地环境是 simulation profile，没有物理 Ascend310P、匹配的 TorchAir/ATC 和真实 OM。因此当前
可以冻结 ABI、实现 host/Fake-ACL 测试和生成真机命令，但不能宣称：

- 2/3/4 OM 中哪个一定更快；
- 不同 Target OM 会共享权重；
- 792 MiB bank 会被编译器复用为理想 workspace；
- receiver 自定义算子能通过动态 gear；
- 已达到闭源框架时延。

这些结论必须由同一台 310P 上的组合 load、零差异验证、未 profile 的 10 次分布和 msprof
诊断共同给出。
