# v43：对齐当前分支 torch_npu 的状态与调度

## 范围与结论

基线是当前分支的 `models.dflash_v1.run_npu → run_rollback → dflash_rollback_greedy`，
不是另一个 Git rollback 分支。v42 的 Draft attention 修复继续保留；本次针对源码 review
确认的另外几处差异，不以修改接受率分母或降低精度来提高数字。

**必须重新导出 AIR、编译 OM，并重建 C++ runner 1.26.0 / report schema 13。**
四个静态 OM、权重、量化文件、KV 容量和张量 ABI 不变，不需要重新量化。
只有更新 runner、manifest 或旧 pbtxt 都不能修复旧 Prefill OM 中的状态提交。

| 环节 | v42 / runner 1.25 | v43 默认静态 split |
| --- | --- | --- |
| 部分 Prefill 的 conv state | 提交物理 64 行后的尾部，包含 padding | 提交逻辑有效行 L 后的历史 |
| 每轮 proposal 数 K | `min(Kmax, remaining-1)`；末 token 走 Decode1 | `min(Kmax, remaining)`；只裁掉末轮超预算 bonus |
| 第一次零接受后 | 默认继续 Draft | 默认 `request-target-only`，本请求后续不再 Draft |
| DFlash target-only | ordinary Decode1 | Verify16，K=0，只提交 FP32 GDR-MTP bank 第 0 槽 |
| infer-cpp / run-e2e-cpp EOS | tokenizer 的 EOS，例如 248046 | 当前 run_npu 锁定的 248044，可显式覆盖 |

EOS 差异不一定影响未遇到 EOS 的 32-token 用例；OM 切换耗时也不能解释 proposal ID 差异。
以上是可验证的源码差异，尚不能断定各自解释了用户接受率差距的多少，更不能保证达到 60%。

## 1. Prefill：物理 padding 不能推进持久 conv state

例如真实 prompt 有 17 行、Prefill OM 固定 64 行。旧路径已按 L=17 提交 GDR state、
cursor 和有效 features，但 causal-conv helper 按 64 行更新 conv state。下一次 Target
调用读到的历史因此与 compact torch_npu 不同。ordinary OM 和 DFlash OM 都复用此
Prefill，所以它们相互零 mismatch 仍不能证明与 torch_npu 对齐。

设历史缓存宽度为 C，进入调用前的缓存为 S，当前物理输入为 X：

```text
history = concat(S, X)
committed_state = history[..., L : L+C]
```

实现使用固定宽度 tensor gather，不使用数据相关 Python slice、动态 SymInt shape 或
动态广播。每个 batch 可有自己的 L。原 native/Torch convolution 继续计算物理输出，
仅纠正持久缓存；不替换算子输出、不改变 dtype 或 kernel 数学。L 等于物理宽度时，
提交规则与原 compact 路径相同。

导出入口要求实际加载的 language model 暴露
`dflash_conv_state_commit_policy="logical-effective-length-v1"`；若外部 receiver 或
旧进程仍加载旧 modeling，立即拒绝导出。应更新实际加载的当前分支 modeling，不能只
手工补这个标记。AIR graph metadata 同时记录此 policy，SOURCE_LOCK 保留真实源码哈希检查。

## 2. 调度与 GDR 精度

当前 eager 的 `block_size` 包含 anchor：Kmax=15 对应 `--block-size 16`。
两端都设 `max_new_tokens=32` 仍不够，还需冻结 prompt token IDs、量化/权重、
block size、EOS 和 fallback。

新 stateful C++ 循环按剩余预算选 K。即使只剩一个 token，Draft 尚启用时也执行 K=1；
全部接受时最多多算一个 bonus，但不把它加入输出，也不允许下一轮读取这个终态。
计数仍是 `accepted_draft_tokens / drafted_tokens`，不为了末轮裁剪而少计 proposals。
cache admission 也预留了末轮最后一个已接受输入的空间。

首次零接受后，当前 eager 继续用 Verify1/GDR-MTP 路径，保留 FP32 recurrent bank。
普通 Decode1 使用 ordinary GDR，其内部 recurrent state 有 FP16 转换边界，因此不能
仅因两者都生成一个 token 就替换。新静态 split 复用常驻 Verify16，置 K=0，anchor
后物理 padding 不可见，选择 bank[0]、cursor+1，只下载单 token compact 前缀。
后续不调用 Draft，也不换到 Decode1 组；ordinary 模式仍使用独立 Decode1 OM。

报告中 `target_decode1_executions` 是兼容的**逻辑单步计数**；
`target_only_verify_executions` 是其子集，不额外计入 `model_executions`。
独立 split 的物理 Decode1 次数为前者减后者，物理 Verify 次数为
`target_verify_commit_executions + target_only_verify_executions`。不要按旧字段名误判换组。

逐轮调度对齐范围是**默认静态 split、sync window=1、separate prefill**。
window>1 必须先消费已排队事务；coalesce-first-verify 可能在观察首 token EOS 前已运行
Verify；旧 fused 没有独立 Verify，target-only 仍退到 ordinary Decode1。这些显式诊断/
历史路径不声称与 eager 逐轮等同；本次没有为了它们新增 OM 或改变默认四图拓扑。

## 3. 更新与复验

从当前分支部署源码根目录设置模块路径，沿用已声明的模型/CANN 环境。产物写入新 run：

```bash
export DFLASH_SOURCE="$PWD"
export PYTHONPATH="$DFLASH_SOURCE/framework/python:$DFLASH_SOURCE${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
"$MODEL_PYTHON" -c 'from qwen35_dflash.ascend310p.quant_factory import _verify_quant_source_lock; print(_verify_quant_source_lock())'
```

按 [默认静态 split 操作步骤](STATIC_SPLIT_OM_DEFAULT.md#从哪里重新运行) 重建 runner、
export-air、compile-om，再 infer-cpp。保留旧 bundle，不手改旧 manifest，不重新量化。
有独立 receiver 源码副本时同步本次 modeling；若已锁定的外部文件有预期变更，审阅后重新
冻结对应 input manifest，不能关闭 hash 校验。

特别检查你**实际传入**的 runner JSON，而不只是仓库模板：

```json
{
  "zero_accept_fallback_policy": "request-target-only",
  "dflash_sync_window": 1,
  "prefill_completion_policy": "separate"
}
```

原 JSON 中显式 `"disabled"` 不会被新默认值覆盖。两个 C++ CLI 入口默认 EOS=248044；
可传重复的 `--eos-token-id` 覆盖，但做本分支 run_npu 对照时要保持一致。既有比较
使用 248046 时，不能与新默认直接混算；新报告记录最终 EOS IDs 和调度 policy。

先用原始失败 prompt、相同 chat 模板和 32-token 预算做诊断，不急于跑十轮时延：

1. 检查新 AIR/OM/source hashes、runner 1.26.0、schema 13，四图 static 和 dtype/bytes
   门禁全部保留；protocol 应为 `torch-npu-remaining-v1` / `static-verify16-k0`。
2. 对比新 OM ordinary 与当前分支 torch_npu ordinary 的完整 token IDs、EOS、stop reason。
3. 对比新 OM DFlash 与当前分支 torch_npu DFlash 的输出和原始 accepted/drafted 计数。
   差异未闭合时，冻结首个分歧轮的 anchor、K、Target features、Draft proposal IDs、
   Target verify IDs、accepted count 和进入/离开的 conv、recurrent、KV 逻辑前缀。
4. 只有上述正确性通过，才做相同输入的 3 warmup + 10 次未 profiling 时延。
   重复一个 prompt 十次是稳定性测试，不是十个独立接受率样本。

现有 `ordinary_parity: PASS` 只比较 OM ordinary 与 OM DFlash，**不包含外部 torch_npu
基准**。不能用它替代第 2、3 步，也不要因切换模型更快就推断接受率问题已解决。

## 回归证据与剩余门禁

- `test_prefill_conv_parity.py`：L=1/3/4/17/63/64、非零历史、FP16/FP32、非零 padding，
  compact 对照、跨块和连续 decode；固定 64 行 strict torch.export 重放。
- 同一测试还贯穿真实 Prefill/Verify wrapper 边界、K=0/1/15、连续 Verify、标量 state、
  FP32 recurrent 和 bank[0]；小模型的计算后端是 CPU fixture，不是原 checkpoint/NPU。
- C++ generation 覆盖预算 1/2/3/16/17/32、EOS、末轮 bonus、零接受后的 VerifyOne；
  fake ACL 覆盖 D2D anchor、多 token 载体、显式 token override、分组驻留和报告闭合。
- 导出拒绝缺少 conv policy 的旧 receiver；报告拒绝旧 schema、旧 budget/target-only policy、
  非法单步 Verify 子集计数；源码锁测试不能绕过。

本地声明的 profile 为 simulation-only；严格设备/ATC preflight 没有可用 310P 或 ATC。
本次主机验证为 Python 788 passed（另 43 subtests）、C++/fake ACL 32/32；
同一 C++ 套件在 ASan/UBSan 下也为 32/32。源码锁校验保留，并随本次 modeling 变更同步。
**未获得本次源码的 AIR/OM 编译、真机数值、接受率或性能 PASS。** 特别是固定物理 Verify16
与 eager Verify1、固定 Draft padding 与 compact Draft、V4444 与原 eager 算子的数值
一致性，仍需真实 checkpoint/device 证据；不据 CPU fixture 推广成真机保证。
