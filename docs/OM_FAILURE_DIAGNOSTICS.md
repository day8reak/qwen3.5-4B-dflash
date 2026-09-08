# v44：OM 首对即停与失败诊断

适用于 incremental C++ runner 1.27.0。用户报告的
`stateful DFlash output differs from ordinary greedy authority` 是正确性失败，
不是接受率阈值，也不是之前的 ACL 500002。本次改善定位能力，不修改 OM 数学、
状态提交、EOS、proposal 预算、fallback 或普通 Target 的权威地位。

## 改了什么

- 每个 warmup 配对、每个 measurement 配对完成后立即比较。首个 warmup 配对不一致
  就退出，不再先跑满 3+10。后续 measurement 同时与各模式第一轮比较，及时发现漂移。
- token 值、长度、stop reason 都要相等；停止原因不同但 token 相同也不放行。
- stderr 给出 phase、1-based run、0-based 首个生成 token 分歧位置、包含 prompt 的
  absolute token index、两边 token/长度/停止原因。缺少一侧 token 用 `<missing>` 表示。
- 失败写独立 JSON，保留两条序列、每个已完成事务的结果和调度信息；退出码仍为 1，
  不生成成功 raw/final 报告。已有文件或临时文件不会被本次运行覆盖。
- ACL launch 错误附上 model role/id、物理行数、Target/Draft ping-pong 槽位；
  compact 结果校验错误附上实际 commit/drafted/accepted/rejected/finished 值。
  异步错误的 launch 上下文不一定就是最初出错的算子，不作因果断言。
- Python 异常保留 C++ 具体错误及诊断路径；旧 runner、进程信号、磁盘异常没有 sidecar
  时仍保留有界日志尾部。报告写入失败不会替换原始错误或变为成功。

OM 图、张量 ABI、四图拓扑和 PASS schema 13 不变。PASS protocol 新增
`pair_validation_policy` 和 `failure_diagnostic_policy`；failure 与 invocation
使用独立 schema 1，不可交给成功报告验证器当成 PASS。

## 升级与运行

沿用已声明的模型/CANN 环境、当前失败用例和已有 v43 bundle。更新本分支源码，
设置其 `framework/python` 模块路径后，在活动 run 中使用**新的** build/output 名称：

```bash
"$MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/cpp-diag127" \
  --output "$AI_RUN_DIR/reports/cpp-build-diag127.json" \
  --ascendcl-root "$ASCEND_HOME_PATH" --device-memory-policy normal-only

"$MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$STATIC_SPLIT_BUNDLE/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/cpp-diag127/qwen35_dflash_incremental_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner-static-split.json" \
  --model-dir "$TARGET_MODEL_DIR" \
  --prompt "世界上最高的山是哪座山？ " --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --eos-token-id 248044 \
  --output "$AI_RUN_DIR/reports/cpp-infer-diag127.json"
```

变量使用现有声明路径；保留实际失败请求的 prompt/chat、EOS 和 runner JSON，
不要为了诊断改参数。以上 EOS 示例对应当前 torch_npu 默认值，若失败请求显式不同，
沿用其实际值。不重新量化或重建已有 v43 OM。若仍是更早 bundle，
先遵循 [v43 图内修复要求](OM_TORCH_NPU_PARITY.md)，不能只改 manifest 或版本号。

该 infer-cpp 仍使用正式 3+10 协议，但发现首对失败就退出。需要独立短复验时，
从下述 invocation JSON 的 `command` 取出**原始 C++ 命令**，保持模型/hash/input/
policy 不变，仅换新 `--output` 并将以下三个参数改为：

```text
--measurement-protocol profile --warmup 1 --repetitions 1
```

这些是原始 C++ runner 参数，不能直接加到 Python infer-cpp。无需启动 msprof；
profile 协议结果是诊断证据，不是正式时延或性能提升证据。不要直接执行来源不明的
invocation 命令，先核对 runner/hash/路径。

## 需要回传的三个文件

以上例子对应：

```text
reports/cpp-infer-diag127-runner-raw.json.failure.json
reports/cpp-infer-diag127-runner-raw.json.invocation.json
log/cpp-infer-diag127-cpp-runner.log
```

invocation 在启动子进程**之前**由 Python 保存，记录完整 argv、runner 与 deployment
manifest SHA-256、各 OM 哈希和声明的 CANN/driver/firmware/device/policy 身份。
`execution_status=not_recorded` 表示它只是启动记录，不是运行结果。
即使进程被信号杀死没有 failure JSON，也可依靠 invocation 与日志追溯。
直接运行 C++ 没有 Python invocation，但 failure JSON 内仍保留 argv、runner version
和输入/OM 预期哈希；`model_hashes_verified` 区分是否已完成全部文件哈希验证。

failure JSON 的关键字段：

- `error`、`stage`、`last_progress`：具体异常与阶段，即使 `--progress false` 也保留。
- `comparison.kind`：`ordinary_dflash_parity` 或 `repetition_stability`。后者两侧
  属于同一模式，`reference_run_index=1`；不能误读为跨模式差异。
- `first_mismatch_index` / `absolute_token_index`：0-based；只有 stop reason
  不同时为 null，不能当成 token 0 出错。
- `expected/actual.measurement.generated_token_ids`、`stop_reason`、`counters`：
  完整两侧结果。measurement 内 `repetition` 为 0-based，顶层 run 为 1-based。
- `expected/actual.transactions`：prefill、speculative-verify、target-only-verify、
  ordinary-decode1 的路径，anchor、K 上限、accepted/rejected、finished、OM 调用数，
  decode iteration、窗口内 0-based step 和 compact 槽位。
  `generated_begin <= first_mismatch_index < generated_end` 对应生成该 token 的事务。
  path 表示调度方法，不单独证明物理 OM：legacy fused 的 VerifyOne 会走 Decode1，
  unified 的 DecodeOne 会走 Verify T=1；结合顶层 `target_only_execution_policy`、
  `models` 和 invocation 中的拓扑判断，默认 static split 才是独立 Verify16 K=0。
- `returned_token_ids` 是图返回值，`[generated_begin, generated_end)` 是实际追加范围；
  末轮全接受但超预算的 bonus 可能在前者中、却不属于最终输出。

运行时/API 失败还没有完整配对时，`comparison=null`，不伪造普通/DFlash 序列。
失败报告可能因权限、磁盘空间、信号等无法写出；以原始非零退出和 durable log 为准。

## 状态信息与结论边界

默认只记录已有 compact D2H 结果、host 逻辑 prefix 和载体位置，不新增 NPU API、
同步或 KV/conv/GDR 全量下载。`compact_slot` 是结果载体槽位，staged=true 时是合并
窗口的 staging slot，**不是某层 KV 或 recurrent state 的 bank 编号**；未知为 -1。
`prefix_tokens_before` 是 host 已提交前缀，不冒充实测 device cursor。
成功计时仍包括轻量 host trace 记账；配对比较、失败 JSON 序列化在模型计时之外。

这些信息可定位首个输出分歧发生在 prefill、speculative commit 或零接受后的 Verify K=0
路径，但不能仅凭路径名字断定是哪一个 kernel/张量先错。若仍需数值根因，应在该边界
定点对比 logits/conv/GDR/KV cursor 与 torch_npu；不能删除校验、改 authority 或放宽阈值。
用户已报告的真机正确性失败仍未闭合，本次不声称接受率恢复或模型逻辑已修复。

回归覆盖：首个 warmup/measurement 配对即停、后续模式漂移、token/长度/EOS 差异、
末轮 bonus、回退轨迹、progress=false、真实 runner 的 fake-ACL 故障注入、
独立 FAIL JSON、旧文件保护和 Python 异常传播。它们是主机诊断功能证据，不是设备精度证据。
另有故障注入覆盖诊断目录不可写时保留原始 token 分歧错误和非零退出码。
