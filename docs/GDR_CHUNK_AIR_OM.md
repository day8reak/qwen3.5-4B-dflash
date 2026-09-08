# 增量 chunk GDR 的 AIR → OM → C++ 路径

原框架只导出整段前缀重算图。新增工厂复用本分支 rollback Target 的原权重、量化模块、
RoPE、GDR 和官方 Draft，导出显式状态接口。旧重算工厂仍可用于对照。

DFlash 使用 **3 个 OM**；加普通生成对照最多 **4 个 OM**：

| OM | 输入行数 | 完整职责 |
|---|---:|---|
| `target_prefill.om` | 64，有效 1..64 | prompt 分块、末行 Top1、8 路特征、Target 状态 |
| `target_verify.om` | 16，有效 1..16 | verify、Top1、接受判断、第二次 GDR、已提交状态 |
| `draft.om` | 新特征 64＋noise 16 | 特征投影、增量 Draft KV 更新、一次草稿生成 |
| `target_decode.om` | 1 | 普通 greedy 对照；DFlash 不使用 |

没有独立 commit OM。verify 在图内比较 proposals 与 Target Top1，求首个不匹配位置 `a`，
再从本轮初始 recurrent state 执行 `effective_length=a+1` 的第二次 GDR。conv 状态选择
同一个前缀，persistent recurrent state 保留原 FP16 边界。capsule 只是图内中间结果。
C++ 再核对接受数量，随后同时发布所有层状态及逻辑 cursor。失败使当前请求失效，必须
reset 才能复用。零接受率关闭后续 Draft，继续调用 verify，`valid_rows=1`，仍执行两次 GDR。

## 构建

当前工厂支持 W8A8 Target＋FP16 Draft；FP16 eager 仍用原入口，没有新增 FP16 导出工厂。
沿用 [原框架文档](QUANT_AIR_OM_FRAMEWORK.md) 的 CANN/TorchAir、接收方 wrapper、量化 YAML、
外部权重和 `lock_quant_inputs.py` 输入锁。不要将这些资源放进仓库。
先启动 workspace session，激活已声明环境；配置副本、生成物及日志均放在 `$AI_RUN_DIR`。
下面 `python` 指模型侧解释器，`PYTHONPATH` 须包含仓库根与 `framework/python`。

复制 [配置样例](../config/gdr_chunk_air_om_factory.example.json) 到 run 目录并填写路径。
`max_sequence_length=C` 是 64 的倍数，范围 64..32704；实际 KV 为 `C+64`，末尾留给 padding
scratch。请求必须满足 `prompt_length + max_new_tokens <= C`。

```bash
python -m qwen35_dflash.ascend310p build-om \
  --factory qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs \
  --factory-config "$AI_RUN_DIR/gdr-factory.json" \
  --bundle-dir "$AI_RUN_DIR/gdr-bundle" \
  --atc "$ASCEND310P_ATC_BIN" --soc-version "$SOC_VERSION"

python -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/cpp-build" \
  --output "$AI_RUN_DIR/cpp-build-report.json" \
  --ascendcl-root "$ASCEND_HOME_PATH"
```

`SOC_VERSION` 必须是实际 ATC 型号，例如 `Ascend310P3`。export-air、compile-om 和
run-e2e-cpp 原有入口也支持此工厂。AIR manifest 保存有序 tensor ABI，编译后完整保留；
C++ 逐项核对实际 OM 的 I/O 数量、dtype、shape、字节数和 hash，跨图状态形状不一致会失败。

## 配对运行与纯 DFlash

普通对照保留 `include_ordinary_decode: true`，用原入口：

```bash
python -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest "$AI_RUN_DIR/gdr-bundle/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/cpp-build/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/runner.json" \
  --model-dir "$TARGET_MODEL_DIR" --prompt '请介绍一下杭州' \
  --max-new-tokens 64 --max-draft-tokens 15 \
  --output "$AI_RUN_DIR/gdr-paired.json"
```

`runner.json` 沿用原样例，填真实设备/CANN/driver/firmware/runtime 身份。增量 bundle 自动
选择 C++ 多图调度，`graph_name` 不再用于选择单一重算 OM。报告包含交错普通/DFlash 的
3 次 warmup＋10 次测量、token/EOS 对照、每次测量的 `stage_ms`。`target_verify` 耗时包括
接受判断和第二次 GDR；这些是同步图调用时间，算子时间仍需 msprof。
每次请求的状态清零计时另记为 `latency_ms.request_reset`，不计入 prefill/model_total；
比较包含请求初始化的完整时延时须加回，不能直接与不同计时范围的结果比较。

只部署 DFlash 时设置 `include_ordinary_decode: false`，即可只导出三个 OM。
也可以保留四图 bundle，仅生成三图加载计划：

```bash
python -m qwen35_dflash.ascend310p prepare-chunk-plan \
  --deployment-manifest "$AI_RUN_DIR/gdr-bundle/deployment-manifest.json" \
  --mode dflash --output "$AI_RUN_DIR/dflash-plan.txt"

read -r PLAN_SHA _ < <(sha256sum "$AI_RUN_DIR/dflash-plan.txt")
"$AI_RUN_DIR/cpp-build/qwen35_dflash_acl_runner" \
  --model-kind chunk --model "$AI_RUN_DIR/dflash-plan.txt" \
  --model-sha256 "$PLAN_SHA" --mode dflash \
  --prompt-token-ids '4,5,6' --max-new-tokens 32 \
  --output "$AI_RUN_DIR/dflash-only.json"
```

示例 token IDs 须换成实际 tokenizer 输出，并通过 `--eos-token-ids` 传入 EOS 集合。
纯 DFlash 不加载、不要求 unused decode OM；纯普通生成用 `--mode ordinary`。
单模式报告标记 `ordinary_parity=NOT_RUN`，不能代替正确性对照。
C++ 热循环不需要 Python、pyACL 或 torch_npu；下面的诊断控制器只需要 Python 标准库。

## C++ 和 Python 的单次 msprof

两个入口共用 `tools/run_msprof.sh`。省略新增参数时保留原 Python DFlash 行为。

| 入口 | 模式参数 | 单独阶段 | `all` 的窗口顺序 |
|---|---|---|---|
| Python NPU，默认 | `--profile-mode ordinary` | `prefill`、`decode` | prefill → decode |
| Python NPU，默认 | `--profile-mode dflash`，默认 | 原来的 9 个阶段 | 见 [NPU 文档](DFLASH_RUN_AND_VALIDATE.md#74-只采一次-prefill-或-draft-生成--target-verify) |
| C++，`--profile-backend cpp` | `--profile-mode ordinary` | `prefill`、`decode` | prefill → decode |
| C++，`--profile-backend cpp` | `--profile-mode dflash` | `prefill`、`draft`、`verify` | prefill → draft → verify |

普通 OM 先从含 decode 的四图 bundle 生成加载计划，C++ 只加载 prefill 和 decode 两个 OM。
下面一条采集命令加载一次模型，分别采一次完整 prefill 和一次真正的一行 decode：

```bash
python -m qwen35_dflash.ascend310p prepare-chunk-plan \
  --deployment-manifest "$AI_RUN_DIR/gdr-bundle/deployment-manifest.json" \
  --mode ordinary --output "$AI_RUN_DIR/ordinary-plan.txt"

read -r PLAN_SHA _ < <(sha256sum "$AI_RUN_DIR/ordinary-plan.txt")
tools/run_msprof.sh \
  --label om-ordinary-all --output-dir "$AI_RUN_DIR/om-ordinary-profile" \
  --python "$MODEL_PYTHON" \
  --profile-backend cpp --profile-mode ordinary \
  --profile-stage all --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$AI_RUN_DIR/cpp-build/qwen35_dflash_acl_runner" \
    --model-kind chunk --model "$AI_RUN_DIR/ordinary-plan.txt" \
    --model-sha256 "$PLAN_SHA" --device-id 0 \
    --prompt-token-ids "$PROMPT_TOKEN_IDS" --eos-token-ids 248044
```

`PROMPT_TOKEN_IDS` 是相同 tokenizer、相同 chat template 得到的逗号分隔整数。
只测一段时将 `all` 改为 `prefill` 或 `decode`，使用新 label/输出路径。
`MODEL_PYTHON` 在 C++ 采集时可以是普通 Python 3.10+，无需安装 torch、torch_npu、pyACL。
设备初始化、执行与同步全部由 C++ AscendCL 完成，wrapper 还会记录 `npu-smi info`。
阶段诊断绕过原来的 3+10 benchmark，`--profile-warmup` 控制窗口外预热；不通过
`--max-new-tokens` 或 `--max-draft-tokens` 截短所测阶段。

纯 DFlash 使用前面生成的 `dflash-plan.txt`，分别采一次 prefill、Draft 和融合 verify：

```bash
read -r PLAN_SHA _ < <(sha256sum "$AI_RUN_DIR/dflash-plan.txt")
tools/run_msprof.sh \
  --label om-dflash-all --output-dir "$AI_RUN_DIR/om-dflash-profile" \
  --python "$MODEL_PYTHON" \
  --profile-backend cpp --profile-mode dflash \
  --profile-stage all --profile-warmup 1 --aic-metrics PipeUtilization \
  -- "$AI_RUN_DIR/cpp-build/qwen35_dflash_acl_runner" \
    --model-kind chunk --model "$AI_RUN_DIR/dflash-plan.txt" \
    --model-sha256 "$PLAN_SHA" --device-id 0 \
    --prompt-token-ids "$PROMPT_TOKEN_IDS" --eos-token-ids 248044
```

C++ 的窗口范围：

- `prefill`：完整 prompt 的所有 Target 64 行分块，包含各 OM 的 Top1；状态清零在窗口外，
  不执行 Draft。一次 prefill 窗口可能包含多次 `target_prefill.om` 调用。
- `decode`：真实 prefill、缓存准备在窗口外，只调用一次 `target_decode.om`，包含图内
  LM head、Top1 和状态更新。
- `draft`：先在窗口外准备 Target prefill 和长 prompt 前面分块的 Draft KV；窗口内只调用
  一次 `draft.om`，包含最后一块特征投影、Draft KV 追加和完整 15 token proposal。
- `verify`：prefill、Draft、EOS 截断与 verify block 整理在窗口外；窗口内只调用一次
  `target_verify.om`，包含两次 GDR、Top1、接受判断和 committed state 输出。
  C++ 接受数量复核与状态指针发布在窗口外。

融合 verify 没有独立 commit OM，因此 C++ 不提供 `accept-commit` 等内部子图采集；
从该 verify 目录的算子明细查看内部算子，或使用原 Python NPU 的细阶段采集作辅助分析。
两条路径的计时范围不完全相同：Python ordinary 的 Top1 在窗口外、cache reset 在 prefill
内；C++ Top1 已在 OM 内、reset 在窗口外，小量输入/输出 H2D/D2H 在窗口内。
不能把两者窗口时长直接当作相同口径比较。

每个阶段先 fresh-state warmup，再从相同 prompt 重建状态。每次 start/stop/quit 都等待
msprof 成功回执；`all` 复用同一个应用 PID，为各阶段独立 attach 和导出。prefill anchor
已是 EOS 时不会伪造 decode/verify 窗口。普通 decode 要留一行 KV，verify/all DFlash 要留
完整 16 行；未覆盖的容量会明确报错。

以上普通 `all` 示例输出：

```text
om-ordinary-profile/
  profile/msprof/om-ordinary-all/prefill/  # PROF_* 和 op_summary*.csv
  profile/msprof/om-ordinary-all/decode/
  om-ordinary-all-stage-report.json       # 每窗口调用数、范围、同步时间、预热一致性
  om-ordinary-all-stage-summary.csv       # mode/backend、同步时间、算子行数、目录
  manifest/om-ordinary-all.json           # 命令、设备、源代码身份与最终状态
  manifest/om-ordinary-all-control.json   # 应用/采集 PID、回执、退出状态
  log/msprof-om-ordinary-all.log
```

单阶段目录直接是 `profile/msprof/<label>/`。优先从 `op_summary*.csv` 查看算子的执行时间；
`profiled_elapsed_ms` 是含 profiling 开销的同步阶段墙钟时间，排除 msprof 控制等待。
任一窗口失败、缺少回执或无算子数据，整体 FAIL，不输出成功汇总。
真实延迟仍用不带 msprof 的 3+10；stage report 明确不声明整段 strict-greedy 正确性。

## 验证范围与代价

已提供 CPU Tensor/torch.export、AIR/ATC fixture 与真实 C++ 代码连接 fake ACL 的测试，
覆盖零/部分/全接受、EOS、跨 block、长 prompt、重复请求、异常接受数量、执行失败、hash
损坏和三图加载；还覆盖真实 C++ socket + msprof 控制器的单次/all 边界、异常退出和空导出。
测试的 fake ACL/msprof 只验证控制流程。这些 **不是实际 AIR/OM 或 Ascend 310P 精度、稳定性、时延证据**。

实机须验证接收方 GDR/fused attention 的 TorchAir 支持，尤其是
`adn_fused_infer_attention` 的 `pse_shift=INT64[1] logical_end` 导出接口；完成 ATC 后对比
原生 ordinary、OM ordinary 和 OM DFlash，要求 token-ID/EOS 零差异，再测 3+10 时延。
配对 OM 一致性不能排除共有的导出错误，原生 ordinary 对照不可省略。

固定 gear 的 DFlash 单行 fallback 使用物理 16 行、chunk_size=64；原生单行是
chunk_size=1。浮点等价仍需设备精度门禁，普通对照则使用真正的一行 decode 图。
Draft 为合并频繁连续执行的结构，新特征采用 64 行 gear；长 prompt 的非末尾分块也运行
Draft 初始化 KV，丢弃当时 proposals。部分接受后的特征 padding 到 64，无效行先清零，
防止未定义的 GDR padding/NaN 污染后续 KV。
共享 prefill OM 为每个分块计算末行 Top1；原 Python prefill 只在最后一块计算 LM head。
这些为控制 OM 数量引入的额外计算需要实测评估，不能预先声明更快。

Target KV 使用 functional row update 和 current/next 两套 device buffer，便于验证状态
归属，但可能拷贝整段 cache。是否换成已验证的原地 CacheUpdate，应由实测决定。
不同 OM 不会自动共享权重；DFlash 加载两张 Target 图和一张 Draft，对照另加 decode 图。
启动日志打印模型数及持久 buffer 字节数，模型加载计时单列，实际设备显存仍需确认。
