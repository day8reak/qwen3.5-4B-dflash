# 临时 Draft RMSNorm A/B 定位

用于定位“输入完全相同，Draft OM 的有效 context K/V 仍发生漂移”。
使用已有 replay 保存的真实 features 和官方 Draft 权重，构建六个小图；
不加载整个 Target/Draft，不修改主模型，也不运行 msprof。
调试结束后可移除此目录及对应测试；C++ 执行部分复用 `tools/debug_gdr/`。

最初，用户的 `private.json` 离线分析显示，12 个 K/V 输出的有效区
`[0,17)` 在 isolated 后续 19 轮均有变化，写入 padding `[17,64)` 和尾部
`[64,576)` 保持稳定。第 0 层的 K/V 就出现小幅数值变化，因此优先检查
所有 context K/V 共同经过的 FC、hidden_norm，以及后续投影。
“改成 adn_rms_norm 后才出现”曾是回归线索；下面的最新对照已经将排查重点移到 FC。

## 比较内容

| Case | 固定输入 | 输出 |
|---|---|---|
| `fc` | 已保存 features，按原图清零无效行 | FC 输出 |
| `norm_tensor` | 冻结一次 native FC 输出 | 原 Tensor 公式的归一化输出 |
| `norm_adn` | **同一份**冻结 FC 输出 | 当前 adn_rms_norm 包装的归一化输出 |
| `vproj` | 冻结一次原 Tensor 公式的归一化输出 | 第 0 层 V 投影 |
| `chain_tensor` | 已保存 features | FC、原归一化、V 投影三个中间结果 |
| `chain_adn` | 同一份 features | FC、新归一化、V 投影三个中间结果 |

所有 case 分别做 native / OM 重放。每次恢复固定输入，不把上一轮输出喂回。
FC 和 V MatMul 保持 FP16 输入与权重。使用的三个 checkpoint tensor 是：

- `fc.weight`
- `hidden_norm.weight`
- `layers.0.self_attn.v_proj.weight`

原公式使用 FP32 计算 `x * rsqrt(mean(x*x) + eps)`，然后转回 FP16、乘 gamma。
新公式按当前代码向 `adn_rms_norm` 传 FP32 x 和全 1 FP32 gamma，取第一个输出，
然后在相同位置转回 FP16、乘真实 gamma。这里 gamma 就是 checkpoint 权重，
不做 `1 + gamma`。A/B 保留这两个精度转换边界。

## 运行

先更新源码，使用正常导出 OM 时的 Python、CANN 和自定义算子环境。
`DRAFT_REPLAY_REPORT` 指向之前保存完整输入的 `private.json` 或 `shared.json`，
不要指向 `*-kv-analysis.json` 或 shared/private 汇总文件。

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/debug_draft_context/run.py" all \
  --run-dir "$AI_RUN_DIR" \
  --replay-report "$DRAFT_REPLAY_REPORT" \
  --factory-config "$AI_RUN_DIR/factory.json" \
  --device-id 0 --repetitions 20
```

脚本从原 `factory.json` 读取 `draft_dir` 和 `adn_rms_norm_ge_op_type`。
如果配置保存在别处，传实际配置路径。也可省略 `--factory-config`，显式传
`--draft-dir "$DRAFT_DIR"`，并让 `--rms-ge-op-type` 与原导出配置一致
（默认 `RmsNorm`，另一个可选值是 `AdnRmsNorm`）。
若 CMake 无法定位已激活的 CANN，再加 `--ascendcl-root "$CANN_ROOT"`。

只需这一条命令：自动做 native 重放、导出六个小 AIR、ATC 编译、构建临时 C++
runner、逐个加载 OM 重放。**不需要重编完整 Target/Draft OM 或主 C++ runner。**
导出 Python 进程结束后才运行 OM，每次只加载一个小图。

输出位于新建的 `$AI_RUN_DIR/debug-draft-context-*/`：

- `summary.json`：每个输出的 native/OM 不一致轮数、误差和新旧归一化对比。
- `native.json`、`om-results/*/report.json`：逐轮有效区 SHA256。
- `*-reference.bin`、`*-first-diff.bin`：参考与首个不同的完整输出，后续轮不会覆盖首个差异。
- `request.json`、`air.json`、`om.json`：源码/输入/官方 checkpoint/AIR/OM 哈希及编译参数。
- `logs/`：导出、编译、构建和运行日志。

20 次调用全部检查，第 0 次是参考，最多有 19 次与参考不同；没有跳过 warmup。
只用有效行判断重复性，数值误差坐标是 `B,S,F`。
退出 0 表示小图有效输出稳定；退出 1 表示观察到变化；退出 2 表示流程错误，查看日志。
跨公式或 native/OM 的固定数值差异单独报告，不作为“重复运行漂移”。

## 如何收窄范围

| 结果 | 下一步 |
|---|---|
| `fc` 重复不稳定 | 优先检查 FC 的 MatMul 与对应编译路径 |
| 同一冻结输入下 `norm_tensor` 稳定、`norm_adn` 不稳定 | 优先检查 adn_rms_norm 的实现、GE 转换及输出缓冲区 |
| 两个单独 norm 都稳定，只有 `chain_adn` 漂移 | 检查组合图的融合、调度和中间缓冲区；还不能只归因于 norm kernel |
| `vproj` 对固定输入不稳定 | 检查 V 投影的 MatMul |
| 六个小图都稳定 | 继续在原 Draft 图内部加边界输出/做算子定位，原故障仍未关闭 |

AIR 检查要求 adn 分支恰好包含一个对应 norm 节点、原 Tensor 分支不包含该节点。
这不等于已经确认最终 OM 的 kernel：ATC 后续仍可能融合 Tensor 公式，需要查看
编译结果才能确定具体 kernel。输出中间结果也会改变原图的融合和内存生命周期；
小图全部稳定不能排除完整 Draft OM 内的竞态或未初始化数据。

本工具不宣称修复已完成，不放宽原 profiling/greedy 一致性检查，计时也不作为正式性能证据。

## 最新结果：漂移在 FC 中独立复现

用户返回的真实设备报告（`cpu_fallback=false`，20 次调用，以第 0 次为参考）：

| 固定输入的小图 | Native 不一致轮数 | OM 不一致轮数 |
|---|---:|---:|
| FC | 19 | 19 |
| 新 adn RMSNorm | 0 | 0 |
| 原 Tensor RMSNorm | 0 | 0 |
| 第 0 层 V 投影 | 0 | 0 |
| FC → 新 norm → V，三个输出各自 | 18 | 19 |
| FC → 原 norm → V，三个输出各自 | 19 | 19 |

新旧 norm 对相同冻结输入的有效输出逐位相同。原 norm 组合链也从 FC 开始变化，
因此这次复现不需要 adn RMSNorm，也不需要 msprof、共享工作区或整个 Draft 图。
先前保留的“norm 算子造成漂移”假设优先级下降。

FC 使用 FP16 `[64,20480] × [20480,2560]`。保存的首个 FC 差异样本只有少量
元素变化，最大为 1 个 FP16 ULP；这不是所有调用的最大误差统计。
浮点累加顺序变化是下一步假设。尚未取得 kernel/tiling 证据，不能断言是
split-K、AtomicAdd 或某个特定实现。小幅 FC 变化经 norm、V 投影传播，与之前
有效 KV 漂移相符；不应把 V 输出变化直接当作 V MatMul 自身不稳定。

## FC 确定性开关对照

设置 `FC_PROBE_DIR` 为上一次六图测试的输出目录，例如用户当前的
`debug-draft-context-fmfxogyf`。复用其中的 AIR、冻结输入及权重来源：

```bash
"$MODEL_PYTHON" -B "$REPO_ROOT/tools/debug_draft_context/fc_determinism.py" all \
  --run-dir "$AI_RUN_DIR" --probe-dir "$FC_PROBE_DIR" \
  --device-id 0 --repetitions 20
```

不重新导出 AIR，不重编完整模型。只加载官方 `fc.weight` 做 native 重放，随后
用同一份 FC AIR 编译两个小 OM。全部结果写到新的 `debug-fc-determinism-*`，
原测试文件保持不变。如果 ATC/CANN 未在环境中声明，补充 `--atc "$ATC_BIN"`
和 `--ascendcl-root "$CANN_ROOT"`。

- Native：分别在新进程、首次 NPU 运算前设置
  `torch.use_deterministic_algorithms(False/True, warn_only=False)`，记录 getter 状态和 PID。
- OM：同一 AIR 分别加 `--deterministic=0`、`--deterministic=1`，每个 OM 在独立 ACL 进程重放。
- FP16 输入、FP16 权重、矩阵乘法公式及 `must_keep_origin_dtype` 保持相同。

ATC 的确定性开关及其对浮点累加顺序的影响见
[官方 ATC 文档](https://www.hiascend.com/document/detail/zh/canncommercial/700/inferapplicationdev/atctool/atlasatc_16_0122.html)。
Native 用法和“同一线程后续切换可能无效”的限制见
[昇腾 PyTorch 文档](https://www.hiascend.com/document/detail/zh/Pytorch/720/apiref/PyTorchNativeapi/ptaoplist_000292.html)。
这也是两个模式分别启动新进程的原因。开关或编译不受当前环境支持时直接报错，
不会静默忽略，也不会改用 CPU。

查看新目录 `summary.json` 的两个 case，分别比较 native 和 OM 的不一致轮数：

| status | 含义 |
|---|---|
| `STABLE_WITH_DETERMINISTIC` | 关闭时至少一条路径复现变化，开启后 native 和 OM 均稳定 |
| `VARIATION_WITH_DETERMINISTIC` | 开启后 native 或 OM 仍有变化，或 OM 出现非有限值 |
| `BASELINE_NOT_REPRODUCED` | 本次关闭时也稳定，不能据此宣称开关修复了问题 |

前两列原始不一致轮数仍需分别看：不能把只修复 native 的结果当作 OM 已修复。
跨开关模式的固定舍入差异单独报告。native getter 只证明设置状态，不证明最终
kernel 身份；稳定的 FC 对照也不代替完整 Draft 重放与 ordinary greedy 一致性验证。
退出 0 表示流程完成且开启模式稳定，退出 1 表示开启模式仍有异常，退出 2 表示流程错误。
确定性模式可能影响性能，主模型是否启用应在完整验证和重新测速后决定。
