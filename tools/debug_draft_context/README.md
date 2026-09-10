# 临时 Draft RMSNorm A/B 定位

用于定位“输入完全相同，Draft OM 的有效 context K/V 仍发生漂移”。
使用已有 replay 保存的真实 features 和官方 Draft 权重，构建六个小图；
不加载整个 Target/Draft，不修改主模型，也不运行 msprof。
调试结束后可移除此目录及对应测试；C++ 执行部分复用 `tools/debug_gdr/`。

当前证据：用户的 `private.json` 离线分析显示，12 个 K/V 输出的有效区
`[0,17)` 在 isolated 后续 19 轮均有变化，写入 padding `[17,64)` 和尾部
`[64,576)` 保持稳定。第 0 层的 K/V 就出现小幅数值变化，因此优先检查
所有 context K/V 共同经过的 FC、hidden_norm，以及后续投影。
“改成 adn_rms_norm 后才出现”是重要的回归线索，尚不能据此确定具体 kernel。

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
