# GDR 单算子临时调试

本目录定位 Ascend310P 上 GDR 在 OM 中耗时异常的问题。所有代码都在本目录，删除本目录即可撤掉调试入口。代码复用仓库的 GDR Fake/Meta、GE converter、AIR 输入顺序校验和 msprof 动态控制器。OM 由独立的 AscendCL C++ 程序执行，不需要 pyACL。

## 1. 准备环境

在 `feature/gdr-chunk-verify` 分支的仓库根目录执行。使用已经能执行 GDR、导出 AIR 和编译 OM 的 CANN / torch_npu / TorchAir 环境，并保留相同的自定义算子环境变量。需要 ATC、CMake、C++17 编译器；采集时还需要 msprof。

`MODEL_PYTHON` 指模型 Python，`AI_RUN_DIR` 指源码目录外的运行目录。每次完整实验使用一个新子目录：

```bash
export GDR_DEBUG_DIR="$AI_RUN_DIR/debug-gdr-$(date +%Y%m%d-%H%M%S)"
"$MODEL_PYTHON" -B tools/debug_gdr/run.py --help
```

默认不读取模型权重，只生成固定随机种子的有限输入。Q/K/V 是 FP16 `[1,16,32,128]`，g 是 FP32，beta 是 FP16，初始 state 是非零 FP32 `[1,32,128,128]`。真实输入回放见第 6 步。

## 2. 导出、编译并执行对照

```bash
"$MODEL_PYTHON" -B tools/debug_gdr/run.py all \
  --work-dir "$GDR_DEBUG_DIR" \
  --device-id 0 --soc-version Ascend310P3 \
  --lengths 16,8 --warmup 3 --repetitions 10
```

需要显式指定工具链时，给 `all` 或 `compile` 加上：

```text
--atc /absolute/path/atc --ascendcl-root /absolute/path/cann-toolkit
```

`ascendcl-root` 应包含 `include/acl/acl.h` 和 AscendCL 库。SoC 必须与设备一致。

实验导出 **3 个临时 OM**，不替换模型的 4 个生产 OM：

| 实验 | GDR 计算属性 | 图输出 |
|---|---|---|
| `both` | chunk_size=64，两个 bool 属性均为 True | core_attn + FP32 state |
| `core` | 同上 | 仅 core_attn |
| `state` | 同上 | 仅 FP32 state |

有效长度是 `INT16[1]` 运行时输入。同一个 OM 文件分别运行长度 16 和 8，不为每个长度重新编译。三种图都只含一个 GDR；`core`/`state` 测试输出是否被消费者使用，不把 `output_final_state` 改成 False。`state` 直接输出 FP32，暂不包含完整模型在 GDR 后的 FP16 转换。

每个长度先运行 torch_npu 的双输出基准，再运行三种 OM。默认每种情况预热 3 次、记录 10 次。输入搬运、模型加载、输出下载、哈希及数值比较都在计时范围之外。C++ 使用持久缓冲区；torch_npu 的调用时间包含输出分配，因此精确的算子时间以 msprof 对照为准。

每次执行都恢复同一份输入，绝不把上一轮输出 state 喂给下一轮；执行后检测输入是否被修改。输出稳定性按有效区域计算，core_attn 超过有效长度的尾部不参与数值比较。

## 3. 查看结果

```bash
cat "$GDR_DEBUG_DIR/summary.md"
```

主要文件：

| 文件 | 内容 |
|---|---|
| `inputs.json` | 输入来源、属性、形状、dtype、哈希和待测长度 |
| `air.json` | 三个 AIR 的哈希、GE 节点数和 7 个运行时输入的顺序审计 |
| `om.json` | 三个 OM 的哈希、ATC 命令和当前 vendor 文件清单 |
| `summary.md` / `summary.json` | 各组时延、稳定性、与 native 输出的差异 |
| `measurements/*/report.json` | 每次调用的时延、有效输出哈希；C++ 还记录实际 OM I/O 描述和缓冲区地址 |
| `measurements/*/*.bin` | 最后一次测量的输出，按逻辑大小保存 |
| `logs/` | AIR 导出、ATC、C++ 构建和执行日志 |

`exact_equal`、`max_abs_diff`、`rmse` 是诊断观察，不设置放宽的正确性阈值。`stable=false` 或有效区域出现 NaN/Inf 时，先定位这些问题。未使用的 core 尾部可能未定义；完整文件哈希会保存，但不会因此误报有效输出不一致。

vendor 文件清单只是当前可见文件的哈希，不证明实际选中了哪份 `.o` 或 TilingKey。C++ 记录的是单算子 OM 的边界缓冲区，不是完整模型内部的 workspace 地址。

结果的下一步：

- 同一 OM 长度 16 慢、8 快：检查有效长度影响的分支、尾块和搬运次数。
- 只有 `core` 慢：检查未使用 state 输出相关的编译及写回处理。
- 三个 OM 都慢、native 快：比较实际 kernel 二进制和 tiling 数据。
- 单算子全部快：尚未复现完整模型的问题，继续用真实输入或保留周边图依赖。不能据此认定完整 OM 正常。

## 4. 追加 output_final_state=False 对照

完成第 2 步后，可以在同一个测试目录追加 `core_no_state`。它和 `core` 都只输出 core_attn，计算属性仅将 `output_final_state` 从 True 改成 False；`initial_state` 仍然传入，chunk_size=64，有效长度仍为 16 / 8。

```bash
"$MODEL_PYTHON" -B tools/debug_gdr/run.py no-state \
  --work-dir "$GDR_DEBUG_DIR" \
  --device-id 0 --soc-version Ascend310P3 \
  --warmup 3 --repetitions 10 --profile
```

命令会打印新目录 `$GDR_DEBUG_DIR/outstate0-<timestamp>-<pid>/`，并执行以下步骤：

1. 复制并校验原来的固定输入，按哈希复用三个基准 OM。
2. 只导出并编译一个新增的 `core_no_state.om`。
3. 使用同一份输入重新测量基准和新实验，共 12 行结果：每个长度包含 native 双输出、native False core，以及四种 OM。
4. 加 `--profile` 时，只对新增的 native / OM False 情况分别采集长度 16 / 8，共 4 个独立窗口，每个窗口一个 GDR。

新目录的 `summary.md` 重点对比 `om-core-L16` 与 `om-core_no_state-L16`，以及对应的 L8 行。`Exact vs native` 始终与同长度的 native **True 双输出**基准比较有效 core，不能用 False 自己当正确性基准。False 的 state 值不下载、不哈希、不比较，因为调用方没有请求它。

`experiment.json` 记录属性变化和基准产物哈希；`air.json` 记录新图实际传入的属性，以及保存前对 GE 节点的校验：`output_final_state=False`、chunk_size=64、state 无消费者。输入清单保留基准属性记录，各实验的实际属性以 graph / case 报告为准。

这个实验使用本目录内的独立导出接口，处理生产 Fake/Meta 只接受 True 的限制；底层仍调用已安装的 `ChunkGatedDeltaRule`，不修改 kernel。GE 算子定义仍有两个输出槽，所以结构图或 msprof 仍可能列出 state 的描述，不能据此认定 False 没有生效。使用 False 后 kernel 是否跳过写回，需要看实测时延和 MTE3 指标。

原测试目录已有的 AIR、OM、报告和 profiling 数据保持原样；再次运行会创建另一个新子目录。

如果 False 的 core 数值一致、时延恢复，则第一遍 verify 关闭 final state 输出有了单算子依据。第二遍 commit、prefill 和普通 decode 需要保存状态，仍应输出 state；完整模型修改另做逐轮正确性和性能验证。

## 5. 预热后只采一次 GDR

先完成第 2 步。下面复用同一个已编译 OM 和同一份输入，每条命令只采一次调用：

```bash
"$MODEL_PYTHON" -B tools/debug_gdr/run.py profile \
  --work-dir "$GDR_DEBUG_DIR" --backend om --variant both \
  --length 16 --device-id 0 --warmup 3 --metrics PipeUtilization

"$MODEL_PYTHON" -B tools/debug_gdr/run.py profile \
  --work-dir "$GDR_DEBUG_DIR" --backend om --variant both \
  --length 8 --device-id 0 --warmup 3 --metrics PipeUtilization

"$MODEL_PYTHON" -B tools/debug_gdr/run.py profile \
  --work-dir "$GDR_DEBUG_DIR" --backend native --variant both \
  --length 16 --device-id 0 --warmup 3 --metrics PipeUtilization
```

`--variant core` / `--variant state` 选择另外两个 OM。`--metrics Memory` 或 `MemoryUB` 采集对应指标。需要时指定 `--msprof-bin /absolute/path/msprof`、`--timeout 600`。

对 False 实验单独追加采集时，`--work-dir` 指向第 4 步打印的 `outstate0-...` 子目录，使用 `--variant core_no_state`。

若要在首次完整实验中一次完成 8 组单算子采集，给第 2 步的 `all` 命令加 `--profile`。这是 8 个独立窗口，每个窗口一个 GDR，不把不同实验的任务时长相加。

每次采集保存在新的 `profiles/<backend>-<variant>-L<length>-<metrics>-<timestamp>/` 中：

- `control.json`：动态 start / stop / quit 的确认记录。
- `gdr-rows.json`：每份导出 CSV 中的 GDR 原始行，保留 MTE3 等全部原始列及单次 Task Duration。每份 CSV 必须恰好有一个 GDR。
- `operators/hotspots.txt`：整个采集窗口的算子列表。
- `capture/`：msprof 原始产物。
- `result/`：这一次执行的输出及校验报告。

控制器日志中的 `stage=verify` 是复用协议的标签，这里只执行一个独立 GDR，不执行完整 target verify。预热和 H2D 在 start 确认前完成，D2H、哈希和输入修改检查在 stop/quit 确认后执行。msprof 耗时与未开启采集的基准分别保存。

## 6. 捕获真实 torch_npu 输入并回放

捕获入口只在当前进程包装 `torch_npu.npu_chunk_gated_delta_rule`，不会修改模型文件。默认保存第一个形状匹配 `[1,16,32,128]` 的调用，包含 7 个输入张量和调用栈。

使用第 1 步的环境，同时准备模型推理所需的 `TARGET_DIR`、`DRAFT_DIR`、`QUANT_CONFIG`、`MAX_SEQUENCE_LENGTH`、`prompt-ids.json` 和外部加载器配置。下面在抓到输入后主动停止该次调试进程，不生成完整推理结果：

```bash
export GDR_CAPTURE_DIR="$AI_RUN_DIR/gdr-inputs-$(date +%Y%m%d-%H%M%S)"
"$MODEL_PYTHON" -B tools/debug_gdr/run.py capture \
  --work-dir "$GDR_CAPTURE_DIR" --device-id 0 --capture-index 0 \
  --stop-after-capture --module models.dflash_v1.run_npu -- \
  --target-dir "$TARGET_DIR" --draft-dir "$DRAFT_DIR" \
  --config "$QUANT_CONFIG" --quant_mode enable \
  --kv-cache-max-len "$MAX_SEQUENCE_LENGTH" --device npu:0 \
  --prompt-json "$AI_RUN_DIR/prompt-ids.json" \
  --block-size 16 --max-new-tokens 32 --execution-mode dflash
```

`--capture-index` 从 0 开始，只计数形状匹配的 GDR 调用。若 prompt 恰为 16 行，第一个匹配调用可能属于 prefill；根据 `capture.json` 的调用栈确认，选择相应索引重跑。也可用该参数选择另一层或 commit 调用。去掉 `--stop-after-capture` 会继续完成推理。

再用一个新目录执行回放：

```bash
export GDR_DEBUG_DIR="$AI_RUN_DIR/debug-gdr-real-$(date +%Y%m%d-%H%M%S)"
"$MODEL_PYTHON" -B tools/debug_gdr/run.py all \
  --work-dir "$GDR_DEBUG_DIR" --inputs "$GDR_CAPTURE_DIR/inputs.npz" \
  --device-id 0 --soc-version Ascend310P3 --lengths 16,8 \
  --warmup 3 --repetitions 10
```

捕获的是 native 输入值，不包含 OM 的内部地址或 tiling；改变回放有效长度是在固定数据上做受控实验。

## 7. 分步执行和撤掉调试代码

可以分别执行 `prepare`、`export`、`compile`、`benchmark`、`summarize`。各命令通过 `--work-dir` 指向同一目录。为保留证据，prepare/export/compile/benchmark 不覆盖已有对应产物；失败日志也会保留。需要完整重试时创建新目录，单次 profiling 可以在已有目录中反复追加。

定位完成后，用 Git 撤销仅新增本目录的调试提交，或删除 `tools/debug_gdr/` 后提交。运行目录里的输入、AIR/OM、采集结果和日志独立保存，删除源码不会删除这些证据。

主机检查命令：

```bash
"$MODEL_PYTHON" -B -m pytest tools/debug_gdr/test_debug_gdr.py -q -p no:cacheprovider \
  --basetemp "$AI_RUN_DIR/gdr-host-tests"
```

主机测试使用模拟 ACL 检查 ABI、输出有效区域、输入修改检测、重复稳定性和采集握手，不能证明真实 GDR 数值或设备性能。
