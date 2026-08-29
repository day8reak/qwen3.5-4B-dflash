# Qwen3.5-4B DFlash AscendCL C++ 低时延推理

## 结论与边界

目标推理热路径已经改为可选的原生 C++ AscendCL runner：Python 只负责 checkpoint/AIR/OM
构建、一次 tokenizer 和证据封装；从预分配输入到循环生成 token 的 ordinary/DFlash 3+10
全部在一个 C++ 进程、一个已加载 OM 和一个 ACL stream 中完成。

当前无 CANN/310P 的 workspace 已完成两类模拟验证：

- 纯 C++ scheduler：全接受、拒绝纠错、bonus、EOS、静态档位、配对稳定性和 SHA-256；
- 同 API 签名的 fake-ACL：编译真实 `acl_executor.cpp` 和 CLI，执行完整 3+10 并生成 PASS
  JSON。

这证明 C++ 源码、ABI 和控制流能编译/连通，但不证明真实 `ascendcl` 链接、OM 加载、310P
时延或与闭源框架相当。真实结论必须按本文在物理设备上测量。

## 为什么改成 C++，以及它不能单独解决什么

Python pyACL 路线适合先闭合功能，但 decode 热路径会经过 Python 对象、NumPy 数组构造、
Python/C API 边界和多次同步。C++ runner 针对这些 host 开销做了以下处理：

- `aclmdlLoadFromFile` 只调用一次，ordinary 与 DFlash 共用同一 model ID；
- `aclrtMallocHost` 的 pinned host buffer、device buffer、dataset 和 data buffer 只分配一次；
- 每次图调用在同一 stream 上依次排入 H2D、`aclmdlExecuteAsync`、D2H，最后只做一次
  `aclrtSynchronizeStream`；
- 输入/输出 `std::vector` 预留并复用，decode 中不重复创建 ACL 对象；
- ordinary 与 DFlash 的 3 warmup + 10 measured 在同一进程交错执行，降低温度和顺序偏差；
- C++ 自己重新计算 OM SHA-256，hash 不一致时不会加载模型。

官方 AscendCL 文档明确说明异步模型执行后必须同步 stream，异步 Host/Device 拷贝应使用
合适的 Host 内存并同步；runner 按这个生命周期实现：

- [aclmdlExecuteAsync](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0299.html)
- [aclrtMemcpyAsync](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/81RC1alpha002/apiref/appdevgapi/aclcppdevg_03_0106.html)
- [aclmdlLoadFromFile](https://www.hiascend.com/document/detail/en/canncommercial/800/apiref/appdevgapi/aclcppdevg_03_0283.html)

但是当前 OM 仍是“完整前缀重算”的一体图。若 profile 显示绝大部分时间在 NPU 图执行，
而不是 host 调度，那么仅把 Python 换成 C++ 不会消除主要差距。真正追平闭源框架时，后续
很可能需要独立实现 target KV、Gated DeltaNet recurrent/conv state 和 draft cache 的
增量提交/回滚；那是下一阶段，必须另设状态分支准确率门禁，不能混进本次基线。

## 源码结构

```text
targets/ascend310p/runtime/cpp/
├── CMakeLists.txt
├── include/qwen35_dflash/
│   ├── acl_executor.hpp       # AscendCL executor API
│   ├── generation.hpp         # ordinary/DFlash scheduler 与报告结构
│   └── sha256.hpp             # 无外部依赖的 OM 完整性检查
├── src/
│   ├── acl_executor.cpp       # pinned host/device buffer、dataset、stream、OM execute
│   ├── generation.cpp         # prompt token → 完整生成 token 循环
│   ├── main.cpp               # 3+10 paired CLI 和 JSON 报告
│   └── sha256.cpp
└── tests/
    ├── test_generation.cpp    # 纯 C++ 调度测试
    └── fake_acl/              # 只用于 host 编译/集成测试，不参与 target runner
```

Python 控制面位于 `model/qwen35_dflash/ascend310p/cpp_runtime.py`，提供：

- `build-cpp`：在当前 `$AI_RUN_DIR` 构建生产 runner 并运行 host tests；
- `infer-cpp`：对已有 OM 执行 C++ paired prompt→token；
- `run-e2e-cpp`：AIR → OM → C++ ordinary/DFlash 3+10 → summary。

## 1. 建立真实 target session

从 workspace 根目录执行：

```bash
kit/bin/ws context qwen3.5-4b --task runtime --target ascend310p
eval "$(kit/bin/ws session start \
  --model qwen3.5-4b --target ascend310p --task runtime --format shell)"
eval "$(kit/bin/ws env qwen3.5-4b \
  --target ascend310p --format shell --activate)"

"$AI_TARGET_PREFLIGHT" "$AI_MODEL_ROOT" \
  --require-model-python --require-atc --require-device
```

目标 profile 必须提供真实 CANN/AscendCL headers 和 `libascendcl.so`、TorchAir、
`torch_npu`、ATC、物理 310P 与具体 SoC 变体。当前 simulation profile 只能跑 host tests。

## 2. 构建 C++ runner

推荐由框架构建，所有产物和日志自动留在当前 run：

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p build-cpp \
  --build-dir "$AI_RUN_DIR/build/dflash-cpp" \
  --output "$AI_RUN_DIR/out/cpp-runner-build.json"
```

如果 profile 没有导出 `ASCEND_HOME_PATH`、`CANN_HOME` 或 `ASCEND_TOOLKIT_HOME`，显式传入
该 profile 声明的 toolkit root：

```bash
... build-cpp \
  --build-dir "$AI_RUN_DIR/build/dflash-cpp" \
  --ascendcl-root "$DECLARED_ASCENDCL_ROOT" \
  --output "$AI_RUN_DIR/out/cpp-runner-build.json"
```

不要写死个人安装路径，也不要把二进制安装回共享 CANN profile。成功报告会给出：

```text
$AI_RUN_DIR/build/dflash-cpp/qwen35_dflash_acl_runner
$AI_RUN_DIR/log/dflash-cpp-build/configure.log
$AI_RUN_DIR/log/dflash-cpp-build/build.log
$AI_RUN_DIR/log/dflash-cpp-build/host-tests.log
$AI_RUN_DIR/out/cpp-runner-build.json
```

仅验证 host 代码时，可在 simulation profile 下执行：

```bash
cmake -S "$AI_MODEL_ROOT/targets/ascend310p/runtime/cpp" \
  -B "$AI_RUN_DIR/build/dflash-cpp-host" \
  -DQWEN35_DFLASH_BUILD_ACL_RUNNER=OFF \
  -DQWEN35_DFLASH_BUILD_TESTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$AI_RUN_DIR/build/dflash-cpp-host" --parallel
ctest --test-dir "$AI_RUN_DIR/build/dflash-cpp-host" --output-on-failure
```

该命令不产生真实 target runner，也不能用于性能结论。

## 3. 准备 runner 身份配置

在 `$AI_RUN_DIR/input/cpp-runner.json` 保存：

```json
{
  "graph_name": "dflash_recompute",
  "pad_token_id": 0,
  "device_model": "<具体 Atlas 产品与 310P 变体>",
  "cann": "<精确 CANN 版本>",
  "driver": "<精确驱动版本>",
  "firmware": "<精确固件版本>",
  "runtime": "qwen35-dflash-ascendcl-cpp-v1 + <libascendcl identity>"
}
```

这些字段用于审计，不能代替真实设备证据。框架还会要求 strict preflight、C++ 动态链接可
启动、OM hash 一致、`cpu_fallback=false` 和实际 ACL execute 全部成立。

## 4. 推荐：AIR → OM → C++ token 的一键目标流程

准备与普通框架相同的 NPU factory 配置后运行：

```bash
CPP_RUNNER="$AI_RUN_DIR/build/dflash-cpp/qwen35_dflash_acl_runner"

PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p run-e2e-cpp \
  --factory-config "$AI_RUN_DIR/input/dflash-fp16-s256.json" \
  --bundle-dir "$AI_RUN_DIR/out/dflash-ascend310p" \
  --atc "$ASCEND310P_ATC_BIN" \
  --soc-version "$EXACT_SOC_VERSION" \
  --atc-arg=--precision_mode=force_fp16 \
  --runner "$CPP_RUNNER" \
  --runner-config "$AI_RUN_DIR/input/cpp-runner.json" \
  --prompt '请简要介绍昇腾310P。' --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --report-dir "$AI_RUN_DIR/out/performance-cpp"
```

命令在加载两份 4B checkpoint 前检查：

- runner 可执行且 `--help` 能启动，从而尽早发现缺失 `libascendcl.so`；
- runner 配置包含具体设备和 runtime 身份；
- 具体 ATC SoC、可执行 ATC、strict target preflight；
- TorchAir、`torch_npu` 和显式 `npu:*` factory device；
- bundle/report 目录均为当前 run 的新目录且互不重叠。

然后按顺序执行 AIR、OM、一次 tokenizer、C++ paired 3+10、零 mismatch 和文本解码。成功
输出：

```text
$AI_RUN_DIR/out/performance-cpp/runner-raw.json
$AI_RUN_DIR/out/performance-cpp/summary.json
$AI_RUN_DIR/log/dflash-cpp-runner.log
```

`summary.json` 同时保留 runner/OM/AIR/preflight hash、普通与 DFlash 的十次原始值、接受率、
graph call 数、生成 token/text 和 C++/closed 对比所需身份。

## 5. 已有 OM 时只运行 C++ 推理

```bash
PYTHONPATH="$AI_MODEL_ROOT/model" \
  "$AI_MODEL_PYTHON" -m qwen35_dflash.ascend310p infer-cpp \
  --deployment-manifest \
    "$AI_RUN_DIR/out/dflash-ascend310p/deployment-manifest.json" \
  --runner "$AI_RUN_DIR/build/dflash-cpp/qwen35_dflash_acl_runner" \
  --runner-config "$AI_RUN_DIR/input/cpp-runner.json" \
  --prompt '请简要介绍昇腾310P。' --chat \
  --max-new-tokens 32 --max-draft-tokens 15 --device-id 0 \
  --output "$AI_RUN_DIR/out/performance-cpp/summary.json"
```

Python 在 runner 启动前完成 tokenizer，C++ 接收精确 `prompt_token_ids` 和 EOS 集合；模型
加载后，warmup 和 measured 的生成循环不再回到 Python。

## 6. C++ 延迟口径

C++ 报告的主性能口径是 pretokenized、model-load-excluded 的同步 model loop：

- `prefill`：第一次 OM 调用，得到第一个普通 target token；
- `decode`：之后所有 proposal/verify OM 调用和 C++ 接受/纠错调度；
- `model_total`：`prefill + decode`；
- `generated_tokens_per_second`：十次生成 token 总数 / 十次 `model_total` 总和；
- `acceptance_rate`：接受的 draft token / 实际 drafted token；
- `startup_ms.acl_and_model_load`：单独报告，不进入 model loop；
- Python 的 tokenize/detokenize 只各记录一次，明确不拼成伪造的 10 次 service latency。

ordinary 与 DFlash 每种固定 3 warmup + 10 measured，测量顺序交错。每个 `Execute` 在一个
stream 中排入两次 H2D、一次 OM execute、两次 D2H，再同步一次；因此 host 计时覆盖提交、
设备执行和取回 Top1 的完整可见成本。

## 7. 与闭源框架做同口径 A/B

“差不多快”不能只比较两张日志里的平均值。先让闭源框架输出
`targets/ascend310p/abi/closed-runtime-baseline-v1.json` 规定的字段，至少包括：

- 同一物理设备、device ID、CANN、driver、firmware；
- 同一 target/draft checkpoint manifest、FP16、静态 `S`；
- 同一 tokenizer 后的 prompt token IDs；
- 同一生成 token IDs 和 stop reason；
- concurrency=1、排除模型加载、显式设备同步；
- 至少 3 warmup + 10 个原始 `model_total` 值。

在测试前冻结“相当”的 median/p90 比率。例如产品要求 C++ 不得慢于闭源框架的某个比率，
把它显式传给比较器；仓库不会替你发明阈值：

```bash
"$AI_MODEL_PYTHON" \
  "$AI_MODEL_ROOT/targets/ascend310p/scripts/compare_cpp_closed_runtime.py" \
  --cpp-report "$AI_RUN_DIR/out/performance-cpp/summary.json" \
  --closed-report "$AI_RUN_DIR/input/closed-runtime-baseline.json" \
  --max-median-ratio "$APPROVED_MEDIAN_RATIO" \
  --max-p90-ratio "$APPROVED_P90_RATIO" \
  --output "$AI_RUN_DIR/out/cpp-vs-closed.json"
```

比较器会重新读取并验证 C++ AIR manifest hash，比较模型、档位、设备、prompt/token/EOS、
CANN/driver/firmware，再从原始值重算 median/p90。任一身份或 token 不一致，即使 C++ 更快
也是 FAIL。

## 8. 性能差距的下一步决策

先采集 `summary.json`、闭源 A/B 和同一 OM 的 profiler，再根据测量结果决定：

| 证据 | 结论 | 下一步 |
| --- | --- | --- |
| C++ 比 Python 明显快，且达到闭源阈值 | host 热路径已闭合 | 固化 runner/profile，扩大 prompt 集 |
| C++ 比 Python快，但仍显著慢于闭源；NPU execute 占主导 | 重计算图是主要瓶颈 | 单独设计增量 KV/DeltaNet/draft state ABI 与回滚门禁 |
| C++ 与 Python接近，host 等待/拷贝占比低 | 语言不是主要瓶颈 | 优化图、layout、算子和重算次数 |
| H2D/D2H 或 stream idle 占比高 | runtime/调度仍有空间 | 检查输出裁剪、异步流水、内存复用与线程绑定 |
| token mismatch 或输出不稳定 | 性能结果无效 | 回到 ordinary authority、anchor/padding/EOS 门禁 |

增量状态优化必须保持普通 target 权威、最终 token-ID mismatch=0，并覆盖每个接受/拒绝位置
的 KV、full-attention 和 Gated DeltaNet recurrent/conv state 分支。未通过这些门禁前，不能
用更快的错误输出替代当前重计算基线。

## 9. 当前仍未完成的目标证据

当前 profile 无真实 CANN headers/library、TorchAir、ATC、AscendCL runtime 或物理 310P，
所以尚未完成：

- 生产 runner 对真实 `libascendcl.so` 的编译/动态链接；
- 真实 AIR/OM；
- `aclmdlLoadFromFile` 和 `aclmdlExecuteAsync` 的物理设备调用；
- C++ vs Python、C++ vs 闭源框架时延；
- 是否需要增量 cache/state 的测量结论。

在这些证据回来前，准确表述是“C++ 低开销推理框架和 host/fake-ACL 验证已完成，真实 310P
功能与性能待测”，不是“已经达到闭源框架时延”。
