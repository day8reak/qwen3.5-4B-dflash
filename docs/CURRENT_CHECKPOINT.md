# Current checkpoint

- Run: `20260829T062557Z-8879-f4312c`
- Target/profile: `ascend310p / ascend310p-local (simulation-only)`
- Active model phase: `target-plan`
- Regression: Python `82 passed, 1 skipped, 7 warnings, 3 subtests passed`；C++/fake-ACL
  `2/2 CTest passed`。
- Hardware-reference policy: `use`; matching measured 310P portrait is currently absent。
- Precision: FP16 仍只获准进入 target 验证，不能由 CPU smoke 直接晋级。

## Phase state

- `contract`: PASS；text-only batch-1 strict greedy，最终 token mismatch 必须为零。
- `reference`: PASS；官方 reduced-source parity、锁定真实 checkpoint FP16 一体图 CPU smoke
  和完整回归已形成可复现的非 target reference。
- `target-plan`: IN PROGRESS；图/ABI、pyACL 参考路径、AscendCL C++ 低时延路径、timing/gate
  和闭源 A/B contract 已定义，但缺少匹配的硬件画像、真实 CANN profile 和 310P。
- `adaptation`、`integration`、`accuracy`、`performance`、`release`: PENDING。

## Preferred C++ target workflow

`run-e2e-cpp` 把以下步骤收敛为一个 fail-fast 命令：

1. 在加载两份 4B checkpoint 前验证具体 ATC SoC、ATC executable，执行 workspace 声明的
   strict ATC/device preflight，再验证 TorchAir；内置图要求 `torch_npu` 和显式 NPU
   device，runner 必须能在当前 AscendCL 动态环境启动；
2. 导出 hash-complete AIR 和外置权重；
3. ATC `framework=1` 编译 OM；
4. C++ 一次加载 OM、预分配 pinned host/device buffer、dataset 和单 stream；
5. 在同一 C++ 进程中交错执行 ordinary 与 DFlash 各 3 warmup + 10 measured；
6. 对 OM、设备/runtime、tokenizer、prompt、token IDs、EOS 和稳定性做门禁；
7. 仅在全部通过后写入 `runner-raw.json` 和 `summary.json`。

最终 summary 包含 prompt 输出、AIR/OM/runner/report hashes、ordinary/DFlash 的十次原始
prefill/decode/model-total、接受率、graph call 数和 median 比值。`run-e2e` + pyACL 仍作为
功能参考。构建、运行和闭源 A/B 见 `docs/DFLASH_ASCEND310P_CPP_RUNTIME.md`。

## C++ host evidence

- 原生 scheduler 覆盖 ordinary、DFlash 全接受/bonus、拒绝 correction、EOS、档位与
  SHA-256；
- 生产 `acl_executor.cpp`/CLI 用同 API 签名 fake-ACL 编译链接，完整 paired 3+10 输出
  JSON 并通过零 mismatch；
- Debug ASan/UBSan 构建的 scheduler 与 fake-ACL runner `2/2` 通过；
- Python 控制面新增 `build-cpp`、`infer-cpp`、`run-e2e-cpp`，以及显式阈值的同设备闭源
  median/p90 比较器；
- 这些都是 host/simulation evidence，不是 `libascendcl.so` 真机或性能证据。

## Locked real-checkpoint CPU evidence

上一封存 run 已用官方 target 与 DFlash 权重完成 FP16、`S=2` 一体图 smoke：

- evidence: `.work/qwen3.5-4b/20260829T041726Z-11379-ad68cc/out/reference/integrated-fp16-s2.json`
- input token IDs: `[1, 2]`
- ordinary next token: `220`
- target/draft shapes: `[1,2]` / `[1,15]`
- draft token IDs: `[17,17,25,6,6,6,11,11,11,11,11,11,11,11,11]`
- checkpoint load / forward: `326876.998751 ms` / `231677.447444 ms`

报告包含 `cpu_fallback=true`；这些时长不是 310P total、prefill 或 decode latency。

## Current target evidence boundary

当前 profile manifest hash 与 `specs/environment.lock.json` 一致，但结果没有变化：

- 声明 preflight：ATC、CAModel、310P device 均 `PENDING`；strict 在
  `--require-atc` 处失败；
- 直接探测：无 ATC、`npu-smi`、`msprof`，无 TorchAir、`torch_npu`、AscendCL/pyACL，
  无 `/dev/davinci_manager`、`/dev/devmm_svm`、`/dev/davinci0`；
- scoped knowledge search：匹配的 measured Ascend 310P capability portrait 数量为零；
- `run-e2e`/`run-e2e-cpp` target readiness：在 checkpoint load、AIR write、OM write 前
  因缺 ATC/AscendCL 返回
  `PENDING`。

因此仍然没有真实 AIR、OM、device output 或 310P latency，不能声称 integration、accuracy
或 performance PASS。

当前 C++ runtime run 证据：

- `.work/qwen3.5-4b/20260829T062557Z-8879-f4312c/log/python-tests-cpp-final.log`
- `.work/qwen3.5-4b/20260829T062557Z-8879-f4312c/log/cpp-cmake-final.log`
- `.work/qwen3.5-4b/20260829T062557Z-8879-f4312c/log/cpp-build-final.log`
- `.work/qwen3.5-4b/20260829T062557Z-8879-f4312c/log/cpp-tests-final.log`
- `.work/qwen3.5-4b/20260829T062557Z-8879-f4312c/log/cpp-sanitize-build.log`
- `.work/qwen3.5-4b/20260829T062557Z-8879-f4312c/log/cpp-sanitize-tests.log`

前一 target readiness run 证据：

- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/unit-tests-final.log`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/ascend310p-preflight-capabilities.log`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/ascend310p-preflight-strict.log`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/ascend310p-direct-capability-probe.log`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/ascend310p-knowledge-search.json`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/dflash-run-e2e-preflight.log`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/run-e2e-readiness.json`
- `.work/qwen3.5-4b/20260829T050816Z-22260-6771fe/log/dflash-e2e-source-sha256.txt`

## Exact next action

切换到 manifest-locked、同时暴露 TorchAir、`torch_npu`、具体 ATC、AscendCL headers/
library 和物理 310P 的 profile；采集匹配的 capability portrait，执行 `build-cpp` 和
`run-e2e-cpp`。随后用同设备、同模型/档位/token 的闭源报告和预先批准的 median/p90 比率
运行比较器。真实 `summary.json` 与 `cpp-vs-closed.json` 均 PASS 之前，目标保持未完成。
