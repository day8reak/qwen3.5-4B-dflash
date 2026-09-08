# AIR/OM/C++ 接口参考

从环境准备开始的完整命令见 [AIR/OM/C++ 从零部署](GDR_CHUNK_AIR_OM.md)。本文集中说明
配置、产物和运行时边界，供执行步骤时查阅。范围为 batch=1、W8A8 Target、FP16 Draft、
strict greedy 的显式状态增量图。

## 1. 处理顺序

```text
Target + Draft checkpoint + W8A8 输入 + receiver 加载器
    → 输入 manifest + factory.json
    → TorchAir：四个 AIR 图及外置权重
    → ATC：四个 OM + deployment-manifest.json
    → C++ AscendCL：普通/DFlash 生成
    → NPU ordinary token 对照
    → 时延报告和分阶段 msprof
```

DFlash 运行加载 `target_prefill`、`target_verify`、`draft`；普通运行加载
`target_prefill`、`target_decode`。prefill 共用，verify 内部完成接受判断与状态提交。

## 2. 输入和导出配置

量化 YAML 必须包含三个绝对路径：

| 字段 | 内容 |
|---|---|
| `quanted_pth` | Target W8A8 Linear 的 `data*.safetensors` 目录 |
| `embedding_weight_path` | INT8 Target embedding raw binary |
| `embedding_scale_path` | 每词表行的 FP32 scale raw binary |

只量化 Target Linear 和输入 embedding。Draft 使用 FP16 embedding、LM head 和主体。
配置和量化输入须来自同一 Target；加载器检查 QLinear 结构和 buffer，拒绝不完整转换。

增量工厂是 `qwen35_dflash.ascend310p.quant_factory:create_quant_incremental_graphs`。
在导出命令中显式指定它，并使用以下 `factory.json` 字段：

| 字段 | 要求 |
|---|---|
| `target_dir` | 完整 Target checkpoint/tokenizer 目录 |
| `draft_dir` | 官方 Qwen3.5-4B-DFlash checkpoint 目录 |
| `quant_config` | 上述三字段 YAML |
| `input_manifest` | `lock_quant_inputs.py` 生成的输入 manifest |
| `receiver_models_dir` | 含 `export_model_wrapper_qwen3_5.py` 及其依赖的外部 models 目录 |
| `max_sequence_length` | 逻辑 KV 容量 C，64 的倍数，64..32704 |
| `include_ordinary_decode` | `true` 导出 4 图；`false` 只导出 DFlash 3 图 |
| `dtype` | `float16` |
| `device` | 例如 `npu:0` |
| `adn_rms_norm_ge_op_type` | 默认 `RmsNorm`，必须是目标环境注册的 GE type |

外部输入不能是 symlink；输入 manifest 冻结后不得修改对应内容。导出器在加载模型前检查
输入 hash 和仓库 `SOURCE_LOCK.json`。物理 KV 容量为 C+64，额外行作为 scratch；
请求须满足 `prompt_tokens + max_new_tokens <= C`。

## 3. OM 有序输入/输出

ABI 标识为 `qwen35-dflash-chunk-v1`，合同见
[图与状态合同](../framework/abi/dflash-chunk-v1.json)。所有图固定 batch=1。
实际 tensor 顺序、dtype、shape 由加载后的模型推导，并冻结在 manifest 的
`metadata.tensor_abi` 中。不要通过文件名猜测输入顺序。

### 3.1 Target 图

三个 Target 图按以下顺序接收输入：

| 输入 | dtype/shape | 含义 |
|---|---|---|
| `input_ids` | INT64 `[1,R]` | prefill R=64；decode R=1；verify R=16 |
| `start_position` | INT64 `[1]` | 已提交前缀之后的绝对起始位置 |
| `valid_rows` | INT16 `[1]` | 本次物理 R 行中的有效行数，1..R |
| `t0_*` 到 `t31_*` | FP16，依层结构确定 | 每层 conv/recurrent 或 paged key/value 状态 |

输出顺序：

| 图 | 输出 |
|---|---|
| `target_prefill` | `target_top1 INT64[1,1]`、`features FP16[1,64,20480]`、完整 Target 状态 |
| `target_decode` | `target_top1 INT64[1,1]`、完整 Target 状态 |
| `target_verify` | `target_top1 INT64[1,16]`、`accepted_count INT64[1]`、`features FP16[1,64,20480]`、完整 Target 状态 |

prefill/decode Top1 对应本次最后一个有效输入行。verify 输入为 `[anchor,d1,...,dK]`，
右侧补齐至 16 行；`valid_rows=K+1`。接受数 `a` 是从第一个 proposal 起连续匹配的长度，
提交 anchor 加 a 个 proposal，共 `a+1` 行。verify 的有效 committed features 右补齐到 64 行。

一次 verify OM 内部执行：

1. 从本轮初始状态计算全部有效 verify 行，完成第一遍 GDR 和 Target Top1。
2. 比较 proposal 与 Target Top1，得到最长连续接受数 a。
3. 从同一本轮初始 recurrent state，以 `effective_length=a+1` 完成第二遍 GDR。
4. 选择对应 conv state，输出 committed features 和完整 Target 状态。

第二遍 GDR 所需的中间量留在图内，没有独立 commit OM，也不回传主机。
GDR 累加及算子 initial/final state 使用 FP32；OM 之间持久保存的 recurrent state 为 FP16。

### 3.2 Draft 图

| 输入顺序 | dtype/shape | 含义 |
|---|---|---|
| `features` | FP16 `[1,64,20480]` | 本次需追加的 committed Target feature，右补齐 |
| `start_position` | INT64 `[1]` | 这批 feature 在上下文中的起始位置 |
| `valid_rows` | INT16 `[1]` | 有效 feature 行数，1..64 |
| `anchor` | INT64 `[1]` | 当前已输出、尚未作为 Target 输入提交的 token |
| `d0_key,d0_value,...,d5_key,d5_value` | FP16 `[1,8,C+64,128]` | 6 层 committed-context KV |

输出为 `draft_top1 INT64[1,15]`，随后是相同顺序的 12 个更新后 Draft KV。
一次调用完成 feature projection、context KV 追加和 anchor+15 mask 的并行 proposal。
用于 proposal 的 transient block KV 不作为 committed cache 输出。

## 4. 自定义算子导出要求

Target 使用 `npu_chunk_gated_delta_rule`、`adn_fused_infer_attention`、`adn_rms_norm`
等 NPU 算子。每个导出节点都需要匹配的 dispatcher schema、Fake/Meta、TorchAir converter
和 GE 注册。GDR `effective_length` 是本次有效行数，不能填写累计 KV 长度。
fused attention 的 `pse_shift` 接收 `INT64[1] logical_end`，需要 receiver 软件栈支持导出。

`adn_rms_norm` 的导出对应关系为：

```text
torch_npu.adn_rms_norm → npu.adn_rms_norm.default → GE RmsNorm
```

Fake/Meta 声明两个输出：第一个与 input 同 shape/dtype；第二个是
`[*input.shape[:-1],1] FP32`。converter 一对一生成注册的 GE 节点，不执行 RMSNorm Tensor
分解。导出要求 converter 命中，并在 `dynamo.pbtxt` 中找到对应 GE 节点；审计结果写入
`air-manifest.json`，编译 OM 前再次检查。

CPU、fixture 或 fake ACL 检查不证明目标环境的导出和算子数值正确。目标机需完成真实
TorchAir/ATC、自定义算子、AscendCL 和 token 精度检查。

## 5. 产物和加载计划

| 文件 | 内容和用途 |
|---|---|
| `quant-input-manifest.json` | checkpoint、量化输入、YAML、receiver 加载器的内容身份 |
| `factory.json` | 图工厂和设备配置 |
| `air-manifest.json` | AIR payload、外置权重、I/O ABI、源码/输入 hash、自定义算子审计 |
| `deployment-manifest.json` | 编译成功的 OM、SHA-256、字节数、ATC 参数和 tensor ABI |
| `cpp-build.json` | runner 构建日志、二进制 hash 和编译身份 |
| `ordinary-plan.txt` / `dflash-plan.txt` | C++ 加载的 OM、tensor 和状态合同 |
| `runner.json` | 具体设备、CANN、驱动、固件、runtime 和 pad token 配置 |

加载计划由 `prepare-chunk-plan` 生成。C++ 调用须同时传计划路径和该计划的 SHA-256；
runner 还检查所选 OM 的 hash，以及 AscendCL 返回的 dtype、维度和字节数。
AIR 外置权重与 manifest 必须随图一起保留，不能仅复制一个 `.air` 文件。

`runner.json` 必填 `device_model`、`cann`、`driver`、`firmware`、`runtime`；
`device_model` 须填写具体产品和型号，不能只写 `Ascend310P`。
ATC 的 `--soc-version` 同样要求设备支持的精确型号。

## 6. CLI 操作顺序

命令模块为 `qwen35_dflash.ascend310p`，通过 `MODEL_PYTHON -B -m` 调用。
完整路径初始化和可执行命令见[部署手册](GDR_CHUNK_AIR_OM.md)。

| 顺序 | 子命令 | 主要参数 | 结果 |
|---|---|---|---|
| 1 | `export-air` | `--factory`、`--factory-config`、`--bundle-dir` | AIR bundle |
| 2 | `compile-om` | `--air-manifest`、`--atc`、`--soc-version` | OM 与 deployment manifest |
| 3 | `build-cpp` | `--build-dir`、`--ascendcl-root`、`--output` | C++ runner 与构建报告 |
| 4 | `infer-cpp` | `--deployment-manifest`、`--runner`、`--runner-config`、`--model-dir`、`--prompt`、`--chat`、`--output` | paired 生成和文本报告 |
| 5 | `prepare-chunk-plan` | `--deployment-manifest`、`--mode ordinary\|dflash\|paired`、`--output` | 单模式或配对计划 |

`infer-cpp` 还支持 `--max-new-tokens`、`--max-draft-tokens`、`--device-id`。
Python 处理 tokenizer、文本和报告；生成热循环在 C++ 内执行。
直接调用 C++ 时使用 `--model-kind chunk --mode ordinary|dflash|paired`，
配合 `--model`、`--model-sha256`、`--prompt-token-ids`、`--eos-token-ids` 和 `--output`。

需要组合步骤时，`build-om` 合并 AIR 导出和 ATC 编译；`run-e2e-cpp` 合并导出、编译和
paired 推理，runner 须已构建。两者都应显式传入本参考第 2 节的增量 `--factory`。

## 7. 状态、正确性和计时

C++ 只加载所选模式的模型一次，持久保留 current/next device buffer。
verify 完成后，主机复核接受数并统一发布状态；异常使本次请求失效，清零后才能继续。
correction 或 bonus 成为下一轮 anchor，本轮不提前把它写入已提交状态。

paired 运行按模式交错执行 3 次预热和 10 次测量，要求普通/DFlash token、EOS 和停止原因
一致。还需与 Python NPU ordinary 比较，覆盖零/部分/全接受、跨 64 行块、长 prompt 和重复
请求，才能排除两个 OM 路径共有的导出误差。

| 时延字段 | 范围 |
|---|---|
| `latency_ms.request_reset` | 请求状态清零 |
| `latency_ms.prefill` | prompt 处理 |
| `latency_ms.decode` | prefill 之后的生成循环 |
| `latency_ms.model_total` | prefill + decode，排除加载、tokenizer 和 request reset |
| `stage_ms` | 按图记录每次同步 OM 调用，包括必要的 H2D/D2H |
| `profiled_elapsed_ms` | 带 msprof 开销的同步采集窗口，排除控制器等待 |

不同 OM 不自动共享权重。固定 gear 的 padding、完整 cache 更新、每块 prefill LM head，
以及长 prompt 的 Draft KV 初始化都可能影响性能。以真实设备测量判断耗时，不能根据 OM
数量或主机模拟时间推断加速比。

## 8. msprof 窗口

统一 wrapper 为 `tools/run_msprof.sh`。C++ 使用 `--profile-backend cpp`，
普通模式选 `--profile-mode ordinary --profile-stage prefill|decode|all`，DFlash 选
`--profile-mode dflash --profile-stage prefill|draft|verify|all`。这些参数放在 wrapper 的
`--` 之前，之后跟 C++ runner 命令。

每个阶段从同一个 prompt 重建状态，窗口外完成预热。`all` 复用一个应用进程，
逐阶段各开一次独立窗口。C++ verify 的窗口包含两遍 GDR、Top1 和接受/提交计算。

Python NPU 支持更细的投影、输入准备、Top1、接受/提交和联合窗口，完整范围见
[Python NPU 手册](DFLASH_RUN_AND_VALIDATE.md)。两个后端均由 msprof 动态 PID CLI 控制，
无需 pyACL。每段算子耗时查看 `op_summary*.csv`，阶段同步耗时查看 stage report/summary。
