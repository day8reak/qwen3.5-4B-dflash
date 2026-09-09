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

单模式 DFlash 加载 `target_prefill`、`target_verify`、`draft`；普通运行加载
`target_prefill`、`target_decode`；paired 加载四个。
prefill 共用，verify 内部完成接受判断与状态提交。

## 2. 输入和导出配置

量化 YAML 必须包含三个绝对路径：

| 字段 | 内容 |
|---|---|
| `quanted_pth` | Target W8A8 Linear 的 `data*.safetensors` 目录 |
| `embedding_weight_path` | INT8 Target embedding raw binary |
| `embedding_scale_path` | 每词表行的 FP32 scale raw binary |

Target 的全部 Linear（包含 `dflash_execution_model.lm_head`）及输入 embedding 使用量化数据。
Draft 使用 FP16 embedding、LM head 和主体；公开 embedding getter 保留的是 Draft head。
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
| `adn_rms_norm_ge_op_type` | 默认 `RmsNorm`；也支持已注册的 `AdnRmsNorm` |

外部输入不能是 symlink；输入 manifest 冻结后不得修改对应内容。导出器在加载模型前检查
输入 hash 和仓库 `SOURCE_LOCK.json`。物理 KV 容量为 C+64，额外行作为 scratch；
请求须满足 `prompt_tokens + max_new_tokens <= C`。

## 3. OM 有序输入/输出

ABI 标识为 `qwen35-dflash-chunk-v3`，合同见
[图与状态合同](../framework/abi/dflash-chunk-v3.json)。所有图固定 batch=1。
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
| `target_verify` | `target_top1 INT64[1,16]`、`accepted_count INT64[1]`、`features FP16[1,64,20480]`、完整 Target 状态、24 份第一遍 GDR 的原始 FP32 state |

prefill/decode Top1 对应本次最后一个有效输入行。verify 输入为 `[anchor,d1,...,dK]`，
右侧补齐至 16 行；`valid_rows=K+1`。接受数 `a` 是从第一个 proposal 起连续匹配的长度，
提交 anchor 加 a 个 proposal，共 `a+1` 行。verify 的有效 committed features 右补齐到 64 行。

一次 verify OM 内部执行：

1. 从本轮初始状态计算全部有效 verify 行，完成第一遍 GDR 和 Target Top1。
2. 比较 proposal 与 Target Top1，得到最长连续接受数 a。
3. 从同一本轮初始 recurrent state，以 `effective_length=a+1` 完成第二遍 GDR。
4. 选择对应 conv state，输出 committed features 和完整 Target 状态；随后附加第一遍 GDR 的原始 FP32 state。

第二遍 GDR 所需的中间量留在图内，没有独立 commit OM，也不回传主机。
GDR 累加及算子 initial/final state 使用 FP32；OM 之间持久保存的 recurrent state 为 FP16。

附加输出命名为 `verify_discard_t<layer>_recurrent`，按线性注意力层的顺序排列，
每份为 FP32 `[1,32,128,128]`。它们直接来自第一遍 GDR 的第二个输出，
保持 `output_final_state=True`，不经过 FP16 转换或零乘法。
标准 4B verify 共 91 个输出，最后 24 个为这些 discard state。

C++ 为每份 discard state 分配独立、持久的设备缓冲区，共 48 MiB；
不分配对应 pinned host 内存，不做 H2D/D2H，也不在请求 reset 时清零。
它们不是任何 OM 的输入，不参与 `current/next` 缓存交换，每轮由 GDR 覆盖。
缓存只接收第二遍 commit 的输出。48 MiB 是这些输出缓冲区的大小，完整模型显存
变化还取决于 ATC 的图内内存规划。普通模式只加载 prefill/decode，不分配这些缓冲区。

导出器在 GE 保存前检查这些输出确实连接到第一遍 GDR 的 raw state，
生成 `air/target_verify/verify-discard-outputs.json`。该报告证明保存前的输出连接；
实际 OM 缓冲区由 C++ 加载时的逐输出 dtype/shape/bytes 校验和分配保证。
接口版本必须与 runner 一致；使用新的空 bundle 目录导出 AIR、编译 OM，并重建 runner。

### 3.2 Draft 图

| 输入顺序 | dtype/shape | 含义 |
|---|---|---|
| `features` | FP16 `[1,64,20480]` | 本次需追加的 committed Target feature，右补齐 |
| `start_position` | INT64 `[1]` | 这批 feature 在上下文中的起始位置 |
| `valid_rows` | INT16 `[1]` | 有效 feature 行数，1..64 |
| `anchor` | INT64 `[1]` | 当前已输出、尚未作为 Target 输入提交的 token |
| `proposal_count` | INT16 `[1]` | 本轮实际草稿数 K，1..15，受请求设置和剩余输出预算约束 |
| `d0_key,d0_value,...,d5_key,d5_value` | FP16 `[1,8,C+64,128]` | 6 层 committed-context KV |

输出为 `draft_top1 INT64[1,15]`，随后是相同顺序的 12 个更新后 Draft KV。
一次调用完成 feature projection、context KV 追加和 anchor+K mask 的并行 proposal。
物理 block 固定 16 行，每层 attention 都排除 K 以后的 noise key，包括最后的非因果层。
输出前 K 项有效，其余项为 0；不能用完整 block 的结果直接截断来替代短 block。
用于 proposal 的 transient block KV 不作为 committed cache 输出。

增量套件的 ATC 编译自动添加 `--precision_mode=must_keep_origin_dtype`，
保留图中显式的 FP32 RMSNorm、RoPE、Softmax、注意力 matmul 和 GDR 状态计算。
也可显式使用等价的 `--precision_mode_v2=origin`；两个参数不能同时使用。
编译器拒绝对该套件使用降精度模式。FP16 checkpoint 不意味着全部中间计算都是 FP16。
参数语义见 [ATC 精度模式说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta3/devaids/atctool/atlasatcparam_16_0068.html)。
部署 manifest 的 `compiler.precision_policy` 为 `preserve_graph_dtypes`，
每张图的 `atc_command` 记录最终参数。

增量 GDR 在 GE 保存前逐节点核对 `core_attn=FP16`、`last_recurrent_state=FP32`，
报告为 `air/<图名>/gdr-output-dtypes.json`，并写入 `runtime_input_abi.gdr_output_dtypes`。
这项检查覆盖未使用的物理输出；它证明 Python 交给 GE 的类型，不能替代接收端
InferShape/InferDataType 和 ATC 的实际编译结果。ATC 失败时异常附带错误摘要，
完整内容仍保存在每图编译日志中。

## 4. 自定义算子导出要求

导出器在加载权重前验证所需 dispatcher schema，保留并验证已有 Meta kernel，
为缺少 Meta 的算子注册 Fake 实现。Fake 仅描述输出 shape/dtype，实际计算仍由目标算子执行。
随后逐图注册所需 converter，并在 `dynamo.pbtxt` 中核对保留的 GE 节点。

| AIR 路径 | PyTorch 前端 | GE 节点与导出处理 |
|---|---|---|
| 三个 Target 图 | `npu::adn_rms_norm` | `RmsNorm` 或 `AdnRmsNorm`；按注册类型使用输入名 `x` 或 `self` |
| 三个 Target 图 | `npu::npu_dynamic_quant` | `DynamicQuant`；保留 TorchAir 内置 converter，审计 GE 节点 |
| 三个 Target 图 | `qwen35_dflash::npu_quant_matmul_v4444` | `QuantBatchMatmulV4444`；专用捕获前端避免覆盖 TorchAir 的量化 matmul converter |
| 三个 Target 图 | `npu::npu_chunk_gated_delta_rule` | `ChunkGatedDeltaRule`；Meta 覆盖 R=1/16/64，保留两个输出和 FP32 recurrent state |
| 三个 Target 图 | `npu::adn_fused_infer_attention` | `AdnFusedInferAttention`；保留 paged KV、mask、block table 和长度输入 |
| 三个 Target 图 | `aten::softplus` | `SoftplusV2`；保留 beta/threshold 属性 |
| 完整前缀图的缓存路径 | `qwen35_dflash::npu_cache_update` | `CacheUpdate`；功能化前端输出更新后的缓存，供 attention 消费 |
| 完整前缀图的可选 scatter 路径 | `npu::npu_scatter_nd_update_` | `ScatterNdUpdate`；验证别名/Meta，保留内置 converter |

增量四图的缓存写入使用函数式 `scatter`，由 TorchAir 映射到 `ScatterElements`，
不要求 `CacheUpdate` 或 `ScatterNdUpdate` 节点。Target 在展平的 paged KV 第 0 维写入，
Draft 在 `[B,H,C,D]` 的第 2 维写入；位置索引用静态 `repeat/Tile` 复制到更新值的形状，
输入缓存保持不变。索引重复因子来自固定形状，不构造动态 `BroadcastTo` shape 输入。
写入位置由连续且不重复的 token 位置构造，结果与整行 `index_copy` 相同。
Draft 的 GQA 在新插入的 group 维使用 `repeat/Tile`，head 顺序为
`[h0,h0,...,h1,h1,...]`，不重复整个 head 序列。
Draft 图使用 Tensor 算子。完整前缀工厂按其实际缓存路径声明算子依赖。
本分支的 verify 和 commit 都使用 `ChunkGatedDeltaRule`，不依赖 `GatedDeltaRuleMTP`。

卷积状态窗口使用静态切片加 `stack`，不调用 TorchAir 尚未实现 GE converter 的
`aten.unfold.default`。卷积宽度 K=4 时只构造四个移位切片，得到
`bank[b,r,c,k] = history[b,c,r+1+k]`；`bank[:,v-1]` 就是消费 v 行后的状态。
prefill、decode 和 verify 的接受前缀提交共用该规则，卷积与 SiLU 计算保持同一公式。

verify 接受长度在图内以整数计算：有效 proposal 范围是 `valid_rows-1`；
将范围内的 mismatch 转为 INT32，经过 `Cumsum` 后，以累计 mismatch 为 0 的有效行
形成连续接受前缀，最后用 INT32 `ReduceSum` 计数并转回 `INT64[1] accepted_count`。
该路径不调用 `amin`、`min(dim=...)` 或 `cumprod`。零接受、全接受及短块 padding
均使用同一规则；第二次 GDR 仍以 `INT16[1](accepted_count+1)` 提交 anchor 和接受前缀。
主机回归覆盖全部 32768 种 proposal 匹配模式和 1～16 的有效行数，并检查捕获图中的
INT32 scan/reduction 及缓存、head 复制的 Tensor 算子。

PyTorch 的 FakeTensor/严格捕获通过，只证明 PyTorch 图有效；标准算子的 GE 支持
仍需单独检查，最终以目标机 TorchAir 导出及 ATC 编译为准。

W8A8 的量化 activation/weight 保持 INT8，weight scale 和 per-token scale 保持 FP32，
matmul 输出为 FP16。专用前端仅在工厂启用的 AIR 捕获期间生效。普通 NPU 推理的量化
调用不变。RMSNorm 的两个输出分别为同 input shape/dtype 的 Tensor，以及
`[*input.shape[:-1],1] FP32` 的 rstd。

GDR 的 `effective_length: INT16[1]` 表示本次有效行数。
Target attention 的三个长度专用输入分别是：

| 前端参数（SymInt[]） | GE Tensor（INT64[1]）内容 |
| --- | --- |
| `all_seq_lengths_q` | 增量静态图为物理缓存容量 `C+64` |
| `actual_seq_lengths_q` | 当前物理行数：prefill 64、decode 1、verify 16 |
| `actual_seq_lengths_kv` | 物理缓存容量 `C+64` |

运行时 mask 只允许 `key_position <= query_position` 且
`key_position < start_position+valid_rows` 的位置参与 attention。
位置和长度保持整数，FP16 mask 只表示精确的 0 和负无穷；有效前缀由 mask 限定，
不能从固定物理长度推断真实已提交长度。该策略仍需通过目标机普通生成和 DFlash
严格 token 对齐检查。

`pse_shift` 是可选的 FP16 attention bias，本模型不传该输入；不可用它携带整数长度，
也不可把长度转成 FP16 后传入。Fake/GE converter 会拒绝非 FP16 的 PSE。
完整前缀工厂同样通过 `all_seq_lengths_q` 传递其固定序列长度。
GDR GE 输入中
`initial_state` 位于 `effective_length` 前；converter 显式按 GE 名称映射，
不把 PyTorch 的参数顺序直接传给 GE。已配置的 vendor 环境需要通过 GDR/attention
prototype 检查。

`air-manifest.json` 保存 `operator_preflight`、每图的
`custom_op_export_contracts`、`custom_op_audit` 和 `standard_op_overrides`。
内置 converter 以 GE 节点审计为准；框架 converter 同时检查调用次数。
解析同时支持 GE dump 的 `op:` 和 `type:` 字段，避免重复计数。
编译前检查整个图集合的审计，缺少任意声明的算子或 SoftplusV2 记录都会拒绝进入 ATC。
增量 bundle 还会检查 attention 导出策略；缺少该策略或使用整数 PSE 的 bundle
需要重新导出 AIR，不能只重新运行 ATC。

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
`--eos-token-id` 可重复传入，用于覆盖 tokenizer 的 EOS 并对齐 NPU 报告。
`--trace-rounds` 为 chunk bundle 记录每轮的 proposal、Target token、接受前缀和输出；
直接 C++ 入口也支持此开关。记录位于每条 measurement 的 `rounds` 中，
性能基线不启用逐轮记录。
Python 处理 tokenizer、文本和报告；生成热循环在 C++ 内执行。
直接调用 C++ 时使用 `--model-kind chunk --mode ordinary|dflash|paired`，
配合 `--model`、`--model-sha256`、`--prompt-token-ids`、`--eos-token-ids` 和 `--output`。

需要组合步骤时，`build-om` 合并 AIR 导出和 ATC 编译；`run-e2e-cpp` 合并导出、编译和
paired 推理，runner 须已构建。两者都应显式传入本参考第 2 节的增量 `--factory`。

## 7. 状态、正确性和计时

C++ 只加载所选模式的模型一次，持久保留 current/next device buffer。
verify 完成后，主机复核接受数并统一发布状态；异常使本次请求失效，清零后才能继续。
correction 或 bonus 成为下一轮 anchor，本轮不提前把它写入已提交状态。
零接受后关闭 Draft，后续轮次优先使用已加载的 `target_decode`；
只加载三图的 DFlash 模式使用 `target_verify` 的 `valid_rows=1`。
报告的 `speculation_disable_events`、`target_only_fallback_rounds` 与 `stage_ms`
分别记录关闭事件、后备轮数和实际调用图。

paired 运行按模式交错执行 3 次预热和 10 次测量，要求普通/DFlash token、EOS 和停止原因
一致。还需与 Python NPU ordinary 比较，覆盖零/部分/全接受、跨 64 行块、长 prompt 和重复
请求，才能排除两个 OM 路径共有的导出误差。

最终 token 相同不代表逐轮一致。模块
`qwen35_dflash.ascend310p.compare_rounds --native NPU_JSON --cpp CPP_JSON --output REPORT`
验证逐轮输出能重建完整 token 序列，并按相同的已提交前缀比较每次 proposal、verify 和接受结果。
固定 16 行 Draft OM 与 NPU 在尾部缩短的 Draft block 可能产生不同 proposal，
需结合输入形状和中间张量继续定位，不能仅按相同轮次编号比较。

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
