# Quant AIR/OM 推理框架

当前 v47 / runner 1.29.0 增加可选[通用逐算子诊断](../docs/OM_OPERATOR_DIAGNOSTICS.md)：
原 OM 运行时选择节点 dump、显式输入/输出逐张量比较、同输入原生 `aten`/`npu`
算子回放。GDR-MTP 只是附加 ABI 检查的专用入口，不限制其他算子的对照。
默认关闭；只更新 Python/C++，不重导 AIR/OM。仍需真机采集和回放，未证明 GDR 或其他内核有错。

v46 保留可选 Verify GDR 数值对照：按普通 Decode1 逐 token 调用
`ChunkGatedDeltaRule(chunk_size=1)`，每步执行相同的 FP16 状态舍入。
默认仍为 GDR-MTP；对照不是 MTP 内核修复，不声称真机 PASS，可能增加 Verify 时延。
仅启用对照时需要重新导出 AIR / 编译 OM；runner 1.28.0 兼容，无需重新量化。
详见 [可选对照与复验](../docs/OM_VERIFY_GDR_REFERENCE.md)。

v45 / runner 1.28.0 增加默认关闭的
[Target 定点状态重放](../docs/OM_TARGET_PARITY_DIAGNOSTICS.md)，使用
`infer-cpp --diagnose-target-parity --diagnostic-max-transactions 2` 开启。
保留 v44 的[首对即停与失败诊断](../docs/OM_FAILURE_DIAGNOSTICS.md)。
本次只需重建 runner、更新 Python 控制面，可复用已有 v43 AIR/OM；不代表已修复真机
ordinary/DFlash 分歧。更早版本缺失的图内修复仍需重新导出。

这个目录是直接加在仓库 `quant` 分支之上的部署层，不替换现有量化、rollback 或 DFlash
实现。基线提交固定为 `28f93e784a2beed87020a80bd93c8788754eab1c`。

完整数据流是：

```text
quant 分支 Target W8A8 + FP16 Draft
        │ TorchAir dynamo_export
        ▼
      AIR + 外置权重
        │ atc --framework=1 --mode=0
        ▼
      静态 OM
        │ C++17 / AscendCL
        ▼
ordinary greedy 与 strict-greedy DFlash 逐 token 生成
```

目录内容：

- `python/qwen35_dflash/ascend310p/`：AIR 导出、ATC 编译、manifest/hash
  门禁、tokenizer 控制面和 C++ runner 启动器；
- `runtime/cpp/`：不经过 Python 热循环的 AscendCL OM runner；
- `abi/`：OM、运行时、性能和闭源框架 A/B 合同；
- `runtime/cpp/qwen35_dflash_om_inspect`：用 `aclmdlQuerySize` 计算多 OM 候选的权重、共享
  workspace 和状态预算，不假设不同 OM 自动共享权重；
- `scripts/compare_cpp_closed_runtime.py`：同设备、同 token、同计时范围的性能对比；
- `FRAMEWORK_LOCK.json`：本分支冻结的量化、图和运行时 ABI。

v41 / runner 1.25.0 的默认构图、迁移命令和验证边界见
[静态四图默认部署指南](../docs/STATIC_SPLIT_OM_DEFAULT.md)：合并 Prefill body/head，
保留 Decode1，独立静态 Draft N=64 和 Verify16。默认 phase-resident，
Draft/Verify 联合常驻、两份权重不重叠；Prefill/Decode1 按阶段替换。
必须重新导出 AIR、编译四个 OM 并重建 runner，不能复用旧 Prefill ABI。
静态真机严格 greedy 对齐后再验证动态。已有 Scatter/GQA Tile 审计继续保留。

[旧静态 fused 基线](../docs/STATIC_FUSED_OM_BASELINE.md) 和
[v40 静态 fused 显存候选](../docs/STATIC_OM_MEMORY.md) 保留为显式回退。
只有 v40 的旧拓扑生命周期改动可复用原正确静态 OM；它的“不用重新导出”不适用于 v41。
通用框架与历史排错参考见 [AIR/OM/C++ 框架](../docs/QUANT_AIR_OM_FRAMEWORK.md)。
增量状态 ABI、2/3/4 OM 选择门禁和内存检查命令见
[docs/INCREMENTAL_OM_PERFORMANCE.md](../docs/INCREMENTAL_OM_PERFORMANCE.md)。

保留的第一版诊断 OM 使用固定 gear 的完整前缀重算，以先冻结可验证的两输入/两输出 ABI。它确实由
C++ 调用 OM 完成 token 推理，但尚未把 `quant` 分支已有的 persistent rollback cache/state
转成显式 OM I/O。因此它是功能基线，不应在真实测量前声称已达到闭源框架时延。

该单图 C++ 诊断基线在第一次完整输入上传后只发送变化区间，并只从 Target 输出下载 scheduler 需要的
尾部 `K+1` 行；JSON 保留实际与“每次完整传输”等价字节计数。这个 exact I/O 优化不改变 OM
数学，也不能替代后续 incremental state OM。

当前 `quant` 基线要求原 GDR 算子接收 `INT16[B] effective_length`。框架不会为此增加第三个
OM 输入，而是在 AIR 图内从 `attention_mask` 计算有效前缀长度；静态物理 gear 与逻辑有效
行数因此可以分别为 64 和 37。

Target modeling 的八个 NPU 自定义算子数值路径保持不变：`npu_dynamic_quant`、
`npu_quant_matmul`、`adn_rms_norm`、`npu_chunk_gated_delta_rule`、`npu_cache_update_`、
`adn_fused_infer_attention`、`npu_scatter_nd_update_` 和 verify 专用的
`npu_gated_delta_rule_mtp`。AIR 导出前，框架逐个锁定 dispatcher
schema，并校验已有 Meta 或在缺失时注册精确 Fake；原位 cache/scatter 的 writable alias 也属于
合同。QuantMatmul 的 AIR 路径使用项目私有 functional frontend，避免与 receiver TorchAir
注册在源 target 上的 V3 builtin converter 冲突，并精确 lowering 为 V4444；普通 eager 为
与可运行的 `quant` 分支完全一致，保留 INT8 activation、INT8 weight、FP32 weight scale 和
FP32 per-token scale；它不会调用 `npu_trans_quant_param` 生成与 INT8 weight 不匹配的 INT64
carrier。AIR 只在 active Dynamo capture 中切换到项目私有 FP32-scale frontend，两条路径不会因
私有 op 的注册状态或残留 factory flag 串路。Attention 则按 receiver 的真实 310P prototype
lower 为单输出 `AdnFusedInferAttention`，并在加载权重前验证 ADN vendor、原型和预编译 kernel；不会再生成
310P3 无 kernel 的 A2 `FusedInferAttentionScore`。最终 `dynamo.pbtxt` 节点计数和 converter 审计写入
`air-manifest.json`，不会通过 Tensor 公式替换绕过自定义算子。
