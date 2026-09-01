# Quant AIR/OM 推理框架

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

详细构建、运行与验证命令见
[docs/QUANT_AIR_OM_FRAMEWORK.md](../docs/QUANT_AIR_OM_FRAMEWORK.md)。
增量状态 ABI、2/3/4 OM 选择门禁和内存检查命令见
[docs/INCREMENTAL_OM_PERFORMANCE.md](../docs/INCREMENTAL_OM_PERFORMANCE.md)。

当前第一版 OM 使用固定 gear 的完整前缀重算，以先冻结可验证的两输入/两输出 ABI。它确实由
C++ 调用 OM 完成 token 推理，但尚未把 `quant` 分支已有的 persistent rollback cache/state
转成显式 OM I/O。因此它是功能基线，不应在真实测量前声称已达到闭源框架时延。

当前 C++ 基线在第一次完整输入上传后只发送变化区间，并只从 Target 输出下载 scheduler 需要的
尾部 `K+1` 行；JSON 保留实际与“每次完整传输”等价字节计数。这个 exact I/O 优化不改变 OM
数学，也不能替代后续 incremental state OM。

当前 `quant` 基线要求原 GDR 算子接收 `INT16[B] effective_length`。框架不会为此增加第三个
OM 输入，而是在 AIR 图内从 `attention_mask` 计算有效前缀长度；静态物理 gear 与逻辑有效
行数因此可以分别为 64 和 37。

Target modeling 的七个 NPU 自定义算子数值路径保持不变：`npu_dynamic_quant`、
`npu_quant_matmul`、`adn_rms_norm`、`npu_chunk_gated_delta_rule`、`npu_cache_update_`、
`adn_fused_infer_attention` 和 `npu_scatter_nd_update_`。AIR 导出前，框架逐个锁定 dispatcher
schema，并校验已有 Meta 或在缺失时注册精确 Fake；原位 cache/scatter 的 writable alias 也属于
合同。QuantMatmul 的 AIR 路径使用项目私有 functional frontend，避免与 receiver TorchAir
注册在源 target 上的 V3 builtin converter 冲突，并精确 lowering 为 V4444；普通 eager 为
ACLNN V4 一次性编码并缓存 INT64/UINT64 scale，最终 helper 也会阻止 FP32 直达 ACLNN；AIR 只在
active Dynamo capture 中保留原始 FP32 scale，两条路径不会因私有 op 的注册状态或残留 factory
flag 串路。Attention 则按 receiver 的真实 310P prototype lower 为单输出
`AdnFusedInferAttention`，并在加载权重前验证 ADN vendor、原型和预编译 kernel；不会再生成
310P3 无 kernel 的 A2 `FusedInferAttentionScore`。最终 `dynamo.pbtxt` 节点计数和 converter 审计写入
`air-manifest.json`，不会通过 Tensor 公式替换绕过自定义算子。
