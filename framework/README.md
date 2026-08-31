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
- `scripts/compare_cpp_closed_runtime.py`：同设备、同 token、同计时范围的性能对比；
- `FRAMEWORK_LOCK.json`：本分支冻结的量化、图和运行时 ABI。

详细构建、运行与验证命令见
[docs/QUANT_AIR_OM_FRAMEWORK.md](../docs/QUANT_AIR_OM_FRAMEWORK.md)。

当前第一版 OM 使用固定 gear 的完整前缀重算，以先冻结可验证的两输入/两输出 ABI。它确实由
C++ 调用 OM 完成 token 推理，但尚未把 `quant` 分支已有的 persistent rollback cache/state
转成显式 OM I/O。因此它是功能基线，不应在真实测量前声称已达到闭源框架时延。

当前 `quant` 基线要求原 GDR 算子接收 `INT16[B] effective_length`。框架不会为此增加第三个
OM 输入，而是在 AIR 图内从 `attention_mask` 计算有效前缀长度；静态物理 gear 与逻辑有效
行数因此可以分别为 64 和 37。
