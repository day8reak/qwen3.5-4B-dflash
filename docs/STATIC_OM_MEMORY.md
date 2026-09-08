# v40：静态 OM 按阶段常驻，先解决权重同时分配的峰值

本页记录 **v40 旧静态 fused 四图**的运行时修复，供显式回退。
v41 默认已改为 [合并 Prefill + Decode1 + 静态 Draft + Verify16](STATIC_SPLIT_OM_DEFAULT.md)，
默认 phase-resident，Draft/Verify 联合常驻，权重预算为
`max(Prefill, Decode1, ALIGN_UP(Draft,512)+Verify)`，而非最大单 OM。
新默认需要重新导出 AIR/OM；下面“不用重新导出”的说明只适用于旧 fused 拓扑。

本候选只改 C++ 运行时生命周期与控制面，不改已有静态 AIR/OM、量化精度、
KV capacity、prompt 或生成长度。必须显式配置 `model_residency_policy: "phase-resident"`；
缺省仍为 `all-resident`，旧的性能基线和动态路线不被静默替换。
v40 只支持经过 manifest 和物理 ABI 双重检查的四个静态 fused 路线 OM。

## 这次日志说明什么

接收端 2026-09-08 02:24:19–20 的日志中，APP 已分配
10,188,557,408 字节（约 9.49 GiB），又请求 8,712,392,192 字节（约 8.12 GiB），
随后 driver 报 out of memory / 207001。仅这两项就约 17.60 GiB；
它不是一个能够证明具体算子中间张量过大的执行期日志。
该片段缺少完整 model-query role 与当时设备空闲量，不能凭大小断言具体失败 OM、
设备总容量或其他进程占用。

旧代码已经共享串行 workspace，但启动时先为所有 OM 分配独立权重，再加载模型。
因此仅在推理结束后卸载 prefill 不够：必须首先避免启动时分配全部权重。
本候选保留一个最大 workspace，另保留一个最大单 OM 的 weight arena：

| 显式分配项 | all-resident | phase-resident |
| --- | --- | --- |
| 模型权重 | sum(weight_bytes) | max(weight_bytes) |
| 串行 workspace | max(work_bytes) | max(work_bytes) |
| KV、recurrent state、输入输出载体 | 保持原配置 | 保持原配置 |
| GE/driver 内部开销 | 另计 | 另计，需要真机测量 |

这里是**先卸载后复用内存**，不是让两个活模型共享权重，更不是假设不同 OM
的权重布局相同。OM 文件大小不变，最大 weight arena 也保持预留，避免每次换模重新申请大块。
失败的显式分配会记录 owner、requested_bytes 与 aclrtGetMemInfo 的状态/空闲量/总量；
空闲量是错误时的快照，不是整次运行的峰值。

## 哪些常驻

启动时逐一加载、读取并复制静态 I/O metadata、校验、卸载，任一时刻最多一个活模型。
真正生成时在模型角色变化前完成 stream 同步并检查卸载成功，再加载新角色：

- prefill body 在 prompt 的连续块之间常驻；最终 head 只运行一次。
- 普通 greedy 的 decode1 连续执行时常驻。
- DFlash 的 fused-speculative-step 连续执行时常驻，不逐 token 重载。
- DFlash 剩余预算为 1，或启用 request-target-only 且触发零接受回退，才切至 decode1。
  尚不能直接删除 decode1；fused 的当前调度路径要求正 proposal count。

应用拥有的 KV/recurrent/compact-result 缓冲区不会随模型卸载。新的 aclmdlDesc 在每次
重载后重建，输入/输出数量、dtype、shape 和 bytes 必须与启动时复制的 ABI 一致，
才允许旧 dataset 执行。运行中的换模重载、卸载或执行失败均停止，不写 PASS 报告。
异步执行即使返回错误也按可能有排队任务处理，释放前先尝试同步。

依据 CANN 的接口约束，调用方提供的 work/weight 内存在模型执行期间必须有效，
卸载前不能仍有接口在使用模型；加载、执行、卸载使用同一 Context。
参见 [LoadFromFileWithMem](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_0285.html)
和 [aclmdlUnload](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/850alpha002/API/appdevgapi/aclcppdevg_03_0291.html)。

## 从哪里重跑

如果已有通过 v39 KV Tile 导出审计的正确静态 bundle，**本次不用重新 export-air /
compile-om**。只需拉取源码、使用原 CANN 环境重新 build-cpp，换用 1.24.0 runner，
然后在原本填写真实硬件身份的 runner JSON 中加入：

```json
"model_residency_policy": "phase-resident"
```

完整模板见 [静态 runner 配置](../config/quant_air_om_static_fused_runner.example.json)。
它不能直接运行，必须填写当前设备/CANN/driver/firmware 身份。初次复现保持
async-memset、fixed-16、sync window 1、separate prefill、disabled fallback。
沿用 [静态基线](STATIC_FUSED_OM_BASELINE.md) 的 build-cpp / infer-cpp 命令，
选择新 build/output 路径与该 runner JSON；保持原失败用例的 prompt、长度、manifest。
直接调用 C++ 时需同时提供 `--model-residency-policy phase-resident` 和真实
`--fused-static-feature-rows N`。旧 runner、动态 OM 或未声明静态 shape 的 manifest
不能用这个候选。

检查报告 `model_residency` 的 peak_resident_models=1，以及 `model_memory_query`
的 allocated_weight_bytes=max(models[].weight_bytes)；
weight_bytes_elided=sum_weight_bytes-allocated_weight_bytes。
explicit_allocated_device_bytes_excluding_runtime 用实际 arena 大小闭合，不再加全部权重。
models[].model_id 在该策略下仅是启动 metadata 检查的 ID，已经卸载，允许 CANN 复用；
诊断 trace 按实际执行 role 和当时的 model_id 归因。
当前 msprof 汇总器只支持常驻模型的 ID 映射，会明确拒绝 phase-resident：
跨加载周期的 ModelId/InferId 可能复用，不能拿启动 ID 去关联执行期 CSV。
保留原始 profile 和带 role 的 runner trace；要支持该汇总需另补加载周期证据。

## 性能、图拆分与后续显存工作

模型切换的同步、卸载、加载和 ABI 检查均计入生成/benchmark wall time，
`model_load_excluded_from_latency=false`；启动 metadata 检查单独统计。
额外 switch synchronization 单独记录并计入总同步数，不能当成 DFlash 多窗口的节省。
phase-resident 不应被当成原 all-resident 稳态延迟直接比较；大 OM 每次请求重载有成本。

以下是 v40 的后续候选分析；其中合并 Prefill 和 Draft/Verify 联合驻留现已由 v41 实现，
设备收益仍待验证。v40 的短 prompt 静态基线可进一步把 prefill body+head 合并；原拆分是为多块 prompt
只在最终块运行一次 head，且 body 已明确排除 head 权重。合并可减少换模，但不保证降低
max(weight/work)；需要新的图 ABI、AIR/OM 与精度门禁，v40 运行时修复不包含这项图变更。
另一合理后续是 prefill、Draft、Verify 三图：prefill 用后卸载，Draft/Verify 热循环常驻。
若删除 decode1，Verify 还需经过严格验证的 target-only 路径，覆盖预算尾部、零接受回退、
EOS、recurrent state 选择与 KV 边界。静态 T=16 跑逻辑 K=0 也会付出多行计算成本。

若最大单 OM+workspace+state+carriers 仍不够，应依据新日志及同时间 npu-smi 空闲量继续定位。
仅减少 max_new_tokens 不会减少静态 OM 权重。减小 KV capacity 需重新导出/编译并收紧
请求长度门禁；减少精度、改 attention 累加方式或裁剪模型不属于本候选。
也不能在未验证量化权重的 OM 编译布局前承诺通过文件压缩降低加载显存。

本地 fake ACL 仅验证受限预算下的分配关系、同步/卸载生命周期和调度计数，
不是真实 CANN 内存峰值、算子执行或性能证据。仍需物理 310P 无 fallback、
ordinary/DFlash token 与 EOS 零差异以及 3 warmup+10 测量后才能确认可部署。
