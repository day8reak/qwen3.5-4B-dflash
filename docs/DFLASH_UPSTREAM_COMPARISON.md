# 当前 rollback 与官方完整 DFlash 的差异

本文只做版本对照，不重复框架和算子说明。“官方完整 DFlash”固定指：

- 算法仓 z-lab/dflash，提交 07ebd93db9f472af339b644bb70221ad8428328a；
- Draft checkpoint z-lab/Qwen3.5-4B-DFlash，项目锁定 revision
  9a1996ccf887b79ab3af4fcbf8c1d1f4b5658bcf；
- 普通 DFlash，不包含 DFlash2、SGLang/vLLM 后续实现或官方 main 分支的新变化。

官方参考源码：

- [Transformers 实现 model.py](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py)
- [Qwen3.5 可用的 MLX 实现 model_mlx.py](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model_mlx.py)
- [该提交的 README 与 backend 支持范围](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/README.md)

## 1. 一句话结论

当前 CPU/GPU/NPU 代码不是官方仓库的逐行复制，也还不是官方全部 sampling 功能集。准确定位是：

~~~text
官方 Qwen3.5-4B DFlash Draft 图和 checkpoint 合同
+ 相同的 block draft / single target verify / contiguous accept 核心算法
+ 为 PyTorch Qwen3.5 与 HIAI/NPU 重写的 transactional rollback runtime
+ 强制 ordinary 对照的 strict-greedy 验证层
~~~

Draft 数学主体和 KV cache 的 committed/transient 生命周期与官方对齐；Target state、sampling、
backend 和运行接口仍存在明确差异。

## 2. 先区分官方两个本地 backend

在锁定提交中，不能简单说存在一个“官方 Qwen3.5 CPU/GPU 版本”：

| 官方 backend | 锁定提交中的范围 | 与当前代码的关系 |
| --- | --- | --- |
| Transformers / PyTorch | README 明确列出 DFlash 的 Qwen3 和 LLaMA-3.1；没有列 Qwen3.5 | 可作为 PyTorch scheduler、Draft cache 和 DynamicCache rollback 参考，但不是 Qwen3.5 原样实现 |
| MLX | README 明确列出 Qwen3.5；model_mlx.py 有 Qwen3.5 GDN state capture/rollback | 是 Qwen3.5 Target 状态语义的主要官方参考，但运行于 Apple MLX，不是 CUDA/HIAI |
| 当前仓库 | Qwen3.5-4B 的 CPU/CUDA PyTorch 与 HIAI/NPU | 是跨 backend port，需要自己的等价性门禁 |

因此当前 CUDA 路线可以叫 Qwen3.5 DFlash rollback port，不能叫官方 CUDA 实现。

## 3. 完整差异矩阵

| 项目 | 官方锁定实现 | 当前 rollback | 判断 |
| --- | --- | --- | --- |
| Draft checkpoint | checkpoint config 决定 Draft 拓扑 | 锁定同一 Qwen3.5-4B checkpoint、6 层、69 tensor 和 hash | 对齐且检查更严格 |
| Target feature | 8 个指定层拼接并投影 | 层 1、5、9、13、17、21、25、29，宽度 20480 | 对齐 |
| Draft block | `block_size` 包含 clean anchor | 相同；B=16 时为 anchor 加 15 个 mask | 口径与数学均对齐 |
| Draft attention | context KV 注入；前层 sliding causal，末层按 config | 相同 6 层与 mask 语义 | 对齐，代码重写 |
| 共享权重 | 使用 Target embedding 和 LM head | 使用并审计 Target embedding 和 LM head | 对齐 |
| Target verify | 一次验证 anchor 加 proposal | 一次 T=K+1 verify | 核心算法对齐 |
| Greedy accept | 最长连续 Top-1 匹配，随后 correction/bonus | 相同 | 对齐 |
| Target full-attention KV | crop/trim rejected tail | CPU/CUDA crop 后 replay；NPU logical cursor commit | 结果目标相同，实现不同 |
| Target GDN recurrent | MLX 捕获输入，拒绝后只重算 accepted+1 状态 | CPU/CUDA 恢复后逐 token replay；NPU GDR MTP state bank | 语义等价目标，实现不同 |
| Target conv state | MLX 从捕获的 conv input 恢复接受位置窗口 | CPU/CUDA snapshot/replay；NPU conv state bank golden | 语义等价目标，实现不同 |
| Draft KV cache | 官方 Transformers 与 MLX 都跨轮维护并 trim/crop | 6 层 request-local K/V；只追加新增 committed feature，当前 block 为 transient，成功后 crop | 语义对齐，cache 类为本项目重写 |
| Feature 生命周期 | 依赖 Draft cache，只保留下一轮需要的 accepted feature rows | prompt 或新提交的 1+a 行只投影一次；被 cache 消费后释放，不累积完整投影 history | 对齐 |
| Sampling | 支持 temperature、top-p、top-k 和 rejection sampling | 只支持 temperature=0 strict greedy | 当前缺失 |
| Ordinary 基线 | 正常 generate 不额外跑 ordinary 对照 | validator 先跑 ordinary，再独立跑 DFlash 并零差异比较 | 当前新增验证层 |
| API | generate/stream 与 acceptance、TPS 统计 | `dflash` 单跑、`validate` 对照、同步 NPU benchmark、JSON audit 和 fail-closed transaction | 工程接口不同 |
| Backend | PyTorch Qwen3/LLaMA；MLX Qwen3.5 等 | PyTorch Qwen3.5 CPU/CUDA；HIAI/NPU | 当前新增 port |
| 性能定位 | 使用 Draft cache，并提供本地生成/性能统计 | 已有 Draft cache 和单跑模式；310P 同边界实测仍待完成 | 尚不能声称达到官方 TPS |

## 4. block_size 口径已对齐

这是当前统一后的公共合同。

官方本地 Transformers 和 MLX runner 都把 block_size 当成 Target verify 的总行数：

~~~text
official block_size = anchor 1 行 + proposal 行数
block_size 16       = anchor + 15 proposals
Target verify T     = 16
~~~

当前仓库的配置、scheduler、CPU/CUDA/NPU CLI、报告和状态 bank 使用同一语义：

~~~text
--block-size 16     = anchor + 15 proposals
current K           = block_size - 1 = 15
Target verify T     = block_size = 16
~~~

官方 block 档位与显式 proposal 诊断档位的换算为：

~~~text
block_size = 2 / 4 / 6 / 8 / 16
K          = 1 / 3 / 5 / 7 / 15
T          = 2 / 4 / 6 / 8 / 16
~~~

`--proposal-counts` 只在接受率诊断中显式表示 K；运行入口只接收官方总行数口径。因此报告
可以直接与官方相同 block_size 档位比较，不再存在额外的 17-row 扩展档。生成尾轮或 proposal
遇到 EOS 时，本轮有效 K/T 可以更小，但始终满足 K≤block_size-1、T≤block_size。

## 5. Draft cache 已补齐的语义与实现差异

官方 Transformers 路线创建 past_key_values_draft，官方 MLX 路线创建 draft_cache。每轮 Draft
只把新 Target feature 和当前 block 接到已提交 Draft cache 上；完成 proposal 后再把 cache
trim/crop 到 committed boundary。

当前 `DFlashDraftKVCache` 复现相同生命周期：

~~~text
round 输入 = 已提交 Draft KV + 新 accepted feature + 当前 block
attention 可见 = old committed + new committed + transient block
round 结束 = 只保留 old committed + new committed
~~~

具体差异是当前没有直接复用 Transformers `DynamicCache` 或 MLX cache 类，而是使用独立、
request-local、逐层 `[B,Hkv,C,D]` cache，并增加 staged round、异常 abort 和 audit。无 cache 路线
保留为 golden。两轮 reduced-shape 测试同时覆盖 sliding causal 层和 final full-attention 层。

已经消除的是 6 层历史 context 的 K/V projection 和完整投影 feature history。仍未消除的是
attention 对 C+T 个 K/V 的读取以及当前 PyTorch cache 拼接；是否需要 paged/static Draft cache
或融合 GQA，必须由 NPU profile 决定。

## 6. Target rollback 的实现不同

三条路线的提交目标相同：

~~~text
verify 输入 [anchor, d1, ..., dK]
接受 a 个 proposal
只提交输入 [anchor, d1, ..., da]
correction/bonus 留作下一轮尚未处理的 anchor
~~~

但实现方式不同。

### 官方 Transformers

官方 model.py 为 Target 和 Draft 创建启用 past recording 的 DynamicCache。一次 verify 后，
通过 cache.crop 把 rejected tail 裁到新的 committed length。

该路径是通用 PyTorch 参考，但锁定 README 没有把 Qwen3.5 列为 Transformers 本地 backend，
不能直接推断其 DynamicCache 对 Qwen3.5 GDN state 已完整可用。

### 官方 MLX Qwen3.5

对于可 trim cache，直接 trim rejected tail。对于不可 trim 的 Qwen3.5 GDN cache，
_GDNStateCapture 会保存当前 q/k/v、gate、初始 recurrent state 和 conv input；知道 accepted 后
仅以 accepted+1 行重算 recurrent state，并从捕获输入恢复 conv window。

### 当前 CPU/CUDA

FrameworkDFlashRollbackTarget 在 verify 前 snapshot attention 长度、GDN conv/recurrent tensor
和初始化标志。得到 a 后先恢复 round-start，再用普通增量路径逐 token 重放
anchor 加 accepted proposals，最多 K+1 次调用。

这更保守，也便于与 ordinary incremental 数值路径对齐，但比官方 cache crop 或一次 accepted
短重算多调用。

### 当前 HIAI/NPU

- npu_gated_delta_rule_mtp 产生每行 recurrent state bank；
- causal-conv golden 产生每行 conv state bank；
- accepted_tokens 在下一轮选择上一轮已提交槽；
- paged KV 允许 provisional 物理写入，commit 只推进 logical cursor 1+a。

这是为 HIAI receiver 和现有自定义算子 ABI 设计的 port，官方仓库没有这套实现。

## 7. Sampling 与 greedy 的差异

官方实现同时支持：

- temperature=0 的 greedy 连续匹配；
- temperature>0 的 proposal 概率；
- Target/Draft 概率比 rejection sampling；
- 拒绝位置的 residual distribution correction；
- top-p 和 top-k。

当前 scheduler 只保留 strict greedy：

~~~text
proposal == Target Top-1 才接受
首次不匹配时使用 Target Top-1 correction
最终必须与 ordinary greedy token stream 零差异
~~~

所以当前不能声称完整覆盖官方 generation 功能。若以后实现 sampling，需要 Draft 概率、Target
概率、随机流和 residual sampling，验证标准也要从单次 token 零差异改为固定随机流与分布门禁。

## 8. 验证模式也不同

官方 generate 直接执行 DFlash，并输出 tokens、acceptance 和性能统计。当前 `run_rollback` /
`run_npu` 有两种模式：

1. `--execution-mode validate` 新建 ordinary persistent session，生成权威 greedy stream，再新建
   DFlash session并严格比较 token ID、EOS 和 stop reason；
2. `--execution-mode dflash` 只执行 DFlash session，避免生产请求额外跑 ordinary；报告明确写
   `NOT_RUN_DFLASH_ONLY`，不能作为本次 exact-match 证据。

ordinary 对照是当前 bring-up 的必要验收，但它会额外执行一次 Target，不能计入 DFlash
生产性能。`benchmark_npu` 把 correctness gate 放在计时区间外，并以分进程、设备同步的 3+10
测量 rollback receiver 内部的 ordinary/DFlash；这里的 ordinary 不是 main 分支原部署工程的
非 DFlash 实现。原模型权威基线仍须运行 `python3 inference.py --config
./config/qwen3.5.ymal --max_token 32`，且不传 `--quant_mode enable`。单次 `dflash` 运行只用于
功能/日常生成，正式性能结论仍来自边界对齐后的独立 benchmark。

## 9. 当前可以怎样表述

| 表述 | 是否准确 | 条件 |
| --- | --- | --- |
| 使用官方 Qwen3.5-4B DFlash checkpoint 与 Draft 拓扑 | 是 | checkpoint/hash 审计通过 |
| greedy DFlash 核心 proposal/verify/accept 算法对齐 | 是 | 最长连续匹配和 1+a commit 通过 |
| 官方完整 CPU/GPU 实现 | 否 | 当前是 Qwen3.5 PyTorch port |
| 官方 block_size=16 原样对齐 | 是 | 默认 B=16/K=15/T=16 |
| 完整官方 generation 功能 | 否 | greedy 与 Draft cache 已有；仍缺 temperature/top-p/top-k rejection sampling |
| Qwen3.5 transactional rollback port | 是 | CPU/CUDA/NPU 各自状态门禁通过 |
| 已达到官方性能 | 否 | 当前没有同边界配对性能证据 |

## 10. 补齐完整原版能力的建议顺序

1. 保持默认 `block_size=16`，即 K=15/T=16、temperature=0。
2. 对每轮比较 Draft proposal、Target Top-1、accepted、bonus 和 emitted token。
3. 比较拒绝后下一 token 的完整 Target state，而不只比较当前输出。
4. 已实现 Draft KV cache 和 committed-boundary trim/crop；继续保持无 cache 路线作为 golden。
5. 已增加不运行 ordinary 对照的 `dflash` 入口；validator 继续保留。
6. 在 310P 上先做 cache on/off 等价、稳定性和相同边界 benchmark。
7. 如果确实需要官方 sampling，再实现概率输出、rejection sampling 和 residual correction。
8. 最后用相同 checkpoint、prompt、K/T、dtype、warmup 和输出长度比较端到端性能。

在第 7 步前，只能声明 greedy DFlash，不应笼统写成支持官方全部解码模式。Draft cache 的代码和
CPU reduced-shape 门禁已经完成，但在真机 cache 等价和 profile 完成前仍不能声称性能闭合。

## 11. 文件映射

| 官方锁定文件 | 当前对应文件 |
| --- | --- |
| dflash/model.py 的 dflash_generate | models/dflash_v1/dflash_rollback_decode.py |
| dflash/model.py 的 DFlashDraftModel | models/dflash_v1/modeling_dflash.py |
| dflash/model.py 的 Draft/Target DynamicCache | DFlashDraftKVCache；Target DynamicCache transaction / NPU logical cursor |
| dflash/model_mlx.py 的 GDN capture/rollback | dflash_rollback_adapter.py 的 snapshot/replay；HIAI state bank |
| dflash/model_mlx.py 的 draft_cache | modeling_dflash.py 的 DFlashDraftKVCache |
| dflash/model.py 的 rejection sampling | 当前缺失 |
| 官方直接 generate/stream | run_rollback.py / run_npu.py 的 `dflash` 单跑；`validate` 为额外门禁 |

当前仓库中的 dflash_reference_decode_v1.py 是本项目早期 full-prefix correctness oracle，不是
官方完整 DFlash 的同义词，也不应拿它代表 z-lab/dflash 的正式 cache/rollback 实现。
