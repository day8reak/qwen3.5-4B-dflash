# 按现有 Qwen3.5 工程接入 DFlash target feature

## 本阶段边界

本阶段只让现有 Qwen3.5 target 按需输出 DFlash 所需的八层 hidden feature。
以下路径保持原样：

- `models/modeling_qwen3_5_hiai_nd.py`；
- 已有 GDN、attention 和 RMSNorm 自定义算子；
- target cache 的布局和更新；
- 普通 `inference.py` 生成流程；
- export wrapper 和 OM 图。

真实工程的 `models/modeling_qwen3_5.py` 没有放在当前 workspace，因此不能安全生成能
自动套用的行号 patch。这里提供的是三个稳定插入点，以及一个可以直接复制到现有
`dflash/` 目录的实现。不要用 Transformers main 分支文件覆盖当前已经能运行的
torch_npu target。

## 与现有目录的映射

```text
qwen3_5/
├── config/
│   ├── qwen3_5.yaml                    # 不改
│   └── dflash.yaml                     # 新增：feature contract
├── models/
│   ├── modeling_qwen3_5.py             # 仅增加三个 collector 插入点
│   ├── modeling_qwen3_5_hiai_nd.py     # 不改
│   ├── export_model_wrapper_qwen3_5.py # 不改
│   └── configuration_qwen3_5.py        # 第一阶段不要求改
├── dflash/
│   ├── __init__.py
│   └── target_features.py              # 新增
├── tests/
│   └── test_target_feature.py          # 新增
├── inference.py                        # 普通调用不改
└── Qwen3.5-4B_infer_golden_data_ifa/   # 不写入新产物
```

当前 workspace 的对应实现是
`model/qwen35_dflash/target_features.py`。交付包会将它映射成现有工程中的
`dflash/target_features.py`。

注意：`model/qwen35_dflash/` 只是本 workspace 为了测试和版本管理使用的内部源码路径，
不是要求现有工程改成这种布局。对外增量包始终以 `qwen3_5/` 为归档根目录；解压到现有
工程的上一级目录后，只新增上图中的 `dflash/`、`config/dflash.yaml`、测试和插桩说明，
不会替换已有的 `models/modeling_qwen3_5.py`。

## 插入点一：import 和 forward 参数

在 `models/modeling_qwen3_5.py` 中引入：

```python
from dflash.target_features import (
    DFlashCausalLMOutputWithPast,
    DFlashFeatureCollector,
    QWEN35_4B_DFLASH_TARGET_FEATURES,
)
```

`QWEN35_4B_DFLASH_TARGET_FEATURES` 的层号和维度来自 `dflash/config.py`，该文件还暴露
共享方案中的兼容映射：

```python
DFLASH_CONFIG = {
    "feature_layers": [1, 5, 9, 13, 17, 21, 25, 29],
    "feature_dim": 20480,
    "target_hidden_size": 2560,
    "target_num_hidden_layers": 32,
}
```

因此不要在 `modeling_qwen3_5.py` 中再写一份层号集合。

不要替换本地 `Qwen3_5ModelOutputWithPast`。在它已有字段的末尾增加：

```python
dflash_features: torch.Tensor | None = None
```

例如当前官方类型还带 `rope_deltas`；替换成普通 `BaseModelOutputWithPast` 会静默丢字段。

给 text model 的 `forward()` 增加显式、默认关闭的参数：

```python
output_dflash_features: bool = False,
detach_dflash_features: bool = True,
clone_dflash_features: bool = True,
```

第一阶段默认 `clone_dflash_features=True`，防止自定义 decoder/算子原地复用上一层
buffer 后污染已捕获值。collector 会一次预分配 `[B,S,20480]`，逐层写入对应 slice，
不会先保留八份 clone 再额外 `cat`。普通 forward 不会执行这些 copy。只有内部确认
各层都是 out-of-place，并通过 feature 对照门禁后，才可显式改为 `False`。

## 插入点二：decoder loop

进入 decoder loop 前创建一次 collector：

```python
dflash_collector = DFlashFeatureCollector(
    QWEN35_4B_DFLASH_TARGET_FEATURES,
    enabled=output_dflash_features,
    detach=detach_dflash_features,
    clone=clone_dflash_features,
)
```

现有 decoder layer 完成后、target 最终 RMSNorm 之前增加一行：

```python
for layer_idx, decoder_layer in enumerate(self.layers):
    hidden_states = decoder_layer(
        hidden_states,
        # 保留现有 torch_npu/GDN/cache 参数
    )
    dflash_collector.capture(layer_idx, hidden_states)

dflash_features = dflash_collector.finalize()
hidden_states = self.norm(hidden_states)
```

层号是 0-based decoder layer ID：`1,5,9,13,17,21,25,29`。这与事后从
Transformers `hidden_states[2,6,10,14,18,22,26,30]` 取值完全对应。不能捕获
decoder 输入，也不能捕获 final RMSNorm 后的结果。

## 插入点三：按需扩展返回值

base model 继续返回现有 `Qwen3_5ModelOutputWithPast`，只是把新字段传进去。字段为
`None` 时 Transformers `ModelOutput` 不会把它加入 tuple，因此普通 tuple 次序不变：

```python
return Qwen3_5ModelOutputWithPast(
    last_hidden_state=hidden_states,
    past_key_values=past_key_values,
    dflash_features=dflash_features,
    # 保留现有字段
)
```

如果本地类名不同，同样只在现有 dataclass 末尾增加字段，不要丢掉本地字段。

`Qwen3_5ForCausalLM.forward()` 同样增加这三个参数并原样传给 `self.model()`。如果本地
已经有 CausalLM output dataclass，就在该类末尾增加同名字段。否则普通路径继续返回
原 `CausalLMOutputWithPast`，只有 feature 路径返回：

```python
return DFlashCausalLMOutputWithPast(
    loss=loss,
    logits=logits,
    past_key_values=outputs.past_key_values,
    dflash_features=outputs.dflash_features,
    # 保留现有字段
)
```

如果本地支持 `return_dict=False`，关闭 flag 时 tuple 必须完全不变；开启 flag 时可以把
`dflash_features` 追加在最后，但要在内部 ABI 中固定这一位置。第一阶段推荐 golden
调用使用 `return_dict=True`，减少与现有自定义返回结构的耦合。

## `inference.py` 的最小调用

普通推理保持：

```python
output = model(input_ids=input_ids, use_cache=True)
```

生成 DFlash feature case 时才开启：

```python
with torch.inference_mode():
    output = model(
        input_ids=input_ids,
        use_cache=True,
        output_dflash_features=True,
    )
target_features = output.dflash_features
assert target_features.shape[-1] == 20480
```

不要同时设置 `output_hidden_states=True`；那会保留全部 32 层。也不要把 feature 写回
现有 `Qwen3.5-4B_infer_golden_data_ifa/`，运行产物应写入独立输出目录。

## 第一阶段门禁

内部拿到真实 target 后必须补三组比较：

1. 同一输入、同一 cache 初态，flag 关闭/开启的 logits 逐 bit 相同；
2. flag 关闭/开启后的所有 target KV、conv 和 recurrent state 逐 bit 相同；
3. selective feature 与一次诊断性 `output_hidden_states=True` 得到的
   `[2,6,10,14,18,22,26,30]` 拼接结果逐元素相同。

当前 portable fixture 已验证 collector 分支不会改变 target logits，并验证 shape、层序、
detach、clone 和缺层失败行为。它不能替代你的真实 torch_npu target/cache 门禁。

## 后续阶段

只有上述三组门禁通过后，才进入 target block verify、GDN speculative state、draft KV
cache 和 scheduler。第一阶段不应增加 `speculative_mode`，因为一个没有对应状态事务
语义的布尔 flag 很容易被误用成“已经支持回滚”。
