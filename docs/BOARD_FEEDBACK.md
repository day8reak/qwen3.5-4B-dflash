# 内部 310P 反馈格式

请回传一个目录（不要只贴结论），至少包含：

```text
manifest.json
preflight.log
ordinary.json
mtp.json
compare.json
artifact-sha256.txt
runtime.log
```

`manifest.json` 至少记录：

- Git commit；
- checkpoint revision 和所有输入文件 SHA-256；
- 具体设备（例如 Atlas 300I Duo / Atlas 200I Pro）、device ID；
- CANN、驱动、固件、内部框架/插件版本；
- dtype、prompt token IDs、max-new-tokens、K；
- backend ID，以及 CPU fallback 是否严格关闭；
- ordinary/MTP 每轮 token IDs；
- drafted、accepted、rejected、fallback counters；
- 10 轮原始耗时（准确率先通过，性能仅作观察）。

如果失败，请保留第一个失败 case 的主模型 target Top1、draft proposals、每个 verify
row Top1、接受长度和 NaN/Inf 统计。这样下一轮可以区分 main 图误差、MTP shift/权重
问题和 scheduler 问题。

固定 `S1/P1` core case 还请原样回传三个输出：`mtp_hidden`、`present_key`、
`present_value`，不要只给 allclose 结论。记录输入顺序、输出文件 SHA-256、转换后
artifact SHA-256、是否出现任何 Host/CPU fallback，以及与 case 内 FP16 expected 和
BF16 reference 分别得到的 max abs、relative L2、cosine、NaN/Inf。
