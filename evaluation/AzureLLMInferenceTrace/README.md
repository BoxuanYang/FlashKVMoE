# Azure LLM Inference Trace

目录只包含两个纯 JSON object：

- `data.json`：19,366 个 request，格式为 `"req_n": "prompt 内容"`。
- `seq_length.json`：相同 key 对应的整数 token 长度。

来源是 Microsoft Azure 官方 2023 Conversation trace：
<https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_conv.csv>

Azure 因客户隐私不公开真实 prompt，只公开 `ContextTokens`。因此 `data.json` 使用可被
GLM-4.5-Air tokenizer 精确编码为指定长度的合成文本（重复 `" t"`），不是用户的真实
prompt。每条合成 prompt 都已实际送入 tokenizer 验证，长度与原 trace 的
`ContextTokens` 完全一致；长度范围为 2–14,050 tokens。

除 `req_n` 和 prompt/长度外，不保留时间戳、生成长度或其他字段。

Tokenizer：`zai-org/GLM-4.5-Air@a24ceef6ce4f3536971efe9b778bdaa1bab18daa`，
不添加 BOS、EOS 或 chat template。

| 文件 | SHA-256 |
| --- | --- |
| `data.json` | `6ceeb9171656858865fc24740cef6340cf95810f4b1882ea2b9ebe5992a25fad` |
| `seq_length.json` | `77d92a104779fb4ef5456038a55a452db6b9e90f8d143d6eda615e1979401cff` |
