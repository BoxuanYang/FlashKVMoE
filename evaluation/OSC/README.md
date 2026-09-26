# OpenAI Summarization Comparisons（OSC）

目录内的数据已经整理为两个纯 JSON object：

- `data.json`：21,471 个去重后的 prompt，格式为 `"req_n": "prompt 内容"`。
- `seq_length.json`：与 `data.json` 使用完全相同的 key，格式为
  `"req_n": token_length`。

示例：

```json
{
  "req_0": "prompt 内容",
  "req_1": "另一个 prompt 内容"
}
```

处理口径：

- 来源为 OpenAI `summarize_from_feedback` 的全部 comparisons train 和 validation 数据。
- prompt 取 `info.post`；当它为空时取 `info.article`。
- 按完整 prompt 文本去重，顺序为 train 在前、validation 在后。
- `data.json` 不含标签、候选摘要、split、worker、batch 等任何其他属性。
- `seq_length.json` 不含 tokenizer 信息或其他元数据，值只有整数。
- token 长度使用
  `zai-org/GLM-4.5-Air@a24ceef6ce4f3536971efe9b778bdaa1bab18daa` tokenizer，
  且不添加 BOS、EOS 或 chat template；最短 1 token，最长 1,960 tokens。

校验：

| 文件 | SHA-256 |
| --- | --- |
| `data.json` | `635772a69a0603d5a0866ec50570e64c0e0e4dc86161034c8b6577f654d1dde3` |
| `seq_length.json` | `fd2dcdd250223fb9e28326fe971353f59179676a522569a5a1d6689c5112aab0` |
