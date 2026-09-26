# ShareGPT

目录只包含两个纯 JSON object：

- `data.json`：92,866 个 request，格式为 `"req_n": "prompt 内容"`。
- `seq_length.json`：相同 key 对应的整数 token 长度。

来源是 `ShareGPT_V3_unfiltered_cleaned_split.json`：
<https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered>。

采用 vLLM ShareGPT benchmark 的标准口径：保留至少有两轮消息的 conversation，并取
第一条 human message 作为推理 prompt。conversation ID、GPT 回复、后续对话轮次和其他
属性均未保留。为保留 benchmark 的请求分布，没有按 prompt 文本去重；长度范围为
1–163,254 tokens。

Tokenizer：`zai-org/GLM-4.5-Air@a24ceef6ce4f3536971efe9b778bdaa1bab18daa`，
不添加 BOS、EOS 或 chat template。

| 文件 | SHA-256 |
| --- | --- |
| `data.json` | `ce5418ffdf499298f9ba0d28f213db4636ee91eeea19a90ea8f43a714235956b` |
| `seq_length.json` | `a3798540460d8cce5c0b9022993f26fe7647ec784772a70ac348961bcdc0ab82` |
