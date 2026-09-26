# LongBench v2

目录只包含两个纯 JSON object：

- `data.json`：503 个完整评测 prompt，格式为 `"req_n": "prompt 内容"`。
- `seq_length.json`：相同 key 对应的整数 token 长度。

来源：<https://huggingface.co/datasets/zai-org/LongBench-v2>。

每个 prompt 严格按照 LongBench v2 官方 `prompts/0shot.txt` 构造，包含完整 context、
question、A–D 四个选项及输出格式要求。原始 `_id`、domain、difficulty、answer 等属性
均未保留。长度范围为 10,081–4,145,061 tokens。

Tokenizer：`zai-org/GLM-4.5-Air@a24ceef6ce4f3536971efe9b778bdaa1bab18daa`，
不添加 BOS、EOS 或 chat template。

| 文件 | SHA-256 |
| --- | --- |
| `data.json` | `6884761adbafbdc00f2dc16d1e170a7a19ebb7ec706c400c4ad6817874634a90` |
| `seq_length.json` | `f8c73b29489b734559e83f9544368d232efa48117c7ad75bd47b4031f23ed87b` |
