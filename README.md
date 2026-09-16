# FlashKVMoE

Single GPU Qwen3 MoE inference with Mini-SGLang and the KTransformers (KT) CPU expert backend. The model uses **one complete GGUF checkpoint for all weights**: KT runs quantized experts on the CPU, while the GPU weights are dequantized to BF16 and Attention runs with FlashInfer.

Supported models are Qwen3-30B-A3B and Qwen3-235B-A22B. Runtime requirements include Linux x86-64, an AVX2 CPU, a BF16-capable NVIDIA GPU, and the KT LLAMAFILE backend. Tensor parallelism and other model architectures are not supported by this integration.

The `--model` path or Hugging Face model ID supplies only the config and tokenizer. It does not supply model weights. `--kt-weight-path` must point to a **complete GGUF file** or a directory containing **every shard of one GGUF quantization**. A GGUF containing only experts is insufficient.

## Quick start: Qwen3-30B-A3B

Follow the [installation instructions](docs/ktransformers.md#1-克隆与安装) to install this repository and build the pinned KT kernel. Download a single GGUF checkpoint:

```bash
hf download Qwen/Qwen3-30B-A3B-GGUF \
  --include 'Qwen3-30B-A3B-Q4_K_M.gguf' \
  --local-dir /data2/models/Qwen3-30B-A3B-GGUF
```

Start the server. Hugging Face will fetch the Qwen3 config and tokenizer for `--model`; it will not download safetensors weights.

```bash
python -m minisgl \
  --model Qwen/Qwen3-30B-A3B \
  --kt-weight-path /data2/models/Qwen3-30B-A3B-GGUF \
  --dtype bfloat16 \
  --tp-size 1 \
  --attention-backend fi \
  --cuda-graph-max-bs 0 \
  --page-size 1 \
  --num-pages 2048 \
  --max-seq-len-override 2048 \
  --max-prefill-length 2048
```

`--moe-backend kt` is the only supported MoE backend and is selected by default. For Qwen3-235B-A22B, download all GGUF shards and point `--kt-weight-path` at their `Q4_K_M` directory. See the [model download and launch instructions](docs/ktransformers.md#2-权重) for the complete commands and hardware considerations.

## Verification

Run the GGUF loader and KT contract tests from the repository root:

```bash
python -m pytest tests/models/test_gguf.py tests/moe -o addopts= -v
```

The Linux CUDA tests exercise real KT experts and FlashInfer Attention. A full 30B/235B server run still needs validation on the target GPU machine.

This project builds on [Mini-SGLang](https://github.com/sgl-project/mini-sglang) and [KTransformers](https://github.com/kvcache-ai/ktransformers). See [docs/ktransformers.md](docs/ktransformers.md) for installation, configuration, and API request examples.
