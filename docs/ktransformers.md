# Qwen3 MoE with standalone KTransformers

This integration targets **Qwen3-30B-A3B and Qwen3-235B-A22B**, using one NVIDIA GPU
and the LLAMAFILE CPU backend with GGUF expert weights. It calls `KTMoEWrapper`
directly; installing SGLang or `sglang-kt` is unnecessary.

## Placement and implementation

| Component | Weights | Execution |
| --- | --- | --- |
| Attention projections, Q/K norms, all RMSNorms, embeddings, LM head | Original BF16 safetensors, GPU | GPU |
| RoPE and KV cache | GPU buffers | GPU |
| MoE router (`mlp.gate`) and top-k | Original BF16 safetensors, GPU | GPU |
| All MoE experts (gate/up/down projections and SiLU) | GGUF, CPU | CPU |

`load_weight(..., skip_experts=True)` filters expert names **before** `get_tensor`;
the model never allocates CUDA expert parameters. KT reads the three expert tensors
per layer from the GGUF mmap and loads them into its CPU kernel. Only activations,
expert IDs, routing probabilities and results cross between CPU and GPU.
The router is distinct from the expert's `gate_proj`.

The adapter is `python/minisgl/moe/ktransformers.py`, selected by
`--moe-backend ktransformers`. Upstream source is a submodule at
`third_party/ktransformers`, pinned to `4882505c9a66a6784b3360a1b6ba9b23d53d0291`.
This version uses `gpu_experts_mask=None` for all-CPU experts, differing from the
older `num_gpu_experts` example in the
[standalone API documentation](https://github.com/kvcache-ai/ktransformers/blob/4882505c9a66a6784b3360a1b6ba9b23d53d0291/kt-kernel/README.md#direct-python-api-usage).
Edit the adapter for Mini-SGLang integration changes; edit the submodule and rebuild
`kt-kernel` for CPU kernel changes.

The adapter creates one `KTMoEWrapper` per layer, calls `load_weights()` once,
then calls `forward(hidden_states, topk_ids, topk_weights, cuda_stream)` per step.
KT handles GGUF loading, CPU buffers, submission, synchronization and result copies;
Mini-SGLang supplies routing and the current CUDA stream. Remote machine access
is not required to implement this integration; the checks below are optional lab verification.

Only BF16 GPU weights, LLAMAFILE, and TP=1 are supported here. Experts on GPU and
deferred experts are fixed to zero. CUDA graphs are disabled automatically.
The existing `fused` backend remains the default when KT is not selected.

## Clone and install (Linux x86-64)

Use a Linux machine with an AVX2-capable CPU, enough host RAM for the chosen GGUF
experts plus KT working buffers, and an NVIDIA GPU with a compatible CUDA toolkit
and driver. The commands below use CUDA 12.8 / PyTorch 2.9.1. `nvcc` must be on PATH.
235B still needs GPU memory for its non-expert parameters and attention workspace;
CPU offload does not eliminate this requirement.

```bash
git clone --branch codex/qwen3-kt-cpu-moe https://github.com/BoxuanYang/FlashKVMoE.git
cd FlashKVMoE
git submodule update --init third_party/ktransformers
# Only the dependencies needed to build kt-kernel; no SGLang checkout is needed.
git -C third_party/ktransformers submodule update --init --recursive \
  third_party/llama.cpp third_party/pybind11

conda create -n minisgl-kt python=3.12 -y
conda activate minisgl-kt
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build libhwloc-dev pkg-config
nvcc --version
nvidia-smi

python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e . 'transformers==4.57.3' \
  'flashinfer-python==0.5.3' 'sgl_kernel==0.3.17.post1' pytest
CPUINFER_USE_CUDA=1 CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_PARALLEL=8 \
  python -m pip install -v ./third_party/ktransformers/kt-kernel
python -m pip check
python -c 'from kt_kernel import KTMoEWrapper, kt_kernel_ext; assert hasattr(kt_kernel_ext.CPUInfer, "submit_with_cuda_stream"); print("KT CUDA stream API available")'
```

After editing KT source, rebuild with:

```bash
CPUINFER_USE_CUDA=1 CPUINFER_CPU_INSTRUCT=NATIVE CPUINFER_PARALLEL=8 \
  CPUINFER_FORCE_REBUILD=1 python -m pip install -v ./third_party/ktransformers/kt-kernel
```

## Weight directories

Use the existing local model directories:

```text
/data2/models/Qwen3-30B-A3B          # config, tokenizer and original BF16 safetensors
/data2/models/Qwen3-30B-A3B-GGUF     # one GGUF quantization, with all its shards
/data2/models/Qwen3-235B-A22B        # config, tokenizer and original BF16 safetensors
/data2/models/Qwen3-235B-A22B-GGUF    # matching 235B GGUF, with all its shards
```

Do not mix different quantizations or model revisions in a GGUF directory.
The BF16 and GGUF directories must represent the same model. A complete original
BF16 checkpoint is accepted, but its expert tensors are never read by Mini-SGLang.
`--kt-weight-path` can also name one complete GGUF file.

## Start 30B

```bash
CUDA_VISIBLE_DEVICES=2 python -m minisgl \
  --host 0.0.0.0 \
  --port 30000 \
  --model /data2/models/Qwen3-30B-A3B \
  --dtype bfloat16 \
  --moe-backend ktransformers \
  --kt-weight-path /data2/models/Qwen3-30B-A3B-GGUF \
  --kt-cpuinfer 128 \
  --kt-threadpool-count 2 \
  --kt-method LLAMAFILE \
  --attention-backend fi \
  --cuda-graph-max-bs 0 \
  --page-size 1 \
  --num-pages 2048 \
  --max-seq-len-override 2048 \
  --max-prefill-length 2048
```

## Start 235B

Stop the 30B process first when reusing the same GPU and port.

```bash
CUDA_VISIBLE_DEVICES=2 python -m minisgl \
  --host 0.0.0.0 \
  --port 30000 \
  --model /data2/models/Qwen3-235B-A22B \
  --dtype bfloat16 \
  --moe-backend ktransformers \
  --kt-weight-path /data2/models/Qwen3-235B-A22B-GGUF \
  --kt-cpuinfer 128 \
  --kt-threadpool-count 2 \
  --kt-method LLAMAFILE \
  --attention-backend fi \
  --cuda-graph-max-bs 0 \
  --page-size 1 \
  --num-pages 2048 \
  --max-seq-len-override 2048 \
  --max-prefill-length 2048
```

`128` CPU threads and `2` NUMA pools preserve the original configuration; adjust
them to the lab machine's physical cores and NUMA topology (`lscpu`). CUDA device
2 becomes logical device 0 inside the process.

Mini-SGLang uses `fi` for FlashInfer and `--cuda-graph-max-bs 0` to disable graph
capture. `--page-size 1 --num-pages 2048` gives a total KV capacity of 2048 tokens
(plus the engine's dummy page). Explicit pages override automatic memory-ratio
sizing, so `--memory-ratio 0.5` would not impose a 50% memory cap here.
The SGLang flags `--watchdog-timeout`, `--skip-server-warmup`,
`--kt-num-gpu-experts`, and `--kt-max-deferred-experts-per-token` are not needed
and are not accepted by this Mini-SGLang CLI.

## Verification

Run the tests before starting a server. The CUDA test creates a small GGUF and
compares real KT outputs with a PyTorch reference over different batch sizes and
layers. It requires no downloaded model weights.

```bash
CUDA_VISIBLE_DEVICES=2 python -m pytest tests/moe/test_ktransformers.py -o addopts= -q
```

Then run each real model through chunked prefill, batched decode, and a repeated
request. These commands also assert GPU non-expert/RoPE placement and CPU expert
masks. Run them sequentially, with the server stopped:

```bash
CUDA_VISIBLE_DEVICES=2 python tests/moe/smoke_qwen3_kt.py \
  --model /data2/models/Qwen3-30B-A3B \
  --kt-weight-path /data2/models/Qwen3-30B-A3B-GGUF

CUDA_VISIBLE_DEVICES=2 python tests/moe/smoke_qwen3_kt.py \
  --model /data2/models/Qwen3-235B-A22B \
  --kt-weight-path /data2/models/Qwen3-235B-A22B-GGUF
```

Once the server is running, test its API:

```bash
curl --fail-with-body http://127.0.0.1:30000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3","messages":[{"role":"user","content":"What is 2 + 2? /no_think"}],"temperature":0,"max_tokens":32}'
```

Startup checks assert that the model has no GPU expert parameters and that all
remaining model weights are on GPU, then log their memory usage. KT prints its
per-layer loading details. Inspect generated answers when validating your GGUF quantization.

Local validation: Python 3.12, PyTorch 2.9.1 and Transformers 4.57.3 on Windows;
18 tests passed, the real CUDA/KT test was skipped. Full 30B/235B inference and the
Linux source build have **not** been executed in this development environment.
