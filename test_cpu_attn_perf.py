"""Compare GPU and CPU decode attention for GLM-4.5-Air layer 8."""

import os
import sys
import time
from pathlib import Path

# Keep the script directly runnable on the same GPU used by the MiniSGL command.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

if sys.platform != "linux":
    raise RuntimeError("Run this benchmark on the Linux CUDA/AVX512 target machine.")

import torch
from flashinfer import single_decode_with_kv_cache
from kt_kernel import kt_kernel_ext as ext
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.layers.marlin import pack_marlin
from minisgl.models.config import ModelConfig
from minisgl.models.gguf import GGUFWeights
from minisgl.models.glm4_moe import Glm4MoeAttention
from minisgl.utils import torch_dtype
from transformers import AutoConfig

MODEL_PATH = "/data1/models/GLM-4.5-Air-GGUF"
WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
LAYER_NUMBER = 8
LAYER_INDEX = LAYER_NUMBER - 1  # Human layer 8 is GGUF blk.7.

SEQUENCE_LENGTHS = [200, 500, 1000, 2048]
REPEATS = 10
WARMUP = 3
CPU_THREADS = 64
NUMA_POOLS = 2
BLOCK_LENGTH = 128
DEVICE = torch.device("cuda:0")
SEED = 42
OUTPUT_PATH = Path("Attn_perf.txt")


def pack_weight(checkpoint: GGUFWeights, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Load one GGUF matrix, pack it for Marlin, and move it to the GPU."""
    weight = checkpoint._dequantize(name, torch.device("cpu"), torch.bfloat16, 16 << 20)
    packed, scales = pack_marlin(weight)
    return packed.to(DEVICE), scales.to(DEVICE)


def load_attention_layer(config: ModelConfig) -> Glm4MoeAttention:
    """Load only layer 8's QKV and output projections."""
    checkpoint = GGUFWeights(WEIGHT_PATH, config)
    prefix = f"blk.{LAYER_INDEX}"

    qkv = torch.cat(
        [
            checkpoint._dequantize(
                f"{prefix}.attn_{name}.weight",
                torch.device("cpu"),
                torch.bfloat16,
                16 << 20,
            )
            for name in ("q", "k", "v")
        ]
    )
    qkv_weight, qkv_scales = pack_marlin(qkv)
    del qkv

    qkv_bias = torch.cat(
        [
            checkpoint._dequantize(
                f"{prefix}.attn_{name}.bias", DEVICE, torch.bfloat16, 16 << 20
            )
            for name in ("q", "k", "v")
        ]
    )
    output_weight, output_scales = pack_weight(
        checkpoint, f"{prefix}.attn_output.weight"
    )

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layer = Glm4MoeAttention(config, LAYER_INDEX)
    layer.load_state_dict(
        {
            "qkv_proj.weight": qkv_weight.to(DEVICE),
            "qkv_proj.scales": qkv_scales.to(DEVICE),
            "qkv_bias": qkv_bias,
            "o_proj.weight": output_weight,
            "o_proj.scales": output_scales,
        }
    )
    return layer


def make_cpu_pool():
    pool_config = ext.WorkerPoolConfig()
    pool_config.subpool_count = NUMA_POOLS
    pool_config.subpool_numa_map = list(range(NUMA_POOLS))
    pool_config.subpool_thread_count = [
        CPU_THREADS // NUMA_POOLS + (pool < CPU_THREADS % NUMA_POOLS)
        for pool in range(NUMA_POOLS)
    ]
    cpu_infer = ext.CPUInfer(pool_config)
    return cpu_infer, cpu_infer.backend_


@torch.inference_mode()
def make_inputs(layer: Glm4MoeAttention, config: ModelConfig, sequence_length: int):
    """Use layer 8 for the current Q/K/V and synthesize the preceding KV history."""
    hidden = torch.randn(1, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    rms = hidden.float().square().mean(-1, keepdim=True).add(config.rms_norm_eps).sqrt()
    hidden = (hidden.float() / rms).bfloat16()

    qkv = layer.qkv_proj.forward(hidden) + layer.qkv_bias
    q_size = config.num_qo_heads * config.head_dim
    kv_size = config.num_kv_heads * config.head_dim
    query, current_key, current_value = qkv.split([q_size, kv_size, kv_size], dim=-1)

    position = torch.tensor([sequence_length - 1], device=DEVICE, dtype=torch.int32)
    query, current_key = layer.attn.rotary.forward(position, query, current_key)

    keys = torch.randn(
        1,
        sequence_length,
        config.num_kv_heads,
        config.head_dim,
        device=DEVICE,
        dtype=torch.bfloat16,
    )
    values = torch.randn_like(keys)
    keys[:, -1].copy_(current_key.view(1, config.num_kv_heads, config.head_dim))
    values[:, -1].copy_(current_value.view(1, config.num_kv_heads, config.head_dim))

    query = query.view(1, config.num_qo_heads, config.head_dim).contiguous()
    return query, keys, values


def gpu_attention(query: torch.Tensor, keys: torch.Tensor, values: torch.Tensor):
    # This is FlashInfer's single-request decode path. GQA=12 uses tensor cores,
    # matching MiniSGL's FlashInfer backend choice for GLM-4.5-Air.
    return single_decode_with_kv_cache(
        query[0],
        keys[0],
        values[0],
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_tensor_cores=True,
    )


def make_cpu_attention(query, keys, values, sequence_length, pool):
    """Create and populate the one-layer CPU paged KV cache."""
    block_count = (sequence_length + BLOCK_LENGTH - 1) // BLOCK_LENGTH
    config = ext.dense_kvcache.KVCacheConfig(
        1,
        8,
        96,
        128,
        BLOCK_LENGTH,
        ext.kvcache.ggml_type.BF16,
        block_count,
        1,
        CPU_THREADS,
    )
    cache = ext.dense_kvcache.KVCache(config)

    query_cpu = query.cpu().contiguous()
    keys_cpu = keys.cpu().contiguous()
    values_cpu = values.cpu().contiguous()
    output_cpu = torch.empty_like(query_cpu)
    lse_cpu = torch.empty(1, 96, dtype=torch.float32)
    block_table = torch.arange(block_count, dtype=torch.int32).view(1, -1)
    empty_length = torch.zeros(1, dtype=torch.int32)
    cache_length = torch.tensor([sequence_length], dtype=torch.int32)

    cache.update_kvcache_bf16(
        keys_cpu.data_ptr(),
        values_cpu.data_ptr(),
        0,
        block_table.data_ptr(),
        1,
        block_count,
        empty_length.data_ptr(),
        sequence_length,
        pool,
    )

    def attention():
        cache.attn(
            query_cpu.data_ptr(),
            output_cpu.data_ptr(),
            lse_cpu.data_ptr(),
            0,
            0,
            1,
            1,
            block_count,
            block_table.data_ptr(),
            cache_length.data_ptr(),
            pool,
        )

    return attention, output_cpu


def average_ms(operation, synchronize=None) -> float:
    for _ in range(WARMUP):
        operation()
    if synchronize:
        synchronize()

    start = time.perf_counter_ns()
    for _ in range(REPEATS):
        operation()
    if synchronize:
        synchronize()
    return (time.perf_counter_ns() - start) / REPEATS / 1e6


def capture_gpu_graph(query, keys, values):
    # Warm up allocations before capture. The returned output is updated by replay().
    for _ in range(WARMUP):
        gpu_attention(query, keys, values)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = gpu_attention(query, keys, values)
    return graph, output


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.set_num_threads(1)
    torch.cuda.set_device(DEVICE)
    set_tp_info(0, 1)
    set_rope_device(DEVICE)

    model_config = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL_PATH))
    expected_shape = (96, 8, 128)
    actual_shape = (
        model_config.num_qo_heads,
        model_config.num_kv_heads,
        model_config.head_dim,
    )
    if not model_config.is_glm4_moe or actual_shape != expected_shape:
        raise ValueError(f"Expected GLM-4.5-Air attention shape {expected_shape}, got {actual_shape}")

    print(f"Loading GLM-4.5-Air layer {LAYER_NUMBER} attention weights ...", flush=True)
    layer = load_attention_layer(model_config)
    cpu_infer, pool = make_cpu_pool()  # Keep cpu_infer alive while pool is in use.
    _ = cpu_infer

    gpu_name = torch.cuda.get_device_name(DEVICE)
    results = []

    for sequence_length in SEQUENCE_LENGTHS:
        query, keys, values = make_inputs(layer, model_config, sequence_length)
        cpu_attention, cpu_output = make_cpu_attention(
            query, keys, values, sequence_length, pool
        )

        gpu_output = gpu_attention(query, keys, values)
        gpu_without_graph_ms = average_ms(
            lambda: gpu_attention(query, keys, values), torch.cuda.synchronize
        )

        graph, graph_output = capture_gpu_graph(query, keys, values)
        gpu_with_graph_ms = average_ms(graph.replay, torch.cuda.synchronize)
        cpu_ms = average_ms(cpu_attention)
        cpu_attention()

        for expected in (gpu_output, graph_output):
            torch.testing.assert_close(
                cpu_output.float(),
                expected.unsqueeze(0).cpu().float(),
                rtol=0.03,
                atol=0.03,
            )

        results.append(
            (sequence_length, cpu_ms, gpu_with_graph_ms, gpu_without_graph_ms)
        )
        graph.reset()

    lines = [
        "GLM-4.5-Air layer 8 decode attention",
        f"GPU: {gpu_name}",
        f"CPU threads: {CPU_THREADS}, NUMA pools: {NUMA_POOLS}, repeats: {REPEATS}",
        "Unit: ms; batch size: 1",
        "Timing excludes QKV projection, RoPE, weight loading and data transfers.",
        "",
        (
            f"{'Sequence':>10} {'CPU':>14} "
            f"{'GPU with CUDA Graph':>22} {'GPU without CUDA Graph':>24}"
        ),
        "-" * 73,
    ]
    lines.extend(
        f"{length:>10} {cpu_ms:>14.4f} {gpu_graph_ms:>22.4f} {gpu_ms:>24.4f}"
        for length, cpu_ms, gpu_graph_ms, gpu_ms in results
    )
    report = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"Results written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
