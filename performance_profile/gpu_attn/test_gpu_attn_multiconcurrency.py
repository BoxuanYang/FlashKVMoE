"""Measure the GLM GPU overlap path at 64K total KV and one short baseline."""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import torch
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.layers import set_rope_device
from minisgl.models.config import ModelConfig
from test_gpu_single_attn_perf import (
    ATTN_BACKEND,
    DEVICE,
    LAYER_INDEX,
    LAYER_NUMBER,
    MODEL_PATH,
    PAGE_SIZE,
    REPEATS,
    WARMUP,
    capture_graph,
    gpu_overlap_path,
    load_decoder_layer,
    measure_gpu_ms,
)
from transformers import AutoConfig

TOTAL_CONTEXT = 64 * 1024
SHORT_SEQUENCE_CONFIG = (1, 10)
CONFIGS = [
    (1, 64 * 1024),
    (2, 32 * 1024),
    (4, 16 * 1024),
    (8, 8 * 1024),
    (16, 4 * 1024),
    (32, 2 * 1024),
    (64, 1 * 1024),
]
OUTPUT_PATH = Path(__file__).with_name("multiconcurrency.txt")
SEED = 42


def initialize_kv_cache(context: Context, config: ModelConfig):
    """Allocate 64K KV slots for layer 7."""
    num_pages = (TOTAL_CONTEXT + PAGE_SIZE - 1) // PAGE_SIZE
    context.kv_cache = MHAKVCache(
        num_kv_heads=config.num_kv_heads,
        num_layers=LAYER_INDEX + 1,
        head_dim=config.head_dim,
        num_pages=num_pages,
        page_size=PAGE_SIZE,
        dtype=torch.bfloat16,
        device=DEVICE,
    )
    context.kv_cache.k_cache(LAYER_INDEX).zero_()
    context.kv_cache.v_cache(LAYER_INDEX).zero_()


def make_inputs(config: ModelConfig, batch_size: int):
    hidden = torch.randn(
        batch_size, config.hidden_size, device=DEVICE, dtype=torch.bfloat16
    )
    return hidden, torch.randn_like(hidden)


def make_decode_batch(context: Context, batch_size: int, sequence_length: int):
    """Give every request its own contiguous section of the KV cache."""
    total_context = batch_size * sequence_length
    if total_context > TOTAL_CONTEXT:
        raise ValueError(f"Total context {total_context} exceeds {TOTAL_CONTEXT}")

    requests = [
        Req(
            input_ids=torch.zeros(sequence_length, dtype=torch.int32),
            table_idx=index,
            cached_len=sequence_length - 1,
            output_len=1,
            uid=index,
            sampling_params=None,  # type: ignore[arg-type]
            cache_handle=None,  # type: ignore[arg-type]
        )
        for index in range(batch_size)
    ]
    batch = Batch(requests, "decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.zeros(batch_size, device=DEVICE, dtype=torch.int32)
    batch.positions = torch.full(
        (batch_size,), sequence_length - 1, device=DEVICE, dtype=torch.int32
    )
    batch.out_loc = (
        torch.arange(1, batch_size + 1, device=DEVICE, dtype=torch.int32)
        * sequence_length
        - 1
    )
    context.page_table = torch.arange(
        total_context, device=DEVICE, dtype=torch.int32
    ).view(batch_size, sequence_length)
    return batch


@torch.inference_mode()
def main():
    if ATTN_BACKEND not in ("fi", "fa"):
        raise ValueError(f"ATTN_BACKEND must be 'fi' or 'fa', got {ATTN_BACKEND!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(DEVICE)
    set_tp_info(0, 1)
    set_rope_device(DEVICE)

    context = Context(page_size=PAGE_SIZE)
    set_global_ctx(context)
    config = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL_PATH))
    if any(batch * sequence != TOTAL_CONTEXT for batch, sequence in CONFIGS):
        raise ValueError("Every multiconcurrency configuration must total 64K tokens")
    if config.rotary_config.max_position < TOTAL_CONTEXT:
        config = replace(
            config,
            rotary_config=replace(
                config.rotary_config, max_position=TOTAL_CONTEXT
            ),
        )

    expected_shape = (96, 8, 128)
    actual_shape = (config.num_qo_heads, config.num_kv_heads, config.head_dim)
    if not config.is_glm4_moe or actual_shape != expected_shape:
        raise ValueError(
            f"Expected GLM-4.5-Air attention shape {expected_shape}, got {actual_shape}"
        )

    print(f"Attention backend: {ATTN_BACKEND}", flush=True)
    print(f"Loading GLM-4.5-Air layer {LAYER_NUMBER} GPU weights ...", flush=True)
    layer = load_decoder_layer(config)
    initialize_kv_cache(context, config)
    gpu_name = torch.cuda.get_device_name(DEVICE)
    results = []

    for batch_size, sequence_length in [SHORT_SEQUENCE_CONFIG, *CONFIGS]:
        hidden, residual = make_inputs(config, batch_size)
        batch = make_decode_batch(context, batch_size, sequence_length)
        backend = create_attention_backend(ATTN_BACKEND, config)
        context.attn_backend = backend
        backend.init_capture_graph(
            max_seq_len=sequence_length, bs_list=[batch_size]
        )
        backend.prepare_for_capture(batch)

        operation = partial(gpu_overlap_path, layer, hidden, residual)
        with context.forward_batch(batch):
            graph, output = capture_graph(operation)
            # Match server replay preparation: FA capture uses placeholder KV lengths.
            backend.prepare_metadata(batch)
            backend.prepare_for_replay(batch)
            latency_ms = measure_gpu_ms(graph)
        total_context = batch_size * sequence_length
        results.append((batch_size, sequence_length, total_context, latency_ms))
        print(
            f"batch={batch_size:>2}, seq/request={sequence_length:>6}: "
            f"{latency_ms:.6f} ms",
            flush=True,
        )

        graph.reset()
        del context.attn_backend
        del graph, output, operation, hidden, residual, batch, backend
        torch.cuda.empty_cache()

    baseline = results[0]
    multiconcurrency_results = results[1:]
    lines = [
        f"GLM-4.5-Air layer {LAYER_NUMBER} GPU overlap path",
        f"GPU: {gpu_name}",
        f"Attention backend: {ATTN_BACKEND}",
        f"CUDA Graph: on, warmup: {WARMUP}, repeats: {REPEATS}",
        "Includes norms, attention, shared expert and MoE router; excludes routed experts.",
        "",
        "## Short-sequence baseline",
        "## Batch   Seq/request   Total KV   GPU Attention (ms)",
        f"{baseline[0]:>8} {baseline[1]:>13} {baseline[2]:>10} {baseline[3]:>20.6f}",
        "",
        "## Fixed 64K total context",
        "## Batch   Seq/request   Total KV   GPU Attention (ms)",
    ]
    lines.extend(
        f"{batch_size:>8} {sequence_length:>13} {total_context:>10} {latency_ms:>20.6f}"
        for batch_size, sequence_length, total_context, latency_ms in multiconcurrency_results
    )
    report = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"Results written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
