"""Measure GLM-4.5-Air decode attention with CUDA Graph."""

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import torch
from flashinfer import single_decode_with_kv_cache

DEVICE = torch.device("cuda:0")
SEQUENCE_LENGTHS = [2000 * i for i in range(1, 80)]
OUTPUT_PATH = Path("gpu_perf.txt")

QUERY_HEADS = 96
KV_HEADS = 8
HEAD_DIM = 128

WARMUP = 10
REPEATS = 100
SEED = 42


def make_inputs(sequence_length: int):
    """Create one synthetic GLM-4.5-Air query and its KV history."""
    shape = (sequence_length, KV_HEADS, HEAD_DIM)
    query = torch.randn(QUERY_HEADS, HEAD_DIM, device=DEVICE, dtype=torch.bfloat16)
    keys = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    values = torch.randn(shape, device=DEVICE, dtype=torch.bfloat16)
    return query, keys, values


def attention(query, keys, values):
    return single_decode_with_kv_cache(
        query,
        keys,
        values,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_tensor_cores=True,
    )


def capture_graph(query, keys, values):
    """Capture one attention call after FlashInfer has finished its setup work."""
    for _ in range(WARMUP):
        attention(query, keys, values)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = attention(query, keys, values)
    return graph, output


def measure_gpu_ms(graph: torch.cuda.CUDAGraph) -> float:
    """Measure average graph execution time on the GPU with CUDA Events."""
    for _ in range(WARMUP):
        graph.replay()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(REPEATS):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / REPEATS


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(DEVICE)

    gpu_name = torch.cuda.get_device_name(DEVICE)
    results = []

    for sequence_length in SEQUENCE_LENGTHS:
        query, keys, values = make_inputs(sequence_length)
        graph, output = capture_graph(query, keys, values)
        results.append((sequence_length, measure_gpu_ms(graph)))

        # Captured tensors stay alive until every replay has finished.
        graph.reset()
        del graph, output, query, keys, values
        torch.cuda.empty_cache()

    lines = [
        "GLM-4.5-Air decode attention",
        f"GPU: {gpu_name}",
        f"Batch size: 1, CUDA Graph: on, repeats: {REPEATS}",
        "Timing uses CUDA Events and covers attention only.",
        "",
        f"{'Sequence':>10} {'GPU ms':>14}",
        "-" * 25,
    ]
    lines.extend(
        f"{sequence_length:>10} {gpu_ms:>14.6f}"
        for sequence_length, gpu_ms in results
    )

    report = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"Results written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
