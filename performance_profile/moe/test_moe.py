"""Measure one GLM-4.5-Air routed MoE layer with KTransformers."""

import os
import statistics
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import torch
from minisgl.models.config import ModelConfig
from transformers import AutoConfig

MODEL_PATH = "/data1/models/GLM-4.5-Air-GGUF"
WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"

LAYER_NUMBER = 8
LAYER_INDEX = LAYER_NUMBER - 1
BATCH_SIZES = range(1, 257)

CPU_THREADS = 64
THREAD_POOL_COUNT = 2
WARMUP = 3
REPEATS = 40
SEED = 42

DEVICE = torch.device("cuda:0")
OUTPUT_PATH = Path(__file__).with_name("moe_perf.txt")


def load_moe(config: ModelConfig):
    """Load only layer 8's routed-expert weights."""
    from kt_kernel import KTMoEWrapper

    wrapper = KTMoEWrapper(
        layer_idx=LAYER_INDEX,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=CPU_THREADS,
        threadpool_count=THREAD_POOL_COUNT,
        weight_path=WEIGHT_PATH,
        chunked_prefill_size=max(BATCH_SIZES),
        method="LLAMAFILE",
        max_deferred_experts_per_token=0,
    )
    expert_map = torch.arange(config.num_experts, dtype=torch.int32)
    wrapper.load_weights(expert_map)
    return wrapper


def make_inputs(config: ModelConfig, batch_size: int):
    """Use identical hidden states and random, non-repeating experts per token."""
    one_token = torch.randn(1, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    hidden_states = one_token.expand(batch_size, -1).contiguous()

    random_scores = torch.rand(batch_size, config.num_experts, device=DEVICE)
    expert_ids = random_scores.topk(config.num_experts_per_tok, dim=-1, sorted=False).indices
    expert_weights = torch.rand(batch_size, config.num_experts_per_tok, device=DEVICE)
    expert_weights /= expert_weights.sum(dim=-1, keepdim=True)
    return hidden_states, expert_ids, expert_weights


def measure_ms(wrapper, hidden_states, expert_ids, expert_weights, stream) -> float:
    """Time KT submit, CPU routed experts, sync, and output H2D."""
    from kt_kernel import KTMoEWrapper
    from kt_kernel.experts_base import KExpertsCPUBuffer

    KTMoEWrapper.clear_buffer_cache()
    KTMoEWrapper.set_capture_batch_sizes([len(hidden_states)])

    def stage_inputs():
        # This is the only input D2H path. Complete it before capture/timing.
        wrapper.copy_inputs_to_cpu_buffers(hidden_states, expert_ids, expert_weights)
        stream.synchronize()

    def run_cpu_moe():
        # Use KT's synchronous pinned-buffer path instead of a CUDA host node.
        # This constructs, submits, and synchronizes a fresh CPU task every time.
        wrapper.run_pinned_forward_sync(hidden_states, stream.cuda_stream)

    stream.wait_stream(torch.cuda.current_stream(DEVICE))
    with torch.cuda.stream(stream):
        stage_inputs()

        buffers = KExpertsCPUBuffer.get_buffer(
            hidden_states.view(-1, hidden_states.shape[-1]),
            wrapper.num_experts_per_tok,
        )
        current_slot = wrapper.layer_idx % KExpertsCPUBuffer.buffer_depth
        output_cpu = buffers[4][current_slot]

        # Warm both the CPU MoE and the pinned-output H2D before capture.
        for _ in range(WARMUP):
            run_cpu_moe()
            output = wrapper.copy_forward_output_to_device(hidden_states)
        stream.synchronize()

        # Re-stage and compute the reference before capture. The graph contains
        # only the routed output H2D; it has no input D2H or fragile host nodes.
        stage_inputs()
        run_cpu_moe()
        reference_cpu = output_cpu.clone()
        reference_nonfinite = torch.count_nonzero(~torch.isfinite(reference_cpu)).item()
        if reference_nonfinite:
            raise RuntimeError(
                "Eager KT CPU MoE produced "
                f"{reference_nonfinite}/{reference_cpu.numel()} non-finite values before graph replay"
            )
        if torch.count_nonzero(reference_cpu).item() == 0:
            raise RuntimeError("Eager KT CPU MoE produced an all-zero reference output")

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            output = wrapper.copy_forward_output_to_device(hidden_states)

        def verify_fresh_output(phase: str):
            nonzero = torch.count_nonzero(output_cpu).item()
            if nonzero == 0:
                raise RuntimeError(
                    f"CUDA Graph {phase} left the cleared CPU output unchanged; "
                    "the KT submit host callback did not produce a result"
                )
            try:
                torch.testing.assert_close(output_cpu, reference_cpu, rtol=1e-3, atol=1e-3)
            except AssertionError as error:
                raise RuntimeError(
                    f"CUDA Graph {phase} CPU output differs from the eager KT reference"
                ) from error

        for warmup_index in range(WARMUP):
            # A finite sentinel avoids contaminating implementations that read
            # the destination while still detecting a skipped CPU task.
            output_cpu.zero_()
            run_cpu_moe()
            graph.replay()
            stream.synchronize()
            verify_fresh_output(f"warmup {warmup_index + 1}")

        samples_ms = []
        for repeat_index in range(REPEATS):
            output_cpu.zero_()
            start = time.perf_counter()
            run_cpu_moe()
            graph.replay()
            stream.synchronize()
            samples_ms.append((time.perf_counter() - start) * 1000)
            verify_fresh_output(f"measurement {repeat_index + 1}")

        latency_ms = statistics.mean(samples_ms)

    graph.reset()
    del graph, output
    KTMoEWrapper.clear_buffer_cache()
    return latency_ms


def save_results(gpu_name: str, results: list[tuple[int, float]]):
    lines = [
        f"GLM-4.5-Air layer {LAYER_NUMBER} routed MoE",
        f"GPU: {gpu_name}",
        f"KTransformers: {CPU_THREADS} CPU threads, {THREAD_POOL_COUNT} thread pools",
        f"CUDA Graph: on, warmup: {WARMUP}, repeats: {REPEATS}",
        "Timed: KT task submit/sync, CPU routed experts, and routed output H2D.",
        "Excluded: shared expert, router, and all input D2H copies.",
        "Hidden states, expert IDs, and expert weights are copied to KT pinned CPU buffers before capture/timing.",
        "Timer: perf_counter around one KT synchronous CPU MoE, output-H2D graph replay, and stream.synchronize.",
        "CUDA Graph contains output H2D; CPU submit/sync runs immediately before each paired replay.",
        "Every replay clears and checks the CPU output against an eager reference.",
        "Expert IDs are random and unique per token.",
        "",
        f"{'Batch size':>10} {'MoE ms':>14}",
        "-" * 25,
    ]
    lines.extend(f"{batch_size:>10} {latency_ms:>14.6f}" for batch_size, latency_ms in results)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if os.environ.get("KT_FORCE_SYNC_SUBMIT") == "1":
        raise RuntimeError("Unset KT_FORCE_SYNC_SUBMIT; KT CUDA Graph needs stream callbacks.")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.cuda.set_device(DEVICE)

    config = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL_PATH))
    if not config.is_glm4_moe:
        raise ValueError("Expected a GLM-4 MoE model")
    if LAYER_INDEX < config.first_k_dense_replace:
        raise ValueError(f"Layer {LAYER_NUMBER} is dense, not MoE")

    print(f"Loading GLM-4.5-Air layer {LAYER_NUMBER} MoE weights ...", flush=True)
    wrapper = load_moe(config)
    stream = torch.cuda.Stream()
    gpu_name = torch.cuda.get_device_name(DEVICE)
    results = []

    print(f"{'Batch size':>10} {'MoE ms':>14}", flush=True)
    print("-" * 25, flush=True)
    for batch_size in BATCH_SIZES:
        inputs = make_inputs(config, batch_size)
        latency_ms = measure_ms(wrapper, *inputs, stream)
        results.append((batch_size, latency_ms))
        save_results(gpu_name, results)
        print(f"{batch_size:>10} {latency_ms:>14.6f}", flush=True)

    torch.cuda.current_stream().wait_stream(stream)
    print(f"Results written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
