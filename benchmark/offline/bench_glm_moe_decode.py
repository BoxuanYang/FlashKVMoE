"""Single-layer GLM-4.5-Air MoE decode benchmark (Linux, CUDA, KT LLAMAFILE)."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import platform
import random
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_BATCH_SIZES = [1, 2, 4, 8, 16, 24, 32, 64, 128]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="zai-org/GLM-4.5-Air", help="HF config ID/directory")
    parser.add_argument(
        "--kt-weight-path", required=True, help="Complete GGUF file/shard directory"
    )
    parser.add_argument("--layer-index", type=int, default=7, help="Zero-based GGUF block index")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=DEFAULT_BATCH_SIZES)
    parser.add_argument("--kt-cpuinfer", type=int, default=64, help="Total KT worker threads")
    parser.add_argument("--kt-threadpool-count", type=int, default=2, help="NUMA nodes 0..N-1")
    parser.add_argument(
        "--warmup", type=int, default=20, help="Replays before each measurement round"
    )
    parser.add_argument(
        "--repeats", type=int, default=100, help="Measured replays per round and size"
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=0, help="Visible CUDA device index")
    parser.add_argument(
        "--hidden-states",
        type=Path,
        help="Optional .npy [N, hidden_size] real decode MoE inputs from this layer",
    )
    parser.add_argument("--output", type=Path, default=Path("results/glm_moe_decode.json"))
    args = parser.parse_args(argv)
    for name in ("kt_cpuinfer", "kt_threadpool_count", "warmup", "repeats", "rounds"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.layer_index < 0 or args.device < 0:
        parser.error("layer/device indices must be nonnegative")
    if any(bs <= 0 for bs in args.batch_sizes) or len(set(args.batch_sizes)) != len(
        args.batch_sizes
    ):
        parser.error("batch sizes must be distinct positive integers")
    if args.kt_threadpool_count > args.kt_cpuinfer:
        parser.error("NUMA pool count cannot exceed total worker threads")
    return args


def parse_cpu_list(value):
    result = set()
    for part in value.strip().split(","):
        if part:
            bounds = [int(x) for x in part.split("-")]
            result.update(range(bounds[0], bounds[-1] + 1))
    return result


def check_numa(args):
    if platform.system() != "Linux":
        raise RuntimeError("This benchmark requires Linux with CUDA and the pinned KT kernel")
    numa = ctypes.CDLL("libnuma.so.1")
    if numa.numa_available() < 0:
        raise RuntimeError("NUMA is unavailable")
    affinity = set(os.sched_getaffinity(0))
    status = Path("/proc/self/status").read_text()
    allowed_mems = next(
        line.split(":", 1)[1]
        for line in status.splitlines()
        if line.startswith("Mems_allowed_list:")
    )
    allowed_nodes = parse_cpu_list(allowed_mems)
    nodes = []
    for node in range(args.kt_threadpool_count):
        cpulist = Path(f"/sys/devices/system/node/node{node}/cpulist")
        if not cpulist.exists() or node not in allowed_nodes:
            raise RuntimeError(f"NUMA node {node} is missing or disallowed by the process cpuset")
        cpus = parse_cpu_list(cpulist.read_text())
        # KT binds by physical core index. Partial external pinning can invalidate that mapping.
        if not cpus <= affinity:
            raise RuntimeError(f"Allow all CPUs of NUMA node {node}; remove external CPU pinning")
        cores = set()
        for cpu in cpus:
            topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
            cores.add(
                (
                    topology.joinpath("physical_package_id").read_text().strip(),
                    topology.joinpath("core_id").read_text().strip(),
                )
            )
        threads = args.kt_cpuinfer // args.kt_threadpool_count + (
            node < args.kt_cpuinfer % args.kt_threadpool_count
        )
        if len(cores) < threads:
            raise RuntimeError(
                f"NUMA node {node} has {len(cores)} cores, needs {threads} KT workers"
            )
        nodes.append(
            {
                "node": node,
                "worker_threads": threads,
                "physical_cores": len(cores),
                "cpus": sorted(cpus),
            }
        )
    return nodes


def load_layer(args, config, device):
    """Mmap/index the checkpoint, but materialize only this block's MoE weights."""
    import torch
    from kt_kernel import KTMoEWrapper
    from kt_kernel.utils.llamafile import LlamafileMoEWrapper
    from minisgl.layers.marlin import pack_marlin
    from minisgl.models.gguf import GGUFWeights
    from minisgl.models.glm4_moe import Glm4MoeSparseMLP

    checkpoint = GGUFWeights(args.kt_weight_path, config)
    prefix = f"blk.{args.layer_index}"

    def dense(name, dtype):
        return checkpoint._dequantize(f"{prefix}.{name}", torch.device("cpu"), dtype, 16 << 20)

    with torch.device("meta"):
        layer = Glm4MoeSparseMLP(config)
    state = {
        "gate.weight": dense("ffn_gate_inp.weight", torch.float32).to(device),
        "gate.e_score_correction_bias": dense("exp_probs_b.bias", torch.float32).to(device),
    }
    gate_up = torch.cat([dense(f"ffn_{p}_shexp.weight", torch.bfloat16) for p in ("gate", "up")])
    for target, weight in (
        ("gate_up_proj", gate_up),
        ("down_proj", dense("ffn_down_shexp.weight", torch.bfloat16)),
    ):
        packed, scales = pack_marlin(weight)
        state[f"shared_experts.{target}.weight"] = packed.to(device)
        state[f"shared_experts.{target}.scales"] = scales.to(device)
    layer.load_state_dict(state)
    KTMoEWrapper.set_capture_batch_sizes(args.batch_sizes)
    LlamafileMoEWrapper._gguf_loaders_by_path[os.path.realpath(args.kt_weight_path)] = checkpoint
    layer._wrapper = KTMoEWrapper(
        layer_idx=args.layer_index,
        num_experts=config.num_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        moe_intermediate_size=config.moe_intermediate_size,
        gpu_experts_mask=None,
        cpuinfer_threads=args.kt_cpuinfer,
        threadpool_count=args.kt_threadpool_count,
        weight_path=args.kt_weight_path,
        chunked_prefill_size=max(args.batch_sizes),
        method="LLAMAFILE",
        max_deferred_experts_per_token=0,
    )
    layer._wrapper.load_weights(torch.arange(config.num_experts, dtype=torch.int32, device="cpu"))
    quantization = {
        p: checkpoint.tensors[f"{prefix}.ffn_{p}_exps.weight"].tensor_type.name
        for p in ("gate", "up", "down")
    }
    return layer, quantization


class InputSource:
    def __init__(self, args, hidden_size, device):
        import numpy as np
        import torch

        self.torch = torch
        self.generator = torch.Generator(device=device).manual_seed(args.seed)
        self.rng = np.random.default_rng(args.seed)
        self.pool = None
        if args.hidden_states:
            self.pool = np.load(args.hidden_states, mmap_mode="r", allow_pickle=False)
            if (
                self.pool.ndim != 2
                or self.pool.shape[1] != hidden_size
                or self.pool.shape[0] < max(args.batch_sizes)
            ):
                raise ValueError("hidden states must be [N, hidden_size], N >= max(batch_sizes)")
            if not np.issubdtype(self.pool.dtype, np.floating):
                raise ValueError("hidden states must have a floating-point dtype")
            # Chunk validation avoids allocating a boolean array as large as the whole trace.
            for start in range(0, len(self.pool), 1024):
                if not np.isfinite(self.pool[start : start + 1024]).all():
                    raise ValueError("hidden states contain NaN/Inf")

    def fill(self, x):
        if self.pool is None:
            x.normal_(generator=self.generator)
        else:
            rows = self.rng.choice(len(self.pool), size=x.shape[0], replace=False)
            x.copy_(self.torch.from_numpy(self.pool[rows]).to(dtype=x.dtype))


def capture_layer(layer, batch_size, config, source, stream):
    import torch
    from kt_kernel.experts_base import KExpertsCPUBuffer

    x = torch.empty(batch_size, config.hidden_size, device=stream.device, dtype=torch.bfloat16)

    def forward():
        # Same operations/order as Glm4MoeSparseMLP.forward; retain graph routing output
        # for diagnostics without adding diagnostic kernels to the captured workload.
        ids, weights = layer.gate.forward(x)
        routed = layer._wrapper.forward(x, ids, weights, stream.cuda_stream)
        return routed + layer.shared_experts.forward(x), ids

    for _ in range(3):
        source.fill(x)
        forward()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output, ids = forward()

    # Verify fresh inputs against production eager forward. Poison both KT output
    # copies: an omitted CPU callback/H2D must not pass using a stale eager answer.
    for _ in range(2):
        source.fill(x)
        expected = layer.forward(x).clone()
        expected_ids = layer.gate.forward(x)[0]
        stream.synchronize()
        buffers = KExpertsCPUBuffer.get_buffer(x, config.num_experts_per_tok)
        for buffer in (*buffers[4], *buffers[6]):
            buffer.fill_(float("nan"))
        output.fill_(float("nan"))
        ids.fill_(-1)
        graph.replay()
        stream.synchronize()
        if not torch.isfinite(output).all().item():
            raise RuntimeError("Graph replay produced nonfinite/stale outputs")
        torch.testing.assert_close(output, expected, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(ids, expected_ids, rtol=0, atol=0)
    return graph, x, output, ids


def summarize(samples, round_means, histogram, unique_experts, unique_routes):
    return {
        "avg_ms": statistics.mean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": min(samples),
        "max_ms": max(samples),
        "num_measurements": len(samples),
        "round_avg_ms": round_means,
        "samples_ms": samples,
        "routing": {
            "expert_selection_counts": histogram,
            "avg_unique_experts_per_batch": statistics.mean(unique_experts),
            "avg_unique_expert_sets_per_batch": statistics.mean(unique_routes),
        },
    }


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(args):
    numa_nodes = check_numa(args)
    if os.environ.get("KT_FORCE_SYNC_SUBMIT") == "1":
        raise RuntimeError("Unset KT_FORCE_SYNC_SUBMIT: CUDA graphs require stream callbacks")

    import torch
    from minisgl.models.config import ModelConfig
    from transformers import AutoConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.cuda.set_device(args.device)
    device = torch.device("cuda", args.device)
    # The requested 64 threads belong to KT; avoid an additional 64-thread Torch pool.
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    config = ModelConfig.from_hf(AutoConfig.from_pretrained(args.model))
    if not config.is_glm4_moe:
        raise ValueError("Expected a GLM-4 MoE config (use GLM-4.5-Air)")
    if not config.first_k_dense_replace <= args.layer_index < config.num_layers:
        raise ValueError("Selected layer is dense or outside the model")
    print(
        f"Loading only blk.{args.layer_index} MoE; {args.kt_cpuinfer} KT threads; "
        f"NUMA nodes {[node['node'] for node in numa_nodes]}",
        flush=True,
    )
    layer, quantization = load_layer(args, config, device)
    source = InputSource(args, config.hidden_size, device)
    data = {
        "schema_version": 1,
        "complete": False,
        "metadata": {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": args.model,
            "weight_path": os.path.realpath(args.kt_weight_path),
            "layer_index": args.layer_index,
            "layer_number": args.layer_index + 1,
            "batch_sizes": args.batch_sizes,
            "cuda_graph": True,
            "kt_cpuinfer": args.kt_cpuinfer,
            "kt_threadpool_count": args.kt_threadpool_count,
            "numa_nodes": numa_nodes,
            "backend": "KT LLAMAFILE",
            "expert_quantization": quantization,
            "shared_experts_quantization": "Marlin INT4 g64",
            "dtype": "bfloat16",
            "router_dtype": "float32",
            "hidden_size": config.hidden_size,
            "num_experts": config.num_experts,
            "top_k": config.num_experts_per_tok,
            "moe_intermediate_size": config.moe_intermediate_size,
            "input_source": str(args.hidden_states.resolve())
            if args.hidden_states
            else "iid_normal",
            "routing": "checkpoint router, independent input per token, fresh inputs every replay",
            "timing": "perf_counter_ns: graph.replay + stream.synchronize, milliseconds per batch",
            "scope": "router + routed CPU experts + shared GPU experts + sum + transfers",
            "warmup_per_round": args.warmup,
            "repeats_per_round": args.repeats,
            "rounds": args.rounds,
            "seed": args.seed,
            "gpu": torch.cuda.get_device_name(device),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "platform": platform.platform(),
            "graph_eager_validation": "two fresh inputs per batch size; KT outputs poisoned",
        },
        "round_batch_orders": [],
        "results": [],
    }
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    records = {
        bs: {
            "samples": [],
            "round_means": [],
            "histogram": [0] * config.num_experts,
            "unique_experts": [],
            "unique_routes": [],
        }
        for bs in args.batch_sizes
    }
    graphs = {}
    order_rng = random.Random(args.seed)
    with torch.inference_mode(), torch.cuda.stream(stream):
        for bs in args.batch_sizes:
            graphs[bs] = capture_layer(layer, bs, config, source, stream)
            print(f"Captured and validated batch_size={bs}", flush=True)
        for round_index in range(args.rounds):
            order = list(args.batch_sizes)
            order_rng.shuffle(order)
            data["round_batch_orders"].append(order)
            for bs in order:
                graph, x, output, ids = graphs[bs]
                record = records[bs]
                for _ in range(args.warmup):
                    source.fill(x)
                    graph.replay()
                    stream.synchronize()
                samples = []
                for _ in range(args.repeats):
                    source.fill(x)
                    stream.synchronize()  # Input generation/copy excluded from the timer.
                    start = time.perf_counter_ns()
                    graph.replay()
                    stream.synchronize()  # Includes CPU callbacks, copies, and shared experts.
                    elapsed_ms = (time.perf_counter_ns() - start) / 1e6
                    samples.append(elapsed_ms)
                    # Read the actual graph routing, only after stopping the timer.
                    selected = ids.cpu().tolist()
                    expert_sets = {tuple(sorted(row)) for row in selected}
                    experts = {expert for row in selected for expert in row}
                    record["unique_experts"].append(len(experts))
                    record["unique_routes"].append(len(expert_sets))
                    for row in selected:
                        for expert in row:
                            record["histogram"][expert] += 1
                if not torch.isfinite(output).all().item():
                    raise RuntimeError(f"Nonfinite output at batch size {bs}")
                record["samples"].extend(samples)
                record["round_means"].append(statistics.mean(samples))
                print(
                    f"Round {round_index + 1}/{args.rounds}, batch_size={bs:3d}: "
                    f"{statistics.mean(samples):.4f} ms",
                    flush=True,
                )
                data["results"] = [
                    {"batch_size": size, **summarize(**records[size])}
                    for size in sorted(records)
                    if records[size]["samples"]
                ]
                write_json(args.output, data)
    data["complete"] = True
    write_json(args.output, data)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    run(parse_args())
