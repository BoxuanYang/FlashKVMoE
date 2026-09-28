"""Measure the GPU work that can overlap CPU routed experts."""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import torch
from flashinfer import single_decode_with_kv_cache
from minisgl.distributed import set_tp_info
from minisgl.layers import set_rope_device
from minisgl.layers.marlin import pack_marlin
from minisgl.models.config import ModelConfig
from minisgl.models.gguf import GGUFWeights
from minisgl.models.glm4_moe import Glm4MoeDecoderLayer
from minisgl.utils import torch_dtype
from transformers import AutoConfig

MODEL_PATH = "/data1/models/GLM-4.5-Air-GGUF"
WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
LAYER_NUMBER = 7
LAYER_INDEX = LAYER_NUMBER - 1  # Human layer 7 is GGUF blk.6.

DEVICE = torch.device("cuda:0")
SEQUENCE_LENGTHS = [2000 * i for i in range(1, 80)]
OUTPUT_PATH = Path("gpu_perf.txt")

WARMUP = 10
REPEATS = 100
SEED = 42


def pack_weight(checkpoint: GGUFWeights, name: str):
    """Load one GGUF matrix, pack it for Marlin, and move it to the GPU."""
    weight = checkpoint._dequantize(name, torch.device("cpu"), torch.bfloat16, 16 << 20)
    packed, scales = pack_marlin(weight)
    return packed.to(DEVICE), scales.to(DEVICE)


def pack_merged_weight(checkpoint: GGUFWeights, names: list[str]):
    """Concatenate related GGUF matrices before packing them for Marlin."""
    weight = torch.cat(
        [
            checkpoint._dequantize(name, torch.device("cpu"), torch.bfloat16, 16 << 20)
            for name in names
        ]
    )
    packed, scales = pack_marlin(weight)
    return packed.to(DEVICE), scales.to(DEVICE)


def load_decoder_layer(config: ModelConfig) -> Glm4MoeDecoderLayer:
    """Load the GPU-resident parts of GLM-4.5-Air layer 7."""
    if LAYER_INDEX < config.first_k_dense_replace:
        raise ValueError(f"Layer {LAYER_NUMBER} is dense and has no MoE router")

    checkpoint = GGUFWeights(WEIGHT_PATH, config)
    prefix = f"blk.{LAYER_INDEX}"

    qkv_weight, qkv_scales = pack_merged_weight(
        checkpoint,
        [f"{prefix}.attn_{name}.weight" for name in ("q", "k", "v")],
    )
    qkv_bias = torch.cat(
        [
            checkpoint._dequantize(f"{prefix}.attn_{name}.bias", DEVICE, torch.bfloat16, 16 << 20)
            for name in ("q", "k", "v")
        ]
    )
    output_weight, output_scales = pack_weight(checkpoint, f"{prefix}.attn_output.weight")
    shared_gate_up_weight, shared_gate_up_scales = pack_merged_weight(
        checkpoint,
        [
            f"{prefix}.ffn_gate_shexp.weight",
            f"{prefix}.ffn_up_shexp.weight",
        ],
    )
    shared_down_weight, shared_down_scales = pack_weight(
        checkpoint, f"{prefix}.ffn_down_shexp.weight"
    )

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        layer = Glm4MoeDecoderLayer(config, LAYER_INDEX)
    layer.load_state_dict(
        {
            "self_attn.qkv_proj.weight": qkv_weight,
            "self_attn.qkv_proj.scales": qkv_scales,
            "self_attn.qkv_bias": qkv_bias,
            "self_attn.o_proj.weight": output_weight,
            "self_attn.o_proj.scales": output_scales,
            "mlp.gate.weight": checkpoint._dequantize(
                f"{prefix}.ffn_gate_inp.weight", DEVICE, torch.float32, 16 << 20
            ),
            "mlp.gate.e_score_correction_bias": checkpoint._dequantize(
                f"{prefix}.exp_probs_b.bias", DEVICE, torch.float32, 16 << 20
            ),
            "mlp.shared_experts.gate_up_proj.weight": shared_gate_up_weight,
            "mlp.shared_experts.gate_up_proj.scales": shared_gate_up_scales,
            "mlp.shared_experts.down_proj.weight": shared_down_weight,
            "mlp.shared_experts.down_proj.scales": shared_down_scales,
            "input_layernorm.weight": checkpoint._dequantize(
                f"{prefix}.attn_norm.weight", DEVICE, torch.bfloat16, 16 << 20
            ),
            "post_attention_layernorm.weight": checkpoint._dequantize(
                f"{prefix}.post_attention_norm.weight",
                DEVICE,
                torch.bfloat16,
                16 << 20,
            ),
        }
    )
    return layer


def make_inputs(config: ModelConfig, sequence_length: int):
    """Create one decode token, its residual, position, and KV history."""
    hidden = torch.randn(1, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    position = torch.tensor([sequence_length - 1], device=DEVICE, dtype=torch.int32)
    kv_shape = (sequence_length, config.num_kv_heads, config.head_dim)
    keys = torch.randn(kv_shape, device=DEVICE, dtype=torch.bfloat16)
    values = torch.randn(kv_shape, device=DEVICE, dtype=torch.bfloat16)
    return hidden, residual, position, keys, values


def gpu_work(layer, config, hidden, residual, position, keys, values):
    """Run every GPU operation before and alongside the CPU routed experts."""
    hidden, residual = layer.input_layernorm.forward(hidden, residual)

    qkv = layer.self_attn.qkv_proj.forward(hidden) + layer.self_attn.qkv_bias
    query_size = config.num_qo_heads * config.head_dim
    kv_size = config.num_kv_heads * config.head_dim
    query, current_key, current_value = qkv.split([query_size, kv_size, kv_size], dim=-1)
    query, current_key = layer.self_attn.attn.rotary.forward(position, query, current_key)

    keys[-1].copy_(current_key.view(config.num_kv_heads, config.head_dim))
    values[-1].copy_(current_value.view(config.num_kv_heads, config.head_dim))
    attention = single_decode_with_kv_cache(
        query.view(config.num_qo_heads, config.head_dim),
        keys,
        values,
        kv_layout="NHD",
        pos_encoding_mode="NONE",
        use_tensor_cores=True,
    )
    hidden = layer.self_attn.o_proj.forward(attention.reshape(1, -1))
    hidden, residual = layer.post_attention_layernorm.forward(hidden, residual)

    expert_ids, expert_weights = layer.mlp.gate.forward(hidden)
    shared_output = layer.mlp.shared_experts.forward(hidden)
    return shared_output, expert_ids, expert_weights, residual


def capture_graph(operation):
    """Capture the complete GPU path after all setup and allocations are warm."""
    for _ in range(WARMUP):
        operation()
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = operation()
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
    set_tp_info(0, 1)
    set_rope_device(DEVICE)

    config = ModelConfig.from_hf(AutoConfig.from_pretrained(MODEL_PATH))
    max_length = max(SEQUENCE_LENGTHS)
    if config.rotary_config.max_position < max_length:
        config = replace(
            config,
            rotary_config=replace(config.rotary_config, max_position=max_length),
        )

    expected_shape = (96, 8, 128)
    actual_shape = (config.num_qo_heads, config.num_kv_heads, config.head_dim)
    if not config.is_glm4_moe or actual_shape != expected_shape:
        raise ValueError(
            f"Expected GLM-4.5-Air attention shape {expected_shape}, got {actual_shape}"
        )

    print(f"Loading GLM-4.5-Air layer {LAYER_NUMBER} GPU weights ...", flush=True)
    layer = load_decoder_layer(config)
    gpu_name = torch.cuda.get_device_name(DEVICE)
    results = []

    for sequence_length in SEQUENCE_LENGTHS:
        inputs = make_inputs(config, sequence_length)
        operation = partial(gpu_work, layer, config, *inputs)
        graph, output = capture_graph(operation)
        results.append((sequence_length, measure_gpu_ms(graph)))

        graph.reset()
        del graph, output, operation, inputs
        torch.cuda.empty_cache()

    lines = [
        f"GLM-4.5-Air layer {LAYER_NUMBER} GPU overlap path",
        f"GPU: {gpu_name}",
        f"Batch size: 1, CUDA Graph: on, repeats: {REPEATS}",
        "Includes norms, attention, shared expert and MoE router; excludes routed experts.",
        "",
        f"{'Sequence':>10} {'GPU ms':>14}",
        "-" * 25,
    ]
    lines.extend(f"{sequence_length:>10} {gpu_ms:>14.6f}" for sequence_length, gpu_ms in results)

    report = "\n".join(lines) + "\n"
    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report, end="")
    print(f"Results written to {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
    main()
