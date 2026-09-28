"""Measure the GPU work that can overlap CPU routed experts."""

import os
from dataclasses import replace
from functools import partial
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "6")

import torch
import torch.nn.functional as F
from minisgl.attention.fi import FlashInferBackend
from minisgl.core import Batch, Context, Req, set_global_ctx
from minisgl.distributed import set_tp_info
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.layers import AttentionLayer, BaseOP, RMSNormFused, set_rope_device
from minisgl.layers.marlin import MarlinLinear, pack_marlin
from minisgl.models.config import ModelConfig
from minisgl.models.gguf import GGUFWeights
from minisgl.utils import torch_dtype
from transformers import AutoConfig

MODEL_PATH = "/data1/models/GLM-4.5-Air-GGUF"
WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
LAYER_NUMBER = 7
LAYER_INDEX = LAYER_NUMBER - 1  # Human layer 7 is GGUF blk.6.

DEVICE = torch.device("cuda:0")
SEQUENCE_LENGTHS = [2000 * i for i in range(1, 80)]
OUTPUT_PATH = Path("gpu_perf.txt")
PAGE_SIZE = 2

WARMUP = 10
REPEATS = 100
SEED = 42


class Glm4MoeMLP(BaseOP):
    def __init__(self, hidden: int, intermediate: int):
        self.gate_up_proj = MarlinLinear(hidden, 2 * intermediate)
        self.down_proj = MarlinLinear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, up = self.gate_up_proj.forward(x).chunk(2, dim=-1)
        return self.down_proj.forward(F.silu(gate) * up)


class Glm4MoeRouter(BaseOP):
    def __init__(self, config: ModelConfig):
        self.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self.e_score_correction_bias = torch.empty(config.num_experts, dtype=torch.float32)
        self._config = config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self._config
        scores = F.linear(x.float(), self.weight).sigmoid()
        choice = scores + self.e_score_correction_bias
        grouped = choice.view(-1, c.n_group, c.num_experts // c.n_group)
        groups = grouped.topk(2, dim=-1).values.sum(-1).topk(c.topk_group, dim=-1).indices
        mask = torch.zeros_like(grouped[..., 0], dtype=torch.bool).scatter_(1, groups, True)
        choice = grouped.masked_fill(~mask.unsqueeze(-1), 0).flatten(1)
        ids = choice.topk(c.num_experts_per_tok, dim=-1, sorted=False).indices
        weights = scores.gather(1, ids)
        if c.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        return ids, weights * c.routed_scaling_factor


class Glm4MoeSparseMLP(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = Glm4MoeRouter(config)
        self.shared_experts = Glm4MoeMLP(
            config.hidden_size, config.n_shared_experts * config.moe_intermediate_size
        )
        self._wrapper = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids, weights = self.gate.forward(x)
        routed = self._wrapper.forward(
            x, ids, weights, torch.cuda.current_stream(x.device).cuda_stream
        )
        return routed + self.shared_experts.forward(x)


class Glm4MoeAttention(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        qkv = (config.num_qo_heads + 2 * config.num_kv_heads) * config.head_dim
        self.qkv_proj = MarlinLinear(config.hidden_size, qkv)
        self.qkv_bias = torch.empty(qkv)
        self.attn = AttentionLayer(
            layer_id,
            config.num_qo_heads,
            config.num_kv_heads,
            config.head_dim,
            config.rotary_config,
        )
        self.o_proj = MarlinLinear(config.num_qo_heads * config.head_dim, config.hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.o_proj.forward(self.attn.forward(self.qkv_proj.forward(x) + self.qkv_bias))


class Glm4MoeDecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Glm4MoeAttention(config, layer_id)
        self.mlp = (
            Glm4MoeMLP(config.hidden_size, config.intermediate_size)
            if layer_id < config.first_k_dense_replace
            else Glm4MoeSparseMLP(config)
        )
        self.input_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, residual: torch.Tensor | None):
        x, residual = self.input_layernorm.forward(x, residual)
        x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)
        return self.mlp.forward(x), residual


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


def initialize_kv_cache(context: Context, config: ModelConfig, max_length: int):
    """Create the same paged MHA cache used by the production backend."""
    num_pages = (max_length + PAGE_SIZE - 1) // PAGE_SIZE
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
    context.page_table = torch.arange(max_length, device=DEVICE, dtype=torch.int32).view(1, -1)


def make_inputs(config: ModelConfig):
    """Create the activation and residual entering decoder layer 7."""
    hidden = torch.randn(1, config.hidden_size, device=DEVICE, dtype=torch.bfloat16)
    residual = torch.randn_like(hidden)
    return hidden, residual


def gpu_overlap_path(
    layer: Glm4MoeDecoderLayer,
    hidden: torch.Tensor,
    residual: torch.Tensor,
):
    """Run the original layer's GPU work, excluding only routed CPU experts."""
    hidden, residual = layer.input_layernorm.forward(hidden, residual)
    hidden = layer.self_attn.forward(hidden)
    hidden, residual = layer.post_attention_layernorm.forward(hidden, residual)

    if not isinstance(layer.mlp, Glm4MoeSparseMLP):
        raise TypeError("The GPU overlap benchmark requires a sparse MoE layer")
    expert_ids, expert_weights = layer.mlp.gate.forward(hidden)
    shared_output = layer.mlp.shared_experts.forward(hidden)
    return shared_output, expert_ids, expert_weights, residual


def make_decode_batch(sequence_length: int) -> Batch:
    """Build a real batch with one decode token and a synthetic history."""
    request = Req(
        input_ids=torch.zeros(sequence_length, dtype=torch.int32),
        table_idx=0,
        cached_len=sequence_length - 1,
        output_len=1,
        uid=0,
        sampling_params=None,  # type: ignore[arg-type]
        cache_handle=None,  # type: ignore[arg-type]
    )
    batch = Batch([request], "decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.zeros(1, device=DEVICE, dtype=torch.int32)
    batch.positions = torch.tensor(
        [sequence_length - 1], device=DEVICE, dtype=torch.int32
    )
    batch.out_loc = torch.tensor(
        [sequence_length - 1], device=DEVICE, dtype=torch.int32
    )
    return batch


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
    context = Context(page_size=PAGE_SIZE)
    set_global_ctx(context)

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
    initialize_kv_cache(context, config, max_length)
    gpu_name = torch.cuda.get_device_name(DEVICE)
    results = []

    for sequence_length in SEQUENCE_LENGTHS:
        hidden, residual = make_inputs(config)
        batch = make_decode_batch(sequence_length)
        backend = FlashInferBackend(config)
        context.attn_backend = backend
        backend.init_capture_graph(max_seq_len=max_length, bs_list=[1])
        backend.prepare_for_capture(batch)
        operation = partial(gpu_overlap_path, layer, hidden, residual)
        with context.forward_batch(batch):
            graph, output = capture_graph(operation)
            results.append((sequence_length, measure_gpu_ms(graph)))

        graph.reset()
        del context.attn_backend
        del graph, output, operation, hidden, residual, batch, backend
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
