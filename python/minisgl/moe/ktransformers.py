from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import torch
from minisgl.utils import init_logger

from .base import BaseMoeBackend
from .fused import fused_topk

if TYPE_CHECKING:
    from minisgl.engine import EngineConfig

logger = init_logger(__name__)


def validate_config(config: EngineConfig) -> None:
    if config.model_config.model_type != "qwen3_moe":
        raise ValueError("KTransformers supports Qwen3 MoE only")
    if config.tp_info.size != 1:
        raise ValueError("KTransformers requires --tp-size 1 (all experts run on CPU)")
    if config.dtype != torch.bfloat16:
        raise ValueError("KTransformers LLAMAFILE requires --dtype bfloat16")
    if config.kt_method != "LLAMAFILE":
        raise ValueError("Only --kt-method LLAMAFILE is supported")
    if not config.kt_weight_path or not Path(config.kt_weight_path).exists():
        raise ValueError("--kt-weight-path must point to an existing GGUF file or directory")
    if config.kt_cpuinfer < config.kt_threadpool_count or config.kt_threadpool_count < 1:
        raise ValueError("Require --kt-cpuinfer >= --kt-threadpool-count >= 1")
    if config.max_forward_len < 1 or config.max_running_req < 1:
        raise ValueError("Prefill length and maximum running requests must be positive")
    if config.use_dummy_weight:
        raise ValueError("KTransformers requires real weights; --dummy-weight is unsupported")
    if getattr(config.hf_config, "quantization_config", None):
        raise ValueError(
            "Use the original BF16 Qwen3 model for --model, and GGUF for --kt-weight-path"
        )


class KTransformersMoe(BaseMoeBackend):
    """Standalone KT experts. Attention, router, norms and RoPE stay on CUDA."""

    cpu_experts = True

    def __init__(self, config: EngineConfig):
        validate_config(config)
        from kt_kernel import KTMoEWrapper

        model = config.model_config
        # KT sizes its C++ buffers at load time; decode batches can exceed the prefill limit.
        self.max_tokens = max(config.max_forward_len, config.max_running_req)
        self.wrappers = []
        for layer_id in range(model.num_layers):
            wrapper = KTMoEWrapper(
                layer_idx=layer_id,
                num_experts=model.num_experts,
                num_experts_per_tok=model.num_experts_per_tok,
                hidden_size=model.hidden_size,
                moe_intermediate_size=model.moe_intermediate_size,
                gpu_experts_mask=None,
                cpuinfer_threads=config.kt_cpuinfer,
                threadpool_count=config.kt_threadpool_count,
                weight_path=config.kt_weight_path,
                chunked_prefill_size=self.max_tokens,
                method=config.kt_method,
                max_deferred_experts_per_token=0,
            )
            wrapper.load_weights()  # Identity expert IDs; GGUF expert tensors are loaded on CPU.
            self.wrappers.append(wrapper)
        logger.info_rank0(
            f"Loaded {model.num_layers} MoE layers on CPU, {model.num_experts} experts/layer; "
            "GPU expert weights: 0"
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor | None,
        w2: torch.Tensor | None,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        layer_id: int = 0,
    ) -> torch.Tensor:
        if w1 is not None or w2 is not None:
            raise ValueError("KTransformers must not receive GPU expert weights")
        if activation != "silu" or apply_router_weight_on_input:
            raise ValueError(
                "KTransformers supports Qwen3 SiLU experts with output routing weights"
            )
        if not hidden_states.is_cuda or hidden_states.dtype != torch.bfloat16:
            raise ValueError("KTransformers expects CUDA BF16 hidden states")
        if hidden_states.shape[0] > self.max_tokens:
            raise ValueError("MoE batch exceeds the allocated KT buffer size")
        topk_weights, topk_ids = fused_topk(hidden_states, gating_output, topk, renormalize)
        stream = torch.cuda.current_stream(hidden_states.device).cuda_stream
        return self.wrappers[layer_id].forward(hidden_states, topk_ids, topk_weights, stream)
