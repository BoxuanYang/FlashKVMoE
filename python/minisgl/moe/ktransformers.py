from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from minisgl.layers import BaseOP

if TYPE_CHECKING:
    from minisgl.engine import EngineConfig
    from minisgl.models.qwen3_moe import Qwen3MoeForCausalLM


def load_ktransformers_experts(model: Qwen3MoeForCausalLM, config: EngineConfig) -> None:
    # Replace meta expert placeholders BEFORE loading any GPU weights.
    for layer_id, layer in enumerate(model.model.layers.op_list):
        layer.mlp.experts = KTransformersMoE(config, layer_id)


class KTransformersMoE(BaseOP):
    """CPU experts owned by KT; routing and the rest of Qwen3 stay on GPU."""

    def __init__(self, config: EngineConfig, layer_id: int):
        from kt_kernel import KTMoEWrapper

        model = config.model_config
        self._top_k = model.num_experts_per_tok
        self._renormalize = model.norm_topk_prob
        self._wrapper = KTMoEWrapper(
            layer_idx=layer_id,
            num_experts=model.num_experts,
            num_experts_per_tok=self._top_k,
            hidden_size=model.hidden_size,
            moe_intermediate_size=model.moe_intermediate_size,
            gpu_experts_mask=None,
            cpuinfer_threads=config.kt_cpuinfer,
            threadpool_count=config.kt_threadpool_count,
            weight_path=config.kt_weight_path,
            # Decode can have more tokens than a configured prefill chunk.
            chunked_prefill_size=max(config.max_forward_len, config.max_running_req),
            method=config.kt_method,
            max_deferred_experts_per_token=0,
        )
        self._wrapper.load_weights(torch.arange(model.num_experts, dtype=torch.int32, device="cpu"))

    def forward(self, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(scores, self._top_k, dim=-1)
        if self._renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)
        return self._wrapper.forward(
            hidden_states,
            ids,
            weights,
            torch.cuda.current_stream(hidden_states.device).cuda_stream,
        )
