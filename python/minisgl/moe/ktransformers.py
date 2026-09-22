from __future__ import annotations

import os
from typing import TYPE_CHECKING

import torch
from minisgl.layers import BaseOP, LinearReplicated

if TYPE_CHECKING:
    from minisgl.engine import EngineConfig
    from minisgl.models import BaseLLMModel
    from minisgl.models.gguf import GGUFWeights


def load_ktransformers_experts(
    model: BaseLLMModel,
    config: EngineConfig,
    cuda_graph_bs: list[int] | None = None,
    *,
    gguf_weights: GGUFWeights | None = None,
) -> None:
    from kt_kernel import KTMoEWrapper

    cuda_graph_bs = cuda_graph_bs or []
    if cuda_graph_bs and os.environ.get("KT_FORCE_SYNC_SUBMIT") == "1":
        raise ValueError(
            "KT CUDA graphs require stream callbacks; unset KT_FORCE_SYNC_SUBMIT "
            "or use --cuda-graph-max-bs 0"
        )
    # Warmup allocates these pinned CPU/GPU buffers. KT retains each size so
    # captured D2H, CPU callbacks and H2D keep valid addresses during replay,
    # even after an eager prefill uses a different batch size.
    KTMoEWrapper.set_capture_batch_sizes(cuda_graph_bs)
    if gguf_weights is not None:
        from kt_kernel.utils.llamafile import LlamafileMoEWrapper

        # Reuse our validated mapping, including byte-split GGUF files.
        LlamafileMoEWrapper._gguf_loaders_by_path[os.path.realpath(config.kt_weight_path)] = (
            gguf_weights
        )
    max_graph_bs = max(cuda_graph_bs, default=0)
    # Replace the entire MLP BEFORE loading GPU weights, retaining its router
    # under mlp.gate so the Hugging Face checkpoint keys remain unchanged.
    for layer_id, layer in enumerate(model.model.layers.op_list):
        if config.model_config.is_mla:
            if layer_id >= config.model_config.first_k_dense_replace:
                layer.mlp._wrapper = _create_kt_wrapper(config, layer_id, max_graph_bs)
            continue
        layer.mlp = KTransformersMoE(
            config, layer_id, gate=layer.mlp.gate, max_graph_bs=max_graph_bs
        )


def _create_kt_wrapper(config: EngineConfig, layer_id: int, max_graph_bs: int):
    from kt_kernel import KTMoEWrapper

    model = config.model_config
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
        chunked_prefill_size=max(config.max_forward_len, config.max_running_req, max_graph_bs),
        method=config.kt_method,
        max_deferred_experts_per_token=0,
    )
    wrapper.load_weights(torch.arange(model.num_experts, dtype=torch.int32, device="cpu"))
    return wrapper


class KTransformersMoE(BaseOP):
    """Complete MoE MLP: GPU router and a direct KT CPU expert wrapper."""

    def __init__(
        self, config: EngineConfig, layer_id: int, *, gate: LinearReplicated, max_graph_bs: int = 0
    ):
        model = config.model_config
        self.gate = gate
        self._top_k = model.num_experts_per_tok
        self._renormalize = model.norm_topk_prob
        self._wrapper = _create_kt_wrapper(config, layer_id, max_graph_bs)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        router_logits = self.gate.forward(hidden_states)
        scores = torch.softmax(router_logits, dim=-1, dtype=torch.float32)
        weights, ids = torch.topk(scores, self._top_k, dim=-1)
        if self._renormalize:
            weights = weights / weights.sum(dim=-1, keepdim=True)

        # type(self._wrapper): <class 'kt_kernel.utils.llamafile.LlamafileMoEWrapper'>
        return self._wrapper.forward(
            hidden_states,
            ids,
            weights,
            torch.cuda.current_stream(hidden_states.device).cuda_stream,
        )
