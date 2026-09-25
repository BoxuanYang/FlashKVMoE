# SPDX-License-Identifier: Apache-2.0
"""保留 GLM 模型层次，并为 MiniSGL 增加双 microbatch decode。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.attention import create_attention_backend
from minisgl.core import Batch, get_global_ctx
from minisgl.distributed import DistributedInfo
from minisgl.layers import AttentionLayer, BaseOP, OPList, RMSNormFused, VocabParallelEmbedding
from minisgl.layers.marlin import MarlinLinear
from minisgl.models.base import BaseLLMModel
from minisgl.models.config import ModelConfig


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


class Glm4MoeModel(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = OPList(
            [Glm4MoeDecoderLayer(config, layer) for layer in range(config.num_layers)]
        )
        self.norm = RMSNormFused(config.hidden_size, config.rms_norm_eps)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x, residual = self.embed_tokens.forward(ids), None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]

    def forward_dual(
        self,
        m0: Batch,
        m1: Batch,
        attention: tuple,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """两份 decode batch 共用模型权重，在同一 CUDA stream 上交错执行。"""
        self.batches = (m0, m1)
        self.attention = attention
        self.stream = stream
        self.hidden_states = [self.embed_tokens.forward(batch.input_ids) for batch in self.batches]
        self.residual: list[torch.Tensor | None] = [None, None]
        self.shared: list[torch.Tensor | None] = [None, None]
        self.topk_ids: list[torch.Tensor | None] = [None, None]
        self.topk_weights: list[torch.Tensor | None] = [None, None]

        ctx = get_global_ctx()
        previous_backend = ctx.attn_backend
        layer_num = len(self.layers.op_list)
        overlap_steps = 2 * layer_num - 1
        try:
            for step in range(overlap_steps + 2):
                if step == 0:
                    self.submit_attn(0, 0)
                elif step == overlap_steps + 1:
                    self.submit_moe(1, layer_num - 1)
                    self.sync_moe(1, layer_num - 1)
                else:
                    attn_batch = step % 2
                    moe_batch = 1 - attn_batch
                    attn_layer = step // 2
                    moe_layer = (step - 1) // 2
                    self.submit_moe(moe_batch, moe_layer)
                    self.submit_attn(attn_batch, attn_layer)
                    self.sync_moe(moe_batch, moe_layer)
        finally:
            ctx.attn_backend = previous_backend

        return (
            self.norm.forward(self.hidden_states[0], self.residual[0])[0],
            self.norm.forward(self.hidden_states[1], self.residual[1])[0],
        )

    def submit_attn(self, mb: int, layer_idx: int) -> None:
        layer = self.layers.op_list[layer_idx]
        ctx = get_global_ctx()
        ctx.attn_backend = self.attention[mb]
        with ctx.forward_batch(self.batches[mb]):
            x, self.residual[mb] = layer.input_layernorm.forward(
                self.hidden_states[mb], self.residual[mb]
            )
            x = layer.self_attn.forward(x)
            x, self.residual[mb] = layer.post_attention_layernorm.forward(x, self.residual[mb])

        if isinstance(layer.mlp, Glm4MoeMLP):
            self.hidden_states[mb] = layer.mlp.forward(x)
            return

        mlp: Glm4MoeSparseMLP = layer.mlp
        self.topk_ids[mb], self.topk_weights[mb] = mlp.gate.forward(x)
        self.shared[mb] = mlp.shared_experts.forward(x)
        self.hidden_states[mb] = x

    def submit_moe(self, mb: int, layer_idx: int) -> None:
        mlp = self.layers.op_list[layer_idx].mlp
        if isinstance(mlp, Glm4MoeMLP):
            return
        mlp._wrapper.submit_forward(
            self.hidden_states[mb],
            self.topk_ids[mb],
            self.topk_weights[mb],
            self.stream.cuda_stream,
        )

    def sync_moe(self, mb: int, layer_idx: int) -> None:
        mlp = self.layers.op_list[layer_idx].mlp
        if isinstance(mlp, Glm4MoeMLP):
            return
        routed = mlp._wrapper.sync_forward(self.hidden_states[mb], self.stream.cuda_stream).clone()
        routed.add_(self.shared[mb])
        self.hidden_states[mb] = routed
        self.shared[mb] = None
        self.topk_ids[mb] = self.topk_weights[mb] = None


class Glm4MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Glm4MoeModel(config)
        self.lm_head = MarlinLinear(config.hidden_size, config.vocab_size)
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        x = self.model.forward(batch.input_ids)
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return self.lm_head.forward(x)

    def forward_dual(
        self,
        m0: Batch,
        m1: Batch,
        attention: tuple,
        stream: torch.cuda.Stream,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h0, h1 = self.model.forward_dual(m0, m1, attention, stream)
        return self.lm_head.forward(h0), self.lm_head.forward(h1)


class GLMDualBatchRunner:
    """持有 MiniSGL Engine 和两份 Attention plan。"""

    def __init__(self, engine, model_config: ModelConfig, attention_backend: str):
        self.engine = engine
        self.causal_lm: Glm4MoeForCausalLM = engine.model
        self.attention = (
            create_attention_backend(attention_backend, model_config),
            create_attention_backend(attention_backend, model_config),
        )

    def forward(self, m0: Batch, m1: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.stream(self.engine.stream):
            for batch, backend in zip((m0, m1), self.attention):
                backend.prepare_metadata(batch)
            return self.causal_lm.model.forward_dual(m0, m1, self.attention, self.engine.stream)

    def forward_logits(self, m0: Batch, m1: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.cuda.stream(self.engine.stream):
            for batch, backend in zip((m0, m1), self.attention):
                backend.prepare_metadata(batch)
            return self.causal_lm.forward_dual(m0, m1, self.attention, self.engine.stream)

    def shutdown(self) -> None:
        self.engine.shutdown()


def load_model(
    model_path: str,
    kt_weight_path: str,
    *,
    max_running_req: int = 256,
    attention_backend: str = "fi",
    kt_cpuinfer: int = 128,
    kt_threadpool_count: int = 2,
    page_size: int = 1,
    memory_ratio: float = 0.9,
    max_seq_len: int | None = None,
    num_pages: int | None = None,
    distributed_timeout: float = 60.0,
    use_pynccl: bool = True,
) -> GLMDualBatchRunner:
    """通过 MiniSGL Engine 加载本文件定义的 GLM 模型和 KT 权重。"""
    from minisgl.engine import Engine, EngineConfig

    config = EngineConfig(
        model_path=model_path,
        tp_info=DistributedInfo(rank=0, size=1),
        dtype=torch.bfloat16,
        max_running_req=max_running_req,
        attention_backend=attention_backend,
        kt_weight_path=kt_weight_path,
        kt_cpuinfer=kt_cpuinfer,
        kt_threadpool_count=kt_threadpool_count,
        page_size=page_size,
        memory_ratio=memory_ratio,
        distributed_timeout=distributed_timeout,
        use_pynccl=use_pynccl,
        max_seq_len_override=max_seq_len,
        num_page_override=num_pages,
        cuda_graph_bs=[],
        cuda_graph_max_bs=0,
    )
    if not config.model_config.is_glm4_moe:
        raise ValueError(f"需要 GLM-4 MoE，当前模型类型为 {config.model_config.model_type!r}")

    engine = Engine(config, model_factory=Glm4MoeForCausalLM)
    try:
        return GLMDualBatchRunner(engine, config.model_config, config.attention_backend)
    except Exception:
        engine.shutdown()
        raise


__all__ = [
    "Glm4MoeMLP",
    "Glm4MoeRouter",
    "Glm4MoeSparseMLP",
    "Glm4MoeAttention",
    "Glm4MoeDecoderLayer",
    "Glm4MoeModel",
    "Glm4MoeForCausalLM",
    "GLMDualBatchRunner",
    "load_model",
]
