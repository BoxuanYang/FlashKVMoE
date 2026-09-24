from __future__ import annotations

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers import AttentionLayer, BaseOP, OPList, RMSNormFused, VocabParallelEmbedding
from minisgl.layers.marlin import MarlinLinear

from .base import BaseLLMModel
from .config import ModelConfig


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


__all__ = ["Glm4MoeForCausalLM"]
