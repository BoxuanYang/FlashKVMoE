# SPDX-License-Identifier: Apache-2.0
# Copyright 2023 DeepSeek-AI and The HuggingFace Inc. team.
# Adapted routing/MLA from KT archive/ktransformers/models/modeling_deepseek.py.
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from minisgl.core import get_global_ctx
from minisgl.layers import (
    BaseOP,
    OPList,
    RMSNorm,
    RMSNormFused,
    VocabParallelEmbedding,
    silu_and_mul,
)
from minisgl.layers.marlin import MarlinLinear
from minisgl.layers.rotary import get_rope
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .config import ModelConfig


def yarn_mscale(factor: float, mscale: float = 1.0) -> float:
    return 1.0 if factor <= 1 else 1.0 + 0.1 * mscale * math.log(factor)


class DeepseekMLP(BaseOP):
    def __init__(self, hidden: int, intermediate: int):
        self.gate_up_proj = MarlinLinear(hidden, 2 * intermediate)
        self.down_proj = MarlinLinear(intermediate, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj.forward(silu_and_mul(self.gate_up_proj.forward(x)))


class DeepseekRouter(BaseOP):
    def __init__(self, config: ModelConfig):
        self.weight = torch.empty(config.num_experts, config.hidden_size, dtype=torch.float32)
        self._config = config

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        c = self._config
        scores = F.linear(x.float(), self.weight).softmax(dim=-1)
        grouped = scores.view(-1, c.n_group, c.num_experts // c.n_group)
        groups = grouped.amax(dim=-1).topk(c.topk_group, dim=-1, sorted=False).indices
        mask = torch.zeros_like(grouped[..., 0], dtype=torch.bool).scatter_(1, groups, True)
        choice = grouped.masked_fill(~mask.unsqueeze(-1), 0).flatten(1)
        weights, ids = choice.topk(c.num_experts_per_tok, dim=-1, sorted=False)
        if c.num_experts_per_tok > 1 and c.norm_topk_prob:
            weights = weights / (weights.sum(-1, keepdim=True) + 1e-20)
        else:
            weights = weights * c.routed_scaling_factor
        return ids, weights


class DeepseekMoE(BaseOP):
    def __init__(self, config: ModelConfig):
        self.gate = DeepseekRouter(config)
        self.shared_experts = DeepseekMLP(
            config.hidden_size, config.n_shared_experts * config.moe_intermediate_size
        )
        self._wrapper = None  # Installed by load_ktransformers_experts before loading GPU weights.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ids, weights = self.gate.forward(x)
        shared = self.shared_experts.forward(x)
        routed = self._wrapper.forward(
            x, ids, weights, torch.cuda.current_stream(x.device).cuda_stream
        )
        return shared + routed


class MLAProjection(BaseOP):
    """Per-head BF16 projection used on either side of absorbed MLA attention."""

    def __init__(self, heads: int, output: int, input: int):
        self.weight = torch.empty(heads, output, input, dtype=torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x.transpose(0, 1), self.weight.transpose(1, 2)).transpose(0, 1)


class DeepseekMLA(BaseOP):
    def __init__(self, c: ModelConfig, layer_id: int):
        self._config, self._layer_id = c, layer_id
        self.q_a_proj = MarlinLinear(c.hidden_size, c.q_lora_rank)
        self.q_a_layernorm = RMSNorm(c.q_lora_rank, c.rms_norm_eps)
        self.q_b_proj = MarlinLinear(c.q_lora_rank, c.num_qo_heads * c.head_dim)
        self.kv_a_proj_with_mqa = MarlinLinear(c.hidden_size, c.kv_lora_rank + c.qk_rope_head_dim)
        self.kv_a_layernorm = RMSNorm(c.kv_lora_rank, c.rms_norm_eps)
        self.k_b_proj = MLAProjection(c.num_qo_heads, c.kv_lora_rank, c.qk_nope_head_dim)
        self.v_b_proj = MLAProjection(c.num_qo_heads, c.v_head_dim, c.kv_lora_rank)
        self.o_proj = MarlinLinear(c.num_qo_heads * c.v_head_dim, c.hidden_size)
        rope = c.rotary_config
        self.rotary = get_rope(
            rope.head_dim,
            rope.rotary_dim,
            rope.max_position,
            rope.base,
            tuple(rope.scaling.items()) if rope.scaling else None,
        )
        scaling = rope.scaling or {}
        factor = scaling.get("factor", 1.0)
        self._rope_scale = yarn_mscale(factor, scaling.get("mscale", 1.0)) / yarn_mscale(
            factor, scaling.get("mscale_all_dim", 0.0)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctx, c = get_global_ctx(), self._config
        q = self.q_b_proj.forward(self.q_a_layernorm.forward(self.q_a_proj.forward(x)))
        q = q.view(-1, c.num_qo_heads, c.head_dim)
        q_nope, q_pe = q.split([c.qk_nope_head_dim, c.qk_rope_head_dim], dim=-1)
        kv = self.kv_a_proj_with_mqa.forward(x)
        latent, k_pe = kv.split([c.kv_lora_rank, c.qk_rope_head_dim], dim=-1)
        latent = self.kv_a_layernorm.forward(latent.contiguous())
        # DeepSeek GGUF preserves interleaved RoPE rows; FlashInfer uses NeoX pairs.
        q_pe = q_pe.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2).contiguous()
        k_pe = k_pe.unflatten(-1, (-1, 2)).transpose(-1, -2).flatten(-2).contiguous()
        q_pe, k_pe = self.rotary.forward(ctx.batch.positions, q_pe.flatten(1), k_pe)
        q_pe = q_pe.view(-1, c.num_qo_heads, c.qk_rope_head_dim) * self._rope_scale
        k_pe = k_pe * self._rope_scale
        q = torch.cat((self.k_b_proj.forward(q_nope), q_pe), dim=-1)
        out = ctx.attn_backend.forward(q, latent, k_pe, self._layer_id, ctx.batch)
        return self.o_proj.forward(self.v_b_proj.forward(out).flatten(1))


class DeepseekDecoderLayer(BaseOP):
    def __init__(self, c: ModelConfig, layer_id: int):
        self._layer_id = layer_id
        self.self_attn = DeepseekMLA(c, layer_id)
        self.mlp = (
            DeepseekMLP(c.hidden_size, c.intermediate_size)
            if layer_id < c.first_k_dense_replace
            else DeepseekMoE(c)
        )
        self.input_layernorm = RMSNormFused(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNormFused(c.hidden_size, c.rms_norm_eps)

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(self, x: torch.Tensor, residual: torch.Tensor | None):
        with torch.cuda.nvtx.range(f"Attention_{self._layer_id}"):
            x, residual = self.input_layernorm.forward(x, residual)
            x = self.self_attn.forward(x)
            x, residual = self.post_attention_layernorm.forward(x, residual)

        with torch.cuda.nvtx.range(f"MoE_{self._layer_id}"):
            y = self.mlp.forward(x)

        return y, residual


class DeepseekModel(BaseOP):
    def __init__(self, c: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(c.vocab_size, c.hidden_size)
        self.layers = OPList([DeepseekDecoderLayer(c, i) for i in range(c.num_layers)])
        self.norm = RMSNormFused(c.hidden_size, c.rms_norm_eps)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x, residual = self.embed_tokens.forward(ids), None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class DeepseekV2ForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = DeepseekModel(config)
        self.lm_head = MarlinLinear(config.hidden_size, config.vocab_size)
        super().__init__()

    def forward(self) -> torch.Tensor:
        batch = get_global_ctx().batch
        x = self.model.forward(batch.input_ids)
        if batch.is_prefill:
            x = x[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        return self.lm_head.forward(x)
