from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch
from minisgl.core import get_global_ctx
from minisgl.layers import BaseOP, OPList, RMSNormFused, VocabParallelEmbedding
from minisgl.layers.marlin import MarlinLinear
from minisgl.utils import nvtx_annotate

from .base import BaseLLMModel
from .utils import MoEMLP as Qwen3MLP
from .utils import RopeAttn as Qwen3Attn

if TYPE_CHECKING:
    from .config import ModelConfig


class Qwen3DecoderLayer(BaseOP):
    def __init__(self, config: ModelConfig, layer_id: int):
        self.self_attn = Qwen3Attn(config, layer_id, has_qk_norm=True)
        self.self_attn.qkv_proj = MarlinLinear(
            config.hidden_size, (config.num_qo_heads + 2 * config.num_kv_heads) * config.head_dim
        )
        self.self_attn.o_proj = MarlinLinear(
            config.num_qo_heads * config.head_dim, config.hidden_size
        )
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

        self._layer_id = layer_id

    @nvtx_annotate("Layer_{}", layer_id_field="_layer_id")
    def forward(
        self, x: torch.Tensor, residual: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, residual = self.input_layernorm.forward(x, residual)
        with torch.cuda.nvtx.range(f"Attention_{self._layer_id}"):
            x = self.self_attn.forward(x)
        x, residual = self.post_attention_layernorm.forward(x, residual)

        # type(self.mlp): <class 'minisgl.moe.ktransformers.KTransformersMoE'>
        with torch.cuda.nvtx.range(f"MoE_{self._layer_id}"):
            x = self.mlp.forward(x)
        return x, residual


class Qwen3Model(BaseOP):
    def __init__(self, config: ModelConfig):
        self.embed_tokens = VocabParallelEmbedding(
            num_embeddings=config.vocab_size,
            embedding_dim=config.hidden_size,
        )
        self.layers = OPList(
            [Qwen3DecoderLayer(config, layer_id) for layer_id in range(config.num_layers)]
        )
        self.norm = RMSNormFused(
            size=config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens.forward(input_ids)
        residual: torch.Tensor | None = None
        for layer in self.layers.op_list:
            x, residual = layer.forward(x, residual)
        return self.norm.forward(x, residual)[0]


class Qwen3MoeForCausalLM(BaseLLMModel):
    def __init__(self, config: ModelConfig):
        self.model = Qwen3Model(config)
        self.lm_head = MarlinLinear(config.hidden_size, config.vocab_size)
        super().__init__()

    def forward(self) -> torch.Tensor:
        output = self.model.forward(get_global_ctx().batch.input_ids)
        batch = get_global_ctx().batch
        if batch.is_prefill:
            output = output[batch.attn_metadata.get_last_indices(batch.size)].contiguous()
        logits = self.lm_head.forward(output)
        return logits


__all__ = ["Qwen3MoeForCausalLM"]
