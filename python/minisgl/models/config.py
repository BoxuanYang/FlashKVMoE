from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from transformers import PretrainedConfig


@dataclass(frozen=True)
class RotaryConfig:
    head_dim: int
    rotary_dim: int
    max_position: int
    base: float
    scaling: Dict[str, Any] | None


@dataclass(frozen=True)
class ModelConfig:
    num_layers: int
    num_qo_heads: int
    num_kv_heads: int
    head_dim: int
    hidden_size: int
    vocab_size: int
    intermediate_size: int
    rms_norm_eps: float
    rotary_config: RotaryConfig
    hidden_act: str
    tie_word_embeddings: bool
    num_experts: int
    num_experts_per_tok: int
    moe_intermediate_size: int
    norm_topk_prob: bool
    model_type: str
    architectures: list[str]
    q_lora_rank: int = 0
    kv_lora_rank: int = 0
    qk_nope_head_dim: int = 0
    qk_rope_head_dim: int = 0
    v_head_dim: int = 0
    first_k_dense_replace: int = 0
    n_shared_experts: int = 0
    n_group: int = 1
    topk_group: int = 1
    routed_scaling_factor: float = 1.0

    @property
    def is_mla(self) -> bool:
        return self.model_type == "deepseek_v2"

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @classmethod
    def from_hf(cls, config: PretrainedConfig) -> ModelConfig:
        if hasattr(config, "text_config") and config.text_config is not None:
            top = config
            config = config.text_config
            for attr in ("architectures", "rope_theta", "rope_scaling"):
                if not getattr(config, attr, None) and getattr(top, attr, None):
                    setattr(config, attr, getattr(top, attr))

        num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        head_dim = (
            getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        )
        tie_word_embeddings = getattr(config, "tie_word_embeddings", False)
        model_type = getattr(config, "model_type", "llama")
        num_experts = getattr(config, "num_local_experts", getattr(config, "num_experts", 0))
        num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        moe_intermediate_size = getattr(config, "moe_intermediate_size", 0)
        norm_topk_prob = getattr(config, "norm_topk_prob", False)
        architectures = getattr(config, "architectures", ["LlamaForCausalLM"])
        mla = model_type == "deepseek_v2"
        if mla:
            if (config.scoring_func, config.topk_method, config.moe_layer_freq) != (
                "softmax",
                "group_limited_greedy",
                1,
            ):
                raise ValueError(
                    "DeepSeek Coder V2 requires softmax/group_limited_greedy routing and moe_layer_freq=1"
                )
            num_experts = config.n_routed_experts
            head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        # Llama/Qwen: rope_theta is a direct attr; Mistral: it's inside rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None:
            rope_scaling = dict(rope_scaling)
            rope_scaling.setdefault("rope_type", rope_scaling.get("type", "default"))
        rope_theta = getattr(config, "rope_theta", None) or rope_scaling["rope_theta"]

        return cls(
            num_layers=config.num_hidden_layers,
            num_qo_heads=config.num_attention_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            hidden_size=config.hidden_size,
            vocab_size=config.vocab_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            rms_norm_eps=config.rms_norm_eps,
            tie_word_embeddings=tie_word_embeddings,
            rotary_config=RotaryConfig(
                head_dim=config.qk_rope_head_dim if mla else head_dim,
                rotary_dim=config.qk_rope_head_dim if mla else head_dim,
                max_position=config.max_position_embeddings,
                base=rope_theta,
                scaling=rope_scaling,
            ),
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            moe_intermediate_size=moe_intermediate_size,
            norm_topk_prob=norm_topk_prob,
            model_type=model_type,
            architectures=architectures,
            **(
                {
                    name: getattr(config, name)
                    for name in (
                        "q_lora_rank",
                        "kv_lora_rank",
                        "qk_nope_head_dim",
                        "qk_rope_head_dim",
                        "v_head_dim",
                        "first_k_dense_replace",
                        "n_shared_experts",
                        "n_group",
                        "topk_group",
                        "routed_scaling_factor",
                    )
                }
                if mla
                else {}
            ),
        )
