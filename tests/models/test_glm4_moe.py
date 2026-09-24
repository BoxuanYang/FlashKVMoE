import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import torch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config
from minisgl.layers.marlin import pack_marlin
from minisgl.models import ModelConfig, create_model
from minisgl.models.gguf import GGUFWeights
from minisgl.models.glm4_moe import Glm4MoeMLP, Glm4MoeRouter, Glm4MoeSparseMLP
from minisgl.moe.ktransformers import load_ktransformers_experts
from minisgl.utils import torch_dtype
from transformers import Glm4MoeConfig
from transformers.models.glm4_moe.modeling_glm4_moe import Glm4MoeTopkRouter


def glm_config(**overrides):
    values = {
        "architectures": ["Glm4MoeForCausalLM"],
        "num_hidden_layers": 2,
        "hidden_size": 256,
        "intermediate_size": 192,
        "moe_intermediate_size": 128,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 128,
        "partial_rotary_factor": 0.5,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "first_k_dense_replace": 1,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "attention_bias": True,
        "vocab_size": 32,
        "max_position_embeddings": 128,
    }
    return Glm4MoeConfig(**(values | overrides))


def write_shards(path, config):
    rng = np.random.default_rng(41)
    tensors = {}

    def add(name, shape, norm=False):
        tensors[name] = (
            np.ones(shape, np.float32) if norm else rng.normal(0, 0.03, shape).astype(np.float32)
        )

    add("token_embd.weight", (config.vocab_size, config.hidden_size))
    add("output.weight", (config.vocab_size, config.hidden_size))
    add("output_norm.weight", (config.hidden_size,), True)
    for layer in range(config.num_layers):
        p = f"blk.{layer}"
        for proj, heads in (
            ("q", config.num_qo_heads),
            ("k", config.num_kv_heads),
            ("v", config.num_kv_heads),
        ):
            add(f"{p}.attn_{proj}.weight", (heads * config.head_dim, config.hidden_size))
            add(f"{p}.attn_{proj}.bias", (heads * config.head_dim,))
        add(
            f"{p}.attn_output.weight",
            (config.hidden_size, config.num_qo_heads * config.head_dim),
        )
        add(f"{p}.attn_norm.weight", (config.hidden_size,), True)
        add(f"{p}.post_attention_norm.weight", (config.hidden_size,), True)
        if layer >= config.first_k_dense_replace:
            add(f"{p}.ffn_gate_inp.weight", (config.num_experts, config.hidden_size))
            add(f"{p}.exp_probs_b.bias", (config.num_experts,))
            for proj in ("gate", "up", "down"):
                shape = (
                    (config.num_experts, config.hidden_size, config.moe_intermediate_size)
                    if proj == "down"
                    else (config.num_experts, config.moe_intermediate_size, config.hidden_size)
                )
                add(f"{p}.ffn_{proj}_exps.weight", shape)
        intermediate = (
            config.intermediate_size
            if layer < config.first_k_dense_replace
            else config.moe_intermediate_size * config.n_shared_experts
        )
        suffix = "" if layer < config.first_k_dense_replace else "_shexp"
        for proj in ("gate", "up", "down"):
            shape = (
                (config.hidden_size, intermediate)
                if proj == "down"
                else (intermediate, config.hidden_size)
            )
            add(f"{p}.ffn_{proj}{suffix}.weight", shape)

    entries = list(tensors.items())
    for shard in range(2):
        writer = gguf.GGUFWriter(str(path / f"glm-{shard + 1:05d}-of-00002.gguf"), "glm4moe")
        writer.add_block_count(config.num_layers + 1)
        writer.add_uint32("glm4moe.nextn_predict_layers", 1)
        writer.add_uint16("split.no", shard)
        writer.add_uint16("split.count", 2)
        for name, tensor in entries[shard::2]:
            writer.add_tensor(name, tensor)
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
    return tensors


def test_config_and_router_match_transformers(monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    hf = glm_config()
    config = ModelConfig.from_hf(hf)
    assert config.is_glm4_moe and config.rotary_config.rotary_dim == 64
    assert config.num_experts == 4 and config.first_k_dense_replace == 1

    router = Glm4MoeRouter(config)
    reference = Glm4MoeTopkRouter(hf)
    torch.manual_seed(11)
    router.weight.normal_()
    router.e_score_correction_bias.normal_()
    reference.load_state_dict(router.state_dict())
    x = torch.randn(7, config.hidden_size, dtype=torch.bfloat16)
    for actual, expected in zip(router.forward(x), reference.forward(x)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_two_gguf_shards_load_model_and_kt(tmp_path, monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    config = ModelConfig.from_hf(glm_config())
    tensors = write_shards(tmp_path, config)
    weights = GGUFWeights(str(tmp_path), config)
    cache = {}
    monkeypatch.setitem(
        sys.modules,
        "kt_kernel.utils.llamafile",
        SimpleNamespace(LlamafileMoEWrapper=SimpleNamespace(_gguf_loaders_by_path=cache)),
    )
    factory = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    monkeypatch.setattr("minisgl.layers.attention.get_rope", lambda **kwargs: None)

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    engine = EngineConfig(
        "unused", DistributedInfo(0, 1), torch.bfloat16, kt_weight_path=str(tmp_path)
    )
    engine.__dict__["model_config"] = config
    _adjust_config(engine)
    load_ktransformers_experts(model, engine, gguf_weights=weights)
    state = dict(weights.weights(torch.device("cpu"), torch.bfloat16))
    model.load_state_dict(state.copy())

    layers = model.model.layers.op_list
    assert isinstance(layers[0].mlp, Glm4MoeMLP)
    assert isinstance(layers[1].mlp, Glm4MoeSparseMLP)
    assert factory.call_count == 1 and factory.call_args.kwargs["layer_idx"] == 1
    assert not any("_exps" in name for name in state)
    expected_bias = torch.cat(
        [torch.from_numpy(tensors[f"blk.0.attn_{p}.bias"]) for p in ("q", "k", "v")]
    ).bfloat16()
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_bias"], expected_bias)
    expected = torch.cat(
        [torch.from_numpy(tensors[f"blk.0.attn_{p}.weight"]) for p in ("q", "k", "v")]
    ).bfloat16()
    packed, scales = pack_marlin(expected)
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_proj.weight"], packed)
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_proj.scales"], scales)
    torch.testing.assert_close(
        state["model.layers.1.mlp.gate.e_score_correction_bias"],
        torch.from_numpy(tensors["blk.1.exp_probs_b.bias"]),
    )
