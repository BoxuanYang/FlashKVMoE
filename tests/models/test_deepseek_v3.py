"""Archive numerical references, real GGUF layouts, KT ordering and CUDA MLA."""

import ast
import math
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import Engine, _adjust_config
from minisgl.kvcache import create_kvcache_pool
from minisgl.layers.marlin import pack_marlin
from minisgl.models import ModelConfig, create_model
from minisgl.models.deepseek_v3 import DeepseekMLA, DeepseekMLP, DeepseekMoEWrapper, DeepseekRouter
from minisgl.models.gguf import GGUFWeights
from minisgl.moe.ktransformers import load_ktransformers_experts
from minisgl.utils import torch_dtype


def hf_config(**overrides):
    values = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "deepseek_v3",
        "num_hidden_layers": 2,
        "hidden_size": 128,
        "vocab_size": 64,
        "intermediate_size": 256,
        "moe_intermediate_size": 256,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "q_lora_rank": 128,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "n_routed_experts": 8,
        "n_shared_experts": 1,
        "n_group": 2,
        "topk_group": 1,
        "num_experts_per_tok": 2,
        "routed_scaling_factor": 2.5,
        "norm_topk_prob": True,
        "scoring_func": "sigmoid",
        "topk_method": "noaux_tc",
        "rms_norm_eps": 1e-6,
        "hidden_act": "silu",
        "attention_bias": False,
        "attention_dropout": 0.0,
        "tie_word_embeddings": False,
        "max_position_embeddings": 8192,
        "rope_theta": 10000.0,
        "rope_scaling": {
            "type": "yarn",
            "factor": 40,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
        },
    }
    return SimpleNamespace(**(values | overrides))


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    return ModelConfig.from_hf(hf_config())


@pytest.fixture(scope="module")
def archive():
    path = Path(__file__).resolve().parents[2] / (
        "third_party/ktransformers/archive/ktransformers/models/modeling_deepseek_v3.py"
    )
    names = {
        "MoEGate",
        "DeepseekV3RMSNorm",
        "DeepseekV3Attention",
        "DeepseekV3RotaryEmbedding",
        "DeepseekV3YarnRotaryEmbedding",
        "rotate_half",
        "apply_rotary_pos_emb",
        "yarn_find_correction_dim",
        "yarn_find_correction_range",
        "yarn_get_mscale",
        "yarn_linear_ramp_mask",
    }
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [
        n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names
    ]
    namespace = {
        "torch": torch,
        "nn": torch.nn,
        "F": F,
        "math": math,
        "Optional": Optional,
        "Tuple": Tuple,
        "Cache": object,
        "DeepseekV3Config": SimpleNamespace,
    }
    exec(compile(tree, str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


@pytest.mark.parametrize("tokens,normalize", [(1, True), (17, True), (17, False)])
def test_router_matches_archive(config, archive, tokens, normalize):
    c = replace(config, norm_topk_prob=normalize)
    gate = DeepseekRouter(c)
    ref = archive.MoEGate(hf_config(norm_topk_prob=normalize))
    torch.manual_seed(31)
    gate.weight.normal_(std=0.03)
    gate.e_score_correction_bias.copy_(torch.linspace(-2, 2, c.num_experts))
    ref.load_state_dict(gate.state_dict())
    x = torch.randn(tokens, c.hidden_size, dtype=torch.bfloat16)
    actual = gate.forward(x)
    expected = ref(x.unsqueeze(0))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Correction bias changes selection only, never the selected experts' weights.
    weights = F.linear(x.float(), gate.weight).sigmoid().gather(1, actual[0])
    if normalize:
        weights /= weights.sum(-1, keepdim=True)
    torch.testing.assert_close(actual[1], weights * c.routed_scaling_factor)


def test_shared_expert_runs_between_kt_submit_and_sync(config, monkeypatch):
    events = []
    layer = DeepseekMoEWrapper(config)
    x = torch.randn(3, config.hidden_size)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=71)
    )

    def submit(actual_x, ids, weights, stream):
        assert actual_x is x and stream == 71
        assert ids.shape == weights.shape == (3, 2)
        events.append("submit")

    def shared(actual_x):
        assert events == ["submit"] and actual_x is x
        events.append("shared")
        return x * 2

    def sync(actual_x, stream):
        assert events == ["submit", "shared"] and actual_x is x and stream == 71
        events.append("sync")
        return x * 3

    layer.gate.weight.zero_()
    layer.gate.e_score_correction_bias.copy_(torch.arange(8))
    layer._wrapper = SimpleNamespace(submit_forward=submit, sync_forward=sync)
    layer.shared_experts = SimpleNamespace(forward=shared)
    torch.testing.assert_close(layer.forward(x), x * 5)
    assert events == ["submit", "shared", "sync"]


def write_v3(path, c, split=False):
    """Independent HF-to-GGUF layout, including expert tensors never unpacked by MiniSGL."""
    rng = np.random.default_rng(17)
    tensors = {}

    def add(name, shape, norm=False):
        tensors[name] = (
            np.ones(shape, np.float32) if norm else rng.normal(0, 0.03, shape).astype(np.float32)
        )

    add("token_embd.weight", (c.vocab_size, c.hidden_size))
    add("output.weight", (c.vocab_size, c.hidden_size))
    add("output_norm.weight", (c.hidden_size,), True)
    for i in range(c.num_layers):
        p = f"blk.{i}"
        add(f"{p}.attn_q_a.weight", (c.q_lora_rank, c.hidden_size))
        add(f"{p}.attn_q_a_norm.weight", (c.q_lora_rank,), True)
        add(f"{p}.attn_q_b.weight", (c.num_qo_heads * c.head_dim, c.q_lora_rank))
        add(f"{p}.attn_kv_a_mqa.weight", (c.kv_lora_rank + c.qk_rope_head_dim, c.hidden_size))
        add(f"{p}.attn_kv_a_norm.weight", (c.kv_lora_rank,), True)
        add(
            f"{p}.attn_kv_b.weight",
            (c.num_qo_heads * (c.qk_nope_head_dim + c.v_head_dim), c.kv_lora_rank),
        )
        add(f"{p}.attn_output.weight", (c.hidden_size, c.num_qo_heads * c.v_head_dim))
        add(f"{p}.attn_norm.weight", (c.hidden_size,), True)
        add(f"{p}.ffn_norm.weight", (c.hidden_size,), True)
        if i >= c.first_k_dense_replace:
            add(f"{p}.ffn_gate_inp.weight", (c.num_experts, c.hidden_size))
            add(f"{p}.exp_probs_b.bias", (c.num_experts,))
            for proj in ("gate", "up", "down"):
                dims = (
                    (c.hidden_size, c.moe_intermediate_size)
                    if proj == "down"
                    else (c.moe_intermediate_size, c.hidden_size)
                )
                add(f"{p}.ffn_{proj}_exps.weight", (c.num_experts, *dims))
        mid = (
            c.intermediate_size
            if i < c.first_k_dense_replace
            else c.moe_intermediate_size * c.n_shared_experts
        )
        suffix = "" if i < c.first_k_dense_replace else "_shexp"
        for proj in ("gate", "up", "down"):
            add(
                f"{p}.ffn_{proj}{suffix}.weight",
                (c.hidden_size, mid) if proj == "down" else (mid, c.hidden_size),
            )
    writer = gguf.GGUFWriter(str(path), "deepseek2")
    writer.add_float32("deepseek2.rope.freq_base", c.rotary_config.base)
    # llama.cpp's metadata describes a single compressed KV head, unlike HF config.
    for key, value in {
        "block_count": c.num_layers,
        "attention.head_count": c.num_qo_heads,
        "attention.head_count_kv": 1,
        "attention.key_length": c.kv_lora_rank + c.qk_rope_head_dim,
        "attention.value_length": c.kv_lora_rank,
        "attention.key_length_mla": c.head_dim,
        "attention.value_length_mla": c.v_head_dim,
        "attention.q_lora_rank": c.q_lora_rank,
        "attention.kv_lora_rank": c.kv_lora_rank,
        "leading_dense_block_count": c.first_k_dense_replace,
        "expert_shared_count": c.n_shared_experts,
    }.items():
        writer.add_uint32(f"deepseek2.{key}", value)
    for name, tensor in tensors.items():
        if split and name.endswith("attn_kv_b.weight"):
            kv = tensor.reshape(c.num_qo_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
            writer.add_tensor(
                name.replace("kv_b", "k_b"), kv[:, : c.qk_nope_head_dim].transpose(0, 2, 1).copy()
            )
            writer.add_tensor(name.replace("kv_b", "v_b"), kv[:, c.qk_nope_head_dim :].copy())
        else:
            writer.add_tensor(name, tensor)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return tensors


@pytest.mark.parametrize("split", [False, True])
def test_gguf_and_model_state(config, tmp_path, monkeypatch, split):
    path = tmp_path / "v3.gguf"
    tensors = write_v3(path, config, split)
    factory = MagicMock(side_effect=lambda **kwargs: MagicMock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    monkeypatch.setattr("minisgl.models.deepseek_v3.get_rope", lambda *args: None)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    engine = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, kt_weight_path=str(path))
    engine.__dict__["model_config"] = config
    _adjust_config(engine)
    load_ktransformers_experts(model, engine)
    assert isinstance(model.model.layers.op_list[0].mlp, DeepseekMLP)
    assert isinstance(model.model.layers.op_list[1].mlp, DeepseekMoEWrapper)
    assert factory.call_count == 1 and factory.call_args.kwargs["layer_idx"] == 1
    loader = GGUFWeights(str(path), config)
    dequantize = loader._dequantize

    def checked(name, *args):
        assert "_exps" not in name
        return dequantize(name, *args)

    monkeypatch.setattr(loader, "_dequantize", checked)
    state = dict(loader.weights(torch.device("cpu"), torch.bfloat16))
    kv = torch.from_numpy(tensors["blk.0.attn_kv_b.weight"]).bfloat16().view(2, 256, 512)
    torch.testing.assert_close(
        state["model.layers.0.self_attn.k_b_proj.weight"], kv[:, :128].transpose(1, 2)
    )
    torch.testing.assert_close(state["model.layers.0.self_attn.v_b_proj.weight"], kv[:, 128:])
    assert state["model.layers.1.mlp.gate.weight"].dtype == torch.float32
    for i, suffix, target in [(0, "", "mlp"), (1, "_shexp", "mlp.shared_experts")]:
        w = torch.cat(
            [
                torch.from_numpy(tensors[f"blk.{i}.ffn_{p}{suffix}.weight"]).bfloat16()
                for p in ("gate", "up")
            ]
        )
        packed, scales = pack_marlin(w)
        torch.testing.assert_close(state[f"model.layers.{i}.{target}.gate_up_proj.weight"], packed)
        torch.testing.assert_close(state[f"model.layers.{i}.{target}.gate_up_proj.scales"], scales)
    model.load_state_dict(state)
    assert not any(".experts." in name for name in model.state_dict())
    with pytest.raises(ValueError, match="rope.freq_base.*does not match"):
        GGUFWeights(
            str(path), replace(config, rotary_config=replace(config.rotary_config, base=20000))
        )


def test_v3_cache_storage_and_budget(config, monkeypatch):
    pool = create_kvcache_pool(config, 17, 1, torch.bfloat16, torch.device("cpu"))
    k, v = torch.randn(3, 512).bfloat16(), torch.randn(3, 64).bfloat16()
    indices = torch.tensor([8, 2, 15], dtype=torch.int32)
    pool.store_kv(k, v, indices, 1)
    torch.testing.assert_close(pool.k_cache(1)[indices.long(), 0], k)
    torch.testing.assert_close(pool.v_cache(1)[indices.long(), 0], v)
    assert pool._buffer.numel() == 2 * 17 * 576
    args = EngineConfig("unused", DistributedInfo(0, 1), torch.bfloat16, memory_ratio=1.0)
    args.__dict__["model_config"] = config
    engine = Engine.__new__(Engine)
    engine.dtype = torch.bfloat16
    per_page = config.num_layers * 576 * 2
    monkeypatch.setattr(engine, "_sync_get_memory", lambda: (per_page * 17, per_page * 17))
    assert engine._determine_num_pages(per_page * 17, args) == 17


def test_full_v3_gpu_weight_budget(config, monkeypatch):
    config = ModelConfig.from_hf(
        hf_config(
            num_hidden_layers=61,
            first_k_dense_replace=3,
            hidden_size=7168,
            vocab_size=129280,
            intermediate_size=18432,
            moe_intermediate_size=2048,
            num_attention_heads=128,
            num_key_value_heads=128,
            q_lora_rank=1536,
            n_routed_experts=256,
            n_group=8,
            topk_group=4,
            num_experts_per_tok=8,
        )
    )
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    monkeypatch.setattr("minisgl.models.deepseek_v3.get_rope", lambda *args: None)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    state = model.state_dict()
    size = sum(t.numel() * t.element_size() for t in state.values())
    assert size < 12 * 1024**3
    assert sum(isinstance(l.mlp, DeepseekMLP) for l in model.model.layers.op_list) == 3
    assert sum(isinstance(l.mlp, DeepseekMoEWrapper) for l in model.model.layers.op_list) == 58
    assert not any(".experts." in name for name in state)


def test_yarn_frequencies_match_archive(config, archive, monkeypatch):
    from minisgl.layers.rotary import _get_rope

    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    rope = _get_rope(64, 64, 8192, 10000, config.rotary_config.scaling)
    ref = archive.DeepseekV3YarnRotaryEmbedding(
        64, 8192, scaling_factor=40, mscale=1, mscale_all_dim=1
    )
    torch.testing.assert_close(
        rope._cos_sin_cache[:, :32], ref.cos_cached[:, :32], atol=0.002, rtol=0.002
    )
    torch.testing.assert_close(
        rope._cos_sin_cache[:, 32:], ref.sin_cached[:, :32], atol=0.002, rtol=0.002
    )


@pytest.mark.parametrize(
    "device_name",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                sys.platform != "linux" or not torch.cuda.is_available(),
                reason="requires Linux CUDA",
            ),
        ),
    ],
)
def test_mla_against_archive_attention(config, archive, tmp_path, monkeypatch, device_name):
    from minisgl.layers import set_rope_device
    from minisgl.layers.rotary import get_rope
    from test_marlin import quantized_reference

    device = torch.device(device_name)
    hf = hf_config(num_attention_heads=16, num_key_value_heads=16, num_hidden_layers=1)
    c = ModelConfig.from_hf(hf)
    path = tmp_path / "mla.gguf"
    tensors = write_v3(path, c)
    ctx = SimpleNamespace(
        kv_cache=create_kvcache_pool(c, 64, 1, torch.bfloat16, device),
        page_table=torch.randperm(64, device=device).int().view(1, -1),
    )
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", ctx)
    get_rope.cache_clear()

    set_rope_device(device)
    if device_name == "cpu":
        # CPU emulation of just the FlashInfer primitives; compare model math
        # against archive's independent, non-absorbed Attention implementation.
        def rmsnorm(x, weight, eps):
            return (
                x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps) * weight
            ).to(x.dtype)

        def apply_rope(positions, query, key, head_size, cos_sin_cache):
            cos, sin = cos_sin_cache[positions.long()].chunk(2, dim=-1)
            cos, sin = cos.repeat(1, 2)[:, None], sin.repeat(1, 2)[:, None]
            for t in (query, key):
                shaped = t.view(t.shape[0], -1, head_size)
                rotated = torch.cat(
                    (-shaped[..., head_size // 2 :], shaped[..., : head_size // 2]), -1
                )
                t.copy_((shaped.float() * cos + rotated.float() * sin).flatten(1).to(t.dtype))

        monkeypatch.setitem(
            sys.modules,
            "flashinfer",
            SimpleNamespace(rmsnorm=rmsnorm, apply_rope_with_cos_sin_cache_inplace=apply_rope),
        )
        from minisgl.models.deepseek_v3 import yarn_mscale

        scale = c.head_dim**-0.5 * yarn_mscale(40, 1) ** 2
        cached_k, cached_v = [], []

        def forward(q, k, v, layer_id, batch):
            cached_k.append(torch.cat((k, v), -1))
            cached_v.append(k)
            keys, values = torch.cat(cached_k), torch.cat(cached_v)
            mask = torch.arange(len(keys))[None, :] <= batch.positions[:, None]
            return (
                F.scaled_dot_product_attention(
                    q.transpose(0, 1).float(),
                    keys[None].float(),
                    values[None].float(),
                    attn_mask=mask,
                    scale=scale,
                )
                .transpose(0, 1)
                .bfloat16()
            )

        ctx.attn_backend = SimpleNamespace(forward=forward)
    else:
        from minisgl.attention import create_attention_backend

        ctx.attn_backend = create_attention_backend("fi", c)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        attn = DeepseekMLA(c, 0)
    state = dict(GGUFWeights(str(path), c).weights(device, torch.bfloat16))
    prefix = "model.layers.0.self_attn."
    attn.load_state_dict(
        {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    )
    ref = (
        archive.DeepseekV3Attention(hf, layer_idx=0).to(device=device, dtype=torch.bfloat16).eval()
    )
    ref_state = {}
    for name, source in {
        "q_a_proj": "attn_q_a",
        "q_a_layernorm": "attn_q_a_norm",
        "q_b_proj": "attn_q_b",
        "kv_a_proj_with_mqa": "attn_kv_a_mqa",
        "kv_a_layernorm": "attn_kv_a_norm",
        "kv_b_proj": "attn_kv_b",
        "o_proj": "attn_output",
    }.items():
        w = torch.from_numpy(tensors[f"blk.0.{source}.weight"]).bfloat16()
        if name in ("q_a_proj", "q_b_proj", "kv_a_proj_with_mqa", "o_proj"):
            w = quantized_reference(w)
            if device_name == "cpu":
                monkeypatch.setattr(getattr(attn, name), "forward", lambda x, w=w: F.linear(x, w))
        ref_state[name + ".weight"] = w.to(device)
    ref.load_state_dict(ref_state)

    class Cache:
        k = v = None

        def get_usable_length(self, length, layer):
            return 0 if self.k is None else self.k.shape[2]

        def update(self, k, v, layer, kwargs):
            self.k = k if self.k is None else torch.cat((self.k, k), 2)
            self.v = v if self.v is None else torch.cat((self.v, v), 2)
            return self.k, self.v

    cache, cached = Cache(), 0
    with torch.inference_mode():
        for length in (5, 1, 3, 1):
            positions = torch.arange(cached, cached + length, device=device)
            ctx.batch = SimpleNamespace(
                positions=positions,
                out_loc=ctx.page_table[0, cached : cached + length],
                padded_reqs=[
                    SimpleNamespace(extend_len=length, device_len=cached + length, table_idx=0)
                ],
            )
            if device_name == "cuda":
                ctx.attn_backend.prepare_metadata(ctx.batch)
            x = torch.randn(length, c.hidden_size, device=device, dtype=torch.bfloat16) * 0.3
            allowed = torch.arange(cached + length, device=device)[None, :] <= positions[:, None]
            mask = torch.zeros(length, cached + length, device=device).masked_fill_(
                ~allowed, -torch.inf
            )[None, None]
            expected = ref(
                x[None], attention_mask=mask, position_ids=positions[None], past_key_value=cache
            )[0][0]
            actual = attn.forward(x)
            torch.testing.assert_close(actual, expected, atol=0.006, rtol=0.06)
            assert (actual.float() - expected.float()).norm() / expected.float().norm() < 0.03
            cached += length
    get_rope.cache_clear()


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(), reason="requires Linux CUDA"
)
def test_flashinfer_mla_noncontiguous_pages_and_graph(config, monkeypatch):
    from minisgl.attention.fi_mla import FlashInferMLABackend
    from minisgl.core import Batch

    c = replace(config, num_qo_heads=128, num_kv_heads=128)
    device = torch.device("cuda")
    ctx = SimpleNamespace(
        kv_cache=create_kvcache_pool(c, 65, 1, torch.bfloat16, device),
        page_table=torch.randperm(64, device=device).int().view(2, 32),
    )
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", ctx)
    backend = FlashInferMLABackend(c)
    lengths, keys, values = [0, 0], [[], []], [[], []]

    def inputs(counts):
        reqs, locs = [], []
        for i, n in enumerate(counts):
            reqs.append(SimpleNamespace(table_idx=i, extend_len=n, device_len=lengths[i] + n))
            locs.append(ctx.page_table[i, lengths[i] : lengths[i] + n])
        batch = Batch(reqs=reqs, phase="decode" if counts == (1, 1) else "prefill")
        batch.padded_reqs = reqs
        batch.out_loc = torch.cat(locs)
        q = torch.randn(sum(counts), 128, 576, device=device, dtype=torch.bfloat16) * 0.1
        k = torch.randn(sum(counts), 512, device=device, dtype=torch.bfloat16)
        v = torch.randn(sum(counts), 64, device=device, dtype=torch.bfloat16)
        expected, start = [], 0
        for i, n in enumerate(counts):
            keys[i].append(torch.cat((k[start : start + n], v[start : start + n]), -1))
            values[i].append(k[start : start + n].clone())
            all_k, all_v = torch.cat(keys[i]), torch.cat(values[i])
            mask = torch.arange(len(all_k), device=device)[None, :] <= (
                lengths[i] + torch.arange(n, device=device)[:, None]
            )
            expected.append(
                F.scaled_dot_product_attention(
                    q[start : start + n].transpose(0, 1).float(),
                    all_k[None].float(),
                    all_v[None].float(),
                    attn_mask=mask,
                    scale=backend.sm_scale,
                )
                .transpose(0, 1)
                .bfloat16()
            )
            lengths[i] += n
            start += n
        return batch, q, k, v, torch.cat(expected)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for counts in ((4, 2), (1, 1), (2, 3), (1, 1)):
            batch, q, k, v, expected = inputs(counts)
            backend.prepare_metadata(batch)
            actual = backend.forward(q, k, v, 0, batch)
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
        backend.init_capture_graph(32, [2])
        backend.prepare_for_capture(batch)
        backend.forward(q, k, v, 0, batch)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = backend.forward(q, k, v, 0, batch)
        for _ in range(3):
            updated, next_q, next_k, next_v, expected = inputs((1, 1))
            q.copy_(next_q)
            k.copy_(next_k)
            v.copy_(next_v)
            batch.out_loc.copy_(updated.out_loc)
            backend.prepare_metadata(updated)
            backend.prepare_for_replay(updated)
            actual.fill_(float("nan"))
            graph.replay()
            stream.synchronize()
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.03)
