"""CPU contract tests; the real CUDA/KT numerical test is in test_ktransformers_cuda.py."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from minisgl.core import Batch
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config
from minisgl.engine.graph import GraphRunner, _determine_cuda_graph_bs
from minisgl.layers import LinearReplicated
from minisgl.models import ModelConfig, create_model
from minisgl.moe.ktransformers import KTransformersMoE, load_ktransformers_experts
from minisgl.server.args import parse_args
from transformers import Qwen3MoeConfig


def make_config(**kwargs):
    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_backend="kt",
        kt_weight_path="experts.gguf",
        attention_backend="fi",
        **kwargs,
    )
    hf = Qwen3MoeConfig(
        architectures=["Qwen3MoeForCausalLM"],
        num_hidden_layers=2,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=512,
        max_position_embeddings=128,
    )
    config.__dict__["model_config"] = ModelConfig.from_hf(hf)
    return config


@pytest.fixture
def single_rank(monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))


@pytest.fixture
def wrapper_factory(monkeypatch, single_rank):
    factory = MagicMock(side_effect=lambda **kwargs: MagicMock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    return factory


@pytest.mark.parametrize(
    "layers,hidden,intermediate,heads", [(48, 2048, 768, 32), (94, 4096, 1536, 64)]
)
def test_full_qwen_shapes_have_no_expert_state(
    wrapper_factory, monkeypatch, layers, hidden, intermediate, heads
):
    config = make_config()
    config.__dict__["model_config"] = replace(
        config.model_config,
        num_layers=layers,
        hidden_size=hidden,
        moe_intermediate_size=intermediate,
        num_qo_heads=heads,
        num_kv_heads=4,
        num_experts=128,
        num_experts_per_tok=8,
        vocab_size=151936,
    )
    monkeypatch.setattr("minisgl.layers.attention.get_rope", lambda **kwargs: None)
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    with torch.device("meta"):
        model = create_model(config.model_config)
    gates = [layer.mlp.gate for layer in model.model.layers.op_list]
    before = model.state_dict()
    load_ktransformers_experts(model, config)
    after = model.state_dict()
    assert set(after) == {name for name in before if ".experts." not in name}
    assert all(after[name] is before[name] for name in after)
    for layer, gate in zip(model.model.layers.op_list, gates):
        assert isinstance(layer.mlp, KTransformersMoE)
        assert layer.mlp.gate is gate
        assert not hasattr(layer.mlp, "experts")
    assert wrapper_factory.call_count == layers
    for layer_id, call in enumerate(wrapper_factory.call_args_list):
        assert call.kwargs["layer_idx"] == layer_id
        assert call.kwargs["gpu_experts_mask"] is None
        assert call.kwargs["max_deferred_experts_per_token"] == 0
        assert call.kwargs["moe_intermediate_size"] == intermediate


@pytest.mark.parametrize("tokens", [1, 17])
@pytest.mark.parametrize("renormalize", [False, True])
def test_routing_and_stream_contract(wrapper_factory, monkeypatch, tokens, renormalize):
    config = make_config(max_seq_len_override=8, max_running_req=32)
    config.__dict__["model_config"] = replace(config.model_config, norm_topk_prob=renormalize)
    gate = LinearReplicated(256, 4, has_bias=False)
    gate.weight = torch.empty(4, 256, device="meta", dtype=torch.bfloat16)
    layer = KTransformersMoE(config, 0, gate=gate)
    router_weight = torch.zeros(4, 256, dtype=torch.bfloat16)
    router_weight[:, 0] = torch.tensor([1.0, -2.0, 4.0, 0.0])
    layer.load_state_dict({"gate.weight": router_weight})
    assert layer.gate.weight is router_weight
    assert set(layer.state_dict()) == {"gate.weight"}
    assert wrapper_factory.call_args.kwargs["chunked_prefill_size"] == 32
    mapping = layer._wrapper.load_weights.call_args.args[0]
    torch.testing.assert_close(mapping, torch.arange(4, dtype=torch.int32))
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123)
    )
    x = torch.randn(tokens, 256, dtype=torch.bfloat16)
    x[:, 0] = 1
    logits = torch.tensor([1.0, -2.0, 4.0, 0.0]).expand(tokens, -1)
    result = layer.forward(x)
    states, ids, weights, stream = layer._wrapper.forward.call_args.args
    assert states is x and stream == 123
    assert result is layer._wrapper.forward.return_value
    expected_ids = torch.tensor([2, 0]).expand(tokens, -1)
    expected_weights = logits.softmax(-1).gather(-1, expected_ids)
    if renormalize:
        expected_weights /= expected_weights.sum(-1, keepdim=True)
    torch.testing.assert_close(ids, expected_ids)
    torch.testing.assert_close(weights, expected_weights)
    assert weights.dtype == torch.float32


@pytest.mark.parametrize(
    "sizes,limit", [(None, 100), ([1, 2], 128), (None, 0), ([], 128), (None, None)]
)
def test_kt_preserves_graph_settings(sizes, limit):
    config = make_config(cuda_graph_bs=sizes, cuda_graph_max_bs=limit)
    _adjust_config(config)
    assert config.cuda_graph_bs == sizes and config.cuda_graph_max_bs == limit


@pytest.mark.parametrize("limit", [0, 1, 2, 3, 4, 7, 8, 17, 100])
def test_graph_sizes_respect_requested_limit(limit):
    sizes = _determine_cuda_graph_bs(None, limit, 0)
    if limit == 0:
        assert sizes == []
    else:
        assert sizes == sorted(set(sizes))
        assert sizes[0] == 1 and sizes[-1] == limit
        assert all(0 < bs <= limit for bs in sizes)


@pytest.mark.parametrize(
    "limit,expected",
    [
        (12, [1, 2, 4, 8, 12]),
        (20, [1, 2, 4, 8, 12, 16, 20]),
        (24, [1, 2, 4, 8, 12, 16, 24]),
        (32, [1, 2, 4, 8, 12, 16, 24, 32]),
        (100, [1, 2, 4, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 100]),
    ],
)
def test_graph_capture_schedule(limit, expected):
    assert _determine_cuda_graph_bs(None, limit, 0) == expected


@pytest.mark.parametrize(
    "phase,size,selected",
    [
        ("decode", 3, 4),
        ("decode", 9, 12),
        ("decode", 12, 12),
        ("decode", 17, 24),
        ("decode", 25, None),
        ("prefill", 9, None),
    ],
)
def test_graph_selective_replay(phase, size, selected):
    runner = GraphRunner.__new__(GraphRunner)
    runner.graph_bs_list = _determine_cuda_graph_bs(None, 24, 0)
    runner.max_graph_bs = 24
    runner.dummy_req = object()
    runner.graph_map = {bs: MagicMock() for bs in runner.graph_bs_list}
    runner.buffer = MagicMock(logits=torch.zeros(24, 2))
    runner.attn_backend = MagicMock()
    batch = Batch(reqs=[object() for _ in range(size)], phase=phase)
    runner.pad_batch(batch)
    assert batch.padded_reqs[:size] == batch.reqs
    if selected is None:
        assert not runner.can_use_cuda_graph(batch)
        assert batch.padded_size == size
    else:
        assert runner.can_use_cuda_graph(batch)
        assert batch.padded_size == selected
        assert all(req is runner.dummy_req for req in batch.padded_reqs[size:])
        assert runner.replay(batch).shape == (size, 2)
        runner.buffer.copy_from.assert_called_once_with(batch)
        runner.attn_backend.prepare_for_replay.assert_called_once_with(batch)
    for bs, graph in runner.graph_map.items():
        assert graph.replay.call_count == int(bs == selected)


def test_kt_graph_buffers_cover_capture_padding(wrapper_factory):
    config = make_config(max_running_req=3, max_seq_len_override=8)
    layers = [
        SimpleNamespace(mlp=SimpleNamespace(gate=LinearReplicated(256, 4, has_bias=False)))
        for _ in range(2)
    ]
    model = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=layers)))
    load_ktransformers_experts(model, config, [1, 2, 4, 16])
    wrapper_factory.set_capture_batch_sizes.assert_called_once_with([1, 2, 4, 16])
    assert wrapper_factory.call_count == 2
    assert all(call.kwargs["chunked_prefill_size"] == 16 for call in wrapper_factory.call_args_list)


def test_kt_graph_rejects_synchronous_submit(wrapper_factory, monkeypatch):
    monkeypatch.setenv("KT_FORCE_SYNC_SUBMIT", "1")
    model = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=[])))
    with pytest.raises(ValueError, match="KT_FORCE_SYNC_SUBMIT"):
        load_ktransformers_experts(model, make_config(), [1])
    wrapper_factory.assert_not_called()
    # The explicit eager mode remains usable with this debugging override.
    load_ktransformers_experts(model, make_config(), [])


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_info": DistributedInfo(0, 2)},
        {"dtype": torch.float16},
        {"kt_weight_path": None},
        {"kt_method": "BF16"},
        {"kt_cpuinfer": 0},
        {"kt_threadpool_count": 0},
        {"max_running_req": 0},
        {"max_seq_len_override": 0},
        {"moe_backend": "fused"},
    ],
)
def test_invalid_config_fails_before_loading(changes):
    config = make_config()
    for key, value in changes.items():
        object.__setattr__(config, key, value)
    with pytest.raises(ValueError):
        _adjust_config(config)


def test_cli():
    config, _ = parse_args(
        [
            "--model",
            "unused",
            "--dtype",
            "bfloat16",
            "--moe-backend",
            "kt",
            "--kt-weight-path",
            "/models/gguf",
            "--kt-cpuinfer",
            "128",
            "--kt-threadpool-count",
            "2",
            "--kt-method",
            "LLAMAFILE",
            "--attention-backend",
            "fi",
            "--num-pages",
            "2048",
            "--page-size",
            "1",
            "--cuda-graph-max-bs",
            "100",
        ]
    )
    assert config.moe_backend == "kt" and config.kt_weight_path == "/models/gguf"
    assert config.kt_cpuinfer == 128 and config.kt_threadpool_count == 2
    assert config.num_page_override * config.page_size == 2048
    assert config.cuda_graph_max_bs == 100
