import sys
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from minisgl.core import Batch, Context, Req
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config
from minisgl.models import ModelConfig
from minisgl.models.gguf import GGUFWeights
from minisgl.models.glm4_moe import Glm4MoeForCausalLM as SequentialGlm4MoeForCausalLM
from minisgl.moe.ktransformers import load_ktransformers_experts
from minisgl.utils import torch_dtype
from test_glm4_moe import glm_config, write_shards

import GLMDualBatchRunner as glm_dual
from GLMDualBatchRunner import Glm4MoeMLP, Glm4MoeSparseMLP


class Stream:
    def __init__(self):
        self.cuda_stream = id(self)
        self.sync_count = 0

    def synchronize(self):
        self.sync_count += 1


def make_batches(sizes, device="cpu", step=0):
    result, row = [], 0
    for size in sizes:
        reqs = [
            Req(torch.zeros(step + 1, dtype=torch.int32), row + i, step, 4, row + i, None, None)
            for i in range(size)
        ]
        batch = Batch(reqs, "decode")
        batch.padded_reqs = reqs
        batch.input_ids = torch.arange(row + 1, row + size + 1, device=device)
        batch.positions = torch.full((size,), step, device=device, dtype=torch.int32)
        batch.out_loc = torch.arange(row, row + size, device=device, dtype=torch.int32)
        result.append(batch)
        row += size
    return tuple(result)


@pytest.fixture
def simulation(monkeypatch):
    model = glm_dual.Glm4MoeModel.__new__(glm_dual.Glm4MoeModel)
    runner = glm_dual.GLMDualBatchRunner.__new__(glm_dual.GLMDualBatchRunner)
    stream = Stream()
    pending, buffers = [], {}
    order = []

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: stream)
    monkeypatch.setattr(torch.cuda, "stream", lambda actual: nullcontext())

    ctx = Context(1)
    original_backend = object()
    ctx.attn_backend = original_backend
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", ctx)

    class Attention:
        def prepare_metadata(self, batch):
            batch.attn_metadata = SimpleNamespace(wrapper=self)

    runner.attention = (Attention(), Attention())

    def norm(x, residual):
        residual = x if residual is None else x + residual
        return residual * 0.625, residual

    def attn(x):
        assert ctx.batch.attn_metadata.wrapper is ctx.attn_backend
        return x * 0.75 + ctx.batch.positions[:, None].to(x.dtype) * 0.01

    def gate(x):
        scores = x[:, :4].float().sigmoid()
        weights, ids = scores.topk(2, sorted=False)
        return ids, weights * 1.25

    def shared(x):
        return x * 0.25

    def routed(x, ids, weights, layer):
        coefficient = ((ids.float() + 1) * weights).sum(-1, keepdim=True)
        return (x.float() * coefficient * (0.01 * layer)).bfloat16()

    class KT:
        def __init__(self, layer):
            self.layer = layer

        def submit_forward(self, x, ids, weights, stream):
            assert stream == runner.engine.stream.cuda_stream and not pending
            pending.append((x, routed(x, ids, weights, self.layer)))
            order.append(("submit", self.layer))

        def sync_forward(self, x, stream):
            assert stream == runner.engine.stream.cuda_stream and len(pending) == 1
            original, output = pending.pop()
            assert original is x
            key = (x.shape[0], self.layer)
            buffer = buffers.setdefault(key, torch.empty_like(x))
            buffer.copy_(output)
            order.append(("sync", self.layer))
            return buffer

    layers = []
    for layer_id in range(3):
        if layer_id == 0:
            mlp = Glm4MoeMLP.__new__(Glm4MoeMLP)
            mlp.forward = lambda x: x * 0.375
        else:
            mlp = Glm4MoeSparseMLP.__new__(Glm4MoeSparseMLP)
            mlp.gate = SimpleNamespace(forward=gate)
            mlp.shared_experts = SimpleNamespace(forward=shared)
            mlp._wrapper = KT(layer_id)
        layers.append(
            SimpleNamespace(
                input_layernorm=SimpleNamespace(forward=norm),
                post_attention_layernorm=SimpleNamespace(forward=norm),
                self_attn=SimpleNamespace(forward=attn),
                mlp=mlp,
            )
        )

    embedding = torch.arange(64 * 8).view(64, 8).bfloat16() / 512
    model.layers = SimpleNamespace(op_list=layers)
    model.embed_tokens = SimpleNamespace(forward=lambda ids: embedding[ids])
    model.norm = SimpleNamespace(forward=norm)
    causal_lm = glm_dual.Glm4MoeForCausalLM.__new__(glm_dual.Glm4MoeForCausalLM)
    causal_lm.model = model
    causal_lm.lm_head = SimpleNamespace(forward=lambda x: x.float() * 2)
    runner.causal_lm = causal_lm
    runner.engine = SimpleNamespace(stream=stream)

    def serial(batch, backend):
        previous = ctx.attn_backend
        ctx.attn_backend = backend
        backend.prepare_metadata(batch)
        with ctx.forward_batch(batch):
            x, residual = embedding[batch.input_ids], None
            for layer_id, layer in enumerate(layers):
                x, residual = norm(x, residual)
                x = attn(x)
                x, residual = norm(x, residual)
                if isinstance(layer.mlp, Glm4MoeSparseMLP):
                    ids, weights = gate(x)
                    x = routed(x, ids, weights, layer_id) + shared(x)
                else:
                    x = layer.mlp.forward(x)
            x = norm(x, residual)[0]
        ctx.attn_backend = previous
        return x

    return SimpleNamespace(
        runner=runner,
        serial=serial,
        ctx=ctx,
        original_backend=original_backend,
        pending=pending,
        order=order,
    )


def test_dual_decode_matches_serial_glm_path(simulation):
    pair = make_batches((2, 3), step=2)
    expected = tuple(
        simulation.serial(batch, backend)
        for batch, backend in zip(pair, simulation.runner.attention)
    )
    actual = simulation.runner.forward(*pair)

    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)
    assert simulation.ctx.attn_backend is simulation.original_backend
    assert simulation.ctx._batch is None and not simulation.pending
    model = simulation.runner.causal_lm.model
    assert model.shared == [None, None]
    assert model.topk_ids == model.topk_weights == [None, None]
    assert simulation.runner.engine.stream.sync_count == 0
    assert simulation.order == [
        ("submit", 1),
        ("sync", 1),
        ("submit", 1),
        ("sync", 1),
        ("submit", 2),
        ("sync", 2),
        ("submit", 2),
        ("sync", 2),
    ]


def test_dual_logits_use_causal_lm_head(simulation):
    pair = make_batches((2, 2))
    expected = tuple(
        simulation.serial(batch, backend).float() * 2
        for batch, backend in zip(pair, simulation.runner.attention)
    )
    actual = simulation.runner.forward_logits(*pair)
    for value, reference in zip(actual, expected):
        torch.testing.assert_close(value, reference, rtol=0, atol=0)


def test_local_model_keeps_sequential_weight_layout(monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    monkeypatch.setattr("minisgl.layers.attention.get_rope", lambda **kwargs: None)
    config = ModelConfig.from_hf(glm_config())

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        sequential = SequentialGlm4MoeForCausalLM(config)
        dual = glm_dual.Glm4MoeForCausalLM(config)

    before = sequential.state_dict()
    after = dual.state_dict()
    assert before.keys() == after.keys()
    assert {name: (value.shape, value.dtype) for name, value in before.items()} == {
        name: (value.shape, value.dtype) for name, value in after.items()
    }
    assert isinstance(dual.model.layers.op_list[0], glm_dual.Glm4MoeDecoderLayer)
    assert isinstance(dual.model.layers.op_list[1].mlp, glm_dual.Glm4MoeSparseMLP)


def test_local_model_accepts_gguf_and_kt_weights(tmp_path, monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    monkeypatch.setattr("minisgl.layers.attention.get_rope", lambda **kwargs: None)
    config = ModelConfig.from_hf(glm_config())
    write_shards(tmp_path, config)
    weights = GGUFWeights(str(tmp_path), config)
    cache = {}
    monkeypatch.setitem(
        sys.modules,
        "kt_kernel.utils.llamafile",
        SimpleNamespace(LlamafileMoEWrapper=SimpleNamespace(_gguf_loaders_by_path=cache)),
    )
    factory = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))

    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = glm_dual.Glm4MoeForCausalLM(config)
    engine_config = EngineConfig(
        "unused", DistributedInfo(0, 1), torch.bfloat16, kt_weight_path=str(tmp_path)
    )
    engine_config.__dict__["model_config"] = config
    _adjust_config(engine_config)
    load_ktransformers_experts(model, engine_config, gguf_weights=weights)
    model.load_state_dict(dict(weights.weights(torch.device("cpu"), torch.bfloat16)))

    assert factory.call_count == 1
    assert factory.call_args.kwargs["layer_idx"] == 1
    assert model.model.layers.op_list[1].mlp._wrapper is factory.return_value


def test_load_model_builds_eager_single_rank_engine(monkeypatch):
    model_config = SimpleNamespace(is_glm4_moe=True, model_type="glm4_moe")
    captured = {}

    class Config:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model_config = model_config
            self.attention_backend = kwargs["attention_backend"]

    engine = SimpleNamespace()
    monkeypatch.setattr("minisgl.engine.EngineConfig", Config)

    def make_engine(config, *, model_factory):
        assert model_factory is glm_dual.Glm4MoeForCausalLM
        return engine

    monkeypatch.setattr("minisgl.engine.Engine", make_engine)
    expected = object()

    def make_runner(actual_engine, actual_config, backend):
        assert actual_engine is engine and actual_config is model_config and backend == "fi"
        return expected

    monkeypatch.setattr(glm_dual, "GLMDualBatchRunner", make_runner)
    actual = glm_dual.load_model(
        "glm-config",
        "glm.gguf",
        max_running_req=32,
        max_seq_len=2048,
        num_pages=1024,
    )

    assert actual is expected
    assert captured["model_path"] == "glm-config"
    assert captured["kt_weight_path"] == "glm.gguf"
    assert captured["dtype"] == torch.bfloat16
    assert captured["tp_info"].size == 1
    assert captured["cuda_graph_bs"] == [] and captured["cuda_graph_max_bs"] == 0
    assert captured["max_seq_len_override"] == 2048
    assert captured["num_page_override"] == 1024


def test_load_model_rejects_non_glm_before_engine(monkeypatch):
    model_config = SimpleNamespace(is_glm4_moe=False, model_type="qwen3_moe")

    class Config:
        def __init__(self, **kwargs):
            self.model_config = model_config

    monkeypatch.setattr("minisgl.engine.EngineConfig", Config)
    monkeypatch.setattr(
        "minisgl.engine.Engine", lambda config: pytest.fail("Engine must not be constructed")
    )
    with pytest.raises(ValueError, match="需要 GLM-4 MoE"):
        glm_dual.load_model("qwen-config", "qwen.gguf")
