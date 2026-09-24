"""验证单流时间槽、KT 暂存区复用，以及真实 KT 和 MLA 的数值一致性。"""

import sys
from types import SimpleNamespace

import pytest
import torch
from minisgl.core import Batch, Context, Req
from minisgl.models.deepseek_v2 import DeepseekMoE

from DeepseekV2DualBatchRunner import DeepseekV2DualBatchRunner


class Stream:
    def __init__(self):
        self.time, self.sync_count = 0, 0
        self.cuda_stream = id(self)

    def synchronize(self):
        self.sync_count += 1


def batches(sizes, device="cpu", step=0):
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
    """模拟一条 GPU 流、异步 CPU 任务及 KT 按层号复用的输出暂存区。"""
    runner = DeepseekV2DualBatchRunner.__new__(DeepseekV2DualBatchRunner)
    runner.device, runner.max_tokens = torch.device("cpu"), 64
    runner.stream, runner.batch_sizes = Stream(), None
    trace, buffers, shared_outputs = [], {}, []
    pending = []

    def stage(name, duration):
        stream = runner.stream
        trace.append((name, stream.time, stream.time + duration))
        stream.time += duration

    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: runner.stream)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    add = torch.Tensor.__add__

    def add_(x, y):
        if any(y is value for value in shared_outputs):
            stage("combine", 1)
        return add(x, y)

    monkeypatch.setattr(torch.Tensor, "__add__", add_)
    ctx = Context(1)
    original_backend = object()
    ctx.attn_backend = original_backend
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", ctx)

    class Attention:
        def prepare_metadata(self, batch):
            batch.attn_metadata = SimpleNamespace(wrapper=self, initialized=False)

        def _initialize(self, metadata):
            metadata.initialized = True

    runner.attention = (Attention(), Attention())

    def norm(x, residual):
        stage("norm", 2)
        residual = x if residual is None else x + residual
        return residual * 0.625, residual

    def attn(x):
        stage("attention", 3)
        assert ctx.batch.attn_metadata.wrapper is ctx.attn_backend
        assert ctx.batch.attn_metadata.initialized
        return x * 0.75 + ctx.batch.positions[:, None].to(x.dtype) * 0.01

    def gate(x):
        stage("router", 3)
        scores = x[:, :4].float().softmax(-1)
        weights, ids = scores.topk(2, sorted=False)
        return ids, weights * 16

    def shared(x):
        stage("shared", 5)
        value = x * 0.25
        shared_outputs.append(value)
        return value

    def routed(x, ids, weights, layer):
        coefficient = ((ids.float() + 1) * weights).sum(-1, keepdim=True)
        return (x.float() * coefficient * (0.01 * layer)).bfloat16()

    class KT:
        def __init__(self, layer):
            self.layer = layer

        def submit_forward(self, x, ids, weights, stream):
            assert stream == runner.stream.cuda_stream and not pending
            stage("d2h", 6)
            start = runner.stream.time
            trace.append(("cpu", start, start + 20))
            pending.append((self.layer, x, routed(x, ids, weights, self.layer), start + 20))
            # 同层两份 microbatch 共用这个槽，提前覆盖以暴露未 clone 的问题。
            key = (x.shape[0], self.layer % 2)
            if key not in buffers:
                buffers[key] = torch.empty_like(x)
            buffers[key].fill_(float("nan"))

        def sync_forward(self, x, stream):
            assert stream == runner.stream.cuda_stream and len(pending) == 1
            layer, original_x, output, done = pending.pop()
            assert layer == self.layer and original_x is x
            runner.stream.time = max(runner.stream.time, done)
            stage("h2d", 3)
            buffer = buffers[(x.shape[0], self.layer % 2)]
            buffer.copy_(output)
            return buffer

    layers = []
    for i in range(4):
        if i == 0:
            mlp = SimpleNamespace(forward=lambda x: x * 0.375)
        else:
            mlp = DeepseekMoE.__new__(DeepseekMoE)
            mlp.gate, mlp.shared_experts = (
                SimpleNamespace(forward=gate),
                SimpleNamespace(forward=shared),
            )
            mlp._wrapper = KT(i)
        layers.append(
            SimpleNamespace(
                input_layernorm=SimpleNamespace(forward=norm),
                post_attention_layernorm=SimpleNamespace(forward=norm),
                self_attn=SimpleNamespace(forward=attn),
                mlp=mlp,
            )
        )
    embedding = torch.arange(64 * 8).view(64, 8).bfloat16() / 512
    runner.layers = layers
    runner.layer_num = len(layers)
    runner.cpu_moe = [None] + [layer.mlp._wrapper for layer in layers[1:]]
    runner.model = SimpleNamespace(
        embed_tokens=SimpleNamespace(forward=lambda ids: embedding[ids]),
        norm=SimpleNamespace(forward=norm),
    )

    def serial(batch, backend):
        ctx.attn_backend = backend
        backend.prepare_metadata(batch)
        backend._initialize(batch.attn_metadata)
        with ctx.forward_batch(batch):
            x, residual = embedding[batch.input_ids], None
            for i, layer in enumerate(layers):
                x, residual = norm(x, residual)
                x = attn(x)
                x, residual = norm(x, residual)
                if i == 0:
                    x = layer.mlp.forward(x)
                else:
                    ids, weights = gate(x)
                    x = shared(x) + routed(x, ids, weights, i)
            x = norm(x, residual)[0]
        ctx.attn_backend = original_backend
        return x

    return SimpleNamespace(
        runner=runner,
        trace=trace,
        serial=serial,
        ctx=ctx,
        original_backend=original_backend,
        buffers=buffers,
        pending=pending,
    )


@pytest.mark.parametrize("sizes", [(1, 1), (2, 3), (17, 17)])
def test_dual_decode_matches_serial_and_reuses_buffers(simulation, sizes):
    s, buffer_ids, previous = simulation, None, None
    for step in range(3):
        pair = batches(sizes, step=step)
        expected = tuple(s.serial(b, a) for b, a in zip(pair, s.runner.attention))
        actual = s.runner.forward(*pair)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        if previous is not None:
            for a, b in zip(*previous):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        previous = actual, tuple(x.clone() for x in actual)
        assert s.ctx.attn_backend is s.original_backend and s.ctx._batch is None
        assert not s.pending
        assert s.runner.topk_ids == s.runner.topk_weights == s.runner.shared == [None, None]
        pointers = tuple(b.data_ptr() for b in s.buffers.values())
        assert buffer_ids is None or buffer_ids == pointers
        buffer_ids = pointers
    assert s.runner.stream.sync_count == 3 * (2 * s.runner.layer_num + 1)


def test_gpu_stages_overlap_cpu_and_transfers_finish_in_slot(simulation):
    s = simulation
    s.runner.forward(*batches((2, 2)))
    trace = s.trace

    def spans(name):
        return [(start, end) for stage, start, end in trace if stage == name]

    def overlaps(a, b):
        return max(a[0], b[0]) < min(a[1], b[1])

    cpu = spans("cpu")
    assert len(cpu) == 6 and all(a[1] <= b[0] for a, b in zip(cpu, cpu[1:]))
    for name in ("norm", "attention", "router", "shared", "combine"):
        assert any(overlaps(a, b) for a in spans(name) for b in cpu), name
    for i, (send, work, receive) in enumerate(zip(spans("d2h"), cpu, spans("h2d"))):
        assert send[1] <= work[0] and work[1] <= receive[0]
        if i + 1 < len(cpu):
            assert receive[1] <= cpu[i + 1][0]
    assert s.runner.stream.sync_count == 2 * s.runner.layer_num + 1


@pytest.mark.parametrize("problem", ["prefill", "duplicate", "padding", "size_change", "capture"])
def test_invalid_batches_fail_before_launch(simulation, monkeypatch, problem):
    runner = simulation.runner
    pair = batches((1, 1))
    if problem == "prefill":
        pair[0].phase = "prefill"
    elif problem == "duplicate":
        pair[1].reqs[0].table_idx = pair[0].reqs[0].table_idx
    elif problem == "padding":
        pair[0].padded_reqs = pair[0].reqs * 2
    elif problem == "size_change":
        runner.forward(*pair)
        pair = batches((2, 1))
    else:
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    before = len(simulation.trace)
    with pytest.raises(ValueError):
        runner.forward(*pair)
    assert len(simulation.trace) == before


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(), reason="requires Linux CUDA and KT"
)
def test_real_mla_shared_and_kt_dual_decode(tmp_path, monkeypatch):
    from minisgl.attention.fi_mla import FlashInferMLABackend
    from minisgl.distributed import DistributedInfo
    from minisgl.engine.config import EngineConfig
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.layers import set_rope_device
    from minisgl.layers.rotary import get_rope
    from minisgl.models import ModelConfig, create_model
    from minisgl.models.gguf import GGUFWeights
    from minisgl.moe.ktransformers import load_ktransformers_experts
    from minisgl.utils import torch_dtype
    from test_deepseek_v2 import hf_config, write_v2

    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    c = ModelConfig.from_hf(
        hf_config(
            num_hidden_layers=4, hidden_size=256, num_attention_heads=16, num_key_value_heads=16
        )
    )
    device = torch.device("cuda", torch.cuda.current_device())
    ctx = Context(1)
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", ctx)
    ctx.page_table = torch.randperm(32, device=device).int().reshape(4, 8)
    serial_cache = create_kvcache_pool(c, 32, 1, torch.bfloat16, device)
    dual_cache = create_kvcache_pool(c, 32, 1, torch.bfloat16, device)
    ctx.kv_cache = serial_cache
    ctx.attn_backend = serial_attention = FlashInferMLABackend(c)
    get_rope.cache_clear()
    set_rope_device(device)
    path = tmp_path / "dual.gguf"
    write_v2(path, c)
    engine = EngineConfig(
        "unused",
        DistributedInfo(0, 1),
        torch.bfloat16,
        kt_weight_path=str(path),
        kt_cpuinfer=2,
        kt_threadpool_count=1,
    )
    engine.__dict__["model_config"] = c
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(c)
    weights = GGUFWeights(str(path), c)
    load_ktransformers_experts(model, engine, gguf_weights=weights)
    model.load_state_dict(dict(weights.weights(device, torch.bfloat16)))
    stream = torch.cuda.Stream(device=device)
    stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(stream), torch.inference_mode():
        ctx.kv_cache = dual_cache
        runner = DeepseekV2DualBatchRunner(model.model)
        for step in range(3):
            pair = batches((2, 2), device=device, step=step)
            for batch in pair:
                rows = [r.table_idx for r in batch.reqs]
                batch.out_loc = ctx.page_table[rows, step]
            expected = []
            ctx.kv_cache, ctx.attn_backend = serial_cache, serial_attention
            for batch in pair:
                serial_attention.prepare_metadata(batch)
                with ctx.forward_batch(batch):
                    expected.append(model.model.forward(batch.input_ids).clone())
            # 先完成普通串行路径的 KT 回调，再由 runner 独占 CPUInfer 队列。
            stream.synchronize()
            ctx.kv_cache = dual_cache
            actual = runner.forward(*pair)
            stream.synchronize()
            for a, b in zip(actual, expected):
                torch.testing.assert_close(a, b, atol=0.02, rtol=0.03)
            used = ctx.page_table[:, : step + 1].flatten().long()
            torch.testing.assert_close(
                dual_cache._buffer[:, used], serial_cache._buffer[:, used], atol=0.02, rtol=0.03
            )
