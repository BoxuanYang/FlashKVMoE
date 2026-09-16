"""Real GGUF -> CPU KT -> CUDA test. Requires Linux, CUDA and the pinned kt-kernel."""

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import minisgl.core as core
import pytest
import torch
import torch.nn.functional as F
from minisgl.core import Batch, Context, Req, SamplingParams
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.graph import GraphRunner
from minisgl.layers import LinearReplicated
from minisgl.models import ModelConfig
from minisgl.moe.ktransformers import load_ktransformers_experts
from transformers import Qwen3MoeConfig

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="requires Linux with CUDA and kt-kernel",
)


@pytest.fixture
def kt_experts(tmp_path):
    import gguf
    import kt_kernel  # Missing/broken KT must fail on the lab machine.

    assert kt_kernel.KTMoEWrapper is not None
    kt_kernel.KTMoEWrapper.clear_buffer_cache()
    torch.manual_seed(42)
    path = tmp_path / "experts.gguf"
    writer = gguf.GGUFWriter(str(path), "qwen3moe")
    references = []
    for layer_id in range(2):
        weights = []
        for proj in ("gate", "up", "down"):
            weight = (torch.randn(4, 256, 256) * 0.02).numpy()
            quantized = gguf.quantize(weight, gguf.GGMLQuantizationType.Q8_0)
            writer.add_tensor(
                f"blk.{layer_id}.ffn_{proj}_exps.weight",
                quantized,
                raw_dtype=gguf.GGMLQuantizationType.Q8_0,
            )
            weights.append(
                torch.from_numpy(gguf.dequantize(quantized, gguf.GGMLQuantizationType.Q8_0)).cuda()
            )
        references.append(weights)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    config = EngineConfig(
        model_path="unused",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_backend="kt",
        kt_weight_path=str(path),
        kt_cpuinfer=2,
        kt_threadpool_count=1,
        max_running_req=17,
        max_seq_len_override=17,
    )
    config.__dict__["model_config"] = ModelConfig.from_hf(
        Qwen3MoeConfig(
            architectures=["Qwen3MoeForCausalLM"],
            hidden_size=256,
            moe_intermediate_size=256,
            num_experts=4,
            num_experts_per_tok=2,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
        )
    )
    layers = []
    for _ in range(2):
        with torch.device("cuda"):
            gate = LinearReplicated(256, 4, has_bias=False)
        gate.weight = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16) * 0.02
        layers.append(SimpleNamespace(mlp=SimpleNamespace(gate=gate)))
    model = SimpleNamespace(model=SimpleNamespace(layers=SimpleNamespace(op_list=layers)))
    load_ktransformers_experts(model, config, [1, 2, 4])
    experts = [layer.mlp for layer in layers]
    assert all(set(layer.state_dict()) == {"gate.weight"} for layer in experts)
    yield experts, references
    torch.cuda.synchronize()
    kt_kernel.KTMoEWrapper.clear_buffer_cache()
    kt_kernel.KTMoEWrapper.set_capture_batch_sizes([])


def test_real_gguf_prefill_decode(kt_experts):
    experts, references = kt_experts

    # Use a non-default stream, as MiniSGL's engine does, and change buffer sizes.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        for tokens in (1, 17, 1):
            for layer, (gate, up, down) in zip(experts, references):
                x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
                logits = F.linear(x, layer.gate.weight)
                scores, ids = logits.float().softmax(-1).topk(2, dim=-1)
                scores /= scores.sum(-1, keepdim=True)
                expected = torch.zeros_like(x, dtype=torch.float32)
                for expert in range(4):
                    expert_out = (
                        F.silu(x.float() @ gate[expert].T) * (x.float() @ up[expert].T)
                    ) @ down[expert].T
                    coefficient = (scores * (ids == expert)).sum(-1, keepdim=True)
                    expected += coefficient * expert_out
                actual = layer.forward(x)
                assert actual.device == x.device and actual.dtype == x.dtype
                stream.synchronize()
                assert torch.isfinite(actual).all()
                relative_error = (actual.float() - expected).norm() / expected.norm()
                assert relative_error.item() < 0.025


def test_real_gguf_graph_replay_after_prefill(kt_experts, monkeypatch):
    """Exercise GraphRunner with real CPU callbacks, padding and changing inputs."""
    from kt_kernel.experts_base import KExpertsCPUBuffer

    experts, _ = kt_experts
    ctx = Context(page_size=1)
    monkeypatch.setattr(core, "_GLOBAL_CTX", ctx)
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    embedding = torch.randn(128, 256, device="cuda", dtype=torch.bfloat16)
    dummy_req = Req(
        input_ids=torch.tensor([0], dtype=torch.int32),
        table_idx=0,
        cached_len=0,
        output_len=1,
        uid=-1,
        sampling_params=SamplingParams(),
        cache_handle=None,
    )

    def forward():
        x = embedding[ctx.batch.input_ids.long()]
        for layer in experts:
            x = x + layer.forward(x)
        return x

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), torch.inference_mode():
        runner = GraphRunner(
            stream=stream,
            device=embedding.device,
            model=SimpleNamespace(forward=forward),
            attn_backend=MagicMock(),
            cuda_graph_bs=[1, 2, 4],
            cuda_graph_max_bs=4,
            free_memory=0,
            max_seq_len=32,
            vocab_size=256,
            dummy_req=dummy_req,
        )
        try:
            assert set(runner.graph_map) == {1, 2, 4}
            capture_buffers = dict(KExpertsCPUBuffer.capture_buffers)
            assert set(capture_buffers) == {1, 2, 4}
            for iteration, tokens in enumerate((1, 2, 3, 4, 17, 1, 3, 2, 17, 4, 1)):
                batch = Batch(
                    reqs=[dummy_req] * tokens, phase="prefill" if tokens == 17 else "decode"
                )
                runner.pad_batch(batch)
                batch.input_ids = (
                    torch.arange(batch.padded_size, device="cuda", dtype=torch.int32)
                    + iteration * 7
                )
                batch.positions = torch.zeros_like(batch.input_ids)
                batch.out_loc = torch.zeros_like(batch.input_ids)
                with ctx.forward_batch(batch):
                    expected = forward().clone()
                if tokens == 17:
                    assert not runner.can_use_cuda_graph(batch)
                    stream.synchronize()
                    continue
                # Eager populated the same buffers with the expected answer.
                # Poison both copies so missing replay callbacks/copies cannot
                # pass by returning that stale answer.
                stream.synchronize()
                buffers = capture_buffers[batch.padded_size]
                for output in (*buffers[4], *buffers[6]):
                    output.fill_(float("nan"))
                # Replay must produce fresh results without invoking Python forward.
                actual = runner.replay(batch).clone()
                stream.synchronize()
                torch.testing.assert_close(actual, expected[:tokens], rtol=1e-3, atol=1e-3)
                assert all(
                    KExpertsCPUBuffer.capture_buffers[bs] is buffers
                    for bs, buffers in capture_buffers.items()
                )
        finally:
            stream.synchronize()
            runner.destroy_cuda_graphs()
