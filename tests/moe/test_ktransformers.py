"""CPU tests for weight separation and the standalone KT integration contract."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from minisgl.core import Context, set_global_ctx
from minisgl.distributed import DistributedInfo, set_tp_info
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import _adjust_config
from minisgl.models import create_model
from minisgl.models.weight import load_weight
from minisgl.moe import create_moe_backend
from minisgl.moe.ktransformers import validate_config
from minisgl.utils import torch_dtype
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", None)
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", None)
    hf = Qwen3MoeConfig(
        num_hidden_layers=2,
        hidden_size=128,
        intermediate_size=256,
        moe_intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        vocab_size=32,
        max_position_embeddings=128,
        architectures=["Qwen3MoeForCausalLM"],
    )
    hf.save_pretrained(tmp_path)
    set_tp_info(0, 1)
    return EngineConfig(
        model_path=str(tmp_path),
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_backend="ktransformers",
        kt_weight_path=str(tmp_path),
        max_running_req=256,
    )


def test_filter_experts_before_reading_tensor(config, monkeypatch):
    import minisgl.models.weight as loader

    dense = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
        "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 128),
        "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 128),
        "model.layers.0.mlp.gate.weight": torch.randn(4, 128),
        "model.layers.0.input_layernorm.weight": torch.randn(128),
    }
    # Both individual and packed expert formats must be skipped before any tensor read.
    tensors = dense | {
        "model.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(256, 128),
        "model.layers.0.mlp.experts.gate_up_proj": torch.randn(4, 512, 128),
    }
    save_file(tensors, config.model_path + "/model.safetensors")
    safe_open = loader.safetensors.safe_open
    read_names = []

    class GuardedReader:
        def __init__(self, *args, **kwargs):
            self.reader = safe_open(*args, **kwargs)

        def __enter__(self):
            self.reader.__enter__()
            return self

        def __exit__(self, *args):
            return self.reader.__exit__(*args)

        def keys(self):
            return self.reader.keys()

        def get_tensor(self, name):
            assert ".experts." not in name
            read_names.append(name)
            return self.reader.get_tensor(name)

    monkeypatch.setattr(loader.safetensors, "safe_open", GuardedReader)
    loaded = dict(load_weight(config.model_path, torch.device("cpu"), skip_experts=True))
    assert set(read_names) == set(dense)
    torch.testing.assert_close(
        loaded["model.layers.0.self_attn.qkv_proj.weight"],
        torch.cat([dense[f"model.layers.0.self_attn.{p}_proj.weight"] for p in ("q", "k", "v")]),
    )
    assert "model.layers.0.mlp.gate.weight" in loaded
    assert "model.layers.0.input_layernorm.weight" in loaded


@pytest.mark.parametrize("cpu_experts", [False, True])
@pytest.mark.parametrize(
    "layers,hidden,heads,intermediate", [(48, 2048, 32, 768), (94, 4096, 64, 1536)]
)
def test_qwen_model_weight_placement(
    config, monkeypatch, cpu_experts, layers, hidden, heads, intermediate
):
    # No GPU kernels execute here; inspect real model construction at both target sizes on meta.
    monkeypatch.setitem(
        sys.modules, "flashinfer", SimpleNamespace(rmsnorm=Mock(), fused_add_rmsnorm=Mock())
    )
    monkeypatch.setattr("minisgl.layers.attention.get_rope", Mock())
    ctx = Context(1)
    ctx.moe_backend = SimpleNamespace(cpu_experts=cpu_experts)
    set_global_ctx(ctx)
    model_config = replace(
        config.model_config,
        num_layers=layers,
        hidden_size=hidden,
        num_qo_heads=heads,
        head_dim=128,
        num_kv_heads=4,
        moe_intermediate_size=intermediate,
        num_experts=128,
        num_experts_per_tok=8,
    )
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(model_config)
    weights = model.state_dict()
    expert_keys = [k for k in weights if ".experts." in k]
    assert len(expert_keys) == (0 if cpu_experts else 2 * layers)
    for i, layer in enumerate(model.model.layers.op_list):
        assert layer.mlp.experts._layer_id == i
        for suffix in (
            "self_attn.qkv_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "mlp.gate.weight",
        ):
            assert f"model.layers.{i}.{suffix}" in weights


def test_standalone_wrapper_initialization(config, monkeypatch):
    factory = Mock(side_effect=lambda **kwargs: Mock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    backend = create_moe_backend("ktransformers", config)
    assert backend.cpu_experts
    assert len(backend.wrappers) == 2
    for i, call in enumerate(factory.call_args_list):
        assert call.kwargs["layer_idx"] == i
        assert call.kwargs["gpu_experts_mask"] is None
        assert call.kwargs["max_deferred_experts_per_token"] == 0
        assert call.kwargs["chunked_prefill_size"] == 256  # decode > prefill
        backend.wrappers[i].load_weights.assert_called_once_with()


@pytest.mark.parametrize(
    "changes",
    [
        {"tp_info": DistributedInfo(0, 2)},
        {"dtype": torch.float16},
        {"kt_weight_path": None},
        {"kt_cpuinfer": 0},
        {"kt_threadpool_count": 0},
        {"kt_method": "BF16"},
        {"use_dummy_weight": True},
        {"max_running_req": 0},
    ],
)
def test_invalid_config(config, changes):
    with pytest.raises(ValueError):
        validate_config(replace(config, **changes))


def test_kt_disables_all_graph_capture(config):
    config = replace(config, cuda_graph_bs=[1, 4], cuda_graph_max_bs=4, attention_backend="fi")
    _adjust_config(config)
    assert config.cuda_graph_bs == []
    assert config.cuda_graph_max_bs == 0


def test_fused_backend_remains_default(config):
    config = replace(config, moe_backend="auto", attention_backend="fi")
    _adjust_config(config)
    assert config.moe_backend == "fused"
    assert not create_moe_backend("fused", config).cpu_experts


def test_forward_uses_layer_and_current_stream(config, monkeypatch):
    factory = Mock(side_effect=lambda **kwargs: Mock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    backend = create_moe_backend("ktransformers", config)
    hidden = SimpleNamespace(is_cuda=True, dtype=torch.bfloat16, shape=(2, 128), device="cuda:0")
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int32)
    weights = torch.tensor([[0.7, 0.3], [0.6, 0.4]])
    monkeypatch.setattr("minisgl.moe.ktransformers.fused_topk", Mock(return_value=(weights, ids)))
    monkeypatch.setattr(
        torch.cuda, "current_stream", Mock(return_value=SimpleNamespace(cuda_stream=123))
    )
    output = backend.forward(hidden, None, None, Mock(), 2, True, layer_id=1)
    backend.wrappers[0].forward.assert_not_called()
    backend.wrappers[1].forward.assert_called_once_with(hidden, ids, weights, 123)
    assert output is backend.wrappers[1].forward.return_value
    hidden.shape = (257, 128)
    with pytest.raises(ValueError, match="buffer size"):
        backend.forward(hidden, None, None, Mock(), 2, True)


def test_launch_arguments(config):
    from minisgl.server.args import parse_args

    args, shell = parse_args(
        [
            "--model",
            config.model_path,
            "--dtype",
            "bfloat16",
            "--moe-backend",
            "ktransformers",
            "--kt-weight-path",
            config.kt_weight_path,
            "--kt-cpuinfer",
            "128",
            "--kt-threadpool-count",
            "2",
            "--kt-method",
            "LLAMAFILE",
            "--attention-backend",
            "fi",
            "--cuda-graph-max-bs",
            "0",
            "--page-size",
            "1",
            "--num-pages",
            "2048",
            "--max-prefill-length",
            "2048",
            "--max-seq-len-override",
            "2048",
            "--host",
            "0.0.0.0",
            "--port",
            "30000",
        ]
    )
    assert not shell
    assert args.moe_backend == "ktransformers"
    assert args.kt_weight_path == config.kt_weight_path
    assert args.num_page_override * args.page_size == 2048
    assert args.max_forward_len == 2048


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires Linux CUDA and compiled kt-kernel"
)
def test_real_kt_gguf_prefill_and_decode(config):
    """Exercise actual GGUF loading, CPU kernels and CUDA stream copies against a reference."""
    import numpy as np
    from gguf import GGMLQuantizationType, GGUFWriter, dequantize, quantize
    from minisgl.moe.ktransformers import KTransformersMoe

    hf = config.hf_config
    hf.hidden_size = 256
    hf.save_pretrained(config.model_path)
    # Use a fresh config because model_config/hf_config are cached properties.
    config = replace(config, kt_cpuinfer=2, kt_threadpool_count=1, max_running_req=128)
    config.__dict__["hf_config"] = hf
    rng = np.random.default_rng(42)
    writer = GGUFWriter(config.model_path + "/experts.gguf", "qwen3moe")
    reference = []
    for layer in range(2):
        projections = []
        for projection in ("gate", "up", "down"):
            raw = (rng.standard_normal((4, 256, 256)) * 0.02).astype(np.float32)
            packed = quantize(raw, GGMLQuantizationType.Q8_0)
            writer.add_tensor(
                f"blk.{layer}.ffn_{projection}_exps.weight",
                packed,
                raw_dtype=GGMLQuantizationType.Q8_0,
            )
            projections.append(torch.from_numpy(dequantize(packed, GGMLQuantizationType.Q8_0)))
        reference.append(projections)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    backend = KTransformersMoe(config)
    stream = torch.cuda.Stream()
    with torch.inference_mode(), torch.cuda.stream(stream):
        for tokens in (1, 32, 2, 16, 1):
            for layer in range(2):
                x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
                logits = torch.randn(tokens, 4, device="cuda")
                output = backend.forward(x, None, None, logits, 2, True, layer_id=layer)
                stream.synchronize()
                probs = logits.cpu().softmax(-1)
                weights, ids = probs.topk(2, dim=-1)
                weights /= weights.sum(-1, keepdim=True)
                expected = torch.zeros(tokens, 256)
                gate, up, down = reference[layer]
                x_cpu = x.cpu().float()
                for token in range(tokens):
                    for slot in range(2):
                        expert = ids[token, slot]
                        activation = torch.nn.functional.silu(gate[expert] @ x_cpu[token])
                        value = down[expert] @ (activation * (up[expert] @ x_cpu[token]))
                        expected[token] += weights[token, slot] * value
                assert output.is_cuda and output.dtype == torch.bfloat16
                torch.testing.assert_close(output.cpu().float(), expected, atol=0.01, rtol=0.05)
