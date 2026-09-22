"""Real GGUF files, CPU loading tests and a Linux/CUDA FlashInfer numerical test."""

import sys
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import gguf
import numpy as np
import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.engine.engine import Engine, _adjust_config
from minisgl.layers.marlin import MarlinLinear, pack_marlin
from minisgl.models import ModelConfig, create_model
from minisgl.models.gguf import GGUFWeights
from minisgl.moe.ktransformers import load_ktransformers_experts
from minisgl.server.args import parse_args
from minisgl.utils import torch_dtype
from transformers import Qwen3MoeConfig

Q = gguf.GGMLQuantizationType


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr("minisgl.distributed.info._TP_INFO", DistributedInfo(0, 1))
    return ModelConfig.from_hf(
        Qwen3MoeConfig(
            architectures=["Qwen3MoeForCausalLM"],
            num_hidden_layers=2,
            hidden_size=256,
            intermediate_size=512,
            moe_intermediate_size=512,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=64,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=32,
            max_position_embeddings=128,
            tie_word_embeddings=False,
        )
    )


def checkpoint_tensors(config):
    # Independent checkpoint layout, including expert tensors that must never be dequantized.
    rng = np.random.default_rng(123)
    tensors = {}

    def add(name, shape, quant=Q.Q8_0):
        values = (rng.standard_normal(shape) * 0.03).astype(np.float32)
        if "norm" in name:
            values += 1
        tensors[name] = (gguf.quantize(values, quant), quant)

    add("token_embd.weight", (32, 256))
    if not config.tie_word_embeddings:
        add("output.weight", (32, 256), Q.BF16)
    add("output_norm.weight", (256,), Q.F32)
    for layer in range(2):
        p = f"blk.{layer}"
        add(f"{p}.attn_q.weight", (256, 256))
        add(f"{p}.attn_k.weight", (128, 256), Q.Q4_0)
        add(f"{p}.attn_v.weight", (128, 256), Q.F16)
        add(f"{p}.attn_output.weight", (256, 256), Q.BF16)
        add(f"{p}.attn_q_norm.weight", (64,), Q.F32)
        add(f"{p}.attn_k_norm.weight", (64,), Q.F32)
        add(f"{p}.attn_norm.weight", (256,), Q.F32)
        add(f"{p}.ffn_norm.weight", (256,), Q.F32)
        add(f"{p}.ffn_gate_inp.weight", (4, 256), Q.F32)
        for proj in ("gate", "up", "down"):
            shape = (4, 256, 512) if proj == "down" else (4, 512, 256)
            add(f"{p}.ffn_{proj}_exps.weight", shape)
    return tensors


def write_checkpoint(path, tensors, *, arch="qwen3moe", split=None, metadata=None):
    writer = gguf.GGUFWriter(str(path), arch)
    if split is not None:
        writer.add_uint16("split.no", split[0])
        writer.add_uint16("split.count", split[1])
    for key, value in (metadata or {}).items():
        writer.add_uint32(key, value)
    for name, (data, quant) in tensors.items():
        writer.add_tensor(name, data, raw_dtype=quant)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


@pytest.fixture
def checkpoint(tmp_path, config):
    tensors = checkpoint_tensors(config)
    path = tmp_path / "model.gguf"
    write_checkpoint(path, tensors)
    return path, tensors


def reference(tensors, name, device="cpu"):
    data, quant = tensors[name]
    return torch.from_numpy(gguf.dequantize(data, quant).copy()).to(
        device=device, dtype=torch.bfloat16
    )


def engine_config(config, path, **kwargs):
    engine = EngineConfig(
        model_path="config-and-tokenizer-only",
        tp_info=DistributedInfo(0, 1),
        dtype=torch.bfloat16,
        moe_backend="kt",
        kt_weight_path=str(path),
        attention_backend="fi",
        **kwargs,
    )
    engine.__dict__["model_config"] = config
    return engine


def test_mixed_gguf_loads_complete_gpu_state(checkpoint, config, monkeypatch):
    path, tensors = checkpoint
    expected_qkv = torch.cat(
        [reference(tensors, f"blk.0.attn_{proj}.weight") for proj in ("q", "k", "v")]
    )
    reader = GGUFWeights(str(path), config)
    real_dequantize = gguf.dequantize
    chunk_rows = []
    expert_bytes = [t.data for n, t in reader.tensors.items() if "_exps" in n]

    def checked_dequantize(data, quant):
        assert not any(np.shares_memory(data, expert) for expert in expert_bytes)
        chunk_rows.append(data.shape[0])
        return real_dequantize(data, quant)

    monkeypatch.setattr(gguf, "dequantize", checked_dequantize)
    state = dict(reader.weights(torch.device("cpu"), torch.bfloat16, chunk_bytes=4 * 256 * 7))
    assert max(chunk_rows) <= 7
    assert len(state) == 4 + 2 * 9
    assert all(t.is_contiguous() for t in state.values())
    assert not any("experts" in name for name in state)
    qkv_weight, qkv_scales = pack_marlin(expected_qkv)
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_proj.weight"], qkv_weight)
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_proj.scales"], qkv_scales)
    for layer in range(2):
        for source, target in (
            ("attn_q_norm", "self_attn.q_norm"),
            ("attn_k_norm", "self_attn.k_norm"),
            ("attn_norm", "input_layernorm"),
            ("ffn_norm", "post_attention_layernorm"),
            ("ffn_gate_inp", "mlp.gate"),
        ):
            torch.testing.assert_close(
                state[f"model.layers.{layer}.{target}.weight"],
                reference(tensors, f"blk.{layer}.{source}.weight"),
            )
        expected_weight, expected_scales = pack_marlin(
            reference(tensors, f"blk.{layer}.attn_output.weight")
        )
        torch.testing.assert_close(
            state[f"model.layers.{layer}.self_attn.o_proj.weight"], expected_weight
        )
        torch.testing.assert_close(
            state[f"model.layers.{layer}.self_attn.o_proj.scales"], expected_scales
        )
    torch.testing.assert_close(
        state["model.embed_tokens.weight"], reference(tensors, "token_embd.weight")
    )
    head, head_scales = pack_marlin(reference(tensors, "output.weight"))
    torch.testing.assert_close(state["lm_head.weight"], head)
    torch.testing.assert_close(state["lm_head.scales"], head_scales)

    # Actual runtime model must accept every key/shape/dtype, not just our reference mapping.
    monkeypatch.setitem(sys.modules, "flashinfer", MagicMock())
    monkeypatch.setattr("minisgl.layers.attention.get_rope", lambda **kwargs: None)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        model = create_model(config)
    factory = MagicMock(side_effect=lambda **kwargs: MagicMock())
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=factory))
    load_ktransformers_experts(model, engine_config(config, path))
    assert all(call.kwargs["weight_path"] == str(path) for call in factory.call_args_list)
    model.load_state_dict(state.copy())
    projection = model.model.layers.op_list[0].self_attn.qkv_proj
    assert isinstance(projection, MarlinLinear) and isinstance(model.lm_head, MarlinLinear)
    assert projection.weight.dtype == torch.int32
    torch.testing.assert_close(projection.weight, qkv_weight)


@pytest.mark.parametrize("quant", [Q.Q4_K, Q.Q6_K, Q.IQ4_XS])
def test_k_quantized_gpu_weights(tmp_path, config, quant):
    tensors = checkpoint_tensors(config)
    _, block_bytes = gguf.GGML_QUANT_SIZES[quant]
    # Finite, nonzero packed K-quant blocks. These quantizers only implement dequantization.
    data = np.full((256, block_bytes), 17, dtype=np.uint8)
    scale = np.array([0.001], dtype=np.float16).view(np.uint8)
    if quant == Q.IQ4_XS:
        data[:, :2] = scale
    elif quant == Q.Q4_K:
        data[:, :2] = scale
        data[:, 2:4] = scale
    else:
        data[:, -2:] = scale
    tensors["blk.0.attn_q.weight"] = data, quant
    path = tmp_path / "k_quant.gguf"
    write_checkpoint(path, tensors)
    state = dict(GGUFWeights(str(path), config).weights(torch.device("cpu"), torch.bfloat16))
    expected = torch.cat([reference(tensors, f"blk.0.attn_{p}.weight") for p in ("q", "k", "v")])
    packed, scales = pack_marlin(expected)
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_proj.weight"], packed)
    torch.testing.assert_close(state["model.layers.0.self_attn.qkv_proj.scales"], scales)


def test_engine_loads_gguf_weights(checkpoint, config):
    path, _ = checkpoint
    engine = Engine.__new__(Engine)
    engine.device, engine.dtype = torch.device("cpu"), torch.bfloat16
    args = engine_config(config, path)
    _adjust_config(args)
    state = engine._load_weight_state_dict(GGUFWeights(args.kt_weight_path, config))
    assert "model.layers.0.self_attn.qkv_proj.weight" in state


def test_shards_and_missing_shard(tmp_path, config, monkeypatch):
    tensors = checkpoint_tensors(config)
    entries = list(tensors.items())
    for index in range(2):
        write_checkpoint(tmp_path / f"part-{index}.gguf", dict(entries[index::2]), split=(index, 2))
    state = dict(GGUFWeights(str(tmp_path), config).weights(torch.device("cpu"), torch.bfloat16))
    assert len(state) == 22
    with pytest.raises(ValueError, match="Incomplete GGUF shards"):
        GGUFWeights(str(tmp_path / "part-0.gguf"), config)

    # llama.cpp stores model metadata only in the first shard.
    readers = [gguf.GGUFReader(str(tmp_path / f"part-{i}.gguf")) for i in range(2)]
    readers[1].fields.pop("general.architecture")
    reader_iter = iter(readers)
    monkeypatch.setattr(gguf, "GGUFReader", lambda *args, **kwargs: next(reader_iter))
    GGUFWeights(str(tmp_path), config)


@pytest.mark.parametrize(
    "problem,match",
    [
        ("missing", "Missing GGUF tensor blk.0.attn_q.weight"),
        ("shape", "expected shape"),
        ("architecture", "general.architecture"),
        ("metadata", "head_count.*does not match"),
        ("duplicate", "Duplicate GGUF tensor"),
    ],
)
def test_bad_checkpoints_fail_before_dequantization(tmp_path, config, monkeypatch, problem, match):
    tensors = checkpoint_tensors(config)
    if problem == "missing":
        del tensors["blk.0.attn_q.weight"]
    if problem == "shape":
        tensors["blk.0.attn_q.weight"] = tensors["blk.0.attn_k.weight"]
    write_checkpoint(
        tmp_path / "model.gguf",
        tensors,
        arch="llama" if problem == "architecture" else "qwen3moe",
        metadata={"qwen3moe.attention.head_count": 8} if problem == "metadata" else None,
    )
    if problem == "duplicate":
        write_checkpoint(tmp_path / "other-quant.gguf", tensors)
    monkeypatch.setattr(
        gguf, "dequantize", MagicMock(side_effect=AssertionError("must not unpack"))
    )
    with pytest.raises(ValueError, match=match):
        GGUFWeights(str(tmp_path), config)


def test_tied_embeddings_need_no_output_tensor(tmp_path, config):
    config = replace(config, tie_word_embeddings=True)
    path = tmp_path / "tied.gguf"
    write_checkpoint(path, checkpoint_tensors(config))
    state = dict(GGUFWeights(str(path), config).weights(torch.device("cpu"), torch.bfloat16))
    packed, scales = pack_marlin(state["model.embed_tokens.weight"])
    torch.testing.assert_close(state["lm_head.weight"], packed)
    torch.testing.assert_close(state["lm_head.scales"], scales)
    assert "model.embed_tokens.weight" in state


def test_cli_requires_gguf_path():
    args, _ = parse_args(
        [
            "--model",
            "config-only",
            "--dtype",
            "bfloat16",
            "--moe-backend",
            "kt",
            "--kt-weight-path",
            "full.gguf",
        ]
    )
    assert args.kt_weight_path == "full.gguf"
    with pytest.raises(SystemExit):
        parse_args(["--model", "config-only", "--load-format", "safetensors"])


def test_gguf_rejects_other_moe_backend(config):
    args = replace(engine_config(config, "model.gguf"), moe_backend="fused")
    with pytest.raises(ValueError, match="Only the KT MoE backend"):
        _adjust_config(args)


def test_modelscope_gguf_does_not_download_weights(monkeypatch):
    download = MagicMock(return_value="local-config-only")
    monkeypatch.setitem(sys.modules, "modelscope", SimpleNamespace(snapshot_download=download))
    args, _ = parse_args(
        [
            "--model",
            "Qwen/config-only",
            "--model-source",
            "modelscope",
            "--dtype",
            "bfloat16",
            "--moe-backend",
            "kt",
            "--kt-weight-path",
            "model.gguf",
        ]
    )
    assert args.model_path == "local-config-only"
    ignored = download.call_args.kwargs["ignore_patterns"]
    assert all(pattern in ignored for pattern in ("*.safetensors", "*.bin", "*.gguf"))


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(), reason="requires Linux CUDA"
)
def test_gguf_flashinfer_prefill_decode(checkpoint, config, monkeypatch):
    """Real QKV/norm/RoPE/FlashInfer/O-proj; SDPA reference includes the growing KV cache."""
    from minisgl.attention.fi import FlashInferBackend
    from minisgl.kvcache import create_kvcache_pool
    from minisgl.layers import set_rope_device
    from minisgl.models.qwen3_moe import Qwen3DecoderLayer
    from test_marlin import quantized_reference

    if not torch.cuda.is_bf16_supported():
        pytest.skip("requires BF16 GPU")
    path, tensors = checkpoint
    device = torch.device("cuda")
    state = dict(GGUFWeights(str(path), config).weights(device, torch.bfloat16))
    ctx = SimpleNamespace(
        kv_cache=create_kvcache_pool(config, 32, 1, torch.bfloat16, device),
        page_table=torch.arange(32, device=device, dtype=torch.int32).view(1, -1),
    )
    monkeypatch.setattr("minisgl.core._GLOBAL_CTX", ctx)
    ctx.attn_backend = FlashInferBackend(config)
    set_rope_device(device)
    with torch.device("meta"), torch_dtype(torch.bfloat16):
        attn = Qwen3DecoderLayer(config, 0).self_attn
    prefix = "model.layers.0.self_attn."
    attn.load_state_dict(
        {k.removeprefix(prefix): v for k, v in state.items() if k.startswith(prefix)}
    )

    keys, values = [], []
    cached = 0
    for length in (5, 1, 1, 3, 1):  # prefill, decode, then extend prefill and decode
        positions = torch.arange(cached, cached + length, device=device)
        ctx.batch = SimpleNamespace(
            positions=positions,
            out_loc=positions.to(torch.int32),
            is_decode=length == 1,
            padded_reqs=[
                SimpleNamespace(
                    extend_len=length, device_len=cached + length, cached_len=cached, table_idx=0
                )
            ],
        )
        ctx.attn_backend.prepare_metadata(ctx.batch)
        x = torch.randn(length, 256, device=device, dtype=torch.bfloat16)
        q, k, v = [
            F.linear(
                x, quantized_reference(reference(tensors, f"blk.0.attn_{p}.weight", device))
            ).view(length, heads, 64)
            for p, heads in (("q", 4), ("k", 2), ("v", 2))
        ]

        def norm_rope(tensor, proj):
            norm = tensor.float() * torch.rsqrt(
                tensor.float().square().mean(-1, keepdim=True) + config.rms_norm_eps
            )
            norm = (norm * reference(tensors, f"blk.0.attn_{proj}_norm.weight", device)).to(
                torch.bfloat16
            )
            freqs = positions.float()[:, None] / config.rotary_config.base ** (
                torch.arange(0, 64, 2, device=device).float() / 64
            )
            cos, sin = (
                torch.cat([freqs.cos()] * 2, -1)[:, None],
                torch.cat([freqs.sin()] * 2, -1)[:, None],
            )
            rotated = torch.cat((-norm[..., 32:], norm[..., :32]), -1)
            return (norm.float() * cos + rotated.float() * sin).to(torch.bfloat16)

        q, k = norm_rope(q, "q"), norm_rope(k, "k")
        keys.append(k)
        values.append(v)
        full_k = torch.cat(keys).repeat_interleave(2, dim=1)
        full_v = torch.cat(values).repeat_interleave(2, dim=1)
        mask = torch.arange(cached + length, device=device)[None, :] <= positions[:, None]
        ref = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).float(),
                full_k.transpose(0, 1).float(),
                full_v.transpose(0, 1).float(),
                attn_mask=mask,
            )
            .transpose(0, 1)
            .reshape(length, -1)
            .to(torch.bfloat16)
        )
        ref = F.linear(
            ref, quantized_reference(reference(tensors, "blk.0.attn_output.weight", device))
        )
        actual = attn.forward(x)
        torch.testing.assert_close(actual, ref, atol=0.008, rtol=0.03)
        cached += length
