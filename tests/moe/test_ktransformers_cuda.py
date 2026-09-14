"""Real GGUF -> CPU KT -> CUDA test. Requires Linux, CUDA and the pinned kt-kernel."""

import sys

import pytest
import torch
import torch.nn.functional as F
from minisgl.distributed import DistributedInfo
from minisgl.engine.config import EngineConfig
from minisgl.models import ModelConfig
from minisgl.moe.ktransformers import KTransformersMoE
from transformers import Qwen3MoeConfig


@pytest.mark.skipif(
    sys.platform != "linux" or not torch.cuda.is_available(),
    reason="requires Linux with CUDA and kt-kernel",
)
def test_real_gguf_prefill_decode(tmp_path):
    import gguf
    import kt_kernel  # Missing/broken KT must fail on the lab machine.

    assert kt_kernel.KTMoEWrapper is not None
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
    experts = [KTransformersMoE(config, i) for i in range(2)]
    assert all(not layer.state_dict() for layer in experts)

    # Use a non-default stream, as MiniSGL's engine does, and change buffer sizes.
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.inference_mode():
        for tokens in (1, 17, 1):
            for layer, (gate, up, down) in zip(experts, references):
                x = torch.randn(tokens, 256, device="cuda", dtype=torch.bfloat16)
                logits = torch.randn(tokens, 4, device="cuda", dtype=torch.bfloat16)
                scores, ids = logits.float().softmax(-1).topk(2, dim=-1)
                scores /= scores.sum(-1, keepdim=True)
                expected = torch.zeros_like(x, dtype=torch.float32)
                for expert in range(4):
                    expert_out = (
                        F.silu(x.float() @ gate[expert].T) * (x.float() @ up[expert].T)
                    ) @ down[expert].T
                    coefficient = (scores * (ids == expert)).sum(-1, keepdim=True)
                    expected += coefficient * expert_out
                actual = layer.forward(x, logits)
                assert actual.device == x.device and actual.dtype == x.dtype
                stream.synchronize()
                assert torch.isfinite(actual).all()
                relative_error = (actual.float() - expected).norm() / expected.norm()
                assert relative_error.item() < 0.025
