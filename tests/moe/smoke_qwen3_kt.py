"""Run on the lab machine with real Qwen3 weights; no mocks or dummy weights."""

import argparse

import torch
from minisgl.core import SamplingParams
from minisgl.llm import LLM


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kt-weight-path", required=True)
    parser.add_argument("--kt-cpuinfer", type=int, default=128)
    parser.add_argument("--kt-threadpool-count", type=int, default=2)
    args = parser.parse_args()
    with torch.inference_mode():
        llm = LLM(
            args.model,
            moe_backend="ktransformers",
            kt_weight_path=args.kt_weight_path,
            kt_cpuinfer=args.kt_cpuinfer,
            kt_threadpool_count=args.kt_threadpool_count,
            attention_backend="fi",
            max_extend_tokens=128,
            max_running_req=2,
            max_seq_len_override=2048,
            num_page_override=2048,
            page_size=1,
            cuda_graph_max_bs=0,
        )
        try:
            weights = llm.engine.model.state_dict()
            assert weights and all(t.is_cuda for t in weights.values())
            assert not any(".experts." in name for name in weights)
            for layer in llm.engine.model.model.layers.op_list:
                assert layer.self_attn.attn.rotary._cos_sin_cache.is_cuda
            for wrapper in llm.engine.moe_backend.wrappers:
                assert wrapper.gpu_experts_mask.is_cpu
                assert not wrapper.gpu_experts_mask.any()
            prompts = ["The capital of France is", "Hello. " * 160 + "Count: one, two,"]
            params = SamplingParams(temperature=0, max_tokens=16, ignore_eos=True)
            # Reuse the engine for two batches to exercise KV cache and KT buffer reuse.
            for _ in range(2):
                results = llm.generate(prompts, params)
                assert len(results) == 2
                for result in results:
                    assert len(result["token_ids"]) == 16
                    assert all(0 <= t < llm.tokenizer.vocab_size for t in result["token_ids"])
                    print(result)
            print("PASS: GPU dense weights/RoPE, CPU experts, chunked prefill and decode")
        finally:
            llm.shutdown()


if __name__ == "__main__":
    main()
