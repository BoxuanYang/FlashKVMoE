"""Qwen3 MoE / DeepSeek V3 GGUF: packed CPU experts and GPU projection weights."""

from __future__ import annotations

from typing import Iterator

import gguf
import numpy as np
import torch
from minisgl.layers.marlin import pack_marlin
from tqdm import tqdm

from .config import ModelConfig
from .gguf_parts import open_gguf_readers


def _gpu_tensor_specs(config: ModelConfig) -> Iterator[tuple[str, str, tuple[int, ...]]]:
    """GGUF name, runtime name, PyTorch shape (GGUF dimensions are reversed)."""
    hidden = config.hidden_size
    yield "token_embd.weight", "model.embed_tokens.weight", (config.vocab_size, hidden)
    yield "output_norm.weight", "model.norm.weight", (hidden,)
    head = "token_embd.weight" if config.tie_word_embeddings else "output.weight"
    yield head, "lm_head.weight", (config.vocab_size, hidden)
    if config.is_mla:
        yield from _deepseek_tensor_specs(config)
        return
    for layer in range(config.num_layers):
        src, dst = f"blk.{layer}", f"model.layers.{layer}"
        for proj, heads in (
            ("q", config.num_qo_heads),
            ("k", config.num_kv_heads),
            ("v", config.num_kv_heads),
        ):
            yield (
                f"{src}.attn_{proj}.weight",
                f"{dst}.self_attn.{proj}_proj.weight",
                (heads * config.head_dim, hidden),
            )
        yield (
            f"{src}.attn_output.weight",
            f"{dst}.self_attn.o_proj.weight",
            (hidden, config.num_qo_heads * config.head_dim),
        )
        for proj in ("q", "k"):
            yield (
                f"{src}.attn_{proj}_norm.weight",
                f"{dst}.self_attn.{proj}_norm.weight",
                (config.head_dim,),
            )
        yield f"{src}.attn_norm.weight", f"{dst}.input_layernorm.weight", (hidden,)
        yield f"{src}.ffn_norm.weight", f"{dst}.post_attention_layernorm.weight", (hidden,)
        yield f"{src}.ffn_gate_inp.weight", f"{dst}.mlp.gate.weight", (config.num_experts, hidden)


def _deepseek_tensor_specs(c: ModelConfig) -> Iterator[tuple[str, str, tuple[int, ...]]]:
    for layer in range(c.num_layers):
        src, dst = f"blk.{layer}", f"model.layers.{layer}"
        for name, target, shape in (
            ("attn_q_a", "q_a_proj", (c.q_lora_rank, c.hidden_size)),
            ("attn_q_a_norm", "q_a_layernorm", (c.q_lora_rank,)),
            ("attn_q_b", "q_b_proj", (c.num_qo_heads * c.head_dim, c.q_lora_rank)),
            (
                "attn_kv_a_mqa",
                "kv_a_proj_with_mqa",
                (c.kv_lora_rank + c.qk_rope_head_dim, c.hidden_size),
            ),
            ("attn_kv_a_norm", "kv_a_layernorm", (c.kv_lora_rank,)),
            ("attn_k_b", "k_b_proj", (c.num_qo_heads, c.kv_lora_rank, c.qk_nope_head_dim)),
            ("attn_v_b", "v_b_proj", (c.num_qo_heads, c.v_head_dim, c.kv_lora_rank)),
            ("attn_output", "o_proj", (c.hidden_size, c.num_qo_heads * c.v_head_dim)),
        ):
            yield f"{src}.{name}.weight", f"{dst}.self_attn.{target}.weight", shape
        yield f"{src}.attn_norm.weight", f"{dst}.input_layernorm.weight", (c.hidden_size,)
        yield f"{src}.ffn_norm.weight", f"{dst}.post_attention_layernorm.weight", (c.hidden_size,)
        dense = layer < c.first_k_dense_replace
        if not dense:
            yield f"{src}.ffn_gate_inp.weight", f"{dst}.mlp.gate.weight", (
                c.num_experts,
                c.hidden_size,
            )
            yield f"{src}.exp_probs_b.bias", f"{dst}.mlp.gate.e_score_correction_bias", (
                c.num_experts,
            )
        intermediate = (
            c.intermediate_size if dense else c.n_shared_experts * c.moe_intermediate_size
        )
        suffix, mlp = ("", f"{dst}.mlp") if dense else ("_shexp", f"{dst}.mlp.shared_experts")
        for proj in ("gate", "up", "down"):
            shape = (
                (c.hidden_size, intermediate) if proj == "down" else (intermediate, c.hidden_size)
            )
            yield f"{src}.ffn_{proj}{suffix}.weight", f"{mlp}.{proj}_proj.weight", shape


class GGUFWeights:
    """Index and validate a complete checkpoint without unpacking any expert tensors.

    Qwen3 MoE's llama.cpp converter preserves HF Q/K row order. Unlike Llama,
    these tensors must NOT be permuted before MiniSGL's NeoX-style RoPE.
    """

    def __init__(self, path: str, config: ModelConfig):
        if config.architectures not in (["Qwen3MoeForCausalLM"], ["DeepseekV3ForCausalLM"]):
            raise ValueError("GGUF GPU weight loading supports Qwen3 MoE and DeepSeek V3 only")
        architecture_name = "deepseek2" if config.is_mla else "qwen3moe"
        # Readers mmap the packed data. Keep them alive until weight iteration completes.
        self.readers = open_gguf_readers(path)
        self.tensors = {}
        self.config = config
        shard_ids = []
        for reader in self.readers:
            if reader.endianess != gguf.GGUFEndian.LITTLE:
                raise ValueError("Only little-endian GGUF checkpoints are supported")
            fields = reader.fields
            architecture = fields.get("general.architecture")
            if architecture is None or architecture.contents() != architecture_name:
                raise ValueError(
                    f"Expected general.architecture={architecture_name} in every GGUF shard"
                )
            count = fields.get("split.count")
            if count is not None:
                if count.contents() != len(self.readers) or "split.no" not in fields:
                    raise ValueError("Incomplete GGUF shards: pass a directory with all shards")
                shard_ids.append(fields["split.no"].contents())
            for tensor in reader.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(
                        f"Duplicate GGUF tensor {tensor.name}; use one quantization only"
                    )
                self.tensors[tensor.name] = tensor
        if shard_ids and sorted(shard_ids) != list(range(len(self.readers))):
            raise ValueError("Inconsistent GGUF split metadata or duplicate shard indices")
        if len(self.readers) > 1 and not shard_ids:
            raise ValueError("Multiple GGUF files without split metadata; select one checkpoint")

        expected = {name: shape for name, _, shape in _gpu_tensor_specs(config)}
        # Older GGUFs keep KV-B combined; current llama.cpp splits/transposes K-B.
        if config.is_mla:
            for layer in range(config.num_layers):
                src = f"blk.{layer}"
                if f"{src}.attn_kv_b.weight" in self.tensors:
                    expected.pop(f"{src}.attn_k_b.weight")
                    expected.pop(f"{src}.attn_v_b.weight")
                    expected[f"{src}.attn_kv_b.weight"] = (
                        config.num_qo_heads * (config.qk_nope_head_dim + config.v_head_dim),
                        config.kv_lora_rank,
                    )
        for layer in range(config.num_layers):
            if config.is_mla and layer < config.first_k_dense_replace:
                continue
            for proj in ("gate", "up", "down"):
                dims = (config.moe_intermediate_size, config.hidden_size)
                if proj == "down":
                    dims = dims[::-1]
                expected[f"blk.{layer}.ffn_{proj}_exps.weight"] = (config.num_experts, *dims)
        for name, shape in expected.items():
            tensor = self.tensors.get(name)
            if tensor is None:
                raise ValueError(f"Missing GGUF tensor {name}; pass a complete GGUF checkpoint")
            actual = tuple(int(dim) for dim in tensor.shape[::-1])
            if actual != shape:
                raise ValueError(f"GGUF tensor {name}: expected shape {shape}, got {actual}")

        # Detect a mismatched config even when tensor shapes alone cannot expose it.
        metadata = {
            "block_count": config.num_layers,
            "embedding_length": config.hidden_size,
            "attention.head_count": config.num_qo_heads,
            "attention.head_count_kv": config.num_kv_heads,
            "attention.key_length": config.head_dim,
            "attention.value_length": config.head_dim,
            "expert_count": config.num_experts,
            "expert_used_count": config.num_experts_per_tok,
            "expert_feed_forward_length": config.moe_intermediate_size,
            "rope.freq_base": config.rotary_config.base,
            "attention.layer_norm_rms_epsilon": config.rms_norm_eps,
        }
        if config.is_mla:
            metadata.update(
                {
                    # GGUF describes the compressed MQA cache here, not HF's expanded heads.
                    "attention.head_count_kv": 1,
                    "attention.key_length": config.kv_lora_rank + config.qk_rope_head_dim,
                    "attention.value_length": config.kv_lora_rank,
                    "attention.key_length_mla": config.head_dim,
                    "attention.value_length_mla": config.v_head_dim,
                    "attention.q_lora_rank": config.q_lora_rank,
                    "attention.kv_lora_rank": config.kv_lora_rank,
                    "rope.dimension_count": config.qk_rope_head_dim,
                    "leading_dense_block_count": config.first_k_dense_replace,
                    "expert_shared_count": config.n_shared_experts,
                }
            )
        for reader in self.readers:
            for key, expected_value in metadata.items():
                field = reader.fields.get(f"{architecture_name}.{key}")
                if field is not None and not np.isclose(
                    field.contents(), expected_value, rtol=1e-6, atol=0
                ):
                    raise ValueError(
                        f"GGUF metadata {architecture_name}.{key} does not match --model config"
                    )

    def get_undequanted_tensor_and_ggml_type(self, name: str):
        """KT's loader protocol: a packed byte view, never a dequantized expert copy."""
        tensor = self.tensors[name]
        return torch.from_numpy(tensor.data.view(np.uint8).reshape(-1)), tensor.tensor_type

    def _dequantize(
        self, name: str, device: torch.device, dtype: torch.dtype, chunk_bytes: int
    ) -> torch.Tensor:
        tensor = self.tensors[name]
        shape = tuple(int(dim) for dim in tensor.shape[::-1])
        result = torch.empty(shape, device=device, dtype=dtype)
        rows = result.reshape(-1, shape[-1])
        packed = tensor.data.reshape(rows.shape[0], -1)
        rows_per_chunk = max(1, chunk_bytes // (shape[-1] * 4))
        for start in range(0, rows.shape[0], rows_per_chunk):
            stop = min(start + rows_per_chunk, rows.shape[0])
            try:
                dense = gguf.dequantize(packed[start:stop], tensor.tensor_type)
            except (NotImplementedError, ValueError) as exc:
                raise ValueError(
                    f"Cannot dequantize GGUF GPU tensor {name} ({tensor.tensor_type.name}): {exc}"
                ) from exc
            # Copy detaches even F32/F16 arrays from the read-only file mapping.
            # Convert on CPU so no temporary FP32 copy is allocated on the GPU.
            chunk = torch.from_numpy(np.array(dense, copy=True)).to(dtype=dtype)
            rows[start:stop].copy_(chunk)
        return result

    def weights(
        self, device: torch.device, dtype: torch.dtype, *, chunk_bytes: int = 16 * 1024 * 1024
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Pack ordinary linear layers on CPU; keep absorbed MLA projections in BF16."""
        merged = []
        combined_kv = None
        specs = list(_gpu_tensor_specs(self.config))
        for source, target, _ in tqdm(specs, desc="Loading GGUF GPU weights"):
            if self.config.is_mla and source.endswith((".attn_k_b.weight", ".attn_v_b.weight")):
                combined_source = source.rsplit(".", 2)[0] + ".attn_kv_b.weight"
                if combined_source in self.tensors:
                    c = self.config
                    if source.endswith(".attn_k_b.weight"):
                        combined_kv = self._dequantize(
                            combined_source, device, dtype, chunk_bytes
                        ).view(c.num_qo_heads, c.qk_nope_head_dim + c.v_head_dim, c.kv_lora_rank)
                        yield target, combined_kv[:, : c.qk_nope_head_dim].transpose(
                            1, 2
                        ).contiguous()
                    else:
                        yield target, combined_kv[:, c.qk_nope_head_dim :].contiguous()
                        combined_kv = None
                    continue
            is_qkv = target.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight"))
            is_gate_up = target.endswith((".gate_proj.weight", ".up_proj.weight"))
            is_marlin = (
                is_qkv
                or is_gate_up
                or target.endswith(
                    (
                        ".o_proj.weight",
                        "lm_head.weight",
                        ".q_a_proj.weight",
                        ".q_b_proj.weight",
                        ".kv_a_proj_with_mqa.weight",
                        ".down_proj.weight",
                    )
                )
            )
            tensor_dtype = torch.float32 if self.config.is_mla and ".mlp.gate." in target else dtype
            weight = self._dequantize(
                source, torch.device("cpu") if is_marlin else device, tensor_dtype, chunk_bytes
            )
            if is_qkv or is_gate_up:
                merged.append(weight)
                if len(merged) < (3 if is_qkv else 2):
                    continue
                target = (
                    target.replace(".v_proj.weight", ".qkv_proj.weight")
                    if is_qkv
                    else target.replace(".up_proj.weight", ".gate_up_proj.weight")
                )
                weight = torch.cat(merged)
                merged.clear()
            if is_marlin:
                packed, scales = pack_marlin(weight)
                yield target, packed.to(device)
                yield target.removesuffix(".weight") + ".scales", scales.to(device)
            else:
                yield target, weight
