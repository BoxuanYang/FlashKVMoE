"""Qwen3 MoE GGUF weights: CPU experts stay packed; GPU weights become dense tensors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import gguf
import numpy as np
import torch
from tqdm import tqdm

from .config import ModelConfig


def _gpu_tensor_specs(config: ModelConfig) -> Iterator[tuple[str, str, tuple[int, ...]]]:
    """GGUF name, runtime name, PyTorch shape (GGUF dimensions are reversed)."""
    hidden = config.hidden_size
    yield "token_embd.weight", "model.embed_tokens.weight", (config.vocab_size, hidden)
    yield "output_norm.weight", "model.norm.weight", (hidden,)
    if not config.tie_word_embeddings:
        yield "output.weight", "lm_head.weight", (config.vocab_size, hidden)
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


class GGUFWeights:
    """Index and validate a complete checkpoint without unpacking any expert tensors.

    Qwen3 MoE's llama.cpp converter preserves HF Q/K row order. Unlike Llama,
    these tensors must NOT be permuted before MiniSGL's NeoX-style RoPE.
    """

    def __init__(self, path: str, config: ModelConfig):
        if config.architectures != ["Qwen3MoeForCausalLM"]:
            raise ValueError("GGUF GPU weight loading currently supports Qwen3 MoE only")
        source = Path(path)
        files = sorted(source.glob("*.gguf")) if source.is_dir() else [source]
        if not files or any(not f.is_file() or f.suffix != ".gguf" for f in files):
            raise FileNotFoundError(f"No GGUF checkpoint found at {path}")
        # Readers mmap the packed data. Keep them alive until weight iteration completes.
        self.readers = [gguf.GGUFReader(str(file), mode="r") for file in files]
        self.tensors = {}
        self.config = config
        shard_ids = []
        for reader in self.readers:
            if reader.endianess != gguf.GGUFEndian.LITTLE:
                raise ValueError("Only little-endian GGUF checkpoints are supported")
            fields = reader.fields
            architecture = fields.get("general.architecture")
            if architecture is None or architecture.contents() != "qwen3moe":
                raise ValueError("Expected general.architecture=qwen3moe in every GGUF shard")
            count = fields.get("split.count")
            if count is not None:
                if count.contents() != len(files) or "split.no" not in fields:
                    raise ValueError("Incomplete GGUF shards: pass a directory with all shards")
                shard_ids.append(fields["split.no"].contents())
            for tensor in reader.tensors:
                if tensor.name in self.tensors:
                    raise ValueError(
                        f"Duplicate GGUF tensor {tensor.name}; use one quantization only"
                    )
                self.tensors[tensor.name] = tensor
        if shard_ids and sorted(shard_ids) != list(range(len(files))):
            raise ValueError("Inconsistent GGUF split metadata or duplicate shard indices")
        if len(files) > 1 and not shard_ids:
            raise ValueError("Multiple GGUF files without split metadata; select one checkpoint")

        expected = {name: shape for name, _, shape in _gpu_tensor_specs(config)}
        for layer in range(config.num_layers):
            for proj in ("gate", "up", "down"):
                dims = (config.moe_intermediate_size, config.hidden_size)
                if proj == "down":
                    dims = dims[::-1]
                expected[f"blk.{layer}.ffn_{proj}_exps.weight"] = (config.num_experts, *dims)
        for name, shape in expected.items():
            tensor = self.tensors.get(name)
            if tensor is None:
                raise ValueError(
                    f"Missing GGUF tensor {name}; pass a complete Qwen3 MoE GGUF checkpoint"
                )
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
        for reader in self.readers:
            for key, expected_value in metadata.items():
                field = reader.fields.get(f"qwen3moe.{key}")
                if field is not None and not np.isclose(
                    field.contents(), expected_value, rtol=1e-6, atol=0
                ):
                    raise ValueError(f"GGUF metadata qwen3moe.{key} does not match --model config")

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
        """Yield only GPU tensors, including fused QKV in Q, K, V order."""
        qkv = []
        specs = list(_gpu_tensor_specs(self.config))
        for source, target, _ in tqdm(specs, desc="Loading GGUF GPU weights"):
            weight = self._dequantize(source, device, dtype, chunk_bytes)
            if target.endswith((".q_proj.weight", ".k_proj.weight", ".v_proj.weight")):
                qkv.append(weight)
                if len(qkv) == 3:
                    yield target.replace(".v_proj.weight", ".qkv_proj.weight"), torch.cat(qkv)
                    qkv.clear()
            else:
                yield target, weight
