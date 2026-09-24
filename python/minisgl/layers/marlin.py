# SPDX-License-Identifier: Apache-2.0
# Adapted from KTransformers (KVCache.AI) and Marlin (Elias Frantar).
"""Fixed W4A16, group-64 Marlin path adapted from the pinned KT archive.

Only packing and the BaseOP adapter live here. The CUDA kernel is compiled
directly from archive/csrc/ktransformers_ext/cuda/gptq_marlin (Apache-2.0).
Packing follows archive's marlin_utils.py, marlin_perms.py and quant_utils.py.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from .base import BaseOP


@lru_cache(maxsize=1)
def marlin_gemm():
    from torch.utils.cpp_extension import load

    root = Path(__file__).resolve().parents[3]
    source = root / "third_party/ktransformers/archive/csrc/ktransformers_ext/cuda/gptq_marlin"
    if not (source / "gptq_marlin.cu").is_file():
        raise RuntimeError("Marlin requires the repository checkout and its KT submodule")
    if torch.cuda.get_device_capability()[0] < 8:
        raise RuntimeError("Marlin requires an Ampere or newer NVIDIA GPU")
    return load(
        name="minisgl_marlin",
        sources=[
            str(root / "python/minisgl/kernel/csrc/src/marlin.cpp"),
            str(source / "gptq_marlin.cu"),
        ],
        extra_include_paths=[str(source)],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        verbose=True,
    ).gemm


@lru_cache(maxsize=1)
def _permutations():
    perm = []
    for i in range(32):
        rows = (2 * (i % 4), 2 * (i % 4) + 1, 2 * (i % 4 + 4), 2 * (i % 4 + 4) + 1)
        tile = [16 * row + i // 4 + 8 * block for block in (0, 1) for row in rows]
        for j in range(4):
            perm.extend(p + 256 * j for p in tile)
    perm = np.asarray(perm).reshape(-1, 8)[:, [0, 2, 4, 6, 1, 3, 5, 7]].ravel()
    return torch.from_numpy(perm.copy()), [i + 8 * j for i in range(8) for j in range(8)]


def pack_marlin(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize CPU BF16 [N,K] weights; limit temporary storage to 1024 rows."""
    if weight.device.type != "cpu" or weight.dtype != torch.bfloat16 or weight.ndim != 2:
        raise ValueError("Marlin packing requires a CPU BF16 matrix")
    n, k = weight.shape
    padded_k = (k + 127) // 128 * 128
    if padded_k != k:
        weight = torch.nn.functional.pad(weight, (0, padded_k - k))
        k = padded_k
    padded_n = (n + 63) // 64 * 64
    packed = torch.empty(k // 16, padded_n * 2, dtype=torch.int32, device="cpu")
    scales = torch.empty(k // 64, padded_n, dtype=torch.bfloat16, device="cpu")
    perm, scale_perm = _permutations()
    for start in range(0, padded_n, 1024):
        end = min(start + 1024, padded_n)
        width = end - start
        block = torch.zeros(k, width, dtype=torch.bfloat16, device="cpu")
        block[:, : min(end, n) - start] = weight[start : min(end, n)].T
        block = block.reshape(k // 64, 64, width)
        s = block.abs().amax(dim=1).mul_(2 / 15)
        # Zero groups (including padding) must not divide by zero.
        s.masked_fill_(s == 0, 1)
        q = torch.round(block / s[:, None, :]).int().add_(8).clamp_(0, 15)
        q = q.reshape(k // 16, 16, width // 16, 16).permute(0, 2, 1, 3)
        q = q.reshape(-1, 1024)[:, perm].reshape(k // 16, width * 16)
        q = q.numpy().astype(np.uint32)
        result = np.zeros((k // 16, width * 2), dtype=np.uint32)
        for i in range(8):
            result |= q[:, i::8] << (4 * i)
        packed[:, start * 2 : end * 2] = torch.from_numpy(result.view(np.int32))
        scales[:, start:end] = s.reshape(-1, 64)[:, scale_perm].reshape(k // 64, width)
    return packed, scales


class MarlinLinear(BaseOP):
    def __init__(self, input_size: int, output_size: int):
        self.input_size, self.output_size = input_size, output_size
        self.padded_input = (input_size + 127) // 128 * 128
        self.padded_output = (output_size + 63) // 64 * 64
        self.weight = torch.empty(
            self.padded_input // 16, self.padded_output * 2, dtype=torch.int32
        )
        self.scales = torch.empty(self.padded_input // 64, self.padded_output, dtype=torch.bfloat16)
        self._workspace = self._empty = None

    def load_state_dict(self, state_dict, *, prefix="", _internal=False):
        super().load_state_dict(state_dict, prefix=prefix, _internal=_internal)
        self._workspace = torch.zeros(
            self.padded_output // 64 * 16, dtype=torch.int32, device=self.weight.device
        )
        self._empty = torch.empty(0, dtype=torch.int32, device=self.weight.device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.device.type != "cuda" or x.dtype != torch.bfloat16:
            raise ValueError("Marlin requires CUDA BF16 activations")
        if self._workspace is None:
            raise RuntimeError("Marlin weights have not been loaded")
        shape = x.shape[:-1]
        x = x.reshape(-1, self.input_size).contiguous()
        if self.padded_input != self.input_size:
            x = torch.nn.functional.pad(x, (0, self.padded_input - self.input_size))
        y = marlin_gemm()(
            x,
            self.weight,
            self.scales,
            self._empty,
            self._empty,
            self._workspace,
            4,
            x.shape[0],
            self.padded_output,
            self.padded_input,
            True,
        )
        return y[:, : self.output_size].reshape(*shape, self.output_size)
