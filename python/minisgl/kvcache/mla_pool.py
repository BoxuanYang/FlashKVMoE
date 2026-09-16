from __future__ import annotations

import torch

from .base import BaseKVCachePool


class MLAKVCache(BaseKVCachePool):
    """One compressed latent and one RoPE key per token, shared by all heads."""

    def __init__(self, config, num_pages, page_size, dtype, device):
        if page_size != 1:
            raise ValueError("MLA requires --page-size 1")
        self._rank = config.kv_lora_rank
        self._buffer = torch.empty(
            config.num_layers,
            num_pages,
            1,
            config.kv_lora_rank + config.qk_rope_head_dim,
            dtype=dtype,
            device=device,
        )

    def k_cache(self, index: int) -> torch.Tensor:
        return self._buffer[index, ..., : self._rank]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._buffer[index, ..., self._rank :]

    def store_kv(self, k, v, out_loc, layer_id) -> None:
        self._buffer[layer_id, :, 0].index_copy_(0, out_loc.long(), torch.cat((k, v), dim=-1))

    @property
    def device(self):
        return self._buffer.device

    @property
    def dtype(self):
        return self._buffer.dtype

    @property
    def num_layers(self):
        return self._buffer.shape[0]
