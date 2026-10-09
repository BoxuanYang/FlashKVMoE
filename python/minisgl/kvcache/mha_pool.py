from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


@lru_cache(maxsize=1)
def _load_kv_shadow_extension():
    from torch.utils.cpp_extension import load

    source = Path(__file__).parents[1] / "kernel/csrc/src/kv_shadow.cpp"
    return load(
        name="minisgl_kv_shadow",
        sources=[str(source)],
        with_cuda=True,
        extra_cflags=["-O3", "-std=c++17"],
    )


class MHAKVCache(BaseKVCachePool):
    """
    Base class for key-value caches.
    This class defines the interface for key-value caches used in LLMs.
    """

    def __init__(
        self,
        num_kv_heads: int,
        num_layers: int,
        head_dim: int,
        num_pages: int,
        page_size: int,
        dtype: torch.dtype,
        device: torch.device,
        num_cpu_pages: int | None = None,
        max_transfer_tokens: int = 1,
    ) -> None:
        tp_info = get_tp_info()
        local_kv_heads = div_even(num_kv_heads, tp_info.size, allow_replicate=True)
        self._kv_buffer = torch.empty(
            (2, num_layers, num_pages, page_size, local_kv_heads, head_dim),
            device=device,
            dtype=dtype,
        )
        self._num_layers = num_layers
        self._k_buffer = self._kv_buffer[0]
        self._v_buffer = self._kv_buffer[1]
        self._device = device
        self._storage_shape = (num_pages * page_size, local_kv_heads, head_dim)
        self._num_pages = num_pages
        self._num_cpu_pages = num_pages if num_cpu_pages is None else num_cpu_pages
        if self._num_cpu_pages < num_pages:
            raise ValueError(
                f"Host KV cache needs at least {num_pages} pages to shadow GPU residency, "
                f"got {self._num_cpu_pages}"
            )
        if max_transfer_tokens <= 0:
            raise ValueError("max_transfer_tokens must be positive")

        # The host cache mirrors GPU physical locations for the currently resident
        # pages. Future eviction code can use the remaining CPU pages as backing.
        pin_memory = device.type == "cuda"
        self._kv_buffer_cpu = torch.empty(
            (
                2,
                num_layers,
                self._num_cpu_pages,
                page_size,
                local_kv_heads,
                head_dim,
            ),
            device="cpu",
            dtype=dtype,
        )
        self._k_buffer_cpu = self._kv_buffer_cpu[0]
        self._v_buffer_cpu = self._kv_buffer_cpu[1]
        self._cpu_storage_shape = (
            self._num_cpu_pages * page_size,
            local_kv_heads,
            head_dim,
        )

        # Two slots are sufficient: layer L+1 waits for layer L before layer L+2
        # reuses its slot. Interleaved [token, K/V, ...] makes KV one D2H DMA.
        self._max_transfer_tokens = max_transfer_tokens
        staging_shape = (2, max_transfer_tokens, 2, local_kv_heads, head_dim)
        self._staging_gpu = torch.empty(staging_shape, device=device, dtype=dtype)
        self._staging_cpu = torch.empty(
            staging_shape, device="cpu", dtype=dtype, pin_memory=pin_memory
        )
        self._indices_gpu = torch.empty(
            (2, max_transfer_tokens), device=device, dtype=torch.int32
        )
        self._indices_cpu = torch.empty(
            (2, max_transfer_tokens), device="cpu", dtype=torch.int32, pin_memory=pin_memory
        )
        self._staged_counts = [0] * num_layers
        self._shadow_enabled = device.type == "cuda"
        self._offload_stream = torch.cuda.Stream(device=device) if self._shadow_enabled else None
        self._ready_events = (
            [torch.cuda.Event() for _ in range(num_layers)] if self._shadow_enabled else []
        )
        self._done_events = (
            [torch.cuda.Event() for _ in range(num_layers)] if self._shadow_enabled else []
        )
        self._last_done_event: torch.cuda.Event | None = None
        self._host_scatter = None

    def k_cache(self, index: int) -> torch.Tensor:
        return self._k_buffer[index]

    def v_cache(self, index: int) -> torch.Tensor:
        return self._v_buffer[index]

    def k_cache_cpu(self, index: int) -> torch.Tensor:
        return self._k_buffer_cpu[index]

    def v_cache_cpu(self, index: int) -> torch.Tensor:
        return self._v_buffer_cpu[index]

    def gather_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """Gather newly generated KV into the graph-stable GPU staging slot."""
        count = out_loc.numel()
        if count > self._max_transfer_tokens:
            raise ValueError(
                f"KV shadow transfer has {count} tokens, exceeding its preallocated "
                f"capacity of {self._max_transfer_tokens}"
            )
        slot = layer_id % 2
        shape = (count, self._storage_shape[1], self._storage_shape[2])
        self._staging_gpu[slot, :count, 0].copy_(k.view(shape))
        self._staging_gpu[slot, :count, 1].copy_(v.view(shape))
        self._indices_gpu[slot, :count].copy_(out_loc)
        self._staged_counts[layer_id] = count

    def store_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        from minisgl.kernel import store_cache

        store_cache(
            k_cache=self._k_buffer[layer_id].view(self._storage_shape),
            v_cache=self._v_buffer[layer_id].view(self._storage_shape),
            indices=out_loc,
            k=k,
            v=v,
        )

    def submit_kv_offload(self, layer_id: int) -> None:
        """Queue the staged layer KV after KT has queued its expert-input D2H copies."""
        count = self._staged_counts[layer_id]
        if count == 0:
            return
        slot = layer_id % 2
        if not self._shadow_enabled:
            self._staging_cpu[slot, :count].copy_(self._staging_gpu[slot, :count])
            self._indices_cpu[slot, :count].copy_(self._indices_gpu[slot, :count])
            self._scatter_staging(layer_id, slot, count)
            return

        assert self._offload_stream is not None
        producer = torch.cuda.current_stream(self._device)
        self._ready_events[layer_id].record(producer)
        with torch.cuda.stream(self._offload_stream):
            self._offload_stream.wait_event(self._ready_events[layer_id])
            self._staging_cpu[slot, :count].copy_(
                self._staging_gpu[slot, :count], non_blocking=True
            )
            self._indices_cpu[slot, :count].copy_(
                self._indices_gpu[slot, :count], non_blocking=True
            )
            if self._host_scatter is None:
                self._host_scatter = _load_kv_shadow_extension().HostScatterLauncher()
            row_bytes = self._storage_shape[1] * self._storage_shape[2] * self.dtype.itemsize
            self._host_scatter.launch(
                self._offload_stream.cuda_stream,
                self.k_cache_cpu(layer_id).data_ptr(),
                self.v_cache_cpu(layer_id).data_ptr(),
                self._staging_cpu[slot].data_ptr(),
                self._indices_cpu[slot].data_ptr(),
                count,
                row_bytes,
                self._cpu_storage_shape[0],
            )
            self._done_events[layer_id].record(self._offload_stream)
        self._last_done_event = self._done_events[layer_id]

    def _scatter_staging(self, layer_id: int, slot: int, count: int) -> None:
        indices = self._indices_cpu[slot, :count].long()
        shape = self._cpu_storage_shape
        self.k_cache_cpu(layer_id).view(shape).index_copy_(
            0, indices, self._staging_cpu[slot, :count, 0]
        )
        self.v_cache_cpu(layer_id).view(shape).index_copy_(
            0, indices, self._staging_cpu[slot, :count, 1]
        )

    def sync_kv_offload(self) -> None:
        """Order the current CUDA stream after the most recent host-shadow update.

        This is a stream wait rather than a host synchronize, so it is safe to use
        while capturing and replaying CUDA graphs.
        """
        if self._shadow_enabled and self._last_done_event is not None:
            torch.cuda.current_stream(self._device).wait_event(self._last_done_event)

    @property
    def shadow_enabled(self) -> bool:
        return True

    @property
    def num_cpu_pages(self) -> int:
        return self._num_cpu_pages

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def dtype(self) -> torch.dtype:
        return self._kv_buffer.dtype

    @property
    def num_layers(self) -> int:
        return self._num_layers
