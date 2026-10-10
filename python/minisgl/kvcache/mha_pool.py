from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import torch
from minisgl.distributed import get_tp_info
from minisgl.utils import div_even

from .base import BaseKVCachePool


def _kv_shadow_extension_paths(*roots: str | None) -> tuple[list[str], list[str]]:
    include_paths: list[str] = []
    linker_flags: list[str] = []
    for root in dict.fromkeys(root for root in roots if root):
        # NVIDIA's conda CUDA packages use a target sysroot instead of the
        # traditional $CUDA_HOME/{include,lib64} layout.  cpp_extension does
        # not add these paths when a .cpp source includes cuda_runtime.h.
        target = Path(root) / "targets" / "x86_64-linux"
        include_dir = target / "include"
        library_dir = target / "lib"
        if (include_dir / "cuda_runtime.h").is_file():
            include_paths.append(str(include_dir))
        if library_dir.is_dir():
            linker_flags.extend((f"-L{library_dir}", f"-Wl,-rpath,{library_dir}"))
    return include_paths, linker_flags


@lru_cache(maxsize=1)
def _load_kv_shadow_extension():
    from torch.utils.cpp_extension import CUDA_HOME, load

    source = Path(__file__).parents[1] / "kernel/csrc/src/kv_shadow.cpp"
    include_paths, linker_flags = _kv_shadow_extension_paths(
        CUDA_HOME, os.environ.get("CONDA_PREFIX")
    )
    return load(
        name="minisgl_kv_shadow",
        sources=[str(source)],
        with_cuda=True,
        extra_cflags=["-O3", "-std=c++17"],
        extra_include_paths=include_paths,
        extra_ldflags=linker_flags,
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

        # Decode buffers are captured by CUDA Graph and must retain both their
        # capacity and addresses. Prefill gets a separate exact-size temporary
        # allocation on its first layer and releases it when that batch ends.
        self._decode_capacity = max_transfer_tokens
        decode_shape = (
            2,
            max_transfer_tokens,
            2,
            local_kv_heads,
            head_dim,
        )
        self._decode_staging_gpu = torch.empty(decode_shape, device=device, dtype=dtype)
        self._decode_staging_cpu = torch.empty(
            decode_shape, device="cpu", dtype=dtype, pin_memory=pin_memory
        )
        self._decode_indices_gpu = torch.empty(
            (2, max_transfer_tokens), device=device, dtype=torch.int32
        )
        self._decode_indices_cpu = torch.empty(
            (2, max_transfer_tokens),
            device="cpu",
            dtype=torch.int32,
            pin_memory=pin_memory,
        )
        self._prefill_staging_gpu: torch.Tensor | None = None
        self._prefill_staging_cpu: torch.Tensor | None = None
        self._prefill_indices_gpu: torch.Tensor | None = None
        self._prefill_indices_cpu: torch.Tensor | None = None
        self._staged_counts = [0] * num_layers
        self._staged_is_prefill = [False] * num_layers
        self._offload_stream = (
            torch.cuda.Stream(device=device) if device.type == "cuda" else None
        )
        self._ready_events = (
            [torch.cuda.Event() for _ in range(num_layers)] if device.type == "cuda" else []
        )
        self._done_events = (
            [torch.cuda.Event() for _ in range(num_layers)] if device.type == "cuda" else []
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

    def gather_decode_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """Gather decode KV into the fixed-address CUDA Graph staging slot."""
        count = out_loc.numel()
        if count > self._decode_capacity:
            raise ValueError(
                f"KV shadow decode transfer has {count} tokens, exceeding its "
                f"staging capacity of {self._decode_capacity}"
            )
        slot = layer_id % 2
        shape = (count, self._storage_shape[1], self._storage_shape[2])
        self._decode_staging_gpu[slot, :count, 0].copy_(k.view(shape))
        self._decode_staging_gpu[slot, :count, 1].copy_(v.view(shape))
        self._decode_indices_gpu[slot, :count].copy_(out_loc)
        self._staged_counts[layer_id] = count
        self._staged_is_prefill[layer_id] = False

    def gather_prefill_kv(
        self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
    ) -> None:
        """Allocate exact-size eager staging on the first prefill layer, then gather."""
        count = out_loc.numel()
        if self._prefill_staging_gpu is None:
            if count <= 0:
                raise ValueError("Prefill KV shadow token count must be positive")
            prefill_shape = (
                2,
                count,
                2,
                self._storage_shape[1],
                self._storage_shape[2],
            )
            self._prefill_staging_gpu = torch.empty(
                prefill_shape, device=self._device, dtype=self.dtype
            )
            self._prefill_staging_cpu = torch.empty(
                prefill_shape,
                device="cpu",
                dtype=self.dtype,
                pin_memory=self._device.type == "cuda",
            )
            self._prefill_indices_gpu = torch.empty(
                (2, count), device=self._device, dtype=torch.int32
            )
            self._prefill_indices_cpu = torch.empty(
                (2, count),
                device="cpu",
                dtype=torch.int32,
                pin_memory=self._device.type == "cuda",
            )
        elif count != self._prefill_staging_gpu.shape[1]:
            raise RuntimeError(
                "Prefill KV token count changed within one batch: "
                f"expected {self._prefill_staging_gpu.shape[1]}, got {count}"
            )
        assert self._prefill_staging_gpu is not None
        assert self._prefill_indices_gpu is not None
        slot = layer_id % 2
        shape = (count, self._storage_shape[1], self._storage_shape[2])
        self._prefill_staging_gpu[slot, :count, 0].copy_(k.view(shape))
        self._prefill_staging_gpu[slot, :count, 1].copy_(v.view(shape))
        self._prefill_indices_gpu[slot, :count].copy_(out_loc)
        self._staged_counts[layer_id] = count
        self._staged_is_prefill[layer_id] = True

    def release_prefill_kv_offload(self) -> None:
        """Wait for the final eager callback, then discard this batch's staging."""
        if self._prefill_staging_gpu is None:
            return
        # The raw pinned-memory pointers are passed to a CUDA host callback.
        # They cannot be released until that callback has finished.
        if self._last_done_event is not None:
            self._last_done_event.synchronize()
        self._prefill_staging_gpu = None
        self._prefill_staging_cpu = None
        self._prefill_indices_gpu = None
        self._prefill_indices_cpu = None

    def get_buffer(
        self, prefill: bool
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Select staging buffers according to prefill/decode phase."""
        if not prefill:
            return (
                self._decode_staging_gpu,
                self._decode_staging_cpu,
                self._decode_indices_gpu,
                self._decode_indices_cpu,
            )
        assert self._prefill_staging_gpu is not None
        assert self._prefill_staging_cpu is not None
        assert self._prefill_indices_gpu is not None
        assert self._prefill_indices_cpu is not None
        return (
            self._prefill_staging_gpu,
            self._prefill_staging_cpu,
            self._prefill_indices_gpu,
            self._prefill_indices_cpu,
        )

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
        staging_gpu, staging_cpu, indices_gpu, indices_cpu = self.get_buffer(
            self._staged_is_prefill[layer_id]
        )
        if self._device.type != "cuda":
            staging_cpu[slot, :count].copy_(staging_gpu[slot, :count])
            indices_cpu[slot, :count].copy_(indices_gpu[slot, :count])
            self._scatter_staging(layer_id, slot, count, staging_cpu, indices_cpu)
            return

        assert self._offload_stream is not None
        producer = torch.cuda.current_stream(self._device)
        self._ready_events[layer_id].record(producer)
        with torch.cuda.stream(self._offload_stream):
            self._offload_stream.wait_event(self._ready_events[layer_id])
            staging_cpu[slot, :count].copy_(staging_gpu[slot, :count], non_blocking=True)
            indices_cpu[slot, :count].copy_(indices_gpu[slot, :count], non_blocking=True)
            if self._host_scatter is None:
                self._host_scatter = _load_kv_shadow_extension().HostScatterLauncher()
            row_bytes = self._storage_shape[1] * self._storage_shape[2] * self.dtype.itemsize
            self._host_scatter.launch(
                self._offload_stream.cuda_stream,
                self.k_cache_cpu(layer_id).data_ptr(),
                self.v_cache_cpu(layer_id).data_ptr(),
                staging_cpu[slot].data_ptr(),
                indices_cpu[slot].data_ptr(),
                count,
                row_bytes,
                self._cpu_storage_shape[0],
            )
            self._done_events[layer_id].record(self._offload_stream)
        self._last_done_event = self._done_events[layer_id]

    def _scatter_staging(
        self,
        layer_id: int,
        slot: int,
        count: int,
        staging_cpu: torch.Tensor,
        indices_cpu: torch.Tensor,
    ) -> None:
        indices = indices_cpu[slot, :count].long()
        shape = self._cpu_storage_shape
        self.k_cache_cpu(layer_id).view(shape).index_copy_(0, indices, staging_cpu[slot, :count, 0])
        self.v_cache_cpu(layer_id).view(shape).index_copy_(0, indices, staging_cpu[slot, :count, 1])

    def sync_kv_offload(self) -> None:
        """Order the current CUDA stream after the most recent host-shadow update.

        This is a stream wait rather than a host synchronize, so it is safe to use
        while capturing and replaying CUDA graphs.
        """
        if self._device.type == "cuda" and self._last_done_event is not None:
            torch.cuda.current_stream(self._device).wait_event(self._last_done_event)

    def reset_kv_offload_sync(self) -> None:
        """Drop event provenance after all eager work has completed or a graph was captured.

        CUDA stream capture may only wait on work that belongs to the same
        capture.  The event objects themselves remain alive in ``_done_events``
        for captured graph nodes; only the Python-side "latest event" marker is
        cleared so the next eager/capture generation starts independently.
        """
        self._last_done_event = None

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
