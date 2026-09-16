from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from minisgl.core import Batch, get_global_ctx
from minisgl.models import ModelConfig
from minisgl.models.deepseek_v3 import yarn_mscale

from .base import BaseAttnBackend, BaseAttnMetadata
from .utils import BaseCaptureData

if TYPE_CHECKING:
    from flashinfer.mla import BatchMLAPagedAttentionWrapper


@dataclass
class MLAMetadata(BaseAttnMetadata):
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len_arr: torch.Tensor
    qo_indptr_gpu: torch.Tensor
    wrapper: BatchMLAPagedAttentionWrapper
    initialized: bool = False

    def get_last_indices(self, bs: int) -> torch.Tensor:
        return self.qo_indptr_gpu[1 : bs + 1] - 1


class FlashInferMLABackend(BaseAttnBackend):
    def __init__(self, config: ModelConfig):
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        self.config = config
        self.kvcache = get_global_ctx().kv_cache
        self.workspace = torch.empty(
            128 * 1024 * 1024, dtype=torch.uint8, device=self.kvcache.device
        )
        self.wrapper = BatchMLAPagedAttentionWrapper(self.workspace, backend="fa2")
        scaling = config.rotary_config.scaling or {}
        self.sm_scale = (
            config.head_dim**-0.5
            * yarn_mscale(scaling.get("factor", 1.0), scaling.get("mscale_all_dim", 0.0)) ** 2
        )
        self.graph_wrappers = {}
        self.capture = None
        self.last_event = torch.cuda.Event()
        self.last_event.record()

    def _initialize(self, metadata: MLAMetadata):
        if metadata.initialized:
            return
        # The wrapper reuses staging/workspace buffers between batches.
        self.last_event.synchronize()
        c = self.config
        metadata.wrapper.plan(
            qo_indptr=metadata.qo_indptr,
            kv_indptr=metadata.kv_indptr,
            kv_indices=metadata.kv_indices,
            kv_len_arr=metadata.kv_len_arr,
            num_heads=c.num_qo_heads,
            head_dim_ckv=c.kv_lora_rank,
            head_dim_kpe=c.qk_rope_head_dim,
            page_size=1,
            causal=True,
            sm_scale=self.sm_scale,
            q_data_type=self.kvcache.dtype,
            kv_data_type=self.kvcache.dtype,
        )
        self.last_event.record()
        metadata.initialized = True

    def prepare_metadata(self, batch: Batch):
        reqs = batch.padded_reqs
        opts = {"dtype": torch.int32, "device": "cpu", "pin_memory": True}
        qo = torch.tensor([0] + [r.extend_len for r in reqs], **opts).cumsum_(0)
        kv = torch.tensor([0] + [r.device_len for r in reqs], **opts).cumsum_(0)
        table = get_global_ctx().page_table
        batch.attn_metadata = MLAMetadata(
            qo,
            kv,
            torch.cat([table[r.table_idx, : r.device_len] for r in reqs]),
            torch.tensor([r.device_len for r in reqs], **opts),
            qo.to(self.kvcache.device, non_blocking=True),
            self.wrapper,
        )

    def forward(self, q, k, v, layer_id, batch):
        metadata = batch.attn_metadata
        self._initialize(metadata)
        self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
        return metadata.wrapper.run(
            q_nope=q[..., : self.config.kv_lora_rank],
            q_pe=q[..., self.config.kv_lora_rank :],
            ckv_cache=self.kvcache.k_cache(layer_id),
            kpe_cache=self.kvcache.v_cache(layer_id),
        )

    def init_capture_graph(self, max_seq_len, bs_list):
        self.capture = BaseCaptureData.create(max(bs_list), max_seq_len, self.kvcache.device)

    def prepare_for_capture(self, batch):
        from flashinfer.mla import BatchMLAPagedAttentionWrapper

        c, bs = self.capture, batch.size
        wrapper = BatchMLAPagedAttentionWrapper(
            self.workspace,
            backend="fa2",
            use_cuda_graph=True,
            qo_indptr=c.cu_seqlens_q[: bs + 1],
            kv_indptr=c.cu_seqlens_k[: bs + 1],
            kv_indices=c.page_table.view(-1),
            kv_len_arr=c.seq_lens[:bs],
        )
        self.graph_wrappers[bs] = wrapper
        self.prepare_metadata(batch)
        batch.attn_metadata.wrapper = wrapper
        self._initialize(batch.attn_metadata)

    def prepare_for_replay(self, batch):
        batch.attn_metadata.wrapper = self.graph_wrappers[batch.padded_size]
        self._initialize(batch.attn_metadata)
