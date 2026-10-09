from types import SimpleNamespace

import minisgl.distributed.info as dist_info
import torch
from minisgl.attention import fi
from minisgl.distributed import DistributedInfo
from minisgl.kvcache.mha_pool import MHAKVCache
from minisgl.moe import ktransformers


def _cpu_store_cache(*, k_cache, v_cache, indices, k, v):
    k_cache.index_copy_(0, indices.long(), k.view_as(k_cache[: len(indices)]))
    v_cache.index_copy_(0, indices.long(), v.view_as(v_cache[: len(indices)]))


def test_gather_then_store_builds_host_shadow(monkeypatch):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    monkeypatch.setattr("minisgl.kernel.store_cache", _cpu_store_cache)
    pool = MHAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=4,
        num_pages=4,
        page_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_cpu_pages=6,
        max_transfer_tokens=3,
    )
    locations = torch.tensor([3, 1], dtype=torch.int32)
    k = torch.arange(8, dtype=torch.float32).view(2, 1, 4)
    v = k + 100

    pool.gather_prefill_kv(k, v, locations, layer_id=1)
    pool.store_kv(k, v, locations, layer_id=1)
    pool.submit_kv_offload(layer_id=1)

    torch.testing.assert_close(pool.k_cache_cpu(1).view(6, 1, 4)[locations.long()], k)
    torch.testing.assert_close(pool.v_cache_cpu(1).view(6, 1, 4)[locations.long()], v)
    assert pool.num_cpu_pages == 6


def test_prefill_staging_is_exact_size_and_released_per_batch(monkeypatch):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    pool = MHAKVCache(
        num_kv_heads=1,
        num_layers=1,
        head_dim=4,
        num_pages=4,
        page_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_cpu_pages=4,
        max_transfer_tokens=1,
    )

    locations = torch.tensor([0, 1, 2], dtype=torch.int32)
    k = torch.zeros(3, 1, 4)
    v = torch.ones(3, 1, 4)
    decode_staging = pool._decode_staging_gpu

    pool.gather_prefill_kv(k, v, locations, layer_id=0)

    assert pool._decode_capacity == 1
    assert pool._decode_staging_gpu is decode_staging
    assert pool._prefill_capacity == 3
    assert pool._prefill_staging_gpu is not None
    assert pool._prefill_staging_gpu.shape[1] == 3

    pool.release_prefill_kv_offload()

    assert pool._prefill_capacity == 0
    assert pool._prefill_staging_gpu is None
    assert pool._prefill_staging_cpu is None
    assert pool._prefill_indices_gpu is None
    assert pool._prefill_indices_cpu is None


def test_flashinfer_selects_phase_gather_before_paged_store(monkeypatch):
    calls = []

    class Cache:
        device = torch.device("cpu")
        dtype = torch.float32

        def gather_decode_kv(self, k, v, out_loc, layer_id):
            calls.append("decode-gather")

        def gather_prefill_kv(self, k, v, out_loc, layer_id):
            calls.append("prefill-gather")

        def store_kv(self, k, v, out_loc, layer_id):
            calls.append("store")

        def k_cache(self, layer_id):
            return torch.zeros(4, 1, 1, 2)

        def v_cache(self, layer_id):
            return torch.zeros(4, 1, 1, 2)

    class Wrapper:
        def run(self, **kwargs):
            calls.append("attention")
            return kwargs["q"]

    class Metadata:
        wrapper = Wrapper()

    monkeypatch.setattr(fi, "FIMetadata", Metadata)
    backend = fi.FlashInferBackend.__new__(fi.FlashInferBackend)
    backend.kvcache = Cache()
    backend._initialize_metadata_once = lambda metadata: None
    q = torch.zeros(1, 1, 2)
    k = torch.ones(1, 1, 2)
    v = torch.full((1, 1, 2), 2.0)

    for is_prefill, expected in (
        (True, "prefill-gather"),
        (False, "decode-gather"),
    ):
        batch = SimpleNamespace(
            attn_metadata=Metadata(),
            out_loc=torch.tensor([0], dtype=torch.int32),
            is_prefill=is_prefill,
        )
        assert backend.forward(q, k, v, layer_id=0, batch=batch) is q
        assert calls == [expected, "store", "attention"]
        calls.clear()


def test_kt_submission_wraps_kv_d2h_around_cpu_moe(monkeypatch):
    calls = []

    class Cache:
        shadow_enabled = True

        def sync_kv_offload(self):
            calls.append("kv-sync")

        def submit_kv_offload(self, layer_id):
            calls.append(("kv-submit", layer_id))

    class Wrapper:
        layer_idx = 7

        def submit_forward(self, hidden, ids, weights, stream):
            calls.append("moe-submit")

        def sync_forward(self, hidden, stream):
            calls.append("moe-sync")
            return hidden

    monkeypatch.setattr(ktransformers, "get_global_ctx", lambda: SimpleNamespace(kv_cache=Cache()))
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda device: SimpleNamespace(cuda_stream=123)
    )
    hidden = torch.zeros(1, 4)
    result = ktransformers.forward_with_kv_shadow(
        Wrapper(), hidden, torch.zeros(1, 1), torch.zeros(1, 1)
    )

    assert result is hidden
    assert calls == ["kv-sync", "moe-submit", ("kv-submit", 7), "moe-sync"]
