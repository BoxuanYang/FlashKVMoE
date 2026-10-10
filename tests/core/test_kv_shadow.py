from types import SimpleNamespace

import minisgl.distributed.info as dist_info
import torch
from minisgl.attention import fi
from minisgl.distributed import DistributedInfo
from minisgl.engine.graph import _isolated_kv_offload_capture
from minisgl.kvcache.mha_pool import MHAKVCache, _kv_shadow_extension_paths
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


def test_decode_uses_layer_private_staging_and_builds_host_shadow(monkeypatch):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    pool = MHAKVCache(
        num_kv_heads=1,
        num_layers=3,
        head_dim=4,
        num_pages=4,
        page_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
        num_cpu_pages=4,
        max_transfer_tokens=2,
    )
    locations = torch.tensor([2, 0], dtype=torch.int32)
    k = torch.arange(8, dtype=torch.float32).view(2, 1, 4)
    v = k + 100

    pool.gather_decode_kv(k, v, locations, layer_id=2)
    plan = pool.submit_kv_offload(layer_id=2)

    assert plan is None  # CPU execution scatters immediately.
    assert pool._decode_staging_gpu.shape[0] == 3
    torch.testing.assert_close(pool.k_cache_cpu(2).view(4, 1, 4)[locations.long()], k)
    torch.testing.assert_close(pool.v_cache_cpu(2).view(4, 1, 4)[locations.long()], v)


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
    assert pool._prefill_staging_gpu is not None
    assert pool._prefill_staging_gpu.shape[1] == 3

    pool.release_prefill_kv_offload()

    assert pool._prefill_staging_gpu is None
    assert pool._prefill_staging_cpu is None
    assert pool._prefill_indices_gpu is None
    assert pool._prefill_indices_cpu is None


def test_reset_kv_offload_sync_forgets_only_latest_event(monkeypatch):
    monkeypatch.setattr(dist_info, "_TP_INFO", DistributedInfo(0, 1))
    pool = MHAKVCache(
        num_kv_heads=1,
        num_layers=2,
        head_dim=4,
        num_pages=4,
        page_size=1,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    done_events = pool._done_events
    pool._last_done_event = object()

    pool.reset_kv_offload_sync()

    assert pool._last_done_event is None
    assert pool._done_events is done_events


def test_capture_boundary_resets_eager_and_captured_event_generations():
    class Cache:
        def __init__(self):
            self.latest = "eager"
            self.resets = 0

        def reset_kv_offload_sync(self):
            self.latest = None
            self.resets += 1

    cache = Cache()
    with _isolated_kv_offload_capture(cache):
        assert cache.latest is None
        cache.latest = "captured"

    assert cache.latest is None
    assert cache.resets == 2


def test_conda_cuda_target_sysroot_is_added_to_extension_build(tmp_path):
    include_dir = tmp_path / "targets" / "x86_64-linux" / "include"
    library_dir = tmp_path / "targets" / "x86_64-linux" / "lib"
    include_dir.mkdir(parents=True)
    library_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime.h").touch()

    includes, flags = _kv_shadow_extension_paths(str(tmp_path), str(tmp_path))

    assert includes == [str(include_dir)]
    assert flags == [f"-L{library_dir}", f"-Wl,-rpath,{library_dir}"]


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
    plan = (1, 2, 3, 4, 5, 6, 7)
    deferred_plan = (8, 9, 10, 11, 12, 13, 14)

    class Cache:
        shadow_enabled = True

        def sync_kv_offload(self):
            calls.append("kv-prefill-sync")

        def submit_kv_offload(self, layer_id):
            calls.append(("kv-submit", layer_id))
            return plan

        def take_deferred_kv_scatters(self):
            calls.append("kv-take-deferred")
            return [deferred_plan]

    class Wrapper:
        layer_idx = 7

        def submit_forward(self, hidden, ids, weights, stream):
            calls.append("moe-submit")

        def sync_forward(self, hidden, stream, kv_scatter_plans=None):
            calls.append(("moe-sync", kv_scatter_plans))
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
    assert calls == [
        "kv-prefill-sync",
        "moe-submit",
        ("kv-submit", 7),
        "kv-take-deferred",
        ("moe-sync", [list(deferred_plan), list(plan)]),
    ]


def test_dense_decode_scatter_is_deferred_to_next_moe_sync(monkeypatch):
    calls = []
    plan = (1, 2, 3, 4, 5, 6, 7)

    class Cache:
        shadow_enabled = True

        def sync_kv_offload(self):
            calls.append("kv-prefill-sync")

        def submit_kv_offload(self, layer_id, *, join_producer=True):
            calls.append(("kv-submit", layer_id, join_producer))
            return plan

        def defer_kv_scatter(self, layer_id, value):
            calls.append(("kv-defer", layer_id, value))

    monkeypatch.setattr(ktransformers, "get_global_ctx", lambda: SimpleNamespace(kv_cache=Cache()))

    ktransformers.submit_dense_layer_kv_shadow(0)

    assert calls == [
        "kv-prefill-sync",
        ("kv-submit", 0, False),
        ("kv-defer", 0, plan),
    ]
