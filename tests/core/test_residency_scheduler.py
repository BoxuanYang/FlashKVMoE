from contextlib import contextmanager
from types import SimpleNamespace

import torch
from minisgl.core import FETCHING, GPU_RESIDENT, HOST_RESIDENT, Batch, Req, SamplingParams
from minisgl.models import glm4_moe
from minisgl.scheduler.decode import MLFQ_QUANTA, DecodeManager
from minisgl.scheduler.scheduler import can_evict, should_priority_boost


class ReadyEvent:
    def __init__(self, ready: bool):
        self.ready = ready

    def query(self) -> bool:
        return self.ready


def make_req(uid: int, *, level: int = 0, state: str = GPU_RESIDENT) -> Req:
    req = Req(
        input_ids=torch.tensor([uid + 1], dtype=torch.int32),
        table_idx=uid,
        cached_len=0,
        output_len=32,
        uid=uid,
        sampling_params=SamplingParams(max_tokens=32),
        cache_handle=None,
    )
    req.mlfq_level = level
    req.resident_state = state
    return req


def test_short_requests_share_the_top_queue():
    manager = DecodeManager(page_size=1)
    reqs = [make_req(uid) for uid in (3, 1, 2)]
    manager.running_reqs.update(reqs)
    assert manager.schedule_next_batch().reqs == sorted(reqs, key=lambda req: req.uid)


def test_mlfq_prefers_new_short_request_and_demotes_by_quantum():
    manager = DecodeManager(page_size=1)
    long_req = make_req(1, level=2)
    short_req = make_req(2)
    manager.running_reqs.update((long_req, short_req))

    assert manager.schedule_next_batch().reqs == [short_req]
    for _ in range(MLFQ_QUANTA[0] - 1):
        assert not manager.record_service(short_req)
    assert manager.record_service(short_req)
    assert short_req.mlfq_level == 1


def test_migration_amortization_is_an_eligibility_gate():
    req = make_req(1)
    req.last_fetch_cost = 0.2
    req.resident_service = 0.39
    assert not can_evict(req)
    req.resident_service = 0.4
    assert can_evict(req)


def test_cost_aware_host_wait_boost_threshold():
    req = make_req(1, state=HOST_RESIDENT)
    req.last_fetch_cost = 0.2
    req.host_wait_start = 10.0
    assert not should_priority_boost(req, 10.79)
    assert should_priority_boost(req, 10.8)


def test_not_ready_fetch_does_not_block_resident_decode():
    manager = DecodeManager(page_size=1)
    resident = make_req(1)
    fetching = make_req(2, state=FETCHING)
    fetching.kv_ready_events = [ReadyEvent(False)]
    manager.running_reqs.update((resident, fetching))

    assert manager.schedule_next_batch().reqs == [resident]
    fetching.kv_ready_events[0].ready = True
    assert manager.schedule_next_batch().reqs == [resident, fetching]


def test_glm_fetching_row_stops_and_resumes_at_layer_boundary(monkeypatch):
    resident = make_req(1)
    fetching = make_req(2, state=FETCHING)
    fetching.kv_ready_events = [ReadyEvent(True), ReadyEvent(False), ReadyEvent(False)]
    batch = Batch([resident, fetching], phase="decode")
    batch.padded_reqs = batch.reqs
    batch.input_ids = torch.tensor([1, 2], dtype=torch.int32)
    batch.positions = torch.tensor([0, 0], dtype=torch.int32)
    batch.out_loc = torch.tensor([0, 1], dtype=torch.int32)

    class Layer:
        def forward(self, x, residual):
            residual = torch.zeros_like(x) if residual is None else residual + 1
            return x + 1, residual

    prepared_sizes = []

    class Backend:
        def prepare_metadata(self, sub_batch):
            prepared_sizes.append(sub_batch.size)
            sub_batch.attn_metadata = None

    active_batch = batch

    @contextmanager
    def switch_batch(next_batch, *, allow_nested=False):
        del allow_nested
        nonlocal active_batch
        previous = active_batch
        active_batch = next_batch
        try:
            yield
        finally:
            active_batch = previous

    class Context:
        attn_backend = Backend()

        @property
        def batch(self):
            return active_batch

        forward_batch = staticmethod(switch_batch)

    context = Context()
    monkeypatch.setattr(glm4_moe, "get_global_ctx", lambda: context)

    model = glm4_moe.Glm4MoeModel.__new__(glm4_moe.Glm4MoeModel)
    model.embed_tokens = SimpleNamespace(forward=lambda ids: ids.float().unsqueeze(1))
    model.layers = SimpleNamespace(op_list=[Layer(), Layer(), Layer()])
    model.norm = SimpleNamespace(forward=lambda x, residual: (x + residual, None))

    output = model._forward_resumable(batch.input_ids, batch)
    assert output.shape == (1, 1)
    assert batch.completed_indices == [0]
    assert resident.current_layer == 3
    assert fetching.current_layer == 1
    assert fetching.intermediate_activation is not None
    assert prepared_sizes == [2, 1, 1]

    fetching.kv_ready_events[1].ready = True
    fetching.kv_ready_events[2].ready = True
    resumed = Batch([fetching], phase="decode")
    resumed.padded_reqs = resumed.reqs
    resumed.input_ids = torch.tensor([2], dtype=torch.int32)
    resumed.positions = torch.tensor([0], dtype=torch.int32)
    resumed.out_loc = torch.tensor([1], dtype=torch.int32)
    active_batch = resumed
    output = model._forward_resumable(resumed.input_ids, resumed)
    assert output.shape == (1, 1)
    assert resumed.completed_indices == [0]
    assert fetching.current_layer == 3
