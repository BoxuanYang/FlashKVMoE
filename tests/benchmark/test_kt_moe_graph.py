"""Exercise the benchmark's capture/replay ordering without CUDA or model weights."""

import ast
import math
import statistics
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("missing_node", [None, "cpu", "h2d"])
def test_full_routed_graph_recomputes_on_replay(monkeypatch, missing_node):
    events = []
    queue = []
    captured = []
    state = SimpleNamespace(capturing=False, replaying=False, timed=False, clock=0.0)

    def enqueue(kind, action):
        node = (kind, action)
        if state.capturing:
            captured.append(node)
        else:
            queue.append(node)

    class Tensor:
        def __init__(self, value, device="cpu"):
            self.value = value
            self.device = device
            self.shape = (1, 4)

        def __len__(self):
            return 1

        def view(self, *shape):
            return self

        def data_ptr(self):
            return id(self)

        def numel(self):
            return 4

        def clone(self):
            assert not queue, "Reference read before stream synchronization"
            return Tensor(self.value, self.device)

        def zero_(self):
            assert not state.timed and not queue
            self.value = 0

        def fill_(self, value):
            assert not state.timed
            enqueue("poison", lambda: setattr(self, "value", value))

        def copy_(self, source, non_blocking):
            assert self.device == "cuda" and source.device == "cpu"
            assert non_blocking
            enqueue("h2d", lambda: setattr(self, "value", source.value))

        def all(self):
            return self

        def item(self):
            return self.value

        def __invert__(self):
            return Tensor(not self.value)

    def synchronize():
        for kind, action in queue:
            if state.replaying and kind == missing_node:
                continue
            events.append((kind, state.replaying, state.timed))
            action()
        queue.clear()
        state.replaying = False

    stream = SimpleNamespace(
        cuda_stream=123, synchronize=synchronize, wait_stream=lambda other: None
    )

    class Graph:
        def replay(self):
            assert [kind for kind, _ in captured] == ["cpu", "sync", "h2d"]
            assert not queue
            state.replaying = True
            queue.extend(captured)

        def reset(self):
            assert not queue
            captured.clear()

    @contextmanager
    def capture(graph, stream):
        assert not queue and not state.timed
        state.capturing = True
        yield
        state.capturing = False

    buffers = [[Tensor(0), Tensor(0)] for _ in range(7)]
    buffers[6] = [Tensor(0, "cuda"), Tensor(0, "cuda")]
    x, ids, weights = Tensor(2, "cuda"), Tensor(3, "cuda"), Tensor(5, "cuda")
    slot = 1

    def stage(*args):
        assert args == (x, ids, weights)
        assert not state.capturing and not state.timed
        for index, source in ((0, x), (1, ids), (3, weights)):
            enqueue("d2h", lambda i=index, s=source: setattr(buffers[i][slot], "value", s.value))

    def forward_task(*args):
        assert args == (
            buffers[5][slot].data_ptr(),
            8,
            buffers[1][slot].data_ptr(),
            buffers[3][slot].data_ptr(),
            buffers[0][slot].data_ptr(),
            buffers[4][slot].data_ptr(),
            False,
        )

        def compute():
            buffers[4][slot].value = buffers[0][slot].value * buffers[3][slot].value

        return compute

    def submit(stream_id, task):
        assert stream_id == 123
        enqueue("cpu", task)

    def sync(stream_id, allow_pending):
        assert stream_id == 123 and allow_pending == 0
        enqueue("sync", lambda: None)

    wrapper = SimpleNamespace(
        layer_idx=7,
        num_experts_per_tok=8,
        copy_inputs_to_cpu_buffers=stage,
        moe=SimpleNamespace(forward_task=forward_task),
        cpu_infer=SimpleNamespace(submit_with_cuda_stream=submit, sync_with_cuda_stream=sync),
    )
    api = SimpleNamespace(
        clear_buffer_cache=lambda: None, set_capture_batch_sizes=lambda sizes: None
    )
    monkeypatch.setitem(sys.modules, "kt_kernel", SimpleNamespace(KTMoEWrapper=api))
    monkeypatch.setitem(
        sys.modules,
        "kt_kernel.experts_base",
        SimpleNamespace(
            KExpertsCPUBuffer=SimpleNamespace(buffer_depth=2, get_buffer=lambda *args: buffers)
        ),
    )

    def assert_close(actual, expected, **kwargs):
        assert math.isclose(actual.value, expected.value)

    def clock():
        state.timed = not state.timed
        state.clock += 0.001
        return state.clock

    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            current_stream=lambda device: stream,
            stream=lambda s: nullcontext(),
            CUDAGraph=Graph,
            graph=capture,
        ),
        isfinite=lambda t: Tensor(math.isfinite(t.value)),
        count_nonzero=lambda t: Tensor(int(t.value != 0)),
        testing=SimpleNamespace(assert_close=assert_close),
    )
    # Load only the function: importing the executable benchmark would import
    # the full model stack and change CUDA_VISIBLE_DEVICES on CPU test hosts.
    path = Path(__file__).resolve().parents[2] / "performance_profile/moe/test_moe.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "measure_ms"
    )
    namespace = {
        "torch": torch,
        "time": SimpleNamespace(perf_counter=clock),
        "statistics": statistics,
        "DEVICE": "cuda:0",
        "WARMUP": 3,
        "REPEATS": 4,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
    if missing_node:
        with pytest.raises((RuntimeError, AssertionError)):
            namespace["measure_ms"](wrapper, x, ids, weights, stream)
    else:
        assert namespace["measure_ms"](wrapper, x, ids, weights, stream) == pytest.approx(1.0)
        timed_events = [kind for kind, _, timed in events if timed]
        assert timed_events == ["cpu", "sync", "h2d"] * 4
        assert sum(kind == "cpu" and replay for kind, replay, _ in events) == 7
