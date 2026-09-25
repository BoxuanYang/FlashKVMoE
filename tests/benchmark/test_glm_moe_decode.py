"""CPU checks for the benchmark's inputs, result aggregation, and regression."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "benchmark" / "offline" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bench = load_script("bench_glm_moe_decode")
analysis = load_script("analyze_glm_moe_decode")


def test_defaults_and_invalid_batch_sizes():
    args = bench.parse_args(["--kt-weight-path", "unused"])
    assert args.batch_sizes == [1, 2, 4, 8, 16, 24, 32, 64, 128]
    assert (args.kt_cpuinfer, args.kt_threadpool_count, args.layer_index) == (64, 2, 7)
    for values in (["0"], ["2", "2"]):
        with pytest.raises(SystemExit):
            bench.parse_args(["--kt-weight-path", "unused", "--batch-sizes", *values])


def test_independent_inputs_change_between_replays():
    args = SimpleNamespace(seed=42, hidden_states=None)
    source = bench.InputSource(args, 64, torch.device("cpu"))
    x = torch.empty(128, 64)
    source.fill(x)
    first = x.clone()
    assert torch.unique(x, dim=0).shape[0] == 128
    source.fill(x)
    assert not torch.equal(x, first)
    # The same seed reproduces the input sequence.
    bench.InputSource(args, 64, torch.device("cpu")).fill(x)
    torch.testing.assert_close(x, first)


def test_real_input_rows_are_sampled_independently(tmp_path):
    path = tmp_path / "inputs.npy"
    pool = np.arange(256 * 64, dtype=np.float32).reshape(256, 64) / 1000
    np.save(path, pool)
    args = SimpleNamespace(seed=42, hidden_states=path, batch_sizes=[1, 128])
    source = bench.InputSource(args, 64, torch.device("cpu"))
    x = torch.empty(128, 64)
    source.fill(x)
    assert torch.unique(x, dim=0).shape[0] == 128
    assert all(any(np.array_equal(row, candidate) for candidate in pool) for row in x.numpy())


def test_json_aggregation(tmp_path):
    result = bench.summarize([1.0, 2.0, 3.0, 4.0], [1.5, 3.5], [4, 4], [2, 2], [1, 2])
    assert result["avg_ms"] == 2.5
    assert result["num_measurements"] == 4
    assert result["routing"]["avg_unique_expert_sets_per_batch"] == 1.5
    path = tmp_path / "result.json"
    bench.write_json(path, result)
    import json

    assert json.loads(path.read_text())["avg_ms"] == 2.5
    assert not path.with_suffix(".json.tmp").exists()


def test_linear_regression_uses_numeric_batch_sizes():
    data = {
        "complete": True,
        "results": [
            {"batch_size": bs, "avg_ms": 0.125 * bs + 2.0}
            for bs in reversed(bench.DEFAULT_BATCH_SIZES)
        ],
    }
    rows, fit = analysis.fit_results(data)
    assert [row["batch_size"] for row in rows] == bench.DEFAULT_BATCH_SIZES
    assert fit["slope_ms_per_token"] == pytest.approx(0.125)
    assert fit["intercept_ms"] == pytest.approx(2.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert fit["rmse_ms"] < 1e-12


@pytest.mark.parametrize(
    "data",
    [
        {"complete": False, "results": []},
        {"complete": True, "results": [{"batch_size": 1, "avg_ms": 1}]},
        {
            "complete": True,
            "results": [{"batch_size": 1, "avg_ms": 1}, {"batch_size": 1, "avg_ms": 2}],
        },
        {
            "complete": True,
            "results": [{"batch_size": 1, "avg_ms": 1}, {"batch_size": 2, "avg_ms": float("nan")}],
        },
    ],
)
def test_regression_rejects_incomplete_or_invalid_data(data):
    with pytest.raises(ValueError):
        analysis.fit_results(data)


def test_constant_latency_has_undefined_r_squared():
    _, fit = analysis.fit_results(
        {
            "complete": True,
            "results": [
                {"batch_size": 1, "avg_ms": 2.0},
                {"batch_size": 128, "avg_ms": 2.0},
            ],
        }
    )
    assert fit["r_squared"] is None
    assert fit["rmse_ms"] < 1e-12


def test_linux_cpu_list():
    assert bench.parse_cpu_list("0-3,8,10-11\n") == {0, 1, 2, 3, 8, 10, 11}
