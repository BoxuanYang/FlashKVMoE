"""CPU checks for callback timing and the standalone analysis script."""

import importlib.util
import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

SCRIPTS = Path(__file__).resolve().parents[2] / "benchmark" / "offline"


def test_callback_times_only_synchronous_expert_forward():
    spec = importlib.util.spec_from_file_location("bench", SCRIPTS / "bench_glm_moe_decode.py")
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    assert bench.MODEL == "/data1/models/GLM-4.5-Air-GGUF"
    assert bench.WEIGHT_PATH == "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
    assert bench.PHYSICAL_GPU == "2"
    assert bench.AUTO_ANALYZE is True
    assert list(bench.BATCH_SIZES) == list(range(1, 129))
    assert bench.REPEATS == 20
    bench.experts = SimpleNamespace(forward=Mock())
    bench.forward_args = (1, 2, 3, 4, 5, 6, False)
    bench.callback_error = None
    bench.time = SimpleNamespace(perf_counter_ns=Mock(side_effect=[100, 2_500_100]))
    bench.timed_experts(None)
    bench.experts.forward.assert_called_once_with(*bench.forward_args)
    assert bench.elapsed_ms == 2.5
    assert bench.callback_error is None
    bench.experts.forward.side_effect = RuntimeError("CPU failure")
    bench.time.perf_counter_ns = Mock(return_value=0)
    bench.timed_experts(None)
    assert str(bench.callback_error) == "CPU failure"


def test_analysis_fits_all_128_sizes_and_writes_plot(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    Path("results").mkdir()
    data = {
        "complete": True,
        "results": [{"batch_size": b, "avg_ms": 2 + 0.125 * b} for b in range(128, 0, -1)],
    }
    Path("results/glm_moe_decode.json").write_text(json.dumps(data), encoding="utf-8")
    runpy.run_path(str(SCRIPTS / "analyze_glm_moe_decode.py"), run_name="__main__")
    fit = json.loads(Path("results/glm_moe_decode_fit.json").read_text())
    assert fit["slope_ms_per_token"] == pytest.approx(0.125)
    assert fit["intercept_ms"] == pytest.approx(2.0)
    assert fit["r_squared"] == pytest.approx(1.0)
    assert Path("results/glm_moe_decode.png").read_bytes().startswith(b"\x89PNG")
    data["complete"] = False
    Path("results/glm_moe_decode.json").write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError):
        runpy.run_path(str(SCRIPTS / "analyze_glm_moe_decode.py"), run_name="__main__")
