"""Benchmark MiniSGLang + KT decode token latency with the Azure LLM trace.

The measured metric deliberately excludes request queueing and prefill/TTFT.  For
each request, only intervals between consecutive streamed output tokens are
included in the aggregate average per-token latency.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
TRACE_DIR = ROOT / "evaluation" / "AzureLLMInferenceTrace"
TRACE_DATA = TRACE_DIR / "data.json"
TRACE_SEQ_LENGTH = TRACE_DIR / "seq_length.json"
OUTPUT_FILE = ROOT / "results" / "azure_llm_inference_trace" / "kt_decode_latency.json"

HOST = "127.0.0.1"
PORT = 30000
BASE_URL = f"http://{HOST}:{PORT}/v1"

MODEL = "/data1/models/GLM-4.5-Air-GGUF"
KT_WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
CUDA_DEVICE = "6"
KT_CPU_THREADS = 64

REQUEST_RATES = [0.5, 2.0, 3.0, 4.0]
REQUESTS_PER_RATE = 100
MIN_INPUT_TOKENS = 1024  # The filter below is strict: seq_len > 1024.
OUTPUT_TOKENS = 128
RANDOM_SEED = 42
WARMUP_REQUESTS = 5

# 40,000 BF16 KV tokens. Long requests that do not fit concurrently remain in
# MiniSGLang's pending queue; the client continues to send at the offered rate.
NUM_KV_TOKENS = 40_000
MAX_SEQ_LEN = 16_384
MAX_RUNNING_REQUESTS = 96
CUDA_GRAPH_MAX_BS = 96

SERVER_START_TIMEOUT = 15 * 60
REQUEST_TIMEOUT = 30 * 60


@dataclass(frozen=True)
class TraceRequest:
    request_id: str
    prompt: str
    input_tokens: int


@dataclass(frozen=True)
class RequestMeasurement:
    request_id: str
    input_tokens: int
    token_times: tuple[float, ...]

    @property
    def decode_intervals(self) -> tuple[float, ...]:
        return tuple(
            later - earlier for earlier, later in zip(self.token_times, self.token_times[1:])
        )


def server_command() -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "minisgl",
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--model",
        MODEL,
        "--dtype",
        "bfloat16",
        "--tp-size",
        "1",
        "--moe-backend",
        "kt",
        "--kt-weight-path",
        KT_WEIGHT_PATH,
        "--kt-cpuinfer",
        str(KT_CPU_THREADS),
        "--kt-threadpool-count",
        "2",
        "--kt-method",
        "LLAMAFILE",
        "--attention-backend",
        "fi",
        "--cuda-graph-max-bs",
        str(CUDA_GRAPH_MAX_BS),
        "--page-size",
        "1",
        "--num-pages",
        str(NUM_KV_TOKENS),
        "--max-seq-len-override",
        str(MAX_SEQ_LEN),
        "--max-prefill-length",
        str(MAX_SEQ_LEN),
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
        # Azure's privacy-preserving prompts are repetitions of the same token.
        # Disable prefix reuse so that this synthetic common prefix does not make
        # nearly every request an unrealistic Radix-cache hit.
        "--cache-type",
        "naive",
    ]


def server_is_ready() -> bool:
    try:
        with urllib.request.urlopen(BASE_URL, timeout=2) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def start_server() -> subprocess.Popen:
    if sys.platform != "linux":
        raise RuntimeError("This benchmark requires Linux, CUDA, and the KT backend.")
    if server_is_ready():
        raise RuntimeError(f"Port {PORT} already has a MiniSGL server. Stop it before this test.")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = CUDA_DEVICE
    checkout_python = str(ROOT / "python")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [checkout_python, env.get("PYTHONPATH")]))

    print("Starting MiniSGLang + KT server...", flush=True)
    return subprocess.Popen(server_command(), cwd=ROOT, env=env, start_new_session=True)


def wait_for_server(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"MiniSGLang exited during startup with code {server.returncode}.")
        if server_is_ready():
            print("MiniSGLang server is ready.", flush=True)
            return
        time.sleep(2)
    raise TimeoutError(f"MiniSGLang did not become ready within {SERVER_START_TIMEOUT} seconds.")


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return

    print("Stopping MiniSGLang server...", flush=True)
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def load_requests() -> tuple[list[TraceRequest], int]:
    prompts = json.loads(TRACE_DATA.read_text(encoding="utf-8"))
    seq_lengths = json.loads(TRACE_SEQ_LENGTH.read_text(encoding="utf-8"))
    if prompts.keys() != seq_lengths.keys():
        missing_prompts = sorted(seq_lengths.keys() - prompts.keys())[:5]
        missing_lengths = sorted(prompts.keys() - seq_lengths.keys())[:5]
        raise ValueError(
            "Trace files have different request IDs: "
            f"missing prompts={missing_prompts}, missing lengths={missing_lengths}"
        )

    eligible = [
        TraceRequest(request_id, prompt, seq_lengths[request_id])
        for request_id, prompt in prompts.items()
        if seq_lengths[request_id] > MIN_INPUT_TOKENS
    ]
    if len(eligible) < REQUESTS_PER_RATE:
        raise ValueError(
            f"Only {len(eligible)} requests have seq_len > {MIN_INPUT_TOKENS}; "
            f"need {REQUESTS_PER_RATE}."
        )

    selected = random.Random(RANDOM_SEED).sample(eligible, REQUESTS_PER_RATE)
    return selected, len(eligible)


async def measure_request(
    client: AsyncOpenAI, model: str, request: TraceRequest
) -> RequestMeasurement:
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": request.prompt}],
        max_tokens=OUTPUT_TOKENS,
        temperature=0,
        stream=True,
        extra_body={"ignore_eos": True, "top_k": 1},
    )

    # Do not record request-start -> first-token. That interval contains queueing
    # and prefill. Only output-token arrival times are used below.
    token_times: list[float] = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].finish_reason is None:
            token_times.append(time.perf_counter())

    if len(token_times) != OUTPUT_TOKENS:
        raise RuntimeError(
            f"{request.request_id}: expected {OUTPUT_TOKENS} output tokens, "
            f"received {len(token_times)}."
        )
    return RequestMeasurement(request.request_id, request.input_tokens, tuple(token_times))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(int(len(ordered) * fraction), len(ordered) - 1)
    return ordered[index]


def summarize(
    measurements: list[RequestMeasurement], request_rate: float, elapsed: float
) -> dict[str, int | float]:
    intervals = list(chain.from_iterable(item.decode_intervals for item in measurements))
    if not intervals:
        raise RuntimeError("No decode token intervals were measured.")

    average_s = statistics.fmean(intervals)
    return {
        "request_rate_per_s": request_rate,
        "completed_requests": len(measurements),
        "output_tokens_per_request": OUTPUT_TOKENS,
        "decode_token_intervals": len(intervals),
        "average_per_token_latency_s": average_s,
        "average_per_token_latency_ms": average_s * 1000,
        "p50_per_token_latency_ms": percentile(intervals, 0.50) * 1000,
        "p90_per_token_latency_ms": percentile(intervals, 0.90) * 1000,
        "p99_per_token_latency_ms": percentile(intervals, 0.99) * 1000,
        "wall_time_s": elapsed,
    }


async def run_setting(
    client: AsyncOpenAI,
    model: str,
    requests: list[TraceRequest],
    request_rate: float,
) -> dict[str, int | float]:
    print(f"\nTesting {request_rate:g} req/s with {len(requests)} requests...", flush=True)
    started_at = time.perf_counter()

    async def send_at(request: TraceRequest, target: float) -> RequestMeasurement:
        await asyncio.sleep(max(0, target - time.perf_counter()))
        return await measure_request(client, model, request)

    tasks = [
        asyncio.create_task(send_at(request, started_at + index / request_rate))
        for index, request in enumerate(requests)
    ]
    measurements = await asyncio.gather(*tasks)
    elapsed = time.perf_counter() - started_at
    result = summarize(measurements, request_rate, elapsed)
    print(
        "Average decode token latency "
        f"(prefill/TTFT excluded): {result['average_per_token_latency_ms']:.3f} ms",
        flush=True,
    )
    return result


def save_results(document: dict) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_file.replace(OUTPUT_FILE)


async def run_benchmark() -> None:
    requests, eligible_count = load_requests()
    selected_lengths = [request.input_tokens for request in requests]
    document = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "MiniSGLang + KTransformers Azure trace decode latency",
        "metric": {
            "name": "average_per_token_latency",
            "definition": "Mean of all consecutive streamed output-token intervals.",
            "excludes": "Request queueing and request-start-to-first-token prefill/TTFT.",
            "weighting": "token-weighted",
        },
        "dataset": {
            "data_file": str(TRACE_DATA.relative_to(ROOT)),
            "seq_length_file": str(TRACE_SEQ_LENGTH.relative_to(ROOT)),
            "filter": f"seq_len > {MIN_INPUT_TOKENS}",
            "eligible_requests": eligible_count,
            "requests_per_rate": len(requests),
            "random_seed": RANDOM_SEED,
            "selected_input_tokens_min": min(selected_lengths),
            "selected_input_tokens_max": max(selected_lengths),
            "selected_input_tokens_average": statistics.fmean(selected_lengths),
        },
        "server": {
            "model": MODEL,
            "kt_weight_path": KT_WEIGHT_PATH,
            "cuda_device": CUDA_DEVICE,
            "kt_cpu_threads": KT_CPU_THREADS,
            "kv_tokens": NUM_KV_TOKENS,
            "max_seq_len": MAX_SEQ_LEN,
            "max_running_requests": MAX_RUNNING_REQUESTS,
            "cache_type": "naive",
        },
        "request_rates_per_s": REQUEST_RATES,
        "output_tokens_per_request": OUTPUT_TOKENS,
        "results": {},
    }
    save_results(document)

    async with AsyncOpenAI(
        base_url=BASE_URL,
        api_key="dummy",
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    ) as client:
        model = (await client.models.list()).data[0].id
        print(
            f"Selected {len(requests)} of {eligible_count} requests with "
            f"seq_len > {MIN_INPUT_TOKENS}.",
            flush=True,
        )
        print(f"Warming up with {WARMUP_REQUESTS} requests...", flush=True)
        await asyncio.gather(
            *(measure_request(client, model, request) for request in requests[:WARMUP_REQUESTS])
        )

        for request_rate in REQUEST_RATES:
            document["results"][f"{request_rate:g}"] = await run_setting(
                client, model, requests, request_rate
            )
            save_results(document)

    print(f"\nResults written to {OUTPUT_FILE}", flush=True)


def main() -> None:
    server = start_server()
    try:
        wait_for_server(server)
        asyncio.run(run_benchmark())
    finally:
        stop_server(server)


if __name__ == "__main__":
    main()
