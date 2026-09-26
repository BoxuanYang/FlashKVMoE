"""Measure MiniSGL + KT decode latency with the OSC workload."""

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
from itertools import chain
from pathlib import Path

from openai import AsyncOpenAI

ROOT = Path(__file__).resolve().parent
OSC_DATA = ROOT / "evaluation" / "OSC" / "data.json"
OUTPUT_FILE = ROOT / "results" / "osc" / "decode_token_latency.json"

HOST = "127.0.0.1"
PORT = 30000
BASE_URL = f"http://{HOST}:{PORT}/v1"

MODEL = "/data1/models/GLM-4.5-Air-GGUF"
KT_WEIGHT_PATH = "/data1/models/GLM-4.5-Air-GGUF/IQ4_XS"
CUDA_DEVICE = "6"
KT_CPU_THREADS = 64

# 40,000 KV tokens / about 426 tokens per OSC request = about 94 concurrent requests.
MAX_RUNNING_REQUESTS = 96
CUDA_GRAPH_MAX_BS = 96

REQUEST_RATES = [0.5, 1, 1.5, 2, 2.5, 3]
REQUESTS_PER_RATE = 600
OUTPUT_TOKENS = 128
RANDOM_SEED = 42
SERVER_START_TIMEOUT = 15 * 60
REQUEST_TIMEOUT = 15 * 60


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
        "2",
        "--num-pages",
        "20000",
        "--max-seq-len-override",
        "7000",
        "--max-prefill-length",
        "7000",
        "--max-running-requests",
        str(MAX_RUNNING_REQUESTS),
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

    print("Starting MiniSGL + KT server...", flush=True)
    return subprocess.Popen(server_command(), cwd=ROOT, env=env, start_new_session=True)


def wait_for_server(server: subprocess.Popen) -> None:
    deadline = time.monotonic() + SERVER_START_TIMEOUT
    while time.monotonic() < deadline:
        if server.poll() is not None:
            raise RuntimeError(f"MiniSGL exited during startup with code {server.returncode}.")
        if server_is_ready():
            print("MiniSGL server is ready.", flush=True)
            return
        time.sleep(2)
    raise TimeoutError(f"MiniSGL did not become ready within {SERVER_START_TIMEOUT} seconds.")


def stop_server(server: subprocess.Popen) -> None:
    if server.poll() is not None:
        return

    print("Stopping MiniSGL server...", flush=True)
    os.killpg(server.pid, signal.SIGTERM)
    try:
        server.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(server.pid, signal.SIGKILL)
        server.wait()


def load_prompts() -> list[str]:
    prompts = list(json.loads(OSC_DATA.read_text(encoding="utf-8")).values())
    if len(prompts) < REQUESTS_PER_RATE:
        raise ValueError(f"OSC contains only {len(prompts)} prompts; need {REQUESTS_PER_RATE}.")
    return random.Random(RANDOM_SEED).sample(prompts, REQUESTS_PER_RATE)


async def measure_request(client: AsyncOpenAI, model: str, prompt: str) -> list[float]:
    stream = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=OUTPUT_TOKENS,
        temperature=0,
        stream=True,
        extra_body={"ignore_eos": True, "top_k": 1},
    )

    token_times = []
    async for chunk in stream:
        if chunk.choices and chunk.choices[0].finish_reason is None:
            token_times.append(time.perf_counter())

    if len(token_times) != OUTPUT_TOKENS:
        raise RuntimeError(f"Expected {OUTPUT_TOKENS} tokens, received {len(token_times)}.")
    return [later - earlier for earlier, later in zip(token_times, token_times[1:])]


async def run_setting(
    client: AsyncOpenAI, model: str, prompts: list[str], request_rate: float
) -> float:
    print(f"\nTesting {request_rate:g} req/s with {len(prompts)} requests...", flush=True)
    start = time.perf_counter()
    tasks = []

    for index, prompt in enumerate(prompts):
        send_at = start + index / request_rate
        await asyncio.sleep(max(0, send_at - time.perf_counter()))
        tasks.append(asyncio.create_task(measure_request(client, model, prompt)))

    decode_intervals = await asyncio.gather(*tasks)
    average = statistics.fmean(chain.from_iterable(decode_intervals))
    print(f"Average decode token latency: {average:.6f} s", flush=True)
    return average


def save_results(results: dict[str, float]) -> None:
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    temporary_file = OUTPUT_FILE.with_suffix(".tmp")
    temporary_file.write_text(
        json.dumps(results, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    temporary_file.replace(OUTPUT_FILE)


async def run_benchmark() -> None:
    prompts = load_prompts()
    results: dict[str, float] = {}

    async with AsyncOpenAI(
        base_url=BASE_URL,
        api_key="dummy",
        max_retries=0,
        timeout=REQUEST_TIMEOUT,
    ) as client:
        model = (await client.models.list()).data[0].id

        print(f"Warming up with 5 requests of {OUTPUT_TOKENS} tokens...", flush=True)
        await asyncio.gather(*(measure_request(client, model, prompt) for prompt in prompts[:5]))
        save_results({})

        for request_rate in REQUEST_RATES:
            results[f"{request_rate:g}"] = await run_setting(client, model, prompts, request_rate)
            save_results(results)

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
