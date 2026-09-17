import os


from datasets import load_dataset
import requests
from concurrent.futures import ThreadPoolExecutor

URL = "http://127.0.0.1:30000/generate"

NUM_REQUESTS = 4
TARGET_CHARS = 15000
MAX_NEW_TOKENS = 128


# 你以前已经下载过，所以直接从本地 HF cache 读取
dataset = load_dataset(
    "THUDM/LongBench-v2",
    split="train",
)


def make_prompt(x, idx):
    context = x["context"][:TARGET_CHARS]

    # 每个请求开头不同，避免 radix cache prefix reuse
    return (
        f"Request-{idx}\n\n"
        f"{context}\n\n"
        f"Question: {x['question']}\n"
        f"A. {x['choice_A']}\n"
        f"B. {x['choice_B']}\n"
        f"C. {x['choice_C']}\n"
        f"D. {x['choice_D']}\n"
        f"Answer:"
    )


def send(prompt):
    r = requests.post(
        URL,
        json={
            "prompt": prompt,
            "max_tokens": MAX_NEW_TOKENS,
            "ignore_eos": False,
        },
        timeout=600,
    )

    if r.status_code != 200:
        print("ERROR:", r.status_code, r.text)
        r.raise_for_status()

    return r.text


# 选 4 个不同样本，并确保 context 足够长
samples = []

for x in dataset:
    if len(x["context"]) >= TARGET_CHARS:
        samples.append(x)

    if len(samples) == NUM_REQUESTS:
        break


prompts = [
    make_prompt(x, i)
    for i, x in enumerate(samples)
]

print(f"prompts: {prompts}")

for i, prompt in enumerate(prompts):
    print(
        f"Request {i}: "
        f"{len(prompt)} chars, "
        f"domain={samples[i]['domain']}, "
        f"length={samples[i]['length']}"
    )


print("\nSending 4 concurrent requests...\n")

with ThreadPoolExecutor(max_workers=4) as pool:
    results = list(pool.map(send, prompts))


for i, result in enumerate(results):
    print(f"\n===== Request {i} =====")
    print(result[:500])