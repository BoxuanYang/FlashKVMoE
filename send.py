"""Send four concurrent Qwen3 requests, each with exactly 4096 input tokens.

Run in the minisgl-kt environment: python send.py
Server: --page-size 1 --num-pages 18432 --max-seq-len-override 4352
        --max-prefill-length 16384 --max-running-requests 4
        --cuda-graph-max-bs 0
"""

import argparse
import asyncio
import json
import time
import uuid
from itertools import combinations
from os.path import commonprefix
from zipfile import ZipFile

import httpx
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

INPUT_TOKENS = 4096
REQUESTS = 4
DATASETS = ("narrativeqa", "qasper", "gov_report", "qmsum")


def make_prompt(tokenizer, body_ids, request_id):
    # Different early prefixes also avoid reusing this run's prompts on later runs.
    prefix = f"{uuid.uuid4().hex}: document {request_id}\n"

    def render(body):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prefix + "请用中文详细总结以下内容：\n" + body}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

    count = INPUT_TOKENS - len(tokenizer.encode(render("")))
    for _ in range(8):
        body = tokenizer.decode(body_ids[:count], clean_up_tokenization_spaces=False)
        prompt = render(body)
        difference = INPUT_TOKENS - len(tokenizer.encode(prompt))
        if difference == 0:
            return prompt
        count += difference
    raise ValueError("无法构造恰好 4096 tokens 的输入，请检查是否使用了服务端同一份 tokenizer")


def load_prompts(tokenizer):
    print(
        "准备 LongBench 数据（首次下载约 114 MB，之后使用本地缓存；不计入请求耗时）。", flush=True
    )
    archive = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    prompts = []
    with ZipFile(archive) as data:
        for request_id, dataset in enumerate(DATASETS, 1):
            with data.open(f"data/{dataset}.jsonl") as records:
                for line in records:
                    record = json.loads(line)
                    body_ids = tokenizer.encode(record["context"], add_special_tokens=False)
                    if len(body_ids) >= INPUT_TOKENS:
                        prompts.append(make_prompt(tokenizer, body_ids, request_id))
                        print(f"请求 {request_id} 上下文：{dataset} / {record['_id']}", flush=True)
                        break
                else:
                    raise ValueError(f"{dataset} 中没有足够长的上下文")
    token_ids = [tokenizer.encode(prompt) for prompt in prompts]
    shared = max(len(commonprefix(pair)) for pair in combinations(token_ids, 2))
    print(f"任意两请求的最长公共前缀：{shared} tokens（含聊天模板）。", flush=True)
    return prompts


async def send_one(client, args, tokenizer, request_id, prompt):
    started = time.perf_counter()
    first_text = None
    parts = []
    try:

        async def consume():
            nonlocal first_text
            async with client.stream(
                "POST",
                args.url.rstrip("/") + "/v1/chat/completions",
                json={
                    "model": args.model,
                    # MiniSGL accepts a raw prompt here; the template is already included.
                    "prompt": prompt,
                    "max_tokens": args.output_tokens,
                    "temperature": 0,
                    "ignore_eos": True,
                    "stream": True,
                },
            ) as response:
                if response.is_error:
                    detail = (await response.aread()).decode(errors="replace")
                    raise RuntimeError(f"HTTP {response.status_code}: {detail[:500]}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        return
                    chunk = json.loads(data)
                    if "error" in chunk:
                        raise RuntimeError(str(chunk["error"]))
                    for choice in chunk.get("choices", []):
                        text = choice.get("delta", {}).get("content")
                        if text:
                            if first_text is None:
                                first_text = time.perf_counter() - started
                                print(f"请求 {request_id} 首段文本：{first_text:.2f}s", flush=True)
                            parts.append(text)
                raise RuntimeError("连接提前结束，未收到 [DONE]")

        await asyncio.wait_for(consume(), timeout=args.timeout)
        elapsed = time.perf_counter() - started
        if not parts:
            raise RuntimeError("服务端返回了空文本")
        text = "".join(parts)
        # The current server reports usage=0. Retokenized text is an estimate,
        # not the actual number of generated token IDs (e.g. EOS is not included).
        output_tokens = len(tokenizer.encode(text, add_special_tokens=False))
        print(
            f"请求 {request_id} 完成：输入={INPUT_TOKENS} tokens，输出≈{output_tokens} tokens，"
            f"首段文本={first_text:.2f}s，总耗时={elapsed:.2f}s\n"
            f"回复预览：{text[:200]!r}",
            flush=True,
        )
        return output_tokens
    except Exception as exc:
        print(
            f"请求 {request_id} 失败（{time.perf_counter() - started:.2f}s）："
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return None


async def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = load_prompts(tokenizer)
    print(
        f"同时提交 {REQUESTS} 个请求：每个输入 {INPUT_TOKENS} tokens，"
        f"输出目标 {args.output_tokens} tokens（ignore_eos=True）。",
        flush=True,
    )
    print("耗时包含排队；首段文本延迟不等同于 GPU 首 token 计算时间。", flush=True)
    async with httpx.AsyncClient(
        timeout=args.timeout,
        trust_env=False,
        limits=httpx.Limits(max_connections=REQUESTS, max_keepalive_connections=REQUESTS),
    ) as client:
        started = time.perf_counter()
        results = await asyncio.gather(
            *[send_one(client, args, tokenizer, i, prompt) for i, prompt in enumerate(prompts, 1)]
        )
        elapsed = time.perf_counter() - started
    successful = [n for n in results if n is not None]
    print(
        f"完成 {len(successful)}/{REQUESTS}，整轮耗时 {elapsed:.2f}s，"
        f"成功请求合计输出吞吐≈{sum(successful) / elapsed:.2f} tokens/s。"
    )
    return 0 if len(successful) == REQUESTS else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--model", default="/data2/models/Qwen3-30B-A3B")
    parser.add_argument("--output-tokens", type=int, default=256)
    parser.add_argument("--timeout", type=float, default=600, help="每个请求的总超时秒数")
    args = parser.parse_args()
    if args.output_tokens < 1 or args.timeout <= 0:
        parser.error("output-tokens 和 timeout 必须大于 0")
    raise SystemExit(asyncio.run(main(args)))
