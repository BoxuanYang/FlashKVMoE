"""python send.py: send four different LongBench contexts concurrently to MiniSGLang."""

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier
from urllib.request import ProxyHandler, Request, build_opener
from zipfile import ZipFile

from huggingface_hub import hf_hub_download

URL = "http://127.0.0.1:30000/v1/chat/completions"
CONTEXT_CHARS = 14000
OUTPUT_TOKENS = 256
DATASETS = ("narrativeqa", "qasper", "gov_report", "qmsum")


def load_contexts():
    print("准备 LongBench（首次下载约 114 MB，之后使用缓存；下载不计时）……", flush=True)
    archive = hf_hub_download("zai-org/LongBench", "data.zip", repo_type="dataset")
    contexts = []
    with ZipFile(archive) as data:
        for dataset in DATASETS:
            with data.open(f"data/{dataset}.jsonl") as records:
                for line in records:
                    record = json.loads(line)
                    context = record["context"][:CONTEXT_CHARS]
                    if len(context) == CONTEXT_CHARS and context not in contexts:
                        contexts.append(context)
                        print(
                            f"请求 {len(contexts)}：{dataset} / {record['_id']}，正文 {len(context)} 字符"
                        )
                        break
                else:
                    raise ValueError(f"{dataset} 没有足够长的不同上下文")
    return contexts


def send_one(request_id, context, barrier):
    # A fresh early prefix avoids reusing previous runs' cached contexts.
    content = f"{uuid.uuid4().hex}\n{context}\n请用中文详细总结上述内容。 /no_think"
    request = Request(
        URL,
        data=json.dumps(
            {
                "model": "Qwen3-30B-A3B",
                "messages": [{"role": "user", "content": content}],
                "max_tokens": OUTPUT_TOKENS,
                "temperature": 0,
                "ignore_eos": True,
                "stream": True,
            }
        ).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    opener = build_opener(ProxyHandler({}))
    barrier.wait()
    started = time.perf_counter()
    first_text = None
    parts = []
    try:
        with opener.open(request, timeout=600) as response:
            for line in response:
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    break
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
            else:
                raise RuntimeError("连接提前结束，未收到 [DONE]")
        if not parts:
            raise RuntimeError("服务端返回空文本")
        elapsed = time.perf_counter() - started
        print(
            f"请求 {request_id} 完成：首段文本 {first_text:.2f}s，总耗时 {elapsed:.2f}s，"
            f"输出 {sum(map(len, parts))} 字符",
            flush=True,
        )
        return True
    except Exception as exc:
        print(f"请求 {request_id} 失败（{time.perf_counter() - started:.2f}s）：{exc}", flush=True)
        return False


def main():
    contexts = load_contexts()
    print(f"并发发送 4 个请求到 {URL}，每个正文 14000 字符，目标输出 {OUTPUT_TOKENS} tokens。")
    print("输入仅按字符截断，约 4k tokens；耗时包含排队，首段文本延迟包含解码缓冲。", flush=True)
    barrier = Barrier(4)
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(send_one, i, context, barrier) for i, context in enumerate(contexts, 1)
        ]
        successful = sum(future.result() for future in futures)
    elapsed = time.perf_counter() - started
    print(f"完成 {successful}/4，总耗时 {elapsed:.2f}s，吞吐 {successful / elapsed:.3f} 请求/s。")
    return 0 if successful == 4 else 1


if __name__ == "__main__":
    raise SystemExit(main())
