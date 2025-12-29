#!/usr/bin/env python
"""
Async image-to-text benchmark runner for OpenAI-compatible VLM backends
(vLLM, HuggingFace TGI with multimodal enabled, TensorRT-LLM serve).

Usage examples:
  python benchmark.py --data ./images --model Qwen/Qwen2.5-VL-7B-Instruct --backend vllm
  python benchmark.py --data dataset.jsonl --endpoint http://server:8000/v1 --model qwen --prompt "Caption the image"
"""

import argparse
import asyncio
import base64
import io
import json
import random
import time
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
from openai import AsyncOpenAI
from openai.types import Model
from PIL import Image
from tqdm.asyncio import tqdm_asyncio


def list_images(path: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    if path.is_file():
        return [path]
    files = [p for p in path.rglob("*") if p.suffix.lower() in exts]
    if not files:
        raise ValueError(f"No images found under {path}")
    return sorted(files)


def load_jsonl(path: Path) -> List[Tuple[str, Optional[str]]]:
    rows: List[Tuple[str, Optional[str]]] = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            img = obj.get("image") or obj.get("image_path")
            if not img:
                raise ValueError("JSONL rows must include 'image' or 'image_path'")
            rows.append((img, obj.get("prompt")))
    return rows


def encode_image(path: Path, max_side: Optional[int] = None) -> str:
    """Return base64 data URL of the image, optionally resizing to max_side px to save bandwidth."""
    data: bytes
    if max_side is None:
        data = path.read_bytes()
    else:
        img = Image.open(path).convert("RGB")
        img.thumbnail((max_side, max_side))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=92)
        data = buf.getvalue()
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


def build_messages(prompt: str, image_url: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }
    ]


async def infer_one(
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
    max_side: Optional[int],
) -> Tuple[str, float, Optional[str], Optional[str]]:
    start = time.perf_counter()
    try:
        image_url = encode_image(image_path, max_side=max_side)
        messages = build_messages(prompt, image_url)
        resp = await asyncio.wait_for(
            client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            ),
            timeout=request_timeout,
        )
        latency_ms = (time.perf_counter() - start) * 1000
        text = resp.choices[0].message.content or ""
        finish_reason = resp.choices[0].finish_reason
        return text, latency_ms, finish_reason, None
    except Exception as exc:  # noqa: BLE001
        latency_ms = (time.perf_counter() - start) * 1000
        return "", latency_ms, None, str(exc)


async def worker(
    sem: asyncio.Semaphore,
    client: AsyncOpenAI,
    model: str,
    prompt: str,
    image_path: Path,
    max_tokens: int,
    temperature: float,
    request_timeout: float,
    max_side: Optional[int],
    retries: int,
    results: list,
):
    async with sem:
        last_error: Optional[str] = None
        for attempt in range(1, retries + 2):
            text, latency_ms, finish_reason, error = await infer_one(
                client,
                model,
                prompt,
                image_path,
                max_tokens,
                temperature,
                request_timeout,
                max_side,
            )
            last_error = error
            if error is None:
                break
        error = last_error
        results.append(
            {
                "image": str(image_path),
                "prompt": prompt,
                "latency_ms": round(latency_ms, 2),
                "finish_reason": finish_reason,
                "output": text,
                "error": error,
            }
        )


async def run_benchmark(
    data: Iterable[Tuple[Path, str]],
    client: AsyncOpenAI,
    model: str,
    max_tokens: int,
    temperature: float,
    concurrency: int,
    request_timeout: float,
    max_side: Optional[int],
    retries: int,
) -> List[dict]:
    sem = asyncio.Semaphore(concurrency)
    results: List[dict] = []
    tasks = [
        asyncio.create_task(
            worker(
                sem,
                client,
                model,
                prompt,
                path,
                max_tokens,
                temperature,
                request_timeout,
                max_side,
                retries,
                results,
            )
        )
        for path, prompt in data
    ]
    await tqdm_asyncio.gather(*tasks)
    return results


def prepare_data(data_path: Path, prompt: str) -> List[Tuple[Path, str]]:
    if data_path.is_dir():
        return [(p, prompt) for p in list_images(data_path)]
    if data_path.suffix == ".jsonl":
        rows = load_jsonl(data_path)
        return [(Path(img), p or prompt) for img, p in rows]
    raise ValueError("data must be a directory of images or a JSONL file")


async def check_endpoint(client: AsyncOpenAI, model: str):
    try:
        models = await client.models.list()
        available: List[Model] = models.data
        names = {m.id for m in available}
        if model not in names:
            print(f"[warn] Model '{model}' not listed by endpoint; proceeding anyway.")
        else:
            print(f"[ok] Endpoint reachable; model '{model}' listed.")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not list models: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VLMs via OpenAI-compatible APIs.")
    parser.add_argument("--data", required=True, help="Image directory or JSONL file with image/prompt.")
    parser.add_argument("--endpoint", default="http://localhost:8000/v1", help="OpenAI-compatible base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key if backend enforces auth.")
    parser.add_argument("--model", required=True, help="Model name as seen by the serving backend.")
    parser.add_argument("--backend", default="vllm", choices=["vllm", "tgi", "trtllm"], help="Backend label for logging.")
    parser.add_argument("--prompt", default="Describe the image.", help="Fallback prompt when dataset rows omit it.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--concurrency", type=int, default=4, help="Number of in-flight requests.")
    parser.add_argument("--output", default="results.csv", help="Where to store the CSV results.")
    parser.add_argument("--max-side", type=int, default=None, help="Optional max image side in px for client-side resize.")
    parser.add_argument("--request-timeout", type=float, default=60.0, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=1, help="Retry count on failure (in addition to first try).")
    parser.add_argument("--samples", type=int, default=None, help="Limit number of samples (after shuffle).")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle dataset before limiting.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_path = Path(args.data)
    data = prepare_data(data_path, args.prompt)
    if args.shuffle:
        random.shuffle(data)
    if args.samples:
        data = data[: args.samples]

    print(f"Found {len(data)} items. Sending to {args.endpoint} ({args.backend}) with concurrency={args.concurrency}.")

    async def runner():
        client = AsyncOpenAI(base_url=args.endpoint, api_key=args.api_key)
        await check_endpoint(client, args.model)
        try:
            return await run_benchmark(
                data,
                client=client,
                model=args.model,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                concurrency=args.concurrency,
                request_timeout=args.request_timeout,
                max_side=args.max_side,
                retries=args.retries,
            )
        finally:
            await client.close()

    results = asyncio.run(runner())

    for row in results:
        row["backend"] = args.backend
        row["model"] = args.model

    df = pd.DataFrame(results)
    df.to_csv(args.output, index=False)
    if not df.empty:
        print(
            f"Wrote {len(df)} rows to {args.output}; "
            f"median latency={df['latency_ms'].median():.1f} ms; "
            f"errors={df['error'].notna().sum()}"
        )
    else:
        print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
