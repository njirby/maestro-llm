"""Generate teacher (prompt, audio) -> (reasoning, answer) data from Qwen3-Omni-30B-A3B-Thinking served by vLLM.

  .venv/bin/python -m ear_graft.gen_teacher --manifest data/ear_corpus/manifest.jsonl \
      --out data/ear_corpus/teacher.jsonl --n 200 --pair-frac 0.25 --seed 0

Sampling: none sent — the server applies the checkpoint's generation_config (T=0.6, top_p=0.95, top_k=20).
Only max_tokens is capped (runaway guard); rows that hit it are kept with finish_reason="length" for filtering.
Resumable: task ids are deterministic from --seed, and ids already in --out are skipped.
"""
import argparse
import asyncio
import base64
import json
import random
import time
from collections import Counter
from pathlib import Path

import httpx

from .prompts import pair_task, single_task

MIME = {".wav": "audio/wav", ".flac": "audio/flac", ".mp3": "audio/mpeg", ".ogg": "audio/ogg"}


def build_tasks(rows, n, pair_frac, seed, paraphrase=False):
    rng = random.Random(seed)
    label_pool = sorted({l for r in rows for l in r.get("meta", {}).get("labels", [])})
    by_source = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    sources = sorted(by_source)
    # sqrt(size) source weights: big sources don't drown small ones, tiny ones aren't repeated to death
    weights = [len(by_source[s]) ** 0.5 for s in sources]
    pick = lambda: rng.choices(sources, weights=weights, k=1)[0]
    tasks = []
    for i in range(n):
        src = pick()
        a = rng.choice(by_source[src])
        if rng.random() < pair_frac:
            same = rng.random() < 0.5
            b = rng.choice(by_source[src] if same else by_source[pick()])
            if b["id"] == a["id"]:
                b = rng.choice(rows)
            t = pair_task(a, b, rng, paraphrase)
            clips = [a, b]
        else:
            t = single_task(a, rng, label_pool, paraphrase)
            clips = [a]
        tasks.append({"id": f"s{seed}-{i:06d}", **t, "clips": [c["id"] for c in clips],
                      "paths": [c["path"] for c in clips], "kinds": [c["kind"] for c in clips]})
    return tasks


def content_for(task):
    parts = []
    for k, p in enumerate(task["paths"]):
        if len(task["paths"]) > 1:
            parts.append({"type": "text", "text": f"Audio {k + 1}:"})
        b64 = base64.b64encode(Path(p).read_bytes()).decode()
        parts.append({"type": "audio_url", "audio_url": {"url": f"data:{MIME[Path(p).suffix.lower()]};base64,{b64}"}})
    parts.append({"type": "text", "text": task["prompt"]})
    return parts


async def worker(q, client, args, out_f, stats):
    while True:
        task = await q.get()
        if task is None:
            return
        body = {"model": args.model, "max_tokens": args.max_tokens,
                "messages": [{"role": "user", "content": content_for(task)}]}
        for attempt in range(4):
            try:
                t0 = time.time()
                r = await client.post(f"{args.url}/v1/chat/completions", json=body)
                r.raise_for_status()
                j = r.json()
                ch = j["choices"][0]
                msg = ch["message"]
                row = {**task, "reasoning": msg.get("reasoning_content") or msg.get("reasoning"),
                       "answer": msg.get("content"), "finish_reason": ch.get("finish_reason"),
                       "usage": j.get("usage"), "latency_s": round(time.time() - t0, 1),
                       "teacher": "Qwen3-Omni-30B-A3B-Thinking", "sampling": "generation_config"}
                out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
                out_f.flush()
                stats[ch.get("finish_reason")] += 1
                stats["completion_tokens"] += (j.get("usage") or {}).get("completion_tokens", 0)
                break
            except Exception as e:  # noqa: BLE001 — log and retry; a bad clip shouldn't kill the run
                if attempt == 3:
                    stats["failed"] += 1
                    print(f"FAILED {task['id']} {task['paths']}: {e!r}"[:400], flush=True)
                await asyncio.sleep(5 * (attempt + 1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ear_corpus/manifest.jsonl")
    ap.add_argument("--out", default="data/ear_corpus/teacher.jsonl")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--pair-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sources", default="", help="comma list; default all sources in the manifest")
    ap.add_argument("--paraphrase", action="store_true", help="sample prompts from templates + prompt_paraphrases.json")
    ap.add_argument("--tasks", default="", help="jsonl of prebuilt tasks instead of sampling from the manifest")
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="omni30b-thinking")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    if args.sources:
        rows = [r for r in rows if r["source"] in args.sources.split(",")]
    if args.tasks:  # prebuilt tasks (e.g. self-generated questions): each needs id, task, prompt, clips, paths, kinds
        tasks = [json.loads(l) for l in open(args.tasks)]
    else:
        tasks = build_tasks(rows, args.n, args.pair_frac, args.seed, args.paraphrase)
    out = Path(args.out)
    done = {json.loads(l)["id"] for l in open(out)} if out.exists() else set()
    todo = [t for t in tasks if t["id"] not in done]
    print(f"{len(rows)} clips, {len(tasks)} tasks, {len(done)} done, {len(todo)} to run;",
          Counter(t["task"] for t in todo).most_common(), flush=True)

    q = asyncio.Queue()
    for t in todo:
        q.put_nowait(t)
    for _ in range(args.concurrency):
        q.put_nowait(None)
    stats = Counter()
    t0 = time.time()
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        with open(out, "a") as f:
            ws = [asyncio.create_task(worker(q, client, args, f, stats)) for _ in range(args.concurrency)]

            async def report():
                while not all(w.done() for w in ws):
                    await asyncio.sleep(60)
                    el = time.time() - t0
                    n = stats["stop"] + stats["length"]
                    print(f"[{el / 60:.1f} min] {n}/{len(todo)} done ({n / el * 3600:.0f}/h), "
                          f"{stats['completion_tokens'] / el:.0f} tok/s, {dict(stats)}", flush=True)

            rep = asyncio.create_task(report())
            await asyncio.gather(*ws)
            rep.cancel()
    print("FINISHED", dict(stats), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
