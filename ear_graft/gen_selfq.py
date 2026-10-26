"""Self-questioned teacher data, step A: the teacher listens to a clip (or a pair) and writes questions about it.

  .venv/bin/python -m ear_graft.gen_selfq --n-clips 3000 --pair-frac 0.2 --seed 4 --out data/ear_corpus/selfq_tasks_s4.jsonl
  .venv/bin/python -m ear_graft.gen_teacher --tasks data/ear_corpus/selfq_tasks_s4.jsonl --out data/ear_corpus/teacher_selfq_s4.jsonl

Step B (gen_teacher --tasks) asks each question fresh, with only the audio and the question, so every answer is the
teacher's response to exactly the prompt that will be trained on. Raw step-A outputs are kept next to the task file.
"""
import argparse
import asyncio
import json
import random
import re
from pathlib import Path

import httpx

from .gen_teacher import content_for

ASK_ONE = """Listen to this audio closely. Write {k} questions that a person might ask about this specific clip, which can only be answered well by listening to it carefully: about particular details, events and their timing, timbre, pitch, rhythm, dynamics, words spoken or sung, the recording space, or how things change over time. Mix question styles (some open-ended, some specific). Don't put the answers in the questions, and don't describe the clip in them beyond what is needed to point at the part you mean.

Finish with a JSON list of the question strings."""
ASK_PAIR = """Listen closely to both clips (Audio 1 and Audio 2). Write {k} questions that a person might ask about these two specific clips together, which can only be answered well by listening to both carefully: comparisons, differences in detail, timing, timbre, pitch, rhythm, dynamics, space, or how one might be turned into the other. Don't put the answers in the questions.

Finish with a JSON list of the question strings."""


def pick_clips(rows, n, pair_frac, seed):
    rng = random.Random(seed)
    by_src = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    srcs = sorted(by_src)
    w = [len(by_src[s]) ** 0.5 for s in srcs]
    out = []
    for i in range(n):
        a = rng.choice(by_src[rng.choices(srcs, weights=w, k=1)[0]])
        clips = [a]
        if rng.random() < pair_frac:
            b = rng.choice(by_src[a["source"]] if rng.random() < 0.5 else by_src[rng.choices(srcs, weights=w, k=1)[0]])
            if b["id"] != a["id"]:
                clips.append(b)
        out.append({"gid": f"q{seed}-{i:06d}", "clips": clips, "k": rng.choice([2, 3, 3, 4])})
    return out


async def worker(q, client, args, raw_f, task_f, stats):
    while (g := await q.get()) is not None:
        ask = (ASK_PAIR if len(g["clips"]) == 2 else ASK_ONE).format(k=g["k"])
        task = {"paths": [c["path"] for c in g["clips"]], "prompt": ask}
        body = {"model": args.model, "max_tokens": args.max_tokens, "messages": [{"role": "user", "content": content_for(task)}]}
        for attempt in range(4):
            try:
                r = await client.post(f"{args.url}/v1/chat/completions", json=body)
                r.raise_for_status()
                text = r.json()["choices"][0]["message"]["content"] or ""
                m = re.findall(r"\[[\s\S]*\]", text)
                qs = [s.strip() for s in (json.loads(m[-1]) if m else []) if isinstance(s, str) and len(s.strip()) > 8]
                raw_f.write(json.dumps({"gid": g["gid"], "clips": [c["id"] for c in g["clips"]], "raw": text, "questions": qs}) + "\n")
                raw_f.flush()
                for j, question in enumerate(qs[:6]):
                    task_f.write(json.dumps({
                        "id": f"{g['gid']}-{j}", "task": "selfq_pair" if len(g["clips"]) == 2 else "selfq",
                        "prompt": question, "clips": [c["id"] for c in g["clips"]], "paths": [c["path"] for c in g["clips"]],
                        "kinds": [c["kind"] for c in g["clips"]]}) + "\n")
                task_f.flush()
                stats["clips"] += 1
                stats["questions"] += len(qs[:6])
                stats["no_json"] += not qs
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 3:
                    stats["failed"] += 1
                    print(f"FAILED {g['gid']}: {e!r}"[:300], flush=True)
                await asyncio.sleep(5 * (attempt + 1))


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/ear_corpus/manifest.jsonl")
    ap.add_argument("--out", default="data/ear_corpus/selfq_tasks_s4.jsonl")
    ap.add_argument("--n-clips", type=int, default=3000)
    ap.add_argument("--pair-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=4)
    ap.add_argument("--concurrency", type=int, default=40)
    ap.add_argument("--max-tokens", type=int, default=12000)
    ap.add_argument("--url", default="http://127.0.0.1:8100")
    ap.add_argument("--model", default="omni30b-thinking")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.manifest)]
    groups = pick_clips(rows, args.n_clips, args.pair_frac, args.seed)
    out = Path(args.out)
    raw_path = out.with_name(out.stem + "_raw.jsonl")
    done = {json.loads(l)["gid"] for l in open(raw_path)} if raw_path.exists() else set()
    todo = [g for g in groups if g["gid"] not in done]
    print(f"{len(groups)} clip groups, {len(done)} done, {len(todo)} to run", flush=True)
    q = asyncio.Queue()
    for g in todo:
        q.put_nowait(g)
    for _ in range(args.concurrency):
        q.put_nowait(None)
    stats = {"clips": 0, "questions": 0, "no_json": 0, "failed": 0}
    async with httpx.AsyncClient(timeout=httpx.Timeout(1800.0)) as client:
        with open(raw_path, "a") as raw_f, open(out, "a") as task_f:
            ws = [asyncio.create_task(worker(q, client, args, raw_f, task_f, stats)) for _ in range(args.concurrency)]
            while not all(w.done() for w in ws):
                await asyncio.sleep(60)
                print(stats, flush=True)
            await asyncio.gather(*ws)
    print("FINISHED", stats, flush=True)


if __name__ == "__main__":
    asyncio.run(main())
