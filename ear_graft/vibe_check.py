"""Full autoregressive rollouts from a trained graft checkpoint, thinking on and off, next to the teacher. Container only:

  for g in 0 1 2 3; do GPU=$g NAME=vibe$g ear_graft/docker_run.sh -m ear_graft.vibe_check \
      --ckpt ear_graft/out/v0/step_1400.pt --shard $g --nshards 4 & done; wait
  .venv/bin/python -m ear_graft.vibe_check --ckpt ear_graft/out/v0/step_1400.pt --report

Prompts: up to --per-task held-out val rows per task (teacher answer shown for reference) plus --pairs two-clip prompts
built from val clips (no teacher answer). Sampling: Qwen3.5's recommended settings (the 4B snapshot ships no
generation_config): thinking T=1.0 top_p=0.95 top_k=20; no-thinking T=0.7 top_p=0.8 top_k=20.
"""
import argparse
import collections
import glob
import json
import random
import time
import zlib
from pathlib import Path

import torch

from .prompts import PAIR

C = "data/ear_corpus"
# Text-only turns: a frozen-4B run (v0) answers these exactly as the base 4B does, so they are the regression check
# for runs that put LoRA on the 4B.
TEXT_PROMPTS = [
    "A train leaves at 3:40 pm and the trip takes 2 hours 35 minutes. What time does it arrive? Show your steps.",
    "Write a Python function that returns the n-th Fibonacci number in O(log n) time.",
    "Explain the difference between a low-pass and a band-pass filter on a subtractive synthesizer.",
    "What notes are in a D minor 7 chord, and what is its function in the key of C major?",
    "Summarize the causes of the 1929 stock market crash in three bullet points.",
    "Reply with only the word 'banana', in lowercase, and nothing else.",
]
SAMPLING = {True: dict(temperature=1.0, top_p=0.95, top_k=20), False: dict(temperature=0.7, top_p=0.8, top_k=20)}


def prompt_set(per_task, n_pairs, seed=0):
    rows = [json.loads(l) for l in open(f"{C}/val.jsonl")]
    rng = random.Random(seed)
    rng.shuffle(rows)
    by_task = collections.defaultdict(list)
    for r in rows:
        if len(r["clips"]) == 1 and len(by_task[r["task"]]) < per_task:
            by_task[r["task"]].append(r)
    items = [{"id": r["id"], "task": r["task"], "kinds": r["kinds"], "audios": r["audios"], "prompt": r["prompt"],
              "teacher_reasoning": r["reasoning"], "teacher_answer": r["answer"]} for t in sorted(by_task) for r in by_task[t]]
    singles = [r for r in rows if len(r["clips"]) == 1]
    for k in range(n_pairs):
        a, b = rng.sample(singles, 2)
        task = ["compare", "which", "compare", "transform"][k % 4]
        items.append({"id": f"pair{k}", "task": "pair_" + task, "kinds": a["kinds"] + b["kinds"],
                      "audios": a["audios"] + b["audios"], "prompt": rng.choice(PAIR[task]),
                      "teacher_reasoning": None, "teacher_answer": None})
    items += [{"id": f"text{k}", "task": "text_only", "kinds": [], "audios": [], "prompt": p,
               "teacher_reasoning": None, "teacher_answer": None} for k, p in enumerate(TEXT_PROMPTS)]
    return items


def rollouts(args):
    from .data import Collator
    from .model import EarGraft
    dev = torch.device("cuda", 0)
    cfg = json.loads((Path(args.ckpt).parent / "config.json").read_text())
    model = EarGraft(device=dev, projector=cfg.get("projector", "linear"), lora_r=cfg.get("lora_r", 0)).to(dev).eval()
    model.load_trainable(args.ckpt)
    collate = Collator()
    eos = [collate.tok.convert_tokens_to_ids("<|im_end|>"), collate.tok.eos_token_id]
    items = prompt_set(args.per_task, args.pairs)[args.shard::args.nshards]
    out = Path(args.ckpt).parent / f"vibe_{Path(args.ckpt).stem}.shard{args.shard}.jsonl"
    done = {(json.loads(l)["id"], json.loads(l)["think"]) for l in open(out)} if out.exists() else set()
    with open(out, "a") as f:
        for it in items:
            for think in (True, False):
                if (it["id"], think) in done:
                    continue
                b = collate([{**it, "answer": None, "think": think}])
                b = {k: v.to(dev) for k, v in b.items() if k != "labels"}
                torch.manual_seed(zlib.crc32(f"{it['id']}|{think}".encode()))
                t0 = time.time()
                ids = model.generate(**b, do_sample=True, **SAMPLING[think], eos_token_id=eos, pad_token_id=eos[0],
                                     max_new_tokens=args.max_think_tokens if think else args.max_answer_tokens)[0]
                hit_eos = bool(ids[-1].item() in eos)
                text = collate.tok.decode(ids, skip_special_tokens=False).replace("<|im_end|>", "").strip()
                f.write(json.dumps({**it, "think": think, "student": text, "n_new": len(ids), "hit_eos": hit_eos,
                                    "sec": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{it['id']} {it['task']} think={think} {len(ids)} tok eos={hit_eos} {time.time() - t0:.0f}s", flush=True)


def report(args):
    stem = Path(args.ckpt).stem
    rows = [json.loads(l) for p in sorted(glob.glob(str(Path(args.ckpt).parent / f"vibe_{stem}.shard*.jsonl"))) for l in open(p)]
    rows.sort(key=lambda r: (r["task"], r["id"], not r["think"]))
    md = [f"# Vibe check: {args.ckpt}\n", f"{len(rows)} rollouts; "
          f"{sum(r['hit_eos'] for r in rows)} ended on <|im_end|>; median {sorted(r['n_new'] for r in rows)[len(rows) // 2]} new tokens\n"]
    for r in rows:
        md.append(f"\n---\n## {r['task']} · {r['id']} · {','.join(r['kinds'])} · think={r['think']}\n")
        md.append(f"**audio:** {' | '.join(r['audios'])}\n\n**prompt:** {r['prompt']}\n")
        md.append(f"**student** ({r['n_new']} tok, eos={r['hit_eos']}):\n\n```\n{r['student']}\n```\n")
        if r["teacher_answer"] and not r["think"]:
            md.append(f"**teacher answer:**\n\n```\n{r['teacher_answer']}\n```\n")
    path = Path(args.ckpt).parent / f"vibe_{stem}.md"
    path.write_text("\n".join(md))
    print(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--per-task", type=int, default=2)
    ap.add_argument("--pairs", type=int, default=4)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--max-think-tokens", type=int, default=6000)
    ap.add_argument("--max-answer-tokens", type=int, default=1500)
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()
    report(args) if args.report else rollouts(args)


if __name__ == "__main__":
    main()
