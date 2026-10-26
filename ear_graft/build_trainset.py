"""Teacher outputs -> data/ear_corpus/{train,val}.jsonl for projector training.

  .venv/bin/python -m ear_graft.build_trainset [--think-frac 0.5] [--val-frac 0.03] [--max-len 6144]

Filters: finish_reason != stop, empty answer/reasoning, refusals, yes/no answers that contradict the label,
self-questions demanding millisecond precision, rows longer than --max-len tokens. Each row gets think=True/False
(seeded, --think-frac): thinking rows train on reasoning + answer, others on the answer alone. Split is by clip: a
row is val if all its clips are held out, train if none are, dropped if mixed.
"""
import argparse
import collections
import glob
import hashlib
import json
import random
import re

from transformers import AutoTokenizer

from .model import QWEN4B_PATH

C = "data/ear_corpus"
REFUSAL = re.compile(r"\b(i can(?:no|')t (?:hear|listen|process)|i'm unable to|as an ai|no audio (?:was|is) (?:provided|attached))", re.I)
# timing/counting precision the teacher can only guess at ("at what exact second", "to the nearest 0.1 s")
TOO_PRECISE = re.compile(r"millisecond|\bms\b|exact (?:second|time|moment|timestamp|timing|interval|duration)"
                         r"|nearest (?:0\.1|tenth|hundredth)|to the (?:tenth|hundredth)", re.I)


def audio_tokens(dur):  # tower output length for a clip of `dur` seconds (100 mel frames/s)
    frames = int(dur * 100)
    leave = frames % 100
    return ((((leave - 1) // 2 + 1) - 1) // 2 + 1 - 1) // 2 + 1 + (frames // 100) * 13


def held_out(clip_id, frac):
    return int(hashlib.md5(clip_id.encode()).hexdigest(), 16) % 10_000 < frac * 10_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", default=f"{C}/teacher_*.jsonl")
    ap.add_argument("--think-frac", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=6144)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--suffix", default="", help="write {train,val}<suffix>.jsonl")
    args = ap.parse_args()

    dur = {}
    for l in open(f"{C}/manifest.jsonl"):
        r = json.loads(l)
        dur[r["id"]] = r["dur"]
    tok = AutoTokenizer.from_pretrained(QWEN4B_PATH)
    rng = random.Random(args.seed)
    drops = collections.Counter()
    out = {"train": [], "val": []}
    for path in sorted(glob.glob(args.inputs)):
        for l in open(path):
            r = json.loads(l)
            answer, reasoning = (r.get("answer") or "").strip(), (r.get("reasoning") or "").strip()
            if r.get("finish_reason") != "stop":
                drops["not_stop"] += 1; continue
            if not answer or not reasoning:
                drops["empty"] += 1; continue
            if REFUSAL.search(answer):
                drops["refusal"] += 1; continue
            if r["task"] == "yes_no" and not answer.lower().lstrip("*# ").startswith(r["check"]["expected"]):
                drops["yes_no_mismatch"] += 1; continue
            if r["task"].startswith("selfq") and TOO_PRECISE.search(r["prompt"]):
                drops["too_precise"] += 1; continue
            think = rng.random() < args.think_frac
            n_tok = (sum(audio_tokens(dur[c]) + 2 for c in r["clips"]) + 30
                     + len(tok(r["prompt"] + answer + (reasoning if think else ""), add_special_tokens=False).input_ids))
            if n_tok > args.max_len:
                drops["too_long"] += 1; continue
            ho = [held_out(c, args.val_frac) for c in r["clips"]]
            if any(ho) and not all(ho):
                drops["mixed_split"] += 1; continue
            out["val" if all(ho) else "train"].append({
                "id": r["id"], "task": r["task"], "kinds": r["kinds"], "clips": r["clips"], "audios": r["paths"],
                "prompt": r["prompt"], "reasoning": reasoning, "answer": answer, "think": think, "n_tok": n_tok,
                "src_file": path.split("/")[-1]})
    for split, rows in out.items():
        rng.shuffle(rows)
        with open(f"{C}/{split}{args.suffix}.jsonl", "w") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        k = collections.Counter(x for row in rows for x in row["kinds"])
        print(f"{split}: {len(rows)} rows, think {sum(r['think'] for r in rows) / max(1, len(rows)):.0%}, "
              f"tokens median {sorted(r['n_tok'] for r in rows)[len(rows) // 2] if rows else 0}, kinds {dict(k)}")
    print("dropped:", dict(drops))


if __name__ == "__main__":
    main()
