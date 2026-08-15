#!/usr/bin/env python
"""Per-kind token distribution of SFT records (3B tokenizer + audio estimate).

Audio tokens estimated at ~25 tok/s of clip duration (Qwen omni audio
encoder rate), matching the method used for the transcription-run sizing.

Usage:
    python scripts/token_stats_records.py outputs/pilot_oc_v3/*.jsonl \
        [--model <hf snapshot>] [--audio-rate 25]
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

DEFAULT_MODEL = ("/home/nate/.cache/huggingface/hub/"
                 "models--Qwen--Qwen2.5-Omni-3B/snapshots")


def _resolve(model: str) -> str:
    p = Path(model)
    if p.name == "snapshots" and p.is_dir():
        subs = sorted(p.iterdir())
        return str(subs[0]) if subs else model
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--audio-rate", type=float, default=25.0)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    import soundfile as sf
    tok = AutoTokenizer.from_pretrained(_resolve(args.model))

    dur_cache: dict[str, float] = {}
    for f in args.files:
        totals, texts, audios = [], [], []
        for line in open(f):
            r = json.loads(line)
            text = "".join(str(m.get("content", "")) for m in r["messages"])
            t_text = len(tok(text)["input_ids"])
            t_audio = 0.0
            for a in r.get("audios", []):
                if a not in dur_cache:
                    try:
                        dur_cache[a] = sf.info(a).duration
                    except Exception:
                        dur_cache[a] = 6.0
                t_audio += dur_cache[a] * args.audio_rate
            overhead = 6 * len(r["messages"]) + 20
            totals.append(t_text + int(t_audio) + overhead)
            texts.append(t_text)
            audios.append(int(t_audio))
        if not totals:
            continue
        totals_s = sorted(totals)
        p95 = totals_s[min(len(totals_s) - 1, int(len(totals_s) * 0.95))]
        print(f"{Path(f).name:<34} n={len(totals):>3}  "
              f"mean={int(st.mean(totals)):>7,}  p95={p95:>7,}  max={max(totals):>7,}  "
              f"(text mean {int(st.mean(texts)):>6,} / audio mean {int(st.mean(audios)):>6,})")


if __name__ == "__main__":
    main()
