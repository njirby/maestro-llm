#!/usr/bin/env python3
"""Count training-context token lengths for SFT rollouts.

Sums text tokens (Qwen3-Omni tokenizer) across all message contents +
adds an audio-token estimate using ~100 tokens/second of audio (an
empirical heuristic for Qwen2.5/3 Omni — see MEMORY.md).

Per-file summary: count, mean, p50, p90, max. Plus per-record breakdown.

Usage:
    python scripts/count_rollout_tokens.py outputs/smoke_v3/main_final8_v19.jsonl ...
"""
from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from statistics import mean

AUDIO_TOKENS_PER_SEC = 100.0  # ~94.5 empirical, rounded for headroom


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(
        "Qwen/Qwen3-Omni-30B-A3B-Instruct", trust_remote_code=True,
    )


def audio_duration_s(path: str) -> float | None:
    try:
        with wave.open(path, "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        try:
            import soundfile as sf
            info = sf.info(path)
            return float(info.frames) / float(info.samplerate)
        except Exception:
            return None


def count_record(rec: dict, tok) -> dict:
    # Concatenate all message contents — tools list is also tokenized as part of
    # the SFT input for ms-swift, so include it.
    text_chunks: list[str] = []
    if "tools" in rec:
        text_chunks.append(json.dumps(rec["tools"], ensure_ascii=False))
    for m in rec.get("messages", []) or []:
        c = m.get("content", "")
        if c:
            text_chunks.append(c)
    text_tokens = sum(len(tok.encode(t, add_special_tokens=False)) for t in text_chunks)

    audio_tokens = 0
    audio_total_s = 0.0
    audio_files = 0
    for p in rec.get("audios", []) or []:
        d = audio_duration_s(p)
        if d is None:
            continue
        audio_total_s += d
        audio_files += 1
    audio_tokens = int(round(audio_total_s * AUDIO_TOKENS_PER_SEC))
    return {
        "id": rec.get("id"),
        "n_messages": len(rec.get("messages", []) or []),
        "n_audios": len(rec.get("audios", []) or []),
        "audio_files_resolved": audio_files,
        "audio_total_s": round(audio_total_s, 2),
        "text_tokens": text_tokens,
        "audio_tokens": audio_tokens,
        "total_tokens": text_tokens + audio_tokens,
    }


def percentile(xs: list[int], p: float) -> int:
    if not xs:
        return 0
    s = sorted(xs)
    k = max(0, min(len(s) - 1, int(round(p * (len(s) - 1)))))
    return s[k]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", type=Path, nargs="+")
    ap.add_argument("--per-record", action="store_true",
                    help="Print one line per record before the summary.")
    args = ap.parse_args()

    tok = load_tokenizer()

    for path in args.paths:
        rows = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        if not rows:
            print(f"{path}: empty", file=sys.stderr)
            continue
        results = [count_record(r, tok) for r in rows]

        if args.per_record:
            print(f"\n=== {path} (per-record) ===")
            for r in results:
                print(
                    f"  {r['id']:36}  text={r['text_tokens']:6}  "
                    f"audio={r['audio_tokens']:5} ({r['audio_total_s']:.1f}s, "
                    f"{r['audio_files_resolved']}/{r['n_audios']} resolved)  "
                    f"total={r['total_tokens']:6}"
                )

        totals = [r["total_tokens"] for r in results]
        text_tokens = [r["text_tokens"] for r in results]
        audio_tokens = [r["audio_tokens"] for r in results]
        print(
            f"\n=== {path} ({len(results)} records) ==="
        )
        print(
            f"  text:  mean={int(mean(text_tokens)):>6}  "
            f"p50={percentile(text_tokens,0.5):>6}  "
            f"p90={percentile(text_tokens,0.9):>6}  "
            f"max={max(text_tokens):>6}"
        )
        print(
            f"  audio: mean={int(mean(audio_tokens)):>6}  "
            f"p50={percentile(audio_tokens,0.5):>6}  "
            f"p90={percentile(audio_tokens,0.9):>6}  "
            f"max={max(audio_tokens):>6}   ({AUDIO_TOKENS_PER_SEC:.0f} tok/s)"
        )
        print(
            f"  TOTAL: mean={int(mean(totals)):>6}  "
            f"p50={percentile(totals,0.5):>6}  "
            f"p90={percentile(totals,0.9):>6}  "
            f"max={max(totals):>6}"
        )


if __name__ == "__main__":
    main()
