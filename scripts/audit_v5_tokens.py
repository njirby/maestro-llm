#!/usr/bin/env python
"""Token audit of a v5 run: per record type (main / transcriber / searcher / judge), chars and text tokens of the
messages (Qwen3.5 tokenizer, message contents only: no chat-template tokens), tokens of the `tools` schema, audio
clips and audio seconds per record. Median / p90 / max. Writes <run>/token_audit.json.

  .venv/bin/python scripts/audit_v5_tokens.py --run outputs/v5_rollouts/pilot20 [--tokenizer Qwen/Qwen3.5-4B]

Audio tokens are estimated at --audio-tok-per-s (12.5/s: the target model's audio path, the Qwen3-Omni audio tower
grafted onto the 4B, measured ~13/s on the graft) and, as a conservative column, at --audio-tok-per-s-conservative
(25/s: one token per 40 ms encoder frame).
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import soundfile as sf

FILES = ("main", "transcriber", "searcher", "judge")


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))] if xs else 0


def summary(xs: list[float]) -> dict:
    return {"median": round(statistics.median(xs), 1) if xs else 0, "p90": round(pct(xs, 0.9), 1),
            "max": round(max(xs), 1) if xs else 0}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--audio-tok-per-s", type=float, default=12.5)
    ap.add_argument("--audio-tok-per-s-conservative", type=float, default=25.0)
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer, local_files_only=True)
    dur: dict[str, float] = {}
    out = {"tokenizer": a.tokenizer, "audio_tok_per_s": a.audio_tok_per_s,
           "audio_tok_per_s_source": "Qwen3-Omni audio tower as grafted onto the 4B: ~12.5 tokens/s "
                                     "(lead, measured ~13/s on the graft)",
           "audio_tok_per_s_conservative": a.audio_tok_per_s_conservative,
           "audio_tok_per_s_conservative_source": "one token per 40 ms encoder frame (25/s), upper bound"}
    for f in FILES:
        rows = [json.loads(l) for l in (a.run / f"{f}.jsonl").read_text().splitlines() if l.strip()]
        ch, tk, tools_tk, clips, secs = [], [], [], [], []
        for r in rows:
            text = "".join(m["content"] for m in r["messages"])
            ch.append(len(text))
            tk.append(sum(len(tok(m["content"], add_special_tokens=False)["input_ids"]) for m in r["messages"]))
            tools_tk.append(len(tok(r["tools"], add_special_tokens=False)["input_ids"]))
            for p in r["audios"]:
                if p not in dur:
                    dur[p] = sf.info(p).duration
            clips.append(len(r["audios"]))
            secs.append(sum(dur[p] for p in r["audios"]))
        out[f] = {"n": len(rows), "chars": summary(ch), "text_tokens": summary(tk), "tools_tokens": summary(tools_tk),
                  "audio_clips": summary(clips), "audio_seconds": summary(secs),
                  "audio_tokens_est": summary([s * a.audio_tok_per_s for s in secs]),
                  "total_tokens_est": summary([t + u + s * a.audio_tok_per_s for t, u, s in zip(tk, tools_tk, secs)]),
                  "audio_tokens_est_conservative": summary([s * a.audio_tok_per_s_conservative for s in secs]),
                  "total_tokens_est_conservative": summary([t + u + s * a.audio_tok_per_s_conservative
                                                            for t, u, s in zip(tk, tools_tk, secs)])}
    (a.run / "token_audit.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
