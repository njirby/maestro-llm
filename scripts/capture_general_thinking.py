"""Capture a model's thinking on everyday prompts (no system prompt, no sampling params), to see its general style.

  .venv/bin/python scripts/capture_general_thinking.py --url http://127.0.0.1:8001/v1 --model qwen3.8-27b-uncensored \
      --out outputs/thinking_capture/general/qwen3.8-27b [--audio-wav target.wav]
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import time
import urllib.error
import urllib.request
from pathlib import Path

PROMPTS = {
    "arith": "What's 17 × 24?",
    "trap": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?",
    "code": "Write a Python function that merges overlapping intervals.",
    "debug": "Why does this print 3 three times instead of 0 1 2?\n\nfs = [lambda: i for i in range(3)]\nfor f in fs:\n    print(f())",
    "casual": "hey, how's it going?",
    "advice": "I'm choosing between a used Civic and a new Corolla for a 40-mile daily commute. What should I think about?",
    "fact": "Why is the sky blue?",
    "haiku": "Write a haiku about a synthesizer.",
    "plan": "Plan a 3-day trip to Kyoto.",
    "opinion": "Tabs or spaces?",
    "synth": "How do I make a supersaw lead in a synth?",
}
AUDIO_PROMPTS = {"audio_describe": "Describe this sound.", "audio_recreate": "How would I recreate this sound on a synth?"}


def call(url, body):
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:500]}") from None
    msg = resp["choices"][0]["message"]
    return {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "content": msg.get("content") or "",
            "finish_reason": resp["choices"][0].get("finish_reason"), "usage": resp.get("usage"),
            "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--audio-wav", help="also ask the audio prompts about this file (audio models only)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(k, [{"role": "user", "content": p}]) for k, p in PROMPTS.items()]
    if a.audio_wav:
        data = base64.b64encode(Path(a.audio_wav).read_bytes()).decode()
        for k, p in AUDIO_PROMPTS.items():
            jobs.append((k, [{"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}, {"type": "text", "text": p}]}]))
    jobs = [(k, i, m) for k, m in jobs for i in range(a.samples)]

    def run(job):
        k, i, msgs = job
        dst = out / f"{k}.{i}.json"
        if dst.exists():
            return dst, "cached"
        try:
            r = call(a.url, {"model": a.model, "messages": msgs})
        except Exception as e:
            return dst, f"FAILED: {e}"
        prompt = msgs[0]["content"] if isinstance(msgs[0]["content"], str) else msgs[0]["content"][-1]["text"] + " [audio]"
        dst.write_text(json.dumps({"prompt_id": k, "sample": i, "prompt": prompt, "model": a.model, **r},
                                  indent=1, ensure_ascii=False))
        return dst, f"{len(r['reasoning'])} reasoning chars, {r['finish_reason']}, {r['seconds']} s"

    with cf.ThreadPoolExecutor(12) as ex:
        for dst, msg in ex.map(run, jobs):
            print(dst.name, msg, flush=True)


if __name__ == "__main__":
    main()
