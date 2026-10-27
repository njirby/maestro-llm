"""Capture what a model's own thinking looks like at real decision points of a v5 rollout.

The replay logs (outputs/replay/<run>/<sid>/io/requests.jsonl) hold the exact request bodies real OpenCode built. This
re-sends a selection of them to a live model and stores its reasoning, text and tool calls. Only the context is ours:
no sampling params and no max_tokens are sent (server defaults), and nothing is appended to the prompt.

Audio parts reference io/media/<sha>.wav. They are inlined as base64 for audio models (--audio) and dropped for
text-only models (the "Attached media from tool result:" text stays).

  .venv/bin/python scripts/capture_natural_thinking.py --url http://127.0.0.1:8000/v1 --model omni-thinking --audio \
      --out outputs/thinking_capture/omni-thinking outputs/replay/v5_pilot20c/s0003 outputs/replay/v5_pilot20c/s0007
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures as cf
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

DROP = {"max_tokens", "max_completion_tokens", "stream", "stream_options", "temperature", "top_p", "top_k",
        "min_p", "presence_penalty", "frequency_penalty", "repetition_penalty", "seed"}


def _text(m) -> str:
    c = m.get("content")
    if isinstance(c, str):
        return c
    return " ".join(p.get("text", "") for p in c or [] if p.get("type") == "text")


def _kind(msgs) -> str:
    last = msgs[-1]
    if len(msgs) == 2:
        return "start"
    if last["role"] == "user" and isinstance(last["content"], list) and any(
            p.get("type") == "input_audio" for p in last["content"]):
        return "listen"
    t = _text(last)
    if t.startswith("<task"):
        return "task_result"
    if t.startswith("<skill_content"):
        return "after_skill"
    if re.search(r"^error:|Traceback", t, re.M):
        return "error"
    if " -> " in t:
        return "edit_applied"
    return "other"


def select(rows) -> list[dict]:
    """A spread of decision points per rollout: main at start / first listen / mid listen / each task result kind /
    an error / an applied edit / the last step; each subagent at its first step and its last listen."""
    out, seen = [], set()
    rows = [r for r in rows if r.get("kind") == "record"]
    by_route = {}
    for r in rows:
        by_route.setdefault(r["route"], []).append(r)
    main = by_route.get("main", [])
    listens = [r for r in main if _kind(r["body"]["messages"]) == "listen"]
    picks = [("main.start", next((r for r in main if _kind(r["body"]["messages"]) == "start"), None)),
             ("main.listen_first", listens[0] if listens else None),
             ("main.listen_mid", listens[len(listens) // 2] if len(listens) > 2 else None),
             ("main.last", main[-1] if main else None)]
    for r in main:
        k = _kind(r["body"]["messages"])
        if k == "task_result":
            sub = "judge" if '"tuple"' in _text(r["body"]["messages"][-1]) else (
                "transcriber" if "notes.json" in _text(r["body"]["messages"][-1]) else "searcher")
            picks.append((f"main.after_{sub}", r))
        elif k in ("after_skill", "error", "edit_applied"):
            picks.append((f"main.{k}", r))
    for route, rs in by_route.items():
        if route == "main" or route is None:
            continue
        role = route.split(".")[-1].rstrip("0123456789")
        sl = [r for r in rs if _kind(r["body"]["messages"]) == "listen"]
        picks += [(f"{role}.first", rs[0]), (f"{role}.last_listen", sl[-1] if sl else None)]
    for name, r in picks:
        if r is None or name in seen:
            continue
        seen.add(name)
        out.append({"point": name, "seq": r["seq"], "route": r["route"], "body": r["body"]})
    return out


def prepare(body: dict, media: Path, model: str, audio: bool) -> dict:
    b = {k: v for k, v in body.items() if k not in DROP}
    b["model"] = model
    msgs = []
    for m in body["messages"]:
        m = dict(m)
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "input_audio":
                    if not audio:
                        continue
                    sha = p["input_audio"]["data"]["sha256"]
                    data = base64.b64encode((media / f"{sha}.wav").read_bytes()).decode()
                    p = {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}
                parts.append(p)
            m["content"] = parts
        msgs.append(m)
    b["messages"] = msgs
    return b


def call(url: str, body: dict) -> dict:
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:  # no retries: an error is reported, not hidden
            resp = json.load(r)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"HTTP {e.code}: {e.read().decode()[:1000]}") from None
    msg = resp["choices"][0]["message"]
    return {"reasoning": msg.get("reasoning_content") or msg.get("reasoning") or "", "content": msg.get("content") or "",
            "tool_calls": msg.get("tool_calls") or [], "finish_reason": resp["choices"][0].get("finish_reason"),
            "usage": resp.get("usage"), "seconds": round(time.time() - t, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rollouts", nargs="+", help="replay dirs, e.g. outputs/replay/v5_pilot20c/s0003")
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--audio", action="store_true", help="inline audio (audio-capable models only)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for d in map(Path, a.rollouts):
        rows = [json.loads(l) for l in open(d / "io" / "requests.jsonl")]
        for p in select(rows):
            jobs.append((d.name, p, prepare(p["body"], d / "io" / "media", a.model, a.audio)))
    print(f"{len(jobs)} decision points")

    def run(job):
        sid, p, body = job
        dst = out / f"{sid}.{p['point']}.json"
        if dst.exists():
            return dst, "cached"
        try:
            r = call(a.url, body)
        except Exception as e:  # reported per point and not written, so a rerun retries it
            return dst, f"FAILED: {e}"
        rec = {"sid": sid, "point": p["point"], "seq": p["seq"], "route": p["route"], "model": a.model,
               "audio": a.audio, "n_messages": len(body["messages"]),
               "last_message": _text(p["body"]["messages"][-1])[:2000], **r}
        dst.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return dst, f"{len(r['reasoning'])} reasoning chars, {r['finish_reason']}, {r['seconds']} s"

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for dst, msg in ex.map(run, jobs):
            print(dst.name, msg, flush=True)


if __name__ == "__main__":
    main()
