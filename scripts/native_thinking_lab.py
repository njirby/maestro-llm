"""One-call thinking: the model takes the record's turn natively (reasoning_content + content + tool_calls) with the
turn given in the prompt, and only the thinking is kept. Compares where the turn goes:

  A  system: the hearing preamble plus "your next turn is ..." in the system prompt; the conversation ends as recorded
  B  system: the preamble plus a fixed instruction; the turn itself in a final user message

  .venv/bin/python scripts/native_thinking_lab.py --placements A B --n 30 --tag it1
  -> outputs/thinking_capture/hearing_lab/native/<tag>/<placement>.jsonl and a side-by-side page view/native_<tag>.html
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import html
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hearing_prompt_lab as h  # noqa: E402

ROOT = h.ROOT
RUN = ROOT / "outputs/v5_rollouts/pilot20d"
REPLAY = ROOT / "outputs/replay/v5_pilot20d"
TWO_CALL = RUN / "thinking"  # rows from the stopped two-call run, for side-by-side
OUT = h.OUT / "native"
META = re.compile(r"\b(already (been )?(decided|chosen)|decided in advance|predetermined|system prompt|instructions?\b|instructed|"
                  r"I'm (told|supposed)|told to|supposed to|the plan (says|line)|given turn|exactly (that|this) turn|as specified|"
                  r"settled|hand(ed|ing) (me|over)|the user (is )?(telling|indicating|passing|passed|presenting|gave|giving|wants me to take)|"
                  r"as[- ]is|execute (it|that|this)|<plan|<next|the given)\b", re.I)

PROMPTS = {
    "A": """# Your next turn
Your next turn in this session is already settled. It is:

<next>
{turn}
</next>

Take that turn now: the same visible text (if any) and the same tool calls with the same arguments. Before it, think the way you naturally would at this point, reaching that decision from what you've heard and seen so far. Your thinking is your own reasoning toward the decision; it never mentions that the turn was settled in advance or anything in this section.""",
    "B_sys": """# Taking a settled turn
Sometimes the user hands you the turn you are about to take. When that happens, take exactly that turn: the same visible text (if any) and the same tool calls with the same arguments. Before it, think the way you naturally would at that point, reaching the decision from what you've heard and seen so far. Your thinking is your own reasoning toward the decision; it never mentions that the turn was handed to you.""",
    "B_user": """<next>
{turn}
</next>""",
}

# placement -> (system addition, final user message or None)
PROMPTS["B2_sys"] = """# Your plan
After you have looked at the latest result, the last message in the session holds the move you've chosen to make next, in <plan> tags. Make exactly that move: the same visible text (if any) and the same tool calls with the same arguments.
Your thinking before it is how you arrive at that move yourself, as if the <plan> weren't there: what you just heard or saw, what it tells you, and why this is the right next step. When you've just listened to something, think through what you heard against what you're after before deciding. Simple steps get a sentence or two; decisions after listening get a short paragraph. Your thinking never mentions the plan, a message about it, or anything in this section."""
PROMPTS["B2_user"] = """<plan>
{turn}
</plan>"""
PROMPTS["A2"] = """# Your next move
You've already worked out the move you make next:

<plan>
{turn}
</plan>

Make exactly that move now: the same visible text (if any) and the same tool calls with the same arguments, and nothing else.
Your thinking before it is how you arrive at that move yourself, as if the <plan> weren't there: what you just heard or saw, what it tells you, and why this is the right next step. When you've just listened to something, think through what you heard against what you're after before deciding. Simple steps get a sentence or two; decisions after listening get a short paragraph. Your thinking never mentions the plan or anything in this section."""
LAYOUT = {"A": ("A", None), "B": ("B_sys", "B_user"), "A2": ("A2", None), "B2": ("B2_sys", "B2_user")}


def sample(sids, n, seed) -> list[tuple[str, str, int, dict, dict]]:
    """Turns spread over kinds and agents: (sid, route, step, request, next request)."""
    pool = {}
    for sid in sids:
        rows = [json.loads(l) for l in open(REPLAY / sid / "io" / "requests.jsonl")]
        by = {}
        for r in rows:
            if r.get("kind") == "record":
                by.setdefault(r["route"], []).append(r)
        for route, rs in by.items():
            agent = "main" if route == "main" else route.split(".", 1)[1].rstrip("0123456789")
            for k, (r, nx) in enumerate(zip(rs, rs[1:])):
                kind = h._kind(r["body"]["messages"])
                if kind == "mid_read":
                    continue
                pool.setdefault((agent, kind), []).append((sid, route, k, r, nx))
    rnd = random.Random(seed)
    keys = sorted(pool)
    out = []
    while len(out) < n and any(pool.values()):
        for key in keys:
            if pool[key] and len(out) < n:
                out.append(pool[key].pop(rnd.randrange(len(pool[key]))))
    return out


def norm(calls) -> list[tuple[str, object]]:
    out = []
    for c in calls:
        try:
            a = json.loads(c["arguments"])
        except (ValueError, TypeError):
            a = c["arguments"]
        out.append((c["name"], a))
    return out


def one(a, placement, sid, route, k, r, nx):
    sub = route != "main"
    h.RUN, h.REPLAY = RUN, REPLAY
    ctx = h.transform(r["body"], sid, h.PREAMBLES["sub" if sub else "v2"], a.model, [])
    turn, m = h.turn_text(r, nx)
    msgs = json.loads(json.dumps(ctx["messages"]))
    sys_key, user_key = LAYOUT[placement]
    msgs[0]["content"] += "\n\n" + PROMPTS[sys_key].format(turn=turn)
    if user_key:
        msgs.append({"role": "user", "content": PROMPTS[user_key].format(turn=turn)})
    body = {**ctx, "messages": msgs, "chat_template_kwargs": {"reasoning_effort": a.effort}}
    t = time.time()
    try:
        res = h.call(a.url, body)
    except Exception as e:  # recorded as a failed row
        return {"sid": sid, "route": route, "step": k, "placement": placement, "error": str(e)[:300]}
    oracle = [t_["function"] for t_ in m.get("tool_calls") or []]
    th = res["reasoning"].strip()
    heard = " ".join(p["text"] for x in ctx["messages"] if isinstance(x.get("content"), list)
                     for p in x["content"] if p.get("text", "").startswith("<heard"))
    call_part = turn.split("[tool call]", 1)[-1] if "[tool call]" in turn else ""
    fails = []
    if not th:
        fails.append("empty")
    exact = norm(res["tool_calls"]) == norm(oracle)
    if h.subs(res["tool_calls"]) != h.subs(oracle):
        fails.append(f"action {sorted(h.subs(res['tool_calls']))} != record {sorted(h.subs(oracle))}")
    for rx, name in ((h.LEAK, "leak"), (META, "meta"), (h.QUOTE, "quotes hearing")):
        if rx.search(th):
            fails.append(f"{name}: {rx.search(th).group(0)}")
    quoted = [q for q in re.findall(r"[\"“]([^\"”]{3,60})[\"”]", th) if len(q.split()) >= 2 and q.lower() in heard.lower()]
    if quoted:
        fails.append(f"quotes hearing: {quoted[0]}")
    if not heard and re.search(r"\bI (just )?(heard|hear)\b|\bthe target (is|sounds|has)\b", th, re.I):
        fails.append("hearing claim before any audio")
    nums = sorted({x.group(0) for x in h.NUM_RE.finditer(th) if call_part and x.group(0) in call_part})
    if nums:
        fails.append(f"copies values: {nums[:5]}")
    return {"sid": sid, "route": route, "step": k, "kind": h._kind(r["body"]["messages"]), "placement": placement,
            "thinking": th, "content": res["content"], "calls": res["tool_calls"], "exact_args": exact,
            "turn": turn, "fails": fails, "seconds": round(time.time() - t, 1)}


def page(tag, placements, rows):
    two = {}
    for f in TWO_CALL.glob("*.jsonl"):
        for l in open(f):
            x = json.loads(l)
            two[(x["sid"], x["route"], x["step"])] = x
    e = html.escape
    by = {}
    for r in rows:
        by.setdefault((r["sid"], r["route"], r["step"]), {})[r["placement"]] = r
    cols = list(placements) + ["two-call"]
    stats = []
    for p in placements:
        rs = [r for r in rows if r["placement"] == p and "error" not in r]
        stats.append(f"<tr><td>{p}</td><td>{len(rs)}</td><td>{sum(not r['fails'] for r in rs)}</td>"
                     f"<td>{sum(r['exact_args'] for r in rs)}</td>"
                     f"<td>{sum(any(f.startswith(('leak', 'meta')) for f in r['fails']) for r in rs)}</td>"
                     f"<td>{int(statistics.median(len(r['thinking']) for r in rs)) if rs else 0}</td>"
                     f"<td>{statistics.median(r['seconds'] for r in rs) if rs else 0}</td></tr>")
    css = """:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--ok:#2e8b57;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#34332f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1500px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}
.cols{display:grid;grid-template-columns:repeat(%d,minmax(0,1fr));gap:10px}@media(max-width:900px){.cols{grid-template-columns:1fr}}
pre{white-space:pre-wrap;font-size:12.5px;margin:4px 0;max-height:380px;overflow:auto;background:var(--bg);padding:6px;border-radius:6px;border:1px solid var(--line)}
.mut{color:var(--mut);font-size:12.5px}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}""" % len(cols)
    body = [f"<h1>One-call thinking ({e(tag)})</h1><p class=mut>A: the turn in the system prompt. B: a fixed system "
            "instruction, the turn in a final user message. two-call: the stopped run's guess-then-replay thinking.</p>"
            "<table><tr><th>placement</th><th>turns</th><th>pass all checks</th><th>exact tool args</th>"
            "<th>leak/meta</th><th>median chars</th><th>median s</th></tr>" + "".join(stats) + "</table>"]
    for key, d in sorted(by.items()):
        anyr = next(iter(d.values()))
        body.append(f"<div class=card><b>{e(key[0])} {e(key[1])} step {key[2]}</b> <span class=mut>{e(anyr.get('kind', ''))}"
                    f"</span><details><summary class=mut>record's turn</summary><pre>{e(anyr.get('turn', ''))}</pre></details>"
                    "<div class=cols>")
        for c in cols:
            if c == "two-call":
                x = two.get(key)
                txt = (x or {}).get("thinking") or ""
                st = "<span class=mut>not run</span>" if not x else (
                    "<span class=ok>ok</span>" if x["ok"] else "<span class=bad>failed</span>")
            else:
                x = d.get(c)
                if not x or "error" in x:
                    body.append(f"<div><b>{c}</b><p class=bad>{e((x or {}).get('error', 'missing'))}</p></div>")
                    continue
                txt = x["thinking"]
                st = ("<span class=ok>ok</span>" if not x["fails"] else f"<span class=bad>{e('; '.join(x['fails']))}</span>") + (
                    "" if x["exact_args"] else " <span class=mut>(args differ)</span>")
            body.append(f"<div><b>{c}</b> {st} <span class=mut>({len(txt)} chars)</span><pre>{e(txt) or '(none)'}</pre></div>")
        body.append("</div></div>")
    dst = h.OUT / "view" / f"native_{tag}.html"
    dst.write_text(f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,"
                   f"initial-scale=1'><title>One-call thinking</title><style>{css}</style></head><body>{''.join(body)}"
                   "</body></html>")
    return dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placements", nargs="+", default=["A", "B"])
    ap.add_argument("--sids", nargs="+", default=["s0000", "s0001", "s0002", "s0003", "s0004", "s0005"])
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b-uncensored")
    a = ap.parse_args()
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    pts = sample(a.sids, a.n, a.seed)
    jobs = [(p, *x) for p in a.placements for x in pts]
    t0 = time.time()
    with cf.ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(lambda j: one(a, *j), jobs))
    for p in a.placements:
        (out / f"{p}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows if r["placement"] == p))
        rs = [r for r in rows if r["placement"] == p and "error" not in r]
        fails = {}
        for r in rs:
            for f in r["fails"]:
                fails[f.split(":")[0].split(" [")[0]] = fails.get(f.split(":")[0].split(" [")[0], 0) + 1
        print(f"{p}: {sum(not r['fails'] for r in rs)}/{len(rs)} pass, exact args {sum(r['exact_args'] for r in rs)}, "
              f"median {int(statistics.median(len(r['thinking']) for r in rs)) if rs else 0} chars, "
              f"errors {sum('error' in r for r in rows if r['placement'] == p)}, fails {fails}")
    print(f"wall {time.time() - t0:.0f}s ->", page(a.tag, a.placements, rows))


if __name__ == "__main__":
    main()
