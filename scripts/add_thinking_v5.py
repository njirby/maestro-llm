"""Thinking for every assistant turn of v5 rollouts: guessed by the 27B, kept only when it passes the checks.

Per turn (main and every subagent, from the replay's exact OpenCode requests):
  1. context = the request, with each heard clip replaced by its verified perception in <heard> (hearing_prompt_lab)
  2. guess: the model writes the thinking that leads to the record's turn (prompt g2, reasoning_effort medium)
  3. checks: the replay with only that thinking takes the record's action (tool + patch parts); no leak, no quoting of
     what was heard, no hearing claim before any audio, no exact values copied from the tool call, not empty
  4. up to --tries guesses; the first that passes is kept, else the turn is marked failed with every attempt kept

  .venv/bin/python scripts/add_thinking_v5.py --sids s0016 s0019 --out outputs/v5_rollouts/pilot20c/thinking
  -> <out>/<sid>.jsonl  one row per turn: {sid, route, step, kind, ok, thinking, tries, attempts: [...]}
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hearing_prompt_lab as h  # noqa: E402
import native_thinking_lab as nl  # noqa: E402

READ_ONLY = re.compile(r"^\s*(ls|cat|head|tail|find)\b|wavetable\.py\s+list\b")


def calls_subs(calls) -> set:
    """Tool names + patch parts; a task dispatch counts as its subagent type, not the words in its prompt."""
    out = set()
    for c in calls:
        if c["name"] == "task":
            try:
                out |= {"task", "task:" + json.loads(c["arguments"]).get("subagent_type", "")}
            except ValueError:
                out.add("task")
        else:
            out |= h.subs([c])
    return out


def read_only(c) -> bool:
    if c["name"] in ("read", "glob", "grep"):
        return True
    if c["name"] == "bash":
        try:
            return bool(READ_ONLY.search(json.loads(c["arguments"]).get("command", "")))
        except ValueError:
            return False
    return False


def same_action(got, want) -> bool:
    """The record's step, allowing extra read-only calls alongside it (a parallel read or wavetable list)."""
    if calls_subs(got) == calls_subs(want):
        return True
    return calls_subs(want) <= calls_subs(got) and calls_subs([c for c in got if not read_only(c)]) <= calls_subs(want)


def static_fails(th, heard, call_part) -> list[str]:
    fails = []
    if not th:
        fails.append("empty")
    for rx, name in ((h.LEAK, "leak"), (nl.META, "meta"), (h.QUOTE, "quotes hearing")):
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
    return fails


def attempt_native(a, sid, route, r, nxt, run):
    """One call: the model takes the record's turn natively with it given as its plan (B2); the thinking is kept.
    Then a cheap replay (thinking prefilled, plan removed) must reach the same action; a miss is replayed once more."""
    sub = route != "main"
    ctx = h.transform(r["body"], sid, h.PREAMBLES["sub" if sub else "v2"], a.model, [])
    turn, m = h.turn_text(r, nxt) if nxt is not None else last_turn(run, sid, route)
    oracle = [t["function"] for t in m.get("tool_calls") or []]
    msgs = json.loads(json.dumps(ctx["messages"]))
    msgs[0]["content"] += "\n\n" + nl.PROMPTS["B2_sys"]
    msgs.append({"role": "user", "content": nl.PROMPTS["B2_user"].format(turn=turn)})
    res = h.call(a.url, {**ctx, "messages": msgs, "chat_template_kwargs": {"reasoning_effort": a.effort}})
    th = res["reasoning"].strip()
    heard = " ".join(p["text"] for x in ctx["messages"] if isinstance(x.get("content"), list)
                     for p in x["content"] if p.get("text", "").startswith("<heard"))
    call_part = turn.split("[tool call]", 1)[-1] if "[tool call]" in turn else ""
    fails = static_fails(th, heard, call_part)
    if calls_subs(res["tool_calls"]) != calls_subs(oracle):
        fails.append(f"native action {sorted(calls_subs(res['tool_calls']))} != record {sorted(calls_subs(oracle))}")
    replays = []
    if not fails and not a.no_verify:
        body = dict(ctx)
        body.update(messages=ctx["messages"] + [{"role": "assistant", "content": f"<think>\n{th}\n</think>\n\n"}],
                    add_generation_prompt=False, continue_final_message=True)
        for _ in range(2):
            rr = h.call(a.url, body)
            calls = rr["tool_calls"] or h.parse_calls(rr["reasoning"] + rr["content"])
            replays.append([c["name"] for c in calls])
            if (not oracle and not calls) or (oracle and same_action(calls, oracle)):
                break
        else:
            fails.append(f"replay action {sorted(calls_subs(calls))} != record {sorted(calls_subs(oracle))}")
    return {"thinking": th, "content": res["content"], "fails": fails, "soft": False, "replays": replays, "turn": turn}


def last_turn(run: Path, sid: str, route: str) -> tuple[str, dict]:
    """The route's final assistant message from the records (no later request carries it)."""
    agent = route.split(".", 1)[1] if "." in route else "main"
    fname = "main" if agent == "main" else agent.rstrip("0123456789")
    rid = f"{sid}.{agent}"
    for line in open(run / f"{fname}.jsonl"):
        rec = json.loads(line)
        if rec["id"] == rid:
            m = [x for x in rec["messages"] if x["role"] == "assistant"][-1]
            return m["content"], {"role": "assistant", "content": m["content"]}
    raise KeyError(rid)


def attempt(a, sid, route, r, nxt, run):
    sub = route != "main"
    ctx = h.transform(r["body"], sid, h.PREAMBLES["sub" if sub else "v2"], a.model, [])
    if nxt is not None:
        turn, m = h.turn_text(r, nxt)
    else:
        turn, m = last_turn(run, sid, route)
    oracle = [t["function"] for t in m.get("tool_calls") or []]
    g = h.call(a.url, {**ctx, "messages": ctx["messages"] + [{"role": "user", "content": h.GUESSES["g2"].format(turn=turn)}],
                       "chat_template_kwargs": {"reasoning_effort": a.effort}})
    thinking = g["content"].strip()
    if a.no_verify:
        # keep the first guess that passes the text checks; no replay-verification call
        text, calls = "", None
    else:
        body = dict(ctx)
        body.update(messages=ctx["messages"] + [{"role": "assistant", "content": f"Think silently\n{thinking}\nThink silently\n\n"}],
                    add_generation_prompt=False, continue_final_message=True)
        res = h.call(a.url, body)
        text = res["reasoning"] + res["content"]
        calls = res["tool_calls"] or h.parse_calls(text)
    heard = " ".join(p["text"] for x in ctx["messages"] if isinstance(x.get("content"), list)
                     for p in x["content"] if p.get("text", "").startswith("<heard"))
    call_part = turn.split("[tool call]", 1)[-1] if "[tool call]" in turn else ""
    fails = []
    if not thinking:
        fails.append("empty")
    soft = False
    if calls is not None and h.subs(oracle) != h.subs(calls):
        # batch steps (several renders in one command) can't be inferred from thinking: a natural replay looks around first.
        # Those pass when the thinking plans the render step itself; the record says the action check was skipped.
        batch = {"render.py", "wavetable.py"} <= h.subs(oracle) or len(h.subs(oracle)) >= 5
        if batch and re.search(r"\brender\w*", thinking, re.I) and re.search(
                r"\b(probe|candidate|combination|combo|each|every|batch|all of them)\w*", thinking, re.I):
            soft = True
        else:
            fails.append(f"replay action {sorted(h.subs(calls))} != record {sorted(h.subs(oracle))}")
    if h.LEAK.search(thinking):
        fails.append(f"leak: {h.LEAK.search(thinking).group(0)}")
    quoted = [q for q in re.findall(r"[\"“]([^\"”]{3,60})[\"”]", thinking) if len(q.split()) >= 2 and q.lower() in heard.lower()]
    if quoted or h.QUOTE.search(thinking):
        fails.append(f"quotes hearing: {quoted or h.QUOTE.search(thinking).group(0)}")
    if not heard and re.search(r"\bI (just )?(heard|hear)\b|\bthe target (is|sounds|has)\b", thinking, re.I):
        fails.append("hearing claim before any audio")
    nums = sorted({x.group(0) for x in h.NUM_RE.finditer(thinking) if call_part and x.group(0) in call_part})
    if nums:
        fails.append(f"copies values: {nums[:5]}")
    return {"thinking": thinking, "fails": fails, "soft": soft,
            "replay": None if calls is None else {"text": text[:2000], "calls": calls}, "turn": turn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=h.RUN)
    ap.add_argument("--replay", type=Path, default=h.REPLAY, help="replay dir holding <sid>/io/requests.jsonl")
    ap.add_argument("--subagents-only", action="store_true", help="skip the main agent (e.g. narration still deferred)")
    ap.add_argument("--sids", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--tries", type=int, default=3)
    ap.add_argument("--no-verify", action="store_true",
                    help="keep the first guess passing the text checks; skip the replay-verification call")
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b-uncensored")
    ap.add_argument("--mode", choices=("guess", "native"), default="guess",
                    help="guess: thinking written for the known turn, then replayed; native: one call takes the turn (B2)")
    a = ap.parse_args()
    h.RUN, h.REPLAY = nl.RUN, nl.REPLAY = a.run, a.replay
    a.out.mkdir(parents=True, exist_ok=True)
    for sid in a.sids:
        rows = [json.loads(l) for l in open(h.REPLAY / sid / "io" / "requests.jsonl")]
        by = {}
        for r in rows:
            if r.get("kind") == "record":
                by.setdefault(r["route"], []).append(r)
        jobs = [(route, k, r, rs[k + 1] if k + 1 < len(rs) else None)
                for route, rs in by.items() for k, r in enumerate(rs) if not (a.subagents_only and route == "main")]
        dst = a.out / f"{sid}.jsonl"
        done = {(x["route"], x["step"]) for x in map(json.loads, open(dst))} if dst.exists() else set()
        jobs = [j for j in jobs if (j[0], j[1]) not in done]

        def one(job):
            route, k, r, nxt = job
            atts = []
            for _ in range(a.tries):
                try:
                    x = (attempt_native if a.mode == "native" else attempt)(a, sid, route, r, nxt, a.run)
                except Exception as e:  # recorded as a failed attempt; the turn is retried
                    x = {"thinking": "", "fails": [f"error: {str(e)[:200]}"]}
                atts.append(x)
                if not x["fails"]:
                    break
            ok = not atts[-1]["fails"]
            return {"sid": sid, "route": route, "step": k, "kind": h._kind(r["body"]["messages"]), "ok": ok,
                    "soft": ok and atts[-1].get("soft", False),
                    "thinking": atts[-1]["thinking"] if ok else None, "tries": len(atts), "attempts": atts}

        with cf.ThreadPoolExecutor(a.workers) as ex, open(dst, "a") as f:
            for rec in ex.map(one, jobs):
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"{sid} {rec['route']:22s} step {rec['step']:2d} {rec['kind']:18s} "
                      f"{'ok' if rec['ok'] else 'FAILED'} tries={rec['tries']}"
                      + ("" if rec["ok"] else f"  {rec['attempts'][-1]['fails']}"), flush=True)
        rs = [json.loads(l) for l in open(dst)]
        print(f"== {sid}: {sum(x['ok'] for x in rs)}/{len(rs)} turns have passing thinking; "
              f"first try {sum(x['ok'] and x['tries'] == 1 for x in rs)}", flush=True)


if __name__ == "__main__":
    main()
