"""Quick lab: let a text model act as if it hears, by putting each clip's verified listening notes where the audio was.

Takes the exact OpenCode requests from a replay (outputs/replay/<run>/<sid>/io/requests.jsonl), replaces every
input_audio part with a <heard> block built from that clip's session notes (narration_slots.jsonl), appends a
generation-only "Hearing" section to the system prompt, and records the model's thinking, text and next action next to
the record's own next action. No sampling params are sent.

  .venv/bin/python scripts/hearing_prompt_lab.py --sid s0003 --points 3 --samples 2 --tag v1 \
      [--flip "Brightness: much brighter.=Brightness: much darker."]
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures as cf
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RUN = ROOT / "outputs/v5_rollouts/pilot20c"
REPLAY = ROOT / "outputs/replay/v5_pilot20c"
OUT = ROOT / "outputs/thinking_capture/hearing_lab"
DROP = {"max_tokens", "max_completion_tokens", "stream", "stream_options", "temperature", "top_p", "top_k", "min_p",
        "presence_penalty", "frequency_penalty", "repetition_penalty", "seed"}

PREAMBLES = {
    "v1": """# Hearing
You can hear audio. When you `read` an audio file, your audio encoder turns the clip into a listening report that appears where the audio would be, inside <heard> tags. That report is your own perception of the clip: treat it exactly as having heard it. Think and talk about sounds in the first person, as things you heard ("I hear...", "it's brighter than the target..."), and never mention reports, tags, encoders or notes. Don't claim to hear anything the report doesn't contain.""",
    "v2": """# Hearing
You can hear. When you `read` an audio file, what you hear reaches you inside <heard> tags, already put into words by your auditory system. It is not a document or a report; it is your hearing. After reading the target you know how it sounds; after reading a render you know how it changed against the patch before your last edit, and how it compares with the target.
Think and talk about sound the way a producer does after listening: in your own words, in the first person ("it's brighter now", "the target swells much more than mine"). Never quote the wording inside <heard>, and never mention tags, reports, notes, labels, lists or how your hearing works. Don't claim to hear anything beyond what you heard.""",
    "sub": """# Hearing
You can hear. When you `read` an audio file, what you hear reaches you inside <heard> tags, already put into words by your auditory system. It is not a document or a report; it is your hearing. After reading the target you know how it sounds; after reading one of your renders you know how it compares with the target.
Think and talk about sound the way a producer does after listening: in your own words, in the first person ("this one's much buzzier than the target", "the 3rd note is a step high"). Never quote the wording inside <heard>, and never mention tags, reports, notes, labels, lists or how your hearing works. Don't claim to hear anything beyond what you heard.""",
}


def _notes(slot) -> list[str]:
    """Bullet lines of the slot's verified notes, minus the plan hint ("- Next: ..."), which is not something heard."""
    n = slot["notes"]
    if isinstance(n, str):
        try:
            n = ast.literal_eval(n)
        except ValueError:  # floats like nan in other fields; only the text is needed
            n = {"text": re.search(r"'text': (\"(?:[^\"\\\\]|\\\\.)*\"|'(?:[^'\\\\]|\\\\.)*')", n).group(1)}
            n["text"] = ast.literal_eval(n["text"])
    return [l for l in n["text"].splitlines() if l.startswith("- ") and not l.startswith("- Next:")]


def perceptions(sid: str) -> dict:
    """wav basename -> <heard> text. target.wav gets the first impression; the k-th render gets the k-th self_edit
    (vs your previous render) + target_vs_current (vs the target) notes."""
    slots = [json.loads(l) for l in open(RUN / "narration_slots.jsonl") if f'"{sid}/' in l]
    fi = next(s for s in slots if s["mode"] == "first_impression")
    se = [s for s in slots if s["mode"] == "self_edit"]
    tvc = [s for s in slots if s["mode"] == "target_vs_current"]
    return {"target.wav": "\n".join(_notes(fi)), "_renders": list(zip(se, tvc))}


def heard_block(name: str, perc: dict, render_ix: dict) -> str:
    if name == "target.wav":
        body = perc["target.wav"]
    elif name in render_ix and render_ix[name] < len(perc["_renders"]):
        se, tvc = perc["_renders"][render_ix[name]]
        body = ("Compared with the patch before your last edit:\n" + "\n".join(_notes(se)) + "\n\nCompared with the target:\n"
                + "\n".join(l for l in _notes(tvc) if not l.startswith("- Your last edit")))
    elif render_ix and render_ix[max(render_ix, key=render_ix.get)] < len(perc["_renders"]):
        # e.g. final.wav: a fresh render of the last patch, so it sounds like the last render against the target
        _, tvc = perc["_renders"][max(render_ix.values())]
        body = "Compared with the target:\n" + "\n".join(
            l for l in _notes(tvc) if not l.startswith(("- Your last edit", "- Did it get you closer")))
    else:
        body = "(nothing distinct)"
    return f"<heard>\n{body}\n</heard>"


PROBES = Path(os.environ.get("PROBES_DIR", OUT / "probe_perceptions"))  # per-run: sample ids repeat across runs


def _ord(k: int) -> str:
    return f"{k}{'th' if 10 <= k % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(k % 10, 'th')}"


def melody_diff(written: dict, sid: str) -> str:
    """What a transcriber hears comparing its render (its notes through the default patch) with the target: the melody
    against the ground-truth notes, in listener words. Exact, since both note lists are known."""
    from maestro import notation   # notes.json may be in seconds or in the skill's bars-and-beats form
    gt = notation.to_seconds((RUN / "oracle" / sid / "notes.json").read_text())[0]
    mine = notation.to_seconds(json.dumps(written))[0]
    key = lambda n: (round(n["start_s"], 3), n["pitch"])
    a, b = sorted(mine, key=key), sorted(gt, key=key)
    L = []
    # align the two sequences (edit distance on onset + pitch) so one missing or extra note doesn't shift every later pair
    cost = lambda x, y: min(abs(x["start_s"] - y["start_s"]) * 4, 2.0) + (0 if x["pitch"] == y["pitch"] else 0.5)
    n, m = len(a), len(b)
    D = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i
    for j in range(1, m + 1):
        D[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            D[i][j] = min(D[i - 1][j] + 1, D[i][j - 1] + 1, D[i - 1][j - 1] + cost(a[i - 1], b[j - 1]))
    pairs, extra, missing, i, j = [], [], [], n, m
    while i or j:
        if i and j and D[i][j] == D[i - 1][j - 1] + cost(a[i - 1], b[j - 1]):
            pairs.append((j, a[i - 1], b[j - 1])); i, j = i - 1, j - 1
        elif i and D[i][j] == D[i - 1][j] + 1:
            extra.append(a[i - 1]); i -= 1
        else:
            missing.append(j); j -= 1
    pairs.reverse()
    if missing:
        L.append(f"- Missing notes: yours leaves out the target's " + ", ".join(_ord(k) for k in sorted(missing)[:4])
                 + (f" and {len(missing) - 4} more" if len(missing) > 4 else "") + (" notes." if len(missing) > 1 else " note."))
    if extra:
        L.append(f"- Extra notes: yours adds {len(extra)} note{'s' if len(extra) > 1 else ''} the target doesn't play.")
    pitch, timing, length = [], [], []
    for i, x, y in pairs:
        if x["pitch"] != y["pitch"]:
            s = x["pitch"] - y["pitch"]
            amt = "an octave" if abs(s) == 12 else f"{abs(s)} semitone{'s' if abs(s) > 1 else ''}"
            pitch.append(f"the {_ord(i)} note is {amt} too {'high' if s > 0 else 'low'}")
        if abs(x["start_s"] - y["start_s"]) > 0.04:
            timing.append(f"the {_ord(i)} note comes in {'late' if x['start_s'] > y['start_s'] else 'early'}")
        r = x["dur_s"] / max(y["dur_s"], 1e-3)
        if r > 1.3 or r < 0.7:
            length.append(f"the {_ord(i)} note is held {'too long' if r > 1 else 'too short'}")
    for lab, xs in (("Pitch", pitch), ("Timing", timing), ("Note lengths", length)):
        if xs:
            more = f", and {len(xs) - 4} more like that" if len(xs) > 4 else ""
            L.append(f"- {lab}: " + "; ".join(xs[:4]) + more + ".")
        else:
            L.append(f"- {lab}: matches the target.")
    L.append("- Tone: the plain default patch, so it sounds nothing like the target's sound; only the notes are being compared.")
    return "Compared with the target:\n" + "\n".join(L)


def transform(body: dict, sid: str, preamble: str, model: str, flips: list[tuple[str, str]]) -> dict:
    perc = perceptions(sid)
    b = {k: v for k, v in body.items() if k not in DROP}
    b["model"] = model
    msgs, pending, render_ix, written = [], [], {}, None
    probes = json.loads((PROBES / f"{sid}.json").read_text()) if (PROBES / f"{sid}.json").exists() else {}
    wd = f"/work/rollouts/{sid}/"
    for m in body["messages"]:
        m = json.loads(json.dumps(m))
        if m["role"] == "system" and not msgs:
            m["content"] = m["content"] + "\n\n" + preamble
        for tc in m.get("tool_calls") or []:
            args = json.loads(tc["function"]["arguments"])
            if tc["function"]["name"] == "read":
                p = args.get("filePath", "")
                if p.endswith(".wav"):
                    pending.append(p[len(wd):] if p.startswith(wd) else Path(p).name)
            if tc["function"]["name"] == "write" and args.get("filePath", "").endswith("notes.json"):
                written = json.loads(args["content"])
        if isinstance(m.get("content"), list):
            parts = []
            for p in m["content"]:
                if p.get("type") == "input_audio":
                    name = pending.pop(0) if pending else "?"
                    if re.match(r"r\d+_", name) and name not in render_ix:
                        render_ix[name] = len(render_ix)
                    if name in probes:
                        txt = f"<heard>\nCompared with the target:\n{probes[name]}\n</heard>"
                    elif name.startswith("transcription") and written is not None:
                        txt = f"<heard>\n{melody_diff(written, sid)}\n</heard>"
                    else:
                        txt = heard_block(name, perc, render_ix)
                    for a, z in flips:
                        txt = txt.replace(a, z)
                    p = {"type": "text", "text": txt}
                parts.append(p)
            m["content"] = parts
        msgs.append(m)
    b["messages"] = msgs
    return b


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
            "tool_calls": [t["function"] for t in msg.get("tool_calls") or []], "seconds": round(time.time() - t, 1)}


def listen_points(sid: str, n: int) -> list[tuple[dict, dict | None]]:
    """The first n main-agent requests that end on a listen, each with the record's next main request (oracle)."""
    rows = [json.loads(l) for l in open(REPLAY / sid / "io" / "requests.jsonl")]
    main = [r for r in rows if r.get("route") == "main" and r.get("kind") == "record"]
    out = []
    for i, r in enumerate(main):
        last = r["body"]["messages"][-1]
        if last["role"] == "user" and isinstance(last["content"], list) and any(
                p.get("type") == "input_audio" for p in last["content"]):
            out.append((r, main[i + 1] if i + 1 < len(main) else None))
    return out[:n]


def oracle_next(r, nxt) -> str:
    if nxt is None:
        return "(end)"
    m = nxt["body"]["messages"][len(r["body"]["messages"])]
    calls = "; ".join(f"{t['function']['name']}({t['function']['arguments'][:200]})" for t in m.get("tool_calls") or [])
    return (m.get("content") or "")[:300] + (" | " if calls else "") + calls


LEAK = re.compile(r"\breport(ed)?\b(?! back)|\bthe listener|<heard|\btags?\b|encoder|\bverified\b|\bthe (next )?turn\b|instruction says|"
                  r"\bconstraint\b|next_turn|I was (shown|given|asked)|\bthis request\b", re.I)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="s0003")
    ap.add_argument("--points", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--only", type=int, help="run just this listen point")
    ap.add_argument("--tag", default="v1")
    ap.add_argument("--flip", action="append", default=[], help="OLD=NEW substitution inside <heard> blocks")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b-uncensored")
    a = ap.parse_args()
    flips = [tuple(f.split("=", 1)) for f in a.flip]
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    for k, (r, nxt) in enumerate(listen_points(a.sid, a.points)):
        if a.only is not None and k != a.only:
            continue
        b = transform(r["body"], a.sid, PREAMBLES[a.tag.split("_")[0]], a.model, flips)
        for s in range(a.samples):
            jobs.append((k, s, r, nxt, b))

    def run(job):
        k, s, r, nxt, b = job
        res = call(a.url, b)
        rec = {"sid": a.sid, "point": k, "sample": s, "flips": flips, "heard": b["messages"][-1]["content"][-1]["text"],
               "oracle_next": oracle_next(r, nxt), **res,
               "leaks": sorted({m.group(0).lower() for m in LEAK.finditer(res["reasoning"] + " " + res["content"])})}
        (out / f"{a.sid}.p{k}.s{s}.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    with cf.ThreadPoolExecutor(8) as ex:
        for rec in ex.map(run, jobs):
            acts = "; ".join(f"{t['name']}({t['arguments'][:160]})" for t in rec["tool_calls"])
            print(f"\n===== {rec['sid']} point {rec['point']} sample {rec['sample']}  leaks={rec['leaks']}")
            print("THINK:", rec["reasoning"].strip()[:1500])
            print("SAY:", rec["content"].strip()[:500])
            print("DO:", acts[:400])
            print("ORACLE:", rec["oracle_next"][:400])




# --------------------------------------------------------------------------- rationalize mode
GUESS = """Above is the session so far. Here is the turn you produced next, exactly:

<next_turn>
{turn}
</next_turn>

Write the private thinking you had immediately before producing that turn: the reasoning, in your own first-person thinking voice, that leads from what you have heard and seen so far to exactly that turn. Use only what was available to you before the turn; don't refer to the turn as something you were shown, and don't explain exact numbers you couldn't have worked out from what you heard. Output only the thinking."""


def turn_text(r, nxt) -> tuple[str, dict]:
    m = nxt["body"]["messages"][len(r["body"]["messages"])]
    calls = "\n".join(f"[tool call] {t['function']['name']}({t['function']['arguments']})" for t in m.get("tool_calls") or [])
    return ((m.get("content") or "").strip() + ("\n" if m.get("content") and calls else "") + calls), m


def parse_calls(text: str) -> list[dict]:
    """Tool calls from raw continuation text (Qwen XML format), since a prefilled turn bypasses the server's parsers."""
    out = []
    for fn, body in re.findall(r"<function=([\w-]+)>(.*?)</function>", text, re.S):
        out.append({"name": fn, "arguments": json.dumps(dict(re.findall(r"<parameter=(\w+)>\n?(.*?)\n?</parameter>", body, re.S)))})
    return out


SUB = re.compile(r"\b(env_\d|filter_\d|lfo_\d|osc_\d|random_\d|mod\.py|wavetable\.py|render\.py|volume|reverb|delay|distortion|eq_|chorus|compressor|phaser|flanger)")


def subs(calls) -> set:
    return {m.group(1) for c in calls for m in SUB.finditer(c["name"] + " " + c["arguments"])} | {c["name"] for c in calls}


def rationalize_main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sid", default="s0003")
    ap.add_argument("--points", type=int, default=3)
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b-uncensored")
    ap.add_argument("--effort", choices=["low", "medium", "xhigh"], help="reasoning_effort for the guessing call only")
    ap.add_argument("--sids", nargs="+", help="several rollouts (overrides --sid)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="rationalize")
    a = ap.parse_args(sys.argv[2:])
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)

    def one(job):
        sid, k, s, r, nxt = job
        dst = out / f"{sid}.p{k}.s{s}.json"
        if dst.exists():
            return json.loads(dst.read_text())
        ctx = transform(r["body"], sid, PREAMBLES["v2"], a.model, [])
        turn, m = turn_text(r, nxt)
        oracle = [t["function"] for t in m.get("tool_calls") or []]
        g = call(a.url, {**ctx, "messages": ctx["messages"] + [{"role": "user", "content": GUESS.format(turn=turn)}],
                         **({"chat_template_kwargs": {"reasoning_effort": a.effort}} if a.effort else {})})
        thinking = g["content"].strip()

        def replay(th):
            body = {k2: v for k2, v in ctx.items() if k2 not in ("tools", "tool_choice")}
            body.update(messages=ctx["messages"] + [{"role": "assistant", "content": f"<think>\n{th}\n</think>\n\n"}],
                        tools=ctx["tools"], add_generation_prompt=False, continue_final_message=True)
            res = call(a.url, body)
            text = res["reasoning"] + res["content"]
            return {"text": text, "calls": res["tool_calls"] or parse_calls(text)}

        with_t, without = replay(thinking), replay("")
        so = subs(oracle)
        rec = {"sid": sid, "point": k, "sample": s, "effort": a.effort or "xhigh(default)", "turn": turn, "guessed_thinking": thinking,
               "guess_model_reasoning": g["reasoning"], "replay_with": with_t, "replay_without": without,
               "oracle_subs": sorted(so), "with_subs": sorted(subs(with_t["calls"])),
               "without_subs": sorted(subs(without["calls"])),
               "leaks": sorted({x.group(0).lower() for x in LEAK.finditer(thinking)})}
        rec["match_with"] = bool(so) and so == set(rec["with_subs"])
        rec["match_without"] = bool(so) and so == set(rec["without_subs"])
        dst.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    jobs = [(sid, k, s, r, nxt) for sid in (a.sids or [a.sid])
            for k, (r, nxt) in enumerate(listen_points(sid, a.points)) if nxt is not None for s in range(a.samples)]

    def safe(job):
        try:
            return one(job)
        except Exception as e:
            return {"sid": job[0], "point": job[1], "error": str(e)[:300]}

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(safe, jobs):
            if "error" in rec:
                print(f"{rec['sid']} p{rec['point']} FAILED {rec['error']}", flush=True)
            else:
                print(f"{rec['sid']} p{rec['point']} s{rec['sample']} chars={len(rec['guessed_thinking'])} "
                      f"with={rec['match_with']} without={rec['match_without']} leaks={rec['leaks']}", flush=True)




# --------------------------------------------------------------------------- all turn kinds + extra checks
def _kind(msgs) -> str:
    last = msgs[-1]
    t = last["content"] if isinstance(last.get("content"), str) else " ".join(
        p.get("text", "") for p in last.get("content") or [] if p.get("type") == "text")
    if last["role"] == "user" and isinstance(last["content"], list) and any(p.get("type") == "input_audio" for p in last["content"]):
        return "listen"
    if len(msgs) == 2:
        return "start"
    if t.startswith("<skill_content"):
        return "after_skill"
    if t.startswith("<task"):
        return "after_judge" if '"tuple"' in t else ("after_transcriber" if "notes.json" in t else "after_searcher")
    if re.search(r"^error:|Traceback", t, re.M):
        return "error"
    if " -> " in t:
        return "edit_applied"
    if t.strip() == "Audio read successfully":
        return "mid_read"
    return "other"


def sub_points(sid: str) -> list[tuple[str, dict, dict]]:
    """Every turn of the transcriber, the first searcher and the judge, each with the record's next request."""
    rows = [json.loads(l) for l in open(REPLAY / sid / "io" / "requests.jsonl")]
    out = []
    for role in ("transcriber", "searcher1", "judge"):
        rs = [r for r in rows if r.get("route") == f"{sid}.{role}" and r.get("kind") == "record"]
        out += [(f"{role.rstrip('1')}.{i}.{_kind(r['body']['messages'])}", r, n) for i, (r, n) in enumerate(zip(rs, rs[1:]))]
    return out


def turn_points(sid: str, per_kind: int) -> list[tuple[str, dict, dict]]:
    """Main-agent requests spread across turn kinds (up to per_kind of each, evenly spaced), with the record's next."""
    rows = [json.loads(l) for l in open(REPLAY / sid / "io" / "requests.jsonl")]
    main = [r for r in rows if r.get("route") == "main" and r.get("kind") == "record"]
    by = {}
    for i, r in enumerate(main[:-1]):
        by.setdefault(_kind(r["body"]["messages"]), []).append((r, main[i + 1]))
    out = []
    for kind, xs in by.items():
        if kind in ("mid_read",):
            continue
        idx = sorted({round(j * (len(xs) - 1) / max(per_kind - 1, 1)) for j in range(min(per_kind, len(xs)))})
        out += [(kind, *xs[j]) for j in idx]
    return out


UNIT = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(ms|s|hz|khz|%|db|st)?\s*$", re.I)


def _num(s: str):
    m = UNIT.match(s.strip().strip("'\""))
    if not m:
        return None
    v, u = float(m.group(1)), (m.group(2) or "").lower()
    if u == "khz":
        v, u = v * 1000, "hz"
    if u == "s":
        v, u = v * 1000, "ms"
    return v, u


def record_changes(nxt_after) -> dict:
    """param -> (old, new) as (value, unit) from set.py output lines 'name: raw (display) -> raw (display)'."""
    out = {}
    for name, old, new in re.findall(r"^(\w+): (.+?) -> (.+)$", nxt_after, re.M):
        disp = lambda x: (re.search(r"\(([^)]*)\)", x).group(1) if "(" in x else x)
        o, n = _num(disp(old)), _num(disp(new))
        if o and n and o[1] == n[1]:
            out[name] = (o, n)
    return out


def direction_agreement(replay_calls, changes) -> tuple[int, int]:
    """(agree, compared) over params both the replay's set.py call and the record changed, judged against the old value."""
    agree = comp = 0
    for c in replay_calls:
        try:
            cmd = json.loads(c["arguments"]).get("command", "")
        except Exception:
            continue
        for name, val in re.findall(r"\b(\w+)=('[^']*'|\"[^\"]*\"|\S+)", cmd):
            if name not in changes:
                continue
            v = _num(val)
            (ov, ou), (nv, nu) = changes[name]
            if not v or v[1] != ou or nv == ov:
                continue
            comp += 1
            agree += (v[0] - ov) * (nv - ov) > 0
    return agree, comp


def slot_checks(sid: str, body: dict) -> dict:
    """The verified-notes check dicts for the most recent listen in this context (target FI, or a render's pair)."""
    slots = [json.loads(l) for l in open(RUN / "narration_slots.jsonl") if f'"{sid}/' in l]
    names = []
    for m in body["messages"]:
        for tc in m.get("tool_calls") or []:
            if tc["function"]["name"] == "read":
                p = json.loads(tc["function"]["arguments"]).get("filePath", "")
                if p.endswith(".wav"):
                    names.append(Path(p).name)
    renders = [n for n in dict.fromkeys(names) if re.match(r"r\d+_", n)]
    last = names[-1] if names else None

    def parsed(s):
        n = s["notes"]
        try:
            return ast.literal_eval(n) if isinstance(n, str) else n
        except ValueError:
            return None
    if last == "target.wav":
        fi = next(s for s in slots if s["mode"] == "first_impression")
        return {"fi": parsed(fi)}
    if last in renders:
        k = renders.index(last)
        se = [s for s in slots if s["mode"] == "self_edit"]
        tvc = [s for s in slots if s["mode"] == "target_vs_current"]
        return {"self_edit": parsed(se[k]) if k < len(se) else None, "tvc": parsed(tvc[k]) if k < len(tvc) else None}
    return {}


QUOTE = re.compile(r"(unchanged or too small to call|same as yours \(|in both: |Did it get you closer|a big, obvious change\.|"
                   r"the edit is clearly audible|\(target: [a-z]+: )", re.I)

GUESSES = {
    "g1": GUESS,
    "g2": """Above is the session so far. Here is the turn you produced next, exactly:

<next_turn>
{turn}
</next_turn>

Write the private thinking you had right before producing that turn, in your own natural thinking voice: what you just heard or saw, what it means, and why that leads to exactly this turn. Keep it about as long as your thinking naturally is at a step like this: usually a few sentences, longer only when the decision is hard. Use only what was available to you before the turn. When the turn sets many values, think about what you're going for and roughly which way to move things, the way you would before typing them; don't walk through the individual numbers. Output only the thinking itself, with nothing about this request.""",
}


def lab2_main():
    from maestro.sounddiff.check import check_delta, check_state
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids", nargs="+", required=True)
    ap.add_argument("--per-kind", type=int, default=2)
    ap.add_argument("--subagents", action="store_true", help="subagent turns instead of main-agent turns")
    ap.add_argument("--guess", default="g1", choices=list(GUESSES))
    ap.add_argument("--effort", default="medium")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b-uncensored")
    a = ap.parse_args(sys.argv[2:])
    out = OUT / a.tag
    out.mkdir(parents=True, exist_ok=True)

    def one(job):
        sid, kind, j, r, nxt = job
        dst = out / f"{sid}.{kind}.{j}.json"
        if dst.exists():
            return json.loads(dst.read_text())
        ctx = transform(r["body"], sid, PREAMBLES["sub" if a.subagents else "v2"], a.model, [])
        turn, m = turn_text(r, nxt)
        oracle = [t["function"] for t in m.get("tool_calls") or []]
        after = nxt["body"]["messages"][len(r["body"]["messages"]) + 1:]
        after_txt = "\n".join(x["content"] for x in after if isinstance(x.get("content"), str))
        g = call(a.url, {**ctx, "messages": ctx["messages"] + [{"role": "user", "content": GUESSES[a.guess].format(turn=turn)}],
                         "chat_template_kwargs": {"reasoning_effort": a.effort}})
        thinking = g["content"].strip()

        def replay(th):
            body = {k2: v for k2, v in ctx.items()}
            body.update(messages=ctx["messages"] + [{"role": "assistant", "content": f"<think>\n{th}\n</think>\n\n"}],
                        add_generation_prompt=False, continue_final_message=True)
            res = call(a.url, body)
            text = res["reasoning"] + res["content"]
            return {"text": text, "calls": res["tool_calls"] or parse_calls(text)}

        w, wo = replay(thinking), replay("")
        so = subs(oracle)
        chk = {} if a.subagents else slot_checks(sid, r["body"])
        claims = []
        if chk.get("fi"):
            claims += check_state(thinking, chk["fi"]["check"]["flags"]) if "check" in chk["fi"] else []
        for key in ("self_edit", "tvc"):
            if chk.get(key) and "check" in chk[key]:
                claims += [f"{key}: {p}" for p in check_delta(thinking, chk[key]["check"])]
        dag, dcomp = direction_agreement(w["calls"], record_changes(after_txt))
        heard_txt = " ".join(p["text"] for x in ctx["messages"] if isinstance(x.get("content"), list)
                             for p in x["content"] if p.get("text", "").startswith("<heard"))
        quoted = [q for q in re.findall(r"[\"“]([^\"”]{3,60})[\"”]", thinking) if q.lower() in heard_txt.lower()]
        early = (not heard_txt) and bool(HEARD_CLAIM.search(thinking))
        nums = sorted({x.group(0) for x in NUM_RE.finditer(thinking) if x.group(0) in turn.split("[tool call]", 1)[-1]}) \
            if "[tool call]" in turn else []
        rec = {"sid": sid, "kind": kind, "j": j, "guess": a.guess, "effort": a.effort, "turn": turn,
               "guessed_thinking": thinking, "replay_with": w, "replay_without": wo,
               "oracle_subs": sorted(so), "with_subs": sorted(subs(w["calls"])), "without_subs": sorted(subs(wo["calls"])),
               "match_with": so == subs(w["calls"]), "match_without": so == subs(wo["calls"]),
               "direction": [dag, dcomp], "claim_problems": claims,
               "leaks": sorted({x.group(0).lower() for x in LEAK.finditer(thinking)}),
               "quotes": sorted({x.group(0) for x in QUOTE.finditer(thinking)} | set(quoted)),
               "claims_hearing_before_any_audio": early, "copied_numbers": nums}
        dst.write_text(json.dumps(rec, indent=1, ensure_ascii=False))
        return rec

    pts = (lambda s: sub_points(s)) if a.subagents else (lambda s: turn_points(s, a.per_kind))
    jobs = [(sid, kind, j, r, nxt) for sid in a.sids for j, (kind, r, nxt) in enumerate(pts(sid))]
    print(len(jobs), "turns", flush=True)

    def safe(job):
        try:
            return one(job)
        except Exception as e:
            return {"sid": job[0], "kind": job[1], "error": str(e)[:300]}

    with cf.ThreadPoolExecutor(a.workers) as ex:
        for rec in ex.map(safe, jobs):
            if "error" in rec:
                print(rec["sid"], rec["kind"], "FAILED", rec["error"], flush=True)
            else:
                print(f"{rec['sid']} {rec['kind']:18s} chars={len(rec['guessed_thinking']):5d} match={rec['match_with']!s:5s} "
                      f"ctrl={rec['match_without']!s:5s} dir={rec['direction']} claims={len(rec['claim_problems'])} "
                      f"leaks={rec['leaks']} quotes={rec['quotes']} early={rec['claims_hearing_before_any_audio']} nums={len(rec['copied_numbers'])}", flush=True)


HEARD_CLAIM = re.compile(r"\bI (just )?(heard|hear|can hear)\b|in my head|\bthe target (is|sounds|has|reads)\b|sounds like", re.I)
NUM_RE = re.compile(r"\d+(\.\d+)?\s?(ms|hz|khz|%|db|st|s\b)|\b\d+\.\d+\b", re.I)


if __name__ == "__main__":
    {"rationalize": rationalize_main, "lab2": lab2_main}.get(sys.argv[1] if len(sys.argv) > 1 else "", main)()
