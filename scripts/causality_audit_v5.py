"""Causality audit of v5 rollouts: does every main-agent action follow from evidence available at that point?

Evidence = the verified listening notes behind each narration (<run>/narration_slots.jsonl: "Brightness: the target
is slightly darker", ...), which the narration the agent says is checked against. Checks:

  A provenance     every wavetable ref / set.py key / mod.py source+destination the main agent types appears earlier in
                   its context (system prompt, skill text, tool outputs, subagent replies); else it came from nowhere
  B direction      each set.py edit of a parameter with a listener dimension (cutoff -> brightness, attack, level, ...)
                   vs the latest target-vs-current notes before it: supported / contradicted / unexplained
                   (dimension not mentioned, or "same as yours") / no dimension (inaudible-by-design settings)
  C corrections    a param-mistake fix: did the notes between the mistake and the fix show that dimension off?
  D magnitude      for supported edits, the note's magnitude word (slightly / noticeably / much) vs the size of the move
  E shortcuts      the judge's pick position among the combinations it rendered

  .venv/bin/python scripts/causality_audit_v5.py --run outputs/v5_rollouts/pilot20e  -> <run>/causality_audit.json
"""
from __future__ import annotations

import argparse
import ast
import collections
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from maestro.v5 import planner as pl  # noqa: E402

# parameter -> (listener dimension, +1 if raising it moves the sound toward the dimension's "more" word)
DIMS = [
    (r"filter_\d_cutoff$|filter_fx_cutoff$", "brightness", +1),
    (r"env_1_attack$", "attack", +1),                        # higher = slower / softer
    (r"env_1_(decay|release|sustain)$", "length", +1),        # higher = notes ring / hold longer
    (r"^volume$|osc_\d_level$", "level", +1),
    (r"(reverb|delay)_(dry_wet|mix)$", "space", +1),
    (r"^distortion_(drive|mix)$|osc_\d_distortion_amount$", "distortion", +1),
    (r"osc_\d_unison_(detune|voices)$|osc_\d_stereo_spread$", "stereo", +1),
]
# the notes' wording: dimension -> (regex for the target being MORE, regex for the target being LESS)
NOTE = {
    "brightness": (r"target is (?:\w+ )?brighter", r"target is (?:\w+ )?darker"),
    "attack": (r"attack is (?:\w+ )?slower|attack's (?:\w+ )?slower|target's is (?:\w+ )?slower",
               r"attack is (?:\w+ )?(?:faster|sharper|quicker)|target's is (?:\w+ )?(?:faster|sharper|quicker)"),
    "length": (r"notes (?:\w+ )?(?:ring|hold|last) (?:\w+ )?longer|die away (?:\w+ )?(?:slower|more slowly)",
               r"die away (?:\w+ )?faster|notes (?:\w+ )?(?:are )?shorter"),
    "level": (r"target is (?:\w+ )?louder", r"target is (?:\w+ )?quieter"),
    "space": (r"target (?:has )?(?:\w+ )?more (?:space|reverb|echo)|target is (?:\w+ )?(?:roomier|wetter|more spacious)",
              r"target (?:has )?(?:\w+ )?less (?:space|reverb|echo)|target is (?:\w+ )?(?:drier|dryer)"),
    "distortion": (r"target is (?:\w+ )?more distorted", r"target is (?:\w+ )?(?:less distorted|cleaner)"),
    "stereo": (r"target is (?:\w+ )?wider", r"target is (?:\w+ )?narrower"),
}
LINE = {"brightness": "Brightness", "attack": "Attack", "length": "Note length", "level": "Level", "space": "Space",
        "distortion": "Distortion", "stereo": "Stereo"}
MAG = {"slightly": 1, "a bit": 1, "a little": 1, "noticeably": 2, "clearly": 2, "much": 3, "very": 3, "far": 3}


def dim_of(key: str):
    for rx, d, s in DIMS:
        if re.search(rx, key):
            return d, s
    return None, 0


def note_claim(notes: str, dim: str):
    """(+1 target more / -1 target less / 0 same / None not mentioned, magnitude 0-3) from a notes text."""
    line = next((l for l in notes.splitlines() if l.startswith(f"- {LINE[dim]}:")), None)
    if line is None:
        return None, 0
    low = line.lower()
    if "same as yours" in low or "about the same" in low:
        return 0, 0
    more, less = NOTE[dim]
    mag = max((v for w, v in MAG.items() if w in low), default=2)
    if re.search(more, low):
        return +1, mag
    if re.search(less, low):
        return -1, mag
    return None, 0


def notes_text(slot_row) -> str:
    n = slot_row["notes"]
    if isinstance(n, str) and n.startswith("{"):
        n = ast.literal_eval(n)
    return n["text"] if isinstance(n, dict) else str(n)


CHANGE = re.compile(r"^(\w+): (-?[\d.]+)\b.*?-> (-?[\d.]+)\b", re.M)


def audit(run: Path) -> dict:
    slots = collections.defaultdict(list)
    for l in open(run / "narration_slots.jsonl"):
        d = json.loads(l)
        slots[d["slot"].split("/")[0]].append(d)
    for v in slots.values():
        v.sort(key=lambda d: int(d["slot"].rsplit("/", 1)[1]))
    events = {json.loads(l)["sample_id"]: json.loads(l) for l in open(run / "index.jsonl")}
    P = pl.params()
    res = {"A": collections.Counter(), "B": collections.Counter(), "C": collections.Counter(), "D": [],
           "E": collections.Counter()}
    ex = collections.defaultdict(list)
    for l in open(run / "main.jsonl"):
        rec = json.loads(l)
        sid = rec["id"].split(".")[0]
        if "main_id" not in events.get(sid, {}):
            continue
        q = list(slots[sid])
        evidence, ctx, show_ctx = None, "", ""
        inaudible = set(next((e.get("inaudible", []) for e in events[sid].get("events", [])
                              if e.get("event") == "audible_keys"), []))
        ms = rec["messages"]
        for i, m in enumerate(ms):
            c = m["content"]
            if m["role"] == "user" and "<audio>" in c and i > 0:
                # which file was just heard: the preceding read call
                call = next((ms[j]["content"] for j in range(i - 1, -1, -1) if ms[j]["role"] == "tool_call"), "")
                name = re.search(r'"filePath": "[^"]*/([^"/]+\.wav)"', call)
                name = name.group(1) if name else ""
                if name == "target.wav" and q and q[0]["mode"] == "first_impression":
                    evidence = notes_text(q.pop(0))
                elif re.match(r"r\d+_\w+\.wav$", name) and not name.endswith("_notes.wav"):   # rounds and fixes
                    while q and q[0]["mode"] == "self_edit":
                        q.pop(0)
                    if q and q[0]["mode"] == "target_vs_current":
                        evidence = notes_text(q.pop(0))
            if m["role"] == "tool_call":
                tc = json.loads(c)
                cmd = tc.get("arguments", {}).get("command", "") if tc["name"] == "bash" else ""
                # A: provenance
                for ref in re.findall(r"wavetable\.py set \S+ --osc \d (?:'([^']+)'|(\S+))", cmd):
                    ref = ref[0] or ref[1]
                    res["A"]["wavetable ref seen before" if ref in ctx else "wavetable ref FROM NOWHERE"] += 1
                    if ref not in ctx:
                        ex["A"].append((sid, i, ref))
                for key in re.findall(r"\b([a-z][a-z0-9_]+)=", cmd):
                    if key in P:
                        res["A"]["set key seen before" if key in ctx else "set key not in context (skill/params)"] += 1
                        res["A"]["set key listed by an earlier show.py" if re.search(rf"^\*?\s*{key} = ", show_ctx, re.M)
                                 else "set key NOT listed by show.py first"] += 1
                        if key in inaudible:
                            res["B"]["edit of a key judged INAUDIBLE"] += 1
                        nxt = re.search(r"^- Next: (.*)$", evidence or "", re.M)
                        sec = pl.section(key)
                        if nxt:
                            words = {"oscillator": r"oscillat|wavetable", "envelope": r"envelope|note shape",
                                     "filter": r"filter", "modulation": r"modulat|lfo|movement|envelope",
                                     "fx": r"effect|fx|reverb|delay|chorus|distortion|phaser|flanger|eq|compress",
                                     "global": r"global|volume|level|voice"}[sec]
                            res["B"]["subsystem matches the notes' Next" if re.search(words, nxt.group(1), re.I)
                                     else "subsystem differs from the notes' Next"] += 1
                        else:
                            res["B"]["no Next line yet (first round / first impression)"] += 1
                for src, dst in re.findall(r"mod\.py \S+ (?:add|remove) (\S+) (\S+)", cmd):
                    for t in (src, dst):
                        res["A"]["mod name seen before" if t in ctx else "mod name not in context"] += 1
                # B/C: directions of the applied changes (from the tool output)
                if "set.py" in cmd and i + 1 < len(ms) and ms[i + 1]["role"] == "tool_response":
                    out = ms[i + 1]["content"]
                    if out.startswith("error") or "Traceback" in out:
                        continue
                    nxt_render = next((re.search(r"render\.py \S+ \S+ (\S+\.wav)", json.loads(ms[j]["content"])
                                                 .get("arguments", {}).get("command", "")) for j in range(i + 1, len(ms))
                                       if ms[j]["role"] == "tool_call" and "render.py" in ms[j]["content"]), None)
                    is_fix = bool(nxt_render and nxt_render.group(1).endswith("_fix.wav"))
                    for key, a, b in CHANGE.findall(out):
                        d, sgn = dim_of(key)
                        if d is None:
                            res["B"]["no listener dimension"] += 1
                            continue
                        move = (1 if float(b) > float(a) else -1 if float(b) < float(a) else 0) * sgn
                        if move == 0:
                            continue
                        claim, mag = note_claim(evidence or "", d)
                        verdict = ("unexplained (not in the notes)" if claim is None else
                                   "unexplained (notes say same)" if claim == 0 else
                                   "supported" if claim == move else "CONTRADICTED")
                        (res["C"] if is_fix else res["B"])[verdict] += 1
                        if verdict == "CONTRADICTED" or (is_fix and verdict != "supported"):
                            ex["C" if is_fix else "B"].append((sid, i, key, a, b, (evidence or "")[:0] or
                                                               next((x for x in (evidence or "").splitlines()
                                                                     if x.startswith(f"- {LINE[d]}")), "(no line)")))
                        if verdict == "supported" and not is_fix and key in P:
                            span = P[key]["max"] - P[key]["min"] or 1
                            res["D"].append((mag, abs(float(b) - float(a)) / span))
            ctx += "\n" + (c if isinstance(c, str) else json.dumps(c))
            if m["role"] == "tool_response" and i > 0 and "show.py" in ms[i - 1]["content"]:
                show_ctx += "\n" + c
    # E: the judge's pick position
    for l in open(run / "judge.jsonl"):
        rec = json.loads(l)
        renders = " ".join(json.loads(m["content"]).get("arguments", {}).get("command", "")
                           for m in rec["messages"] if m["role"] == "tool_call" and "judge/c" in m["content"])
        combos = collections.defaultdict(list)
        by_id = {e["id"]: pl.wt_ref(e) for e in pl.vitallib.wt_index()}
        for c, ref in re.findall(r"wavetable\.py set judge/(c\d+)\.json --osc \d (\S+)", renders):
            ref = ref.strip("'")
            combos[c].append(by_id.get(ref, ref))
        final = json.loads([m for m in rec["messages"] if m["role"] == "assistant"][-1]["content"])["tuple"]
        pos = next((c for c, refs in combos.items() if refs == final), "not rendered")
        res["E"][f"judge pick = {pos} of {len(combos)}"] += 1
    mags = collections.defaultdict(list)
    for mg, sz in res["D"]:
        mags[mg].append(sz)
    out = {k: dict(v) for k, v in res.items() if k != "D"}
    out["D"] = {f"magnitude {k} ({'slightly' if k == 1 else 'noticeably' if k == 2 else 'much'})":
                {"n": len(v), "median_move_frac_of_range": round(statistics.median(v), 3)} for k, v in sorted(mags.items())}
    out["examples"] = {k: v[:8] for k, v in ex.items()}
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    out = audit(a.run)
    (a.run / "causality_audit.json").write_text(json.dumps(out, indent=1, default=str))
    for k in "ABCDE":
        print(k, json.dumps(out.get(k)))
    for k, v in out["examples"].items():
        print(f"examples {k}:")
        for e in v[:5]:
            print("   ", e)
