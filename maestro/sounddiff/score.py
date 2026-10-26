"""Narration filter: reject a narration that claims a perceptual dimension its facts do not support, contradicts a
measured direction, says "no difference" on a clearly audible change (or describes changes on an inaudible one), or
quotes parameter numbers / parameter names. narrate.py retries once with the problem named in the prompt, then falls
back to a template. Keyword-based (with a short negation guard), so it is strict on vocabulary, lenient on negations.

    from maestro.sounddiff.score import check
    r = check(facts, text)   # {"ok": bool, "problems": [...], "claims": {dim: dir}, "unsupported": [...]}

  .venv/bin/python -m maestro.sounddiff.score outputs/sounddiff/probe1    # stats over narr_*.jsonl
"""
from __future__ import annotations

import collections
import glob
import json
import re
import sys
from pathlib import Path

from .facts import TONE

MOVE = {"tonal movement", "volume movement"}
GRIT = "grit / harmonic richness"
# (pattern, dims any of which supports the word)
LEX = [
    (r"bright|dark|top[- ]?end|\bhighs\b|treble|dull|muffl|sizzl|sparkl|\bopen(s|ed)? up|closed[- ]off|\bshiny|"
     r"\bhigh end|low[- ]?pass|filtered", {"brightness"}),
    (r"\bwarm", {"brightness", TONE}),
    (r"\bair(y|ier)?\b", {"brightness", "noisiness"}),
    (r"grit|saturat|crunch|\bbite\b|biting|\bedg(e|y)|distort|dirt|fuzz|\bbuzz|rasp|aggressive|harmonic", {GRIT, "noisiness"}),
    (r"nois|breath|hiss", {"noisiness"}),
    (r"\bloud|quiet|hotter|volume(?! (movement|puls|swell))|\blevel\b", {"loudness"}),
    (r"\bwide|wider|narrow|stereo|\bmono\b|width|spread", {"stereo width"}),
    (r"attack|snap|punch|onset|swell|transient|\bclick|front[- ]end|fades? in|bloom", {"attack"}),
    (r"sustain|die[sd]? (away|off|out)|dying|decay|holds?\b|fades? (out|away)|drops? off|pluck", {"sustain"}),
    (r"\btail|reverb|spac(e|ious)|echo|delay|rings? out|ringing|\bhang|trail|ambien|\broom\b|wash", {"tail after the last note", "sound between notes"}),
    (r"(into|in|fills?) the gaps|between (the )?notes|bleed|smear", {"sound between notes", "tail after the last note"}),
    (r"wobbl|warbl|vibrato|sweep|moving|movement|modulat|swirl|shimmer|chorus|phas(er|ing|ey)|flang|beating|"
     r"\bwaver|undulat|\bdrift", MOVE),
    (r"puls|pump|tremolo|throb|chopp|flutter", MOVE),
    (r"\btone\b|tonal(?! movement)|timbre|character|colou?r|hollow|glass|metallic|nasal|reedy|bell|woody|vocal|"
     r"formant|thick|thin|full(er)?\b|rich|\bbody\b|fatter|\bfat\b|organ|brass", {TONE, GRIT, "brightness"}),
]
POLARITY = {  # dim: (up pattern, down pattern), checked for self_edit_delta / first_impression directions
    "brightness": (r"brighter|more (top|high|air|sparkle)|opens? up|opened up", r"darker|duller|more muffled|less (top|high|bright)|closed"),
    "loudness": (r"louder|hotter|more level", r"quieter|softer overall|lower (in )?level"),
    "stereo width": (r"wider|more (stereo|spread)", r"narrower|more mono|less (wide|stereo)"),
    "tail after the last note": (r"longer tail|more (reverb|space|echo)|rings? out (longer|more)", r"shorter tail|drier|less (reverb|space|echo)"),
}
NEG = re.compile(r"(\b(no|not|without|never|nothing|same|barely|hardly)\b|n[’']t\b)[^.,;]{0,25}$", re.I)
NO_DIFF = re.compile(r"can.?t (really )?(hear|tell)|no (real |audible |obvious )?(difference|change)|"
                     r"(sounds|is) (basically |pretty much )?(the same|identical)|barely (hear|notice)|nothing changed|"
                     r"don.?t hear", re.I)
TIC = re.compile(r"surpris|unexpected|didn.?t expect|wasn.?t expecting|did not expect", re.I)
NUMBERS = re.compile(r"\d+(\.\d+)?\s?(ms|hz|khz|st|db|%|percent|semitones?|cents?|s\b|sec)|\b\d+\s*(voices|x)\b", re.I)
PARAMS = re.compile(r"\b(osc|env|lfo|filter|mod|random)_\d|_(cutoff|resonance|dry_wet|level|amount)\b|\bwave_frame\b", re.I)
FI_DIM = {"brightness": "brightness", "harmonics": GRIT, "noise": "noisiness", "attack": "attack", "body": "sustain",
          "width": "stereo width", "ending": "tail after the last note", "movement": "tonal movement",
          "spectral balance": "brightness", TONE: TONE}


NEG_CLAUSE = re.compile(r"\b(no|not|never|without|barely|hardly|instead of)\b|n[’']t\b", re.I)


def _clause(text: str, i: int) -> str:
    a = max(text.rfind(c, 0, i) for c in ".;!?") + 1
    ends = [j for j in (text.find(c, i) for c in ".;!?") if j >= 0]
    return text[a:min(ends) if ends else len(text)]


def claims(text: str) -> dict:
    """{word_dims (frozenset): matched word} for every lexicon hit not in a negated clause ("the tail I expected
    didn't show up" reports an expected effect that did NOT happen: true, and the most informative kind)."""
    out = {}
    for pat, dims in LEX:
        for m in re.finditer(pat, text, re.I):
            if NEG.search(text[:m.start()]) or NEG_CLAUSE.search(_clause(text, m.start())):
                continue
            out[frozenset(dims)] = m.group(0)
    return out


def _supported_dims(facts: dict) -> set:
    sup = set(facts["supported"])
    if facts["mode"] == "first_impression":
        sup = {FI_DIM.get(k, k) for k in sup}
        if "tonal movement" in sup:
            sup.add("volume movement")
    if sup & MOVE:
        sup |= MOVE
    if "tail after the last note" in sup or "sound between notes" in sup:
        sup |= {"tail after the last note", "sound between notes"}
    return sup


def check(facts: dict, text: str) -> dict:
    problems, unsupported = [], []
    if not text.strip():
        return {"ok": False, "problems": ["empty"], "claims": {}, "unsupported": []}
    if NUMBERS.search(text):
        problems.append("quotes numbers")
    if PARAMS.search(text):
        problems.append("names parameters")
    sup = _supported_dims(facts)
    found = claims(text)
    nodiff = bool(NO_DIFF.search(text))
    aud = facts.get("audibility")
    if facts["mode"] in ("self_edit_delta", "target_vs_current") and TIC.search(text):
        problems.append("talks about being surprised / something being unexpected; just describe what you hear")
    if facts["mode"] == "self_edit_delta":
        if aud == "none":
            if not nodiff:
                problems.append("inaudible edit but the narration does not say it can't really be heard")
        elif aud == "clear" and nodiff and not found:
            problems.append("says no difference on a clearly audible change")
    # words the facts themselves offer (hedged CLAP tone words, audible patch features like "chorus") are fair game
    offered = " ".join(l for l in facts["text"].split("\n") if "tone word" in l or l.startswith("Audible patch")).lower()
    for dims, word in found.items():
        if not (dims & sup) and word.lower() not in offered:
            unsupported.append(word)
    if facts["mode"] == "self_edit_delta" and aud != "none":
        for dim, (up, down) in POLARITY.items():
            d = facts["supported"].get(dim)
            if d in (1, -1) and re.search(down if d > 0 else up, text, re.I):
                unsupported.append(f"{dim} in the wrong direction")
    if facts["mode"] == "first_impression":
        b = facts["buckets"].get("brightness", "")
        if b.startswith(("dark", "warm")) and re.search(r"\bbright|sparkl|sizzl|cutting", text, re.I):
            unsupported.append("bright (measured dark/warm)")
        if b.startswith(("bright", "very bright")) and re.search(r"\bdark|dull|muffl|\bwarm", text, re.I):
            unsupported.append("dark/warm (measured bright)")
    if unsupported:
        problems.append("claims not supported by the facts: " + ", ".join(sorted(set(unsupported))))
    return {"ok": not problems, "problems": problems, "unsupported": sorted(set(unsupported)),
            "claims": sorted({w for w in found.values()})}


def main():
    """Stats per narrator tag and mode from narr_*.jsonl -> printed table + <dir>/stats.json (report.py shows it)."""
    import statistics
    from .narrate import load_probe  # here: narrate imports this module
    d = Path(sys.argv[1])
    fx = {(f["id"], f["mode"]): f for _, f in load_probe(d)[1]}
    aud = {k: f["audibility"] for k, f in fx.items()}
    cols = ["narrator", "mode", "n", "first-try rejected", "first-try unsupported claim", "numbers/params",
            "inaudible -> says can't hear", "template fallback", "kept LLM text failing current filter", "median reasoning chars"]
    table = []
    for p in sorted(glob.glob(str(d / "narr_*.jsonl"))):
        if p.endswith("_r1.jsonl"):
            continue
        rows = [json.loads(l) for l in open(p)]
        by = collections.defaultdict(list)
        for r in rows:
            by[r["mode"]].append(r)
        for mode, rs in sorted(by.items()):
            n = len(rs)
            first = [check(fx[(r["id"], r["mode"])], r["attempts"][0]["text"]) for r in rs]  # current filter
            pct = lambda k: f"{k} ({k / max(n, 1):.0%})"
            num = sum(any(x in pr for pr in c["problems"] for x in ("numbers", "parameters")) for c in first)
            inaud = [c for c in first if any("inaudible" in pr for pr in c["problems"])]
            final = [check(fx[(r["id"], r["mode"])], r["text"]) for r in rs if not r["fallback"]]
            n_inaud = sum(1 for r in rs if aud.get((r["id"], r["mode"])) == "none")
            table.append([rows[0]["tag"], mode, n, pct(sum(not c["ok"] for c in first)),
                          pct(sum(bool(c["unsupported"]) for c in first)), num,
                          f"{n_inaud - len(inaud)}/{n_inaud}" if mode == "self_edit_delta" else "-",
                          pct(sum(r["fallback"] for r in rs)), sum(not c["ok"] for c in final),
                          int(statistics.median(r["attempts"][0]["reasoning_chars"] for r in rs))])
    for row in table:
        print("  ".join(f"{c}" for c in row))
    (d / "stats.json").write_text(json.dumps({"columns": cols, "rows": table}, indent=1))

if __name__ == "__main__":
    main()
