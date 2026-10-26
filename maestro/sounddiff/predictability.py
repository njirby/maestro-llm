"""How much of a grounded narration could have been written from the text context alone?

For each delta item (self_edit_delta, target_vs_current) the narrator also writes a text-only prediction from the
edit list (no measurements). An LLM extractor turns both narrations into (dimension, direction) claims (+ "no audible
difference", + closer/further for target_vs_current), and an item carries information beyond the text when the claim
sets differ. A second, extraction-free view compares the prediction with the measured facts directly. The share of
items beyond the text sets the dataset mix (aim: >= 30% surprise deltas; facts["surprise"] is the cheap proxy).

  .venv/bin/python -m maestro.sounddiff.predictability outputs/sounddiff/probe1 --narr narr_27b-xhigh.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .facts import TONE, edit_lines, melody_line
from .narrate import COMMON, Narrator, load_probe

KEYS = ["brightness", "grit", "noisiness", "loudness", "width", "attack", "sustain", "space", "movement", "tone"]
FKEY = {"brightness": "brightness", "grit / harmonic richness": "grit", "noisiness": "noisiness",
        "loudness": "loudness", "stereo width": "width", "attack": "attack", "sustain": "sustain",
        "tail after the last note": "space", "sound between notes": "space", "tonal movement": "movement",
        "volume movement": "movement", TONE: "tone"}

PREDICT = {
    "self_edit_delta": """Sound: a {arch} patch in Vital playing {melody}.

Edit you just made (before -> after):
{edit}

You haven't played it back yet. In one to three sentences, say how you expect the sound changed, the way you'd say \
it to a collaborator. If you expect no audible difference, say so. """ + COMMON,
    "target_vs_current": """Sound: a {arch} patch in Vital playing {melody}. You're recreating a target sound.

Your last edit (before -> after):
{edit}

You haven't A/B'd it against the target yet. In one to three sentences, say how you expect your patch now compares \
with the target and whether the edit got you closer. """ + COMMON,
}

EXTRACT = """Read this sentence a music producer said about a synth sound and list the claims it makes.

Mode: {mode_desc}
Sentence: "{text}"

Dimensions: brightness (up = brighter), grit (up = grittier / more bite / more harmonics), noisiness (up = noisier / \
breathier), loudness (up = louder), width (up = wider stereo), attack (up = slower/softer attack, down = snappier), \
sustain (up = notes hold longer, down = die away faster), space (up = more tail / reverb / echo / ringing between \
notes), movement (up = more wobble / modulation / pulsing), tone (overall timbre/character changed; dir "changed").

Return JSON: {{"no_difference": true/false (the speaker says they can't really hear a difference), "claims": [{{"dim": \
one of the dimensions, "dir": "up" | "down" | "changed"}}], "closer": "closer" | "further" | "same" | "unknown" \
(target mode only: did the edit get them closer to the target)}}. {claim_scope} Only include claims the sentence \
actually makes; a hedged claim still counts."""
SCOPE = {"self_edit_delta": "Claims are about how the sound changed with the edit.",
         "target_vs_current": "Claims are about how the TARGET differs from the speaker's current patch (dir: the "
                              "target is more/less X than theirs); ignore statements about what the edit did."}
MODE_DESC = {"self_edit_delta": "describing how an edit changed the sound",
             "target_vs_current": "comparing their patch with a target sound they are recreating"}


def predict_prompt(item):
    return PREDICT[item["mode"]].format(arch=item["archetype"].lower(), melody=melody_line(item),
                                        edit="\n".join(edit_lines(item["edit"])))


def _parse(txt: str) -> dict:
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        j = json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        j = {}
    claims = {(c.get("dim"), c.get("dir")) for c in j.get("claims", []) if c.get("dim") in KEYS}
    claims = {(d, "changed" if d == "tone" else r) for d, r in claims}
    return {"no_difference": bool(j.get("no_difference")), "claims": sorted(claims), "closer": j.get("closer", "unknown")}


def extract(texts_modes, ext: Narrator) -> list[dict]:
    def one(tm):
        text, mode = tm
        p = EXTRACT.format(mode_desc=MODE_DESC[mode], text=text.replace('"', "'"), claim_scope=SCOPE[mode])
        out, _ = ext.call(p, system="You extract structured claims from text. Reply with JSON only.", json_mode=True)
        return _parse(out)
    with ThreadPoolExecutor(ext.workers) as ex:
        return list(ex.map(one, texts_modes))


def sig(e: dict, mode: str) -> frozenset:
    s = set(map(tuple, e["claims"])) | ({("no_difference", "")} if e["no_difference"] else set())
    if mode == "target_vs_current":
        s.add(("closer", e["closer"]))
    return frozenset(s)


def truth(f: dict) -> frozenset:
    """Measured claim set from the facts (dims merged to extractor keys)."""
    if f["mode"] == "self_edit_delta":
        if f["audibility"] == "none":
            return frozenset({("no_difference", "")})
        return frozenset((FKEY[c["dim"]], "changed" if c["dim"] == TONE else ("up" if c["dir"] > 0 else "down"))
                         for c in f["changes"])
    s = {(FKEY[c["dim"]], "changed" if c["dim"] == TONE else ("up" if c["dir"] > 0 else "down")) for c in f["gaps"]}
    return frozenset(s | {("closer", f["trajectory"])})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--narr", default="narr_27b-xhigh.jsonl")
    ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
    ap.add_argument("--model", default="qwen3.8-27b")
    a = ap.parse_args()
    d = Path(a.dir)
    _, items = load_probe(d)
    items = [(it, f) for it, f in items if it["mode"] in PREDICT]
    narr = {(r["id"], r["mode"]): r for r in map(json.loads, open(d / a.narr))}
    nar = Narrator(a.url, a.model)
    with ThreadPoolExecutor(nar.workers) as ex:
        preds = list(ex.map(lambda x: nar.call(predict_prompt(x[0]))[0], items))
    ext = Narrator(a.url, a.model, think=False, temperature=0.0, max_tokens=600)
    e_pred = extract([(p, it["mode"]) for p, (it, _) in zip(preds, items)], ext)
    e_grnd = extract([(narr[(it["id"], it["mode"])]["text"], it["mode"]) for it, _ in items], ext)
    rows, stats = [], collections.defaultdict(collections.Counter)
    for (it, f), p, ep, eg in zip(items, preds, e_pred, e_grnd):
        mode = it["mode"]
        sp, sg, st = sig(ep, mode), sig(eg, mode), truth(f)
        beyond = sp != sg
        # extraction-free view: the prediction misses a measured change or claims one that was not measured
        dims = lambda s: {x for x in s if x[0] not in ("closer",)}
        wrong_vs_facts = dims(sp) != dims(st) if mode == "self_edit_delta" else \
            (ep["closer"] != f["trajectory"] or not dims(sp) <= dims(st))
        traj_wrong = mode == "target_vs_current" and ep["closer"] != f["trajectory"]
        s = stats[mode]
        s["n"] += 1
        s["beyond_text"] += beyond
        s["pred_wrong_vs_facts"] += wrong_vs_facts
        s["facts_surprise"] += bool(f.get("surprise"))
        s["traj_wrong"] += traj_wrong
        s["grounded_matches_facts"] += dims(sg) <= dims(st) | {("no_difference", "")}
        rows.append({"id": it["id"], "mode": mode, "prediction": p, "grounded": narr[(it["id"], mode)]["text"],
                     "pred_claims": sorted(map(list, sp)), "grounded_claims": sorted(map(list, sg)),
                     "fact_claims": sorted(map(list, st)), "beyond_text": beyond, "pred_wrong_vs_facts": wrong_vs_facts})
    with open(d / "predictability.jsonl", "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    summary = {}
    for mode, s in stats.items():
        n = s["n"]
        summary[mode] = {k: f"{v}/{n} ({v / n:.0%})" for k, v in s.items() if k != "n"}
        print(mode, n, summary[mode])
    (d / "predictability_summary.json").write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
