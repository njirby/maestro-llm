"""Probe CLI for the comparison modes (self_edit, target_vs_current) on outputs/sounddiff/probe1: build the verified
notes, run the head-to-head arms, write the blind pages.

  .venv/bin/python -m maestro.sounddiff.delta_probe notes self_edit [--ids e000,e001] [--n 40]
  .venv/bin/python -m maestro.sounddiff.delta_probe run self_edit omni --url http://localhost:8000/v1
  .venv/bin/python -m maestro.sounddiff.delta_probe run self_edit text --url http://localhost:8001/v1
  .venv/bin/python -m maestro.sounddiff.delta_probe score self_edit      # re-check every arm with the current checker
  .venv/bin/python -m maestro.sounddiff.delta_probe page self_edit       # -> outputs/sounddiff/delta/self_edit/index.html

The head-to-head uses `selected(mode)`: 40 items per mode, stratified (every audible self-edit plus inaudible ones to
fill; every tvc item whose last edit moved it closer / further, then audible gaps, then matches).
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .delta import narrate_self_edit, narrate_target_vs_current, notes_self_edit, notes_target_vs_current
from .renderer import REPO, ContainerRenderer

P1 = REPO / "outputs/sounddiff/probe1"
OUT = REPO / "outputs/sounddiff/delta"
PMODE = {"self_edit": "self_edit_delta", "target_vs_current": "target_vs_current"}


def probe_items(mode: str) -> list[dict]:
    return [it for it in map(json.loads, open(P1 / "items.jsonl")) if it["mode"] == PMODE[mode]]


def build_notes(mode: str, ids: list[str] | None, n: int) -> None:
    items = [it for it in probe_items(mode) if not ids or it["id"] in ids]
    d = OUT / mode
    d.mkdir(parents=True, exist_ok=True)
    f = d / "notes.json"
    res = json.loads(f.read_text()) if f.exists() else {}
    with ContainerRenderer() as r:
        for it in items:
            if len([k for k, v in res.items() if not v.get("skip")]) >= n and it["id"] not in res:
                break
            pr = lambda role: P1 / "presets" / f"{it['id']}_{role}.vital"
            notes = {"bpm": it["bpm"], "notes": it["notes"]}
            t0 = time.time()
            if mode == "self_edit":
                x = notes_self_edit(pr("a"), pr("b"), notes, renderer=r)
            else:
                x = notes_target_vs_current(pr("t"), pr("c0"), pr("c1"), notes, renderer=r,
                                            next_subsystem=it.get("next_subsystem"))
            # silent clips can't be narrated from audio: kept out of the head-to-head
            x["skip"] = "silent" in x["text"]
            x["moves"] = {k: list(v) for k, v in x["moves"].items()}
            res[it["id"]] = x
            print(it["id"], f"{time.time() - t0:.0f}s", x["overall"], x.get("trajectory", ""), flush=True)
            f.write_text(json.dumps(res, indent=1))


def selected(mode: str, n: int = 40) -> list[str]:
    notes = {k: v for k, v in json.loads((OUT / mode / "notes.json").read_text()).items() if not v.get("skip")}
    if mode == "self_edit":
        key = lambda k: (notes[k]["overall"] == "none", k)
    else:
        key = lambda k: (notes[k]["trajectory"] == "same", notes[k]["overall"] == "none", k)
    keep = sorted(notes, key=key)[:n]
    return sorted(keep)


def score(mode: str) -> dict:
    """First-try and final (after one regenerate) flag rates per arm on the selected items, re-checked with the
    current checker (the retry decision itself was made by the checker at run time)."""
    from .check import check_delta, hit
    d = OUT / mode
    notes = json.loads((d / "notes.json").read_text())
    ids = set(selected(mode))
    out = {}
    for arm in ("omni", "text", "audio"):
        p = d / f"arm_{arm}.jsonl"
        if not p.exists():
            continue
        rows = [r for r in map(json.loads, open(p)) if r["id"] in ids]
        first = [check_delta(r["attempts"][0]["text"], notes[r["id"]]["check"]) for r in rows]
        # final = the recipe replayed with the current checker: a clean first sample is kept, a flagged one is replaced
        # by the regenerated sample when the run made one (when the run-time checker passed the first sample, there
        # is no second one and the first stays flagged)
        final = [f if not f or len(r["attempts"]) < 2 else check_delta(r["attempts"][1]["text"], notes[r["id"]]["check"])
                 for f, r in zip(first, rows)]
        texts = [r["attempts"][0]["text"] if not f or len(r["attempts"]) < 2 else r["attempts"][1]["text"]
                 for f, r in zip(first, rows)]
        labels = {}
        for pr in first:
            for x in pr:
                labels[x.split(",")[0]] = labels.get(x.split(",")[0], 0) + 1
        n = len(rows)
        hits = sum(hit(t, notes[r["id"]]["check"]) for t, r in zip(texts, rows))
        out[arm] = {"n": n, "hit_rate": round(hits / n, 3), "first_flagged": sum(map(bool, first)), "final_flagged": sum(map(bool, final)),
                    "first_rate": round(sum(map(bool, first)) / n, 3), "final_rate": round(sum(map(bool, final)) / n, 3),
                    "regenerated": sum(len(r["attempts"]) > 1 for r in rows),
                    "top_problems": sorted(labels.items(), key=lambda kv: -kv[1])[:6]}
        print(mode, arm, {k: v for k, v in out[arm].items() if k != "top_problems"})
        print("   ", out[arm]["top_problems"])
    (d / "score.json").write_text(json.dumps(out, indent=1))
    return out


def run_arm(mode: str, arm: str, url: str, samples: int, workers: int) -> None:
    d = OUT / mode
    notes = {k: v for k, v in json.loads((d / "notes.json").read_text()).items() if not v.get("skip")}
    fn = narrate_self_edit if mode == "self_edit" else narrate_target_vs_current
    jobs = [(iid, k) for iid in notes for k in range(samples)]

    def one(job):
        return fn(notes[job[0]], url=url, arm=arm)
    with ThreadPoolExecutor(workers) as ex:
        outs = list(ex.map(one, jobs))
    with open(d / f"arm_{arm}.jsonl", "w") as f:
        for (iid, k), o in zip(jobs, outs):
            f.write(json.dumps({"id": iid, "k": k, **o}, ensure_ascii=False) + "\n")
    n = len(outs)
    first = sum(bool(o["attempts"][0]["problems"]) for o in outs)
    final = sum(bool(o["problems"]) for o in outs)
    print(f"{mode}/{arm}: {n} narrations; first try flagged {first} ({first / n:.0%}); final {final} ({final / n:.0%})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["notes", "run", "score", "page"])
    ap.add_argument("mode", choices=list(PMODE))
    ap.add_argument("arm", nargs="?")
    ap.add_argument("--ids")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--samples", type=int, default=2)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    if a.cmd == "notes":
        build_notes(a.mode, a.ids.split(",") if a.ids else None, a.n)
    elif a.cmd == "run":
        run_arm(a.mode, a.arm, a.url, a.samples, a.workers)
    elif a.cmd == "score":
        score(a.mode)
    else:
        from .delta_page import page
        page(a.mode)


if __name__ == "__main__":
    main()
