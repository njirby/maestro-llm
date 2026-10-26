"""Probe CLI over the probe1 first-impression targets, through the library API (regression for the recipe).

  .venv/bin/python -m maestro.sounddiff.first_impression notes            # -> outputs/sounddiff/fi_lib/notes.json
  .venv/bin/python -m maestro.sounddiff.first_impression regress          # diff vs the round-4 notes (fi_notes/notes.json)
  .venv/bin/python -m maestro.sounddiff.first_impression run --url http://localhost:8000/v1   # -> fi_lib/narr.jsonl
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor

from ..renderer import REPO, ContainerRenderer
from . import narrate_first_impression, notes_first_impression

P1 = REPO / "outputs/sounddiff/probe1"
OUT = REPO / "outputs/sounddiff/fi_lib"
ROUND4 = REPO / "outputs/sounddiff/fi_notes/notes.json"


def notes() -> None:
    items = [it for it in map(json.loads, open(P1 / "items.jsonl")) if it["mode"] == "first_impression"]
    OUT.mkdir(parents=True, exist_ok=True)
    with ContainerRenderer() as r:
        res = {it["id"]: notes_first_impression(P1 / it["settings_target"], {"bpm": it["bpm"], "notes": it["notes"]},
                                                renderer=r) for it in items}
    (OUT / "notes.json").write_text(json.dumps(res, indent=1))
    print(f"{len(res)} notes -> {OUT / 'notes.json'}")


def regress() -> None:
    new, old = json.loads((OUT / "notes.json").read_text()), json.loads(ROUND4.read_text())
    same = [k for k in old if k in new and new[k]["text"] == old[k]["text"]]
    print(f"{len(same)}/{len(old)} identical to round 4")
    for k in old:
        if k in new and new[k]["text"] != old[k]["text"]:
            a, b = old[k]["text"].splitlines(), new[k]["text"].splitlines()
            for x, y in zip(a, b):
                if x != y:
                    print(f"{k}\n  - {x}\n  + {y}")


def run(url: str, samples: int) -> None:
    res = {k: v for k, v in json.loads((OUT / "notes.json").read_text()).items() if not v["silent"]}
    jobs = [(k, s) for k in res for s in range(samples)]
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda j: narrate_first_impression(res[j[0]]["audio"], res[j[0]], url=url), jobs))
    with open(OUT / "narr.jsonl", "w") as f:
        for (k, s), o in zip(jobs, outs):
            f.write(json.dumps({"id": k, "k": s, **o}, ensure_ascii=False) + "\n")
    n = len(outs)
    first = sum(bool(o["attempts"][0]["problems"]) for o in outs)
    final = sum(bool(o["problems"]) for o in outs)
    print(f"{n} impressions; first try flagged {first} ({first / n:.0%}); after one retry {final} ({final / n:.0%})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["notes", "regress", "run"])
    ap.add_argument("--url", default="http://localhost:8000/v1")
    ap.add_argument("--samples", type=int, default=2)
    a = ap.parse_args()
    {"notes": notes, "regress": regress, "run": lambda: run(a.url, a.samples)}[a.cmd]()
