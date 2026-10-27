"""judge: loads the skill, listens to the target, renders the GT combination and one or two alternatives from the pool
(init patch, oscillators on, tables swapped), listens to each combination one at a time, replies
{"tuple": [...], "reasoning": ...}.
The choice is oracle (the GT tables in oscillator order)."""
from __future__ import annotations

import json
import shlex

from ..ctx import Ctx
from ..record import Record
from .common import INIT, RENDER, listen, open_session, run
from .searcher import SET, WT

AGENT = "judge"


def combo_line(refs: list[str], path: str) -> str:
    parts = [f"cp {INIT} {path}.json"]
    on = [f"osc_{n}_on=on" for n in range(2, len(refs) + 1)]
    if on:
        parts.append(f"{SET} {path}.json {' '.join(on)} >/dev/null")
    parts += [f"{WT} set {path}.json --osc {n} {shlex.quote(r)} >/dev/null" for n, r in enumerate(refs, 1)]
    parts.append(f"{RENDER} {path}.json notes.json {path}.wav")
    return " && ".join(parts)


def run_judge(ctx: Ctx, prompt: str, rid: str, pool: list[str], gt_refs: list[str]) -> dict:
    rng = ctx.rng
    others = [r for r in pool if r not in gt_refs]
    combos = [list(gt_refs)]
    if len(gt_refs) == 1:
        combos += [[r] for r in others[:2]]
    else:
        if others:
            i = rng.randrange(len(gt_refs))
            combos.append(gt_refs[:i] + [rng.choice(others)] + gt_refs[i + 1:])
        combos.append(gt_refs[:-1])
    rng.shuffle(combos)
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)
    paths = [f"judge/c{i + 1}" for i in range(len(combos))]
    run(rec, ctx, "\n".join(["mkdir -p judge"] + [combo_line(c, p) for c, p in zip(combos, paths)]))
    for p in paths:
        listen(rec, ctx, f"{p}.wav")
    oscs = " + ".join(f"{r} (osc {n})" for n, r in enumerate(gt_refs, 1))
    reasoning = f"{oscs} came closest to the target of the {len(combos)} combinations I rendered."
    ctx.note(agent=AGENT, combos=combos, choice=gt_refs)
    return rec.finish(json.dumps({"tuple": gt_refs, "reasoning": reasoning}, ensure_ascii=False))
