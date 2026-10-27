"""judge: loads the skill, listens to the target, renders the GT configuration and up to three recombos from the
pooled configurations (one table from one configuration, another from another; init patch, oscillators on,
tables swapped), listens to each one at a time, replies {"tuple": [...], "reasoning": ...}.
The choice is oracle (the GT tables in oscillator order)."""
from __future__ import annotations

import json

from ..ctx import Ctx
from ..planner import wt_ref
from ..record import Record
from .common import INIT, RENDER, listen, open_session, run
from .searcher import SET, WT
from ..search import batched, combo_lines

AGENT = "judge"
MAX_ALTS = 3


def _recombos(pool: list[list[dict]], gt: list[dict]) -> list[list[dict]]:
    """Cross-config recombos: for each GT slot, swap in a non-GT table from a pool configuration that contains
    that slot's GT table; plus the first configuration sharing no tables with the GT at all. Capped, deduped."""
    gt_ids = [e["id"] for e in gt]
    alts, seen = [], {tuple(gt_ids)}

    def add(combo: list[dict]) -> None:
        key = tuple(e["id"] for e in combo)
        if key not in seen:
            seen.add(key)
            alts.append(list(combo))

    for i, g in enumerate(gt):
        for cfg in pool:
            ids = [e["id"] for e in cfg]
            if g["id"] not in ids or ids == gt_ids:
                continue
            for t in cfg:
                if t["id"] != g["id"] and t["id"] not in gt_ids:
                    add([e for e in gt][:i] + [t] + [e for e in gt][i + 1:])
    for cfg in pool:
        if len(cfg) == len(gt) and not ({e["id"] for e in cfg} & set(gt_ids)):
            add(list(cfg))
    return alts[:MAX_ALTS]


def run_judge(ctx: Ctx, prompt: str, rid: str, pool: list[list[dict]], gt: list[dict]) -> dict:
    combos = [list(gt)] + _recombos(pool, gt)
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)
    out_dir = "judge"
    for step in batched(combo_lines([(f"c{i + 1}", c) for i, c in enumerate(combos)], out_dir, INIT, SET, WT, RENDER)):
        run(rec, ctx, step)
    for i in range(len(combos)):
        listen(rec, ctx, f"{out_dir}/c{i + 1}.wav")
    oscs = " + ".join(f"{wt_ref(e)} (osc {n})" for n, e in enumerate(gt, 1))
    reasoning = f"{oscs} came closest to the target of the {len(combos)} combinations I rendered."
    gt_refs = [wt_ref(e) for e in gt]
    ctx.note(agent=AGENT, combos=[[e["id"] for e in c] for c in combos], choice=gt_refs, pool=[[e["id"] for e in c] for c in pool])
    return rec.finish(json.dumps({"tuple": gt_refs, "reasoning": reasoning}, ensure_ascii=False))
