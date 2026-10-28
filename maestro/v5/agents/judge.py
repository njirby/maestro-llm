"""judge: loads the skill, listens to the target, ranks every assignment of the pooled tables to the target's
oscillators (the superposition ranking, maestro/v5/wt_rank.py), renders the best one and up to three runners-up
(init patch, oscillators on, tables swapped), listens to each one at a time, replies {"tuple": [...], "reasoning": ...}.
The choice is the ranking's top configuration: the GT tables when the pool holds them (or frame-0 duplicates of them,
which are equally correct), else the closest the pool allows."""
from __future__ import annotations

import json

from .. import wt_rank
from ..ctx import Ctx
from ..planner import wt_ref
from ..record import Record
from .common import INIT, RENDER, listen, open_session, run
from .searcher import SET, WT
from ..search import batched, combo_lines

AGENT = "judge"
MAX_ALTS = 3


def run_judge(ctx: Ctx, prompt: str, rid: str, pool: list[dict[int, dict]], gt_preset: dict, renderer,
              gt_tables: list[dict] = ()) -> tuple[dict, dict[int, dict]]:
    """pool: the searchers' configurations ({oscillator slot: library entry}). Returns (record, the chosen
    configuration). gt_tables only feeds the diagnostics in the event log."""
    cands = {e["id"]: e for c in pool for e in c.values()}
    slots = wt_rank.active_slots(gt_preset)
    ranked = wt_rank.rank_subsets(renderer, gt_preset, list(cands.values()))
    picks: list[tuple[dict[int, dict], float]] = []
    for r in ranked:
        cfg = {n: cands[t] for n, t in zip(slots, r.tables) if t is not None}
        if not cfg or any(len(cfg) == len(p) and all(n in p and wt_rank.same_table(p[n]["id"], e["id"])
                                                      for n, e in cfg.items()) for p, _ in picks):
            continue           # an empty assignment, or the same configuration up to frame-0 duplicates
        picks.append((cfg, r.score_db))
        if len(picks) == 1 + MAX_ALTS:
            break
    choice = picks[0][0]
    combos = [[e for _, e in sorted(c.items())] for c, _ in picks]

    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)
    out_dir = "judge"
    for step in batched(combo_lines([(f"c{i + 1}", c) for i, c in enumerate(combos)], out_dir, INIT, SET, WT, RENDER)):
        run(rec, ctx, step)
    for i in range(len(combos)):
        listen(rec, ctx, f"{out_dir}/c{i + 1}.wav")
    oscs = " + ".join(f"{wt_ref(e)} (osc {n})" for n, e in sorted(choice.items()))
    reasoning = f"{oscs} came closest to the target of the {len(combos)} combinations I rendered."
    refs = [wt_ref(e) for _, e in sorted(choice.items())]
    gt = {n: e["id"] for n, e in zip(slots, gt_tables)}
    ctx.note(agent=AGENT, combos=[{n: e["id"] for n, e in c.items()} for c, _ in picks],
             scores=[round(sc, 2) for _, sc in picks], choice=refs,
             choice_is_gt=all(n in choice and wt_rank.same_table(choice[n]["id"], t) for n, t in gt.items())
             if gt else None, pool=[{n: e["id"] for n, e in c.items()} for c in pool])
    return rec.finish(json.dumps({"tuple": refs, "reasoning": reasoning}, ensure_ascii=False)), choice
