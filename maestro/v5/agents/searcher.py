"""searcher: loads the skill, listens to the target, lists its library slice, renders a wide net of candidates through
the session notes (init patch, only the osc 1 wavetable swapped), listens to each probe one at a time, replies with a
2-4 name shortlist.

Oracle: every GT table inside the slice is auditioned and shortlisted. The other candidates are drawn from the slice;
the rest of the shortlist is the candidates nearest (audio.distance) to a probe of a GT table, so the order is measured.
"""
from __future__ import annotations

from maestro.synth.preset_gen import _SILENT_WAVETABLES

from ..audio import distance, peak_dbfs
from ..ctx import Ctx
from ..planner import vitallib, wt_ref
from ..record import Record
from .common import INIT, RENDER, listen, open_session, run

AGENT = "searcher"
WT = "python3 $VITAL_SKILL/scripts/wavetable.py"
SET = "python3 $VITAL_SKILL/scripts/set.py"
P_SILENT = 0.10      # share of searcher records that audition one silent table
QUIET_DBFS = -60.0


def probe_lines(entries: list[dict], out_dir: str) -> str:
    lines = [f"mkdir -p {out_dir}"]
    for e in entries:
        p = f"{out_dir}/{e['id']}"
        frame = f" && {SET} {p}.json osc_1_wave_frame=128 >/dev/null" if e.get("silent_frame0") else ""
        lines.append(f"cp {INIT} {p}.json && {WT} set {p}.json --osc 1 {e['id']} >/dev/null{frame}"
                     f" && {RENDER} {p}.json notes.json {p}.wav")
    return "\n".join(lines)


def shortlist_line(e: dict, why: str) -> str:
    ref, wid = wt_ref(e), e["id"]
    return f"- {ref}: {why}" if ref == wid else f"- {ref} ({wid}): {why}"


def run_searcher(ctx: Ctx, k: int, prompt: str, rid: str, start: int, end: int, gt: list[dict],
                 gt_probes: dict[str, str]) -> tuple[dict, list[str]]:
    """Returns (record, shortlisted refs in reply order). gt_probes: GT table id -> host wav of its oracle probe."""
    rng = ctx.rng
    idx = vitallib.wt_index()
    sl = idx[start:end]
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)        # skill, then the target: candidates are picked after hearing it
    run(rec, ctx, f"{WT} list --ids {sl[0]['id']}-{sl[-1]['id']}")

    gt_ids = {e["id"] for e in gt}
    mine = [e for e in sl if e["id"] in gt_ids]
    # distractors: mostly tables with a distinct real name (what a searcher picks from a listing), some unnamed ones;
    # silent-on-init tables (WS-C's list; never GT) only via P_SILENT below
    pool = [e for e in sl if e["id"] not in gt_ids and e["name"] not in _SILENT_WAVETABLES]
    named, seen = [], {e["name"].lower() for e in mine}
    for e in pool:
        if e["name"] not in ("Init", "(unnamed)") and e["name"].lower() not in seen:
            named.append(e)
            seen.add(e["name"].lower())
    unnamed = [e for e in pool if e not in named]
    n_other = max(rng.randint(8, 12) - len(mine), 4)
    n_unnamed = min(len(unnamed), sum(rng.random() < 0.2 for _ in range(n_other)))
    cands = (mine + rng.sample(named, min(len(named), n_other - n_unnamed))
             + rng.sample(unnamed, min(len(unnamed), n_other - min(len(named), n_other - n_unnamed))))
    # now and then one table that renders silent: heard as near-silence and dismissed (a true listening example)
    silent = [e for e in sl if e["id"] not in gt_ids and e["name"] in _SILENT_WAVETABLES]
    rng.shuffle(cands)
    if silent and rng.random() < P_SILENT:
        cands.insert(rng.randrange(len(cands)), rng.choice(silent))  # never last: its note must not join the reply
    out_dir = f"probes/s{k}"
    run(rec, ctx, probe_lines(cands, out_dir))
    dist = {}
    quiet = set()
    for e in cands:
        host = listen(rec, ctx, f"{out_dir}/{e['id']}.wav")
        dist[e["id"]] = min(distance(host, ref) for ref in gt_probes.values())
        if peak_dbfs(host) < QUIET_DBFS:  # measured, so the agent may say it
            quiet.add(e["id"])
            rec.say(f"{wt_ref(e)} is close to silent on this patch; dropping it.")

    n_short = rng.randint(2, min(4, len(cands)))
    rest = sorted((e for e in cands if e["id"] not in gt_ids and e["id"] not in quiet), key=lambda e: dist[e["id"]])
    short = mine + rest[:max(n_short - len(mine), 0)]
    lines = []
    for i, e in enumerate(short):
        why = ("closest to the target in this slice" if i == 0 else
               "also close" if e["id"] in gt_ids or i == 1 else "close, a bit further off")
        lines.append(shortlist_line(e, why))
    ctx.note(agent=AGENT, k=k, slice=[start, end], candidates=[e["id"] for e in cands],
             shortlist=[e["id"] for e in short], dist={i: round(d, 2) for i, d in dist.items()})
    return rec.finish("\n".join(lines)), [wt_ref(e) for e in short]
