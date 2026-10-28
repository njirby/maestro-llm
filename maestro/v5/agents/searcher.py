"""searcher: loads the skill, listens to the target, lists its chunk, then the full audition: every table in the
chunk rendered as an osc-1 probe (init patch) and listened to in batches; then its closest combinations rendered and
listened to; replies with the best configurations for the target in this chunk (tables per oscillator).

What counts as "closest" is the superposition ranking (maestro/v5/wt_rank.py, docs/wavetable_search_heuristic.md):
every assignment of the chunk's tables to the target's oscillators, scored exactly against the target from
single-oscillator renders of the stripped target patch (oracle, host-side). Nothing is forced: the reply is the
ranking's top configurations, whether or not they hold the GT tables (frame-0 duplicates of a GT table are equally
correct).
"""
from __future__ import annotations

from .. import wt_rank
from ..audio import peak_dbfs
from ..ctx import Ctx
from ..planner import vitallib, wt_ref
from ..record import Record
from .common import INIT, RENDER, listen_many, open_session, run
from ..search import BEAM, TOP_K, batched, combo_lines

AGENT = "searcher"
WT = "python3 $VITAL_SKILL/scripts/wavetable.py"
SET = "python3 $VITAL_SKILL/scripts/set.py"
QUIET_DBFS = -60.0
LISTEN_BATCH = 8      # probes listened to per step; the agent comments after each batch


def _batch_impression(batch: list[dict], dist: dict, quiet_now: set, running_best: dict | None, name) -> str:
    """One short comment after a batch of listens — what the batch is/isn't close to, plus near-silent drops
    (all measured, so the agent may say them). `name` maps an item to its display label."""
    dropped = [name(it) for it in batch if it["id"] in quiet_now]
    audible = [it for it in batch if it["id"] not in quiet_now]
    if not audible:
        verb = "render" if len(dropped) > 1 else "renders"
        pron = "them" if len(dropped) > 1 else "it"
        return f"{' and '.join(dropped)} {verb} near-silent on this patch; dropping {pron}."
    best = min(audible, key=lambda it: dist[it["id"]])
    if running_best is None or dist[best["id"]] < dist[running_best["id"]]:
        base = f"{name(best)} is the closest of that batch, but still not the target."
    else:
        base = f"Nothing in that batch beats {name(running_best)}."
    if dropped:
        verb = "render" if len(dropped) > 1 else "renders"
        pron = "them" if len(dropped) > 1 else "it"
        base += f" {' and '.join(dropped)} {verb} near-silent on this patch; dropping {pron}."
    return base


def _listen_batched(rec: Record, ctx: Ctx, out_dir: str, items: list[dict], dist: dict, quiet: set, name) -> dict | None:
    """Listen to rendered probes in batches of LISTEN_BATCH and say one short impression per batch. `dist` holds the
    ranking's score per item (lower = closer). Returns the running-best item (or None)."""
    running_best = None
    for i in range(0, len(items), LISTEN_BATCH):
        b = items[i:i + LISTEN_BATCH]
        hosts = {it["id"]: ctx.fetch(f"{out_dir}/{it['id']}.wav") for it in b}
        quiet_now = {k for k, h in hosts.items() if peak_dbfs(h) < QUIET_DBFS}
        quiet |= quiet_now
        listen_many(rec, ctx, [f"{out_dir}/{it['id']}.wav" for it in b])
        rec.say(_batch_impression(b, dist, quiet_now, running_best, name))
        audible = [it for it in b if it["id"] not in quiet_now]
        if audible:
            best = min(audible, key=lambda it: dist[it["id"]])
            if running_best is None or dist[best["id"]] < dist[running_best["id"]]:
                running_best = best
    return running_best


def config_line(i: int, cfg: dict[int, dict], why: str) -> str:
    """cfg: {oscillator slot: library entry}."""
    refs = []
    for n, e in sorted(cfg.items()):
        ref, wid = wt_ref(e), e["id"]
        core = f"{ref} (osc {n})" if len(cfg) > 1 or n != 1 else ref
        refs.append(f"{core} ({wid})" if ref != wid else core)
    return f"{i}. " + " + ".join(refs) + f": {why}"


def _why(i: int) -> str:
    return "closest to the target in this chunk" if i == 0 else "also close" if i == 1 else "close, a bit further off"


def _label(cfg: dict[int, dict]) -> str:
    return "_".join(e["id"] for _, e in sorted(cfg.items()))


def run_searcher(ctx: Ctx, k: int, prompt: str, rid: str, start: int, end: int, gt_preset: dict, renderer,
                 gt_tables: list[dict] = ()) -> tuple[dict, list[dict[int, dict]]]:
    """Returns (record, configurations in reply order — each {oscillator slot: library entry}). gt_tables only feeds
    the diagnostics in the event log (did the reply hold the chunk's GT tables?)."""
    idx = vitallib.wt_index()
    chunk = idx[start:end]
    by_id = {e["id"]: e for e in chunk}
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)        # skill, then the target: candidates are picked after hearing it
    run(rec, ctx, f"{WT} list --ids {chunk[0]['id']}-{chunk[-1]['id']}")
    label = f"{chunk[0]['id']}-{chunk[-1]['id']}"
    out_dir = f"probes/s{k}"

    # the ranking (oracle, host-side): every assignment of this chunk's tables to the target's oscillators
    slots = wt_rank.active_slots(gt_preset)
    ranked = wt_rank.rank_subsets(renderer, gt_preset, chunk)
    tscore = wt_rank.table_scores(ranked)

    # ---- the full audition — every table in the chunk as an osc-1 probe -------------------------------------
    for step in batched(combo_lines([(e["id"], [e]) for e in chunk], out_dir, INIT, SET, WT, RENDER)):
        run(rec, ctx, step)
    dist = {e["id"]: tscore.get(e["id"], float("inf")) for e in chunk}
    quiet: set[str] = set()
    _listen_batched(rec, ctx, out_dir, list(chunk), dist, quiet, wt_ref)

    # ---- the closest configurations: render the multi-table ones (singles were just heard) ---------------------
    configs, seen = [], set()
    for r in ranked:
        cfg = {slot: by_id[t] for slot, t in zip(slots, r.tables) if t is not None}
        key = tuple(sorted((n, e["id"]) for n, e in cfg.items()))
        if key in seen or any(e["id"] in quiet for e in cfg.values()):
            continue
        seen.add(key)
        configs.append((cfg, r.score_db))
        if len(configs) == BEAM:
            break
    combos = [(cfg, sc) for cfg, sc in configs if len(cfg) > 1]
    if combos:
        for step in batched(combo_lines([(_label(c), [e for _, e in sorted(c.items())]) for c, _ in combos],
                                        out_dir, INIT, SET, WT, RENDER)):
            run(rec, ctx, step)
        items = [{"id": _label(c), "cfg": c} for c, _ in combos]
        d2 = {_label(c): sc for c, sc in combos}
        _listen_batched(rec, ctx, out_dir, items, d2, set(),
                        lambda it: " + ".join(wt_ref(e) for _, e in sorted(it["cfg"].items())))
    top = [cfg for cfg, _ in configs[:TOP_K]]

    header = f"Auditioned all {len(chunk)} tables in {label}"
    header += f", then {len(combos)} of the closest combinations." if combos else " with the session notes."
    lines = [header, "My best configurations for the target in this chunk:" if len(slots) > 1
             else "My best for the target in this chunk:"]
    lines += [config_line(i + 1, cfg, _why(i)) for i, cfg in enumerate(top)]
    need = {n: e["id"] for n, e in zip(slots, gt_tables) if e["id"] in by_id}
    held = any(all(n in c and wt_rank.same_table(c[n]["id"], t) for n, t in need.items()) for c in top) if need else None
    ctx.note(agent=AGENT, k=k, chunk=[start, end], label=label, need=need, quiet=sorted(quiet),
             reply_holds_gt=held, configs=[{n: e["id"] for n, e in c.items()} for c in top],
             scores=[round(sc, 2) for _, sc in configs[:TOP_K]])
    return rec.finish("\n".join(lines)), top
