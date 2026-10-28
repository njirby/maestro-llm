"""searcher: loads the skill, listens to the target, lists its chunk, then the full audition: every table in the
chunk rendered as an osc-1 probe (init patch) and listened to one at a time; the B closest singles beamed into
oriented pairs (both orders), rendered and listened to; replies with the best 2-3 configurations for the target
in this chunk (tables in oscillator order).

Oracle: distances are measured against the TARGET only — never against the GT. The GT matters only to the local
gate: every GT table in the chunk must appear in one of the returned configurations (in relative order), else the
judge can never reassemble the answer. If the search misses, widen the beam once, then rescue the GT configuration
into the reply.
"""
from __future__ import annotations

from ..audio import distance, peak_dbfs
from ..ctx import Ctx
from ..planner import vitallib, wt_ref
from ..record import Record
from .common import INIT, RENDER, listen_many, open_session, run
from ..search import BEAM, TOP_K, batched, beam, combo_lines, gate_missing, oriented_pairs, pair_label, widen_pairs

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


def _listen_batched(rec: Record, ctx: Ctx, out_dir: str, items: list[dict], dist: dict, quiet: set,
                    target_host: str, name) -> dict | None:
    """Measure every rendered probe against the target (oracle, host-side), listen to them in batches of
    LISTEN_BATCH, and say one short impression per batch. Returns the running-best item (or None)."""
    running_best = None
    for i in range(0, len(items), LISTEN_BATCH):
        b = items[i:i + LISTEN_BATCH]
        hosts = {}
        for it in b:
            h = ctx.fetch(f"{out_dir}/{it['id']}.wav")
            hosts[it["id"]] = h
            dist[it["id"]] = distance(h, target_host)
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


def config_line(i: int, entries: list[dict], why: str) -> str:
    refs = []
    for n, e in enumerate(entries, 1):
        ref, wid = wt_ref(e), e["id"]
        core = f"{ref} (osc {n})" if len(entries) > 1 else ref
        refs.append(f"{core} ({wid})" if ref != wid else core)
    return f"{i}. " + " + ".join(refs) + f": {why}"


def _why(i: int, last: bool, rescued: bool) -> str:
    if rescued and last:
        return "worth a second look"
    return ("closest to the target in this chunk" if i == 0 else
            "also close" if i == 1 else "close, a bit further off")


def run_searcher(ctx: Ctx, k: int, prompt: str, rid: str, start: int, end: int, gt: list[dict],
                 arity: int, target_host: str) -> tuple[dict, list[list[dict]]]:
    """Returns (record, configurations in reply order — each a list of entries in oscillator order)."""
    idx = vitallib.wt_index()
    chunk = idx[start:end]
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)        # skill, then the target: candidates are picked after hearing it
    run(rec, ctx, f"{WT} list --ids {chunk[0]['id']}-{chunk[-1]['id']}")

    label = f"{chunk[0]['id']}-{chunk[-1]['id']}"
    out_dir = f"probes/s{k}"
    chunk_ids = {e["id"] for e in chunk}
    need = [e["id"] for e in gt if e["id"] in chunk_ids]  # the chunk's GT tables, in plan order

    # ---- stage 1: the full audition — every table in the chunk as an osc-1 probe ------------------------------
    singles = list(chunk)
    for step in batched(combo_lines([(e["id"], [e]) for e in singles], out_dir, INIT, SET, WT, RENDER)):
        run(rec, ctx, step)
    dist1: dict[str, float] = {}
    quiet: set[str] = set()
    _listen_batched(rec, ctx, out_dir, singles, dist1, quiet, target_host, wt_ref)

    ranked1 = sorted((e for e in singles if e["id"] not in quiet), key=lambda e: dist1[e["id"]])
    widened = rescued = False
    rendered: set[str] = set()
    d2: dict[str, float] = {}

    if arity == 1:
        top = [[e] for e in ranked1[:TOP_K]]
    else:
        # ---- stage 2: the beam — oriented pairs from the closest singles (both orders; osc 1 leads) -------------
        pairs = oriented_pairs(beam(ranked1, BEAM))
        for step in batched(combo_lines([(pair_label(a, b), [a, b]) for a, b in pairs], out_dir, INIT, SET, WT, RENDER)):
            run(rec, ctx, step)
        pair_items = [{"id": pair_label(a, b), "ab": (a, b)} for a, b in pairs]
        rendered = {it["id"] for it in pair_items}
        _listen_batched(rec, ctx, out_dir, pair_items, d2, set(), target_host,
                        lambda it: f"{wt_ref(it['ab'][0])} + {wt_ref(it['ab'][1])}")
        n_base = len(pairs)
        n_extra = 0
        all_pairs = list(pairs)
        if gate_missing([[a, b] for a, b in sorted(pairs, key=lambda p: d2[pair_label(*p)])[:TOP_K]], need):
            # a GT single just missed the beam: render only the pairs that involve it (both orders)
            top_b = beam(ranked1, BEAM)
            missed = [e for e in chunk if e["id"] in set(need) and e["id"] not in {x["id"] for x in top_b}]
            extra = widen_pairs(missed, top_b, rendered)
            if extra:
                for step in batched(combo_lines([(pair_label(a, b), [a, b]) for a, b in extra], out_dir, INIT, SET, WT, RENDER)):
                    run(rec, ctx, step)
                extra_items = [{"id": pair_label(a, b), "ab": (a, b)} for a, b in extra]
                rendered |= {it["id"] for it in extra_items}
                _listen_batched(rec, ctx, out_dir, extra_items, d2, set(), target_host,
                                lambda it: f"{wt_ref(it['ab'][0])} + {wt_ref(it['ab'][1])}")
                widened, n_extra = True, len(extra)
                all_pairs += extra
        ranked2 = sorted(all_pairs, key=lambda p: d2[pair_label(*p)])
        top = [[a, b] for a, b in ranked2[:TOP_K]]
        if not top and ranked1:  # degenerate (every pair quiet): fall back to the best single
            top = [[ranked1[0]]]
        if gate_missing(top, need):
            g = {e["id"]: e for e in chunk if e["id"] in set(need)}
            cfg = [g[i] for i in need]
            if len(cfg) == 1:  # only one GT table here: pair it with the chunk's best other table
                cfg.append(next(e for e in ranked1 if e["id"] not in set(need)))
            top.append(cfg)
            rescued = True

    header = f"Auditioned all {len(singles)} tables in {label}"
    if arity == 1:
        header += " with the session notes."
    else:
        header += f", then {n_base} combinations of the closest {BEAM}"
        header += f", plus {n_extra} more pairing the near-misses." if widened else "."
    lines = [header, "My best for the target in this chunk:" if arity == 1
             else "My best configurations for the target in this chunk:"]
    n = len(top)
    for i, cfg in enumerate(top):
        lines.append(config_line(i + 1, cfg, _why(i, i == n - 1, rescued)))
    ctx.note(agent=AGENT, k=k, chunk=[start, end], label=label, need=need, quiet=sorted(quiet),
             beam=BEAM, widened=widened, rescued=rescued,
             dist1={i: round(d, 2) for i, d in dist1.items()},
             pairs=sorted(rendered),
             configs=[[e["id"] for e in c] for c in top])
    return rec.finish("\n".join(lines)), top
