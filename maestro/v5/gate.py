"""Does a rollout land on the ground truth? Two gates on the final patch vs the GT patch.

structural (primary; both presets are known exactly):
  - every scalar key audible in the GT (maestro.sounddiff.vital_facts.why_dead(gt, k) is None; HOST_IGNORED aside;
    mod-slot scalars are compared as routes) matches the final within
    set.py's rounding: |final - gt| <= max(ABS_TOL, REL_TOL * |gt|)
  - live mod routes (vital_facts.live_routes whose destination is audible): same (source, destination) set, amounts
    within ABS_TOL
  - wavetables of the three oscillators: same content (hash of the table's groups)
  - LFO shapes of the LFOs those live routes use: identical
audio (secondary): renders through the GT notes on the rollout's session, host-side; GT_TAKES takes of the GT
  - mean mel_d(final, GT take) <= MEL_SCALE * floor + MEL_MARGIN_DB, floor = mean pairwise mel_d of the GT takes;
    same length
  - |rms(final) - mean rms(GT)| <= RMS_TOL_DB + rms spread of the GT takes   (catches level-only residuals)
  a structural pass with an audio fail is re-tested once on RETEST_TAKES GT takes and RETEST_FINAL final takes: on noisy
  patches (random phase, unison) 4 takes can put the floor well below its true value (pilot20f s0008: 4-take floor 1.62,
  12-take ~1.9, final within the GT's own take spread)
"""
from __future__ import annotations

import hashlib
import json

from maestro.sounddiff import vital_facts as vf

ABS_TOL, REL_TOL = 0.005, 0.01
MEL_SCALE, MEL_MARGIN_DB = 1.25, 0.3
RMS_TOL_DB = 0.5
GT_TAKES = 4
RETEST_TAKES, RETEST_FINAL = 12, 4
# keys vital_facts counts as live that cannot change the render: the host sets the tempo from notes.json (SKILL.md
# rule 7; the skill's vitallib.why_dead says so too)
HOST_IGNORED = {"beats_per_minute"}


def _wt_hash(t: dict) -> str:
    return hashlib.md5(json.dumps(t.get("groups"), sort_keys=True).encode()).hexdigest()


def _same_wt(a: dict, b: dict) -> bool:
    """Same table, or library tables whose frame-0 renders are identical (maestro/v5/wt_rank.same_table)."""
    ha, hb = _wt_hash(a), _wt_hash(b)
    if ha == hb:
        return True
    from .planner import _wt_by_hash
    from .wt_rank import same_table
    ea, eb = _wt_by_hash().get(ha), _wt_by_hash().get(hb)
    return bool(ea and eb and same_table(ea["id"], eb["id"]))


def _routes(s: dict) -> dict[tuple[str, str], float]:
    return {(r["source"], r["destination"]): float(r["amount"]) for r in vf.live_routes(s)
            if vf.why_dead(s, r["destination"]) is None}


def _live(g: dict, k: str) -> bool:
    return not isinstance(g[k], (list, dict)) and not k.startswith("modulation_") and k not in HOST_IGNORED \
        and vf.why_dead(g, k) is None


def structural(gt: dict, final: dict, ignore: set[str] = frozenset()) -> dict:
    """ignore: keys whose GT change was judged inaudible (wt_rank.audible_changes); they need not match."""
    g, f = gt["settings"], final["settings"]
    params = []
    for k, v in g.items():
        if not _live(g, k) or k in ignore:
            continue
        fv = f.get(k)
        if fv is None or abs(float(fv) - float(v)) > max(ABS_TOL, REL_TOL * abs(float(v))):
            params.append({"key": k, "gt": v, "final": fv})
    rg, rf = _routes(g), _routes(f)
    routes = sorted(f"{s}->{d}" for s, d in set(rg) ^ set(rf))
    routes += sorted(f"{s}->{d} amount {rg[(s, d)]:.3f} vs {rf[(s, d)]:.3f}" for s, d in set(rg) & set(rf)
                     if abs(rg[(s, d)] - rf[(s, d)]) > ABS_TOL)
    wts = [n + 1 for n in range(3) if not _same_wt(g["wavetables"][n], f["wavetables"][n])]
    lfos = sorted({int(s.split("_")[1]) for s, _ in rg if s.startswith("lfo_")})
    shapes = [n for n in lfos if g["lfos"][n - 1] != f["lfos"][n - 1]]
    return {"pass": not (params or routes or wts or shapes), "n_live_params": sum(_live(g, k) for k in g),
            "param_diffs": params[:20], "n_param_diffs": len(params), "route_diffs": routes,
            "wavetable_diffs": wts, "lfo_shape_diffs": shapes, "n_live_routes": len(rg)}


def audio(renderer, gt: dict, final: dict, notes: dict, takes: int = GT_TAKES, final_takes: int = 1) -> dict:
    """GT_TAKES renders of the GT (render-noise floor = mean pairwise mel distance; a single pair understates noisy
    patches) and one of the final; final-vs-GT = mean mel distance to the GT takes."""
    from itertools import combinations
    from maestro.sounddiff.verified import mel_d, rms_dbfs
    names = ["gt"] + [f"gt#{k}" for k in range(2, takes + 1)]
    fins = ["final"] + [f"final#{k}" for k in range(2, final_takes + 1)]
    w = renderer({**{n: gt for n in names}, **{f: final for f in fins}}, notes)
    floor = sum(mel_d(w[a], w[b]) for a, b in combinations(names, 2)) / (takes * (takes - 1) / 2)
    d = sum(mel_d(w[f], w[n]) for f in fins for n in names) / (takes * final_takes)
    r_gt = [rms_dbfs(w[n]) for n in names]
    r_mean, r_f = sum(r_gt) / takes, sum(rms_dbfs(w[f]) for f in fins) / final_takes
    limit = MEL_SCALE * floor + MEL_MARGIN_DB
    rms_limit = RMS_TOL_DB + (max(r_gt) - min(r_gt))
    ok = d <= limit and abs(r_f - r_mean) <= rms_limit and all(len(w[f]) == len(w[n]) for f in fins for n in names)
    return {"pass": bool(ok), "final_vs_gt_db": round(d, 3), "gt_vs_gt_db": round(floor, 3),
            "limit_db": round(limit, 3), "rms_final_dbfs": round(r_f, 2), "rms_gt_dbfs": round(r_mean, 2),
            "rms_diff_db": round(r_f - r_mean, 2), "rms_limit_db": round(rms_limit, 2), "gt_takes": takes, "final_takes": final_takes,
            "len_final": len(w["final"]), "len_gt": len(w["gt"])}


def same_in_context(renderer, gt: dict, final: dict, slots: list[int]) -> list[int]:
    """Of the wavetable-diff `slots`, those where the final tables sound identical to the GT's in the GT's own
    (stripped) patch: wt_rank distance < SAME_EPS with the final's tables swapped in. A table pair can be identical
    under one patch's oscillator settings without being on the fixed frame-0 duplicate list."""
    import copy
    from . import wt_rank
    base = wt_rank.stripped(gt)
    alt = copy.deepcopy(base)
    for n in slots:
        alt["settings"]["wavetables"][n - 1] = final["settings"]["wavetables"][n - 1]
    y = renderer({"gt": base, "alt": alt}, wt_rank.PATTERN)
    d = float(wt_rank.distance(wt_rank.sustain_frames(y["gt"]), wt_rank.sustain_frames(y["alt"])))
    return list(slots) if d < wt_rank.SAME_EPS else []


def gate(renderer, gt: dict, final: dict, notes: dict, ignore: set[str] = frozenset()) -> dict:
    st, au = structural(gt, final, ignore), audio(renderer, gt, final, notes)
    if ignore:
        st["ignored_inaudible"] = sorted(ignore)
    if st["wavetable_diffs"]:
        same = same_in_context(renderer, gt, final, st["wavetable_diffs"])
        if same:
            st["wavetables_same_in_context"] = same
            st["wavetable_diffs"] = [n for n in st["wavetable_diffs"] if n not in same]
            st["pass"] = not (st["n_param_diffs"] or st["route_diffs"] or st["wavetable_diffs"] or st["lfo_shape_diffs"])
    if st["pass"] and not au["pass"]:
        au = {**audio(renderer, gt, final, notes, RETEST_TAKES, RETEST_FINAL), "first_try": au}
    return {"pass": st["pass"] and au["pass"], "structural": st, "audio": au}
