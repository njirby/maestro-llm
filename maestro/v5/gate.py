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
"""
from __future__ import annotations

import hashlib
import json

from maestro.sounddiff import vital_facts as vf

ABS_TOL, REL_TOL = 0.005, 0.01
MEL_SCALE, MEL_MARGIN_DB = 1.25, 0.3
RMS_TOL_DB = 0.5
GT_TAKES = 4
# keys vital_facts counts as live that cannot change the render: the host sets the tempo from notes.json (SKILL.md
# rule 7; the skill's vitallib.why_dead says so too)
HOST_IGNORED = {"beats_per_minute"}


def _wt_hash(t: dict) -> str:
    return hashlib.md5(json.dumps(t.get("groups"), sort_keys=True).encode()).hexdigest()


def _routes(s: dict) -> dict[tuple[str, str], float]:
    return {(r["source"], r["destination"]): float(r["amount"]) for r in vf.live_routes(s)
            if vf.why_dead(s, r["destination"]) is None}


def _live(g: dict, k: str) -> bool:
    return not isinstance(g[k], (list, dict)) and not k.startswith("modulation_") and k not in HOST_IGNORED \
        and vf.why_dead(g, k) is None


def structural(gt: dict, final: dict) -> dict:
    g, f = gt["settings"], final["settings"]
    params = []
    for k, v in g.items():
        if not _live(g, k):
            continue
        fv = f.get(k)
        if fv is None or abs(float(fv) - float(v)) > max(ABS_TOL, REL_TOL * abs(float(v))):
            params.append({"key": k, "gt": v, "final": fv})
    rg, rf = _routes(g), _routes(f)
    routes = sorted(f"{s}->{d}" for s, d in set(rg) ^ set(rf))
    routes += sorted(f"{s}->{d} amount {rg[(s, d)]:.3f} vs {rf[(s, d)]:.3f}" for s, d in set(rg) & set(rf)
                     if abs(rg[(s, d)] - rf[(s, d)]) > ABS_TOL)
    wts = [n + 1 for n in range(3) if _wt_hash(g["wavetables"][n]) != _wt_hash(f["wavetables"][n])]
    lfos = sorted({int(s.split("_")[1]) for s, _ in rg if s.startswith("lfo_")})
    shapes = [n for n in lfos if g["lfos"][n - 1] != f["lfos"][n - 1]]
    return {"pass": not (params or routes or wts or shapes), "n_live_params": sum(_live(g, k) for k in g),
            "param_diffs": params[:20], "n_param_diffs": len(params), "route_diffs": routes,
            "wavetable_diffs": wts, "lfo_shape_diffs": shapes, "n_live_routes": len(rg)}


def audio(renderer, gt: dict, final: dict, notes: dict) -> dict:
    """GT_TAKES renders of the GT (render-noise floor = mean pairwise mel distance; a single pair understates noisy
    patches) and one of the final; final-vs-GT = mean mel distance to the GT takes."""
    from itertools import combinations
    from maestro.sounddiff.verified import mel_d, rms_dbfs
    names = ["gt"] + [f"gt#{k}" for k in range(2, GT_TAKES + 1)]
    w = renderer({**{n: gt for n in names}, "final": final}, notes)
    floor = sum(mel_d(w[a], w[b]) for a, b in combinations(names, 2)) / (GT_TAKES * (GT_TAKES - 1) / 2)
    d = sum(mel_d(w["final"], w[n]) for n in names) / GT_TAKES
    r_gt = [rms_dbfs(w[n]) for n in names]
    r_mean, r_f = sum(r_gt) / GT_TAKES, rms_dbfs(w["final"])
    limit = MEL_SCALE * floor + MEL_MARGIN_DB
    rms_limit = RMS_TOL_DB + (max(r_gt) - min(r_gt))
    ok = d <= limit and abs(r_f - r_mean) <= rms_limit and all(len(w["final"]) == len(w[n]) for n in names)
    return {"pass": bool(ok), "final_vs_gt_db": round(d, 3), "gt_vs_gt_db": round(floor, 3),
            "limit_db": round(limit, 3), "rms_final_dbfs": round(r_f, 2), "rms_gt_dbfs": round(r_mean, 2),
            "rms_diff_db": round(r_f - r_mean, 2), "rms_limit_db": round(rms_limit, 2), "gt_takes": GT_TAKES,
            "len_final": len(w["final"]), "len_gt": len(w["gt"])}


def gate(renderer, gt: dict, final: dict, notes: dict) -> dict:
    st, au = structural(gt, final), audio(renderer, gt, final, notes)
    return {"pass": st["pass"] and au["pass"], "structural": st, "audio": au}
