"""Quick test: where does the GT osc-1 wavetable rank among its 32-table chunk, by reference?

Probes are rendered as the searcher does (init patch, the table on osc 1, the session notes). References:
  target       the GT patch as heard (today's ranking)
  fx_off       the GT patch with every FX unit off
  fx_filt_off  ... and both filters off
  osc1_only    ... and only osc 1 on (upper bound: the probe's own situation)
Distances: mel (mean |log-mel diff|) and mel_norm (both clips scaled to the same RMS first).

  .venv/bin/python scripts/wt_rank_test.py --run outputs/v5_rollouts/pilot20d --n 20
"""
from __future__ import annotations

import argparse
import copy
import json
import statistics
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.sounddiff.verified import mel_d  # noqa: E402
from maestro.v5 import planner as pl  # noqa: E402

vl = pl.vitallib
INIT = json.loads((pl.SKILL_DIR / "data" / "init_preset.json").read_text())
CHUNK = 32


def variants(gt: dict) -> dict[str, dict]:
    s = gt["settings"]
    fx = copy.deepcopy(gt)
    for u in vl.FX:
        fx["settings"][f"{u}_on"] = 0.0
    filt = copy.deepcopy(fx)
    for k in ("filter_1_on", "filter_2_on", "filter_fx_on"):
        if k in filt["settings"]:
            filt["settings"][k] = 0.0
    one = copy.deepcopy(filt)
    for n in (2, 3):
        one["settings"][f"osc_{n}_on"] = 0.0
    out = {"target": gt, "fx_off": fx, "fx_filt_off": filt, "osc1_only": one}
    if not ABLATE:
        return out
    # osc-1 processing back to init, one group at a time and all together (test 2)
    i = INIT["settings"]

    def reset(base, keys):
        x = copy.deepcopy(base)
        for k in keys:
            x["settings"][k] = i[k]
        return x
    for name, keys in OSC_GROUPS.items():
        out[f"osc1_only-{name}"] = reset(one, keys)
    flat = [k for g, ks in OSC_GROUPS.items() if g != "frame" for k in ks]
    out["osc1_flat"] = reset(one, flat)
    out["osc1_flat_frame0"] = reset(one, flat + OSC_GROUPS["frame"])
    out["target-transpose"] = reset(gt, OSC_GROUPS["transpose"])
    out["target-osc_flat"] = reset(gt, flat)
    return out


ABLATE = False
PROBE = "init"
OSC_GROUPS = {
    "transpose": ["osc_1_transpose", "osc_1_tune"],
    "morph": ["osc_1_spectral_morph_type", "osc_1_spectral_morph_amount"],
    "dist": ["osc_1_distortion_type", "osc_1_distortion_amount"],
    "unison": ["osc_1_unison_voices", "osc_1_unison_detune", "osc_1_unison_blend"],
    "frame": ["osc_1_wave_frame", "osc_1_frame_spread"],
}


def probe(entry: dict, gt: dict | None = None, mode: str = "init") -> dict:
    """init: the searcher's probe today. sketch: init plus what a listener can hear and set first (the amp envelope,
    osc 1's transpose and unison). in_context: the GT patch with only the osc-1 table swapped."""
    if mode == "in_context":
        p = copy.deepcopy(gt)
        p["settings"]["wavetables"][0] = vl.wt_load(entry)
        return p
    p = copy.deepcopy(INIT)
    p["settings"]["wavetables"][0] = vl.wt_load(entry)
    if entry.get("silent_frame0"):
        p["settings"]["osc_1_wave_frame"] = 128.0
    if mode == "sketch":
        g = gt["settings"]
        for k in [k for k in g if k.startswith("env_1_")] + OSC_GROUPS["transpose"] + OSC_GROUPS["unison"]:
            if k in g and not isinstance(g[k], (list, dict)):
                p["settings"][k] = g[k]
    return p


def norm(y):
    r = np.sqrt((y ** 2).mean()) + 1e-12
    return y / r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "outputs/v5_rollouts/pilot20d")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/wt_rank/test1.json")
    ap.add_argument("--ablate", action="store_true", help="test 2: osc-1 processing reset to init, by group")
    ap.add_argument("--probe", choices=("init", "sketch", "in_context"), default="init", help="test 3: probe context")
    a = ap.parse_args()
    global ABLATE, PROBE
    ABLATE, PROBE = a.ablate, a.probe
    idx = vl.wt_index()
    by_hash = pl._wt_by_hash()
    rows = []
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with ContainerRenderer(ROOT / "outputs/wt_rank/render_cache", jobs=8) as r:
        r._session()
        for d in sorted((a.run / "oracle").iterdir())[: a.n]:
            gt = json.loads((d / "gt_preset.json").read_text())
            notes = json.loads((d / "notes.json").read_text())
            if float(gt["settings"].get("osc_1_on", 0)) <= 0.5:
                print(d.name, "osc 1 off in GT; skipped", flush=True)
                continue
            e = by_hash.get(pl._wt_hash(gt["settings"]["wavetables"][0]))
            if e is None:
                print(d.name, "GT osc-1 table not in library; skipped", flush=True)
                continue
            pos = next(i for i, x in enumerate(idx) if x["id"] == e["id"])
            lo = pos // CHUNK * CHUNK
            chunk = idx[lo:lo + CHUNK]
            refs = r.render(variants(gt), notes)
            probes = r.render({x["id"]: probe(x, gt, PROBE) for x in chunk}, notes)
            row = {"sid": d.name, "gt": e["id"], "chunk": [lo, lo + len(chunk)],
                   "n_osc": sum(float(gt["settings"].get(f"osc_{n}_on", 0)) > 0.5 for n in (1, 2, 3)),
                   "fx": [u for u in vl.FX if float(gt["settings"].get(f"{u}_on", 0)) > 0.5],
                   "filters": [k for k in ("filter_1_on", "filter_2_on") if float(gt["settings"].get(k, 0)) > 0.5],
                   "ranks": {}}
            for rn, ref in refs.items():
                for dn, f in (("mel", lambda x: x), ("mel_norm", norm)):
                    dist = {k: mel_d(f(ref), f(y)) for k, y in probes.items()}
                    order = sorted(dist, key=dist.get)
                    row["ranks"][f"{rn}/{dn}"] = order.index(e["id"]) + 1
            rows.append(row)
            print(d.name, e["id"], row["ranks"], flush=True)
            a.out.write_text(json.dumps(rows, indent=1))
    keys = list(rows[0]["ranks"]) if rows else []
    print(f"\n{len(rows)} samples, chunk {CHUNK}: rank of the GT osc-1 table")
    print(f"{'reference/distance':24s} top1  top3  top8  median")
    for k in keys:
        rk = [x["ranks"][k] for x in rows]
        print(f"{k:24s} {sum(v == 1 for v in rk):4d}  {sum(v <= 3 for v in rk):4d}  {sum(v <= 8 for v in rk):4d}  "
              f"{statistics.median(rk):6.1f}")


if __name__ == "__main__":
    main()
