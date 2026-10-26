"""Build narration items for the three modes (and the calibration reference set) as preset files + renders.jsonl,
ready for render_in_container.sh and measure.py.

Modes (see facts.py / narrate.py):
- target_vs_current: a target preset T and a realistic mid-rollout recreation state. C0 = T with 1-3 subsystems
  (osc/env/filter/lfo/fx/mod) reverted toward init (maestro/synth/init_preset.json, fully or part-way) or perturbed by
  a producer move; the agent's last edit C0 -> C1 then moves 1-4 live parameters of one of those subsystems back toward
  T: restore exactly, part-way, overshoot past T, or (sometimes) the wrong way. Whether it actually got closer is
  measured, not assumed. Renders: t, t2 (floor), c0, c1.
- self_edit_delta: base preset A, edited B (edits.py moves: live / dead / tiny). Renders: a, a2 (floor), b.
- first_impression: a single preset. Render: t (CLAP only, no Omni frames needed).

  .venv/bin/python -m maestro.sounddiff.items probe --out outputs/sounddiff/probe1 --tvc 60 --sed 50 --fi 40
  .venv/bin/python -m maestro.sounddiff.items refset --out outputs/sounddiff/refset --n 480
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
from functools import lru_cache
from pathlib import Path

from maestro.synth.preset_gen import ARCHETYPES, _get_init_preset_json, generate_preset
from maestro.synth.wavetable_lib import load_wavetable_lib

from . import edits
from . import vital_facts as vf

SUBSYSTEMS = ("osc", "env", "filter", "lfo", "fx", "mod")
_PREFIX = {"osc": ("osc_", "sample_"), "env": ("env_",), "filter": ("filter_",), "lfo": ("lfo_", "random_"),
           "fx": tuple(f"{fx}_" for fx in vf.FX), "mod": ("modulation_",)}
# producer moves (edits.py) that perturb each subsystem
_MOVES = {"osc": [edits.m_unison_wide, edits.m_unison_off, edits.m_wavetable_swap, edits.m_frame, edits.m_octave,
                  edits.m_add_osc],
          "env": [edits.m_attack_slow, edits.m_attack_fast, edits.m_release_long, edits.m_release_short, edits.m_pluck],
          "filter": [edits.m_filter_close, edits.m_filter_open, edits.m_resonance],
          "fx": [edits.m_distortion, edits.m_reverb, edits.m_fx_off, edits.m_delay, edits.m_chorus],
          "mod": [edits.m_wobble], "lfo": [edits.m_wobble]}
_DISCRETE_SUFFIX = ("_on", "_type", "_model", "_style", "_voices", "_destination", "_sync", "_style", "_spectral_morph_type",
                    "_filter_input", "_stack_style", "_order")


@lru_cache(maxsize=1)
def init_settings() -> dict:
    return _get_init_preset_json()["settings"]


def params_of(s: dict, sub: str) -> list[str]:
    """Editable units of a subsystem: scalar settings keys, plus pseudo-params for list-valued state."""
    keys = [k for k, v in s.items() if isinstance(v, (int, float)) and k.startswith(_PREFIX[sub])]
    if sub == "osc":
        keys += ["osc_1_wavetable", "osc_2_wavetable", "osc_3_wavetable"]
    if sub == "mod":
        keys.append("mod_matrix")  # the route list; amounts are modulation_N_amount keys
    if sub == "lfo":
        keys.append("lfo_shapes")
    return keys


def _get(s, p):
    m = re.match(r"osc_(\d)_wavetable$", p)
    if m:
        return s["wavetables"][int(m.group(1)) - 1]
    if p == "mod_matrix":
        return s["modulations"]
    if p == "lfo_shapes":
        return s.get("lfos")
    return s.get(p)


def _put(s, p, v):
    v = copy.deepcopy(v)
    m = re.match(r"osc_(\d)_wavetable$", p)
    if m:
        s["wavetables"][int(m.group(1)) - 1] = v
    elif p == "mod_matrix":
        s["modulations"] = v
    elif p == "lfo_shapes":
        s["lfos"] = v
    else:
        edits._set(s, p, v)


def _discrete(p) -> bool:
    if p in ("mod_matrix", "lfo_shapes") or p.endswith("_wavetable"):
        return True
    r = vf.ranges().get(p)
    return bool(r and r.get("discrete")) or p.endswith(_DISCRETE_SUFFIX)


def _live_in(s, p) -> bool:
    m = re.match(r"osc_(\d)_wavetable$", p)
    if m:
        return vf.why_dead(s, f"osc_{m.group(1)}_wave_frame") is None
    if p == "mod_matrix":
        return bool(vf.live_routes(s))
    if p == "lfo_shapes":
        return any(r["source"].startswith("lfo_") for r in vf.live_routes(s))
    return vf.why_dead(s, p) is None


def live_diff(x: dict, y: dict, sub: str) -> list[str]:
    """Params of `sub` that differ between settings x and y and are audible in at least one of them."""
    out = []
    for p in params_of(y, sub):
        a, b = _get(x, p), _get(y, p)
        if a == b or (isinstance(a, float) and isinstance(b, float) and abs(a - b) < 1e-6):
            continue
        if p == "mod_matrix" and {(r["source"], r["destination"]) for r in vf.live_routes(x)} == \
                {(r["source"], r["destination"]) for r in vf.live_routes(y)}:
            continue
        if _live_in(x, p) or _live_in(y, p):
            out.append(p)
    return out


def revert(s: dict, sub: str, alpha: float, rng) -> None:
    """Move subsystem `sub` of settings s toward init by alpha (1 = all the way). Discrete params flip to init with
    probability alpha."""
    init = init_settings()
    for p in params_of(s, sub):
        a, b = _get(s, p), _get(init, p)
        if b is None or a == b:
            continue
        if _discrete(p) or not isinstance(a, (int, float)):
            if rng.random() < alpha:
                _put(s, p, b)
        else:
            _put(s, p, a + alpha * (b - a))


def perturb(s: dict, sub: str, rng, lib) -> None:
    rng.choice(_MOVES[sub])(s, rng, lib)


def move_back(c0: dict, t: dict, sub: str, rng, lib) -> tuple[dict, str, list[str]] | None:
    """The agent's last edit: 1-4 live differing params of `sub` moved from C0 toward T. Returns (C1, kind, params)."""
    cand = live_diff(c0, t, sub)
    if not cand:
        return None
    kind = rng.choices(["restore", "partial", "overshoot", "wrong"], weights=[45, 25, 15, 15])[0]
    ps = rng.sample(cand, min(len(cand), rng.choice([1, 1, 2, 3, 4])))
    c1 = copy.deepcopy(c0)
    for p in ps:
        a, b = _get(c0, p), _get(t, p)
        cont = isinstance(a, (int, float)) and isinstance(b, (int, float)) and not _discrete(p)
        if kind == "restore" or (not cont and kind != "wrong"):
            _put(c1, p, b)
        elif kind == "partial":
            _put(c1, p, a + rng.uniform(0.4, 0.7) * (b - a))
        elif kind == "overshoot":
            _put(c1, p, b + rng.uniform(0.5, 1.0) * (b - a))
        elif cont:  # wrong way
            _put(c1, p, a - rng.uniform(0.5, 1.0) * (b - a))
        elif p.endswith("_wavetable"):
            _put(c1, p, rng.choice(lib))
        elif p in ("mod_matrix", "lfo_shapes"):
            perturb(c1, sub if sub in _MOVES else "mod", rng, lib)
        else:
            r = vf.ranges().get(p, {"min": 0, "max": 1})
            opts = [v for v in range(int(r["min"]), int(r["max"]) + 1) if v not in (a, b)]
            _put(c1, p, rng.choice(opts) if opts else b)
    if not live_diff(c0, c1, sub) and not any(d["live"] for d in vf.diff(c0, c1)):
        return None
    return c1, kind, ps


def make_tvc(t_preset: dict, rng, lib):
    """(C0, C1, plan) for a target preset, or None if no usable state came out."""
    t = t_preset["settings"]
    for _ in range(20):
        c0 = copy.deepcopy(t)
        subs = rng.sample(SUBSYSTEMS, rng.choices([1, 2, 3], weights=[3, 4, 2])[0])
        how = {}
        for sub in subs:
            if rng.random() < 0.6:
                alpha = 1.0 if rng.random() < 0.6 else rng.uniform(0.5, 0.85)
                revert(c0, sub, alpha, rng)
                how[sub] = f"revert {alpha:.2f}"
            else:
                perturb(c0, sub, rng, lib)
                how[sub] = "perturb"
        touched = [sub for sub in subs if live_diff(c0, t, sub)]
        if not touched:
            continue
        sub = rng.choice(touched)
        mb = move_back(c0, t, sub, rng, lib)
        if mb is None:
            continue
        c1, kind, ps = mb
        return c0, c1, {"state": how, "edit_subsystem": sub, "edit_kind": kind, "edit_params": ps}
    return None


def next_subsystem(c1: dict, t: dict, plan: dict, rng) -> str | None:
    """The planned next edit from the ground truth: a subsystem that still differs audibly from the target after the
    last edit, preferring the ones the state was built from (what a rollout would fix next)."""
    left = [sub for sub in SUBSYSTEMS if live_diff(c1, t, sub)]
    pref = [sub for sub in left if sub in plan["state"]]
    return rng.choice(pref or left) if left else None


def _wrap(base: dict, settings: dict) -> dict:
    p = copy.deepcopy(base)
    p["settings"] = settings
    return p


def _clip_info(c):
    return {k: c[k] for k in ("midi_path", "track_idx", "start_s", "program")}


def build_probe(out: Path, n_tvc: int, n_sed: int, n_fi: int, seed: int, catalog: str, wavetables: str):
    rng = random.Random(seed)
    lib = load_wavetable_lib(wavetables)
    clips = edits.load_clips(catalog)
    (out / "presets").mkdir(parents=True, exist_ok=True)
    items, renders = [], []

    def save(iid, role, preset, notes, bpm, omni=True):
        preset["settings"]["beats_per_minute"] = bpm / 60.0
        (out / "presets" / f"{iid}_{role}.vital").write_text(json.dumps(preset, separators=(",", ":")))
        renders.append({"name": f"{iid}_{role}", "preset": f"presets/{iid}_{role}.vital", "notes": notes, "bpm": bpm,
                        "omni": omni})

    i = 0
    while sum(it["mode"] == "target_vs_current" for it in items) < n_tvc:
        t = generate_preset(rng.choice(ARCHETYPES), rng, wavetable_lib=lib)
        r = make_tvc(t, rng, lib)
        if r is None:
            continue
        c0, c1, plan = r
        plan["next_subsystem"] = next_subsystem(c1, t["settings"], plan, rng)
        c, notes = edits.pick_clip(clips, rng)
        iid = f"t{i:03d}"
        i += 1
        for role, s in (("t", t["settings"]), ("t2", t["settings"]), ("c0", c0), ("c1", c1)):
            save(iid, role, _wrap(t, copy.deepcopy(s)), notes, c["bpm"])
        items.append({"id": iid, "mode": "target_vs_current", "archetype": t["preset_style"], "bpm": c["bpm"],
                      "clip": _clip_info(c), "notes": notes, "plan": plan,
                      "wavs": {"target": f"{iid}_t", "target2": f"{iid}_t2", "before": f"{iid}_c0", "after": f"{iid}_c1"},
                      "edit": vf.diff(c0, c1), "remaining_before": vf.diff(c0, t["settings"]),
                      "remaining_after": vf.diff(c1, t["settings"]), "next_subsystem": plan["next_subsystem"],
                      "summary_target": vf.summary(t["settings"])})

    n_dead, n_tiny = round(n_sed * 0.15), round(n_sed * 0.08)
    kinds = ["dead"] * n_dead + ["tiny"] * n_tiny + ["live"] * (n_sed - n_dead - n_tiny)
    rng.shuffle(kinds)
    for j, kind in enumerate(kinds):
        a = generate_preset(rng.choice(ARCHETYPES), rng, wavetable_lib=lib)
        b = copy.deepcopy(a)
        moves = rng.sample(edits.LIVE, rng.choices([1, 2], weights=[7, 3])[0]) if kind == "live" else \
            [rng.choice(edits.DEAD if kind == "dead" else edits.TINY)]
        for mv in moves:
            mv(b["settings"], rng, lib)
        c, notes = edits.pick_clip(clips, rng)
        iid = f"e{j:03d}"
        for role, p in (("a", a), ("a2", a), ("b", b)):
            save(iid, role, copy.deepcopy(p), notes, c["bpm"])
        items.append({"id": iid, "mode": "self_edit_delta", "kind": kind, "moves": [m.__name__ for m in moves],
                      "archetype": a["preset_style"], "bpm": c["bpm"], "clip": _clip_info(c), "notes": notes,
                      "wavs": {"before": f"{iid}_a", "before2": f"{iid}_a2", "after": f"{iid}_b"},
                      "edit": vf.diff(a["settings"], b["settings"]), "summary_before": vf.summary(a["settings"])})

    for j in range(n_fi):
        t = generate_preset(ARCHETYPES[j % len(ARCHETYPES)], rng, wavetable_lib=lib)
        c, notes = edits.pick_clip(clips, rng)
        iid = f"f{j:03d}"
        save(iid, "t", t, notes, c["bpm"], omni=False)
        items.append({"id": iid, "mode": "first_impression", "archetype": t["preset_style"], "bpm": c["bpm"],
                      "clip": _clip_info(c), "notes": notes, "wavs": {"target": f"{iid}_t"},
                      "summary_target": vf.summary(t["settings"]), "settings_target": f"presets/{iid}_t.vital"})

    with open(out / "items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(out / "renders.jsonl", "w") as f:
        for r in renders:
            f.write(json.dumps(r) + "\n")
    print(f"{len(items)} items, {len(renders)} renders -> {out}")


def build_refset(out: Path, n: int, seed: int, catalog: str, wavetables: str):
    """Reference renders across archetypes for rank-calibrating absolute buckets and CLAP adjective z-scores."""
    rng = random.Random(seed)
    lib = load_wavetable_lib(wavetables)
    clips = edits.load_clips(catalog)
    (out / "presets").mkdir(parents=True, exist_ok=True)
    with open(out / "renders.jsonl", "w") as f:
        for j in range(n):
            arch = ARCHETYPES[j % len(ARCHETYPES)]
            p = generate_preset(arch, rng, wavetable_lib=lib)
            c, notes = edits.pick_clip(clips, rng)
            p["settings"]["beats_per_minute"] = c["bpm"] / 60.0
            name = f"r{j:04d}"
            (out / "presets" / f"{name}.vital").write_text(json.dumps(p, separators=(",", ":")))
            f.write(json.dumps({"name": name, "preset": f"presets/{name}.vital", "notes": notes, "bpm": c["bpm"],
                                "omni": False, "archetype": arch}) + "\n")
    print(f"{n} reference renders -> {out}/renders.jsonl")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["probe", "refset"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--tvc", type=int, default=60)
    ap.add_argument("--sed", type=int, default=50)
    ap.add_argument("--fi", type=int, default=40)
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--catalog", default="outputs/midi_clips/lakh_catalog_small.jsonl")
    ap.add_argument("--wavetables", default="data/wavetable_lib.json")
    a = ap.parse_args()
    if a.what == "probe":
        build_probe(Path(a.out), a.tvc, a.sed, a.fi, a.seed, a.catalog, a.wavetables)
    else:
        build_refset(Path(a.out), a.n, a.seed + 1000, a.catalog, a.wavetables)


if __name__ == "__main__":
    main()
