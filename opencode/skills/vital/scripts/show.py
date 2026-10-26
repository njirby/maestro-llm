#!/usr/bin/env python3
"""Show what is live in a Vital preset: on-gates, signal routing, modulation, values in human units.
Patch state only (reads preset.json) - it does not render or measure audio.

  show.py preset.json                 overview of everything that can currently be heard
  show.py preset.json SECTION         every parameter of one section: osc | sample | env | filter | lfo | random |
                                      mod | fx | macro | global
  show.py preset.json KEY_PREFIX      parameters whose key starts with the prefix (e.g. filter_1_, reverb_,
                                      osc_2_unison) with range, enum labels and why a key has no effect

Parameter lines read `key = <raw> (<human>)`; `*` marks a non-default value; `! <reason>` marks a key that cannot
affect the sound in this patch (module off, mode not selected, nothing routed into it, ...).
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitallib import (FX, FX_MIX, UsageError, _on, human, live_routes, load_preset, params, run_main,  # noqa
                      why_dead)

MODULES = {
    "osc": [f"osc_{n}_" for n in (1, 2, 3)],
    "sample": ["sample_"],
    "env": [f"env_{n}_" for n in range(1, 7)],
    "filter": ["filter_1_", "filter_2_", "filter_fx_"],
    "lfo": [f"lfo_{n}_" for n in range(1, 9)],
    "random": [f"random_{n}_" for n in range(1, 5)],
    "fx": [f"{fx}_" for fx in FX],
    "macro": [f"macro_control_{n}" for n in range(1, 5)],
}
GATE = {"osc_1_": "osc_1_on", "osc_2_": "osc_2_on", "osc_3_": "osc_3_on", "sample_": "sample_on",
        "filter_1_": "filter_1_on", "filter_2_": "filter_2_on", "filter_fx_": "filter_fx_on",
        **{f"{fx}_": f"{fx}_on" for fx in FX}}


def val(s, k):
    return s.get(k, params()[k]["default"])


def fmt(s, k):
    v = val(s, k)
    raw, h = f"{float(v):g}", human(k, v)
    star = "*" if abs(float(v) - params()[k]["default"]) > 1e-6 else " "
    out = f"{star} {k} = {raw}" + ("" if h == raw else f" ({h})")
    dead = why_dead(s, k)
    return out + (f"  ! {dead}" if dead else "")


def keys_with_prefix(prefix):
    return sorted(k for k in params() if k.startswith(prefix))


def wt_name(s, n):
    w = s.get("wavetables", [{}] * 3)
    return (w[n - 1] or {}).get("name") or "(unnamed)"


def lfo_shape(s, n):
    lfos = s.get("lfos") or []
    return (lfos[n - 1] or {}).get("name", "?") if n - 1 < len(lfos) else "?"


def routes_from(s, src):
    return [f"{r['destination']} {r['amount'] * 100:+.0f}%" for r in live_routes(s)
            if r["source"] == src and why_dead(s, r["destination"]) is None]


def overview(s):
    out = []
    for n in (1, 2, 3):
        if not _on(s, f"osc_{n}_on"):
            out.append(f"osc {n}: off")
            continue
        d = why_dead(s, f"osc_{n}_wave_frame")
        line = (f"osc {n}: {wt_name(s, n)!r} {human('osc_1_wave_frame', val(s, f'osc_{n}_wave_frame'))}, "
                f"level {human('osc_1_level', val(s, f'osc_{n}_level'))}, "
                f"transpose {human('osc_1_transpose', val(s, f'osc_{n}_transpose'))}")
        if float(val(s, f"osc_{n}_unison_voices")) > 1:
            line += (f", unison {val(s, f'osc_{n}_unison_voices'):g} voices detune "
                     f"{human('osc_1_unison_detune', val(s, f'osc_{n}_unison_detune'))}")
        if int(val(s, f"osc_{n}_distortion_type")):
            line += (f", {human(f'osc_{n}_distortion_type', val(s, f'osc_{n}_distortion_type'))} "
                     f"{human('osc_1_distortion_amount', val(s, f'osc_{n}_distortion_amount'))}")
        if int(val(s, f"osc_{n}_spectral_morph_type")):
            line += (f", {human('osc_1_spectral_morph_type', val(s, f'osc_{n}_spectral_morph_type'))} "
                     f"{human('osc_1_spectral_morph_amount', val(s, f'osc_{n}_spectral_morph_amount'))}")
        dest = int(val(s, f"osc_{n}_destination"))
        line += f" -> {human('osc_1_destination', dest)}"
        if dest in (0, 1, 2):
            fs = {0: ["1"], 1: ["2"], 2: ["1", "2"]}[dest]
            off = [f for f in fs if not _on(s, f"filter_{f}_on")]
            if off and len(off) == len(fs):
                line += " (filter off: passes unfiltered)"
        out.append(line + (f"  ! {d}" if d else ""))
    if _on(s, "sample_on"):
        smp = (s.get("sample") or {}).get("name", "?")
        out.append(f"sample: {smp!r}, level {human('sample_level', val(s, 'sample_level'))} -> "
                   f"{human('sample_destination', val(s, 'sample_destination'))}")
    else:
        out.append("sample: off")
    out.append("amp env (env 1): " + ", ".join(f"{p} {human(f'env_1_{p}', val(s, f'env_1_{p}'))}"
                                             for p in ("delay", "attack", "hold", "decay", "sustain", "release")))
    for f in ("1", "2", "fx"):
        if not _on(s, f"filter_{f}_on"):
            out.append(f"filter {f}: off")
            continue
        d = why_dead(s, f"filter_{f}_cutoff")
        out.append(f"filter {f}: {human('filter_1_model', val(s, f'filter_{f}_model'))} "
                   f"{human('filter_1_style', val(s, f'filter_{f}_style'))}, cutoff "
                   f"{human('filter_1_cutoff', val(s, f'filter_{f}_cutoff'))}, resonance "
                   f"{human('filter_1_resonance', val(s, f'filter_{f}_resonance'))}, drive "
                   f"{human('filter_1_drive', val(s, f'filter_{f}_drive'))}, mix "
                   f"{human('filter_1_mix', val(s, f'filter_{f}_mix'))}" + (f"  ! {d}" if d else ""))
    srcs = [f"env_{n}" for n in range(2, 7)] + [f"lfo_{n}" for n in range(1, 9)] + [f"random_{n}" for n in range(1, 5)]
    for src in srcs:
        r = routes_from(s, src)
        if not r:
            continue
        kind, n = src.split("_")
        if kind == "env":
            desc = ", ".join(f"{p} {human(f'env_{n}_{p}', val(s, f'env_{n}_{p}'))}"
                             for p in ("attack", "decay", "sustain", "release"))
        elif kind == "lfo":
            sync = int(val(s, f"lfo_{n}_sync"))
            rate = human(f"lfo_{n}_frequency", val(s, f"lfo_{n}_frequency")) if sync in (0, 4) else \
                human(f"lfo_{n}_tempo", val(s, f"lfo_{n}_tempo"))
            desc = (f"{lfo_shape(s, int(n))} shape, {human(f'lfo_{n}_sync', sync)} {rate}, "
                    f"{human(f'lfo_{n}_sync_type', val(s, f'lfo_{n}_sync_type'))}")
        else:
            sync = int(val(s, f"random_{n}_sync"))
            rate = human(f"random_{n}_frequency", val(s, f"random_{n}_frequency")) if sync == 0 else \
                human(f"random_{n}_tempo", val(s, f"random_{n}_tempo"))
            desc = f"{human(f'random_{n}_style', val(s, f'random_{n}_style'))}, {human(f'random_{n}_sync', sync)} {rate}"
        out.append(f"{src.replace('_', ' ')}: {desc} -> {', '.join(r)}")
    other = [f"{r['source']} -> {r['destination']} {r['amount'] * 100:+.0f}%" for r in live_routes(s)
             if not re.match(r"(env_[2-6]|lfo_\d|random_\d)$", r["source"]) and why_dead(s, r["destination"]) is None]
    if other:
        out.append("other mod routes: " + ", ".join(other))
    fx_on = []
    for fx in FX:
        if not _on(s, f"{fx}_on"):
            continue
        mix = FX_MIX.get(fx)
        fx_on.append(f"{fx} {human(mix, val(s, mix))} wet" if mix else fx)
    out.append("fx on: " + (", ".join(fx_on) if fx_on else "none"))
    out.append(f"global: volume {human('volume', val(s, 'volume'))}, polyphony {val(s, 'polyphony'):g}, "
               f"legato {human('legato', val(s, 'legato'))}, glide "
               f"{'none' if val(s, 'portamento_time') <= -10 else human('portamento_time', val(s, 'portamento_time'))}, "
               f"velocity_track {human('velocity_track', val(s, 'velocity_track'))}")
    return out


def section(s, name):
    if name == "mod":
        mods = s.get("modulations", [])
        out = []
        for i, m in enumerate(mods):
            if not (m.get("source") or m.get("destination")):
                continue
            amt = float(s.get(f"modulation_{i + 1}_amount", 0))
            d = why_dead(s, f"modulation_{i + 1}_amount") if amt else "amount is 0"
            out.append(f"slot {i + 1}  {m.get('source')} -> {m.get('destination')}  {amt * 100:+.0f}%"
                       + (f"  ! {d}" if d else ""))
        return out or ["no modulation routes"]
    if name == "global":
        used = tuple(p for ps in MODULES.values() for p in ps) + ("modulation_",)
        return [fmt(s, k) for k in sorted(params()) if not k.startswith(used)]
    out = []
    for prefix in MODULES[name]:
        gate = GATE.get(prefix)
        if gate and not _on(s, gate):
            out.append(f"  {gate} = 0 (Off)  [{len(keys_with_prefix(prefix)) - 1} more keys inactive]")
            continue
        if name in ("env", "lfo", "random") and prefix != "env_1_":
            dead = why_dead(s, keys_with_prefix(prefix)[0])
            if dead:
                out.append(f"  {prefix.rstrip('_')}: {dead}  [{len(keys_with_prefix(prefix))} keys inactive]")
                continue
        if name == "osc":
            out.append(f"  wavetable = {wt_name(s, int(prefix[4]))!r}")
        if name == "lfo":
            out.append(f"  shape = {lfo_shape(s, int(prefix[4]))!r}")
        out += [fmt(s, k) for k in keys_with_prefix(prefix)]
        if name == "macro":
            r = routes_from(s, prefix)
            out.append(f"    routes: {', '.join(r) if r else 'none'}")
    return out


def prefix_view(s, prefix):
    ks = keys_with_prefix(prefix)
    if not ks:
        raise UsageError(f"no parameter key starts with {prefix!r} (sections: {', '.join([*MODULES, 'mod', 'global'])})")
    out = []
    for k in ks:
        p = params()[k]
        rng = f"  [{p['min']:g}..{p['max']:g}]"
        if "labels" in p:
            rng = "  [" + ", ".join(f"{p['min'] + i:g}={lab}" for i, lab in enumerate(p["labels"])) + "]"
        out.append(fmt(s, k) + rng)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset")
    ap.add_argument("what", nargs="?", help="section or key prefix")
    a = ap.parse_args()
    s = load_preset(a.preset)["settings"]
    if not a.what:
        lines = overview(s)
    elif a.what in MODULES or a.what in ("mod", "global"):
        lines = section(s, a.what)
    else:
        lines = prefix_view(s, a.what)
    print("\n".join(lines))


if __name__ == "__main__":
    run_main(main)
