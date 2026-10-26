"""Routing-aware view of a Vital preset: which parameters can actually be heard, and a human-unit diff of two presets.

A parameter change is only worth narrating if the parameter is live in the signal path. Dead cases (all from the
vital skill's section map, skills_dawdreamer/vital/SKILL.md): a module switched off (osc/sample/filter/FX `_on`), an
oscillator at level 0, an FX at dry/wet 0, a filter nothing is routed into, env 2-6 / LFOs / randoms with no live mod
route, a mod slot with amount 0 / bypassed / empty, sub-params of a mode that is not selected (distortion amount with
type None, formant controls without the formant model, unison detune with 1 voice). Oscillator FM/RM warps (checked by
render, sounddiff probe1/_fmtest): "osc A/B" are the other two oscillators in ascending order (osc 1: A=2, B=3); FM
from a source that is off does nothing (amount dead); RM from a source that is off crossfades the oscillator toward
silence (amount is live, at 100% the whole oscillator is muted); a source oscillator at level 0 still drives FM/RM, so
its own parameters stay live.
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

RANGES_PATH = Path(__file__).resolve().parents[1] / "synth" / "param_ranges.json"

FX = ("chorus", "delay", "reverb", "phaser", "flanger", "distortion", "eq", "compressor")
FX_MIX = {"chorus": "chorus_dry_wet", "delay": "delay_dry_wet", "reverb": "reverb_dry_wet", "phaser": "phaser_dry_wet",
          "flanger": "flanger_dry_wet", "distortion": "distortion_mix", "compressor": "compressor_mix"}
OSC_DEST = {0: "filter 1", 1: "filter 2", 2: "filters 1+2", 3: "effects", 4: "direct out"}
FILTER_MODELS = ["analog", "dirty", "ladder", "digital", "diode", "formant", "comb", "phaser"]
FILTER_STYLES = ["12dB", "24dB", "notch blend", "notch spread", "B/P/N"]
OSC_DISTORTION = ["none", "sync", "formant", "quantize", "bend", "squeeze", "pulse", "FM (osc A)", "FM (osc B)",
                  "FM (sample)", "RM (osc A)", "RM (osc B)", "RM (sample)"]


@lru_cache(maxsize=1)
def ranges() -> dict:
    return json.loads(RANGES_PATH.read_text())


def _on(s: dict, k: str) -> bool:
    return float(s.get(k, 0.0)) > 0.5


_WARP_SRC = {7: "A", 8: "B", 9: "sample", 10: "A", 11: "B", 12: "sample"}  # osc_N_distortion_type -> modulator


def warp_source(s: dict, n: int) -> tuple[str, bool] | None:
    """(label, is_on) of the FM/RM modulator of oscillator n, None if its warp is not FM/RM."""
    src = _WARP_SRC.get(int(s.get(f"osc_{n}_distortion_type", 0)))
    if src is None:
        return None
    if src == "sample":
        return "the sample source", _on(s, "sample_on")
    m = [k for k in (1, 2, 3) if k != n][0 if src == "A" else 1]
    return f"oscillator {m}", _on(s, f"osc_{m}_on")


def _is_warp_source(s: dict, n: int) -> bool:
    """Oscillator n modulates another live oscillator via FM/RM (then it matters even at level 0)."""
    for m in (1, 2, 3):
        if m == n or not _on(s, f"osc_{m}_on"):
            continue
        ws = warp_source(s, m)
        if ws and ws[0] == f"oscillator {n}" and (float(s.get(f"osc_{m}_distortion_amount", 0)) > 0
                                                  or _modulated(s, f"osc_{m}_distortion_amount")):
            return True
    return False


def live_routes(s: dict) -> list[dict]:
    """Mod-matrix slots that do something: source+destination set, amount != 0, not bypassed."""
    out = []
    for i, m in enumerate(s.get("modulations", [])):
        if not m.get("source") or not m.get("destination"):
            continue
        amt = float(s.get(f"modulation_{i + 1}_amount", 0.0))
        if amt == 0.0 or _on(s, f"modulation_{i + 1}_bypass"):
            continue
        out.append({"slot": i + 1, "source": m["source"], "destination": m["destination"], "amount": amt})
    return out


def _filter_fed(s: dict, f: int, depth: int = 0) -> bool:
    """Some live source is routed into filter f (1 or 2)."""
    want = {0, 2} if f == 1 else {1, 2}
    for n in (1, 2, 3):
        if _on(s, f"osc_{n}_on") and int(s.get(f"osc_{n}_destination", 0)) in want:
            return True
    if _on(s, "sample_on") and int(s.get("sample_destination", 3)) in want:
        return True
    other = 2 if f == 1 else 1
    return depth == 0 and _on(s, f"filter_{f}_filter_input") and _on(s, f"filter_{other}_on") and _filter_fed(s, other, 1)


def why_dead(s: dict, key: str, depth: int = 0) -> str | None:
    """None if `key` can affect the sound in preset settings `s`, else the reason it cannot."""
    if depth > 3:
        return None
    if key in ("bypass",) or key.startswith(("view_", "osc_1_view", "osc_2_view", "osc_3_view")) or key.endswith("_view_2d"):
        return "UI-only" if key != "bypass" else None
    m = re.match(r"osc_(\d)_(.+)", key)
    if m:
        n, p = m.group(1), m.group(2)
        if p != "on" and not _on(s, f"osc_{n}_on"):
            return f"oscillator {n} is off"
        if p not in ("on", "level") and float(s.get(f"osc_{n}_level", 0)) == 0 and not _modulated(s, f"osc_{n}_level") \
                and not _is_warp_source(s, int(n)):
            return f"oscillator {n} level is 0"
        ws = warp_source(s, int(n))
        if ws and not ws[1]:
            rm = int(s.get(f"osc_{n}_distortion_type", 0)) in (10, 11, 12)
            if rm and p not in ("on", "level", "distortion_type", "distortion_amount") \
                    and float(s.get(f"osc_{n}_distortion_amount", 0)) >= 0.999 \
                    and not _modulated(s, f"osc_{n}_distortion_amount"):
                return f"oscillator {n} is muted: ring-modulated at 100% by {ws[0]}, which is off"
            if p.startswith("distortion_") and p != "distortion_type" and (not rm or p != "distortion_amount"):
                return f"oscillator {n} {'RM' if rm else 'FM'} source ({ws[0]}) is off"
        if p.startswith("distortion_") and p != "distortion_type" and int(s.get(f"osc_{n}_distortion_type", 0)) == 0:
            return f"oscillator {n} warp/distortion type is None"
        if p.startswith("spectral_morph_") and p != "spectral_morph_type" and int(s.get(f"osc_{n}_spectral_morph_type", 0)) == 0:
            return f"oscillator {n} spectral morph type is None"
        if p.startswith(("unison_detune", "unison_blend", "unison_stack", "detune_", "stack_style", "spectral_unison")) \
                and float(s.get(f"osc_{n}_unison_voices", 1)) <= 1:
            return f"oscillator {n} has a single unison voice"
        return None
    if key.startswith("sample_") and key != "sample_on" and not _on(s, "sample_on"):
        return "sample source is off"
    m = re.match(r"filter_(1|2|fx)_(.+)", key)
    if m:
        f, p = m.group(1), m.group(2)
        if p != "on" and not _on(s, f"filter_{f}_on"):
            return f"filter {f} is off"
        if f in ("1", "2") and p != "on" and not _filter_fed(s, int(f)):
            return f"nothing is routed into filter {f}"
        if p.startswith("formant_") and int(s.get(f"filter_{f}_model", 0)) != 5:
            return f"filter {f} is not the formant model"
        if p == "comb_blend_offset" and int(s.get(f"filter_{f}_model", 0)) != 6:
            return f"filter {f} is not the comb model"
        return None
    for fx in FX:
        if key.startswith(fx + "_") and key != f"{fx}_on":
            if not _on(s, f"{fx}_on"):
                return f"{fx} is off"
            mix = FX_MIX.get(fx)
            if mix and key != mix and float(s.get(mix, 1.0)) == 0.0 and not _modulated(s, mix):
                return f"{fx} dry/wet is 0"
            return None
    m = re.match(r"(env|lfo|random)_(\d)_", key)
    if m:
        src = f"{m.group(1)}_{m.group(2)}"
        if src == "env_1":
            return None  # master amp envelope
        if not any(r["source"] == src and why_dead(s, r["destination"], depth + 1) is None for r in live_routes(s)):
            return f"{src.replace('_', ' ')} is not routed anywhere audible"
        return None
    m = re.match(r"modulation_(\d+)_(.+)", key)
    if m:
        i = int(m.group(1))
        mods = s.get("modulations", [])
        slot = mods[i - 1] if i - 1 < len(mods) else {}
        if not slot.get("source") or not slot.get("destination"):
            return f"mod slot {i} is empty"
        if m.group(2) != "amount" and float(s.get(f"modulation_{i}_amount", 0)) == 0:
            return f"mod slot {i} amount is 0"
        if m.group(2) != "bypass" and _on(s, f"modulation_{i}_bypass"):
            return f"mod slot {i} is bypassed"
        return why_dead(s, slot["destination"], depth + 1)
    if key.startswith("portamento_") and key != "portamento_on" and not _on(s, "portamento_on"):
        return "portamento is off"
    return None


def _modulated(s: dict, dest: str) -> bool:
    return any(r["destination"] == dest for r in live_routes(s))


def human(key: str, v) -> str:
    """Value in the units a producer would read it in (units table in the vital skill)."""
    if isinstance(v, (list, dict)) or v is None:
        return str(v)
    v = float(v)
    if re.search(r"(cutoff|_center)$", key):
        return f"{440 * 2 ** ((v - 69) / 12):.0f} Hz"
    if re.match(r"env_\d_(attack|decay|release|delay|hold)$", key):
        t = v ** 4
        return f"{t * 1000:.1f} ms" if t < 0.01 else f"{t * 1000:.0f} ms" if t < 1 else f"{t:.2f} s"
    if re.match(r"env_\d_sustain$", key) or key.endswith(("_level", "_mix", "_dry_wet", "_resonance", "_amount", "_blend")):
        return f"{v * 100:.0f}%"
    if key.endswith("_transpose") and key.startswith(("osc", "sample", "voice")):
        return f"{v:+.0f} st"
    if key.endswith("_unison_voices"):
        return f"{v:.0f} voices"
    if key.endswith("_unison_detune"):
        return f"{(v / 10) ** 2 * 100:.0f}%"
    if key.endswith("_drive") and key.startswith("filter"):
        return f"{v:.1f} dB"
    if key == "distortion_drive":
        return f"{v:+.1f} dB"
    if key in ("delay_feedback", "chorus_feedback", "flanger_feedback", "phaser_feedback") or key.endswith("_keytrack"):
        return f"{v * 100:+.0f}%"
    if key == "reverb_decay_time":
        return f"{2 ** v:.2f} s"
    if key == "reverb_size":
        return f"{v * 100:.0f}%"
    if re.match(r"(lfo|random)_\d_frequency$", key) or key in ("delay_frequency", "chorus_frequency"):
        return f"{2 ** (-v):.3g} s period"
    if key.endswith("_on"):
        return "on" if v > 0.5 else "off"
    if key.endswith("_destination"):
        return OSC_DEST.get(int(v), f"route {int(v)}")
    if re.match(r"filter_\w+_model$", key):
        return FILTER_MODELS[int(v)] if int(v) < len(FILTER_MODELS) else str(v)
    if re.match(r"filter_\w+_style$", key):
        return FILTER_STYLES[int(v)] if int(v) < len(FILTER_STYLES) else str(v)
    if key.endswith("_distortion_type"):
        return OSC_DISTORTION[int(v)] if int(v) < len(OSC_DISTORTION) else str(v)
    r = ranges().get(key)
    if r and r["max"] > r["min"] and not r.get("discrete"):
        return f"{(v - r['min']) / (r['max'] - r['min']) * 100:.0f}% of range"
    return f"{v:g}"


def _entry(param, before, after, dead_a, dead_b, sign=0) -> dict:
    """live_before/live_after: whether the value could be heard in A / in B. A change is live if either is.
    sign: +1 / -1 if the raw value went up / down (off->on and none->route count as up), 0 if not ordered."""
    return {"param": param, "before": before, "after": after, "live": dead_a is None or dead_b is None,
            "live_before": dead_a is None, "live_after": dead_b is None,
            "dead_reason": dead_b if (dead_a and dead_b) else None, "sign": sign}


def _sign(va, vb) -> int:
    if isinstance(va, (int, float)) and isinstance(vb, (int, float)):
        return 1 if vb > va else -1
    return 0


def _wt_names(s: dict) -> list[str]:
    return [w.get("name", "?") for w in s.get("wavetables", [])]


def diff(a: dict, b: dict) -> list[dict]:
    """Changed parameters between presets a and b (settings dicts), each tagged live/dead in context.

    A change is live if the parameter is live in either preset (turning reverb on: dead before, live after)."""
    out = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if k in ("modulations", "wavetables", "lfos", "sample") or va == vb:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and math.isclose(va, vb, rel_tol=1e-6, abs_tol=1e-9):
            continue
        out.append(_entry(k, human(k, va), human(k, vb), why_dead(a, k), why_dead(b, k), _sign(va, vb)))
    for i, (wa, wb) in enumerate(zip(_wt_names(a), _wt_names(b))):
        if wa != wb:
            k = f"osc_{i + 1}_wavetable"
            out.append(_entry(k, wa, wb, why_dead(a, f"osc_{i + 1}_wave_frame"), why_dead(b, f"osc_{i + 1}_wave_frame")))
    ra = {(r["source"], r["destination"]): r["amount"] for r in live_routes(a)}
    rb = {(r["source"], r["destination"]): r["amount"] for r in live_routes(b)}
    for sd in sorted(set(ra) | set(rb)):
        if ra.get(sd) != rb.get(sd):
            out.append(_entry(f"mod {sd[0]} -> {sd[1]}", f"{ra[sd] * 100:+.0f}%" if sd in ra else "none",
                              f"{rb[sd] * 100:+.0f}%" if sd in rb else "none",
                              why_dead(a, sd[1]) if sd in ra else "no route", why_dead(b, sd[1]) if sd in rb else "no route",
                              _sign(abs(ra.get(sd, 0.0)), abs(rb.get(sd, 0.0)))))
    # mod-slot keys are reported through the route view above
    return [d for d in out if not re.match(r"modulation_\d+_(amount|bypass)$", d["param"])]


def summary(s: dict) -> dict:
    """What is live in a preset, in human units: the facts for a one-preset ("what is this sound") description."""
    oscs = []
    for n in (1, 2, 3):
        if why_dead(s, f"osc_{n}_wave_frame") is None:
            o = {"osc": n, "wavetable": _wt_names(s)[n - 1], "level": human("osc_1_level", s[f"osc_{n}_level"]),
                 "transpose": human("osc_1_transpose", s[f"osc_{n}_transpose"]),
                 "unison": human("osc_1_unison_voices", s[f"osc_{n}_unison_voices"]),
                 "route": human("osc_1_destination", s[f"osc_{n}_destination"])}
            if float(s[f"osc_{n}_unison_voices"]) > 1:
                o["unison_detune"] = human("osc_1_unison_detune", s[f"osc_{n}_unison_detune"])
            if int(s.get(f"osc_{n}_distortion_type", 0)):
                o["warp"] = f"{human('osc_1_distortion_type', s[f'osc_{n}_distortion_type'])} " \
                            f"{human('osc_1_distortion_amount', s[f'osc_{n}_distortion_amount'])}"
            oscs.append(o)
    filters = []
    for f in ("1", "2", "fx"):
        if why_dead(s, f"filter_{f}_cutoff") is None:
            filters.append({"filter": f, "model": human("filter_1_model", s[f"filter_{f}_model"]),
                            "style": human("filter_1_style", s[f"filter_{f}_style"]),
                            "cutoff": human("filter_1_cutoff", s[f"filter_{f}_cutoff"]),
                            "resonance": human("filter_1_resonance", s[f"filter_{f}_resonance"])})
    amp = {p: human(f"env_1_{p}", s[f"env_1_{p}"]) for p in ("attack", "decay", "sustain", "release")}
    fx = {fx: human(FX_MIX[fx], s[FX_MIX[fx]]) if fx in FX_MIX else "on" for fx in FX
          if _on(s, f"{fx}_on") and why_dead(s, FX_MIX.get(fx, f"{fx}_on")) is None}
    routes = [f"{r['source']} -> {r['destination']} {r['amount'] * 100:+.0f}%" for r in live_routes(s)
              if why_dead(s, r["destination"]) is None]
    return {"oscillators": oscs, "filters": filters, "amp_envelope": amp, "fx": fx, "modulation": routes,
            "legato": _on(s, "legato"), "portamento": _on(s, "portamento_on")}
