"""Sample (preset, edit, melody) jobs for the sound-delta probe.

Each job: a generated base preset A, an edited copy B, a ground-truth Lakh clip (notes + BPM from the MIDI tempo map).
Edits are producer moves ("open the filter", "add unison", "more reverb") plus deliberately dead ones (a change on a
module that is off or unrouted) and near-null ones (half a semitone of cutoff), which calibrate the audibility check.

  .venv/bin/python -m maestro.sounddiff.edits --n 60 --out outputs/sounddiff/probe0
"""
from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import pretty_midi

from maestro.synth.preset_gen import ARCHETYPES, generate_preset
from maestro.synth.wavetable_lib import load_wavetable_lib

from . import vital_facts as vf

MAX_CLIP_S = 8.0


def _set(s, k, v):
    r = vf.ranges().get(k)
    if r:
        v = min(max(v, r["min"]), r["max"])
    s[k] = float(v)


def _live_osc(s):
    return [n for n in (1, 2, 3) if vf.why_dead(s, f"osc_{n}_wave_frame") is None] or [1]


def _route_free_slot(s):
    return next(i for i, m in enumerate(s["modulations"]) if not m.get("source"))


# --- moves: (name, kind, fn(settings, rng, wtlib) -> None). kind: live | dead | tiny -------------------------------

def m_filter_close(s, rng, _):
    _set(s, "filter_1_on", 1.0)
    for n in _live_osc(s):
        s[f"osc_{n}_destination"] = 0.0
    _set(s, "filter_1_cutoff", rng.uniform(45, 70))


def m_filter_open(s, rng, _):
    if vf.why_dead(s, "filter_1_cutoff"):
        m_filter_close(s, rng, _)
    _set(s, "filter_1_cutoff", s["filter_1_cutoff"] + rng.uniform(18, 40))


def m_resonance(s, rng, _):
    if vf.why_dead(s, "filter_1_cutoff"):
        m_filter_close(s, rng, _)
    _set(s, "filter_1_resonance", rng.uniform(0.7, 0.95))


def m_attack_slow(s, rng, _):
    _set(s, "env_1_attack", rng.uniform(0.85, 1.05))  # ~0.5-1.2 s


def m_attack_fast(s, rng, _):
    _set(s, "env_1_attack", rng.uniform(0.0, 0.15))


def m_release_long(s, rng, _):
    _set(s, "env_1_release", rng.uniform(1.25, 1.5))  # ~2.5-5 s


def m_release_short(s, rng, _):
    _set(s, "env_1_release", rng.uniform(0.2, 0.4))


def m_pluck(s, rng, _):
    _set(s, "env_1_attack", 0.0)
    _set(s, "env_1_decay", rng.uniform(0.55, 0.75))
    _set(s, "env_1_sustain", rng.uniform(0.0, 0.1))


def m_unison_wide(s, rng, _):
    n = rng.choice(_live_osc(s))
    _set(s, f"osc_{n}_unison_voices", rng.choice([5, 7, 9]))
    _set(s, f"osc_{n}_unison_detune", rng.uniform(4.5, 7.0))
    _set(s, f"osc_{n}_stereo_spread", 1.0)


def m_unison_off(s, rng, _):
    for n in _live_osc(s):
        _set(s, f"osc_{n}_unison_voices", 1)


def m_wavetable_swap(s, rng, lib):
    n = rng.choice(_live_osc(s))
    s["wavetables"][n - 1] = copy.deepcopy(rng.choice(lib))


def m_frame(s, rng, _):
    n = rng.choice(_live_osc(s))
    _set(s, f"osc_{n}_wave_frame", (s[f"osc_{n}_wave_frame"] + rng.uniform(90, 160)) % 256)


def m_octave(s, rng, _):
    n = rng.choice(_live_osc(s))
    _set(s, f"osc_{n}_transpose", s[f"osc_{n}_transpose"] + rng.choice([-12, 12]))


def m_distortion(s, rng, _):
    _set(s, "distortion_on", 1.0)
    _set(s, "distortion_type", rng.choice([0, 1, 2, 3]))
    _set(s, "distortion_drive", rng.uniform(12, 24))
    _set(s, "distortion_mix", 1.0)


def m_reverb(s, rng, _):
    _set(s, "reverb_on", 1.0)
    _set(s, "reverb_dry_wet", rng.uniform(0.4, 0.6))
    _set(s, "reverb_decay_time", rng.uniform(1.0, 2.5))
    _set(s, "reverb_size", rng.uniform(0.6, 0.9))


def m_fx_off(s, rng, _):
    on = [fx for fx in vf.FX if s.get(f"{fx}_on", 0) > 0.5]
    for fx in on or ["reverb"]:
        s[f"{fx}_on"] = 0.0


def m_delay(s, rng, _):
    _set(s, "delay_on", 1.0)
    _set(s, "delay_dry_wet", rng.uniform(0.3, 0.5))
    _set(s, "delay_feedback", rng.uniform(0.4, 0.6))


def m_chorus(s, rng, _):
    _set(s, "chorus_on", 1.0)
    _set(s, "chorus_dry_wet", rng.uniform(0.5, 0.8))


def m_wobble(s, rng, _):
    if vf.why_dead(s, "filter_1_cutoff"):
        m_filter_close(s, rng, _)
    i = _route_free_slot(s)
    s["modulations"][i] = {"source": "lfo_8", "destination": "filter_1_cutoff"}
    _set(s, f"modulation_{i + 1}_amount", rng.uniform(0.4, 0.7))
    _set(s, f"modulation_{i + 1}_bypass", 0.0)
    _set(s, "lfo_8_sync", 0.0)  # seconds mode
    _set(s, "lfo_8_frequency", rng.uniform(1.5, 3.0))  # ~3-8 Hz -> period 2^-x s


def m_add_osc(s, rng, lib):
    off = [n for n in (2, 3) if s.get(f"osc_{n}_on", 0) < 0.5]
    if not off:
        return m_wavetable_swap(s, rng, lib)
    n = off[0]
    s["wavetables"][n - 1] = copy.deepcopy(rng.choice(lib))
    _set(s, f"osc_{n}_on", 1.0)
    _set(s, f"osc_{n}_level", rng.uniform(0.6, 0.9))
    _set(s, f"osc_{n}_transpose", rng.choice([-12, 0, 7, 12]))
    s[f"osc_{n}_destination"] = s.get("osc_1_destination", 0.0)


# dead moves: change something that cannot be heard in context
def d_filter_off(s, rng, _):
    _set(s, "filter_2_on", 0.0)
    _set(s, "filter_2_cutoff", s["filter_2_cutoff"] + rng.choice([-30, 30]))


def d_fx_off_param(s, rng, _):
    off = [fx for fx in ("reverb", "delay", "chorus", "phaser", "flanger") if s.get(f"{fx}_on", 0) < 0.5]
    fx = rng.choice(off or ["phaser"])
    s[f"{fx}_on"] = 0.0
    _set(s, vf.FX_MIX[fx], rng.uniform(0.6, 1.0))


def d_unrouted_lfo(s, rng, _):
    for i, m in enumerate(s["modulations"]):
        if m.get("source") == "lfo_7":
            s["modulations"][i] = {"source": "", "destination": ""}
    _set(s, "lfo_7_frequency", rng.uniform(-3, 6))


def d_unrouted_env(s, rng, _):
    for i, m in enumerate(s["modulations"]):
        if m.get("source") == "env_6":
            s["modulations"][i] = {"source": "", "destination": ""}
    _set(s, "env_6_attack", rng.uniform(0.8, 1.4))


def d_osc_off_param(s, rng, _):
    off = [n for n in (2, 3) if s.get(f"osc_{n}_on", 0) < 0.5]
    if not off:
        s["osc_3_on"] = 0.0
        off = [3]
    _set(s, f"osc_{off[0]}_transpose", rng.choice([-12, 12]))


# tiny moves: live but probably below what anyone hears
def t_cutoff_nudge(s, rng, _):
    if vf.why_dead(s, "filter_1_cutoff"):
        m_filter_close(s, rng, _)
        _set(s, "filter_1_cutoff", 90)
    _set(s, "filter_1_cutoff", s["filter_1_cutoff"] + rng.choice([-0.5, 0.5]))


def t_level_nudge(s, rng, _):
    n = rng.choice(_live_osc(s))
    _set(s, f"osc_{n}_level", s[f"osc_{n}_level"] * 0.97)


LIVE = [m_filter_close, m_filter_open, m_resonance, m_attack_slow, m_attack_fast, m_release_long, m_release_short,
        m_pluck, m_unison_wide, m_unison_off, m_wavetable_swap, m_frame, m_octave, m_distortion, m_reverb, m_fx_off,
        m_delay, m_chorus, m_wobble, m_add_osc]
DEAD = [d_filter_off, d_fx_off_param, d_unrouted_lfo, d_unrouted_env, d_osc_off_param]
TINY = [t_cutoff_nudge, t_level_nudge]


def clip_notes(c: dict) -> list[dict]:
    pm = pretty_midi.PrettyMIDI(c["midi_path"])
    t0, t1 = c["start_s"], c["start_s"] + min(c["duration_s"], MAX_CLIP_S)
    notes = [n for n in pm.instruments[c["track_idx"]].notes if t0 <= n.start < t1]
    return [{"pitch": n.pitch, "velocity": n.velocity, "start_s": round(n.start - t0, 4),
             "dur_s": round(min(n.end, t1) - n.start, 4)} for n in notes]


def load_clips(catalog: str) -> list[dict]:
    """Melodic, playable Lakh clips: >= 3 s, 6-60 notes, mid register, plausible tempo, not percussion/FX programs."""
    clips = [json.loads(l) for l in open(catalog)]
    return [c for c in clips if 3 <= c["duration_s"] and 6 <= c["n_notes"] <= 60 and 36 <= c["pitch_min"]
            and c["pitch_max"] <= 96 and 60 <= c["bpm"] <= 180 and c["program"] < 112]


def pick_clip(clips: list[dict], rng: random.Random) -> tuple[dict, list[dict]]:
    while True:
        c = rng.choice(clips)
        notes = clip_notes(c)
        if len(notes) >= 3:
            return c, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--catalog", default="outputs/midi_clips/lakh_catalog_small.jsonl")
    ap.add_argument("--wavetables", default="data/wavetable_lib.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    lib = load_wavetable_lib(args.wavetables)
    clips = load_clips(args.catalog)
    out = Path(args.out)
    (out / "presets").mkdir(parents=True, exist_ok=True)
    jobs = []
    n_dead, n_tiny = round(args.n * 0.15), round(args.n * 0.08)
    kinds = ["dead"] * n_dead + ["tiny"] * n_tiny + ["live"] * (args.n - n_dead - n_tiny)
    rng.shuffle(kinds)
    for j, kind in enumerate(kinds):
        a = generate_preset(rng.choice(ARCHETYPES), rng, wavetable_lib=lib)
        b = copy.deepcopy(a)
        if kind == "live":
            moves = rng.sample(LIVE, rng.choices([1, 2, 3], weights=[6, 3, 1])[0])
        else:
            moves = [rng.choice(DEAD if kind == "dead" else TINY)]
        for mv in moves:
            mv(b["settings"], rng, lib)
        c, notes = pick_clip(clips, rng)
        for p in (a, b):
            p["settings"]["beats_per_minute"] = c["bpm"] / 60.0
        jid = f"j{j:03d}"
        (out / "presets" / f"{jid}_a.vital").write_text(json.dumps(a, separators=(",", ":")))
        (out / "presets" / f"{jid}_b.vital").write_text(json.dumps(b, separators=(",", ":")))
        jobs.append({"id": jid, "kind": kind, "moves": [m.__name__ for m in moves], "archetype": a["preset_style"],
                     "bpm": c["bpm"], "clip": {k: c[k] for k in ("midi_path", "track_idx", "start_s", "program")},
                     "notes": notes, "diff": vf.diff(a["settings"], b["settings"]),
                     "summary_a": vf.summary(a["settings"]), "summary_b": vf.summary(b["settings"])})
    with open(out / "jobs.jsonl", "w") as f:
        for jb in jobs:
            f.write(json.dumps(jb) + "\n")
    print(f"{len(jobs)} jobs -> {out}/jobs.jsonl  kinds: " + ", ".join(f"{k} {kinds.count(k)}" for k in set(kinds)))


if __name__ == "__main__":
    main()
