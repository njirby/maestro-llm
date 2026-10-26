"""Fact builders for the three narration modes: measured audio + routing-aware patch diff -> the plain-text facts a
text-only narrator sees, plus the structured record the filter (score.py) checks narrations against.

API (the rollout builder calls these; narrate.py turns facts into text):

    from maestro.sounddiff.facts import Measured, facts_for
    meas = Measured("outputs/sounddiff/probe1")        # a measured dir: renders.jsonl, wav/, desc.jsonl, emb/
    f = facts_for("target_vs_current", item, meas)     # or "self_edit_delta", "first_impression"
    f["text"]        # facts block for the narration prompt
    f["supported"]   # {dim: direction or None}: what a narration may claim (score.check uses it)
    f["audibility"]  # clear / subtle / none (mode-specific: the edit, or current vs target)

`item` fields (items.py writes them; a rollout builder fills the same keys):
  common: notes [{pitch, velocity, start_s, dur_s}], bpm, archetype
  target_vs_current: wavs {target, target2 (second render of target: noise floor), before, after}, edit (vf.diff before->after),
                     next_subsystem (optional: osc/env/filter/lfo/fx/mod the planned next edit works on; the narration
                     then names the biggest measured gap that subsystem can address as the next move)
  self_edit_delta:   wavs {before, before2 (floor), after}, edit (vf.diff)
  first_impression:  wavs {target}, summary_target (vf.summary)

Every measured change is judged against a render-noise floor (the same preset rendered twice), so a change only counts
when it is bigger than the difference between two renders of the same patch. Absolute buckets and CLAP tone adjectives
are rank-calibrated on a reference set (calibrate.py -> calib.json).
"""
from __future__ import annotations

import json
import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import measure

CALIB_PATH = Path(__file__).with_name("calib.json")

# (key, label, how to compute the change, (slight, clear, big) thresholds, words for down/up, min |raw abs change|)
DIMS = [
    ("centroid_w_hz", "brightness", "log2", (0.15, 0.4, 0.9), ("darker", "brighter"), 0),
    ("harm_rich_db", "grit / harmonic richness", "diff", (3.0, 6.0, 12.0),
     ("smoother, fewer upper harmonics", "more upper harmonics: more bite / grit"), 0),
    ("flatness", "noisiness", "log2", (0.5, 1.0, 2.0), ("cleaner / more tonal", "noisier / breathier"), 0.01),
    ("loudness_db", "loudness", "diff", (1.5, 4.0, 8.0), ("quieter", "louder"), 0),
    ("width_db", "stereo width", "diff", (2.0, 5.0, 10.0), ("narrower", "wider"), 0),
    ("attack_ms", "attack", "attack", (15.0, 60.0, 200.0), ("snappier attack", "slower, softer attack"), 0),
    ("sustain_ratio", "sustain", "diff", (0.1, 0.25, 0.5), ("notes die away faster", "notes hold longer"), 0),
    ("tail_s", "tail after the last note", "diff", (0.3, 1.0, 2.5), ("shorter tail", "longer tail"), 0),
    ("gap_fill_db", "sound between notes", "diff", (3.0, 8.0, 15.0), ("drier gaps", "more sound ringing into the gaps"), 0),
    ("mod_cent", "tonal movement", "rdiff", (0.06, 0.15, 0.3), ("steadier tone", "more tonal movement / wobble"), 0),
    ("mod_amp", "volume movement", "rdiff", (1.0, 2.5, 5.0), ("steadier level", "more volume pulsing"), 0),
]
DIM_KEYS = {label: key for key, label, *_ in DIMS}
TONE = "tone / character"
# how the target compares with the current patch, per dim: (target lower, target higher); {s} = slightly/clearly/much
TARGET_PHRASE = {
    "brightness": ("is {s} darker", "is {s} brighter"),
    "grit / harmonic richness": ("is {s} smoother (fewer upper harmonics)", "has {s} more upper harmonics (more bite / grit)"),
    "noisiness": ("is {s} cleaner / more tonal", "is {s} noisier / breathier"),
    "loudness": ("is {s} quieter", "is {s} louder"),
    "stereo width": ("is {s} narrower", "is {s} wider"),
    "attack": ("has a {s} snappier attack", "has a {s} slower, softer attack"),
    "sustain": ("dies away {s} faster", "holds its notes {s} longer"),
    "tail after the last note": ("has a {s} shorter tail", "has a {s} longer tail"),
    "sound between notes": ("has {s} drier gaps between notes", "has {s} more sound ringing into the gaps"),
    "tonal movement": ("is {s} steadier in tone", "has {s} more tonal movement / wobble"),
    "volume movement": ("is {s} steadier in level", "has {s} more volume pulsing"),
}
WORDS = ("slightly", "clearly", "much")
AUD_TEXT = {"clear": "clearly audible", "subtle": "subtle - small but above render noise",
            "none": "none measurable - no bigger than two renders of the same patch"}


# --- measurements ------------------------------------------------------------------------------------------------------

class Measured:
    """Descriptors, CLAP adjective sims and pairwise distances of one measured dir (see measure.py)."""

    def __init__(self, d):
        self.d = Path(d)
        self.rows = measure.load_dir(self.d)
        self._pairs = {}

    def desc(self, name):
        return self.rows[name]["desc"]

    def adj_z(self, name):
        return adj_z(self.rows[name]["adj"])

    def dist(self, a, b):
        k = (a, b)
        if k not in self._pairs:
            self._pairs[k] = measure.pair(self.d, a, b)
        return self._pairs[k]


@lru_cache(maxsize=1)
def calib() -> dict:
    return json.loads(CALIB_PATH.read_text())


def adj_z(adj: dict) -> dict:
    """CLAP adjective similarity, centered over the vocabulary for this sound and z-scored per adjective against the
    reference set: > 1.5 means CLAP prefers this word for this sound unusually strongly."""
    c = calib()["adj"]
    v = np.array([adj[a] for a in measure.ADJECTIVES])
    v = v - v.mean()
    return {a: float((x - c[a]["mean"]) / c[a]["std"]) for a, x in zip(measure.ADJECTIVES, v)}


def pct(key: str, v, bands=False) -> float:
    """Percentile of value v for descriptor `key` in the reference set."""
    q = calib()["bands_q" if bands else "q"][key]
    return float(np.interp(v, q, calib()["pcts"]))


# --- changes ----------------------------------------------------------------------------------------------------------

def _change(kind, a, b):
    if a is None or b is None:
        return None
    if kind == "log2":
        return math.log2((b + 1e-9) / (a + 1e-9))
    if kind == "rdiff":  # difference, but only when it is also a >= 1.3x ratio (envelope reshaping leaks into these)
        return b - a if max(a, b) / max(min(a, b), 1e-6) >= 1.3 else 0.0
    if kind == "attack":  # ms difference, but small attacks are perceptually compressed
        return b - a if max(a, b) > 20 and max(a, b) / max(min(a, b), 1.0) > 1.4 else 0.0
    return b - a


def changes(da: dict, db: dict, fa: dict, fb: dict) -> list[dict]:
    """Dimensions that moved from descriptors da to db by more than the threshold and twice the floor (fa vs fb: two
    renders of the same preset). octave_offset never appears: the detector misfires on distortion/FX (probe0)."""
    out = []
    for key, label, kind, th, (down, up), abs_min in DIMS:
        ch = _change(kind, da.get(key), db.get(key))
        if ch is None:
            continue
        noise = abs(_change(kind, fa.get(key), fb.get(key)) or 0.0)
        if abs(ch) < max(th[0], 2 * noise) or abs(db[key] - da[key]) < abs_min:
            continue
        level = WORDS[sum(abs(ch) >= t for t in th) - 1]
        out.append({"dim": label, "dir": 1 if ch > 0 else -1, "direction": up if ch > 0 else down, "size": level,
                    "value": round(ch, 3), "noise": round(noise, 3)})
    return out


def excess(d: dict, f: dict) -> dict:
    """Distance minus the render-noise floor, per space."""
    return {"mel": d["mel_db"] - f["mel_db"], "clap": d["clap"] - f["clap"],
            "omni": d.get("omni_frames", 0.0) - f.get("omni_frames", 0.0),
            "omni_r": d.get("omni_frames", 0.0) / max(f.get("omni_frames", 0.0), 0.01)}


def audibility(d: dict, f: dict) -> str:
    """'clear' / 'subtle' / 'none' from distances vs the render-noise floor (thresholds from probe0 calibration)."""
    x = excess(d, f)
    if x["mel"] > 4 or (x["mel"] > 2 and (x["omni_r"] > 2 or x["clap"] > 0.05)):
        return "clear"
    if x["mel"] > 0.8 or x["omni_r"] > 1.5 or x["clap"] > 0.03:
        return "subtle"
    return "none"


def tone_score(d: dict, f: dict) -> float:
    """Overall timbre change from the embedding spaces (CLAP, Omni frames) beyond the floor; ~0.5 slight, 1.5 clear.
    Catches timbre moves the descriptors miss (probe0 j027: wavetable frame 0 -> 49%)."""
    x = excess(d, f)
    return max(x["clap"] / 0.06, x["omni"] / 0.08, 0.0)


def distance_score(d: dict, f: dict) -> float:
    """One number for 'how far apart' (floor-subtracted, each space scaled by its 'clear' threshold)."""
    x = excess(d, f)
    return (max(x["mel"], 0) / 4 + max(x["clap"], 0) / 0.05 + max(x["omni"], 0) / 0.08) / 3


def tone_change(d, f, aud, ch) -> dict | None:
    ts = tone_score(d, f)
    if aud == "none" or ts < 0.5:
        return None
    size = "much" if ts >= 4 else "clearly" if ts >= 1.5 else "slightly"
    return {"dim": TONE, "dir": 0, "direction": "different overall timbre", "size": size, "value": round(ts, 2)}


# adjectives that only restate a measured dimension (brightness, envelope, tail) are left to the descriptors, so a CLAP
# word can never contradict a measurement
NON_TONE = {"dark", "bright", "warm", "airy", "plucky", "percussive", "punchy", "spacey", "soft", "wobbly"}


def adj_shift(za: dict, zb: dict, k=1, thr=1.0) -> list[str]:
    """Tone words CLAP moved most from a to b (hedged; only adjectives that passed the calibration sanity check)."""
    drop = set(calib().get("adj_drop", [])) | NON_TONE
    dz = sorted(((zb[a] - za[a], a) for a in measure.ADJECTIVES if a not in drop), reverse=True)
    more = [f"more {a}" for v, a in dz[:k] if v >= thr]
    less = [f"less {a}" for v, a in dz[::-1][:k] if v <= -thr]
    return more + less


def top_adjectives(z: dict, k=2, thr=1.5) -> list[str]:
    drop = set(calib().get("adj_drop", [])) | NON_TONE
    return [a for v, a in sorted(((z[a], a) for a in measure.ADJECTIVES if a not in drop), reverse=True)[:k] if v >= thr]


# --- shared text pieces ---------------------------------------------------------------------------------------------

def melody_line(item: dict) -> str:
    notes = [n for n in item["notes"] if n["dur_s"] > 0]
    lo, hi = min(n["pitch"] for n in notes), max(n["pitch"] for n in notes)
    name = lambda p: "C C# D D# E F F# G G# A A# B".split()[p % 12] + str(p // 12 - 1)
    poly = any(a["start_s"] < b["start_s"] + b["dur_s"] and b["start_s"] < a["start_s"] + a["dur_s"]
               for i, a in enumerate(notes) for b in notes[i + 1:])
    avg = sum(n["dur_s"] for n in notes) / len(notes)
    return (f"{len(notes)} notes at {item['bpm']:.0f} BPM, range {name(lo)}-{name(hi)}, "
            f"{'chords/overlapping notes' if poly else 'single-note line'}, average note length {avg:.2f} s")


def edit_lines(edit: list[dict]) -> list[str]:
    out = []
    for d in edit:
        if d["live"]:
            note = "" if d["live_before"] else "  (not in the signal path before this edit)"
            out.append(f"- {d['param']}: {d['before']} -> {d['after']}{note}")
    for d in edit:
        if not d["live"]:
            out.append(f"- {d['param']}: {d['before']} -> {d['after']}  (NOT audible: {d['dead_reason']})")
    return out


def _order(ch: list[dict]) -> list[dict]:
    """Biggest first; a tone/character change leads when everything else is only slight, so a faint side-measurement
    (j027: 'slightly softer attack') cannot dominate a real timbre change."""
    rank = {w: i for i, w in enumerate(WORDS)}
    return sorted(ch, key=lambda c: (-rank[c["size"]], c["dim"] != TONE))


# --- expected effects of an edit (self_edit_delta) ------------------------------------------------------------------

# (param regex, [(dim, sign relative to the param's change sign: +1 same, -1 opposite, 0 either)]); first dim = main
EXPECT = [
    (r"filter_\w+_cutoff$", [("brightness", 1), ("grit / harmonic richness", 1)]),
    (r"filter_\w+_(resonance|drive)$", [(TONE, 0), ("brightness", 0), ("grit / harmonic richness", 1)]),
    (r"filter_\w+_on$", [("brightness", -1), (TONE, 0), ("grit / harmonic richness", -1), ("loudness", 0)]),
    (r"filter_\w+_(model|style|blend|keytrack|formant_\w+|mix)$", [(TONE, 0), ("brightness", 0)]),
    (r"osc_\d_(wave_frame|wavetable|distortion_\w+|spectral_morph_\w+|frame_spread)$",
     [(TONE, 0), ("brightness", 0), ("grit / harmonic richness", 0), ("noisiness", 0)]),
    (r"osc_\d_transpose$", [("brightness", 1), (TONE, 0)]),
    (r"osc_\d_(level|on)$|volume$|voice_amplitude", [("loudness", 1), (TONE, 0), ("brightness", 0)]),
    (r"osc_\d_(unison_\w+|stereo_spread|detune_\w+)$", [("stereo width", 1), (TONE, 0), ("tonal movement", 1),
                                                        ("loudness", 1), ("volume movement", 0)]),
    (r"env_1_attack$", [("attack", 1), ("loudness", -1), ("sound between notes", 0)]),
    (r"env_1_(decay|sustain|hold)$", [("sustain", 1), ("loudness", 1), ("sound between notes", 1)]),
    (r"env_1_release$", [("tail after the last note", 1), ("sound between notes", 1), ("sustain", 0)]),
    (r"reverb_\w+$", [("tail after the last note", 1), ("sound between notes", 1), ("stereo width", 1), (TONE, 0),
                      ("brightness", 0)]),
    (r"delay_\w+$", [("tail after the last note", 1), ("sound between notes", 1), ("stereo width", 1),
                     ("volume movement", 0)]),
    (r"chorus_\w+$", [("stereo width", 1), (TONE, 0), ("tonal movement", 1), ("volume movement", 0)]),
    (r"(flanger|phaser)_\w+$", [("tonal movement", 1), (TONE, 0), ("stereo width", 0), ("brightness", 0)]),
    (r"distortion_\w+$", [("grit / harmonic richness", 1), ("loudness", 1), ("brightness", 1), ("noisiness", 1),
                          (TONE, 0)]),
    (r"(compressor|eq)_\w+$", [("loudness", 0), (TONE, 0), ("brightness", 0)]),
    (r"^mod lfo_\d -> .*cutoff", [("tonal movement", 1), ("brightness", 0), ("loudness", 0), (TONE, 0)]),
    (r"^mod lfo_\d -> .*(level|amplitude|volume)", [("volume movement", 1), ("loudness", 0)]),
    (r"^mod (lfo|random)_\d -> ", [("tonal movement", 1), ("volume movement", 0), (TONE, 0)]),
    (r"^mod env_\d -> ", [("attack", 0), ("sustain", 0), ("brightness", 0), (TONE, 0)]),
    (r"^mod ", [(TONE, 0), ("tonal movement", 0)]),
    (r"(lfo|random)_\d_\w+$", [("tonal movement", 0), ("volume movement", 0)]),
    (r"env_[2-6]_\w+$", [("brightness", 0), (TONE, 0), ("attack", 0), ("sustain", 0)]),
    (r"portamento_\w+|legato", [(TONE, 0)]),
]


def expected_effects(edit: list[dict]) -> dict:
    """{dim: {"dir": +1/-1/0, "main": bool, "why": param}} from the live entries of a vf.diff."""
    out = {}
    for d in edit:
        if not d["live"]:
            continue
        for pat, effs in EXPECT:
            if re.match(pat, d["param"]):
                for i, (dim, rel) in enumerate(effs):
                    sgn = rel * d.get("sign", 0) if rel else 0
                    prev = out.get(dim)
                    if prev is None:
                        out[dim] = {"dir": sgn, "main": i == 0, "why": d["param"]}
                    else:
                        if prev["dir"] != sgn:
                            prev["dir"] = 0
                        prev["main"] |= i == 0
                break
    return out


def split_expected(ch: list[dict], exp: dict) -> list[dict]:
    for c in ch:
        e = exp.get(c["dim"])
        c["expected"] = bool(e) and (e["dir"] == 0 or c["dir"] == 0 or e["dir"] == c["dir"])
    return ch


# --- level gate, salience, tone-word consistency --------------------------------------------------------------------

SILENT_PEAK_DB, SILENT_RMS_DBFS, DROP_DB = -40.0, -60.0, 20.0
GATED_DIMS = {"loudness"}  # the only dimension still meaningful next to a near-silent render


def near_silent(d: dict) -> bool:
    return d["peak_db"] < SILENT_PEAK_DB or d.get("rms_dbfs", 0.0) < SILENT_RMS_DBFS


def level_gate(da: dict, db: dict, a="before", b="after") -> str | None:
    """A note when one side is (nearly) silent or the level falls/rises by >= 20 dB; then every ratio- or
    normalisation-based measurement (grit, noisiness, width, movement, sustain, CLAP/Omni tone, even brightness) is
    computed on the noise floor of the quiet side and is suppressed (probe1 e031: cutoff 186 Hz under B4-B5 notes gave
    bogus 'more grit, more wobble, narrower')."""
    sa, sb = near_silent(da), near_silent(db)
    ld = db.get("rms_dbfs", 0.0) - da.get("rms_dbfs", 0.0)
    if sa and sb:
        return f"{a} and {b} are both nearly silent - there is very little to hear in either"
    if sb or ld <= -DROP_DB:
        return f"{b}: nearly silent / almost gone compared with {a}"
    if sa or ld >= DROP_DB:
        return f"{a} was nearly silent; {b} is clearly there"
    return None


def _ratio(c: dict) -> float:
    if c["dim"] == TONE:
        return c["value"] / 0.5
    th = next(t for k, label, kind, t, *_ in DIMS if label == c["dim"])
    return abs(c["value"]) / th[0]


def salient(ch: list[dict], k: int = 3) -> list[dict]:
    """What the narrator hears about: the top k changes by size class then by magnitude relative to their threshold,
    plus any other 'much' change; slight extras are dropped once two clear/much changes are there."""
    rank = {w: i for i, w in enumerate(WORDS)}
    srt = sorted(ch, key=lambda c: (-rank[c["size"]], -_ratio(c)))
    keep = [c for i, c in enumerate(srt) if i < k or c["size"] == "much"]
    if sum(c["size"] != "slightly" for c in keep) >= 2:
        keep = [c for c in keep if c["size"] != "slightly"]
    return keep


# CLAP words that imply a measured dimension's direction: dropped when they contradict a measurement
ADJ_DIM = {"gritty": ("grit / harmonic richness", 1), "buzzy": ("grit / harmonic richness", 1),
           "harsh": ("grit / harmonic richness", 1), "distorted": ("grit / harmonic richness", 1),
           "smooth": ("grit / harmonic richness", -1), "clean": ("grit / harmonic richness", -1),
           "breathy": ("noisiness", 1), "thick": ("stereo width", 1), "lush": ("stereo width", 1)}


def consistent_words(words: list[str], measured: list[dict]) -> list[str]:
    """Keep 'more X' / 'less X' tone words whose implied direction does not contradict a measured change (e031: 'less
    gritty' next to 'more grit')."""
    dirs = {c["dim"]: c["dir"] for c in measured}
    out = []
    for w in words:
        sign, adj = (1, w[5:]) if w.startswith("more ") else (-1, w[5:]) if w.startswith("less ") else (1, w)
        dim = ADJ_DIM.get(adj)
        if dim and dirs.get(dim[0]) and dirs[dim[0]] != sign * dim[1]:
            continue
        out.append(w)
    return out


# --- modes ------------------------------------------------------------------------------------------------------------

def self_edit_delta(item: dict, m: Measured) -> dict:
    """The narrator sees the edit (context) and the salient measured changes only. Expected / surprise is computed
    here for metadata, stats and the report, never shown: the narration must come from the measurements."""
    w = item["wavs"]
    da, db, df = m.desc(w["before"]), m.desc(w["after"]), m.desc(w["before2"])
    d, f = m.dist(w["before"], w["after"]), m.dist(w["before"], w["before2"])
    aud = audibility(d, f)
    gate = level_gate(da, db) if aud != "none" else None
    ch = changes(da, db, da, df) if aud != "none" else []
    tc = None if gate else tone_change(d, f, aud, ch)
    if tc:
        ch.append(tc)
    if gate:
        ch = [c for c in ch if c["dim"] in GATED_DIMS]
    exp = expected_effects(item["edit"])
    ch = _order(split_expected(ch, exp))
    shown = salient(ch)
    measured = {c["dim"] for c in ch}
    missing = [dim for dim, e in exp.items() if e["main"] and dim not in measured and dim != TONE]
    live_edit = any(e["live"] for e in item["edit"])
    surprise = [c for c in ch if not c["expected"]]
    lines = [f"Sound: a {item['archetype'].lower()} patch in Vital playing {melody_line(item)}.", "",
             "Edit you made (before -> after):"] + edit_lines(item["edit"])
    lines += ["", "Overall difference between before and after renders: " + AUD_TEXT[aud]]
    if gate:
        lines.append("Level: " + gate)
    if shown:
        lines.append("What stands out most (after vs before, beyond render noise, biggest first):")
        lines += [f"- {c['dim']}: {c['size']} {c['direction']}" for c in shown]
        if tc and tc in shown:
            words = consistent_words(adj_shift(m.adj_z(w["before"]), m.adj_z(w["after"])), ch)
            if words:
                lines.append(f"  (tone words an audio-text model moved most, hedge these: {', '.join(words)})")
    elif aud != "none" and not gate:
        lines.append("Measured changes: none of the tracked dimensions moved beyond render noise; the difference is "
                     "small and in the overall timbre.")
    if live_edit and aud == "none":
        lines.append("The edit is in the signal path but made no measurable difference here.")
    return {"mode": "self_edit_delta", "text": "\n".join(lines), "audibility": aud, "gate": gate, "changes": ch,
            "shown": shown, "expected": exp, "missing": missing, "supported": {c["dim"]: c["dir"] for c in shown},
            "surprise": bool(surprise) or (live_edit and aud == "none") or (bool(missing) and aud != "none"),
            "n_surprise": len(surprise)}


def _dir_word(dim, sgn):
    for key, label, kind, th, (down, up), _ in DIMS:
        if label == dim:
            return up if sgn > 0 else down
    return ""


# which measured gaps each subsystem's next edit can address (target_vs_current next-step focus)
SUB_DIMS = {"osc": [TONE, "brightness", "grit / harmonic richness", "noisiness", "stereo width", "loudness"],
            "env": ["attack", "sustain", "tail after the last note", "sound between notes", "loudness"],
            "filter": ["brightness", "grit / harmonic richness", TONE],
            "lfo": ["tonal movement", "volume movement"], "mod": ["tonal movement", "volume movement", TONE],
            "fx": ["tail after the last note", "sound between notes", "stereo width", "grit / harmonic richness",
                   "tonal movement"]}
SUB_LABEL = {"osc": "the oscillators", "env": "the envelopes", "filter": "the filter", "lfo": "the LFOs",
             "mod": "the modulation", "fx": "the effects"}


def target_vs_current(item: dict, m: Measured) -> dict:
    w = item["wavs"]
    dt, dt2, d0, d1 = (m.desc(w[k]) for k in ("target", "target2", "before", "after"))
    f = m.dist(w["target"], w["target2"])
    dist0, dist1, dedit = m.dist(w["before"], w["target"]), m.dist(w["after"], w["target"]), m.dist(w["before"], w["after"])
    aud_now, aud_edit = audibility(dist1, f), audibility(dedit, f)
    gate_now, gate0 = level_gate(d1, dt, "your patch", "the target"), level_gate(d0, dt, "your patch", "the target")
    gate_edit = level_gate(d0, d1)
    s0, s1 = distance_score(dist0, f), distance_score(dist1, f)
    # gaps: how the target differs from the current patch (target relative to current), per dimension
    gap0 = {c["dim"]: c for c in changes(d0, dt, dt, dt2) if not gate0 or c["dim"] in GATED_DIMS}
    gap1 = {c["dim"]: c for c in changes(d1, dt, dt, dt2) if not gate_now or c["dim"] in GATED_DIMS} \
        if aud_now != "none" else {}
    tc = None if gate_now else tone_change(dist1, f, aud_now, [])
    if tc:
        gap1[TONE] = tc
    if not gate0 and (t0 := tone_change(dist0, f, audibility(dist0, f), [])):
        gap0[TONE] = t0
    if s1 < s0 * 0.85 and s0 - s1 > 0.15:
        traj = "closer"
    elif s1 > s0 * 1.15 and s1 - s0 > 0.15:
        traj = "further"
    else:
        traj = "same"
    rank = {w_: i + 1 for i, w_ in enumerate(WORDS)}
    per_dim = []
    for dim in sorted(set(gap0) | set(gap1)) if aud_edit != "none" else []:  # an inaudible edit moves no gap
        g0, g1 = gap0.get(dim), gap1.get(dim)
        r0, r1 = (rank[g0["size"]] if g0 else 0), (rank[g1["size"]] if g1 else 0)
        if g0 and g1 and g0["dir"] and g1["dir"] and g0["dir"] != g1["dir"]:
            per_dim.append((dim, "overshot", g0, g1, 9))
        elif r1 < r0:
            per_dim.append((dim, "closed" if r1 == 0 else "narrowed", g0, g1, r0 - r1))
        elif r1 > r0:
            per_dim.append((dim, "opened" if r0 == 0 else "widened", g0, g1, r1 - r0))
    per_dim = sorted(per_dim, key=lambda x: -x[4])[:2]
    shown = salient(list(gap1.values()))
    # next-step focus: the biggest measured remaining gap the planned next edit's subsystem can address
    nxt = item.get("next_subsystem")
    focus = next((c for c in _order(list(gap1.values())) if nxt and c["dim"] in SUB_DIMS[nxt]), None)
    if focus and focus not in shown:
        shown.append(focus)
    lines = [f"Sound: a {item['archetype'].lower()} patch in Vital playing {melody_line(item)}.", "",
             "You are recreating a target sound. Your last edit to your patch (before -> after):"] + edit_lines(item["edit"])
    lines += ["", "Your last edit on its own: " + AUD_TEXT[aud_edit] + (f" ({gate_edit})" if gate_edit else ""),
              "Your patch now vs the target: " + {"clear": "clearly different", "subtle": "close, but subtly different",
                                                  "none": "basically matches (no bigger than render noise)"}[aud_now]]
    if gate_now:
        lines.append("Level: " + gate_now)
    if shown:
        lines.append("Biggest differences between the TARGET and your current patch (beyond render noise):")
        lines += [f"- {c['dim']}: the target {_tphrase(c)}" for c in shown]
        if tc and tc in shown:
            words = consistent_words(adj_shift(m.adj_z(w["after"]), m.adj_z(w["target"])), list(gap1.values()))
            if words:
                lines.append(f"  (tone words, target vs yours, from an audio-text model; hedge these: {', '.join(words)})")
    lines.append("Did the last edit get you closer? " + {
        "closer": "yes - overall clearly closer to the target than before the edit",
        "further": "no - overall further from the target than before the edit",
        "same": "overall about the same distance from the target as before the edit"}[traj])
    for dim, how, g0, g1, _ in per_dim:
        was = f"before: the target {_tphrase(g0)}" if g0 else "no gap before"
        now = f"now: the target {_tphrase(g1)}" if g1 else "no gap now"
        lines.append(f"- {dim}: gap {how} ({was}; {now})")
    if focus:
        lines.append(f"Your next move: work on {SUB_LABEL[nxt]}, going after the {focus['dim']} difference.")
    supported = {c["dim"]: None for c in shown} | {dim: None for dim, *_ in per_dim}
    edit_ch = changes(d0, d1, dt, dt2) if aud_edit != "none" and not gate_edit else []
    for c in salient(edit_ch, 2):  # what the edit itself most obviously did is fair to mention
        supported.setdefault(c["dim"], None)
    if aud_edit != "none" and not gate_edit and tone_change(dedit, f, aud_edit, []):
        supported.setdefault(TONE, None)
    return {"mode": "target_vs_current", "text": "\n".join(lines), "audibility": aud_now, "edit_audibility": aud_edit,
            "gate": gate_now, "trajectory": traj, "score_before": round(s0, 3), "score_after": round(s1, 3),
            "gaps": list(gap1.values()), "shown": shown, "per_dim": [(d_, h) for d_, h, *_ in per_dim],
            "next_subsystem": nxt, "focus": focus["dim"] if focus else None, "supported": supported}


def _tphrase(c):
    if c["dim"] == TONE:
        return f"sounds {c['size']} different in overall timbre"
    return TARGET_PHRASE[c["dim"]][c["dir"] > 0].format(s=c["size"])


# --- first impression ------------------------------------------------------------------------------------------------

def buckets(desc: dict) -> dict:
    """Absolute description {trait: (text, distinctiveness 0-50)}, rank-calibrated against the reference set where the
    scale is arbitrary (brightness, grit, noise, width, movement) and in perceptual units where it is not (attack ms,
    sustain ratio, tail s). Distinctiveness is how far from typical the sound is on that trait."""
    out = {}
    if near_silent(desc):
        return {"level": ("very quiet, nearly silent", 50)}
    p = pct("centroid_w_hz", desc["centroid_w_hz"])
    out["brightness"] = ("dark" if p < 15 else "warm / rounded" if p < 35 else "balanced, neither dark nor bright"
                         if p < 65 else "bright" if p < 85 else "very bright / cutting", 0 if 35 <= p < 65 else abs(p - 50))
    if desc.get("harm_rich_db") is not None:
        p = pct("harm_rich_db", desc["harm_rich_db"])
        if p < 20:
            out["harmonics"] = ("smooth, few upper harmonics (sine-like)", 50 - p)
        elif p > 80:
            out["harmonics"] = ("lots of upper harmonics: buzzy / edgy", p - 50)
    p = pct("flatness", desc["flatness"])
    if p > 85 and desc["flatness"] > 0.02:
        out["noise"] = ("a noticeable noisy / breathy component", p - 50)
    att = desc.get("attack_ms") or 0
    out["attack"] = ("sharp, immediate", 15) if att < 20 else ("quick", 3) if att < 80 else ("soft", 25) if att < 300 \
        else ("slow swell", 40)
    sus = desc.get("sustain_ratio")
    if sus is not None:
        out["body"] = ("plucky, decays fast", 30) if sus < 0.3 else ("decays a bit", 3) if sus < 0.7 else ("sustained", 20)
    pw = pct("width_db", desc["width_db"])
    out["width"] = ("mono / centered", 12) if desc["width_db"] < -30 else ("narrow", 8) if pw < 35 \
        else ("some stereo width", 2) if pw < 75 else ("wide", pw - 50)
    out["ending"] = ("dry ending", 8) if desc["tail_s"] < 0.4 else ("short tail", 3) if desc["tail_s"] < 1.2 \
        else ("long tail / ambience", 35)
    mv = [k for k in ("mod_cent", "mod_amp") if desc.get(k) is not None and pct(k, desc[k]) > 85]
    if mv:
        out["movement"] = ("audible movement within notes (" + " and ".join(
            {"mod_cent": "tone", "mod_amp": "volume"}[k] for k in mv) + ")", max(pct(k, desc[k]) for k in mv) - 50)
    bands = desc["bands"]
    hi = [b for b in ("sub", "low", "mid", "high_mid", "air") if pct(b, bands[b], bands=True) > 85]
    if hi:
        out["spectral balance"] = ("unusually strong " + " and ".join(b.replace("_", "-") for b in hi) + " range",
                                   max(pct(b, bands[b], bands=True) for b in hi) - 50)
    return out


def _octave_layer(oscs: list[dict]) -> bool:
    """Two live oscillators an octave (or more) apart with levels within 6 dB: only then is a stacked octave
    plausibly heard (it comes from patch transposes, not from a measurement)."""
    lv = [(int(o["transpose"].replace("st", "").strip()), float(o["level"].rstrip("%"))) for o in oscs]
    lv = [(t, 20 * math.log10(l / 100)) for t, l in lv if l > 0]
    if len(lv) < 2:
        return False
    top = max(x for _, x in lv)
    near = sorted({t for t, x in lv if x >= top - 6})
    return any((b - a) % 12 == 0 and b != a for i, a in enumerate(near) for b in near[i + 1:])


def robust_patch_facts(s: dict) -> list[str]:
    """Patch facts a listener can reliably hear: unison stacks, octave layers, obvious FX. No wavetable names, warp
    types or mod-matrix dumps (not audible as such, and they invite parameter talk)."""
    out = []
    oscs = s["oscillators"]
    for o in oscs:
        v = int(o["unison"].split()[0])
        if v >= 3 and float(o.get("unison_detune", "0%").rstrip("%")) >= 8:
            out.append(f"a detuned unison stack ({v} voices) on one oscillator")
            break
    if _octave_layer(oscs):
        out.append("oscillators layered an octave (or more) apart")
    fx = s["fx"]
    pctv = lambda v: float(str(v).rstrip("%")) if str(v).rstrip("%").replace(".", "").isdigit() else 100.0
    for name, thr, word in (("reverb", 25, "reverb"), ("delay", 20, "an audible delay / echo"), ("chorus", 30, "chorus"),
                            ("distortion", 50, "distortion"), ("phaser", 30, "a phaser"), ("flanger", 30, "a flanger")):
        if name in fx and pctv(fx[name]) >= thr:
            out.append(word)
    return out


def first_impression(item: dict, m: Measured) -> dict:
    """Short facts: the 2-3 most distinctive traits, at most one (consistent) tone word and one patch feature."""
    w = item["wavs"]["target"]
    allb = buckets(m.desc(w))
    top = dict(sorted(((k, v) for k, v in allb.items() if v[1] > 5), key=lambda kv: -kv[1][1])[:3])
    b = {k: t for k, (t, _) in top.items()}
    gr = allb.get("harmonics", ("", 0))[0]
    adjs = [a for a in top_adjectives(m.adj_z(w))
            if not (ADJ_DIM.get(a, ("", 0))[0].startswith("grit") and gr and
                    (ADJ_DIM[a][1] > 0) != gr.startswith("lots"))][:1] if "level" not in allb else []
    pf = robust_patch_facts(item["summary_target"])[:1] if "level" not in allb else []
    lines = [f"Sound: a Vital patch playing {melody_line(item)}.", "", "What stands out in the render (compared with "
             "a large set of other synth patches):"] + [f"- {k}: {v}" for k, v in b.items()]
    if adjs:
        lines.append(f"- tone word an audio-text model rates unusually high for this sound (not certain, hedge it): "
                     f"{adjs[0]}")
    if pf:
        lines += ["", "Audible patch feature: " + pf[0]]
    return {"mode": "first_impression", "text": "\n".join(lines), "audibility": "n/a",
            "buckets": {k: t for k, (t, _) in allb.items()}, "shown": b, "adjectives": adjs, "patch_facts": pf,
            "supported": {k: None for k in b} | ({TONE: None} if adjs else {})}


MODES = {"target_vs_current": target_vs_current, "self_edit_delta": self_edit_delta, "first_impression": first_impression}


def facts_for(mode: str, item: dict, m: Measured) -> dict:
    """Facts for one item in one mode (see module docstring)."""
    f = MODES[mode](item, m)
    f["id"] = item.get("id")
    return f
