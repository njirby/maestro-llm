"""Shared helpers for the vital skill scripts (python3 stdlib + numpy + soundfile + dawdreamer only).

Patch-state facts only: parameter table, human units, which parameters are live in the signal path, preset state
encoding for DawDreamer, and the vendored wavetable library. Nothing here measures audio.
"""
from __future__ import annotations

import gzip
import json
import math
import os
import re
import struct
import sys
from functools import lru_cache
from pathlib import Path

SKILL_DIR = Path(__file__).resolve().parents[1]
DATA = SKILL_DIR / "data"

FX = ("chorus", "delay", "reverb", "phaser", "flanger", "distortion", "eq", "compressor")
FX_MIX = {"chorus": "chorus_dry_wet", "delay": "delay_dry_wet", "reverb": "reverb_dry_wet", "phaser": "phaser_dry_wet",
          "flanger": "flanger_dry_wet", "distortion": "distortion_mix", "compressor": "compressor_mix"}
GATES = ("osc_1_on", "osc_2_on", "osc_3_on", "sample_on", "filter_1_on", "filter_2_on", "filter_fx_on",
         *(f"{fx}_on" for fx in FX))
MOD_SOURCES = (*(f"lfo_{i}" for i in range(1, 9)), *(f"random_{i}" for i in range(1, 5)),
               *(f"env_{i}" for i in range(1, 7)), *(f"macro_control_{i}" for i in range(1, 5)),
               "mod_wheel", "pitch_wheel", "velocity", "aftertouch", "note", "note_in_octave", "lift", "stereo", "random")
N_SLOTS = 64

# notes.json: the melody written as measures and note values (the builder's copy lives in maestro/notation.py;
# the two are pinned in step by tests/test_notation.py).
NOTE_VALUES = {
    "whole": 4.0, "dotted half": 3.0, "half": 2.0, "dotted quarter": 1.5, "quarter": 1.0,
    "dotted eighth": 0.75, "eighth": 0.5, "dotted sixteenth": 0.375, "sixteenth": 0.25,
    "dotted thirty-second": 0.1875, "thirty-second": 0.125, "triplet quarter": 2.0 / 3,
    "triplet eighth": 1.0 / 3, "triplet sixteenth": 1.0 / 6, "triplet thirty-second": 1.0 / 12,
}


def parse_time_signature(ts):
    """`W/X` -> (beats per bar, beat length in beats). 4/4 counts quarters; 6/8 is compound (dotted-quarter beat)."""
    ts = ts or "4/4"
    try:
        w, x = (int(t) for t in ts.split("/"))
    except ValueError:
        raise UsageError(f"time signature {ts!r} is not W/X") from None
    if x == 4 and 2 <= w <= 7:
        return w, 1.0
    if x == 8 and w in (6, 9, 12):
        return w // 3, 1.5
    raise UsageError(f"time signature {ts!r} unsupported (use 4/4, 3/4, 6/8, 9/8 or 12/8)")


class UsageError(Exception):
    """Bad input from the caller: printed as `error: ...` with exit code 2."""


def die(msg: str, code: int = 2):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(code)


def run_main(fn):
    try:
        sys.exit(fn() or 0)
    except UsageError as e:
        die(str(e))
    except BrokenPipeError:
        sys.exit(0)


# --------------------------------------------------------------------------- preset I/O

def load_preset(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_file():
        raise UsageError(f"{p}: no such preset file")
    try:
        preset = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise UsageError(f"{p}: not valid JSON ({e})")
    if not isinstance(preset, dict) or not isinstance(preset.get("settings"), dict):
        raise UsageError(f"{p}: not a Vital preset (no 'settings' object)")
    return preset


def save_preset(path: str | Path, preset: dict) -> None:
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(preset, separators=(",", ":")))
    os.replace(tmp, p)


def load_notes(path: str | Path) -> tuple[list[dict], float | None]:
    """notes.json -> (notes in seconds, bpm or None). The documented form is measures and note values
    ({"bpm", "time_signature", "notes": [{pitch, velocity, bar, beat, duration}]}); the legacy
    start_s/dur_s form is still accepted. A bare list of notes also works (120 bpm)."""
    p = Path(path)
    if not p.is_file():
        raise UsageError(f"{p}: no such notes file")
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise UsageError(f"{p}: not valid JSON ({e})")
    notes, bpm = (d, None) if isinstance(d, list) else (d.get("notes"), d.get("bpm"))
    if not isinstance(notes, list) or not notes:
        raise UsageError(f"{p}: needs a non-empty \"notes\" list")
    eff_bpm = float(bpm) if bpm is not None else 120.0
    bpb, beat_beats = parse_time_signature(None if isinstance(d, list) else d.get("time_signature"))
    spb = 60.0 / eff_bpm * beat_beats  # seconds per beat
    out = []
    for i, n in enumerate(notes):
        try:
            pitch, vel = int(n["pitch"]), int(n.get("velocity", 100))
            if "start_s" in n:  # legacy form
                start_s, dur_s = float(n["start_s"]), float(n["dur_s"])
            else:  # measures and note values
                bar, beat, val = int(n["bar"]), float(n["beat"]), str(n["duration"])
                if val not in NOTE_VALUES:
                    raise UsageError(f"{p}: note {i} has unknown note value {val!r} "
                                     f"(one of: {', '.join(NOTE_VALUES)})")
                if not (bar >= 1 and beat >= 1):
                    raise UsageError(f"{p}: note {i} starts before bar 1 beat 1: {n!r}")
                start_s = ((bar - 1) * bpb + (beat - 1)) * spb
                dur_s = NOTE_VALUES[val] * spb
        except UsageError:
            raise
        except (KeyError, TypeError, ValueError):
            raise UsageError(f"{p}: note {i} needs pitch and bar/beat/duration or start_s/dur_s "
                             f"(velocity optional): {n!r}")
        if not (0 <= pitch <= 127 and 0 <= vel <= 127 and dur_s > 0 and start_s >= 0):
            raise UsageError(f"{p}: note {i} out of range: {n!r}")
        out.append({"pitch": pitch, "velocity": vel, "start_s": start_s, "dur_s": dur_s})
    return out, (float(bpm) if bpm is not None else None)


# --------------------------------------------------------------------------- parameter table

@lru_cache(maxsize=1)
def params() -> dict:
    return json.loads((DATA / "params.json").read_text())


def _on(s: dict, k: str) -> bool:
    return float(s.get(k, 0.0)) > 0.5


def osc_others(n: int) -> tuple[int, int]:
    """The (A, B) oscillators an FM/RM distortion type on osc n reads from."""
    return n % 3 + 1, (n + 1) % 3 + 1


def label(key: str, v) -> str | None:
    p = params().get(key)
    if not p or "labels" not in p:
        return None
    i = int(round(float(v) - p["min"]))
    if not 0 <= i < len(p["labels"]):
        return None
    lab = p["labels"][i]
    m = re.match(r"osc_(\d)_distortion_type$", key)
    if m and lab.endswith(("Osc A", "Osc B")):
        a, b = osc_others(int(m.group(1)))
        lab = f"{lab[:-5]}osc {a if lab.endswith('A') else b}"
    return lab


def human(key: str, v) -> str:
    """Value in the units a producer reads (SKILL.md units table)."""
    if isinstance(v, (list, dict, str)) or v is None:
        return str(v)
    v = float(v)
    lab = label(key, v)
    if lab is not None:
        return lab
    if re.search(r"(cutoff|_center)$", key):
        return f"{440 * 2 ** ((v - 69) / 12):.0f} Hz"
    if re.match(r"env_\d_(attack|decay|release|delay|hold)$", key):
        t = v ** 4
        return f"{t * 1000:.1f} ms" if t < 0.01 else f"{t * 1000:.0f} ms" if t < 1 else f"{t:.2f} s"
    if key == "volume":
        return f"{math.sqrt(max(v, 0.0)) - 80:+.1f} dB"
    if key == "chorus_voices":
        return f"{v * 4:.0f} voices"
    if re.match(r"env_\d_sustain$", key) or key.endswith(("_level", "_mix", "_dry_wet", "_resonance", "_amount",
                                                            "_blend", "_size", "macro_control_1", "macro_control_2",
                                                            "macro_control_3", "macro_control_4")):
        return f"{v * 100:.0f}%"
    if key.endswith("_transpose"):
        return f"{v:+.0f} st"
    if key.endswith("_tune"):
        return f"{v * 100:+.0f} cents"
    if key.endswith("_unison_voices"):
        return f"{v:.0f} voices"
    if key.endswith("_unison_detune"):
        return f"{(v / 10) ** 2 * 100:.0f}%"
    if key.endswith("_detune_range"):
        return f"{v:.1f} cents"
    if key.endswith("_drive") and key.startswith("filter"):
        return f"{v:.1f} dB"
    if key == "distortion_drive" or key.endswith("_gain"):
        return f"{v:+.1f} dB"
    if key in ("delay_feedback", "chorus_feedback", "flanger_feedback", "phaser_feedback", "velocity_track") \
            or key.endswith("_keytrack"):
        return f"{v * 100:+.0f}%"
    if key in ("reverb_decay_time", "portamento_time") or re.match(r"lfo_\d_smooth_time$", key):
        return f"{2 ** v:.3g} s"
    if re.match(r"(lfo|random)_\d_frequency$", key) or key in ("delay_frequency", "delay_aux_frequency",
                                                                  "chorus_frequency", "phaser_frequency",
                                                                  "flanger_frequency", "reverb_chorus_frequency"):
        return f"{2 ** (-v):.3g} s period"
    if key.endswith(("_frame", "_wave_frame")):
        return f"frame {v:.0f}/256"
    p = params().get(key)
    if p and not p["discrete"] and (p["min"], p["max"]) == (0.0, 1.0):
        return f"{v * 100:.0f}%"
    return f"{v:g}"


def parse_value(key: str, text: str) -> float:
    """CLI value -> JSON value. Accepts a number, an enum label (case-insensitive), on/off, or a unit suffix:
    Hz (cutoff/center keys), ms / s (env times), % (0..1 params), dB (volume), st (transposes)."""
    p = params()[key]
    t = text.strip()
    tl = t.lower()
    if "labels" in p:
        labs = [lab.lower() for lab in p["labels"]]
        if tl in labs:
            return p["min"] + labs.index(tl)
        if tl in ("on", "off") and labs[:2] == ["off", "on"]:
            return p["min"] + labs.index(tl)
        m = re.match(r"osc_(\d)_distortion_type$", key)
        if m:  # "FM <- osc 2" style, as show.py prints it
            a, b = osc_others(int(m.group(1)))
            for i, lab in enumerate(p["labels"]):
                if lab.endswith(("Osc A", "Osc B")):
                    n = a if lab.endswith("A") else b
                    if tl.replace(" ", "") == f"{lab[:-5]}osc {n}".lower().replace(" ", ""):
                        return p["min"] + i
    if tl in ("on", "off") and p["discrete"] and p["min"] == 0 and p["max"] == 1:
        return 1.0 if tl == "on" else 0.0
    m = re.fullmatch(r"([-+]?\d*\.?\d+(?:e[-+]?\d+)?)\s*(hz|khz|ms|s|%|db|st)?", tl)
    if not m:
        opts = f"; labels: {', '.join(p['labels'])}" if "labels" in p else ""
        raise UsageError(f"{key}: cannot parse value {text!r}{opts}")
    x, unit = float(m.group(1)), m.group(2)
    if unit is None:
        return x
    if unit in ("hz", "khz") and re.search(r"(cutoff|_center)$", key):
        hz = x * (1000 if unit == "khz" else 1)
        if hz <= 0:
            raise UsageError(f"{key}: frequency must be > 0")
        return 69 + 12 * math.log2(hz / 440)
    if unit in ("ms", "s") and re.match(r"env_\d_(attack|decay|release|delay|hold)$", key):
        return (x / (1000 if unit == "ms" else 1)) ** 0.25
    if unit == "%" and (p["min"], p["max"]) in ((0.0, 1.0), (-1.0, 1.0)) and not p["discrete"]:
        return x / 100
    if unit == "db" and key == "volume":
        return (x + 80) ** 2
    if unit == "db" and (key.endswith("_drive") or key.endswith("_gain")):
        return x
    if unit == "st" and key.endswith("_transpose"):
        return x
    raise UsageError(f"{key}: unit {unit!r} does not apply to this key (give the raw JSON value)")


# --------------------------------------------------------------------------- routing-aware liveness

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


def _modulated(s: dict, dest: str) -> bool:
    return any(r["destination"] == dest for r in live_routes(s))


def _filter_fed(s: dict, f: int, depth: int = 0) -> bool:
    """Some live source is routed into filter f (1 or 2)."""
    want = {0, 2} if f == 1 else {1, 2}
    for n in (1, 2, 3):
        if _on(s, f"osc_{n}_on") and int(s.get(f"osc_{n}_destination", 0)) in want:
            return True
    if _on(s, "sample_on") and int(s.get("sample_destination", 3)) in want:
        return True
    other = 2 if f == 1 else 1
    return depth == 0 and _on(s, f"filter_{f}_filter_input") and _on(s, f"filter_{other}_on") \
        and _filter_fed(s, other, 1)


def why_dead(s: dict, key: str, depth: int = 0) -> str | None:
    """None if `key` can affect the sound in preset settings `s`, else the reason it cannot."""
    if depth > 3:
        return None
    if key.endswith("_view_2d") or key.startswith("view_"):
        return "UI-only"
    if key == "beats_per_minute":
        return "the host sets the tempo (notes.json \"bpm\" / render.py --bpm)"
    m = re.match(r"osc_(\d)_(.+)", key)
    if m:
        n, p = m.group(1), m.group(2)
        if p != "on" and not _on(s, f"osc_{n}_on"):
            return f"oscillator {n} is off"
        if p not in ("on", "level") and float(s.get(f"osc_{n}_level", 0)) == 0 and not _modulated(s, f"osc_{n}_level"):
            return f"oscillator {n} level is 0"
        if p.startswith("distortion_") and p != "distortion_type" and int(s.get(f"osc_{n}_distortion_type", 0)) == 0:
            return f"oscillator {n} distortion type is None"
        if p.startswith("spectral_morph_") and p != "spectral_morph_type" \
                and int(s.get(f"osc_{n}_spectral_morph_type", 0)) == 0:
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
            return f"filter {f} is not the Formant model"
        if p == "comb_blend_offset" and int(s.get(f"filter_{f}_model", 0)) != 6:
            return f"filter {f} is not the Comb model"
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
            return f"{src} is not routed anywhere audible"
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
    # portamento_on is not a gate: glide follows portamento_time alone (-10 = no glide); verified by render
    if key == "portamento_on":
        return "no audible effect (glide is set by portamento_time)"
    if key.startswith("portamento_") and key != "portamento_time" and float(s.get("portamento_time", -10.0)) <= -10.0:
        return "portamento time is -10 (no glide)"
    return None


# --------------------------------------------------------------------------- wavetable library

@lru_cache(maxsize=1)
def wt_index() -> list[dict]:
    return json.loads((DATA / "wavetables.idx.json").read_text())


def wt_load(entry: dict) -> dict:
    with open(DATA / "wavetables.bin", "rb") as f:
        f.seek(entry["offset"])
        return json.loads(gzip.decompress(f.read(entry["length"])))


def wt_resolve(ref: str) -> dict:
    """Library entry by id (wt042) or by exact name (case-insensitive; first variant if several)."""
    idx = wt_index()
    if re.fullmatch(r"wt\d{3}", ref):
        for e in idx:
            if e["id"] == ref:
                return e
        raise UsageError(f"no wavetable with id {ref}")
    hits = [e for e in idx if e["name"].lower() == ref.lower()]
    if not hits:
        near = [e["name"] for e in idx if ref.lower() in e["name"].lower()][:8]
        raise UsageError(f"no wavetable named {ref!r}" + (f"; did you mean: {', '.join(near)}" if near else
                                                         " (wavetable.py list --query ...)"))
    return hits[0]


# --------------------------------------------------------------------------- DawDreamer state encoding

_JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def _juce_b64(data: bytes) -> str:
    """JUCE MemoryBlock::toBase64Encoding — 'size.chars', 6 bits LSB-first."""
    size, out = len(data), []
    for i in range(((size * 8) + 5) // 6):
        bit = i * 6
        byte, off = bit >> 3, bit & 7
        word = data[byte] | (data[byte + 1] << 8) if byte + 1 < size else (data[byte] if byte < size else 0)
        out.append(_JUCE_B64[(word >> off) & 0x3F])
    return f"{size}.{''.join(out)}"


def build_state(preset: dict) -> bytes:
    """Vital VST3 state blob as Vital's own save_state writes it (full preset: wavetables, LFO shapes, mod matrix)."""
    json_bytes = json.dumps(preset, separators=(",", ":")).encode() + b"\x00"
    chunk = json_bytes + b"\x00" * 16 + b"JUCEPrivateData"
    body = 4 + 4 + 4 + 4 + 4 + 128 + 4 + len(chunk)
    icomp = (b"VstW" + struct.pack(">III", 8, 1, 0) + b"CcnK" + struct.pack(">i", body)
             + b"FBCh" + struct.pack(">i", 2) + b"Vita" + struct.pack(">i", 0x00010600)
             + struct.pack(">i", 0) + b"\x00" * 128 + struct.pack(">i", len(chunk)) + chunk)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           f"<VST3PluginState><IComponent>{_juce_b64(icomp)}</IComponent></VST3PluginState>").encode() + b"\x00"
    return struct.pack("<II", 0x21324356, len(xml)) + xml


def install_x11_error_handler() -> None:
    """Must run before importing dawdreamer: Vital's state load touches X11, and an unhandled X error aborts the
    process under Xvfb. Handlers are (Display*, XErrorEvent*) and (Display*); references are kept alive."""
    import ctypes
    import ctypes.util
    path = ctypes.util.find_library("X11")
    if not path:
        return
    libx11 = ctypes.CDLL(path)
    libx11.XInitThreads()
    handler = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)(lambda d, e: 0)
    io_handler = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)(lambda d: 0)
    libx11.XSetErrorHandler(handler)
    libx11.XSetIOErrorHandler(io_handler)
    install_x11_error_handler._keep = (handler, io_handler)
