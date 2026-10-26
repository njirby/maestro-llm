#!/usr/bin/env python3
"""Shared fixtures for the candidate-probe renderer bake-off.

Defines the one thing every candidate render path must reproduce identically:
8 probe presets that differ *only* in settings["wavetables"][0], plus one
fixed melody. A path is correct iff the 8 renders are audibly distinct.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

WAVETABLE_LIB = REPO / "data" / "wavetable_lib.json"
INIT_PRESET = REPO / "maestro" / "synth" / "init_preset.json"

# Maximally diverse by timbre class: pure sine, saw, square/pulse, noise,
# inharmonic bell, metallic, formant/vox. If a path collapses these to the
# same signature it is not applying wavetables.
PROBE_WAVETABLES = [
    "Vital Sine",
    "Quad Saw",
    "ESW Analog - Juno 60 Square",
    "White Noise",
    "Bell",
    "Pulse Width",
    "ESW Metallic - Waves",
    "woolf female vox 1",
]

# 8 notes, MIDI 55-79 (~2 octaves, mid register), 4.8s + release tail.
MELODY = [
    {"pitch": 55, "velocity": 100, "start_s": 0.0, "dur_s": 0.55},
    {"pitch": 62, "velocity": 100, "start_s": 0.6, "dur_s": 0.55},
    {"pitch": 67, "velocity": 100, "start_s": 1.2, "dur_s": 0.55},
    {"pitch": 71, "velocity": 100, "start_s": 1.8, "dur_s": 0.55},
    {"pitch": 74, "velocity": 100, "start_s": 2.4, "dur_s": 0.55},
    {"pitch": 79, "velocity": 100, "start_s": 3.0, "dur_s": 0.55},
    {"pitch": 72, "velocity": 100, "start_s": 3.6, "dur_s": 0.55},
    {"pitch": 60, "velocity": 100, "start_s": 4.2, "dur_s": 0.80},
]

SAMPLE_RATE = 44100


def load_init_preset() -> dict[str, Any]:
    with open(INIT_PRESET) as f:
        return json.load(f)


def load_wavetable_lib() -> list[dict[str, Any]]:
    with open(WAVETABLE_LIB) as f:
        return json.load(f)


def pick_probe_wavetables() -> list[dict[str, Any]]:
    """Resolve PROBE_WAVETABLES to library entries (first match per name)."""
    lib = load_wavetable_lib()
    by_name: dict[str, dict[str, Any]] = {}
    for wt in lib:
        by_name.setdefault(wt.get("name", ""), wt)
    picked = []
    for name in PROBE_WAVETABLES:
        if name not in by_name:
            raise KeyError(f"wavetable {name!r} not in {WAVETABLE_LIB}")
        picked.append(by_name[name])
    return picked


def build_probe_preset(init_preset: dict[str, Any], wavetable: dict[str, Any],
                       frame_idx: int = 0) -> dict[str, Any]:
    """Mirror of _build_probe_preset in build_wavetable_retrieval_baseline.py:
    isolate oscillator 1, kill FX, swap in the candidate wavetable."""
    preset = copy.deepcopy(init_preset)
    settings = preset["settings"]
    settings["sample_on"] = 0.0

    settings["osc_1_on"] = 1.0
    settings["osc_1_level"] = 1.0
    settings["osc_1_wave_frame"] = float(frame_idx)
    settings["osc_1_unison_voices"] = 1.0
    settings["osc_1_unison_detune"] = 0.0
    settings["osc_1_pan"] = 0.0

    settings["osc_2_on"] = 0.0
    settings["osc_2_level"] = 0.0
    settings["osc_3_on"] = 0.0
    settings["osc_3_level"] = 0.0

    for k in ("reverb_on", "delay_on", "chorus_on", "distortion_on",
              "phaser_on", "flanger_on", "compressor_on"):
        if k in settings:
            settings[k] = 0.0

    if isinstance(settings.get("wavetables"), list) and settings["wavetables"]:
        settings["wavetables"][0] = copy.deepcopy(wavetable)
    return preset


def build_probe_set() -> list[tuple[str, dict[str, Any]]]:
    """[(safe_name, preset), ...] for the 8 probes."""
    init = load_init_preset()
    out = []
    for wt in pick_probe_wavetables():
        name = wt.get("name", "unnamed")
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
        out.append((safe, build_probe_preset(init, wt)))
    return out


def notes_as_tuples() -> list[tuple[int, int, float, float]]:
    return [(n["pitch"], n["velocity"], n["start_s"], n["dur_s"]) for n in MELODY]


# ---------------------------------------------------------------------------
# Correctness analysis
# ---------------------------------------------------------------------------

def analyze_wavs(wav_paths: list[Path]) -> dict[str, Any]:
    """Spectral centroid + time-averaged log-mel signature per render.

    A path passes if the wavetables actually reached the synth: centroid
    spread across renders should be large (hundreds of Hz) and pairwise
    mel-signature cosines meaningfully below 1.
    """
    import librosa
    import soundfile as sf

    rows = []
    sigs = []
    for p in wav_paths:
        audio, sr = sf.read(str(p), always_2d=True)
        mono = audio.mean(axis=1).astype(np.float32)
        peak = float(np.abs(mono).max()) if mono.size else 0.0
        rms = float(np.sqrt(np.mean(mono ** 2))) if mono.size else 0.0
        # Normalize so loudness differences don't masquerade as timbre.
        norm = mono / peak if peak > 0 else mono
        cent = float(np.mean(librosa.feature.spectral_centroid(y=norm, sr=sr)))
        mel = librosa.feature.melspectrogram(y=norm, sr=sr, n_mels=64)
        sig = np.log(mel + 1e-8).mean(axis=1)
        sigs.append(sig)
        rows.append({
            "file": p.name,
            "duration_s": round(len(mono) / sr, 3),
            "sr": sr,
            "peak": round(peak, 5),
            "rms": round(rms, 5),
            "centroid_hz": round(cent, 1),
        })

    S = np.stack(sigs)
    Sn = S / (np.linalg.norm(S, axis=1, keepdims=True) + 1e-12)
    cos = Sn @ Sn.T
    iu = np.triu_indices(len(S), k=1)
    pairwise = cos[iu]
    cents = np.array([r["centroid_hz"] for r in rows])

    return {
        "rows": rows,
        "centroid_min_hz": float(cents.min()),
        "centroid_max_hz": float(cents.max()),
        "centroid_spread_hz": float(cents.max() - cents.min()),
        "centroid_std_hz": float(cents.std()),
        "min_pairwise_mel_cos": float(pairwise.min()),
        "mean_pairwise_mel_cos": float(pairwise.mean()),
        "max_pairwise_mel_cos": float(pairwise.max()),
        "all_nonsilent": bool(all(r["rms"] > 1e-4 for r in rows)),
    }


def verdict(stats: dict[str, Any]) -> tuple[bool, str]:
    ok_spread = stats["centroid_spread_hz"] > 500.0
    ok_cos = stats["min_pairwise_mel_cos"] < 0.98
    ok_audio = stats["all_nonsilent"]
    passed = ok_spread and ok_cos and ok_audio
    why = (f"spread={stats['centroid_spread_hz']:.0f}Hz(>500:{ok_spread}) "
           f"min_cos={stats['min_pairwise_mel_cos']:.4f}(<0.98:{ok_cos}) "
           f"nonsilent={ok_audio}")
    return passed, why


if __name__ == "__main__":
    probes = build_probe_set()
    for safe, preset in probes:
        wt = preset["settings"]["wavetables"][0]
        groups = wt.get("groups", [])
        n_kf = sum(len(c.get("keyframes", []))
                   for g in groups for c in g.get("components", []))
        has_data = any(bool(k.get("wave_data"))
                       for g in groups for c in g.get("components", [])
                       for k in c.get("keyframes", []))
        print(f"{safe:32s} name={wt.get('name','?'):32s} keyframes={n_kf} wave_data={has_data}")
    print(f"\n{len(probes)} probes, melody {len(MELODY)} notes, "
          f"end={max(n['start_s']+n['dur_s'] for n in MELODY):.2f}s")
