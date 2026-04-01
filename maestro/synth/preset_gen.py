"""
Structured Vital synthesizer preset generator.

Generates .vital-compatible preset dicts using archetype-based parameter
sampling.  Six archetypes (bass, lead, pad, keys, pluck, sequence) each
carry per-section parameter ranges tuned for audible, musically plausible
results.

All parameter values are in Vital's native units (not normalized 0-1):
  - envelope times: seconds [0, 2.378]
  - filter_cutoff: MIDI note units [8, 136]
  - lfo_frequency: log2-scale [-7, 9] → 2^x Hz
  - volume: amplitude units [0, 7399]
  - discrete params (unison_voices, polyphony, etc.) are integers stored as float

Usage:
    from maestro.synth.preset_gen import generate_preset, generate_preset_to_file
    import random
    rng = random.Random(42)
    preset_dict = generate_preset("bass", rng)
    generate_preset_to_file("pad", "/tmp/my_pad.vital", rng)
"""

from __future__ import annotations

import copy
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from maestro.synth.lfo_shapes import build_lfos_array
from maestro.synth.wavetable_lib import get_default_wavetable, get_n_frames, sample_wavetable

ARCHETYPES = ["bass", "lead", "pad", "keys", "pluck", "sequence"]

# ---------------------------------------------------------------------------
# Archetype specifications
# ---------------------------------------------------------------------------

@dataclass
class ArchetypeSpec:
    name: str

    # Oscillator
    n_osc_weights: list[float]          # P(1 osc), P(2 osc), P(3 osc)
    unison_voices_range: tuple[int, int]
    unison_detune_range: tuple[float, float]  # cents-ish, actual range 0-1

    # Envelope 1 (amplitude)
    env_attack_range: tuple[float, float]   # seconds
    env_decay_range: tuple[float, float]    # seconds
    env_sustain_range: tuple[float, float]  # 0-1
    env_release_range: tuple[float, float]  # seconds

    # Filter
    filter_on_prob: float
    filter_cutoff_range: tuple[float, float]   # MIDI note units
    filter_resonance_range: tuple[float, float]
    filter2_on_prob: float

    # Polyphony
    polyphony_choices: list[int]
    polyphony_weights: list[float]

    # FX enable probabilities
    reverb_prob: float
    reverb_wet_range: tuple[float, float]
    delay_prob: float
    delay_wet_range: tuple[float, float]
    chorus_prob: float
    chorus_wet_range: tuple[float, float]
    distortion_prob: float
    distortion_drive_range: tuple[float, float]
    phaser_prob: float
    flanger_prob: float
    compressor_prob: float

    # Modulation
    n_mod_routes_range: tuple[int, int]

    # LFO
    n_active_lfos_range: tuple[int, int]
    lfo_freq_range: tuple[float, float]  # log2 Hz [-7, 9]

    # Mod target group weights
    group_weights: dict[str, float]


ARCHETYPE_SPECS: dict[str, ArchetypeSpec] = {
    "bass": ArchetypeSpec(
        name="bass",
        n_osc_weights=[0.35, 0.45, 0.20],
        unison_voices_range=(1, 6),
        unison_detune_range=(0.0, 0.7),
        env_attack_range=(0.0, 0.4),
        env_decay_range=(0.05, 1.5),
        env_sustain_range=(0.3, 1.0),
        env_release_range=(0.05, 1.0),
        filter_on_prob=0.75,
        filter_cutoff_range=(20.0, 110.0),
        filter_resonance_range=(0.0, 0.7),
        filter2_on_prob=0.15,
        polyphony_choices=[1, 2, 4],
        polyphony_weights=[0.45, 0.35, 0.20],
        reverb_prob=0.35,
        reverb_wet_range=(0.05, 0.30),
        delay_prob=0.30,
        delay_wet_range=(0.05, 0.35),
        chorus_prob=0.20,
        chorus_wet_range=(0.05, 0.30),
        distortion_prob=0.70,
        distortion_drive_range=(-5.0, 20.0),
        phaser_prob=0.10,
        flanger_prob=0.10,
        compressor_prob=0.35,
        n_mod_routes_range=(3, 10),
        n_active_lfos_range=(1, 3),
        lfo_freq_range=(-1.0, 3.0),  # ~0.5–8 Hz
        group_weights={
            "filter_dynamics": 0.40,
            "osc_timbre": 0.20,
            "osc_level": 0.15,
            "pitch": 0.10,
            "fx_depth": 0.15,
        },
    ),
    "lead": ArchetypeSpec(
        name="lead",
        n_osc_weights=[0.25, 0.45, 0.30],
        unison_voices_range=(1, 12),
        unison_detune_range=(0.0, 1.0),
        env_attack_range=(0.0, 0.6),
        env_decay_range=(0.05, 2.0),
        env_sustain_range=(0.2, 1.0),
        env_release_range=(0.05, 1.5),
        filter_on_prob=0.70,
        filter_cutoff_range=(30.0, 128.0),
        filter_resonance_range=(0.0, 0.75),
        filter2_on_prob=0.20,
        polyphony_choices=[4, 8, 16],
        polyphony_weights=[0.35, 0.45, 0.20],
        reverb_prob=0.65,
        reverb_wet_range=(0.10, 0.45),
        delay_prob=0.50,
        delay_wet_range=(0.10, 0.40),
        chorus_prob=0.45,
        chorus_wet_range=(0.10, 0.45),
        distortion_prob=0.55,
        distortion_drive_range=(-5.0, 15.0),
        phaser_prob=0.20,
        flanger_prob=0.20,
        compressor_prob=0.25,
        n_mod_routes_range=(3, 12),
        n_active_lfos_range=(1, 3),
        lfo_freq_range=(-1.0, 3.5),
        group_weights={
            "filter_dynamics": 0.35,
            "osc_timbre": 0.25,
            "osc_level": 0.10,
            "pitch": 0.15,
            "fx_depth": 0.15,
        },
    ),
    "pad": ArchetypeSpec(
        name="pad",
        n_osc_weights=[0.15, 0.40, 0.45],
        unison_voices_range=(3, 12),
        unison_detune_range=(0.1, 0.8),
        env_attack_range=(0.2, 1.8),
        env_decay_range=(0.3, 2.0),
        env_sustain_range=(0.6, 1.0),
        env_release_range=(0.5, 2.3),
        filter_on_prob=0.65,
        filter_cutoff_range=(50.0, 110.0),
        filter_resonance_range=(0.0, 0.55),
        filter2_on_prob=0.25,
        polyphony_choices=[8, 16, 32],
        polyphony_weights=[0.35, 0.45, 0.20],
        reverb_prob=0.88,
        reverb_wet_range=(0.25, 0.70),
        delay_prob=0.55,
        delay_wet_range=(0.10, 0.45),
        chorus_prob=0.60,
        chorus_wet_range=(0.15, 0.55),
        distortion_prob=0.30,
        distortion_drive_range=(-10.0, 8.0),
        phaser_prob=0.25,
        flanger_prob=0.20,
        compressor_prob=0.20,
        n_mod_routes_range=(4, 14),
        n_active_lfos_range=(1, 4),
        lfo_freq_range=(-4.0, 1.0),  # very slow: 0.06–2 Hz
        group_weights={
            "filter_dynamics": 0.20,
            "osc_timbre": 0.30,
            "osc_level": 0.20,
            "pitch": 0.05,
            "fx_depth": 0.25,
        },
    ),
    "keys": ArchetypeSpec(
        name="keys",
        n_osc_weights=[0.25, 0.50, 0.25],
        unison_voices_range=(1, 6),
        unison_detune_range=(0.0, 0.4),
        env_attack_range=(0.0, 0.20),
        env_decay_range=(0.2, 1.5),
        env_sustain_range=(0.0, 0.8),
        env_release_range=(0.2, 1.5),
        filter_on_prob=0.60,
        filter_cutoff_range=(45.0, 110.0),
        filter_resonance_range=(0.0, 0.60),
        filter2_on_prob=0.20,
        polyphony_choices=[8, 16],
        polyphony_weights=[0.50, 0.50],
        reverb_prob=0.80,
        reverb_wet_range=(0.15, 0.50),
        delay_prob=0.45,
        delay_wet_range=(0.10, 0.40),
        chorus_prob=0.35,
        chorus_wet_range=(0.10, 0.35),
        distortion_prob=0.30,
        distortion_drive_range=(-10.0, 10.0),
        phaser_prob=0.15,
        flanger_prob=0.10,
        compressor_prob=0.30,
        n_mod_routes_range=(2, 10),
        n_active_lfos_range=(1, 2),
        lfo_freq_range=(-2.0, 2.0),
        group_weights={
            "filter_dynamics": 0.25,
            "osc_timbre": 0.20,
            "osc_level": 0.15,
            "pitch": 0.10,
            "fx_depth": 0.30,
        },
    ),
    "pluck": ArchetypeSpec(
        name="pluck",
        n_osc_weights=[0.40, 0.40, 0.20],
        unison_voices_range=(1, 2),
        unison_detune_range=(0.0, 0.25),
        env_attack_range=(0.0, 0.02),
        env_decay_range=(0.05, 0.8),
        env_sustain_range=(0.05, 0.35),  # floor raised: 0 sustain + fast decay = silence
        env_release_range=(0.02, 0.4),
        filter_on_prob=0.65,
        filter_cutoff_range=(58.0, 115.0),  # floor raised: 45 MIDI (~234Hz) attenuates pluck fundamentals
        filter_resonance_range=(0.0, 0.70),
        filter2_on_prob=0.15,
        polyphony_choices=[1, 4, 8],
        polyphony_weights=[0.35, 0.40, 0.25],
        reverb_prob=0.55,
        reverb_wet_range=(0.10, 0.40),
        delay_prob=0.35,
        delay_wet_range=(0.05, 0.30),
        chorus_prob=0.25,
        chorus_wet_range=(0.05, 0.30),
        distortion_prob=0.40,
        distortion_drive_range=(-5.0, 12.0),
        phaser_prob=0.10,
        flanger_prob=0.10,
        compressor_prob=0.20,
        n_mod_routes_range=(2, 8),
        n_active_lfos_range=(0, 2),
        lfo_freq_range=(0.0, 4.0),
        group_weights={
            "filter_dynamics": 0.35,
            "osc_timbre": 0.30,
            "osc_level": 0.10,
            "pitch": 0.10,
            "fx_depth": 0.15,
        },
    ),
    "sequence": ArchetypeSpec(
        name="sequence",
        n_osc_weights=[0.30, 0.45, 0.25],
        unison_voices_range=(1, 4),
        unison_detune_range=(0.0, 0.5),
        env_attack_range=(0.0, 0.15),
        env_decay_range=(0.05, 1.0),
        env_sustain_range=(0.5, 1.0),
        env_release_range=(0.05, 0.5),
        filter_on_prob=0.70,
        filter_cutoff_range=(30.0, 100.0),
        filter_resonance_range=(0.0, 0.70),
        filter2_on_prob=0.20,
        polyphony_choices=[1, 4, 8],
        polyphony_weights=[0.30, 0.40, 0.30],
        reverb_prob=0.60,
        reverb_wet_range=(0.10, 0.40),
        delay_prob=0.55,
        delay_wet_range=(0.15, 0.50),
        chorus_prob=0.35,
        chorus_wet_range=(0.10, 0.35),
        distortion_prob=0.55,
        distortion_drive_range=(-5.0, 18.0),
        phaser_prob=0.20,
        flanger_prob=0.15,
        compressor_prob=0.30,
        n_mod_routes_range=(5, 16),
        n_active_lfos_range=(1, 4),
        lfo_freq_range=(-1.0, 4.0),
        group_weights={
            "filter_dynamics": 0.30,
            "osc_timbre": 0.20,
            "osc_level": 0.15,
            "pitch": 0.10,
            "fx_depth": 0.25,
        },
    ),
}


# ---------------------------------------------------------------------------
# Modulation sources and target groups
# ---------------------------------------------------------------------------

MOD_SOURCES = [
    ("lfo_1", 0.25), ("lfo_2", 0.14), ("lfo_3", 0.08), ("lfo_4", 0.05),
    ("env_2", 0.12), ("env_3", 0.06),
    ("velocity", 0.10), ("note", 0.06), ("mod_wheel", 0.07),
]
# macro_control sources removed: they default to 0 during render and never move.
_MOD_SOURCE_NAMES = [s for s, _ in MOD_SOURCES]
_MOD_SOURCE_WEIGHTS = [w for _, w in MOD_SOURCES]

MOD_TARGET_GROUPS: dict[str, list[str]] = {
    "filter_dynamics": [
        "filter_1_cutoff", "filter_2_cutoff",
    ],
    "osc_timbre": [
        "osc_1_wave_frame", "osc_2_wave_frame", "osc_3_wave_frame",
        "osc_1_distortion_amount", "osc_2_distortion_amount",
        "osc_1_spectral_morph_amount",
    ],
    "osc_level": [
        "osc_1_level", "osc_2_level", "osc_3_level",
    ],
    "pitch": [
        "osc_1_tune", "osc_2_tune", "osc_1_transpose",
    ],
    "fx_depth": [
        "reverb_dry_wet", "delay_dry_wet", "chorus_dry_wet",
        "distortion_drive", "phaser_dry_wet", "flanger_dry_wet",
    ],
}

# Sources that should be rate-limited when targeting level params
_LEVEL_SENSITIVE_TARGETS = {"osc_1_level", "osc_2_level", "osc_3_level", "voice_amplitude"}
# LFO sources (for frequency constraining)
_LFO_SOURCES = {"lfo_1", "lfo_2", "lfo_3", "lfo_4"}


# ---------------------------------------------------------------------------
# Section samplers
# ---------------------------------------------------------------------------

def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _sample_osc(spec: ArchetypeSpec, rng: random.Random, osc_idx: int,
                wavetable: dict) -> dict:
    """Sample settings for oscillator osc_idx (1-based)."""
    n_frames = get_n_frames(wavetable)
    p = {}
    p[f"osc_{osc_idx}_on"] = 1.0
    # Level: primary osc gets higher floor; secondary oscs can be quieter
    level_lo = 0.50 if osc_idx == 1 else 0.20
    p[f"osc_{osc_idx}_level"] = rng.uniform(level_lo, 1.0)
    # Wave frame: uniform over actual available frames
    p[f"osc_{osc_idx}_wave_frame"] = rng.uniform(0.0, float(max(1, n_frames - 1)))
    # Transpose: +/- 2 octaves (discrete)
    # Transpose: archetype-biased so bass leans low, lead/sequence lean high
    _TRANSPOSE_POOLS = {
        "bass":     [-24, -24, -12, -12, -5, 0, 0, 7, 12],
        "lead":     [-12, -5, 0, 0, 0, 7, 12, 12, 24],
        "sequence": [-12, -5, 0, 0, 0, 7, 12, 12, 24],
        "pad":      [-12, -5, 0, 0, 0, 0, 7, 12, 24],
        "keys":     [-12, -5, 0, 0, 0, 7, 12, 24, 24],
        "pluck":    [-12, -5, 0, 0, 0, 7, 12, 24, 24],
    }
    pool = _TRANSPOSE_POOLS.get(spec.name, [-24, -12, -5, 0, 0, 0, 7, 12, 24])
    p[f"osc_{osc_idx}_transpose"] = float(rng.choice(pool))
    # Fine tune: ±50 cents
    p[f"osc_{osc_idx}_tune"] = rng.uniform(-0.5, 0.5)
    # Unison
    lo_v, hi_v = spec.unison_voices_range
    voices = rng.randint(lo_v, hi_v)
    p[f"osc_{osc_idx}_unison_voices"] = float(voices)
    p[f"osc_{osc_idx}_unison_detune"] = rng.uniform(*spec.unison_detune_range)
    p[f"osc_{osc_idx}_unison_blend"] = rng.uniform(0.5, 1.0)
    p[f"osc_{osc_idx}_unison_stack_type"] = 0.0
    # Distortion
    dist_type = rng.randint(0, 12)
    p[f"osc_{osc_idx}_distortion_type"] = float(dist_type)
    if dist_type > 0:
        p[f"osc_{osc_idx}_distortion_amount"] = rng.uniform(0.0, 0.7)
    else:
        p[f"osc_{osc_idx}_distortion_amount"] = 0.0
    # Pan
    p[f"osc_{osc_idx}_pan"] = rng.uniform(-0.4, 0.4)
    # Spectral morph
    p[f"osc_{osc_idx}_spectral_morph_type"] = float(rng.randint(0, 8))
    p[f"osc_{osc_idx}_spectral_morph_amount"] = rng.uniform(0.0, 0.5)
    # Phase
    p[f"osc_{osc_idx}_phase"] = rng.uniform(0.0, 1.0)
    p[f"osc_{osc_idx}_random_phase"] = rng.uniform(0.0, 0.5)
    # Frame spread (for complex wavetables with multiple frames)
    p[f"osc_{osc_idx}_frame_spread"] = rng.uniform(0.0, 0.3)
    return p


def _sample_env(spec: ArchetypeSpec, rng: random.Random, env_idx: int) -> dict:
    """Sample ADSR envelope. env_idx=1 is amp env; 2+ are utility."""
    p = {}
    if env_idx == 1:
        attack = rng.uniform(*spec.env_attack_range)
        decay = rng.uniform(*spec.env_decay_range)
        sustain = rng.uniform(*spec.env_sustain_range)
        release = rng.uniform(*spec.env_release_range)
    else:
        # Utility envelopes: biased toward fast attack, variable decay
        attack = rng.uniform(0.0, 0.5)
        decay = rng.uniform(0.05, 1.5)
        sustain = rng.uniform(0.0, 1.0)
        release = rng.uniform(0.05, 1.0)
    p[f"env_{env_idx}_attack"] = _clamp(attack, 0.0, 2.378)
    p[f"env_{env_idx}_decay"] = _clamp(decay, 0.0, 2.378)
    p[f"env_{env_idx}_sustain"] = _clamp(sustain, 0.0, 1.0)
    p[f"env_{env_idx}_release"] = _clamp(release, 0.0, 2.378)
    # Curve shapes (power): 0 = linear, negative = convex, positive = concave
    p[f"env_{env_idx}_attack_power"] = rng.uniform(-3.0, 3.0)
    p[f"env_{env_idx}_decay_power"] = rng.uniform(-3.0, 3.0)
    p[f"env_{env_idx}_release_power"] = rng.uniform(-3.0, 3.0)
    p[f"env_{env_idx}_hold"] = 0.0
    p[f"env_{env_idx}_delay"] = 0.0
    return p


def _sample_filter(spec: ArchetypeSpec, rng: random.Random, filter_idx: int) -> dict:
    """Sample filter section. filter_idx 1 or 2."""
    p = {}
    on_prob = spec.filter_on_prob if filter_idx == 1 else spec.filter2_on_prob
    on = 1.0 if rng.random() < on_prob else 0.0
    p[f"filter_{filter_idx}_on"] = on
    p[f"filter_{filter_idx}_cutoff"] = rng.uniform(*spec.filter_cutoff_range)
    p[f"filter_{filter_idx}_resonance"] = _clamp(
        rng.uniform(*spec.filter_resonance_range), 0.0, 0.80
    )
    # Filter style: 0=LP12, 1=LP24, 2=BP12, 3=BP24, 4=HP12, 5=HP24, 6=Notch, ...
    p[f"filter_{filter_idx}_style"] = float(rng.choices(
        [0, 1, 2, 3, 4, 5, 6],
        weights=[0.30, 0.20, 0.15, 0.10, 0.10, 0.08, 0.07],
        k=1,
    )[0])
    # Filter model: 0=Analog, 1=Dirty, 2=Ladder, 3=Digital, etc.
    p[f"filter_{filter_idx}_model"] = float(rng.choices(
        [0, 1, 2, 3, 4, 5, 6],
        weights=[0.35, 0.15, 0.15, 0.15, 0.10, 0.05, 0.05],
        k=1,
    )[0])
    p[f"filter_{filter_idx}_drive"] = rng.uniform(0.0, 0.4)
    p[f"filter_{filter_idx}_blend"] = rng.uniform(0.0, 1.0)
    p[f"filter_{filter_idx}_mix"] = 1.0
    p[f"filter_{filter_idx}_keytrack"] = 0.0
    return p


def _sample_lfo(spec: ArchetypeSpec, rng: random.Random, lfo_idx: int,
                is_active: bool) -> dict:
    """Sample LFO settings. lfo_idx is 1-based."""
    p = {}
    p[f"lfo_{lfo_idx}_frequency"] = rng.uniform(*spec.lfo_freq_range)
    p[f"lfo_{lfo_idx}_phase"] = rng.uniform(0.0, 1.0)
    p[f"lfo_{lfo_idx}_sync"] = float(rng.random() < 0.3)  # 30% tempo-synced
    p[f"lfo_{lfo_idx}_sync_type"] = 0.0
    p[f"lfo_{lfo_idx}_smooth_mode"] = float(rng.random() < 0.2)
    p[f"lfo_{lfo_idx}_smooth_time"] = rng.uniform(-7.0, -3.0)
    p[f"lfo_{lfo_idx}_delay_time"] = rng.uniform(-7.0, -2.0)
    p[f"lfo_{lfo_idx}_fade_time"] = rng.uniform(-7.0, 0.0)
    p[f"lfo_{lfo_idx}_stereo"] = float(rng.random() < 0.2)
    p[f"lfo_{lfo_idx}_keytrack_transpose"] = 0.0
    p[f"lfo_{lfo_idx}_keytrack_tune"] = 0.0
    return p


def _sample_fx(spec: ArchetypeSpec, rng: random.Random) -> dict:
    """Sample all FX section parameters."""
    p = {}

    # Reverb
    reverb_on = rng.random() < spec.reverb_prob
    p["reverb_on"] = 1.0 if reverb_on else 0.0
    p["reverb_dry_wet"] = rng.uniform(*spec.reverb_wet_range) if reverb_on else 0.25
    p["reverb_decay_time"] = rng.uniform(-4.0, 4.0)
    p["reverb_pre_low_cutoff"] = rng.uniform(20.0, 60.0)
    p["reverb_pre_high_cutoff"] = rng.uniform(80.0, 120.0)
    p["reverb_low_shelf_cutoff"] = rng.uniform(20.0, 60.0)
    p["reverb_low_shelf_gain"] = rng.uniform(-6.0, 2.0)
    p["reverb_high_shelf_cutoff"] = rng.uniform(80.0, 120.0)
    p["reverb_high_shelf_gain"] = rng.uniform(-6.0, 0.0)
    p["reverb_size"] = rng.uniform(0.3, 1.0)
    p["reverb_feedback"] = rng.uniform(0.5, 0.95)
    p["reverb_chorus_amount"] = rng.uniform(0.0, 0.3)
    p["reverb_chorus_frequency"] = rng.uniform(-3.0, 3.0)

    # Delay
    delay_on = rng.random() < spec.delay_prob
    p["delay_on"] = 1.0 if delay_on else 0.0
    p["delay_dry_wet"] = rng.uniform(*spec.delay_wet_range) if delay_on else 0.333
    p["delay_feedback"] = rng.uniform(0.1, 0.6)
    p["delay_filter_cutoff"] = rng.uniform(40.0, 100.0)
    p["delay_filter_spread"] = rng.uniform(0.0, 0.5)
    p["delay_frequency"] = rng.uniform(-2.0, 3.0)
    p["delay_style"] = float(rng.randint(0, 3))
    p["delay_sync"] = float(rng.random() < 0.5)
    p["delay_tempo"] = float(rng.randint(8, 14))  # tempo-sync subdivisions

    # Chorus
    chorus_on = rng.random() < spec.chorus_prob
    p["chorus_on"] = 1.0 if chorus_on else 0.0
    p["chorus_dry_wet"] = rng.uniform(*spec.chorus_wet_range) if chorus_on else 0.5
    p["chorus_voices"] = float(rng.choice([2, 4, 6]))
    p["chorus_frequency"] = rng.uniform(-3.0, 2.0)
    p["chorus_feedback"] = rng.uniform(0.0, 0.5)
    p["chorus_mod_depth"] = rng.uniform(0.1, 0.8)
    p["chorus_cutoff"] = rng.uniform(60.0, 100.0)
    p["chorus_spread"] = rng.uniform(0.0, 1.0)
    p["chorus_delay_1"] = rng.uniform(0.005, 0.02)
    p["chorus_delay_2"] = rng.uniform(0.01, 0.04)
    p["chorus_sync"] = 0.0
    p["chorus_tempo"] = float(rng.randint(8, 14))

    # Distortion
    dist_on = rng.random() < spec.distortion_prob
    p["distortion_on"] = 1.0 if dist_on else 0.0
    p["distortion_drive"] = rng.uniform(*spec.distortion_drive_range) if dist_on else 0.0
    p["distortion_type"] = float(rng.randint(0, 5))
    p["distortion_filter_cutoff"] = rng.uniform(40.0, 110.0)
    p["distortion_filter_resonance"] = rng.uniform(0.0, 0.6)
    p["distortion_filter_blend"] = rng.uniform(0.0, 1.0)
    p["distortion_mix"] = rng.uniform(0.3, 1.0) if dist_on else 1.0

    # Phaser
    phaser_on = rng.random() < spec.phaser_prob
    p["phaser_on"] = 1.0 if phaser_on else 0.0
    p["phaser_dry_wet"] = rng.uniform(0.1, 0.5) if phaser_on else 0.5
    p["phaser_frequency"] = rng.uniform(-4.0, 2.0)
    p["phaser_feedback"] = rng.uniform(0.3, 0.85)
    p["phaser_mod_depth"] = rng.uniform(0.3, 1.0)
    p["phaser_center"] = rng.uniform(60.0, 100.0)
    p["phaser_phase_offset"] = rng.uniform(0.0, 1.0)
    p["phaser_blend"] = rng.uniform(0.5, 1.0)
    p["phaser_sync"] = 0.0
    p["phaser_tempo"] = float(rng.randint(8, 14))

    # Flanger
    flanger_on = rng.random() < spec.flanger_prob
    p["flanger_on"] = 1.0 if flanger_on else 0.0
    p["flanger_dry_wet"] = rng.uniform(0.1, 0.5) if flanger_on else 0.5
    p["flanger_frequency"] = rng.uniform(-4.0, 2.0)
    p["flanger_feedback"] = rng.uniform(0.3, 0.85)
    p["flanger_mod_depth"] = rng.uniform(0.3, 1.0)
    p["flanger_center"] = rng.uniform(60.0, 100.0)
    p["flanger_phase_offset"] = rng.uniform(0.0, 1.0)
    p["flanger_sync"] = 0.0
    p["flanger_tempo"] = float(rng.randint(8, 14))

    # Compressor
    comp_on = rng.random() < spec.compressor_prob
    p["compressor_on"] = 1.0 if comp_on else 0.0
    p["compressor_mix"] = rng.uniform(0.3, 1.0)
    p["compressor_attack"] = rng.uniform(-3.0, 1.0)
    p["compressor_release"] = rng.uniform(-3.0, 1.0)
    p["compressor_low_gain"] = rng.uniform(-6.0, 6.0)
    p["compressor_band_gain"] = rng.uniform(-6.0, 6.0)
    p["compressor_high_gain"] = rng.uniform(-6.0, 6.0)
    p["compressor_low_upper_ratio"] = rng.uniform(0.5, 1.0)
    p["compressor_band_upper_ratio"] = rng.uniform(0.5, 1.0)
    p["compressor_high_upper_ratio"] = rng.uniform(0.5, 1.0)
    p["compressor_low_lower_ratio"] = 0.0
    p["compressor_band_lower_ratio"] = 0.0
    p["compressor_high_lower_ratio"] = 0.0
    p["compressor_enabled_bands"] = float(rng.randint(1, 7))

    # EQ (always on, subtle)
    p["eq_on"] = 1.0
    p["eq_low_cutoff"] = rng.uniform(20.0, 50.0)
    p["eq_low_resonance"] = rng.uniform(0.5, 1.5)
    p["eq_low_gain"] = rng.uniform(-6.0, 3.0)
    p["eq_low_mode"] = 0.0
    p["eq_band_cutoff"] = rng.uniform(60.0, 90.0)
    p["eq_band_resonance"] = rng.uniform(0.5, 2.0)
    p["eq_band_gain"] = rng.uniform(-4.0, 4.0)
    p["eq_band_mode"] = 1.0
    p["eq_high_cutoff"] = rng.uniform(90.0, 120.0)
    p["eq_high_resonance"] = rng.uniform(0.5, 1.5)
    p["eq_high_gain"] = rng.uniform(-6.0, 2.0)
    p["eq_high_mode"] = 0.0

    return p


def _sample_modulations(
    spec: ArchetypeSpec,
    rng: random.Random,
    active_oscs: list[int],
    active_filters: list[int],
    active_lfos: set[int],
    fx_params: dict,
    archetype: str = "",
) -> tuple[list[dict], dict, set[int]]:
    """
    Build 64-slot modulation array and associated parameter values.

    Returns:
        modulations:  list of 64 {source, destination} dicts
        mod_params:   dict of modulation_N_* settings keys
        level_lfos:   set of LFO indices targeting level params (need rate constraint)
    """
    n_routes = rng.randint(*spec.n_mod_routes_range)
    group_names = list(spec.group_weights.keys())

    # Build available targets per group, filtering to active sections only.
    # Disabled-FX targets and inactive osc/filter targets are excluded entirely
    # rather than kept — routing to them wastes slots and produces silence.
    _FX_ON_KEYS = {
        "reverb_dry_wet": "reverb_on",
        "delay_dry_wet": "delay_on",
        "chorus_dry_wet": "chorus_on",
        "phaser_dry_wet": "phaser_on",
        "flanger_dry_wet": "flanger_on",
    }
    available_targets: dict[str, list[str]] = {}
    for group, targets in MOD_TARGET_GROUPS.items():
        active = []
        for t in targets:
            # Skip targets for inactive oscillators
            if "osc_" in t:
                parts = t.split("_")
                if len(parts) > 1 and parts[1].isdigit():
                    if int(parts[1]) not in active_oscs:
                        continue
            # Skip targets for inactive filters
            if "filter_" in t and "_cutoff" in t:
                parts = t.split("_")
                if len(parts) > 1 and parts[1].isdigit():
                    if int(parts[1]) not in active_filters:
                        continue
            # Skip targets for disabled FX
            if t in _FX_ON_KEYS and fx_params.get(_FX_ON_KEYS[t], 0.0) < 0.5:
                continue
            active.append(t)
        if active:
            available_targets[group] = active

    if not available_targets:
        available_targets = {g: list(t) for g, t in MOD_TARGET_GROUPS.items()}

    # Build source pool: only active LFOs, no macro controls.
    # Macro controls default to 0 at render time and never move.
    active_lfo_names = {f"lfo_{n}" for n in active_lfos}
    filtered_sources = [
        (name, weight) for name, weight in MOD_SOURCES
        if not name.startswith("lfo_") or name in active_lfo_names
    ]
    if not filtered_sources:
        filtered_sources = list(MOD_SOURCES)
    src_names = [s for s, _ in filtered_sources]
    src_weights = [w for _, w in filtered_sources]

    modulations = [{"source": "", "destination": ""} for _ in range(64)]
    mod_params: dict[str, float] = {}
    used_pairs: set[tuple[str, str]] = set()
    level_lfos: set[int] = set()
    route_idx = 0

    def _add_route(source: str, target: str, amount: float) -> bool:
        """Commit one route. Returns True if added, False if slot full or duplicate."""
        nonlocal route_idx
        if route_idx >= 64:
            return False
        pair = (source, target)
        if pair in used_pairs:
            return False
        used_pairs.add(pair)
        if source in _LFO_SOURCES and target in _LEVEL_SENSITIVE_TARGETS:
            lfo_n = int(source.split("_")[1])
            level_lfos.add(lfo_n)
        modulations[route_idx] = {"source": source, "destination": target}
        slot = route_idx + 1
        amt = amount
        if target in _LEVEL_SENSITIVE_TARGETS:
            amt = _clamp(abs(amt), 0.05, 0.50)
        if rng.random() < 0.15:
            amt = -abs(amt)
        mod_params[f"modulation_{slot}_amount"] = amt
        mod_params[f"modulation_{slot}_bipolar"] = 0.0
        mod_params[f"modulation_{slot}_bypass"] = 0.0
        mod_params[f"modulation_{slot}_power"] = rng.uniform(-2.0, 2.0)
        mod_params[f"modulation_{slot}_stereo"] = float(rng.random() < 0.1)
        route_idx += 1
        return True

    # --- Archetype signature routes ---
    # These pre-seed the most defining modulation for each archetype.
    # Placed first so they always get a slot even if n_routes is small.
    first_active_lfo = f"lfo_{min(active_lfos)}" if active_lfos else None

    if archetype in ("bass", "pluck") and 1 in active_filters:
        # Filter envelope: env_2 → filter cutoff sweep is the defining characteristic
        prob = 0.90 if archetype == "pluck" else 0.82
        if rng.random() < prob:
            amount = rng.uniform(0.30, 0.75)
            _add_route("env_2", "filter_1_cutoff", amount)

    elif archetype in ("lead", "sequence") and 1 in active_filters and first_active_lfo:
        # Rhythmic/expressive filter sweep
        prob = 0.75
        if rng.random() < prob:
            amount = rng.uniform(0.25, 0.65)
            _add_route(first_active_lfo, "filter_1_cutoff", amount)

    elif archetype == "pad" and 1 in active_oscs and first_active_lfo:
        # Slow wavetable morph — gives pads their evolving texture
        if rng.random() < 0.85:
            amount = rng.uniform(0.20, 0.70)
            _add_route(first_active_lfo, "osc_1_wave_frame", amount)

    elif archetype == "keys" and 1 in active_filters:
        # Velocity → filter cutoff: expressiveness from playing dynamics
        if rng.random() < 0.65:
            amount = rng.uniform(0.20, 0.55)
            _add_route("velocity", "filter_1_cutoff", amount)

    # --- Random routes ---
    for _ in range(n_routes):
        if route_idx >= 64:
            break
        source = rng.choices(src_names, weights=src_weights, k=1)[0]
        avail_groups = [g for g in group_names if g in available_targets]
        if not avail_groups:
            continue
        avail_w = [spec.group_weights[g] for g in avail_groups]
        group = rng.choices(avail_groups, weights=avail_w, k=1)[0]
        target = rng.choice(available_targets[group])
        amount = abs(rng.gauss(0.35, 0.20))
        amount = _clamp(amount, 0.05, 1.0)
        _add_route(source, target, amount)

    # --- Complexity floor ---
    # Every preset must have at least one "movement route": a time-varying source
    # (LFO or envelope) driving an audibly significant target (filter cutoff or
    # wavetable frame) with a non-trivial amount.  Without this a preset is a
    # static tone regardless of how many routes it has.
    _MOVEMENT_SOURCES = _LFO_SOURCES | {"env_2", "env_3"}
    _MOVEMENT_TARGETS = {
        "filter_1_cutoff", "filter_2_cutoff",
        "osc_1_wave_frame", "osc_2_wave_frame", "osc_3_wave_frame",
    }
    has_movement = any(
        modulations[i]["source"] in _MOVEMENT_SOURCES
        and modulations[i]["destination"] in _MOVEMENT_TARGETS
        and abs(mod_params.get(f"modulation_{i + 1}_amount", 0.0)) >= 0.15
        for i in range(route_idx)
    )
    if not has_movement and route_idx < 64:
        # Pick best available movement source and target
        mv_src = None
        if first_active_lfo:
            mv_src = first_active_lfo
        elif "env_2" in src_names:
            mv_src = "env_2"
        mv_tgt = None
        if 1 in active_filters:
            mv_tgt = "filter_1_cutoff"
        elif active_oscs:
            mv_tgt = f"osc_{active_oscs[0]}_wave_frame"
        if mv_src and mv_tgt:
            _add_route(mv_src, mv_tgt, rng.uniform(0.25, 0.65))

    # Fill remaining slots with bypassed defaults
    for i in range(route_idx, 64):
        slot = i + 1
        mod_params[f"modulation_{slot}_amount"] = 0.0
        mod_params[f"modulation_{slot}_bipolar"] = 0.0
        mod_params[f"modulation_{slot}_bypass"] = 1.0
        mod_params[f"modulation_{slot}_power"] = 0.0
        mod_params[f"modulation_{slot}_stereo"] = 0.0

    return modulations, mod_params, level_lfos


def _constrain_level_lfos(settings: dict, level_lfos: set[int]) -> None:
    """Constrain LFOs targeting level params: slow rate + non-zero start phase."""
    for lfo_n in level_lfos:
        freq_key = f"lfo_{lfo_n}_frequency"
        phase_key = f"lfo_{lfo_n}_phase"
        if freq_key in settings:
            # Cap at log2(4) ≈ 2.0 so max 4 Hz, preventing silent note starts
            settings[freq_key] = min(settings[freq_key], 2.0)
        if phase_key in settings:
            # Start at 25% so amplitude is never at minimum at note onset
            settings[phase_key] = max(settings[phase_key], 0.25)


_PARAM_RANGES: Optional[dict] = None


def _get_param_ranges() -> dict:
    """Load the param ranges table (min/max for every Vital control)."""
    global _PARAM_RANGES
    if _PARAM_RANGES is None:
        ranges_path = Path(__file__).parent / "param_ranges.json"
        if ranges_path.exists():
            with open(ranges_path) as f:
                _PARAM_RANGES = json.load(f)
        else:
            _PARAM_RANGES = {}
    return _PARAM_RANGES


def _clamp_settings_to_ranges(settings: dict) -> None:
    """Clamp every float settings value to its valid [min, max] range in-place."""
    ranges = _get_param_ranges()
    for k, v in settings.items():
        if not isinstance(v, (int, float)):
            continue
        r = ranges.get(k)
        if r is None:
            continue
        clamped = max(r["min"], min(r["max"], float(v)))
        if clamped != v:
            settings[k] = clamped


def _get_init_preset_json() -> dict:
    """Return a deep copy of the Vital init preset dict (used as structural template).

    The init preset is loaded from the bundled JSON file so we never need to
    create a second vita.Synth instance (creating two in the same process segfaults).
    The file is generated once via:  python -m maestro.synth.dump_init_preset
    """
    init_path = Path(__file__).parent / "init_preset.json"
    if not init_path.exists():
        raise FileNotFoundError(
            f"Init preset not found at {init_path}. "
            "Run: python -m maestro.synth.dump_init_preset"
        )
    with open(init_path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_preset(
    archetype: str,
    rng: random.Random,
    wavetable_lib: Optional[list[dict]] = None,
    name: Optional[str] = None,
) -> dict:
    """
    Generate a complete .vital-compatible preset dict.

    Args:
        archetype:      One of ARCHETYPES ('bass', 'lead', 'pad', 'keys', 'pluck', 'sequence')
        rng:            Seeded random.Random for determinism
        wavetable_lib:  List of wavetable dicts from wavetable_lib.load_wavetable_lib().
                        If None or empty, Vital's Init wavetable is used for all oscillators.
        name:           Optional preset name; auto-generated if None.

    Returns:
        A dict ready to be serialized as JSON and loaded by vita.Synth.load_json().
    """
    if archetype not in ARCHETYPE_SPECS:
        raise ValueError(f"Unknown archetype '{archetype}'. Choose from: {ARCHETYPES}")
    spec = ARCHETYPE_SPECS[archetype]

    # Start from the init preset to get a complete settings dict with all defaults
    base = _get_init_preset_json()
    settings: dict = base["settings"]

    # 1. Decide active oscillators
    n_oscs = rng.choices([1, 2, 3], weights=spec.n_osc_weights, k=1)[0]
    active_oscs = list(range(1, n_oscs + 1))

    # 2. Sample oscillators; assign one wavetable per oscillator
    for i in active_oscs:
        wt = sample_wavetable(wavetable_lib or [], rng)
        osc_params = _sample_osc(spec, rng, i, wt)
        settings.update(osc_params)
        # Embed wavetable in the correct slot (0-indexed)
        if len(settings["wavetables"]) >= i:
            settings["wavetables"][i - 1] = wt

    # Turn off inactive oscillators
    for i in range(n_oscs + 1, 4):
        settings[f"osc_{i}_on"] = 0.0
        settings[f"osc_{i}_level"] = 0.0

    # 3. Sample envelope sections
    for env_idx in range(1, 7):
        settings.update(_sample_env(spec, rng, env_idx))

    # 4. Sample filters
    active_filters = []
    for f_idx in [1, 2]:
        fparams = _sample_filter(spec, rng, f_idx)
        settings.update(fparams)
        if fparams[f"filter_{f_idx}_on"] > 0.5:
            active_filters.append(f_idx)

    # 5. Sample FX
    fx_params = _sample_fx(spec, rng)
    settings.update(fx_params)

    # 6. Plan LFOs: decide how many are active
    n_active_lfos = rng.randint(*spec.n_active_lfos_range)
    active_lfo_set = set(range(1, n_active_lfos + 1))
    for lfo_idx in range(1, 9):
        lfo_params = _sample_lfo(spec, rng, lfo_idx, lfo_idx in active_lfo_set)
        settings.update(lfo_params)

    # 7. Build modulation matrix
    modulations, mod_params, level_lfos = _sample_modulations(
        spec, rng, active_oscs, active_filters, active_lfo_set, fx_params, archetype
    )
    settings.update(mod_params)
    settings["modulations"] = modulations

    # 8. Constrain level-targeting LFOs
    _constrain_level_lfos(settings, level_lfos)

    # 9. Build LFO waveform shapes array
    all_active_lfos = active_lfo_set | level_lfos
    # Also add any LFO sources actually used in modulations
    for mod in modulations:
        src = mod.get("source", "")
        if src in _LFO_SOURCES:
            all_active_lfos.add(int(src.split("_")[1]))
    settings["lfos"] = build_lfos_array(all_active_lfos, archetype, rng)

    # 10. Global params
    settings["polyphony"] = float(
        rng.choices(spec.polyphony_choices, weights=spec.polyphony_weights, k=1)[0]
    )
    settings["voice_amplitude"] = rng.uniform(0.7, 1.0)
    # volume: near default 0 dB (5473), ± ~6 dB; floor at 3500 (~-4 dB) avoids near-silence
    settings["volume"] = _clamp(rng.gauss(5473.0, 1200.0), 3500.0, 7399.0)
    settings["beats_per_minute"] = 3.0  # default
    settings["legato"] = float(rng.random() < 0.15)
    settings["portamento_on"] = float(rng.random() < 0.10)
    settings["portamento_time"] = rng.uniform(-10.0, -5.0)
    settings["voice_priority"] = float(rng.randint(0, 5))
    settings["voice_transpose"] = 0.0
    settings["pitch_wheel"] = 0.0
    settings["pitch_bend_range"] = float(rng.choice([2, 4, 7, 12, 24]))
    settings["stereo_routing"] = float(rng.random() < 0.3)

    # Disable sample layer (don't accidentally use sample oscillator)
    settings["sample_on"] = 0.0

    # Clamp all values to valid ranges — catches any generator bugs before vita sees them
    _clamp_settings_to_ranges(settings)

    preset_name = name or f"{archetype}_{rng.randint(0, 99999):05d}"
    return {
        "author": "maestro-preset-gen",
        "comments": f"archetype={archetype}",
        "macro1": "MACRO 1",
        "macro2": "MACRO 2",
        "macro3": "MACRO 3",
        "macro4": "MACRO 4",
        "preset_name": preset_name,
        "preset_style": archetype.capitalize(),
        "settings": settings,
        "synth_version": "99999.9.9",
    }


def generate_preset_to_file(
    archetype: str,
    output_path: str | Path,
    rng: random.Random,
    wavetable_lib: Optional[list[dict]] = None,
    name: Optional[str] = None,
) -> str:
    """Generate a preset and write it as a .vital JSON file. Returns the output path."""
    preset = generate_preset(archetype, rng, wavetable_lib=wavetable_lib, name=name)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(preset, f, separators=(",", ":"))
    return str(path)


def generate_preset_batch(
    n: int,
    output_dir: str | Path,
    seed: int = 42,
    archetypes: Optional[list[str]] = None,
    wavetable_lib: Optional[list[dict]] = None,
) -> list[str]:
    """
    Generate n presets and write them to output_dir.

    Args:
        n:             Number of presets to generate.
        output_dir:    Directory to write .vital files into.
        seed:          Master seed (each preset gets a derived sub-seed).
        archetypes:    List to sample from; None = uniform over all 6 archetypes.
        wavetable_lib: Pre-loaded wavetable library.

    Returns:
        List of written file paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(seed)
    archetype_pool = archetypes or ARCHETYPES
    paths = []
    for i in range(n):
        archetype = master_rng.choice(archetype_pool)
        sub_seed = master_rng.randint(0, 2**32 - 1)
        preset_name = f"{archetype}_{i:06d}"
        path = generate_preset_to_file(
            archetype,
            out / f"{preset_name}.vital",
            random.Random(sub_seed),
            wavetable_lib=wavetable_lib,
            name=preset_name,
        )
        paths.append(path)
    return paths
