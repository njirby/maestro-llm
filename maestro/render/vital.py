"""
Library for rendering single-track WAV files using the Vital synthesizer
via the Vita Python bindings (direct C++ engine, no VST3/REAPER required).
"""

import random
import warnings
from pathlib import Path

import numpy as np
import pretty_midi
import vita as _vita

SAMPLE_RATE = 44100


# ---------------------------------------------------------------------------
# MIDI extraction
# ---------------------------------------------------------------------------

def extract_track(midi_path: str, rng: random.Random):
    """
    Parse a MIDI file and select one random non-drum track.

    Returns:
        (notes, track_idx, track_name, program, duration_s)
        or (None, None, None, None, None) if no usable melodic tracks.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            pm = pretty_midi.PrettyMIDI(midi_path)
        except Exception:
            return None, None, None, None, None

    candidates = [
        (i, inst)
        for i, inst in enumerate(pm.instruments)
        if not inst.is_drum and len(inst.notes) > 0
    ]
    if not candidates:
        return None, None, None, None, None

    track_idx, inst = rng.choice(candidates)
    duration_s = max(n.end for n in inst.notes)
    return inst.notes, track_idx, inst.name or "", inst.program, duration_s


# ---------------------------------------------------------------------------
# Vital synth management
# ---------------------------------------------------------------------------

def _load_vital() -> _vita.Synth:
    """Create a fresh Vita Synth instance (Vital C++ engine)."""
    return _vita.Synth()


def apply_preset(synth: _vita.Synth, preset_path: str) -> None:
    """Load a .vital preset file into a Vita Synth instance."""
    synth.load_preset(str(preset_path))


def jitter_preset(synth: _vita.Synth, preset_path: str, rng: random.Random, jitter_std: float = 0.05) -> None:
    """
    Apply small random perturbations to preset parameters using normalized (0-1) values.
    A jitter_std of 0.03 means roughly ±3% of the parameter's normalized range.
    """
    controls = synth.get_controls()
    for name, ctrl in controls.items():
        try:
            current_norm = ctrl.get_normalized()
            delta = rng.gauss(0, jitter_std)
            ctrl.set_normalized(max(0.0, min(1.0, current_norm + delta)))
        except Exception:
            pass


def randomize_preset(
    synth: _vita.Synth,
    rng: random.Random,
    amount: float = 1.0,
) -> None:
    """
    Randomize synth controls in normalized space.

    amount:
        0.0 -> keep current state
        1.0 -> fully randomize each control
    """
    amt = max(0.0, min(1.0, float(amount)))
    if amt == 0.0:
        return

    controls = synth.get_controls()
    for _, ctrl in controls.items():
        try:
            current_norm = ctrl.get_normalized()
            target_norm = rng.random()
            new_norm = current_norm + (target_norm - current_norm) * amt
            ctrl.set_normalized(max(0.0, min(1.0, new_norm)))
        except Exception:
            pass


def constrain_random_preset_for_audibility(synth: _vita.Synth) -> None:
    """
    Clamp critical controls to reduce the chance of silent generated presets.
    """
    controls = synth.get_controls()

    def _set_floor(name: str, minimum: float) -> None:
        ctrl = controls.get(name)
        if ctrl is None:
            return
        try:
            value = ctrl.get_normalized()
            ctrl.set_normalized(max(minimum, min(1.0, value)))
        except Exception:
            pass

    def _set_ceiling(name: str, maximum: float) -> None:
        ctrl = controls.get(name)
        if ctrl is None:
            return
        try:
            value = ctrl.get_normalized()
            ctrl.set_normalized(max(0.0, min(maximum, value)))
        except Exception:
            pass

    def _set_value(name: str, value: float) -> None:
        ctrl = controls.get(name)
        if ctrl is None:
            return
        try:
            ctrl.set_normalized(max(0.0, min(1.0, value)))
        except Exception:
            pass

    # Ensure at least one clearly audible source is active.
    _set_value("osc_1_on", 1.0)
    _set_floor("osc_1_level", 0.30)

    # Keep global output from collapsing.
    _set_floor("volume", 0.35)
    _set_floor("voice_amplitude", 0.35)

    # Avoid very long attack/zero sustain amp shapes that mute short notes.
    _set_ceiling("env_1_attack", 0.30)
    _set_floor("env_1_sustain", 0.20)
    _set_floor("env_1_decay", 0.02)
    _set_floor("env_1_release", 0.02)

    # Prevent extreme low-pass lockouts when filters are enabled.
    _set_floor("filter_1_cutoff", 0.08)
    _set_floor("filter_2_cutoff", 0.08)
    _set_floor("filter_fx_cutoff", 0.08)


def render_probe_note(
    synth: _vita.Synth,
    midi_note: int,
    midi_velocity: float,
    note_dur: float,
    tail_s: float = 0.15,
) -> np.ndarray:
    """
    Render a short probe note used to estimate audibility.
    """
    pitch = max(0, min(127, int(midi_note)))
    velocity = max(0.01, min(1.0, float(midi_velocity)))
    dur = max(0.01, float(note_dur))
    render_dur = dur + max(0.01, float(tail_s))
    return synth.render(
        midi_note=pitch,
        midi_velocity=velocity,
        note_dur=dur,
        render_dur=render_dur,
    )


# ---------------------------------------------------------------------------
# Note helpers
# ---------------------------------------------------------------------------

def transpose_notes(notes, semitones: int) -> list:
    """Return a new list of notes with pitches shifted by semitones, clamped to [0, 127]."""
    result = []
    for note in notes:
        new_note = pretty_midi.Note(
            velocity=note.velocity,
            pitch=max(0, min(127, note.pitch + semitones)),
            start=note.start,
            end=note.end,
        )
        result.append(new_note)
    return result


# ---------------------------------------------------------------------------
# Audio rendering
# ---------------------------------------------------------------------------

NOTE_TAIL_S = 0.15   # short tail per note (release ring); full tail_s added at end


def _render_note_list(synth: _vita.Synth, notes, sample_rate: int, tail_s: float) -> np.ndarray:
    """
    Render a list of pretty_midi.Note objects through a Vita Synth instance.

    Each note is rendered individually and mixed into a shared stereo buffer at
    its correct time offset. A short per-note tail captures the release; the
    full tail_s is reflected in the output buffer length only.

    Returns:
        Stereo float32 numpy array of shape (2, num_samples).
    """
    if not notes:
        return np.zeros((2, int(sample_rate * tail_s)), dtype=np.float32)

    total_dur = max(n.end for n in notes) + tail_s
    total_samples = int(total_dur * sample_rate)
    mix = np.zeros((2, total_samples), dtype=np.float32)

    for note in notes:
        pitch = max(0, min(127, int(note.pitch)))
        velocity = max(0.01, min(1.0, note.velocity / 127.0))
        note_dur = max(0.01, note.end - note.start)
        render_dur = note_dur + NOTE_TAIL_S

        chunk = synth.render(
            midi_note=pitch,
            midi_velocity=velocity,
            note_dur=note_dur,
            render_dur=render_dur,
        )  # shape: (2, n_samples)

        offset = int(note.start * sample_rate)
        n = chunk.shape[1]
        space = total_samples - offset
        if space <= 0:
            continue
        if n > space:
            chunk = chunk[:, :space]
            n = space
        mix[:, offset:offset + n] += chunk

    peak = np.abs(mix).max()
    if peak > 1.0:
        mix /= peak

    return mix


def render_notes(
    notes,
    preset_path: str,
    sample_rate: int = 44100,
    tail_s: float = 2.0,
) -> np.ndarray:
    """
    Render a list of pretty_midi.Note objects through Vital with the given preset.

    Returns:
        Stereo float32 numpy array of shape (2, num_samples).
    """
    synth = _load_vital()
    apply_preset(synth, preset_path)
    return _render_note_list(synth, notes, sample_rate, tail_s)
