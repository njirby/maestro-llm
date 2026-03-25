import math

import pretty_midi

from scripts.generate_reaper_tuples import MAX_CLIP_S, MIN_CLIP_S, _clip_bars, _extract_clips


def _note(start: float, end: float, pitch: int = 60, velocity: int = 90) -> pretty_midi.Note:
    return pretty_midi.Note(velocity=velocity, pitch=pitch, start=start, end=end)


def test_clip_bars_duration_stays_in_bounds():
    bpm = 120.0
    for clip_idx in range(20):
        bars = _clip_bars("artist/track.mid", 0, clip_idx, bpm)
        dur = bars * (240.0 / bpm)
        assert dur >= MIN_CLIP_S
        assert dur <= MAX_CLIP_S


def test_extract_clips_aligns_to_bar_boundaries():
    bpm = 120.0
    bar_len = 240.0 / bpm
    notes = [
        _note(1.20, 1.60, 60),
        _note(2.25, 2.60, 62),
        _note(3.00, 3.20, 64),
        _note(4.10, 4.40, 65),
    ]

    clips = _extract_clips(notes, "artist/track.mid", 0, bpm)
    assert clips, "expected at least one clip"
    first = clips[0]

    assert math.isclose(first["clip_start_s"] % bar_len, 0.0, abs_tol=1e-9)
    assert first["clip_start_bar"] == int(first["clip_start_s"] / bar_len) + 1
    assert math.isclose(first["clip_dur_s"] / bar_len, round(first["clip_dur_s"] / bar_len), abs_tol=1e-9)
