"""The notes.json contract of the `vital` skill: the melody written the way a musician writes it —
measures and note values, not absolute seconds.

    {"bpm": 148.0, "time_signature": "4/4", "notes": [
        {"pitch": 74, "velocity": 116, "bar": 1, "beat": 1, "duration": "eighth"}, ...]}

`bar`/`beat` place the note on the grid (beat 1 is the first beat of the bar; fractions land on
sixteenths); `duration` is a note value from VALUE_BEATS. Seconds live only in the renderer:
`to_seconds` is the single bridge, and `quantize` snaps a seconds list onto the grid so a seconds
list and its beat form round-trip exactly. The legacy {"pitch", "velocity", "start_s", "dur_s"}
form is still accepted by `to_seconds` (old rollouts) but no longer written.

The skill's own scripts (opencode/skills/vital/scripts/vitallib.py) are standalone and carry a
copy of VALUE_BEATS / parse_time_signature; tests/test_notation.py pins them in step.
"""
from __future__ import annotations

import copy
import json
import math
import random

GRID = 0.25  # the grid is in sixteenths: beat fractions are multiples of 0.25

# note value -> beats (a quarter note is one beat). The full vocabulary a DAW's quantize menu
# offers down to 32nds: straight and dotted values, plus the triplet values (three in the time
# of two: a triplet eighth is 3-in-2 eighths = 1/3 beat).
VALUE_BEATS = {
    "whole": 4.0,
    "dotted half": 3.0,
    "half": 2.0,
    "dotted quarter": 1.5,
    "quarter": 1.0,
    "dotted eighth": 0.75,
    "eighth": 0.5,
    "dotted sixteenth": 0.375,
    "sixteenth": 0.25,
    "dotted thirty-second": 0.1875,
    "thirty-second": 0.125,
    "triplet quarter": 2.0 / 3,
    "triplet eighth": 1.0 / 3,
    "triplet sixteenth": 1.0 / 6,
    "triplet thirty-second": 1.0 / 12,
}
_VALUE_ORDER = sorted(VALUE_BEATS, key=VALUE_BEATS.get)


class NotationError(ValueError):
    pass


def parse_time_signature(ts: str | None) -> tuple[int, float]:
    """`W/X` -> (beats per bar, beat length in beats). 4/4 and 3/4 count quarters; 6/8, 9/8 and 12/8
    are compound (the beat is a dotted quarter). Anything else is not used in this corpus."""
    ts = ts or "4/4"
    try:
        w, x = (int(t) for t in ts.split("/"))
    except ValueError:
        raise NotationError(f"time signature {ts!r} is not W/X") from None
    if x == 4 and 2 <= w <= 7:
        return w, 1.0
    if x == 8 and w in (6, 9, 12):
        return w // 3, 1.5
    raise NotationError(f"time signature {ts!r} unsupported (use 4/4, 3/4, 6/8, 9/8 or 12/8)")


def nearest_value(dur_beats: float) -> str:
    """The note value a musician would write for a duration: nearest in log space (halving/doubling)."""
    if dur_beats <= 0:
        raise NotationError(f"duration {dur_beats} beats is not > 0")
    return min(_VALUE_ORDER, key=lambda v: abs(math.log2(dur_beats / VALUE_BEATS[v])))


def beat_note_to_seconds(bar: int, beat: float, duration: str, bpm: float,
                         beats_per_bar: int = 4, beat_beats: float = 1.0) -> tuple[float, float]:
    if duration not in VALUE_BEATS:
        raise NotationError(f"unknown note value {duration!r} (one of: {', '.join(_VALUE_ORDER)})")
    if not (1 <= int(bar) and beat >= 1):
        raise NotationError(f"note starts before bar 1 beat 1: bar={bar} beat={beat}")
    spb = 60.0 / bpm * beat_beats  # seconds per beat
    return ((bar - 1) * beats_per_bar + (beat - 1)) * spb, VALUE_BEATS[duration] * spb


def to_seconds(notes) -> tuple[list[dict], float]:
    """Any notes.json shape -> (canonical seconds notes, bpm). Accepts the beat form, the legacy
    start_s/dur_s form, a bare list, or a notes dict. Raises NotationError on bad input."""
    d, is_dict = notes, isinstance(notes, (dict, str))
    if isinstance(notes, str):
        d = json.loads(notes)
    if isinstance(d, dict):
        bpm = float(d.get("bpm") or 120.0)
        d = d.get("notes")
    else:
        bpm = 120.0
    if not isinstance(d, list) or not d:
        raise NotationError("needs a non-empty notes list")
    bpb, beat_beats = parse_time_signature(None)  # 4/4: the only signature the beat form is written in
    out = []
    for i, n in enumerate(d):
        pitch, vel = int(n["pitch"]), int(n.get("velocity", 100))
        if not (0 <= pitch <= 127 and 0 <= vel <= 127):
            raise NotationError(f"note {i} out of range: {n!r}")
        if "start_s" in n:  # legacy form
            start_s, dur_s = float(n["start_s"]), float(n["dur_s"])
        else:
            start_s, dur_s = beat_note_to_seconds(int(n["bar"]), float(n["beat"]), str(n["duration"]),
                                                  bpm, bpb, beat_beats)
        if not (start_s >= 0 and dur_s > 0):
            raise NotationError(f"note {i} out of range: {n!r}")
        out.append({"pitch": pitch, "velocity": vel, "start_s": start_s, "dur_s": dur_s})
    return out, (bpm if is_dict else 120.0)


def quantize(bpm: float, notes: list[dict]) -> list[dict]:
    """Snap a seconds list onto the sixteenth grid (onset to the grid, duration to the nearest
    note value) and drop grid-merged duplicates (same onset, same pitch). The corpus' ground truth
    is quantized this way before it is rendered or written, so the beat form round-trips exactly."""
    spb = 60.0 / bpm
    out, seen = [], set()
    for n in sorted(notes, key=lambda x: (x["start_s"], x["pitch"])):
        slot = int(round(n["start_s"] / (spb * GRID)))
        start_s = slot * spb * GRID
        dur_s = VALUE_BEATS[nearest_value(n["dur_s"] / spb)] * spb
        key = (start_s, n["pitch"])
        if key in seen:
            continue
        seen.add(key)
        out.append({"pitch": n["pitch"], "velocity": n["velocity"], "start_s": start_s, "dur_s": dur_s})
    return out


def to_beat_form(bpm: float, notes: list[dict], time_signature: str = "4/4") -> dict:
    """Seconds notes (on the grid) — or already beat-form notes — -> the notes.json contract:
    {"bpm", "time_signature", "notes": [{pitch, velocity, bar, beat, duration}]}."""
    bpb, beat_beats = parse_time_signature(time_signature)
    slots_per_bar = int(bpb * 4)
    spb = 60.0 / bpm * beat_beats
    out = []
    for n in notes:
        if "start_s" in n:
            slot = int(round(n["start_s"] / (spb * GRID)))
            val = nearest_value(n["dur_s"] / spb)
        else:
            slot = (int(n["bar"]) - 1) * slots_per_bar + int(round((float(n["beat"]) - 1) / GRID))
            val = str(n["duration"])
        bar = slot // slots_per_bar + 1
        beat = (slot % slots_per_bar) / 4.0 + 1.0
        out.append({"pitch": n["pitch"], "velocity": n["velocity"], "bar": bar,
                    "beat": int(beat) if beat == int(beat) else beat, "duration": val})
    return {"bpm": float(bpm), "time_signature": time_signature, "notes": out}


def beat_form_json(bpm: float, notes: list[dict], time_signature: str = "4/4") -> str:
    """The file exactly as an agent writes it: one note per line."""
    d = to_beat_form(bpm, notes, time_signature)
    rows = ",\n".join("  " + json.dumps(n, ensure_ascii=False) for n in d["notes"])
    return f'{{"bpm": {json.dumps(d["bpm"])}, "time_signature": "{d["time_signature"]}", "notes": [\n{rows}\n]}}\n'


# --------------------------------------------------------------------------- transcription errors

_MUTATION_NARRATION = {
    "pitch_shift": [
        "One of the pitches is off compared to the target. Re-transcribing.",
        "A note sounds a bit flat against the target. Starting over.",
        "The pitch is wrong in one place. Re-transcribing.",
    ],
    "octave": [
        "A note is an octave off from the target. Re-transcribing.",
        "I'm hearing one note a full octave wrong. Starting over.",
        "The register is off on one note. Re-transcribing.",
    ],
    "note_deletion": [
        "The melody sounds incomplete — a note seems to be missing. Re-transcribing.",
        "Compared to the target, the transcription is missing a note. Starting over.",
        "The phrase feels like it’s missing a beat — a note was dropped. Re-transcribing.",
    ],
    "note_insertion": [
        "There’s an extra note that wasn’t in the original melody. Re-transcribing.",
        "I’m hearing a note that doesn’t belong in this phrase. Starting over.",
        "The transcription has an extra event that the target doesn’t. Re-transcribing.",
    ],
    "timing_shift": [
        "The rhythm is off — a note’s timing doesn’t match the target. Re-transcribing.",
        "One note sounds out of time compared to the original. Starting over.",
        "The timing doesn’t line up for one of the notes. Re-transcribing.",
    ],
    "multi_note_corruption": [
        "Multiple notes sound wrong compared to the target. Re-transcribing from scratch.",
        "A couple of notes don’t match the original melody. Starting over.",
        "The melody deviates in several places. Re-transcribing.",
    ],
}
_COMPOUND_NARRATIONS = [
    "The melody doesn't match — several notes sound off compared to the target. Re-transcribing from scratch.",
    "This doesn't sound right. Multiple issues — some pitches are wrong and the phrasing is off. Starting over.",
    "The transcription deviates from the target in several places. Re-transcribing.",
    "Comparing against the target, the melody has multiple errors. Starting over from scratch.",
    "The transcription has noticeable differences from the target — re-transcribing.",
]


def _slot_of(n: dict) -> int:
    return (n["bar"] - 1) * 16 + int(round((n["beat"] - 1) / GRID))


def _set_slot(n: dict, slot: int) -> None:
    n["bar"] = slot // 16 + 1
    beat = (slot % 16) / 4.0 + 1.0
    n["beat"] = int(beat) if beat == int(beat) else beat


def _shift_start(n: dict, delta_slots: int) -> None:
    """Move a beat-form note by whole grid slots (the onset may cross into a neighbouring bar)."""
    _set_slot(n, max(0, _slot_of(n) + delta_slots))


def mutate_transcription(notes_beat: list[dict], rng: random.Random, n_muts: int | None = None,
                         kinds: tuple[str, ...] | None = None) -> tuple[list[dict], list[dict], str] | None:
    """Corrupt a beat-form transcription the way a hurried transcriber does: a wrong pitch, an
    octave, a dropped or added note, a note that lands on the wrong grid slot. Returns
    (wrong_notes, info_list, narration) or None if the phrase is too short to corrupt.
    n_muts: how many mutations (default 2..6 by phrase length); kinds: the mutation types to draw from."""
    if len(notes_beat) < 3:
        return None

    if n_muts is None:
        n_muts = rng.randint(2, max(2, min(6, len(notes_beat) // 5)))
    wrong = copy.deepcopy(notes_beat)
    infos: list[dict] = []

    for _ in range(n_muts):
        name = rng.choice(list(kinds or _MUTATION_NARRATION))
        idx = rng.randrange(len(wrong))
        if name == "pitch_shift":
            d = rng.choice([-2, -1, 1, 2])
            orig = wrong[idx]["pitch"]
            wrong[idx]["pitch"] = max(0, min(127, orig + d))
            infos.append({"type": name, "note_idx": idx, "delta": d})
        elif name == "octave":
            d = rng.choice([-12, 12])
            orig = wrong[idx]["pitch"]
            wrong[idx]["pitch"] = max(0, min(127, orig + d))
            infos.append({"type": name, "note_idx": idx, "delta": d})
        elif name == "note_deletion":
            if len(wrong) < 3:
                continue
            removed = wrong.pop(idx)
            infos.append({"type": name, "note_idx": idx, "removed_pitch": removed["pitch"]})
        elif name == "note_insertion":
            if not wrong:
                continue
            pitches = [n["pitch"] for n in wrong]
            phantom_pitch = rng.randint(max(0, min(pitches) - 3), min(127, max(pitches) + 3))
            slots = max(0, min(_slot_of(rng.choice(wrong)) + rng.randint(-4, 4),
                               max(map(_slot_of, wrong)) + 8))
            phantom = {"pitch": phantom_pitch, "velocity": rng.randint(60, 110),
                       "duration": rng.choice(["eighth", "quarter", "dotted eighth"])}
            _set_slot(phantom, slots)
            wrong.append(phantom)
            wrong.sort(key=lambda n: (_slot_of(n), n["pitch"]))
            infos.append({"type": name, "phantom": phantom})
        elif name == "timing_shift":
            shift = rng.choice([-1, 1]) * rng.randint(1, 3)
            _shift_start(wrong[idx], shift)
            infos.append({"type": name, "note_idx": idx, "shift_slots": shift})
        else:  # multi_note_corruption
            for j in rng.sample(range(len(wrong)), min(2, len(wrong))):
                d = rng.choice([-2, -1, 1, 2])
                orig = wrong[j]["pitch"]
                wrong[j]["pitch"] = max(0, min(127, orig + d))
                infos.append({"type": name, "note_idx": j, "delta": d})

    if not infos or wrong == copy.deepcopy(notes_beat):
        return None
    narration = rng.choice(_COMPOUND_NARRATIONS) if len(infos) > 2 else rng.choice(_MUTATION_NARRATION[infos[-1]["type"]])
    return wrong, infos, narration
