"""maestro.notation: the skill's notes.json contract (measures and note values) round-trips, and the skill's
standalone copy of the vocabulary stays in step with the builder's."""
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from maestro import notation  # noqa: E402

VITAL = REPO / "opencode" / "skills" / "vital"


def test_round_trip_is_exact():
    bpm = 148.0
    secs = [
        {"pitch": 74, "velocity": 116, "start_s": 0.3026, "dur_s": 0.5263},
        {"pitch": 65, "velocity": 110, "start_s": 0.9342, "dur_s": 1.1908},
        {"pitch": 74, "velocity": 109, "start_s": 3.4605, "dur_s": 0.2632},
    ]
    q = notation.quantize(bpm, secs)
    form = notation.to_beat_form(bpm, q)
    back, bpm2 = notation.to_seconds(form)
    assert bpm2 == bpm
    assert [(n["start_s"], n["dur_s"], n["pitch"], n["velocity"]) for n in q] == \
           [(n["start_s"], n["dur_s"], n["pitch"], n["velocity"]) for n in back]
    assert notation.to_beat_form(bpm, form["notes"]) == form  # idempotent on beat form


def test_beat_form_json_shape():
    bpm = 95.0
    q = notation.quantize(bpm, [{"pitch": 74, "velocity": 116, "start_s": 0.3026, "dur_s": 0.5263}])
    text = notation.beat_form_json(bpm, q)
    d = json.loads(text)
    assert d["time_signature"] == "4/4" and d["bpm"] == bpm
    n = d["notes"][0]
    assert set(n) == {"pitch", "velocity", "bar", "beat", "duration"}
    assert n["duration"] in notation.VALUE_BEATS
    assert n["bar"] == 1 and 1 <= n["beat"] <= 4
    assert text.count("\n") == 2 + len(d["notes"])  # one note per line


def test_quantize_snaps_to_grid_and_drops_merged():
    bpm = 120.0  # 16th = 0.125 s
    q = notation.quantize(bpm, [
        {"pitch": 60, "velocity": 100, "start_s": 0.01, "dur_s": 0.5},
        {"pitch": 64, "velocity": 90, "start_s": 0.14, "dur_s": 0.5},   # snaps to the same grid slot as the next
        {"pitch": 64, "velocity": 100, "start_s": 0.135, "dur_s": 0.5},  # same slot, same pitch: merged away
    ])
    spb16 = 0.125
    assert [n["start_s"] for n in q] == [0.0, spb16]
    assert [n["pitch"] for n in q] == [60, 64]
    for n in q:
        assert abs(n["start_s"] / spb16 - round(n["start_s"] / spb16)) < 1e-9


def test_triplet_values():
    bpm = 120.0  # 16th = 0.25 s
    # the corpus shape: onsets on the 16th grid, durations that only a triplet value can write
    q = notation.quantize(bpm, [{"pitch": 60 + i, "velocity": 100, "start_s": i * 0.125, "dur_s": 1 / 6}
                                for i in range(3)])
    form = notation.to_beat_form(bpm, q)
    # 1/6 s at 120 bpm = 1/3 beat = a triplet eighth (3-in-2 eighths)
    assert [n["duration"] for n in form["notes"]] == ["triplet eighth"] * 3
    assert [n["beat"] for n in form["notes"]] == [1, 1.25, 1.5]
    back, _ = notation.to_seconds(form)
    assert [n["dur_s"] for n in back] == [n["dur_s"] for n in q]
    for name, beats in [("triplet quarter", 2 / 3), ("triplet eighth", 1 / 3),
                        ("triplet sixteenth", 1 / 6), ("triplet thirty-second", 1 / 12)]:
        assert notation.nearest_value(beats) == name
    assert notation.nearest_value(0.5) == "eighth"  # not ambiguous with the triplets


def test_legacy_seconds_still_accepted():
    notes, bpm = notation.to_seconds({"bpm": 120.0, "notes": [
        {"pitch": 60, "velocity": 100, "start_s": 0.0, "dur_s": 0.5}]})
    assert notes == [{"pitch": 60, "velocity": 100, "start_s": 0.0, "dur_s": 0.5}] and bpm == 120.0
    notes, _ = notation.to_seconds('[{"pitch": 60, "start_s": 0.5, "dur_s": 1.0}]')
    assert notes[0]["velocity"] == 100


def test_bad_input_rejected():
    with pytest.raises(notation.NotationError):
        notation.to_seconds({"bpm": 120, "notes": [{"pitch": 60, "bar": 1, "beat": 1, "duration": "semitone"}]})
    with pytest.raises(notation.NotationError):
        notation.to_seconds({"bpm": 120, "notes": [{"pitch": 60, "bar": 0, "beat": 1, "duration": "eighth"}]})
    with pytest.raises(notation.NotationError):
        notation.parse_time_signature("7/7")


@pytest.mark.parametrize("ts, want", [("4/4", (4, 1.0)), ("3/4", (3, 1.0)), ("6/8", (2, 1.5)), ("9/8", (3, 1.5))])
def test_time_signatures(ts, want):
    assert notation.parse_time_signature(ts) == want


def test_skill_vocabulary_matches_builder():
    """The skill script is standalone: it carries its own NOTE_VALUES / parse_time_signature. Run it in a
    clean interpreter and compare against the builder's copies."""
    code = (
        "import json, sys; sys.path.insert(0, %r); import vitallib; "
        "print(json.dumps([vitallib.NOTE_VALUES, [vitallib.parse_time_signature(t) for t in ('4/4','3/4','6/8')]]))"
    ) % str(VITAL / "scripts")
    got = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                    check=True).stdout)
    assert got[0] == notation.VALUE_BEATS
    assert [list(x) for x in got[1]] == [list(notation.parse_time_signature(t)) for t in ("4/4", "3/4", "6/8")]


def test_mutations_change_the_phrase_and_parse():
    bpm = 140.0
    q = notation.quantize(bpm, [{"pitch": 72, "velocity": 100, "start_s": 0.12 * i, "dur_s": 0.2} for i in range(8)])
    notes_b = notation.to_beat_form(bpm, q)["notes"]
    rng = random.Random(3)
    made = 0
    for _ in range(8):
        mut = notation.mutate_transcription(notes_b, rng)
        if mut is None:
            continue
        made += 1
        wrong, infos, narration = mut
        assert wrong != notes_b and infos and narration
        notation.to_seconds({"bpm": bpm, "notes": wrong})  # still a valid beat form
    assert made >= 3
    assert notation.mutate_transcription(notes_b[:2], rng) is None
