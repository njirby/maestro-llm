"""Tests for score_transcription_record (task_type=melody_transcription)."""
from __future__ import annotations

import json

from scripts.grade_agent_sft import score_transcription_record


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

def _make_transcription_record(
    *,
    oracle_notes: list[dict] | None = None,
    emitted_n_notes: int | None = None,
    has_reapy_insert: bool = True,
    has_audio_in_user: bool = True,
    last_role_assistant: bool = True,
    inject_snake_case: bool = False,
    inject_bold: bool = False,
) -> dict:
    oracle_notes = oracle_notes if oracle_notes is not None else [
        {"pitch": 60, "start_s": 0.0, "end_s": 0.5, "velocity": 95},   # C4
        {"pitch": 64, "start_s": 0.5, "end_s": 1.0, "velocity": 90},   # E4
        {"pitch": 67, "start_s": 1.0, "end_s": 1.5, "velocity": 92},   # G4
    ]
    n_oracle = len(oracle_notes)
    emitted_n_notes = emitted_n_notes if emitted_n_notes is not None else n_oracle
    duration_s = max(n["end_s"] for n in oracle_notes) if oracle_notes else 0.0

    messages = []

    # User — dispatch prompt with audio if requested
    user_content = "Transcribe this to MIDI on track 0." if not has_audio_in_user else (
        "<audio>\nTranscribe this to MIDI on track 0. Save JSON to /tmp/x.json."
    )
    messages.append({"role": "user", "content": user_content})

    # Assistant with per-note list naming pitches so pitch_coverage passes
    reason = "Listening to the target.\n\n"
    for i, n in enumerate(oracle_notes):
        reason += f"{i + 1}. MIDI {n['pitch']} at {n['start_s']:.2f}s, dur {n['end_s'] - n['start_s']:.2f}s, vel {n['velocity']}\n"
    if inject_snake_case:
        reason += "\nalso mention env_1_attack_mod_depth for testing\n"
    if inject_bold:
        reason += "\n**PLAN:** something\n"
    messages.append({"role": "assistant", "content": reason})

    # reapy MIDI insert call (optional)
    if has_reapy_insert:
        # Encode oracle pitches into the insert cmd so grader can match on pitches
        pitches_in_cmd = ", ".join(str(n["pitch"]) for n in oracle_notes)
        insert_cmd = (
            "python - <<'PY'\n"
            "import reapy\n"
            f"# pitches: {pitches_in_cmd}\n"
            "for p in []:\n"
            "    MIDI_InsertNote(take, False, False, 0, 100, 0, p, 90, False)\n"
            "PY"
        )
        messages.append({
            "role": "tool_call",
            "content": json.dumps({"name": "Bash", "arguments": {"command": insert_cmd}}),
        })
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "notes_inserted": emitted_n_notes}),
        })

    # (Sub-agents no longer write output files — the framework captures results.
    #  The MIDI insert tool_response already carries notes_inserted.)

    if last_role_assistant:
        messages.append({
            "role": "assistant",
            "content": f"Transcription complete. {emitted_n_notes} notes on track 0.",
        })

    return {
        "id": "sample_1_transcription",
        "task_type": "melody_transcription",
        "tools": "[]",
        "messages": messages,
        "audios": [],
        "meta": {
            "pipeline_version": "v3_transcription",
            "sample_id": "sample_1",
            "archetype": "keys",
            "source_midi_path": "/tmp/x.mid",
            "n_notes": n_oracle,
            "duration_s": duration_s,
            "pitch_range": [min(n["pitch"] for n in oracle_notes), max(n["pitch"] for n in oracle_notes)] if oracle_notes else [0, 0],
            "notes": oracle_notes,
            "output_file": "/tmp/x.json",
            "track_idx": 0,
        },
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_transcription_happy_path():
    record = _make_transcription_record()
    scores = score_transcription_record(record)
    assert scores["has_midi_insert"] == 1.0
    assert scores["result_communicated"] == 1.0
    assert scores["note_count_match"] == 1.0
    assert scores["pitch_coverage"] == 1.0
    assert scores["has_render_listen"] == 1.0
    assert scores["closing_assistant"] == 1.0
    assert scores["snake_case_clean"] == 1.0
    assert scores["format_consistent"] == 1.0
    assert scores["overall"] == 1.0


def test_transcription_missing_reapy_insert_dings():
    record = _make_transcription_record(has_reapy_insert=False)
    scores = score_transcription_record(record)
    assert scores["has_midi_insert"] == 0.0
    assert scores["overall"] < 1.0


def test_transcription_missing_midi_insert_dings_result():
    record = _make_transcription_record(has_reapy_insert=False)
    scores = score_transcription_record(record)
    assert scores["result_communicated"] == 0.0


def test_transcription_note_count_mismatch():
    record = _make_transcription_record(emitted_n_notes=2)  # oracle has 3
    scores = score_transcription_record(record)
    assert scores["note_count_match"] == 0.0


def test_transcription_no_audio_in_user_dings():
    record = _make_transcription_record(has_audio_in_user=False)
    scores = score_transcription_record(record)
    assert scores["has_render_listen"] == 0.0


def test_transcription_last_turn_not_assistant():
    record = _make_transcription_record(last_role_assistant=False)
    scores = score_transcription_record(record)
    assert scores["closing_assistant"] == 0.0


def test_transcription_snake_case_dings():
    record = _make_transcription_record(inject_snake_case=True)
    scores = score_transcription_record(record)
    assert scores["snake_case_clean"] < 1.0


def test_transcription_bold_dings():
    record = _make_transcription_record(inject_bold=True)
    scores = score_transcription_record(record)
    assert scores["format_consistent"] < 1.0


def test_transcription_pitch_coverage_partial():
    # Oracle has 3 pitches but assistant text only mentions 1
    oracle = [
        {"pitch": 60, "start_s": 0.0, "end_s": 0.5, "velocity": 95},
        {"pitch": 64, "start_s": 0.5, "end_s": 1.0, "velocity": 90},
        {"pitch": 67, "start_s": 1.0, "end_s": 1.5, "velocity": 92},
    ]
    record = _make_transcription_record(oracle_notes=oracle)
    # Replace the reasoning text to only mention one pitch — and strip the insert cmd's
    # pitch list so combined coverage is 1/3.
    record["messages"][1]["content"] = "Listening. I heard one note: MIDI 60"
    # Replace insert_cmd to not leak pitches
    for m in record["messages"]:
        if m.get("role") == "tool_call":
            c = json.loads(m["content"])
            cmd = c["arguments"]["command"]
            if "MIDI_InsertNote" in cmd:
                c["arguments"]["command"] = (
                    "python - <<'PY'\nimport reapy\nMIDI_InsertNote(t,False,False,0,100,0,0,0,False)\nPY"
                )
                m["content"] = json.dumps(c)
                break
    scores = score_transcription_record(record)
    # Exactly 1 of 3 oracle pitches (60) in text
    assert 0 < scores["pitch_coverage"] < 0.5
