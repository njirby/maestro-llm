#!/usr/bin/env python3
"""Build melody-transcription SFT v3.

The main agent invokes this subagent right after creating a MIDI track in
REAPER. The subagent's job: listen to the target audio and use reapy to
insert MIDI notes on the REAPER track, then save the final note list as JSON.

Flow per record (5 messages):
  user:       <audio> + "transcribe to MIDI on track {N}, save to <file>"
  assistant:  "Transcribing the target melody to track {N}."
  tool_call:  bash python heredoc — single call that:
                1) uses reapy to insert MIDI notes via MIDI_InsertNote
                2) writes the note list to transcription.json
  tool_resp:  {"status":"ok","notes_inserted":N,"file":"/tmp/.../transcription.json","duration_s":X}
  assistant:  "Done. N notes on track {N} over ~Xs."

Build-time oracle:
  Notes come from pretty_midi.PrettyMIDI(source_midi_path) — the same MIDI
  file that drove `vita` to render the target audio. The transcription is
  exactly correct at build time; no Omni call is needed during synthesis.

Note representation:
  Each note is {"pitch": 36, "start_s": 0.0, "dur_s": 1.25, "velocity": 90}.
  Pitch is a MIDI int (matches what RPR_MIDI_InsertNote takes directly —
  zero conversion). `dur_s` instead of `end_s` aligns with how notes are
  described verbally ("dur 1.25s") and saves a redundant subtraction at
  insert time.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pretty_midi

from scripts.agent_sft_common import (  # type: ignore
    assert_valid_ms_swift_multiturn_record,
    load_manifest_entries,
    _bash_tool_response,
    _REAPY_HELPER,
    _tool_call,
    _wrap_as_bash,
)


# Tools available to the transcription agent at inference
_TRANSCRIPTION_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": (
                "Execute shell/Python commands. Used here to dispatch Lua that "
                "inserts MIDI notes on a REAPER track and writes the transcription "
                "JSON in a single call."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    }
]


def load_notes_from_midi(midi_path: str | Path) -> list[dict]:
    """Return a sorted list of note dicts loaded from a MIDI file.

    Each note: {"pitch": int, "start_s": float, "dur_s": float, "velocity": int}.
    Pitch is a MIDI int — matches RPR_MIDI_InsertNote directly.
    """
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes: list[dict] = []
    for inst in pm.instruments:
        for n in inst.notes:
            start_s = round(float(n.start), 4)
            end_s = round(float(n.end), 4)
            notes.append({
                "pitch": int(n.pitch),
                "start_s": start_s,
                "dur_s": round(end_s - start_s, 4),
                "velocity": int(n.velocity),
            })
    notes.sort(key=lambda n: (n["start_s"], n["pitch"]))
    return notes


def build_insert_and_write_cmd(
    notes: list[dict],
    output_path: str | Path,
    *,
    track_idx: int = 0,
) -> str:
    """Build a bash command that uses reapy to insert MIDI notes and
    writes the transcription JSON file.
    """
    n_notes = len(notes)
    duration_s = round(
        max((n["start_s"] + n["dur_s"] for n in notes), default=0.0),
        2,
    )
    item_end = round(
        max((n["start_s"] + n["dur_s"] for n in notes), default=1.0) + 0.25,
        4,
    )
    output_path_s = str(output_path)
    output_dir_s = str(Path(output_path).parent)

    notes_literal = json.dumps(notes, ensure_ascii=False)
    payload_json = json.dumps(
        {"notes": notes, "n_notes": n_notes, "duration_s": duration_s},
        ensure_ascii=False,
    )

    snippet = (
        _REAPY_HELPER
        + "from pathlib import Path\n"
        f"notes = {notes_literal}\n"
        f"with reapy.inside_reaper():\n"
        f"    track = RPR.GetTrack(0, {track_idx})\n"
        f"    item = RPR.CreateNewMIDIItemInProj(track, 0.0, {item_end}, False)[0]\n"
        f"    take = RPR.GetActiveTake(item)\n"
        f"    bpm = 120\n"
        f"    ppb = 960\n"
        f"    for n in notes:\n"
        f"        start_ppq = int(n['start_s'] * (bpm / 60.0) * ppb)\n"
        f"        end_ppq = int((n['start_s'] + n['dur_s']) * (bpm / 60.0) * ppb)\n"
        f"        RPR.MIDI_InsertNote(take, False, False, start_ppq, end_ppq, 0, n['pitch'], n['velocity'], True)\n"
        f"    RPR.MIDI_Sort(take)\n"
        f"Path({output_dir_s!r}).mkdir(parents=True, exist_ok=True)\n"
        f"Path({output_path_s!r}).write_text({payload_json!r} + '\\n')\n"
        f"print(json.dumps({{'status': 'ok', 'notes_inserted': {n_notes}, "
        f"'file': {output_path_s!r}, 'duration_s': {duration_s}}}))\n"
    )
    return _wrap_as_bash(snippet)


def build_transcription_record(
    *,
    sample_id: str,
    archetype: str,
    target_audio_path: Path,
    source_midi_path: str | Path,
    output_dir: Path,
    track_idx: int = 0,
) -> dict | None:
    """Build one SFT transcription record. Returns None if the MIDI has no notes.

    No Omni calls at build time — notes come straight from the oracle MIDI file.
    """
    notes = load_notes_from_midi(source_midi_path)
    if not notes:
        return None

    n_notes = len(notes)
    duration_s = round(
        max(n["start_s"] + n["dur_s"] for n in notes),
        2,
    )
    pitch_range = [
        min(n["pitch"] for n in notes),
        max(n["pitch"] for n in notes),
    ]

    # Use claw-code-style agent IDs and place the output under the agent's
    # per-sample directory. The path embeds the deterministic agent ID so
    # SFT records mirror the runtime harness's outputFile naming.
    from scripts.agent_sft_common import make_agent_id  # type: ignore
    agent_id = make_agent_id(sample_id, "melody_transcription")
    output_file = Path(output_dir) / sample_id / f"{agent_id}.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    messages: list[dict] = []
    audio_assets = [str(target_audio_path)]

    # 1. User dispatch message
    messages.append({
        "role": "user",
        "content": (
            f"<audio>\n"
            f"Transcribe this melody into MIDI notes on REAPER track {track_idx} "
            f"(the main agent just created the track). Write Python code "
            f"(reapy → MIDI_InsertNote) that inserts the "
            f"notes, and write the final note list as JSON to {output_file} "
            f"with shape {{'notes': [...], 'n_notes': N, 'duration_s': X}}."
        ),
    })

    # 2. Brief acknowledgment — no narration, no per-note enumeration
    messages.append({
        "role": "assistant",
        "content": f"Transcribing the target melody to track {track_idx}.",
    })

    # 3. Single bash call: insert notes + write JSON
    cmd = build_insert_and_write_cmd(notes, output_file, track_idx=track_idx)
    messages.append(_tool_call("Bash", {"command": cmd}))

    # 4. Tool response — persist the file to disk for real so it exists at
    # the path the transcript references (live-exec grading reuses this).
    payload = {
        "status": "completed",
        "notes": notes,
        "n_notes": n_notes,
        "duration_s": duration_s,
    }
    with open(output_file, "w") as f:
        json.dump(payload, f)
        f.write("\n")
    _insert_stdout = json.dumps({
        "status": "ok",
        "notes_inserted": n_notes,
        "file": str(output_file),
        "duration_s": duration_s,
    }) + "\n"
    messages.append(_bash_tool_response(_insert_stdout))

    # 5. Closing assistant — validator requires last message to be assistant.
    messages.append({
        "role": "assistant",
        "content": (
            f"Done. {n_notes} notes on track {track_idx} over ~{duration_s:.1f}s."
        ),
    })

    record = {
        "id": f"{sample_id}_transcription",
        "task_type": "melody_transcription",
        "tools": _TRANSCRIPTION_TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "meta": {
            "pipeline_version": "v3_transcription",
            "sample_id": sample_id,
            "archetype": archetype,
            "source_midi_path": str(source_midi_path),
            "n_notes": n_notes,
            "duration_s": duration_s,
            "pitch_range": pitch_range,
            "notes": notes,
            "agent_id": agent_id,
            "output_file": str(output_file),
            "track_idx": track_idx,
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return record


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build melody-transcription SFT v3. One record per manifest entry. "
            "No Omni calls at build time — notes come from the oracle MIDI file."
        )
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/transcription_outputs"),
        help="Where transcription.json files land (must match the path the main "
             "agent's dispatch-prompt references).",
    )
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument(
        "--track-idx", type=int, default=0,
        help="REAPER track index to insert notes into (default 0 — matches main-agent's create-track turn).",
    )
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args()

    entries = load_manifest_entries(args.manifest, max_samples=args.max_samples)
    all_records: list[dict] = []

    def _process(entry: dict) -> dict | None:
        sample_id = str(entry["sample_id"])
        archetype = str(entry.get("archetype", "synth"))
        target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
        source_midi_path = entry.get("source_midi_path")
        if not source_midi_path:
            return None
        return build_transcription_record(
            sample_id=sample_id,
            archetype=archetype,
            target_audio_path=target_audio_path,
            source_midi_path=source_midi_path,
            output_dir=args.output_dir,
            track_idx=args.track_idx,
        )

    def _safe_process(entry):
        try:
            return entry, _process(entry), None
        except Exception as exc:
            import traceback
            return entry, None, (exc, traceback.format_exc())

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = [pool.submit(_safe_process, e) for e in entries]
            for i, fut in enumerate(as_completed(futs)):
                entry, rec, err = fut.result()
                if err:
                    print(f"WARNING: {entry.get('sample_id', '?')} failed: {err[0]}")
                    print(err[1])
                elif rec:
                    all_records.append(rec)
                    print(
                        f"[{i + 1}/{len(entries)}] {entry['sample_id']}: "
                        f"{rec['meta']['n_notes']} notes, "
                        f"{rec['meta']['duration_s']:.2f}s",
                        flush=True,
                    )
    else:
        for i, entry in enumerate(entries):
            entry, rec, err = _safe_process(entry)
            if err:
                print(f"WARNING: {entry.get('sample_id', '?')} failed: {err[0]}")
                print(err[1])
            elif rec:
                all_records.append(rec)
                print(
                    f"[{i + 1}/{len(entries)}] {entry['sample_id']}: "
                    f"{rec['meta']['n_notes']} notes, "
                    f"{rec['meta']['duration_s']:.2f}s",
                    flush=True,
                )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(
        f"Wrote {len(all_records)} transcription records to {args.out_jsonl}",
        flush=True,
    )


if __name__ == "__main__":
    main()
