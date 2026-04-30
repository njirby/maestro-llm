#!/usr/bin/env python3
"""Build melody-transcription SFT v4 — self-verifying transcription agent.

The transcription agent listens to the target audio, transcribes it to MIDI,
renders its own output through the init preset, listens to verify, and
re-transcribes from scratch if the melody doesn't match. Up to 4 total
attempts; the last always succeeds naturally.

Flow per record (variable length):
  user:       <audio> + "transcribe to MIDI on track {N}, save to <file>"
  assistant:  "Transcribing the target melody to track {N}."
  tool_call:  bash — reapy MIDI insert + write JSON
  tool_resp:  {status: ok, notes_inserted: N, file: path}
  assistant:  "Verifying — rendering transcribed notes through default preset."
  tool_call:  bash — DawDreamer render verify
  tool_resp:  {listen_probe: ...}
  assistant:  "Listening to the verification render."
  tool_call:  Read (verify WAV)
  tool_resp:  <audio>
  --- if mismatch, perceptual narration + retry from insert ---
  assistant:  "The melody matches the target."

Mistake injection:
  ~15% of samples get 1-3 wrong attempts before succeeding. Each wrong
  attempt applies multiple mutations (2-6, scaled to melody length) from
  the catalog (pitch shift, octave error, note deletion, insertion,
  timing shift, multi-note corruption). Detection narration is
  perceptual — no exact note indices or semitone deltas.

Code mistakes (from CodeMutation) are a separate layer and don't count
as transcription retries.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random as _random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pretty_midi

from maestro.render.dawdreamer import notes_from_dicts, render_preset_audio
from scripts.agent_sft_common import (  # type: ignore
    apply_transcription_mutations,
    assert_valid_ms_swift_multiturn_record,
    build_render_verify_snippet,
    load_manifest_entries,
    make_agent_id,
    _bash_tool_response,
    _emit_listen_sequence,
    _REAPY_HELPER,
    _tool_call,
    _wrap_as_bash,
)


_TRANSCRIPTION_V4_TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "Bash",
            "description": (
                "Execute shell/Python commands. Used to insert MIDI notes on a "
                "REAPER track via reapy and to render verification audio via "
                "DawDreamer."
            ),
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Read",
            "description": "Read a file. Used here to listen to rendered audio.",
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string"}},
                "required": ["file_path"],
            },
        },
    },
]


@dataclass
class TranscriptionResult:
    record: dict | None
    notes: list[dict] = field(default_factory=list)
    output_file: str = ""
    n_notes: int = 0
    duration_s: float = 0.0


def load_notes_from_midi(midi_path: str | Path) -> list[dict]:
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


def build_insert_cmd(
    notes: list[dict],
    *,
    track_idx: int = 0,
) -> str:
    n_notes = len(notes)
    duration_s = round(
        max((n["start_s"] + n["dur_s"] for n in notes), default=0.0), 2,
    )
    item_end = round(
        max((n["start_s"] + n["dur_s"] for n in notes), default=1.0) + 0.25, 4,
    )
    notes_literal = json.dumps(notes, ensure_ascii=False)
    snippet = (
        _REAPY_HELPER
        + f"notes = {notes_literal}\n"
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
        f"print(json.dumps({{'status': 'ok', 'notes_inserted': {n_notes}, "
        f"'duration_s': {duration_s}}}))\n"
    )
    return _wrap_as_bash(snippet)


def _load_init_preset() -> dict:
    with open(ROOT / "maestro" / "synth" / "init_preset.json") as f:
        return json.load(f)


_INIT_PRESET: dict | None = None


def _get_init_preset() -> dict:
    global _INIT_PRESET
    if _INIT_PRESET is None:
        _INIT_PRESET = _load_init_preset()
    return _INIT_PRESET


def _pick_n_retries(rng: _random.Random) -> int:
    """Among mistake samples, pick how many retries: 1 (70%), 2 (20%), 3 (10%)."""
    r = rng.random()
    if r < 0.70:
        return 1
    elif r < 0.90:
        return 2
    return 3


def build_transcription_record_v4(
    *,
    sample_id: str,
    archetype: str,
    target_audio_path: Path,
    source_midi_path: str | Path,
    output_dir: Path,
    track_idx: int = 0,
    mistake_rate: float = 0.0,
    seed: int = 42,
) -> TranscriptionResult:
    notes = load_notes_from_midi(source_midi_path)
    if not notes:
        return TranscriptionResult(record=None)

    n_notes = len(notes)
    duration_s = round(max(n["start_s"] + n["dur_s"] for n in notes), 2)
    pitch_range = [min(n["pitch"] for n in notes), max(n["pitch"] for n in notes)]
    init_preset = _get_init_preset()

    agent_id = make_agent_id(sample_id, "melody_transcription")
    output_file = Path(output_dir) / sample_id / f"{agent_id}.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Deterministic RNG for this sample
    sid_int = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
    rng = _random.Random(seed + sid_int + 7919)

    # Mistake decision
    inject_mistake = (
        mistake_rate > 0.0
        and rng.random() < mistake_rate
        and n_notes >= 3
    )
    n_retries = _pick_n_retries(rng) if inject_mistake else 0
    total_attempts = n_retries + 1

    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path)]
    mutations_applied: list[list[dict]] = []

    # ── Phase 1: User dispatch ──
    messages.append({
        "role": "user",
        "content": (
            f"<audio>\n"
            f"Transcribe this melody into MIDI notes on REAPER track {track_idx} "
            f"(the main agent just created the track). Write Python code "
            f"(reapy → MIDI_InsertNote) that inserts the notes. "
            f"After inserting, render your output through the default Vital "
            f"preset and listen to verify it matches the target melody. "
            f"If it doesn't match, re-transcribe from scratch."
        ),
    })

    messages.append({
        "role": "assistant",
        "content": f"Transcribing the target melody to track {track_idx}.",
    })

    # ── Phase 2-4: Attempt loop ──
    for attempt_idx in range(total_attempts):
        is_last = (attempt_idx == total_attempts - 1)

        # Pick notes for this attempt
        if is_last:
            attempt_notes = notes  # oracle (correct)
            attempt_narration = None
        else:
            result = apply_transcription_mutations(notes, rng)
            if result is None:
                attempt_notes = notes
                attempt_narration = None
                is_last = True
            else:
                attempt_notes, infos, attempt_narration = result
                mutations_applied.append(infos)

        attempt_n_notes = len(attempt_notes)
        attempt_duration_s = round(
            max((n["start_s"] + n["dur_s"] for n in attempt_notes), default=0.0), 2,
        )

        # Build the output path for this attempt
        if is_last:
            attempt_output = output_file
        else:
            attempt_agent_id = make_agent_id(
                sample_id, "melody_transcription", f"attempt{attempt_idx + 1}",
            )
            attempt_output = Path(output_dir) / sample_id / f"{attempt_agent_id}.md"

        # ── Insert MIDI notes ──
        cmd = build_insert_cmd(
            attempt_notes, track_idx=track_idx,
        )
        messages.append(_tool_call("Bash", {"command": cmd}))

        attempt_output.parent.mkdir(parents=True, exist_ok=True)

        insert_stdout = json.dumps({
            "status": "ok",
            "notes_inserted": attempt_n_notes,
            "duration_s": attempt_duration_s,
        }) + "\n"
        messages.append(_bash_tool_response(insert_stdout))

        # ── Render verify ──
        verify_wav = str(
            Path(output_dir)
            / sample_id
            / f"transcription_verify_{attempt_idx + 1}.wav"
        )
        note_tuples = notes_from_dicts(attempt_notes)

        messages.append({
            "role": "assistant",
            "content": (
                "Verifying — rendering the transcribed notes through the "
                "default Vital preset to compare against the target."
                if attempt_idx == 0
                else "Verifying the corrected transcription."
            ),
        })

        verify_snippet = build_render_verify_snippet(
            out_path=verify_wav, notes_override=note_tuples,
        )
        messages.append(_tool_call("Bash", {"command": _wrap_as_bash(verify_snippet)}))

        # Render at build time
        try:
            Path(verify_wav).parent.mkdir(parents=True, exist_ok=True)
            render_preset_audio(init_preset, note_tuples, out_path=verify_wav, tail_s=1.0)
        except Exception as exc:
            print(f"  WARNING: verify render failed for {sample_id} attempt {attempt_idx + 1}: {exc}")

        audio_assets.append(verify_wav)

        probe_stdout = json.dumps({
            "listen_probe": {"path": verify_wav, "exists": True},
        }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, verify_wav,
            probe_stdout=probe_stdout,
            listen_text=(
                "Listening to the verification render."
                if attempt_idx == 0
                else "Listening to the corrected transcription."
            ),
        )

        # ── Verdict ──
        if is_last:
            messages.append({
                "role": "assistant",
                "content": "The melody matches the target.",
            })
        else:
            messages.append({
                "role": "assistant",
                "content": attempt_narration,
            })

    # Persist the final assistant message to disk so the main agent's reference
    # path is valid.  In claw-code, outputFile = agent's last response (text).
    with open(output_file, "w") as f:
        f.write("The melody matches the target.\n")

    record = {
        "id": f"{sample_id}_transcription",
        "task_type": "melody_transcription",
        "tools": _TRANSCRIPTION_V4_TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "meta": {
            "pipeline_version": "v4_transcription",
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
            "n_retries": n_retries,
            "mutations_applied": mutations_applied,
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return TranscriptionResult(
        record=record,
        notes=notes,
        output_file=str(output_file),
        n_notes=n_notes,
        duration_s=duration_s,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build melody-transcription SFT v4. Self-verifying: transcribe → "
            "render → listen → retry up to 4 attempts."
        )
    )
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--out-jsonl", required=True, type=Path)
    ap.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/agents"),
    )
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--track-idx", type=int, default=0)
    ap.add_argument("--transcription-mistake-rate", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
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
        result = build_transcription_record_v4(
            sample_id=sample_id,
            archetype=archetype,
            target_audio_path=target_audio_path,
            source_midi_path=source_midi_path,
            output_dir=args.output_dir,
            track_idx=args.track_idx,
            mistake_rate=args.transcription_mistake_rate,
            seed=args.seed,
        )
        return result.record

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
                    retries = rec["meta"]["n_retries"]
                    muts = rec["meta"]["mutations_applied"]
                    n_total = sum(len(a) for a in muts)
                    print(
                        f"[{i + 1}/{len(entries)}] {entry['sample_id']}: "
                        f"{rec['meta']['n_notes']} notes, "
                        f"{rec['meta']['duration_s']:.2f}s, "
                        f"{retries} retries"
                        f"{f' ({n_total} mutations)' if muts else ''}",
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
                retries = rec["meta"]["n_retries"]
                muts = rec["meta"]["mutations_applied"]
                n_total = sum(len(a) for a in muts)
                print(
                    f"[{i + 1}/{len(entries)}] {entry['sample_id']}: "
                    f"{rec['meta']['n_notes']} notes, "
                    f"{rec['meta']['duration_s']:.2f}s, "
                    f"{retries} retries"
                    f"{f' ({n_total} mutations)' if muts else ''}",
                    flush=True,
                )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_jsonl, "w") as f:
        for r in all_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_mistake = sum(1 for r in all_records if r["meta"]["n_retries"] > 0)
    print(
        f"Wrote {len(all_records)} transcription records to {args.out_jsonl} "
        f"({n_mistake} with mistakes)",
        flush=True,
    )


if __name__ == "__main__":
    main()
