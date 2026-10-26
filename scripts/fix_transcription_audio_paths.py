#!/usr/bin/env python3
"""Re-render transcription verification WAVs and remap audio paths.

The unified builder wrote verification renders to /tmp/agents/*/transcription_verify_*.wav.
These are DawDreamer renders of the init preset + transcribed MIDI notes.
3,000 of 8,851 have been cleaned. This script re-renders ALL of them to a
persistent location and rewrites the rollout audio paths.

Usage:
    python scripts/fix_transcription_audio_paths.py \
        --input outputs/sft_32k/rollouts/transcription_final_v1.jsonl \
        --output outputs/sft_32k/rollouts/transcription_final_v2.jsonl \
        --render-dir outputs/sft_32k/transcription_verify_audio
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import validate_ms_swift_multiturn_record


def _extract_notes_from_rollout(row: dict) -> list[list[dict]]:
    """Extract note lists from each InsertNote tool_call in the rollout.

    Returns a list of note lists — one per verification render (multiple if
    the transcription had retries with corrected notes).
    """
    all_notes: list[list[dict]] = []
    for m in row.get("messages", []):
        if m.get("role") != "tool_call":
            continue
        content = m.get("content", "")
        try:
            tc = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            continue
        cmd = tc.get("arguments", {}).get("command", "")
        if "MIDI_InsertNote" not in cmd:
            continue

        py_code = cmd.split("<<'PY'")[1].split("\nPY")[0] if "<<'PY'" in cmd else cmd
        match = re.search(r"notes\s*=\s*(\[.*?\])", py_code, re.DOTALL)
        if match:
            notes_str = match.group(1).replace("\\n", "")
            try:
                notes = json.loads(notes_str)
                all_notes.append(notes)
            except json.JSONDecodeError:
                pass
    return all_notes


def _render_verify_wav(notes: list[dict], out_path: Path) -> bool:
    """Render verification WAV using DawDreamer (init preset + notes)."""
    from maestro.render.dawdreamer import notes_from_dicts, render_preset_audio

    init_path = ROOT / "maestro" / "synth" / "init_preset.json"
    with open(init_path) as f:
        init_preset = json.load(f)

    note_tuples = notes_from_dicts(notes)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        render_preset_audio(init_preset, note_tuples, out_path=str(out_path), tail_s=1.0)
        return True
    except Exception as exc:
        print(f"  WARNING: render failed for {out_path.name}: {exc}", file=sys.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=ROOT / "outputs/sft_32k/rollouts/transcription_final_v1.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "outputs/sft_32k/rollouts/transcription_final_v2.jsonl")
    ap.add_argument("--render-dir", type=Path, default=ROOT / "outputs/sft_32k/transcription_verify_audio")
    ap.add_argument("--skip-existing", action="store_true",
                    help="Skip re-rendering if output WAV already exists")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    print("Step 1: Scanning rollouts and rendering verification WAVs...", flush=True)

    stats = {"total": 0, "rewritten": 0, "rendered": 0, "render_failed": 0, "skipped_existing": 0}
    rows_out: list[dict] = []

    with open(args.input) as fin:
        for line_idx, line in enumerate(fin):
            row = json.loads(line)
            stats["total"] += 1
            sample_id = row.get("id", f"row_{line_idx}")

            tmp_indices = []
            for i, audio_path in enumerate(row.get("audios", [])):
                if "/tmp/" in audio_path and "transcription_verify" in audio_path:
                    tmp_indices.append(i)

            if not tmp_indices:
                rows_out.append(row)
                stats["rewritten"] += 1
                continue

            note_lists = _extract_notes_from_rollout(row)

            if len(note_lists) != len(tmp_indices):
                print(f"  WARNING [{sample_id}]: {len(note_lists)} note lists vs "
                      f"{len(tmp_indices)} verify WAVs — skipping", file=sys.stderr)
                continue

            new_audios = list(row["audios"])
            all_ok = True

            for j, (audio_idx, notes) in enumerate(zip(tmp_indices, note_lists)):
                out_wav = args.render_dir / sample_id / f"transcription_verify_{j + 1}.wav"

                if args.skip_existing and out_wav.exists():
                    stats["skipped_existing"] += 1
                else:
                    ok = _render_verify_wav(notes, out_wav)
                    if ok:
                        stats["rendered"] += 1
                    else:
                        stats["render_failed"] += 1
                        all_ok = False
                        break

                new_audios[audio_idx] = str(out_wav)

            if all_ok:
                row["audios"] = new_audios
                rows_out.append(row)
                stats["rewritten"] += 1

            if (line_idx + 1) % 1000 == 0:
                print(f"  [{line_idx + 1}/{stats['total']}] "
                      f"rendered={stats['rendered']}, rewritten={stats['rewritten']}", flush=True)

    print(f"\n  Total: {stats['total']}", flush=True)
    print(f"  Rewritten: {stats['rewritten']}", flush=True)
    print(f"  Rendered: {stats['rendered']}", flush=True)
    print(f"  Render failed: {stats['render_failed']}", flush=True)
    if stats["skipped_existing"]:
        print(f"  Skipped existing: {stats['skipped_existing']}", flush=True)

    print(f"\nStep 2: Writing {args.output}...", flush=True)
    with open(args.output, "w") as fout:
        for row in rows_out:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not args.skip_validate:
        print("\nStep 3: Validating...", flush=True)
        valid = 0
        missing_audio = 0
        errors: dict[str, int] = {}
        with open(args.output) as f:
            for line in f:
                row = json.loads(line)
                errs = validate_ms_swift_multiturn_record(row)
                if errs:
                    for e in errs:
                        errors[e] = errors.get(e, 0) + 1
                else:
                    valid += 1
                for a in row.get("audios", []):
                    if not Path(a).exists():
                        missing_audio += 1

        print(f"  {valid}/{len(rows_out)} valid, {missing_audio} missing audio files", flush=True)
        if errors:
            for e, c in sorted(errors.items(), key=lambda x: -x[1])[:5]:
                print(f"    {e}: {c}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
