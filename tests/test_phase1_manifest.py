from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maestro.data.phase1 import (
    build_manifest_from_sources,
    write_manifest,
)


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


def test_build_manifest_from_mixed_sources(tmp_path: Path):
    multi_song = tmp_path / "multi" / "song_a"
    _touch(multi_song / "mix.wav")
    _touch(multi_song / "drums.wav")
    _touch(multi_song / "bass.wav")
    _touch(multi_song / "arrangement.mid")

    midi_root = tmp_path / "midi"
    _touch(midi_root / "artist" / "track_01.mid")

    specs = [
        {
            "name": "toy_multi",
            "kind": "multitrack",
            "root": str(tmp_path / "multi"),
            "split": "train",
            "recursive": True,
            "min_stems": 2,
        },
        {
            "name": "toy_midi",
            "kind": "midi_only",
            "root": str(midi_root),
            "split": "train",
            "recursive": True,
        },
    ]

    rows = build_manifest_from_sources(specs, strict_missing_roots=True)
    assert len(rows) == 2

    by_source = {row.source: row for row in rows}
    multi = by_source["toy_multi"]
    assert multi.mix_path is not None
    assert len(multi.stems) == 2
    assert len(multi.midi_paths) == 1
    assert "remove_entire_track" in multi.perturbations
    assert "quantize_midi_too_hard" in multi.perturbations

    midi = by_source["toy_midi"]
    assert midi.mix_path is None
    assert len(midi.stems) == 0
    assert len(midi.midi_paths) == 1
    assert "quantize_midi_too_hard" in midi.perturbations
    assert "remove_entire_track" not in midi.perturbations


def test_write_manifest_outputs_summary(tmp_path: Path):
    song_dir = tmp_path / "slakh_like" / "song_b"
    _touch(song_dir / "mix.wav")
    _touch(song_dir / "piano.wav")
    _touch(song_dir / "vocal.wav")

    specs = [
        {
            "name": "toy_source",
            "kind": "multitrack",
            "root": str(tmp_path / "slakh_like"),
            "split": "train",
            "recursive": True,
            "min_stems": 2,
        }
    ]
    rows = build_manifest_from_sources(specs, strict_missing_roots=True)

    manifest_path = tmp_path / "out" / "manifest.jsonl"
    summary_path = tmp_path / "out" / "summary.json"
    summary = write_manifest(rows, str(manifest_path), str(summary_path))

    assert manifest_path.exists()
    assert summary_path.exists()
    assert summary["total_songs"] == 1
    assert summary["by_source"]["toy_source"]["songs"] == 1

    lines = manifest_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["source"] == "toy_source"
    assert row["stem_count"] == 2
