"""
Phase 1 SFT data manifest builder for stem/MIDI corpora.

This module intentionally stays simple:
- no license gating
- minimal assumptions about dataset internals
- one unified JSONL manifest format for downstream perturbation generation
"""

from __future__ import annotations

import json
import os
import pathlib as pl
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Optional

AUDIO_EXTS = {
    ".wav",
    ".flac",
    ".aif",
    ".aiff",
    ".mp3",
    ".ogg",
    ".m4a",
}
MIDI_EXTS = {".mid", ".midi"}

MIX_BASENAME_PRIORITY = [
    "mix",
    "mixture",
    "mixdown",
    "full_mix",
    "master",
    "song",
]

PERTURBATION_ORDER = [
    "remove_entire_track",
    "remove_section",
    "remove_automation",
    "swap_instrument_timbre",
    "remove_fx_processing",
    "quantize_midi_too_hard",
    "remove_harmony_parts",
    "remove_full_song_section",
]


@dataclass
class StemRecord:
    name: str
    path: str
    extension: str


@dataclass
class SongRecord:
    source: str
    kind: str
    split: str
    song_id: str
    song_dir: str
    mix_path: Optional[str]
    stems: List[StemRecord]
    midi_paths: List[str]
    perturbations: List[str]


def _is_audio(path: pl.Path) -> bool:
    return path.is_file() and path.suffix.lower() in AUDIO_EXTS


def _is_midi(path: pl.Path) -> bool:
    return path.is_file() and path.suffix.lower() in MIDI_EXTS


def _iter_song_dirs(root: pl.Path, recursive: bool) -> Iterable[pl.Path]:
    if not root.exists():
        return []

    if not recursive:
        return [p for p in root.iterdir() if p.is_dir()]

    out: List[pl.Path] = []
    for dirpath, _, filenames in os.walk(root):
        if not filenames:
            continue
        has_supported = any(pl.Path(name).suffix.lower() in (AUDIO_EXTS | MIDI_EXTS) for name in filenames)
        if has_supported:
            out.append(pl.Path(dirpath))
    return out


def _choose_mix_path(audio_files: List[pl.Path]) -> Optional[pl.Path]:
    if not audio_files:
        return None

    by_name = {p.stem.lower(): p for p in audio_files}
    for base in MIX_BASENAME_PRIORITY:
        if base in by_name:
            return by_name[base]

    contains_mix = [p for p in audio_files if "mix" in p.stem.lower()]
    if contains_mix:
        contains_mix.sort(key=lambda p: p.name)
        return contains_mix[0]

    return None


def infer_perturbations(stem_count: int, midi_count: int, has_mix: bool) -> List[str]:
    chosen: List[str] = []

    def add(name: str) -> None:
        if name not in chosen:
            chosen.append(name)

    if has_mix or stem_count > 0:
        add("remove_section")
        add("remove_full_song_section")

    if stem_count >= 2:
        add("remove_entire_track")
        add("remove_automation")
        add("swap_instrument_timbre")
        add("remove_fx_processing")
        add("remove_harmony_parts")

    if midi_count > 0:
        add("quantize_midi_too_hard")
        add("swap_instrument_timbre")
        add("remove_harmony_parts")

    return [name for name in PERTURBATION_ORDER if name in chosen]


def scan_generic_multitrack(
    source_name: str,
    root_dir: str,
    split: str = "unspecified",
    recursive: bool = True,
    min_stems: int = 1,
) -> List[SongRecord]:
    root = pl.Path(root_dir)
    rows: List[SongRecord] = []

    for song_dir in sorted(_iter_song_dirs(root, recursive=recursive)):
        files = [p for p in song_dir.iterdir() if p.is_file()]
        audio_files = sorted([p for p in files if _is_audio(p)], key=lambda p: p.name)
        midi_files = sorted([p for p in files if _is_midi(p)], key=lambda p: p.name)

        if not audio_files and not midi_files:
            continue

        mix_path = _choose_mix_path(audio_files)
        stem_files = [p for p in audio_files if mix_path is None or p != mix_path]

        if len(stem_files) < min_stems and not midi_files:
            continue

        song_id = str(song_dir.relative_to(root)) if song_dir != root else song_dir.name
        perturbations = infer_perturbations(
            stem_count=len(stem_files),
            midi_count=len(midi_files),
            has_mix=mix_path is not None,
        )

        rows.append(
            SongRecord(
                source=source_name,
                kind="multitrack",
                split=split,
                song_id=song_id,
                song_dir=str(song_dir.resolve()),
                mix_path=str(mix_path.resolve()) if mix_path else None,
                stems=[
                    StemRecord(name=path.stem, path=str(path.resolve()), extension=path.suffix.lower())
                    for path in stem_files
                ],
                midi_paths=[str(path.resolve()) for path in midi_files],
                perturbations=perturbations,
            )
        )

    return rows


def scan_midi_only(
    source_name: str,
    root_dir: str,
    split: str = "unspecified",
    recursive: bool = True,
) -> List[SongRecord]:
    root = pl.Path(root_dir)
    pattern = "**/*" if recursive else "*"
    rows: List[SongRecord] = []

    for midi_path in sorted(root.glob(pattern)):
        if not _is_midi(midi_path):
            continue

        song_id = str(midi_path.relative_to(root))
        rows.append(
            SongRecord(
                source=source_name,
                kind="midi_only",
                split=split,
                song_id=song_id,
                song_dir=str(midi_path.parent.resolve()),
                mix_path=None,
                stems=[],
                midi_paths=[str(midi_path.resolve())],
                perturbations=infer_perturbations(stem_count=0, midi_count=1, has_mix=False),
            )
        )

    return rows


def build_manifest_from_sources(
    source_specs: List[Dict[str, Any]],
    strict_missing_roots: bool = False,
) -> List[SongRecord]:
    rows: List[SongRecord] = []

    for spec in source_specs:
        name = str(spec["name"])
        kind = str(spec["kind"])
        root = pl.Path(str(spec["root"])).expanduser()
        split = str(spec.get("split", "unspecified"))
        recursive = bool(spec.get("recursive", True))

        if not root.exists():
            msg = f"Source root not found for '{name}': {root}"
            if strict_missing_roots:
                raise FileNotFoundError(msg)
            print(f"WARNING: {msg} (skipping)")
            continue

        if kind == "multitrack":
            min_stems = int(spec.get("min_stems", 1))
            scanned = scan_generic_multitrack(
                source_name=name,
                root_dir=str(root),
                split=split,
                recursive=recursive,
                min_stems=min_stems,
            )
        elif kind == "midi_only":
            scanned = scan_midi_only(
                source_name=name,
                root_dir=str(root),
                split=split,
                recursive=recursive,
            )
        else:
            raise ValueError(f"Unsupported source kind for '{name}': {kind}")

        rows.extend(scanned)

    return rows


def serialize_records(records: Iterable[SongRecord]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for rec in records:
        row = asdict(rec)
        row["stem_count"] = len(rec.stems)
        row["midi_count"] = len(rec.midi_paths)
        out.append(row)
    return out


def build_summary(records: List[SongRecord]) -> Dict[str, Any]:
    by_source: Dict[str, Dict[str, int]] = {}
    by_perturbation: Dict[str, int] = {}

    for rec in records:
        bucket = by_source.setdefault(
            rec.source,
            {"songs": 0, "stems": 0, "midi_files": 0},
        )
        bucket["songs"] += 1
        bucket["stems"] += len(rec.stems)
        bucket["midi_files"] += len(rec.midi_paths)

        for perturb in rec.perturbations:
            by_perturbation[perturb] = by_perturbation.get(perturb, 0) + 1

    return {
        "total_songs": len(records),
        "by_source": by_source,
        "by_perturbation": dict(sorted(by_perturbation.items())),
    }


def write_manifest(
    records: List[SongRecord],
    manifest_path: str,
    summary_path: Optional[str] = None,
) -> Dict[str, Any]:
    manifest = pl.Path(manifest_path)
    manifest.parent.mkdir(parents=True, exist_ok=True)

    serialized = serialize_records(records)
    with manifest.open("w", encoding="utf-8") as f:
        for row in serialized:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")

    summary = build_summary(records)
    summary["manifest_path"] = str(manifest.resolve())

    if summary_path is not None:
        summary_file = pl.Path(summary_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        with summary_file.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        summary["summary_path"] = str(summary_file.resolve())

    return summary
