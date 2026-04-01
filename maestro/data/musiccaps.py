"""
MusicCaps dataset builder.

Loads metadata from HuggingFace and filters to clips that have been
downloaded to disk. Returns a HuggingFace Dataset of {audio_path, caption, ytid}.
Audio is loaded lazily inside the DataCollator at training time.
"""

import json
import pathlib as pl
from typing import Dict, List, Optional

from datasets import Dataset, load_dataset

MUSICCAPS_HF_ID = "google/MusicCaps"
PREPARED_MANIFEST_FIELDS = {
    "row_id",
    "ytid",
    "caption",
    "start_s",
    "end_s",
    "clip_path",
    "duration_s",
    "sample_rate",
    "channels",
    "status",
    "error",
}


def build_dataset(
    audio_dir: str,
    split: str = "train",
    limit: Optional[int] = None,
) -> Dataset:
    """
    Load MusicCaps metadata and filter to clips present on disk.

    Args:
        audio_dir: Directory containing {ytid}.mp3 files.
        split: HuggingFace dataset split (only "train" exists for MusicCaps).
        limit: If set, cap the dataset at this many examples (useful for smoke tests).

    Returns:
        HuggingFace Dataset with columns: audio_path (str), caption (str), ytid (str).
    """
    audio_path = pl.Path(audio_dir)

    print(f"Loading MusicCaps metadata from HuggingFace...")
    raw = load_dataset(MUSICCAPS_HF_ID, split=split)

    records = []
    for ex in raw:
        mp3_file = audio_path / f"{ex['ytid']}.mp3"
        if mp3_file.exists():
            records.append({
                "audio_path": str(mp3_file),
                "caption": ex["caption"],
                "ytid": ex["ytid"],
            })

    if limit is not None:
        records = records[:limit]

    print(f"Found {len(records)} downloaded clips (out of {len(raw)} total)")
    return Dataset.from_list(records)


def build_prepared_dataset(
    prepared_dir: str,
    limit: Optional[int] = None,
) -> Dataset:
    """
    Load a preprocessed manifest and return rows ready for training.

    Args:
        prepared_dir: Directory containing manifest.jsonl and clips/.
        limit: Optional cap for smoke tests.

    Returns:
        HuggingFace Dataset with columns: audio_path (str), caption (str), ytid (str).
    """
    root = pl.Path(prepared_dir)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Prepared manifest not found: {manifest_path}. "
            "Run `python scripts/prepare_musiccaps.py` first."
        )

    records: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            missing = PREPARED_MANIFEST_FIELDS - set(rec.keys())
            if missing:
                raise ValueError(
                    f"Invalid prepared manifest row {lineno}: missing fields {sorted(missing)}"
                )
            if rec["status"] != "ok":
                continue

            clip_path = root / rec["clip_path"]
            if not clip_path.exists():
                continue

            records.append(
                {
                    "audio_path": str(clip_path),
                    "caption": str(rec["caption"]),
                    "ytid": str(rec["ytid"]),
                }
            )
            if limit is not None and len(records) >= limit:
                break

    if not records:
        raise ValueError(
            f"No valid prepared rows found in {manifest_path}. "
            "Check prepare_report.json for failures."
        )

    print(f"Loaded {len(records)} prepared clips from {manifest_path}")
    return Dataset.from_list(records)
