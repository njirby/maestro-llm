"""
Prepare canonical 10-second MusicCaps clips for stable training.

This module trims each local source audio file to the row's [start_s, end_s]
segment, resamples to 48 kHz mono, validates duration, and writes a manifest.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pathlib as pl
import subprocess
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

from maestro.audio.processing import CLAP_SAMPLE_RATE
from maestro.data.musiccaps import MUSICCAPS_HF_ID

DURATION_TOLERANCE_S = 0.2
DEFAULT_PREPARED_DIR = "./data/prepared/musiccaps_v1"


@dataclass
class PrepareResult:
    row_id: int
    ytid: str
    caption: str
    start_s: float
    end_s: float
    clip_path: str
    duration_s: float
    sample_rate: int
    channels: int
    status: str
    error: Optional[str] = None


@dataclass
class PrepareSummary:
    total_rows: int
    ok_rows: int
    failed_rows: int
    failed_ratio: float
    manifest_path: str
    report_path: str


def _default_workers() -> int:
    return max(1, (os.cpu_count() or 2) - 1)


def _format_seconds(seconds: float) -> str:
    return f"{seconds:.3f}"


def _validate_prepared_clip(path: pl.Path) -> Tuple[bool, float, int, int, Optional[str]]:
    if not path.exists():
        return False, 0.0, 0, 0, "missing_output"

    try:
        info = sf.info(str(path))
    except Exception as exc:  # pragma: no cover - defensive
        return False, 0.0, 0, 0, f"soundfile_error:{exc}"

    duration_s = float(info.frames) / float(info.samplerate)
    if info.samplerate != CLAP_SAMPLE_RATE:
        return False, duration_s, int(info.samplerate), int(info.channels), "wrong_sample_rate"
    if info.channels != 1:
        return False, duration_s, int(info.samplerate), int(info.channels), "wrong_channels"

    expected = 10.0
    if abs(duration_s - expected) > DURATION_TOLERANCE_S:
        return False, duration_s, int(info.samplerate), int(info.channels), "wrong_duration"

    return True, duration_s, int(info.samplerate), int(info.channels), None


def _trim_one(task: Dict[str, Any]) -> PrepareResult:
    row_id = int(task["row_id"])
    ytid = str(task["ytid"])
    caption = str(task["caption"])
    start_s = float(task["start_s"])
    end_s = float(task["end_s"])
    src_path = pl.Path(task["src_path"])
    out_path = pl.Path(task["out_path"])
    out_rel_path = str(task["out_rel_path"])
    overwrite = bool(task["overwrite"])

    if not src_path.exists():
        return PrepareResult(
            row_id=row_id,
            ytid=ytid,
            caption=caption,
            start_s=start_s,
            end_s=end_s,
            clip_path=out_rel_path,
            duration_s=0.0,
            sample_rate=0,
            channels=0,
            status="failed",
            error="missing_source",
        )

    if out_path.exists() and not overwrite:
        ok, dur, sr, ch, err = _validate_prepared_clip(out_path)
        return PrepareResult(
            row_id=row_id,
            ytid=ytid,
            caption=caption,
            start_s=start_s,
            end_s=end_s,
            clip_path=out_rel_path,
            duration_s=dur,
            sample_rate=sr,
            channels=ch,
            status="ok" if ok else "failed",
            error=err,
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-y",
        "-ss",
        _format_seconds(start_s),
        "-to",
        _format_seconds(end_s),
        "-i",
        str(src_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        str(CLAP_SAMPLE_RATE),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]

    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return PrepareResult(
            row_id=row_id,
            ytid=ytid,
            caption=caption,
            start_s=start_s,
            end_s=end_s,
            clip_path=out_rel_path,
            duration_s=0.0,
            sample_rate=0,
            channels=0,
            status="failed",
            error="ffmpeg_timeout",
        )
    except Exception as exc:  # pragma: no cover - defensive
        return PrepareResult(
            row_id=row_id,
            ytid=ytid,
            caption=caption,
            start_s=start_s,
            end_s=end_s,
            clip_path=out_rel_path,
            duration_s=0.0,
            sample_rate=0,
            channels=0,
            status="failed",
            error=f"ffmpeg_error:{exc}",
        )

    if proc.returncode != 0:
        return PrepareResult(
            row_id=row_id,
            ytid=ytid,
            caption=caption,
            start_s=start_s,
            end_s=end_s,
            clip_path=out_rel_path,
            duration_s=0.0,
            sample_rate=0,
            channels=0,
            status="failed",
            error="ffmpeg_nonzero",
        )

    ok, dur, sr, ch, err = _validate_prepared_clip(out_path)
    return PrepareResult(
        row_id=row_id,
        ytid=ytid,
        caption=caption,
        start_s=start_s,
        end_s=end_s,
        clip_path=out_rel_path,
        duration_s=dur,
        sample_rate=sr,
        channels=ch,
        status="ok" if ok else "failed",
        error=err,
    )


def _to_record(result: PrepareResult) -> Dict[str, Any]:
    return {
        "row_id": result.row_id,
        "ytid": result.ytid,
        "caption": result.caption,
        "start_s": result.start_s,
        "end_s": result.end_s,
        "clip_path": result.clip_path,
        "duration_s": result.duration_s,
        "sample_rate": result.sample_rate,
        "channels": result.channels,
        "status": result.status,
        "error": result.error,
    }


def _write_jsonl(path: pl.Path, records: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=True) + "\n")


def prepare_musiccaps(
    audio_dir: str,
    out_dir: str = DEFAULT_PREPARED_DIR,
    num_workers: Optional[int] = None,
    limit: Optional[int] = None,
    overwrite: bool = False,
    failure_ratio_threshold: float = 0.01,
) -> PrepareSummary:
    """
    Prepare canonical 10-second clips and manifest for training.

    Raises RuntimeError when failures exceed failure_ratio_threshold.
    """
    out_root = pl.Path(out_dir)
    clips_dir = out_root / "clips"
    manifest_path = out_root / "manifest.jsonl"
    report_path = out_root / "prepare_report.json"
    clips_dir.mkdir(parents=True, exist_ok=True)

    print("Loading MusicCaps metadata from HuggingFace...")
    raw_rows = list(load_dataset(MUSICCAPS_HF_ID, split="train"))
    src_root = pl.Path(audio_dir)
    rows = [ex for ex in raw_rows if (src_root / f"{ex['ytid']}.mp3").exists()]
    print(f"Found {len(rows)} local source files (out of {len(raw_rows)} metadata rows)")
    if limit is not None:
        rows = rows[:limit]

    workers = _default_workers() if num_workers is None else max(1, int(num_workers))
    print(f"Preparing {len(rows)} rows using {workers} worker(s)...")

    tasks: List[Dict[str, Any]] = []
    for idx, ex in enumerate(rows):
        ytid = str(ex["ytid"])
        clip_name = f"{idx:05d}_{ytid}.wav"
        out_path = clips_dir / clip_name
        tasks.append(
            {
                "row_id": idx,
                "ytid": ytid,
                "caption": ex["caption"],
                "start_s": float(ex["start_s"]),
                "end_s": float(ex["end_s"]),
                "src_path": str(src_root / f"{ytid}.mp3"),
                "out_path": str(out_path),
                "out_rel_path": str(pl.Path("clips") / clip_name),
                "overwrite": overwrite,
            }
        )

    records: List[Dict[str, Any]] = []
    with multiprocessing.Pool(processes=workers) as pool:
        for result in tqdm(pool.imap_unordered(_trim_one, tasks), total=len(tasks), desc="Preparing"):
            records.append(_to_record(result))

    records.sort(key=lambda r: int(r["row_id"]))
    ok_rows = [r for r in records if r["status"] == "ok"]
    failed_rows = [r for r in records if r["status"] != "ok"]

    _write_jsonl(manifest_path, records)

    summary = {
        "total_rows": len(records),
        "ok_rows": len(ok_rows),
        "failed_rows": len(failed_rows),
        "failed_ratio": (len(failed_rows) / len(records)) if records else 0.0,
        "failure_ratio_threshold": failure_ratio_threshold,
        "workers": workers,
        "audio_dir": str(pl.Path(audio_dir).resolve()),
        "out_dir": str(out_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "failures": [
            {
                "row_id": r["row_id"],
                "ytid": r["ytid"],
                "error": r["error"],
            }
            for r in failed_rows[:200]
        ],
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Prepared rows: total={summary['total_rows']}, ok={summary['ok_rows']}, "
        f"failed={summary['failed_rows']} ({summary['failed_ratio']:.2%})"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Report:   {report_path}")

    if summary["failed_ratio"] > failure_ratio_threshold:
        raise RuntimeError(
            "Prepared dataset failed quality threshold: "
            f"failed_ratio={summary['failed_ratio']:.2%} > "
            f"threshold={failure_ratio_threshold:.2%}."
        )

    return PrepareSummary(
        total_rows=summary["total_rows"],
        ok_rows=summary["ok_rows"],
        failed_rows=summary["failed_rows"],
        failed_ratio=float(summary["failed_ratio"]),
        manifest_path=str(manifest_path),
        report_path=str(report_path),
    )


def validate_prepared_manifest(
    out_dir: str = DEFAULT_PREPARED_DIR,
    failure_ratio_threshold: float = 0.01,
) -> PrepareSummary:
    """Validate existing manifest and clip files without re-trimming."""
    out_root = pl.Path(out_dir)
    manifest_path = out_root / "manifest.jsonl"
    report_path = out_root / "prepare_report.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    records: List[Dict[str, Any]] = []
    with manifest_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    validated: List[Dict[str, Any]] = []
    for rec in tqdm(records, total=len(records), desc="Validating"):
        clip_path = out_root / rec["clip_path"]
        ok, dur, sr, ch, err = _validate_prepared_clip(clip_path)
        rec = dict(rec)
        rec["duration_s"] = dur
        rec["sample_rate"] = sr
        rec["channels"] = ch
        rec["status"] = "ok" if ok else "failed"
        rec["error"] = err
        validated.append(rec)

    ok_rows = [r for r in validated if r["status"] == "ok"]
    failed_rows = [r for r in validated if r["status"] != "ok"]
    _write_jsonl(manifest_path, validated)

    summary = {
        "total_rows": len(validated),
        "ok_rows": len(ok_rows),
        "failed_rows": len(failed_rows),
        "failed_ratio": (len(failed_rows) / len(validated)) if validated else 0.0,
        "failure_ratio_threshold": failure_ratio_threshold,
        "audio_dir": None,
        "out_dir": str(out_root.resolve()),
        "manifest_path": str(manifest_path.resolve()),
        "failures": [
            {
                "row_id": r.get("row_id"),
                "ytid": r.get("ytid"),
                "error": r.get("error"),
            }
            for r in failed_rows[:200]
        ],
    }

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if summary["failed_ratio"] > failure_ratio_threshold:
        raise RuntimeError(
            "Prepared dataset failed quality threshold: "
            f"failed_ratio={summary['failed_ratio']:.2%} > "
            f"threshold={failure_ratio_threshold:.2%}."
        )

    return PrepareSummary(
        total_rows=summary["total_rows"],
        ok_rows=summary["ok_rows"],
        failed_rows=summary["failed_rows"],
        failed_ratio=float(summary["failed_ratio"]),
        manifest_path=str(manifest_path),
        report_path=str(report_path),
    )
