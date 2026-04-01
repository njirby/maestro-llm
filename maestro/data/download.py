"""
Download MusicCaps audio clips from YouTube.

Uses yt-dlp + ffmpeg to fetch source audio files and validates decodeability.
Canonical fixed 10s training clips are produced separately by prepare_musiccaps.py.
"""

import multiprocessing
import os
import pathlib as pl
import subprocess
from typing import Optional

from datasets import load_dataset
from tqdm import tqdm

MUSICCAPS_HF_ID = "google/MusicCaps"


def _format_seconds(seconds: float) -> str:
    hours, remainder = divmod(int(seconds), 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def validate_mp3(path: pl.Path) -> bool:
    """Return True if path exists and ffmpeg can decode it without errors."""
    if not path.exists():
        return False
    result = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "null", "-"],
        capture_output=True,
    )
    return result.returncode == 0


def download_clip(ytid: str, output_path: pl.Path, start_sec: float, end_sec: float) -> bool:
    """
    Download a single audio clip from YouTube and trim to [start_sec, end_sec].

    Returns True on success, False on failure. Skips if already downloaded and valid.
    """
    mp3_path = pl.Path(str(output_path) + ".mp3")

    if validate_mp3(mp3_path):
        return True

    if mp3_path.exists():
        print(f"Removing corrupt file: {mp3_path}")
        os.remove(mp3_path)

    cmd = [
        "yt-dlp",
        "-x",
        "--audio-format", "mp3",
        "--audio-quality", "0",
        "--postprocessor-args", f"-ss {_format_seconds(start_sec)} -to {_format_seconds(end_sec)}",
        "-o", str(mp3_path),
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        f"https://www.youtube.com/watch?v={ytid}",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, timeout=120)
        if result.returncode == 0 and validate_mp3(mp3_path):
            return True
    except subprocess.TimeoutExpired:
        print(f"Timeout downloading {ytid}")
    except Exception as e:
        print(f"Error downloading {ytid}: {e}")

    if mp3_path.exists():
        os.remove(mp3_path)
    return False


def _download_worker(args):
    ytid, output_path, start_sec, end_sec = args
    return ytid, download_clip(ytid, output_path, start_sec, end_sec)


def download_musiccaps(
    out_dir: str,
    num_workers: int = 4,
    limit: Optional[int] = None,
):
    """
    Download all (or a subset of) MusicCaps clips to out_dir.

    Args:
        out_dir: Directory to save .mp3 files.
        num_workers: Parallel download workers.
        limit: Cap the number of clips (useful for quick tests).
    """
    out_path = pl.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading MusicCaps metadata from HuggingFace...")
    dataset = load_dataset(MUSICCAPS_HF_ID, split="train")

    examples = list(dataset)
    if limit is not None:
        examples = examples[:limit]

    print(f"Downloading {len(examples)} clips with {num_workers} workers...")

    tasks = [
        (ex["ytid"], out_path / ex["ytid"], ex["start_s"], ex["end_s"])
        for ex in examples
    ]

    success, failed = 0, 0
    with multiprocessing.Pool(processes=num_workers) as pool:
        for _, ok in tqdm(pool.imap_unordered(_download_worker, tasks), total=len(tasks), desc="Downloading"):
            if ok:
                success += 1
            else:
                failed += 1

    print(f"\nDone. Success: {success}, Failed: {failed}")
    print(f"Audio saved to: {out_path.resolve()}")
