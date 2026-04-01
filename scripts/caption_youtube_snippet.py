#!/usr/bin/env python3
"""
Download a YouTube audio snippet and caption it with the autoregressive audio model.

Example:
  python scripts/caption_youtube_snippet.py \
    --url "https://www.youtube.com/watch?v=dQw4w9WgXcQ" \
    --run-dir outputs/audio_caption_qwen4b/runs/Mar05_06-47-08_ml-workstation \
    --device cuda --clap-device cuda
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path


AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".webm", ".opus", ".flac", ".ogg"}


def format_seconds(seconds: float) -> str:
    total = int(seconds)
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02}:{minutes:02}:{secs:02}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Caption a 10-second YouTube snippet with the audio model.")
    parser.add_argument("--url", type=str, required=True, help="YouTube URL")
    parser.add_argument("--start-sec", type=float, default=0.0, help="Snippet start time in seconds")
    parser.add_argument("--duration-sec", type=float, default=10.0, help="Snippet duration in seconds")
    parser.add_argument(
        "--snippet-dir",
        type=str,
        default=None,
        help="Directory for downloaded snippets (default: system temp dir)",
    )
    parser.add_argument(
        "--delete-snippet",
        action="store_true",
        help="Delete downloaded snippet after captioning (default: keep it in temp storage)",
    )

    parser.add_argument("--config", type=str, default="configs/audio_caption_qwen4b_stable.yaml")
    parser.add_argument("--run-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default=None)
    parser.add_argument("--projection-path", type=str, default=None)

    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--dynamic-chunking", action="store_true")
    parser.add_argument("--stream", action="store_true", help="Stream generated tokens as they are produced")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--clap-device", type=str, choices=["cpu", "cuda"], default=None)

    parser.add_argument("--output-json", type=str, default=None, help="Optional JSON output path for prediction row")
    return parser.parse_args()


def require_tool(name: str):
    if shutil.which(name) is None:
        raise RuntimeError(f"Required tool not found on PATH: {name}")


def find_downloaded_audio(snippet_dir: Path, prefix: str) -> Path:
    candidates = sorted(
        [
            p
            for p in snippet_dir.glob(f"{prefix}.*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            f"yt-dlp completed but no audio snippet found for prefix={prefix} in {snippet_dir}"
        )
    return candidates[0]


def download_snippet(url: str, start_sec: float, duration_sec: float, snippet_dir: Path) -> Path:
    require_tool("yt-dlp")
    require_tool("ffmpeg")

    if start_sec < 0:
        raise ValueError("--start-sec must be >= 0")
    if duration_sec <= 0:
        raise ValueError("--duration-sec must be > 0")

    end_sec = start_sec + duration_sec
    snippet_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"yt_snippet_{ts}_{int(start_sec)}_{int(end_sec)}"
    out_template = str(snippet_dir / f"{prefix}.%(ext)s")

    section = f"*{format_seconds(start_sec)}-{format_seconds(end_sec)}"
    cmd = [
        "yt-dlp",
        "--download-sections",
        section,
        "-x",
        "--audio-format",
        "wav",
        "--audio-quality",
        "0",
        "-o",
        out_template,
        "--no-playlist",
        "--quiet",
        "--no-warnings",
        url,
    ]
    print("Downloading snippet with yt-dlp...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        stderr = result.stderr.strip()
        stdout = result.stdout.strip()
        detail = stderr or stdout or "yt-dlp failed without error text"
        raise RuntimeError(f"yt-dlp failed: {detail}")

    snippet_path = find_downloaded_audio(snippet_dir, prefix)
    print(f"Snippet saved: {snippet_path}")
    return snippet_path


def run_caption(args: argparse.Namespace, snippet_path: Path):
    cmd = [
        sys.executable,
        "scripts/predict_audio_caption_autoreg.py",
        "--config",
        args.config,
        "--audio-path",
        str(snippet_path),
        "--batch-size",
        "1",
        "--max-new-tokens",
        str(args.max_new_tokens),
        "--device",
        args.device,
    ]

    if args.run_dir is not None:
        cmd.extend(["--run-dir", args.run_dir])
    if args.output_dir is not None:
        cmd.extend(["--output-dir", args.output_dir])
    if args.checkpoint_step is not None:
        cmd.extend(["--checkpoint-step", str(args.checkpoint_step)])
    if args.checkpoint_dir is not None:
        cmd.extend(["--checkpoint-dir", args.checkpoint_dir])
    if args.projection_path is not None:
        cmd.extend(["--projection-path", args.projection_path])

    if args.clap_device is not None:
        cmd.extend(["--clap-device", args.clap_device])
    if args.dynamic_chunking:
        cmd.append("--dynamic-chunking")
    if args.stream:
        cmd.append("--stream")
    if args.output_json is not None:
        cmd.extend(["--output-json", args.output_json])

    print("Running autoregressive captioning...")
    subprocess.run(cmd, check=True)


def main() -> int:
    args = parse_args()

    if args.snippet_dir is not None:
        snippet_dir = Path(args.snippet_dir)
    else:
        snippet_dir = Path(tempfile.gettempdir()) / "maestro_youtube_snippets"
    snippet_path = download_snippet(
        url=args.url,
        start_sec=args.start_sec,
        duration_sec=args.duration_sec,
        snippet_dir=snippet_dir,
    )
    print(f"Temp audio file: {snippet_path.resolve()}")

    try:
        run_caption(args, snippet_path)
    finally:
        if args.delete_snippet and snippet_path.exists():
            snippet_path.unlink()
            print(f"Removed temporary snippet: {snippet_path}")
        elif snippet_path.exists():
            print(f"Kept temporary snippet: {snippet_path.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
