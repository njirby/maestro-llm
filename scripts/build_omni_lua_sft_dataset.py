#!/usr/bin/env python3
"""
Build an MS-Swift multimodal SFT dataset from tuple MP3 + Lua pairs.

Each output row has:
  - user message: "<audio>\\n<PROMPT>"
  - assistant message: full Lua text
  - audios: [relative/path/to/file.mp3]

Usage:
    python scripts/build_omni_lua_sft_dataset.py \
      --audio-dir data/processed/reaper_tuples_lakh/wavs \
      --lua-dir data/processed/reaper_tuples_lakh/luas \
      --prompt-file data/prompts/omni_lua_user_prompt.txt \
      --out-dir data/prepared/omni_lua_sft \
      --val-count 256
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
from pathlib import Path


REQUIRED_LUA_MARKERS = (
    "MIDI_InsertNote",
    "TrackFX_AddByName(track,\"Vital\"",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--audio-dir",
        type=Path,
        default=Path("data/processed/reaper_tuples_lakh/wavs"),
        help="Directory containing tuple audio files (default: data/processed/reaper_tuples_lakh/wavs).",
    )
    parser.add_argument(
        "--lua-dir",
        type=Path,
        default=Path("data/processed/reaper_tuples_lakh/luas"),
        help="Directory containing tuple lua files (default: data/processed/reaper_tuples_lakh/luas).",
    )
    parser.add_argument(
        "--prompt-file",
        type=Path,
        required=True,
        help="Path to text file containing the exact user prompt to use for every sample.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data/prepared/omni_lua_sft"),
        help="Output directory for all.jsonl/train.jsonl/val.jsonl/stats.json.",
    )
    parser.add_argument(
        "--audio-exts",
        default=".mp3",
        help='Comma-separated audio extensions to include (default: ".mp3").',
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio in [0,1] (default: 0.1).",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=None,
        help="If set, use exactly this many validation rows (deterministic), overriding --val-ratio.",
    )
    parser.add_argument(
        "--val-max-audio-seconds",
        type=float,
        default=None,
        help="If set, only select validation rows whose audio duration is <= this value.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic seed used for stable split and optional sampling.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="If set, randomly subsample at most this many basename pairs before writing output.",
    )
    parser.add_argument(
        "--require-lua-markers",
        action="store_true",
        help="Require key Lua markers (MIDI_InsertNote, Vital plugin load).",
    )
    parser.add_argument(
        "--sample-preview-count",
        type=int,
        default=5,
        help="How many sample IDs to include in stats preview (default: 5).",
    )
    return parser.parse_args()


def _normalize_audio_exts(ext_csv: str) -> set[str]:
    exts = set()
    for raw in ext_csv.split(","):
        e = raw.strip().lower()
        if not e:
            continue
        if not e.startswith("."):
            e = f".{e}"
        exts.add(e)
    if not exts:
        raise ValueError("No valid audio extensions were provided.")
    return exts


def _collect_by_stem(root: Path, exts: set[str]) -> dict[str, Path]:
    by_stem: dict[str, Path] = {}
    if not root.exists():
        return by_stem
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in exts:
            continue
        by_stem[path.stem] = path
    return by_stem


def _stable_split_key(sample_id: str, seed: int) -> float:
    digest = hashlib.md5(f"{sample_id}|{seed}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64 - 1)


def _load_prompt(prompt_file: Path) -> str:
    prompt = prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {prompt_file}")
    return prompt


def _validate_lua_text(lua_text: str, require_markers: bool) -> tuple[bool, str | None]:
    if not lua_text.strip():
        return False, "empty_lua"
    if require_markers:
        for marker in REQUIRED_LUA_MARKERS:
            if marker not in lua_text:
                return False, f"missing_marker:{marker}"
    return True, None


def _build_record(sample_id: str, prompt: str, lua_text: str, audio_path: Path, out_dir: Path) -> dict:
    resolved_audio = str(audio_path.resolve())
    return {
        "id": sample_id,
        "messages": [
            {"role": "user", "content": f"<audio>\n{prompt}"},
            {"role": "assistant", "content": lua_text},
        ],
        "audios": [resolved_audio],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _audio_duration_seconds(path: str) -> float | None:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    try:
        out = subprocess.check_output(cmd, text=True).strip()
    except Exception:
        return None
    if not out:
        return None
    try:
        return float(out)
    except ValueError:
        return None


def main() -> None:
    args = parse_args()

    if not (0.0 <= args.val_ratio <= 1.0):
        raise ValueError(f"--val-ratio must be in [0,1], got {args.val_ratio}")
    if args.val_count is not None and args.val_count < 0:
        raise ValueError(f"--val-count must be >= 0, got {args.val_count}")
    if args.val_max_audio_seconds is not None and args.val_max_audio_seconds <= 0:
        raise ValueError(
            f"--val-max-audio-seconds must be > 0 when set, got {args.val_max_audio_seconds}"
        )

    audio_exts = _normalize_audio_exts(args.audio_exts)
    prompt = _load_prompt(args.prompt_file)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    audio_by_stem = _collect_by_stem(args.audio_dir, audio_exts)
    lua_by_stem = _collect_by_stem(args.lua_dir, {".lua"})

    audio_stems = set(audio_by_stem.keys())
    lua_stems = set(lua_by_stem.keys())
    paired_stems = sorted(audio_stems & lua_stems)

    if args.max_samples is not None and args.max_samples > 0 and len(paired_stems) > args.max_samples:
        rng = random.Random(args.seed)
        paired_stems = sorted(rng.sample(paired_stems, args.max_samples))

    dropped: dict[str, int] = {
        "missing_audio": len(lua_stems - audio_stems),
        "missing_lua": len(audio_stems - lua_stems),
        "invalid_audio": 0,
        "read_error": 0,
        "invalid_lua": 0,
    }

    all_rows: list[dict] = []
    for sample_id in paired_stems:
        lua_path = lua_by_stem[sample_id]
        audio_path = audio_by_stem[sample_id]

        if not audio_path.exists() or audio_path.stat().st_size <= 0:
            dropped["invalid_audio"] += 1
            continue

        try:
            lua_text = lua_path.read_text(encoding="utf-8")
        except Exception:
            dropped["read_error"] += 1
            continue

        ok, reason = _validate_lua_text(lua_text, args.require_lua_markers)
        if not ok:
            dropped["invalid_lua"] += 1
            continue

        all_rows.append(_build_record(sample_id, prompt, lua_text, audio_path, args.out_dir))

    train_rows: list[dict] = []
    val_rows: list[dict] = []
    val_filtered_long_audio = 0
    val_filtered_unknown_audio = 0
    if args.val_count is not None:
        target_val = min(args.val_count, len(all_rows))
        ordered = sorted(
            all_rows,
            key=lambda row: (_stable_split_key(row["id"], args.seed), row["id"]),
        )
        val_ids: set[str] = set()
        for row in ordered:
            if len(val_ids) >= target_val:
                break
            if args.val_max_audio_seconds is not None:
                dur_s = _audio_duration_seconds(row["audios"][0])
                if dur_s is None:
                    val_filtered_unknown_audio += 1
                    continue
                if dur_s > args.val_max_audio_seconds:
                    val_filtered_long_audio += 1
                    continue
            val_ids.add(row["id"])
        for row in all_rows:
            if row["id"] in val_ids:
                val_rows.append(row)
            else:
                train_rows.append(row)
    else:
        for row in all_rows:
            k = _stable_split_key(row["id"], args.seed)
            if k < args.val_ratio:
                val_rows.append(row)
            else:
                train_rows.append(row)

        if all_rows and not train_rows:
            train_rows.append(val_rows.pop())
        if all_rows and args.val_ratio > 0.0 and not val_rows:
            val_rows.append(train_rows.pop())

    all_path = args.out_dir / "all.jsonl"
    train_path = args.out_dir / "train.jsonl"
    val_path = args.out_dir / "val.jsonl"
    stats_path = args.out_dir / "stats.json"

    _write_jsonl(all_path, all_rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(val_path, val_rows)

    preview_ids = [row["id"] for row in all_rows[: max(0, args.sample_preview_count)]]
    stats = {
        "audio_dir": str(args.audio_dir),
        "lua_dir": str(args.lua_dir),
        "prompt_file": str(args.prompt_file),
        "out_dir": str(args.out_dir),
        "audio_exts": sorted(audio_exts),
        "seed": args.seed,
        "val_ratio": args.val_ratio,
        "val_count": args.val_count,
        "val_max_audio_seconds": args.val_max_audio_seconds,
        "split_mode": ("count" if args.val_count is not None else "ratio"),
        "paired_candidates": len(paired_stems),
        "written": {
            "all": len(all_rows),
            "train": len(train_rows),
            "val": len(val_rows),
        },
        "dropped": dropped,
        "val_filter": {
            "long_audio_rejected": val_filtered_long_audio,
            "unknown_audio_duration_rejected": val_filtered_unknown_audio,
        },
        "preview_ids": preview_ids,
    }
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"Wrote: {all_path}")
    print(f"Wrote: {train_path}")
    print(f"Wrote: {val_path}")
    print(f"Wrote: {stats_path}")
    if args.val_count is not None:
        split_desc = f"val_count={args.val_count}"
    else:
        split_desc = f"val_ratio={args.val_ratio:.3f}"
    print(f"Rows: all={len(all_rows):,}, train={len(train_rows):,}, val={len(val_rows):,} ({split_desc})")
    if any(v > 0 for v in dropped.values()):
        print(f"Dropped: {json.dumps(dropped)}")


if __name__ == "__main__":
    main()
