#!/usr/bin/env python3
"""
Smoke-test an Omni Lua SFT dataset exported as JSONL.

Checks:
  - row schema (messages + audios)
  - user content includes <audio>
  - assistant content is non-empty Lua text
  - audio path exists on disk
  - optional Lua markers

Usage:
    python scripts/smoke_test_omni_lua_dataset.py \
      --dataset-dir data/prepared/omni_lua_sft \
      --split train \
      --max-rows 256 \
      --require-lua-markers
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


REQUIRED_LUA_MARKERS = (
    "MIDI_InsertNote",
    "TrackFX_AddByName(track,\"Vital\"",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/prepared/omni_lua_sft"),
        help="Directory containing all.jsonl/train.jsonl/val.jsonl.",
    )
    parser.add_argument(
        "--split",
        choices=["all", "train", "val"],
        default="train",
        help="Which split to validate (default: train).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=256,
        help="Validate at most this many rows (default: 256).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for row sampling when max-rows < total rows.",
    )
    parser.add_argument(
        "--require-lua-markers",
        action="store_true",
        help="Require key Lua markers (MIDI_InsertNote and Vital plugin load).",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=None,
        help="Optional JSON report output path.",
    )
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
    return rows


def _sample_rows(rows: list[dict], max_rows: int, seed: int) -> list[dict]:
    if max_rows <= 0 or len(rows) <= max_rows:
        return rows
    rng = random.Random(seed)
    return rng.sample(rows, max_rows)


def _validate_row(row: dict, dataset_dir: Path, require_lua_markers: bool) -> list[str]:
    errors: list[str] = []

    msgs = row.get("messages")
    audios = row.get("audios")
    if not isinstance(msgs, list) or len(msgs) < 2:
        return ["messages_missing_or_short"]
    if not isinstance(audios, list) or len(audios) != 1:
        errors.append("audios_missing_or_not_singleton")
    else:
        audio_ref = audios[0]
        if not isinstance(audio_ref, str) or not audio_ref:
            errors.append("audio_path_invalid_type")
        else:
            audio_path = (dataset_dir / audio_ref).resolve()
            if not audio_path.exists():
                errors.append("audio_missing")
            elif audio_path.stat().st_size <= 0:
                errors.append("audio_empty")

    user = msgs[0]
    assistant = msgs[1]
    if user.get("role") != "user":
        errors.append("user_role_invalid")
    if assistant.get("role") != "assistant":
        errors.append("assistant_role_invalid")

    user_content = user.get("content")
    if not isinstance(user_content, str) or "<audio>" not in user_content:
        errors.append("user_content_missing_audio_token")

    lua_text = assistant.get("content")
    if not isinstance(lua_text, str) or not lua_text.strip():
        errors.append("assistant_content_empty")
    elif require_lua_markers:
        for marker in REQUIRED_LUA_MARKERS:
            if marker not in lua_text:
                errors.append(f"assistant_missing_marker:{marker}")

    return errors


def main() -> None:
    args = parse_args()
    split_path = args.dataset_dir / f"{args.split}.jsonl"
    if not split_path.exists():
        raise FileNotFoundError(f"Split not found: {split_path}")

    rows = _load_rows(split_path)
    sampled = _sample_rows(rows, args.max_rows, args.seed)

    failures: list[dict] = []
    for row in sampled:
        row_id = str(row.get("id", "<missing-id>"))
        errs = _validate_row(row, args.dataset_dir, args.require_lua_markers)
        if errs:
            failures.append({"id": row_id, "errors": errs})

    report = {
        "dataset_dir": str(args.dataset_dir),
        "split": args.split,
        "rows_total": len(rows),
        "rows_checked": len(sampled),
        "failed": len(failures),
        "ok": len(failures) == 0,
        "failure_examples": failures[:10],
    }
    text = json.dumps(report, indent=2)
    print(text)

    if args.report_out:
        args.report_out.parent.mkdir(parents=True, exist_ok=True)
        args.report_out.write_text(text + "\n", encoding="utf-8")
        print(f"Report written to: {args.report_out}")

    if failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
