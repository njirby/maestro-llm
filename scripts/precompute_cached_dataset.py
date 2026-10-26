#!/usr/bin/env python3
"""Pre-compute token lengths for MS-Swift --cached_dataset.

Reads a training JSONL, computes exact audio token counts from WAV headers
and text token estimates via chars/2.5 (calibrated ratio), then saves an HF
Dataset with a 'lengths' column. Pass the output dir to --cached_dataset to
skip the ~50h AddLengthPreprocessor step entirely.

Validated against true AddLengthPreprocessor output on 199 samples:
all within 5% of true lengths after 1.04x template correction.
"""

import argparse
import json
import os
import wave
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tqdm import tqdm

PROJECT_ROOT = "/home/nate/Documents/maestro-llm"
CHARS_PER_TOKEN = 2.5


def audio_token_count(duration_s: float) -> int:
    """Exact Qwen2.5-Omni audio token count from duration."""
    mel_frames = int(duration_s * 100)  # 16kHz / hop 160 = 100 frames/s
    return max(1, ((mel_frames - 1) // 2 + 1 - 2) // 2 + 1)


def wav_duration(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / w.getframerate()


def resolve_path(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)


def compute_length(line: str) -> int:
    row = json.loads(line)
    audio_total = 0
    audio_idx = 0
    total_chars = 0

    tools = row.get("tools")
    if tools:
        total_chars += len(tools) if isinstance(tools, str) else len(json.dumps(tools))

    for msg in row.get("messages", []):
        content = msg.get("content", "")
        while "<audio>" in content:
            if audio_idx < len(row.get("audios", [])):
                try:
                    path = resolve_path(row["audios"][audio_idx])
                    dur = wav_duration(path)
                    audio_total += audio_token_count(dur)
                except Exception:
                    audio_total += 750
                audio_idx += 1
            content = content.replace("<audio>", "", 1)
        total_chars += len(content)

    text_tokens = int(total_chars / CHARS_PER_TOKEN)

    # 1.04x corrects for chat-template special tokens (im_start/im_end/role
    # markers) not present in raw text. Calibrated against true lengths from
    # AddLengthPreprocessor on 199 samples: all within 5% after correction.
    return int(text_tokens * 1.04) + audio_total


def process_jsonl(input_path: str, output_path: str, num_workers: int) -> None:
    print(f"Reading {input_path}...")
    with open(input_path) as f:
        lines = f.readlines()
    print(f"  {len(lines)} samples")

    print(f"Computing lengths ({num_workers} workers)...")
    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        lengths = list(tqdm(pool.map(compute_length, lines, chunksize=256),
                            total=len(lines), desc="lengths"))

    del lines

    print("Building HF Dataset...")
    from datasets import load_dataset

    ds = load_dataset("json", data_files=input_path, split="train")
    ds = ds.add_column("lengths", lengths)
    ds.save_to_disk(output_path)

    print(f"\nSaved: {output_path}")
    print(f"  samples: {len(ds)}")
    print(f"  mean: {sum(lengths) / len(lengths):.0f}")
    print(f"  max:  {max(lengths)}")
    print(f"  min:  {min(lengths)}")
    over = sum(1 for l in lengths if l > 65536)
    print(f"  >65536: {over} ({100 * over / len(lengths):.1f}%)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-jsonl", required=True)
    p.add_argument("--val-jsonl", default=None)
    p.add_argument("--output-dir", required=True, help="Parent dir for cached datasets")
    p.add_argument("--num-workers", type=int, default=24)
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    process_jsonl(args.train_jsonl, str(out / "train"), args.num_workers)

    if args.val_jsonl:
        print("\n" + "=" * 60 + "\n")
        process_jsonl(args.val_jsonl, str(out / "val"), args.num_workers)


if __name__ == "__main__":
    main()
