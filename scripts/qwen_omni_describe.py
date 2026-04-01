"""
Compare Qwen3-Omni audio descriptions against MusicCaps ground-truth captions.

Usage:
    python scripts/qwen_omni_describe.py [--n 10] [--data-dir data/prepared/musiccaps_v1]
"""

import argparse
import json
import textwrap
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

MODEL_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
SAMPLING_RATE = 16000
PROMPT = "Describe this audio clip in detail. What instruments, vocals, genre, mood, or sonic qualities do you hear?"


def load_audio(path: Path, target_sr: int = SAMPLING_RATE) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        g = gcd(sr, target_sr)
        audio = resample_poly(audio, target_sr // g, sr // g).astype("float32")
    return audio


def load_entries(manifest_path: Path, n: int) -> list[dict]:
    entries = []
    with open(manifest_path) as f:
        for line in f:
            entry = json.loads(line)
            if entry.get("status") == "ok":
                entries.append(entry)
            if len(entries) >= n:
                break
    return entries


def describe(processor, model, audio: np.ndarray, device: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio", "audio": audio},
                {"type": "text", "text": PROMPT},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=text, audio=audio, sampling_rate=SAMPLING_RATE, return_tensors="pt"
    ).to(device)
    # Cast float tensors to model dtype to avoid bfloat16/float32 mismatch
    model_dtype = next(model.parameters()).dtype
    inputs = {k: v.to(model_dtype) if v.dtype.is_floating_point else v for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            use_audio_in_video=False,
            return_audio=False,
        )

    generated = output_ids[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(generated, skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="Number of clips to describe")
    parser.add_argument(
        "--data-dir",
        default="data/prepared/musiccaps_v1",
        help="Path to prepared musiccaps directory",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    manifest_path = data_dir / "manifest.jsonl"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {MODEL_ID} on {device}...")

    processor = Qwen3OmniMoeProcessor.from_pretrained(MODEL_ID)
    model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
    )
    if device == "cpu":
        model = model.to(device)
    model.eval()

    entries = load_entries(manifest_path, args.n)
    print(f"\nDescribing {len(entries)} clips...\n{'=' * 80}\n")

    for i, entry in enumerate(entries, 1):
        clip_path = data_dir / entry["clip_path"]
        print(f"[{i}/{len(entries)}] {clip_path.name}")
        print(f"  ytid: {entry['ytid']}  ({entry['start_s']:.0f}s – {entry['end_s']:.0f}s)")
        print()

        gt = textwrap.fill(
            entry["caption"], width=76, initial_indent="  GT:    ", subsequent_indent="         "
        )
        print(gt)
        print()

        audio = load_audio(clip_path)
        prediction = describe(processor, model, audio, device)
        pred = textwrap.fill(
            prediction, width=76, initial_indent="  Qwen:  ", subsequent_indent="         "
        )
        print(pred)
        print(f"\n{'-' * 80}\n")


if __name__ == "__main__":
    main()
