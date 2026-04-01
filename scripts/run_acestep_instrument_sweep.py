#!/usr/bin/env python3
"""
Generate a variety of instrument-focused clips with ACE-Step.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from acestep.handler import AceStepHandler
from acestep.inference import GenerationConfig, GenerationParams, generate_music

from run_acestep_toy import choose_device, ensure_minimal_checkpoints


@dataclass
class InstrumentPrompt:
    name: str
    caption: str
    bpm: int
    keyscale: str


INSTRUMENT_PROMPTS: list[InstrumentPrompt] = [
    InstrumentPrompt("acoustic_piano", "solo acoustic piano, expressive dynamics, close-mic studio tone", 82, "C Major"),
    InstrumentPrompt("electric_guitar", "clean electric guitar melody with subtle delay, modern indie tone", 104, "E minor"),
    InstrumentPrompt("violin", "solo violin lyrical phrase, warm bow texture, cinematic chamber feel", 76, "D minor"),
    InstrumentPrompt("trumpet", "jazz trumpet lead with muted articulation and smoky club ambience", 112, "Bb minor"),
    InstrumentPrompt("flute", "breathy concert flute motif, airy and intimate, gentle vibrato", 78, "G Major"),
    InstrumentPrompt("saxophone", "tenor sax riff with blues phrasing and warm analog room response", 98, "F minor"),
    InstrumentPrompt("cello", "solo cello legato lines, rich low mids, film-score mood", 70, "A minor"),
    InstrumentPrompt("harp", "solo concert harp arpeggios, shimmering transients, soft reverb tail", 72, "Db Major"),
    InstrumentPrompt("marimba", "dry marimba groove, woody attack, precise rhythmic pattern", 120, "C minor"),
    InstrumentPrompt("drum_kit", "acoustic drum kit groove only, punchy kick and crisp snare", 126, "N/A"),
    InstrumentPrompt("electric_bass", "fingerstyle electric bass groove, round low end, tight timing", 110, "E minor"),
    InstrumentPrompt("synth_lead", "mono analog synth lead, bright saw tone, energetic phrasing", 128, "A minor"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ACE-Step instrument sweep.")
    parser.add_argument("--model-config", default="acestep-v15-sft")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--inference-steps", type=int, default=12)
    parser.add_argument("--audio-format", default="wav", choices=["wav", "flac", "mp3", "wav32", "opus", "aac"])
    parser.add_argument("--seed", type=int, default=1234, help="Base seed; each instrument uses seed+i.")
    parser.add_argument("--limit", type=int, default=0, help="Limit number of instruments (0 = all).")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--skip-bootstrap", action="store_true")
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "acestep_instrument_sweep"
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    if not args.skip_bootstrap:
        ensure_minimal_checkpoints(project_root=project_root, model_config=args.model_config)

    handler = AceStepHandler()
    status, ok = handler.initialize_service(
        project_root=str(project_root),
        config_path=args.model_config,
        device=device,
    )
    print(status)
    if not ok:
        print("Initialization failed.")
        return 1

    prompts = INSTRUMENT_PROMPTS if args.limit <= 0 else INSTRUMENT_PROMPTS[: args.limit]
    summary: dict[str, Any] = {
        "model_config": args.model_config,
        "device": device,
        "duration": args.duration,
        "inference_steps": args.inference_steps,
        "audio_format": args.audio_format,
        "run_dir": str(run_dir),
        "results": [],
    }

    for idx, prompt in enumerate(prompts):
        seed = args.seed + idx
        params = GenerationParams(
            task_type="text2music",
            caption=prompt.caption,
            lyrics="[Instrumental]",
            instrumental=True,
            bpm=prompt.bpm,
            keyscale=prompt.keyscale,
            duration=args.duration,
            inference_steps=args.inference_steps,
            seed=seed,
            thinking=False,
            use_cot_metas=False,
            use_cot_caption=False,
            use_cot_language=False,
            use_constrained_decoding=False,
        )
        config = GenerationConfig(batch_size=1, audio_format=args.audio_format, use_random_seed=False, seeds=[seed])

        t0 = time.perf_counter()
        result = generate_music(
            dit_handler=handler,
            llm_handler=None,
            params=params,
            config=config,
            save_dir=str(run_dir),
        )
        elapsed = time.perf_counter() - t0

        audio_paths = [a.get("path") for a in result.audios]
        print(f"{prompt.name}: {'OK' if result.success else 'FAIL'} -> {audio_paths}")
        summary["results"].append(
            {
                "instrument": prompt.name,
                "caption": prompt.caption,
                "seed": seed,
                "success": bool(result.success),
                "error": result.error,
                "elapsed_sec": round(elapsed, 3),
                "audio_paths": audio_paths,
            }
        )

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    success_count = sum(1 for r in summary["results"] if r["success"])
    print(f"{success_count}/{len(summary['results'])} succeeded")
    print(f"summary: {summary_path}")
    return 0 if success_count == len(summary["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
