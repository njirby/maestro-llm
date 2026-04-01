#!/usr/bin/env python3
"""
Run a toy ACE-Step scenario suite using `acestep-v15-sft`.

This script initializes ACE-Step locally, runs a few short generation tasks
(text2music + optional edit tasks), and stores audio plus a run summary JSON.

Example:
    python scripts/run_acestep_toy.py --device cuda --inference-steps 24 --duration 12
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

import torch

try:
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.model_downloader import SUBMODEL_REGISTRY
except ImportError as exc:
    raise SystemExit(
        "Missing ACE-Step runtime. Install it first, for example:\n"
        "  pip install -e /tmp/ACE-Step-1.5 --no-deps\n"
        "Then install required deps (transformers<4.58, diffusers, loguru, "
        "vector-quantize-pytorch, torchaudio).\n"
        f"Original import error: {exc}"
    )

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


@dataclass
class Scenario:
    name: str
    params: dict[str, Any]

def build_text_scenarios(duration: float, inference_steps: int, seed: int | None) -> list[Scenario]:
    common = {
        "duration": duration,
        "inference_steps": inference_steps,
        "thinking": False,
        "use_cot_metas": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "use_constrained_decoding": False,
        "seed": -1 if seed is None else int(seed),
    }
    return [
        Scenario(
            name="ambient_instrumental",
            params={
                **common,
                "task_type": "text2music",
                "caption": "warm ambient pad with soft piano, cinematic but minimal, gentle swells",
                "lyrics": "[Instrumental]",
                "instrumental": True,
                "bpm": 82,
                "keyscale": "D minor",
            },
        ),
        Scenario(
            name="indie_vocal_hook",
            params={
                **common,
                "task_type": "text2music",
                "caption": "indie pop with female vocal, bright guitars, punchy drums, catchy chorus",
                "lyrics": (
                    "[Verse]\n"
                    "Streetlights blur like watercolor skies,\n"
                    "I hear your name in passing cars tonight.\n\n"
                    "[Chorus]\n"
                    "If we run, we run toward the morning,\n"
                    "holding on till the city starts to shine."
                ),
                "instrumental": False,
                "vocal_language": "en",
                "bpm": 116,
                "keyscale": "G Major",
            },
        ),
        Scenario(
            name="house_rhythm_focus",
            params={
                **common,
                "task_type": "text2music",
                "caption": "tight house groove with deep bass, crisp hats, club-ready punch",
                "lyrics": "[Instrumental]",
                "instrumental": True,
                "bpm": 126,
                "keyscale": "F minor",
            },
        ),
    ]


def build_edit_scenarios(
    src_audio: Path,
    duration: float,
    inference_steps: int,
    seed: int | None,
) -> list[Scenario]:
    common = {
        "src_audio": str(src_audio),
        "duration": duration,
        "inference_steps": inference_steps,
        "thinking": False,
        "use_cot_metas": False,
        "use_cot_caption": False,
        "use_cot_language": False,
        "use_constrained_decoding": False,
        "seed": -1 if seed is None else int(seed),
    }
    return [
        Scenario(
            name="cover_lofi_transform",
            params={
                **common,
                "task_type": "cover",
                "caption": "lofi hip-hop remix, vinyl texture, mellow chords, relaxed swing",
                "lyrics": "[Instrumental]",
                "audio_cover_strength": 0.65,
            },
        ),
        Scenario(
            name="repaint_middle_section",
            params={
                **common,
                "task_type": "repaint",
                "caption": "replace middle phrase with a short expressive guitar lead and smooth transition",
                "lyrics": "[Instrumental]",
                "repainting_start": 3.0,
                "repainting_end": max(5.0, min(duration - 1.0, 8.0)),
            },
        ),
    ]


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


def _has_files(base: Path, required_files: list[str]) -> bool:
    return all((base / rel).exists() for rel in required_files)


def _resolve_dit_repo_id(model_config: str) -> str | None:
    repo_id = SUBMODEL_REGISTRY.get(model_config)
    if repo_id:
        return repo_id
    if model_config.startswith("acestep-"):
        return f"ACE-Step/{model_config}"
    return None


def ensure_minimal_checkpoints(project_root: Path, model_config: str) -> None:
    if snapshot_download is None:
        print("[bootstrap] huggingface_hub not installed; skipping minimal bootstrap.")
        return

    checkpoint_root = project_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    # ACE-Step initializer requires these directories as part of "main model" checks.
    (checkpoint_root / "acestep-v15-turbo").mkdir(exist_ok=True)
    (checkpoint_root / "acestep-5Hz-lm-1.7B").mkdir(exist_ok=True)

    vae_dir = checkpoint_root / "vae"
    text_dir = checkpoint_root / "Qwen3-Embedding-0.6B"
    dit_dir = checkpoint_root / model_config

    need_vae = not _has_files(vae_dir, ["config.json", "diffusion_pytorch_model.safetensors"])
    need_text = not _has_files(text_dir, ["config.json", "model.safetensors", "tokenizer.json"])
    need_dit = not _has_files(dit_dir, ["config.json", "model.safetensors", "silence_latent.pt"])

    if need_vae or need_text:
        print("[bootstrap] downloading shared components (vae + text encoder)...")
        allow_patterns: list[str] = []
        if need_vae:
            allow_patterns.append("vae/*")
        if need_text:
            allow_patterns.append("Qwen3-Embedding-0.6B/*")
        snapshot_download(
            repo_id="ACE-Step/Ace-Step1.5",
            local_dir=str(checkpoint_root),
            allow_patterns=allow_patterns,
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )

    if need_dit:
        repo_id = _resolve_dit_repo_id(model_config)
        if repo_id is None:
            raise RuntimeError(f"Could not resolve Hugging Face repo for model_config='{model_config}'")
        print(f"[bootstrap] downloading DiT model '{model_config}' from {repo_id}...")
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(dit_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            max_workers=4,
        )


def run_one(
    dit_handler: AceStepHandler,
    scenario: Scenario,
    out_dir: Path,
    batch_size: int,
    audio_format: str,
    random_seed: bool,
    explicit_seed: int | None,
) -> dict[str, Any]:
    params = GenerationParams(**scenario.params)
    config = GenerationConfig(
        batch_size=batch_size,
        audio_format=audio_format,
        use_random_seed=random_seed,
        seeds=[explicit_seed] if explicit_seed is not None else None,
    )

    start = time.perf_counter()
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=None,
        params=params,
        config=config,
        save_dir=str(out_dir),
    )
    elapsed = time.perf_counter() - start

    audio_paths = [a.get("path") for a in result.audios]
    seeds = [a.get("params", {}).get("seed") for a in result.audios]
    return {
        "name": scenario.name,
        "task_type": scenario.params.get("task_type"),
        "success": bool(result.success),
        "error": result.error,
        "status_message": result.status_message,
        "elapsed_sec": round(elapsed, 3),
        "audio_paths": audio_paths,
        "seeds": seeds,
        "params": scenario.params,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run toy ACE-Step v1.5 scenarios.")
    parser.add_argument("--model-config", default="acestep-v15-sft", help="DiT checkpoint folder name.")
    parser.add_argument("--device", default="auto", help="Device for DiT (auto/cuda/cpu/xpu).")
    parser.add_argument("--duration", type=float, default=12.0, help="Target duration (seconds).")
    parser.add_argument("--inference-steps", type=int, default=20, help="Diffusion inference steps.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size per scenario.")
    parser.add_argument("--seed", type=int, default=None, help="Fixed seed for reproducible runs.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output root directory (default: <repo>/outputs/acestep_toy).",
    )
    parser.add_argument("--audio-format", default="wav", choices=["wav", "flac", "mp3", "wav32", "opus", "aac"])
    parser.add_argument("--max-scenarios", type=int, default=0, help="Limit number of scenarios (0 = all).")
    parser.add_argument("--skip-editing", action="store_true", help="Skip cover/repaint tasks.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failed scenario.")
    parser.add_argument(
        "--skip-bootstrap",
        action="store_true",
        help="Skip minimal checkpoint bootstrap and use ACE-Step's default downloader.",
    )
    return parser.parse_args()


def main() -> int:
    os.environ.setdefault("ACESTEP_DISABLE_TQDM", "1")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    args = parse_args()

    project_root = Path(__file__).resolve().parents[1]
    output_root = Path(args.output_dir) if args.output_dir else project_root / "outputs" / "acestep_toy"
    run_dir = output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    print(f"[init] project_root={project_root}")
    print(f"[init] output_dir={run_dir}")
    print(f"[init] model_config={args.model_config} device={device}")

    if not args.skip_bootstrap:
        ensure_minimal_checkpoints(project_root=project_root, model_config=args.model_config)

    dit_handler = AceStepHandler()
    status, ok = dit_handler.initialize_service(
        project_root=str(project_root),
        config_path=args.model_config,
        device=device,
    )
    print(f"[init] {status}")
    if not ok:
        print("[error] failed to initialize ACE-Step model")
        return 1

    text_scenarios = build_text_scenarios(args.duration, args.inference_steps, args.seed)
    run_scenarios = list(text_scenarios)
    if args.max_scenarios > 0:
        run_scenarios = run_scenarios[: args.max_scenarios]

    summary: dict[str, Any] = {
        "model_config": args.model_config,
        "device": device,
        "duration": args.duration,
        "inference_steps": args.inference_steps,
        "batch_size": args.batch_size,
        "seed": args.seed,
        "audio_format": args.audio_format,
        "run_dir": str(run_dir),
        "results": [],
    }

    anchor_audio: Path | None = None
    for idx, scenario in enumerate(run_scenarios, start=1):
        print(f"[run {idx}/{len(run_scenarios)}] {scenario.name}")
        result = run_one(
            dit_handler=dit_handler,
            scenario=scenario,
            out_dir=run_dir,
            batch_size=args.batch_size,
            audio_format=args.audio_format,
            random_seed=args.seed is None,
            explicit_seed=args.seed,
        )
        summary["results"].append(result)
        if result["audio_paths"] and anchor_audio is None:
            anchor_audio = Path(result["audio_paths"][0])
        if result["success"]:
            print(f"[ok] {scenario.name} -> {result['audio_paths']}")
        else:
            print(f"[fail] {scenario.name}: {result['error']}")
            if args.fail_fast:
                break

    if not args.skip_editing and anchor_audio is not None:
        edit_scenarios = build_edit_scenarios(
            src_audio=anchor_audio,
            duration=args.duration,
            inference_steps=args.inference_steps,
            seed=args.seed,
        )
        if args.max_scenarios > 0:
            remaining = max(0, args.max_scenarios - len(summary["results"]))
            edit_scenarios = edit_scenarios[:remaining]

        for offset, scenario in enumerate(edit_scenarios, start=1):
            print(f"[run edit {offset}/{len(edit_scenarios)}] {scenario.name}")
            result = run_one(
                dit_handler=dit_handler,
                scenario=scenario,
                out_dir=run_dir,
                batch_size=args.batch_size,
                audio_format=args.audio_format,
                random_seed=args.seed is None,
                explicit_seed=args.seed,
            )
            summary["results"].append(result)
            if result["success"]:
                print(f"[ok] {scenario.name} -> {result['audio_paths']}")
            else:
                print(f"[fail] {scenario.name}: {result['error']}")
                if args.fail_fast:
                    break
    elif not args.skip_editing:
        print("[warn] skipping edit scenarios because no anchor audio was produced.")

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    success_count = sum(1 for r in summary["results"] if r["success"])
    print(f"[done] {success_count}/{len(summary['results'])} scenarios succeeded")
    print(f"[done] summary: {summary_path}")

    return 0 if success_count == len(summary["results"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
