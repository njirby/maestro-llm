#!/usr/bin/env python3
"""Experiment: can Omni meaningfully compare multiple wavetable candidate audios in one prompt?

Sends target audio + N candidate wavetable probe audios to Omni and checks whether
the model produces distinct, per-candidate assessments.

Usage:
    python scripts/experiment_omni_batch_listen.py \
        --manifest outputs/smoke_test_v10/manifest.jsonl \
        --wavetable-lib data/wavetable_lib.json \
        --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
        --omni-server http://localhost:8000 \
        --candidates-per-batch 8 \
        --n-samples 3
"""
from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import (
    ensure_candidate_probes_for_names,
    load_manifest_entries,
    load_wavetable_lib,
    load_index_rows,
    select_probe_rows_by_name,
)
from scripts.build_main_agent_sft_v2 import _llm_post
from scripts.build_wavetable_retrieval_baseline import _extract_gt_wavetable_names


def _b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def run_batch_listen(
    target_wav: Path,
    candidate_names: list[str],
    candidate_audio: dict[str, Path],
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """Send target + N candidates to Omni, return the raw response."""
    content: list[dict] = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(target_wav)}"}},
    ]
    for name in candidate_names:
        if name in candidate_audio:
            content.append(
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(candidate_audio[name])}"}}
            )

    # Build the text prompt
    candidate_list = "\n".join(
        f"  Audio {i+2}: \"{name}\""
        for i, name in enumerate(candidate_names)
        if name in candidate_audio
    )
    n_candidates = sum(1 for n in candidate_names if n in candidate_audio)

    content.append({"type": "text", "text": (
        f"You are a sound design assistant evaluating wavetable candidates.\n\n"
        f"Audio 1 is the TARGET {archetype} sound we want to recreate.\n"
        f"Audios 2-{n_candidates + 1} are candidate wavetables, each rendered through "
        f"a default synthesizer preset with the same notes.\n\n"
        f"Candidate list:\n{candidate_list}\n\n"
        f"For EACH candidate, write exactly one sentence describing how similar or "
        f"different it sounds compared to the target. Use the format:\n"
        f"  \"{candidate_names[0]}\": <one sentence>\n"
        f"  \"{candidate_names[1]}\": <one sentence>\n"
        f"  ...etc\n\n"
        f"Be specific about what matches or doesn't match: harmonic character, "
        f"brightness, texture, attack shape. Be concise."
    )})

    payload = {
        "model": omni_model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 600,
        "temperature": 0.4,
    }
    result = _llm_post(f"{omni_server}/v1/chat/completions", payload, timeout=180.0)
    return result["choices"][0]["message"]["content"].strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--omni-server", default="http://localhost:8000")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--candidates-per-batch", type=int, default=8)
    ap.add_argument("--n-samples", type=int, default=3)
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))
    args = ap.parse_args()

    entries = load_manifest_entries(args.manifest, max_samples=args.n_samples)
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    all_names = sorted(selected_by_name.keys())

    candidate_audio: dict[str, Path] = {}

    for entry in entries:
        sample_id = entry["sample_id"]
        archetype = entry.get("archetype", "synth")
        target_wav = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))

        # Get GT wavetable names
        path_file = entry.get("path_file")
        target_preset_path = entry.get("target_preset_path")
        if not target_preset_path and path_file:
            with open(path_file) as f:
                pd = json.load(f)
            target_preset_path = pd.get("target_preset_path")

        gt_names = _extract_gt_wavetable_names(Path(target_preset_path)) if target_preset_path else []

        # Pick candidates: GT + random others
        import random
        rng = random.Random(42)
        non_gt = [n for n in all_names if n not in gt_names]
        rng.shuffle(non_gt)
        candidates = list(gt_names) + non_gt[:args.candidates_per_batch - len(gt_names)]
        rng.shuffle(candidates)

        # Ensure probe audio exists
        ensure_candidate_probes_for_names(
            names=candidates,
            wavetable_lib=wavetable_lib,
            selected_rows=selected_by_name,
            out_dir=args.probe_dir,
            cache=candidate_audio,
        )

        print(f"\n{'='*80}")
        print(f"Sample: {sample_id} ({archetype})")
        print(f"GT wavetables: {gt_names}")
        print(f"Candidates: {candidates}")
        print(f"Sending target + {len(candidates)} candidates to Omni...")
        print(f"{'='*80}\n")

        try:
            response = run_batch_listen(
                target_wav=target_wav,
                candidate_names=candidates,
                candidate_audio=candidate_audio,
                archetype=archetype,
                omni_server=args.omni_server,
                omni_model=args.omni_model,
            )
            print(response)

            # Check: how many candidates were addressed?
            addressed = sum(1 for name in candidates if name.lower() in response.lower() or name[:15].lower() in response.lower())
            print(f"\n--- Addressed {addressed}/{len(candidates)} candidates by name ---")
            if addressed < len(candidates):
                missing = [n for n in candidates if n.lower() not in response.lower() and n[:15].lower() not in response.lower()]
                print(f"    Missing: {missing}")

        except Exception as exc:
            print(f"ERROR: {exc}")


if __name__ == "__main__":
    main()
