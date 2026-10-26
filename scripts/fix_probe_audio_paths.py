#!/usr/bin/env python3
"""Remap search/judge rollout audio paths from /tmp/ to standard probes.

Commit be8c282 changed audios lists to point to /tmp/search_probes/ and
/tmp/judge_probes/ paths that were never rendered.  The Omni descriptions
in these rollouts were generated from the pre-cached standard probes in
outputs/agent_sft/candidate_probes/.  This script remaps the audios back
to those existing files.

Usage:
    python scripts/fix_probe_audio_paths.py \
        --rollouts-dir outputs/sft_32k/rollouts \
        --probe-dir outputs/agent_sft/candidate_probes
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import validate_ms_swift_multiturn_record


def build_slug_to_probe_path(probe_dir: Path) -> dict[str, str]:
    """Map rollout wavetable slug → absolute path of standard probe WAV.

    The standard probes are named {slug}__src{N}_f{N}.wav.  Rollout filenames
    are wt_{idx}_{slug}.wav.  Both use the same slugify(), so we can match
    directly without going through the wavetable library.
    """
    probe_files = sorted(probe_dir.glob("*.wav"))

    mapping: dict[str, str] = {}
    for p in probe_files:
        stem = p.stem
        idx = stem.find("__src")
        if idx == -1:
            continue
        slug = stem[:idx]
        if slug not in mapping:
            mapping[slug] = str(p)

    return mapping


def rewrite_rollouts(
    input_path: Path,
    output_path: Path,
    slug_to_probe: dict[str, str],
) -> dict:
    """Rewrite audios lists, replacing /tmp/ paths with standard probe paths."""
    stats = {"total": 0, "rewritten": 0, "dropped": 0, "paths_fixed": 0, "paths_unfixed": 0}

    with open(input_path) as fin, open(output_path, "w") as fout:
        for line in fin:
            row = json.loads(line)
            stats["total"] += 1
            audios = row.get("audios", [])
            new_audios = []
            all_resolved = True

            for audio_path in audios:
                if "/tmp/search_probes/" in audio_path or "/tmp/judge_probes/" in audio_path:
                    fname = Path(audio_path).name
                    m = re.match(r"wt_\d+_(.+)\.wav", fname)
                    if m and m.group(1) in slug_to_probe:
                        new_audios.append(slug_to_probe[m.group(1)])
                        stats["paths_fixed"] += 1
                    else:
                        new_audios.append(audio_path)
                        stats["paths_unfixed"] += 1
                        all_resolved = False
                else:
                    new_audios.append(audio_path)

            if not all_resolved:
                stats["dropped"] += 1
                continue

            row["audios"] = new_audios
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rewritten"] += 1

    return stats


def validate_output(path: Path) -> dict:
    """Run MS-Swift validation and check audio file existence."""
    total = 0
    valid = 0
    missing_audio = 0
    errors: dict[str, int] = {}

    with open(path) as f:
        for line in f:
            row = json.loads(line)
            total += 1

            errs = validate_ms_swift_multiturn_record(row)
            if errs:
                for e in errs:
                    errors[e] = errors.get(e, 0) + 1
            else:
                valid += 1

            for a in row.get("audios", []):
                if not Path(a).exists():
                    missing_audio += 1

    return {"total": total, "valid": valid, "missing_audio": missing_audio, "errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rollouts-dir", type=Path, default=ROOT / "outputs/sft_32k/rollouts")
    ap.add_argument("--probe-dir", type=Path, default=ROOT / "outputs/agent_sft/candidate_probes")
    ap.add_argument("--skip-validate", action="store_true")
    args = ap.parse_args()

    print("Step 1: Building slug → standard probe mapping...", flush=True)
    slug_to_probe = build_slug_to_probe_path(args.probe_dir)
    print(f"  {len(slug_to_probe)} wavetable slugs mapped to standard probes", flush=True)

    print("\nStep 2: Rewriting rollouts...", flush=True)
    for agent in ["search", "judge"]:
        src = args.rollouts_dir / f"{agent}_final_v1.jsonl"
        dst = args.rollouts_dir / f"{agent}_final_v2.jsonl"
        if not src.exists():
            print(f"  SKIP {agent}: {src} not found", flush=True)
            continue
        stats = rewrite_rollouts(src, dst, slug_to_probe)
        print(f"  {agent}: {stats['rewritten']} rewritten, "
              f"{stats['dropped']} dropped, "
              f"{stats['paths_fixed']} paths fixed, "
              f"{stats['paths_unfixed']} unfixable", flush=True)

    if not args.skip_validate:
        print("\nStep 3: Validating rewritten rollouts...", flush=True)
        for agent in ["search", "judge"]:
            path = args.rollouts_dir / f"{agent}_final_v2.jsonl"
            if not path.exists():
                continue
            result = validate_output(path)
            print(f"  {agent}: {result['valid']}/{result['total']} valid, "
                  f"{result['missing_audio']} missing audio files", flush=True)
            if result["errors"]:
                top = sorted(result["errors"].items(), key=lambda x: -x[1])[:5]
                for e, c in top:
                    print(f"    {e}: {c}", flush=True)

    print("\nDone. Next steps:", flush=True)
    print("  python scripts/merge_agent_sft.py \\", flush=True)
    print("    --input outputs/sft_32k/rollouts/main_final_v1.jsonl \\", flush=True)
    print("    --input outputs/sft_32k/rollouts/search_final_v2.jsonl \\", flush=True)
    print("    --input outputs/sft_32k/rollouts/judge_final_v2.jsonl \\", flush=True)
    print("    --input outputs/sft_32k/rollouts/transcription_final_v1.jsonl \\", flush=True)
    print("    --output outputs/sft_32k/sft_train_v2.jsonl \\", flush=True)
    print("    --shuffle --dedupe-id --drop-invalid", flush=True)


if __name__ == "__main__":
    main()
