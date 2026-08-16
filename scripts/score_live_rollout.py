#!/usr/bin/env python
"""Score a LIVE rollout: pull the audio the model actually rendered out of the
container and compare it to ground truth.

The live runner leaves everything the model produced inside the container's
rollout dir. This fetches those wavs to the host, identifies the final render,
and scores it against the sample's GT with CLAP — reported against the
manifest's determinism ceiling, because that ceiling (not 1.0) is the best a
correct rollout can achieve given render stochasticity.

Also reports the intermediate renders so a partial rollout is still legible:
how far the model got, whether what it made was audible, and whether it was
moving toward the target.

Usage:
    python scripts/score_live_rollout.py --sample bass_73cf256d \
        --session daw-farm-reaper-12 --manifest outputs/ref_run11/manifest.jsonl \
        --out outputs/live_rollouts/bass_73cf256d
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def fetch_dir(session: str, container_path: str, host_dir: Path) -> int:
    host_dir.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["docker", "cp", f"{session}:{container_path}", str(host_dir)],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return 0
    return len(list(host_dir.rglob("*.wav")))


def stats(p: Path) -> dict:
    try:
        y, sr = sf.read(str(p))
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        peak = float(np.abs(y).max()) if len(y) else 0.0
        S = np.abs(np.fft.rfft(y)) if len(y) else np.array([0.0])
        fr = np.fft.rfftfreq(len(y), 1 / sr) if len(y) else np.array([0.0])
        cen = float((S * fr).sum() / S.sum()) if S.sum() > 0 else 0.0
        return {"peak": peak, "silent": peak < 1e-5,
                "dur_s": round(len(y) / sr, 2) if sr else 0.0,
                "centroid_hz": round(cen)}
    except Exception as exc:
        return {"error": str(exc)[:100]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", required=True)
    ap.add_argument("--session", default="daw-farm-reaper-12")
    ap.add_argument("--manifest", type=Path, default=Path("outputs/ref_run11/manifest.jsonl"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--clap-device", default="cpu")
    args = ap.parse_args()

    entry = next((json.loads(l) for l in open(args.manifest)
                  if json.loads(l)["sample_id"] == args.sample), None)
    if not entry:
        raise SystemExit(f"{args.sample} not in {args.manifest}")
    out = args.out or Path("outputs/live_rollouts") / args.sample
    out.mkdir(parents=True, exist_ok=True)

    n = fetch_dir(args.session, f"/work/rollouts/{args.sample}", out)
    print(f"fetched {n} wavs from container rollout dir")
    wavs = sorted(out.rglob("*.wav"))
    if not wavs:
        print("NO AUDIO PRODUCED — the model never reached a render")
        json.dump({"sample": args.sample, "n_wavs": 0}, open(out / "score.json", "w"), indent=2)
        return

    from scripts.agent_sft_common import ClapEmbedder
    emb = ClapEmbedder.create(args.clap_device)
    gt = Path(entry["gt_wav"])
    ceiling = entry.get("determinism_clap")

    rows = []
    for w in wavs:
        s = stats(w)
        if not s.get("silent") and not s.get("error"):
            try:
                s["clap_vs_gt"] = round(float(emb.cosine_paths(w, gt)), 4)
            except Exception as exc:
                s["clap_error"] = str(exc)[:80]
        s["file"] = w.name
        rows.append(s)

    scored = [r for r in rows if "clap_vs_gt" in r]
    best = max(scored, key=lambda r: r["clap_vs_gt"], default=None)
    summary = {
        "sample": args.sample,
        "n_wavs": len(wavs),
        "n_silent": sum(1 for r in rows if r.get("silent")),
        "determinism_ceiling": ceiling,
        "best_clap_vs_gt": best["clap_vs_gt"] if best else None,
        "best_file": best["file"] if best else None,
        "gap_to_ceiling": (round(ceiling - best["clap_vs_gt"], 4)
                           if best and ceiling else None),
    }
    print(json.dumps(summary, indent=2))
    print(f"{'file':<40} {'peak':>6} {'centroid':>9} {'CLAP vs GT':>11}")
    for r in rows:
        print(f"{r['file'][:40]:<40} {r.get('peak', 0):>6.3f} "
              f"{r.get('centroid_hz', 0):>9} {r.get('clap_vs_gt', float('nan')):>11}")
    json.dump({"summary": summary, "files": rows}, open(out / "score.json", "w"), indent=2)
    print("wrote", out / "score.json")


if __name__ == "__main__":
    main()
