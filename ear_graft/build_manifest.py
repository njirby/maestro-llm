"""Build data/ear_corpus/manifest.jsonl from local audio + downloaded HF parquet sources.

  .venv/bin/python -m ear_graft.build_manifest [--sources musiccaps,samples,...] [--seed 0]

Every clip the teacher hears is exactly what the student will hear: <= 30 s, stored at native sample rate / channels
(kept for the later wide-band/stereo work; the student front-end downmixes + resamples). Files already <= 30 s are
referenced in place; longer files get one non-silent 10-30 s segment written to data/ear_corpus/clips/<source>/ as FLAC.
Re-running appends only sources not yet in the manifest.
"""
import argparse
import glob
import io
import json
import os
import random
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path("/home/nate/Documents/maestro-llm/data")
CORPUS = ROOT / "ear_corpus"
MAX_S = 30.0

# name: (glob or dir, kind, cap) — local sources
LOCAL = {
    "musiccaps": ("prepared/musiccaps_v1/clips/*.wav", "music", 5000),
    "youtube_songs": ("audio/*.mp3", "music", 1500),
    "samples": ("raw/phase1_sources/samples/**/*.wav", "synth", 1400),
    "vital_renders": ("processed/reaper_tuples_lakh/wavs/*.mp3", "synth", 3000),
    "slakh_stems": ("raw/phase1_sources/slakh2100/slakh2100_flac_redux/*/*/stems/*.flac", "synth", 2000),
    "musdb_stems": ("raw/phase1_sources/musdb18_hq/*/*/*.wav", "music", 750),
    "medleydb": ("raw/phase1_sources/medleydb/**/*_STEMS/*.wav", "music", 200),
}


def segment(path, rng, min_s=10.0, max_s=MAX_S, tries=6):
    """Return (audio, sr, offset_s) for a non-silent window of a long file, or None."""
    info = sf.info(path)
    L = rng.uniform(min_s, max_s)
    best = None
    for _ in range(tries):
        off = rng.uniform(0, max(0.0, info.duration - L))
        y, sr = sf.read(path, start=int(off * info.samplerate), frames=int(L * info.samplerate), dtype="float32", always_2d=True)
        rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
        if best is None or rms > best[3]:
            best = (y, sr, off, rms)
        if 20 * np.log10(rms) > -40:
            break
    return None if 20 * np.log10(best[3]) < -55 else best[:3]


def local_rows(name, pattern, kind, cap, rng):
    files = sorted(glob.glob(str(ROOT / pattern), recursive=True))
    rng.shuffle(files)
    out_dir = CORPUS / "clips" / name
    rows = []
    for f in files:
        if len(rows) >= cap:
            break
        try:
            info = sf.info(f)
        except Exception:
            continue
        if info.duration < 1.0:
            continue
        rid = f"{name}/{len(rows):06d}"
        meta = {"orig": os.path.relpath(f, ROOT)}
        if name == "musiccaps":
            meta["ytid"] = Path(f).stem.split("_", 1)[1]
        if info.duration <= MAX_S + 0.05:
            path, dur, sr, ch = f, info.duration, info.samplerate, info.channels
        else:
            seg = segment(f, rng)
            if seg is None:
                continue  # silent stem
            y, sr, off = seg
            out_dir.mkdir(parents=True, exist_ok=True)
            path = str(out_dir / f"{len(rows):06d}.flac")
            sf.write(path, y, sr, format="FLAC")
            dur, ch, meta["offset_s"] = len(y) / sr, y.shape[1], round(off, 2)
        rows.append({"id": rid, "source": name, "kind": kind, "path": path, "dur": round(dur, 2), "sr": sr,
                     "channels": ch, "meta": meta})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=",".join(LOCAL))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    CORPUS.mkdir(parents=True, exist_ok=True)
    man = CORPUS / "manifest.jsonl"
    have = {json.loads(l)["source"] for l in open(man)} if man.exists() else set()
    with open(man, "a") as f:
        for name in args.sources.split(","):
            if name in have:
                print(f"{name}: already in manifest, skip")
                continue
            rng = random.Random(f"{args.seed}-{name}")
            if name in LOCAL:
                rows = local_rows(name, *LOCAL[name], rng)
            else:
                from .parquet_sources import parquet_rows
                rows = parquet_rows(name, rng)
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            print(f"{name}: {len(rows)} clips, {sum(r['dur'] for r in rows) / 3600:.1f} h")


if __name__ == "__main__":
    main()
