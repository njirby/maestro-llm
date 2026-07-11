#!/usr/bin/env python
"""Re-render a stage-A manifest's model-visible audio through daw-farm.

Rollouts train the model to act inside a daw-farm REAPER session, so the
audio it hears — above all the ground-truth target — must come from that
exact environment. Stage A (render_iter_presets.py) renders with vita, a
different build of the Vital engine; this post-pass re-renders `gt_wav` and
`default_wav` for every manifest entry through a real session (same plugin,
same REAPER master chain, same render action the rollout snippets use) and
overwrites the files in place. The vita originals are kept alongside as
`<name>.vita.wav`.

Run BEFORE build_main_agent_sft_v3.py --daw-farm.

Usage:
    python scripts/rerender_manifest_dawfarm.py \
        --manifest outputs/<run>/manifest.jsonl \
        [--daw-farm docker] [--daw-farm-vital-data data/prepared/wavetable_lib_vital_dir] \
        [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from maestro.reaper import dawfarm


def _load_notes(source_midi_path: str) -> list[dict]:
    from scripts.build_transcription_agent_sft_v4 import load_notes_from_midi
    return load_notes_from_midi(source_midi_path)


_embedder = None
_embedder_lock = None


def _clap_cosine(a, b) -> float:
    global _embedder, _embedder_lock
    import threading
    from scripts.agent_sft_common import ClapEmbedder
    if _embedder_lock is None:
        _embedder_lock = threading.Lock()
    with _embedder_lock:
        if _embedder is None:
            _embedder = ClapEmbedder.create("cpu")
        return float(_embedder.cosine_paths(Path(a), Path(b)))


def _rerender_entry(pool: dawfarm.DawFarmPool, entry: dict, vital_data: str) -> str:
    sid = entry["sample_id"]
    pd = json.load(open(entry["path_file"]))
    target = json.load(open(pd["target_preset_path"]))
    if pd.get("start_preset_path"):
        start = json.load(open(pd["start_preset_path"]))
    else:
        start = json.load(open(ROOT / "maestro" / "synth" / "init_preset.json"))
    notes = _load_notes(entry["source_midi_path"])

    jobs = [(target, entry.get("gt_wav")), (start, entry.get("default_wav"))]
    with pool.acquire() as s:
        dawfarm.sync_vital_data(s, vital_data)
        for preset, wav in jobs:
            if not wav:
                continue
            wav = Path(wav)
            backup = wav.with_suffix(".vita.wav")
            if wav.exists() and not backup.exists():
                shutil.copy2(wav, backup)
            dawfarm.render_preset_in_reaper(s, preset, notes, wav, tag=f"stage_a_{sid}")
            audio, sr = sf.read(wav)
            peak = float(np.abs(audio).max())
            if peak < 1e-4:
                raise RuntimeError(f"{sid}: env render of {wav.name} is silent (peak={peak})")
        # Determinism ceiling: render the target a second time and CLAP the
        # two GT renders. Patches with random osc phase / free-running LFOs
        # are render-stochastic — final CLAP is only interpretable relative
        # to this per-sample self-similarity.
        if entry.get("gt_wav"):
            gt_b = Path(entry["gt_wav"]).with_name(Path(entry["gt_wav"]).stem + "_b.wav")
            dawfarm.render_preset_in_reaper(s, target, notes, gt_b, tag=f"stage_a_{sid}_b")
            entry["determinism_clap"] = round(_clap_cosine(entry["gt_wav"], gt_b), 4)
        dawfarm.reset_project(s)
    return sid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, type=Path)
    ap.add_argument("--daw-farm", default="docker")
    ap.add_argument("--daw-farm-vital-data",
                    default=str(ROOT / "data" / "prepared" / "wavetable_lib_vital_dir"))
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    entries = [json.loads(l) for l in open(args.manifest)]
    pool = dawfarm.DawFarmPool.from_spec(args.daw_farm)

    ok, failed = 0, []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_rerender_entry, pool, e, args.daw_farm_vital_data): e["sample_id"]
                for e in entries}
        for fut in as_completed(futs):
            sid = futs[fut]
            try:
                fut.result()
                ok += 1
                print(f"[{ok}/{len(entries)}] {sid} re-rendered", flush=True)
            except Exception as exc:
                failed.append(sid)
                print(f"WARNING: {sid} failed: {exc}", flush=True)

    # Write back the manifest (entries gained determinism_clap), atomically.
    tmp = args.manifest.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    tmp.replace(args.manifest)
    dets = [e["determinism_clap"] for e in entries if "determinism_clap" in e]
    if dets:
        print(f"determinism_clap: min={min(dets):.3f} mean={sum(dets)/len(dets):.3f} "
              f"(n={len(dets)}; final CLAP is only interpretable relative to this ceiling)")
    print(f"Done: {ok} re-rendered, {len(failed)} failed{': ' + ', '.join(failed) if failed else ''}")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
