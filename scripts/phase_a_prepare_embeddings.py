#!/usr/bin/env python3
"""Phase A of two-phase generation: pre-render + pre-embed everything.

Renders exactly the probe audio the Phase-B pilot (build_unified_sft_v4 in
daw-farm mode -> build_search_record_v3) will ask the CLAP embedder about,
at the IDENTICAL host paths (the embed cache is path-keyed), then embeds the
full file set with one ClapEmbedder per GPU and merges the shard caches into
a single --clap-cache npz. Phase B then runs with GPUs devoted to the 30B
generator and hits the cache for every planned embed; misses log loudly.

Per manifest sample this replicates the v3 search stage's renders:
  - per agent shard (compute_search_partition, same slice size):
      {out_dir}/search_probe_audio/{sid}_agent{N}/wt_{idx:04d}_{slug}.wav
      {out_dir}/search_probe_audio/{sid}_agent{N}_gt/wt_{k:04d}_{slug}.wav
  - notes json from the manifest's source_midi_path via the same loader the
    transcription builder uses, staged at /tmp/agents/{sid}/{sid}_notes.json
    in-container (probe renders play the transcribed melody).
Startup set embedded too: manifest gt/gt_probe/default wavs + the global
candidate-probe cache dir.

Round-2 re-search probes (paths keyed by later rounds, if any divergence)
are NOT pre-rendered — acceptable stragglers, they'll warn at generation.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.reaper import dawfarm as _dawfarm
from maestro.reaper.dawfarm import DawFarmPool, assert_clean, create_vital_track, reset_project
from scripts.agent_sft_common import (
    _wrap_as_bash,
    build_render_probes_snippet,
    extract_gt_wavetable_names,
    load_manifest_entries,
    load_wavetable_lib,
)
from scripts.build_search_agent_sft_v3 import MIN_CENTROID_SPREAD_HZ, audio_descriptors
from scripts.build_transcription_agent_sft_v4 import load_notes_from_midi
from scripts.build_unified_sft_v4 import compute_search_partition


def _slugify(s: str) -> str:
    return (re.sub(r"[^a-zA-Z0-9]+", "_", s).strip("_") or "unnamed")[:80]


def render_sample(session, entry: dict, *, name_to_idx, idx_to_name,
                  slice_starts, slice_size, total_named,
                  search_probe_dir: Path, recycle: bool,
                  vital_data: str | None) -> list[str]:
    """Render all shard + GT probes for one sample; return fetched host wavs."""
    sid = entry["sample_id"]
    if recycle and hasattr(session, "recycle"):
        session.recycle()
        _dawfarm.sync_vital_data(session, vital_data)
    assert_clean(session)
    reset_project(session)
    create_vital_track(session)

    notes = load_notes_from_midi(entry["source_midi_path"])
    notes_path = f"/tmp/agents/{sid}/{sid}_notes.json"
    payload = json.dumps({"notes": notes, "n_notes": len(notes)})
    r = session.exec_bash(
        f"mkdir -p /tmp/agents/{sid} && cat > {notes_path} <<'NJ'\n{payload}\nNJ")
    if not r.ok:
        raise RuntimeError(f"{sid}: notes staging failed: {r.stderr[-200:]}")

    gt_names = extract_gt_wavetable_names(Path(entry["target_preset_path"]))
    fetched: list[str] = []
    for ai, start in enumerate(slice_starts):
        end = min(start + slice_size, total_named)
        agent_idx = ai + 1
        shard = [idx_to_name[i] for i in range(start, end) if i in idx_to_name]
        idxs = [name_to_idx[n] for n in shard]

        cdir = f"/tmp/phase_a/{sid}_a{agent_idx}"
        host_dir = search_probe_dir / f"{sid}_agent{agent_idx}"
        expected = [host_dir / f"wt_{name_to_idx[n]:04d}_{_slugify(n)}.wav" for n in shard]
        if not all(p.exists() for p in expected):
            snip = build_render_probes_snippet(idxs=idxs, out_dir=cdir,
                                               midi_path=notes_path)
            rr = session.exec_bash("DISPLAY=:99 " + _wrap_as_bash(snip), timeout=900.0)
            if not rr.ok:
                raise RuntimeError(f"{sid} a{agent_idx}: render failed: {rr.stderr[-300:]}")
            host_dir.mkdir(parents=True, exist_ok=True)
            session.get_dir(cdir, host_dir)
            missing = [p for p in expected if not p.exists()]
            if missing:
                raise RuntimeError(f"{sid} a{agent_idx}: fetch missing {len(missing)}")
            cents = [audio_descriptors(p)[0]["centroid"] for p in expected]
            spread = max(cents) - min(cents)
            if spread < MIN_CENTROID_SPREAD_HZ:
                raise RuntimeError(
                    f"{sid} a{agent_idx}: DISCRIMINABILITY GATE FAILED "
                    f"({spread:.0f} Hz)")
        fetched.extend(str(p) for p in expected)

        gdir = f"/tmp/phase_a/{sid}_a{agent_idx}_gt"
        ghost = search_probe_dir / f"{sid}_agent{agent_idx}_gt"
        gexp = [ghost / f"wt_{k:04d}_{_slugify(n)}.wav" for k, n in enumerate(gt_names)]
        if not all(p.exists() for p in gexp):
            gsnip = build_render_probes_snippet(names=gt_names, out_dir=gdir,
                                                midi_path=notes_path)
            gr = session.exec_bash("DISPLAY=:99 " + _wrap_as_bash(gsnip), timeout=600.0)
            if not gr.ok:
                raise RuntimeError(f"{sid} a{agent_idx}: gt render failed: {gr.stderr[-300:]}")
            ghost.mkdir(parents=True, exist_ok=True)
            session.get_dir(gdir, ghost)
        fetched.extend(str(p) for p in gexp if p.exists())
        session.exec_bash(f"rm -rf {cdir} {gdir}")
    return fetched


def _embed_worker(device: str, files: list[str], shard_out: str) -> int:
    """Run in a spawned process: one ClapEmbedder on one GPU."""
    import sys as _s
    _s.path.insert(0, str(ROOT))
    from scripts.agent_sft_common import ClapEmbedder
    emb = ClapEmbedder.create(device)
    n = 0
    for f in files:
        try:
            emb.embed_audio_path(Path(f))
            n += 1
        except Exception as exc:
            print(f"[{device}] embed failed {f}: {exc}", file=_s.stderr, flush=True)
    emb.save_cache(Path(shard_out))
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-dir", type=Path, required=True,
                    help="the Phase-B pilot --out-dir (host probe paths key off it)")
    ap.add_argument("--clap-cache", type=Path, required=True)
    ap.add_argument("--daw-farm", default="docker")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--gpus", type=int, default=4)
    ap.add_argument("--candidates-per-slice", type=int, default=48)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--candidate-probe-dir", type=Path,
                    default=Path("outputs/agent_sft/candidate_probes"))
    ap.add_argument("--recycle-containers", type=lambda v: v.lower() != "false", default=True)
    ap.add_argument("--daw-farm-vital-data", default=None)
    ap.add_argument("--render-only", action="store_true")
    args = ap.parse_args()

    entries = load_manifest_entries(args.manifest, args.max_samples)
    lib = load_wavetable_lib(args.wavetable_lib)
    names_sorted = sorted({wt["name"] for wt in lib
                           if isinstance(wt, dict) and wt.get("name")})
    total_named = len(names_sorted)
    name_to_idx = {n: i for i, n in enumerate(names_sorted)}
    idx_to_name = {i: n for n, i in name_to_idx.items()}
    slice_starts = compute_search_partition(total_named, args.candidates_per_slice)
    search_probe_dir = args.out_dir / "search_probe_audio"
    print(f"Phase A: {len(entries)} samples x {len(slice_starts)} shards "
          f"({total_named} wavetables)", flush=True)

    # ---- render (container fleet) ----
    pool = DawFarmPool.from_spec(args.daw_farm)
    all_files: list[str] = []
    lock = threading.Lock()

    def _one(entry):
        with pool.acquire() as sess:
            files = render_sample(
                sess, entry, name_to_idx=name_to_idx, idx_to_name=idx_to_name,
                slice_starts=slice_starts, slice_size=args.candidates_per_slice,
                total_named=total_named, search_probe_dir=search_probe_dir,
                recycle=args.recycle_containers,
                vital_data=args.daw_farm_vital_data)
        with lock:
            all_files.extend(files)
        return entry["sample_id"], len(files)

    n_workers = min(args.workers, len(pool.sessions), len(entries))
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = [ex.submit(_one, e) for e in entries]
        for fut in as_completed(futs):
            sid, n = fut.result()
            print(f"  rendered {sid}: {n} probe files", flush=True)

    # ---- startup set ----
    for e in entries:
        for k in ("gt_wav", "gt_probe_wav", "default_wav"):
            if e.get(k) and Path(e[k]).exists():
                all_files.append(str(e[k]))
    if args.candidate_probe_dir.is_dir():
        all_files.extend(str(p) for p in sorted(args.candidate_probe_dir.glob("*.wav")))

    uniq = sorted({str(Path(f).resolve()) for f in all_files if Path(f).exists()})
    print(f"render phase complete: {len(uniq)} unique files to embed", flush=True)
    if args.render_only:
        return

    # ---- embed on N GPUs (spawned processes, shard caches, merge) ----
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    shards = [uniq[i::args.gpus] for i in range(args.gpus)]
    shard_paths = [str(args.clap_cache.with_suffix(f".shard{i}.npz"))
                   for i in range(args.gpus)]
    procs = []
    for i, (files, sp) in enumerate(zip(shards, shard_paths)):
        pr = ctx.Process(target=_embed_worker, args=(f"cuda:{i}", files, sp))
        pr.start()
        procs.append(pr)
    for pr in procs:
        pr.join()
        if pr.exitcode != 0:
            raise RuntimeError(f"embed worker exited {pr.exitcode}")

    merged: dict[str, np.ndarray] = {}
    if args.clap_cache.exists():
        data = np.load(str(args.clap_cache), allow_pickle=False)
        for i, k in enumerate(list(data["keys"])):
            if f"emb_{i}" in data:
                merged[str(k)] = data[f"emb_{i}"]
        print(f"merging into existing cache ({len(merged)} entries)", flush=True)
    for sp in shard_paths:
        if not Path(sp).exists():
            continue
        data = np.load(sp, allow_pickle=False)
        for i, k in enumerate(list(data["keys"])):
            if f"emb_{i}" in data:
                merged[str(k)] = data[f"emb_{i}"]
        Path(sp).unlink()
    args.clap_cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(args.clap_cache), keys=list(merged.keys()),
                        **{f"emb_{i}": v for i, v in enumerate(merged.values())})

    # ---- verification ----
    import random
    rng = random.Random(0)
    probe_files = [f for f in uniq if "/search_probe_audio/" in f]
    sample = rng.sample(probe_files, min(10, len(probe_files)))
    hits = sum(1 for f in sample if f in merged)
    size_mb = args.clap_cache.stat().st_size / 1e6
    print(json.dumps({
        "files_rendered_or_present": len(probe_files),
        "files_embedded_total": len(uniq),
        "cache_entries": len(merged),
        "cache_file_mb": round(size_mb, 1),
        "verification_hits": f"{hits}/{len(sample)}",
    }, indent=2), flush=True)
    if hits != len(sample):
        raise SystemExit("VERIFICATION FAILED: expected rollout paths missing from cache")


if __name__ == "__main__":
    main()
