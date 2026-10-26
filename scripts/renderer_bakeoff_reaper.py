#!/usr/bin/env python3
"""Bake-off candidate (a): REAPER chunk path, via daw-farm DockerSession.

Renders the 8 bake-off probes through the exact production rollout path
(live REAPER + Vital VST3 chunk load + offline render action 42230), then
reports the correctness gate and per-render timings.

Usage:
    python scripts/renderer_bakeoff_reaper.py correctness [--container NAME]
    python scripts/renderer_bakeoff_reaper.py throughput  [--container NAME] [--n 16]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from maestro.reaper.dawfarm import (  # noqa: E402
    DockerSession,
    apply_vital_preset,
    create_vital_track,
    insert_midi_notes,
    reset_project,
    set_project_tempo,
    ROLLOUT_ROOT,
    _REAPER_RENDER_PY,
)
from scripts.renderer_bakeoff_common import (  # noqa: E402
    MELODY,
    analyze_wavs,
    build_probe_set,
    verdict,
)

OUT_DIR = REPO / "outputs" / "renderer_bakeoff" / "reaper"


def render_one(session: DockerSession, preset: dict, notes: list[dict],
               host_out: Path, tag: str = "_bakeoff") -> dict[str, float]:
    """render_preset_in_reaper, instrumented per stage so we can see where
    the per-render cost actually goes."""
    timings: dict[str, float] = {}

    def stage(name, fn):
        t0 = time.perf_counter()
        fn()
        timings[name] = time.perf_counter() - t0

    stage("reset", lambda: reset_project(session))
    bpm = 60.0 * float(preset.get("settings", {}).get("beats_per_minute", 2.0))
    stage("tempo", lambda: set_project_tempo(session, bpm))
    stage("track", lambda: create_vital_track(session))
    stage("preset", lambda: apply_vital_preset(session, preset))
    stage("notes", lambda: insert_midi_notes(session, notes))

    container_out = f"{ROLLOUT_ROOT}/{tag}/{host_out.name}"

    def _render():
        res = session.exec_python(
            _REAPER_RENDER_PY.format(out_path=container_out), timeout=300)
        if not res.ok:
            raise RuntimeError(f"render failed: {res.stderr[-1500:]}")

    stage("render", _render)
    stage("fetch", lambda: session.get(container_out, host_out))
    timings["total"] = sum(v for k, v in timings.items() if k != "total")
    return timings


def cmd_correctness(args) -> int:
    session = DockerSession(args.container)
    if not session.healthy():
        print(f"container {args.container} not healthy", file=sys.stderr)
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    probes = build_probe_set()
    paths, all_timings = [], []
    for i, (safe, preset) in enumerate(probes, 1):
        out = OUT_DIR / f"{i:02d}_{safe}.wav"
        t = render_one(session, preset, MELODY, out)
        all_timings.append(t)
        paths.append(out)
        print(f"[{i}/{len(probes)}] {safe:32s} {t['total']:6.2f}s "
              f"(preset={t['preset']:.2f} render={t['render']:.2f})", flush=True)

    stats = analyze_wavs(paths)
    passed, why = verdict(stats)
    for r, (safe, _) in zip(stats["rows"], probes):
        print(f"  {safe:32s} centroid={r['centroid_hz']:8.1f}Hz "
              f"rms={r['rms']:.4f} dur={r['duration_s']}s")
    print(f"\nGATE: {'PASS' if passed else 'FAIL'} — {why}")
    mean_total = sum(t["total"] for t in all_timings) / len(all_timings)
    print(f"mean sec/render (correctness pass): {mean_total:.2f}")

    (OUT_DIR / "stats.json").write_text(json.dumps(
        {"stats": stats, "passed": passed, "why": why,
         "timings": all_timings,
         "labels": [s for s, _ in probes]}, indent=2))
    return 0 if passed else 2


def cmd_throughput(args) -> int:
    session = DockerSession(args.container)
    if not session.healthy():
        print(f"container {args.container} not healthy", file=sys.stderr)
        return 1
    tmp = OUT_DIR / "throughput"
    tmp.mkdir(parents=True, exist_ok=True)

    probes = build_probe_set()
    timings = []
    t_wall = time.perf_counter()
    for i in range(args.n):
        safe, preset = probes[i % len(probes)]
        out = tmp / f"t{i:02d}_{safe}.wav"
        t = render_one(session, preset, MELODY, out)
        timings.append(t)
        print(f"[{i+1}/{args.n}] {t['total']:6.2f}s", flush=True)
    wall = time.perf_counter() - t_wall

    keys = ["reset", "tempo", "track", "preset", "notes", "render", "fetch", "total"]
    print("\nmean per-stage seconds:")
    for k in keys:
        print(f"  {k:8s} {sum(t[k] for t in timings)/len(timings):6.3f}")
    per = wall / args.n
    print(f"\nwall={wall:.1f}s over {args.n} renders -> {per:.2f} s/render")
    print(f"probes/hour, 1 container: {3600/per:.0f}")
    print(f"probes/hour, 20 containers: {20*3600/per:.0f}")

    (tmp / "throughput.json").write_text(json.dumps(
        {"n": args.n, "wall_s": wall, "sec_per_render": per,
         "timings": timings}, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["correctness", "throughput"])
    ap.add_argument("--container", default="renderer-bakeoff")
    ap.add_argument("--n", type=int, default=16)
    args = ap.parse_args()
    return cmd_correctness(args) if args.cmd == "correctness" else cmd_throughput(args)


if __name__ == "__main__":
    raise SystemExit(main())
