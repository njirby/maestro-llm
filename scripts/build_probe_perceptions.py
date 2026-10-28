"""Verified listening notes for the clips subagents hear: each searcher probe and judge combination against the target.

Uses sounddiff's target-vs-current builder with the ground-truth preset as the target and the probe preset (from the
replay workdir) as the current patch. Trajectory lines don't apply to a probe and are dropped. Renders run in one
--network none container.

  .venv/bin/python scripts/build_probe_perceptions.py s0001 s0005 ...
  -> outputs/thinking_capture/hearing_lab/probe_perceptions/<sid>.json  {"probes/s1/wt047.wav": "- ...\n- ...", ...}
"""
from __future__ import annotations

import os

# one thread per process: numpy's thread pools on a 48-core box spent ~70 CPU-s per probe fighting each other
# (1.9 s single-threaded); parallelism comes from one process per sample instead (scripts/run_probe_perceptions.sh)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import concurrent.futures as cf
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maestro.sounddiff import delta  # noqa: E402
from maestro.sounddiff import verified as V  # noqa: E402

_state_cache: dict = {}
_build_state = V.build_state


def _cached_build_state(preset, notes, wavs):
    """The target's state is the same for every probe of a sample: analyze it once (renders are cached, so the
    same preset + notes always gives the same wavs)."""
    key = json.dumps(preset.get("settings", preset), sort_keys=True) + json.dumps(notes, sort_keys=True)
    if key not in _state_cache:
        _state_cache.clear() if len(_state_cache) > 8 else None
        _state_cache[key] = _build_state(preset, notes, wavs)
    return _state_cache[key]


V.build_state = delta.build_state = _cached_build_state
from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402


REPLAY = Path(os.environ.get("REPLAY_DIR", ROOT / "outputs/replay/v5_pilot20c"))
ORACLE = Path(os.environ.get("RUN_DIR", ROOT / "outputs/v5_rollouts/pilot20c")) / "oracle"
OUT = Path(os.environ.get("PROBES_DIR", ROOT / "outputs/thinking_capture/hearing_lab/probe_perceptions"))
SKIP = ("- Your last edit", "- Did it get you closer", "- Next:")


def prewarm(r, gt: Path, probes: list[Path], notes: dict) -> None:
    """Render every probe's verification jobs (and the target's) in ONE batch, so the per-probe notes below only hit
    the cache: per-probe calls carry ~2 renders each and left the render pool idle; one call also can't race itself."""
    from maestro.sounddiff.verified import render_state_jobs
    jobs = {}
    for i, path in enumerate([gt] + probes):
        for k, q in render_state_jobs(json.loads(path.read_text())).items():
            jobs[f"{i}/{k}"] = q   # the '#2' take suffix is kept, so cache keys match delta's own calls
    r.render(jobs, notes)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with ContainerRenderer(OUT / "render_cache", jobs=int(os.environ.get("RENDER_JOBS", 24))) as r:
        r._session()  # start the container once, before worker threads race to start it
        for sid in sys.argv[1:]:
            dst = OUT / f"{sid}.json"
            done = json.loads(dst.read_text()) if dst.exists() else {}
            wd = REPLAY / sid / "rollouts" / sid
            gt, notes = ORACLE / sid / "gt_preset.json", json.loads((ORACLE / sid / "notes.json").read_text())
            probes = sorted(wd.glob("probes/*/*.json")) + sorted(wd.glob("judge/*.json"))
            todo = [p for p in probes if str(p.with_suffix(".wav").relative_to(wd)) not in done]
            if todo:
                prewarm(r, gt, todo, notes)

            def one(p):
                n = delta.notes_target_vs_current(gt, p, p, notes, renderer=r)
                return str(p.with_suffix(".wav").relative_to(wd)), "\n".join(
                    l for l in n["text"].splitlines() if l.startswith("- ") and not l.startswith(SKIP))

            with cf.ThreadPoolExecutor(1) as ex:  # identical presets across threads race on the input files
                for k, txt in ex.map(one, todo):
                    done[k] = txt
                    dst.write_text(json.dumps(done, indent=1))
                    print(sid, k, txt.count("\n") + 1, "lines", flush=True)


if __name__ == "__main__":
    main()
