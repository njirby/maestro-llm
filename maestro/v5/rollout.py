"""One sample end to end in one DawDreamer session: workdir setup, the main agent (subagents included), outputs.

Workdir at start (what the agents see): target.wav only. The user message names it (USER_PROMPT); see
agents/main.py and the "target audio entry" note in scripts/build_rollouts_v5.py.
"""
from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

from maestro.dawfarm_dd import DawDreamerSession

from .agents.main import Main
from .audio import peak_dbfs
from .ctx import Ctx
from .inputs import Sample
from .start import KINDS as START_KINDS, START_FILE, start_patch

SILENCE_DBFS = -60.0
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")


def snapshot_workdir(workdir: Path, oracle_dir: Path) -> Path:
    """The agents' workdir as the rollout leaves it (the pool purges it at the next rollout), host-side for WS-D's
    final-state check: oracle/<sid>/final_workdir.json lists every file (path relative to the workdir, size, sha256,
    audio flag), and oracle/<sid>/final_workdir/ keeps full copies of the non-audio files."""
    import hashlib
    files, copy_root = [], oracle_dir / "final_workdir"
    if copy_root.exists():
        shutil.rmtree(copy_root)
    for f in sorted(p for p in workdir.rglob("*") if p.is_file()):
        rel = str(f.relative_to(workdir))
        audio = f.suffix.lower() in AUDIO_EXT
        files.append({"path": rel, "size": f.stat().st_size, "sha256": hashlib.sha256(f.read_bytes()).hexdigest(),
                      "audio": audio})
        if not audio:
            (copy_root / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(f, copy_root / rel)
    out = oracle_dir / "final_workdir.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"workdir": str(workdir), "files": files}, indent=1))
    return out


class SkipSample(RuntimeError):
    pass


def run_one(sess: DawDreamerSession, sample: Sample, run_dir: Path, seed: int = 0, **main_kw) -> dict:
    """Returns {"main": record, "transcriber": [..], "searcher": [..], "judge": [..], "links": [..], "events": [..]}."""
    wd = sess.begin_rollout(sample.id)
    ctx = Ctx(sess=sess, sample=sample, rng=random.Random(f"{seed}/{sample.id}"), run_dir=run_dir, wd=wd)
    if sample.target_wav_path:
        ctx.put(sample.target_wav_path, "target.wav")
    else:
        tgt = ctx.oracle_render(sample.gt_preset(), "target")
        ctx.put(tgt, "target.wav")
    peak = peak_dbfs(ctx.host("target.wav"))
    if peak < SILENCE_DBFS:
        raise SkipSample(f"{sample.id}: target peak {peak:.1f} dBFS")
    # the workdir as the first step finds it (target.wav only): WS-D's replay --init-dir
    init_dir = run_dir / "init" / sample.id
    if init_dir.exists():
        shutil.rmtree(init_dir)
    # a starting patch from the user (start_patch_rate): written before the first step, so it is in the init dir
    start = None
    start_rate = main_kw.pop("start_patch_rate", 0.0)
    if start_rate and ctx.rng.random() < start_rate:
        kind = ctx.rng.choice(START_KINDS)
        start = start_patch(sample.gt_preset(), sample.init_preset(), ctx.rng, kind, sample.archetype)
        ctx.host(START_FILE).write_text(json.dumps(start, separators=(",", ":")))
        ctx.note(agent="maestro", event="start_patch", kind=kind)
    shutil.copytree(sess.host_path(wd), init_dir)
    out = Main(ctx, start=start, **main_kw).run()
    out["events"] = ctx.events
    out["final_workdir"] = str(snapshot_workdir(sess.host_path(wd), ctx.oracle_host_dir))
    out["init_dir"] = str(init_dir.resolve())
    return out
