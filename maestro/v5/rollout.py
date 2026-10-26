"""One sample end to end in one DawDreamer session: workdir setup, the main agent (subagents included), outputs.

Workdir at start (what the agents see): target.wav only. The user message names it (USER_PROMPT); see
agents/main.py and the "target audio entry" note in scripts/build_rollouts_v5.py.
"""
from __future__ import annotations

import random
import shutil
from pathlib import Path

from maestro.dawfarm_dd import DawDreamerSession

from .agents.main import Main
from .audio import peak_dbfs
from .ctx import Ctx
from .inputs import Sample

SILENCE_DBFS = -60.0


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
    shutil.copytree(sess.host_path(wd), init_dir)
    out = Main(ctx, **main_kw).run()
    out["events"] = ctx.events
    out["init_dir"] = str(init_dir.resolve())
    return out
