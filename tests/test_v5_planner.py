"""maestro/v5 planner + code mistakes: the plan, run through the skill CLIs on the host, rebuilds the GT patch."""
import json
import os
import random
import subprocess

import pytest

from maestro.v5 import mistakes, planner as pl
from maestro.v5.agents.main import Main  # noqa: F401  (import check: the builder modules load)

INIT = json.loads((pl.SKILL_DIR / "data" / "init_preset.json").read_text())


def gt_preset(seed):
    from pathlib import Path

    from maestro.synth.preset_gen import ARCHETYPES, generate_preset
    from maestro.synth.wavetable_lib import load_wavetable_lib
    lib = load_wavetable_lib(Path(pl.SKILL_DIR.parents[2] / "data" / "wavetable_lib.json"))
    return generate_preset(ARCHETYPES[seed % len(ARCHETYPES)], random.Random(seed), wavetable_lib=lib)


def sh(cmd, cwd):
    env = dict(os.environ, VITAL_SKILL=str(pl.SKILL_DIR))
    return subprocess.run(["bash", "-c", cmd], cwd=cwd, env=env, capture_output=True, text=True)


@pytest.mark.parametrize("key, v", [("filter_1_cutoff", 81.389), ("env_1_release", 0.795), ("volume", 5473.04),
                                    ("filter_1_model", 5.0), ("osc_1_transpose", -12.0), ("osc_1_level", 0.577),
                                    ("lfo_1_tempo", 9.0), ("osc_1_unison_voices", 6.0), ("delay_feedback", -0.3)])
def test_fmt_value_roundtrip(key, v):
    assert pl.parsed(key, v) == pytest.approx(v, rel=2e-3, abs=2e-3)


@pytest.mark.parametrize("seed, rate", [(0, 0.0), (1, 0.0), (2, 0.3), (3, 0.3)])
def test_plan_rebuilds_gt(tmp_path, seed, rate):
    gt = gt_preset(seed)
    plan = pl.build_plan(gt, INIT, random.Random(seed), mistake_rate=rate)
    (tmp_path / "preset.json").write_text(json.dumps(INIT))
    cmds = [f"python3 {pl.S}/wavetable.py set preset.json --osc {n} '{pl.wt_ref(e)}'" for n, e in plan.wavetables]
    for r in plan.rounds:
        cmds += [e.command() for e in r.edits] + ([r.fix.command()] if r.fix else [])
        if rate == 0:
            assert not r.mistakes and r.fix is None
    for c in cmds:
        p = sh(c, tmp_path)
        assert p.returncode == 0 and "error" not in p.stdout + p.stderr, (c, p.stdout, p.stderr)
    got = json.loads((tmp_path / "preset.json").read_text())["settings"]
    s = gt["settings"]
    for k in plan.changed:
        tol = 15 if k == "volume" else 1e-3 * (1 + abs(s[k]))  # volume is typed in 0.1 dB steps (~15 raw)
        assert got[k] == pytest.approx(s[k], abs=tol), k
    live = lambda st: sorted((r["source"], r["destination"], round(r["amount"], 3))  # noqa: E731
                             for r in pl.vitallib.live_routes(st)
                             if pl.vitallib.why_dead(st, r["destination"]) is None)
    assert live(got) == live(s)
    for n, e in plan.wavetables:
        assert pl._wt_hash(got["wavetables"][n - 1]) == pl._wt_hash(s["wavetables"][n - 1])


def test_code_mistakes_fail_for_real(tmp_path):
    gt = gt_preset(4)
    plan = pl.build_plan(gt, INIT, random.Random(4), mistake_rate=0.0)
    (tmp_path / "preset.json").write_text(json.dumps(INIT))
    rng = random.Random(0)
    kinds = set()
    edits = [pl.Edit("wavetable", f"python3 {pl.S}/wavetable.py set preset.json --osc 1 wt012")]
    edits += [e for r in plan.rounds for e in r.edits]
    for e in edits * 3:
        m = mistakes.code_mistake(e, rng)
        if m is None:
            continue
        before = (tmp_path / "preset.json").read_text()
        p = sh(m.cmd, tmp_path)
        assert p.returncode != 0 or "error" in p.stdout + p.stderr, m
        if e.tool != "mod":  # set.py / wavetable.py never half-write
            assert (tmp_path / "preset.json").read_text() == before
        assert m.diagnosis
        kinds.add(m.kind)
        assert sh(e.command(), tmp_path).returncode == 0
    assert {"bad_key", "out_of_range", "bad_osc"} <= kinds
