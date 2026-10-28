"""maestro/v5 injected mistakes beyond the planner's: render-call mistakes, transcriber retries / residual errors, and
starting patches other than init (the plan from a starting patch, run through the skill CLIs, rebuilds the GT)."""
import json
import random
import types

import pytest

from maestro import notation
from maestro.v5 import gate, mistakes, planner as pl, start
from maestro.v5.agents import main as mainmod
from test_v5_planner import INIT, gt_preset, sh

RENDER = "python3 $VITAL_SKILL/scripts/render.py preset.json notes.json r03_filter.wav"
NOTES = {"bpm": 120, "time_signature": "4/4", "notes": [
    {"pitch": 60 + i, "velocity": 100, "bar": 1 + i // 4, "beat": 1 + i % 4, "duration": "quarter"} for i in range(8)]}


def test_render_mistakes_fail_before_rendering(tmp_path):
    """Each kind errors out on its arguments, before Vital is touched (render.py checks paths first; the missing-
    output and wrong-extension kinds fail in argparse / the .wav check)."""
    (tmp_path / "preset.json").write_text(json.dumps(INIT))
    (tmp_path / "notes.json").write_text(json.dumps(NOTES))
    rng, kinds = random.Random(0), set()
    for _ in range(40):
        m = mistakes.render_mistake(RENDER, rng)
        assert m is not None and m.diagnosis and m.cmd != RENDER
        kinds.add(m.kind)
        if m.kind == "render_ext":   # on the host the VST3 check comes first; in the session the .wav check fails
            assert "r03_filter.wav" not in m.cmd
            continue
        p = sh(m.cmd, tmp_path)
        assert p.returncode != 0 and ("error" in (p.stdout + p.stderr).lower()), (m, p.stderr)
        assert not (tmp_path / "r03_filter.wav").exists()
    assert kinds == {"render_swapped", "render_typo", "render_no_out", "render_ext"}


def test_render_typo_keeps_directory():
    rng = random.Random(1)
    for _ in range(30):
        m = mistakes.render_mistake(
            "python3 $VITAL_SKILL/scripts/render.py $VITAL_SKILL/data/init_preset.json notes.json t.wav", rng)
        if m.kind == "render_typo":
            assert "$VITAL_SKILL/scripts/render.py" in m.cmd
            assert "$VITAL_SKILL/data/" in m.cmd or "notes.json" not in m.cmd


def test_mutate_transcription_n_and_kinds():
    beat = notation.to_beat_form(120, notation.to_seconds(json.dumps(NOTES))[0])["notes"]
    rng = random.Random(0)
    for _ in range(30):
        mut = notation.mutate_transcription(beat, rng, n_muts=1, kinds=("pitch_shift", "timing_shift"))
        if mut is None:
            continue
        wrong, infos, narr = mut
        assert len(infos) == 1 and infos[0]["type"] in ("pitch_shift", "timing_shift") and narr
        assert len(wrong) == len(beat)


def test_residual_error_keeps_note_order():
    fake = types.SimpleNamespace(s=types.SimpleNamespace(bpm=120.0), rng=random.Random(3),
                                 notes=notation.to_seconds(json.dumps(NOTES))[0])
    gt_b = notation.to_beat_form(120, fake.notes)["notes"]
    seen = set()
    for _ in range(40):
        r = mainmod.Main.residual_error(fake)
        assert r is not None
        wrong, (info,), _ = r
        seen.add(info["type"])
        slots = [notation._slot_of(n) for n in wrong]
        assert slots == sorted(slots)
        i = info["note_idx"]
        if info["type"] == "timing_shift":
            assert slots[i] - notation._slot_of(gt_b[i]) == info["shift_slots"] != 0
        else:
            assert wrong[i]["pitch"] - gt_b[i]["pitch"] == info["delta"]
        assert mainmod.residual_problem(info).startswith("the ")
    assert seen == {"pitch_shift", "timing_shift"}


@pytest.mark.parametrize("kind, seed", [(k, s) for k in start.KINDS for s in (0, 1, 2, 5)])
def test_plan_from_start_patch_rebuilds_gt(tmp_path, kind, seed):
    gt = gt_preset(seed)
    rng = random.Random(seed)
    st = start.start_patch(gt, INIT, rng, kind)
    assert not gate.structural(gt, st)["pass"]
    plan = pl.build_plan(gt, st, rng, mistake_rate=0.1)
    (tmp_path / "preset.json").write_text(json.dumps(st))
    cmds = [f"python3 {pl.S}/wavetable.py set preset.json --osc {n} '{pl.wt_ref(e)}'" for n, e in plan.wavetables]
    for r in plan.rounds:
        cmds += [e.command() for e in r.edits] + ([r.fix.command()] if r.fix else [])
    for c in cmds:
        p = sh(c, tmp_path)
        assert p.returncode == 0 and "error" not in p.stdout + p.stderr, (c, p.stdout, p.stderr)
    g = gate.structural(gt, json.loads((tmp_path / "preset.json").read_text()))
    assert g["pass"], g


def test_init_start_plan_unchanged():
    """No starting patch: no route removals (init has no routes), so plans are what they were."""
    assert pl.start_routes(INIT) == []
    plan = pl.build_plan(gt_preset(4), INIT, random.Random(4), mistake_rate=0.0)
    assert all(" remove " not in e.cmd for r in plan.rounds for e in r.edits)
