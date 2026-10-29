"""maestro/v5/wt_rank: the metric and the "sounds identical in context" rule (pure functions + the gate's use of it)."""
import copy
import json

import numpy as np

from maestro.v5 import gate, planner as pl, wt_rank as wr

SR = wr.SR
INIT = json.loads((pl.SKILL_DIR / "data" / "init_preset.json").read_text())


def tone(f0, amps, seconds=6.0):
    t = np.arange(int(seconds * SR)) / SR
    y = sum(a * np.sin(2 * np.pi * f0 * (h + 1) * t) for h, a in enumerate(amps))
    return np.stack([y, y], 1).astype(np.float32)


def frames(y):
    return wr.sustain_frames(y)


def test_identical_is_zero_and_same():
    y = tone(220, [1, 0.5, 0.25])
    d = float(wr.distance(frames(y), frames(y)))
    assert d == 0.0 and wr.sounds_like_target(d)


def test_louder_missing_layer_costs_more():
    """Dropping a loud layer must cost more than dropping a quiet one (the check the unweighted metric failed)."""
    base, loud, quiet = tone(220, [1, 0.5]), tone(330, [0.8, 0.4]), tone(330, [0.02, 0.01])
    T_loud, T_quiet = frames(base + loud), frames(base + quiet)
    d_loud = float(wr.distance(T_loud, frames(base)))
    d_quiet = float(wr.distance(T_quiet, frames(base)))
    assert d_loud > d_quiet > 0
    assert not wr.sounds_like_target(d_loud)


def test_distance_vectorizes_over_candidates():
    T = frames(tone(220, [1, 0.5]))
    S = np.stack([T, frames(tone(220, [1, 0.6])), frames(tone(247, [1, 0.5]))])
    d = wr.distance(T, S)
    assert d.shape == (3,) and d[0] == 0 and d[1] < d[2]


def test_dupe_list_loads():
    assert wr.same_table("wt004", "wt028") and not wr.same_table("wt004", "wt005") and wr.same_table(None, None)


class FakeRenderer:
    """Renders 'gt' as one tone and 'alt' as another, so the gate's in-context check sees what we choose."""

    def __init__(self, same: bool):
        self.same = same

    def __call__(self, presets, notes):
        y = tone(220, [1, 0.5])
        return {k: (y if k == "gt" or self.same else tone(220, [1, 0.9])) for k in presets}


def _gt_final():
    gt = copy.deepcopy(INIT)
    gt["settings"]["osc_1_on"] = 1.0
    final = copy.deepcopy(gt)
    final["settings"]["wavetables"][0] = {"groups": [{"different": True}]}
    return gt, final


def test_same_in_context_clears_wavetable_diff():
    gt, final = _gt_final()
    st = gate.structural(gt, final)
    assert st["wavetable_diffs"] == [1]
    assert gate.same_in_context(FakeRenderer(same=True), gt, final, [1]) == [1]
    assert gate.same_in_context(FakeRenderer(same=False), gt, final, [1]) == []


def test_structural_ignores_inaudible_keys():
    gt, _ = _gt_final()
    final = copy.deepcopy(gt)
    gt["settings"]["polyphony"] = 4.0
    final["settings"]["polyphony"] = 8.0
    assert not gate.structural(gt, final)["pass"]
    assert gate.structural(gt, final, ignore={"polyphony"})["pass"]


def test_build_plan_edits_only_audible_keys():
    import random
    from test_v5_planner import gt_preset
    gt = gt_preset(3)
    keys = pl.live_changed_keys(gt, INIT)
    keep = set(keys[::2])
    plan = pl.build_plan(gt, INIT, random.Random(3), mistake_rate=0.0, audible=keep)
    edited = {k for r in plan.rounds for e in r.edits if e.tool == "set" for k in e.values}
    assert edited <= keep and plan.changed == [k for k in keys if k in keep]


class CountingRenderer:
    """Renders 'gt' as a tone; a key revert is audible only for keys in `loud` (a different tone)."""

    def __init__(self, loud):
        self.loud = loud

    def render(self, presets, notes):
        base = tone(220, [1, 0.5])
        out = {}
        for k in presets:
            out[k] = tone(220, [1, 0.9]) if k in {f"k/{x}" for x in self.loud} else base
        return out


def test_audible_changes_splits_keys():
    gt, _ = _gt_final()
    keys = ["filter_1_cutoff", "polyphony", "osc_1_unison_blend"]
    a = wr.audible_changes(CountingRenderer({"filter_1_cutoff"}), gt, gt, keys, {"notes": []})
    assert a["audible"] == ["filter_1_cutoff"] and set(a["inaudible"]) == {"polyphony", "osc_1_unison_blend"}
