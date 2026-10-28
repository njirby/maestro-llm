"""Starting patches other than the skill's init preset: the user hands the agent a patch to work from.

  start = start_patch(gt, init, rng, kind, archetype)
  partial   init with 1-3 of the GT's envelope / filter / fx / global sections already dialed in (the agent finds
            the rest; wavetables and modulation stay at init)
  foreign   a different patch of the same archetype (preset_gen): wrong in every section, routes and LFO shapes
            included, so the plan also removes routes and re-shapes LFOs (planner.build_plan with init=start)

The patch is written into the workdir as START_FILE before the rollout's first step; the user message names it.
"""
from __future__ import annotations

import copy
import random
from pathlib import Path

from . import planner as pl

START_FILE = "my_patch.json"
KINDS = ("partial", "foreign")
PARTIAL_SECTIONS = ("envelope", "filter", "fx", "global")


def start_patch(gt: dict, init: dict, rng: random.Random, kind: str, archetype: str | None = None) -> dict:
    if kind == "partial":
        out = copy.deepcopy(init)
        keys = pl.live_changed_keys(gt, init)
        secs = sorted({pl.section(k) for k in keys} & set(PARTIAL_SECTIONS))
        if not secs:
            return out
        pick = set(rng.sample(secs, rng.randint(1, min(3, len(secs)))))
        for k in keys:
            if pl.section(k) in pick:
                out["settings"][k] = gt["settings"][k]
        return out
    if kind == "foreign":
        from maestro.synth.preset_gen import ARCHETYPES, generate_preset
        from maestro.synth.wavetable_lib import load_wavetable_lib
        lib = load_wavetable_lib(Path(pl.SKILL_DIR.parents[2] / "data" / "wavetable_lib.json"))
        arch = archetype if archetype in ARCHETYPES else rng.choice(ARCHETYPES)
        out = generate_preset(arch, random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
        # an oscillator the GT leaves off keeps the GT's table: off, it is silent (the plan never touches it), and
        # the GT gate compares all three tables
        for n in (1, 2, 3):
            if float(gt["settings"].get(f"osc_{n}_on", 0)) <= 0.5:
                out["settings"]["wavetables"][n - 1] = copy.deepcopy(gt["settings"]["wavetables"][n - 1])
        return out
    raise ValueError(kind)
