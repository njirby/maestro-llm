from __future__ import annotations

import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from maestro.synth.path_gen import (
    MAX_ITERATIONS,
    MIN_ITERATIONS,
    _param_family,
    generate_preset_path,
)


def test_iteration_count_varies_across_samples():
    rng = random.Random(1234)
    counts = []
    for i in range(30):
        path = generate_preset_path("lead", rng, output_dir=None, sample_id=f"lead_test_{i}")
        counts.append(path["n_iterations"])
    assert len(set(counts)) > 1
    assert all(MIN_ITERATIONS <= c <= MAX_ITERATIONS for c in counts)


def test_no_empty_iteration_steps_when_params_changed():
    rng = random.Random(2026)
    path = generate_preset_path("bass", rng, output_dir=None, sample_id="bass_test_nonempty")
    assert path["n_changed_params"] >= path["n_iterations"]
    for step in path["iterations"]:
        assert step["params_applied"], f"step {step['step']} should not be empty"


def test_step_planner_metadata_and_family_purity():
    rng = random.Random(7)
    path = generate_preset_path("pad", rng, output_dir=None, sample_id="pad_test_planner")
    for step in path["iterations"]:
        names = list(step["params_applied"].keys())
        assert step.get("primary_family"), "missing primary_family"
        assert "planner_stage" in step
        assert isinstance(step.get("intent_tags"), list)
        assert isinstance(step.get("planned_param_names"), list)
        assert isinstance(step.get("planned_primary_names"), list)
        assert isinstance(step.get("allowed_primary_controls"), list)
        assert isinstance(step.get("allowed_support_controls"), list)
        assert isinstance(step.get("intended_edit_controls"), list)
        assert step.get("search_scope_type") in {"family", "mod_slot"}
        assert isinstance(step.get("checkpoint_revisit"), bool)
        assert step.get("search_keyword"), "missing search_keyword"
        if not names:
            continue
        primary = step["primary_family"]
        primary_share = sum(_param_family(n) == primary for n in names) / len(names)
        assert primary_share >= 0.70
        if primary == "modulation":
            assert re.fullmatch(r"modulation \d+", step["search_keyword"]), step["search_keyword"]
            assert step.get("search_scope_type") == "mod_slot"
            slot = step["search_keyword"].split()[-1]
            for name in step.get("allowed_primary_controls", []):
                assert name.startswith(f"modulation_{slot}_"), name
