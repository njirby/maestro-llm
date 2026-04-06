from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_iter_sft_dataset import (
    _format_remaining_delta_context,
    _prepare_delta_maps,
)


def _sample_path_data() -> dict:
    return {
        "iterations": [
            {
                "step": 1,
                "params_changed": {"osc_1_level": 0.2, "filter_1_cutoff": 0.9},
                "params_applied": {"osc_1_level": 0.4},
                "params_delta": [
                    {
                        "name": "osc_1_level",
                        "from_norm": 0.7,
                        "to_norm": 0.4,
                        "target_norm": 0.2,
                        "mistake": False,
                    }
                ],
            },
            {
                "step": 2,
                "params_changed": {"filter_1_cutoff": 0.9},
                "params_applied": {"filter_1_cutoff": 0.85},
                "params_delta": [
                    {
                        "name": "filter_1_cutoff",
                        "from_norm": 0.5,
                        "to_norm": 0.85,
                        "target_norm": 0.9,
                        "mistake": False,
                    }
                ],
            },
        ]
    }


def test_prepare_delta_maps_collects_targets_and_initials():
    maps = _prepare_delta_maps(_sample_path_data())
    assert maps["target_norm"]["osc_1_level"] == 0.2
    assert maps["target_norm"]["filter_1_cutoff"] == 0.9
    assert maps["initial_norm"]["osc_1_level"] == 0.7
    assert maps["initial_norm"]["filter_1_cutoff"] == 0.5
    assert len(maps["applied_per_step"]) == 2


def test_remaining_delta_context_step1_shows_both_unresolved_params():
    maps = _prepare_delta_maps(_sample_path_data())
    ctx = _format_remaining_delta_context(step_num=1, delta_maps=maps)
    assert "before step 1" in ctx
    assert "2/2 tracked changed parameters still differ" in ctx
    assert "Oscillator 1 Level" in ctx
    assert "Filter 1 Cutoff" in ctx


def test_remaining_delta_context_step2_reflects_prior_progress():
    maps = _prepare_delta_maps(_sample_path_data())
    ctx = _format_remaining_delta_context(step_num=2, delta_maps=maps)
    assert "before step 2" in ctx
    # After step 1, osc_1_level moved from 0.7 -> 0.4 toward target 0.2.
    assert "[0.40→0.20]" in ctx
