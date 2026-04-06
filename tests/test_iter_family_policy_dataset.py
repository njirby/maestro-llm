from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_iter_family_policy_dataset import _aggregate_family_stats


def test_aggregate_family_stats_prefers_early_high_gain_families():
    rows = [
        {
            "sample_id": "a",
            "steps": [
                {
                    "best_family": "osc",
                    "best_gain": 0.20,
                    "candidates": [
                        {"family": "osc", "gain": 0.20},
                        {"family": "modulation", "gain": 0.05},
                    ],
                },
                {
                    "best_family": "filter1",
                    "best_gain": 0.10,
                    "candidates": [
                        {"family": "filter1", "gain": 0.10},
                        {"family": "modulation", "gain": 0.08},
                    ],
                },
                {
                    "best_family": "modulation",
                    "best_gain": 0.04,
                    "candidates": [
                        {"family": "modulation", "gain": 0.04},
                        {"family": "osc", "gain": 0.01},
                    ],
                },
            ],
        }
    ]

    summary = _aggregate_family_stats(rows)
    fam = summary["family_stats"]
    assert fam["osc"]["avg_gain_by_bucket"]["early"] > fam["modulation"]["avg_gain_by_bucket"]["early"]
    assert summary["recommended_order"][0] in {"osc", "filter1"}
    assert fam["modulation"]["win_count"] == 1

