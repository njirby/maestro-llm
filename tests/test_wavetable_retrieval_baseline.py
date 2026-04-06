from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_wavetable_retrieval_baseline import (  # noqa: E402
    _collapse_row_scores_to_name_scores,
    _compute_recall_hits,
    _extract_gt_wavetable_names_from_preset_dict,
    _resolve_target_preset_path,
)


def test_extract_gt_wavetable_names_uses_active_oscillators_only():
    preset = {
        "settings": {
            "osc_1_on": 1.0,
            "osc_2_on": 0.0,
            "osc_3_on": 1.0,
            "wavetables": [
                {"name": "A"},
                {"name": "B"},
                {"name": "C"},
            ],
        }
    }
    assert _extract_gt_wavetable_names_from_preset_dict(preset) == ["A", "C"]


def test_collapse_row_scores_uses_max_score_per_name():
    scores = np.array([0.2, 0.9, 0.3, 0.6], dtype=np.float32)
    rows = [
        {"wavetable_name": "A"},
        {"wavetable_name": "A"},
        {"wavetable_name": "B"},
        {"wavetable_name": "B"},
    ]
    out = _collapse_row_scores_to_name_scores(scores, rows)
    assert out["A"] == pytest.approx(0.9)
    assert out["B"] == pytest.approx(0.6)


def test_compute_recall_hits_any_gt_in_topk():
    gt = ["A", "C"]
    ranked = ["B", "A", "D"]
    hits = _compute_recall_hits(gt, ranked, [1, 2, 10])
    assert hits["r@1"] == 0
    assert hits["r@2"] == 1
    assert hits["r@10"] == 1


def test_resolve_target_preset_path_from_path_file_json(tmp_path: Path):
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("")

    paths_dir = tmp_path / "paths"
    paths_dir.mkdir(parents=True)
    target = paths_dir / "bass_abc_target.vital"
    target.write_text("{}")

    path_file = paths_dir / "bass_abc.json"
    path_file.write_text(json.dumps({"target_preset_path": str(target)}))

    row = {
        "sample_id": "bass_abc",
        "path_file": str(path_file),
    }
    resolved = _resolve_target_preset_path(manifest, row)
    assert resolved == target
