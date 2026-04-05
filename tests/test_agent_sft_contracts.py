from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.agent_sft_common import (
    build_disjoint_shards,
    build_oracle_mix_candidates,
    validate_ms_swift_multiturn_record,
)
from scripts.build_judge_agent_sft import _build_rank_labels


def test_build_oracle_mix_candidates_keeps_gt_in_pool():
    out = build_oracle_mix_candidates(
        gt_names=["GT_A", "GT_B"],
        universe_names=["GT_A", "GT_B", "N1", "N2", "N3", "N4", "N5", "N6"],
        mix_size=8,
        rng=__import__("random").Random(7),
        hard_negative_names=["N1", "N2", "N3", "N4", "N5", "N6"],
    )
    assert "GT_A" in out and "GT_B" in out
    assert len(out) == 8


def test_build_disjoint_shards_no_duplication():
    candidates = [f"C{i}" for i in range(10)]
    shards = build_disjoint_shards(candidates, 3)
    flat = [x for s in shards for x in s]
    assert sorted(flat) == sorted(candidates)
    assert len(flat) == len(set(flat))


def test_build_rank_labels_puts_gt_first():
    ordered = ["A", "B", "C", "D"]
    gt_names = {"C"}
    scores = {"A": 0.8, "B": 0.7, "C": 0.2, "D": 0.1}
    ranking, selected, gt_ids, score_map = _build_rank_labels(ordered, gt_names, scores, topk_select=2)
    assert ranking[0] == "C3"
    assert selected == ranking[:2]
    assert gt_ids == ["C3"]
    assert set(score_map.keys()) == {"C1", "C2", "C3", "C4"}


def test_ms_swift_validator_accepts_minimal_valid_record():
    row = {
        "id": "x",
        "messages": [
            {"role": "user", "content": "<audio>\nTarget clip."},
            {"role": "assistant", "content": "Done."},
        ],
        "audios": ["/tmp/a.wav"],
    }
    assert validate_ms_swift_multiturn_record(row) == []


def test_ms_swift_validator_rejects_bad_shapes():
    row = {
        "id": "x",
        "messages": [
            {"role": "user", "content": {"bad": "shape"}},
            {"role": "assistant", "content": "A"},
            {"role": "assistant", "content": "B"},
        ],
        "audios": [],
    }
    errors = validate_ms_swift_multiturn_record(row)
    assert any("content_not_string" in e for e in errors)
    assert any("duplicate_adjacent_role:assistant" in e for e in errors)
