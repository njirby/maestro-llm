from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.eval_wavetable_retrieval_qwen import (  # noqa: E402
    _build_oracle_mix_candidates,
    _parse_qwen_listwise,
    _parse_qwen_score,
    _select_probe_rows_by_name,
)


def test_select_probe_rows_prefers_lowest_frame_idx():
    rows = [
        {"wavetable_name": "A", "frame_idx": 3, "source_wavetable_idx": 5},
        {"wavetable_name": "A", "frame_idx": 1, "source_wavetable_idx": 5},
        {"wavetable_name": "B", "frame_idx": 2, "source_wavetable_idx": 7},
    ]
    out = _select_probe_rows_by_name(rows)
    assert out["A"]["frame_idx"] == 1
    assert out["B"]["frame_idx"] == 2


def test_parse_qwen_score_accepts_json():
    parsed = _parse_qwen_score('{"score": 0.73, "confidence": 0.88, "reason": "close harmonic profile"}')
    assert parsed.score == 0.73
    assert parsed.confidence == 0.88
    assert "harmonic" in parsed.reason


def test_parse_qwen_score_accepts_fenced_or_mixed_json():
    parsed = _parse_qwen_score("Sure. ```json\n{\"score\": 0.52, \"confidence\": 0.67, \"reason\": \"partial match\"}\n```")
    assert parsed.score == 0.52
    assert parsed.confidence == 0.67


def test_parse_qwen_score_fallback_numeric_percent():
    parsed = _parse_qwen_score("I would give this about 62 out of 100")
    assert parsed.score == 0.62
    assert parsed.confidence == 0.0


def test_parse_qwen_listwise_accepts_json_ranking_and_scores():
    text = """```json
{"ranking":["C3","C1","C2"],"scores":{"C3":0.9,"C1":0.7,"C2":0.4},"reason":"clear spectral match"}
```"""
    ranking, scores, reason = _parse_qwen_listwise(text, ["C1", "C2", "C3"])
    assert ranking == ["C3", "C1", "C2"]
    assert scores["C3"] == 0.9
    assert "spectral" in reason


def test_build_oracle_mix_candidates_includes_gt():
    cands = _build_oracle_mix_candidates(
        gt_names=["GT_A"],
        universe_names=["GT_A", "B", "C", "D", "E", "F", "G", "H"],
        mix_size=8,
        rng=__import__("random").Random(1),
        hard_negative_names=["B", "C", "D", "E", "F", "G", "H"],
    )
    assert "GT_A" in cands
    assert len(cands) == 8


def test_build_oracle_mix_candidates_includes_all_gt_when_fit():
    cands = _build_oracle_mix_candidates(
        gt_names=["GT_A", "GT_B", "GT_C"],
        universe_names=["GT_A", "GT_B", "GT_C", "D", "E", "F", "G", "H"],
        mix_size=8,
        rng=__import__("random").Random(2),
        hard_negative_names=["D", "E", "F", "G", "H"],
    )
    assert "GT_A" in cands and "GT_B" in cands and "GT_C" in cands
    assert len(cands) == 8
