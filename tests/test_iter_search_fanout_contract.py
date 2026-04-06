from __future__ import annotations

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_iter_sft_dataset import (
    _build_search_fanout_bundle,
    _build_search_handoff_interpretation,
    _build_search_queries,
)


def _sample_step_data() -> dict:
    return {
        "search_keyword": "oscillator 2",
        "search_result": "Oscillator 2 Level: 0.6123\nOscillator 2 Wave Frame: 0.2011",
        "params_delta": [
            {"name": "osc_2_level", "from_norm": 0.6123, "to_norm": 0.7100},
            {"name": "osc_2_wave_frame", "from_norm": 0.2011, "to_norm": 0.5800},
            {"name": "filter_1_cutoff", "from_norm": 0.3200, "to_norm": 0.6400},
        ],
    }


def test_build_search_queries_respects_fanout_and_dedupes():
    queries = _build_search_queries(
        primary_keyword="oscillator 2",
        primary_family="osc",
        support_family="filter1",
        fanout=3,
    )
    assert queries[0] == "oscillator 2"
    assert len(queries) == 3
    assert len(set(queries)) == len(queries)


def test_build_search_fanout_bundle_emits_structured_reports():
    payload = _build_search_fanout_bundle(
        step_data=_sample_step_data(),
        step_num=4,
        primary_keyword="oscillator 2",
        primary_family="osc",
        support_family="filter1",
        fanout=3,
        bad_result_prob=0.0,
        bad_result_max=1,
        rng=random.Random(7),
    )
    bundle = payload["bundle"]
    assert len(payload["reports"]) == 3
    assert bundle["step"] == 4
    assert bundle["primary_query"] == "oscillator 2"
    assert "reports" in bundle and len(bundle["reports"]) == 3
    assert "consensus_controls" in bundle
    assert bundle["quality_summary"]["bad_injected"] == 0


def test_build_search_fanout_bundle_can_inject_bad_results():
    payload = _build_search_fanout_bundle(
        step_data=_sample_step_data(),
        step_num=2,
        primary_keyword="oscillator 2",
        primary_family="osc",
        support_family="filter1",
        fanout=2,
        bad_result_prob=1.0,
        bad_result_max=1,
        rng=random.Random(42),
    )
    bundle = payload["bundle"]
    assert bundle["quality_summary"]["bad_injected"] == 1
    assert any(r["bad_type"] for r in bundle["reports"])
    assert any(r["status"] == "low_confidence" for r in bundle["reports"])


def test_handoff_interpretation_mentions_summary():
    payload = _build_search_fanout_bundle(
        step_data=_sample_step_data(),
        step_num=1,
        primary_keyword="oscillator 2",
        primary_family="osc",
        support_family=None,
        fanout=2,
        bad_result_prob=0.0,
        bad_result_max=0,
        rng=random.Random(0),
    )
    text = _build_search_handoff_interpretation(payload["bundle"], "oscillator 2")
    assert "Search handoff:" in text
    assert "agents" in text
    assert "consensus" in text
