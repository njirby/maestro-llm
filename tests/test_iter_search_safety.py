from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_iter_sft_dataset import (
    _build_set_turn_content,
    _build_search_snippet,
    _build_search_result_from_step,
    _choose_search_snippet,
    _infer_search_keyword,
    _validate_set_turn_content,
)
from maestro.synth.path_gen import _generate_search_result, _search_keyword


def test_path_gen_search_keyword_fallback_is_non_empty():
    assert _search_keyword([]) == "filter"


def test_infer_search_keyword_ignores_blank_explicit_keyword():
    step = {"search_keyword": "   ", "params_applied": {}, "params_delta": []}
    assert _infer_search_keyword(step) == "filter"


def test_choose_search_snippet_rejects_unbounded_empty_keyword_query():
    step = {
        "search_snippet": (
            "import reapy\n"
            "with reapy.inside_reaper():\n"
            "    fx = reapy.Project().tracks[0].fxs[0]\n"
            "    hits = [(p.name, p.normalized) for p in fx.params if '' in p.name.lower()]\n"
            "    print(hits)\n"
        )
    }
    snippet = _choose_search_snippet(step, "oscillator 1")
    assert "if '' in p.name.lower()" not in snippet
    assert "'oscillator 1' in p.name.lower()" in snippet


def test_build_search_result_ignores_broad_cached_result_when_keyword_blank():
    step = {
        "search_keyword": "",
        "search_result": "ALL PARAMS ...",
        "params_delta": [
            {"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.4},
        ],
    }
    result = _build_search_result_from_step(step, "filter 1")
    assert "ALL PARAMS" not in result
    assert "Filter 1 Cutoff" in result


def test_build_search_snippet_uses_exact_match_for_modulation_slots():
    snippet = _build_search_snippet("modulation 1")
    assert "startswith('modulation 1 ')" in snippet
    assert "'modulation 1' in p.name.lower()" not in snippet


def test_choose_search_snippet_rewrites_broad_modulation_candidate():
    step = {
        "search_snippet": (
            "import reapy\n"
            "with reapy.inside_reaper():\n"
            "    fx = reapy.Project().tracks[0].fxs[0]\n"
            "    hits = [(p.name, p.normalized) for p in fx.params if 'modulation 1' in p.name.lower()]\n"
            "    print(hits)\n"
        )
    }
    snippet = _choose_search_snippet(step, "modulation 1")
    assert "startswith('modulation 1 ')" in snippet
    assert "'modulation 1' in p.name.lower()" not in snippet


def test_build_search_result_filters_modulation_slot_exactly():
    step = {
        "search_keyword": "modulation 1",
        "search_result": (
            "Modulation 1 Amount: 0.5000\n"
            "Modulation 10 Amount: 0.5000\n"
            "Modulation 11 Amount: 0.5000\n"
        ),
        "params_delta": [],
    }
    result = _build_search_result_from_step(step, "modulation 1")
    assert "Modulation 1 Amount" in result
    assert "Modulation 10 Amount" not in result
    assert "Modulation 11 Amount" not in result


def test_path_gen_generate_search_result_filters_modulation_slot_exactly():
    settings = {
        "modulation_1_amount": 0.2,
        "modulation_10_amount": 0.4,
        "modulation_11_amount": 0.8,
    }
    result = _generate_search_result("modulation 1", settings)
    assert "Modulation 1 Amount" in result
    assert "Modulation 10 Amount" not in result
    assert "Modulation 11 Amount" not in result


def test_build_set_turn_content_keeps_primary_edits_within_keyword_scope():
    content = _build_set_turn_content(
        step_num=3,
        keyword="lfo 5",
        params_delta=[
            {"name": "lfo_3_phase", "from_norm": 0.2, "to_norm": 0.9},
            {"name": "lfo_5_sync", "from_norm": 0.0, "to_norm": 1.0},
            {"name": "lfo_5_phase", "from_norm": 0.5, "to_norm": 0.2},
        ],
        is_mistake=False,
        allowed_primary_names=["lfo_5_sync", "lfo_5_phase"],
        planned_primary_names=["lfo_3_phase", "lfo_5_sync", "lfo_5_phase"],
    )
    assert "LFO 5 Sync" in content
    assert "LFO 5 Phase" in content
    assert "LFO 3 Phase" not in content


def test_validate_set_turn_content_rejects_out_of_scope_primary_edit():
    set_content = (
        "Search check: found 12 lfo 5 controls; key values LFO 5 Delay Time=0.0000. "
        "Step 6: applying the planned lfo 5 updates now. Primary edits: LFO 3 Phase."
    )
    ok, reason = _validate_set_turn_content(
        set_content=set_content,
        keyword="lfo 5",
        allowed_primary_names=["lfo_5_sync", "lfo_5_phase"],
    )
    assert not ok
    assert "outside allowed primary controls" in reason or "does not match search keyword" in reason
