from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.build_iter_sft_dataset import _validate_commentary


def test_validate_commentary_accepts_compact_uncertain_output():
    commentary = (
        "HEARD: Audio A is brighter and wider than Audio B.\n\n"
        "HYPOTHESIS: The mismatch likely comes from lower filter cutoff and weaker chorus depth.\n\n"
        "PLAN: I will search oscillator 1 controls first, then raise filter cutoff and chorus mix, then listen again."
    )
    ok, reason = _validate_commentary(
        commentary,
        {"filter_1_cutoff", "chorus_dry_wet"},
        search_keyword="oscillator 1",
        archetype="bass",
    )
    assert ok, reason


def test_validate_commentary_rejects_missing_uncertainty():
    commentary = (
        "HEARD: Audio A has more high-end bite.\n\n"
        "HYPOTHESIS: The mismatch comes from filter and distortion settings.\n\n"
        "PLAN: I will search filter controls first and then increase the cutoff."
    )
    ok, reason = _validate_commentary(commentary, {"filter_1_cutoff"}, search_keyword="filter")
    assert not ok
    assert "uncertainty" in reason


def test_validate_commentary_rejects_programmatic_param_reference():
    commentary = (
        "HEARD: Audio A has a slower release and less click.\n\n"
        "HYPOTHESIS: The difference might be envelope timing.\n\n"
        "PLAN: Search envelope 1 controls first, then lower env_1_attack."
    )
    ok, reason = _validate_commentary(commentary, {"env_1_attack"}, search_keyword="envelope 1")
    assert not ok
    assert "snake_case" in reason


def test_validate_commentary_rejects_duplicate_sections():
    commentary = (
        "HEARD: Audio A is brighter.\n\n"
        "HYPOTHESIS: This likely comes from higher cutoff.\n\n"
        "PLAN: Increase filter_1_cutoff.\n\n"
        "HEARD: Extra duplicate section."
    )
    ok, reason = _validate_commentary(commentary, {"filter_1_cutoff"}, search_keyword="filter")
    assert not ok
    assert "exactly one" in reason


def test_validate_commentary_rejects_missing_search_focus_in_plan():
    commentary = (
        "HEARD: Audio A is duller and narrower than Audio B.\n\n"
        "HYPOTHESIS: This may come from envelope and filter shape differences.\n\n"
        "PLAN: I will inspect unison controls first and then apply the next edits."
    )
    ok, reason = _validate_commentary(commentary, set(), search_keyword="filter 1")
    assert not ok
    assert "search focus" in reason


def test_validate_commentary_rejects_archetype_mismatch():
    commentary = (
        "HEARD: Audio A has a slower bloom and wider tail than Audio B.\n\n"
        "HYPOTHESIS: The gap likely comes from filter and reverb balance.\n\n"
        "PLAN: I will search filter controls first; this sounds like a pad preset so I will keep smoothing transients."
    )
    ok, reason = _validate_commentary(commentary, set(), search_keyword="filter", archetype="lead")
    assert not ok
    assert "archetype mismatch" in reason


def test_validate_commentary_rejects_exact_control_name_before_search():
    commentary = (
        "HEARD: Audio A has less upper harmonics and less movement than Audio B.\n\n"
        "HYPOTHESIS: The gap may come from oscillator and filter voicing.\n\n"
        "PLAN: I will search oscillator 1 controls first, then raise oscillator 1 random phase and listen again."
    )
    ok, reason = _validate_commentary(
        commentary,
        {"osc_1_random_phase"},
        search_keyword="oscillator 1",
        archetype="bass",
    )
    assert not ok
    assert "control families" in reason


def test_validate_commentary_rejects_lfo_alias_for_modulation_slot_focus():
    commentary = (
        "HEARD: Audio A has stronger rhythmic motion than Audio B.\n\n"
        "HYPOTHESIS: The mismatch might come from a missing modulation route depth.\n\n"
        "PLAN: I will inspect LFO 10 controls first, then apply the next update and listen again."
    )
    ok, reason = _validate_commentary(
        commentary,
        {"modulation_10_amount"},
        search_keyword="modulation 10",
        archetype="pad",
        primary_family="modulation",
    )
    assert not ok
    assert "search focus" in reason or "modulation-slot wording" in reason
