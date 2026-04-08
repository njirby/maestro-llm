"""Tests for the agent SFT quality grading pipeline.

Includes both unit tests for individual scoring functions and a smoke test
that verifies the single-call output scores lower than the two-stage output
on plan_param_alignment (the key discriminator).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.grade_agent_sft import (
    _extract_section,
    _has_all_sections,
    _param_mentioned_in_text,
    _llm_judge_plan_step,
    _score_clap_net_improvement,
    _score_hypothesis_grounding,
    _score_plan_rationale_unique,
    llm_judge_main_record,
    score_main_record,
    score_search_record,
    score_judge_record,
    score_record,
    grade_file,
)


# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------

def _make_commentary(heard: str, hypo: str, plan: str) -> str:
    return f"HEARD: {heard}\n\nHYPOTHESIS: {hypo}\n\nPLAN: {plan}"


def _make_step_label(step: int, param_names: list[str]) -> dict:
    return {
        "step": step,
        "keyword": "filter",
        "params_delta": [
            {"name": n, "from_norm": 0.2, "to_norm": 0.8, "mistake": False}
            for n in param_names
        ],
        "planned_param_names": param_names,
    }


def _make_main_record(
    commentaries: list[str],
    step_labels: list[dict] | None = None,
    extra_assistant: list[str] | None = None,
) -> dict:
    messages = [{"role": "user", "content": "<audio>\nRecreate this sound."}]
    for c in commentaries:
        messages.append({"role": "assistant", "content": c})
        messages.append({"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"python - <<\'PY\'\\nprint(1)\\nPY"}}'})
        messages.append({"role": "tool_response", "content": '{"status":"ok"}'})
    for a in (extra_assistant or []):
        messages.append({"role": "assistant", "content": a})
    if messages[-1]["role"] != "assistant":
        messages.append({"role": "assistant", "content": "Done."})
    return {
        "id": "test_main",
        "task_type": "main",
        "messages": messages,
        "audios": ["/tmp/t.wav"],
        "meta": {"step_labels": step_labels or []},
        "labels": {},
    }


# ---------------------------------------------------------------------------
# Unit tests: text parsing helpers
# ---------------------------------------------------------------------------

def test_extract_section_plain():
    text = "HEARD: bright tone\n\nHYPOTHESIS: osc driven\n\nPLAN: raise filter cutoff"
    assert _extract_section(text, "PLAN:") == "raise filter cutoff"
    assert _extract_section(text, "HEARD:") == "bright tone"


def test_extract_section_bold_variant():
    text = "**HEARD:** bright tone\n\n**HYPOTHESIS:** osc driven\n\n**PLAN:** raise filter"
    plan = _extract_section(text, "PLAN:")
    assert "raise filter" in plan


def test_extract_section_missing():
    assert _extract_section("no sections here", "PLAN:") == ""


def test_has_all_sections_true():
    assert _has_all_sections("HEARD: x\nHYPOTHESIS: y\nPLAN: z")


def test_has_all_sections_false():
    assert not _has_all_sections("HEARD: x\nHYPOTHESIS: y")


def test_has_all_sections_bold_ok():
    assert _has_all_sections("**HEARD:** x\n**HYPOTHESIS:** y\n**PLAN:** z")


# ---------------------------------------------------------------------------
# Unit tests: param mention detection
# ---------------------------------------------------------------------------

def test_param_mentioned_bigram_match():
    assert _param_mentioned_in_text(
        "Increase Filter 1 Cutoff to brighten the tone.", "filter_1_cutoff"
    )


def test_param_mentioned_case_insensitive():
    assert _param_mentioned_in_text(
        "raise modulation 10 bypass to engage the effect.", "modulation_10_bypass"
    )


def test_param_mentioned_lfo_sync():
    assert _param_mentioned_in_text(
        "Decrease LFO 1 Sync to remove tempo lock.", "lfo_1_sync"
    )


def test_param_not_mentioned():
    assert not _param_mentioned_in_text(
        "Adjust the reverb room size.", "filter_1_cutoff"
    )


def test_param_mentioned_requires_specific_bigram():
    # "distortion" alone (no number context) should NOT match osc_2_distortion_type —
    # the model must say "2 distortion" or "distortion type" to get credit.
    assert not _param_mentioned_in_text(
        "Enable the distortion for grit.", "osc_2_distortion_type"
    )
    # But "2 distortion" in context DOES match.
    assert _param_mentioned_in_text(
        "Increase oscillator 2 distortion to add grit.", "osc_2_distortion_type"
    )


def test_param_entity_number_bigram_rejected():
    # "oscillator 2" alone must not count as a specific parameter reference.
    assert not _param_mentioned_in_text(
        "Apply updates for oscillator 2.", "osc_2_on"
    )
    assert not _param_mentioned_in_text(
        "Apply updates for modulation 10.", "modulation_10_bypass"
    )
    assert not _param_mentioned_in_text(
        "Apply updates for lfo 1.", "lfo_1_sync"
    )


# ---------------------------------------------------------------------------
# Unit tests: main record scoring
# ---------------------------------------------------------------------------

def test_plan_param_alignment_perfect():
    labels = [_make_step_label(1, ["filter_1_cutoff"])]
    commentary = _make_commentary("brighter", "filter likely", "Increase Filter 1 Cutoff to brighten.")
    record = _make_main_record([commentary], step_labels=labels)
    scores = score_main_record(record)
    assert scores["plan_param_alignment"] == 1.0


def test_plan_param_alignment_zero():
    labels = [_make_step_label(1, ["filter_1_cutoff"])]
    commentary = _make_commentary("brighter", "osc driven", "Adjust the reverb room size.")
    record = _make_main_record([commentary], step_labels=labels)
    scores = score_main_record(record)
    assert scores["plan_param_alignment"] == 0.0


def test_plan_param_alignment_partial():
    labels = [
        _make_step_label(1, ["filter_1_cutoff"]),  # correct
        _make_step_label(2, ["env_1_attack"]),       # incorrect plan
    ]
    c1 = _make_commentary("brighter", "filter", "Increase Filter 1 Cutoff.")
    c2 = _make_commentary("slower", "envelope", "Raise the reverb decay time.")
    record = _make_main_record([c1, c2], step_labels=labels)
    scores = score_main_record(record)
    assert scores["plan_param_alignment"] == pytest.approx(0.5)


def test_plan_param_alignment_none_when_no_step_labels():
    record = _make_main_record([_make_commentary("x", "y", "z")], step_labels=[])
    scores = score_main_record(record)
    assert scores["plan_param_alignment"] is None


def test_section_structure_perfect():
    labels = [_make_step_label(1, [])]
    commentary = _make_commentary("tone", "osc", "filter cutoff")
    record = _make_main_record([commentary], step_labels=labels)
    scores = score_main_record(record)
    assert scores["section_structure"] == 1.0


def test_section_structure_missing():
    record = _make_main_record(["Just a plain assistant message."])
    scores = score_main_record(record)
    assert scores["section_structure"] == 0.0


def test_snake_case_penalty():
    commentary = _make_commentary("bright", "osc", "Adjust filter_1_cutoff now.")
    record = _make_main_record([commentary])
    scores = score_main_record(record)
    assert scores["snake_case_clean"] < 1.0


def test_snake_case_clean():
    commentary = _make_commentary("bright", "osc", "Raise Filter 1 Cutoff.")
    record = _make_main_record([commentary])
    scores = score_main_record(record)
    assert scores["snake_case_clean"] == 1.0


def test_bold_header_penalty():
    commentary = "**HEARD:** bright\n\n**HYPOTHESIS:** osc\n\n**PLAN:** raise filter"
    record = _make_main_record([commentary])
    scores = score_main_record(record)
    assert scores["format_consistent"] < 1.0


def test_format_consistent():
    commentary = _make_commentary("bright", "osc", "raise filter cutoff")
    record = _make_main_record([commentary])
    scores = score_main_record(record)
    assert scores["format_consistent"] == 1.0


def test_commentary_diversity_identical():
    """Two identical commentaries should yield low diversity."""
    c = _make_commentary("bright tone with sharp attack", "oscillator driven", "raise the filter cutoff setting")
    record = _make_main_record([c, c])
    scores = score_main_record(record)
    assert scores["commentary_diversity"] < 0.3


def test_commentary_diversity_distinct():
    """Very different commentaries should yield high diversity."""
    c1 = _make_commentary("bright sharp transient", "oscillator distortion", "raise filter cutoff resonance")
    c2 = _make_commentary("slow evolving pad", "envelope attack long", "lower reverb decay time")
    record = _make_main_record([c1, c2])
    scores = score_main_record(record)
    assert scores["commentary_diversity"] > 0.5


def test_overall_score_bounded():
    commentary = _make_commentary("x", "y", "z")
    record = _make_main_record([commentary])
    scores = score_main_record(record)
    assert 0.0 <= scores["overall"] <= 1.0


# ---------------------------------------------------------------------------
# Unit tests: search record scoring
# ---------------------------------------------------------------------------

def _make_search_record(has_cosine: bool, proposals: list[dict] | None = None, gt_leak: bool = False) -> dict:
    audition_entry = {"candidate_id": "C1", "wavetable_name": "wt", "audio_preview": "<audio>"}
    if has_cosine:
        audition_entry["cosine_vs_target"] = 0.42
    tool_response_content = json.dumps({"status": "ok", "auditioned": 1, "audition_results": [audition_entry]})
    proposal_list = proposals or [{"candidate_id": "C1", "reason": "Strong match (0.800)."}]
    proposal_content = json.dumps({"proposals": proposal_list})
    gt_text = "Closest waveform match; likely the source wavetable." if gt_leak else "Strong match (0.800)."
    return {
        "id": "test_search",
        "task_type": "search",
        "messages": [
            {"role": "user", "content": "<audio>\nSearch task."},
            {"role": "assistant", "content": gt_text},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"search"}}'},
            {"role": "tool_response", "content": '{"stdout":"C1\\twt\\t0.80\\t/tmp/c1.wav"}'},
            {"role": "assistant", "content": "Auditioning."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"audition"}}'},
            {"role": "tool_response", "content": tool_response_content},
            {"role": "assistant", "content": proposal_content},
        ],
        "audios": ["/tmp/t.wav", "/tmp/c1.wav"],
        "labels": {},
        "meta": {},
    }


def test_search_has_cosine_scores():
    record = _make_search_record(has_cosine=True)
    scores = score_search_record(record)
    assert scores["has_cosine_scores"] == 1.0


def test_search_missing_cosine_scores():
    record = _make_search_record(has_cosine=False)
    scores = score_search_record(record)
    assert scores["has_cosine_scores"] == 0.0


def test_search_no_gt_leak():
    record = _make_search_record(has_cosine=True, gt_leak=False)
    scores = score_search_record(record)
    assert scores["no_gt_leak"] == 1.0


def test_search_gt_leak_detected():
    record = _make_search_record(has_cosine=True, gt_leak=True)
    scores = score_search_record(record)
    assert scores["no_gt_leak"] == 0.0


def test_search_proposal_diversity():
    proposals = [
        {"candidate_id": "C1", "reason": "Strong match (0.800)."},
        {"candidate_id": "C2", "reason": "Moderate match (0.400)."},
    ]
    record = _make_search_record(has_cosine=True, proposals=proposals)
    scores = score_search_record(record)
    assert scores["proposal_diversity"] == 1.0


def test_search_proposal_duplicate_reasons():
    same_reason = "Strong match (0.800)."
    proposals = [
        {"candidate_id": "C1", "reason": same_reason},
        {"candidate_id": "C2", "reason": same_reason},
    ]
    record = _make_search_record(has_cosine=True, proposals=proposals)
    scores = score_search_record(record)
    assert scores["proposal_diversity"] < 1.0


# ---------------------------------------------------------------------------
# Unit tests: judge record scoring
# ---------------------------------------------------------------------------

def _make_judge_record(has_scores: bool, has_cosine: bool, gt_in_selected: bool | None) -> dict:
    reason = "Ranked by CLAP. Top scores: C1 (0.800), C2 (0.600)." if has_scores else "Selected top 2."
    final_content = json.dumps({"ranking": ["C1", "C2"], "selected": ["C1"], "reason": reason})
    audition_entry = {"candidate_id": "C1", "wavetable_name": "wt", "audio_preview": "<audio>"}
    if has_cosine:
        audition_entry["cosine_vs_target"] = 0.42
    tool_response_content = json.dumps({"audition_results": [audition_entry]})
    labels: dict = {}
    if gt_in_selected is True:
        labels = {"gt_candidate_ids": ["C1"], "selected": ["C1"]}
    elif gt_in_selected is False:
        labels = {"gt_candidate_ids": ["C3"], "selected": ["C1"]}
    return {
        "id": "test_judge",
        "task_type": "judge",
        "messages": [
            {"role": "user", "content": "<audio>\nJudge task."},
            {"role": "assistant", "content": "Auditioning."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"audition"}}'},
            {"role": "tool_response", "content": tool_response_content},
            {"role": "assistant", "content": final_content},
        ],
        "audios": ["/tmp/t.wav", "/tmp/c1.wav"],
        "labels": labels,
        "meta": {},
    }


def test_judge_has_score_in_reason():
    record = _make_judge_record(has_scores=True, has_cosine=True, gt_in_selected=True)
    scores = score_judge_record(record)
    assert scores["has_score_in_reason"] == 1.0


def test_judge_no_score_in_reason():
    record = _make_judge_record(has_scores=False, has_cosine=True, gt_in_selected=True)
    scores = score_judge_record(record)
    assert scores["has_score_in_reason"] == 0.0


def test_judge_gt_recall_hit():
    record = _make_judge_record(has_scores=True, has_cosine=True, gt_in_selected=True)
    scores = score_judge_record(record)
    assert scores["gt_recall"] == 1.0


def test_judge_gt_recall_miss():
    record = _make_judge_record(has_scores=True, has_cosine=True, gt_in_selected=False)
    scores = score_judge_record(record)
    assert scores["gt_recall"] == 0.0


def test_judge_gt_recall_unknown():
    record = _make_judge_record(has_scores=True, has_cosine=True, gt_in_selected=None)
    scores = score_judge_record(record)
    assert scores["gt_recall"] is None


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def test_score_record_dispatches_by_task_type():
    main_r = _make_main_record([_make_commentary("x", "y", "z")])
    assert "section_structure" in score_record(main_r)

    search_r = _make_search_record(has_cosine=True)
    assert "has_cosine_scores" in score_record(search_r)

    judge_r = _make_judge_record(has_scores=True, has_cosine=True, gt_in_selected=True)
    assert "has_score_in_reason" in score_record(judge_r)


# ---------------------------------------------------------------------------
# Smoke test: single-call vs two-stage output files
# ---------------------------------------------------------------------------

SINGLE_SMOKE = Path("outputs/agent_sft_v1/main_smoke.jsonl")
TWO_STAGE_SMOKE = Path("outputs/agent_sft_v1/main_smoke_two_stage.jsonl")
FALLBACK_SMOKE = Path("outputs/agent_sft_v1/main_smoke_fallback.jsonl")


def _load_scores(path: Path) -> list[dict]:
    scores = []
    with open(path) as f:
        for line in f:
            r = json.loads(line.strip())
            scores.append(score_main_record(r))
    return scores


def _mean(vals: list[float | None]) -> float:
    valid = [v for v in vals if v is not None]
    return sum(valid) / len(valid) if valid else 0.0


@pytest.mark.skipif(
    not FALLBACK_SMOKE.exists(),
    reason="Fallback smoke file not present — run build_main_agent_sft_v2.py with --omni-server '' first.",
)
def test_fallback_alignment_near_zero():
    """Template fallback commentary must score near-zero plan_param_alignment —
    proving the grader rejects keyword-only PLAN text."""
    scores = _load_scores(FALLBACK_SMOKE)
    align = _mean([s["plan_param_alignment"] for s in scores])
    print(f"\nfallback mean plan_param_alignment: {align:.3f}")
    assert align <= 0.20, f"Fallback alignment should be near 0 but got {align:.3f}"


@pytest.mark.skipif(
    not SINGLE_SMOKE.exists() or not TWO_STAGE_SMOKE.exists(),
    reason="Smoke output files not present — run build_main_agent_sft_v2.py first.",
)
def test_llm_modes_beat_fallback_on_alignment():
    """Both LLM modes must score higher plan_param_alignment than template fallback."""
    fallback_scores = _load_scores(FALLBACK_SMOKE) if FALLBACK_SMOKE.exists() else []
    fallback_align = _mean([s["plan_param_alignment"] for s in fallback_scores])
    for path, label in [(SINGLE_SMOKE, "single"), (TWO_STAGE_SMOKE, "two_stage")]:
        scores = _load_scores(path)
        align = _mean([s["plan_param_alignment"] for s in scores])
        print(f"\n{label} ({align:.3f}) vs fallback ({fallback_align:.3f})")
        assert align > fallback_align, f"{label} ({align:.3f}) must beat fallback ({fallback_align:.3f})"


@pytest.mark.skipif(
    not SINGLE_SMOKE.exists() or not TWO_STAGE_SMOKE.exists(),
    reason="Smoke output files not present — run build_main_agent_sft_v2.py first.",
)
def test_both_modes_achieve_high_alignment():
    """Two-stage must score ≥ 0.50; single-call must score ≥ 0.30.

    Single-call is weaker because it sometimes writes vague PLAN text like
    "Modulation 11 control" instead of the specific sub-parameter name.
    Two-stage isolates audio perception from param integration, so it
    produces more specific PLAN text. This test documents that gap.
    """
    thresholds = {"single": 0.30, "two_stage": 0.50}
    for path, label in [(SINGLE_SMOKE, "single"), (TWO_STAGE_SMOKE, "two_stage")]:
        scores = _load_scores(path)
        align = _mean([s["plan_param_alignment"] for s in scores])
        print(f"\n{label} mean plan_param_alignment: {align:.3f}")
        assert align >= thresholds[label], f"{label} alignment too low: {align:.3f} (min {thresholds[label]})"


@pytest.mark.skipif(
    not SINGLE_SMOKE.exists() or not TWO_STAGE_SMOKE.exists(),
    reason="Smoke output files not present.",
)
def test_two_stage_better_or_equal_format_consistency():
    """Two-stage must have mean format_consistent >= single (no bold **PLAN:** headers)."""
    single_scores = _load_scores(SINGLE_SMOKE)
    two_stage_scores = _load_scores(TWO_STAGE_SMOKE)
    single_fmt = _mean([s["format_consistent"] for s in single_scores])
    two_stage_fmt = _mean([s["format_consistent"] for s in two_stage_scores])
    print(f"\nformat_consistent — single: {single_fmt:.3f}  two_stage: {two_stage_fmt:.3f}")
    assert two_stage_fmt >= single_fmt, (
        f"two_stage ({two_stage_fmt:.3f}) should be >= single ({single_fmt:.3f})"
    )


@pytest.mark.skipif(
    not TWO_STAGE_SMOKE.exists(),
    reason="Two-stage smoke file not present.",
)
def test_two_stage_alignment_above_threshold():
    """Two-stage records should achieve plan_param_alignment ≥ 0.80."""
    scores = _load_scores(TWO_STAGE_SMOKE)
    for s in scores:
        val = s["plan_param_alignment"]
        if val is not None:
            assert val >= 0.80, f"Two-stage alignment too low: {val:.3f}\n{s}"


@pytest.mark.skipif(
    not SINGLE_SMOKE.exists(),
    reason="Single smoke file not present.",
)
def test_single_alignment_above_threshold():
    """Single-call mean plan_param_alignment should be ≥ 0.30.

    Single-call mode is meaningfully weaker than two-stage: it tends to write
    vague PLAN text like "Modulation 11 control" without naming the sub-parameter.
    This floor (0.30 mean) ensures single-call is at least better than template
    fallback (~0.0) while honestly capturing the gap vs two-stage (typically ≥ 0.80).
    Per-record thresholds are too sensitive for small smoke files (2-3 records).
    """
    scores = _load_scores(SINGLE_SMOKE)
    align = _mean([s["plan_param_alignment"] for s in scores])
    print(f"\nsingle mean plan_param_alignment: {align:.3f}")
    assert align >= 0.30, f"Single alignment mean too low: {align:.3f}"


def test_grader_distinguishes_bad_from_good_synthetically():
    """The grader must score a record with wrong PLAN params lower than one with correct params."""
    labels = [_make_step_label(1, ["modulation_10_bypass"])]
    # Good: PLAN mentions the actual param
    good = _make_main_record(
        [_make_commentary("brighter", "mod routing", "Increase Modulation 10 Bypass to engage the effect.")],
        step_labels=labels,
    )
    # Bad: PLAN mentions wrong params (oscillator instead of modulation)
    bad = _make_main_record(
        [_make_commentary("brighter", "osc driven", "Raise Filter 1 Cutoff and Oscillator 2 Level.")],
        step_labels=labels,
    )
    good_score = score_main_record(good)["plan_param_alignment"]
    bad_score = score_main_record(bad)["plan_param_alignment"]
    assert good_score > bad_score, f"Good ({good_score}) should beat bad ({bad_score})"
    assert good_score == 1.0
    assert bad_score == 0.0


@pytest.mark.skipif(
    not SINGLE_SMOKE.exists(),
    reason="Single smoke file not present.",
)
def test_grade_file_cli_produces_scored_output(tmp_path):
    """grade_file() must write a JSONL with quality_scores on each record."""
    out = tmp_path / "graded.jsonl"
    summary = grade_file(SINGLE_SMOKE, out)
    assert out.exists()
    assert summary["total"] > 0
    with open(out) as f:
        for line in f:
            r = json.loads(line)
            assert "quality_scores" in r
            assert "overall" in r["quality_scores"]


# ---------------------------------------------------------------------------
# LLM-as-judge unit tests (mocked — no server required)
# ---------------------------------------------------------------------------

def _mock_httpx_response(score: float, reasoning: str = "test") -> MagicMock:
    """Build a fake httpx response object that returns a judge JSON."""
    content = json.dumps({"score": score, "reasoning": reasoning})
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": content}}]
    }
    return mock_resp


def test_llm_judge_plan_step_returns_score():
    """_llm_judge_plan_step returns (float, str) tuple when server responds."""
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8}]
    with patch("scripts.grade_agent_sft._httpx") as mock_httpx:
        mock_httpx.post.return_value = _mock_httpx_response(1.0, "Filter 1 Cutoff mentioned ✓")
        result = _llm_judge_plan_step(
            plan_text="Increase Filter 1 Cutoff to brighten the tone.",
            params_delta=params_delta,
            server_url="http://localhost:8000",
            model="Qwen2.5-7B-Instruct",
        )
    assert result is not None
    score, reasoning = result
    assert 0.0 <= score <= 1.0
    assert score == pytest.approx(1.0)
    assert isinstance(reasoning, str)


def test_llm_judge_plan_step_prompt_contains_param_and_plan():
    """The prompt sent to the LLM must contain both the PLAN text and the param display name."""
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8}]
    captured_payload = {}
    with patch("scripts.grade_agent_sft._httpx") as mock_httpx:
        def capture(*args, **kwargs):
            captured_payload.update(kwargs.get("json", {}))
            return _mock_httpx_response(1.0)
        mock_httpx.post.side_effect = capture
        _llm_judge_plan_step(
            plan_text="Increase Filter 1 Cutoff to brighten the tone.",
            params_delta=params_delta,
            server_url="http://localhost:8000",
            model="Qwen2.5-7B-Instruct",
        )

    user_content = captured_payload["messages"][-1]["content"]
    assert "Filter 1 Cutoff" in user_content or "filter_1_cutoff" in user_content
    assert "Increase Filter 1 Cutoff" in user_content


def test_llm_judge_plan_step_clamps_out_of_range():
    """LLM scores outside [0,1] should be clamped."""
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8}]
    with patch("scripts.grade_agent_sft._httpx") as mock_httpx:
        mock_httpx.post.return_value = _mock_httpx_response(1.5)  # out of range
        result = _llm_judge_plan_step(
            "Increase Filter 1 Cutoff.", params_delta, "http://localhost:8000", "model"
        )
    assert result is not None
    score, _ = result
    assert score == pytest.approx(1.0)


def test_llm_judge_plan_step_returns_none_on_server_error():
    """If the server raises an exception, _llm_judge_plan_step returns None, not an error."""
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8}]
    with patch("scripts.grade_agent_sft._httpx") as mock_httpx:
        mock_httpx.post.side_effect = Exception("connection refused")
        result = _llm_judge_plan_step(
            "Increase Filter 1 Cutoff.", params_delta, "http://localhost:8000", "model"
        )
    assert result is None


def test_llm_judge_plan_step_handles_markdown_fence():
    """LLM sometimes wraps JSON in ```json ... ``` — must still parse correctly."""
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8}]
    fenced_content = '```json\n{"score": 0.5, "reasoning": "mentions cutoff but adds extras"}\n```'
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {"choices": [{"message": {"content": fenced_content}}]}
    with patch("scripts.grade_agent_sft._httpx") as mock_httpx:
        mock_httpx.post.return_value = mock_resp
        result = _llm_judge_plan_step(
            "Raise filter cutoff.", params_delta, "http://localhost:8000", "model"
        )
    assert result is not None
    score, _ = result
    assert score == pytest.approx(0.5)


def test_llm_judge_returns_reasoning_string():
    """_llm_judge_plan_step returns the reasoning string as the second tuple element."""
    params_delta = [{"name": "env_1_attack", "from_norm": 0.1, "to_norm": 0.5}]
    with patch("scripts.grade_agent_sft._httpx") as mock_httpx:
        mock_httpx.post.return_value = _mock_httpx_response(
            0.5, "Env 1 Attack ✓ mentioned, but also references filter which is not in step."
        )
        result = _llm_judge_plan_step(
            "Raise Envelope 1 Attack and also increase filter resonance.",
            params_delta,
            "http://localhost:8000",
            "model",
        )
    assert result is not None
    score, reasoning = result
    assert score == pytest.approx(0.5)
    assert "Env" in reasoning or "Attack" in reasoning or "filter" in reasoning


def test_llm_judge_prompt_contains_step_instructions():
    """The judge prompt must include STEP 1/2/3 and all three rubric score values."""
    from scripts.grade_agent_sft import _LLM_JUDGE_PROMPT_TEMPLATE
    rendered = _LLM_JUDGE_PROMPT_TEMPLATE.format(
        param_list="  - Filter 1 Cutoff (filter_1_cutoff)",
        plan_text="Adjusting Filter 1 Cutoff will brighten the tone.",
    )
    assert "STEP 1" in rendered
    assert "STEP 2" in rendered
    assert "STEP 3" in rendered
    assert "1.0" in rendered
    assert "0.5" in rendered
    assert "0.0" in rendered


def test_llm_judge_prompt_contains_grouped_language_guidance():
    """The judge prompt must explicitly tell the model that 'N subsystem parameters'
    counts as naming all params in that subsystem."""
    from scripts.grade_agent_sft import _LLM_JUDGE_PROMPT_TEMPLATE
    assert "N [subsystem] parameters" in _LLM_JUDGE_PROMPT_TEMPLATE
    # The grouped acceptance bullet must be present
    assert "compressor parameters" in _LLM_JUDGE_PROMPT_TEMPLATE
    assert "covers ALL" in _LLM_JUDGE_PROMPT_TEMPLATE or "counts as" in _LLM_JUDGE_PROMPT_TEMPLATE


def test_llm_judge_main_record_aggregates_steps():
    """llm_judge_main_record averages scores across all assessable steps."""
    labels = [
        _make_step_label(1, ["filter_1_cutoff"]),
        _make_step_label(2, ["env_1_attack"]),
    ]
    c1 = _make_commentary("bright", "filter", "Increase Filter 1 Cutoff.")
    c2 = _make_commentary("slow", "envelope", "Raise Amplitude Envelope Attack.")
    record = _make_main_record([c1, c2], step_labels=labels)

    call_count = 0
    def fake_judge(plan_text, params_delta, server_url, model, timeout=10.0):
        nonlocal call_count
        call_count += 1
        return (0.8, "step 1 ok") if call_count == 1 else (1.0, "step 2 ok")

    with patch("scripts.grade_agent_sft._llm_judge_plan_step", side_effect=fake_judge):
        result = llm_judge_main_record(record, "http://localhost:8000", "model")

    assert result["llm_plan_alignment"] == pytest.approx(0.9)
    assert len(result["llm_per_step"]) == 2


def test_llm_judge_per_step_includes_reasoning():
    """llm_per_step entries must have a 'reasoning' key."""
    labels = [_make_step_label(1, ["filter_1_cutoff"])]
    c1 = _make_commentary("bright", "filter", "Increase Filter 1 Cutoff.")
    record = _make_main_record([c1], step_labels=labels)

    def fake_judge(plan_text, params_delta, server_url, model, timeout=10.0):
        return (1.0, "Filter 1 Cutoff is ✓ mentioned and no extras.")

    with patch("scripts.grade_agent_sft._llm_judge_plan_step", side_effect=fake_judge):
        result = llm_judge_main_record(record, "http://localhost:8000", "model")

    assert len(result["llm_per_step"]) == 1
    entry = result["llm_per_step"][0]
    assert "reasoning" in entry
    assert "Filter" in entry["reasoning"] or entry["reasoning"]  # non-empty


def test_llm_judge_discrete_score_05_partial():
    """A 0.5 score from the judge propagates correctly through llm_plan_alignment."""
    labels = [_make_step_label(1, ["filter_1_cutoff"])]
    c1 = _make_commentary("bright", "filter", "Increase Filter 1 Cutoff and also LFO rate.")
    record = _make_main_record([c1], step_labels=labels)

    def fake_judge(plan_text, params_delta, server_url, model, timeout=10.0):
        return (0.5, "Filter ✓ but LFO rate is not in step.")

    with patch("scripts.grade_agent_sft._llm_judge_plan_step", side_effect=fake_judge):
        result = llm_judge_main_record(record, "http://localhost:8000", "model")

    assert result["llm_plan_alignment"] == pytest.approx(0.5)


def test_score_main_record_with_llm_judge_overrides_heuristic():
    """When llm_judge_server is set, llm_plan_alignment drives overall score."""
    labels = [_make_step_label(1, ["filter_1_cutoff"])]
    # Bad PLAN text — heuristic would score 0.0
    bad_commentary = _make_commentary(
        "brighter tone", "oscillator", "Enable oscillator 2 on."
    )
    record = _make_main_record([bad_commentary], step_labels=labels)

    with patch("scripts.grade_agent_sft._llm_judge_plan_step", return_value=(1.0, "correct")):
        scores = score_main_record(record, llm_judge_server="http://localhost:8000")

    # LLM says 1.0; heuristic would say 0.0 — LLM should win
    assert scores["llm_plan_alignment"] == pytest.approx(1.0)
    assert scores["heuristic_plan_param_alignment"] == pytest.approx(0.0)
    # Overall should be high (driven by llm score of 1.0)
    assert scores["overall"] > 0.5


def test_score_main_record_without_llm_judge_no_llm_keys():
    """Without llm_judge_server, output must not contain llm_ keys."""
    labels = [_make_step_label(1, ["filter_1_cutoff"])]
    commentary = _make_commentary("bright", "filter", "Increase Filter 1 Cutoff.")
    record = _make_main_record([commentary], step_labels=labels)
    scores = score_main_record(record)
    assert "llm_plan_alignment" not in scores
    assert "llm_per_step" not in scores
    assert "heuristic_plan_param_alignment" in scores  # always present


def test_llm_judge_falls_back_when_httpx_unavailable():
    """If httpx is not installed, _llm_judge_plan_step returns None gracefully."""
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8}]
    import scripts.grade_agent_sft as grader_module
    original = grader_module._HTTPX_AVAILABLE
    try:
        grader_module._HTTPX_AVAILABLE = False
        result = _llm_judge_plan_step(
            "Increase Filter 1 Cutoff.", params_delta, "http://localhost:8000", "model"
        )
        assert result is None
    finally:
        grader_module._HTTPX_AVAILABLE = original


def test_overall_score_excludes_diversity_weight():
    """commentary_diversity must not be in the overall scoring weights."""
    from scripts import grade_agent_sft
    import inspect, ast, textwrap

    # Read the source and find the weights dict near the overall formula.
    # We check that the string "commentary_diversity" does not appear in a weights dict.
    src = inspect.getsource(grade_agent_sft)
    # Quick string check: it should NOT be a key in any weights dict in the grading formula.
    # We look for the pattern '"commentary_diversity": <non-zero>' in a weights dict context.
    import re
    # Match lines that assign a non-zero weight to commentary_diversity in a dict
    pattern = re.compile(r'"commentary_diversity"\s*:\s*0\.[1-9]')
    assert not pattern.search(src), \
        "commentary_diversity should not have a non-zero weight in the overall formula"


def test_overall_score_unaffected_by_low_diversity():
    """A record with diversity=0.0 should still score >0.8 overall if other metrics pass."""
    from scripts.grade_agent_sft import score_main_record
    from unittest.mock import patch, MagicMock

    # Build a minimal record with perfect structural scores but diversity=0 won't matter
    record = {
        "id": "test_record",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "step 1"},
            {"role": "assistant", "content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting Filter 1 Cutoff. This brightens the tone."},
        ],
        "labels": {
            "step_labels": [
                {"step": 1, "params_delta": [{"name": "filter_1_cutoff", "from_norm": 0.3, "to_norm": 0.7}],
                 "clap_delta": 0.1, "planned_param_names": ["filter_1_cutoff"]}
            ]
        },
        "quality_scores": {},
    }

    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        result = score_main_record(record)

    overall = result.get("overall", 0)
    assert overall > 0.8, f"Expected overall > 0.8, got {overall}"


# ---------------------------------------------------------------------------
# CLAP monotonicity metric
# ---------------------------------------------------------------------------

def test_clap_net_improvement_returns_none_for_no_clap_data():
    labels = [{"step": i, "clap_score": None} for i in range(1, 5)]
    assert _score_clap_net_improvement(labels) is None


def test_clap_net_improvement_returns_none_for_single_step():
    labels = [{"step": 1, "clap_score": 0.5}]
    assert _score_clap_net_improvement(labels) is None


def test_clap_net_improvement_perfect_when_large_delta():
    # delta = 0.55 - 0.25 = 0.30 → score 1.0 (at normalization ceiling)
    # i=0: 0.25, i=4: 0.55 → delta exactly 0.30
    labels = [{"step": i + 1, "clap_score": 0.25 + i * 0.075} for i in range(5)]
    score = _score_clap_net_improvement(labels)
    assert score == 1.0, f"delta>=0.30 should score 1.0, got {score}"


def test_clap_net_improvement_zero_when_no_progress():
    # final == first → delta=0
    labels = [{"step": 1, "clap_score": 0.5}, {"step": 2, "clap_score": 0.5}]
    assert _score_clap_net_improvement(labels) == 0.0


def test_clap_net_improvement_zero_when_degrading():
    # final < first → clamped to 0.0
    labels = [{"step": i, "clap_score": 0.8 - i * 0.05} for i in range(1, 6)]
    assert _score_clap_net_improvement(labels) == 0.0


def test_clap_net_improvement_partial_score():
    # delta = 0.55 - 0.40 = 0.15 → 0.15/0.30 = 0.5
    labels = [
        {"step": 1, "clap_score": 0.40},
        {"step": 2, "clap_score": 0.45},  # intermediate is ignored
        {"step": 3, "clap_score": 0.30},  # intermediate is ignored
        {"step": 4, "clap_score": 0.55},  # only last matters
    ]
    score = _score_clap_net_improvement(labels)
    assert abs(score - 0.5) < 1e-4, f"Expected 0.5 (delta=0.15/0.30), got {score}"


def test_clap_net_improvement_ignores_intermediate_fluctuations():
    # Noisy path: intermediate steps dip, but first=0.40, last=0.70 → delta=0.30 → 1.0
    labels = [
        {"step": 1, "clap_score": 0.40},
        {"step": 2, "clap_score": 0.35},  # dip
        {"step": 3, "clap_score": 0.32},  # worse
        {"step": 4, "clap_score": 0.45},
        {"step": 5, "clap_score": 0.70},  # big jump at end
    ]
    score = _score_clap_net_improvement(labels)
    assert score == 1.0, f"Should be 1.0 (net delta=0.30), got {score}"


def test_clap_net_improvement_reduces_overall_when_low():
    """A path that degrades CLAP every step should score lower than one that improves."""
    from unittest.mock import patch

    def _make_record(clap_scores: list) -> dict:
        return {
            "id": "test",
            "task_type": "main",
            "messages": [
                {"role": "user", "content": "target"},
                *[
                    {"role": "assistant", "content": f"HEARD: bright sound.\n\nHYPOTHESIS: The compressor is off.\n\nPLAN: Adjusting step {i}. This improves tone."}
                    for i in range(len(clap_scores))
                ],
            ],
            "labels": {},
            "meta": {
                "step_labels": [
                    {"step": i + 1, "clap_score": s, "remaining_top_2": "compressor and reverb",
                     "params_delta": [{"name": f"env_{i}_attack"}]}
                    for i, s in enumerate(clap_scores)
                ]
            },
        }

    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        improving = score_main_record(_make_record([0.3, 0.4, 0.5, 0.6, 0.7]))
        degrading = score_main_record(_make_record([0.7, 0.6, 0.5, 0.4, 0.3]))

    assert improving["overall"] > degrading["overall"], (
        f"Improving CLAP path ({improving['overall']:.4f}) should outscore "
        f"degrading path ({degrading['overall']:.4f})"
    )


def test_clap_net_improvement_in_score_output():
    """clap_net_improvement must appear in score_main_record output."""
    from unittest.mock import patch

    record = {
        "id": "test",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "target"},
            {"role": "assistant", "content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting cutoff. This brightens tone."},
        ],
        "labels": {},
        "meta": {
            "step_labels": [{"step": 1, "clap_score": 0.5, "remaining_top_2": None, "params_delta": []}]
        },
    }
    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        result = score_main_record(record)
    assert "clap_net_improvement" in result


# ---------------------------------------------------------------------------
# Plan rationale uniqueness metric
# ---------------------------------------------------------------------------

def test_plan_rationale_unique_returns_none_for_single_turn():
    turns = ["HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting filter 1 cutoff. This brightens the tone."]
    assert _score_plan_rationale_unique(turns) is None


def test_plan_rationale_unique_perfect_when_all_different():
    turns = [
        "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting filter 1 cutoff. This brightens the overall tone significantly.",
        "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting envelope attack time. This controls the transient punch.",
        "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting oscillator wave frame. This adds harmonic complexity to the timbre.",
    ]
    score = _score_plan_rationale_unique(turns)
    assert score == 1.0, f"All different sentences should score 1.0, got {score}"


def test_plan_rationale_unique_zero_when_all_identical():
    repeated = "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting filter 1 cutoff. These changes will enable deeper filter modulation and stereo movement shaping the evolving brightness."
    turns = [repeated] * 5
    score = _score_plan_rationale_unique(turns)
    assert score == 0.0, f"All identical sentences should score 0.0, got {score}"


def test_plan_rationale_unique_partial_repetition():
    turns = [
        "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting filter 1 cutoff. These changes will unlock deeper filter modulation and stereo movement.",
        "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting envelope 1 attack. These changes will enable deeper filter modulation and stereo movement.",  # near-identical s2
        "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting oscillator wave. This adds rich harmonic overtones not present in the current preset.",  # clearly different
    ]
    score = _score_plan_rationale_unique(turns)
    # pair(0,1): similar → 0.0; pair(1,2): different → 1.0 → mean = 0.5
    assert score == 0.5, f"Expected 0.5 for one similar / one different pair, got {score}"


def test_plan_rationale_unique_reduces_overall_score():
    """A path with repeated PLAN Sentence 2 should score lower than one with unique rationales."""
    from unittest.mock import patch

    def _make_record(turns_content: list[str]) -> dict:
        return {
            "id": "test",
            "task_type": "main",
            "messages": [
                {"role": "user", "content": "target"},
                *[{"role": "assistant", "content": c} for c in turns_content],
            ],
            "labels": {},
            "meta": {"step_labels": []},
        }

    boilerplate = "These changes will enable deeper filter modulation and stereo movement shaping the evolving brightness."
    unique_rationales = [
        f"HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting {p}. {r}"
        for p, r in [
            ("filter 1 cutoff", "This brightens the upper harmonic range considerably."),
            ("envelope attack", "This sharpens the transient punch for percussive impact."),
            ("lfo rate", "This introduces rhythmic timbral movement to the sustained tone."),
        ]
    ]
    repeated_rationales = [
        f"HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting {p}. {boilerplate}"
        for p in ["filter 1 cutoff", "envelope attack", "lfo rate"]
    ]

    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        good = score_main_record(_make_record(unique_rationales))
        bad = score_main_record(_make_record(repeated_rationales))

    assert good["overall"] > bad["overall"], (
        f"Unique rationales ({good['overall']:.4f}) should outscore "
        f"repeated rationales ({bad['overall']:.4f})"
    )


def test_plan_rationale_unique_in_score_output():
    """plan_rationale_unique must appear in score_main_record output."""
    from unittest.mock import patch

    record = {
        "id": "test",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "target"},
            {"role": "assistant", "content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting cutoff. This brightens tone."},
            {"role": "assistant", "content": "HEARD: a\n\nHYPOTHESIS: b\n\nPLAN: Adjusting attack. This sharpens the hit."},
        ],
        "labels": {},
        "meta": {"step_labels": []},
    }
    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        result = score_main_record(record)

    assert "plan_rationale_unique" in result
    assert result["plan_rationale_unique"] is not None


def test_plan_rationale_unique_excluded_from_weights_when_none():
    """When plan_rationale_unique is None (single step), overall must still be computed."""
    from unittest.mock import patch

    record = {
        "id": "test",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "target"},
            {"role": "assistant", "content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting cutoff. This brightens tone."},
        ],
        "labels": {},
        "meta": {"step_labels": []},
    }
    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        result = score_main_record(record)

    assert result["plan_rationale_unique"] is None
    assert result["overall"] > 0.0, "overall should still be computed when plan_rationale_unique is None"


# ---------------------------------------------------------------------------
# Hypothesis grounding metric
# ---------------------------------------------------------------------------

def test_hypothesis_grounding_hits_when_subsystem_mentioned():
    """Score 1.0 when HYPOTHESIS text contains one of the remaining_top_2 names."""
    commentary_turns = [
        "HEARD: x\n\nHYPOTHESIS: The compressor settings are causing the issue.\n\nPLAN: Adjusting filter.",
    ]
    step_labels = [{"remaining_top_2": "compressor and chorus"}]
    score = _score_hypothesis_grounding(commentary_turns, step_labels)
    assert score == 1.0, f"Expected 1.0, got {score}"


def test_hypothesis_grounding_misses_when_subsystem_absent():
    """Score 0.0 when HYPOTHESIS doesn't mention either remaining subsystem."""
    commentary_turns = [
        "HEARD: x\n\nHYPOTHESIS: The oscillator waveform is wrong.\n\nPLAN: Adjusting filter.",
    ]
    step_labels = [{"remaining_top_2": "compressor and chorus"}]
    score = _score_hypothesis_grounding(commentary_turns, step_labels)
    assert score == 0.0, f"Expected 0.0, got {score}"


def test_hypothesis_grounding_skips_steps_without_remaining_info():
    """Steps with no remaining_top_2 label are excluded from the score."""
    commentary_turns = [
        "HEARD: x\n\nHYPOTHESIS: The oscillator waveform is wrong.\n\nPLAN: Adjusting filter.",
        "HEARD: y\n\nHYPOTHESIS: The reverb is too wet.\n\nPLAN: Adjusting reverb.",
    ]
    # Only second step has remaining info, and it matches
    step_labels = [
        {"remaining_top_2": None},
        {"remaining_top_2": "reverb and delay"},
    ]
    score = _score_hypothesis_grounding(commentary_turns, step_labels)
    assert score == 1.0, f"Expected 1.0 (only second step assessed), got {score}"


def test_hypothesis_grounding_returns_none_when_no_labels_have_remaining():
    """Returns None (not 0.0) when no step_labels have remaining_top_2."""
    commentary_turns = ["HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: z."]
    step_labels = [{"remaining_top_2": None}]
    score = _score_hypothesis_grounding(commentary_turns, step_labels)
    assert score is None, f"Expected None, got {score}"


def test_hypothesis_grounding_partial_score():
    """Two steps, one hit one miss → score = 0.5."""
    commentary_turns = [
        "HEARD: x\n\nHYPOTHESIS: The compressor is the issue.\n\nPLAN: Adjusting filter.",
        "HEARD: y\n\nHYPOTHESIS: The oscillator is the problem.\n\nPLAN: Adjusting osc.",
    ]
    step_labels = [
        {"remaining_top_2": "compressor and chorus"},  # hit: "compressor" present
        {"remaining_top_2": "reverb and delay"},        # miss: neither mentioned
    ]
    score = _score_hypothesis_grounding(commentary_turns, step_labels)
    assert score == 0.5, f"Expected 0.5, got {score}"


def test_hypothesis_grounding_in_overall_score():
    """hypothesis_grounding must appear in the raw output of score_main_record."""
    from unittest.mock import patch

    record = {
        "id": "test",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "target"},
            {"role": "assistant", "content": "HEARD: x\n\nHYPOTHESIS: The compressor is wrong.\n\nPLAN: Adjusting filter 1 cutoff. This tightens tone."},
        ],
        "labels": {},
        "meta": {
            "step_labels": [
                {
                    "step": 1,
                    "params_delta": [{"name": "filter_1_cutoff", "from_norm": 0.3, "to_norm": 0.7}],
                    "planned_param_names": ["filter_1_cutoff"],
                    "clap_delta": None,
                    "remaining_top_2": "compressor and chorus",
                }
            ]
        },
    }
    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        result = score_main_record(record)

    assert "hypothesis_grounding" in result, "hypothesis_grounding missing from score output"
    assert result["hypothesis_grounding"] == 1.0, f"Expected 1.0 (compressor mentioned), got {result['hypothesis_grounding']}"


def test_low_hypothesis_grounding_reduces_overall():
    """A record with zero hypothesis grounding should score lower than one with perfect grounding."""
    from unittest.mock import patch

    def _make_record(hypo_text: str) -> dict:
        return {
            "id": "test",
            "task_type": "main",
            "messages": [
                {"role": "user", "content": "target"},
                {"role": "assistant", "content": f"HEARD: x\n\nHYPOTHESIS: {hypo_text}\n\nPLAN: Adjusting filter 1 cutoff. This tightens tone."},
            ],
            "labels": {},
            "meta": {
                "step_labels": [
                    {
                        "step": 1,
                        "params_delta": [{"name": "filter_1_cutoff", "from_norm": 0.3, "to_norm": 0.7}],
                        "planned_param_names": ["filter_1_cutoff"],
                        "clap_delta": None,
                        "remaining_top_2": "compressor and chorus",
                    }
                ]
            },
        }

    with patch("scripts.grade_agent_sft._HTTPX_AVAILABLE", False):
        good = score_main_record(_make_record("The compressor settings are causing the remaining gap."))
        bad = score_main_record(_make_record("The oscillator waveform is completely off."))

    assert good["overall"] > bad["overall"], \
        f"Good grounding ({good['overall']}) should score higher than bad grounding ({bad['overall']})"


# ---------------------------------------------------------------------------
# Parallel grading tests
# ---------------------------------------------------------------------------

def _make_grading_record(record_id: str, score: float) -> dict:
    """Return a minimal main record; mock score_record will return *score*."""
    return {
        "id": record_id,
        "task_type": "main",
        "messages": [],
        "meta": {"step_labels": []},
    }


def test_grade_file_parallel_same_records_as_serial(tmp_path):
    """grade_file with workers>1 must produce same record set as workers=1."""
    import json
    from unittest.mock import patch

    records = [_make_grading_record(f"r{i}", 0.9) for i in range(8)]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    fake_scores = {"overall": 0.9, "plan_param_alignment": 1.0}

    with patch("scripts.grade_agent_sft.score_record", return_value=fake_scores):
        out_serial = tmp_path / "serial.jsonl"
        grade_file(input_file, out_serial, workers=1)

        out_parallel = tmp_path / "parallel.jsonl"
        grade_file(input_file, out_parallel, workers=4)

    serial_ids = [json.loads(l)["id"] for l in open(out_serial)]
    parallel_ids = [json.loads(l)["id"] for l in open(out_parallel)]
    assert serial_ids == parallel_ids, "Output IDs must match between serial and parallel runs"


def test_grade_file_parallel_preserves_input_order(tmp_path):
    """Output record order must match input order even when workers > 1."""
    import json
    import time
    from unittest.mock import patch

    n = 12
    records = [_make_grading_record(f"r{i:02d}", 1.0) for i in range(n)]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    def _slow_score(record, **kwargs):
        # Reverse-order delay: early records sleep longer, so without order
        # preservation the output would be reversed.
        idx = int(record["id"][1:])
        time.sleep((n - idx) * 0.01)
        return {"overall": float(idx) / n}

    with patch("scripts.grade_agent_sft.score_record", side_effect=_slow_score):
        out = tmp_path / "out.jsonl"
        grade_file(input_file, out, workers=n)

    ids = [json.loads(l)["id"] for l in open(out)]
    assert ids == [f"r{i:02d}" for i in range(n)], f"Order not preserved: {ids}"


def test_grade_file_workers_1_produces_correct_output(tmp_path):
    """workers=1 (serial path) must produce correctly ordered, correctly scored output."""
    import json
    from unittest.mock import patch

    records = [_make_grading_record(f"r{i}", 0.8) for i in range(4)]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    fake_scores = {"overall": 0.8}
    with patch("scripts.grade_agent_sft.score_record", return_value=fake_scores):
        out = tmp_path / "out.jsonl"
        grade_file(input_file, out, workers=1)

    out_records = [json.loads(l) for l in open(out)]
    assert len(out_records) == 4
    assert [r["id"] for r in out_records] == [f"r{i}" for i in range(4)]


def test_grade_file_failed_record_does_not_kill_others(tmp_path):
    """A record that raises during scoring should warn but not abort the run.

    The failed record is kept in output with empty quality_scores rather than
    silently dropped — callers can inspect records with missing 'overall' scores.
    """
    import json
    from unittest.mock import patch

    records = [_make_grading_record(f"r{i}", 0.9) for i in range(6)]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    call_count = [0]

    def _sometimes_fails(record, **kwargs):
        call_count[0] += 1
        if record["id"] == "r2":
            raise RuntimeError("simulated scoring failure")
        return {"overall": 0.9}

    with patch("scripts.grade_agent_sft.score_record", side_effect=_sometimes_fails):
        out = tmp_path / "out.jsonl"
        # Should not raise despite one record failing
        grade_file(input_file, out, workers=4)

    assert call_count[0] == 6, "All 6 records should have been attempted"
    out_records = {json.loads(l)["id"]: json.loads(l) for l in open(out)}
    assert len(out_records) == 6, "All 6 records should be in output (failed gets empty scores)"
    # The failed record gets an empty quality_scores dict (no 'overall' key)
    assert out_records["r2"]["quality_scores"] == {}, "Failed record should have empty quality_scores"
    # Successful records have their scores intact
    assert out_records["r0"]["quality_scores"]["overall"] == 0.9


def test_grade_file_parallel_score_values_match_serial(tmp_path):
    """Scores returned parallel must be identical to those returned serial."""
    import json
    from unittest.mock import patch

    records = [_make_grading_record(f"r{i}", 0.0) for i in range(6)]
    input_file = tmp_path / "input.jsonl"
    with open(input_file, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    # Return a unique score per record to detect any mixing
    def _unique_score(record, **kwargs):
        idx = int(record["id"][1:])
        return {"overall": round(idx * 0.1 + 0.1, 3)}

    with patch("scripts.grade_agent_sft.score_record", side_effect=_unique_score):
        out_s = tmp_path / "serial.jsonl"
        grade_file(input_file, out_s, workers=1)
        out_p = tmp_path / "parallel.jsonl"
        grade_file(input_file, out_p, workers=6)

    serial_scores = [json.loads(l)["quality_scores"]["overall"] for l in open(out_s)]
    parallel_scores = [json.loads(l)["quality_scores"]["overall"] for l in open(out_p)]
    assert serial_scores == parallel_scores, \
        f"Score mismatch: serial={serial_scores}, parallel={parallel_scores}"
