"""Tests for score_search_v2_record and score_judge_v3_record."""
from __future__ import annotations

import json

from scripts.grade_agent_sft import score_search_v2_record, score_judge_v3_record


# ---------------------------------------------------------------------------
# search_v2 fixtures
# ---------------------------------------------------------------------------

def _make_search_v2(
    *,
    gt_in_shard: list[str] | None = None,
    gt_on_shortlist: list[str] | None = None,
    final_shortlist: list[str] | None = None,
    has_render_probes_call: bool = True,
    render_probes_style: str = "legacy",
    has_final_write: bool = True,
    last_role_assistant: bool = True,
    inject_snake_case: bool = False,
    inject_bold: bool = False,
) -> dict:
    gt_in_shard = gt_in_shard if gt_in_shard is not None else []
    gt_on_shortlist = gt_on_shortlist if gt_on_shortlist is not None else []
    final_shortlist = final_shortlist if final_shortlist is not None else ["A", "B"]

    messages = [
        {"role": "user", "content": "<audio>\nEvaluate slice 0-47..."},
    ]
    if has_render_probes_call:
        if render_probes_style == "dawdreamer":
            render_cmd = "python -c 'import dawdreamer; render_vital_preset(preset, \"/tmp/x/wt_0000_Init.wav\", midi_notes)'"
        else:
            render_cmd = "python skills/vital/scripts/render_probes.py --idxs 0,1,2 --out-dir /tmp/x"
        messages.append({"role": "assistant", "content": "Rendering batch 1."})
        messages.append({
            "role": "tool_call",
            "content": json.dumps({
                "name": "bash",
                "arguments": {"command": render_cmd},
            }),
        })
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "rendered": [{"name": "A", "audio": "<audio>"}]}),
        })
    # Some Stage-2 reasoning
    middle = "'A': Good candidate. Selected."
    if inject_snake_case:
        middle += " has osc_1_level_knob_parameter set high."
    if inject_bold:
        middle += "\n**PLAN:** final."
    messages.append({"role": "assistant", "content": middle})
    if last_role_assistant:
        if has_final_write:
            shortlist_str = ", ".join(f'"{n}"' for n in final_shortlist)
            messages.append({
                "role": "assistant",
                "content": (
                    f"Final narration.\n\nShortlist: [{shortlist_str}]. "
                    f"{len(final_shortlist)} candidate(s) flagged for the judge agent."
                ),
            })
        else:
            messages.append({
                "role": "assistant",
                "content": "Done evaluating this shard.",
            })

    return {
        "id": "s1_search",
        "task_type": "search_v2",
        "tools": "[]",
        "messages": messages,
        "audios": [],
        "meta": {
            "pipeline_version": "v2_search",
            "sample_id": "s1",
            "archetype": "keys",
            "shard_size": 48,
            "shard_start": 0,
            "shard_end": 48,
            "n_batches": 6,
            "candidates_per_batch": 8,
            "gt_in_shard": gt_in_shard,
            "final_shortlist": final_shortlist,
            "gt_on_shortlist": gt_on_shortlist,
            "shortlist_output_file": "/tmp/agents/s1_search_1.json",
        },
    }


def test_search_v2_happy_path_full_gt_recovery():
    record = _make_search_v2(
        gt_in_shard=["01 Basic Shapes"],
        gt_on_shortlist=["01 Basic Shapes"],
        final_shortlist=["01 Basic Shapes", "Pink Noise"],
    )
    scores = score_search_v2_record(record)
    assert scores["gt_recovery"] == 1.0
    assert scores["result_communicated"] == 1.0
    assert scores["closing_assistant"] == 1.0
    assert scores["has_render_probes"] == 1.0
    assert scores["shortlist_nonempty"] == 1.0
    assert scores["snake_case_clean"] == 1.0
    assert scores["format_consistent"] == 1.0
    assert scores["overall"] == 1.0


def test_search_v2_partial_gt_recovery():
    record = _make_search_v2(
        gt_in_shard=["GT1", "GT2"],
        gt_on_shortlist=["GT1"],
    )
    scores = score_search_v2_record(record)
    assert scores["gt_recovery"] == 0.5


def test_search_v2_gt_recovery_none_when_no_gts_in_shard():
    record = _make_search_v2(gt_in_shard=[], gt_on_shortlist=[])
    scores = score_search_v2_record(record)
    assert scores["gt_recovery"] is None
    # Overall should still compute from other axes
    assert scores["overall"] > 0


def test_search_v2_missing_final_write_is_penalised():
    record = _make_search_v2(has_final_write=False)
    scores = score_search_v2_record(record)
    assert scores["result_communicated"] == 0.0
    assert scores["overall"] < 0.9


def test_search_v2_dawdreamer_render_detected():
    record = _make_search_v2(render_probes_style="dawdreamer")
    scores = score_search_v2_record(record)
    assert scores["has_render_probes"] == 1.0


def test_search_v2_missing_render_probes_is_penalised():
    record = _make_search_v2(has_render_probes_call=False)
    scores = score_search_v2_record(record)
    assert scores["has_render_probes"] == 0.0


def test_search_v2_empty_shortlist_is_penalised():
    record = _make_search_v2(gt_in_shard=["GT1"], final_shortlist=[])
    scores = score_search_v2_record(record)
    assert scores["shortlist_nonempty"] == 0.0


def test_search_v2_snake_case_dings():
    record = _make_search_v2(inject_snake_case=True)
    scores = score_search_v2_record(record)
    assert scores["snake_case_clean"] < 1.0


def test_search_v2_bold_headers_ding():
    record = _make_search_v2(inject_bold=True)
    scores = score_search_v2_record(record)
    assert scores["format_consistent"] < 1.0


# ---------------------------------------------------------------------------
# judge_v3 fixtures
# ---------------------------------------------------------------------------

def _make_judge_v3(
    *,
    pool: list[str] | None = None,
    n_osc_slots: int = 2,
    selected_tuple: list[str] | None = None,
    gts_in_pool: list[str] | None = None,
    gt_wavetable_names: list[str] | None = None,
    judge_correct: bool = True,
    has_render_probes: bool = True,
    has_output_write: bool = True,
    last_role_assistant: bool = True,
    mentions_candidates: bool = True,
) -> dict:
    pool = pool if pool is not None else ["A", "B", "C", "D"]
    selected_tuple = selected_tuple if selected_tuple is not None else ["A", "B"]
    gts_in_pool = gts_in_pool if gts_in_pool is not None else []
    gt_wavetable_names = gt_wavetable_names if gt_wavetable_names is not None else []

    messages = [
        {"role": "user", "content": f"<audio>\nPool candidates: {json.dumps(pool)}. Pick {n_osc_slots}."},
    ]
    if has_render_probes:
        messages.append({"role": "assistant", "content": f"Rendering probes for all {len(pool)} pool candidates."})
        messages.append({
            "role": "tool_call",
            "content": json.dumps({
                "name": "bash",
                "arguments": {"command": f"python skills/vital/scripts/render_probes.py --names {','.join(pool)} --out-dir /tmp/judge"},
            }),
        })
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "rendered": [{"name": n, "audio": "<audio>"} for n in pool]}),
        })
    deliberation = "Listening now."
    if mentions_candidates:
        for n in pool:
            deliberation += f"\n'{n}': Candidate evaluated."
    deliberation += f"\n\nSELECTED: [{', '.join(repr(n) for n in selected_tuple)}]: rationale."
    messages.append({"role": "assistant", "content": deliberation})
    if last_role_assistant:
        tuple_str = ", ".join(repr(n) for n in selected_tuple)
        if has_output_write:
            messages.append({
                "role": "assistant",
                "content": f"Final tuple: [{tuple_str}]. The main agent can now apply the chosen wavetables.",
            })
        else:
            messages.append({
                "role": "assistant",
                "content": "Done evaluating pool.",
            })

    return {
        "id": "s1_judge",
        "task_type": "judge",
        "tools": "[]",
        "messages": messages,
        "audios": [],
        "meta": {
            "pipeline_version": "v3_judge",
            "sample_id": "s1",
            "pool": pool,
            "pool_size": len(pool),
            "n_osc_slots": n_osc_slots,
            "active_oscs": list(range(n_osc_slots)),
            "gt_wavetable_names": gt_wavetable_names,
            "gts_in_pool": gts_in_pool,
            "selected_tuple": selected_tuple,
            "judge_correct": judge_correct,
            "output_file": "/tmp/judge/judge_s1.json",
        },
    }


def test_judge_v3_happy_path():
    record = _make_judge_v3()
    scores = score_judge_v3_record(record)
    assert scores["judge_correct"] == 1.0
    assert scores["tuple_size_correct"] == 1.0
    assert scores["tuple_names_in_pool"] == 1.0
    assert scores["result_communicated"] == 1.0
    assert scores["pool_candidates_discussed"] == 1.0
    assert scores["has_render_probes"] == 1.0
    assert scores["closing_assistant"] == 1.0
    assert scores["overall"] == 1.0


def test_judge_v3_incorrect_picks():
    record = _make_judge_v3(judge_correct=False)
    scores = score_judge_v3_record(record)
    assert scores["judge_correct"] == 0.0
    assert scores["overall"] < 1.0


def test_judge_v3_tuple_size_mismatch():
    record = _make_judge_v3(n_osc_slots=3, selected_tuple=["A", "B"])
    scores = score_judge_v3_record(record)
    assert scores["tuple_size_correct"] == 0.0


def test_judge_v3_hallucinated_tuple_name():
    record = _make_judge_v3(
        pool=["A", "B", "C"],
        selected_tuple=["A", "HALLUCINATED"],
    )
    scores = score_judge_v3_record(record)
    assert scores["tuple_names_in_pool"] == 0.0


def test_judge_v3_missing_output_write_is_penalised():
    record = _make_judge_v3(has_output_write=False)
    scores = score_judge_v3_record(record)
    assert scores["result_communicated"] == 0.0


def test_judge_v3_pool_discussion_partial():
    # Use distinct multi-char names so substring matching isn't fooled by common letters.
    record = _make_judge_v3(
        pool=["alpha_wave", "beta_drift", "gamma_chord", "delta_pulse", "epsilon_tone"],
        mentions_candidates=False,  # deliberation only mentions SELECTED line's names
        selected_tuple=["alpha_wave", "beta_drift"],
    )
    scores = score_judge_v3_record(record)
    # Only alpha_wave, beta_drift mentioned (from SELECTED line), out of 5 pool = 0.4
    assert scores["pool_candidates_discussed"] == 0.4


def test_judge_v3_has_render_probes_check():
    record = _make_judge_v3(has_render_probes=False)
    scores = score_judge_v3_record(record)
    assert scores["has_render_probes"] == 0.0
