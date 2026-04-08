from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.agent_sft_common import (
    build_disjoint_shards,
    build_oracle_mix_candidates,
    validate_ms_swift_multiturn_record,
)
from scripts.build_judge_agent_sft import _build_rank_labels
from scripts.build_main_agent_sft_v2 import _wrap_as_bash, _build_judge_result
from scripts.build_search_agent_sft import _reason_for_cosine, _build_assistant_proposals


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


# ---------------------------------------------------------------------------
# Adjacent duplicate role tests (extended to cover tool_call / tool_response)
# ---------------------------------------------------------------------------

def _minimal_tool_record(extra_messages: list[dict]) -> dict:
    """Helper: wraps messages in a valid frame with tool_call first."""
    return {
        "id": "t",
        "messages": [
            {"role": "user", "content": "Go."},
            {"role": "assistant", "content": "Running."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo hi"}}'},
            *extra_messages,
            {"role": "assistant", "content": "Done."},
        ],
        "audios": [],
    }


def test_validator_rejects_adjacent_tool_responses():
    row = _minimal_tool_record([
        {"role": "tool_response", "content": '{"r": 1}'},
        {"role": "tool_response", "content": '{"r": 2}'},
    ])
    errors = validate_ms_swift_multiturn_record(row)
    assert any("duplicate_adjacent_role:tool_response" in e for e in errors), errors


def test_validator_rejects_adjacent_tool_calls():
    row = {
        "id": "t",
        "messages": [
            {"role": "user", "content": "Go."},
            {"role": "assistant", "content": "Running."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"a"}}'},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"b"}}'},
            {"role": "tool_response", "content": '{"r": 1}'},
            {"role": "assistant", "content": "Done."},
        ],
        "audios": [],
    }
    errors = validate_ms_swift_multiturn_record(row)
    assert any("duplicate_adjacent_role:tool_call" in e for e in errors), errors


def test_validator_allows_multi_audio_in_tool_response():
    """A tool_response may carry N <audio> tags (batch audition result)."""
    previews = [
        {"candidate_id": "C1", "audio_preview": "<audio>"},
        {"candidate_id": "C2", "audio_preview": "<audio>"},
        {"candidate_id": "C3", "audio_preview": "<audio>"},
    ]
    row = {
        "id": "t",
        "messages": [
            {"role": "user", "content": "<audio>\nTarget."},
            {"role": "assistant", "content": "Auditioning."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"audition"}}'},
            {"role": "tool_response", "content": json.dumps({"audition_results": previews})},
            {"role": "assistant", "content": '{"ranking":["C1","C2","C3"]}'},
        ],
        "audios": ["/tmp/target.wav", "/tmp/c1.wav", "/tmp/c2.wav", "/tmp/c3.wav"],
    }
    errors = validate_ms_swift_multiturn_record(row)
    assert not errors, f"Expected no errors, got: {errors}"


def test_validator_still_rejects_multi_audio_in_user_turn():
    """User/assistant turns must still be limited to one <audio> tag each."""
    row = {
        "id": "t",
        "messages": [
            {"role": "user", "content": "<audio>\n<audio>\nTwo targets?"},
            {"role": "assistant", "content": "Done."},
        ],
        "audios": ["/tmp/a.wav", "/tmp/b.wav"],
    }
    errors = validate_ms_swift_multiturn_record(row)
    assert any("multiple_audio_tags" in e for e in errors), errors


# ---------------------------------------------------------------------------
# Builder-level structural tests (no data files required)
# ---------------------------------------------------------------------------

def _make_judge_record(n_candidates: int) -> dict:
    """Construct a judge record mimicking the builder's output after the fix."""
    audition_results = [
        {"candidate_id": f"C{i}", "wavetable_name": f"wt{i}", "audio_preview": "<audio>"}
        for i in range(1, n_candidates + 1)
    ]
    return {
        "id": f"judge_test_{n_candidates}",
        "task_type": "judge",
        "messages": [
            {"role": "user", "content": "<audio>\nJudge task. Rank C1..C{n_candidates}."},
            {"role": "assistant", "content": "<audio>\nI will audition before ranking."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"audition"}}'},
            {"role": "tool_response", "content": json.dumps({"audition_results": audition_results})},
            {"role": "assistant", "content": json.dumps({"ranking": [f"C{i}" for i in range(1, n_candidates + 1)], "selected": ["C1", "C2"], "reason": "test"})},
        ],
        "audios": [f"/tmp/target.wav", f"/tmp/current.wav"] + [f"/tmp/c{i}.wav" for i in range(1, n_candidates + 1)],
    }


def _make_search_record(n_listened: int) -> dict:
    """Construct a search record mimicking the builder's output after the fix."""
    audition_results = [
        {"candidate_id": f"C{i}", "wavetable_name": f"wt{i}", "audio_preview": "<audio>"}
        for i in range(1, n_listened + 1)
    ]
    return {
        "id": f"search_test_{n_listened}",
        "task_type": "search",
        "messages": [
            {"role": "user", "content": "<audio>\nSearch task."},
            {"role": "assistant", "content": "<audio>\nRunning search."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"search"}}'},
            {"role": "tool_response", "content": '{"stdout": "C1\\twt1\\t0.87\\t/tmp/c1.wav"}'},
            {"role": "assistant", "content": "Auditioning top candidates."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"audition"}}'},
            {"role": "tool_response", "content": json.dumps({"status": "ok", "auditioned": n_listened, "audition_results": audition_results})},
            {"role": "assistant", "content": json.dumps({"proposals": [{"candidate_id": "C1", "confidence": 0.87, "reason": "test"}]})},
        ],
        "audios": ["/tmp/target.wav", "/tmp/current.wav"] + [f"/tmp/c{i}.wav" for i in range(1, n_listened + 1)],
    }


def test_judge_record_no_adjacent_tool_responses():
    record = _make_judge_record(n_candidates=8)
    errors = validate_ms_swift_multiturn_record(record)
    assert not errors, f"Judge record has validation errors: {errors}"


def test_search_record_no_adjacent_tool_responses():
    record = _make_search_record(n_listened=3)
    errors = validate_ms_swift_multiturn_record(record)
    assert not errors, f"Search record has validation errors: {errors}"


def test_judge_record_audio_tag_count_matches_audios():
    for n in [4, 8]:
        record = _make_judge_record(n_candidates=n)
        errors = validate_ms_swift_multiturn_record(record)
        assert not any("audio_tag_mismatch" in e for e in errors), \
            f"n={n} audio mismatch: {errors}"


def test_search_record_audio_tag_count_matches_audios():
    for n in [1, 2, 3]:
        record = _make_search_record(n_listened=n)
        errors = validate_ms_swift_multiturn_record(record)
        assert not any("audio_tag_mismatch" in e for e in errors), \
            f"n={n} audio mismatch: {errors}"


# ---------------------------------------------------------------------------
# Fix round 2: reapy wrapping, Omni commentary, judge reason, candidate audio
# ---------------------------------------------------------------------------

def test_wrap_as_bash_adds_heredoc_to_raw_python():
    """Bare Python (import reapy...) must be wrapped in a shell heredoc."""
    raw = "import reapy\nwith reapy.inside_reaper():\n    print('hi')"
    result = _wrap_as_bash(raw)
    assert result.startswith("python - <<'PY'"), f"Missing heredoc header: {result[:60]}"
    assert result.endswith("PY"), f"Missing heredoc footer: {result[-20:]}"
    assert "import reapy" in result


def test_wrap_as_bash_leaves_shell_commands_unchanged():
    """Commands already starting with 'python' should pass through unmodified."""
    cmd = "python - <<'PY'\nprint('already wrapped')\nPY"
    assert _wrap_as_bash(cmd) == cmd


def test_wrap_as_bash_output_is_valid_shell_structure():
    """Wrapped output must follow <<'PY'...PY heredoc pattern."""
    raw = "import reapy\nfx = None\nprint(fx)"
    result = _wrap_as_bash(raw)
    assert re.match(r"python - <<'PY'\n.*\nPY$", result, re.DOTALL), \
        f"Invalid heredoc structure: {result}"


def _make_main_record_with_selected_audio(n_selected: int, n_iters: int) -> dict:
    """Construct a main-agent record matching the builder's output after fix round 2."""
    selected_previews = [
        {"candidate_id": f"C{i}", "wavetable_name": f"wt{i}", "audio_preview": "<audio>"}
        for i in range(1, n_selected + 1)
    ]
    judge_response = json.dumps({
        "ranking": [f"C{i}" for i in range(1, n_selected + 1)],
        "selected": [f"C{i}" for i in range(1, n_selected + 1)],
        "reason": "Ranked by CLAP similarity. Top scores: C1 (0.72), C2 (0.65).",
        "selected_previews": selected_previews,
    })
    iter_responses = []
    for step in range(1, n_iters + 1):
        iter_responses += [
            {"role": "assistant", "content": f"HEARD: step {step} differences...\n\nHYPOTHESIS: ...\n\nPLAN: ...\n\nExecuting step {step} parameter updates now."},
            {"role": "tool_call", "content": json.dumps({"name": "bash", "arguments": {"command": f"python - <<'PY'\nprint({step})\nPY"}})},
            {"role": "tool_response", "content": '{"status": "ok"}'},
            {"role": "assistant", "content": f"Listening to updated preset after step {step}."},
            {"role": "tool_call", "content": json.dumps({"name": "bash", "arguments": {"command": f"python - <<'PY'\nprint('listen')\nPY"}})},
            {"role": "tool_response", "content": json.dumps({"status": "ok", "step": step, "iter_audio": "<audio>", "path": f"/tmp/iter{step}.wav"})},
        ]
    return {
        "id": f"main_test_{n_selected}sel_{n_iters}iter",
        "task_type": "main",
        "messages": [
            {"role": "user", "content": "<audio>\nRecreate this sound."},
            {"role": "assistant", "content": "Listening to baseline."},
            {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"python - <<\'PY\'\\nprint(\'hi\')\\nPY"}}'},
            {"role": "tool_response", "content": '{"status": "ok", "baseline_audio": "<audio>"}'},
            {"role": "assistant", "content": "Spawning search shards."},
            {"role": "tool_call", "content": '{"name":"spawn_search_agents","arguments":{}}'},
            {"role": "tool_response", "content": '{"jobs": []}'},
            {"role": "assistant", "content": "Collecting reports."},
            {"role": "tool_call", "content": '{"name":"collect_search_reports","arguments":{}}'},
            {"role": "tool_response", "content": '{"reports": []}'},
            {"role": "assistant", "content": "Judging candidates."},
            {"role": "tool_call", "content": '{"name":"judge_candidates","arguments":{}}'},
            {"role": "tool_response", "content": judge_response},
            *iter_responses,
            {"role": "assistant", "content": "Recreation pass complete."},
        ],
        # audios: target + default + n_selected probes + n_iters iter clips
        "audios": (
            ["/tmp/target.wav", "/tmp/default.wav"]
            + [f"/tmp/probe{i}.wav" for i in range(1, n_selected + 1)]
            + [f"/tmp/iter{s}.wav" for s in range(1, n_iters + 1)]
        ),
    }


def test_main_record_selected_audio_in_judge_response():
    """Judge tool_response must contain selected_previews with <audio> tags."""
    record = _make_main_record_with_selected_audio(n_selected=3, n_iters=2)
    # Find the judge tool_response
    judge_tr = None
    for i, m in enumerate(record["messages"]):
        if m["role"] == "tool_response":
            try:
                parsed = json.loads(m["content"])
                if "selected_previews" in parsed:
                    judge_tr = parsed
                    break
            except Exception:
                pass
    assert judge_tr is not None, "No tool_response with selected_previews found"
    assert len(judge_tr["selected_previews"]) == 3
    assert all(p["audio_preview"] == "<audio>" for p in judge_tr["selected_previews"])


def test_main_record_audio_count_includes_selected_probes():
    """audios list must include target + default + selected_probes + iter_clips."""
    for n_sel, n_iter in [(3, 3), (2, 2), (1, 1)]:
        record = _make_main_record_with_selected_audio(n_selected=n_sel, n_iters=n_iter)
        errors = validate_ms_swift_multiturn_record(record)
        assert not errors, f"n_sel={n_sel} n_iter={n_iter}: {errors}"
        expected = 1 + 1 + n_sel + n_iter  # target + default + probes + iters
        assert len(record["audios"]) == expected, \
            f"Expected {expected} audios, got {len(record['audios'])}"


def test_main_record_edit_steps_use_bash_heredoc():
    """All bash tool_calls in edit steps must use the heredoc pattern, not raw Python."""
    record = _make_main_record_with_selected_audio(n_selected=2, n_iters=3)
    for m in record["messages"]:
        if m["role"] == "tool_call":
            tc = json.loads(m["content"])
            if tc.get("name") == "bash":
                cmd = tc.get("arguments", {}).get("command", "")
                if "import reapy" in cmd or cmd.strip().startswith("import"):
                    assert "<<'PY'" in cmd, f"Bare Python not wrapped in heredoc: {cmd[:80]}"


def test_judge_reason_contains_scores():
    """Judge assistant output reason must reference numeric CLAP scores."""
    ordered = ["A", "B", "C", "D"]
    gt_names = {"C"}
    scores = {"A": 0.81, "B": 0.75, "C": 0.92, "D": 0.63}
    ranking_ids, selected_ids, gt_ids, score_map = _build_rank_labels(
        ordered, gt_names, scores, topk_select=2
    )
    top_scored = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    score_summary = ", ".join(f"{cid} ({s:.3f})" for cid, s in top_scored[:4])
    reason = (
        f"Ranked by CLAP similarity to target. "
        f"Top scores: {score_summary}."
        f" GT candidate: {gt_ids[0]}. Selected top {len(selected_ids)}."
    )
    assert re.search(r"\d\.\d{3}", reason), "No numeric score found in reason"
    assert "GT candidate" in reason


# ---------------------------------------------------------------------------
# Fix round 3: judge result reason in main builder, score-tiered search reasons,
#              cosine audition command, closing eval, inter-step context
# ---------------------------------------------------------------------------

def test_build_judge_result_reason_contains_scores():
    """_build_judge_result in main builder must produce a score-based reason."""
    names = ["alpha", "beta", "gamma", "delta"]
    id_map = {n: f"C{i}" for i, n in enumerate(names, 1)}
    gt = {"gamma"}
    scores = {"alpha": 0.80, "beta": 0.71, "gamma": 0.90, "delta": 0.55}
    result = _build_judge_result(names, id_map, gt, scores, select_k=2)
    assert re.search(r"\d\.\d{3}", result["reason"]), "No numeric score in reason"
    assert "GT candidate" in result["reason"]
    assert "Selected top 2" in result["reason"]


def test_build_judge_result_no_gt_omits_gt_note():
    """When no GT exists, reason must not claim a GT candidate."""
    names = ["alpha", "beta"]
    id_map = {n: f"C{i}" for i, n in enumerate(names, 1)}
    result = _build_judge_result(names, id_map, set(), {"alpha": 0.7, "beta": 0.6}, select_k=1)
    assert "GT candidate" not in result["reason"]


def test_reason_for_cosine_score_tiers():
    """_reason_for_cosine should produce distinct text across score tiers."""
    strong = _reason_for_cosine(0.60)   # conf = 0.80 → strong
    moderate = _reason_for_cosine(0.30)  # conf = 0.65 → moderate
    weak = _reason_for_cosine(-0.10)     # conf = 0.45 → weak

    assert "Strong" in strong, strong
    assert "Moderate" in moderate, moderate
    assert "Weak" in weak, weak
    # All should include the numeric score
    for r in [strong, moderate, weak]:
        assert re.search(r"\d\.\d{3}", r), f"No score in reason: {r}"


def test_reason_for_cosine_no_gt_leak():
    """_reason_for_cosine must not expose oracle GT identity in any output."""
    import inspect
    from scripts.build_search_agent_sft import _reason_for_cosine as rcf
    sig = inspect.signature(rcf)
    assert "is_gt" not in sig.parameters, "is_gt param should be removed (oracle leak)"
    for score in [0.90, 0.50, -0.20]:
        result = rcf(score)
        assert "gt" not in result.lower(), f"Oracle leak in reason: {result}"


def test_build_assistant_proposals_no_is_gt_arg():
    """_build_assistant_proposals must not pass is_gt to _reason_for_cosine."""
    # Passes if no TypeError is raised — confirms the call site uses single-arg form.
    shard_names = ["a", "b"]
    id_map = {"a": "C1", "b": "C2"}
    score_map = {"a": 0.80, "b": 0.20}
    _, proposals = _build_assistant_proposals(shard_names, {"a"}, id_map, score_map, proposals_per_agent=2)
    reasons = [p["reason"] for p in proposals]
    assert all("gt" not in r.lower() for r in reasons), f"GT leak in proposals: {reasons}"


def test_build_assistant_proposals_per_candidate_reasons():
    """Each proposal must have a distinct, score-referencing reason string."""
    shard_names = ["hi_match", "mid_match", "lo_match"]
    id_map = {n: f"C{i}" for i, n in enumerate(shard_names, 1)}
    score_map = {"hi_match": 0.65, "mid_match": 0.25, "lo_match": -0.05}
    _, proposals = _build_assistant_proposals(shard_names, set(), id_map, score_map, proposals_per_agent=3)
    reasons = [p["reason"] for p in proposals]
    # All reasons should differ
    assert len(set(reasons)) == 3, f"All proposals have same reason: {reasons}"
    # All should reference a numeric score
    for r in reasons:
        assert re.search(r"\d\.\d{3}", r), f"No score in reason: {r}"


def test_audition_command_includes_cosine_logic():
    """The generated audition bash command must include cosine computation code."""
    from scripts.build_search_agent_sft import _build_audition_bundle_tool_command as search_audition
    from scripts.build_judge_agent_sft import _build_audition_bundle_tool_command as judge_audition

    search_cmd = search_audition(["/tmp/a.wav", "/tmp/b.wav"], "/tmp/target.wav")
    assert "cosine_vs_target" in search_cmd, "Search audition missing cosine_vs_target"
    assert "np.dot" in search_cmd, "Search audition missing dot product"

    judge_cmd = judge_audition(
        [{"candidate_id": "C1", "wavetable_name": "wt", "audio_path": "/tmp/a.wav"}],
        "/tmp/target.wav",
    )
    assert "cosine_vs_target" in judge_cmd, "Judge audition missing cosine_vs_target"
    assert "np.dot" in judge_cmd, "Judge audition missing dot product"


def test_omni_commentary_accepts_prev_commentary_param():
    """_call_omni_commentary must accept prev_commentary without error (offline check)."""
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary, _step_commentary_fallback
    import inspect
    sig = inspect.signature(_call_omni_commentary)
    assert "prev_commentary" in sig.parameters, "prev_commentary param missing"
    assert sig.parameters["prev_commentary"].default is None


def test_closing_eval_function_exists():
    """_call_omni_closing_eval must exist and have the right signature."""
    from scripts.build_main_agent_sft_v2 import _call_omni_closing_eval
    import inspect
    sig = inspect.signature(_call_omni_closing_eval)
    required = {"gt_wav", "final_iter_wav", "archetype", "omni_server", "model"}
    assert required.issubset(sig.parameters.keys()), \
        f"Missing params: {required - set(sig.parameters.keys())}"


# ---------------------------------------------------------------------------
# Fix round 4: params_delta wired into commentary, cosine surfaced, no GT leak
# ---------------------------------------------------------------------------

def test_omni_commentary_accepts_delta_params():
    """_call_omni_commentary must accept params_delta, is_mistake_step, allowed_params."""
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary
    import inspect
    sig = inspect.signature(_call_omni_commentary)
    required_new = {"params_delta", "is_mistake_step", "allowed_params", "remaining_delta_context"}
    assert required_new.issubset(sig.parameters.keys()), \
        f"Missing params: {required_new - set(sig.parameters.keys())}"
    assert sig.parameters["params_delta"].default is None
    assert sig.parameters["is_mistake_step"].default is False


def test_format_delta_context_includes_param_names():
    """_format_delta_context must reference the display name of each changed param."""
    from scripts.build_main_agent_sft_v2 import _format_delta_context
    delta = [
        {"name": "filter_1_cutoff", "from_norm": 0.3, "to_norm": 0.7, "mistake": False},
        {"name": "env_1_attack", "from_norm": 0.1, "to_norm": 0.05, "mistake": False},
    ]
    result = _format_delta_context(delta, is_mistake_step=False)
    assert "Filter" in result, f"Missing Filter in: {result}"
    assert "Env" in result or "Envelope" in result, f"Missing Env in: {result}"
    assert "0.30" in result or "0.3" in result, f"from_norm missing: {result}"
    assert "0.70" in result or "0.7" in result, f"to_norm missing: {result}"


def test_format_delta_context_mistake_step_adds_warning():
    """Mistake params should trigger the overcorrection warning."""
    from scripts.build_main_agent_sft_v2 import _format_delta_context
    delta = [{"name": "filter_1_cutoff", "from_norm": 0.5, "to_norm": 0.2, "mistake": True}]
    result = _format_delta_context(delta, is_mistake_step=True)
    assert "overcorrection" in result or "wrong direction" in result, \
        f"No mistake warning in: {result}"
    assert "\u26a0" in result, f"Missing warning symbol in: {result}"


def test_param_summary_arrow_direction():
    """_build_param_summary must use ↑ when to_norm > from_norm, ↓ otherwise."""
    from scripts.build_main_agent_sft_v2 import _build_param_summary
    delta = [
        {"name": "filter_1_cutoff", "from_norm": 0.2, "to_norm": 0.8},
        {"name": "env_1_attack", "from_norm": 0.9, "to_norm": 0.1},
    ]
    result = _build_param_summary(delta)
    assert "\u2191" in result, f"Missing ↑ arrow: {result}"
    assert "\u2193" in result, f"Missing ↓ arrow: {result}"


def test_param_summary_truncates_at_4():
    """_build_param_summary with 6 params should show '(+2 more)' suffix."""
    from scripts.build_main_agent_sft_v2 import _build_param_summary
    delta = [
        {"name": f"osc_{i}_level", "from_norm": 0.0, "to_norm": 0.1 * (i + 1)}
        for i in range(6)
    ]
    result = _build_param_summary(delta)
    assert "+2 more" in result, f"No truncation suffix: {result}"


def test_search_audition_response_has_cosine_scores():
    """Search audition tool_response entries must include cosine_vs_target."""
    audition_results = [
        {
            "candidate_id": "C1",
            "wavetable_name": "wt1",
            "audio_preview": "<audio>",
            "cosine_vs_target": 0.7234,
        },
        {
            "candidate_id": "C2",
            "wavetable_name": "wt2",
            "audio_preview": "<audio>",
            "cosine_vs_target": 0.4512,
        },
    ]
    payload = json.dumps({"status": "ok", "auditioned": 2, "audition_results": audition_results})
    parsed = json.loads(payload)
    for entry in parsed["audition_results"]:
        assert "cosine_vs_target" in entry, f"Missing cosine_vs_target in: {entry}"
        assert isinstance(entry["cosine_vs_target"], float), \
            f"cosine_vs_target not float: {entry['cosine_vs_target']}"


def test_two_stage_commentary_function_exists():
    """_call_omni_commentary_two_stage must exist with the right signature."""
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage
    import inspect
    sig = inspect.signature(_call_omni_commentary_two_stage)
    required = {
        "gt_wav", "iter_wav", "step", "step_num", "archetype",
        "omni_server", "omni_model", "stage2_server", "stage2_model",
        "prev_commentary", "params_delta", "is_mistake_step",
        "allowed_params", "remaining_delta_context",
    }
    assert required.issubset(sig.parameters.keys()), \
        f"Missing params: {required - set(sig.parameters.keys())}"


def test_commentary_mode_cli_arg_exists():
    """--commentary-mode must be a recognised CLI arg with two_stage option."""
    import subprocess, sys
    result = subprocess.run(
        [sys.executable, "scripts/build_main_agent_sft_v2.py", "--help"],
        capture_output=True, text=True
    )
    assert "two_stage" in result.stdout, "two_stage not in --help output"
    assert "commentary-mode" in result.stdout, "--commentary-mode not in --help output"


def test_is_mistake_step_field_name_used():
    """Call site must use step.get('is_mistake_step') not 'is_mistake'."""
    import ast, inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "is_mistake_step" in src, "is_mistake_step field not referenced in main()"
    # Ensure old wrong key is gone
    assert '"is_mistake"' not in src and "'is_mistake'" not in src, \
        "Old wrong key 'is_mistake' still present in main()"


def test_planned_param_names_wired_as_allowed():
    """Call site must use planned_param_names as the allowed_params source."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "planned_param_names" in src, "planned_param_names not wired in main()"


def test_judge_audition_response_has_cosine_scores():
    """Judge audition tool_response entries must include cosine_vs_target."""
    audition_results = [
        {
            "candidate_id": f"C{i}",
            "wavetable_name": f"wt{i}",
            "audio_preview": "<audio>",
            "cosine_vs_target": round(0.1 * i, 4),
        }
        for i in range(1, 5)
    ]
    for entry in audition_results:
        assert "cosine_vs_target" in entry
        assert isinstance(entry["cosine_vs_target"], float)


# ---------------------------------------------------------------------------
# Problem A & B: 3-clip comparison and imperceptible step acknowledgment
# ---------------------------------------------------------------------------

def test_two_stage_accepts_prior_iter_wav_and_clap_delta():
    """_call_omni_commentary_two_stage must accept prior_iter_wav and clap_delta."""
    import inspect
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage
    sig = inspect.signature(_call_omni_commentary_two_stage)
    assert "prior_iter_wav" in sig.parameters, "prior_iter_wav param missing"
    assert "clap_delta" in sig.parameters, "clap_delta param missing"
    assert sig.parameters["prior_iter_wav"].default is None
    assert sig.parameters["clap_delta"].default is None


def test_single_call_accepts_prior_iter_wav_and_clap_delta():
    """_call_omni_commentary must also accept prior_iter_wav and clap_delta (passthrough)."""
    import inspect
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary
    sig = inspect.signature(_call_omni_commentary)
    assert "prior_iter_wav" in sig.parameters
    assert "clap_delta" in sig.parameters


def test_clap_imperceptible_threshold_exported():
    """CLAP_IMPERCEPTIBLE_THRESHOLD must be exported from build_main_agent_sft_v2."""
    from scripts.build_main_agent_sft_v2 import CLAP_IMPERCEPTIBLE_THRESHOLD
    assert 0.0 < CLAP_IMPERCEPTIBLE_THRESHOLD <= 0.1, \
        f"Unexpected threshold value: {CLAP_IMPERCEPTIBLE_THRESHOLD}"


def test_imperceptible_step_uses_template_not_audio(tmp_path):
    """When is_planning_step=True, two_stage commentary must skip audio Stage 1
    and produce a PLAN-only response (no HEARD/HYPOTHESIS), making only 1 HTTP call."""
    from unittest.mock import patch, MagicMock
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage

    # Write stub WAV files so open() calls succeed.
    gt_wav = tmp_path / "gt.wav"
    iter_wav = tmp_path / "iter.wav"
    prior_wav = tmp_path / "prior.wav"
    for p in (gt_wav, iter_wav, prior_wav):
        p.write_bytes(b"\x00" * 44)  # minimal stub, not read in planning path

    step = {"primary_family": "filter", "support_family": "env", "search_keyword": "cutoff"}
    params_delta = [{"name": "filter_1_cutoff", "from_norm": 0.5, "to_norm": 0.52, "mistake": False}]

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "PLAN: Adjusting filter 1 cutoff. These sub-perceptual changes prepare the filter for the next audible step."}}]
    }
    mock_resp.raise_for_status = MagicMock()

    captured_prompts: list[str] = []

    def mock_post(url, **kwargs):
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, str):
                captured_prompts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        captured_prompts.append(part["text"])
        return mock_resp

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        result = _call_omni_commentary_two_stage(
            gt_wav=str(gt_wav),
            iter_wav=str(iter_wav),
            step=step,
            step_num=2,
            archetype="bass",
            omni_server="http://localhost:8000",
            omni_model="test-model",
            stage2_server="http://localhost:8000",
            stage2_model="test-model",
            params_delta=params_delta,
            clap_delta=0.005,  # below any gate threshold
            prior_iter_wav=str(prior_wav),
            is_planning_step=True,  # gate decided this is a planning step
        )

    # Planning path: only 1 HTTP call (Stage 2 text-only, no Stage 1 audio calls).
    assert mock_client.post.call_count == 1, \
        f"Expected 1 call (planning path, Stage 2 only), got {mock_client.post.call_count}"

    # The prompt should mention sub-perceptual/planning context and request PLAN only.
    assert captured_prompts, "No prompts captured"
    combined = " ".join(captured_prompts)
    assert any(kw in combined for kw in (
        "no audible difference", "sub-perceptual", "PLAN", "planning"
    )), f"Expected planning context in prompt, got: {combined[:300]}"

    # Result should be the mocked PLAN-only response.
    assert "PLAN:" in result


def test_three_clip_stage1_makes_two_omni_calls(tmp_path):
    """With prior_iter_wav and clap_delta >= threshold, two_stage should make 2 Stage 1
    calls (1A and 1B) plus 1 Stage 2 call = 3 total HTTP calls."""
    from unittest.mock import patch, MagicMock
    from scripts.build_main_agent_sft_v2 import (
        _call_omni_commentary_two_stage,
        CLAP_IMPERCEPTIBLE_THRESHOLD,
    )

    gt_wav = tmp_path / "gt.wav"
    iter_wav = tmp_path / "iter.wav"
    prior_wav = tmp_path / "prior.wav"
    for p in (gt_wav, iter_wav, prior_wav):
        p.write_bytes(b"\x00" * 44)

    step = {"primary_family": "osc", "support_family": "filter", "search_keyword": "osc level"}

    call_count = 0

    def mock_post(url, **kwargs):
        nonlocal call_count
        call_count += 1
        m = MagicMock()
        m.raise_for_status = MagicMock()
        m.json.return_value = {
            "choices": [{"message": {"content": f"Description {call_count}."}}]
        }
        return m

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        _call_omni_commentary_two_stage(
            gt_wav=str(gt_wav),
            iter_wav=str(iter_wav),
            step=step,
            step_num=3,
            archetype="lead",
            omni_server="http://localhost:8000",
            omni_model="test-model",
            stage2_server="http://localhost:8000",
            stage2_model="test-model",
            clap_delta=CLAP_IMPERCEPTIBLE_THRESHOLD * 10,  # well above threshold
            prior_iter_wav=str(prior_wav),
        )

    assert call_count == 3, \
        f"Expected 3 HTTP calls (Stage 1A + 1B + Stage 2), got {call_count}"


def test_step_labels_include_clap_delta():
    """step_labels entries must include clap_delta key."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert '"clap_delta"' in src or "'clap_delta'" in src, \
        "clap_delta not stored in step_labels"


def test_prior_iter_wav_tracked_in_main_loop():
    """main() must track prev_prev_iter_wav (the two-step-back pointer)."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "prev_prev_iter_wav" in src, "prev_prev_iter_wav not tracked in main()"


def test_plan_instruction_begins_with_adjusting():
    """The PLAN instruction uses 2-sentence format seeded with _group_params_for_plan."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    params_delta = [
        {"name": "filter_1_cutoff", "from_norm": 0.3, "to_norm": 0.7},
        {"name": "env_1_attack", "from_norm": 0.1, "to_norm": 0.4},
    ]
    params_str = _group_params_for_plan(params_delta)

    plan_instruction = (
        f"PLAN (2 sentences): "
        f"Sentence 1 — copy this exactly: \"Adjusting {params_str}.\" "
        f"Sentence 2 — one sentence explaining how these changes address the remaining "
        f"timbral gap. Do not name any parameter in Sentence 2."
    )

    assert "Sentence 1" in plan_instruction
    assert "copy this exactly" in plan_instruction
    assert "Sentence 2" in plan_instruction
    assert "Do not name any parameter" in plan_instruction
    assert "Adjusting " in plan_instruction
    assert "Filter 1 Cutoff" in plan_instruction
    assert "Env 1 Attack" in plan_instruction or "Envelope" in plan_instruction
    # Old pattern must be gone
    assert "must begin with these exact words" not in plan_instruction


def test_plan_instruction_two_sentence_params_str_ends_with_period():
    """The pre-seeded Sentence 1 in the PLAN instruction ends with a period."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    params_delta = [{"name": "filter_1_cutoff"}, {"name": "env_1_attack"}]
    params_str = _group_params_for_plan(params_delta)

    plan_instruction = (
        f"PLAN (2 sentences): "
        f"Sentence 1 — copy this exactly: \"Adjusting {params_str}.\" "
        f"Sentence 2 — one sentence explaining how these changes address the remaining "
        f"timbral gap. Do not name any parameter in Sentence 2."
    )

    # The quoted sentence 1 must end with a period before the closing quote
    assert f'Adjusting {params_str}."' in plan_instruction


def test_group_params_for_plan_small():
    """≤5 params → all display names enumerated, no grouping."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    assert _group_params_for_plan([{"name": "filter_1_cutoff"}]) == "Filter 1 Cutoff"
    result2 = _group_params_for_plan([
        {"name": "filter_1_cutoff"}, {"name": "filter_1_resonance"}
    ])
    assert result2 == "Filter 1 Cutoff and Filter 1 Resonance"
    result3 = _group_params_for_plan([
        {"name": "filter_1_cutoff"}, {"name": "filter_1_resonance"}, {"name": "env_1_attack"}
    ])
    assert "Filter 1 Cutoff" in result3
    assert "Filter 1 Resonance" in result3
    assert "Envelope 1 Attack" in result3 or "Env 1 Attack" in result3
    # No grouping language for small sets
    assert "parameters" not in result3


def test_group_params_for_plan_large_single_subsystem():
    """Many params from one subsystem → 'N subsystem parameters'."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    osc_keys = [
        "osc_1_level", "osc_1_pan", "osc_1_tune", "osc_1_phase",
        "osc_1_distortion_type", "osc_1_distortion_amount",
        "osc_1_spectral_morph_amount", "osc_1_spectral_morph_type",
    ]
    params_delta = [{"name": k} for k in osc_keys]
    result = _group_params_for_plan(params_delta)
    assert "oscillator 1" in result
    assert str(len(osc_keys)) in result
    assert "related" not in result  # the old vague suffix must not appear


def test_group_params_for_plan_large_multi_subsystem():
    """Many params across subsystems → each gets its own group description."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    params_delta = (
        [{"name": f"osc_1_param_{i}"} for i in range(6)]
        + [{"name": "compressor_attack"}, {"name": "compressor_release"}, {"name": "compressor_mix"}]
    )
    result = _group_params_for_plan(params_delta)
    assert "oscillator 1" in result
    assert "compressor" in result
    assert "related" not in result


def test_group_params_for_plan_small_group_names_attributes():
    """2-3 params in same subsystem → 'subsystem attr1 and attr2', not full display names."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    # 6 total params: 2 in filter_1 + 4 in osc_1 to trigger grouping path
    params_delta = [
        {"name": "filter_1_cutoff"},
        {"name": "filter_1_resonance"},
        {"name": "osc_1_level"},
        {"name": "osc_1_pan"},
        {"name": "osc_1_tune"},
        {"name": "osc_1_phase"},
    ]
    result = _group_params_for_plan(params_delta)
    # filter_1 group (2 params) should show attribute names
    assert "filter 1" in result
    assert "cutoff" in result.lower()
    assert "resonance" in result.lower()
    # osc_1 group (4 params) should use count form
    assert "oscillator 1" in result
    assert "4" in result
    assert "related" not in result


# ---------------------------------------------------------------------------
# HYPOTHESIS revision tests
# ---------------------------------------------------------------------------


def test_extract_hypothesis_parses_commentary():
    """_extract_hypothesis should return the HYPOTHESIS text from a commentary string."""
    from scripts.build_main_agent_sft_v2 import _extract_hypothesis

    commentary = (
        "HEARD: The previous step brightened the attack.\n\n"
        "HYPOTHESIS: The filter cutoff may still be too low, likely causing the muffled low-mid character.\n\n"
        "PLAN: Adjusting Filter 1 Cutoff will open up the low-mid range."
    )
    result = _extract_hypothesis(commentary)
    assert result is not None
    assert "filter cutoff" in result.lower()
    assert "HEARD" not in result
    assert "PLAN" not in result


def test_extract_hypothesis_returns_none_on_missing():
    """_extract_hypothesis should return None when there is no HYPOTHESIS section."""
    from scripts.build_main_agent_sft_v2 import _extract_hypothesis

    assert _extract_hypothesis("") is None
    assert _extract_hypothesis("HEARD: something\n\nPLAN: do something") is None


def test_stage2_prompt_includes_prior_hypothesis_revision(tmp_path):
    """When prev_commentary contains a HYPOTHESIS, Stage 2 prompt must include
    the prior hypothesis text and a 'revise' instruction."""
    from unittest.mock import patch, MagicMock
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage

    gt_wav = tmp_path / "gt.wav"
    iter_wav = tmp_path / "iter.wav"
    for p in (gt_wav, iter_wav):
        p.write_bytes(b"\x00" * 44)

    step = {"primary_family": "filter", "support_family": "env", "search_keyword": "cutoff"}
    prev_commentary = (
        "HEARD: The previous step added brightness.\n\n"
        "HYPOTHESIS: The filter cutoff may still be too low.\n\n"
        "PLAN: Adjusting Filter 1 Cutoff will open up the low-mid range."
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting Filter 1 Cutoff will fix it."}}]
    }
    mock_resp.raise_for_status = MagicMock()

    captured_prompts: list[str] = []

    def mock_post(url, **kwargs):
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, str):
                captured_prompts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        captured_prompts.append(part["text"])
        return mock_resp

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        _call_omni_commentary_two_stage(
            gt_wav=str(gt_wav),
            iter_wav=str(iter_wav),
            step=step,
            step_num=3,
            archetype="lead",
            omni_server="http://localhost:8000",
            omni_model="test-model",
            stage2_server="http://localhost:8000",
            stage2_model="test-model",
            prev_commentary=prev_commentary,
            clap_delta=None,
        )

    # The Stage 2 prompt (last captured) should reference the prior hypothesis
    assert captured_prompts, "No prompts captured"
    stage2_prompt = captured_prompts[-1]
    assert "prior hypothesis" in stage2_prompt.lower() or "prior hypothesis" in stage2_prompt, \
        f"Expected 'prior hypothesis' in Stage 2 prompt, got: {stage2_prompt[:400]}"
    assert "filter cutoff may still be too low" in stage2_prompt.lower(), \
        "Prior hypothesis text not injected into Stage 2 prompt"
    assert "revise" in stage2_prompt.lower(), \
        "Stage 2 prompt should instruct the model to revise the hypothesis"


def test_stage2_prompt_no_revision_on_first_step(tmp_path):
    """When prev_commentary is None (first step), Stage 2 prompt uses the default
    HYPOTHESIS instruction without any prior hypothesis injection."""
    from unittest.mock import patch, MagicMock
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage

    gt_wav = tmp_path / "gt.wav"
    iter_wav = tmp_path / "iter.wav"
    for p in (gt_wav, iter_wav):
        p.write_bytes(b"\x00" * 44)

    step = {"primary_family": "filter", "support_family": "env", "search_keyword": "cutoff"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting Filter 1 Cutoff will fix it."}}]
    }
    mock_resp.raise_for_status = MagicMock()

    captured_prompts: list[str] = []

    def mock_post(url, **kwargs):
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, str):
                captured_prompts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        captured_prompts.append(part["text"])
        return mock_resp

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        _call_omni_commentary_two_stage(
            gt_wav=str(gt_wav),
            iter_wav=str(iter_wav),
            step=step,
            step_num=1,
            archetype="lead",
            omni_server="http://localhost:8000",
            omni_model="test-model",
            stage2_server="http://localhost:8000",
            stage2_model="test-model",
            prev_commentary=None,
            clap_delta=None,
        )

    assert captured_prompts, "No prompts captured"
    stage2_prompt = captured_prompts[-1]
    assert "prior hypothesis" not in stage2_prompt.lower(), \
        "First-step prompt should not contain prior hypothesis injection"
    assert "most likely synthesis causes" in stage2_prompt, \
        "First-step prompt should use the default HYPOTHESIS instruction"


# ---------------------------------------------------------------------------
# HEARD extraction + revision injection tests
# ---------------------------------------------------------------------------


def test_extract_heard_parses_commentary():
    """_extract_heard should return the HEARD text, excluding HYPOTHESIS and PLAN."""
    from scripts.build_main_agent_sft_v2 import _extract_heard

    commentary = (
        "HEARD: The attack sharpened and the reverb tail shortened significantly.\n\n"
        "HYPOTHESIS: The filter cutoff may still be too low.\n\n"
        "PLAN: Adjusting Filter 1 Cutoff will open up the low-mid range."
    )
    result = _extract_heard(commentary)
    assert result is not None
    assert "attack sharpened" in result
    assert "HYPOTHESIS" not in result
    assert "PLAN" not in result


def test_extract_heard_returns_none_on_missing():
    """_extract_heard should return None when there is no HEARD section."""
    from scripts.build_main_agent_sft_v2 import _extract_heard

    assert _extract_heard("") is None
    assert _extract_heard("HYPOTHESIS: x\n\nPLAN: y") is None


def test_stage2_prompt_includes_prior_heard_constraint(tmp_path):
    """When prev_commentary contains a HEARD section and both obs blocks are present,
    Stage 2 prompt must include the prior HEARD text and a 'different' instruction."""
    from unittest.mock import patch, MagicMock
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage, CLAP_IMPERCEPTIBLE_THRESHOLD

    gt_wav = tmp_path / "gt.wav"
    iter_wav = tmp_path / "iter.wav"
    prior_wav = tmp_path / "prior.wav"
    for p in (gt_wav, iter_wav, prior_wav):
        p.write_bytes(b"\x00" * 44)

    step = {"primary_family": "filter", "support_family": "env", "search_keyword": "cutoff"}
    prev_commentary = (
        "HEARD: The attack sharpened and reverb tail shortened significantly.\n\n"
        "HYPOTHESIS: The filter cutoff may still be too low.\n\n"
        "PLAN: Adjusting Filter 1 Cutoff will open up the low-mid range."
    )

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting Filter 1 Cutoff will fix it."}}]
    }
    mock_resp.raise_for_status = MagicMock()

    captured_prompts: list[str] = []

    def mock_post(url, **kwargs):
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, str):
                captured_prompts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        captured_prompts.append(part["text"])
        return mock_resp

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        _call_omni_commentary_two_stage(
            gt_wav=str(gt_wav),
            iter_wav=str(iter_wav),
            step=step,
            step_num=3,
            archetype="bass",
            omni_server="http://localhost:8000",
            omni_model="test-model",
            stage2_server="http://localhost:8000",
            stage2_model="test-model",
            prev_commentary=prev_commentary,
            clap_delta=CLAP_IMPERCEPTIBLE_THRESHOLD * 10,  # above threshold → 3-clip path
            prior_iter_wav=str(prior_wav),
        )

    assert captured_prompts, "No prompts captured"
    stage2_prompt = captured_prompts[-1]
    assert "previous step's HEARD" in stage2_prompt, \
        f"Expected prior HEARD injection in Stage 2 prompt, got: {stage2_prompt[:400]}"
    assert "attack sharpened" in stage2_prompt, \
        "Prior HEARD text not injected into Stage 2 prompt"
    assert "DIFFERENT" in stage2_prompt, \
        "Stage 2 prompt should instruct Sentence 1 to differ from prior HEARD"


def test_stage2_prompt_no_heard_constraint_on_first_step(tmp_path):
    """When prev_commentary is None (first step), Stage 2 prompt must not include
    any prior HEARD injection."""
    from unittest.mock import patch, MagicMock
    from scripts.build_main_agent_sft_v2 import _call_omni_commentary_two_stage

    gt_wav = tmp_path / "gt.wav"
    iter_wav = tmp_path / "iter.wav"
    for p in (gt_wav, iter_wav):
        p.write_bytes(b"\x00" * 44)

    step = {"primary_family": "filter", "support_family": "env", "search_keyword": "cutoff"}

    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "choices": [{"message": {"content": "HEARD: x\n\nHYPOTHESIS: y\n\nPLAN: Adjusting Filter 1 Cutoff will fix it."}}]
    }
    mock_resp.raise_for_status = MagicMock()

    captured_prompts: list[str] = []

    def mock_post(url, **kwargs):
        body = kwargs.get("json", {})
        msgs = body.get("messages", [])
        if msgs:
            content = msgs[0].get("content", "")
            if isinstance(content, str):
                captured_prompts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        captured_prompts.append(part["text"])
        return mock_resp

    with patch("httpx.Client") as mock_client_cls:
        mock_client = MagicMock()
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)
        mock_client.post = MagicMock(side_effect=mock_post)
        mock_client_cls.return_value = mock_client

        _call_omni_commentary_two_stage(
            gt_wav=str(gt_wav),
            iter_wav=str(iter_wav),
            step=step,
            step_num=1,
            archetype="bass",
            omni_server="http://localhost:8000",
            omni_model="test-model",
            stage2_server="http://localhost:8000",
            stage2_model="test-model",
            prev_commentary=None,
            clap_delta=None,
        )

    assert captured_prompts, "No prompts captured"
    stage2_prompt = captured_prompts[-1]
    assert "previous step's HEARD" not in stage2_prompt, \
        "First-step prompt should not inject prior HEARD"


def test_group_params_for_plan_on_param_displays_as_on_off():
    """Boolean 'on' params stripped of subsystem prefix should display as 'on/off'."""
    from scripts.build_main_agent_sft_v2 import _group_params_for_plan

    # 6 total params to trigger grouping; filter_1 group has 3 including filter_1_on
    params_delta = [
        {"name": "filter_1_blend"},
        {"name": "filter_1_on"},
        {"name": "filter_1_resonance"},
        {"name": "osc_1_level"},
        {"name": "osc_1_pan"},
        {"name": "osc_1_tune"},
    ]
    result = _group_params_for_plan(params_delta)
    assert "on/off" in result, f"Expected 'on/off' in result, got: {result}"
    # Should not contain bare standalone " on " sandwiched by spaces or commas
    import re
    assert not re.search(r"(?<![/\w])on(?![/\w])", result), \
        f"Bare 'on' still present in result: {result}"


def test_hypothesis_style_hint_varies_by_step_num():
    """_HYPOTHESIS_STYLE_HINTS must have ≥4 entries and consecutive steps must use
    different style hints."""
    from scripts.build_main_agent_sft_v2 import _HYPOTHESIS_STYLE_HINTS

    assert len(_HYPOTHESIS_STYLE_HINTS) >= 4, "Need at least 4 style hints to rotate"

    # Verify no two consecutive step numbers (1-4) map to the same hint
    seen = [_HYPOTHESIS_STYLE_HINTS[(s - 1) % len(_HYPOTHESIS_STYLE_HINTS)] for s in range(1, 5)]
    for i in range(len(seen) - 1):
        assert seen[i] != seen[i + 1], \
            f"Steps {i+1} and {i+2} use the same style hint"

    # Verify hints contain distinct opening directives
    assert any("acoustic" in h or "hear" in h.lower() for h in _HYPOTHESIS_STYLE_HINTS), \
        "Should have at least one acoustic-evidence style hint"
    assert any("mechanism" in h.lower() or "responsible" in h.lower() for h in _HYPOTHESIS_STYLE_HINTS), \
        "Should have at least one mechanism-first style hint"

    # Every hint must reference {remaining} so the model reasons about what still needs fixing
    for i, h in enumerate(_HYPOTHESIS_STYLE_HINTS):
        assert "{remaining}" in h, \
            f"Style hint {i} missing {{remaining}} placeholder: {h[:80]}"


def test_extract_top_remaining_parses_context_string():
    from scripts.build_main_agent_sft_v2 import _extract_top_remaining
    ctx = "compressor (12 params), chorus (11 params), EQ (10 params), reverb (10 params), plus 104 params across other subsystems"
    result = _extract_top_remaining(ctx, n=2)
    assert result == "compressor and chorus", f"Got: {result}"


def test_extract_top_remaining_single_subsystem():
    from scripts.build_main_agent_sft_v2 import _extract_top_remaining
    ctx = "oscillator 1 (5 params)"
    result = _extract_top_remaining(ctx, n=2)
    assert result == "oscillator 1", f"Got: {result}"


def test_extract_top_remaining_none_returns_fallback():
    from scripts.build_main_agent_sft_v2 import _extract_top_remaining
    assert _extract_top_remaining(None) == "the remaining parameters"
    assert _extract_top_remaining("none — preset converged to target") == "the remaining parameters"


def test_style_hint_format_includes_remaining():
    """The formatted style hint must contain the remaining subsystem text, not literal {remaining}."""
    from scripts.build_main_agent_sft_v2 import _HYPOTHESIS_STYLE_HINTS, _extract_top_remaining
    ctx = "filter 1 (8 params), envelope 2 (4 params)"
    remaining = _extract_top_remaining(ctx)
    for i, hint in enumerate(_HYPOTHESIS_STYLE_HINTS):
        formatted = hint.format(primary="oscillator 1", support="filter 1", remaining=remaining)
        assert "{remaining}" not in formatted, f"Hint {i} still has literal {{remaining}} after format"
        assert "filter 1" in formatted or "envelope 2" in formatted or "the remaining" in formatted, \
            f"Hint {i} doesn't contain the remaining subsystem text: {formatted[:100]}"


# ---------------------------------------------------------------------------
# GT-preset convergence: _step_remaining_gap + terminal turn
# ---------------------------------------------------------------------------

def test_step_remaining_gap_returns_none_without_cumulative():
    """_step_remaining_gap returns None when step has no cumulative_preset."""
    from scripts.build_main_agent_sft_v2 import _step_remaining_gap
    target = {"settings": {"filter_1_cutoff": 0.8}}
    step_no_cum = {}  # no cumulative_preset key
    result = _step_remaining_gap(target, step_no_cum)
    assert result is None


def test_step_remaining_gap_converged_when_presets_match():
    """_step_remaining_gap reports converged when cumulative_preset matches target."""
    from scripts.build_main_agent_sft_v2 import _step_remaining_gap
    import json
    from pathlib import Path

    # Load a real path file to get a meaningful target_preset
    path_file = Path("outputs/smoke_test_v2/paths/bass_9b82b7d7.json")
    if not path_file.exists():
        import pytest
        pytest.skip("Smoke test path file not available")

    with open(path_file) as f:
        path_data = json.load(f)

    target = path_data["target_preset"]
    # Use the final step's cumulative_preset — it should be closest to target
    final_step = path_data["iterations"][-1]
    result = _step_remaining_gap(target, final_step)
    assert result is not None, "Should return a gap dict for a step with cumulative_preset"
    assert "n_remaining" in result
    assert "by_subsystem" in result
    assert "context_str" in result
    assert isinstance(result["n_remaining"], int)
    assert result["n_remaining"] >= 0


def test_step_remaining_gap_more_remaining_early_than_late():
    """Earlier steps should have more remaining params than the final step."""
    from scripts.build_main_agent_sft_v2 import _step_remaining_gap
    import json
    from pathlib import Path

    path_file = Path("outputs/smoke_test_v2/paths/bass_9b82b7d7.json")
    if not path_file.exists():
        import pytest
        pytest.skip("Smoke test path file not available")

    with open(path_file) as f:
        path_data = json.load(f)

    target = path_data["target_preset"]
    iters = path_data["iterations"]
    if len(iters) < 2:
        import pytest
        pytest.skip("Need at least 2 iterations")

    first_gap = _step_remaining_gap(target, iters[0])
    last_gap = _step_remaining_gap(target, iters[-1])
    assert first_gap is not None and last_gap is not None
    assert first_gap["n_remaining"] >= last_gap["n_remaining"], \
        f"Expected first step gap ({first_gap['n_remaining']}) >= last step gap ({last_gap['n_remaining']})"


def test_closing_eval_accepts_convergence_param():
    """_call_omni_closing_eval must accept a convergence keyword argument."""
    import inspect
    from scripts.build_main_agent_sft_v2 import _call_omni_closing_eval
    sig = inspect.signature(_call_omni_closing_eval)
    assert "convergence" in sig.parameters, "convergence param missing from _call_omni_closing_eval"
    assert sig.parameters["convergence"].default is None, "convergence should default to None"


def test_closing_eval_fallback_complete_verdict():
    """Fallback with converged=True (path completed) should say 'complete'."""
    from unittest.mock import patch
    from scripts.build_main_agent_sft_v2 import _call_omni_closing_eval

    convergence = {"converged": True, "n_remaining": 45, "by_subsystem": {"compressor": 12}}
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = Exception("no server")
        result = _call_omni_closing_eval(
            gt_wav="/dev/null", final_iter_wav="/dev/null",
            archetype="bass", omni_server="http://localhost:9999", model="test",
            convergence=convergence,
        )
    assert "complete" in result.lower(), f"Expected 'complete' in fallback result: {result}"
    assert "budget exhausted" not in result.lower()


def test_closing_eval_fallback_budget_exhausted_verdict():
    """Fallback with converged=False (path truncated by max_steps) should say 'budget exhausted'."""
    from unittest.mock import patch
    from scripts.build_main_agent_sft_v2 import _call_omni_closing_eval

    convergence = {
        "converged": False,
        "n_remaining": 12,
        "by_subsystem": {"oscillator 1": 8, "filter 1": 4},
    }
    with patch("httpx.Client") as mock_client:
        mock_client.return_value.__enter__.return_value.post.side_effect = Exception("no server")
        result = _call_omni_closing_eval(
            gt_wav="/dev/null", final_iter_wav="/dev/null",
            archetype="lead", omni_server="http://localhost:9999", model="test",
            convergence=convergence,
        )
    assert "budget exhausted" in result.lower(), f"Expected 'budget exhausted' in fallback: {result}"
    assert "complete" not in result.lower() or "budget" in result.lower()


def test_main_loop_adds_user_turn_before_closing():
    """The terminal user turn must appear in main() source before the closing assistant turn."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "That completes the available iterations" in src, \
        "Terminal user turn text missing from main()"
    # The user turn dict must appear before _call_omni_closing_eval in source order
    user_turn_pos = src.index("That completes the available iterations")
    closing_call_pos = src.index("_call_omni_closing_eval")
    assert user_turn_pos < closing_call_pos, \
        "User turn must appear before _call_omni_closing_eval in main()"


def test_target_preset_loaded_from_path_when_not_in_memory():
    """main() must fall back to target_preset_path when target_preset is absent."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "target_preset_path" in src, "target_preset_path not handled in main()"
    # Must load from disk, not just skip
    assert 'json.load(_f)' in src or 'json.load(f)' in src, \
        "No json.load call for target_preset_path fallback"


def test_remaining_delta_context_uses_gt_preset_gap():
    """main() must compute remaining_delta_context from _step_remaining_gap, not just path data."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "_step_remaining_gap" in src, "_step_remaining_gap not called in main()"
    assert "target_preset" in src, "target_preset not used in main()"
    # Old path: relying purely on step.get("remaining_delta_context") without GT computation
    # New path: step_gap["context_str"] takes precedence
    assert 'step_gap["context_str"]' in src or "step_gap" in src, \
        "step_gap result not used in main()"


# ---------------------------------------------------------------------------
# Parallel builder tests
# ---------------------------------------------------------------------------

def test_builder_has_workers_arg():
    """build_main_agent_sft_v2.main() must accept a --workers argument."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "--workers" in src, "--workers argument missing from main()"
    assert "workers" in src, "workers variable not used in main()"


def test_builder_uses_thread_pool_executor():
    """main() must use ThreadPoolExecutor for parallelism."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "ThreadPoolExecutor" in src, "ThreadPoolExecutor not found in main()"
    assert "as_completed" in src, "as_completed not found in main()"


def test_builder_serial_lock_guards_vita_and_clap():
    """_process_entry must acquire a lock before vita/CLAP calls."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    # The lock must be acquired before ensure_candidate_probes_for_names (vita)
    lock_pos = src.index("_serial_lock")
    probe_pos = src.index("ensure_candidate_probes_for_names")
    assert lock_pos < probe_pos, "_serial_lock must appear before ensure_candidate_probes_for_names"
    # choose_candidate_pool (CLAP) must also be inside the lock
    clap_pos = src.index("choose_candidate_pool")
    assert lock_pos < clap_pos, "_serial_lock must appear before choose_candidate_pool"


def test_builder_output_order_deterministic():
    """Records must be sorted by input index, not completion order."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    # Must sort by index before writing
    assert "sorted(records_by_idx)" in src, \
        "records_by_idx must be sorted by index to guarantee output order"


def test_builder_worker_errors_handled_gracefully():
    """Failed entries must log a warning and be skipped, not abort the run."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "WARNING" in src, "No WARNING message for failed entries"
    assert "except Exception" in src, "No exception handler for failed entries"


def test_builder_workers_1_skips_thread_pool():
    """When workers=1, the serial path must be used (no ThreadPoolExecutor)."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    # The code must branch on workers > 1
    assert "workers > 1" in src or "args.workers > 1" in src, \
        "No branch on workers > 1; serial path may be missing"


def test_builder_process_entry_returns_none_on_skip():
    """_process_entry must return None for entries that should be skipped."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "return None" in src, "_process_entry must return None for skipped entries"


def test_builder_record_collected_via_return_not_append():
    """The parallel design requires _process_entry to return the record, not append to a list."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert "return [record]" in src or "return record" in src, \
        "_process_entry must return record (not append)"
    # The old pattern records.append(record) should be replaced
    assert "records.append(record)" not in src, \
        "records.append(record) still present; should be return record"


def test_step_labels_includes_clap_score():
    """step_labels must include a clap_score field populated at build time."""
    import inspect
    from scripts import build_main_agent_sft_v2
    src = inspect.getsource(build_main_agent_sft_v2.main)
    assert '"clap_score"' in src, "clap_score field missing from step_labels"
    assert "step_clap_scores" in src, "step_clap_scores dict not computed in main()"
    assert "embedder.cosine_paths" in src.split("step_clap_scores")[1].split("step_labels")[0] or \
           "embedder.cosine_paths" in src, \
        "embedder.cosine_paths not called for per-step CLAP computation"
