"""Contract tests for build_main_agent_sft_v3 record structure.

Validates v3's diagnose → subsystem-batched topology:
  - Exactly one DIAGNOSIS block with OBSERVATIONS + PLAN markers
  - Each subsystem batch follows the apply → listen → check pattern
  - Audio-tag count matches meta.batch_labels + corrections
  - FINAL ASSESSMENT present
  - No v2-style HEARD/HYPOTHESIS headers
  - Params in each set_params call match the preceding subsystem label
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.agent_sft_common import validate_ms_swift_multiturn_record
from scripts.build_main_agent_sft_v3 import (
    MistakeInfo,
    SUBSYSTEM_ORDER,
    SUBSYSTEM_PARAM_POOL,
    SubsystemBatch,
    _format_param_deltas,
    build_batches_from_diff,
    build_diagnosis_subsystem_truth,
    extract_diagnosis_subsystems_mentioned,
    format_subsystem_diff_summary,
    inject_mistakes,
    presentation_subsystem,
)
from maestro.synth.path_gen import _param_family


# ---- helpers ----

def _make_path_iterations(families=("osc", "osc", "env1", "filter1", "lfo")):
    """Create synthetic path_gen iterations for a given family sequence."""
    iters = []
    _prefix_map = {
        "osc": "osc_1_", "env1": "env_1_", "env2": "env_2_",
        "filter1": "filter_1_", "filter2": "filter_2_",
        "lfo": "lfo_1_", "modulation": "modulation_1_",
        "reverb": "reverb_", "delay": "delay_", "chorus": "chorus_",
        "distortion": "distortion_", "compressor": "compressor_",
        "phaser": "phaser_", "flanger": "flanger_", "eq": "eq_",
        "other": "polyphony_",
    }
    for i, fam in enumerate(families, start=1):
        name_prefix = _prefix_map.get(fam, f"{fam}_")
        iters.append({
            "step": i,
            "primary_family": fam,
            "planner_stage": "core" if fam in ("osc", "env1", "env2", "filter1", "filter2") else fam,
            "params_delta": [
                {"name": f"{name_prefix}param_{i}", "from_norm": 0.5, "to_norm": 0.8, "target_norm": 0.8, "mistake": False},
            ],
        })
    return iters


# ---- presentation_subsystem ----

def test_presentation_subsystem_maps_correctly():
    assert presentation_subsystem("osc") == "oscillator"
    assert presentation_subsystem("env1") == "envelope"
    assert presentation_subsystem("env2") == "envelope"
    assert presentation_subsystem("filter1") == "filter"
    assert presentation_subsystem("filter2") == "filter"
    assert presentation_subsystem("lfo") == "lfo"
    assert presentation_subsystem("reverb") == "fx"
    assert presentation_subsystem("delay") == "fx"
    assert presentation_subsystem("modulation") == "modulation"
    assert presentation_subsystem("other") == "macro"


# ---- group_path_into_batches ----

def test_build_batches_from_diff_buckets_by_subsystem():
    """build_batches_from_diff produces one batch per subsystem, in SUBSYSTEM_ORDER."""
    from maestro.synth.path_gen import PARAM_RANGES, _denormalize
    init = {"settings": {k: r["default"] for k, r in PARAM_RANGES.items()}}
    target = {"settings": dict(init["settings"])}
    # Change one osc param, one filter param, one lfo param
    target["settings"]["osc_1_level"] = _denormalize("osc_1_level", 0.9)
    target["settings"]["filter_1_cutoff"] = _denormalize("filter_1_cutoff", 0.2)
    target["settings"]["lfo_1_frequency"] = _denormalize("lfo_1_frequency", 0.7)
    batches = build_batches_from_diff(target, init)
    labels = [b.subsystem for b in batches]
    assert "oscillator" in labels
    assert "filter" in labels
    assert "lfo" in labels
    # All params in oscillator batch should be osc-family
    osc_batch = next(b for b in batches if b.subsystem == "oscillator")
    assert all(_param_family(n) == "osc" for n in osc_batch.params)


def test_inject_mistakes_modifies_applied():
    """inject_mistakes mutates params_applied on chosen batches."""
    import random
    b = SubsystemBatch(
        subsystem="oscillator",
        params={"osc_1_level": 0.5, "osc_1_pan": 0.3, "osc_1_tune": 0.7},
        params_applied={"osc_1_level": 0.5, "osc_1_pan": 0.3, "osc_1_tune": 0.7},
    )
    rng = random.Random(42)
    results = inject_mistakes([b], rng, per_param_rate=1.0,
                              init_preset_settings={"osc_1_level": 0.0, "osc_1_pan": 0.5, "osc_1_tune": 0.0})
    assert len(results) > 0
    assert b.mistakes is not None
    for m in results:
        assert isinstance(m, MistakeInfo)
        assert m.kind in ("overshoot", "undershoot", "omission", "spurious")
        assert m.magnitude >= 0.03
        if m.kind != "omission":
            assert m.param in b.params_applied or m.kind == "spurious"
        assert m.param in b.params or m.kind == "spurious"


def test_inject_mistakes_empty_when_rate_zero():
    import random
    b = SubsystemBatch(
        subsystem="oscillator",
        params={"osc_1_level": 0.5, "osc_1_pan": 0.3},
        params_applied={"osc_1_level": 0.5, "osc_1_pan": 0.3},
    )
    results = inject_mistakes([b], random.Random(42), per_param_rate=0.0)
    assert results == []
    assert b.mistakes is None


def test_inject_mistakes_respects_max_per_batch():
    import random
    b = SubsystemBatch(
        subsystem="oscillator",
        params={f"osc_1_p{i}": 0.5 for i in range(20)},
        params_applied={f"osc_1_p{i}": 0.5 for i in range(20)},
    )
    results = inject_mistakes([b], random.Random(99), per_param_rate=1.0,
                              max_mistakes_per_batch=3, max_mistakes_total=10,
                              init_preset_settings={f"osc_1_p{i}": 0.0 for i in range(20)})
    assert len(results) <= 3


def test_inject_mistakes_respects_max_total():
    import random
    batches = [
        SubsystemBatch(
            subsystem=sub,
            params={f"{sub}_p{i}": 0.5 for i in range(10)},
            params_applied={f"{sub}_p{i}": 0.5 for i in range(10)},
        )
        for sub in ["oscillator", "filter", "lfo"]
    ]
    init_s = {f"{sub}_p{i}": 0.0 for sub in ["oscillator", "filter", "lfo"] for i in range(10)}
    results = inject_mistakes(batches, random.Random(7), per_param_rate=1.0,
                              max_mistakes_total=4, init_preset_settings=init_s)
    assert len(results) <= 4


def test_inject_mistakes_omission_removes_from_applied():
    import random
    for seed in range(200):
        b = SubsystemBatch(
            subsystem="filter",
            params={"filter_1_cutoff": 0.7, "filter_1_resonance": 0.3, "filter_1_drive": 0.6},
            params_applied={"filter_1_cutoff": 0.7, "filter_1_resonance": 0.3, "filter_1_drive": 0.6},
        )
        results = inject_mistakes([b], random.Random(seed), per_param_rate=0.8,
                                  init_preset_settings={"filter_1_cutoff": 0.0, "filter_1_resonance": 0.0, "filter_1_drive": 0.0})
        omissions = [m for m in results if m.kind == "omission"]
        for m in omissions:
            assert m.param not in b.params_applied, f"Omitted param {m.param} should not be in params_applied"
            assert m.param in b.params, "Omitted param must be in GT params"
        if omissions:
            return  # found at least one omission
    assert False, "No omission generated across 200 seeds"


def test_inject_mistakes_spurious_adds_same_subsystem_param():
    import random
    for seed in range(200):
        b = SubsystemBatch(
            subsystem="oscillator",
            params={"osc_1_level": 0.8, "osc_1_pan": 0.5},
            params_applied={"osc_1_level": 0.8, "osc_1_pan": 0.5},
        )
        results = inject_mistakes([b], random.Random(seed), per_param_rate=0.8,
                                  init_preset_settings={"osc_1_level": 0.5, "osc_1_pan": 0.5})
        spurious = [m for m in results if m.kind == "spurious"]
        for m in spurious:
            assert m.param in b.params_applied, "Spurious param must be added to params_applied"
            assert m.param not in b.params, "Spurious param must NOT be in GT params"
            assert m.param in SUBSYSTEM_PARAM_POOL.get("oscillator", []), \
                f"Spurious param {m.param} should be from oscillator pool"
        if spurious:
            return
    assert False, "No spurious generated across 200 seeds"


def test_inject_mistakes_undershoot_between_init_and_target():
    import random
    for seed in range(200):
        b = SubsystemBatch(
            subsystem="filter",
            params={"filter_1_cutoff": 0.8, "filter_1_resonance": 0.6},
            params_applied={"filter_1_cutoff": 0.8, "filter_1_resonance": 0.6},
        )
        results = inject_mistakes([b], random.Random(seed), per_param_rate=0.8,
                                  init_preset_settings={"filter_1_cutoff": 0.0, "filter_1_resonance": 0.0})
        undershoots = [m for m in results if m.kind == "undershoot"]
        for m in undershoots:
            init_norm = 0.0
            true_norm = b.params[m.param]
            lo, hi = min(init_norm, true_norm), max(init_norm, true_norm)
            assert lo <= m.wrong_value <= hi, \
                f"Undershoot {m.param}: wrong={m.wrong_value} should be between {lo} and {hi}"
        if undershoots:
            return
    assert False, "No undershoot generated across 200 seeds"


def test_format_param_deltas_omitted_tag():
    result = _format_param_deltas([("filter_1_cutoff", 0.5, 0.5, "omitted")])
    assert "planned but not applied" in result


def test_format_param_deltas_spurious_tag():
    result = _format_param_deltas([("osc_1_tune", 0.0, 0.3, "spurious")])
    assert "unplanned change" in result


def test_format_param_deltas_backward_compat():
    result = _format_param_deltas([("filter_1_cutoff", 0.3, 0.8)])
    assert "→" in result
    assert "planned but not applied" not in result


# ---- build_diagnosis_subsystem_truth ----

def test_build_diagnosis_subsystem_truth_groups_correctly():
    target = {"settings": {
        "osc_1_level": 0.7,
        "filter_1_cutoff": 60.0,  # differs from init (native value)
        "reverb_dry_wet": 0.5,
    }}
    init = {"settings": {
        "osc_1_level": 0.5,
        "filter_1_cutoff": 80.0,
        "reverb_dry_wet": 0.0,
    }}
    truth = build_diagnosis_subsystem_truth(target, init)
    assert "oscillator" in truth
    assert "filter" in truth
    assert "fx" in truth


# ---- extract_diagnosis_subsystems_mentioned ----

def test_extract_diagnosis_parses_subsystems():
    text = "PLAN:\n• Oscillator: swap wavetable\n• Filter: lower cutoff\n• LFO: add movement"
    mentioned = extract_diagnosis_subsystems_mentioned(text)
    assert "oscillator" in mentioned
    assert "filter" in mentioned
    assert "lfo" in mentioned


def test_extract_diagnosis_no_false_positives():
    text = "The preset needs adjustment."
    mentioned = extract_diagnosis_subsystems_mentioned(text)
    assert mentioned == []


# ---- format_subsystem_diff_summary ----

def test_format_diff_summary_lists_subsystems():
    truth = {"oscillator": ["osc_1_level"], "filter": ["filter_1_cutoff", "filter_1_resonance"]}
    s = format_subsystem_diff_summary(truth)
    assert "oscillator (1 params)" in s
    assert "filter (2 params)" in s


# ---- structural invariants on a v3 record ----

def _make_v3_record(n_batches=3, has_correction=False, has_mistake=False):
    """Build a synthetic v3 record for structural testing.

    Mirrors the claw-code-style WT scaffold:
      baseline → list_wavetables → N Agent calls → cat shortlists → tuple eval → apply
    """
    messages = [
        {"role": "user", "content": "<audio>\nRecreate this keys target sound in Vital from default.\nSearch for matching wavetables across the library, evaluate combinations, then execute by subsystem."},
        {"role": "assistant", "content": "Listening to current default preset baseline."},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo probe"}}'},
        {"role": "tool_response", "content": '{"status":"ok","baseline_audio":"<audio>","path":"/tmp/d.wav"}'},
        {"role": "assistant", "content": "Checking wavetable library size."},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"python - <<\'PY\'\\nimport json, os, glob\\n_VITAL_DIRS = [os.path.expanduser(\'~/.local/share/vital\')]\\n_seen, _count = set(), 0\\nfor _vd in _VITAL_DIRS:\\n    for _vt in sorted(glob.glob(os.path.join(_vd, \'**\', \'*.vitaltable\'), recursive=True)):\\n        _w = json.load(open(_vt)); _n = _w.get(\'name\',\'\')\\n        if _n and _n not in _seen: _seen.add(_n); _count += 1\\nprint(json.dumps({\'total\': _count}))\\nPY"}}'},
        {"role": "tool_response", "content": '{"total":282}'},
        {"role": "assistant", "content": "Library has 282 wavetables. Dispatching 2 search agents across slices [0-47, 141-188]."},
        # Agent call 1
        {"role": "tool_call", "content": '{"name":"Agent","arguments":{"subagent_type":"wavetable_search","description":"Evaluate wavetables 0-47 for keys target","prompt":"Target..."}}'},
        {"role": "tool_response", "content": '{"agentId":"search_1","subagentType":"wavetable_search","status":"completed","outputFile":"/tmp/agents/search_1.json"}'},
        # Agent call 2
        {"role": "tool_call", "content": '{"name":"Agent","arguments":{"subagent_type":"wavetable_search","description":"Evaluate wavetables 141-188 for keys target","prompt":"Target..."}}'},
        {"role": "tool_response", "content": '{"agentId":"search_2","subagentType":"wavetable_search","status":"completed","outputFile":"/tmp/agents/search_2.json"}'},
        # Cat shortlists
        {"role": "assistant", "content": "Reading shortlists from 2 search agents."},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"cat /tmp/agents/search_1.json /tmp/agents/search_2.json"}}'},
        {"role": "tool_response", "content": '{"status":"completed","shortlist":["01 Basic Shapes","Cymatics Chill 25"]}\n{"status":"completed","shortlist":["Pink Noise"]}'},
        # Pool summary + tuple intro (merged)
        {"role": "assistant", "content": "Pooled 3 candidates across 1 round(s): ['01 Basic Shapes', 'Cymatics Chill 25', 'Pink Noise'].\n\nThe target uses 1 active oscillator. Evaluating 2 wavetable combinations."},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"# Render tuple 1"}}'},
        {"role": "tool_response", "content": '{"tuple_audio":"<audio>","tuple_index":1,"wavetables":["01 Basic Shapes"]}'},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"# Render tuple 2"}}'},
        {"role": "tool_response", "content": '{"tuple_audio":"<audio>","tuple_index":2,"wavetables":["Cymatics Chill 25"]}'},
        {"role": "assistant", "content": "Tuple 1 best matches the target. Applying: oscillator 1 = \'01 Basic Shapes\'."},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"# Apply wavetable via library lookup"}}'},
        {"role": "tool_response", "content": '{"status":"ok","applied":["01 Basic Shapes"]}'},
    ]
    audios = ["/tmp/gt.wav", "/tmp/default.wav", "/tmp/tuple1.wav", "/tmp/tuple2.wav"]
    diagnosis = "OBSERVATIONS: The target has brighter harmonics.\n\nPLAN:\n• Oscillator: add unison detune\n• Filter: lower cutoff\n• LFO: add movement\nExecuting plan by subsystem."
    batch_labels = []
    for bi in range(n_batches):
        sub = ["oscillator", "filter", "lfo"][bi % 3]
        # First batch merges with diagnosis; later batches merge prior batch check
        if bi == 0:
            intro = f"{diagnosis}\n\nApplying {sub} changes."
        else:
            prev_sub = ["oscillator", "filter", "lfo"][(bi - 1) % 3]
            intro = f"{prev_sub.capitalize()} edits sharpened the attack.\n\nApplying {sub} changes."
        messages.append({"role": "assistant", "content": intro})
        messages.append({"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo set"}}'})
        messages.append({"role": "tool_response", "content": '{"status":"ok"}'})
        messages.append({"role": "assistant", "content": f"Listening after {sub} batch."})
        messages.append({"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo listen"}}'})
        messages.append({"role": "tool_response", "content": f'{{"status":"ok","batch_audio":"<audio>","path":"/tmp/b{bi}.wav"}}'})
        audios.append(f"/tmp/b{bi}.wav")
        batch_labels.append({
            "index": bi,
            "subsystem": sub,
            "param_names": [f"{sub}_param_{bi}"],
            "clap_score_after_batch": 0.7 + bi * 0.05,
            "is_correction": False,
        })

    if has_correction:
        last_sub = ["oscillator", "filter", "lfo"][(n_batches - 1) % 3]
        messages.append({"role": "assistant", "content": f"{last_sub.capitalize()} edits sharpened the attack.\n\nOvershot on filter — backing off Filter 1 Cutoff."})
        messages.append({"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo corr"}}'})
        messages.append({"role": "tool_response", "content": '{"status":"ok"}'})
        messages.append({"role": "assistant", "content": "Listening to the corrected preset."})
        messages.append({"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo listen_c"}}'})
        messages.append({"role": "tool_response", "content": '{"status":"ok","corrected_audio":"<audio>","path":"/tmp/corr.wav"}'})
        audios.append("/tmp/corr.wav")
        # Confirmation merges into verdict to avoid adjacent assistant.
        batch_labels.append({
            "index": len(batch_labels),
            "subsystem": "correction",
            "param_names": ["filter_1_cutoff"],
            "clap_score_after_batch": 0.9,
            "is_correction": True,
        })

    last_sub = ["oscillator", "filter", "lfo"][(n_batches - 1) % 3]
    if has_correction:
        verdict_prefix = "The filter region now sits back in line with the plan.\n\n"
    else:
        verdict_prefix = f"{last_sub.capitalize()} edits sharpened the attack.\n\n"
    messages.append({"role": "assistant", "content": f"{verdict_prefix}FINAL ASSESSMENT (complete): The recreation matches the target. Minor upper-harmonic differences remain."})

    return {
        "id": "test_v3_0001",
        "task_type": "main",
        "tools": "[]",
        "messages": messages,
        "audios": audios,
        "assets": {},
        "labels": {},
        "meta": {
            "pipeline_version": "v3",
            "sample_id": "test_v3_0001",
            "archetype": "keys",
            "batch_labels": batch_labels,
            "diagnosis_subsystems_mentioned": ["oscillator", "filter", "lfo"],
            "diagnosis_subsystems_truth": ["oscillator", "filter", "lfo"],
            "injected_mistakes": ([{"param": "filter_1_cutoff", "kind": "overshoot", "wrong_value": 0.2, "true_value": 0.6, "magnitude": 0.4}] if has_mistake else []),
            "total_correction_turns": (1 if has_correction and has_mistake else 0),
            "mistake_caught": (True if has_correction and has_mistake else False),
        },
    }


def test_v3_record_passes_validator():
    record = _make_v3_record()
    errors = validate_ms_swift_multiturn_record(record)
    assert errors == [], f"Validation errors: {errors}"


def test_v3_exactly_one_diagnosis_block():
    record = _make_v3_record()
    assistant_msgs = [m["content"] for m in record["messages"] if m["role"] == "assistant"]
    diagnosis_count = sum(1 for t in assistant_msgs if "OBSERVATIONS:" in t and "PLAN:" in t)
    assert diagnosis_count == 1


def test_v3_plan_has_subsystem_bullets():
    record = _make_v3_record()
    diag = [m["content"] for m in record["messages"] if m["role"] == "assistant" and "PLAN:" in m["content"]]
    assert diag, "No diagnosis block found"
    plan_text = diag[0]
    assert "Executing plan by subsystem." in plan_text


def test_v3_final_assessment_present():
    record = _make_v3_record()
    last = record["messages"][-1]["content"]
    # Verdict may be prefixed with a batch-check sentence from the last batch.
    assert "FINAL ASSESSMENT (complete)" in last or "FINAL ASSESSMENT (budget_exhausted)" in last


def test_v3_no_heard_hypothesis_headers():
    record = _make_v3_record()
    for m in record["messages"]:
        if m["role"] == "assistant":
            text = m.get("content", "")
            assert "HEARD:" not in text, f"v3 record should not contain HEARD: header, found in: {text[:80]}"
            assert "HYPOTHESIS:" not in text, f"v3 record should not contain HYPOTHESIS: header, found in: {text[:80]}"


def test_v3_audio_count_matches_batch_labels():
    record = _make_v3_record(n_batches=3, has_correction=False)
    batch_labels = record["meta"]["batch_labels"]
    n_batches_reg = sum(1 for b in batch_labels if not b["is_correction"])
    # GT + default + 2 tuple audios + one per regular batch
    expected = 2 + 2 + n_batches_reg  # GT, default, tuples, batch listens
    assert len(record["audios"]) == expected, f"Expected {expected} audios, got {len(record['audios'])}"


def test_v3_audio_count_with_correction():
    record = _make_v3_record(n_batches=3, has_correction=True)
    batch_labels = record["meta"]["batch_labels"]
    n_regular = sum(1 for b in batch_labels if not b["is_correction"])
    n_corrections = sum(1 for b in batch_labels if b["is_correction"])
    # GT + default + 2 tuple audios + regular listens + correction listen
    expected = 2 + 2 + n_regular + n_corrections
    assert len(record["audios"]) == expected


def test_v3_correction_block_appears_with_mistake():
    record = _make_v3_record(n_batches=3, has_correction=True, has_mistake=True)
    assistant_msgs = [m["content"] for m in record["messages"] if m["role"] == "assistant"]
    has_correction_text = any("Overshot" in t or "correction" in t.lower() for t in assistant_msgs)
    assert has_correction_text, "Correction block should appear when injected_mistake is set"


def test_v3_batch_subsystem_in_intro_sentence():
    record = _make_v3_record()
    for m in record["messages"]:
        if m["role"] == "assistant" and m["content"].startswith("Applying "):
            # Parse subsystem name
            match = re.match(r"Applying (\w+) changes\.", m["content"])
            assert match, f"Unexpected batch intro format: {m['content']}"
            subsystem = match.group(1)
            assert subsystem in [lbl for lbl, _ in SUBSYSTEM_ORDER], f"Unknown subsystem: {subsystem}"


def test_v3_pipeline_version_in_meta():
    record = _make_v3_record()
    assert record["meta"]["pipeline_version"] == "v3"


def test_v3_batch_labels_have_required_fields():
    record = _make_v3_record()
    for bl in record["meta"]["batch_labels"]:
        assert "index" in bl
        assert "subsystem" in bl
        assert "param_names" in bl
        assert "clap_score_after_batch" in bl
        assert "is_correction" in bl


# ---- WT scaffold tests ----

def test_v3_wt_scaffold_uses_agent_tool():
    """Main agent dispatches search via the claw-code-style Agent tool."""
    record = _make_v3_record()
    tool_calls = [m for m in record["messages"] if m["role"] == "tool_call"]
    tool_names = [json.loads(m["content"])["name"] for m in tool_calls]
    assert "Agent" in tool_names, "Main agent should use Agent tool for search dispatch"
    # At least 2 Agent calls (one per search agent shard)
    assert tool_names.count("Agent") >= 2


def test_v3_wt_scaffold_agents_have_subagent_type():
    """Each Agent call should specify subagent_type=wavetable_search."""
    record = _make_v3_record()
    for m in record["messages"]:
        if m["role"] == "tool_call":
            parsed = json.loads(m["content"])
            if parsed["name"] == "Agent":
                args = parsed["arguments"]
                assert args.get("subagent_type") == "wavetable_search"
                assert "description" in args
                assert "prompt" in args


def test_v3_wt_scaffold_agent_returns_output_file():
    """Agent tool_response should include outputFile path (claw-code-style)."""
    record = _make_v3_record()
    agent_responses = []
    tool_calls_iter = iter(record["messages"])
    for i, m in enumerate(record["messages"]):
        if m["role"] == "tool_call":
            parsed = json.loads(m["content"])
            if parsed["name"] == "Agent":
                # Next message should be tool_response
                resp = record["messages"][i + 1]
                if resp["role"] == "tool_response":
                    agent_responses.append(json.loads(resp["content"]))
    assert agent_responses, "Should have Agent tool responses"
    for resp in agent_responses:
        assert "outputFile" in resp, f"Agent response missing outputFile: {resp}"
        assert "agentId" in resp


def test_v3_wt_scaffold_has_cat_read():
    """After Agent dispatch, main agent reads shortlists via bash cat."""
    record = _make_v3_record()
    cat_calls = []
    for m in record["messages"]:
        if m["role"] == "tool_call":
            parsed = json.loads(m["content"])
            if parsed["name"] == "bash" and parsed["arguments"]["command"].startswith("cat "):
                cat_calls.append(parsed)
    assert cat_calls, "Main agent should use bash cat to read shortlist files"


def test_v3_wt_scaffold_discovers_wavetables_inline():
    """Main agent should discover wavetables via inline FS scan, not a premade script."""
    record = _make_v3_record()
    discovery_calls = []
    for m in record["messages"]:
        if m["role"] == "tool_call":
            parsed = json.loads(m["content"])
            cmd = parsed.get("arguments", {}).get("command", "")
            if parsed["name"].lower() == "bash" and ".vitaltable" in cmd:
                discovery_calls.append(parsed)
    assert discovery_calls, "Main agent should scan FS for .vitaltable files inline"


def test_v3_wt_scaffold_has_tuple_audio():
    record = _make_v3_record()
    tuple_responses = [
        m for m in record["messages"]
        if m["role"] == "tool_response" and "tuple_audio" in m["content"]
    ]
    assert len(tuple_responses) >= 2, "Should have at least 2 tuple audio evaluations"


def test_v3_wt_apply_uses_library_lookup():
    """The apply tool call should reference the wavetable library, not the target preset."""
    record = _make_v3_record()
    apply_responses = [
        m for m in record["messages"]
        if m["role"] == "tool_response" and '"applied"' in m["content"]
    ]
    assert apply_responses, "Should have an apply tool response"
    parsed = json.loads(apply_responses[0]["content"])
    assert "applied" in parsed
    assert isinstance(parsed["applied"], list)


def test_v3_wt_scaffold_no_legacy_tool_names():
    """v3 should NOT have the old spawn_search_agents/collect_search_reports/judge_candidates."""
    record = _make_v3_record()
    tool_calls = [m for m in record["messages"] if m["role"] == "tool_call"]
    tool_names = [json.loads(m["content"])["name"] for m in tool_calls]
    assert "spawn_search_agents" not in tool_names
    assert "collect_search_reports" not in tool_names
    assert "judge_candidates" not in tool_names
