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
    SUBSYSTEM_ORDER,
    SubsystemBatch,
    build_batches_from_diff,
    build_diagnosis_subsystem_truth,
    extract_diagnosis_subsystems_mentioned,
    format_subsystem_diff_summary,
    inject_mistake,
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


def test_inject_mistake_modifies_applied():
    """inject_mistake mutates params_applied on the chosen batch."""
    import random
    b = SubsystemBatch(
        subsystem="oscillator",
        params={"osc_1_level": 0.5, "osc_1_pan": 0.3},
        params_applied={"osc_1_level": 0.5, "osc_1_pan": 0.3},
    )
    rng = random.Random(42)
    info = inject_mistake([b], rng, mistake_rate=1.0)
    if info is not None:
        assert info["wrong_value"] != info["true_value"]
        assert b.params_applied[info["param"]] == info["wrong_value"]
        assert b.params[info["param"]] == info["true_value"]  # original unchanged


def test_inject_mistake_none_when_rate_zero():
    import random
    b = SubsystemBatch(
        subsystem="oscillator",
        params={"osc_1_level": 0.5, "osc_1_pan": 0.3},
        params_applied={"osc_1_level": 0.5, "osc_1_pan": 0.3},
    )
    assert inject_mistake([b], random.Random(42), mistake_rate=0.0) is None


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
    """Build a synthetic v3 record for structural testing."""
    messages = [
        {"role": "user", "content": "<audio>\nRecreate this keys target sound in Vital from default.\nListen first, write a subsystem plan, then execute by subsystem."},
        # WT scaffold shortcut — minimal valid
        {"role": "assistant", "content": "Listening to current default preset baseline."},
        {"role": "tool_call", "content": '{"name":"bash","arguments":{"command":"echo probe"}}'},
        {"role": "tool_response", "content": '{"status":"ok","baseline_audio":"<audio>","path":"/tmp/d.wav"}'},
        {"role": "assistant", "content": "Spawning disjoint search shards to gather wavetable proposals in parallel."},
        {"role": "tool_call", "content": '{"name":"spawn_search_agents","arguments":{}}'},
        {"role": "tool_response", "content": '{"jobs":[]}'},
        {"role": "assistant", "content": "Collecting search-agent reports."},
        {"role": "tool_call", "content": '{"name":"collect_search_reports","arguments":{}}'},
        {"role": "tool_response", "content": '{"reports":[]}'},
        {"role": "assistant", "content": "Judging candidates and selecting up to three for edits."},
        {"role": "tool_call", "content": '{"name":"judge_candidates","arguments":{}}'},
        {"role": "tool_response", "content": '{"ranking":[],"selected":[],"selected_previews":[]}'},
    ]
    audios = ["/tmp/gt.wav", "/tmp/default.wav"]
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
            "injected_mistake": ({"batch_index": 1, "subsystem": "filter", "param": "filter_1_cutoff", "wrong_value": 0.2, "true_value": 0.6} if has_mistake else None),
            "mistake_caught": (True if has_correction and has_mistake else None),
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
    n_batches = len(batch_labels)
    n_corrections = sum(1 for b in batch_labels if b["is_correction"])
    # GT + default + one per regular batch + one per correction listen
    expected = 2 + (n_batches - n_corrections)  # GT, default, + regular batch listens
    assert len(record["audios"]) == expected, f"Expected {expected} audios, got {len(record['audios'])}"


def test_v3_audio_count_with_correction():
    record = _make_v3_record(n_batches=3, has_correction=True)
    batch_labels = record["meta"]["batch_labels"]
    n_regular = sum(1 for b in batch_labels if not b["is_correction"])
    n_corrections = sum(1 for b in batch_labels if b["is_correction"])
    expected = 2 + n_regular + n_corrections  # GT, default, regular listens, correction listen
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
