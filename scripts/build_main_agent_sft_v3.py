#!/usr/bin/env python3
"""Main-agent SFT pipeline v3 — Diagnose → Subsystem-Batched Execute.

Topology (one conversation per sample):

    Block 0  — WT search / judge scaffold (copied verbatim from v2)
    Block 1  — DIAGNOSIS: listen to GT + default, emit subsystem plan
    Block 2..K — SUBSYSTEM BATCHES: apply params per subsystem in one bash call,
                 listen after each batch, one-sentence perceptual check
    Block K+1 — CORRECTION (optional): name the overshoot, set corrective value
    Block K+2 — FINAL ASSESSMENT

Replaces v2's per-step HEARD/HYPOTHESIS/PLAN narration. The root quality issue
with v2 was that the LLM was asked to post-hoc rationalize an oracle's 17-step
plan, one step at a time. v3 flips the structure: one upfront plan grounded in
the GT-vs-default subsystem diff, then subsystem-batched execution with one
listen at the end of each batch.
"""
from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import random
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.agent_sft_common import (
    ClapEmbedder,
    assert_valid_ms_swift_multiturn_record,
    build_clap_shortlist_data,
    build_gt_similarity_pool,
    build_list_wavetables_slice_snippet,
    build_list_wavetables_total_snippet,
    build_name_embedding_map,
    build_render_probes_snippet,
    build_render_tuple_snippet,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    make_agent_id,
    select_probe_rows_by_name,
    write_agent_manifest,
    _bash_tool_response,
    _BUILD_CHUNK_HELPER,
    _emit_listen_sequence,
    _read_tool_response_audio,
    _REAPY_HELPER,
    _tool_call,
    _wrap_as_bash,
)
from scripts.build_main_agent_sft_v2 import (
    _build_listen_probe_command,
    _check_server_reachable,
    _llm_post,
    _step_remaining_gap,
)

# Build-time only: authoritative json_key → REAPER display name mapping,
# generated once from a live REAPER+Vital session via TrackFX_GetParamName.
# Covers 756 of 776 param_ranges keys; the 20 missing are macros and obscure
# random/LFO keytrack_tune params that never appear in generated presets.
_VITAL_DISPLAY_NAMES_PATH = ROOT / "maestro" / "synth" / "vital_display_names.json"
with open(_VITAL_DISPLAY_NAMES_PATH) as _f:
    _VITAL_DISPLAY_NAMES: dict[str, str] = json.load(_f)

# v3-specific tool specs: claw-code-style Agent tool + bash. Replaces the
# v2 spawn_search_agents / collect_search_reports / judge_candidates trio.
_V3_TOOL_SPECS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "Bash",
                "description": "Execute shell/Python commands for Vital search, edit, and listen passes.",
                "parameters": {
                    "type": "object",
                    "properties": {"command": {"type": "string"}},
                    "required": ["command"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Agent",
                "description": (
                    "Spawn a background sub-agent to perform a delegated task. "
                    "Returns immediately with an agentId, status, and outputFile path. "
                    "Read outputFile (e.g. via bash cat) to consume the sub-agent's result."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "subagent_type": {"type": "string", "description": "Type of sub-agent (e.g. wavetable_search)."},
                        "description": {"type": "string", "description": "Short task description."},
                        "prompt": {"type": "string", "description": "Full task prompt for the sub-agent."},
                        "name": {"type": "string", "description": "Optional name for the sub-agent."},
                    },
                    "required": ["subagent_type", "description", "prompt"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Read",
                "description": (
                    "Read a file from the filesystem. For audio files (.wav, .mp3, .flac), "
                    "returns the audio content for listening. For text/JSON files, returns "
                    "the file content as a string."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "file_path": {"type": "string", "description": "Absolute path to the file to read."},
                    },
                    "required": ["file_path"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "Skill",
                "description": (
                    "Load a named skill's SKILL.md contents into the conversation context. "
                    "Use at the start of a session to pick up plugin-specific instructions "
                    "and helper-script paths (e.g. Skill('vital') to load the Vital synth skill). "
                    "Available skills are listed in the system context by name + description — "
                    "pick the one whose description matches the task."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string", "description": "Skill name (e.g. 'vital')."},
                        "args": {"type": "string", "description": "Optional args to parameterise the skill."},
                    },
                    "required": ["skill"],
                },
            },
        },
    ],
    ensure_ascii=False,
)
from maestro.synth import path_gen as _pg
from maestro.synth.path_gen import _denormalize, _normalize, _param_family


# ---------------------------------------------------------------------------
# Subsystem taxonomy
# ---------------------------------------------------------------------------

# Order follows path_gen._STAGE_ORDER: core (osc, env, filter) → motion (lfo) →
# space (fx) → modulation → macro. Within core, split into osc / env / filter
# for narrative clarity.
SUBSYSTEM_ORDER: list[tuple[str, tuple[str, ...]]] = [
    ("oscillator", ("osc",)),
    ("envelope", ("env1", "env2")),
    ("filter", ("filter1", "filter2")),
    ("lfo", ("lfo",)),
    ("fx", ("chorus", "delay", "reverb", "distortion", "compressor", "phaser", "flanger", "eq")),
    ("modulation", ("modulation",)),
    ("macro", ("other",)),
]

_FAM_TO_SUBSYSTEM: dict[str, str] = {}
for _label, _fams in SUBSYSTEM_ORDER:
    for _f in _fams:
        _FAM_TO_SUBSYSTEM[_f] = _label


def presentation_subsystem(fam: str) -> str:
    """Map a path_gen _param_family string to a v3 presentation subsystem label."""
    return _FAM_TO_SUBSYSTEM.get(fam, "macro")


def _json_key_to_display(key: str) -> str:
    """Vital param key → display words (local copy — v2's variant is slightly different)."""
    abbrevs = {"osc": "Oscillator", "env": "Envelope", "lfo": "LFO", "eq": "EQ", "fx": "FX"}
    parts = key.split("_")
    return " ".join(abbrevs.get(p.lower(), p.capitalize()) for p in parts)


def _json_key_to_reaper_display(key: str) -> str:
    """Map a Vital JSON key to the exact REAPER display name.

    Uses the authoritative mapping built from a live REAPER+Vital session.
    Falls back to heuristic title-case expansion for any key not in the map
    (shouldn't happen for params in param_ranges.json).
    Build-time only — the model never sees this function.
    """
    exact = _VITAL_DISPLAY_NAMES.get(key)
    if exact:
        return exact
    return _json_key_to_display(key)


# ---------------------------------------------------------------------------
# Batch construction — diff-based, independent of path_gen steps
# ---------------------------------------------------------------------------


@dataclass
class SubsystemBatch:
    subsystem: str                      # presentation label: "oscillator", "filter", ...
    params: dict[str, float]            # {name: target_norm}  (always GT truth)
    params_applied: dict[str, float]    # {name: applied_norm} (= params unless mistake injected)
    is_correction: bool = False
    audio_wav: Path | None = None       # rendered per-batch audio
    mistake: dict | None = None         # {"name": str, "wrong_value": float, "true_value": float}


def build_batches_from_diff(
    target_preset: dict,
    init_preset: dict,
    threshold: float = 0.05,
) -> list[SubsystemBatch]:
    """Diff target vs init, bucket changed params by subsystem, return ordered batches.

    No dependency on path_gen iterations.  One batch per presentation subsystem
    that has at least one param differing by more than ``threshold`` normalized units.
    """
    truth = build_diagnosis_subsystem_truth(target_preset, init_preset, threshold)
    target_settings = target_preset.get("settings", {})

    batches: list[SubsystemBatch] = []
    for label, _ in SUBSYSTEM_ORDER:
        names = truth.get(label)
        if not names:
            continue
        params: dict[str, float] = {}
        for n in names:
            norm = _normalize(n, target_settings[n])
            if norm is not None:
                params[n] = float(norm)
        if params:
            batches.append(SubsystemBatch(
                subsystem=label,
                params=params,
                params_applied=dict(params),  # copy — identical until mistake injected
            ))
    return batches


def inject_mistake(
    batches: list[SubsystemBatch],
    rng: "random.Random",
    mistake_rate: float = 0.20,
) -> dict | None:
    """With probability ``mistake_rate``, pick one param in one batch and overshoot it.

    Mutates ``batch.params_applied`` in place.  Returns a mistake-info dict
    ``{batch_index, subsystem, param, wrong_value, true_value}`` or None.
    """
    import random as _random
    if not batches or rng.random() >= mistake_rate:
        return None
    # Pick a batch with enough params to make the mistake meaningful.
    eligible = [(i, b) for i, b in enumerate(batches) if len(b.params) >= 2]
    if not eligible:
        return None
    bi, batch = rng.choice(eligible)
    param_name = rng.choice(sorted(batch.params.keys()))
    true_norm = batch.params[param_name]

    # Overshoot: push away from target by ≥0.20 norm.
    direction = 1.0 if true_norm < 0.5 else -1.0
    wrong_norm = true_norm + direction * rng.uniform(0.25, 0.50)
    wrong_norm = max(0.0, min(1.0, wrong_norm))
    if abs(wrong_norm - true_norm) < 0.20:
        return None  # clamping collapsed the gap; skip

    batch.params_applied[param_name] = wrong_norm
    info = {
        "batch_index": bi,
        "subsystem": batch.subsystem,
        "param": param_name,
        "wrong_value": float(wrong_norm),
        "true_value": float(true_norm),
    }
    batch.mistake = info
    return info


def corrupt_transcription_notes(notes: list[dict], rng) -> tuple[list[dict], dict]:
    """Create a 'wrong' version of a transcribed note list by altering one
    note's pitch by +-1 or +-2 semitones. Returns (wrong_notes, info).

    Notes are dicts with keys {"pitch", "start_s", "dur_s", "velocity"}.
    """
    import copy
    if not notes:
        return notes, {}
    wrong = copy.deepcopy(notes)
    idx = rng.randrange(len(wrong))
    delta = rng.choice([-2, -1, 1, 2])
    original_pitch = int(wrong[idx]["pitch"])
    wrong[idx]["pitch"] = max(0, min(127, original_pitch + delta))
    return wrong, {
        "note_idx": idx,                            # which note was altered (0-based)
        "start_s": float(wrong[idx]["start_s"]),
        "wrong_pitch": int(wrong[idx]["pitch"]),
        "correct_pitch": int(original_pitch),
        "delta_semitones": int(delta),
    }


# ---------------------------------------------------------------------------
# Vita rendering — fresh per-batch audio (no iter_wav dependency)
# ---------------------------------------------------------------------------

import numpy as np
import soundfile as sf
from maestro.render.vital import SAMPLE_RATE
from maestro.render.dawdreamer import render_preset_audio, make_probe_notes, notes_from_dicts


def render_cumulative_audio(
    cumulative_preset: dict,
    notes: list,
    out_path: Path,
    tail_s: float = 1.0,
) -> Path:
    """Render audio for a cumulative preset state via DawDreamer and write to ``out_path``."""
    render_preset_audio(cumulative_preset, notes, out_path=out_path, tail_s=tail_s)
    return out_path


# ---------------------------------------------------------------------------
# Subsystem diff summary (ground-truth diff of target vs init preset)
# ---------------------------------------------------------------------------


def build_diagnosis_subsystem_truth(
    target_preset: dict,
    init_preset: dict,
    threshold: float = 0.05,
) -> dict[str, list[str]]:
    """Compute {presentation_subsystem → [param_names]} of all params that differ.

    Mirrors path_gen's change-detection (norm diff > threshold), with modulation slots
    filtered to those that are occupied by a real route in the target.
    """
    target_settings = target_preset.get("settings", {})
    init_settings = init_preset.get("settings", {})

    occupied: set[str] = set()
    for i, mod in enumerate(target_settings.get("modulations", []) or []):
        if mod.get("source") or mod.get("destination"):
            occupied.add(str(i + 1))
    mod_re = re.compile(r"modulation_(\d+)_(?:bypass|amount|stereo|bipolar|power)")

    out: dict[str, list[str]] = {label: [] for label, _ in SUBSYSTEM_ORDER}
    for name, tgt in target_settings.items():
        if isinstance(tgt, (list, dict)):
            continue
        m = mod_re.fullmatch(name)
        if m and m.group(1) not in occupied:
            continue
        norm_tgt = _normalize(name, tgt)
        if norm_tgt is None:
            continue
        init_native = init_settings.get(name, tgt)
        norm_init = _normalize(name, init_native)
        if norm_init is None:
            out[presentation_subsystem(_param_family(name))].append(name)
            continue
        if abs(norm_tgt - norm_init) > threshold:
            out[presentation_subsystem(_param_family(name))].append(name)
    return {k: v for k, v in out.items() if v}


def format_subsystem_diff_summary(truth: dict[str, list[str]]) -> str:
    """Render diff dict as prose: 'oscillator (12 params); envelope (4); ...'"""
    if not truth:
        return "none — the default preset already matches the target."
    parts = []
    for label, _ in SUBSYSTEM_ORDER:
        names = truth.get(label)
        if not names:
            continue
        parts.append(f"{label} ({len(names)} params)")
    return "; ".join(parts)


# ---------------------------------------------------------------------------
# Action snippet — apply a batch of params via Lua ReaScript in REAPER
# ---------------------------------------------------------------------------


def build_batch_action_snippet(
    params_display_norm: dict[str, float],
    track_idx: int = 0,
    fx_idx: int = 0,
) -> str:
    """Emit a Python+reapy snippet that sets plugin params via REAPER's native API.

    params_display_norm — {reaper_display_name: normalized_0_to_1_value}.
    """
    params_dict = json.dumps(
        {k: round(float(v), 6) for k, v in sorted(params_display_norm.items())},
        ensure_ascii=False,
    )
    snippet = (
        _REAPY_HELPER
        + f"params = {params_dict}\n"
        f"with reapy.inside_reaper():\n"
        f"    track = RPR.GetTrack(0, {track_idx})\n"
        f"    n = RPR.TrackFX_GetNumParams(track, {fx_idx})\n"
        f"    idx_of = {{}}\n"
        f"    for i in range(n):\n"
        f"        _, _, _, _, name, _ = RPR.TrackFX_GetParamName(track, {fx_idx}, i, '', 2048)\n"
        f"        if name in params:\n"
        f"            idx_of[name] = i\n"
        f"    applied = 0\n"
        f"    not_found = []\n"
        f"    for name, val in params.items():\n"
        f"        if name in idx_of:\n"
        f"            RPR.TrackFX_SetParam(track, {fx_idx}, idx_of[name], val)\n"
        f"            applied += 1\n"
        f"        else:\n"
        f"            not_found.append(name)\n"
        f"print(json.dumps({{'status': 'ok', 'applied': applied, 'not_found': not_found}}))\n"
    )
    return _wrap_as_bash(snippet)


def denormalize_batch_params(params_norm: dict[str, float]) -> dict[str, float]:
    """Map {name: norm} to {name: native} using path_gen's param ranges."""
    return {n: _denormalize(n, v) for n, v in params_norm.items()}


# ---------------------------------------------------------------------------
# Omni Stage 1 (audio) helpers
# ---------------------------------------------------------------------------


def _b64(path: str | Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def omni_stage1_diagnose(
    gt_wav: str,
    target_preset: dict,
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """Stage 1: Omni listens to TARGET only, grounded by a perceptual preset summary.

    Multi-audio target/default comparison is out-of-distribution for Omni — it
    tends to hallucinate modulation, flip attack direction, and produce generic
    differential prose. Instead we compute a perceptual-bucket summary from the
    target preset (no numbers, no param names) and pass it as a grounding prior
    alongside the target audio. The prompt explicitly forbids citing details
    only available from the summary; the summary is a hallucination safety net,
    not a cheat sheet.

    Output goes into Stage 2's DIAGNOSIS prompt as observations.
    """
    from scripts.preset_perceptual_summary import (
        summarize_preset_perceptual,
        GROUNDED_OBSERVATIONS_PROMPT_TEMPLATE,
    )
    preset_summary = summarize_preset_perceptual(target_preset)
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(gt_wav)}"}},
        {"type": "text", "text": GROUNDED_OBSERVATIONS_PROMPT_TEMPLATE.format(
            preset_summary=preset_summary,
        )},
    ]
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {
                "model": omni_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 260,
                "temperature": 0.4,
            },
            timeout=180.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            "The sound has a distinct timbral character and envelope shape that "
            "should be reproducible through careful oscillator, envelope, filter, "
            "and effect choices."
        )


def _describe_audio_solo(
    wav: str,
    prompt_text: str,
    omni_server: str,
    omni_model: str,
    max_tokens: int = 140,
    fallback: str = "",
) -> str:
    """Single-audio Omni call."""
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(wav)}"}},
        {"type": "text", "text": prompt_text},
    ]
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model,
             "messages": [{"role": "user", "content": content}],
             "max_tokens": max_tokens, "temperature": 0.4},
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return fallback


def omni_stage1_verdict(
    gt_wav: str,
    final_wav: str,
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """Single-audio policy for the final verdict: describe target and final
    recreation in separate calls, then synthesize the verdict via a text-only
    call. Avoids the position confusion that the original two-audio comparison
    prompt could produce."""
    describe_prompt = (
        f"Describe this {archetype} sound's TIMBRE in 1-2 sentences: harmonic content, "
        f"brightness, texture, attack/decay shape, and distinctive effects you hear. "
        f"Focus on timbral qualities only."
    )
    target_desc = _describe_audio_solo(
        gt_wav, describe_prompt, omni_server, omni_model, max_tokens=160,
        fallback="The target sound has a clear timbral character.",
    )
    final_desc = _describe_audio_solo(
        final_wav, describe_prompt, omni_server, omni_model, max_tokens=160,
        fallback="The recreation has a clear timbral character.",
    )

    synth_prompt = (
        f"You are a music production AI doing a final review.\n\n"
        f"Target timbre (described from an isolated listen):\n  {target_desc}\n\n"
        f"Recreation timbre (described from an isolated listen):\n  {final_desc}\n\n"
        f"In 2 sentences: what matches well, and what (if anything) still differs. "
        f"Be specific and honest. No snake_case. No kHz numbers."
    )
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {"model": omni_model,
             "messages": [{"role": "user", "content": synth_prompt}],
             "max_tokens": 180, "temperature": 0.4},
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            "The recreation is close to the target in overall timbre and envelope shape. "
            "Fine details in the upper harmonics and modulation movement may still differ slightly."
        )


# ---------------------------------------------------------------------------
# Stage 2 (text) helpers — diagnosis / batch check / correction / verdict
# ---------------------------------------------------------------------------


def stage2_diagnosis(
    perceptual_obs: str,
    subsystem_diff_summary: str,
    subsystems_needed: list[str],
    archetype: str,
    stage2_server: str,
    stage2_model: str,
) -> str:
    """Stage 2: write OBSERVATIONS + PLAN from Stage 1 obs + subsystem diff.

    Plan is qualitative: one bullet per subsystem that needs changes.
    """
    bullets_hint = ", ".join(subsystems_needed) if subsystems_needed else "none"
    prompt = (
        f"You are a music production AI agent writing the upfront plan for recreating "
        f"a synth sound in Vital.\n\n"
        f"--- Perceptual observations (from listening to the target) ---\n"
        f"{perceptual_obs}\n\n"
        f"--- Ground truth: subsystems that need changes ---\n"
        f"{subsystem_diff_summary}\n\n"
        f"Write exactly two sections:\n\n"
        f"OBSERVATIONS: 2-3 sentences describing the TARGET sound's perceptual character "
        f"in your own voice, grounded in the perceptual observations above. Describe the "
        f"sound directly (what it sounds like, its tonal color, attack feel, motion). "
        f"DO NOT use comparative framing like 'the target has', 'in contrast', 'unlike the "
        f"default', 'the default lacks'. DO NOT mention a baseline or default preset at all. "
        f"Talk about the sound as if you just heard it fresh, not against any reference. "
        f"No snake_case parameter names, no **bold** headers, no kHz numbers.\n\n"
        f"PLAN: A bulleted list. Exactly one bullet per subsystem in this list (in order): "
        f"{bullets_hint}. Each bullet starts with the subsystem name capitalised (e.g. "
        f"\"• Oscillator:\"), followed by one short qualitative sentence about what change "
        f"is needed there (e.g. \"swap to a brighter wavetable and add unison detune\"). "
        f"Do NOT write any numeric values or exact param names. Do NOT use snake_case. "
        f"End the PLAN section with the exact line: \"Executing plan by subsystem.\"\n\n"
        f"Be concise and specific."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.7,
            },
            timeout=180.0,
        )
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        bullets = "\n".join(
            f"• {s.capitalize()}: adjust to move the preset toward the target."
            for s in subsystems_needed
        )
        return (
            f"OBSERVATIONS: {perceptual_obs[:240]}\n\n"
            f"PLAN:\n{bullets}\nExecuting plan by subsystem."
        )


def _extract_plan_bullet(plan_text: str, subsystem: str) -> str:
    """Pull the single plan bullet for the given subsystem from the PLAN section.

    Plan bullets look like: "• Oscillator: swap to brighter wavetable, add unison detune".
    Returns just the sentence part (no subsystem prefix, no bullet glyph), or a
    generic fallback if not found.
    """
    if not plan_text or not subsystem:
        return ""
    body = plan_text.split("PLAN:", 1)[-1] if "PLAN:" in plan_text else plan_text
    sub_lower = subsystem.lower()
    for line in body.splitlines():
        line = line.strip().lstrip("•-* \t")
        if ":" not in line:
            continue
        label, _, rest = line.partition(":")
        if label.strip().lower() == sub_lower:
            return rest.strip()
    return ""


def _humanize_param_name(p: str) -> str:
    """env_1_attack → 'env 1 attack', lfo_2_frequency → 'lfo 2 frequency'."""
    return p.replace("_", " ").strip()


def _format_param_deltas(
    param_before_after: list[tuple[str, float, float]], max_rows: int = 10
) -> str:
    """Format param deltas for the batch-check prompt.

    Shows each param as 'name: before → after' with a direction word appended.
    Numbers are included so the model can judge magnitude, but the prompt
    forbids echoing them in the output.
    """
    rows: list[str] = []
    for name, before, after in param_before_after[:max_rows]:
        try:
            before_f = float(before) if before is not None else None
            after_f = float(after) if after is not None else None
        except (TypeError, ValueError):
            before_f, after_f = None, None
        direction = ""
        if before_f is not None and after_f is not None:
            if abs(after_f - before_f) < 1e-6:
                direction = "(unchanged)"
            elif abs(after_f) < 1e-4 and abs(before_f) > 1e-4:
                direction = "(disengaged)"
            elif abs(before_f) < 1e-4 and abs(after_f) > 1e-4:
                direction = "(engaged)"
            elif after_f > before_f:
                pct = (after_f - before_f) / max(abs(before_f), 1e-4)
                direction = "(substantially increased)" if pct > 0.5 else "(slightly increased)"
            else:
                pct = (before_f - after_f) / max(abs(before_f), 1e-4)
                direction = "(substantially decreased)" if pct > 0.5 else "(slightly decreased)"
        bstr = f"{before_f:.3f}" if before_f is not None else "?"
        astr = f"{after_f:.3f}" if after_f is not None else "?"
        rows.append(f"  - {_humanize_param_name(name)}: {bstr} → {astr} {direction}")
    if len(param_before_after) > max_rows:
        rows.append(f"  - (+{len(param_before_after) - max_rows} more)")
    return "\n".join(rows) if rows else "  (no params changed)"


def stage2_batch_check(
    subsystem: str,
    plan_bullet: str,
    param_deltas: list[tuple[str, float, float]],
    prior_checks: list[str],
    archetype: str,
    stage2_server: str,
    stage2_model: str,
    is_final: bool = False,
    n_params_applied: int = 0,
) -> str:
    """Stage 2: write one short sentence describing what THIS batch changed.

    Grounded in the actual plan bullet (so narrations don't contradict the plan)
    and in concrete before→after param values (so narrations describe real
    direction and magnitude, not templated prose). Variation across samples
    comes naturally from different presets having different deltas.
    """
    recent = " / ".join(prior_checks[-2:]) if prior_checks else ""
    recent_hint = (
        f"Prior batch narrations (do not repeat these phrasings): \"{recent}\"" if recent else ""
    )
    final_hint = (
        "This is the LAST planned batch — the preset is now nearly complete." if is_final else ""
    )
    plan_hint = (
        f"Plan bullet for this subsystem: \"{plan_bullet}\"" if plan_bullet else
        f"No explicit plan bullet for {subsystem}; describe the changes directly."
    )
    delta_block = _format_param_deltas(param_deltas)

    prompt = (
        f"You are a music production AI. You just applied {n_params_applied} {subsystem} "
        f"parameter edits for a Vital preset.\n\n"
        f"{plan_hint}\n\n"
        f"Actual param changes this batch (before → after):\n{delta_block}\n\n"
        f"{recent_hint}\n{final_hint}\n\n"
        f"Write EXACTLY ONE sentence (under 28 words) describing what THIS batch did to "
        f"the sound. Translate the numeric changes into perceptual effects — a producer "
        f"reading your sentence should be able to tell what direction the {subsystem} "
        f"moved (longer/shorter, brighter/darker, engaged/disengaged, deeper/subtler).\n\n"
        f"Rules:\n"
        f"  - Stay consistent with the plan bullet above — if it says 'disengage', do NOT "
        f"describe added motion.\n"
        f"  - Ground your claims in the actual param deltas shown — describe the real "
        f"direction of change, not generic subsystem prose.\n"
        f"  - Do NOT cite parameter names, numbers, snake_case tokens, or kHz values.\n"
        f"  - Do NOT mention effects from other subsystems (no 'chorus' or 'phaser' in an "
        f"LFO batch, no 'reverb' in an FX batch that only edited chorus).\n"
        f"  - Natural prose, no **bold**."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 120,
                "temperature": 0.7,
            },
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip().split("\n")[0]
    except Exception:
        return f"{subsystem.capitalize()} edits applied consistent with the plan."


def stage2_correction_intro(
    subsystem: str,
    param_display_names: list[str],
    mistake_info: dict | None,
    archetype: str,
    stage2_server: str,
    stage2_model: str,
) -> str:
    """Stage 2: one sentence naming the overshoot and announcing the fix."""
    if mistake_info:
        disp = _json_key_to_display(mistake_info["param"])
        direction = "too far" if abs(mistake_info["wrong_value"]) > abs(mistake_info["true_value"]) else "off target"
        gist = f"{disp} was set {direction} during the earlier {mistake_info['subsystem']} batch"
    else:
        gist = f"values from an earlier batch need correction in {subsystem}"
    params_str = ", ".join(param_display_names[:3]) or "several parameters"
    prompt = (
        f"You are a music production AI. A previous edit overshot: {gist}. "
        f"Now correcting {params_str} back to the planned target values.\n\n"
        f"Write EXACTLY ONE sentence announcing the correction, naming the subsystem and "
        f"the direction of the overshoot. Natural language, under 30 words, no snake_case, "
        f"no **bold**."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0.7,
            },
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip().split("\n")[0]
    except Exception:
        return f"Overshot on {subsystem} — backing off {params_str} to the planned values."


def stage2_verdict(
    perceptual_obs: str,
    residual_delta_summary: str,
    path_complete: bool = True,
    archetype: str = "",
    stage2_server: str = "",
    stage2_model: str = "",
) -> str:
    """Stage 2: write FINAL ASSESSMENT grounded in a perceptual residual-delta summary.

    The residual summary lists the top 5 concrete differences between the target
    preset and the final cumulative preset, in perceptual language (no numbers,
    no param names). The prompt forbids generic 'envelope N' pattern-matching
    and requires the model to cite one of the specific residuals.
    """
    prompt = (
        f"You are a music production AI writing the final assessment of a "
        f"synth recreation.\n\n"
        f"Perceptual review of the recreation:\n{perceptual_obs}\n\n"
        f"Actual residual differences (what still differs between target and final, "
        f"sorted by magnitude):\n{residual_delta_summary}\n\n"
        f"Write a single line beginning with 'FINAL ASSESSMENT: ' followed "
        f"by exactly 2 short sentences.\n"
        f"  Sentence 1: what the recreation captures well about the target's character.\n"
        f"  Sentence 2: the most important remaining difference — MUST cite one of the "
        f"specific residuals from the list above in perceptual terms (e.g. 'attack is still "
        f"too plucky', 'filter should be darker', 'needs more unison detune'). Do NOT "
        f"default to generic pattern-matching like 'envelope 6' — describe the actual "
        f"audible problem.\n"
        f"Rules: no snake_case tokens, no **bold**, no kHz numbers, no parameter names."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 160,
                "temperature": 0.7,
            },
            timeout=120.0,
        )
        text = r["choices"][0]["message"]["content"].strip()
        if not text.startswith("FINAL ASSESSMENT"):
            text = f"FINAL ASSESSMENT: {text}"
        return text
    except Exception:
        if path_complete:
            return (
                f"FINAL ASSESSMENT: The recreation captures the target's core "
                f"character. No significant residuals remain."
            )
        first_residual = residual_delta_summary.split("\n", 1)[0].lstrip("- ").rstrip()
        return (
            f"FINAL ASSESSMENT: The recreation captures the target's core "
            f"timbre. The most notable remaining difference is that {first_residual}."
        )


# ---------------------------------------------------------------------------
# Parsing helpers (for meta labels)
# ---------------------------------------------------------------------------


_SUBSYSTEM_ALIASES = {
    "oscillator": ["oscillator", "oscillators", "osc"],
    "envelope": ["envelope", "envelopes", "amp envelope", "amp env"],
    "filter": ["filter", "filters", "low-pass", "lowpass", "high-pass", "highpass", "band-pass"],
    "lfo": ["lfo", "lfos"],
    "fx": ["fx", "effects", "reverb", "delay", "chorus", "distortion", "compressor", "phaser", "flanger", "eq", "equalizer"],
    "modulation": ["modulation", "mod matrix", "mod routes", "modulation routes", "routes"],
    "macro": ["macro", "macros"],
}


def extract_diagnosis_subsystems_mentioned(diagnosis_text: str) -> list[str]:
    """Parse DIAGNOSIS text and return presentation subsystems that appear."""
    text = diagnosis_text.lower()
    out: list[str] = []
    for label, aliases in _SUBSYSTEM_ALIASES.items():
        if any(re.search(rf"\b{re.escape(a)}\b", text) for a in aliases):
            out.append(label)
    # preserve canonical order
    return [lbl for lbl, _ in SUBSYSTEM_ORDER if lbl in out]


# ---------------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------------


def _init_preset() -> dict:
    return copy.deepcopy(_pg._INIT_PRESET)


def build_record(
    *,
    entry: dict,
    args: argparse.Namespace,
    embedder: ClapEmbedder,
    shortlist_data: dict,
    selected_by_name: dict,
    wavetable_lib: list[dict],
    index_rows: list[dict],
    candidate_audio: dict[str, Path],
    stage2_server: str,
    stage2_model: str,
    serial_lock: threading.Lock,
    notes: list,
) -> dict | None:
    """Build one v3 SFT record. Returns None to skip."""
    import time as _time
    _t0 = _time.monotonic()
    def _log(msg: str) -> None:
        elapsed = _time.monotonic() - _t0
        print(f"  [{sample_id}] {elapsed:6.1f}s  {msg}", flush=True)

    sample_id = str(entry["sample_id"])
    archetype = str(entry.get("archetype", "synth"))
    target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
    default_audio_path = Path(entry["default_wav"]) if entry.get("default_wav") else None
    if default_audio_path is None:
        return None

    # Load probe notes from this sample's source MIDI so every probe render
    # plays the same melody as the target. Falls back to the caller-supplied
    # `notes` (legacy fixed 4-triad pattern) if no source MIDI is available.
    source_midi_path = entry.get("source_midi_path")
    if source_midi_path and Path(source_midi_path).exists():
        try:
            from scripts.build_transcription_agent_sft_v3 import load_notes_from_midi  # type: ignore
            _midi_notes = load_notes_from_midi(source_midi_path)
            if _midi_notes:
                notes = notes_from_dicts(_midi_notes)
        except Exception:
            pass

    # Resolve target preset from manifest entry (path_file → target_preset_path).
    target_preset_path = entry.get("target_preset_path")
    if not target_preset_path:
        path_file = entry.get("path_file")
        if path_file:
            with open(path_file) as f:
                pd = json.load(f)
            target_preset_path = pd.get("target_preset_path")
    if not target_preset_path:
        return None
    with open(target_preset_path) as f:
        target_preset = json.load(f)

    init_preset = _init_preset()

    # Per-sample deterministic RNG
    sid_seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
    sample_rng = random.Random(int(args.seed) + sid_seed)

    # ---- WT search (claw-code-style Agent dispatch + file-based handoff) ----
    gt_names_list = list(extract_gt_wavetable_names(Path(target_preset_path)))
    if not gt_names_list:
        return None

    # Build dense name↔idx mapping (matches list_wavetables.py / search agent)
    _seen_wt: set[str] = set()
    _unique_wts: list[str] = []
    for wt in wavetable_lib:
        if not isinstance(wt, dict) or "name" not in wt:
            continue
        if wt["name"] in _seen_wt:
            continue
        _seen_wt.add(wt["name"])
        _unique_wts.append(wt["name"])
    idx_to_name_full: dict[int, str] = {i: n for i, n in enumerate(_unique_wts)}
    name_to_idx_full: dict[str, int] = {n: i for i, n in idx_to_name_full.items()}
    total_named = len(_unique_wts)

    # Active oscillators from target preset (determines how many WTs we need)
    active_oscs: list[int] = []
    for osc_idx in range(3):
        on_key = f"osc_{osc_idx + 1}_on"
        level_key = f"osc_{osc_idx + 1}_level"
        osc_on = float(target_preset.get("settings", {}).get(on_key, 0.0)) > 0.5
        osc_level = float(target_preset.get("settings", {}).get(level_key, 0.0)) > 0.01
        if osc_on and osc_level:
            active_oscs.append(osc_idx)
    if not active_oscs:
        active_oscs = [0]

    # Slice planning
    slice_size = int(getattr(args, "candidates_per_slice", 48))
    n_agents = int(args.num_agents)
    max_rounds = int(getattr(args, "max_search_rounds", 3))
    stride = max(1, total_named // n_agents)

    def _compute_slices(base: int) -> list[int]:
        starts = []
        for i in range(n_agents):
            start = (base + i * stride) % total_named
            if start + slice_size > total_named:
                start = max(0, total_named - slice_size)
            starts.append(start)
        return starts

    # Base offset: for most samples, rotate to cover GT in round 1.
    # For ~force_research_rate of samples, deliberately skip the rotation so
    # the first-round search misses GT and the re-search branch triggers.
    gt_idxs = [name_to_idx_full[n] for n in gt_names_list if n in name_to_idx_full]
    force_research_rate = float(getattr(args, "force_research_rate", 0.30))
    force_miss = sample_rng.random() < force_research_rate

    base_offset = sample_rng.randrange(stride)
    slice_starts = _compute_slices(base_offset)
    if gt_idxs and not force_miss:
        step = max(1, stride // 4)
        tried = 0
        while not any(s <= gi < (s + slice_size) for s in slice_starts for gi in gt_idxs) and tried < total_named:
            base_offset = (base_offset + step) % stride
            slice_starts = _compute_slices(base_offset)
            tried += step

    # ---- Begin messages ----
    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path), str(default_audio_path)]

    # Edge case (~5%): user says "recreate this sound" without selecting an audio
    # clip in REAPER. Model should recognise the missing <audio> and ask the
    # user to select a clip. Single-turn conversation, no search/diagnose flow.
    no_audio_rate = float(getattr(args, "no_audio_rate", 0.05))
    if sample_rng.random() < no_audio_rate:
        messages.append({
            "role": "user",
            "content": "Recreate this sound in Vital.",
        })
        messages.append({
            "role": "assistant",
            "content": (
                "I don't see an audio clip attached to your message. Please select "
                "the audio item in REAPER that you want me to recreate (click on the "
                "clip so it's highlighted) and try again — I need to hear the target "
                "sound before I can start matching it in Vital."
            ),
        })
        record = {
            "id": sample_id,
            "task_type": "main",
            "tools": _V3_TOOL_SPECS,
            "messages": messages,
            "audios": [],  # no audios attached
            "assets": {
                "target_audio": "", "current_audio": "",
                "candidate_audio": [], "selected_candidates": [], "selected_tuples": [],
            },
            "labels": {"gt_wavetable_names": gt_names_list, "applied_wavetables": []},
            "meta": {
                "pipeline_version": "v3",
                "sample_id": sample_id,
                "archetype": archetype,
                "agent": "main",
                "variant": "no_audio_selected",  # flag for grader to skip normal metrics
                "batch_labels": [],
                "diagnosis_subsystems_mentioned": [],
                "diagnosis_subsystems_truth": [],
                "injected_mistake": None,
                "mistake_caught": None,
                "path_complete": False,
                "n_remaining": 0,
                "commentary_mode": "two_stage",
                "num_agents": int(args.num_agents),
                "pool_top_k": int(args.pool_top_k),
                "max_batches": int(args.max_batches),
                "mistake_rate": float(args.mistake_rate),
            },
        }
        assert_valid_ms_swift_multiturn_record(record)
        return record

    messages.append({
        "role": "user",
        "content": (
            "<audio>\nRecreate this sound in Vital."
        ),
    })

    # Skill discovery + load. Matches claw-code's pattern: the agent lists the
    # SKILL.md files it can see, picks the one whose description matches the
    # task, then invokes the Skill tool to load the contents. At inference the
    # harness resolves the skill name via its discovery roots and returns the
    # SKILL.md as a tool response; at build time we read the same file from
    # disk so train-time tokens match.
    _skill_name = "vital"
    _skills_root = Path(__file__).resolve().parents[1] / "skills"
    _skill_md_path = _skills_root / _skill_name / "SKILL.md"
    _available_skill_paths = sorted(str(p.relative_to(_skills_root.parent)) for p in _skills_root.glob("*/SKILL.md"))
    try:
        _skill_md_text = _skill_md_path.read_text()
    except Exception:
        _skill_md_text = ""
    # Parse description from YAML frontmatter (best-effort — stops at next top-level key)
    _skill_description = ""
    if _skill_md_text.startswith("---"):
        _fm_end = _skill_md_text.find("---", 3)
        if _fm_end != -1:
            _fm = _skill_md_text[3:_fm_end]
            _in_desc = False
            _desc_lines: list[str] = []
            for _line in _fm.splitlines():
                if _line.startswith("description:"):
                    _in_desc = True
                    _desc_lines.append(_line[len("description:"):].strip())
                    continue
                if _in_desc:
                    # continuation if indented; stop at next top-level key
                    if _line.startswith((" ", "\t")):
                        _desc_lines.append(_line.strip())
                    else:
                        break
            _skill_description = " ".join(_desc_lines).strip()

    _log("start — skill discovery + transcription")
    # Step 1: discover available skills via filesystem
    messages.append({
        "role": "assistant",
        "content": "Let me see which skills are available for this plugin.",
    })
    messages.append(_tool_call("Bash", {"command": "ls skills/*/SKILL.md"}))
    messages.append(_bash_tool_response("\n".join(_available_skill_paths) + "\n"))

    # Step 2: load the matching skill via the Skill tool
    messages.append({
        "role": "assistant",
        "content": f"The {_skill_name} skill matches. Loading it for the plugin-specific instructions, helper-script paths, and recreation strategy.",
    })
    messages.append(_tool_call("Skill", {"skill": _skill_name, "args": ""}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps({
            "skill": _skill_name,
            "path": str(_skill_md_path),
            "args": "",
            "description": _skill_description,
            "prompt": _skill_md_text,
        }, ensure_ascii=False),
    })

    # Listen to baseline: probe → BashCommandOutput, then read → audio
    messages.append({"role": "assistant", "content": "Skill loaded. Probing default preset baseline."})
    messages.append(_tool_call("Bash", {"command": _build_listen_probe_command(default_audio_path)}))
    _emit_listen_sequence(messages, audio_assets, default_audio_path,
                          listen_text="Listening to the default preset.")

    # --- TRANSCRIPTION BLOCK ---
    # Before search, create a REAPER track and dispatch the melody_transcription
    # subagent to populate it with MIDI notes. The subagent's Agent tool_response
    # already reports {status: completed, outputFile}; the main agent never
    # actually inspects the note list — it just acknowledges and moves on —
    # so there is no `cat` of the transcription file.
    source_midi_path = entry.get("source_midi_path")
    transcription_summary_text = ""
    _trans_mistake_info: dict | None = None
    _trans_output_file: str | None = None
    if source_midi_path and Path(source_midi_path).exists():
        from scripts.build_transcription_agent_sft_v3 import (  # type: ignore
            load_notes_from_midi,
        )
        try:
            _trans_notes = load_notes_from_midi(source_midi_path)
        except Exception:
            _trans_notes = []
        if _trans_notes:
            _trans_n_notes = len(_trans_notes)
            _trans_duration_s = round(
                max(n["start_s"] + n["dur_s"] for n in _trans_notes), 2,
            )
            _trans_track_idx = 0
            _trans_track_name = "target_melody"
            _trans_agent_id = make_agent_id(sample_id, "melody_transcription")
            _trans_agent_dir = f"/tmp/agents/{sample_id}"
            _trans_output_file = f"{_trans_agent_dir}/{_trans_agent_id}.json"
            _trans_manifest_file = f"{_trans_agent_dir}/{_trans_agent_id}.manifest.json"

            # Transcription-mistake decision: deterministic per sample_id hash
            # so the same sample always gets the same mistake behavior across
            # re-runs. When injected, the first Agent dispatch returns a
            # corrupted note list; the main agent's verify-listen detects the
            # mismatch and re-dispatches; the second attempt is correct.
            import random as _trans_random
            _trans_mistake_rate = float(getattr(args, "transcription_mistake_rate", 0.0) or 0.0)
            _sid_int = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
            _trans_mistake_rng = _trans_random.Random(int(args.seed) + _sid_int + 7919)
            _trans_inject_mistake = (
                _trans_mistake_rate > 0.0
                and _trans_mistake_rng.random() < _trans_mistake_rate
                and _trans_n_notes >= 3
            )
            _trans_wrong_notes: list[dict] | None = None
            if _trans_inject_mistake:
                _trans_wrong_notes, _trans_mistake_info = corrupt_transcription_notes(
                    _trans_notes, _trans_mistake_rng,
                )

            # Persist the transcription output file on disk so the path is real
            # at build time (live-exec grading and downstream tools can read it).
            # In mistake samples we also persist the wrong-first-attempt file
            # at a sibling path so its contents are inspectable.
            _trans_payload = {
                "status": "completed",
                "notes": _trans_notes,
                "n_notes": _trans_n_notes,
                "duration_s": _trans_duration_s,
            }
            Path(_trans_output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(_trans_output_file, "w") as _trf:
                json.dump(_trans_payload, _trf)
                _trf.write("\n")
            # Wrong-attempt manifest paths (only used when injecting a mistake)
            _wrong_agent_id = make_agent_id(sample_id, "melody_transcription", "attempt1")
            _wrong_file = f"{_trans_agent_dir}/{_wrong_agent_id}.json"
            _wrong_manifest_file = f"{_trans_agent_dir}/{_wrong_agent_id}.manifest.json"
            _wrong_verify_wav = f"{_trans_agent_dir}/{_wrong_agent_id}.verify.wav"
            if _trans_wrong_notes is not None:
                with open(_wrong_file, "w") as _trf:
                    json.dump({
                        "status": "completed",
                        "notes": _trans_wrong_notes,
                        "n_notes": len(_trans_wrong_notes),
                        "duration_s": _trans_duration_s,
                    }, _trf)
                    _trf.write("\n")

            # Step 1: create a REAPER track + load Vital on it
            _trans_create_snippet = (
                _REAPY_HELPER
                + f"track_name = {json.dumps(_trans_track_name)}\n"
                  f"with reapy.inside_reaper():\n"
                  f"    RPR.InsertTrackAtIndex(0, True)\n"
                  f"    track = RPR.GetTrack(0, 0)\n"
                  f"    RPR.GetSetMediaTrackInfo_String(track, 'P_NAME', track_name, True)\n"
                  f"    RPR.TrackFX_AddByName(track, 'Vital', False, 1)\n"
                  f"print(json.dumps({{'status': 'ok', 'track_idx': 0, 'track_name': track_name}}))\n"
            )
            _trans_create_cmd = _wrap_as_bash(_trans_create_snippet)
            messages.append({
                "role": "assistant",
                "content": "Creating a REAPER track to hold the transcribed MIDI before I search the wavetable library.",
            })
            messages.append(_tool_call("Bash", {"command": _trans_create_cmd}))
            _track_stdout = json.dumps({
                "status": "ok",
                "track_idx": _trans_track_idx,
                "track_name": _trans_track_name,
            }) + "\n"
            messages.append(_bash_tool_response(_track_stdout))

            # Step 2a (mistake samples only): emit the WRONG first attempt —
            # dispatch, verify-listen of the corrupted notes, detection, and
            # the re-dispatch intro. This teaches the model to catch+recover
            # from transcription errors instead of proceeding with bad notes.
            if _trans_wrong_notes is not None and _trans_mistake_info is not None:
                _n_wrong = len(_trans_wrong_notes)

                # Write the manifest matching claw-code's runtime convention.
                _wrong_dispatch_prompt = (
                    f"Target: {target_audio_path}. Track: {_trans_track_idx}. Write Python "
                    f"(reapy → MIDI_InsertNote) that inserts the MIDI "
                    f"notes on that track, and save the final note list as JSON to "
                    f"{_wrong_file} with shape "
                    f'{{"notes": [...], "n_notes": N, "duration_s": X}}.'
                )
                write_agent_manifest(
                    agent_id=_wrong_agent_id,
                    subagent_type="melody_transcription",
                    output_file=_wrong_file,
                    manifest_file=_wrong_manifest_file,
                    prompt=_wrong_dispatch_prompt,
                )

                messages.append({
                    "role": "assistant",
                    "content": "Dispatching the transcription subagent to listen to the target and populate the track with MIDI notes.",
                })
                messages.append(_tool_call("Agent", {
                    "subagent_type": "melody_transcription",
                    "description": f"Transcribe target melody to MIDI on track {_trans_track_idx}",
                    "prompt": _wrong_dispatch_prompt,
                    "name": f"transcribe-{sample_id}-1",
                }))
                messages.append({
                    "role": "tool_response",
                    "content": json.dumps({
                        "agentId": _wrong_agent_id,
                        "subagentType": "melody_transcription",
                        "status": "completed",
                        "outputFile": _wrong_file,
                        "manifestFile": _wrong_manifest_file,
                        "createdAt": f"build-time:{_wrong_agent_id}",
                        "startedAt": f"build-time:{_wrong_agent_id}",
                        "n_notes": _n_wrong,
                        "duration_s": _trans_duration_s,
                    }, ensure_ascii=False),
                })

                # Render the WRONG verify probe and expose it so the agent can
                # "hear" the mismatch. At build time we know which note is off
                # and embed that in the detection narration for grounded text.
                try:
                    _wrong_tuples = notes_from_dicts(_trans_wrong_notes)
                    render_preset_audio(init_preset, _wrong_tuples, out_path=_wrong_verify_wav, tail_s=1.0)
                    audio_assets.append(_wrong_verify_wav)
                    wrong_verify_ok = True
                except Exception:
                    wrong_verify_ok = False

                messages.append({
                    "role": "assistant",
                    "content": (
                        "Verifying the first transcription attempt — rendering the returned "
                        "notes through the default Vital preset so I can compare the melody "
                        "to the target."
                    ),
                })
                # Same pattern as _build_listen_probe_command: read the wav
                # we already rendered at build time and emit listen_probe shape.
                # Categorises as listen_probe in the grader, so path matching
                # uses the listen_probe.path fallback.
                _wrong_verify_cmd = (
                    "python - <<'PY'\n"
                    "import json\n"
                    "from pathlib import Path\n"
                    "import soundfile as sf\n"
                    f"payload = json.loads('''{json.dumps({'path': _wrong_verify_wav, 'notes': _n_wrong}, ensure_ascii=False)}''')\n"
                    "p = Path(payload['path'])\n"
                    "out = {'path': str(p), 'exists': p.exists(), 'notes': payload['notes']}\n"
                    "if out['exists']:\n"
                    "    try:\n"
                    "        x, sr = sf.read(p, always_2d=True)\n"
                    "        out['duration_s'] = round(float(len(x) / max(1, sr)), 4)\n"
                    "    except Exception:\n"
                    "        out['duration_s'] = None\n"
                    "print(json.dumps({'listen_probe': out}, ensure_ascii=False))\n"
                    "PY"
                )
                messages.append(_tool_call("Bash", {"command": _wrong_verify_cmd}))
                _wrong_probe_stdout = json.dumps({
                    "listen_probe": {
                        "path": _wrong_verify_wav,
                        "exists": True,
                        "notes": _n_wrong,
                    },
                }) + "\n"
                if wrong_verify_ok:
                    _emit_listen_sequence(
                        messages, audio_assets, _wrong_verify_wav,
                        probe_stdout=_wrong_probe_stdout,
                        listen_text="Listening to the first transcription attempt.",
                    )
                else:
                    messages.append(_bash_tool_response(_wrong_probe_stdout))

            # Step 2 (primary dispatch): when no mistake was injected this is
            # the one-and-only transcription call. When a mistake was injected
            # this becomes the RETRY after detection; we merge the detection
            # narration + re-dispatch intro into a single assistant turn to
            # satisfy the no-adjacent-assistants validator.
            _retry_note = ""
            if _trans_wrong_notes is not None and _trans_mistake_info is not None:
                _m = _trans_mistake_info
                _direction = "high" if _m["delta_semitones"] > 0 else "low"
                _abs_delta = abs(_m["delta_semitones"])
                _retry_note = (
                    f" Previous attempt had a pitch error on note "
                    f"{_m['note_idx'] + 1} at ~{_m['start_s']:.2f}s (off by "
                    f"{_m['delta_semitones']:+d} semitones). Please re-listen "
                    f"and correct that note."
                )
                _dispatch_prose = (
                    f"The verify render doesn't match — note {_m['note_idx'] + 1} at "
                    f"~{_m['start_s']:.2f}s sounds {_abs_delta} semitone"
                    f"{'s' if _abs_delta != 1 else ''} too {_direction}. "
                    f"Re-dispatching the transcription subagent with a correction hint "
                    f"so it can fix the off-pitch note."
                )
            else:
                _dispatch_prose = (
                    "Dispatching the transcription subagent to listen to the target "
                    "and populate the track with MIDI notes."
                )
            messages.append({
                "role": "assistant",
                "content": _dispatch_prose,
            })
            _primary_dispatch_prompt = (
                f"Target: {target_audio_path}. Track: {_trans_track_idx}. Write Python "
                f"(reapy → MIDI_InsertNote) that inserts the MIDI "
                f"notes on that track, and save the final note list as JSON to "
                f"{_trans_output_file} with shape "
                f'{{"notes": [...], "n_notes": N, "duration_s": X}}.' + _retry_note
            )
            write_agent_manifest(
                agent_id=_trans_agent_id,
                subagent_type="melody_transcription",
                output_file=_trans_output_file,
                manifest_file=_trans_manifest_file,
                prompt=_primary_dispatch_prompt,
            )
            messages.append(_tool_call("Agent", {
                "subagent_type": "melody_transcription",
                "description": f"Transcribe target melody to MIDI on track {_trans_track_idx}",
                "prompt": _primary_dispatch_prompt,
                "name": f"transcribe-{sample_id}-2" if _retry_note else f"transcribe-{sample_id}",
            }))
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "agentId": _trans_agent_id,
                    "subagentType": "melody_transcription",
                    "status": "completed",
                    "outputFile": _trans_output_file,
                    "manifestFile": _trans_manifest_file,
                    "createdAt": f"build-time:{_trans_agent_id}",
                    "startedAt": f"build-time:{_trans_agent_id}",
                    "n_notes": _trans_n_notes,
                    "duration_s": _trans_duration_s,
                }, ensure_ascii=False),
            })

            # Step 3: verify the transcription by rendering the transcribed notes
            # through the default Vital preset and listening to them. If the note
            # content matches the target melody (ignoring timbre), the transcription
            # is correct and we proceed. At build time we know it's correct (oracle
            # notes); the verdict prose is always "matches", teaching the model the
            # verify-and-proceed protocol.
            _verify_wav = f"/tmp/agents/{sample_id}/transcription_verify.wav"
            Path(_verify_wav).parent.mkdir(parents=True, exist_ok=True)
            # Render the verify probe live: transcribed notes + default init preset.
            try:
                render_preset_audio(init_preset, notes, out_path=_verify_wav, tail_s=1.0)
                audio_assets.append(_verify_wav)
                verify_wav_ok = True
            except Exception:
                verify_wav_ok = False

            messages.append({
                "role": "assistant",
                "content": (
                    "Verifying the transcription: rendering the transcribed notes "
                    "through the default Vital preset so I can compare the melody "
                    "(note content only) to the target."
                ),
            })
            # Same pattern as _build_listen_probe_command: read the verify wav
            # (already rendered at build time) so the grader categorises this
            # as listen_probe and uses the listen_probe.path fallback for
            # path matching.
            _verify_cmd = (
                "python - <<'PY'\n"
                "import json\n"
                "from pathlib import Path\n"
                "import soundfile as sf\n"
                f"payload = json.loads('''{json.dumps({'path': _verify_wav, 'notes': _trans_n_notes}, ensure_ascii=False)}''')\n"
                "p = Path(payload['path'])\n"
                "out = {'path': str(p), 'exists': p.exists(), 'notes': payload['notes']}\n"
                "if out['exists']:\n"
                "    try:\n"
                "        x, sr = sf.read(p, always_2d=True)\n"
                "        out['duration_s'] = round(float(len(x) / max(1, sr)), 4)\n"
                "    except Exception:\n"
                "        out['duration_s'] = None\n"
                "print(json.dumps({'listen_probe': out}, ensure_ascii=False))\n"
                "PY"
            )
            messages.append(_tool_call("Bash", {"command": _verify_cmd}))
            _verify_probe_stdout = json.dumps({
                "listen_probe": {
                    "path": _verify_wav,
                    "exists": True,
                    "notes": _trans_n_notes,
                },
            }) + "\n"
            if verify_wav_ok:
                _emit_listen_sequence(
                    messages, audio_assets, _verify_wav,
                    probe_stdout=_verify_probe_stdout,
                    listen_text="Listening to the transcription verify render.",
                )
            else:
                messages.append(_bash_tool_response(_verify_probe_stdout))

            _verify_prefix = (
                "Transcription verified on retry"
                if _trans_wrong_notes is not None else "Transcription verified"
            )
            transcription_summary_text = (
                f"{_verify_prefix} — {_trans_n_notes} notes match the target "
                f"melody. MIDI ready on track {_trans_track_idx}. "
            )

    # Check library size — the agent needs to discover this at inference time
    # (it's what drives how many search agents to dispatch and how to slice).
    # Merge the post-transcription "MIDI ready" summary into this turn's opener
    # if we emitted a transcription block (keeps validator happy — no adjacent
    # assistants).
    messages.append({
        "role": "assistant",
        "content": f"{transcription_summary_text}Checking wavetable library size.",
    })
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(build_list_wavetables_total_snippet())}))
    messages.append(_bash_tool_response(json.dumps({"total": total_named}) + "\n"))

    # Agent output directory (real path; runtime executor writes files here at inference)
    agent_out_dir = f"/tmp/agents/{sample_id}"

    # CLAP name→embedding for fallback tuple selection (pick best proxy per osc)
    _wt_name_to_emb = build_name_embedding_map(shortlist_data["embeddings"], index_rows)

    # Setup for tuple rendering
    tuple_audio_dir = Path(args.out_jsonl).parent / "tuple_audio" / sample_id
    tuple_audio_dir.mkdir(parents=True, exist_ok=True)
    wt_lib_by_name = {wt["name"]: wt for wt in wavetable_lib if "name" in wt}

    def _build_tuple_preset(tup: list, oscs: list[int], lib: dict, base: dict) -> dict:
        preset = copy.deepcopy(base)
        for oi in oscs:
            name = tup[oi]
            if name and name in lib:
                if oi < len(preset["settings"].get("wavetables", [])):
                    preset["settings"]["wavetables"][oi] = copy.deepcopy(lib[name])
                preset["settings"][f"osc_{oi + 1}_on"] = 1.0
                preset["settings"][f"osc_{oi + 1}_level"] = 0.7
            else:
                preset["settings"][f"osc_{oi + 1}_on"] = 0.0
                preset["settings"][f"osc_{oi + 1}_level"] = 0.0
        return preset

    _log("search loop start")
    # ---- Multi-round search loop ----
    pool: list[str] = []
    rounds_used = 0
    round_offsets_used: list[list[int]] = []
    verdicts_by_round: list[str] = []
    judge_exhausted_fallback = False

    def _simulate_shortlist(start: int, end: int) -> list[str]:
        """Build-time simulation of search agent shortlist for a given shard.

        Runtime executor would produce this by running the search agent model.
        At build time we use oracle knowledge: GTs in the shard + 2 non-GT picks.
        """
        names_in = [idx_to_name_full[i] for i in range(start, end) if i in idx_to_name_full]
        gts_in = [n for n in names_in if n in gt_names_list]
        non_gt = [n for n in names_in if n not in gt_names_list]
        # Pick 2 non-GT by deterministic sampling
        sample_rng.shuffle(non_gt)
        picks = list(gts_in) + non_gt[: max(1, 3 - len(gts_in))]
        return picks[:4]

    def _write_shortlist_file(agent_id: str, start: int, end: int, shortlist: list[str]) -> tuple[str, str]:
        """Write single-line JSON shortlist file + manifest. Returns (output_file, manifest_file)."""
        out_dir = Path(agent_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{agent_id}.json"
        manifest_path = out_dir / f"{agent_id}.manifest.json"
        with open(out_path, "w") as f:
            json.dump({
                "status": "completed",
                "agentId": agent_id,
                "shardStart": start,
                "shardEnd": end,
                "shortlist": shortlist,
            }, f)
            f.write("\n")
        write_agent_manifest(
            agent_id=agent_id,
            subagent_type="wavetable_search",
            output_file=str(out_path),
            manifest_file=str(manifest_path),
            extra={"shardStart": start, "shardEnd": end},
        )
        return str(out_path), str(manifest_path)

    def _pool_covers_gt(pool_list: list[str], threshold: float = 0.92) -> float:
        """Fraction of GT wavetables covered by pool (exact or CLAP-similar proxy)."""
        if not gt_names_list:
            return 1.0
        # Pre-compute GT embeddings
        index_embs = shortlist_data["embeddings"]
        # Build name → mean-pooled embedding from the index
        name_to_rows: dict[str, list[int]] = {}
        for i, row in enumerate(index_rows):
            name_to_rows.setdefault(row["wavetable_name"], []).append(i)
        name_to_emb: dict[str, np.ndarray] = {}
        for nm in set(gt_names_list) | set(pool_list):
            if nm in name_to_rows:
                embs = index_embs[name_to_rows[nm]]
                emb = embs.mean(axis=0)
                name_to_emb[nm] = emb / (np.linalg.norm(emb) + 1e-12)

        covered = 0
        for gt in gt_names_list:
            if gt in pool_list:
                covered += 1
                continue
            if gt not in name_to_emb:
                continue
            gt_emb = name_to_emb[gt]
            best = 0.0
            for p in pool_list:
                if p in name_to_emb:
                    sim = float(gt_emb @ name_to_emb[p])
                    if sim > best:
                        best = sim
            if best >= threshold:
                covered += 1
        return covered / len(gt_names_list)

    _research_prefix = ""  # merged into round 2+ intro to avoid adjacent assistant

    while rounds_used < max_rounds:
        rounds_used += 1
        round_offsets_used.append(list(slice_starts))

        # Announce round (merge re-search prefix from previous round if any)
        if rounds_used == 1:
            intro = (
                f"Library has {total_named} wavetables. Dispatching {n_agents} search "
                f"agents in parallel across slices "
                f"[{', '.join(f'{s}-{s + slice_size - 1}' for s in slice_starts)}]."
            )
        else:
            intro = (
                f"{_research_prefix}"
                f"Expanding to different library regions with {n_agents} more search agents "
                f"in parallel: "
                f"[{', '.join(f'{s}-{s + slice_size - 1}' for s in slice_starts)}]."
            )
            _research_prefix = ""
        messages.append({"role": "assistant", "content": intro})

        # Step 2: Agent tool calls — emit ALL tool_calls first, then ALL tool_responses.
        # This represents parallel dispatch (multiple tool_use blocks from a single assistant
        # turn in Anthropic / claw-code protocol). Serial emit would read as sequential.
        round_output_files: list[str] = []
        round_agent_ids: list[str] = []
        # (start, end, agent_id, output_file, manifest_file, shortlist)
        round_agent_meta: list[tuple[int, int, str, str, str, list[str]]] = []
        for ai, start in enumerate(slice_starts):
            end = min(start + slice_size, total_named)
            if end <= start:
                continue
            agent_id = make_agent_id(sample_id, "wavetable_search", rounds_used, ai)
            round_agent_ids.append(agent_id)
            sl = _simulate_shortlist(start, end)
            output_file, manifest_file = _write_shortlist_file(agent_id, start, end, sl)
            round_output_files.append(output_file)
            round_agent_meta.append((start, end, agent_id, output_file, manifest_file, sl))

        # Emit all Agent tool_calls back-to-back (parallel dispatch)
        for ai_idx, (start, end, agent_id, _out, _manifest, _sl) in enumerate(round_agent_meta):
            _search_prompt_parts = [f"Target: {target_audio_path}."]
            if _trans_output_file:
                _search_prompt_parts.append(f"Transcription MIDI: {_trans_output_file}.")
            _search_prompt_parts.append(
                f"Evaluate wavetables at indices {start}-{end - 1}. "
                f"Load data/wavetable_lib.json, list names in your range, "
                f"swap each into the synth, render with DawDreamer using the "
                f"transcription MIDI, and listen. Return a JSON shortlist of "
                f"2-4 wavetable names."
            )
            messages.append(_tool_call("Agent", {
                "subagent_type": "wavetable_search",
                "description": f"Evaluate wavetables {start}-{end - 1} for target sound",
                "prompt": "\n".join(_search_prompt_parts),
                "name": f"search-r{rounds_used}-a{ai_idx + 1}",
            }))

        # Then all tool_responses (one per parallel call)
        for _start, _end, agent_id, output_file, manifest_file, _sl in round_agent_meta:
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "agentId": agent_id,
                    "subagentType": "wavetable_search",
                    "status": "completed",
                    "outputFile": output_file,
                    "manifestFile": manifest_file,
                    "createdAt": f"build-time:{agent_id}",
                    "startedAt": f"build-time:{agent_id}",
                }, ensure_ascii=False),
            })

        # Step 3: read all output files in one cat call
        cat_cmd = "cat " + " ".join(round_output_files)
        messages.append({
            "role": "assistant",
            "content": f"Reading shortlists from {len(round_output_files)} search agents.",
        })
        messages.append(_tool_call("Bash", {"command": cat_cmd}))
        # Concatenate the file contents (single-line JSON per file)
        cat_output_lines = []
        round_shortlists: list[list[str]] = []
        for out_file in round_output_files:
            try:
                with open(out_file) as f:
                    content = f.read().strip()
                cat_output_lines.append(content)
                parsed = json.loads(content)
                round_shortlists.append(parsed.get("shortlist", []))
            except Exception:
                cat_output_lines.append("")
                round_shortlists.append([])
        messages.append(_bash_tool_response("\n".join(cat_output_lines) + "\n"))

        # Pool in new shortlists
        for sl in round_shortlists:
            for name in sl:
                if name not in pool:
                    pool.append(name)

        # Judge agent: ALWAYS runs after search agents to filter the pool down
        # to a final tuple. Search agents do local filtering (per-slice); the
        # judge does global ranking — it's the only step with access to the
        # combined pool in one auditory view, which matters when GTs are
        # scattered across slices (each search agent sees one GT + some
        # false positives; only the judge can pick the correct combination).
        #
        # Build-time simulation: judge's "selected tuple" is computed via
        # GT-if-in-pool + CLAP-best-proxy per osc slot. This is the oracle
        # answer — at inference the judge model does this perceptually by
        # listening to the combined pool.

        cur_tuple: list[str | None] = [None, None, None]
        used_in_tuple: set[str] = set()
        for osc_idx in active_oscs:
            wts = target_preset.get("settings", {}).get("wavetables", [])
            if osc_idx < len(wts):
                wt_name = wts[osc_idx].get("name", "")
                if wt_name and wt_name in pool:
                    cur_tuple[osc_idx] = wt_name
                    used_in_tuple.add(wt_name)
                    continue
                # Fallback: best CLAP proxy from pool for this osc's GT
                gt_emb = _wt_name_to_emb.get(wt_name)
                if gt_emb is not None:
                    candidates = [n for n in pool if n not in used_in_tuple and n in _wt_name_to_emb]
                    if candidates:
                        best = max(candidates, key=lambda n: float(_wt_name_to_emb[n] @ gt_emb))
                        cur_tuple[osc_idx] = best
                        used_in_tuple.add(best)
                        continue
            # Last resort: first unused pool member
            for n in pool:
                if n not in used_in_tuple:
                    cur_tuple[osc_idx] = n
                    used_in_tuple.add(n)
                    break

        cur_active_names = [cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]]
        n_osc_slots = len(cur_active_names)

        # VERDICT: judge says "good" iff every active-osc-slot's GT name is in
        # the pool. Otherwise "no_match" — the main agent will branch into a
        # re-search round (or exit with best-available if budget exhausted).
        needed_gt_names = [
            target_preset.get("settings", {}).get("wavetables", [None, None, None])[oi].get("name", "")
            if oi < len(target_preset.get("settings", {}).get("wavetables", []))
            else ""
            for oi in active_oscs
        ]
        needed_gt_names = [n for n in needed_gt_names if n]
        all_gt_in_pool = bool(needed_gt_names) and all(n in pool for n in needed_gt_names)
        judge_verdict = "good" if all_gt_in_pool else "no_match"
        # Pre-compute the missing-character hint for the no_match case via the
        # judge builder's Omni-derived helper. CRITICAL: must NOT leak the GT
        # wavetable name into the main agent's view — the main agent at
        # inference will only see what the judge tells it, and the judge
        # speaks in perceptual terms ("metallic FM buzz"), never names of
        # candidate wavetables. Earlier draft used the GT name directly,
        # which leaked the answer into the main-agent SFT — regression that
        # the model would memorise as a free shortcut.
        if judge_verdict == "no_match":
            try:
                from scripts.build_judge_agent_sft_v3 import (  # type: ignore
                    _derive_missing_character,
                )
                missing_character = _derive_missing_character(
                    target_wav=target_audio_path,
                    omni_server=args.omni_server,
                    omni_model=args.omni_model,
                )
            except Exception:
                missing_character = "the target's distinctive timbral character"
        else:
            missing_character = ""

        verdicts_by_round.append(judge_verdict)

        # Write judge output file (simulates runtime judge agent's output) —
        # new schema with verdict + missing_character + nullable tuple.
        judge_agent_id = make_agent_id(sample_id, "wavetable_judge", rounds_used)
        judge_output_file = Path(agent_out_dir) / f"{judge_agent_id}.json"
        judge_manifest_file = Path(agent_out_dir) / f"{judge_agent_id}.manifest.json"
        judge_output_file.parent.mkdir(parents=True, exist_ok=True)
        if judge_verdict == "good":
            judge_reasoning = (
                f"After listening to the {len(pool)} pool candidates against the target, "
                f"{', '.join(repr(n) for n in cur_active_names)} together captures the "
                f"target's character most closely."
            )
            judge_output_payload = {
                "status": "completed",
                "agentId": judge_agent_id,
                "verdict": "good",
                "missing_character": "",
                "tuple": cur_active_names,
                "n_osc_slots": n_osc_slots,
                "reasoning": judge_reasoning,
            }
        else:
            judge_reasoning = (
                f"Pool of {len(pool)} candidates lacks any wavetable with the "
                f"{missing_character} the target relies on. Recommending re-search "
                f"across unexplored library regions."
            )
            judge_output_payload = {
                "status": "completed",
                "agentId": judge_agent_id,
                "verdict": "no_match",
                "missing_character": missing_character,
                "tuple": None,
                "n_osc_slots": n_osc_slots,
                "reasoning": judge_reasoning,
            }
        with open(judge_output_file, "w") as jf:
            json.dump(judge_output_payload, jf)
            jf.write("\n")

        # Dispatch judge agent
        pool_str = ", ".join(f"'{n}'" for n in pool[:6])
        if len(pool) > 6:
            pool_str += f", ...+{len(pool) - 6} more"
        messages.append({
            "role": "assistant",
            "content": (
                f"Pool has {len(pool)} candidates across {len(round_agent_meta)} slices: "
                f"[{pool_str}]. Dispatching judge agent to audition the combined pool and "
                f"select the best {n_osc_slots}-oscillator combination."
            ),
        })
        _judge_dispatch_prompt = (
            f"Target: {target_audio_path}.\n"
            f"Pool candidates from search agents: {json.dumps(pool)}.\n"
            f"Target uses {n_osc_slots} active oscillator(s). Swap each candidate "
            f"wavetable into the synth via chunk manipulation, render, and listen "
            f"alongside the target, then select the {n_osc_slots} candidates that "
            f"together best capture the target. Write a JSON file with `tuple` (list "
            f"of names), `n_osc_slots`, and `reasoning`."
        )
        write_agent_manifest(
            agent_id=judge_agent_id,
            subagent_type="wavetable_judge",
            output_file=str(judge_output_file),
            manifest_file=str(judge_manifest_file),
            prompt=_judge_dispatch_prompt,
        )
        messages.append(_tool_call("Agent", {
            "subagent_type": "wavetable_judge",
            "description": f"Select best {n_osc_slots}-osc tuple from pool of {len(pool)} candidates",
            "prompt": _judge_dispatch_prompt,
            "name": f"judge-{rounds_used}",
        }))
        messages.append({
            "role": "tool_response",
            "content": json.dumps({
                "agentId": judge_agent_id,
                "subagentType": "wavetable_judge",
                "status": "completed",
                "outputFile": str(judge_output_file),
                "manifestFile": str(judge_manifest_file),
                "createdAt": f"build-time:{judge_agent_id}",
                "startedAt": f"build-time:{judge_agent_id}",
            }, ensure_ascii=False),
        })

        # Read judge's verdict + tuple via cat. The main agent's branch
        # decision is now driven by `verdict`, not by an oracle peek.
        messages.append({
            "role": "assistant",
            "content": "Reading judge's verdict and selection.",
        })
        messages.append(_tool_call("Bash", {"command": f"cat {judge_output_file}"}))
        with open(judge_output_file) as jf:
            judge_content = jf.read().strip()
        messages.append(_bash_tool_response(judge_content + "\n"))

        # Branch on judge verdict.
        if judge_verdict == "no_match":
            # Judge said the pool is insufficient. Skip tuple render entirely
            # and go to a re-search round (or exit with best-available if budget
            # exhausted).
            if rounds_used >= max_rounds:
                judge_exhausted_fallback = True
                break
            _research_prefix = (
                f"The judge reports the pool of {len(pool)} candidates doesn't contain "
                f"any wavetable with the {missing_character} of the target. "
                f"Expanding search to unexplored library regions. "
            )
            base_offset = (base_offset + stride // 2) % stride
            slice_starts = _compute_slices(base_offset)
            continue

        # verdict == "good" → render tuple, listen, break out of search loop.
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}.wav"
        osc_names = {oi: cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]}
        render_cmd = _wrap_as_bash(build_render_tuple_snippet(
            osc_names=osc_names, out_path=str(tuple_wav),
            midi_path=_trans_output_file,
        ))
        render_cumulative_audio(_build_tuple_preset(cur_tuple, active_oscs, wt_lib_by_name, init_preset), notes, tuple_wav)

        tuple_names_str = ", ".join(f"'{n}'" for n in cur_active_names)
        messages.append({
            "role": "assistant",
            "content": (
                f"Judge confirmed pool is sufficient and selected [{tuple_names_str}]. "
                f"Rendering the tuple as a final pre-apply sanity check."
            ),
        })
        messages.append(_tool_call("Bash", {"command": render_cmd}))
        audio_assets.append(str(tuple_wav))
        _tuple_stdout = json.dumps({
            "status": "ok", "out": str(tuple_wav), "wavetables": cur_active_names,
        }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, tuple_wav,
            probe_stdout=_tuple_stdout,
            listen_text="Tuple rendered. Listening.",
        )
        # Verdict was good → done with search, exit loop.
        break

    # The best tuple from the last search round is cur_tuple. For verdict="good"
    # rounds it was already rendered + heard inside the loop. For the
    # judge_exhausted_fallback case (max_rounds reached on no_match) we have
    # not rendered the tuple yet — render it here so the apply step has a
    # tuple_wav to reference and the model gets a final listen.
    if judge_exhausted_fallback:
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}_fallback.wav"
        osc_names = {oi: cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]}
        render_cmd = _wrap_as_bash(build_render_tuple_snippet(
            osc_names=osc_names, out_path=str(tuple_wav),
            midi_path=_trans_output_file,
        ))
        render_cumulative_audio(
            _build_tuple_preset(cur_tuple, active_oscs, wt_lib_by_name, init_preset),
            notes, tuple_wav,
        )
        cur_active_names = [cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]]
        tuple_names_str = ", ".join(f"'{n}'" for n in cur_active_names)
        messages.append({
            "role": "assistant",
            "content": (
                f"Search budget exhausted after {max_rounds} rounds; the judge still "
                f"reports no_match on the final pool. Rendering the best-available "
                f"combination [{tuple_names_str}] to make progress before parameter tuning."
            ),
        })
        messages.append(_tool_call("Bash", {"command": render_cmd}))
        audio_assets.append(str(tuple_wav))
        _fb_tuple_stdout = json.dumps({
            "status": "ok", "out": str(tuple_wav), "wavetables": cur_active_names,
        }) + "\n"
        _emit_listen_sequence(
            messages, audio_assets, tuple_wav,
            probe_stdout=_fb_tuple_stdout,
            listen_text="Fallback tuple rendered. Listening.",
        )

    gt_tuple = cur_tuple
    apply_names = [gt_tuple[oi] for oi in active_oscs if gt_tuple[oi]]
    osc_assignments = ", ".join(
        f"oscillator {oi + 1} = '{gt_tuple[oi]}'" for oi in active_oscs if gt_tuple[oi]
    )
    final_verdict = verdicts_by_round[-1] if verdicts_by_round else "good"
    if final_verdict == "good" and not judge_exhausted_fallback:
        selection_text = f"This tuple matches the target well. Applying: {osc_assignments}."
    else:
        selection_text = (
            f"Search budget exhausted; applying best-available combination: "
            f"{osc_assignments}."
        )

    apply_assignments = repr(
        [(oi, gt_tuple[oi]) for oi in active_oscs if gt_tuple[oi]]
    )
    apply_snippet = (
        _REAPY_HELPER
        + _BUILD_CHUNK_HELPER
        + "import base64\n"
        'wt_lib = json.load(open("data/wavetable_lib.json"))\n'
        "name_to_wt = {wt['name']: wt for wt in wt_lib if 'name' in wt}\n"
        'preset = json.load(open("maestro/synth/init_preset.json"))\n'
        f"for osc_idx, wt_name in {apply_assignments}:\n"
        "    if wt_name in name_to_wt:\n"
        "        preset['settings']['wavetables'][osc_idx] = name_to_wt[wt_name]\n"
        "chunk = build_vital_chunk(preset)\n"
        "encoded = base64.b64encode(chunk).decode('ascii')\n"
        "with reapy.inside_reaper():\n"
        "    track = RPR.GetTrack(0, 0)\n"
        "    if not RPR.TrackFX_SetNamedConfigParm(track, 0, 'vst3_chunk', encoded):\n"
        "        RPR.TrackFX_SetNamedConfigParm(track, 0, 'vst_chunk', encoded)\n"
        f"print(json.dumps({{'status': 'ok', 'applied': {json.dumps(apply_names)}}}))"
    )
    messages.append({"role": "assistant", "content": selection_text})
    messages.append(_tool_call("Bash", {"command": _wrap_as_bash(apply_snippet)}))
    _apply_stdout = json.dumps({"status": "ok", "applied": apply_names}) + "\n"
    messages.append(_bash_tool_response(_apply_stdout))

    # ---- DIAGNOSIS BLOCK (block 1) ----
    subsystem_truth_map = build_diagnosis_subsystem_truth(target_preset, init_preset)
    subsystems_truth = [lbl for lbl, _ in SUBSYSTEM_ORDER if lbl in subsystem_truth_map]
    diff_summary = format_subsystem_diff_summary(subsystem_truth_map)

    if args.omni_server:
        stage1_obs = omni_stage1_diagnose(
            str(target_audio_path), target_preset,
            archetype, args.omni_server, args.omni_model,
        )
    else:
        stage1_obs = "Target differs from default in several subsystems."
    _dt = _time.monotonic()
    diagnosis_text = stage2_diagnosis(
        stage1_obs, diff_summary, subsystems_truth,
        archetype, stage2_server, stage2_model,
    )
    _log(f"diagnosis omni {_time.monotonic()-_dt:.1f}s")

    _diagnosis_text = diagnosis_text

    # ---- SUBSYSTEM BATCHES (diff-based) ----
    batches = build_batches_from_diff(target_preset, init_preset)
    batches = batches[:int(args.max_batches)]
    path_complete = True  # diff-based always applies everything within max_batches

    # Wavetables + sample + lfos come from target (same as path_gen's approach).
    cumulative = copy.deepcopy(init_preset)
    for key in ("wavetables", "sample", "lfos"):
        if key in target_preset.get("settings", {}):
            cumulative["settings"][key] = copy.deepcopy(target_preset["settings"][key])

    # Inject a mistake (~20% of samples, seeded by sample_id).
    import random as _random
    sid_seed = int(hashlib.sha1(sample_id.encode()).hexdigest()[:8], 16)
    mistake_rng = _random.Random(int(args.seed) + sid_seed)
    injected_mistake = inject_mistake(batches, mistake_rng, args.mistake_rate)

    batch_labels: list[dict] = []
    prior_checks: list[str] = []
    pending_check: str | None = None
    last_batch_audio: Path | None = None

    batch_audio_dir = Path(args.out_jsonl).parent / "batch_audio" / sample_id
    batch_audio_dir.mkdir(parents=True, exist_ok=True)
    _log(f"batch loop start — {len(batches)} batches")

    for bi, b in enumerate(batches):
        # Snapshot BEFORE values for the params in this batch so we can describe
        # the perceptual change in Stage 2 narration.
        batch_before_values: dict[str, float] = {
            name: float(cumulative["settings"].get(name, 0.0) or 0.0)
            for name in b.params_applied
        }
        # Apply batch params to cumulative (uses params_applied which may contain mistake)
        for name, norm in b.params_applied.items():
            cumulative["settings"][name] = _denormalize(name, norm)

        batch_wav = batch_audio_dir / f"batch_{bi}_{b.subsystem}.wav"
        _bt = _time.monotonic()
        render_cumulative_audio(cumulative, notes, batch_wav)
        _log(f"  batch {bi}/{len(batches)-1} ({b.subsystem}) render {_time.monotonic()-_bt:.1f}s")

        # CLAP score vs GT
        with serial_lock:
            try:
                clap_after = float(embedder.cosine_paths(batch_wav, target_audio_path))
            except Exception:
                clap_after = None

        b.audio_wav = batch_wav
        display_norm_params = {
            _json_key_to_reaper_display(n): float(v)
            for n, v in b.params_applied.items()
        }
        action_snippet = build_batch_action_snippet(display_norm_params)

        # Intro message — merge diagnosis (for first batch) or prior check
        intro = f"Applying {b.subsystem} changes."
        if bi == 0 and _diagnosis_text:
            intro = f"{_diagnosis_text}\n\n{intro}"
            _diagnosis_text = None
        elif pending_check:
            intro = f"{pending_check}\n\n{intro}"
            pending_check = None
        messages.append({"role": "assistant", "content": intro})
        messages.append(_tool_call("Bash", {"command": action_snippet}))
        _action_stdout = json.dumps({"status": "ok", "applied": len(display_norm_params), "not_found": []}) + "\n"
        messages.append(_bash_tool_response(_action_stdout))

        # Listen
        audio_assets.append(str(batch_wav))
        messages.append({"role": "assistant", "content": f"Listening after {b.subsystem} batch."})
        messages.append(_tool_call("Bash", {"command": _build_listen_probe_command(batch_wav)}))
        _emit_listen_sequence(
            messages, audio_assets, batch_wav,
            listen_text=f"Reading {b.subsystem} batch audio.",
        )
        last_batch_audio = batch_wav

        # Remaining-gap check
        gap = _step_remaining_gap(target_preset, {"cumulative_preset": cumulative})
        gap_str = gap["context_str"] if gap else "several parameters"
        is_last = bi == len(batches) - 1

        if args.omni_server:
            plan_bullet = _extract_plan_bullet(diagnosis_text, b.subsystem)
            param_deltas: list[tuple[str, float, float]] = [
                (name, batch_before_values.get(name, 0.0), float(cumulative["settings"].get(name, 0.0) or 0.0))
                for name in sorted(b.params_applied.keys())
            ]
            _ot = _time.monotonic()
            check_sentence = stage2_batch_check(
                subsystem=b.subsystem,
                plan_bullet=plan_bullet,
                param_deltas=param_deltas,
                prior_checks=prior_checks,
                archetype=archetype,
                stage2_server=stage2_server,
                stage2_model=stage2_model,
                is_final=is_last and not b.mistake,
                n_params_applied=len(b.params),
            )
            _log(f"  batch {bi}/{len(batches)-1} ({b.subsystem}) omni {_time.monotonic()-_ot:.1f}s")
        else:
            check_sentence = f"{b.subsystem.capitalize()} edits applied consistent with the plan."
        prior_checks.append(check_sentence)
        pending_check = check_sentence

        batch_labels.append({
            "index": bi,
            "subsystem": b.subsystem,
            "param_names": sorted(b.params.keys()),
            "clap_score_after_batch": clap_after,
            "is_correction": False,
        })

        # ---- INLINE CORRECTION (immediately after the mistaken batch) ----
        if b.mistake:
            # Fix the mistake param back to its true value
            true_norm = b.mistake["true_value"]
            cumulative["settings"][b.mistake["param"]] = _denormalize(b.mistake["param"], true_norm)

            corr_wav = batch_audio_dir / f"batch_{bi}_correction.wav"
            render_cumulative_audio(cumulative, notes, corr_wav)
            with serial_lock:
                try:
                    corr_clap = float(embedder.cosine_paths(corr_wav, target_audio_path))
                except Exception:
                    corr_clap = None

            disp = _json_key_to_display(b.mistake["param"])
            corr_prefix = f"{pending_check}\n\n" if pending_check else ""
            pending_check = None
            if args.omni_server:
                corr_intro = corr_prefix + stage2_correction_intro(
                    subsystem=b.subsystem,
                    param_display_names=[disp],
                    mistake_info=b.mistake,
                    archetype=archetype,
                    stage2_server=stage2_server,
                    stage2_model=stage2_model,
                )
            else:
                corr_intro = f"{corr_prefix}Overshot on {b.subsystem} — backing off {disp} to the planned value."
            messages.append({"role": "assistant", "content": corr_intro})

            corr_display_norm = {_json_key_to_reaper_display(b.mistake["param"]): float(true_norm)}
            messages.append(_tool_call("Bash", {"command": build_batch_action_snippet(corr_display_norm)}))
            _corr_stdout = json.dumps({"status": "ok", "applied": 1, "not_found": []}) + "\n"
            messages.append(_bash_tool_response(_corr_stdout))

            audio_assets.append(str(corr_wav))
            messages.append({"role": "assistant", "content": "Listening to the corrected preset."})
            messages.append(_tool_call("Bash", {"command": _build_listen_probe_command(corr_wav)}))
            _emit_listen_sequence(
                messages, audio_assets, corr_wav,
                listen_text="Reading corrected audio.",
            )
            last_batch_audio = corr_wav
            pending_check = f"The {b.subsystem} region now sits back in line with the plan."

            batch_labels.append({
                "index": len(batch_labels),
                "subsystem": "correction",
                "param_names": [b.mistake["param"]],
                "clap_score_after_batch": corr_clap,
                "is_correction": True,
            })

    # If no batches, flush diagnosis
    if _diagnosis_text:
        messages.append({"role": "assistant", "content": _diagnosis_text})

    # ---- FINAL ASSESSMENT ----
    final_gap = _step_remaining_gap(target_preset, {"cumulative_preset": cumulative})
    final_gap_str = final_gap["context_str"] if final_gap else "several parameters"
    truth_for_verdict = []
    if final_gap:
        for subsys_display in final_gap.get("by_subsystem", {}):
            truth_for_verdict.append(str(subsys_display).lower())

    final_audio_wav = last_batch_audio

    if args.omni_server and final_audio_wav:
        try:
            verdict_obs = omni_stage1_verdict(
                str(target_audio_path), str(final_audio_wav),
                archetype, args.omni_server, args.omni_model,
            )
        except Exception:
            verdict_obs = "The recreation is close to the target but some timbral details still differ."
    else:
        verdict_obs = "The recreation captures the target's core character."

    fully_converged = final_gap is not None and final_gap["n_remaining"] == 0

    # Compute perceptual residual delta (what's still different between target
    # and final cumulative preset) — grounds the verdict in concrete audible
    # differences instead of pattern-matched 'envelope N'.
    from scripts.preset_perceptual_summary import summarize_residual_delta_perceptual
    residual_delta_summary = summarize_residual_delta_perceptual(target_preset, cumulative)

    _vt = _time.monotonic()
    verdict_text = stage2_verdict(
        perceptual_obs=verdict_obs,
        residual_delta_summary=residual_delta_summary,
        path_complete=fully_converged,
        archetype=archetype,
        stage2_server=stage2_server,
        stage2_model=stage2_model,
    )
    _log(f"verdict omni {_time.monotonic()-_vt:.1f}s")
    if pending_check:
        verdict_text = f"{pending_check}\n\n{verdict_text}"
    messages.append({"role": "assistant", "content": verdict_text})

    # ---- Record assembly ----
    diagnosis_subs_mentioned = extract_diagnosis_subsystems_mentioned(diagnosis_text)
    mistake_caught = injected_mistake is not None  # inline correction always emitted

    record = {
        "id": sample_id,
        "task_type": "main",
        "tools": _V3_TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "assets": {
            "target_audio": str(target_audio_path),
            "current_audio": str(default_audio_path),
            "candidate_audio": [],
            "selected_candidates": [],
            "selected_tuples": [],
        },
        "labels": {
            "gt_wavetable_names": gt_names_list,
            "applied_wavetables": [gt_tuple[oi] for oi in active_oscs if gt_tuple[oi]],
            "search_pool_size": len(pool),
            "search_rounds_used": rounds_used,
            "search_slice_starts_per_round": round_offsets_used,
            "search_judge_verdicts": verdicts_by_round,
            "search_final_verdict": verdicts_by_round[-1] if verdicts_by_round else None,
            "search_rounds_exhausted_on_no_match": judge_exhausted_fallback,
        },
        "meta": {
            "pipeline_version": "v3",
            "pipeline_version_notes": (
                "WT scaffold uses GT-CLAP-similarity pool + tuple render+listen. "
                "Audio is rendered fresh per-batch using vita — each batch's audio "
                "reflects exactly the cumulative preset state after that batch. "
                "Wavetables applied via library lookup by name."
            ),
            "sample_id": sample_id,
            "archetype": archetype,
            "start_type": entry.get("start_type", "init"),
            "agent": "main",
            "num_agents": int(args.num_agents),
            "pool_top_k": int(args.pool_top_k),
            "max_batches": int(args.max_batches),
            "mistake_rate": float(args.mistake_rate),
            "commentary_mode": "two_stage",
            "path_complete": path_complete,
            "n_remaining": final_gap["n_remaining"] if final_gap else 0,
            "batch_labels": batch_labels,
            "diagnosis_subsystems_mentioned": diagnosis_subs_mentioned,
            "diagnosis_subsystems_truth": subsystems_truth,
            "injected_mistake": injected_mistake,
            "mistake_caught": mistake_caught if injected_mistake else None,
            "transcription_mistake": _trans_mistake_info,
            "transcription_mistake_caught": bool(_trans_mistake_info),
        },
    }

    assert_valid_ms_swift_multiturn_record(record)
    return record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Build main-agent SFT dataset v3 (diagnose → subsystem batches).")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-batches", type=int, default=16,
        help="Cap on regular (non-correction) subsystem batches per conversation.")

    ap.add_argument("--pool-top-k", type=int, default=48,
        help="[LEGACY] ignored in index-based mode.")
    ap.add_argument("--num-agents", type=int, default=4,
        help="Number of search agents dispatched per round.")
    ap.add_argument("--candidates-per-slice", type=int, default=48,
        help="Wavetables per search agent slice (default 48).")
    ap.add_argument("--max-search-rounds", type=int, default=3,
        help="Max search rounds before giving up and using best available pool (default 3).")
    ap.add_argument("--force-research-rate", type=float, default=0.30,
        help="Fraction of samples where GT-oracle rotation is skipped, forcing re-search (default 0.30).")
    ap.add_argument("--no-audio-rate", type=float, default=0.05,
        help="Fraction of samples where the user's first message has no <audio> tag (~5%%), "
             "producing a single-turn 'please select an audio clip' refusal. Teaches the model "
             "to recognise the missing attachment instead of proceeding with a fabricated target.")
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))

    ap.add_argument("--mistake-rate", type=float, default=0.20,
        help="Probability of injecting one deliberate overshoot per sample (default 0.20).")
    ap.add_argument("--transcription-mistake-rate", type=float, default=0.15,
        help="Probability of injecting a transcription mistake per sample (default 0.15). "
             "When injected, the first transcription dispatch returns notes with one pitch "
             "altered by +-1-2 semitones; the main agent's verify-listen detects the "
             "mismatch and re-dispatches transcription with a hint; the second attempt is "
             "correct. Teaches the model to catch+recover from transcription errors.")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument("--omni-server", default="", help="Omni audio server URL (empty = template fallback).")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--stage2-server", default="", help="Stage 2 text model server (defaults to --omni-server).")
    ap.add_argument("--stage2-model", default="", help="Stage 2 model name (defaults to --omni-model).")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    stage2_server = args.stage2_server or args.omni_server
    stage2_model = args.stage2_model or args.omni_model

    if args.omni_server:
        _check_server_reachable(args.omni_server, "Omni")
        if stage2_server and stage2_server != args.omni_server:
            _check_server_reachable(stage2_server, "Stage2")

    entries = load_manifest_entries(Path(args.manifest), max_samples=args.max_samples)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    embedder = ClapEmbedder.create(args.clap_device)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)

    _notes = make_probe_notes("lead", clip_duration_s=10.0)

    # Pre-populate CLAP embedding cache (same dance as v2 — GPU in main thread).
    clap_paths: list[Path] = []
    for e in entries:
        for k in ("gt_wav", "gt_probe_wav", "default_wav"):
            if e.get(k):
                clap_paths.append(Path(e[k]))
    probe_dir = Path(args.probe_dir)
    if probe_dir.exists():
        for pp in sorted(probe_dir.glob("*.wav")):
            clap_paths.append(pp)
    clap_paths = list(dict.fromkeys(clap_paths))
    print(f"Pre-computing CLAP embeddings for {len(clap_paths)} audio files...", flush=True)
    for p in clap_paths:
        if p.exists():
            try:
                embedder.embed_audio_path(p)
            except Exception as exc:
                print(f"  WARNING: CLAP embed failed for {p.name}: {exc}")
    print(f"CLAP pre-computation done ({len(embedder._cache)} cached).", flush=True)
    if args.clap_device != "cpu":
        try:
            embedder.model = embedder.model.to("cpu")
            embedder.device = "cpu"
            print("CLAP moved to CPU for worker-thread safety.", flush=True)
        except Exception as exc:
            print(f"WARNING: could not move CLAP to CPU: {exc}")

    candidate_audio: dict[str, Path] = {}
    serial_lock = threading.Lock()  # guards CLAP model (not thread-safe for concurrent forward())

    def _process(entry: dict) -> dict | None:
        return build_record(
            entry=entry,
            args=args,
            embedder=embedder,
            shortlist_data=shortlist_data,
            selected_by_name=selected_by_name,
            wavetable_lib=wavetable_lib,
            index_rows=index_rows,
            candidate_audio=candidate_audio,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            serial_lock=serial_lock,
            notes=_notes,
        )

    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records_by_idx: dict[int, dict] = {}
    write_lock = threading.Lock()

    def _record(i: int, rec: dict | None) -> None:
        if rec is None:
            return
        with write_lock:
            records_by_idx[i] = rec
            sid = rec["meta"]["sample_id"]
            print(f"[{len(records_by_idx)}/{len(entries)}] {sid} OK", flush=True)

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process, e): i for i, e in enumerate(entries)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    _record(i, fut.result())
                except Exception as exc:
                    import traceback
                    sid = entries[i].get("sample_id", f"entry_{i}")
                    print(f"WARNING: {sid} failed: {exc}")
                    traceback.print_exc()
    else:
        for i, e in enumerate(entries):
            try:
                _record(i, _process(e))
            except Exception as exc:
                import traceback
                print(f"WARNING: {e.get('sample_id','?')} failed: {exc}")
                traceback.print_exc()

    records = [records_by_idx[i] for i in sorted(records_by_idx)]
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {out_path}", flush=True)


if __name__ == "__main__":
    main()
