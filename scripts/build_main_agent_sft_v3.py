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
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
)
from scripts.build_main_agent_sft_v2 import (
    _build_listen_probe_command,
    _check_server_reachable,
    _harden_vital_snippet_for_reapy,
    _llm_post,
    _step_remaining_gap,
    _tool_call,
    _wrap_as_bash,
)

# v3-specific tool specs: claw-code-style Agent tool + bash. Replaces the
# v2 spawn_search_agents / collect_search_reports / judge_candidates trio.
_V3_TOOL_SPECS = json.dumps(
    [
        {
            "type": "function",
            "function": {
                "name": "bash",
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


# ---------------------------------------------------------------------------
# Vita rendering — fresh per-batch audio (no iter_wav dependency)
# ---------------------------------------------------------------------------

import numpy as np
import soundfile as sf
from maestro.render.vital import SAMPLE_RATE, _load_vital, _render_note_list, make_probe_notes, trim_silence


def render_cumulative_audio(
    synth,
    cumulative_preset: dict,
    notes: list,
    out_path: Path,
    tail_s: float = 1.0,
) -> Path:
    """Render audio for a cumulative preset state and write to ``out_path``."""
    synth.load_json(json.dumps(cumulative_preset))
    audio = _render_note_list(synth, notes, SAMPLE_RATE, tail_s=tail_s)
    audio = trim_silence(audio, SAMPLE_RATE, min_duration_s=0.5)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio.T, SAMPLE_RATE)
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
# Action snippet — apply a batch of params atomically via VitalController
# ---------------------------------------------------------------------------


def build_batch_action_snippet(params_native: dict[str, float]) -> str:
    """Emit a Python snippet that calls vc.set_params({...}) with native values.

    params_native — {name: native_value} (already denormalized).
    """
    params_repr = ", ".join(
        f'"{name}": {val!r}' for name, val in sorted(params_native.items())
    )
    snippet = (
        "import sys, json\n"
        "sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
        "from vital_tools import VitalController\n"
        "vc = VitalController()\n"
        "vc.discover()\n"
        f"result = vc.set_params({{{params_repr}}})\n"
        'print(json.dumps({"status": "ok", "applied": result["applied"], "not_found": result["not_found"]}))'
    )
    return _harden_vital_snippet_for_reapy(snippet)


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
    default_wav: str,
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """Stage 1: Omni listens to GT + default, returns a perceptual comparison.

    Output goes into Stage 2's DIAGNOSIS prompt as observations.
    """
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(gt_wav)}"}},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(default_wav)}"}},
        {"type": "text", "text": (
            f"You are a music production AI. Listen to two synthesizer clips.\n"
            f"AUDIO A: the TARGET {archetype} sound we need to recreate.\n"
            f"AUDIO B: the current DEFAULT preset.\n\n"
            "Describe the perceptual differences: frequency balance (bright/warm/dark), "
            "harmonic character (clean/buzzy/rich), envelope shape (sharp/slow attack, "
            "short/long decay, sustain level), and any motion or modulation.\n"
            "3-5 short sentences. Focus on what the TARGET has that the default lacks. "
            "Use natural production language, no snake_case parameter names, no kHz numbers."
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
            f"The target {archetype} has a distinct timbral character and envelope shape that "
            "the default sine preset does not. Differences in harmonic content, filter "
            "behaviour, and modulation motion are all audible."
        )


def omni_stage1_verdict(
    gt_wav: str,
    final_wav: str,
    archetype: str,
    omni_server: str,
    omni_model: str,
) -> str:
    """Stage 1 for final verdict: compare final recreation vs GT."""
    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(gt_wav)}"}},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{_b64(final_wav)}"}},
        {"type": "text", "text": (
            f"You are a music production AI doing a final review.\n"
            f"AUDIO A is the target {archetype} sound.\n"
            f"AUDIO B is the final recreation after all subsystem edits.\n\n"
            "In 2 sentences: what matches well, and what (if anything) still differs. "
            "Be specific and honest. No snake_case. No kHz numbers."
        )},
    ]
    try:
        r = _llm_post(
            f"{omni_server}/v1/chat/completions",
            {
                "model": omni_model,
                "messages": [{"role": "user", "content": content}],
                "max_tokens": 180,
                "temperature": 0.4,
            },
            timeout=180.0,
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
        f"a {archetype} synth sound in Vital.\n\n"
        f"--- Perceptual observations (listening to target vs default) ---\n"
        f"{perceptual_obs}\n\n"
        f"--- Ground truth: subsystems that need changes ---\n"
        f"{subsystem_diff_summary}\n\n"
        f"Write exactly two sections:\n\n"
        f"OBSERVATIONS: 2-3 sentences rewording the perceptual observations above in "
        f"your own voice. Describe what the target has that the current preset lacks. "
        f"No snake_case parameter names, no **bold** headers, no kHz numbers.\n\n"
        f"PLAN: A bulleted list. Exactly one bullet per subsystem in this list (in order): "
        f"{bullets_hint}. Each bullet starts with the subsystem name capitalised (e.g. "
        f"\"• Oscillator:\"), followed by one short qualitative sentence about what change "
        f"is needed there (e.g. \"swap to a brighter wavetable and add unison detune\"). "
        f"Do NOT write any numeric values or exact param names. Do NOT use snake_case. "
        f"End the PLAN section with the exact line: \"Executing plan by subsystem.\"\n\n"
        f"Archetype: {archetype}."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.4,
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


def stage2_batch_check(
    subsystem: str,
    remaining_gap_str: str,
    prior_checks: list[str],
    archetype: str,
    stage2_server: str,
    stage2_model: str,
    is_final: bool = False,
    n_params_applied: int = 0,
) -> str:
    """Stage 2: write one short sentence describing what THIS batch specifically changed.

    Focuses on the concrete effect of the current subsystem's edits, not the
    full remaining diff (which causes repetitive "dynamics and spatial effects"
    phrasing across all batches).
    """
    recent = " / ".join(prior_checks[-2:]) if prior_checks else ""
    recent_hint = (
        f"Prior batch checks (DO NOT repeat these phrasings): \"{recent}\"" if recent else ""
    )
    final_hint = (
        "This is the LAST planned batch — note that the preset is now nearly complete." if is_final else ""
    )
    prompt = (
        f"You are a music production AI. You just applied {n_params_applied} {subsystem} "
        f"parameter edits for a {archetype} preset.\n\n"
        f"After this batch, remaining subsystem gaps: {remaining_gap_str}.\n\n"
        f"{recent_hint}\n{final_hint}\n\n"
        f"Write EXACTLY ONE sentence. Focus on what the {subsystem} edits specifically "
        f"changed about the sound's character (e.g. 'filter darkened the tone and added "
        f"resonant sweep' or 'LFO introduced rhythmic pulsing to the brightness'). "
        f"Do NOT list the remaining subsystems — focus on the effect of THIS batch. "
        f"Natural language, no snake_case, no **bold**, under 25 words."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 100,
                "temperature": 0.5,
            },
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip().split("\n")[0]
    except Exception:
        return f"{subsystem.capitalize()} edits applied; remaining gap is in {remaining_gap_str}."


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
                "temperature": 0.5,
            },
            timeout=120.0,
        )
        return r["choices"][0]["message"]["content"].strip().split("\n")[0]
    except Exception:
        return f"Overshot on {subsystem} — backing off {params_str} to the planned values."


def stage2_verdict(
    perceptual_obs: str,
    final_remaining_gap: str,
    path_complete: bool,
    archetype: str,
    subsystems_truth: list[str],
    stage2_server: str,
    stage2_model: str,
) -> str:
    """Stage 2: write FINAL ASSESSMENT grounded in residual diff."""
    status_tag = "(complete)" if path_complete else "(budget_exhausted)"
    truth_hint = ", ".join(subsystems_truth[:3]) if subsystems_truth else "nothing"
    prompt = (
        f"You are a music production AI writing the final assessment of a {archetype} "
        f"synth recreation.\n\n"
        f"Perceptual review:\n{perceptual_obs}\n\n"
        f"Residual subsystem differences from target: {final_remaining_gap}\n"
        f"Path status: {'converged' if path_complete else 'budget exhausted'}.\n\n"
        f"Write a single line beginning with 'FINAL ASSESSMENT {status_tag}: ' followed "
        f"by exactly 2 short sentences. Sentence 1 = what matches well. Sentence 2 = the "
        f"most important remaining difference (reference one of these subsystems if any "
        f"still differ: {truth_hint}). No snake_case. No **bold**. No kHz numbers."
    )
    try:
        r = _llm_post(
            f"{stage2_server}/v1/chat/completions",
            {
                "model": stage2_model,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 160,
                "temperature": 0.4,
            },
            timeout=120.0,
        )
        text = r["choices"][0]["message"]["content"].strip()
        if not text.startswith("FINAL ASSESSMENT"):
            text = f"FINAL ASSESSMENT {status_tag}: {text}"
        return text
    except Exception:
        suffix = final_remaining_gap if not path_complete else "the target is closely matched"
        return (
            f"FINAL ASSESSMENT {status_tag}: The recreation captures the target's core "
            f"timbre. The most notable remaining difference is in {suffix}."
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
    synth: Any,
    notes: list,
) -> dict | None:
    """Build one v3 SFT record. Returns None to skip."""
    sample_id = str(entry["sample_id"])
    archetype = str(entry.get("archetype", "synth"))
    target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
    default_audio_path = Path(entry["default_wav"]) if entry.get("default_wav") else None
    if default_audio_path is None:
        return None

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

    messages.append({
        "role": "user",
        "content": (
            f"<audio>\nRecreate this {archetype} target sound in Vital from default.\n"
            "Search for matching wavetables across the library, evaluate combinations, "
            "then execute by subsystem."
        ),
    })

    # Listen to baseline
    messages.append({"role": "assistant", "content": "Listening to current default preset baseline."})
    messages.append(_tool_call("bash", {"command": _build_listen_probe_command(default_audio_path)}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps(
            {"status": "ok", "baseline_audio": "<audio>", "path": str(default_audio_path)},
            ensure_ascii=False,
        ),
    })

    # Step 1: check library size
    messages.append({"role": "assistant", "content": "Checking wavetable library size."})
    messages.append(_tool_call("bash", {"command": "python scripts/list_wavetables.py --total"}))
    messages.append({"role": "tool_response", "content": json.dumps({"total": total_named}, ensure_ascii=False)})

    # Agent output directory (real path; runtime executor writes files here at inference)
    agent_out_dir = f"/tmp/agents/{sample_id}"

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

    # ---- Multi-round search loop ----
    pool: list[str] = []
    rounds_used = 0
    round_offsets_used: list[list[int]] = []

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

    def _write_shortlist_file(agent_id: str, start: int, end: int, shortlist: list[str]) -> str:
        """Write single-line JSON shortlist file (simulates search agent runtime output)."""
        out_dir = Path(agent_out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"{agent_id}.json"
        with open(path, "w") as f:
            json.dump({
                "status": "completed",
                "agentId": agent_id,
                "shardStart": start,
                "shardEnd": end,
                "shortlist": shortlist,
            }, f)
            f.write("\n")
        return str(path)

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
                f"agents across slices [{', '.join(f'{s}-{s + slice_size - 1}' for s in slice_starts)}]."
            )
        else:
            intro = (
                f"{_research_prefix}"
                f"Expanding to different library regions with {n_agents} more search agents: "
                f"[{', '.join(f'{s}-{s + slice_size - 1}' for s in slice_starts)}]."
            )
            _research_prefix = ""
        messages.append({"role": "assistant", "content": intro})

        # Step 2: Agent tool call per shard
        round_output_files: list[str] = []
        round_agent_ids: list[str] = []
        for ai, start in enumerate(slice_starts):
            end = min(start + slice_size, total_named)
            if end <= start:
                continue
            agent_id = f"wavetable_search_{sample_id}_r{rounds_used}_a{ai + 1}"
            round_agent_ids.append(agent_id)

            # Simulate the shortlist this agent would return (build-time)
            sl = _simulate_shortlist(start, end)
            output_file = _write_shortlist_file(agent_id, start, end, sl)
            round_output_files.append(output_file)

            # Agent tool call
            messages.append(_tool_call("Agent", {
                "subagent_type": "wavetable_search",
                "description": f"Evaluate wavetables {start}-{end - 1} for {archetype} target",
                "prompt": (
                    f"Target: {target_audio_path}. Archetype: {archetype}.\n"
                    f"Evaluate wavetables at indices {start}-{end - 1}. "
                    f"Use `python scripts/list_wavetables.py --start {start} --end {end}` "
                    f"and `python scripts/render_wavetable_probes.py --idxs ... --out-dir ...` "
                    f"to hear each candidate. Return a JSON shortlist of 2-4 wavetable names."
                ),
                "name": f"search-{rounds_used}-{ai + 1}",
            }))
            messages.append({
                "role": "tool_response",
                "content": json.dumps({
                    "agentId": agent_id,
                    "subagentType": "wavetable_search",
                    "status": "completed",
                    "outputFile": output_file,
                }, ensure_ascii=False),
            })

        # Step 3: read all output files in one cat call
        cat_cmd = "cat " + " ".join(round_output_files)
        messages.append({
            "role": "assistant",
            "content": f"Reading shortlists from {len(round_output_files)} search agents.",
        })
        messages.append(_tool_call("bash", {"command": cat_cmd}))
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
        messages.append({
            "role": "tool_response",
            "content": "\n".join(cat_output_lines),
        })

        # Pool in new shortlists
        for sl in round_shortlists:
            for name in sl:
                if name not in pool:
                    pool.append(name)

        # Build best tuple from current pool
        cur_tuple: list[str | None] = [None, None, None]
        for osc_idx in active_oscs:
            wts = target_preset.get("settings", {}).get("wavetables", [])
            if osc_idx < len(wts):
                wt_name = wts[osc_idx].get("name", "")
                if wt_name and wt_name in pool:
                    cur_tuple[osc_idx] = wt_name
        # Fill empty slots with best available from pool
        _non_gt_pool = [n for n in pool if n not in gt_names_list]
        _fb = 0
        for osc_idx in active_oscs:
            if cur_tuple[osc_idx] is None and _non_gt_pool:
                cur_tuple[osc_idx] = _non_gt_pool[min(_fb, len(_non_gt_pool) - 1)]
                _fb += 1

        # Render the tuple to hear it
        cur_active_names = [cur_tuple[oi] for oi in active_oscs if cur_tuple[oi]]
        osc_args = " ".join(
            f"--osc{oi + 1} {json.dumps(cur_tuple[oi])}" for oi in active_oscs if cur_tuple[oi]
        )
        tuple_wav = tuple_audio_dir / f"tuple_r{rounds_used}.wav"
        render_cmd = (
            f"python scripts/render_wavetable_tuple.py {osc_args} "
            f"--out {tuple_wav}"
        )
        with serial_lock:
            render_cumulative_audio(synth, _build_tuple_preset(cur_tuple, active_oscs, wt_lib_by_name, init_preset), notes, tuple_wav)

        pool_str = ", ".join(f"'{n}'" for n in pool[:6])
        if len(pool) > 6:
            pool_str += f", ...+{len(pool) - 6} more"
        tuple_names_str = ", ".join(f"'{n}'" for n in cur_active_names)

        messages.append({
            "role": "assistant",
            "content": f"Pooled {len(pool)} candidates: [{pool_str}]. Rendering tuple [{tuple_names_str}] to evaluate.",
        })
        messages.append(_tool_call("bash", {"command": render_cmd}))
        audio_assets.append(str(tuple_wav))
        messages.append({
            "role": "tool_response",
            "content": json.dumps({
                "status": "ok", "tuple_audio": "<audio>",
                "out": str(tuple_wav), "wavetables": cur_active_names,
            }, ensure_ascii=False),
        })

        # Re-search decision: are ALL GT wavetable names in the pool?
        all_gt_found = all(gt in pool for gt in gt_names_list)
        if all_gt_found or rounds_used >= max_rounds:
            break

        # Tuple doesn't have all GTs → re-search.
        # Store re-search text to merge into next round's dispatch intro
        # (avoids adjacent assistant messages).
        _research_prefix = (
            "The rendered tuple doesn't capture the target's character closely enough. "
        )

        # Shift offsets for next round — avoid re-searching same regions
        base_offset = (base_offset + stride // 2) % stride
        slice_starts = _compute_slices(base_offset)

    # The best tuple from the last search round is cur_tuple (already rendered + heard).
    # Use it for the apply step.
    gt_tuple = cur_tuple
    apply_names = [gt_tuple[oi] for oi in active_oscs if gt_tuple[oi]]
    osc_assignments = ", ".join(
        f"oscillator {oi + 1} = '{gt_tuple[oi]}'" for oi in active_oscs if gt_tuple[oi]
    )
    if all_gt_found:
        selection_text = f"This tuple matches the target well. Applying: {osc_assignments}."
    else:
        selection_text = f"Search budget exhausted. Applying best available: {osc_assignments}."

    apply_assignments = ", ".join(
        f'({oi}, {json.dumps(gt_tuple[oi])})' for oi in active_oscs if gt_tuple[oi]
    )
    apply_snippet = (
        "import sys, json\n"
        "sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
        "from vital_tools import VitalController\n"
        "vc = VitalController()\n"
        "vc.discover()\n"
        f"wt_lib = json.load(open({json.dumps(str(args.wavetable_lib))}))\n"
        "name_to_wt = {wt['name']: wt for wt in wt_lib}\n"
        "preset = vc.get_preset()\n"
        f"for osc_idx, wt_name in [{apply_assignments}]:\n"
        "    if wt_name in name_to_wt:\n"
        "        preset['settings']['wavetables'][osc_idx] = name_to_wt[wt_name]\n"
        "vc.set_preset(preset)\n"
        f"print(json.dumps({{'status': 'ok', 'applied': {json.dumps(apply_names)}}}))"
    )
    apply_snippet = _harden_vital_snippet_for_reapy(apply_snippet)
    messages.append({"role": "assistant", "content": selection_text})
    messages.append(_tool_call("bash", {"command": _wrap_as_bash(apply_snippet)}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps({"status": "ok", "applied": apply_names}, ensure_ascii=False),
    })

    # ---- DIAGNOSIS BLOCK (block 1) ----
    subsystem_truth_map = build_diagnosis_subsystem_truth(target_preset, init_preset)
    subsystems_truth = [lbl for lbl, _ in SUBSYSTEM_ORDER if lbl in subsystem_truth_map]
    diff_summary = format_subsystem_diff_summary(subsystem_truth_map)

    if args.omni_server:
        stage1_obs = omni_stage1_diagnose(
            str(target_audio_path), str(default_audio_path),
            archetype, args.omni_server, args.omni_model,
        )
    else:
        stage1_obs = f"Target {archetype} differs from default in several subsystems."
    diagnosis_text = stage2_diagnosis(
        stage1_obs, diff_summary, subsystems_truth,
        archetype, stage2_server, stage2_model,
    )

    _diagnosis_text = diagnosis_text

    # ---- SUBSYSTEM BATCHES (diff-based, vita-rendered) ----
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

    for bi, b in enumerate(batches):
        # Apply batch params to cumulative (uses params_applied which may contain mistake)
        for name, norm in b.params_applied.items():
            cumulative["settings"][name] = _denormalize(name, norm)

        # Render fresh audio for this cumulative state (under serial lock — one Synth per process)
        batch_wav = batch_audio_dir / f"batch_{bi}_{b.subsystem}.wav"
        with serial_lock:
            render_cumulative_audio(synth, cumulative, notes, batch_wav)

        # CLAP score vs GT
        with serial_lock:
            try:
                clap_after = float(embedder.cosine_paths(batch_wav, target_audio_path))
            except Exception:
                clap_after = None

        b.audio_wav = batch_wav
        native_params = {n: _denormalize(n, v) for n, v in b.params_applied.items()}
        action_snippet = build_batch_action_snippet(native_params)

        # Intro message — merge diagnosis (for first batch) or prior check
        intro = f"Applying {b.subsystem} changes."
        if bi == 0 and _diagnosis_text:
            intro = f"{_diagnosis_text}\n\n{intro}"
            _diagnosis_text = None
        elif pending_check:
            intro = f"{pending_check}\n\n{intro}"
            pending_check = None
        messages.append({"role": "assistant", "content": intro})
        messages.append(_tool_call("bash", {"command": _wrap_as_bash(action_snippet)}))
        messages.append({"role": "tool_response", "content": json.dumps({"status": "ok"}, ensure_ascii=False)})

        # Listen
        audio_assets.append(str(batch_wav))
        messages.append({"role": "assistant", "content": f"Listening after {b.subsystem} batch."})
        messages.append(_tool_call("bash", {"command": _build_listen_probe_command(batch_wav)}))
        messages.append({
            "role": "tool_response",
            "content": json.dumps(
                {"status": "ok", "batch_audio": "<audio>", "path": str(batch_wav)},
                ensure_ascii=False,
            ),
        })
        last_batch_audio = batch_wav

        # Remaining-gap check
        gap = _step_remaining_gap(target_preset, {"cumulative_preset": cumulative})
        gap_str = gap["context_str"] if gap else "several parameters"
        is_last = bi == len(batches) - 1

        if args.omni_server:
            check_sentence = stage2_batch_check(
                b.subsystem, gap_str, prior_checks, archetype,
                stage2_server, stage2_model, is_final=is_last and not b.mistake,
                n_params_applied=len(b.params),
            )
        else:
            check_sentence = f"{b.subsystem.capitalize()} edits applied; remaining gap is in {gap_str}."
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
            with serial_lock:
                render_cumulative_audio(synth, cumulative, notes, corr_wav)
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

            corr_native = {b.mistake["param"]: _denormalize(b.mistake["param"], true_norm)}
            messages.append(_tool_call("bash", {"command": _wrap_as_bash(build_batch_action_snippet(corr_native))}))
            messages.append({"role": "tool_response", "content": json.dumps({"status": "ok"}, ensure_ascii=False)})

            audio_assets.append(str(corr_wav))
            messages.append({"role": "assistant", "content": "Listening to the corrected preset."})
            messages.append(_tool_call("bash", {"command": _build_listen_probe_command(corr_wav)}))
            messages.append({
                "role": "tool_response",
                "content": json.dumps(
                    {"status": "ok", "corrected_audio": "<audio>", "path": str(corr_wav)},
                    ensure_ascii=False,
                ),
            })
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
    verdict_text = stage2_verdict(
        perceptual_obs=verdict_obs,
        final_remaining_gap=final_gap_str,
        path_complete=fully_converged,
        archetype=archetype,
        subsystems_truth=truth_for_verdict,
        stage2_server=stage2_server,
        stage2_model=stage2_model,
    )
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
    ap.add_argument("--max-batches", type=int, default=6,
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
    ap.add_argument("--probe-dir", type=Path, default=Path("outputs/agent_sft/candidate_probes"))

    ap.add_argument("--mistake-rate", type=float, default=0.20,
        help="Probability of injecting one deliberate overshoot per sample (default 0.20).")
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

    # Pre-load vita synth + probe notes (singleton, serialized via lock)
    _synth = _load_vital()
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
    serial_lock = threading.Lock()

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
            synth=_synth,
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
