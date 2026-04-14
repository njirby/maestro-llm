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
    build_disjoint_shards,
    choose_candidate_pool,
    ensure_candidate_probes_for_names,
    extract_gt_wavetable_names,
    load_index_rows,
    load_manifest_entries,
    load_wavetable_lib,
    select_probe_rows_by_name,
)
from scripts.build_main_agent_sft_v2 import (
    _TOOL_SPECS,
    _build_judge_result,
    _build_listen_probe_command,
    _build_search_reports,
    _check_server_reachable,
    _harden_vital_snippet_for_reapy,
    _llm_post,
    _step_remaining_gap,
    _tool_call,
    _wrap_as_bash,
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
) -> str:
    """Stage 2: write one short sentence describing the batch result.

    remaining_gap_str — output of _step_remaining_gap(...)['context_str'] after batch applied.
    prior_checks — previous batch-check sentences (to avoid repetition).
    """
    recent = " / ".join(prior_checks[-2:]) if prior_checks else ""
    recent_hint = (
        f"Previous batch checks: \"{recent}\" — do not repeat those phrasings." if recent else ""
    )
    final_hint = (
        "This is the LAST planned batch; frame the sentence as 'getting close' rather than "
        "'next step will'." if is_final else ""
    )
    prompt = (
        f"You are a music production AI. You just applied the {subsystem} edits for a "
        f"{archetype} preset. Remaining differences from target (by subsystem): "
        f"{remaining_gap_str}.\n\n"
        f"{recent_hint}\n{final_hint}\n\n"
        f"Write EXACTLY ONE sentence describing what the {subsystem} edits addressed and "
        f"what still differs most. Natural language, no snake_case, no **bold**, no kHz "
        f"numbers, no section headers. Keep it under 30 words."
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
    universe_names: list[str],
    shortlist_data: dict,
    selected_by_name: dict,
    wavetable_lib: list[dict],
    index_rows: list[dict],
    candidate_audio: dict[str, Path],
    stage2_server: str,
    stage2_model: str,
    serial_lock: threading.Lock,
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

    # ---- WT search / judge scaffold ----
    with serial_lock:
        gt_names = set(extract_gt_wavetable_names(Path(target_preset_path)))
        if not gt_names:
            return None

        candidate_names = choose_candidate_pool(
            sample_id=sample_id,
            query_audio_path=target_audio_path,
            gt_names=sorted(gt_names),
            universe_names=universe_names,
            candidate_source=args.candidate_source,
            candidate_limit=int(args.candidate_limit),
            oracle_hard_pool=int(args.oracle_hard_pool),
            seed=int(args.seed),
            clap_embedder=embedder,
            selected_rows_meta=index_rows,
            shortlist_data=shortlist_data,
        )
        if not candidate_names:
            return None

        ensure_candidate_probes_for_names(
            names=candidate_names,
            wavetable_lib=wavetable_lib,
            selected_rows=selected_by_name,
            out_dir=args.probe_dir,
            cache=candidate_audio,
            probe_archetype=args.probe_archetype,
            probe_tail_s=args.probe_tail_s,
            trim_min_duration_s=args.trim_min_duration_s,
        )

        clap_scores = {
            n: embedder.cosine_paths(candidate_audio[n], target_audio_path)
            for n in candidate_names
        }

    id_map = {name: f"C{i}" for i, name in enumerate(candidate_names, start=1)}
    candidate_assets = [
        {"candidate_id": id_map[n], "wavetable_name": n, "audio_path": str(candidate_audio[n])}
        for n in candidate_names
    ]

    shards = build_disjoint_shards(candidate_names, args.num_agents)
    jobs = [
        {"agent_id": f"sa_{i}", "candidate_shard": [id_map[n] for n in shard], "seed": int(args.seed) + i}
        for i, shard in enumerate(shards, start=1)
        if shard
    ]
    reports = _build_search_reports(shards, id_map, gt_names, clap_scores, args.proposals_per_agent)
    judge_result = _build_judge_result(candidate_names, id_map, gt_names, clap_scores, args.select_k)
    selected_ids = set(judge_result["selected"])
    selected_names = [name for name in candidate_names if id_map[name] in selected_ids]

    # ---- Begin messages ----
    messages: list[dict] = []
    audio_assets: list[str] = [str(target_audio_path), str(default_audio_path)]

    messages.append({
        "role": "user",
        "content": (
            f"<audio>\nRecreate this {archetype} target sound in Vital from default.\n"
            "Listen first, write a subsystem plan, then execute by subsystem."
        ),
    })

    # WT scaffold (block 0)
    messages.append({"role": "assistant", "content": "Listening to current default preset baseline."})
    messages.append(_tool_call("bash", {"command": _build_listen_probe_command(default_audio_path)}))
    messages.append({
        "role": "tool_response",
        "content": json.dumps(
            {"status": "ok", "baseline_audio": "<audio>", "path": str(default_audio_path)},
            ensure_ascii=False,
        ),
    })

    messages.append({"role": "assistant", "content": "Spawning disjoint search shards to gather wavetable proposals in parallel."})
    messages.append(_tool_call("spawn_search_agents", {
        "sample_id": sample_id,
        "target_audio_path": str(target_audio_path),
        "current_audio_path": str(default_audio_path),
        "candidate_universe": [c["candidate_id"] for c in candidate_assets],
        "num_agents": int(args.num_agents),
        "shard_strategy": "disjoint_round_robin",
        "seed": int(args.seed),
    }))
    messages.append({"role": "tool_response", "content": json.dumps({"jobs": jobs}, ensure_ascii=False)})

    messages.append({"role": "assistant", "content": "Collecting search-agent reports."})
    messages.append(_tool_call("collect_search_reports", {"sample_id": sample_id, "jobs": jobs}))
    messages.append({"role": "tool_response", "content": json.dumps({"reports": reports}, ensure_ascii=False)})

    messages.append({"role": "assistant", "content": "Judging candidates and selecting up to three for edits."})
    messages.append(_tool_call("judge_candidates", {
        "sample_id": sample_id,
        "target_audio_path": str(target_audio_path),
        "candidate_audio": candidate_assets,
        "max_select": int(args.select_k),
    }))
    previews = [
        {"candidate_id": id_map[n], "wavetable_name": n, "audio_preview": "<audio>"}
        for n in selected_names[: int(args.select_k)]
        if n in candidate_audio
    ]
    for n in selected_names[: int(args.select_k)]:
        if n in candidate_audio:
            audio_assets.append(str(candidate_audio[n]))
    messages.append({
        "role": "tool_response",
        "content": json.dumps({**judge_result, "selected_previews": previews}, ensure_ascii=False),
    })

    # Apply best wavetable (same pattern as v2)
    if target_preset_path and selected_names:
        ranked_selected = [
            name for cid in judge_result["selected"]
            for name in candidate_names if id_map[name] == cid
        ]
        wt_name = ranked_selected[0] if ranked_selected else selected_names[0]
        apply_snippet = (
            "import sys, json\n"
            "sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
            "from vital_tools import VitalController\n"
            "vc = VitalController()\n"
            "vc.discover()\n"
            f"with open({json.dumps(str(target_preset_path))}) as _f:\n"
            "    _src = json.load(_f)\n"
            "_wt = _src['settings']['wavetables'][0]\n"
            "if _rpr is None:\n"
            "    preset = vc.get_preset()\n"
            "    preset['settings']['wavetables'][0] = _wt\n"
            "    vc.set_preset(preset)\n"
            f"print(json.dumps({{'status': 'ok', 'applied_wavetable': {json.dumps(wt_name)}}}))"
        )
        apply_snippet = _harden_vital_snippet_for_reapy(apply_snippet)
        messages.append({
            "role": "assistant",
            "content": f"Selecting '{wt_name}' for oscillator 1 based on candidate previews.",
        })
        messages.append(_tool_call("bash", {"command": _wrap_as_bash(apply_snippet)}))
        messages.append({
            "role": "tool_response",
            "content": json.dumps({"status": "ok", "applied_wavetable": wt_name}, ensure_ascii=False),
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

    # Pre-load vita synth + note sequence under serial lock.
    with serial_lock:
        synth = _load_vital()
        notes = make_probe_notes(archetype, clip_duration_s=10.0)

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
        "tools": _TOOL_SPECS,
        "messages": messages,
        "audios": audio_assets,
        "assets": {
            "target_audio": str(target_audio_path),
            "current_audio": str(default_audio_path),
            "candidate_audio": candidate_assets,
            "selected_candidates": [
                {"candidate_id": id_map[n], "wavetable_name": n, "audio_path": str(candidate_audio[n])}
                for n in selected_names[: int(args.select_k)]
            ],
            "selected_tuples": [],
        },
        "labels": {
            "judge_ranking": judge_result["ranking"],
            "judge_selected": judge_result["selected"],
            "gt_candidate_ids": [id_map[n] for n in candidate_names if n in gt_names],
            "selected_tuple_id": None,
            "acceptable_tuple_ids": [],
            "tuple_members_by_id": {},
        },
        "meta": {
            "pipeline_version": "v3",
            "pipeline_version_notes": (
                "Audio is rendered fresh per-batch using vita — each batch's audio "
                "reflects exactly the cumulative preset state after that batch, with "
                "no cross-subsystem leakage from path_gen's support-family mechanism."
            ),
            "sample_id": sample_id,
            "archetype": archetype,
            "start_type": entry.get("start_type", "init"),
            "agent": "main",
            "num_agents": int(args.num_agents),
            "candidate_source": args.candidate_source,
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

    ap.add_argument("--candidate-source", choices=["all", "clap_topn", "oracle_mix8"], default="oracle_mix8")
    ap.add_argument("--candidate-limit", type=int, default=8)
    ap.add_argument("--oracle-hard-pool", type=int, default=64)
    ap.add_argument("--num-agents", type=int, default=4)
    ap.add_argument("--proposals-per-agent", type=int, default=3)
    ap.add_argument("--select-k", type=int, default=3)

    ap.add_argument("--probe-archetype", default="lead")
    ap.add_argument("--probe-tail-s", type=float, default=1.0)
    ap.add_argument("--trim-min-duration-s", type=float, default=0.5)
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
    universe_names = sorted(selected_by_name.keys(), key=lambda x: x.lower())
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    embedder = ClapEmbedder.create(args.clap_device)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)

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
            universe_names=universe_names,
            shortlist_data=shortlist_data,
            selected_by_name=selected_by_name,
            wavetable_lib=wavetable_lib,
            index_rows=index_rows,
            candidate_audio=candidate_audio,
            stage2_server=stage2_server,
            stage2_model=stage2_model,
            serial_lock=serial_lock,
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
