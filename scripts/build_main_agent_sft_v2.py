#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

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

_PARAM_LABELS = {
    "osc_": "oscillator",
    "env_1_": "amplitude envelope",
    "env_2_": "modulation envelope",
    "filter_1_": "filter",
    "filter_2_": "secondary filter",
    "unison_": "unison",
    "lfo_": "LFO",
    "reverb_": "reverb",
    "delay_": "delay",
    "chorus_": "chorus",
    "distortion_": "distortion",
}

_PARAM_KEY_ABBREVS = {
    "osc": "oscillator",
    "env": "envelope",
    "lfo": "lfo",
    "eq": "eq",
}

_SUBSYSTEM_DISPLAY = {
    "osc_1": "oscillator 1",
    "osc_2": "oscillator 2",
    "osc_3": "oscillator 3",
    "filter_1": "filter 1",
    "filter_2": "filter 2",
    "filter": "filter",
    "env_1": "envelope 1",
    "env_2": "envelope 2",
    "env_3": "envelope 3",
    "env_4": "envelope 4",
    "env_5": "envelope 5",
    "env_6": "envelope 6",
    "lfo_1": "LFO 1",
    "lfo_2": "LFO 2",
    "lfo_3": "LFO 3",
    "lfo_4": "LFO 4",
    "lfo_5": "LFO 5",
    "lfo_6": "LFO 6",
    "lfo_7": "LFO 7",
    "lfo_8": "LFO 8",
    "sample": "sample oscillator",
    "random_1": "random LFO 1",
    "random_2": "random LFO 2",
    "random_3": "random LFO 3",
    "random_4": "random LFO 4",
    "modulation": "mod matrix",
    "chorus": "chorus",
    "compressor": "compressor",
    "delay": "delay",
    "distortion": "distortion",
    "eq": "EQ",
    "flanger": "flanger",
    "phaser": "phaser",
    "reverb": "reverb",
    "voice": "voice",
    "pitch": "pitch",
    "stereo": "stereo",
    "volume": "volume",
    "velocity": "velocity",
    "portamento": "portamento",
    "legato": "legato",
    "polyphony": "polyphony",
    "beats": "sync",
    "oversampling": "oversampling",
    "effect": "effect chain",
    "bypass": "bypass",
    "mpe": "MPE",
    "macro": "macro",
}

# Rotating style hints for HYPOTHESIS instructions — indexed by (step_num - 1) % 4.
# Each hint steers the model toward a different sentence-opening pattern AND explicitly
# asks the model to reason about {remaining} — the subsystems that still need to change
# according to the GT-preset gap. This prevents both uniform phrasing and the failure mode
# where the model only discusses what was just applied rather than what still needs fixing.
_HYPOTHESIS_STYLE_HINTS = (
    "Use hedged language (likely/may/could). Explain why {remaining} haven't yet matched "
    "the target, referencing {primary} as the most likely cause.",
    "Begin with the acoustic evidence you hear that indicates {remaining} still need "
    "adjustment (e.g. 'The [X] character in the audio suggests {remaining} are still off '). "
    "Then name the synthesis mechanism responsible.",
    "Begin with the synthesis mechanism directly: why {primary} alone isn't enough and "
    "what {remaining} still need to do to close the remaining timbral gap.",
    "Begin with what the previous step failed to fix (e.g. 'Despite [what last step "
    "addressed], {remaining} still cause [perceptual problem] because...').",
)


def _extract_top_remaining(remaining_delta_context: str | None, n: int = 2) -> str:
    """Pull the top N subsystem names out of a remaining_delta_context string.

    Input:  "compressor (12 params), chorus (11 params), EQ (10 params), ..."
    Output: "compressor and chorus"   (n=2)

    Returns a generic fallback when context is absent or already converged.
    """
    import re as _re
    if not remaining_delta_context or remaining_delta_context.startswith("none"):
        return "the remaining parameters"
    names = _re.findall(r"([A-Za-z0-9 ]+?)\s*\(\d+\s*param", remaining_delta_context)
    top = [nm.strip() for nm in names[:n]]
    if not top:
        return "the remaining parameters"
    if len(top) == 1:
        return top[0]
    return f"{top[0]} and {top[1]}"


def _get_param_subsystem(key: str) -> str:
    """Return the subsystem key used for grouping (e.g. 'osc_1', 'filter_1', 'compressor')."""
    parts = key.split("_")
    if len(parts) >= 2 and parts[1].isdigit():
        if parts[0] == "modulation":
            return "modulation"  # collapse all mod-matrix slots into one group
        return f"{parts[0]}_{parts[1]}"
    return parts[0]


def _group_params_for_plan(params_delta: list[dict]) -> str:
    """Convert params_delta to a natural-language string for PLAN prefix seeding.

    ≤5 params  → enumerate all display names.
    >5 params  → group by subsystem; small groups name the attributes, large groups
                 say "N subsystem parameters".
    """
    if not params_delta:
        return "these parameters"

    if len(params_delta) <= 5:
        names = [_json_key_to_display(d["name"]) for d in params_delta]
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    # Group by subsystem, preserving insertion order
    groups: dict[str, list[str]] = {}
    for d in params_delta:
        sub = _get_param_subsystem(d["name"])
        groups.setdefault(sub, []).append(d["name"])

    parts: list[str] = []
    for sub, keys in groups.items():
        label = _SUBSYSTEM_DISPLAY.get(sub, sub.replace("_", " "))
        n = len(keys)
        if n == 1:
            parts.append(_json_key_to_display(keys[0]))
        elif n <= 3:
            # Strip subsystem prefix to get just the attribute token(s)
            prefix = sub + "_"
            attrs = []
            for k in keys:
                attr_key = k[len(prefix):] if k.startswith(prefix) else k
                attr_disp = _json_key_to_display(attr_key).lower()
                if attr_disp == "on":   # boolean enable/disable param
                    attr_disp = "on/off"
                attrs.append(attr_disp)
            if len(attrs) == 2:
                parts.append(f"{label} {attrs[0]} and {attrs[1]}")
            else:
                parts.append(f"{label} {', '.join(attrs[:-1])}, and {attrs[-1]}")
        else:
            parts.append(f"{n} {label} parameters")

    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def _extract_hypothesis(commentary: str) -> str | None:
    """Extract the HYPOTHESIS section from a step commentary string.

    Returns the raw HYPOTHESIS text (1-2 sentences), or None if not found.
    """
    import re
    m = re.search(
        r"\bHYPOTHESIS[:\s]+(.+?)(?=\n\s*\nPLAN[:\s]|\Z)",
        commentary,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


def _extract_heard(commentary: str) -> str | None:
    """Extract the HEARD section text from a step commentary string.

    Returns the raw HEARD text (1-2 sentences), or None if not found.
    """
    import re
    m = re.search(
        r"\bHEARD[:\s]+(.+?)(?=\n\s*\nHYPOTHESIS[:\s]|\Z)",
        commentary,
        re.DOTALL,
    )
    if m:
        return m.group(1).strip()
    return None


def _step_remaining_gap(target_preset: dict, step: dict) -> dict | None:
    """Compute remaining parameter gap between a step's cumulative preset and GT preset.

    Uses compare_preset_path from path_gen (same normalisation logic as the path builder)
    so the "remaining" signal is grounded in the same parameter-space truth as the path.

    Returns {"n_remaining": int, "by_subsystem": dict[str, int], "context_str": str}
    or None if no cumulative preset is available.
    """
    from maestro.synth.path_gen import compare_preset_path

    cumulative = step.get("cumulative_preset")
    if cumulative is None and step.get("cumulative_preset_path"):
        with open(step["cumulative_preset_path"]) as f:
            cumulative = json.load(f)
    if cumulative is None:
        return None

    synthetic = {
        "target_preset": target_preset,
        "iterations": [{"cumulative_preset": cumulative}],
    }
    try:
        cmp = compare_preset_path(synthetic)
    except Exception:
        return None

    noisy = cmp["noisy"]
    by_subsystem: dict[str, int] = {}
    for p in noisy:
        sub = _get_param_subsystem(p["name"])
        display = _SUBSYSTEM_DISPLAY.get(sub, sub.replace("_", " "))
        by_subsystem[display] = by_subsystem.get(display, 0) + 1

    sorted_subs = sorted(by_subsystem.items(), key=lambda x: -x[1])

    if not sorted_subs:
        context_str = "none — preset converged to target"
    else:
        parts = [f"{s} ({n} param{'s' if n > 1 else ''})" for s, n in sorted_subs[:4]]
        context_str = ", ".join(parts)
        if len(sorted_subs) > 4:
            extra = sum(v for _, v in sorted_subs[4:])
            context_str += f", plus {extra} params across other subsystems"

    return {
        "n_remaining": len(noisy),
        "by_subsystem": dict(sorted_subs),
        "context_str": context_str,
    }


def _json_key_to_display(key: str) -> str:
    """Convert Vital JSON key style (e.g. 'filter_1_cutoff') to display words."""
    parts = key.split("_")
    words = []
    for p in parts:
        low = p.lower()
        if low in _PARAM_KEY_ABBREVS:
            words.append(_PARAM_KEY_ABBREVS[low].capitalize() if low not in ("lfo", "eq") else low.upper())
        elif p.isdigit():
            words.append(p)
        else:
            words.append(p.capitalize())
    return " ".join(words)


def _param_label(name: str) -> str:
    for prefix, label in _PARAM_LABELS.items():
        if name.startswith(prefix):
            return label
    return "synth parameter"


def _format_delta_context(params_delta: list, is_mistake_step: bool) -> str:
    """Format params_delta list into a natural-language string for the Omni prompt."""
    if not params_delta:
        return ""
    lines = []
    mistake_params = []
    for d in params_delta:
        name = d["name"]
        display = _json_key_to_display(name)
        from_n = float(d.get("from_norm", 0))
        to_n = float(d.get("to_norm", 0))
        label = _param_label(name)
        magnitude = abs(to_n - from_n)
        if magnitude < 0.01:
            direction = "unchanged"
        elif to_n > from_n:
            direction = "increased"
        else:
            direction = "decreased"
        mag_str = "slightly" if magnitude < 0.15 else ("significantly" if magnitude > 0.35 else "moderately")
        if d.get("mistake"):
            lines.append(
                f"- {display} ({label}): {direction} {mag_str} [{from_n:.2f}\u2192{to_n:.2f}] \u26a0 moved away from target"
            )
            mistake_params.append(display)
        else:
            lines.append(f"- {display} ({label}): {direction} {mag_str} [{from_n:.2f}\u2192{to_n:.2f}]")
    context = "Parameters being changed in this step:\n" + "\n".join(lines)
    if mistake_params:
        context += (
            f"\n\nNote: {', '.join(mistake_params)} moved in the wrong direction (overcorrection). "
            "The description should acknowledge this mistake and what needs to be fixed next."
        )
    return context


def _build_param_summary(params_delta: list[dict]) -> str:
    """One-line step summary: top 4 changes by magnitude with ↑/↓ arrows."""
    if not params_delta:
        return "(no changes)"
    top = sorted(params_delta, key=lambda d: abs(d.get("to_norm", 0) - d.get("from_norm", 0)), reverse=True)[:4]
    parts = []
    for d in top:
        arrow = "\u2191" if d.get("to_norm", 0) > d.get("from_norm", 0) else "\u2193"
        parts.append(f"{_json_key_to_display(d['name'])} {arrow} {abs(d.get('to_norm', 0) - d.get('from_norm', 0)):.2f}")
    suffix = f" (+{len(params_delta) - 4} more)" if len(params_delta) > 4 else ""
    return "; ".join(parts) + suffix


CLAP_IMPERCEPTIBLE_THRESHOLD = 0.05  # Steps with CLAP delta below this are perceptually silent.

_TOOL_SPECS = json.dumps(
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
                "name": "spawn_search_agents",
                "description": "Fan out disjoint candidate shards to search workers.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "collect_search_reports",
                "description": "Collect proposals from search workers.",
                "parameters": {"type": "object"},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "judge_candidates",
                "description": "Rank and select up to K candidate IDs in token space.",
                "parameters": {"type": "object"},
            },
        },
    ],
    ensure_ascii=False,
)


def _build_search_reports(
    shards: list[list[str]],
    id_map: dict[str, str],
    gt_names: set[str],
    clap_scores: dict[str, float],
    proposals_per_agent: int,
) -> list[dict]:
    reports: list[dict] = []
    for i, shard in enumerate(shards, start=1):
        if not shard:
            continue
        ranked = sorted(
            shard,
            key=lambda n: (1 if n in gt_names else 0, float(clap_scores.get(n, 0.0))),
            reverse=True,
        )
        selected = ranked[: max(1, int(proposals_per_agent))]
        proposals = []
        for n in selected:
            proposals.append(
                {
                    "candidate_id": id_map[n],
                    "wavetable_name": n,
                    "confidence": round(0.9 if n in gt_names else max(0.35, min(0.88, clap_scores[n])), 3),
                    "reason": (
                        "Likely core source candidate from harmonic structure."
                        if n in gt_names
                        else "Possible match in harmonic envelope and brightness."
                    ),
                }
            )
        reports.append({"agent_id": f"sa_{i}", "proposals": proposals})
    return reports


def _build_judge_result(
    candidate_names: list[str],
    id_map: dict[str, str],
    gt_names: set[str],
    clap_scores: dict[str, float],
    select_k: int,
) -> dict:
    ranked = sorted(
        candidate_names,
        key=lambda n: (1 if n in gt_names else 0, float(clap_scores.get(n, 0.0))),
        reverse=True,
    )
    ranking = [id_map[n] for n in ranked]
    selected = ranking[: max(1, int(select_k))]
    score_map = {id_map[n]: float(clap_scores.get(n, 0.0)) for n in candidate_names}
    gt_ids = [id_map[n] for n in candidate_names if n in gt_names]
    top_scored = sorted(score_map.items(), key=lambda kv: kv[1], reverse=True)
    score_summary = ", ".join(f"{cid} ({s:.3f})" for cid, s in top_scored[:4])
    gt_note = f" GT candidate: {gt_ids[0]}." if gt_ids else ""
    reason = (
        f"Ranked by CLAP similarity to target. "
        f"Top scores: {score_summary}.{gt_note} "
        f"Selected top {len(selected)}."
    )
    return {
        "ranking": ranking,
        "selected": selected,
        "reason": reason,
    }


def _wrap_as_bash(python_code: str) -> str:
    """Wrap bare Python code in a shell-executable heredoc."""
    stripped = python_code.strip()
    if stripped.startswith("python"):
        return stripped  # already a shell command
    return f"python - <<'PY'\n{stripped}\nPY"


def _step_commentary_fallback(step: dict, step_num: int) -> str:
    keyword = str(step.get("search_keyword") or "target controls")
    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    return (
        f"HEARD: Step {step_num} still has mismatch in tone and movement.\n\n"
        f"HYPOTHESIS: Primary mismatch is in {primary}, with possible interaction from {support}.\n\n"
        f"PLAN: Apply the programmed updates for {keyword}, then listen again."
    )


def _call_omni_commentary(
    gt_wav: str,
    iter_wav: str | None,
    step: dict,
    step_num: int,
    archetype: str,
    omni_server: str,
    model: str,
    prev_commentary: str | None = None,
    params_delta: list | None = None,
    is_mistake_step: bool = False,
    allowed_params: list | None = None,
    remaining_delta_context: str | None = None,
    prior_iter_wav: str | None = None,  # unused in single-call mode
    clap_delta: float | None = None,  # unused in single-call mode
    is_planning_step: bool = False,
    planner_stage: str | None = None,
) -> str:
    """Call the hosted Omni model for grounded HEARD/HYPOTHESIS/PLAN commentary.
    Falls back to the template string if the server is unavailable."""
    import httpx

    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    keyword = str(step.get("search_keyword") or "target controls")

    delta_ctx = _format_delta_context(params_delta or [], is_mistake_step)
    param_summary = _build_param_summary(params_delta or [])
    allowed_str = ", ".join(allowed_params) if allowed_params else "all"
    mistake_note = (
        "MISTAKE STEP: these changes were intentional regressions for training robustness.\n"
        if is_mistake_step else ""
    )
    remaining_str = (
        f"Still needs to change: {remaining_delta_context}\n"
        if remaining_delta_context else ""
    )

    # Planning steps: text-only, PLAN section only — no HEARD/HYPOTHESIS, no audio.
    if is_planning_step:
        if params_delta:
            params_str = _group_params_for_plan(params_delta)
            plan_sentence1 = f"Adjusting {params_str}."
        else:
            plan_sentence1 = "Applying parameter updates."
        planning_prompt = (
            f"You are a music production AI agent. Step {step_num} applies parameter changes "
            f"that produce no audible difference yet — the timbral effect will become perceptible "
            f"after additional parameters are adjusted.\n\n"
            f"Parameter changes: {param_summary}\n"
            f"{delta_ctx}\n"
            f"{remaining_str}"
            f"Write a PLAN section only (2 sentences):\n"
            f"Sentence 1 — copy this exactly: \"{plan_sentence1}\"\n"
            f"Sentence 2 — one sentence explaining how these structural changes set up the "
            f"next audible step toward the target sound.\n\n"
            f"Output only: PLAN: <sentence 1> <sentence 2>"
        )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": planning_prompt}],
            "max_tokens": 120,
            "temperature": 0.4,
        }
        try:
            with httpx.Client() as client:
                resp = client.post(
                    f"{omni_server}/v1/chat/completions", json=payload, timeout=60.0
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return f"PLAN: {plan_sentence1} These precision adjustments prepare the preset for the next audible improvement step."

    with open(gt_wav, "rb") as f:
        gt_b64 = base64.b64encode(f.read()).decode()

    content: list[dict] = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}}
    ]
    if iter_wav:
        with open(iter_wav, "rb") as f:
            iter_b64 = base64.b64encode(f.read()).decode()
        content.append({"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}})
        prev_context = (
            f"\nPrevious step assessment:\n{prev_commentary}\n"
            if prev_commentary else ""
        )
        prompt = (
            f"You are a music production AI agent comparing synthesizer audio.\n"
            f"AUDIO A is the target {archetype} sound. AUDIO B is the current preset (after step {step_num - 1}).\n"
            f"Step {step_num} summary: {param_summary}\n"
            f"\n{delta_ctx}\n"
            f"Allowed parameters this step: {allowed_str}\n"
            f"{mistake_note}"
            f"{remaining_str}"
            f"{prev_context}\n"
            f"Write exactly 3 short sections, each 1-2 concise sentences:\n"
            f"HEARD: concrete perceptual differences between A and B now — reference what changed vs previous step if relevant.\n"
            f"HYPOTHESIS: likely synthesis causes — use words like likely/may/could.\n"
            f"PLAN: what this next parameter update ({keyword}) targets; "
            f"reference {primary} controls specifically. PLAN must only reference parameters listed above.\n\n"
            f"Use natural control names only (no snake_case). Archetype: {archetype}. "
            f"Support family: {support}."
        )
    else:
        prompt = (
            f"You are a music production AI agent. You just heard a target {archetype} sound.\n\n"
            f"Step {step_num} summary: {param_summary}\n"
            f"\n{delta_ctx}\n"
            f"Allowed parameters this step: {allowed_str}\n"
            f"Write exactly 3 short sections, each 1-2 concise sentences:\n"
            f"HEARD: describe the timbre, texture, and movement you hear.\n"
            f"HYPOTHESIS: what synthesis elements likely produce this sound.\n"
            f"PLAN: how to start recreating it — begin with {primary} controls. "
            f"PLAN must only reference parameters listed above.\n\n"
            f"Use natural control names only (no snake_case). Archetype: {archetype}."
        )
    content.append({"type": "text", "text": prompt})

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 600,
        "temperature": 0.5,
    }
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{omni_server}/v1/chat/completions", json=payload, timeout=90.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return _step_commentary_fallback(step, step_num)


def _call_omni_commentary_two_stage(
    gt_wav: str,
    iter_wav: str | None,
    step: dict,
    step_num: int,
    archetype: str,
    omni_server: str,
    omni_model: str,
    stage2_server: str,
    stage2_model: str,
    prev_commentary: str | None = None,
    params_delta: list | None = None,
    is_mistake_step: bool = False,
    allowed_params: list | None = None,
    remaining_delta_context: str | None = None,
    prior_iter_wav: str | None = None,
    clap_delta: float | None = None,
    planner_stage: str | None = None,
    is_planning_step: bool = False,
) -> str:
    """Two-stage commentary: Stage 1 Omni describes audio only; Stage 2 text-only
    model integrates the description with params_delta to produce HEARD/HYPOTHESIS/PLAN.

    Stage 1 uses three clips when prior_iter_wav is available:
      - 1A: prior_iter_wav → iter_wav (what changed in the last step)
      - 1B: iter_wav → gt_wav (remaining gap to target)

    When clap_delta < CLAP_IMPERCEPTIBLE_THRESHOLD the step is perceptually silent;
    Stage 1 is skipped entirely and a template acknowledgment is used instead.

    Falls back to the single-call version if either stage fails.
    """
    import httpx

    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    keyword = str(step.get("search_keyword") or "target controls")
    delta_ctx = _format_delta_context(params_delta or [], is_mistake_step)
    param_summary = _build_param_summary(params_delta or [])
    allowed_str = ", ".join(allowed_params) if allowed_params else "all"
    mistake_note = (
        "MISTAKE STEP: these changes were intentional regressions for training robustness.\n"
        if is_mistake_step else ""
    )
    remaining_str = (
        f"Still needs to change: {remaining_delta_context}\n"
        if remaining_delta_context else ""
    )

    # Planning steps: text-only Stage 2 call — PLAN section only, no HEARD/HYPOTHESIS, no audio.
    if is_planning_step:
        if params_delta:
            params_str = _group_params_for_plan(params_delta)
            plan_sentence1 = f"Adjusting {params_str}."
        else:
            plan_sentence1 = "Applying parameter updates."
        planning_prompt = (
            f"You are a music production AI agent. Step {step_num} applies parameter changes "
            f"that produce no audible difference yet — the timbral effect will become perceptible "
            f"after additional parameters are adjusted.\n\n"
            f"Parameter changes: {param_summary}\n"
            f"{delta_ctx}\n"
            f"{remaining_str}"
            f"Write a PLAN section only (2 sentences):\n"
            f"Sentence 1 — copy this exactly: \"{plan_sentence1}\"\n"
            f"Sentence 2 — one sentence explaining how these structural changes set up the "
            f"next audible step toward the target {primary} sound.\n\n"
            f"Output only: PLAN: <sentence 1> <sentence 2>"
        )
        s2_payload = {
            "model": stage2_model,
            "messages": [{"role": "user", "content": planning_prompt}],
            "max_tokens": 120,
            "temperature": 0.4,
        }
        try:
            with httpx.Client() as client:
                s2_resp = client.post(
                    f"{stage2_server}/v1/chat/completions", json=s2_payload, timeout=60.0
                )
                s2_resp.raise_for_status()
                return s2_resp.json()["choices"][0]["message"]["content"].strip()
        except Exception:
            return f"PLAN: {plan_sentence1} These precision adjustments prepare the preset for the next audible improvement step."

    # --- Stage 1: audio-only perceptual description ---
    # Outputs two separate observations that Stage 2 will use distinctly:
    #   step_change_obs  — what the previous step achieved (from Stage 1A, 3-clip only)
    #   remaining_gap_obs — what still differs from the target (from Stage 1B or GT-only)
    #
    # Note: planning steps (is_planning_step=True) already returned early above, so by the time
    # we reach here, the step is a genuine listening step — audio gate has confirmed |delta| is
    # large enough to hear. Disable the old imperceptible skip so Stage 1 always runs for
    # listening steps and the model actually describes what it hears.
    is_imperceptible = False

    step_change_obs: str | None = None
    remaining_gap_obs: str | None = None

    if is_imperceptible:
        param_cat = _param_label(params_delta[0]["name"]) if params_delta else "synth"
        step_change_obs = (
            f"The previous edit produced no audible difference — CLAP perceptual distance was "
            f"below threshold. The {param_cat} adjustments are sub-perceptual precision refinements."
        )
    else:
        with open(gt_wav, "rb") as f:
            gt_b64 = base64.b64encode(f.read()).decode()

        if iter_wav and prior_iter_wav:
            # 3-clip: Stage 1A (prior→current) + Stage 1B (current→target).
            with open(prior_iter_wav, "rb") as f:
                prior_b64 = base64.b64encode(f.read()).decode()
            with open(iter_wav, "rb") as f:
                iter_b64 = base64.b64encode(f.read()).decode()

            s1a_content: list[dict] = [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{prior_b64}"}},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}},
                {"type": "text", "text": (
                    "You are a music production AI. Listen to two synthesizer clips.\n"
                    "AUDIO A: the preset BEFORE the last parameter edit.\n"
                    "AUDIO B: the preset AFTER the last parameter edit.\n\n"
                    "Describe ONLY what changed between A and B: be specific about frequency region "
                    "(low/mid/high), envelope shape (attack/decay/sustain), harmonic character "
                    "(brightness, buzz, shimmer), or modulation movement.\n"
                    "2-3 sentences. No recommendations. No 'lacks' — describe what IS different."
                )},
            ]
            s1b_content: list[dict] = [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}},
                {"type": "text", "text": (
                    f"You are a music production AI. Listen to two synthesizer clips.\n"
                    f"AUDIO A: the current {archetype} preset (after step {step_num - 1}).\n"
                    f"AUDIO B: the target sound to match.\n\n"
                    "Describe the remaining timbral gap: what specific qualities does the target have "
                    "that the current preset does not? Focus on the most prominent differences in "
                    "frequency balance, harmonic density, envelope shape, or modulation.\n"
                    "2-3 sentences. No recommendations."
                )},
            ]
            try:
                with httpx.Client() as client:
                    s1a_resp = client.post(
                        f"{omni_server}/v1/chat/completions",
                        json={"model": omni_model, "messages": [{"role": "user", "content": s1a_content}],
                              "max_tokens": 160, "temperature": 0.4},
                        timeout=90.0,
                    )
                    s1a_resp.raise_for_status()
                    step_change_obs = s1a_resp.json()["choices"][0]["message"]["content"].strip()

                    s1b_resp = client.post(
                        f"{omni_server}/v1/chat/completions",
                        json={"model": omni_model, "messages": [{"role": "user", "content": s1b_content}],
                              "max_tokens": 160, "temperature": 0.4},
                        timeout=90.0,
                    )
                    s1b_resp.raise_for_status()
                    remaining_gap_obs = s1b_resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                return _step_commentary_fallback(step, step_num)

        elif iter_wav:
            # Step 1 or no prior: only current→target gap available.
            with open(iter_wav, "rb") as f:
                iter_b64 = base64.b64encode(f.read()).decode()
            s1_content: list[dict] = [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}},
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}},
                {"type": "text", "text": (
                    f"You are a music production AI. Listen to two synthesizer clips.\n"
                    f"AUDIO A: the current {archetype} preset (starting point).\n"
                    f"AUDIO B: the target sound to match.\n\n"
                    "Describe the specific timbral gap: what qualities does the target have that the "
                    "current preset does not? Be precise about frequency balance, harmonic texture, "
                    "envelope character, and any modulation or movement.\n"
                    "3-4 sentences. No recommendations."
                )},
            ]
            try:
                with httpx.Client() as client:
                    s1_resp = client.post(
                        f"{omni_server}/v1/chat/completions",
                        json={"model": omni_model, "messages": [{"role": "user", "content": s1_content}],
                              "max_tokens": 220, "temperature": 0.4},
                        timeout=90.0,
                    )
                    s1_resp.raise_for_status()
                    remaining_gap_obs = s1_resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                return _step_commentary_fallback(step, step_num)

        else:
            # No iter clip — GT description only (first step, no default preset rendered yet).
            s1_content = [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}},
                {"type": "text", "text": (
                    f"You are a music production AI. Listen to this target {archetype} synthesizer sound.\n\n"
                    "Describe what you hear: be specific about frequency balance (bright/warm/dark), "
                    "harmonic character (clean/buzzy/rich), envelope shape (sharp/slow attack, long/short decay), "
                    "and any movement or modulation.\n"
                    "3-4 sentences. No recommendations."
                )},
            ]
            try:
                with httpx.Client() as client:
                    s1_resp = client.post(
                        f"{omni_server}/v1/chat/completions",
                        json={"model": omni_model, "messages": [{"role": "user", "content": s1_content}],
                              "max_tokens": 220, "temperature": 0.4},
                        timeout=90.0,
                    )
                    s1_resp.raise_for_status()
                    remaining_gap_obs = s1_resp.json()["choices"][0]["message"]["content"].strip()
            except Exception:
                return _step_commentary_fallback(step, step_num)

    # --- Stage 2: text-only integration ---
    # Build the observation block, keeping step_change and remaining_gap clearly labelled
    # so Stage 2 can reference each in the appropriate sentence of HEARD.
    prior_heard = _extract_heard(prev_commentary) if prev_commentary else None

    obs_block = ""
    if step_change_obs and remaining_gap_obs:
        obs_block = (
            f"[WHAT THE LAST STEP CHANGED]\n{step_change_obs}\n\n"
            f"[REMAINING GAP TO TARGET]\n{remaining_gap_obs}"
        )
        heard_base = (
            "HEARD (2 sentences): Sentence 1 describes what the previous step changed, "
            "drawn from [WHAT THE LAST STEP CHANGED]. "
            "Sentence 2 states the most important remaining timbral gap, "
            "drawn from [REMAINING GAP TO TARGET]. "
            "Use specific sonic language (e.g. 'the attack softened', 'high-mid shimmer increased', "
            "'filter cutoff moved lower') — avoid generic phrases like 'lacks texture' or 'still missing'."
        )
        if prior_heard:
            heard_instruction = (
                heard_base + " "
                f"The previous step's HEARD began: \"{prior_heard[:180]}\" "
                f"— Sentence 1 must describe something DIFFERENT from this prior observation."
            )
        else:
            heard_instruction = heard_base
    elif step_change_obs:
        # Imperceptible step — only the sub-threshold note, no remaining gap measured.
        obs_block = f"[PERCEPTUAL OBSERVATION]\n{step_change_obs}"
        heard_instruction = (
            "HEARD (1-2 sentences): Acknowledge that this step produced no audible change — "
            "the edit is a sub-perceptual precision adjustment. Explain briefly why these small "
            "changes still matter for convergence toward the target."
        )
    else:
        # First step or no prior — only remaining gap available.
        obs_block = f"[REMAINING GAP TO TARGET]\n{remaining_gap_obs}"
        heard_instruction = (
            "HEARD (1-2 sentences): Describe the most prominent timbral gap between the current "
            "preset and the target, drawn from [REMAINING GAP TO TARGET]. "
            "Use specific sonic language — avoid generic phrases like 'lacks texture'."
        )

    prev_context = (
        f"Previous step assessment:\n{prev_commentary}\n\n"
        if prev_commentary else ""
    )
    mistake_instruction = (
        "NOTE: This step contains intentional regression moves for training robustness — "
        "HEARD should note that the previous step moved away from target.\n"
        if is_mistake_step else ""
    )

    # PLAN: 2-sentence format — Sentence 1 is the pre-seeded param inventory (machine-verifiable),
    # Sentence 2 is the model's rationale (the actual training signal).
    # Splitting them keeps the "why" uncluttered by the param list.
    if params_delta:
        params_str = _group_params_for_plan(params_delta)
        plan_instruction = (
            f"PLAN (2 sentences): "
            f"Sentence 1 — copy this exactly: \"Adjusting {params_str}.\" "
            f"Sentence 2 — one sentence explaining how these changes address the remaining "
            f"timbral gap. Do not name any parameter in Sentence 2."
        )
    else:
        plan_instruction = (
            "PLAN (1 sentence): Explain the rationale for the parameter adjustments listed above."
        )

    # HYPOTHESIS: rotate style hints by step_num so consecutive steps use different
    # sentence-opening patterns. Each hint embeds {remaining} — the top subsystems that
    # still differ from the GT preset — so the model reasons about what still needs fixing,
    # not just what was applied in the current step.
    remaining_top = _extract_top_remaining(remaining_delta_context)
    style_hint = _HYPOTHESIS_STYLE_HINTS[(step_num - 1) % len(_HYPOTHESIS_STYLE_HINTS)].format(
        primary=primary, support=support, remaining=remaining_top
    )

    # When a prior hypothesis exists, require revision rather than restatement.
    prior_hypothesis = _extract_hypothesis(prev_commentary) if prev_commentary else None
    if planner_stage == "correction":
        # Correction steps fix params that were deliberately set wrong in an earlier step.
        # Hypothesis must: (1) identify what was wrong from audio evidence, (2) explain the
        # correction, and (3) note what timbral gap still remains after this fix.
        correction_params = [p["name"].replace("_", " ") for p in (params_delta or [])[:3]]
        correction_str = ", ".join(correction_params) if correction_params else "earlier parameters"
        hypothesis_instruction = (
            f"HYPOTHESIS (2 sentences): Sentence 1 — based on what you hear, explain what was "
            f"wrong with the earlier {correction_str} settings (too high, too low, wrong "
            f"direction) and why correcting them helps close the gap. "
            f"Sentence 2 — describe what timbral difference still remains between the current "
            f"render and the target after this correction, referencing {remaining_top}."
        )
    elif prior_hypothesis:
        hypothesis_instruction = (
            f"HYPOTHESIS (1-2 sentences): Revise your synthesis hypothesis in light of what "
            f"this step's audio just revealed. The prior hypothesis was: "
            f"\"{prior_hypothesis[:280]}\" "
            f"— your new hypothesis must either update what still needs to change or identify "
            f"a new cause for the remaining gap. Do not simply restate the prior hypothesis. "
            f"{style_hint}"
        )
    else:
        hypothesis_instruction = (
            f"HYPOTHESIS (1-2 sentences): Describe the most likely synthesis causes for the "
            f"remaining timbral gap. {style_hint}"
        )

    s2_prompt = (
        f"You are a music production AI agent writing step {step_num} analysis for recreating "
        f"a {archetype} synth sound.\n\n"
        f"--- Perceptual observations from audio ---\n{obs_block}\n\n"
        f"--- Parameter changes for step {step_num} ---\n"
        f"Summary: {param_summary}\n"
        f"{delta_ctx}\n"
        f"{mistake_instruction}"
        f"{remaining_str}"
        f"{prev_context}"
        f"Write exactly 3 sections (no markdown bold, no headers other than the labels below):\n\n"
        f"{heard_instruction}\n\n"
        f"{hypothesis_instruction}\n\n"
        f"{plan_instruction}\n\n"
        f"Archetype: {archetype}. Be concise and specific."
    )

    s2_payload = {
        "model": stage2_model,
        "messages": [{"role": "user", "content": s2_prompt}],
        "max_tokens": 400,
        "temperature": 0.4,
    }

    fallback_obs = step_change_obs or remaining_gap_obs or "No audio observation available."
    try:
        with httpx.Client() as client:
            s2_resp = client.post(
                f"{stage2_server}/v1/chat/completions", json=s2_payload, timeout=90.0
            )
            s2_resp.raise_for_status()
            return s2_resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        return (
            f"HEARD: {fallback_obs[:200]}\n\n"
            f"HYPOTHESIS: Likely caused by {primary} parameter interactions.\n\n"
            f"PLAN: Apply the programmed {keyword} updates."
        )


def _call_omni_closing_eval(
    gt_wav: str,
    final_iter_wav: str,
    archetype: str,
    omni_server: str,
    model: str,
    convergence: dict | None = None,
) -> str:
    """Final A/B eval after all iteration steps.

    convergence — output of _step_remaining_gap on the final step, or None.
      {"n_remaining": int, "by_subsystem": dict, "context_str": str}

    The prompt branches on convergence status so the agent's verdict is grounded
    in parameter-space truth rather than free-form speculation.
    """
    import httpx

    with open(gt_wav, "rb") as f:
        gt_b64 = base64.b64encode(f.read()).decode()
    with open(final_iter_wav, "rb") as f:
        iter_b64 = base64.b64encode(f.read()).decode()

    if convergence is None:
        verdict_instruction = (
            "Write 2-3 sentences: what aspects now match well, and what still differs most. "
            "Be specific about timbre, texture, or envelope."
        )
        verdict_label = "FINAL ASSESSMENT:"
    elif convergence.get("converged"):
        verdict_instruction = (
            "All planned iterations have been applied. "
            "Write 2 sentences: honestly describe how close the final render is to the target "
            "(note any timbral, textural, or envelope differences you can still hear), "
            "then confirm the iteration path is complete even if a small gap remains."
        )
        verdict_label = "FINAL ASSESSMENT (complete):"
    else:
        n = convergence.get("n_remaining", 0)
        subs = list(convergence.get("by_subsystem", {}).keys())[:3]
        subs_str = ", ".join(subs) if subs else "several subsystems"
        verdict_instruction = (
            f"The iteration budget is exhausted before the path completed. "
            f"{n} parameters in {subs_str} still differ from the target preset. "
            "Write 2 sentences: describe what timbral qualities were successfully recreated, "
            "then describe the most significant remaining gap."
        )
        verdict_label = "FINAL ASSESSMENT (budget exhausted):"

    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}},
        {"type": "text", "text": (
            f"You are a music production AI agent doing a final review.\n"
            f"AUDIO A is the target {archetype} sound. AUDIO B is the final recreated preset.\n\n"
            f"{verdict_instruction}\n"
            f"Begin your response with \"{verdict_label}\""
        )},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200,
        "temperature": 0.4,
    }
    try:
        with httpx.Client() as client:
            resp = client.post(
                f"{omni_server}/v1/chat/completions", json=payload, timeout=90.0
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        if convergence and convergence.get("converged"):
            return "FINAL ASSESSMENT (complete): All planned iterations have been applied. The recreation path is complete."
        elif convergence:
            subs = list(convergence.get("by_subsystem", {}).keys())[:2]
            subs_str = ", ".join(subs) if subs else "several parameters"
            return f"FINAL ASSESSMENT (budget exhausted): Iterations exhausted — {subs_str} still differ from the target."
        return "Recreation pass complete."


def _build_listen_probe_command(audio_path: Path) -> str:
    payload = {"path": str(audio_path)}
    return (
        "python - <<'PY'\n"
        "import json\n"
        "from pathlib import Path\n"
        "import soundfile as sf\n"
        f"payload = json.loads('''{json.dumps(payload, ensure_ascii=False)}''')\n"
        "p = Path(payload['path'])\n"
        "out = {'path': str(p), 'exists': p.exists()}\n"
        "if out['exists']:\n"
        "    try:\n"
        "        x, sr = sf.read(p, always_2d=True)\n"
        "        out['duration_s'] = round(float(len(x) / max(1, sr)), 4)\n"
        "    except Exception:\n"
        "        out['duration_s'] = None\n"
        "print(json.dumps({'listen_probe': out}, ensure_ascii=False))\n"
        "PY"
    )


def _tool_call(name: str, arguments: dict) -> dict:
    return {"role": "tool_call", "content": json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)}


def main() -> None:
    ap = argparse.ArgumentParser(description="Build main-agent SFT dataset v2 with search/judge hierarchy.")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--index-npy", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index.npz"))
    ap.add_argument("--index-meta", type=Path, default=Path("outputs/wt_retrieval_baseline/wt_index_meta.json"))
    ap.add_argument("--wavetable-lib", type=Path, default=Path("data/wavetable_lib.json"))
    ap.add_argument("--out-jsonl", required=True)
    ap.add_argument("--max-samples", type=int, default=256)
    ap.add_argument("--max-steps", type=int, default=6)

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

    ap.add_argument("--audio-gate-threshold", type=float, default=0.01,
        help="Absolute CLAP delta threshold. Steps with |delta| below this get PLAN-only turns "
             "(no audio listen). Set to 0 to disable gating (listen every step). Default 0.01.")
    ap.add_argument("--reanchor-gt-audio", action="store_true", default=False,
        help="Re-attach GT audio to the intro of each chunked block 2+. Disabled by default: "
             "the model is expected to reason from conversation context without re-listening to GT "
             "every block, which produces higher-quality training signal and saves ~3000 tokens per block.")

    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--clap-device", default="cuda:0")
    ap.add_argument("--omni-server", default="", help="Omni model server URL (empty = use template fallback).")
    ap.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of samples to process concurrently (default: 4). Omni calls are I/O-bound "
             "so threading scales well. Vita (probe rendering) and CLAP (GPU) are serialized "
             "internally via a shared lock; only Omni calls run in parallel.",
    )
    ap.add_argument(
        "--commentary-mode",
        choices=["single", "two_stage"],
        default="single",
        help="single: one Omni call with injected delta context; two_stage: audio-only Stage 1 + text-only Stage 2.",
    )
    ap.add_argument("--stage2-server", default="", help="Stage 2 text model server URL (defaults to --omni-server).")
    ap.add_argument("--stage2-model", default="", help="Stage 2 model name (defaults to --omni-model).")
    ap.add_argument(
        "--window-tokens", type=int, default=None,
        help="If set, split each conversation into blocks of at most this many tokens. "
             "Each block re-includes GT audio + current-state audio as context prefix. "
             "Requires soundfile + transformers tokenizer. Recommended: 12288.",
    )
    ap.add_argument(
        "--audio-token-rate", type=float, default=94.5,
        help="Tokens per second of audio for block budget estimation (default 94.5).",
    )
    args = ap.parse_args()

    stage2_server = args.stage2_server or args.omni_server
    stage2_model = args.stage2_model or args.omni_model

    entries = load_manifest_entries(Path(args.manifest), max_samples=args.max_samples)
    index_rows = load_index_rows(args.index_meta)
    selected_by_name = select_probe_rows_by_name(index_rows)
    universe_names = sorted(selected_by_name.keys(), key=lambda x: x.lower())
    wavetable_lib = load_wavetable_lib(args.wavetable_lib)
    embedder = ClapEmbedder.create(args.clap_device)
    shortlist_data = build_clap_shortlist_data(args.index_npy, index_rows)

    # Load tokenizer once if chunking is requested.
    _tok = None
    if args.window_tokens:
        from transformers import AutoTokenizer
        _tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Omni-7B", trust_remote_code=True)
        print(f"Chunking enabled: window_tokens={args.window_tokens}, tokenizer loaded.")

    # Shared mutable probe cache. Serialized via _serial_lock.
    candidate_audio: dict[str, Path] = {}

    # Serializes vita (probe rendering) and CLAP (GPU) calls. Omni HTTP calls run freely
    # outside this lock — they are the dominant bottleneck and benefit from parallelism.
    _serial_lock = threading.Lock()

    def _estimate_block_tokens(messages: list[dict], audio_paths: list[str]) -> int:
        """Estimate total token count (text + audio) for a block."""
        import soundfile as _sf
        text = sum(
            len(_tok.encode(m["content"], add_special_tokens=False))
            for m in messages
            if isinstance(m.get("content"), str)
        )
        audio = 0
        for p in audio_paths:
            try:
                audio += int(_sf.info(p).duration * args.audio_token_rate)
            except Exception:
                pass
        return text + audio

    def _make_block_intro(
        current_state_audio: str,
        steps_completed: int,
        gt_audio_path: str | None = None,
    ) -> tuple[list[dict], list[str]]:
        """Build the 4-message intro for block 2+.

        Mirrors the existing baseline-listen pattern so the format validator passes:
          user → assistant → tool_call → tool_response (current state audio)

        When gt_audio_path is provided, the user message includes a GT audio re-anchor
        (<audio> tag) so the model can re-orient to the target. When None (default),
        the GT audio is omitted — the model is expected to reason from context alone,
        which produces higher-quality training signal and saves ~3000 tokens per block.
        """
        user_content = (
            f"{'<audio>' + chr(10) if gt_audio_path else ''}"
            f"{'Target sound re-established. ' if gt_audio_path else ''}"
            f"Current preset state after "
            f"{steps_completed} iteration{'s' if steps_completed != 1 else ''}. "
            f"Continue iterating toward the target."
        )
        messages = [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Listening to current preset state."},
            _tool_call("bash", {"command": _build_listen_probe_command(current_state_audio)}),
            {
                "role": "tool_response",
                "content": json.dumps(
                    {"status": "ok", "baseline_audio": "<audio>", "path": current_state_audio},
                    ensure_ascii=False,
                ),
            },
        ]
        audios = ([gt_audio_path] if gt_audio_path else []) + [current_state_audio]
        return messages, audios

    def _chunk_into_blocks(
        full_messages: list[dict],
        full_audios: list[str],
        preamble_msg_count: int,
        preamble_audio_count: int,
        step_boundaries: list[tuple[int, int, str | None]],
        closing_messages: list[dict],
        gt_audio_path: str,
        default_audio_path: str | None,
        base_record: dict,
        reanchor_gt_audio: bool = False,
    ) -> list[dict]:
        """Split a fully-built record into window_tokens-sized blocks.

        Each block gets current-state audio as a context prefix.
        When reanchor_gt_audio=True, GT audio is also re-attached to block 2+ intros.
        Block 1 includes the full search preamble; block 2+ get a synthetic intro.
        The closing user+assistant turns appear only in the final block.

        step_boundaries: list of (msg_start, msg_end, iter_audio_path) per step.
          iter_audio_path may be None if no iter audio was emitted for that step.
        """
        preamble_messages = full_messages[:preamble_msg_count]
        preamble_audios = full_audios[:preamble_audio_count]

        blocks: list[dict] = []
        block_idx = 0
        current_state_audio = default_audio_path  # updated after each block flush
        accumulated_msgs: list[dict] = []
        accumulated_audios: list[str] = []
        block_step_indices: list[int] = []  # which step indices are in current block
        steps_completed_before_block = 0

        def _flush_block(include_closing: bool) -> None:
            nonlocal block_idx, accumulated_msgs, accumulated_audios, block_step_indices
            nonlocal steps_completed_before_block, current_state_audio

            if block_idx == 0:
                intro_msgs = preamble_messages
                intro_audios = preamble_audios
            else:
                intro_msgs, intro_audios = _make_block_intro(
                    current_state_audio,
                    steps_completed_before_block,
                    gt_audio_path=gt_audio_path if reanchor_gt_audio else None,
                )

            block_messages = intro_msgs + accumulated_msgs
            block_audio_list = intro_audios + accumulated_audios

            if include_closing:
                block_messages = block_messages + closing_messages
            else:
                # Non-final blocks must end with an assistant message (format constraint).
                # Add a brief transition turn so the next block can start with a user message.
                block_messages = block_messages + [
                    {
                        "role": "assistant",
                        "content": "Iteration block complete. Continuing to the next set of edits.",
                    }
                ]

            # Build per-block step_labels from the base record's full step_labels
            full_step_labels = base_record["meta"]["step_labels"]
            block_step_labels = [full_step_labels[i] for i in block_step_indices]

            block_record = {
                "id": f"{base_record['id']}_block{block_idx}",
                "task_type": base_record["task_type"],
                "tools": base_record["tools"],
                "messages": block_messages,
                "audios": block_audio_list,
                "assets": base_record["assets"],
                "labels": base_record["labels"],
                "meta": {
                    **{k: v for k, v in base_record["meta"].items() if k != "step_labels"},
                    "block_idx": block_idx,
                    "total_blocks": -1,  # filled in after all blocks known
                    "source_sample_id": base_record["id"],
                    "step_labels": block_step_labels,
                },
            }
            assert_valid_ms_swift_multiturn_record(block_record)
            blocks.append(block_record)

            # Update state for next block
            steps_completed_before_block += len(block_step_indices)
            if block_step_indices:
                last_step_i = block_step_indices[-1]
                current_state_audio = step_boundaries[last_step_i][2] or current_state_audio
            block_idx += 1
            accumulated_msgs = []
            accumulated_audios = []
            block_step_indices = []

        for step_i, (msg_start, msg_end, iter_audio) in enumerate(step_boundaries):
            step_msgs = full_messages[msg_start:msg_end]
            step_audios = [iter_audio] if iter_audio else []

            # Decide context for projection
            if block_idx == 0:
                ctx_msgs = preamble_messages
                ctx_audios = preamble_audios
            else:
                ctx_msgs, ctx_audios = _make_block_intro(
                    gt_audio_path, current_state_audio, steps_completed_before_block
                )

            projected = _estimate_block_tokens(
                ctx_msgs + accumulated_msgs + step_msgs + closing_messages,
                ctx_audios + accumulated_audios + step_audios,
            )

            if projected > args.window_tokens and accumulated_msgs:
                # Current step would overflow — flush what we have, start fresh
                _flush_block(include_closing=False)

            accumulated_msgs.extend(step_msgs)
            accumulated_audios.extend(step_audios)
            block_step_indices.append(step_i)

        # Flush final block with closing turns
        _flush_block(include_closing=True)

        # Fill in total_blocks now that we know the count
        for blk in blocks:
            blk["meta"]["total_blocks"] = len(blocks)

        return blocks

    def _process_entry(entry: dict) -> list[dict] | None:
        """Build one or more SFT records. Returns None if the entry should be skipped."""
        sample_id = str(entry["sample_id"])
        target_audio_path = Path(entry.get("gt_wav") or entry.get("gt_probe_wav"))
        default_audio_path = Path(entry.get("default_wav")) if entry.get("default_wav") else None
        iter_wavs = [Path(p) for p in (entry.get("iter_wavs") or [])]

        with open(Path(entry["path_file"])) as f:
            path_data = json.load(f)
        if "target_preset_path" in path_data:
            gt_names = set(extract_gt_wavetable_names(Path(path_data["target_preset_path"])))
        elif "target_preset" in path_data:
            from scripts.build_wavetable_retrieval_baseline import _extract_gt_wavetable_names_from_preset_dict
            gt_names = set(_extract_gt_wavetable_names_from_preset_dict(path_data["target_preset"]))
        else:
            return None
        if not gt_names:
            return None

        # Serialize vita (probe rendering) and CLAP (GPU) — these are fast relative to
        # Omni calls. The per-step Omni calls below run outside this lock.
        with _serial_lock:
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

            clap_scores = {n: embedder.cosine_paths(candidate_audio[n], target_audio_path) for n in candidate_names}

            # Per-step CLAP scores: computed upfront (under serial lock) so we can gate
            # audio listening turns before the main step loop runs.
            step_clap_scores: dict[int, float] = {}
            default_clap_score: float | None = None
            if iter_wavs:
                if default_audio_path:
                    try:
                        default_clap_score = float(
                            embedder.cosine_paths(default_audio_path, target_audio_path)
                        )
                    except Exception:
                        pass
                for _step in path_data.get("iterations", [])[: int(args.max_steps)]:
                    _snum = int(_step.get("step", 0))
                    _idx = _snum - 1
                    if 0 <= _idx < len(iter_wavs):
                        try:
                            step_clap_scores[_snum] = float(
                                embedder.cosine_paths(iter_wavs[_idx], target_audio_path)
                            )
                        except Exception:
                            pass

            # |delta| per step vs the previous step's audio (or default for step 1).
            # Used to decide whether a step produced an audible change.
            step_clap_deltas: dict[int, float | None] = {}
            _prev_clap = default_clap_score
            for _step in path_data.get("iterations", [])[: int(args.max_steps)]:
                _snum = int(_step.get("step", 0))
                _cur = step_clap_scores.get(_snum)
                if _cur is not None and _prev_clap is not None:
                    step_clap_deltas[_snum] = abs(_cur - _prev_clap)
                _prev_clap = _cur if _cur is not None else _prev_clap

        id_map = {name: f"C{i}" for i, name in enumerate(candidate_names, start=1)}
        candidate_assets = [
            {"candidate_id": id_map[name], "wavetable_name": name, "audio_path": str(candidate_audio[name])}
            for name in candidate_names
        ]

        shards = build_disjoint_shards(candidate_names, args.num_agents)
        jobs = [
            {
                "agent_id": f"sa_{i}",
                "candidate_shard": [id_map[n] for n in shard],
                "seed": int(args.seed) + i,
            }
            for i, shard in enumerate(shards, start=1)
            if shard
        ]
        reports = _build_search_reports(shards, id_map, gt_names, clap_scores, args.proposals_per_agent)
        judge_result = _build_judge_result(candidate_names, id_map, gt_names, clap_scores, args.select_k)
        selected_ids = set(judge_result["selected"])
        selected_names = [name for name in candidate_names if id_map[name] in selected_ids]

        messages: list[dict] = []
        used_iter_audio_paths: list[str] = []
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<audio>\nRecreate this {entry.get('archetype', 'synth')} target sound in Vital from default.\n"
                    "Run hierarchical wavetable search, keep <=3 candidates, then continue iterative edits."
                ),
            }
        )

        if default_audio_path is not None:
            messages.append({"role": "assistant", "content": "Listening to current working preset baseline."})
            messages.append(_tool_call("bash", {"command": _build_listen_probe_command(default_audio_path)}))
            messages.append(
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        {"status": "ok", "baseline_audio": "<audio>", "path": str(default_audio_path)},
                        ensure_ascii=False,
                    ),
                }
            )

        messages.append(
            {
                "role": "assistant",
                "content": "Spawning disjoint search shards to gather wavetable proposals in parallel.",
            }
        )
        messages.append(
            _tool_call(
                "spawn_search_agents",
                {
                    "sample_id": sample_id,
                    "target_audio_path": str(target_audio_path),
                    "current_audio_path": str(default_audio_path) if default_audio_path else None,
                    "candidate_universe": [c["candidate_id"] for c in candidate_assets],
                    "num_agents": int(args.num_agents),
                    "shard_strategy": "disjoint_round_robin",
                    "seed": int(args.seed),
                },
            )
        )
        messages.append({"role": "tool_response", "content": json.dumps({"jobs": jobs}, ensure_ascii=False)})

        messages.append({"role": "assistant", "content": "Collecting search-agent reports."})
        messages.append(_tool_call("collect_search_reports", {"sample_id": sample_id, "jobs": jobs}))
        messages.append({"role": "tool_response", "content": json.dumps({"reports": reports}, ensure_ascii=False)})

        messages.append({"role": "assistant", "content": "Judging candidates and selecting up to three for edits."})
        messages.append(
            _tool_call(
                "judge_candidates",
                {
                    "sample_id": sample_id,
                    "target_audio_path": str(target_audio_path),
                    "candidate_audio": candidate_assets,
                    "max_select": int(args.select_k),
                },
            )
        )
        # Include audio previews for selected candidates so the main agent hears
        # what was chosen before starting iterative edits.
        selected_previews = [
            {
                "candidate_id": id_map[n],
                "wavetable_name": n,
                "audio_preview": "<audio>",
            }
            for n in selected_names[: int(args.select_k)]
            if n in candidate_audio
        ]
        judge_response_content = {**judge_result, "selected_previews": selected_previews}
        messages.append(
            {"role": "tool_response", "content": json.dumps(judge_response_content, ensure_ascii=False)}
        )

        archetype = str(entry.get("archetype", "synth"))
        prev_iter_wav: str | None = str(default_audio_path) if default_audio_path else None
        prev_prev_iter_wav: str | None = None  # audio two steps back for 3-clip Stage 1A
        last_listened_iter_wav: str | None = None  # most recently heard audio (skip planning steps)
        _snum_is_planning: dict[int, bool] = {}  # step_num → True if audio gate suppressed listen
        prev_commentary: str | None = None
        # Load target_preset — in-memory for old smoke-test paths, on-disk for
        # --generate mode paths which strip the dict and write target_preset_path instead.
        target_preset = path_data.get("target_preset")
        if target_preset is None and path_data.get("target_preset_path"):
            with open(path_data["target_preset_path"]) as _f:
                target_preset = json.load(_f)
        step_gaps_cache: dict[int, dict | None] = {}  # step_num → remaining gap after that step

        # Track per-step message boundaries and iter audio paths for chunking.
        preamble_msg_count = len(messages)
        step_boundaries: list[tuple[int, int, str | None]] = []  # (msg_start, msg_end, iter_audio)

        for step in path_data.get("iterations", [])[: int(args.max_steps)]:
            _step_msg_start = len(messages)
            step_num = int(step.get("step", 0))
            action_snippet = step.get("action_snippet") or step.get("python_script") or "print('noop')"
            prefix = (
                "Selected candidates: " + ", ".join(selected_names[: int(args.select_k)]) + ".\n\n"
                if step_num == 1 and selected_names
                else ""
            )
            # Decide whether this step needs a full listen or just a plan.
            _abs_delta = step_clap_deltas.get(step_num)
            _is_listening_step = (
                args.audio_gate_threshold == 0
                or _abs_delta is None
                or _abs_delta >= args.audio_gate_threshold
            )
            _snum_is_planning[step_num] = not _is_listening_step

            if args.omni_server:
                # Compute remaining parameter gap from GT preset vs this step's cumulative
                # preset. This grounds the HYPOTHESIS in parameter-space truth rather than
                # relying on the (often absent) remaining_delta_context field in path files.
                step_gap = _step_remaining_gap(target_preset, step) if target_preset else None
                step_gaps_cache[step_num] = step_gap
                remaining_delta_context = (
                    step_gap["context_str"] if step_gap else step.get("remaining_delta_context")
                )
                _common_kwargs = dict(
                    gt_wav=str(target_audio_path),
                    iter_wav=prev_iter_wav,
                    step=step,
                    step_num=step_num,
                    archetype=archetype,
                    prev_commentary=prev_commentary,
                    params_delta=step.get("params_delta") or [],
                    is_mistake_step=bool(step.get("is_mistake_step", False)),
                    # planned_param_names is the closest proxy for allowed_params in path data
                    allowed_params=step.get("planned_param_names") or step.get("allowed_params"),
                    remaining_delta_context=remaining_delta_context,
                    # Use last_listened_iter_wav for Stage 1A so the "before/after" comparison
                    # reflects the last time audio actually changed, not the last rendered state.
                    prior_iter_wav=last_listened_iter_wav,
                    clap_delta=step_clap_deltas.get(step_num),
                    is_planning_step=not _is_listening_step,
                    planner_stage=step.get("planner_stage"),
                )
                if args.commentary_mode == "two_stage":
                    commentary = _call_omni_commentary_two_stage(
                        omni_server=args.omni_server,
                        omni_model=args.omni_model,
                        stage2_server=stage2_server,
                        stage2_model=stage2_model,
                        **_common_kwargs,
                    )
                else:
                    commentary = _call_omni_commentary(
                        omni_server=args.omni_server,
                        model=args.omni_model,
                        **_common_kwargs,
                    )
                prev_commentary = commentary
            else:
                commentary = _step_commentary_fallback(step, step_num)
            messages.append(
                {
                    "role": "assistant",
                    "content": (
                        prefix
                        + commentary
                        + f"\n\nExecuting step {step_num} parameter updates now."
                    ),
                }
            )
            messages.append(_tool_call("bash", {"command": _wrap_as_bash(action_snippet)}))
            if _is_listening_step:
                messages.append({"role": "tool_response", "content": json.dumps({"status": "ok"}, ensure_ascii=False)})
            else:
                messages.append({"role": "tool_response", "content": json.dumps(
                    {"status": "ok", "note": "Sub-perceptual edit — no significant timbral shift. No audio update."},
                    ensure_ascii=False)})

            _step_iter_audio: str | None = None
            if step_num - 1 < len(iter_wavs):
                iter_audio = iter_wavs[step_num - 1]
                prev_prev_iter_wav = prev_iter_wav  # always advance pointer (needed for Stage 1B)
                prev_iter_wav = str(iter_audio)
                if _is_listening_step:
                    used_iter_audio_paths.append(str(iter_audio))
                    last_listened_iter_wav = str(iter_audio)
                    _step_iter_audio = str(iter_audio)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": f"Listening to updated preset after step {step_num}.",
                        }
                    )
                    messages.append(_tool_call("bash", {"command": _build_listen_probe_command(iter_audio)}))
                    messages.append(
                        {
                            "role": "tool_response",
                            "content": json.dumps(
                                {"status": "ok", "step": step_num, "iter_audio": "<audio>", "path": str(iter_audio)},
                                ensure_ascii=False,
                            ),
                        }
                    )
                # planning steps: no audio messages, _step_iter_audio stays None

            step_boundaries.append((_step_msg_start, len(messages), _step_iter_audio))

        # Determine convergence: a path is "complete" only when the final cumulative
        # preset actually matches GT (n_remaining == 0). Budget exhausted means we ran
        # out of steps regardless of whether the preset converged.
        final_steps = path_data.get("iterations", [])[: int(args.max_steps)]
        n_path_iterations = len(path_data.get("iterations", []))

        # Reuse cached gap for the last applied step (already computed during the loop).
        last_step_num = int(final_steps[-1].get("step", 0)) if final_steps else 0
        final_gap = step_gaps_cache.get(last_step_num) if last_step_num else None

        # A path is complete only when the final cumulative preset matches GT (n_remaining == 0).
        path_complete = final_gap is not None and final_gap["n_remaining"] == 0

        convergence = {
            "converged": path_complete,
            "n_remaining": final_gap["n_remaining"] if final_gap else 0,
            "by_subsystem": final_gap["by_subsystem"] if final_gap else {},
        }

        # User turn that prompts the closing assessment (makes this a proper Q&A exchange).
        messages.append(
            {
                "role": "user",
                "content": "That completes the available iterations. Compare your final render to the target and give your final assessment.",
            }
        )

        if args.omni_server and used_iter_audio_paths:
            closing = _call_omni_closing_eval(
                gt_wav=str(target_audio_path),
                final_iter_wav=used_iter_audio_paths[-1],
                archetype=archetype,
                omni_server=args.omni_server,
                model=args.omni_model,
                convergence=convergence,
            )
        else:
            if convergence["converged"]:
                closing = "FINAL ASSESSMENT (complete): All planned iterations have been applied. The recreation path is complete."
            else:
                subs = list(convergence["by_subsystem"].keys())[:2]
                subs_str = ", ".join(subs) if subs else "several parameters"
                closing = (
                    f"FINAL ASSESSMENT (budget exhausted): Iterations exhausted — "
                    f"{subs_str} still differ from the target."
                )
        messages.append({"role": "assistant", "content": closing})
        # Closing messages = last 2 messages (user prompt + assistant assessment).
        closing_messages = messages[-2:]

        audio_assets = [str(target_audio_path)]
        if default_audio_path is not None:
            audio_assets.append(str(default_audio_path))
        # Selected candidate probes appear right after default (matching selected_previews in judge response).
        for n in selected_names[: int(args.select_k)]:
            if n in candidate_audio:
                audio_assets.append(str(candidate_audio[n]))
        # preamble_audio_count = audios before the iter renders (gt + default + candidates).
        preamble_audio_count = len(audio_assets)
        audio_assets.extend(used_iter_audio_paths)

        # Pre-compute the last main-loop step's remaining gap context for use as
        # fallback in correction steps (which are appended after the main loop and
        # have no step_gaps_cache entry of their own).
        _last_main_ctx: str | None = None
        for _s in path_data.get("iterations", [])[: int(args.max_steps)]:
            if _s.get("planner_stage") != "correction":
                _snum = int(_s.get("step", 0))
                _gap = step_gaps_cache.get(_snum)
                if _gap:
                    _last_main_ctx = _gap.get("context_str")

        record = {
            "id": sample_id,
            "task_type": "main",
            "tools": _TOOL_SPECS,
            "messages": messages,
            "audios": audio_assets,
            "assets": {
                "target_audio": str(target_audio_path),
                "current_audio": str(default_audio_path) if default_audio_path else None,
                "candidate_audio": candidate_assets,
                "selected_candidates": [
                    {"candidate_id": id_map[n], "wavetable_name": n, "audio_path": str(candidate_audio[n])}
                    for n in selected_names[: int(args.select_k)]
                ],
            },
            "labels": {
                "judge_ranking": judge_result["ranking"],
                "judge_selected": judge_result["selected"],
                "gt_candidate_ids": [id_map[n] for n in candidate_names if n in gt_names],
            },
            "meta": {
                "sample_id": sample_id,
                "archetype": str(entry.get("archetype", "synth")),
                "agent": "main",
                "num_agents": int(args.num_agents),
                "candidate_source": args.candidate_source,
                "max_steps": int(args.max_steps),
                "commentary_mode": args.commentary_mode,
                "path_complete": path_complete,
                "n_remaining": convergence["n_remaining"],
                # step_labels carries ground-truth params per step for offline grading.
                # remaining_top_2: top 2 subsystem names that still differ from GT after
                # this step — used by the grader to score HYPOTHESIS grounding quality.
                # For correction steps (appended after main loop), fall back to the last
                # main-loop step's remaining gap so the label is still meaningful.
                "step_labels": [
                    {
                        "step": int(s.get("step", 0)),
                        "keyword": str(s.get("search_keyword", "")),
                        "planner_stage": s.get("planner_stage"),
                        "params_delta": s.get("params_delta") or [],
                        "planned_param_names": s.get("planned_param_names") or [],
                        "clap_delta": (float(s["clap_delta"]) if s.get("clap_delta") is not None else None),
                        # clap_score: cosine similarity between this step's rendered audio
                        # and the GT audio. Populated at build time for use by the grader.
                        "clap_score": step_clap_scores.get(int(s.get("step", 0))),
                        "remaining_top_2": _extract_top_remaining(
                            step_gaps_cache.get(int(s.get("step", 0)), {}).get("context_str")
                            if step_gaps_cache.get(int(s.get("step", 0)))
                            else _last_main_ctx,
                            n=2,
                        ),
                        # is_planning_step: True when the audio gate suppressed the listen turn
                        # (|clap_delta| < threshold). Used by grader to exempt planning steps
                        # from section_structure and hypothesis_grounding checks.
                        "is_planning_step": _snum_is_planning.get(int(s.get("step", 0)), False),
                    }
                    for s in path_data.get("iterations", [])[: int(args.max_steps)]
                ],
            },
        }
        assert_valid_ms_swift_multiturn_record(record)

        if args.window_tokens and _tok is not None:
            # Chunk the full record into window-sized blocks.
            return _chunk_into_blocks(
                full_messages=messages,
                full_audios=audio_assets,
                preamble_msg_count=preamble_msg_count,
                preamble_audio_count=preamble_audio_count,
                step_boundaries=step_boundaries,
                closing_messages=closing_messages,
                gt_audio_path=str(target_audio_path),
                default_audio_path=str(default_audio_path) if default_audio_path else None,
                base_record=record,
                reanchor_gt_audio=args.reanchor_gt_audio,
            )
        return [record]

    # Process entries in parallel. Vita (probe rendering) and CLAP (GPU) are serialized
    # via _serial_lock inside _process_entry; Omni HTTP calls run freely in parallel.
    # Results are written incrementally as each future completes so progress is visible
    # and partial results survive early termination.
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _write_lock = threading.Lock()
    records_by_idx: dict[int, list[dict]] = {}
    total_blocks_written = 0

    def _write_records(recs: list[dict], i: int) -> None:
        nonlocal total_blocks_written
        with _write_lock:
            records_by_idx[i] = recs
            # Append-write completed records immediately for progress visibility.
            # Final file is re-written in order after all futures complete.
            total_blocks_written += len(recs)
            sid = recs[0]["meta"].get("sample_id", "?") if recs else "?"
            n_done = len(records_by_idx)
            print(f"[{n_done}/{len(entries)}] {sid}: {len(recs)} block(s) written ({total_blocks_written} total)")

    if args.workers > 1:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_process_entry, entry): i for i, entry in enumerate(entries)}
            for fut in as_completed(futs):
                i = futs[fut]
                try:
                    recs = fut.result()
                    if recs is not None:
                        _write_records(recs, i)
                except Exception as exc:
                    import traceback
                    sample_id = entries[i].get("sample_id", f"entry_{i}")
                    print(f"WARNING: {sample_id} failed: {exc}")
                    traceback.print_exc()
    else:
        for i, entry in enumerate(entries):
            recs = _process_entry(entry)
            if recs is not None:
                _write_records(recs, i)

    # Write final output in deterministic input order.
    records = [r for i in sorted(records_by_idx) for r in records_by_idx[i]]
    with open(out_path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n_source = len(records_by_idx)
    print(f"Wrote {len(records)} records ({n_source} samples → {len(records)} blocks) to {out_path}")


if __name__ == "__main__":
    main()
