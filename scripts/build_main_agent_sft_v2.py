#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import random
import sys
import threading
from itertools import combinations
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


def _group_params_for_plan(params_delta: list[dict], modulations_changed: list | None = None) -> str:
    """Convert params_delta to a natural-language string for PLAN prefix seeding.

    ≤5 params  → enumerate all display names.
    >5 params  → group by subsystem; small groups name the attributes, large groups
                 say "N subsystem parameters".

    For modulation params, substitutes route descriptions (src→dst) instead of
    raw slot numbers when modulations_changed is provided.
    """
    import re as _re
    if not params_delta:
        return "these parameters"

    # Build slot→route lookup for modulation substitution.
    _slot_to_route: dict[str, dict] = {}
    for i, route in enumerate(modulations_changed or []):
        slot_key = route.get("slot") or str(i + 1)
        _slot_to_route[slot_key] = route

    # Expand modulation params into route descriptions, deduplicated by slot.
    _mod_slots_seen: set[str] = set()
    expanded_delta: list[dict] = []
    mod_routes_seen: list[str] = []
    for d in params_delta:
        name = d["name"]
        _mod_match = _re.match(r"modulation_(\d+)_(bypass|amount|stereo|bipolar|power)", name)
        if _mod_match and _slot_to_route:
            slot_num = _mod_match.group(1)
            if slot_num not in _mod_slots_seen:
                _mod_slots_seen.add(slot_num)
                route = _slot_to_route.get(slot_num)
                if route:
                    src, dst = route.get("source", ""), route.get("destination", "")
                    if src or dst:
                        mod_routes_seen.append(f"{src}→{dst}")
        else:
            expanded_delta.append(d)

    if len(expanded_delta) <= 5 and not mod_routes_seen:
        names = [_json_key_to_display(d["name"]) for d in expanded_delta]
        if not names:
            return "these parameters"
        if len(names) == 1:
            return names[0]
        if len(names) == 2:
            return f"{names[0]} and {names[1]}"
        return ", ".join(names[:-1]) + f", and {names[-1]}"

    # Group non-mod params by subsystem
    groups: dict[str, list[str]] = {}
    for d in expanded_delta:
        sub = _get_param_subsystem(d["name"])
        groups.setdefault(sub, []).append(d["name"])

    parts: list[str] = []
    for sub, keys in groups.items():
        label = _SUBSYSTEM_DISPLAY.get(sub, sub.replace("_", " "))
        n = len(keys)
        if n == 1:
            parts.append(_json_key_to_display(keys[0]))
        elif n <= 3:
            prefix = sub + "_"
            attrs = []
            for k in keys:
                attr_key = k[len(prefix):] if k.startswith(prefix) else k
                attr_disp = _json_key_to_display(attr_key).lower()
                if attr_disp == "on":
                    attr_disp = "on/off"
                attrs.append(attr_disp)
            if len(attrs) == 2:
                parts.append(f"{label} {attrs[0]} and {attrs[1]}")
            else:
                parts.append(f"{label} {', '.join(attrs[:-1])}, and {attrs[-1]}")
        else:
            parts.append(f"{n} {label} parameters")

    # Append mod route descriptions
    if mod_routes_seen:
        if len(mod_routes_seen) <= 3:
            parts.extend(mod_routes_seen)
        else:
            # Summarise: list first 3 routes + count remainder
            shown = ", ".join(mod_routes_seen[:3])
            parts.append(f"modulation routes {shown} (+{len(mod_routes_seen)-3} more)")

    if not parts:
        return "these parameters"
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


def _format_delta_context(params_delta: list, is_mistake_step: bool, modulations_changed: list | None = None) -> str:
    """Format params_delta list into a natural-language string for the Omni prompt.

    When the step is modulation-heavy, substitutes raw modulation_N_bypass/amount entries
    with human-readable source→destination route descriptions from modulations_changed.
    """
    if not params_delta:
        return ""

    # Build slot→route lookup.  Routes from _target_mod_routes always have a "slot" key;
    # fall back to sequential i+1 indexing for legacy path files that lack it.
    _slot_to_route: dict[str, dict] = {}
    for i, route in enumerate((modulations_changed or [])):
        slot_key = route.get("slot") or str(i + 1)
        _slot_to_route[slot_key] = route

    lines = []
    mistake_params = []
    mod_slots_seen: set[str] = set()
    mod_route_lines: list[str] = []

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

        # For modulation slot params, emit route description instead of raw name.
        import re as _re
        _mod_match = _re.match(r"modulation_(\d+)_(bypass|amount|stereo|bipolar|power)", name)
        if _mod_match and modulations_changed:
            slot_num = _mod_match.group(1)
            if slot_num not in mod_slots_seen:
                mod_slots_seen.add(slot_num)
                route = _slot_to_route.get(slot_num)
                if route:
                    src = route.get("source", "?")
                    dst = route.get("destination", "?")
                    amt = route.get("amount", 0.0)
                    mistake_flag = " ⚠ moved away from target" if d.get("mistake") else ""
                    mod_route_lines.append(f"- {src} → {dst} (amount {amt:+.2f}){mistake_flag}")
                    if d.get("mistake"):
                        mistake_params.append(f"{src}→{dst}")
            continue  # handled via mod_route_lines

        if d.get("mistake"):
            lines.append(
                f"- {display} ({label}): {direction} {mag_str} [{from_n:.2f}\u2192{to_n:.2f}] \u26a0 moved away from target"
            )
            mistake_params.append(display)
        else:
            lines.append(f"- {display} ({label}): {direction} {mag_str} [{from_n:.2f}\u2192{to_n:.2f}]")

    # Emit non-mod params first, then mod routes grouped under a header.
    all_lines = lines[:]
    if mod_route_lines:
        all_lines.append(f"Modulation routes ({len(mod_route_lines)} active):")
        all_lines.extend(mod_route_lines)

    context = "Parameters being changed in this step:\n" + "\n".join(all_lines)
    if mistake_params:
        context += (
            f"\n\nNote: {', '.join(mistake_params)} moved in the wrong direction (overcorrection). "
            "The description should acknowledge this mistake and what needs to be fixed next."
        )
    return context


def _build_param_summary(params_delta: list[dict], modulations_changed: list | None = None) -> str:
    """One-line step summary: top 4 changes by magnitude with ↑/↓ arrows.

    For modulation-heavy steps, substitutes raw modulation_N_* names with
    source→destination route descriptions when modulations_changed is provided.
    """
    import re as _re
    if not params_delta:
        return "(no changes)"

    _slot_to_route: dict[str, dict] = {}
    for i, route in enumerate(modulations_changed or []):
        slot_key = route.get("slot") or str(i + 1)
        _slot_to_route[slot_key] = route

    top = sorted(params_delta, key=lambda d: abs(d.get("to_norm", 0) - d.get("from_norm", 0)), reverse=True)[:4]
    parts = []
    for d in top:
        name = d["name"]
        arrow = "\u2191" if d.get("to_norm", 0) > d.get("from_norm", 0) else "\u2193"
        mag = abs(d.get("to_norm", 0) - d.get("from_norm", 0))
        _mod_match = _re.match(r"modulation_(\d+)_(bypass|amount|stereo|bipolar|power)", name)
        if _mod_match and _slot_to_route:
            route = _slot_to_route.get(_mod_match.group(1))
            if route:
                src = route.get("source", "?")
                dst = route.get("destination", "?")
                parts.append(f"{src}→{dst} {arrow} {mag:.2f}")
                continue
        parts.append(f"{_json_key_to_display(name)} {arrow} {mag:.2f}")
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


def _build_tuple_candidates(
    candidate_names: list[str],
    *,
    max_tuple_size: int,
    max_tuples: int,
    seed: int,
    preferred_names: list[str] | None = None,
) -> list[dict[str, object]]:
    """Build tuple candidates (size 1..max_tuple_size) from candidate names."""
    if not candidate_names:
        return []
    max_tuple_size = max(1, min(3, int(max_tuple_size)))
    max_tuples = max(1, int(max_tuples))

    rng = random.Random(int(seed))
    top_pool = list(candidate_names)[: min(len(candidate_names), 12)]

    tuples: list[tuple[str, ...]] = []
    for sz in range(1, max_tuple_size + 1):
        tuples.extend(list(combinations(top_pool, sz)))
    rng.shuffle(tuples)

    out: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()

    pref = [n for n in (preferred_names or []) if n in candidate_names]
    if pref:
        pref_tuple = tuple(pref[:max_tuple_size])
        if pref_tuple:
            seen.add(pref_tuple)
            out.append({"tuple_id": "T1", "members": list(pref_tuple)})

    for t in tuples:
        if len(out) >= max_tuples:
            break
        if t in seen:
            continue
        seen.add(t)
        out.append({"tuple_id": f"T{len(out) + 1}", "members": list(t)})
    return out


def _tuple_teacher_score(members: list[str], clap_scores: dict[str, float]) -> float:
    vals = [float(clap_scores.get(n, 0.0)) for n in members]
    if not vals:
        return 0.0
    # Slightly reward richer tuples while keeping single-table options competitive.
    return float(sum(vals) / len(vals) + (0.01 * (len(members) - 1)))


def _role_hints_for_tuple_size(n: int) -> list[str]:
    if n <= 1:
        return ["fundamental"]
    if n == 2:
        return ["fundamental", "harmonic_texture"]
    return ["fundamental", "harmonic_texture", "air_noise"]


def _acceptable_tuple_ids(
    tuple_rows: list[dict[str, object]],
    tuple_scores: dict[str, float],
    *,
    epsilon: float,
    max_accept: int,
) -> list[str]:
    if not tuple_rows:
        return []
    ranked = sorted(
        [str(t["tuple_id"]) for t in tuple_rows],
        key=lambda tid: float(tuple_scores.get(tid, 0.0)),
        reverse=True,
    )
    best = float(tuple_scores.get(ranked[0], 0.0))
    band = [tid for tid in ranked if float(tuple_scores.get(tid, 0.0)) >= (best - float(epsilon))]
    if len(band) < 2:
        band = ranked[: min(2, len(ranked))]
    return band[: max(1, int(max_accept))]


def _build_tuple_search_reports(
    shards: list[list[str]],
    tuple_members_by_id: dict[str, list[str]],
    tuple_scores: dict[str, float],
    proposals_per_agent: int,
    name_to_candidate_id: dict[str, str],
) -> list[dict]:
    reports: list[dict] = []
    for i, shard in enumerate(shards, start=1):
        if not shard:
            continue
        ranked = sorted(shard, key=lambda tid: float(tuple_scores.get(tid, 0.0)), reverse=True)
        selected = ranked[: max(1, int(proposals_per_agent))]
        proposals: list[dict[str, object]] = []
        for tid in selected:
            members = tuple_members_by_id.get(tid, [])
            candidate_ids = [name_to_candidate_id[m] for m in members if m in name_to_candidate_id]
            proposals.append(
                {
                    "tuple_id": tid,
                    "candidate_ids": candidate_ids,
                    "role_hints": _role_hints_for_tuple_size(len(members)),
                    "confidence_band": (
                        "high"
                        if float(tuple_scores.get(tid, 0.0)) >= 0.60
                        else "medium" if float(tuple_scores.get(tid, 0.0)) >= 0.45
                        else "low"
                    ),
                    "reason": (
                        "Closest overall source blend across transient/body/sustain."
                        if len(members) >= 2
                        else "Strong foundational source candidate."
                    ),
                }
            )
        reports.append(
            {
                "agent_id": f"sa_{i}",
                "considered": len(shard),
                "uncertainty": "low" if len(selected) >= 2 else "medium",
                "proposed_tuples": proposals,
            }
        )
    return reports


def _build_tuple_judge_result(
    tuple_rows: list[dict[str, object]],
    tuple_scores: dict[str, float],
    *,
    select_k: int,
) -> dict[str, object]:
    ranked = sorted(
        [str(t["tuple_id"]) for t in tuple_rows],
        key=lambda tid: float(tuple_scores.get(tid, 0.0)),
        reverse=True,
    )
    selected = ranked[: max(1, int(select_k))]
    return {
        "ranking": ranked,
        "selected": selected,
        "reason": "Ranked by audible source plausibility across search reports. Selected top blends.",
    }


def _ensure_tuple_preview_audio(
    *,
    sample_id: str,
    tuple_id: str,
    member_names: list[str],
    candidate_audio: dict[str, Path],
    out_dir: Path,
    cache: dict[str, Path],
) -> Path | None:
    """Create a mixed preview wav for a tuple from rendered candidate probes."""
    import numpy as np
    import soundfile as sf

    cache_key = f"{sample_id}:{tuple_id}"
    if cache_key in cache:
        return cache[cache_key]

    paths = [candidate_audio[n] for n in member_names if n in candidate_audio]
    if not paths:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{sample_id}_{tuple_id}.wav"
    if out_path.exists():
        cache[cache_key] = out_path
        return out_path

    waves = []
    sr_ref = None
    for p in paths:
        x, sr = sf.read(p, always_2d=True)
        if sr_ref is None:
            sr_ref = sr
        if sr != sr_ref:
            continue
        waves.append(x.astype(np.float32))
    if not waves:
        return None

    min_len = min(w.shape[0] for w in waves)
    min_ch = min(w.shape[1] for w in waves)
    stack = np.stack([w[:min_len, :min_ch] for w in waves], axis=0)
    mix = np.mean(stack, axis=0)
    peak = float(np.max(np.abs(mix))) if mix.size else 0.0
    if peak > 1e-6:
        mix = 0.95 * (mix / peak)
    sf.write(out_path, mix, sr_ref or 44100)
    cache[cache_key] = out_path
    return out_path


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


def _assert_no_similarity_score_leak(messages: list[dict]) -> None:
    """Guardrail for no-score-leak tuple-search mode."""
    banned = ("cosine_vs_target", "top scores:", "clap similarity")
    for i, msg in enumerate(messages):
        content = str(msg.get("content", "")).lower()
        if any(token in content for token in banned):
            raise ValueError(f"score_leak_detected_at_message_{i}")


import threading as _llm_threading

class _LlmPostStats:
    def __init__(self):
        self._lock = _llm_threading.Lock()
        self.calls = 0
        self.successes = 0
        self.retries = 0
        self.failures = 0
    def record_success(self, retries: int = 0):
        with self._lock:
            self.calls += 1
            self.successes += 1
            self.retries += retries
    def record_failure(self, retries: int = 0):
        with self._lock:
            self.calls += 1
            self.failures += 1
            self.retries += retries
    def summary(self) -> str:
        return (f"calls={self.calls}  ok={self.successes}  "
                f"failed={self.failures}  retries={self.retries}")

llm_post_stats = _LlmPostStats()


def _llm_post(server_url: str, payload: dict, timeout: float = 120.0, max_retries: int = 3) -> dict:
    """POST to an OpenAI-compatible completions endpoint with retry on timeout/5xx/connect errors."""
    import httpx, time
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            with httpx.Client() as client:
                resp = client.post(server_url, json=payload, timeout=timeout)
                resp.raise_for_status()
                llm_post_stats.record_success(retries=attempt)
                return resp.json()
        except (httpx.TimeoutException, httpx.HTTPStatusError, httpx.ConnectError) as exc:
            last_exc = exc
            _tag = type(exc).__name__
            print(f"  LLM_POST retry {attempt+1}/{max_retries} ({_tag}: {exc})", flush=True)
            if attempt < max_retries - 1:
                time.sleep(3)
    assert last_exc is not None
    llm_post_stats.record_failure(retries=max_retries)
    raise last_exc


def _check_server_reachable(server_url: str, label: str) -> None:
    """Raise a clear RuntimeError if the server is not reachable before the build starts."""
    import httpx
    base = server_url.rstrip("/").rsplit("/v1", 1)[0]
    try:
        with httpx.Client() as client:
            resp = client.get(f"{base}/v1/models", timeout=5.0)
            resp.raise_for_status()
    except Exception as exc:
        raise RuntimeError(
            f"{label} server at {server_url} is not reachable: {exc}\n"
            "Start the model server before running this script."
        ) from exc


def _wrap_as_bash(python_code: str) -> str:
    """Wrap bare Python code in a shell-executable heredoc."""
    stripped = python_code.strip()
    if stripped.startswith("python"):
        return stripped  # already a shell command
    return f"python - <<'PY'\n{stripped}\nPY"


_LEGACY_VITAL_SNIPPET_PREFIX = (
    "import sys, json\n"
    "sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
    "from vital_tools import VitalController\n"
    "vc = VitalController()\n"
    "vc.discover()\n"
)


def _vital_reapy_bootstrap() -> str:
    """Return robust VitalController setup code for both ReaScript and external Python."""
    return (
        "import sys, json\n"
        "import atexit\n"
        "try:\n"
        "    from maestro.reaper.vital_tools import VitalController\n"
        "except Exception:\n"
        "    sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
        "    from vital_tools import VitalController\n"
        "_rpr = None\n"
        "_vc_ctx = None\n"
        "if 'RPR_CountTracks' in globals():\n"
        "    vc = VitalController()\n"
        "else:\n"
        "    import reapy\n"
        "    _vc_ctx = reapy.inside_reaper()\n"
        "    _vc_ctx.__enter__()\n"
        "    _api = reapy.reascript_api\n"
        "    _rpr = {f'RPR_{fn}': getattr(_api, fn) for fn in dir(_api) if not fn.startswith('_')}\n"
        "    vc = VitalController(_rpr=_rpr)\n"
        "def _rpr_call(_name, *_args):\n"
        "    if _rpr is not None and _name in _rpr:\n"
        "        return _rpr[_name](*_args)\n"
        "    _fn = globals().get(_name)\n"
        "    if _fn is None:\n"
        "        raise RuntimeError(f'RPR function {_name!r} unavailable')\n"
        "    return _fn(*_args)\n"
        "def _ensure_vital_loaded():\n"
        "    try:\n"
        "        vc.discover()\n"
        "        return\n"
        "    except Exception:\n"
        "        pass\n"
        "    _proj = 0\n"
        "    if int(_rpr_call('RPR_CountTracks', _proj)) == 0:\n"
        "        _rpr_call('RPR_InsertTrackAtIndex', 0, True)\n"
        "    _track = _rpr_call('RPR_GetTrack', _proj, 0)\n"
        "    for _fx_name in (\n"
        "        'Vital',\n"
        "        'VST3i: Vital',\n"
        "        'VSTi: Vital',\n"
        "        'VST3: Vital (Vital Audio)',\n"
        "        'VST3i: Vital (Vital Audio)',\n"
        "        'VST3: Vital',\n"
        "    ):\n"
        "        _idx = _rpr_call('RPR_TrackFX_AddByName', _track, _fx_name, False, 1)\n"
        "        if isinstance(_idx, (tuple, list)):\n"
        "            _idx = _idx[0] if _idx else -1\n"
        "        if int(_idx) >= 0:\n"
        "            break\n"
        "    vc.discover()\n"
        "_ensure_vital_loaded()\n"
        "if _vc_ctx is not None:\n"
        "    atexit.register(lambda: _vc_ctx.__exit__(None, None, None))\n"
    )


def _harden_vital_snippet_for_reapy(python_code: str) -> str:
    """Rewrite legacy VitalController snippets to run via reapy when outside ReaScript."""
    stripped = python_code.strip()
    if "from vital_tools import VitalController" not in stripped or "vc.discover()" not in stripped:
        return stripped

    if stripped.startswith(_LEGACY_VITAL_SNIPPET_PREFIX):
        body = stripped[len(_LEGACY_VITAL_SNIPPET_PREFIX):]
        return f"{_vital_reapy_bootstrap()}{body}"

    body = stripped
    for line in (
        "import sys, json\n",
        "sys.path.append('/home/nate/.config/REAPER/Scripts')\n",
        "from vital_tools import VitalController\n",
        "vc = VitalController()\n",
        "vc.discover()\n",
    ):
        body = body.replace(line, "", 1)
    return f"{_vital_reapy_bootstrap()}{body.strip()}"


def _step_commentary_fallback(step: dict, step_num: int, is_planning_step: bool = False) -> str:
    keyword = str(step.get("search_keyword") or "target controls")
    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    if is_planning_step:
        return f"PLAN: Adjusting {keyword} parameters. These structural changes set up the next audible step toward the target {primary} sound."
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
    modulations_changed: list | None = None,
    is_mistake_step: bool = False,
    allowed_params: list | None = None,
    remaining_delta_context: str | None = None,
    prior_iter_wav: str | None = None,  # unused in single-call mode
    clap_delta: float | None = None,  # unused in single-call mode
    is_planning_step: bool = False,
    planner_stage: str | None = None,
    steps_since_last_listen: int = 1,  # unused in single-call mode
) -> str:
    """Call the hosted Omni model for grounded HEARD/HYPOTHESIS/PLAN commentary.
    Falls back to the template string if the server is unavailable."""

    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    keyword = str(step.get("search_keyword") or "target controls")

    delta_ctx = _format_delta_context(params_delta or [], is_mistake_step, modulations_changed=modulations_changed)
    param_summary = _build_param_summary(params_delta or [], modulations_changed=modulations_changed)
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
            params_str = _group_params_for_plan(params_delta, modulations_changed=modulations_changed)
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
            result = _llm_post(f"{omni_server}/v1/chat/completions", payload, timeout=120.0)
            return result["choices"][0]["message"]["content"].strip()
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
        result = _llm_post(f"{omni_server}/v1/chat/completions", payload, timeout=180.0)
        return result["choices"][0]["message"]["content"].strip()
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
    modulations_changed: list | None = None,
    is_mistake_step: bool = False,
    allowed_params: list | None = None,
    remaining_delta_context: str | None = None,
    prior_iter_wav: str | None = None,
    clap_delta: float | None = None,
    planner_stage: str | None = None,
    is_planning_step: bool = False,
    steps_since_last_listen: int = 1,
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

    primary = str(step.get("primary_family") or "synth")
    support = str(step.get("support_family") or "none")
    keyword = str(step.get("search_keyword") or "target controls")
    delta_ctx = _format_delta_context(params_delta or [], is_mistake_step, modulations_changed=modulations_changed)
    param_summary = _build_param_summary(params_delta or [], modulations_changed=modulations_changed)
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
            params_str = _group_params_for_plan(params_delta, modulations_changed=modulations_changed)
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
            result = _llm_post(f"{stage2_server}/v1/chat/completions", s2_payload, timeout=120.0)
            return result["choices"][0]["message"]["content"].strip()
        except Exception:
            return f"PLAN: {plan_sentence1} These precision adjustments prepare the preset for the next audible improvement step."

    # Build dynamic edit-range description for Stage 1A prompt and HEARD attribution.
    # When multiple planning steps preceded this listen, name the full range of steps.
    if steps_since_last_listen <= 1:
        _edit_range_str = "the parameter edit just applied"
        _s1a_before_label = "the preset BEFORE the last parameter edit"
        _s1a_after_label = "the preset AFTER the last parameter edit"
    else:
        _first_step = step_num - steps_since_last_listen + 1
        _edit_range_str = (
            f"the {steps_since_last_listen} parameter edits applied since your last listen "
            f"(steps {_first_step}–{step_num})"
        )
        _s1a_before_label = f"the preset BEFORE these edits (your last listened state)"
        _s1a_after_label = f"the preset NOW (after steps {_first_step}–{step_num})"

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
                    f"You are a music production AI. Listen to two synthesizer clips.\n"
                    f"AUDIO A: {_s1a_before_label}.\n"
                    f"AUDIO B: {_s1a_after_label}.\n\n"
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
                url = f"{omni_server}/v1/chat/completions"
                s1a_result = _llm_post(url, {"model": omni_model, "messages": [{"role": "user", "content": s1a_content}],
                                             "max_tokens": 160, "temperature": 0.4}, timeout=180.0)
                step_change_obs = s1a_result["choices"][0]["message"]["content"].strip()
                s1b_result = _llm_post(url, {"model": omni_model, "messages": [{"role": "user", "content": s1b_content}],
                                             "max_tokens": 160, "temperature": 0.4}, timeout=180.0)
                remaining_gap_obs = s1b_result["choices"][0]["message"]["content"].strip()
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
                url = f"{omni_server}/v1/chat/completions"
                s1_result = _llm_post(url, {"model": omni_model, "messages": [{"role": "user", "content": s1_content}],
                                            "max_tokens": 220, "temperature": 0.4}, timeout=180.0)
                remaining_gap_obs = s1_result["choices"][0]["message"]["content"].strip()
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
                url = f"{omni_server}/v1/chat/completions"
                s1_result = _llm_post(url, {"model": omni_model, "messages": [{"role": "user", "content": s1_content}],
                                            "max_tokens": 220, "temperature": 0.4}, timeout=180.0)
                remaining_gap_obs = s1_result["choices"][0]["message"]["content"].strip()
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
            "HEARD (2 sentences): "
            f"Sentence 1 describes what {_edit_range_str} achieved, drawn from [WHAT THE LAST STEP CHANGED]. "
            "Sentence 2 names the single most important remaining timbral gap, drawn from [REMAINING GAP TO TARGET] — "
            "describe it as a positive quality the TARGET HAS (e.g. 'the target has a brighter upper harmonic shelf', "
            "'the target has a sharper transient click') not as something the current preset lacks. "
            "Use specific sonic language (frequency region, envelope shape, harmonic character). "
            "BANNED phrases for Sentence 2: 'still lacks', 'still missing', 'still absent', "
            "'still needs', 'still has no', 'remains absent', 'not yet present', 'continues to lack'."
        )
        if prior_heard:
            heard_instruction = (
                heard_base + " "
                f"The previous step's HEARD began: \"{prior_heard[:180]}\" "
                f"— Sentence 2 must describe a DIFFERENT remaining gap than what that step's Sentence 2 focused on. "
                f"If the same raw gap persists, describe it from a different angle: different frequency region, "
                f"different property (texture vs. envelope vs. movement), or different comparison point."
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
            "Use specific sonic language — describe what the TARGET HAS as a positive quality "
            "(e.g. 'the target has a brighter upper harmonic shelf', 'the target has a sharper "
            "transient click') not as something the current preset lacks. "
            "BANNED phrases: 'still lacks', 'still missing', 'still absent', 'still needs', "
            "'still has no', 'remains absent', 'not yet present', 'continues to lack', "
            "'has no', 'is missing'."
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
        params_str = _group_params_for_plan(params_delta, modulations_changed=modulations_changed)
        plan_instruction = (
            f"PLAN (2 sentences): "
            f"Sentence 1 — copy this exactly: \"Adjusting {params_str}.\" "
            f"Sentence 2 — one sentence explaining how these changes will address the remaining "
            f"timbral gap. Do not name any parameter in Sentence 2. "
            f"Use future tense only (will, should, needs to) — never use past tense to describe "
            f"what will happen."
        )
    else:
        plan_instruction = (
            "PLAN (1 sentence): Explain the rationale for the parameter adjustments listed above. "
            "Use future tense only."
        )

    # HYPOTHESIS: audio observation about the remaining gap — what you just heard reveals
    # about the distance still left to close. NOT a prescription for the next step (that's PLAN).
    # This keeps temporal logic clean: HYPOTHESIS reflects what was heard, PLAN says what to do.
    remaining_top = _extract_top_remaining(remaining_delta_context)

    prior_hypothesis = _extract_hypothesis(prev_commentary) if prev_commentary else None
    if planner_stage == "correction":
        # Correction steps fix params that were deliberately set wrong in an earlier step.
        correction_params = [p["name"].replace("_", " ") for p in (params_delta or [])[:3]]
        correction_str = ", ".join(correction_params) if correction_params else "earlier parameters"
        hypothesis_instruction = (
            f"HYPOTHESIS (1 sentence): Based on what you just heard, describe what the "
            f"{correction_str} correction achieved perceptually, and what single timbral quality "
            f"the audio reveals as the most prominent remaining gap. "
            f"Ground it in the audio — name a specific perceptual quality. "
            f"Start with a concrete audio observation, not 'Despite' or 'Although'."
        )
    elif remaining_delta_context and not remaining_delta_context.startswith("none"):
        # Standard case: describe what the audio reveals about the remaining gap.
        prior_hyp_ctx = (
            f" Prior hypothesis: \"{prior_hypothesis[:160]}\" — must describe a DIFFERENT timbral quality."
            if prior_hypothesis else ""
        )
        hypothesis_instruction = (
            f"HYPOTHESIS (1 sentence): Describe the most important timbral quality that still "
            f"separates the current sound from the target, based on what you just heard. "
            f"Ground it in the audio — name a specific perceptual quality "
            f"(e.g. brightness, attack transient, movement speed, harmonic density).{prior_hyp_ctx} "
            f"Use hedged language (appears to, seems to, suggests). "
            f"Do NOT prescribe a next step, name a parameter family, or use 'Despite'/'Although'/'still lacks'."
        )
    else:
        # Converged or no remaining context — brief honest summary.
        prior_hyp_converged_ctx = (
            f" Prior hypothesis: \"{prior_hypothesis[:160]}\" — "
            f"revise: describe a DIFFERENT timbral quality if the gap has changed."
            if prior_hypothesis else ""
        )
        hypothesis_instruction = (
            "HYPOTHESIS (1 sentence): Describe what timbral quality still separates the "
            "current render from the target, if anything, based on what you hear. "
            f"If the gap is very small, say so directly.{prior_hyp_converged_ctx}"
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
        result = _llm_post(f"{stage2_server}/v1/chat/completions", s2_payload, timeout=180.0)
        return result["choices"][0]["message"]["content"].strip()
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

    with open(gt_wav, "rb") as f:
        gt_b64 = base64.b64encode(f.read()).decode()
    with open(final_iter_wav, "rb") as f:
        iter_b64 = base64.b64encode(f.read()).decode()

    if convergence is None:
        verdict_instruction = (
            "Write 2-3 sentences: what aspects now match well, and what still differs most. "
            "Be specific about timbre, texture, or envelope."
        )
    elif convergence.get("converged"):
        verdict_instruction = (
            "All planned iterations have been applied. "
            "Write 2 sentences: honestly describe how close the final render is to the target "
            "(note any timbral, textural, or envelope differences you can still hear), "
            "then confirm the iteration path is complete even if a small gap remains."
        )
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

    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}},
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}},
        {"type": "text", "text": (
            f"You are a music production AI agent doing a final review.\n"
            f"AUDIO A is the target {archetype} sound. AUDIO B is the final recreated preset.\n\n"
            f"{verdict_instruction}"
        )},
    ]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 200,
        "temperature": 0.4,
    }
    try:
        result = _llm_post(f"{omni_server}/v1/chat/completions", payload, timeout=180.0)
        return result["choices"][0]["message"]["content"].strip()
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
    ap.add_argument(
        "--tuple-search-v2",
        action="store_true",
        help="Enable tuple-based parallel search traces (1..3 wavetable tuples, no similarity-score leakage).",
    )
    ap.add_argument("--tuple-max-size", type=int, default=3, help="Max tuple size for tuple-search-v2 (default: 3).")
    ap.add_argument("--tuple-max-candidates", type=int, default=28, help="Max tuple candidates per sample (default: 28).")
    ap.add_argument("--tuple-epsilon", type=float, default=0.015, help="Acceptable-set score band epsilon (default: 0.015).")
    ap.add_argument("--tuple-accept-max", type=int, default=6, help="Max acceptable tuple labels per sample (default: 6).")
    ap.add_argument("--tuple-preview-dir", type=Path, default=Path("outputs/agent_sft/tuple_previews"))

    ap.add_argument("--audio-gate-threshold", type=float, default=0.01,
        help="Absolute CLAP delta threshold. Steps with |delta| below this get PLAN-only turns "
             "(no audio listen). Set to 0 to disable gating (listen every step). Default 0.01.")
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

    # Pre-populate the CLAP embedding cache for all per-sample audio files (target, default,
    # iter_wavs, existing candidate probes) sequentially in the main process BEFORE launching
    # worker threads.  Running GPU CLAP on the main thread avoids worker-thread CUDA context
    # issues (cudaErrorOperatingSystem / VA-space exhaustion) that arise after vLLM starts
    # filling its KV cache.  After the batch precomputation the embedder is moved to CPU so
    # any NEW probe files rendered at runtime (new candidate wavetables) can still be embedded
    # safely without touching the GPU from worker threads.
    _clap_audio_paths: list[Path] = []
    for _e in entries:
        for _key in ("gt_wav", "gt_probe_wav", "default_wav"):
            _p = _e.get(_key)
            if _p:
                _clap_audio_paths.append(Path(_p))
        for _p in _e.get("iter_wavs") or []:
            _clap_audio_paths.append(Path(_p))
    # Also pre-embed existing candidate probe files so workers hit the cache.
    _probe_dir_path = Path(args.probe_dir)
    if _probe_dir_path.exists():
        for _pp in sorted(_probe_dir_path.glob("*.wav")):
            _clap_audio_paths.append(_pp)
    _clap_audio_paths = list(dict.fromkeys(_clap_audio_paths))  # deduplicate, preserve order
    print(f"Pre-computing CLAP embeddings for {len(_clap_audio_paths)} audio files...", flush=True)
    for _ap in _clap_audio_paths:
        if _ap.exists():
            try:
                embedder.embed_audio_path(_ap)
            except Exception as _ce:
                print(f"  WARNING: CLAP embed failed for {_ap.name}: {_ce}")
    print(f"CLAP pre-computation done ({len(embedder._cache)} embeddings cached).", flush=True)
    # Move CLAP to CPU after GPU-accelerated precomputation.  Worker threads will call
    # embed_audio_path for newly rendered probes that weren't pre-embedded; CPU is safe
    # from the CUDA VA-space issues that affect worker threads.
    if args.clap_device != "cpu":
        try:
            embedder.model = embedder.model.to("cpu")
            embedder.device = "cpu"
            print("CLAP model moved to CPU for worker-thread safety.", flush=True)
        except Exception as _move_exc:
            print(f"WARNING: could not move CLAP to CPU: {_move_exc}")

    # Shared mutable probe cache. Serialized via _serial_lock.
    candidate_audio: dict[str, Path] = {}
    tuple_preview_cache: dict[str, Path] = {}

    # Serializes vita (probe rendering) and CLAP (GPU) calls. Omni HTTP calls run freely
    # outside this lock — they are the dominant bottleneck and benefit from parallelism.
    _serial_lock = threading.Lock()

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
        use_tuple_search = bool(args.tuple_search_v2)

        tuple_rows: list[dict[str, object]] = []
        tuple_members_by_id: dict[str, list[str]] = {}
        tuple_scores: dict[str, float] = {}
        tuple_preview_paths: dict[str, Path] = {}
        acceptable_tuple_ids: list[str] = []
        selected_tuple_ids: list[str] = []

        if use_tuple_search:
            pref = [n for n in candidate_names if n in gt_names]
            tuple_rows = _build_tuple_candidates(
                candidate_names,
                max_tuple_size=int(args.tuple_max_size),
                max_tuples=int(args.tuple_max_candidates),
                seed=int(args.seed),
                preferred_names=pref,
            )
            tuple_members_by_id = {
                str(t["tuple_id"]): [str(x) for x in list(t.get("members", []))]
                for t in tuple_rows
            }
            tuple_scores = {
                tid: _tuple_teacher_score(members, clap_scores)
                for tid, members in tuple_members_by_id.items()
            }
            acceptable_tuple_ids = _acceptable_tuple_ids(
                tuple_rows,
                tuple_scores,
                epsilon=float(args.tuple_epsilon),
                max_accept=int(args.tuple_accept_max),
            )
            tuple_ids = [str(t["tuple_id"]) for t in tuple_rows]
            tuple_shards = build_disjoint_shards(tuple_ids, args.num_agents)
            jobs = [
                {
                    "agent_id": f"sa_{i}",
                    "candidate_shard": shard,
                    "seed": int(args.seed) + i,
                }
                for i, shard in enumerate(tuple_shards, start=1)
                if shard
            ]
            reports = _build_tuple_search_reports(
                tuple_shards,
                tuple_members_by_id,
                tuple_scores,
                args.proposals_per_agent,
                id_map,
            )
            judge_result = _build_tuple_judge_result(tuple_rows, tuple_scores, select_k=args.select_k)
            selected_tuple_ids = [str(x) for x in list(judge_result["selected"])]
            selected_names = []  # handled through selected tuples
        else:
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
        start_type = entry.get("start_type", "init")
        archetype_str = entry.get('archetype', 'synth')
        if start_type == "random":
            start_phrase = f"Adapt the current {archetype_str} preset to match this target sound."
        else:
            start_phrase = f"Recreate this {archetype_str} target sound in Vital from default."
        messages.append(
            {
                "role": "user",
                "content": (
                    f"<audio>\n{start_phrase}\n"
                    +
                    (
                        "Run hierarchical tuple search, keep <=3 wavetable tuples, then continue iterative edits."
                        if use_tuple_search
                        else "Run hierarchical wavetable search, keep <=3 candidates, then continue iterative edits."
                    )
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
                    "candidate_universe": (
                        [str(t["tuple_id"]) for t in tuple_rows]
                        if use_tuple_search
                        else [c["candidate_id"] for c in candidate_assets]
                    ),
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

        messages.append(
            {
                "role": "assistant",
                "content": (
                    "Judging tuple proposals and selecting up to three for edits."
                    if use_tuple_search
                    else "Judging candidates and selecting up to three for edits."
                ),
            }
        )
        messages.append(
            _tool_call(
                "judge_candidates",
                {
                    "sample_id": sample_id,
                    "target_audio_path": str(target_audio_path),
                    "candidate_audio": candidate_assets if not use_tuple_search else None,
                    "tuple_candidates": (
                        [
                            {
                                "tuple_id": str(t["tuple_id"]),
                                "candidate_ids": [id_map[n] for n in tuple_members_by_id.get(str(t["tuple_id"]), []) if n in id_map],
                            }
                            for t in tuple_rows
                        ]
                        if use_tuple_search
                        else None
                    ),
                    "max_select": int(args.select_k),
                },
            )
        )
        # Include selected previews so the main agent can hear what was chosen before edits.
        if use_tuple_search:
            selected_previews = []
            for tid in selected_tuple_ids[: int(args.select_k)]:
                members = tuple_members_by_id.get(tid, [])
                preview = _ensure_tuple_preview_audio(
                    sample_id=sample_id,
                    tuple_id=tid,
                    member_names=members,
                    candidate_audio=candidate_audio,
                    out_dir=args.tuple_preview_dir,
                    cache=tuple_preview_cache,
                )
                if preview is not None:
                    tuple_preview_paths[tid] = preview
                    selected_previews.append(
                        {
                            "tuple_id": tid,
                            "candidate_ids": [id_map[m] for m in members if m in id_map],
                            "audio_preview": "<audio>",
                        }
                    )
            judge_response_content = {**judge_result, "selected_tuple_previews": selected_previews}
        else:
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
        steps_since_last_listen: int = 0  # tracks how many steps back the last listen was
        _snum_is_planning: dict[int, bool] = {}  # step_num → True if audio gate suppressed listen
        prev_commentary: str | None = None
        # Load target_preset — in-memory for old smoke-test paths, on-disk for
        # --generate mode paths which strip the dict and write target_preset_path instead.
        target_preset = path_data.get("target_preset")
        if target_preset is None and path_data.get("target_preset_path"):
            with open(path_data["target_preset_path"]) as _f:
                target_preset = json.load(_f)
        # Apply selected source tuple/wavetable so search output is a real action.
        _apply_preset_path = path_data.get("target_preset_path")
        selected_tuple_id: str | None = selected_tuple_ids[0] if selected_tuple_ids else None
        selected_tuple_members: list[str] = tuple_members_by_id.get(selected_tuple_id, []) if selected_tuple_id else []
        if _apply_preset_path:
            if use_tuple_search and selected_tuple_members:
                wt_payload = []
                for n in selected_tuple_members[:3]:
                    row = selected_by_name.get(n)
                    if not row:
                        continue
                    src_idx = int(row["source_wavetable_idx"])
                    wt_payload.append(wavetable_lib[src_idx])
                if wt_payload:
                    _apply_tuple_snippet = (
                        "import sys, json\n"
                        "sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
                        "from vital_tools import VitalController\n"
                        "vc = VitalController()\n"
                        "vc.discover()\n"
                        f"_wts = json.loads('''{json.dumps(wt_payload, ensure_ascii=False)}''')\n"
                        "preset = vc.get_preset()\n"
                        "for _i, _wt in enumerate(_wts[:3]):\n"
                        "    preset['settings']['wavetables'][_i] = _wt\n"
                        "vc.set_preset(preset)\n"
                        f"print(json.dumps({{'status': 'ok', 'applied_tuple_id': {json.dumps(selected_tuple_id)}, 'applied_wavetable_tuple': {json.dumps(selected_tuple_members[:3], ensure_ascii=False)}}}))"
                    )
                    _apply_tuple_snippet = _harden_vital_snippet_for_reapy(_apply_tuple_snippet)
                    messages.append({
                        "role": "assistant",
                        "content": (
                            f"Applying selected tuple {selected_tuple_id}: "
                            + ", ".join(selected_tuple_members[:3]) + "."
                        ),
                    })
                    messages.append(_tool_call("bash", {"command": _wrap_as_bash(_apply_tuple_snippet)}))
                    messages.append({
                        "role": "tool_response",
                        "content": json.dumps(
                            {
                                "status": "ok",
                                "applied_tuple_id": selected_tuple_id,
                                "applied_wavetable_tuple": selected_tuple_members[:3],
                            },
                            ensure_ascii=False,
                        ),
                    })
            elif (not use_tuple_search) and selected_names:
                # Select by judge order only (no GT override).
                _ranked_selected = [
                    name for cid in judge_result["selected"]
                    for name in candidate_names
                    if id_map[name] == cid
                ]
                _wt_name = _ranked_selected[0] if _ranked_selected else selected_names[0]
                _apply_wt_snippet = (
                    "import sys, json\n"
                    "sys.path.append('/home/nate/.config/REAPER/Scripts')\n"
                    "from vital_tools import VitalController\n"
                    "vc = VitalController()\n"
                    "vc.discover()\n"
                f"with open({json.dumps(str(_apply_preset_path))}) as _f:\n"
                "    _src = json.load(_f)\n"
                "_wt = _src['settings']['wavetables'][0]\n"
                "if _rpr is None:\n"
                "    preset = vc.get_preset()\n"
                "    preset['settings']['wavetables'][0] = _wt\n"
                "    vc.set_preset(preset)\n"
                f"print(json.dumps({{'status': 'ok', 'applied_wavetable': {json.dumps(_wt_name)}}}))"
            )
                _apply_wt_snippet = _harden_vital_snippet_for_reapy(_apply_wt_snippet)
                messages.append({
                    "role": "assistant",
                    "content": (
                        f"Having listened to the candidate previews, '{_wt_name}' matches the target "
                        f"most closely. Applying it to oscillator 1."
                    ),
                })
                messages.append(_tool_call("bash", {"command": _wrap_as_bash(_apply_wt_snippet)}))
                messages.append({
                    "role": "tool_response",
                    "content": json.dumps(
                        {"status": "ok", "applied_wavetable": _wt_name}, ensure_ascii=False
                    ),
                })

        step_gaps_cache: dict[int, dict | None] = {}  # step_num → remaining gap after that step

        # Build authoritative slot→route lookup from target preset (1-indexed slot number →
        # {source, destination, amount}).  Used to annotate modulation_N_* params with their
        # actual routing context rather than raw bypass/amount names.
        _target_mod_routes: dict[str, dict] = {}
        if target_preset:
            for _i, _mod in enumerate(target_preset.get("settings", {}).get("modulations", [])):
                _src = _mod.get("source", "")
                _dst = _mod.get("destination", "")
                if _src or _dst:
                    _slot_key = str(_i + 1)
                    _amt_key = f"modulation_{_slot_key}_amount"
                    _target_mod_routes[_slot_key] = {
                        "source": _src,
                        "destination": _dst,
                        "amount": float(target_preset["settings"].get(_amt_key, 0.0)),
                        "slot": _slot_key,
                    }

        for step in path_data.get("iterations", [])[: int(args.max_steps)]:
            step_num = int(step.get("step", 0))
            action_snippet = step.get("action_snippet") or step.get("python_script") or "print('noop')"
            action_snippet = _harden_vital_snippet_for_reapy(action_snippet)
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

            # Always increment before generating commentary.
            # Reset to 0 happens after the listen probe block (listening steps only).
            # This way, for a listening step that follows 2 planning steps, the value
            # is 3 (not 1), so Stage 1A correctly describes "the 3 edits since your
            # last listen" rather than "the parameter edit just applied".
            steps_since_last_listen += 1

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
                    modulations_changed=list(_target_mod_routes.values()),
                    is_mistake_step=bool(step.get("is_mistake_step", False)),
                    # planned_param_names is the closest proxy for allowed_params in path data
                    allowed_params=step.get("planned_param_names") or step.get("allowed_params"),
                    remaining_delta_context=remaining_delta_context,
                    # Stage 1A "before/after": use last_listened_iter_wav as the prior.
                    # When consecutive listening steps occur, last_listened_iter_wav == prev_iter_wav
                    # (both point at iter_{N-1}), which would make Stage 1A compare a file to itself.
                    # In that case fall back to prev_prev_iter_wav (iter_{N-2}) — a meaningful prior.
                    prior_iter_wav=(
                        prev_prev_iter_wav
                        if (last_listened_iter_wav is not None
                            and last_listened_iter_wav == prev_iter_wav)
                        else last_listened_iter_wav
                    ),
                    clap_delta=step_clap_deltas.get(step_num),
                    is_planning_step=not _is_listening_step,
                    planner_stage=step.get("planner_stage"),
                    steps_since_last_listen=steps_since_last_listen,
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
                commentary = _step_commentary_fallback(step, step_num, is_planning_step=not _is_listening_step)

            # When planning steps preceded this listening step, the last message in
            # context is {"status":"ok"} — not <audio>.  Emit a catch-up listen probe
            # so HEARD+HYPOTHESIS immediately follows an <audio> tool_response.
            if _is_listening_step and steps_since_last_listen > 1 and prev_iter_wav:
                _n_planning = steps_since_last_listen - 1  # planning steps before this one
                _first_planning = step_num - _n_planning
                _catchup_label = (
                    f"after steps {_first_planning}–{step_num - 1}"
                    if _n_planning > 1
                    else f"after step {step_num - 1}"
                )
                used_iter_audio_paths.append(prev_iter_wav)
                messages.append({
                    "role": "assistant",
                    "content": f"Listening to accumulated changes {_catchup_label}.",
                })
                messages.append(_tool_call("bash", {"command": _build_listen_probe_command(Path(prev_iter_wav))}))
                messages.append({
                    "role": "tool_response",
                    "content": json.dumps(
                        {"status": "ok", "iter_audio": "<audio>", "path": prev_iter_wav},
                        ensure_ascii=False,
                    ),
                })

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
                    {"status": "ok"},
                    ensure_ascii=False)})

            if step_num - 1 < len(iter_wavs):
                iter_audio = iter_wavs[step_num - 1]
                prev_prev_iter_wav = prev_iter_wav  # always advance pointer (needed for Stage 1B)
                prev_iter_wav = str(iter_audio)
                if _is_listening_step:
                    used_iter_audio_paths.append(str(iter_audio))
                    last_listened_iter_wav = str(iter_audio)
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
                    # Reset counter now that we've listened — next planning steps start
                    # accumulating from 0 again.
                    steps_since_last_listen = 0

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

        # If the last step was a planning step (never listened), insert a final listen turn
        # so the model hears its own last state before being asked to assess.
        if prev_iter_wav is not None and prev_iter_wav != last_listened_iter_wav:
            used_iter_audio_paths.append(prev_iter_wav)
            last_listened_iter_wav = prev_iter_wav
            messages.append({"role": "assistant", "content": "Listening to the final preset state."})
            messages.append(_tool_call("bash", {"command": _build_listen_probe_command(Path(prev_iter_wav))}))
            messages.append(
                {
                    "role": "tool_response",
                    "content": json.dumps(
                        {"status": "ok", "iter_audio": "<audio>", "path": prev_iter_wav},
                        ensure_ascii=False,
                    ),
                }
            )

        # Budget-exhausted case: inject a user message so the model knows to wrap up.
        # Convergence case: no user message — the model just listened and should conclude naturally.
        if not path_complete:
            messages.append({
                "role": "user",
                "content": "That completes the available iterations. Compare your final render to the target and give your final assessment.",
            })

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
                closing = "All planned iterations have been applied. The recreation path is complete."
            else:
                subs = list(convergence["by_subsystem"].keys())[:2]
                subs_str = ", ".join(subs) if subs else "several parameters"
                closing = f"Iterations exhausted — {subs_str} still differ from the target."
        messages.append({"role": "assistant", "content": closing})
        audio_assets = [str(target_audio_path)]
        if default_audio_path is not None:
            audio_assets.append(str(default_audio_path))
        if use_tuple_search:
            for tid in selected_tuple_ids[: int(args.select_k)]:
                if tid in tuple_preview_paths:
                    audio_assets.append(str(tuple_preview_paths[tid]))
        else:
            for n in selected_names[: int(args.select_k)]:
                if n in candidate_audio:
                    audio_assets.append(str(candidate_audio[n]))
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
                "selected_tuples": (
                    [
                        {
                            "tuple_id": tid,
                            "members": tuple_members_by_id.get(tid, []),
                            "audio_path": str(tuple_preview_paths[tid]) if tid in tuple_preview_paths else None,
                        }
                        for tid in selected_tuple_ids[: int(args.select_k)]
                    ]
                    if use_tuple_search
                    else []
                ),
            },
            "labels": {
                "judge_ranking": judge_result["ranking"],
                "judge_selected": judge_result["selected"],
                "gt_candidate_ids": [id_map[n] for n in candidate_names if n in gt_names],
                "selected_tuple_id": selected_tuple_id if use_tuple_search else None,
                "acceptable_tuple_ids": acceptable_tuple_ids if use_tuple_search else [],
                "tuple_members_by_id": tuple_members_by_id if use_tuple_search else {},
            },
            "meta": {
                "sample_id": sample_id,
                "archetype": str(entry.get("archetype", "synth")),
                "start_type": start_type,
                "agent": "main",
                "num_agents": int(args.num_agents),
                "candidate_source": args.candidate_source,
                "tuple_search_mode": use_tuple_search,
                "score_visibility": "hidden" if use_tuple_search else "explicit",
                "teacher_tuple_scores": tuple_scores if use_tuple_search else {},
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
                        "clap_delta": step_clap_deltas.get(int(s.get("step", 0))),
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
        if use_tuple_search:
            _assert_no_similarity_score_leak(messages)
        assert_valid_ms_swift_multiturn_record(record)
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
            print(f"[{n_done}/{len(entries)}] {sid}: {len(recs)} block(s) written ({total_blocks_written} total)", flush=True)

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
    print(f"Wrote {len(records)} records ({n_source} samples → {len(records)} blocks) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
