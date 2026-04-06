#!/usr/bin/env python3
"""Assemble multi-turn ms-swift JSONL from a render manifest + Omni commentary."""

import argparse
import asyncio
import base64
from collections import Counter
import hashlib
import json
import os
import random
import re
import sys
from pathlib import Path

import httpx

# Rough human-readable label for common param name prefixes
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

_SYNTHETIC_PROJECT_STATE = json.dumps({
    "tracks": [
        {"idx": 0, "name": "Reference"},
        {"idx": 1, "name": "Vital Synth", "fx": "Vital"},
    ]
})

_LISTEN_CMD = "python scripts/reaper_render_probe.py && aplay /tmp/probe.wav"
_LISTEN_RESULT = "<audio>\nRendered to /tmp/probe.wav"
_SEARCH_AGENT_TOOL_NAME = "search_agent"
_SECTION_HEADERS = ("HEARD:", "HYPOTHESIS:", "PLAN:")
_SEARCH_ABBREVS = {
    "osc": "oscillator",
    "env": "envelope",
    "lfo": "lfo",
    "eq": "eq",
}
_ARCHETYPE_TOKENS = {"bass", "lead", "pad", "pluck", "keys"}
_PLANNER_FAMILY_LABELS = {
    "osc": "oscillator",
    "env1": "envelope 1",
    "env2": "envelope 2",
    "filter1": "filter 1",
    "filter2": "filter 2",
    "lfo": "lfo",
    "reverb": "reverb",
    "delay": "delay",
    "chorus": "chorus",
    "distortion": "distortion",
    "compressor": "compressor",
    "phaser": "phaser",
    "flanger": "flanger",
    "eq": "eq",
    "modulation": "modulation",
    "other": "synth controls",
}
_FAMILY_TO_SEARCH_KEYWORD = {
    "osc": "oscillator",
    "env1": "envelope 1",
    "env2": "envelope 2",
    "filter1": "filter 1",
    "filter2": "filter 2",
    "lfo": "lfo",
    "reverb": "reverb",
    "delay": "delay",
    "chorus": "chorus",
    "distortion": "distortion",
    "compressor": "compressor",
    "phaser": "phaser",
    "flanger": "flanger",
    "eq": "eq",
    "modulation": "modulation",
    "other": "synth",
}
_BAD_SEARCH_TYPES = (
    "empty_false_negative",
    "off_family_false_positive",
    "stale_values",
    "truncated_output",
)


def _load_default_norms() -> dict[str, float]:
    """Load default normalized values from param_ranges.json.

    Falls back to an empty dict if the file is unavailable.
    """
    param_ranges_path = Path(__file__).resolve().parents[1] / "maestro" / "synth" / "param_ranges.json"
    try:
        with open(param_ranges_path) as f:
            ranges = json.load(f)
    except Exception:
        return {}

    defaults: dict[str, float] = {}
    for name, r in ranges.items():
        try:
            span = float(r["max"]) - float(r["min"])
            if span == 0:
                defaults[name] = 0.0
            else:
                defaults[name] = max(0.0, min(1.0, (float(r["default"]) - float(r["min"])) / span))
        except Exception:
            continue
    return defaults


_DEFAULT_NORMS = _load_default_norms()


def _param_label(name: str) -> str:
    for prefix, label in _PARAM_LABELS.items():
        if name.startswith(prefix):
            return label
    return "synth parameter"


def _format_delta_context(params_delta: list, is_mistake_step: bool) -> str:
    """Format params_delta list into a string for the Omni prompt."""
    if not params_delta:
        return ""

    lines = []
    mistake_params = []
    for d in params_delta:
        name = d["name"]
        display = _json_key_to_display(name)
        from_n = d["from_norm"]
        to_n = d["to_norm"]
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


def _prepare_delta_maps(path_data: dict) -> dict:
    """Build compact per-sample maps for residual-delta tracking.

    Returns:
      - target_norm: target value per tracked parameter
      - initial_norm: baseline value before step 1
      - applied_per_step: list of per-step applied values
    """
    target_norm: dict[str, float] = {}
    initial_norm: dict[str, float] = {}
    applied_per_step: list[dict[str, float]] = []

    for it in path_data.get("iterations", []):
        params_delta = it.get("params_delta", []) or []
        for d in params_delta:
            name = d.get("name")
            if not name:
                continue
            if name not in initial_norm and d.get("from_norm") is not None:
                initial_norm[name] = float(d["from_norm"])
            if d.get("target_norm") is not None:
                target_norm[name] = float(d["target_norm"])

        # Backfill target values from params_changed when target_norm is absent.
        params_changed = it.get("params_changed", {})
        if isinstance(params_changed, dict):
            for name, val in params_changed.items():
                try:
                    target_norm.setdefault(name, float(val))
                except Exception:
                    continue

        # Prefer explicit params_applied; fallback to params_delta to_norm.
        params_applied = it.get("params_applied", {})
        step_applied: dict[str, float] = {}
        if isinstance(params_applied, dict) and params_applied:
            for name, val in params_applied.items():
                try:
                    step_applied[name] = float(val)
                except Exception:
                    continue
        else:
            for d in params_delta:
                name = d.get("name")
                if not name or d.get("to_norm") is None:
                    continue
                step_applied[name] = float(d["to_norm"])
        applied_per_step.append(step_applied)

    for name in target_norm:
        initial_norm.setdefault(name, _DEFAULT_NORMS.get(name, 0.5))

    return {
        "target_norm": target_norm,
        "initial_norm": initial_norm,
        "applied_per_step": applied_per_step,
    }


def _format_remaining_delta_context(step_num: int, delta_maps: dict, top_k: int = 10) -> str:
    """Summarize current-vs-target parameter gaps before a step is applied."""
    target_norm: dict[str, float] = delta_maps.get("target_norm", {})
    initial_norm: dict[str, float] = delta_maps.get("initial_norm", {})
    applied_per_step: list[dict[str, float]] = delta_maps.get("applied_per_step", [])
    if not target_norm:
        return ""

    # step_num is 1-indexed. Apply steps [0, step_num-2] to get current state.
    current_norm = dict(initial_norm)
    for i in range(max(0, step_num - 1)):
        if i >= len(applied_per_step):
            break
        for name, val in applied_per_step[i].items():
            if name in current_norm:
                current_norm[name] = val

    unresolved: list[tuple[float, str, float, float]] = []
    for name, tgt in target_norm.items():
        cur = current_norm.get(name, initial_norm.get(name, _DEFAULT_NORMS.get(name, 0.5)))
        diff = tgt - cur
        if abs(diff) <= 0.01:
            continue
        unresolved.append((abs(diff), name, cur, tgt))
    unresolved.sort(reverse=True)

    lines = [
        (
            f"Current preset vs target before step {step_num}: "
            f"{len(unresolved)}/{len(target_norm)} tracked changed parameters still differ (|Δ|>0.01)."
        )
    ]
    if not unresolved:
        lines.append("Largest unresolved groups: none.")
        return "\n".join(lines)

    group_counts = Counter(_param_label(name) for _, name, _, _ in unresolved)
    groups = ", ".join(f"{label}={count}" for label, count in group_counts.most_common(6))
    lines.append(f"Largest unresolved groups: {groups}.")
    lines.append("Largest unresolved parameter gaps:")
    for mag, name, cur, tgt in unresolved[:top_k]:
        relation = "below" if tgt > cur else "above"
        lines.append(
            f"- {_json_key_to_display(name)} ({_param_label(name)}): "
            f"current is {relation} target by {mag:.2f} [{cur:.2f}→{tgt:.2f}]"
        )
    return "\n".join(lines)


def _build_param_summary(it: dict | None) -> str:
    """Build a short human-readable param-change summary from an iteration dict.

    Uses ``params_delta`` (list of dicts with 'name'/'from_norm'/'to_norm') when
    available, falling back to the simpler ``params_changed`` list.  Returns an
    empty string when no useful data is present.
    """
    if not it:
        return ""

    # Prefer the richer list form: [{"name": ..., "from_norm": ..., "to_norm": ...}]
    delta_list = it.get("params_delta", [])
    if delta_list:
        top = sorted(delta_list, key=lambda d: abs(d.get("to_norm", 0) - d.get("from_norm", 0)), reverse=True)[:6]
        parts = []
        for d in top:
            name = d["name"]
            diff = d.get("to_norm", 0) - d.get("from_norm", 0)
            arrow = "↑" if diff > 0 else "↓"
            parts.append(f"{_json_key_to_display(name)} {arrow} {abs(diff):.2f}")
        return "; ".join(parts)

    # Fallback: simple dict form {name: delta}
    delta_dict = it.get("params_delta_dict", {})
    if delta_dict:
        top = sorted(delta_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:6]
        return "; ".join(
            f"{_json_key_to_display(name)} {'↑' if d > 0 else '↓'} {abs(d):.2f}"
            for name, d in top
        )

    # Last resort: just list the changed param names
    changed = it.get("params_changed", [])
    if changed:
        return "params changed: " + ", ".join(changed[:6])

    return ""


def _json_key_to_display(key: str) -> str:
    """Convert Vital JSON key style to REAPER display-style words."""
    parts = key.split("_")
    words = []
    for p in parts:
        low = p.lower()
        if low in _SEARCH_ABBREVS:
            words.append(_SEARCH_ABBREVS[low].capitalize() if low != "lfo" else "LFO")
        elif p.isdigit():
            words.append(p)
        else:
            words.append(p.capitalize())
    return " ".join(words)


def _infer_search_keyword(step_data: dict) -> str:
    """Infer a compact keyword like 'filter 1' or 'oscillator 2' from step params."""
    if step_data.get("search_keyword"):
        kw = str(step_data["search_keyword"]).strip().lower()
        if kw:
            return kw

    names = []
    if isinstance(step_data.get("params_applied"), dict):
        names.extend(step_data["params_applied"].keys())
    if not names and isinstance(step_data.get("params_delta"), list):
        names.extend([d.get("name") for d in step_data["params_delta"] if d.get("name")])
    if not names:
        return "filter"

    buckets: Counter[str] = Counter()
    for name in names:
        parts = name.split("_")
        if len(parts) >= 2 and parts[1].isdigit():
            head = _SEARCH_ABBREVS.get(parts[0], parts[0])
            buckets[f"{head} {parts[1]}"] += 1
        else:
            head = _SEARCH_ABBREVS.get(parts[0], parts[0])
            buckets[head] += 1
    kw = buckets.most_common(1)[0][0].lower().strip()
    return kw if kw else "filter"


def _sanitize_search_keyword(keyword: str | None) -> str:
    kw = re.sub(r"\s+", " ", str(keyword or "").strip().lower())
    return kw if kw else "filter"


def _modulation_index_from_keyword(keyword: str) -> str | None:
    m = re.fullmatch(r"modulation\s+(\d+)", _sanitize_search_keyword(keyword))
    return m.group(1) if m else None


def _display_matches_keyword(display: str, keyword: str) -> bool:
    """Keyword matching with exact behavior for modulation slot searches."""
    display_l = display.lower()
    kw = _sanitize_search_keyword(keyword)
    mod_idx = _modulation_index_from_keyword(kw)
    if mod_idx is not None:
        return display_l.startswith(f"modulation {mod_idx} ")
    return kw in display_l


def _json_name_matches_keyword(name: str, keyword: str) -> bool:
    return _display_matches_keyword(_json_key_to_display(name), keyword)


def _search_focus_display(keyword: str) -> str:
    kw = _sanitize_search_keyword(keyword)
    mod_idx = _modulation_index_from_keyword(kw)
    if mod_idx is not None:
        return f"modulation slot {mod_idx}"
    return kw


def _build_search_snippet(keyword: str) -> str:
    """Generate reapy Python search snippet for a keyword."""
    kw = _sanitize_search_keyword(keyword).replace("'", "\\'")
    mod_idx = _modulation_index_from_keyword(kw)
    if mod_idx is not None:
        filter_expr = f"p.name.lower().startswith('modulation {mod_idx} ')"
    else:
        filter_expr = f"'{kw}' in p.name.lower()"
    return (
        "import reapy\n"
        "with reapy.inside_reaper():\n"
        "    fx = reapy.Project().tracks[0].fxs[0]\n"
        "    hits = []\n"
        "    for p in fx.params:\n"
        f"        if {filter_expr}:\n"
        "            hits.append((p.name, round(p.normalized, 4)))\n"
        "    for name, val in hits[:16]:\n"
        '        print(f"{name}: {val:.4f}")'
    )


def _build_search_result_from_step(step_data: dict, keyword: str) -> str:
    """Synthesize search output from step metadata when no explicit result exists."""
    raw_kw = str(step_data.get("search_keyword") or "").strip()
    if step_data.get("search_result") and raw_kw:
        cached = str(step_data["search_result"])
        # Old cached modulation searches used broad substring matching (e.g., "modulation 1").
        # Re-filter to exact slot matching to avoid giant noisy result dumps.
        if _modulation_index_from_keyword(raw_kw):
            lines = [ln for ln in cached.splitlines() if ":" in ln]
            exact = [ln for ln in lines if _display_matches_keyword(ln.split(":", 1)[0].strip(), raw_kw)]
            if exact:
                return "\n".join(exact[:16])
        return "\n".join(cached.splitlines()[:16])

    lines: list[str] = []
    for d in step_data.get("params_delta", []) or []:
        name = d.get("name")
        if not name:
            continue
        display = _json_key_to_display(name)
        if not _display_matches_keyword(display, keyword):
            continue
        v = d.get("from_norm")
        if v is None:
            v = d.get("to_norm", 0.0)
        try:
            lines.append(f"{display}: {float(v):.4f}")
        except Exception:
            continue

    if not lines:
        # Fallback: show a small pre-step view of changed params even if keyword is broad.
        for d in (step_data.get("params_delta", []) or [])[:12]:
            name = d.get("name")
            if not name:
                continue
            display = _json_key_to_display(name)
            v = d.get("from_norm")
            if v is None:
                v = d.get("to_norm", 0.0)
            try:
                lines.append(f"{display}: {float(v):.4f}")
            except Exception:
                continue
    return "\n".join(lines[:16])


def _is_unbounded_search_snippet(snippet: str) -> bool:
    text = snippet or ""
    return bool(
        re.search(r"if\s+['\"]\s*['\"]\s+in\s+p\.name\.lower\(\)", text)
        or re.search(r"if\s+True\s*:", text)
    )


def _choose_search_snippet(step_data: dict, keyword: str) -> str:
    keyword = _sanitize_search_keyword(keyword)
    candidate = str(step_data.get("search_snippet") or "").strip()
    if not candidate:
        return _build_search_snippet(keyword)
    if _is_unbounded_search_snippet(candidate):
        return _build_search_snippet(keyword)
    if "p.name.lower()" not in candidate:
        return _build_search_snippet(keyword)
    mod_idx = _modulation_index_from_keyword(keyword)
    if mod_idx is not None and f"startswith('modulation {mod_idx} ')" not in candidate:
        return _build_search_snippet(keyword)
    return candidate


def _extract_section_text(commentary: str, section: str) -> str:
    """Extract section body for HEARD/HYPOTHESIS/PLAN from model output."""
    text = commentary.replace("*", "")
    pattern = rf"{section}\s*(.*?)(?=(HEARD:|HYPOTHESIS:|PLAN:|$))"
    m = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    return (m.group(1).strip() if m else "")


def _normalize_commentary(commentary: str) -> str:
    """Normalize style to plain sectioned text (no markdown emphasis)."""
    text = commentary.replace("**", "").strip()
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def _find_programmatic_tokens(text: str) -> list[str]:
    """Find snake_case tokens that look like internal parameter ids."""
    return sorted(set(re.findall(r"\b[a-z]+(?:_[a-z0-9]+){1,}\b", text)))


def _plan_mentions_search_focus(plan: str, search_keyword: str) -> bool:
    """Check that PLAN references the same control family we will search."""
    kw = _sanitize_search_keyword(search_keyword)
    if not kw:
        return True
    plan_l = plan.lower()
    mod_idx = _modulation_index_from_keyword(kw)
    if mod_idx is not None:
        return bool(re.search(rf"\bmodulation(?:\s+slot)?\s+{re.escape(mod_idx)}\b", plan_l))
    if kw in plan_l:
        return True

    kw_tokens = [t for t in re.findall(r"[a-z0-9]+", kw) if t]
    if not kw_tokens:
        return True

    head = kw_tokens[0]
    synonyms = {head}
    if head == "osc":
        synonyms.add("oscillator")
    if head == "env":
        synonyms.add("envelope")
    if head == "lfo":
        synonyms.add("lfo")
    if head == "eq":
        synonyms.add("eq")

    nums = [t for t in kw_tokens if t.isdigit()]
    has_head = any(s in plan_l for s in synonyms)
    has_nums = all(n in plan_l for n in nums)
    return has_head and has_nums


def _validate_archetype_language(commentary: str, archetype: str) -> tuple[bool, str]:
    """Reject explicit mismatched preset-type labels (pad/lead/bass/etc)."""
    expected = (archetype or "").lower().strip()
    if expected not in _ARCHETYPE_TOKENS:
        return True, ""

    text = commentary.lower()
    patterns = [
        r"\b(bass|lead|pad|pluck|keys)(?:-like)?\s+(preset|patch|sound)\b",
        r"\blike\s+(?:a|an)\s+(bass|lead|pad|pluck|keys)\b",
    ]
    mentions: set[str] = set()
    for pat in patterns:
        for m in re.finditer(pat, text):
            token = m.group(1)
            mentions.add(token)
    bad = sorted(t for t in mentions if t != expected)
    if bad:
        return False, f"archetype mismatch in commentary: expected '{expected}', found '{bad[0]}'"
    return True, ""


def _family_to_scope_category(family: str | None) -> str:
    f = (family or "").lower().strip()
    if f in {"env1", "env2"}:
        return "envelope"
    if f in {"filter1", "filter2"}:
        return "filter"
    if f == "osc":
        return "oscillator"
    return _planner_family_label(f).split()[0]


def _validate_plan_family_scope(plan: str, primary_family: str, support_family: str | None) -> tuple[bool, str]:
    primary_cat = _family_to_scope_category(primary_family)
    support_cat = _family_to_scope_category(support_family) if support_family else ""
    allowed = {c for c in (primary_cat, support_cat) if c and c != "synth"}
    if not allowed:
        return True, ""

    cat_patterns = {
        "oscillator": r"\boscillator\b",
        "envelope": r"\benvelope\b",
        "filter": r"\bfilter\b",
        "lfo": r"\blfo\b",
        "reverb": r"\breverb\b",
        "delay": r"\bdelay\b",
        "chorus": r"\bchorus\b",
        "distortion": r"\bdistortion\b",
        "compressor": r"\bcompressor\b",
        "phaser": r"\bphaser\b",
        "flanger": r"\bflanger\b",
        "eq": r"\beq\b",
        "modulation": r"\bmodulation\b",
    }
    plan_l = plan.lower()
    mentioned = {cat for cat, pat in cat_patterns.items() if re.search(pat, plan_l)}
    bad = sorted(m for m in mentioned if m not in allowed)
    if bad:
        return False, f"plan mentions unrelated control families ({', '.join(bad[:3])}); allowed: {', '.join(sorted(allowed))}"
    return True, ""


def _validate_commentary(
    commentary: str,
    allowed_params: set[str],
    search_keyword: str = "",
    archetype: str = "synth",
    primary_family: str = "",
    support_family: str | None = None,
) -> tuple[bool, str]:
    """Gate commentary quality so synthetic data stays compact and grounded."""
    normalized = _normalize_commentary(commentary)
    upper = normalized.upper()

    for h in _SECTION_HEADERS:
        if h not in upper:
            return False, f"missing required section header '{h}'"
        if upper.count(h) != 1:
            return False, f"expected exactly one '{h}' section"

    if len(normalized) > 1200:
        return False, "too long; keep sections compact"

    snake_case_tokens = _find_programmatic_tokens(normalized)
    if snake_case_tokens:
        return False, f"use human-readable control names, not snake_case ids: {', '.join(snake_case_tokens[:6])}"

    hypothesis = _extract_section_text(normalized, "HYPOTHESIS:")
    uncertainty_words = ("likely", "probably", "may", "might", "could", "seems")
    if not any(w in hypothesis.lower() for w in uncertainty_words):
        return False, "hypothesis must use uncertainty language (likely/probably/may/might/could/seems)"

    plan = _extract_section_text(normalized, "PLAN:")
    plan_l = plan.lower()
    if not any(w in plan_l for w in ("inspect", "search", "check", "look", "find")):
        return False, "plan must state what controls will be inspected/searched first"
    if not _plan_mentions_search_focus(plan, search_keyword):
        return False, f"plan must reference the search focus for this step ('{search_keyword}')"
    mod_idx = _modulation_index_from_keyword(search_keyword)
    if mod_idx is not None and re.search(rf"\blfo\s+{re.escape(mod_idx)}\b", plan_l):
        return False, f"for search focus '{search_keyword}', PLAN should use modulation-slot wording (not LFO {mod_idx})"
    scope_ok, scope_reason = _validate_plan_family_scope(plan, primary_family, support_family)
    if not scope_ok:
        return False, scope_reason

    # Pre-search PLAN should stay at control-family level (don't claim exact controls yet).
    if allowed_params:
        exact_controls = []
        for name in allowed_params:
            display = _json_key_to_display(name).lower()
            if len(display.split()) >= 3 and display in plan_l:
                exact_controls.append(display)
        if exact_controls:
            return False, "plan should reference control families before search results, not exact control names"

    # Defensive check for snake_case ids that slipped through normalization.
    if allowed_params:
        mentioned = set(re.findall(r"\b[a-z]+(?:_[a-z0-9]+){1,}\b", plan))
        bad = sorted(p for p in mentioned if p not in allowed_params)
        if bad:
            return False, f"PLAN mentions params not in this step: {', '.join(bad[:6])}"

    archetype_ok, archetype_reason = _validate_archetype_language(normalized, archetype)
    if not archetype_ok:
        return False, archetype_reason

    return True, ""


def _fallback_commentary(it: dict | None, search_keyword: str = "target controls") -> str:
    """Deterministic fallback if Omni output repeatedly fails quality gates."""
    focus = _search_focus_display(search_keyword)
    return (
        "HEARD: The current preset is closer but still differs from the target in tone, movement, and space.\n\n"
        "HYPOTHESIS: The remaining mismatch likely comes from unresolved oscillator balance, filter shape, and FX dynamics.\n\n"
        f"PLAN: I will inspect {focus} controls first, apply this step's programmed updates, and then listen again."
    )


def _build_set_turn_content(
    step_num: int,
    keyword: str,
    params_delta: list,
    is_mistake: bool,
    allowed_primary_names: list[str] | None = None,
    planned_primary_names: list[str] | None = None,
) -> str:
    """Compact, varied set-turn narration grounded in step metadata."""
    focus = keyword or "target"
    deltas = [d for d in (params_delta or []) if d.get("name")]
    keyword_matched = [d for d in deltas if _json_name_matches_keyword(d["name"], focus)]

    allowed_set = set(allowed_primary_names or [])
    if allowed_set:
        keyword_matched = [d for d in keyword_matched if d["name"] in allowed_set]

    preferred_set = set(planned_primary_names or [])
    preferred = [d for d in keyword_matched if d["name"] in preferred_set] if preferred_set else []
    pool = preferred if preferred else keyword_matched
    top = sorted(
        pool,
        key=lambda d: abs(float(d.get("to_norm", 0.0)) - float(d.get("from_norm", 0.0))),
        reverse=True,
    )[:2]
    controls = ", ".join(_json_key_to_display(d["name"]) for d in top)
    if is_mistake:
        base = f"Step {step_num}: correcting over-adjusted {focus} controls now."
    else:
        base = f"Step {step_num}: applying the planned {focus} updates now."
    if controls:
        return f"{base} Primary edits: {controls}."
    return base


def _planner_family_label(family: str | None) -> str:
    if not family:
        return "synth controls"
    return _PLANNER_FAMILY_LABELS.get(str(family), str(family).replace("_", " "))


def _build_search_interpretation(keyword: str, search_result: str) -> str:
    """One-line interpretation from search output before applying a set step."""
    rows = [ln for ln in (search_result or "").splitlines() if ":" in ln]
    if not rows:
        return f"Search check: no direct matches for {keyword}; proceeding with the planned update group."
    preview = []
    for ln in rows[:2]:
        name, val = ln.split(":", 1)
        preview.append(f"{name.strip()}={val.strip()}")
    return (
        f"Search check: found {len(rows)} {keyword} controls; "
        f"key values {', '.join(preview)}."
    )


def _family_search_keyword(family: str | None) -> str:
    return _FAMILY_TO_SEARCH_KEYWORD.get(str(family or "").lower().strip(), "filter")


def _generalize_search_keyword(keyword: str) -> str:
    kw = _sanitize_search_keyword(keyword)
    mod_idx = _modulation_index_from_keyword(kw)
    if mod_idx is not None:
        return "modulation"
    parts = kw.split()
    if len(parts) >= 2 and parts[1].isdigit():
        return parts[0]
    return kw


def _build_search_queries(
    primary_keyword: str,
    primary_family: str,
    support_family: str | None,
    fanout: int,
) -> list[str]:
    fanout = max(1, int(fanout))
    queries: list[str] = []

    def _add(query: str | None) -> None:
        q = _sanitize_search_keyword(query)
        if q and q not in queries:
            queries.append(q)

    _add(primary_keyword)
    _add(_generalize_search_keyword(primary_keyword))
    _add(_family_search_keyword(primary_family))
    if support_family:
        _add(_family_search_keyword(support_family))
    _add("oscillator")
    _add("filter")
    return queries[:fanout] if queries else ["filter"]


def _controls_from_search_result(search_result: str, max_rows: int = 16) -> list[dict]:
    controls: list[dict] = []
    for ln in (search_result or "").splitlines():
        if ":" not in ln:
            continue
        name, val = ln.split(":", 1)
        try:
            norm = float(val.strip())
        except Exception:
            continue
        controls.append({"name": name.strip(), "value_norm": max(0.0, min(1.0, norm))})
        if len(controls) >= max_rows:
            break
    return controls


def _controls_to_search_result(controls: list[dict], max_rows: int = 16) -> str:
    rows: list[str] = []
    for c in controls[:max_rows]:
        name = str(c.get("name", "")).strip()
        try:
            val = float(c.get("value_norm", 0.0))
        except Exception:
            continue
        if not name:
            continue
        rows.append(f"{name}: {max(0.0, min(1.0, val)):.4f}")
    return "\n".join(rows)


def _build_synthetic_search_result_for_query(step_data: dict, keyword: str) -> str:
    target_kw = _sanitize_search_keyword(keyword)
    raw_kw = _sanitize_search_keyword(step_data.get("search_keyword"))
    if raw_kw == target_kw and step_data.get("search_result"):
        return _build_search_result_from_step(step_data, target_kw)

    lines: list[str] = []
    for d in step_data.get("params_delta", []) or []:
        name = d.get("name")
        if not name or not _json_name_matches_keyword(name, target_kw):
            continue
        display = _json_key_to_display(name)
        v = d.get("from_norm")
        if v is None:
            v = d.get("to_norm", 0.0)
        try:
            lines.append(f"{display}: {float(v):.4f}")
        except Exception:
            continue

    if not lines:
        for d in (step_data.get("params_delta", []) or [])[:10]:
            name = d.get("name")
            if not name:
                continue
            display = _json_key_to_display(name)
            v = d.get("from_norm")
            if v is None:
                v = d.get("to_norm", 0.0)
            try:
                lines.append(f"{display}: {float(v):.4f}")
            except Exception:
                continue

    return "\n".join(lines[:16])


def _build_search_report(
    step_data: dict,
    query: str,
    agent_idx: int,
    primary_query: str,
) -> dict:
    search_snippet = (
        _choose_search_snippet(step_data, query)
        if _sanitize_search_keyword(query) == _sanitize_search_keyword(primary_query)
        else _build_search_snippet(query)
    )
    search_result = _build_synthetic_search_result_for_query(step_data, query)
    controls = _controls_from_search_result(search_result)
    status = "ok" if controls else "low_confidence"
    confidence = 0.86 if controls else 0.32
    return {
        "agent_id": f"sa_{agent_idx}",
        "query": _sanitize_search_keyword(query),
        "status": status,
        "confidence": confidence,
        "controls": controls[:8],
        "issues": [],
        "latency_ms": 35 + (agent_idx * 13),
        "search_snippet": search_snippet,
        "search_result": search_result,
        "bad_type": "",
    }


def _apply_bad_search_noise(report: dict, step_data: dict, rng: random.Random) -> None:
    bad_type = rng.choice(_BAD_SEARCH_TYPES)
    report["bad_type"] = bad_type
    report["status"] = "low_confidence"
    report["confidence"] = min(float(report.get("confidence", 0.5)), 0.45)
    issues = list(report.get("issues") or [])

    if bad_type == "empty_false_negative":
        report["controls"] = []
        report["search_result"] = ""
        issues.append("no_matches_found")
    elif bad_type == "off_family_false_positive":
        off_controls: list[dict] = []
        query = str(report.get("query", ""))
        for d in step_data.get("params_delta", []) or []:
            name = d.get("name")
            if not name or _json_name_matches_keyword(name, query):
                continue
            display = _json_key_to_display(name)
            try:
                val = float(d.get("from_norm", d.get("to_norm", 0.5)))
            except Exception:
                continue
            off_controls.append({"name": display, "value_norm": max(0.0, min(1.0, val))})
            if len(off_controls) >= 8:
                break
        if not off_controls:
            off_controls = [{"name": "Compressor Mix", "value_norm": 0.5000}]
        report["controls"] = off_controls
        report["search_result"] = _controls_to_search_result(off_controls)
        issues.append("off_family_hits")
    elif bad_type == "stale_values":
        controls = list(report.get("controls") or [])
        if not controls:
            controls = [{"name": "Filter 1 Cutoff", "value_norm": 0.5000}]
        for c in controls:
            c["value_norm"] = max(0.0, min(1.0, float(c["value_norm"]) + rng.uniform(-0.2, 0.2)))
        report["controls"] = controls
        report["search_result"] = _controls_to_search_result(controls)
        issues.append("stale_values")
    else:
        lines = [ln for ln in str(report.get("search_result", "")).splitlines() if ln.strip()]
        if not lines:
            lines = ["[truncated output]"]
        else:
            lines = lines[:1] + ["[truncated output]"]
        report["search_result"] = "\n".join(lines)
        report["controls"] = _controls_from_search_result(report["search_result"], max_rows=1)
        issues.append("truncated_output")

    report["issues"] = issues


def _build_search_fanout_bundle(
    step_data: dict,
    step_num: int,
    primary_keyword: str,
    primary_family: str,
    support_family: str | None,
    fanout: int,
    bad_result_prob: float,
    bad_result_max: int,
    rng: random.Random,
) -> dict:
    queries = _build_search_queries(primary_keyword, primary_family, support_family, fanout)
    reports = [
        _build_search_report(step_data, q, idx + 1, primary_keyword)
        for idx, q in enumerate(queries)
    ]

    bad_result_prob = max(0.0, min(1.0, float(bad_result_prob)))
    bad_result_max = max(0, int(bad_result_max))
    injected_bad = 0
    if reports and bad_result_prob > 0.0 and bad_result_max > 0:
        order = list(range(len(reports)))
        rng.shuffle(order)
        for idx in order:
            if injected_bad >= bad_result_max:
                break
            if rng.random() <= bad_result_prob:
                _apply_bad_search_noise(reports[idx], step_data, rng)
                injected_bad += 1

    ok_reports = [r for r in reports if r.get("status") == "ok"]
    best_report = (
        max(ok_reports, key=lambda r: float(r.get("confidence", 0.0)))
        if ok_reports
        else max(reports, key=lambda r: float(r.get("confidence", 0.0)))
    )
    consensus_controls = list(best_report.get("controls") or [])[:6]

    value_map: dict[str, list[float]] = {}
    for report in reports:
        for c in report.get("controls", []):
            value_map.setdefault(str(c.get("name", "")).strip(), []).append(float(c.get("value_norm", 0.0)))
    conflicts = []
    for name, vals in value_map.items():
        if len(vals) < 2:
            continue
        spread = max(vals) - min(vals)
        if spread > 0.08:
            conflicts.append({
                "name": name,
                "spread": round(spread, 4),
                "values": [round(v, 4) for v in vals[:4]],
            })
    conflicts = conflicts[:6]

    summary = {
        "ok": sum(1 for r in reports if r.get("status") == "ok"),
        "low_confidence": sum(1 for r in reports if r.get("status") == "low_confidence"),
        "failed": sum(1 for r in reports if r.get("status") == "failed"),
        "bad_injected": injected_bad,
    }

    bundle = {
        "step": step_num,
        "primary_query": _sanitize_search_keyword(primary_keyword),
        "queries": queries,
        "reports": [
            {
                "agent_id": r["agent_id"],
                "query": r["query"],
                "status": r["status"],
                "confidence": round(float(r["confidence"]), 3),
                "controls": r["controls"],
                "issues": list(r.get("issues") or []),
                "latency_ms": int(r.get("latency_ms", 0)),
                "bad_type": r.get("bad_type", ""),
            }
            for r in reports
        ],
        "consensus_controls": consensus_controls,
        "conflicts": conflicts,
        "quality_summary": summary,
    }

    return {
        "queries": queries,
        "reports": reports,
        "bundle": bundle,
        "primary_search_snippet": reports[0]["search_snippet"],
        "primary_search_result": reports[0]["search_result"],
        "tool_content": json.dumps(bundle, ensure_ascii=False),
    }


def _build_search_handoff_interpretation(bundle: dict, keyword: str) -> str:
    summary = bundle.get("quality_summary", {})
    reports = list(bundle.get("reports") or [])
    if not reports:
        return f"Search handoff: no reports for {keyword}; applying planned updates conservatively."

    best = max(reports, key=lambda r: float(r.get("confidence", 0.0)))
    controls = list(bundle.get("consensus_controls") or [])[:2]
    if controls:
        preview = ", ".join(
            f"{c['name']}={float(c['value_norm']):.4f}"
            for c in controls
            if c.get("name") is not None
        )
    else:
        preview = "none"
    return (
        "Search handoff: "
        f"{len(reports)} agents (ok={summary.get('ok', 0)}, "
        f"low_confidence={summary.get('low_confidence', 0)}, failed={summary.get('failed', 0)}, "
        f"bad_injected={summary.get('bad_injected', 0)}). "
        f"Using {best.get('agent_id', 'sa_1')} on {best.get('query', keyword)} "
        f"(confidence {float(best.get('confidence', 0.0)):.2f}); consensus {preview}."
    )


def _build_main_search_tool_call(
    step_num: int,
    keyword: str,
    queries: list[str] | None,
    search_snippet: str,
    mode: str,
) -> dict:
    """Build the main-agent search tool call in bash or delegated-agent mode."""
    if mode == "search_agent":
        focus = _search_focus_display(keyword)
        arguments = {
            "step": step_num,
            "query": keyword,
            "queries": list(queries or [keyword]),
            "parallelism": max(1, len(list(queries or [keyword]))),
            "output_contract": "json_bundle_with_reports_and_consensus",
            "goal": (
                f"Inspect {focus} controls in Vital with parallel search agents "
                "and return structured reports plus a consensus handoff."
            ),
        }
        return {
            "id": f"tc_search_{step_num}",
            "type": "function",
            "function": {
                "name": _SEARCH_AGENT_TOOL_NAME,
                "arguments": json.dumps(arguments),
            },
        }
    return {
        "id": f"tc_search_{step_num}",
        "type": "function",
        "function": {
            "name": "bash",
            "arguments": json.dumps({"command": search_snippet}),
        },
    }


def _build_search_agent_record(
    sample_id: str,
    step_num: int,
    archetype: str,
    keyword: str,
    search_snippet: str,
    search_result: str,
    primary_family: str,
    support_family: str | None,
    intent_tags: list[str] | None = None,
    agent_idx: int = 1,
    n_agents: int = 1,
    status: str = "ok",
    confidence: float = 0.8,
    issues: list[str] | None = None,
    bad_type: str = "",
) -> dict:
    """Create one step-level SFT conversation for the search specialist agent."""
    focus = _search_focus_display(keyword)
    primary = _planner_family_label(primary_family)
    support = _planner_family_label(support_family) if support_family else "none"
    tags = ", ".join(intent_tags or []) or "none"
    tc_id = f"tc_search_step_{step_num}_a{agent_idx}"
    rows = [ln for ln in (search_result or "").splitlines() if ":" in ln]
    issue_text = ", ".join(issues or []) or "none"
    bad_note = f" bad_type={bad_type}." if bad_type else ""
    found_summary = (
        f"Found {len(rows)} matching controls for {focus}. "
        f"Status={status}, confidence={float(confidence):.2f}, issues={issue_text}.{bad_note}"
        if rows
        else f"No direct matches for {focus}; status={status}, confidence={float(confidence):.2f}, issues={issue_text}.{bad_note}"
    )
    return {
        "id": f"{sample_id}_search_step{step_num:02d}_agent{agent_idx:02d}",
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Main agent request (sample={sample_id}, step={step_num}, archetype={archetype}). "
                    f"You are search agent {agent_idx}/{max(1, n_agents)}. "
                    f"Search focus: {focus}. "
                    f"Planned edit families: primary={primary}, support={support}. "
                    f"Intent tags: {tags}. "
                    "Write reapy Python in a bash tool call that lists matching Vital controls as "
                    "'Name: 0.1234' (max 16 rows)."
                ),
            },
            {
                "role": "assistant",
                "content": f"I'll query {focus} controls and return exact display names with current values.",
                "tool_calls": [{
                    "id": tc_id,
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": search_snippet}),
                    },
                }],
            },
            {
                "role": "tool",
                "tool_call_id": tc_id,
                "content": search_result or f"[no params matching '{keyword}']",
            },
            {
                "role": "assistant",
                "content": f"{found_summary} Returning this to the main agent for the next set step.",
            },
        ],
        "meta": {
            "sample_id": sample_id,
            "step": step_num,
            "archetype": archetype,
            "agent_idx": agent_idx,
            "n_agents": n_agents,
            "search_keyword": keyword,
            "primary_family": primary_family,
            "support_family": support_family,
            "intent_tags": list(intent_tags or []),
            "status": status,
            "confidence": float(confidence),
            "issues": list(issues or []),
            "bad_type": bad_type,
        },
    }


def _extract_primary_edits(set_content: str) -> list[str]:
    m = re.search(r"Primary edits:\s*(.*?)(?:\.\s*$|$)", set_content, flags=re.IGNORECASE)
    if not m:
        return []
    raw = m.group(1).strip()
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def _search_snippet_matches_keyword(search_snippet: str, keyword: str) -> bool:
    kw = _sanitize_search_keyword(keyword)
    mod_idx = _modulation_index_from_keyword(kw)
    snippet = search_snippet or ""
    if mod_idx is not None:
        return f"startswith('modulation {mod_idx} ')" in snippet
    return f"'{kw}' in p.name.lower()" in snippet


def _validate_set_turn_content(
    set_content: str,
    keyword: str,
    allowed_primary_names: list[str] | None,
) -> tuple[bool, str]:
    edits = _extract_primary_edits(set_content)
    if not edits:
        return True, ""

    allowed_display = {_json_key_to_display(n).lower() for n in (allowed_primary_names or [])}
    for edit in edits:
        edit_l = edit.lower()
        if allowed_display and edit_l not in allowed_display:
            return False, f"primary edit '{edit}' is outside allowed primary controls"
        if not _display_matches_keyword(edit, keyword):
            return False, f"primary edit '{edit}' does not match search keyword '{keyword}'"
    return True, ""


def _validate_step_alignment(
    commentary: str,
    search_snippet: str,
    set_content: str,
    keyword: str,
    archetype: str,
    primary_family: str,
    support_family: str | None,
    allowed_param_names: set[str],
    allowed_primary_names: list[str],
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    ok_commentary, reason = _validate_commentary(
        commentary,
        allowed_param_names,
        search_keyword=keyword,
        archetype=archetype,
        primary_family=primary_family,
        support_family=support_family,
    )
    if not ok_commentary:
        errors.append(reason)
    if not _search_snippet_matches_keyword(search_snippet, keyword):
        errors.append(f"search snippet does not match keyword '{keyword}'")
    ok_set, set_reason = _validate_set_turn_content(set_content, keyword, allowed_primary_names)
    if not ok_set:
        errors.append(set_reason)
    return len(errors) == 0, errors


def _build_step_contract(step_data: dict, params_delta: list, keyword: str) -> dict:
    planned_primary_names = list(step_data.get("planned_primary_names") or [])
    planned_param_names = list(step_data.get("planned_param_names") or [])
    allowed_primary = list(step_data.get("allowed_primary_controls") or planned_primary_names or [])
    allowed_support = list(step_data.get("allowed_support_controls") or [])
    intended = list(step_data.get("intended_edit_controls") or [])

    if not intended:
        intended = [d["name"] for d in (params_delta or []) if d.get("name")]
    if not intended:
        intended = list(step_data.get("params_applied", {}).keys())
    if not allowed_primary:
        allowed_primary = [n for n in intended if _json_name_matches_keyword(n, keyword)]
    if not allowed_primary:
        allowed_primary = list(intended[:8])

    primary_family = str(step_data.get("primary_family") or "")
    support_family = step_data.get("support_family")
    search_scope_type = str(
        step_data.get("search_scope_type")
        or ("mod_slot" if _modulation_index_from_keyword(keyword) else "family")
    )

    return {
        "step": int(step_data.get("step", 0)),
        "search_keyword": keyword,
        "search_scope_type": search_scope_type,
        "primary_family": primary_family,
        "support_family": support_family,
        "planned_param_names": planned_param_names,
        "planned_primary_names": planned_primary_names,
        "allowed_primary_controls": allowed_primary,
        "allowed_support_controls": allowed_support,
        "intended_edit_controls": intended,
    }


async def get_commentary(
    gt_wav: str,
    iter_wav: str | None,
    step_num: int,
    client: httpx.AsyncClient,
    server: str,
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    params_delta: list | None = None,
    is_mistake_step: bool = False,
    it: dict | None = None,
    remaining_delta_context: str = "",
    allowed_params: set[str] | None = None,
    search_keyword: str = "",
    archetype: str = "synth",
    primary_family: str = "",
    support_family: str | None = None,
    intent_tags: list[str] | None = None,
    planned_param_names: list[str] | None = None,
    allowed_primary_names: list[str] | None = None,
    extra_feedback: str = "",
) -> tuple[str, dict]:
    delta_context = _format_delta_context(params_delta or [], is_mistake_step)

    # Build grounded description prompt from the iteration dict when available
    param_summary = _build_param_summary(it)
    if param_summary:
        description_prompt = (
            f"Describe this synthesizer audio clip in 2-3 sentences. "
            f"Focus on timbre, texture, and movement. "
            f"Context: the following parameters were just adjusted — {param_summary}. "
            f"Let that context inspire the description without being too literal."
        )
    else:
        description_prompt = (
            "Describe this synthesizer audio clip in 2-3 sentences. "
            "Focus on timbre, texture, and movement."
        )

    with open(gt_wav, "rb") as f:
        gt_b64 = base64.b64encode(f.read()).decode()

    primary_label = _planner_family_label(primary_family)
    support_label = _planner_family_label(support_family) if support_family else "none"
    planned_count = len(planned_param_names or [])
    primary_controls = ", ".join(_json_key_to_display(n) for n in (allowed_primary_names or [])[:6]) or "none"
    intent_hint = ", ".join(intent_tags or [])
    search_focus_display = _search_focus_display(search_keyword)
    contract_text = (
        "Planned edit contract:\n"
        f"- Primary family: {primary_label}\n"
        f"- Support family: {support_label}\n"
        f"- Planned control count: {planned_count}\n"
        f"- Intent tags: {intent_hint if intent_hint else 'none'}\n"
        f"- Search focus to use in PLAN: {search_focus_display}\n"
        f"- Allowed primary controls (sample): {primary_controls}\n"
    )

    content = [
        {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{gt_b64}"}}
    ]

    if iter_wav:
        with open(iter_wav, "rb") as f:
            iter_b64 = base64.b64encode(f.read()).decode()
        content.append(
            {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{iter_b64}"}}
        )
        text = (
            "You are a music production AI agent planning the NEXT preset edit.\n"
            "AUDIO A is the target. AUDIO B is the current working preset.\n"
            "Use both the audio comparison and the numeric parameter deltas.\n\n"
            "Write exactly 3 short sections, each 1-2 concise sentences:\n"
            "HEARD: concrete perceptual A/B differences.\n"
            "HYPOTHESIS: likely synthesis causes (use uncertainty words like likely/may/might/could).\n"
            "PLAN: what this next parameter update is trying to fix, then what control family you'll inspect/search first before applying edits.\n"
            "In PLAN, explicitly reference the step search focus, and use natural control names only.\n"
            "Keep PLAN aligned with the planned edit contract (primary/support families only).\n"
            "If the search focus is a modulation index, refer to it as a modulation slot.\n"
            "Do NOT use snake_case/programmatic parameter ids.\n"
            "Do NOT relabel the preset type; if you mention type, keep it aligned with the archetype hint.\n\n"
            f"Archetype hint: {archetype}\n"
            f"Step search focus: {search_focus_display}\n\n"
            + contract_text + "\n"
            + description_prompt
        )
        if remaining_delta_context:
            text += "\n\n" + remaining_delta_context
        if delta_context:
            text += "\n\n" + delta_context
        if is_mistake_step:
            text += (
                "\n\nNote: this step includes overcorrections on some parameters. "
                "Acknowledge what became worse and why."
            )
    else:
        text = (
            "You are a music production AI agent. You just heard the target sound (above). "
            + description_prompt + "\n\n"
            "Write exactly 3 short sections (HEARD, HYPOTHESIS, PLAN), each 1-2 concise sentences. "
            "In HYPOTHESIS, use uncertainty words like likely/may/might/could. "
            "In PLAN, include what you'll inspect/search first using the provided step search focus. "
            "Use natural control names only (no snake_case/programmatic ids). "
            "If the search focus is a modulation index, refer to it as a modulation slot. "
            "Do not relabel the preset type."
            f"\nArchetype hint: {archetype}\nStep search focus: {search_focus_display}\n\n{contract_text}"
        )
        if remaining_delta_context:
            text += "\n\n" + remaining_delta_context
        if delta_context:
            text += "\n\n" + delta_context

    content.append({"type": "text", "text": text})

    quality_feedback = str(extra_feedback or "").strip()
    allowed = allowed_params or set()
    for attempt in range(6):
        text_with_feedback = text
        if quality_feedback:
            text_with_feedback += (
                "\n\nREVISION REQUIRED:\n"
                f"{quality_feedback}\n"
                "Rewrite fully. Keep all sections concise and complete."
            )
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": content[:-1] + [{"type": "text", "text": text_with_feedback}]}],
            "max_tokens": 420,
            "temperature": 0.5,
        }
        resp = await client.post(
            f"{server}/v1/chat/completions", json=payload, timeout=60.0
        )
        resp.raise_for_status()
        commentary = _normalize_commentary(resp.json()["choices"][0]["message"]["content"].strip())
        ok, reason = _validate_commentary(
            commentary,
            allowed,
            search_keyword=search_keyword,
            archetype=archetype,
            primary_family=primary_family,
            support_family=support_family,
        )
        if ok:
            return commentary, {"attempts": attempt + 1, "fallback_used": False, "reason": ""}
        quality_feedback = reason

    return (
        _fallback_commentary(it, search_keyword=search_keyword or "target controls"),
        {"attempts": 6, "fallback_used": True, "reason": quality_feedback or "quality gate failed"},
    )


async def process_sample(
    manifest_entry: dict,
    omni_server: str,
    sem: asyncio.Semaphore,
    client: httpx.AsyncClient,
    model: str = "Qwen/Qwen3-Omni-30B-A3B-Instruct",
    main_search_mode: str = "bash",
    emit_search_records: bool = False,
    search_fanout: int = 1,
    search_bad_result_prob: float = 0.0,
    search_bad_result_max: int = 1,
    search_seed: int = 1337,
) -> tuple[dict | None, list[dict]]:
    sample_id = manifest_entry["sample_id"]
    with open(manifest_entry["path_file"]) as f:
        path_data = json.load(f)
    delta_maps = _prepare_delta_maps(path_data)

    gt_wav = manifest_entry.get("gt_probe_wav") or manifest_entry["gt_wav"]
    default_wav = manifest_entry.get("default_wav")
    iter_wavs = manifest_entry["iter_wavs"]  # N entries (all steps rendered)
    n_iterations = manifest_entry["n_iterations"]
    archetype = manifest_entry.get("archetype", "synth")

    messages = []
    audios = [os.path.abspath(gt_wav)]
    search_records: list[dict] = []

    # Turn 0: user / GT audio + task
    if main_search_mode == "search_agent":
        workflow_line = (
            "At each step: ask the search_agent tool to run parallel control searches, then set params, then listen again."
        )
    else:
        workflow_line = (
            "At each step: search relevant params by keyword, set params, then listen again."
        )
    user_content = (
        f"<audio>\n"
        f"This is a target synthesizer sound ({archetype}). "
        f"Recreate it in Vital starting from the default preset using terminal reapy Python. "
        f"{workflow_line}"
    )
    messages.append({
        "role": "user",
        "content": user_content,
    })

    # Baseline listen turn from default preset (if rendered).
    if default_wav:
        default_listen_id = "tc_listen_default"
        audios.append(os.path.abspath(default_wav))
        messages.append({
            "role": "assistant",
            "content": "I will first listen to the default Vital preset as a baseline.",
            "tool_calls": [{
                "id": default_listen_id, "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": _LISTEN_CMD}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": default_listen_id,
            "content": _LISTEN_RESULT,
        })

    prev_iter_wav_for_commentary = default_wav if default_wav else None
    step_quality_rows: list[dict] = []
    for step_idx in range(n_iterations):
        step_data = path_data["iterations"][step_idx]
        action_snippet = (
            step_data.get("action_snippet")
            or step_data.get("python_script")  # backwards compat
            or "python - <<'PY'\nprint('No-op step')\nPY"
        )
        params_delta = step_data.get("params_delta", [])
        is_mistake = step_data.get("is_mistake_step", False)
        params_applied = step_data.get("params_applied", {})
        modulations_changed = step_data.get("modulations_changed", [])
        if not params_delta and not params_applied and not modulations_changed:
            continue
        n_params = len(params_applied)
        n_mods = len(modulations_changed)
        step_num = step_data["step"]
        remaining_delta_context = _format_remaining_delta_context(step_num, delta_maps)
        allowed_param_names = {d.get("name") for d in params_delta if d.get("name")}
        keyword = _infer_search_keyword(step_data)
        keyword = _sanitize_search_keyword(keyword)
        contract = _build_step_contract(step_data, params_delta, keyword)
        primary_family = contract["primary_family"]
        support_family = contract["support_family"]
        intent_tags = list(step_data.get("intent_tags") or [])
        planned_param_names = contract["planned_param_names"]
        planned_primary_names = contract["planned_primary_names"]
        allowed_primary_names = contract["allowed_primary_controls"]
        allowed_support_names = contract["allowed_support_controls"]
        intended_edit_names = contract["intended_edit_controls"]
        search_scope_type = contract["search_scope_type"]

        # iter_wav for Omni's commentary: baseline default first, then latest listened state.
        iter_wav_for_commentary = prev_iter_wav_for_commentary

        seed_text = f"{search_seed}:{sample_id}:{step_num}"
        seed_int = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8], "big")
        step_rng = random.Random(seed_int)

        step_fanout = max(1, int(search_fanout if main_search_mode == "search_agent" else 1))
        fanout_payload = _build_search_fanout_bundle(
            step_data=step_data,
            step_num=step_num,
            primary_keyword=keyword,
            primary_family=primary_family,
            support_family=support_family,
            fanout=step_fanout,
            bad_result_prob=search_bad_result_prob,
            bad_result_max=search_bad_result_max,
            rng=step_rng,
        )
        search_queries = list(fanout_payload["queries"])
        search_reports = list(fanout_payload["reports"])
        search_bundle = dict(fanout_payload["bundle"])
        search_snippet = str(fanout_payload["primary_search_snippet"])
        search_result = str(fanout_payload["primary_search_result"])
        search_tool_result_content = (
            str(fanout_payload["tool_content"])
            if main_search_mode == "search_agent"
            else (search_result or f"[no params matching '{keyword}']")
        )
        search_handoff_text = (
            _build_search_handoff_interpretation(search_bundle, keyword)
            if main_search_mode == "search_agent"
            else _build_search_interpretation(keyword, search_result)
        )
        # Get Omni commentary + strict alignment validation (hard retries, then deterministic fallback).
        step_alignment_errors: list[str] = []
        commentary_attempts = 0
        used_fallback = False
        commentary = ""
        set_content = ""
        feedback = ""
        step_ok = False

        for _ in range(3):
            try:
                async with sem:
                    commentary, meta = await get_commentary(
                        gt_wav, iter_wav_for_commentary, step_num, client, omni_server,
                        model=model,
                        params_delta=params_delta,
                        is_mistake_step=is_mistake,
                        it=step_data,
                        remaining_delta_context=remaining_delta_context,
                        allowed_params=allowed_param_names,
                        search_keyword=keyword,
                        archetype=archetype,
                        primary_family=primary_family,
                        support_family=support_family,
                        intent_tags=intent_tags,
                        planned_param_names=planned_param_names,
                        allowed_primary_names=allowed_primary_names,
                        extra_feedback=feedback,
                    )
            except Exception as e:
                meta = {"attempts": 1, "fallback_used": True, "reason": f"request failed: {e}"}
                commentary = _fallback_commentary(step_data, search_keyword=keyword)
            commentary_attempts += int(meta.get("attempts", 1))

            set_content = (
                search_handoff_text
                + " "
                + _build_set_turn_content(
                    step_num,
                    keyword,
                    params_delta,
                    is_mistake,
                    allowed_primary_names=allowed_primary_names,
                    planned_primary_names=planned_primary_names,
                )
            )

            step_ok, step_alignment_errors = _validate_step_alignment(
                commentary=commentary,
                search_snippet=search_snippet,
                set_content=set_content,
                keyword=keyword,
                archetype=archetype,
                primary_family=primary_family,
                support_family=support_family,
                allowed_param_names=allowed_param_names,
                allowed_primary_names=allowed_primary_names,
            )
            if step_ok:
                break
            used_fallback = bool(meta.get("fallback_used", False))
            feedback = "Alignment errors: " + "; ".join(step_alignment_errors[:3])
            if used_fallback:
                break

        if not step_ok:
            used_fallback = True
            commentary = _fallback_commentary(step_data, search_keyword=keyword)
            set_content = (
                search_handoff_text
                + " "
                + _build_set_turn_content(
                    step_num,
                    keyword,
                    params_delta,
                    is_mistake,
                    allowed_primary_names=allowed_primary_names,
                    planned_primary_names=allowed_primary_names,
                )
            )
            step_ok, step_alignment_errors = _validate_step_alignment(
                commentary=commentary,
                search_snippet=search_snippet,
                set_content=set_content,
                keyword=keyword,
                archetype=archetype,
                primary_family=primary_family,
                support_family=support_family,
                allowed_param_names=allowed_param_names,
                allowed_primary_names=allowed_primary_names,
            )

        if not step_ok:
            print(
                f"  WARNING: dropping sample {sample_id} at step {step_num} due unresolved alignment errors: "
                + "; ".join(step_alignment_errors[:4]),
                file=sys.stderr,
            )
            return None, []

        step_quality_rows.append({
            "step": step_num,
            "alignment_pass": step_ok,
            "alignment_errors": step_alignment_errors,
            "commentary_attempts": commentary_attempts,
            "fallback_used": used_fallback,
            "search_scope_type": search_scope_type,
            "search_keyword": keyword,
            "primary_family": primary_family,
            "support_family": support_family,
            "allowed_primary_controls": allowed_primary_names,
            "allowed_support_controls": allowed_support_names,
            "intended_edit_controls": intended_edit_names,
            "search_fanout": len(search_queries),
            "search_bad_injected": int(search_bundle.get("quality_summary", {}).get("bad_injected", 0)),
            "search_quality_summary": search_bundle.get("quality_summary", {}),
            "search_queries": search_queries,
        })

        # 1) Describe + plan, then search before setting.
        search_call = _build_main_search_tool_call(
            step_num=step_num,
            keyword=keyword,
            queries=search_queries,
            search_snippet=search_snippet,
            mode=main_search_mode,
        )
        search_id = search_call["id"]
        messages.append({
            "role": "assistant",
            "content": commentary,
            "tool_calls": [search_call],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": search_id,
            "content": search_tool_result_content,
        })
        if emit_search_records:
            for idx, report in enumerate(search_reports, start=1):
                search_records.append(
                    _build_search_agent_record(
                        sample_id=sample_id,
                        step_num=step_num,
                        archetype=archetype,
                        keyword=str(report.get("query", keyword)),
                        search_snippet=str(report.get("search_snippet", search_snippet)),
                        search_result=str(report.get("search_result", "")),
                        primary_family=primary_family,
                        support_family=support_family,
                        intent_tags=intent_tags,
                        agent_idx=idx,
                        n_agents=len(search_reports),
                        status=str(report.get("status", "ok")),
                        confidence=float(report.get("confidence", 0.0)),
                        issues=list(report.get("issues") or []),
                        bad_type=str(report.get("bad_type", "")),
                    )
                )

        # vital set tool_call
        set_id = f"tc_set_{step_num}"
        set_result = f"OK: set {n_params} param(s)"
        if n_mods:
            set_result += f", {n_mods} mod route(s)"
        messages.append({
            "role": "assistant",
            "content": (
                set_content
            ),
            "tool_calls": [{
                "id": set_id, "type": "function",
                "function": {
                    "name": "bash",
                    "arguments": json.dumps({"command": action_snippet}),
                },
            }],
        })
        messages.append({
            "role": "tool",
            "tool_call_id": set_id,
            "content": set_result,
        })

        # vital listen tool_call — only when a rendered clip exists for this step.
        if step_idx < len(iter_wavs):
            listen_id = f"tc_listen_{step_num}"
            iter_wav = iter_wavs[step_idx]
            audios.append(os.path.abspath(iter_wav))
            prev_iter_wav_for_commentary = iter_wav
            messages.append({
                "role": "assistant",
                "content": "Let me listen to the current state.",
                "tool_calls": [{
                    "id": listen_id, "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": json.dumps({"command": _LISTEN_CMD}),
                    },
                }],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": listen_id,
                "content": _LISTEN_RESULT,
            })

    # Final assistant turn: conclusion (no tool_call)
    messages.append({
        "role": "assistant",
        "content": "Recreation complete.",
    })

    critical_mismatches = sum(1 for row in step_quality_rows if not row.get("alignment_pass"))
    fallback_steps = sum(1 for row in step_quality_rows if row.get("fallback_used"))
    return (
        {
            "id": sample_id,
            "messages": messages,
            "audios": audios,
            "quality": {
                "critical_mismatch_count": critical_mismatches,
                "fallback_steps": fallback_steps,
                "steps": step_quality_rows,
            },
            "meta": {
                "agent": "main",
                "main_search_mode": main_search_mode,
                "search_fanout": max(1, int(search_fanout if main_search_mode == "search_agent" else 1)),
                "search_bad_result_prob": float(search_bad_result_prob),
                "search_bad_result_max": int(search_bad_result_max),
            },
        },
        search_records,
    )


def validate_record(record: dict) -> None:
    n_audio_tags = sum(
        (m.get("content") or "").count("<audio>") for m in record["messages"]
    )
    assert n_audio_tags == len(record["audios"]), (
        f"{record['id']}: audio tag count ({n_audio_tags}) != audios len ({len(record['audios'])})"
    )
    assert record["messages"][-1]["role"] == "assistant", (
        f"{record['id']}: last message is not assistant"
    )
    assert "tool_calls" not in record["messages"][-1], (
        f"{record['id']}: final assistant message must not have tool_calls"
    )
    assert record["messages"][0]["role"] == "user", (
        f"{record['id']}: first message is not user"
    )


def validate_search_record(record: dict) -> None:
    assert record["messages"][0]["role"] == "user", (
        f"{record['id']}: first message is not user"
    )
    assert record["messages"][-1]["role"] == "assistant", (
        f"{record['id']}: last message is not assistant"
    )
    assert "tool_calls" not in record["messages"][-1], (
        f"{record['id']}: final assistant message must not have tool_calls"
    )
    tool_calls = [
        tc
        for m in record["messages"]
        if m.get("role") == "assistant"
        for tc in m.get("tool_calls", [])
    ]
    assert tool_calls, f"{record['id']}: no assistant tool call found"
    assert any(tc.get("function", {}).get("name") == "bash" for tc in tool_calls), (
        f"{record['id']}: search agent record must include a bash tool call"
    )
    tool_ids = {tc.get("id") for tc in tool_calls}
    for m in record["messages"]:
        if m.get("role") == "tool":
            assert m.get("tool_call_id") in tool_ids, (
                f"{record['id']}: tool response references unknown tool_call_id"
            )


async def main():
    parser = argparse.ArgumentParser(description="Build iterative SFT dataset from render manifest")
    parser.add_argument("--manifest", required=True, help="Path to manifest.jsonl")
    parser.add_argument("--omni-server", default="http://localhost:8000", help="Omni model server URL")
    parser.add_argument("--omni-model", default="Qwen/Qwen3-Omni-30B-A3B-Instruct", help="Model name to pass to the inference API")
    parser.add_argument("--output", required=True, help="Output JSONL path")
    parser.add_argument(
        "--main-search-mode",
        choices=["bash", "search_agent"],
        default="bash",
        help="How main-agent search turns are represented in output conversations.",
    )
    parser.add_argument(
        "--search-output",
        default="",
        help="Optional output JSONL path for step-level search-agent SFT conversations.",
    )
    parser.add_argument(
        "--search-fanout",
        type=int,
        default=1,
        help="Number of parallel search-agent shards to synthesize per step (search_agent mode).",
    )
    parser.add_argument(
        "--search-bad-result-prob",
        type=float,
        default=0.0,
        help="Probability of injecting a degraded search-agent output per shard.",
    )
    parser.add_argument(
        "--search-bad-result-max",
        type=int,
        default=1,
        help="Maximum number of degraded search-agent outputs injected per step.",
    )
    parser.add_argument(
        "--search-seed",
        type=int,
        default=1337,
        help="Base seed for deterministic search fanout and bad-result injection.",
    )
    parser.add_argument("--concurrency", type=int, default=8, help="Max concurrent Omni API calls")
    args = parser.parse_args()

    # Load manifest
    with open(args.manifest) as f:
        entries = [json.loads(line) for line in f if line.strip()]
    print(f"Loaded {len(entries)} samples from manifest")

    sem = asyncio.Semaphore(args.concurrency)
    async with httpx.AsyncClient() as client:
        tasks = [
            process_sample(
                entry,
                args.omni_server,
                sem,
                client,
                model=args.omni_model,
                main_search_mode=args.main_search_mode,
                emit_search_records=bool(args.search_output),
                search_fanout=args.search_fanout,
                search_bad_result_prob=args.search_bad_result_prob,
                search_bad_result_max=args.search_bad_result_max,
                search_seed=args.search_seed,
            )
            for entry in entries
        ]
        records = []
        search_records: list[dict] = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                record, sample_search_records = await coro
                if record is not None:
                    records.append(record)
                if sample_search_records:
                    search_records.extend(sample_search_records)
            except Exception as e:
                print(f"\n  ERROR processing sample: {e}", file=sys.stderr)
            print(f"  Processed {i + 1}/{len(entries)}", end="\r")
    print()

    # Validate all records
    for record in records:
        validate_record(record)
    print(f"All {len(records)} records valid")

    # Sort by id for deterministic output
    records.sort(key=lambda r: r["id"])

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} records to {args.output}")

    if args.search_output:
        for record in search_records:
            validate_search_record(record)
        search_records.sort(key=lambda r: r["id"])
        os.makedirs(os.path.dirname(os.path.abspath(args.search_output)), exist_ok=True)
        with open(args.search_output, "w") as f:
            for record in search_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"Wrote {len(search_records)} search-agent records to {args.search_output}")


if __name__ == "__main__":
    asyncio.run(main())
