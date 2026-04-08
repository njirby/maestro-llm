"""
Generates N-step parameter paths from Vital's default preset to a generated
target preset. Used to create training data for an iterative sound recreation
model.

Each path starts from the init preset and applies groups of changed parameters
in priority order, producing a sequence of cumulative preset states that
progressively converge to the target.

Usage:
    from maestro.synth.path_gen import generate_preset_path
    import random
    rng = random.Random(42)
    result = generate_preset_path("bass", rng)
"""

from __future__ import annotations

import copy
import json
import math
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Module-level data (loaded once)
# ---------------------------------------------------------------------------

_DATA_DIR = Path(__file__).parent

with open(_DATA_DIR / "param_ranges.json") as _f:
    PARAM_RANGES: dict = json.load(_f)

with open(_DATA_DIR / "init_preset.json") as _f:
    _INIT_PRESET: dict = json.load(_f)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MISTAKE_PROB = 0.08       # 8% chance any individual param assignment is wrong
CORRECTION_BATCH_SIZE = 4  # max params per correction step

# Iteration count controls.
# Instead of a fixed table that can saturate to 20 when many params differ,
# sample a realistic "params edited per step" density and derive N from that.
MIN_ITERATIONS = 6
MAX_ITERATIONS = 20
MAX_ITERATIONS_HARD_CAP = 40  # absolute ceiling including overflow steps
PARAMS_PER_STEP_MIN = 14.0
PARAMS_PER_STEP_MAX = 24.0
SEARCH_RESULT_MAX_LINES = 16
CHECKPOINT_EVERY_STEPS = 3

# Lower index = higher priority (appears earlier in path)
PARAM_PRIORITY_PREFIXES = [
    "osc_",         # priority 1
    "env_1_",       # priority 2
    "env_2_",       # priority 3
    "filter_1_",    # priority 4
    "filter_2_",    # priority 5
    "unison_",      # priority 6
    "lfo_",         # priority 7
    "reverb_",      # priority 8
    "delay_",       # priority 8
    "chorus_",      # priority 8
    "distortion_",  # priority 9
    "compressor_",  # priority 9
    "phaser_",      # priority 9
    "flanger_",     # priority 9
    "eq_",          # priority 10
]


# ---------------------------------------------------------------------------
# Display name helpers (used for search/set snippets in training conversations)
# ---------------------------------------------------------------------------

_VITAL_ABBREVS = {
    "osc": "Oscillator",
    "env": "Envelope",
    "lfo": "LFO",
    "eq":  "EQ",
    "fx":  "FX",
}


def _json_key_to_display(key: str) -> str:
    """Convert a Vital JSON parameter key to its REAPER display name.

    Examples:
        osc_1_level      → "Oscillator 1 Level"
        filter_1_cutoff  → "Filter 1 Cutoff"
        lfo_2_frequency  → "LFO 2 Frequency"
    """
    parts = key.split("_")
    return " ".join(_VITAL_ABBREVS.get(p.lower(), p.capitalize()) for p in parts)


# display_name → json_key reverse map (built once from PARAM_RANGES)
_DISPLAY_TO_KEY: dict = {_json_key_to_display(k): k for k in PARAM_RANGES}

_FAMILY_TO_KEYWORD = {
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

_STAGE_ORDER = [
    ("core", ("osc", "env1", "env2", "filter1", "filter2")),
    ("motion", ("lfo",)),
    ("space", ("chorus", "delay", "reverb", "distortion", "compressor", "phaser", "flanger", "eq")),
    ("macro", ("other",)),
    ("modulation", ("modulation",)),
]
MAX_MODULATION_STREAK = 2
SOFT_MAX_MODULATION_RATIO = 0.55


def _modulation_index_from_keyword(keyword: str) -> str | None:
    m = re.fullmatch(r"modulation\s+(\d+)", (keyword or "").strip().lower())
    return m.group(1) if m else None


def _modulation_slot(name: str) -> str | None:
    m = re.fullmatch(r"modulation_(\d+)_(?:amount|stereo|bipolar|bypass|power)", name)
    return m.group(1) if m else None


def _param_family(name: str) -> str:
    if name.startswith(("osc_", "unison_")):
        return "osc"
    if name.startswith("env_1_"):
        return "env1"
    if name.startswith("env_2_"):
        return "env2"
    if name.startswith("filter_1_"):
        return "filter1"
    if name.startswith("filter_2_"):
        return "filter2"
    if name.startswith("lfo_"):
        return "lfo"
    if name.startswith("reverb_"):
        return "reverb"
    if name.startswith("delay_"):
        return "delay"
    if name.startswith("chorus_"):
        return "chorus"
    if name.startswith("distortion_"):
        return "distortion"
    if name.startswith("compressor_"):
        return "compressor"
    if name.startswith("phaser_"):
        return "phaser"
    if name.startswith("flanger_"):
        return "flanger"
    if name.startswith("eq_"):
        return "eq"
    if name.startswith("modulation_"):
        return "modulation"
    return "other"


def _stage_index_for_family(family: str) -> int:
    for idx, (_, fams) in enumerate(_STAGE_ORDER):
        if family in fams:
            return idx
    return len(_STAGE_ORDER) - 1


def _families_in_stage(stage_idx: int) -> tuple[str, ...]:
    return _STAGE_ORDER[max(0, min(stage_idx, len(_STAGE_ORDER) - 1))][1]


def _residual(name: str, target_norm: float, init_norm: dict[str, float]) -> float:
    cur = init_norm.get(name, 0.5)
    return abs(target_norm - cur)


def _build_search_filter(keyword: str) -> str:
    kw = (keyword or "").strip().lower()
    mod_idx = _modulation_index_from_keyword(kw)
    if mod_idx is not None:
        return f"p.name.lower().startswith('modulation {mod_idx} ')"
    if kw == "modulation":
        return "p.name.lower().startswith('modulation ')"
    safe = kw.replace("'", "\\'")
    return f"'{safe}' in p.name.lower()"


def _display_matches_keyword(display: str, keyword: str) -> bool:
    display_l = display.lower()
    mod_idx = _modulation_index_from_keyword(keyword)
    if mod_idx is not None:
        return display_l.startswith(f"modulation {mod_idx} ")
    return (keyword or "").strip().lower() in display_l


def _name_matches_keyword(name: str, keyword: str) -> bool:
    return _display_matches_keyword(_json_key_to_display(name), keyword)


def _keyword_coverage(names: list[str], keyword: str) -> float:
    if not names:
        return 0.0
    hits = sum(1 for n in names if _name_matches_keyword(n, keyword))
    return hits / len(names)


def _build_search_snippet(keyword: str) -> str:
    filt = _build_search_filter(keyword)
    return (
        "import reapy\n"
        "with reapy.inside_reaper():\n"
        "    fx = reapy.Project().tracks[0].fxs[0]\n"
        "    hits = []\n"
        "    for p in fx.params:\n"
        f"        if {filt}:\n"
        "            hits.append((p.name, round(p.normalized, 4)))\n"
        f"    for name, val in hits[:{SEARCH_RESULT_MAX_LINES}]:\n"
        '        print(f"{name}: {val:.4f}")'
    )


def _plan_step_groups(
    changed_scalar_params: dict[str, float],
    init_scalars_native: dict[str, float],
    n_iterations: int,
    rng: random.Random,
) -> tuple[list[list[tuple[str, float]]], list[dict]]:
    """Plan per-step parameter groups with primary/support family structure."""
    if not changed_scalar_params:
        return [[]], [{
            "primary_family": "other",
            "support_family": None,
            "planner_stage": "core",
            "checkpoint_revisit": False,
            "planned_param_names": [],
            "planned_primary_names": [],
            "allowed_primary_controls": [],
            "allowed_support_controls": [],
            "intended_edit_controls": [],
            "search_scope_type": "family",
            "intent_tags": ["focus:other", "stage:core"],
            "search_keyword": "filter",
        }]

    init_norm: dict[str, float] = {}
    for name in changed_scalar_params:
        val = _normalize(name, init_scalars_native.get(name, 0.0))
        init_norm[name] = 0.0 if val is None else float(val)

    remaining: dict[str, float] = dict(changed_scalar_params)
    steps_params: list[list[tuple[str, float]]] = []
    plans: list[dict] = []
    current_stage_idx = 0
    modulation_steps_taken = 0
    modulation_streak = 0

    step_idx = 0
    while remaining and step_idx < MAX_ITERATIONS_HARD_CAP:
        # Estimate steps still needed based on remaining param count and expected
        # step density. This ensures step_target stays reasonable as overflow steps
        # mop up any params left after the originally-planned n_iterations.
        avg_step_size = max(1.0, (PARAMS_PER_STEP_MIN + PARAMS_PER_STEP_MAX) / 2.0)
        steps_left = max(1, int(math.ceil(len(remaining) / avg_step_size)))
        step_target = max(1, int(math.ceil(len(remaining) / steps_left)))
        desired_primary = max(1, int(math.ceil(step_target * 0.7)))
        max_support = max(0, step_target - desired_primary)

        fam_items: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for name, tgt in remaining.items():
            family = _param_family(name)
            fam_items[family].append((name, tgt, _residual(name, tgt, init_norm)))
        for family in fam_items:
            fam_items[family].sort(key=lambda x: (x[2], -_param_priority(x[0])), reverse=True)
        available_families = set(fam_items.keys())

        while (
            current_stage_idx < len(_STAGE_ORDER) - 1
            and not any(f in available_families for f in _families_in_stage(current_stage_idx))
        ):
            current_stage_idx += 1

        checkpoint = step_idx > 0 and ((step_idx + 1) % CHECKPOINT_EVERY_STEPS == 0)
        candidate_families = [f for f in _families_in_stage(current_stage_idx) if f in available_families]
        if checkpoint:
            earlier = [
                f
                for idx in range(current_stage_idx)
                for f in _families_in_stage(idx)
                if f in available_families
            ]
            if earlier:
                candidate_families = earlier

        if not candidate_families:
            candidate_families = sorted(available_families)

        # Soft cap modulation-heavy trajectories when non-modulation families remain.
        non_mod_available = any(f != "modulation" for f in available_families)
        soft_mod_cap = max(2, int(math.ceil(n_iterations * SOFT_MAX_MODULATION_RATIO)))
        current_mod_ratio = modulation_steps_taken / max(1, step_idx)
        if non_mod_available:
            if (
                modulation_steps_taken >= soft_mod_cap
                or modulation_streak >= MAX_MODULATION_STREAK
                or (step_idx >= 2 and current_mod_ratio >= 0.50)
            ):
                candidate_families = [f for f in candidate_families if f != "modulation"] or candidate_families

        # If current-stage families cannot satisfy purity, advance to a family that can.
        max_candidate_count = max((len(fam_items[f]) for f in candidate_families), default=0)
        if max_candidate_count < desired_primary:
            capable = [f for f in available_families if len(fam_items[f]) >= desired_primary]
            if capable:
                candidate_families = sorted(
                    capable,
                    key=lambda f: (_stage_index_for_family(f) < current_stage_idx, _stage_index_for_family(f)),
                )
            else:
                # No family can satisfy desired_primary — batch multiple small families.
                # Use PARAMS_PER_STEP_MAX so the multi-family fill packs them together.
                step_target = min(int(PARAMS_PER_STEP_MAX), len(remaining))
                desired_primary = max(1, int(math.ceil(step_target * 0.7)))
                max_support = max(0, step_target - desired_primary)
                candidate_families = sorted(available_families, key=lambda f: len(fam_items[f]), reverse=True)

        def _family_score(family: str) -> tuple[int, float, int, int]:
            items = fam_items[family]
            total_res = sum(r for _, _, r in items)
            has_primary_capacity = 1 if len(items) >= desired_primary else 0
            return (
                has_primary_capacity,
                total_res,
                len(items),
                -_stage_index_for_family(family),
            )

        primary_family = max(candidate_families, key=_family_score)
        primary_items = fam_items[primary_family]
        chosen_mod_slot: str | None = None

        if primary_family == "modulation":
            # Batch multiple modulation slots per step rather than one slot per step.
            # Sort slots by total residual (highest-residual slots first), then take
            # up to step_target params spanning as many slots as needed.
            slot_items: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
            for item in primary_items:
                slot = _modulation_slot(item[0]) or "0"
                slot_items[slot].append(item)
            # Build batched list: fill up to step_target params from top-residual slots
            slots_by_residual = sorted(
                slot_items.keys(),
                key=lambda s: (sum(r for _, _, r in slot_items[s]), len(slot_items[s])),
                reverse=True,
            )
            batched: list[tuple[str, float, float]] = []
            for slot in slots_by_residual:
                batched.extend(slot_items[slot])
                if len(batched) >= step_target:
                    break
            primary_items = batched[:step_target]
            # For keyword: use the dominant slot if all from same slot, else "modulation"
            slots_used = sorted({_modulation_slot(item[0]) or "0" for item in primary_items})
            chosen_mod_slot = slots_used[0] if len(slots_used) == 1 else None

        primary_count = min(
            len(primary_items),
            max(desired_primary, step_target - max_support),
        )
        primary_count = min(primary_count, step_target)
        selected_primary = primary_items[:primary_count]
        selected_names = [name for name, _, _ in selected_primary]
        selected_tuples = [(name, tgt) for name, tgt, _ in selected_primary]

        remaining_slots = step_target - len(selected_tuples)
        support_family = None
        support_cap = 0 if primary_family == "modulation" else max_support

        support_candidates = [f for f in candidate_families if f != primary_family]
        if not support_candidates:
            support_candidates = [f for f in available_families if f != primary_family]
        if remaining_slots > 0 and support_candidates and support_cap > 0:
            support_family = max(
                support_candidates,
                key=lambda f: (sum(r for _, _, r in fam_items[f]), len(fam_items[f])),
            )
            take = min(remaining_slots, support_cap, len(fam_items[support_family]))
            for name, tgt, _ in fam_items[support_family][:take]:
                selected_tuples.append((name, tgt))
                selected_names.append(name)
            remaining_slots = step_target - len(selected_tuples)

        # Fill from primary first to preserve purity, then from support.
        if remaining_slots > 0:
            used_primary = set(name for name, _, _ in selected_primary)
            for name, tgt, _ in primary_items:
                if name in used_primary:
                    continue
                selected_tuples.append((name, tgt))
                selected_names.append(name)
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break

        if remaining_slots > 0 and support_family is not None:
            used = set(selected_names)
            for name, tgt, _ in fam_items[support_family]:
                if name in used:
                    continue
                selected_tuples.append((name, tgt))
                selected_names.append(name)
                remaining_slots -= 1
                if remaining_slots <= 0:
                    break

        # Multi-family fill: when slots remain after primary+support, pull from any
        # remaining family rather than leaving them for isolated 1-2 param steps.
        # This prevents a long tail of tiny steps for small families.
        if remaining_slots > 0:
            used = set(selected_names)
            extra_families = sorted(
                [f for f in available_families if f not in (primary_family, support_family)],
                key=lambda f: (len(fam_items[f]), sum(r for _, _, r in fam_items[f])),
                reverse=True,
            )
            for fam in extra_families:
                if remaining_slots <= 0:
                    break
                for name, tgt, _ in fam_items[fam]:
                    if name in used:
                        continue
                    selected_tuples.append((name, tgt))
                    selected_names.append(name)
                    remaining_slots -= 1
                    used.add(name)
                    if remaining_slots <= 0:
                        break

        if not selected_tuples:
            # Safety fallback: ensure progress if all capacity checks collapsed.
            best_name, best_tgt, _ = max(
                ((n, t, r) for fam in fam_items.values() for (n, t, r) in fam),
                key=lambda x: x[2],
            )
            selected_tuples = [(best_name, best_tgt)]
            selected_names = [best_name]
            primary_family = _param_family(best_name)
            support_family = None

        # For modulation-heavy steps spanning many slots, keep keyword broad.
        primary_names = [name for name, _ in selected_tuples if _param_family(name) == primary_family]
        if primary_family == "modulation":
            slots = sorted({slot for slot in (_modulation_slot(n) for n in primary_names) if slot is not None})
            if chosen_mod_slot is not None:
                search_keyword = f"modulation {chosen_mod_slot}"
            elif len(slots) == 1:
                search_keyword = f"modulation {slots[0]}"
            else:
                search_keyword = "modulation"
        else:
            specific_keyword = _search_keyword(primary_names or selected_names)
            family_keyword = _FAMILY_TO_KEYWORD.get(primary_family, "filter")
            if not specific_keyword:
                search_keyword = family_keyword
            else:
                # If the specific keyword is too narrow for this step's primary edits,
                # back off to family-level search to keep search/edit alignment.
                coverage = _keyword_coverage(primary_names or selected_names, specific_keyword)
                search_keyword = specific_keyword if coverage >= 0.70 else family_keyword

        stage_name = _STAGE_ORDER[_stage_index_for_family(primary_family)][0]
        intent_tags = [f"focus:{primary_family}", f"stage:{stage_name}"]
        if support_family:
            intent_tags.append(f"support:{support_family}")
        if checkpoint:
            intent_tags.append("checkpoint_revisit")
        current_stage_idx = max(current_stage_idx, _stage_index_for_family(primary_family))

        if primary_family == "modulation":
            modulation_steps_taken += 1
            modulation_streak += 1
        else:
            modulation_streak = 0

        support_names = [name for name in selected_names if _param_family(name) != primary_family]
        search_scope_type = "mod_slot" if _modulation_index_from_keyword(search_keyword) else "family"

        plans.append({
            "primary_family": primary_family,
            "support_family": support_family,
            "planner_stage": stage_name,
            "checkpoint_revisit": checkpoint,
            "planned_param_names": selected_names,
            "planned_primary_names": primary_names,
            "allowed_primary_controls": primary_names,
            "allowed_support_controls": support_names,
            "intended_edit_controls": selected_names,
            "search_scope_type": search_scope_type,
            "intent_tags": intent_tags,
            "search_keyword": search_keyword,
        })
        steps_params.append(selected_tuples)

        for name, _ in selected_tuples:
            remaining.pop(name, None)

        step_idx += 1

    return steps_params, plans


def _search_keyword(param_names: list) -> str:
    """Derive a short search keyword from a list of json param keys.

    Takes the first two underscore-parts of the dominant prefix, converts to
    display words (lowercase).  E.g.:
        ["filter_1_cutoff", "filter_1_resonance"] → "filter 1"
        ["osc_1_level", "osc_1_transpose"]        → "oscillator 1"
        ["env_1_attack", "env_2_decay"]            → "envelope 1"
    """
    from collections import Counter as _Counter
    prefixes = []
    for name in param_names:
        parts = name.split("_")
        prefixes.append("_".join(parts[:min(2, len(parts))]))
    if not prefixes:
        return "filter"
    dominant = _Counter(prefixes).most_common(1)[0][0]
    return _json_key_to_display(dominant).lower()


def _generate_search_result(keyword: str, cumulative_settings: dict) -> str:
    """Generate synthetic search output: display names matching keyword + current values.

    Values are the cumulative normalized state BEFORE the step is applied.
    """
    mod_idx = _modulation_index_from_keyword(keyword)
    lines = []
    for display in sorted(_DISPLAY_TO_KEY):
        display_l = display.lower()
        if mod_idx is not None:
            if not display_l.startswith(f"modulation {mod_idx} "):
                continue
        elif keyword not in display_l:
            continue
        key = _DISPLAY_TO_KEY[display]
        r = PARAM_RANGES.get(key)
        if r is None:
            continue
        native = cumulative_settings.get(key, r["default"])
        span = r["max"] - r["min"]
        norm = (native - r["min"]) / span if span != 0 else 0.0
        norm = max(0.0, min(1.0, norm))
        lines.append(f"{display}: {norm:.4f}")
    return "\n".join(lines[:SEARCH_RESULT_MAX_LINES])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize(name: str, native_val: float) -> float | None:
    """Normalize a native Vital parameter value to [0, 1].

    Returns None if the parameter is not in PARAM_RANGES.
    """
    r = PARAM_RANGES.get(name)
    if r is None:
        return None
    span = r["max"] - r["min"]
    if span == 0:
        return 0.0
    return (native_val - r["min"]) / span


def _denormalize(name: str, norm_val: float) -> float:
    """Convert a [0, 1] normalized value back to native Vital scale."""
    r = PARAM_RANGES.get(name)
    if r is None:
        return norm_val
    val = r["min"] + norm_val * (r["max"] - r["min"])
    if r.get("discrete"):
        val = round(val)
    return float(val)


def _param_priority(name: str) -> int:
    """Return sort priority for a parameter name (lower = earlier)."""
    for i, prefix in enumerate(PARAM_PRIORITY_PREFIXES):
        if name.startswith(prefix):
            return i
    return len(PARAM_PRIORITY_PREFIXES)  # unknown → last


def _extract_scalar_params(settings: dict) -> dict[str, float]:
    """Return {name: value} for all scalar (non-list, non-dict) params.

    All modulation slot scalars (amount, stereo, bipolar, bypass, power) are
    included — they are audible and are in PARAM_RANGES.  Only the top-level
    list/dict keys (modulations, lfos, wavetables, sample) are skipped.
    """
    _SKIP_KEYS = {"modulations", "lfos", "wavetables", "sample"}
    return {
        k: v
        for k, v in settings.items()
        if k not in _SKIP_KEYS
        and not isinstance(v, (list, dict))
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_preset_path(
    archetype: str,
    rng: random.Random,
    wavetable_lib: list | None = None,
    output_dir: Path | None = None,
    sample_id: str | None = None,
) -> dict:
    """Generate an N-step parameter path from init preset to a generated target.

    Parameters
    ----------
    archetype : str
        One of: bass, lead, pad, keys, pluck, sequence
    rng : random.Random
        Seeded RNG for reproducible generation.
    wavetable_lib : list | None
        Optional list of wavetable dicts (from load_wavetable_lib). If None,
        generate_preset will use the default wavetable.
    output_dir : Path | None
        If provided, each cumulative preset is saved as
        {output_dir}/{sample_id}_step{N}.vital
    sample_id : str | None
        Identifier for this sample. Auto-generated if not provided.

    Returns
    -------
    dict with keys:
        sample_id, archetype, n_iterations, n_changed_params,
        target_preset, iterations
    """
    # Lazy import to avoid circular imports
    from maestro.synth.preset_gen import generate_preset

    # --- IDs and setup ---
    if sample_id is None:
        sample_id = f"{archetype}_{uuid.uuid4().hex[:8]}"

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # --- Generate target preset ---
    target = generate_preset(archetype, rng, wavetable_lib=wavetable_lib)
    init_preset = copy.deepcopy(_INIT_PRESET)

    # --- Extract scalar params ---
    target_scalars = _extract_scalar_params(target["settings"])
    init_scalars = _extract_scalar_params(init_preset["settings"])

    # --- Compute changed params (normalized diff > 0.05) ---
    changed_scalar_params: dict[str, float] = {}  # {name: norm_val_in_target}
    skipped_unknown: list[str] = []

    for name, tgt_native in target_scalars.items():
        norm_tgt = _normalize(name, tgt_native)
        if norm_tgt is None:
            skipped_unknown.append(name)
            continue

        init_native = init_scalars.get(name)
        if init_native is None:
            # Param exists in target but not init — treat as changed
            changed_scalar_params[name] = norm_tgt
            continue

        norm_init = _normalize(name, init_native)
        if norm_init is None:
            changed_scalar_params[name] = norm_tgt
            continue

        if abs(norm_tgt - norm_init) > 0.05:
            changed_scalar_params[name] = norm_tgt

    # --- Extract modulation diffs ---
    # Vital stores mod route source/destination in settings["modulations"][i] but
    # stores amounts as separate scalar keys: modulation_{i+1}_amount (1-indexed).
    target_mods = target["settings"].get("modulations", [])
    modulations_changed: list[dict] = []
    for i, mod in enumerate(target_mods):
        source = mod.get("source", "")
        destination = mod.get("destination", "")
        if source or destination:
            amount_key = f"modulation_{i + 1}_amount"
            amount = float(target["settings"].get(amount_key, 0.0))
            modulations_changed.append({
                "source": source,
                "destination": destination,
                "amount": amount,
            })

    # --- Determine N iterations ---
    n_changed = len(changed_scalar_params)
    if n_changed <= 0:
        n_iterations = 1
    else:
        density = rng.uniform(PARAMS_PER_STEP_MIN, PARAMS_PER_STEP_MAX)
        n_iterations = int(math.ceil(n_changed / density))
    n_iterations = max(MIN_ITERATIONS, n_iterations)
    n_iterations = min(MAX_ITERATIONS, n_iterations)
    if n_changed > 0:
        # Never create more steps than changed params; avoids empty no-op tails.
        n_iterations = min(n_iterations, n_changed)
    else:
        n_iterations = 1

    # Track params that were applied with a deliberate mistake value so we can
    # schedule correction steps after the main loop.
    # Maps param name → correct target_norm value.
    mistaken_params: dict[str, float] = {}

    # --- Precompute init normalized values for from_norm tracking ---
    init_scalars_native = _extract_scalar_params(init_preset["settings"])

    # --- Plan step groups with primary/support family structure ---
    steps_params, step_plans = _plan_step_groups(
        changed_scalar_params=changed_scalar_params,
        init_scalars_native=init_scalars_native,
        n_iterations=n_iterations,
        rng=rng,
    )
    n_iterations = max(1, len(steps_params))

    # --- Build cumulative presets and iteration records ---
    cumulative = copy.deepcopy(init_preset)
    # Propagate the target's wavetables and sample into every cumulative preset
    # so that iteration clips share the same oscillator timbres as the GT audio.
    for key in ("wavetables", "sample", "lfos"):
        if key in target["settings"]:
            cumulative["settings"][key] = copy.deepcopy(target["settings"][key])
    iterations = []

    for step_idx, step_param_list in enumerate(steps_params):
        planner = step_plans[step_idx] if step_idx < len(step_plans) else {
            "primary_family": "other",
            "support_family": None,
            "planner_stage": "core",
            "checkpoint_revisit": False,
            "planned_param_names": [n for n, _ in step_param_list],
            "planned_primary_names": [n for n, _ in step_param_list],
            "allowed_primary_controls": [n for n, _ in step_param_list],
            "allowed_support_controls": [],
            "intended_edit_controls": [n for n, _ in step_param_list],
            "search_scope_type": "family",
            "intent_tags": ["focus:other", "stage:core"],
            "search_keyword": _search_keyword([n for n, _ in step_param_list]),
        }
        step_num = step_idx + 1
        params_changed: dict[str, float] = {}
        params_applied: dict[str, float] = {}
        params_delta: list[dict] = []

        # Snapshot cumulative settings BEFORE applying this step's params.
        # Used by _generate_search_result so search results show pre-step values.
        cumulative_settings_before = dict(cumulative["settings"])

        for name, norm_val in step_param_list:
            # from_norm: current cumulative normalized value BEFORE this step
            from_norm = _normalize(
                name,
                cumulative["settings"].get(name, init_scalars_native.get(name, 0.0)),
            )
            if from_norm is None:
                from_norm = 0.0

            # Target norm value (ground truth)
            target_norm = norm_val

            # With MISTAKE_PROB: apply a deliberately wrong value and schedule
            # a correction step later. Three mistake types chosen uniformly:
            #   overshoot  — go past the target (too far in the right direction)
            #   undershoot — only go partway toward target
            #   wrong dir  — move away from target
            is_param_mistake = False
            applied_norm = target_norm
            if rng.random() < MISTAKE_PROB:
                mistake_type = rng.randint(0, 2)
                if mistake_type == 0:  # overshoot
                    overshoot = (target_norm - from_norm) * rng.uniform(0.3, 0.7)
                    wrong_norm = target_norm + overshoot
                elif mistake_type == 1:  # undershoot
                    wrong_norm = from_norm + (target_norm - from_norm) * rng.uniform(0.2, 0.6)
                else:  # wrong direction
                    wrong_norm = from_norm - (target_norm - from_norm) * rng.uniform(0.2, 0.5)
                wrong_norm = max(0.0, min(1.0, wrong_norm))
                # Only flag if it actually differs from target by >0.02
                if abs(wrong_norm - target_norm) > 0.02:
                    applied_norm = wrong_norm
                    is_param_mistake = True
                    mistaken_params[name] = target_norm

            params_changed[name] = target_norm
            params_applied[name] = applied_norm

            # Apply the noisy/mistake value (via denormalize) to cumulative preset
            cumulative["settings"][name] = _denormalize(name, applied_norm)

            params_delta.append({
                "name": name,
                "from_norm": from_norm,
                "to_norm": applied_norm,
                "target_norm": target_norm,
                "mistake": is_param_mistake,
            })

        # Save cumulative preset to disk if requested
        cumulative_preset_path = None
        if output_dir is not None:
            preset_filename = f"{sample_id}_step{step_num}.vital"
            preset_filepath = output_dir / preset_filename
            with open(preset_filepath, "w") as f:
                json.dump(copy.deepcopy(cumulative), f)
            cumulative_preset_path = str(preset_filepath)

        # --- action_snippet: reapy high-level API, display names, normalized values ---
        # vital listen / render is NOT included here — added by the conversation
        # builder so the training data shows the model explicitly choosing to listen.
        set_lines = [
            "import reapy",
            "with reapy.inside_reaper():",
            "    fx = reapy.Project().tracks[0].fxs[0]",
        ]
        for name in sorted(params_applied.keys()):
            display = _json_key_to_display(name)
            norm = params_applied[name]
            set_lines.append(f'    fx.params["{display}"].value = {norm:.4f}')
        set_lines.append('    print("Done")')
        action_snippet = "\n".join(set_lines)

        # --- search_snippet + search_result: targeted param lookup before setting ---
        keyword = planner.get("search_keyword") or _search_keyword(list(params_applied.keys()))
        search_snippet = _build_search_snippet(keyword)
        search_result = _generate_search_result(keyword, cumulative_settings_before)

        # Only include in-memory cumulative_preset copy when there's no disk save
        cumulative_copy = None if output_dir is not None else copy.deepcopy(cumulative)

        iterations.append({
            "step": step_num,
            "params_changed": params_changed,
            "modulations_changed": modulations_changed,
            "cumulative_preset": cumulative_copy,
            "cumulative_preset_path": cumulative_preset_path,
            "action_snippet": action_snippet,
            "search_snippet": search_snippet,
            "search_result": search_result,
            "search_keyword": keyword,
            "primary_family": planner.get("primary_family"),
            "support_family": planner.get("support_family"),
            "planner_stage": planner.get("planner_stage"),
            "checkpoint_revisit": bool(planner.get("checkpoint_revisit", False)),
            "intent_tags": planner.get("intent_tags", []),
            "planned_param_names": planner.get("planned_param_names", []),
            "planned_primary_names": planner.get("planned_primary_names", []),
            "allowed_primary_controls": planner.get("allowed_primary_controls", []),
            "allowed_support_controls": planner.get("allowed_support_controls", []),
            "intended_edit_controls": planner.get("intended_edit_controls", []),
            "search_scope_type": planner.get("search_scope_type", "family"),
            "is_mistake_step": False,
            "params_applied": params_applied,
            "params_delta": params_delta,
        })

    # --- Build correction steps for any params that were set incorrectly ---
    # Group mistaken params into batches and append as dedicated correction steps.
    # Each correction step applies the correct target value, bringing cumulative
    # preset back on track. This guarantees n_remaining == 0 after all steps.
    correction_items = list(mistaken_params.items())  # [(name, target_norm), ...]
    rng.shuffle(correction_items)
    for batch_start in range(0, len(correction_items), CORRECTION_BATCH_SIZE):
        batch = correction_items[batch_start: batch_start + CORRECTION_BATCH_SIZE]
        step_num = len(iterations) + 1
        corr_params_applied: dict[str, float] = {}
        corr_params_delta: list[dict] = []
        cumulative_settings_before = dict(cumulative["settings"])

        for name, target_norm in batch:
            from_norm = _normalize(
                name,
                cumulative["settings"].get(name, init_scalars_native.get(name, 0.0)),
            ) or 0.0
            cumulative["settings"][name] = _denormalize(name, target_norm)
            corr_params_applied[name] = target_norm
            corr_params_delta.append({
                "name": name,
                "from_norm": from_norm,
                "to_norm": target_norm,
                "target_norm": target_norm,
                "mistake": False,
            })

        # Persist cumulative preset to disk if output_dir is set
        corr_preset_path = None
        if output_dir is not None:
            preset_filename = f"{sample_id}_step{step_num}.vital"
            preset_filepath = output_dir / preset_filename
            with open(preset_filepath, "w") as f:
                json.dump(copy.deepcopy(cumulative), f)
            corr_preset_path = str(preset_filepath)

        keyword = _search_keyword(list(corr_params_applied.keys()))
        set_lines = [
            "import reapy",
            "with reapy.inside_reaper():",
            "    fx = reapy.Project().tracks[0].fxs[0]",
        ]
        for name in sorted(corr_params_applied.keys()):
            display = _json_key_to_display(name)
            norm = corr_params_applied[name]
            set_lines.append(f'    fx.params["{display}"].value = {norm:.4f}')
        set_lines.append('    print("Done")')

        iterations.append({
            "step": step_num,
            "params_changed": {n: tn for n, tn in batch},
            "modulations_changed": [],
            "cumulative_preset": None if output_dir is not None else copy.deepcopy(cumulative),
            "cumulative_preset_path": corr_preset_path,
            "action_snippet": "\n".join(set_lines),
            "search_snippet": _build_search_snippet(keyword),
            "search_result": _generate_search_result(keyword, cumulative_settings_before),
            "search_keyword": keyword,
            "primary_family": _param_family(batch[0][0]) if batch else "other",
            "support_family": None,
            "planner_stage": "correction",
            "checkpoint_revisit": True,
            "intent_tags": ["stage:correction"],
            "planned_param_names": [n for n, _ in batch],
            "planned_primary_names": [n for n, _ in batch],
            "allowed_primary_controls": [n for n, _ in batch],
            "allowed_support_controls": [],
            "intended_edit_controls": [n for n, _ in batch],
            "search_scope_type": "correction",
            "is_mistake_step": False,
            "params_applied": corr_params_applied,
            "params_delta": corr_params_delta,
        })

    n_iterations = len(iterations)

    return {
        "sample_id": sample_id,
        "archetype": archetype,
        "n_iterations": n_iterations,
        "n_changed_params": n_changed,
        "target_preset": target,
        "has_mistake_step": len(mistaken_params) > 0,
        "n_corrections": len(correction_items),
        "iterations": iterations,
    }


def compare_preset_path(path_result: dict) -> dict:
    """Compare target preset vs final cumulative preset from a path_result.

    Categorises every scalar parameter by why it differs (or doesn't):

      untracked    — not in PARAM_RANGES; never normalised, never applied
      below_thresh — in PARAM_RANGES but |norm_tgt - norm_init| ≤ 0.05; skipped by design
      noisy        — was tracked and applied but final still differs from target by > 0.01 norm
      exact        — tracked and matches within 0.01 norm (success)
      missing      — key present in target settings but absent from final settings entirely

    Returns a dict with those five keys plus a top-level 'summary' string.
    """
    import re as _re
    _IGNORE_RE = _re.compile(r"(bypass|power|line_mapping)")

    target_settings = path_result["target_preset"]["settings"]
    init_settings   = _INIT_PRESET["settings"]

    # Get final cumulative settings from the last iteration
    last_it = path_result["iterations"][-1]
    if last_it.get("cumulative_preset"):
        final_settings = last_it["cumulative_preset"]["settings"]
    elif last_it.get("cumulative_preset_path"):
        import json as _json
        with open(last_it["cumulative_preset_path"]) as _f:
            final_settings = _json.load(_f)["settings"]
    else:
        raise RuntimeError("No cumulative preset available in path_result — pass output_dir to generate_preset_path()")

    untracked    = []
    below_thresh = []
    noisy        = []
    exact        = 0
    missing      = []

    for key, tgt_native in target_settings.items():
        if isinstance(tgt_native, (list, dict)):
            continue
        if _IGNORE_RE.search(key):
            continue

        if key not in final_settings:
            missing.append({"name": key, "target_native": tgt_native})
            continue

        final_native = final_settings[key]
        init_native  = init_settings.get(key, tgt_native)

        norm_tgt   = _normalize(key, tgt_native)
        norm_init  = _normalize(key, init_native)
        norm_final = _normalize(key, final_native)

        if norm_tgt is None:
            untracked.append({
                "name": key,
                "target_native": tgt_native,
                "init_native": init_native,
            })
            continue

        diff_from_init  = abs(norm_tgt - (norm_init or 0.0))
        diff_from_final = abs(norm_tgt - (norm_final or 0.0))

        if diff_from_init <= 0.05:
            below_thresh.append({
                "name": key,
                "target_native": tgt_native,
                "init_native": init_native,
                "norm_diff": diff_from_init,
            })
        elif diff_from_final > 0.01:
            noisy.append({
                "name": key,
                "target_native": tgt_native,
                "final_native": final_native,
                "target_norm": norm_tgt,
                "final_norm": norm_final,
                "norm_diff": diff_from_final,
            })
        else:
            exact += 1

    # Sort noisiest first
    noisy.sort(key=lambda x: x["norm_diff"], reverse=True)
    below_thresh.sort(key=lambda x: x["norm_diff"], reverse=True)

    summary = (
        f"{exact} exact | "
        f"{len(noisy)} noisy | "
        f"{len(below_thresh)} below_thresh | "
        f"{len(untracked)} untracked | "
        f"{len(missing)} missing"
    )

    return {
        "untracked":    untracked,
        "below_thresh": below_thresh,
        "noisy":        noisy,
        "exact":        exact,
        "missing":      missing,
        "summary":      summary,
    }
