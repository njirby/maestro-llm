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
import uuid
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

NOISE_STD = 0.08          # normalized units — ~8% of param range
NOISE_PROB = 0.35         # 35% of params in each step get noise applied
MISTAKE_PROB = 0.25       # 25% of paths get a deliberate mistake step

# (inclusive_upper_bound, n_iterations)
# Calibrated to the actual param-change distribution from generate_preset()
# at threshold=0.15: typically 96-133 changed params per archetype.
# Percentile-based splits: ~p20=102, p40=108, p60=114, p80=122.
N_TABLE = [(56, 8), (72, 10), (96, 13), (120, 17), (999, 20)]

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
        return ""
    dominant = _Counter(prefixes).most_common(1)[0][0]
    return _json_key_to_display(dominant).lower()


def _generate_search_result(keyword: str, cumulative_settings: dict) -> str:
    """Generate synthetic search output: display names matching keyword + current values.

    Values are the cumulative normalized state BEFORE the step is applied.
    """
    lines = []
    for display in sorted(_DISPLAY_TO_KEY):
        if keyword in display.lower():
            key = _DISPLAY_TO_KEY[display]
            r = PARAM_RANGES.get(key)
            if r is None:
                continue
            native = cumulative_settings.get(key, r["default"])
            span = r["max"] - r["min"]
            norm = (native - r["min"]) / span if span != 0 else 0.0
            norm = max(0.0, min(1.0, norm))
            lines.append(f"{display}: {norm:.4f}")
    return "\n".join(lines)


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
    target_scalars_all = _extract_scalar_params(target["settings"])  # pre-filter copy
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
    n_iterations = next(n for threshold, n in N_TABLE if n_changed <= threshold)
    n_iterations = max(2, n_iterations)
    n_iterations = min(n_iterations, 20)

    # --- Sort changed params by priority ---
    sorted_params = sorted(
        changed_scalar_params.items(),
        key=lambda kv: (_param_priority(kv[0]), kv[0]),
    )

    # --- Partition params across N iterations (greedy fill) ---
    budget = math.ceil(n_changed / n_iterations) if n_changed > 0 else 1
    steps_params: list[list[tuple[str, float]]] = [[] for _ in range(n_iterations)]
    param_iter = iter(sorted_params)
    for step_idx in range(n_iterations):
        while len(steps_params[step_idx]) < budget:
            try:
                steps_params[step_idx].append(next(param_iter))
            except StopIteration:
                break

    # --- Decide mistake step ---
    has_mistake_step = rng.random() < MISTAKE_PROB
    mistake_step_idx = None
    if has_mistake_step and n_iterations >= 2:
        # Pick any step except the last
        mistake_step_idx = rng.randint(0, n_iterations - 2)
    else:
        has_mistake_step = False

    # --- Precompute init normalized values for from_norm tracking ---
    init_scalars_native = _extract_scalar_params(init_preset["settings"])

    # --- Build cumulative presets and iteration records ---
    cumulative = copy.deepcopy(init_preset)
    # Propagate the target's wavetables and sample into every cumulative preset
    # so that iteration clips share the same oscillator timbres as the GT audio.
    for key in ("wavetables", "sample", "lfos"):
        if key in target["settings"]:
            cumulative["settings"][key] = copy.deepcopy(target["settings"][key])
    iterations = []

    for step_idx, step_param_list in enumerate(steps_params):
        step_num = step_idx + 1
        is_mistake = step_idx == mistake_step_idx
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

            # Compute applied value: start with target, then optionally add noise
            applied_norm = target_norm

            # Mistake step: 25% of params move in wrong direction
            is_param_mistake = False
            if is_mistake and rng.random() < 0.25:
                init_norm = _normalize(name, init_scalars_native.get(name, 0.0))
                if init_norm is None:
                    init_norm = 0.0
                wrong_norm = init_norm - (target_norm - init_norm) * rng.uniform(0.3, 0.7)
                wrong_norm = max(0.0, min(1.0, wrong_norm))
                # Only flag as mistake if it actually moved away from target
                if abs(wrong_norm - from_norm) > 0.01:
                    applied_norm = wrong_norm
                    is_param_mistake = True
            elif rng.random() < NOISE_PROB:
                # Noisy progression
                applied_norm = target_norm + rng.gauss(0, NOISE_STD)
                applied_norm = max(0.0, min(1.0, applied_norm))

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

        # On the final step: silently close all remaining gaps so the saved
        # preset == target and the rendered audio matches the GT exactly.
        # Do NOT touch params_applied — the action_snippet should only show
        # the natural params for this step, not a massive cleanup dump.
        is_last = step_idx == n_iterations - 1
        if is_last:
            _skip = {"modulations", "lfos", "wavetables", "sample"}
            for _k, _tgt_v in target["settings"].items():
                if isinstance(_tgt_v, (list, dict)) or _k in _skip:
                    continue
                cumulative["settings"][_k] = _tgt_v
            cumulative["settings"]["modulations"] = copy.deepcopy(
                target["settings"].get("modulations", [])
            )

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
        keyword = _search_keyword(list(params_applied.keys()))
        search_snippet = (
            "import reapy\n"
            "with reapy.inside_reaper():\n"
            "    fx = reapy.Project().tracks[0].fxs[0]\n"
            f"    hits = [(p.name, round(p.normalized, 4)) for p in fx.params\n"
            f"            if '{keyword}' in p.name.lower()]\n"
            "    for name, val in hits:\n"
            '        print(f"{name}: {val:.4f}")'
        )
        search_result = _generate_search_result(keyword, cumulative_settings_before)

        # Only include in-memory cumulative_preset copy when there's no disk save
        cumulative_copy = None if output_dir is not None else copy.deepcopy(cumulative)

        iterations.append({
            "step": step_num,
            "params_changed": params_changed,
            "modulations_changed": modulations_changed if is_last else [],
            "cumulative_preset": cumulative_copy,
            "cumulative_preset_path": cumulative_preset_path,
            "action_snippet": action_snippet,
            "search_snippet": search_snippet,
            "search_result": search_result,
            "search_keyword": keyword,
            "is_mistake_step": is_mistake,
            "params_applied": params_applied,
            "params_delta": params_delta,
        })

    return {
        "sample_id": sample_id,
        "archetype": archetype,
        "n_iterations": n_iterations,
        "n_changed_params": n_changed,
        "target_preset": target,
        "has_mistake_step": has_mistake_step,
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
