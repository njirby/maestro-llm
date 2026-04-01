"""
Canonical LFO waveform shape library for the Vital preset generator.

Each shape is a dict matching the format used in settings['lfos'][i].
Inactive LFO slots get the Triangle default; active slots get a shape
sampled from SHAPES weighted by archetype.
"""

# The Triangle is Vital's default LFO shape.
_TRIANGLE = {
    "name": "Triangle",
    "num_points": 3,
    "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
    "powers": [0.0, 0.0, 0.0],
    "smooth": False,
}

_SINE = {
    "name": "Sine",
    "num_points": 3,
    "points": [0.0, 1.0, 0.5, 0.0, 1.0, 1.0],
    "powers": [0.0, 0.0, 0.0],
    "smooth": True,
}

_SAW_UP = {
    "name": "Saw Up",
    "num_points": 3,
    "points": [0.0, 0.0, 0.999, 1.0, 1.0, 0.0],
    "powers": [0.0, 0.0, 0.0],
    "smooth": False,
}

_SAW_DOWN = {
    "name": "Saw Down",
    "num_points": 3,
    "points": [0.0, 1.0, 0.001, 0.0, 1.0, 1.0],
    "powers": [0.0, 0.0, 0.0],
    "smooth": False,
}

_SQUARE = {
    "name": "Square",
    "num_points": 5,
    "points": [0.0, 1.0, 0.0, 1.0, 0.499, 1.0, 0.501, 0.0, 1.0, 0.0],
    "powers": [0.0, 0.0, 0.0, 0.0, 0.0],
    "smooth": False,
}

_RAMP_UP = {
    "name": "Ramp Up",
    "num_points": 3,
    "points": [0.0, 0.0, 0.5, 0.5, 1.0, 1.0],
    "powers": [0.0, 0.0, 0.0],
    "smooth": False,
}

_RANDOM_STEP = {
    "name": "Random Step",
    "num_points": 9,
    "points": [
        0.0, 0.8, 0.125, 0.2, 0.25, 0.6,
        0.375, 0.9, 0.5, 0.3, 0.625, 0.7,
        0.75, 0.4, 0.875, 0.85, 1.0, 0.8,
    ],
    "powers": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "smooth": False,
}

SHAPES: dict[str, dict] = {
    "triangle": _TRIANGLE,
    "sine": _SINE,
    "saw_up": _SAW_UP,
    "saw_down": _SAW_DOWN,
    "square": _SQUARE,
    "ramp_up": _RAMP_UP,
    "random_step": _RANDOM_STEP,
}

# Sampling weights per archetype.  Triangle dominates as in real presets (~95% of slots).
# For the minority of *active* LFO slots, these weights control shape selection.
SHAPE_WEIGHTS: dict[str, dict[str, float]] = {
    "bass": {
        "triangle": 0.20, "sine": 0.15, "saw_up": 0.20, "saw_down": 0.15,
        "square": 0.20, "ramp_up": 0.05, "random_step": 0.05,
    },
    "lead": {
        "triangle": 0.25, "sine": 0.20, "saw_up": 0.15, "saw_down": 0.10,
        "square": 0.15, "ramp_up": 0.10, "random_step": 0.05,
    },
    "pad": {
        "triangle": 0.15, "sine": 0.45, "saw_up": 0.10, "saw_down": 0.10,
        "square": 0.05, "ramp_up": 0.10, "random_step": 0.05,
    },
    "keys": {
        "triangle": 0.30, "sine": 0.30, "saw_up": 0.10, "saw_down": 0.10,
        "square": 0.10, "ramp_up": 0.05, "random_step": 0.05,
    },
    "pluck": {
        "triangle": 0.25, "sine": 0.20, "saw_up": 0.20, "saw_down": 0.10,
        "square": 0.10, "ramp_up": 0.10, "random_step": 0.05,
    },
    "sequence": {
        "triangle": 0.10, "sine": 0.15, "saw_up": 0.15, "saw_down": 0.15,
        "square": 0.20, "ramp_up": 0.10, "random_step": 0.15,
    },
}

DEFAULT_WEIGHTS = {
    "triangle": 0.50, "sine": 0.20, "saw_up": 0.08, "saw_down": 0.07,
    "square": 0.08, "ramp_up": 0.04, "random_step": 0.03,
}


def sample_lfo_shape(archetype: str, rng) -> dict:
    """Return a deep copy of a randomly sampled LFO shape dict for the given archetype."""
    import copy
    weights = SHAPE_WEIGHTS.get(archetype, DEFAULT_WEIGHTS)
    names = list(weights.keys())
    w = [weights[n] for n in names]
    chosen = rng.choices(names, weights=w, k=1)[0]
    return copy.deepcopy(SHAPES[chosen])


def default_lfo_shape() -> dict:
    """Return a deep copy of the Triangle (inactive LFO default)."""
    import copy
    return copy.deepcopy(_TRIANGLE)


def build_lfos_array(active_lfo_indices: set[int], archetype: str, rng) -> list[dict]:
    """
    Build the settings['lfos'] 8-entry array.
    Active LFO slots get a sampled shape; others get Triangle.
    LFO indices are 1-based (lfo_1..lfo_8) but the array is 0-indexed.
    """
    result = []
    for i in range(8):
        lfo_num = i + 1  # 1-based
        if lfo_num in active_lfo_indices:
            result.append(sample_lfo_shape(archetype, rng))
        else:
            result.append(default_lfo_shape())
    return result
