"""
Wavetable library builder and loader for the Vital preset generator.

Builds a JSON library of wavetable dicts by reading:
  1. .vitaltable files (direct JSON in the same format as settings['wavetables'][i])
  2. Embedded wavetables extracted from downloaded .vital preset files

Usage (CLI):
    python -m maestro.synth.wavetable_lib \
        --vital-dir ~/.local/share/vital \
        --presets-dir ~/Downloads/vital_presets \
        --output data/wavetable_lib.json

The output is a JSON list of wavetable dicts, each suitable for direct
use as settings['wavetables'][i] in a generated preset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path


def _fingerprint(wt: dict) -> str:
    """Stable hash of a wavetable dict, used for deduplication."""
    return hashlib.md5(json.dumps(wt, sort_keys=True).encode()).hexdigest()


def _load_vitaltable(path: Path) -> dict | None:
    """Load a .vitaltable file as a wavetable dict."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        if "groups" in d:
            return d
    except Exception:
        pass
    return None


def _extract_wavetables_from_preset(path: Path) -> list[dict]:
    """Extract the wavetables array from a .vital preset file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        wts = d.get("settings", {}).get("wavetables", [])
        return [w for w in wts if isinstance(w, dict) and "groups" in w]
    except Exception:
        return []


def build_wavetable_lib(
    vital_dir: Path | None = None,
    presets_dir: Path | None = None,
    max_from_presets: int = 2000,
    seed: int = 42,
) -> list[dict]:
    """
    Build a deduplicated list of wavetable dicts from:
      - All .vitaltable files under vital_dir
      - Wavetables embedded in preset files under presets_dir (up to max_from_presets)

    Returns a list of unique wavetable dicts ready for use as settings['wavetables'][i].
    """
    seen: dict[str, dict] = {}  # fingerprint → wavetable dict

    # 1. .vitaltable files (direct JSON, highest quality)
    if vital_dir and vital_dir.exists():
        for p in vital_dir.rglob("*.vitaltable"):
            wt = _load_vitaltable(p)
            if wt is not None:
                fp = _fingerprint(wt)
                if fp not in seen:
                    seen[fp] = wt
        print(f"  .vitaltable files: {len(seen)} unique wavetables", file=sys.stderr)

    # 2. Embedded wavetables from preset files (subsample for speed)
    if presets_dir and presets_dir.exists():
        preset_files = list(presets_dir.rglob("*.vital"))
        rng = random.Random(seed)
        rng.shuffle(preset_files)
        preset_files = preset_files[:max_from_presets]

        before = len(seen)
        for p in preset_files:
            for wt in _extract_wavetables_from_preset(p):
                fp = _fingerprint(wt)
                if fp not in seen:
                    seen[fp] = wt
        print(
            f"  Preset files scanned: {len(preset_files):,}  "
            f"new wavetables found: {len(seen) - before}",
            file=sys.stderr,
        )

    result = list(seen.values())
    print(f"  Total unique wavetables in library: {len(result)}", file=sys.stderr)
    return result


VITAL_BUNDLED_DATA_DIRS = [
    Path.home() / ".local" / "share" / "vital",  # Linux
    Path.home() / "Library" / "Application Support" / "Vital",  # macOS
]


def _discover_bundled_wavetables() -> list[dict]:
    """Load wavetables from Vital's own installed assets as fallback.

    Scans the Vital data directory (which contains pack subdirectories like
    Factory/Wavetables/, Glorkglunk/Wavetables/, etc.) for .vitaltable files.
    """
    for d in VITAL_BUNDLED_DATA_DIRS:
        if d.exists():
            return build_wavetable_lib(vital_dir=d)
    return []


def load_wavetable_lib(path: Path) -> list[dict]:
    """Load a previously built wavetable library from JSON file.

    Falls back to Vital's bundled wavetables if the JSON file is missing.
    """
    p = Path(path)
    if p.exists():
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"WARNING: failed to load wavetable lib at {path}: {e}", file=sys.stderr)

    print(
        f"WARNING: wavetable lib not found at {path}. "
        "Falling back to Vital's bundled wavetables. "
        f"Run: python -m maestro.synth.wavetable_lib --output {path} to build the full library.",
        file=sys.stderr,
    )
    bundled = _discover_bundled_wavetables()
    if bundled:
        print(f"  Loaded {len(bundled)} bundled wavetables from Vital installation.", file=sys.stderr)
        return bundled

    print(
        "  WARNING: Vital bundled wavetables not found either. "
        "All presets will use Init wavetable (no timbral diversity).",
        file=sys.stderr,
    )
    return []


def get_default_wavetable() -> dict:
    """Return Vital's Init wavetable (sine) as a fallback."""
    return {
        "author": "",
        "full_normalize": True,
        "groups": [{"components": [{"interpolation": 1, "interpolation_style": 1,
                                    "keyframes": [{"position": 0, "wave_data": ""}],
                                    "type": "Wave Source"}]}],
        "name": "Init",
        "remove_all_dc": True,
        "version": "99999.9.9",
    }


def _dedup_wavetable_lib(lib: list[dict]) -> list[dict]:
    """Return a name-deduplicated view of the lib (one entry per unique name).

    The corpus-extracted library contains many duplicates — e.g. 121 'Init'
    entries — because many presets share the default wavetable.  Deduplicating
    by name gives each unique timbre equal sampling probability.
    """
    seen: dict[str, dict] = {}
    for wt in lib:
        name = wt.get("name", "")
        if name not in seen:
            seen[name] = wt
    return list(seen.values())


# Module-level cache so dedup runs once per process.
_DEDUPED_LIB: list[dict] | None = None
_DEDUPED_SOURCE: list[dict] | None = None


def sample_wavetable(lib: list[dict], rng) -> dict:
    """Sample a random wavetable from the library, or return Init if lib is empty.

    Deduplicates by name before sampling so heavily-repeated wavetables (Init,
    Basic Shapes) don't dominate.  The deduplicated view is cached on first call.
    """
    import copy
    global _DEDUPED_LIB, _DEDUPED_SOURCE
    if not lib:
        return get_default_wavetable()
    if lib is not _DEDUPED_SOURCE:
        _DEDUPED_LIB = _dedup_wavetable_lib(lib)
        _DEDUPED_SOURCE = lib
    return copy.deepcopy(rng.choice(_DEDUPED_LIB))


def get_n_frames(wt: dict) -> int:
    """Return the number of keyframes in a wavetable (used to set wave_frame range)."""
    try:
        keyframes = wt["groups"][0]["components"][0]["keyframes"]
        return max(1, len(keyframes))
    except (KeyError, IndexError):
        return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--vital-dir", type=Path,
                   default=Path("~/.local/share/vital").expanduser())
    p.add_argument("--presets-dir", type=Path,
                   default=Path("~/Downloads/vital_presets").expanduser())
    p.add_argument("--output", type=Path, default=Path("data/wavetable_lib.json"))
    p.add_argument("--max-from-presets", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    print("Building wavetable library...", file=sys.stderr)
    lib = build_wavetable_lib(
        vital_dir=args.vital_dir,
        presets_dir=args.presets_dir,
        max_from_presets=args.max_from_presets,
        seed=args.seed,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(lib, f)
    print(f"Wrote {len(lib)} wavetables to {args.output}", file=sys.stderr)


if __name__ == "__main__":
    _main()
