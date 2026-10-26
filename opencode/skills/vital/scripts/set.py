#!/usr/bin/env python3
"""Set Vital parameters in preset.json, validated against the parameter table.

  set.py preset.json key=value [key=value ...] [--dry-run]

value: a raw JSON number (units table in SKILL.md), an enum label (case-insensitive, e.g. filter_1_model=Formant,
osc_1_destination="DIRECT OUT", reverb_on=on), or a number with a unit: Hz/kHz (cutoff/center keys), ms/s (env
times), % (0..1 and -1..1 keys), dB (volume, *_drive, *_gain), st (transposes).

All edits are validated first; nothing is written if any is invalid (exit 2). Prints one line per key:
  key: <before raw> (<human>) -> <after raw> (<human>)        (or "key: <raw> (<human>) (unchanged)")
"""
import argparse
import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitallib import UsageError, human, load_preset, params, parse_value, run_main, save_preset  # noqa


def fmt(key, v):
    raw = f"{float(v):g}" if isinstance(v, (int, float)) else repr(v)
    h = human(key, v)
    return raw if h == raw else f"{raw} ({h})"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset")
    ap.add_argument("edits", nargs="+", metavar="key=value")
    ap.add_argument("--dry-run", action="store_true", help="validate and print, do not write")
    a = ap.parse_args()

    preset = load_preset(a.preset)
    s = preset["settings"]
    table = params()
    edits = []
    for e in a.edits:
        key, sep, text = e.partition("=")
        key = key.strip()
        if not sep or not key or not text.strip():
            raise UsageError(f"expected key=value, got {e!r}")
        if key not in table:
            near = difflib.get_close_matches(key, table, n=4, cutoff=0.6)
            hint = f"; did you mean: {', '.join(near)}" if near else ""
            if key in ("wavetables", "modulations", "lfos", "sample"):
                hint = "; use wavetable.py / mod.py"
            raise UsageError(f"unknown parameter {key!r}{hint}")
        v = parse_value(key, text)
        p = table[key]
        if p["discrete"]:
            if abs(v - round(v)) > 1e-6:
                raise UsageError(f"{key}: discrete parameter needs a whole number, got {v:g}")
            v = float(round(v))
        lo, hi = p["min"], p["max"]
        if not lo - 1e-9 <= v <= hi + 1e-9:
            labels = f" ({', '.join(f'{lo + i:g}={l}' for i, l in enumerate(p['labels']))})" if "labels" in p else ""
            raise UsageError(f"{key}={v:g} out of range {lo:g}..{hi:g}{labels}")
        edits.append((key, min(max(v, lo), hi)))

    before = {k: s.get(k, table[k]["default"]) for k, _ in edits}
    for k, v in edits:
        s[k] = v
    for k, v in edits:
        b = before[k]
        line = f"{k}: {fmt(k, b)} (unchanged)" if float(b) == v else f"{k}: {fmt(k, b)} -> {fmt(k, v)}"
        print(line)
    if not a.dry_run:
        save_preset(a.preset, preset)


if __name__ == "__main__":
    run_main(main)
