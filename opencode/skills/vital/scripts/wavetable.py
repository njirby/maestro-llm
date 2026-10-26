#!/usr/bin/env python3
"""Search the vendored Vital wavetable library and swap a table into an oscillator.

  wavetable.py list [--query Q] [--ids wtA-wtB] [--limit N]
  wavetable.py set preset.json --osc N NAME_OR_ID

list: one table per line, `<id>  <name>  - <author>  <frames> frames  <source>[  silent at frame 0]`.
      Query terms (case-insensitive, all must match) search name, author and source (wave | audio | line).
      --ids wt101-wt166 keeps that id range (inclusive) and lists all of it unless --limit is given.
      source "audio" = resampled audio (vocals, instruments, noise); "line" = drawn line shapes; "wave" = drawn/
      generated single-cycle frames. "silent at frame 0" (library metadata) = the table's first keyframe is blank;
      it only sounds with osc_N_wave_frame > 0.
set:  copies the library table verbatim into settings.wavetables[N-1]. NAME matches exactly (case-insensitive,
      first variant if several share a name); use the id from `list` to pick a specific variant.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitallib import UsageError, load_preset, run_main, save_preset, wt_index, wt_load, wt_resolve  # noqa


def row(e):
    author = f"  - {e['author']}" if e["author"] else ""
    flag = "  silent at frame 0" if e.get("silent_frame0") else ""
    return f"{e['id']}  {e['name']}{author}  {e['frames']} frames  {e['source']}{flag}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("list", help="search the library")
    pl.add_argument("--query", "-q", default="")
    pl.add_argument("--ids", help="id range, e.g. wt101-wt166 (inclusive)")
    pl.add_argument("--limit", type=int)
    ps = sub.add_parser("set", help="swap a library table into an oscillator")
    ps.add_argument("preset")
    ps.add_argument("--osc", type=int, required=True, choices=(1, 2, 3))
    ps.add_argument("table", help="table name or id (wt042)")
    a = ap.parse_args()

    if a.cmd == "list":
        terms = a.query.lower().split()
        hits = [e for e in wt_index()
                if all(t in f"{e['name']} {e['author']} {e['source']}".lower() for t in terms)]
        what = [f"match {a.query!r}"] if terms else []
        if a.ids:
            m = re.fullmatch(r"\s*(wt\d{3})\s*-\s*(wt\d{3})\s*", a.ids)
            if not m or m.group(1) > m.group(2):
                raise UsageError(f"--ids takes a range like wt101-wt166, got {a.ids!r}")
            hits = [e for e in hits if m.group(1) <= e["id"] <= m.group(2)]
            what.append(f"in {m.group(1)}-{m.group(2)}")
        if a.limit is None:
            a.limit = len(hits) if a.ids else 40
        print(f"{len(hits)} of {len(wt_index())} tables" + "".join(f" {w}" for w in what))
        for e in hits[:max(a.limit, 0)]:
            print(row(e))
        if len(hits) > a.limit:
            print(f"... {len(hits) - a.limit} more (narrow with --query or raise --limit)")
        return

    preset = load_preset(a.preset)
    s = preset["settings"]
    e = wt_resolve(a.table)
    wts = s.get("wavetables")
    if not isinstance(wts, list) or len(wts) < 3:
        raise UsageError("preset has no settings.wavetables[3] list")
    old = wts[a.osc - 1].get("name") or "(unnamed)"
    wts[a.osc - 1] = wt_load(e)
    save_preset(a.preset, preset)
    n = a.osc
    print(f"osc {n} wavetable: {old!r} -> {e['name']!r} ({e['id']}{', ' + e['author'] if e['author'] else ''}, "
          f"{e['frames']} frames, {e['source']})")


if __name__ == "__main__":
    run_main(main)
