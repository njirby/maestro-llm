#!/usr/bin/env python3
"""Rebuild the vendored data of the vital OpenCode skill (opencode/skills/vital/data/) from the repo sources.

  python3 scripts/build_vital_skill_data.py              # rebuild from sources (stdlib only, ~10 s)
  python3 scripts/build_vital_skill_data.py --measure    # also re-measure the silent-at-frame-0 flags (docker, ~1 min)

Outputs (deterministic; gzip mtime=0):
  params.json            key -> {min, max, default, discrete, labels?}: maestro/synth/param_ranges.json + the live
                         engine enum labels (skills_dawdreamer/vital/explore/crosscheck.json), with the known range-table
                         errors corrected from the engine label counts (transposes ±48, unison 1..16, delay tempo
                         4..12, filter_fx_on is the Off/On switch, ...)
  init_preset.json       copy of maestro/synth/init_preset.json
  engine_params.json, engine_enum_values.json   copies of the live Vital 1.6.0 dump in skills_dawdreamer/vital/
  wavetables.idx.json    [{id, name, author, frames, source, offset, length, silent_frame0?}], one per distinct table
                         (distinct `groups` content) of data/wavetable_lib.json, in library order (ids wt001..)
  wavetables.bin         concatenated gzip members, one per table (offset/length from the index)

silent_frame0 = stereo RMS < 1e-4 for a C4 0.4 s note on osc 1 alone (level 1, wave_frame 0), rendered through the
skill's own render path (DawDreamer + Vital in the daw-farm image, 4 s pre-roll). The 2026-10-25 measurement is
recorded in SILENT_FRAME0 below; --measure re-renders every table (frames 0/128/256) and replaces it.
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "opencode/skills/vital/data"
IMAGE = "daw-farm/reaper:latest"

# Measured 2026-10-25 (none of the 499 tables is silent at frames 128/256; these 18 are silent at frame 0 only).
SILENT_FRAME0 = ["wt121", "wt125", "wt166", "wt167", "wt184", "wt191", "wt211", "wt243", "wt296", "wt301", "wt302",
                 "wt303", "wt304", "wt329", "wt345", "wt384", "wt447", "wt477"]

# Runs inside the daw-farm container: python3 - SHARD NSHARDS OUT.jsonl (skill scripts dir on sys.path).
_MEASURE_PY = r'''
import copy, json, sys
sys.path.insert(0, sys.argv[4])
import vitallib as V
shard, nsh, outp = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
V.install_x11_error_handler()
import dawdreamer as daw
import numpy as np
SR, PRE = 44100, 4.0
eng = daw.RenderEngine(SR, 512)
syn = eng.make_plugin_processor("v", "/usr/lib/vst3/Vital.vst3")
eng.load_graph([(syn, [])])
eng.set_bpm(120.0)
init = json.load(open(V.DATA / "init_preset.json"))
state = f"/tmp/ms_{shard}.state"
with open(outp, "w") as fo:
    for e in V.wt_index()[shard::nsh]:
        wt = V.wt_load(e)
        res = {"id": e["id"], "name": e["name"]}
        for frame in (0, 128, 256):
            p = copy.deepcopy(init)
            s = p["settings"]
            s["wavetables"][0] = wt
            s.update(osc_1_on=1.0, osc_1_level=1.0, osc_1_wave_frame=float(frame))
            open(state, "wb").write(V.build_state(p))
            syn.load_state(state)
            syn.clear_midi()
            syn.add_midi_note(60, 100, PRE, 0.4)
            eng.render(PRE + 0.4)
            a = np.asarray(syn.get_audio())[:, int(PRE * SR):]
            res[f"rms_f{frame}"] = float(np.sqrt((a ** 2).mean()))
        fo.write(json.dumps(res) + "\n")
'''


def numeric_label(s):
    s = s.replace(" voices", "").rstrip("v")
    try:
        float(s)
        return True
    except ValueError:
        return False


def build_params():
    ranges = json.loads((REPO / "maestro/synth/param_ranges.json").read_text())
    rows = {r["key"]: r for r in
            json.loads((REPO / "skills_dawdreamer/vital/explore/crosscheck.json").read_text())["enum_rows"]}
    params = {}
    for k, r in sorted(ranges.items()):
        p = {"min": r["min"], "max": r["max"], "default": r["default"], "discrete": bool(r["discrete"])}
        row = rows.get(k)
        if k == "filter_fx_on":  # range table maps it to the FX filter *model* (engine idx 142); the switch is idx 143
            p.update(max=1.0, labels=["Off", "On"])
        elif row:
            labels = [v for v in row["values"] if v != ""]
            if k.endswith("_destination") and labels[-1] == "Filter 1":  # stray 15th label in the engine dump
                labels = labels[:-1]
            if k.endswith("_distortion_type") and k.startswith("osc_"):
                labels = ["None", "Sync", "Formant", "Quantize", "Bend", "Squeeze", "Pulse", "FM <- Osc A",
                          "FM <- Osc B", "FM <- Sample", "RM <- Osc A", "RM <- Osc B", "RM <- Sample"]
            p["max"] = p["min"] + len(labels) - 1  # engine label count is authoritative (range table is off for some)
            if not all(numeric_label(v) for v in labels):
                p["labels"] = labels
        params[k] = p
    (OUT / "params.json").write_text(json.dumps(params, indent=0, sort_keys=True))
    return params


def build_wavetables(silent):
    lib = json.loads((REPO / "data/wavetable_lib.json").read_text())
    seen, index = set(), []
    with open(OUT / "wavetables.bin", "wb") as fb:
        for e in lib:
            h = hashlib.md5(json.dumps(e.get("groups"), sort_keys=True).encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            comps = [c for g in e.get("groups", []) for c in g.get("components", [])]
            frames = max((len(c.get("keyframes", [])) for c in comps), default=0)
            types = [c.get("type") for c in comps]
            source = "audio" if "Audio File Source" in types else "line" if "Line Source" in types else "wave"
            blob = gzip.compress(json.dumps(e, separators=(",", ":")).encode(), 6, mtime=0)
            index.append({"id": f"wt{len(index) + 1:03d}", "name": e.get("name") or "(unnamed)",
                          "author": e.get("author") or "", "frames": frames, "source": source,
                          "offset": fb.tell(), "length": len(blob)})
            fb.write(blob)
    write_index(index, silent)
    return index


def write_index(index, silent):
    for e in index:
        e.pop("silent_frame0", None)
        if e["id"] in silent:
            e["silent_frame0"] = True
    (OUT / "wavetables.idx.json").write_text(json.dumps(index, indent=0))


def measure(nshards=8):
    """Render every table in the daw-farm image; return ids silent at frame 0."""
    scripts = REPO / "opencode/skills/vital/scripts"
    with tempfile.TemporaryDirectory(dir=REPO / "outputs") as td:
        Path(td, "measure.py").write_text(_MEASURE_PY)
        cmd = " ".join(f"python3 {td}/measure.py {i} {nshards} {td}/s{i}.jsonl {scripts} & p{i}=$!;"
                       for i in range(nshards))
        cmd += " rc=0; " + " ".join(f"wait $p{i} || rc=1;" for i in range(nshards)) + " exit $rc"
        boot = "Xvfb :99 -screen 0 1024x768x24 -nolisten tcp >/dev/null 2>&1 & sleep 2; export DISPLAY=:99; "
        subprocess.run(["docker", "run", "--rm", "--user", "1000:1000", "--network", "none",
                        "-e", "PYTHONDONTWRITEBYTECODE=1", "-v", f"{REPO}:{REPO}", "--entrypoint", "bash",
                        IMAGE, "-c", boot + cmd], check=True)
        rows = [json.loads(line) for i in range(nshards) for line in open(f"{td}/s{i}.jsonl")]
    never = [r["id"] for r in rows if max(r["rms_f0"], r["rms_f128"], r["rms_f256"]) < 1e-4]
    if never:
        print(f"tables silent at all measured frames: {never}", file=sys.stderr)
    return sorted(r["id"] for r in rows if r["rms_f0"] < 1e-4)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--measure", action="store_true", help="re-measure silent_frame0 flags in the daw-farm image")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    params = build_params()
    shutil.copyfile(REPO / "maestro/synth/init_preset.json", OUT / "init_preset.json")
    for f in ("engine_params.json", "engine_enum_values.json"):
        shutil.copyfile(REPO / "skills_dawdreamer/vital" / f, OUT / f)
    index = build_wavetables(set(SILENT_FRAME0))
    if a.measure:
        silent = measure()
        write_index(index, set(silent))
        if silent != SILENT_FRAME0:
            print(f"silent_frame0 changed; update SILENT_FRAME0 to: {json.dumps(silent)}", file=sys.stderr)
    n_silent = sum(1 for e in index if e.get("silent_frame0"))
    print(f"{len(params)} params; {len(index)} tables ({n_silent} silent at frame 0); "
          f"{(OUT / 'wavetables.bin').stat().st_size / 1e6:.1f} MB -> {os.path.relpath(OUT, REPO)}")


if __name__ == "__main__":
    main()
