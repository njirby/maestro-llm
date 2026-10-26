"""Reference set for calibration (calibrate.py -> calib.json): N generated presets (maestro.synth.preset_gen, with its
loudness normalization and level QC, as in scripts/build_v5_manifest.py), each playing a Lakh clip, rendered through the
skill's render.py in one network-less daw-farm session (renderer.ContainerRenderer), i.e. the same renders the agent
and the verified notes hear.

  .venv/bin/python -m maestro.sounddiff.refset outputs/sounddiff/refset_v2 --n 480
  # then descriptors + CLAP adjectives (ear-graft container, --network none; CPU is enough for CLAP):
  docker run --rm --network none ... ear-graft:26.09 python -m maestro.sounddiff.refset measure outputs/sounddiff/refset_v2
  .venv/bin/python -m maestro.sounddiff.calibrate outputs/sounddiff/refset_v2

Writes <out>/presets/rNNNN.vital, <out>/wav/rNNNN.wav, <out>/renders.jsonl (measure.py's input) and qc.jsonl.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

MAX_TRIES = 8


def candidate(seed: int, j: int, k: int, lib, clips) -> dict:
    from maestro.synth.preset_gen import ARCHETYPES, generate_preset
    from . import edits
    from .renderer import clean_notes
    rng = random.Random(f"refset-{seed}-{j}-{k}")
    arch = ARCHETYPES[j % len(ARCHETYPES)]
    p = generate_preset(arch, rng, wavetable_lib=lib)
    c, notes = edits.pick_clip(clips, rng)
    return {"preset": p, "notes": clean_notes({"bpm": c["bpm"], "notes": notes}), "archetype": arch,
            "normalized": False, "try": k}


def build(out: Path, n: int, seed: int, catalog: str, wavetables: str) -> None:
    # host-side imports (the measure step runs in the ear-graft image, which has no pretty_midi / DawDreamer client)
    from maestro.synth.preset_gen import level_qc, normalize_volume, render_level
    from maestro.synth.wavetable_lib import load_wavetable_lib
    from . import edits
    from .renderer import ContainerRenderer
    lib = load_wavetable_lib(wavetables)
    clips = edits.load_clips(catalog)
    for sub in ("presets", "wav"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    accepted, current, tries = {}, {}, {j: 0 for j in range(n)}
    with ContainerRenderer(cache_dir=out / "render_cache") as r, open(out / "qc.jsonl", "w") as qc:
        while len(accepted) < n:
            for j in range(n):
                if j not in accepted and j not in current:
                    if tries[j] >= MAX_TRIES:
                        sys.exit(f"QC failed {MAX_TRIES} candidates for r{j:04d}")
                    current[j] = candidate(seed, j, tries[j], lib, clips)
                    tries[j] += 1
            # one render call per notes clip (the renderer batches presets that share notes)
            by_notes = {}
            for j, cand in current.items():
                by_notes.setdefault(json.dumps(cand["notes"], sort_keys=True), []).append(j)
            wavs = {}
            for js in by_notes.values():
                wavs.update(r.render({str(j): current[j]["preset"] for j in js}, current[js[0]]["notes"]))
            for j in list(current):
                cand, y = current[j], wavs[str(j)]
                s = cand["preset"]["settings"]
                end = max(x["start_s"] + x["dur_s"] for x in cand["notes"]["notes"])
                lv = render_level(y, 44100, end)
                ok, why = level_qc(lv)
                new_vol = normalize_volume(s, lv) if lv["finite"] and lv["peak_dbfs"] > -100 and not cand["normalized"] \
                    else None
                qc.write(json.dumps({"name": f"r{j:04d}", "try": cand["try"], "archetype": cand["archetype"],
                                     "normalized": cand["normalized"], "ok": ok and new_vol is None, "reason": why,
                                     "new_volume": new_vol, **lv}) + "\n")
                if new_vol is not None:
                    s["volume"] = new_vol
                    cand["normalized"] = True
                    continue
                del current[j]
                if ok:
                    name = f"r{j:04d}"
                    (out / "presets" / f"{name}.vital").write_text(json.dumps(cand["preset"], separators=(",", ":")))
                    shutil.copyfile(r.path_of(cand["preset"], cand["notes"]), out / "wav" / f"{name}.wav")
                    accepted[j] = {"name": name, "preset": f"presets/{name}.vital", "notes": cand["notes"]["notes"],
                                   "bpm": cand["notes"]["bpm"], "omni": False, "archetype": cand["archetype"],
                                   "rms_dbfs": round(lv["rms_dbfs"], 2), "tries": cand["try"] + 1}
            print(f"accepted {len(accepted)}/{n}, re-rendering {len(current)} normalized, "
                  f"{n - len(accepted) - len(current)} to regenerate", flush=True)
    with open(out / "renders.jsonl", "w") as f:
        for j in range(n):
            f.write(json.dumps(accepted[j]) + "\n")
    print(f"{n} reference renders -> {out}/renders.jsonl")


def measure(d: Path) -> None:
    """Descriptors + CLAP adjective sims for every render (measure.measure_dir), CLAP only, on the GPU if one is
    visible, else CPU. Runs inside the ear-graft container."""
    import torch
    from . import measure as m
    dev = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    m.measure_dir(d, m.Embedders(dev, omni=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", default="build", choices=["build", "measure"])
    ap.add_argument("out")
    ap.add_argument("--n", type=int, default=480)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--catalog", default="outputs/midi_clips/lakh_catalog_small.jsonl")
    ap.add_argument("--wavetables", default="data/wavetable_lib.json")
    a = ap.parse_args()
    if a.cmd == "build":
        build(Path(a.out), a.n, a.seed, a.catalog, a.wavetables)
    else:
        measure(Path(a.out))
