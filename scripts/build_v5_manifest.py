#!/usr/bin/env python3
"""GT manifest for the v5 builder: N (GT preset, Lakh clip + BPM, init preset, notes.json, rendered target) samples.

  .venv/bin/python scripts/build_v5_manifest.py --name pilot20 [--n 20] [--seed 0] [--jobs 6]

Per sample: an archetype from a balanced, seeded mix over preset_gen.ARCHETYPES; a preset from preset_gen; a melodic
Lakh clip (sounddiff.edits.load_clips/clip_notes, zero-length notes dropped); the target rendered with the skill's
render.py in the daw-farm container (exactly what the agent's own renders go through). A render outside
preset_gen.LOUDNESS_BAND_DBFS gets its master volume normalized (preset_gen.normalize_volume) and is rendered again;
a candidate failing preset_gen.level_qc (digital silence, non-finite, too quiet) after that is regenerated with the
next attempt seed.

Writes outputs/v5_manifest/<name>/{manifest.jsonl, presets/, notes/, targets/, qc.jsonl}. Row schema:
  {id, archetype, gt_preset_path, init_preset_path, notes_path, target_wav_path, bpm,
   clip: {midi_path, track_idx, start_s, program}, target_rms_dbfs}
Paths are absolute. target_rms_dbfs is the RMS of the whole target wav. Same --seed -> same presets/clips; the QC
outcome only depends on the render, so a sample can differ only if a candidate sits on the QC threshold.
--qc-off accepts every first candidate as rendered (level statistics of the raw generator).
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from maestro.sounddiff.edits import clip_notes, load_clips  # noqa: E402
from maestro.synth.preset_gen import ARCHETYPES, generate_preset, level_qc, normalize_volume, render_level  # noqa: E402
from maestro.synth.wavetable_lib import load_wavetable_lib  # noqa: E402

SKILL = REPO / "opencode/skills/vital"
INIT_PRESET = SKILL / "data/init_preset.json"
IMAGE = "daw-farm/reaper:latest"
MAX_TRIES = 8


def pick_clip(clips: list[dict], rng: random.Random) -> tuple[dict, list[dict]]:
    """Like edits.pick_clip, minus zero-length notes (render.py rejects them)."""
    while True:
        c = rng.choice(clips)
        notes = [n for n in clip_notes(c) if n["dur_s"] > 0]
        if len(notes) >= 3:
            return c, notes


def candidate(seed: int, i: int, k: int, archetype: str, lib: list[dict], clips: list[dict]) -> dict:
    """Attempt k of sample i: own RNG per (seed, i, k) so samples do not shift when another one is regenerated."""
    rng = random.Random(f"v5:{seed}:{i}:{k}")
    preset = generate_preset(archetype, rng, wavetable_lib=lib, name=f"v5_{seed}_{i:04d}")
    c, notes = pick_clip(clips, rng)
    preset["settings"]["beats_per_minute"] = c["bpm"] / 60.0  # as sounddiff.edits; Vital takes host tempo anyway
    return {"preset": preset, "clip": c, "notes": notes, "bpm": float(c["bpm"])}


class Renderer:
    """One long-lived daw-farm container (--network none, Xvfb inside); each batch is a `docker exec` running the
    skill's render.py in parallel processes, one fresh Vital per render as the agent gets."""

    def __init__(self, jobs: int):
        self.jobs, self.name = jobs, f"wsc-manifest-{os.getpid()}"
        subprocess.run(["docker", "run", "-d", "--rm", "--network", "none", "--name", self.name, "--user",
                        f"{os.getuid()}:{os.getgid()}", "-v", "/home/nate:/home/nate", "-e",
                        "VITAL_VST3=/usr/lib/vst3/Vital.vst3", "-e", "DISPLAY=:99", "--entrypoint", "bash", IMAGE,
                        "-c", "Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 & exec sleep infinity"],
                       check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["docker", "exec", self.name, "bash", "-c",
                        "for i in $(seq 100); do [ -e /tmp/.X11-unix/X99 ] && exit 0; sleep 0.1; done; exit 1"],
                       check=True)

    def render(self, work: Path, names: list[str]) -> None:
        """work/presets/<n>.vital + work/notes/<n>.json -> work/targets/<n>.wav"""
        (work / "targets").mkdir(exist_ok=True)
        (work / "_batch.txt").write_text("\n".join(names) + "\n")
        cmd = (f"cd {work}; xargs -P {self.jobs} -I{{}} sh -c 'python3 {SKILL}/scripts/render.py presets/{{}}.vital "
               f"notes/{{}}.json targets/{{}}.wav >/dev/null 2>&1 || echo render failed: {{}}' < _batch.txt")
        subprocess.run(["docker", "exec", self.name, "bash", "-c", cmd], check=True)
        (work / "_batch.txt").unlink()

    def close(self) -> None:
        subprocess.run(["docker", "kill", self.name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--catalog", default=str(REPO / "outputs/midi_clips/lakh_catalog_small.jsonl"))
    ap.add_argument("--wavetables", default=str(REPO / "data/wavetable_lib.json"))
    ap.add_argument("--qc-off", action="store_true", help="accept every first candidate (generator level stats)")
    a = ap.parse_args()

    out = REPO / "outputs/v5_manifest" / a.name
    if out.exists():
        shutil.rmtree(out)
    for d in ("presets", "notes", "targets"):
        (out / d).mkdir(parents=True)
    lib = load_wavetable_lib(a.wavetables)
    clips = load_clips(a.catalog)
    mix = [ARCHETYPES[i % len(ARCHETYPES)] for i in range(a.n)]
    random.Random(f"v5mix:{a.seed}").shuffle(mix)

    accepted: dict[int, dict] = {}
    tries = {i: 0 for i in range(a.n)}
    current: dict[int, dict] = {}  # sample -> candidate awaiting a render (fresh, or volume-normalized)
    qc_log = open(out / "qc.jsonl", "w")
    renderer = Renderer(a.jobs)
    try:
        build(a, out, mix, lib, clips, renderer, accepted, tries, current, qc_log)
    finally:
        renderer.close()
    qc_log.close()
    with open(out / "manifest.jsonl", "w") as f:
        for i in range(a.n):
            f.write(json.dumps(accepted[i]) + "\n")
    lv = np.array([r["target_rms_dbfs"] for r in accepted.values()])
    print(f"{a.n} samples -> {out}/manifest.jsonl  target RMS dBFS: median {np.median(lv):.1f}, "
          f"min {lv.min():.1f}, max {lv.max():.1f}; candidates generated: {sum(tries.values())}")


def build(a, out, mix, lib, clips, renderer, accepted, tries, current, qc_log) -> None:
    """Generate, render, normalize and QC until every sample is accepted (fills `accepted`)."""
    while len(accepted) < a.n:
        pending = [i for i in range(a.n) if i not in accepted]
        batch = {}
        for i in pending:
            sid = f"s{i:04d}"
            if i not in current:
                if tries[i] >= MAX_TRIES:
                    sys.exit(f"QC failed {MAX_TRIES} candidates in a row for sample {sid}")
                current[i] = {**candidate(a.seed, i, tries[i], mix[i], lib, clips), "try": tries[i], "normalized": False}
                tries[i] += 1
            cand = current[i]
            (out / "presets" / f"{sid}.vital").write_text(json.dumps(cand["preset"], separators=(",", ":")))
            (out / "notes" / f"{sid}.json").write_text(json.dumps({"bpm": cand["bpm"], "notes": cand["notes"]}))
            (out / "targets" / f"{sid}.wav").unlink(missing_ok=True)
            batch[sid] = i
        renderer.render(out, list(batch))
        for sid, i in batch.items():
            cand, wav = current[i], out / "targets" / f"{sid}.wav"
            s = cand["preset"]["settings"]
            if wav.exists():
                audio, sr = sf.read(wav, dtype="float32")
                notes_end = max(n["start_s"] + n["dur_s"] for n in cand["notes"])
                lv = render_level(audio, sr, notes_end)
                ok, why = level_qc(lv)
            else:
                lv, ok, why = {}, False, "render failed"
            ok = ok or (a.qc_off and wav.exists())
            new_vol = None
            if lv and lv["finite"] and lv["peak_dbfs"] > -100 and not cand["normalized"] and not a.qc_off:
                new_vol = normalize_volume(s, lv)
            qc_log.write(json.dumps({"id": sid, "try": cand["try"], "archetype": mix[i], "normalized": cand["normalized"],
                                     "ok": ok and new_vol is None, "reason": why, "volume": s["volume"],
                                     "new_volume": new_vol, **lv}) + "\n")
            qc_log.flush()
            if new_vol is not None:  # outside the loudness band: fix the master volume and render again
                s["volume"] = new_vol
                cand["normalized"] = True
                continue
            del current[i]
            if ok:
                accepted[i] = {"id": sid, "archetype": mix[i],
                               "gt_preset_path": str(out / "presets" / f"{sid}.vital"),
                               "init_preset_path": str(INIT_PRESET),
                               "notes_path": str(out / "notes" / f"{sid}.json"),
                               "target_wav_path": str(wav), "bpm": cand["bpm"],
                               "clip": {k: cand["clip"][k] for k in ("midi_path", "track_idx", "start_s", "program")},
                               "target_rms_dbfs": round(lv["rms_dbfs"], 2)}
        print(f"batch of {len(batch)}: {sum(i in accepted for i in batch.values())} accepted, "
              f"{sum(i in current for i in batch.values())} volume-normalized, "
              f"{a.n - len(accepted) - len(current)} to regenerate", flush=True)

if __name__ == "__main__":
    main()
