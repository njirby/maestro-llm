#!/usr/bin/env python3
"""Render-parity check: the skill renderer (opencode/skills/vital/scripts/render.py, one fresh Vital per render, bar-aligned
pre-roll, 3 s tail) vs the sound-delta renderer (maestro/sounddiff/render_pairs.py, one long-lived Vital per shard, flat
4 s pre-roll, 2 s tail) on the same presets + notes.

Each preset is rendered three times: pairs, skill, skill2. skill vs skill2 is the render-noise floor (random phase,
free-running LFOs); pairs vs skill is compared against it. Both are cut to the common length (notes + 2 s).

  .venv/bin/python scripts/check_render_parity.py [--n-probe 30] [--n-ref 30] [--out outputs/render_parity/run1]

Writes <out>/{renders.jsonl, presets/, notes/, wav/, report.json} and prints one line per flagged preset. Exit 1 if
any preset is silent in one path and not the other.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
SKILL = REPO / "opencode/skills/vital"
IMAGE = "daw-farm/reaper:latest"
SR = 44100
SILENT_DBFS = -100.0  # digital silence renders at -240; anything below this is treated as silent
COMMON_TAIL_S = 2.0
MUST = {"f000_t"}  # the original silent-render report


def rms_dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x, dtype=np.float64))) + 1e-12))


def logmel(x: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(y=x.mean(axis=1).astype(np.float32), sr=SR, n_fft=2048, hop_length=512,
                                       n_mels=64)
    return librosa.power_to_db(m, ref=1.0, amin=1e-10, top_db=None).clip(min=-100.0)


def mel_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |log-mel difference| in dB over frames where either render is above -80 dB (avoids silence-vs-silence)."""
    ma, mb = logmel(a), logmel(b)
    n = min(ma.shape[1], mb.shape[1])
    ma, mb = ma[:, :n], mb[:, :n]
    mask = np.maximum(ma, mb) > -80.0
    return float(np.abs(ma - mb)[mask].mean()) if mask.any() else 0.0


def pick_rows(n_probe: int, n_ref: int) -> list[dict]:
    rows = []
    for src, n, keep in ((REPO / "outputs/sounddiff/probe1", n_probe, lambda r: r["name"].endswith("_t")),
                         (REPO / "outputs/sounddiff/refset", n_ref, lambda r: True)):
        cand = [r for r in map(json.loads, open(src / "renders.jsonl")) if keep(r)]
        got = [cand[i * len(cand) // n] for i in range(min(n, len(cand)))]  # spread over the set
        got += [r for r in cand if r["name"] in MUST and r not in got]
        for r in got:
            rows.append({**r, "src": src.name, "preset_abs": str(src / r["preset"])})
    return rows


def prepare(out: Path, rows: list[dict]) -> None:
    for d in ("presets", "notes", "wav"):
        (out / d).mkdir(parents=True, exist_ok=True)
    with open(out / "renders.jsonl", "w") as f:
        for r in rows:
            name = f"{r['src']}_{r['name']}"
            shutil.copyfile(r["preset_abs"], out / "presets" / f"{name}.vital")
            notes = [n for n in r["notes"] if float(n["dur_s"]) > 0]  # both paths drop zero-length notes
            (out / "notes" / f"{name}.json").write_text(json.dumps({"bpm": r["bpm"], "notes": notes}))
            f.write(json.dumps({"name": f"{name}__pairs", "preset": f"presets/{name}.vital", "notes": notes,
                                "bpm": r["bpm"]}) + "\n")


def render(out: Path, jobs: int) -> None:
    rel = out.relative_to(REPO)
    skill_cmd = (f"ls {rel}/notes | sed 's/.json$//' | xargs -P {jobs} -I{{}} sh -c "
                 f"'for k in skill skill2; do [ -f {rel}/wav/{{}}__$k.wav ] || DISPLAY=:99 python3 "
                 f"{SKILL}/scripts/render.py {rel}/presets/{{}}.vital {rel}/notes/{{}}.json {rel}/wav/{{}}__$k.wav "
                 f">/dev/null; done'")
    pairs_cmd = (f"pids=; for s in $(seq 0 {jobs - 1}); do DISPLAY=:99 python3 maestro/sounddiff/render_pairs.py "
                 f"{rel} $s {jobs} >/dev/null & pids=\"$pids $!\"; done; wait $pids")
    cmd = f"Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 & sleep 2; {pairs_cmd}; {skill_cmd}"
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--name", f"wsc-parity-{os.getpid()}", "--user", f"{os.getuid()}:{os.getgid()}",
                    "-v", "/home/nate:/home/nate", "-w", str(REPO), "-e", f"VITAL_VST3=/usr/lib/vst3/Vital.vst3",
                    "--entrypoint", "bash", IMAGE, "-c", cmd], check=True)


def analyze(out: Path) -> dict:
    res = []
    for r in map(json.loads, open(out / "renders.jsonl")):
        name = r["name"].removesuffix("__pairs")
        notes_end = max(n["start_s"] + n["dur_s"] for n in r["notes"])
        n = int(round((notes_end + COMMON_TAIL_S) * SR))
        w = {k: sf.read(out / "wav" / f"{name}__{k}.wav", dtype="float32")[0][:n] for k in ("pairs", "skill", "skill2")}
        lv = {k: rms_dbfs(v) for k, v in w.items()}
        floor = mel_dist(w["skill"], w["skill2"])
        d = mel_dist(w["pairs"], w["skill"])
        silent = {k: v < SILENT_DBFS for k, v in lv.items()}
        res.append({"name": name, **{f"rms_{k}": round(v, 2) for k, v in lv.items()},
                    "mel_pairs_skill": round(d, 2), "mel_floor": round(floor, 2),
                    "silent_mismatch": silent["pairs"] != silent["skill"] or silent["skill"] != silent["skill2"],
                    "both_silent": silent["pairs"] and silent["skill"]})
    for x in res:
        x["diverges"] = (not x["both_silent"] and not x["silent_mismatch"]
                         and (x["mel_pairs_skill"] > max(3.0, 2.0 * x["mel_floor"])
                              or abs(x["rms_pairs"] - x["rms_skill"]) > max(3.0, 2.0 * abs(x["rms_skill"] - x["rms_skill2"]))))
    md = np.array([x["mel_pairs_skill"] for x in res])
    fl = np.array([x["mel_floor"] for x in res])
    summary = {"n": len(res), "silent_mismatch": sum(x["silent_mismatch"] for x in res),
               "both_silent": sum(x["both_silent"] for x in res), "diverges": sum(x["diverges"] for x in res),
               "mel_pairs_skill_median": round(float(np.median(md)), 2), "mel_floor_median": round(float(np.median(fl)), 2),
               "rms_delta_median": round(float(np.median([abs(x["rms_pairs"] - x["rms_skill"]) for x in res])), 2)}
    report = {"summary": summary, "presets": res}
    (out / "report.json").write_text(json.dumps(report, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-probe", type=int, default=30)
    ap.add_argument("--n-ref", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--out", default="outputs/render_parity/run1")
    ap.add_argument("--analyze-only", action="store_true")
    a = ap.parse_args()
    out = (REPO / a.out).resolve()
    if not a.analyze_only:
        prepare(out, pick_rows(a.n_probe, a.n_ref))
        render(out, a.jobs)
    rep = analyze(out)
    for x in rep["presets"]:
        if x["silent_mismatch"] or x["diverges"] or x["both_silent"]:
            tag = "SILENT-MISMATCH" if x["silent_mismatch"] else "both-silent" if x["both_silent"] else "diverges"
            print(f"{tag:16s} {x['name']:24s} rms pairs {x['rms_pairs']:7.1f} skill {x['rms_skill']:7.1f} "
                  f"skill2 {x['rms_skill2']:7.1f}  mel {x['mel_pairs_skill']:5.2f} (floor {x['mel_floor']:5.2f})")
    print(json.dumps(rep["summary"]))
    sys.exit(1 if rep["summary"]["silent_mismatch"] else 0)


if __name__ == "__main__":
    main()
