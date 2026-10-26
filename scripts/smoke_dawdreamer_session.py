#!/usr/bin/env python3
"""Smoke test for maestro/dawfarm_dd.py + the vital skill scripts (opencode/skills/vital/scripts).

Starts 2 DawDreamerSessions, and in each (concurrently): stages the init preset and a Lakh clip's notes.json (BPM +
notes via maestro.sounddiff.edits.clip_notes), drives set.py / wavetable.py / mod.py / show.py / render.py through
session.bash exactly as an agent would, fetches the wavs and checks:
  - renders are non-silent with the expected duration (default 3 s tail and --tail),
  - tool output carries no audio-derived or audibility verdicts (ears-only),
  - no bleed: a reverb-heavy render followed by a render whose first note starts at 0.3 s is silent for 50 ms,
  - determinism: two renders of the same preset compared in log-mel (records the floor),
  - bash semantics: combined stdout/stderr, exit codes, timeout kill.
Containers are removed at the end (also on failure).

  .venv/bin/python scripts/smoke_dawdreamer_session.py [--clip-index N] [--keep]
"""
from __future__ import annotations

import argparse
import atexit
import json
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from maestro.dawfarm_dd import DawDreamerPool, cleanup_stale  # noqa: E402
from maestro.sounddiff.edits import clip_notes  # noqa: E402

INIT = REPO / "opencode/skills/vital/data/init_preset.json"
CATALOG = REPO / "outputs/midi_clips/lakh_catalog_small.jsonl"
SR = 44100


def log_mel(x: np.ndarray, n_fft=2048, hop=512, n_mels=64) -> np.ndarray:
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if len(mono) < n_fft:
        mono = np.pad(mono, (0, n_fft - len(mono)))
    win = np.hanning(n_fft)
    frames = np.lib.stride_tricks.sliding_window_view(mono, n_fft)[::hop] * win
    spec = np.abs(np.fft.rfft(frames, axis=1)) ** 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)  # noqa: E731
    imel = lambda m: 700 * (10 ** (m / 2595) - 1)  # noqa: E731
    edges = imel(np.linspace(mel(20), mel(SR / 2), n_mels + 2))
    freqs = np.fft.rfftfreq(n_fft, 1 / SR)
    fb = np.zeros((n_mels, len(freqs)))
    for i in range(n_mels):
        lo, c, hi = edges[i:i + 3]
        fb[i] = np.clip(np.minimum((freqs - lo) / (c - lo), (hi - freqs) / (hi - c)), 0, None)
    return 10 * np.log10(spec @ fb.T + 1e-10)


def mel_compare(a: np.ndarray, b: np.ndarray) -> dict:
    n = min(len(a), len(b))
    ma, mb = log_mel(a[:n]), log_mel(b[:n])
    va, vb = ma.ravel() - ma.mean(), mb.ravel() - mb.mean()
    return {"mel_cos": float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12)),
            "mel_mae_db": float(np.abs(ma - mb).mean()), "max_abs_sample_diff": float(np.abs(a[:n] - b[:n]).max())}


def rms(x):
    return float(np.sqrt((x ** 2).mean())) if len(x) else 0.0


def pick_clip(index: int) -> dict:
    clips = [json.loads(line) for line in open(CATALOG)]
    clips = [c for c in clips if 3 <= c["duration_s"] and 6 <= c["n_notes"] <= 40 and 36 <= c["pitch_min"]
             and c["pitch_max"] <= 96 and 60 <= c["bpm"] <= 180]
    c = clips[index % len(clips)]
    return {"bpm": float(c["bpm"]), "notes": clip_notes(c), "src": f"{Path(c['midi_path']).name}@{c['start_s']}"}


def check(cond, msg, failures):
    print(("  ok   " if cond else "  FAIL ") + msg, flush=True)
    if not cond:
        failures.append(msg)


def run_session(sess, clip, tag, failures, report):
    S = f"{sess.skill_mount}/scripts"
    sess.begin_rollout(f"smoke-{tag}")
    sess.put(INIT, "preset.json")
    sess.put_text(json.dumps({"bpm": clip["bpm"], "notes": clip["notes"]}), "notes.json")
    shifted = [{**n, "start_s": round(n["start_s"] + 0.3, 4)} for n in clip["notes"]]
    sess.put_text(json.dumps({"bpm": clip["bpm"], "notes": shifted}), "notes_late.json")
    notes_end = max(n["start_s"] + n["dur_s"] for n in clip["notes"])
    log = []

    def sh(cmd, timeout_ms=None):
        r = sess.bash(cmd, timeout_ms)
        log.append((cmd, r))
        print(f"[{sess.name}] $ {cmd}\n" + "\n".join("    " + line for line in r.output.splitlines()[:14])
              + f"\n    (exit {r.exit_code}, {r.seconds:.2f}s)", flush=True)
        return r

    # agent-style edit chain
    r = sh(f"python3 {S}/wavetable.py list --query 'basic shapes' --limit 3")
    check(r.exit_code == 0 and "Basic Shapes" in r.output, f"{tag}: wavetable list", failures)
    r = sh(f"python3 {S}/wavetable.py set preset.json --osc 1 'Basic Shapes'")
    check(r.exit_code == 0 and "-> 'Basic Shapes'" in r.output, f"{tag}: wavetable set", failures)
    r = sh(f"python3 {S}/set.py preset.json osc_1_wave_frame=128 filter_1_on=on filter_1_cutoff=1200Hz "
           f"filter_1_resonance=35% env_1_release=300ms")
    check(r.exit_code == 0 and "filter_1_cutoff" in r.output, f"{tag}: set.py", failures)
    r = sh(f"python3 {S}/mod.py preset.json add lfo_1 filter_1_cutoff 30%")
    check(r.exit_code == 0 and "slot 1" in r.output, f"{tag}: mod.py add", failures)
    r = sh(f"python3 {S}/set.py preset.json filter_2_cutoff=500Hz")
    check(r.exit_code == 0 and r.output.strip() == "filter_2_cutoff: 60 (262 Hz) -> 71.2131 (500 Hz)",
          f"{tag}: set.py prints only the change (no audibility verdict on a dead key)", failures)
    r = sh(f"python3 {S}/wavetable.py set preset.json --osc 3 wt166")
    check(r.exit_code == 0 and len(r.output.strip().splitlines()) == 1,
          f"{tag}: wavetable set on an off osc / frame-0-silent table prints one line, no warnings", failures)
    r = sh(f"python3 {S}/show.py preset.json filter_2_cutoff")
    check("! filter 2 is off" in r.output, f"{tag}: show.py still explains dead keys on request", failures)
    r = sh(f"python3 {S}/set.py preset.json filter_1_model=11")
    check(r.exit_code == 2 and "out of range" in r.output, f"{tag}: set.py rejects bad enum (exit 2)", failures)
    r = sh(f"python3 {S}/show.py preset.json")
    check(r.exit_code == 0 and "filter 1: Analog" in r.output and "lfo 1:" in r.output, f"{tag}: show.py", failures)

    # renders
    times = []
    for name in ("a", "b"):
        r = sh(f"python3 {S}/render.py preset.json notes.json {name}.wav")
        times.append(r.seconds)
        check(r.exit_code == 0 and r.output.strip() == f"{name}.wav  (notes {notes_end:.2f} s, {clip['bpm']:g} bpm)",
              f"{tag}: render {name} prints only path + note span + bpm", failures)
    r = sh(f"python3 {S}/render.py preset.json notes.json fixed.wav --tail 2")
    times.append(r.seconds)
    # reverb-heavy render, then the late-start render: its first 50 ms must be silent
    r = sh(f"cp preset.json wet.json && python3 {S}/set.py wet.json reverb_on=on reverb_dry_wet=0.8 "
           f"reverb_decay_time=3 && python3 {S}/render.py wet.json notes.json wet.wav && "
           f"python3 {S}/render.py preset.json notes_late.json late.wav")
    check(r.exit_code == 0, f"{tag}: wet + late renders", failures)

    with tempfile.TemporaryDirectory() as td:
        w = {n: sf.read(sess.get(f"{n}.wav", Path(td) / f"{n}.wav"), dtype="float32", always_2d=True)[0]
             for n in ("a", "b", "fixed", "wet", "late")}
    for n, x in w.items():
        check(x.shape[1] == 2 and np.abs(x).max() > 1e-3, f"{tag}: {n}.wav stereo, non-silent (peak "
              f"{np.abs(x).max():.3f})", failures)
    dur = {n: len(x) / SR for n, x in w.items()}
    for n in ("a", "b", "wet"):
        check(abs(dur[n] - (notes_end + 3.0)) <= 1 / SR + 1e-9,
              f"{tag}: {n}.wav default-tail duration {dur[n]:.4f}s == notes_end+3 = {notes_end + 3:.4f}s", failures)
    check(abs(dur["fixed"] - (notes_end + 2.0)) <= 1 / SR + 1e-9,
          f"{tag}: --tail 2 duration {dur['fixed']:.4f}s == notes_end+2 = {notes_end + 2:.4f}s", failures)
    lead = rms(w["late"][:int(0.05 * SR)])
    check(lead < 1e-6, f"{tag}: no bleed after reverb render, first 50 ms RMS {lead:.2e}", failures)
    det = mel_compare(w["a"], w["b"])
    print(f"  [{sess.name}] determinism a vs b: {det}", flush=True)
    report[tag] = {"render_s": [round(t, 2) for t in times], "durations": {k: round(v, 3) for k, v in dur.items()},
                   "notes_end": notes_end, "determinism": det, "first50ms_rms_after_wet": lead}
    return w["a"]


def bash_semantics(sess, failures):
    sess.begin_rollout("smoke-bash")
    r = sess.bash("echo out; echo err >&2; exit 3")
    check(r.exit_code == 3 and r.output == "out\nerr\n", f"bash: combined output + exit code ({r.output!r})", failures)
    r = sess.bash("pwd; echo $DISPLAY $VITAL_SKILL")
    check(r.output.split()[0] == sess.workdir and ":99" in r.output, f"bash: cwd + env ({r.output.split()})",
          failures)
    r = sess.bash("sleep 10; echo never", timeout_ms=1500)
    check(r.timed_out and r.exit_code is None and "exceeding timeout 1500 ms" in r.output and r.seconds < 6,
          f"bash: timeout kill after {r.seconds:.1f}s", failures)
    r = sess.bash("true")
    check(r.output == "(no output)" and r.exit_code == 0, "bash: empty output -> (no output)", failures)
    r = sess.bash("seq 1 5000")
    check(r.truncated and r.output.startswith("...output truncated...") and r.output.endswith("5000\n"),
          "bash: tail truncation at 2000 lines", failures)
    r = sess.bash("ls /home/nate 2>&1 | head -1; touch x.txt && ls")
    check("x.txt" in r.output and sess.host_path("x.txt").exists(), "bash: workdir is the host bind mount", failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip-index", type=int, default=7)
    ap.add_argument("--keep", action="store_true", help="leave containers running (remove with cleanup_stale())")
    a = ap.parse_args()
    clip = pick_clip(a.clip_index)
    print(f"clip {clip['src']}: {len(clip['notes'])} notes, {clip['bpm']} bpm", flush=True)
    failures, report = [], {}
    t0 = time.monotonic()
    pool = DawDreamerPool.start(2, prefix="dd-smoke-")
    print(f"pool start {time.monotonic() - t0:.1f}s", flush=True)
    try:
        bash_semantics(pool.sessions[0], failures)
        with ThreadPoolExecutor(2) as ex:
            futs = [ex.submit(lambda s=s, i=i: run_session(s, clip, f"s{i}", failures, report))
                    for i, s in enumerate(pool.sessions)]
            outs = [f.result() for f in futs]
        cross = mel_compare(outs[0], outs[1])
        print(f"cross-session determinism (s0 a vs s1 a): {cross}")
        report["cross_session"] = cross
    finally:
        if a.keep:
            atexit.unregister(pool.close)
            print("containers kept:", [s.name for s in pool.sessions])
        else:
            pool.close()
            cleanup_stale("dd-smoke-")
    print(json.dumps(report, indent=1))
    print(f"{'FAILED' if failures else 'PASSED'}: {len(failures)} failure(s)", *failures, sep="\n  ")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
