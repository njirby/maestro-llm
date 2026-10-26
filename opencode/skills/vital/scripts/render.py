#!/usr/bin/env python3
"""Render preset.json + notes.json to a wav through the Vital VST3 in headless DawDreamer.

  render.py preset.json notes.json out.wav [--bpm N] [--tail S]

Loads the FULL preset state (wavetables, LFO shapes, mod matrix), sets the host tempo (Vital ignores the preset's
beats_per_minute inside a host: tempo comes from notes.json "bpm", or --bpm, else 120), plays the notes after a
silent bar-aligned pre-roll of >= 4 s that is cut from the output, and writes 32-bit float stereo at 44.1 kHz.

Tail: the render always runs a fixed --tail seconds past the last note-off (default 3.0 s).

Prints one line: <out.wav>  (notes <t> s, <bpm> bpm)   - note span from notes.json, nothing derived from the audio.
"""
import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitallib import UsageError, build_state, install_x11_error_handler, load_notes, load_preset, run_main  # noqa

VITAL = os.environ.get("VITAL_VST3", "/usr/lib/vst3/Vital.vst3")
SR, BLOCK = 44100, 512
PREROLL_MIN_S = 4.0  # Vital keeps FX buffers/voices across state loads; the pre-roll flushes them
DEFAULT_TAIL_S = 3.0


@contextmanager
def quiet_fds():
    """Silence native stdout/stderr chatter from JUCE/Vital (e.g. 'attempt to map invalid URI')."""
    sys.stdout.flush(), sys.stderr.flush()
    saved = os.dup(1), os.dup(2)
    null = os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 1), os.dup2(null, 2)
    try:
        yield
    finally:
        os.dup2(saved[0], 1), os.dup2(saved[1], 2)
        for fd in (*saved, null):
            os.close(fd)


def ensure_display():
    """Vital needs an X display (Xvfb). Use $DISPLAY if set, else start a private Xvfb for this process."""
    if os.environ.get("DISPLAY"):
        return None
    for n in range(100 + os.getpid() % 400, 600):
        if not Path(f"/tmp/.X11-unix/X{n}").exists() and not Path(f"/tmp/.X{n}-lock").exists():
            break
    try:
        proc = subprocess.Popen(["Xvfb", f":{n}", "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        raise UsageError("DISPLAY is not set and Xvfb is not installed (Vital needs an X display)")
    for _ in range(100):
        if Path(f"/tmp/.X11-unix/X{n}").exists():
            break
        time.sleep(0.05)
    os.environ["DISPLAY"] = f":{n}"
    return proc


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset")
    ap.add_argument("notes")
    ap.add_argument("out")
    ap.add_argument("--bpm", type=float, help="host tempo (overrides notes.json \"bpm\"; default 120)")
    ap.add_argument("--tail", type=float, default=DEFAULT_TAIL_S,
                    help=f"seconds rendered after the last note-off (default {DEFAULT_TAIL_S})")
    a = ap.parse_args()

    preset = load_preset(a.preset)
    notes, nbpm = load_notes(a.notes)
    bpm = a.bpm if a.bpm is not None else nbpm if nbpm is not None else 120.0
    if not 20 <= bpm <= 300:
        raise UsageError(f"bpm {bpm:g} out of range 20..300")
    if not 0 <= a.tail <= 30:
        raise UsageError("--tail must be within 0..30 s")
    if not Path(VITAL).exists():
        raise UsageError(f"Vital VST3 not found at {VITAL} (set VITAL_VST3)")
    out = Path(a.out)
    if out.suffix.lower() != ".wav":
        raise UsageError(f"{out}: output must be a .wav path")
    out.parent.mkdir(parents=True, exist_ok=True)

    notes_end = max(n["start_s"] + n["dur_s"] for n in notes)
    bar_s = 240.0 / bpm  # 4/4: pre-roll is whole bars so transport-synced LFOs/delays line up with the notes
    preroll = math.ceil(PREROLL_MIN_S / bar_s) * bar_s
    tail = a.tail

    xvfb = ensure_display()
    try:
        install_x11_error_handler()
        import numpy as np
        import soundfile as sf
        with quiet_fds():
            import dawdreamer as daw
            eng = daw.RenderEngine(SR, BLOCK)
            synth = eng.make_plugin_processor("vital", VITAL)
            eng.load_graph([(synth, [])])
            with tempfile.NamedTemporaryFile(suffix=".vitalstate", delete=False) as f:
                f.write(build_state(preset))
            try:
                synth.load_state(f.name)
            finally:
                os.unlink(f.name)
            eng.set_bpm(float(bpm))
            synth.clear_midi()
            for n in notes:
                synth.add_midi_note(n["pitch"], n["velocity"], preroll + n["start_s"], n["dur_s"])
            eng.render(preroll + notes_end + tail)
            audio = np.asarray(synth.get_audio(), dtype=np.float32).T[int(round(preroll * SR)):]
    finally:
        if xvfb:
            xvfb.terminate()

    audio = audio[:int(round(notes_end * SR)) + int(round(tail * SR))]
    sf.write(str(out), audio, SR, subtype="FLOAT")
    print(f"{out}  (notes {notes_end:.2f} s, {bpm:g} bpm)")


if __name__ == "__main__":
    run_main(main)
