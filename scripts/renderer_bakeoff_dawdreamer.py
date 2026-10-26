#!/usr/bin/env python3
"""Bake-off candidate (b): DawDreamer with FULL preset state via load_state().

Runs inside a daw-farm container (dawdreamer + Vital.vst3 + Xvfb on :99).
Self-contained on purpose — the container has no maestro checkout.

Key finding this file encodes
-----------------------------
maestro/render/dawdreamer.py already builds a byte-correct state blob
(``_build_dawdreamer_state``; verified sha1-identical to ``build_state``
below), but ``_render_worker`` never calls ``load_state()`` — it only calls
``_apply_preset_params()``. That is why every probe renders on Vital's
default wavetable. The module docstring's claim that "load_state doesn't
apply on headless Linux (Vital requires a GUI editor window)" does NOT hold
in the daw-farm container: with Xvfb on :99 and a non-fatal X11 error
handler installed *before* importing dawdreamer, load_state commits full
state — wavetables included — with no editor window. Without the error
handler the process aborts with "X Error of failed request: BadAtom".

Usage (inside the container):
    DISPLAY=:99 python3 renderer_bakeoff_dawdreamer.py <out_dir> [--n N]

Presets are read from /work/bakeoff/presets (8 probe JSONs + melody.json),
as staged by the bake-off driver.
"""
import json
import os
import struct
import sys
import time

PRESET_DIR = os.environ.get("BAKEOFF_PRESETS", "/work/bakeoff/presets")
VITAL = os.environ.get("VITAL_VST3", "/usr/lib/vst3/Vital.vst3")
SR = 44100
BLOCK = 512
TAIL_S = 1.0

_JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def juce_b64_encode(data: bytes) -> str:
    """JUCE MemoryBlock::toBase64Encoding — 'size.chars', 6 bits LSB-first."""
    size = len(data)
    out = []
    for i in range(((size * 8) + 5) // 6):
        bit = i * 6
        byte, off = bit >> 3, bit & 7
        if byte + 1 < size:
            word = data[byte] | (data[byte + 1] << 8)
        else:
            word = data[byte] if byte < size else 0
        out.append(_JUCE_B64[(word >> off) & 0x3F])
    return f"{size}.{''.join(out)}"


def build_state(preset: dict) -> bytes:
    """Vital VST3 state file, byte-identical to what Vital's save_state emits
    (VC2! magic, XML wrapper, JUCE-b64 IComponent, preset JSON at offset 176)."""
    json_bytes = json.dumps(preset, separators=(",", ":")).encode() + b"\x00"
    chunk = json_bytes + b"\x00" * 16 + b"JUCEPrivateData"
    body = 4 + 4 + 4 + 4 + 4 + 128 + 4 + len(chunk)
    icomp = (
        b"VstW" + struct.pack(">III", 8, 1, 0)
        + b"CcnK" + struct.pack(">i", body)
        + b"FBCh" + struct.pack(">i", 2)
        + b"Vita" + struct.pack(">i", 0x00010600)
        + struct.pack(">i", 0)
        + b"\x00" * 128
        + struct.pack(">i", len(chunk))
        + chunk
    )
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n'
           f"<VST3PluginState><IComponent>{juce_b64_encode(icomp)}</IComponent>"
           "</VST3PluginState>").encode() + b"\x00"
    return struct.pack("<II", 0x21324356, len(xml)) + xml


def install_x11_error_handler() -> None:
    """MUST run before importing dawdreamer. Vital's state load touches the
    GUI layer; an unhandled X error aborts the whole process under Xvfb."""
    import ctypes
    import ctypes.util
    path = ctypes.util.find_library("X11")
    if not path:
        return
    libx11 = ctypes.CDLL(path)
    libx11.XInitThreads()
    HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    ignore = HANDLER(lambda d, e: 0)
    libx11.XSetErrorHandler(ignore)
    IOHANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
    io_ignore = IOHANDLER(lambda d: 0)
    libx11.XSetIOErrorHandler(io_ignore)
    install_x11_error_handler._keep = (ignore, io_ignore)


class DawDreamerProbeRenderer:
    """One long-lived engine + Vital instance, reused across renders.

    Safe to run several of these as separate *processes* in one container
    (near-linear scaling to at least 8); not safe across threads — JUCE and
    the Python GIL do not co-operate.
    """

    def __init__(self, vst3_path: str = VITAL, sample_rate: int = SR,
                 block_size: int = BLOCK):
        install_x11_error_handler()
        import dawdreamer as daw
        self.sample_rate = sample_rate
        self.engine = daw.RenderEngine(sample_rate, block_size)
        self.synth = self.engine.make_plugin_processor("v", vst3_path)
        self.engine.load_graph([(self.synth, [])])
        self._state_path = f"/tmp/dd_state_{os.getpid()}"

    def render(self, preset: dict, notes: list[dict], tail_s: float = TAIL_S):
        """Apply full preset state (wavetables included), play notes, return (2, N)."""
        with open(self._state_path, "wb") as f:
            f.write(build_state(preset))
        self.synth.load_state(self._state_path)

        self.synth.clear_midi()
        for n in notes:
            self.synth.add_midi_note(int(n["pitch"]), int(n["velocity"]),
                                     float(n["start_s"]), float(n["dur_s"]))
        duration = max(n["start_s"] + n["dur_s"] for n in notes) + tail_s
        self.engine.render(duration)
        return self.synth.get_audio()


def main() -> int:
    out_dir = sys.argv[1]
    repeat_to = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 0
    os.makedirs(out_dir, exist_ok=True)

    melody = json.load(open(os.path.join(PRESET_DIR, "melody.json")))
    preset_files = sorted(f for f in os.listdir(PRESET_DIR)
                          if f.endswith(".json") and f != "melody.json")

    t0 = time.perf_counter()
    renderer = DawDreamerProbeRenderer()
    setup_s = time.perf_counter() - t0
    print(json.dumps({"event": "setup", "seconds": round(setup_s, 3)}), flush=True)

    import numpy as np
    import soundfile as sf

    timings = []
    for i in range(repeat_to or len(preset_files)):
        pf = preset_files[i % len(preset_files)]
        label = os.path.splitext(pf)[0]
        preset = json.load(open(os.path.join(PRESET_DIR, pf)))
        t_a = time.perf_counter()
        audio = renderer.render(preset, melody)
        elapsed = time.perf_counter() - t_a
        name = f"t{i:02d}_{label}.wav" if repeat_to else f"{label}.wav"
        sf.write(os.path.join(out_dir, name), np.asarray(audio).T, SR)
        timings.append(elapsed)
        print(json.dumps({"event": "render", "i": i, "label": label,
                          "seconds": round(elapsed, 4)}), flush=True)

    print(json.dumps({"event": "done", "setup_s": round(setup_s, 3),
                      "n": len(timings),
                      "mean_s": round(sum(timings) / len(timings), 4)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
