"""DawDreamer-based rendering backend for Vital VST3.

Loads the same Vital.vst3 binary that REAPER uses at inference time,
eliminating the training/inference audio mismatch that vita (C++ engine)
caused — especially for time-based FX like delay and reverb.

JUCE is fundamentally incompatible with Python threading — even a single
render blocks all other threads via the GIL/message-loop interaction.
We solve this by running a dedicated render worker in a spawned subprocess
(created once, reused for all renders via a pair of queues).  A threading
lock serializes callers so request/response pairs never interleave.

Preset loading uses a constructed binary state file that wraps the .vital
JSON in the 5-layer format DawDreamer/JUCE expects:
  VC2! header → XML envelope → JUCE base64 → VstW/FBCh wrapper → JSON
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import struct
import tempfile
import threading
from pathlib import Path

import numpy as np
import soundfile as sf

SAMPLE_RATE = 44100
BLOCK_SIZE = 512
VITAL_VST3_PATH = os.environ.get(
    "VITAL_VST3", os.path.expanduser("~/.vst3/Vital.vst3")
)

NoteTuple = tuple[int, int, float, float]

_lock = threading.Lock()
_ctx = mp.get_context("spawn")
_req_q: mp.Queue | None = None
_resp_q: mp.Queue | None = None
_worker_proc: mp.Process | None = None


# ---------------------------------------------------------------------------
# JUCE-compatible base64 encoding
# ---------------------------------------------------------------------------

_JUCE_B64_TABLE = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"


def _juce_base64_encode(data: bytes) -> str:
    """Encode bytes using JUCE's custom base64 (MemoryBlock::toBase64Encoding).

    JUCE uses a bit-level encoding: for each group of 6 bits (read
    little-endian from the byte stream), emit the corresponding character
    from the JUCE alphabet.  The encoded string is prefixed with the
    decimal byte count and a dot separator.
    """
    size = len(data)
    num_chars = ((size * 8) + 5) // 6
    chars = []
    for i in range(num_chars):
        bit_pos = i * 6
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        if byte_idx + 1 < size:
            word = data[byte_idx] | (data[byte_idx + 1] << 8)
        else:
            word = data[byte_idx] if byte_idx < size else 0
        val = (word >> bit_off) & 0x3F
        chars.append(_JUCE_B64_TABLE[val])
    return f"{size}.{''.join(chars)}"


# ---------------------------------------------------------------------------
# Build a DawDreamer-compatible state file from a .vital preset dict
# ---------------------------------------------------------------------------

def _build_dawdreamer_state(preset_dict: dict) -> bytes:
    """Convert a Vital preset dict into a binary blob that DawDreamer's
    ``load_state()`` accepts — full wavetables, modulation, and all.

    The format mirrors what JUCE's VST3 hosting code writes via
    ``copyXmlToBinary`` and what Vital's ``setStateInformation`` parses.
    """
    json_bytes = json.dumps(preset_dict, separators=(",", ":")).encode("utf-8") + b"\x00"
    juce_trailer = b"\x00" * 16 + b"JUCEPrivateData"
    chunk_data = json_bytes + juce_trailer
    chunk_size = len(chunk_data)

    # VstW + FBCh header (same as REAPER chunk, but without the REAPER-specific
    # 24-byte outer wrapper — starts directly at VstW).
    future = b"\x00" * 128
    # body_size covers: FBCh(4) + version(4) + Vita(4) + fxVersion(4) +
    #                   numPrograms(4) + future(128) + chunkSize(4) + chunk_data
    body_size = 4 + 4 + 4 + 4 + 4 + 128 + 4 + chunk_size

    icomp = (
        b"VstW"
        + struct.pack(">III", 8, 1, 0)
        + b"CcnK"
        + struct.pack(">i", body_size)
        + b"FBCh"
        + struct.pack(">i", 2)
        + b"Vita"
        + struct.pack(">i", 0x00010600)  # Vital 1.6.0
        + struct.pack(">i", 0)  # numPrograms
        + future
        + struct.pack(">i", chunk_size)
        + chunk_data
    )

    b64 = _juce_base64_encode(icomp)
    xml_str = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<VST3PluginState><IComponent>{b64}</IComponent></VST3PluginState>"
    )
    xml_bytes = xml_str.encode("utf-8") + b"\x00"
    # VC2! magic + string-length + xml
    return struct.pack("<II", 0x21324356, len(xml_bytes)) + xml_bytes


# ---------------------------------------------------------------------------
# Render worker
# ---------------------------------------------------------------------------

def _render_worker(req_q, resp_q, vst3_path, sample_rate, block_size):
    """Long-lived subprocess that owns the single DawDreamer engine."""
    import dawdreamer as daw
    engine = daw.RenderEngine(sample_rate, block_size)
    synth = engine.make_plugin_processor("vital", vst3_path)

    while True:
        msg = req_q.get()
        if msg is None:
            break
        preset_json, notes, tail_s = msg
        try:
            preset_dict = json.loads(preset_json)
            state_blob = _build_dawdreamer_state(preset_dict)

            with tempfile.NamedTemporaryFile(suffix=".bin", delete=False) as f:
                f.write(state_blob)
                tmp = f.name
            try:
                synth.load_state(tmp)
            finally:
                os.unlink(tmp)

            synth.clear_midi()
            for pitch, vel, start, dur in notes:
                synth.add_midi_note(int(pitch), int(vel), float(start), float(dur))

            duration = max((s + d for _, _, s, d in notes), default=1.0) + tail_s
            engine.load_graph([(synth, [])])
            engine.render(duration)
            audio = synth.get_audio()
            resp_q.put(("ok", audio))
        except Exception as exc:
            resp_q.put(("err", str(exc)))


def _ensure_worker():
    """Start the render subprocess if not already running.  Caller must hold _lock."""
    global _req_q, _resp_q, _worker_proc
    if _worker_proc is not None and _worker_proc.is_alive():
        return
    _req_q = _ctx.Queue()
    _resp_q = _ctx.Queue()
    _worker_proc = _ctx.Process(
        target=_render_worker,
        args=(_req_q, _resp_q, VITAL_VST3_PATH, SAMPLE_RATE, BLOCK_SIZE),
        daemon=True,
    )
    _worker_proc.start()


def render_preset_audio(
    preset_dict: dict,
    notes: list[NoteTuple],
    out_path: str | Path | None = None,
    tail_s: float = 1.0,
    sample_rate: int = SAMPLE_RATE,
) -> np.ndarray:
    """Render a Vital preset through DawDreamer. Returns (2, N) float32 array.

    If *out_path* is provided, writes a trimmed WAV file as well.
    """
    preset_json = json.dumps(preset_dict, separators=(",", ":"))

    with _lock:
        _ensure_worker()
        _req_q.put((preset_json, notes, tail_s))
        status, payload = _resp_q.get()

    if status == "err":
        raise RuntimeError(f"DawDreamer render failed: {payload}")
    audio = payload

    if out_path is not None:
        from maestro.render.vital import trim_silence
        trimmed = trim_silence(audio, sample_rate, min_duration_s=0.5)
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), trimmed.T, sample_rate)

    return audio


def notes_from_pretty_midi(pm_notes: list) -> list[NoteTuple]:
    """Convert pretty_midi.Note objects to DawDreamer (pitch, vel, start, dur) tuples."""
    return [
        (int(n.pitch), int(n.velocity), float(n.start), float(n.end - n.start))
        for n in pm_notes
    ]


def notes_from_dicts(note_dicts: list[dict]) -> list[NoteTuple]:
    """Convert {"pitch", "velocity", "start_s", "dur_s"} dicts to DawDreamer tuples."""
    return [
        (int(n["pitch"]), int(n["velocity"]), float(n["start_s"]), float(n["dur_s"]))
        for n in note_dicts
    ]


def make_probe_notes(
    archetype: str = "bass",
    clip_duration_s: float = 10.0,
) -> list[NoteTuple]:
    """4 major triads (C2–C5), matching vital.make_probe_notes but returning DawDreamer tuples."""
    roots = [36, 48, 60, 72]
    intervals = [0, 4, 7]
    note_dur = clip_duration_s / len(roots)
    notes: list[NoteTuple] = []
    for i, root in enumerate(roots):
        start = i * note_dur
        for interval in intervals:
            pitch = max(0, min(127, root + interval))
            notes.append((pitch, 85, start, note_dur))
    return notes
