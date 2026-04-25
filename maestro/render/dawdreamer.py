"""DawDreamer-based rendering backend for Vital VST3.

Loads the same Vital.vst3 binary that REAPER uses at inference time,
eliminating the training/inference audio mismatch that vita (C++ engine)
caused — especially for time-based FX like delay and reverb.

JUCE is fundamentally incompatible with Python threading — even a single
render blocks all other threads via the GIL/message-loop interaction.
We solve this by running a dedicated render worker in a spawned subprocess
(created once, reused for all renders via a pair of queues).  A threading
lock serializes callers so request/response pairs never interleave.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
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
            with tempfile.NamedTemporaryFile(suffix=".vital", mode="w", delete=False) as f:
                f.write(preset_json)
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
