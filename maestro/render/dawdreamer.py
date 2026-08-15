"""DawDreamer-based rendering backend for Vital VST3.

Loads the same Vital.vst3 binary that REAPER uses at inference time,
eliminating the training/inference audio mismatch that vita (C++ engine)
caused — especially for time-based FX like delay and reverb.

JUCE is fundamentally incompatible with Python threading — even a single
render blocks all other threads via the GIL/message-loop interaction.
We solve this by running a dedicated render worker in a spawned subprocess
(created once, reused for all renders via a pair of queues).  A threading
lock serializes callers so request/response pairs never interleave.

Preset application uses ``set_parameter()`` for all ~760 numeric automation
params (filter cutoff, oscillator levels, FX, envelopes, LFOs, etc.).
A complete Vital JSON key → DawDreamer param index mapping handles naming
divergences between the two systems (e.g. ``osc_1_on`` → "Oscillator 1
Switch", ``chorus_cutoff`` → "Chorus Filter Cutoff").

Non-automatable state (wavetables, modulation routing, LFO shapes) cannot
be set via ``set_parameter()``. It IS applied via ``load_state()``, which
works headless when an X display (e.g. Xvfb) is available and the non-fatal
X11 error handler is installed before importing dawdreamer (verified in the
daw-farm containers, renderer bake-off 2026-08-15). The historical claim
that load_state requires a GUI editor window is false under Xvfb.
For wavetable changes, use vita or REAPER+reapy with the chunk API.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import re
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
    """Encode bytes using JUCE's custom base64 (MemoryBlock::toBase64Encoding)."""
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
    """Convert a Vital preset dict into a binary blob for ``load_state()``."""
    json_bytes = json.dumps(preset_dict, separators=(",", ":")).encode("utf-8") + b"\x00"
    juce_trailer = b"\x00" * 16 + b"JUCEPrivateData"
    chunk_data = json_bytes + juce_trailer
    chunk_size = len(chunk_data)

    future = b"\x00" * 128
    body_size = 4 + 4 + 4 + 4 + 4 + 128 + 4 + chunk_size

    icomp = (
        b"VstW"
        + struct.pack(">III", 8, 1, 0)
        + b"CcnK"
        + struct.pack(">i", body_size)
        + b"FBCh"
        + struct.pack(">i", 2)
        + b"Vita"
        + struct.pack(">i", 0x00010600)
        + struct.pack(">i", 0)
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
    return struct.pack("<II", 0x21324356, len(xml_bytes)) + xml_bytes


# ---------------------------------------------------------------------------
# Vital JSON key → DawDreamer parameter index mapping
# ---------------------------------------------------------------------------

def _build_param_mapping(synth) -> tuple[dict[str, int], dict]:
    """Build Vital JSON key → DawDreamer param index mapping.

    Returns (vital_key_to_dd_index, param_ranges) where param_ranges is
    loaded from the bundled JSON file for normalization.
    """
    n = synth.get_plugin_parameter_size()
    dd_norm: dict[str, int] = {}
    for i in range(n):
        name = synth.get_parameter_name(i)
        norm = re.sub(r"[^a-z0-9]", "", name.lower())
        dd_norm[norm] = i

    ranges_path = Path(__file__).resolve().parent.parent / "synth" / "param_ranges.json"
    with open(ranges_path) as f:
        param_ranges = json.load(f)

    mapping: dict[str, int] = {}
    for vital_key in param_ranges:
        norm = vital_key.replace("_", "")

        candidates = [norm]

        # osc → oscillator, env → envelope
        expanded = norm.replace("osc", "oscillator").replace("env", "envelope")
        candidates.append(expanded)

        for base in list(candidates):
            # on → switch
            if base.endswith("on"):
                candidates.append(base[:-2] + "switch")
            # dry_wet → mix
            if "drywet" in base:
                candidates.append(base.replace("drywet", "mix"))
            # chorus/delay cutoff/spread → chorus/delay filter cutoff/spread
            for prefix in ("chorus", "delay"):
                for suffix in ("cutoff", "spread"):
                    if base.startswith(prefix) and base.endswith(suffix) and "filter" not in base:
                        candidates.append(base.replace(prefix + suffix, prefix + "filter" + suffix))
            # compressor ratio/threshold → strip prefix
            if base.startswith("compressor") and ("ratio" in base or "threshold" in base):
                candidates.append(base.replace("compressor", "", 1))
            # lfo delay_time → delay, fade_time → fadein
            if "delaytime" in base:
                candidates.append(base.replace("delaytime", "delay"))
            if "fadetime" in base:
                candidates.append(base.replace("fadetime", "fadein"))
            # lfo keytrack_transpose → transpose, keytrack_tune → tune
            if "keytracktranspose" in base:
                candidates.append(base.replace("keytracktranspose", "transpose"))
            if "keytracktune" in base:
                candidates.append(base.replace("keytracktune", "tune"))
            # random_N → randomlfoN
            m = re.match(r"random(\d)(.*)", base)
            if m and not base.startswith("randomlfo"):
                candidates.append(f"randomlfo{m.group(1)}{m.group(2)}")
            # macro_control_N → macroN
            if "macrocontrol" in base:
                candidates.append(base.replace("macrocontrol", "macro"))
            # reverb shelf cutoff/gain
            if "shelf" in base:
                candidates.append(base.replace("shelfcutoff", "cutoff").replace("shelfgain", "gain"))
            # osc frame_spread → unison frame spread
            if "framespread" in base and "unison" not in base:
                candidates.append(base.replace("framespread", "unisonframespread"))
            # osc random_phase → phase randomization
            if "randomphase" in base:
                candidates.append(base.replace("randomphase", "phaserandomization"))
            # osc unison_stack_type → stack style
            if "unisonstacktype" in base:
                candidates.append(base.replace("unisonstacktype", "stackstyle"))
            # spectral_morph → frequency morph
            if "spectralmorph" in base:
                candidates.append(base.replace("spectralmorph", "frequencymorph"))
            # portamento_on → portamento force
            if base == "portamentoon":
                candidates.append("portamentoforce")

        for c in candidates:
            if c in dd_norm:
                mapping[vital_key] = dd_norm[c]
                break

    return mapping, param_ranges


def _apply_preset_params(synth, preset_dict: dict, mapping: dict[str, int],
                         param_ranges: dict) -> int:
    """Set all numeric preset params via ``set_parameter()``. Returns count set."""
    settings = preset_dict.get("settings", preset_dict)
    count = 0
    for vital_key, dd_idx in mapping.items():
        val = settings.get(vital_key)
        if val is None or not isinstance(val, (int, float)):
            continue
        r = param_ranges.get(vital_key)
        if not r:
            continue
        lo, hi = r["min"], r["max"]
        if hi <= lo:
            continue
        norm = max(0.0, min(1.0, (float(val) - lo) / (hi - lo)))
        synth.set_parameter(dd_idx, norm)
        count += 1
    return count


# ---------------------------------------------------------------------------
# Render worker
# ---------------------------------------------------------------------------

def _install_x11_error_handler():
    """Install a non-fatal X11 error handler so load_state's editor-window
    hack doesn't crash when running under Xvfb."""
    try:
        import ctypes, ctypes.util
        libx11_path = ctypes.util.find_library("X11")
        if not libx11_path:
            return
        libx11 = ctypes.CDLL(libx11_path)
        libx11.XInitThreads()
        HANDLER = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
        _ignore = HANDLER(lambda d, e: 0)
        libx11.XSetErrorHandler(_ignore)
        _install_x11_error_handler._prevent_gc = _ignore
    except Exception:
        pass


def _render_worker(req_q, resp_q, vst3_path, sample_rate, block_size):
    """Long-lived subprocess that owns the single DawDreamer engine."""
    _install_x11_error_handler()
    import dawdreamer as daw
    engine = daw.RenderEngine(sample_rate, block_size)
    synth = engine.make_plugin_processor("vital", vst3_path)
    mapping, param_ranges = _build_param_mapping(synth)

    while True:
        msg = req_q.get()
        if msg is None:
            break
        preset_json, notes, tail_s = msg
        try:
            preset_dict = json.loads(preset_json)
            # Full-state application via load_state(): the ONLY path that
            # commits non-automatable state (wavetables, mod routing, LFO
            # shapes). Verified working headless under the daw-farm Xvfb
            # with the X11 error handler installed above (renderer bake-off,
            # 2026-08-15: 9,422 Hz probe centroid spread vs 2 Hz without).
            # set_parameter-only application silently renders the default
            # wavetable and MUST NOT be the primary path.
            try:
                state_blob = _build_dawdreamer_state(preset_dict)
                state_path = os.path.join(
                    tempfile.gettempdir(), f"dd_state_{os.getpid()}.vstate")
                with open(state_path, "wb") as sf_:
                    sf_.write(state_blob)
                synth.load_state(state_path)
            except Exception as state_exc:
                # Param-only fallback: numeric params apply, wavetables DO NOT.
                # Loud on purpose — silent use of this path produced an entire
                # corpus of identical probe renders (2026-08-15 postmortem).
                print(
                    f"[dawdreamer] WARNING: load_state failed ({state_exc}); "
                    "falling back to set_parameter-only — WAVETABLES/MOD/LFO "
                    "STATE WILL NOT APPLY. Renders may be timbrally invalid.",
                    file=sys.stderr, flush=True)
                _apply_preset_params(synth, preset_dict, mapping, param_ranges)

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
