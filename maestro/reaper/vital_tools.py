"""
vital_tools.py — Python ReaScript standard library for controlling the Vital
synthesizer inside a live REAPER session.

.. deprecated::
    VitalController is no longer used in generated SFT training data.
    Training snippets now use generic REAPER APIs (TrackFX_SetParam,
    TrackFX_GetNamedConfigParm) via raw reapy calls so the model learns
    patterns that transfer to any plugin.  This file is retained for
    build-time debugging and the ``_DISPLAY_NAME_ALIASES`` lookup table
    used at build time to convert Vital JSON keys to exact REAPER display
    names.

This file is designed to run via ``reaper -nonewinst script.py``.  All REAPER
API calls use the RPR_ prefix.  No external packages may be imported; only the
Python standard library is available.

Mockability
-----------
Pass an ``_rpr`` dict to ``VitalController.__init__`` to inject fake RPR_
functions for unit testing without a real REAPER process.  Every API call
goes through ``self._call()``, which falls back to ``globals()[fn_name]`` when
the function is not mocked.

VST chunk encoding
------------------
REAPER stores the VST plugin state as a base64-encoded blob (the "chunk").
Vital embeds its full preset as a UTF-8 JSON string inside that blob.  The
blob typically has a small binary header before the ``{`` that opens the JSON.

get_preset() strategy:
  1. Fetch raw chunk string via RPR_TrackFX_GetNamedConfigParm.
  2. Base64-decode it.
  3. Scan forward for the first ``{`` byte to skip any binary header.
  4. Parse JSON from that point.
  5. Cache (raw_prefix, raw_suffix) so set_preset() can reconstruct the chunk.

set_preset() strategy:
  1. Re-encode updated JSON as UTF-8.
  2. Prefix the original binary header; append the original binary suffix.
  3. Base64-encode and write back via RPR_TrackFX_SetNamedConfigParm.
"""

try:
    from reaper_python import *  # injected by REAPER at runtime
except ImportError:
    pass  # running outside REAPER (unit tests with _rpr mocks)

import base64
import difflib
import gc
import json
import os
import re
import time


# ---------------------------------------------------------------------------
# Vital parameter ranges (bundled alongside vital_tools.py)
# ---------------------------------------------------------------------------

def _load_param_ranges() -> dict:
    """Load param_ranges.json from the package data directory, if available."""
    try:
        _dir = os.path.dirname(os.path.abspath(__file__))
        # Walk up to maestro/synth/param_ranges.json
        candidate = os.path.join(_dir, "..", "synth", "param_ranges.json")
        candidate = os.path.normpath(candidate)
        if os.path.exists(candidate):
            with open(candidate) as _f:
                return json.load(_f)
    except Exception:
        pass
    return {}


_PARAM_RANGES: dict = _load_param_ranges()

# Abbreviation expansion used when mapping Vital JSON keys → REAPER display names
_VITAL_ABBREVS = {
    "osc":   "Oscillator",
    "env":   "Envelope",
    "lfo":   "LFO",
    "eq":    "EQ",
    "fx":    "FX",
}


def _json_key_to_display(key: str) -> str:
    """Heuristically convert a Vital JSON parameter key to its REAPER display name.

    Examples::

        osc_1_level    → "Oscillator 1 Level"
        filter_1_cutoff → "Filter 1 Cutoff"
        lfo_2_frequency → "LFO 2 Frequency"
    """
    parts = key.split("_")
    return " ".join(_VITAL_ABBREVS.get(p.lower(), p.capitalize()) for p in parts)


# Explicit aliases for keys whose REAPER display names differ materially from
# Vital JSON naming.
_DISPLAY_NAME_ALIASES = {
    "osc_1_random_phase": "Oscillator 1 Phase Randomization",
    "osc_2_random_phase": "Oscillator 2 Phase Randomization",
    "osc_3_random_phase": "Oscillator 3 Phase Randomization",
    "osc_1_spectral_morph_amount": "Oscillator 1 Frequency Morph Amount",
    "osc_2_spectral_morph_amount": "Oscillator 2 Frequency Morph Amount",
    "osc_3_spectral_morph_amount": "Oscillator 3 Frequency Morph Amount",
    "osc_1_spectral_morph_type": "Oscillator 1 Frequency Morph Type",
    "osc_2_spectral_morph_type": "Oscillator 2 Frequency Morph Type",
    "osc_3_spectral_morph_type": "Oscillator 3 Frequency Morph Type",
    "osc_1_unison_blend": "Oscillator 1 Spectral Unison",
    "osc_2_unison_blend": "Oscillator 2 Spectral Unison",
    "osc_3_unison_blend": "Oscillator 3 Spectral Unison",
    "portamento_on": "Portamento Force",
}


_FAMILY_TOKEN_OVERRIDES = {
    "osc": "oscillator",
    "env": "envelope",
    "eq": "eq",
}


def _canonical_display_name(name: str) -> str:
    """Normalize display names for tolerant lookup (case/punct/space-insensitive)."""
    lowered = str(name).lower()
    return re.sub(r"[^a-z0-9]+", " ", lowered).strip()


def _display_candidates_for_json_key(json_key: str) -> list[str]:
    """Generate plausible REAPER display-name variants for a Vital JSON key."""
    candidates: list[str] = []

    def _add(name: str | None) -> None:
        if isinstance(name, str) and name and name not in candidates:
            candidates.append(name)

    base = _json_key_to_display(json_key)
    _add(_DISPLAY_NAME_ALIASES.get(json_key))
    _add(base)
    _add(base.replace(" Keytrack Transpose", " Transpose"))
    _add(base.replace(" Spectral Morph ", " Frequency Morph "))
    _add(base.replace(" Random Phase", " Phase Randomization"))
    _add(base.replace(" Unison Blend", " Spectral Unison"))
    _add(base.replace(" High Shelf ", " High "))
    _add(base.replace(" Low Shelf ", " Low "))

    if base.endswith(" On"):
        _add(base[:-3] + " Switch")
    if base.endswith(" Dry Wet"):
        _add(base[:-8] + " Mix")
    if base == "Chorus Cutoff":
        _add("Chorus Filter Cutoff")
    if base == "Chorus Spread":
        _add("Chorus Filter Spread")
    if base.startswith("Compressor ") and (" Lower Ratio" in base or " Upper Ratio" in base):
        _add(base[len("Compressor "):])  # Vital UI omits "Compressor" prefix for these

    return candidates


def _native_to_normalized(name: str, native: float) -> float:
    """Convert a Vital native value to a 0–1 REAPER parameter value.

    Uses the linear mapping from param_ranges.json::

        normalized = (native - min) / (max - min)

    Falls back to clamping the value to [0, 1] when the parameter is unknown.
    """
    r = _PARAM_RANGES.get(name)
    if r is None:
        return _clamp01(native)
    span = r["max"] - r["min"]
    if span == 0:
        return 0.0
    return _clamp01((native - r["min"]) / span)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _find_json_start(data: bytes) -> int:
    """Return index of first '{' in *data*, or -1 if not found."""
    for i, b in enumerate(data):
        if b == ord("{"):
            return i
    return -1


def _find_json_end(data: bytes, start: int) -> int:
    """
    Find the closing '}' that matches the '{' at *start* using a simple
    bracket counter.  Returns the index of the closing '}' + 1 (exclusive
    end), or -1 if unbalanced.
    """
    depth = 0
    in_string = False
    escape_next = False
    i = start
    while i < len(data):
        b = data[i]
        if escape_next:
            escape_next = False
            i += 1
            continue
        if b == ord("\\") and in_string:
            escape_next = True
            i += 1
            continue
        if b == ord('"'):
            in_string = not in_string
            i += 1
            continue
        if not in_string:
            if b == ord("{"):
                depth += 1
            elif b == ord("}"):
                depth -= 1
                if depth == 0:
                    return i + 1
        i += 1
    return -1


# ---------------------------------------------------------------------------
# VitalController
# ---------------------------------------------------------------------------

class VitalController:
    """
    High-level interface for Vital running as a VST/VST3 FX inside REAPER.

    Parameters
    ----------
    _rpr:
        Optional dict mapping RPR_ function names to callables.  Inject mocks
        here for unit tests.  When a name is absent the real global RPR_
        function is used.
    """

    @classmethod
    def from_reapy(cls) -> "VitalController":
        """Create a VitalController backed by reapy for external-to-REAPER use.

        Requires reapy installed (``pip install python-reapy``) and REAPER running
        with the reapy server script active.  One-time setup::

            python -c "import reapy; reapy.configure_reaper()"

        Must be called inside a ``with reapy.inside_reaper():`` block so that
        reapy's RPC context is active when RPR_ functions are dispatched.

        Example::

            import reapy
            from maestro.reaper.vital_tools import VitalController

            with reapy.inside_reaper():
                vc = VitalController.from_reapy()
                vc.discover()
                vc.set_params({"filter_1_cutoff": 84.6, "osc_1_level": 0.778})
                vc.render_to("/tmp/probe.wav")
        """
        import reapy  # noqa: PLC0415
        api = reapy.reascript_api
        # reapy.reascript_api exposes functions WITHOUT the RPR_ prefix
        # (e.g. CountTracks, GetTrack), but VitalController calls them as
        # RPR_CountTracks, RPR_GetTrack.  Build a mapping with the prefix added.
        rpr = {
            f"RPR_{fn}": getattr(api, fn)
            for fn in dir(api)
            if not fn.startswith("_")
        }
        return cls(_rpr=rpr)

    def __init__(self, _rpr: dict = None):
        self._rpr = _rpr or {}

        # State cached by discover()
        self._track = None
        self._track_idx: int = -1
        self._fx_idx: int = -1
        self._track_name: str = ""
        self._fx_name: str = ""

        # Param name → FX parameter index (built lazily by get_params())
        self._param_cache: dict = {}  # name -> int index

        # display_name → param index, built at discover() time
        # Used by set_params/set_param to route writes through TrackFX_SetParam.
        self._display_to_idx: dict = {}  # e.g. "Filter 1 Cutoff" → 104
        # canonical(display_name) → param index for punctuation/case-insensitive lookup.
        self._display_norm_to_idx: dict = {}

        # VST chunk split point (set by get_preset, used by set_preset)
        self._chunk_prefix: bytes = b""
        self._chunk_suffix: bytes = b""
        self._chunk_loaded: bool = False  # True after first successful get_preset()

        # Settle duration: time.sleep() is called inside set_preset() for this
        # duration BEFORE each SetNamedConfigParm write.  Releasing the Python
        # GIL via sleep() lets REAPER's internal VST3 processing thread drain
        # any pending state updates; without it, the 5th+ write blocks
        # indefinitely.  Increase if crashes occur under heavy write load.
        self._VITAL_WRITE_SETTLE_S: float = 0.15  # 150 ms

    # ------------------------------------------------------------------
    # Internal dispatch
    # ------------------------------------------------------------------

    def _call(self, fn_name: str, *args):
        """Dispatch to mocked or real RPR_ function."""
        if fn_name in self._rpr:
            return self._rpr[fn_name](*args)
        fn = globals().get(fn_name)
        if fn is None:
            raise RuntimeError(
                f"RPR function '{fn_name}' not found in globals and not mocked."
            )
        return fn(*args)

    # ------------------------------------------------------------------
    # discover
    # ------------------------------------------------------------------

    def discover(self) -> dict:
        """
        Scan all tracks and their FX chains for a plugin whose name contains
        "vital" (case-insensitive).  Caches the track object, track index, and
        FX index for later use.

        Returns
        -------
        dict with keys: track_idx, fx_idx, track_name, fx_name

        Raises
        ------
        RuntimeError if no Vital FX is found on any track.
        """
        proj = 0  # 0 = current project
        n_tracks = self._call("RPR_CountTracks", proj)
        for ti in range(n_tracks):
            track = self._call("RPR_GetTrack", proj, ti)
            n_fx = self._call("RPR_TrackFX_GetCount", track)
            for fi in range(n_fx):
                # GetFXName returns (retval, track, fx, buf, bufsz)
                result = self._call(
                    "RPR_TrackFX_GetFXName", track, fi, "", 512
                )
                # Accommodate both (retval, track, fx, name, sz) and plain str
                if isinstance(result, (tuple, list)):
                    fx_name = result[3] if len(result) > 3 else ""
                else:
                    fx_name = str(result)

                if "vital" not in fx_name.lower():
                    continue

                # Get track name
                tr_name_result = self._call(
                    "RPR_GetTrackName", track, "", 512
                )
                if isinstance(tr_name_result, (tuple, list)):
                    # RPR_GetTrackName → (retval, track, buf_filled, bufsz)
                    track_name = tr_name_result[2] if len(tr_name_result) > 2 else ""
                else:
                    track_name = str(tr_name_result)

                self._track = track
                self._track_idx = ti
                self._fx_idx = fi
                self._track_name = track_name
                self._fx_name = fx_name
                # Reset caches when re-discovering
                self._param_cache = {}
                self._chunk_prefix = b""
                self._chunk_suffix = b""
                cached = self._load_param_cache(ti, fi)
                if cached is not None:
                    self._display_to_idx = cached
                else:
                    self._display_to_idx = self._build_param_idx_map(track, fi)
                    self._save_param_cache(ti, fi, self._display_to_idx)
                self._display_norm_to_idx = {
                    _canonical_display_name(name): idx
                    for name, idx in self._display_to_idx.items()
                }

                return {
                    "track_idx": ti,
                    "fx_idx": fi,
                    "track_name": track_name,
                    "fx_name": fx_name,
                }

        raise RuntimeError(
            "No Vital FX found in the current REAPER project. "
            "Ensure Vital is loaded on a track and the project is open."
        )

    _PARAM_CACHE_PATH: str = "/tmp/maestro/vital_param_cache.json"

    def _require_discovered(self):
        if self._track is None:
            raise RuntimeError(
                "Call discover() before using VitalController."
            )

    def _load_param_cache(self, track_idx: int, fx_idx: int) -> dict | None:
        try:
            with open(self._PARAM_CACHE_PATH) as f:
                data = json.load(f)
            if data.get("track_idx") == track_idx and data.get("fx_idx") == fx_idx:
                return {k: int(v) for k, v in data["display_to_idx"].items()}
        except Exception:
            pass
        return None

    def _save_param_cache(self, track_idx: int, fx_idx: int, mapping: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self._PARAM_CACHE_PATH), exist_ok=True)
            tmp = self._PARAM_CACHE_PATH + ".tmp"
            with open(tmp, "w") as f:
                json.dump({"track_idx": track_idx, "fx_idx": fx_idx, "display_to_idx": mapping}, f)
            os.replace(tmp, self._PARAM_CACHE_PATH)
        except Exception:
            pass

    def _build_param_idx_map(self, track, fi: int) -> dict:
        """Build display_name → param_index for every parameter on *fi*.

        Called once at discover() time so set_param/set_params can use
        TrackFX_SetParam (which works reliably) instead of the chunk write
        mechanism (SetNamedConfigParm/vst_chunk), which returns success but
        does not apply state for VST3 plugins when called via reapy RPC.
        """
        mapping = {}
        # RPR_TrackFX_GetNumParams returns the param count.
        n = self._call("RPR_TrackFX_GetNumParams", track, fi)
        if isinstance(n, (list, tuple)):
            n = n[0]
        try:
            n = int(n)
        except (TypeError, ValueError):
            return mapping
        for i in range(n):
            result = self._call("RPR_TrackFX_GetParamName", track, fi, i, "", 256)
            if isinstance(result, (list, tuple)):
                # (retval, track, fx, idx, name, bufsz)
                name = result[4] if len(result) > 4 else ""
            else:
                name = str(result)
            if name:
                mapping[name] = i
        return mapping

    def _set_param_by_name(self, json_key: str, native_value: float) -> bool:
        """Set a single Vital parameter using TrackFX_SetParam.

        Converts the native value to a 0–1 normalized value using PARAM_RANGES,
        then looks up the REAPER parameter index by heuristically mapping the
        JSON key to a display name.

        Returns True if the parameter was found and set, False otherwise.
        """
        idx = self._resolve_param_index(json_key)
        if idx is None:
            return False
        norm = _native_to_normalized(json_key, native_value)
        self._call("RPR_TrackFX_SetParam", self._track, self._fx_idx, idx, norm)
        return True

    def _resolve_param_index(self, json_key: str) -> int | None:
        """Resolve a Vital JSON key to a REAPER parameter index."""
        candidates = _display_candidates_for_json_key(json_key)
        for display in candidates:
            idx = self._display_to_idx.get(display)
            if idx is not None:
                return idx
            idx = self._display_norm_to_idx.get(_canonical_display_name(display))
            if idx is not None:
                return idx

        # Conservative fuzzy fallback for minor naming drift.
        if not candidates or not self._display_to_idx:
            return None
        target_norm = _canonical_display_name(candidates[0])
        family = str(json_key).split("_", 1)[0].lower()
        family_token = _FAMILY_TOKEN_OVERRIDES.get(family, family)

        best_idx = None
        best_score = 0.0
        second_best = 0.0
        for name, idx in self._display_to_idx.items():
            name_norm = _canonical_display_name(name)
            if family_token and family_token not in name_norm:
                # Compressor ratio params are named "High/Low/Band ... Ratio"
                # without "Compressor" prefix in REAPER.
                if not (family == "compressor" and "ratio" in target_norm and "ratio" in name_norm):
                    continue
            score = difflib.SequenceMatcher(None, target_norm, name_norm).ratio()
            if score > best_score:
                second_best = best_score
                best_score = score
                best_idx = idx
            elif score > second_best:
                second_best = score

        if best_idx is None:
            return None
        # Require a strong match and clear margin to avoid accidental misrouting.
        if best_score >= 0.92 and (best_score - second_best) >= 0.03:
            return int(best_idx)
        return None

    # ------------------------------------------------------------------
    # get_params / set_param / set_params
    # ------------------------------------------------------------------

    def get_params(self) -> dict:
        """
        Return all scalar parameters as ``{name: native_value}`` from the
        preset JSON (Vital's internal parameter namespace).

        Note: values are in Vital's native units (e.g. filter cutoff in
        semitones, envelope times in seconds), NOT normalised to 0–1.
        Use this to inspect the current state; pass the same names and native
        values back to set_params() to update the preset.
        """
        self._require_discovered()
        preset = self.get_preset()
        settings = preset.get("settings", preset)
        return {
            k: v
            for k, v in settings.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        }

    def set_param(self, name: str, value: float) -> bool:
        """
        Set a single scalar parameter by its Vital internal name.

        The value must be in Vital's native units (same scale as the preset
        JSON).  Uses ``TrackFX_SetParam`` (reliable via reapy) by mapping the
        JSON key to a REAPER display-name param index.

        Returns True if the parameter was found and set, False otherwise.
        """
        self._require_discovered()
        return self._set_param_by_name(name, float(value))

    def set_params(self, params: dict) -> dict:
        """
        Batch-set scalar parameters via TrackFX_SetParam.

        Parameters
        ----------
        params : dict
            Mapping of Vital internal parameter name → native value.

        Returns
        -------
        dict with keys:
          - "applied": int — number of parameters successfully set
          - "not_found": list[str] — names not mappable to a REAPER param index
        """
        self._require_discovered()
        applied = 0
        not_found = []
        for name, value in params.items():
            if self._set_param_by_name(name, float(value)):
                applied += 1
            else:
                not_found.append(name)
        return {"applied": applied, "not_found": not_found}

    # ------------------------------------------------------------------
    # VST chunk helpers
    # ------------------------------------------------------------------

    def _decode_chunk(self, raw_chunk: str) -> tuple:
        """
        Decode a raw VST chunk string into (prefix_bytes, json_dict,
        suffix_bytes).

        The chunk may come back as a base64 string or (less commonly) as a
        raw binary-as-string.  We try base64 first; if that fails we treat the
        string as UTF-8 bytes directly.

        Returns (prefix: bytes, preset: dict, suffix: bytes)
        Raises ValueError if the JSON cannot be located.
        """
        # Try base64 decode; fall back to raw UTF-8
        try:
            data = base64.b64decode(raw_chunk)
        except Exception:
            data = raw_chunk.encode("utf-8", errors="replace")

        start = _find_json_start(data)
        if start == -1:
            raise ValueError(
                "Cannot locate JSON '{' in VST chunk. "
                "The chunk may be in an unexpected format."
            )

        end = _find_json_end(data, start)
        if end == -1:
            raise ValueError(
                "Unbalanced braces in VST chunk JSON — cannot find closing '}'."
            )

        prefix = data[:start]
        json_bytes = data[start:end]
        suffix = data[end:]

        preset = json.loads(json_bytes.decode("utf-8"))
        return prefix, preset, suffix

    def _encode_chunk(
        self, prefix: bytes, preset: dict, suffix: bytes
    ) -> str:
        """
        Re-encode a (prefix, preset_dict, suffix) triple into the base64
        string that REAPER expects.
        """
        json_bytes = json.dumps(preset, separators=(",", ":")).encode("utf-8")
        blob = prefix + json_bytes + suffix
        return base64.b64encode(blob).decode("ascii")

    # ------------------------------------------------------------------
    # get_preset / set_preset
    # ------------------------------------------------------------------

    def get_preset(self) -> dict:
        """
        Retrieve the full Vital preset from REAPER as a parsed Python dict.

        The preset JSON is extracted from the VST chunk.  The binary header
        and trailer around the JSON are cached internally so that set_preset()
        can reconstruct a byte-for-byte compatible chunk.
        """
        self._require_discovered()

        track = self._track
        fi = self._fx_idx

        # Try VST2 chunk name first, then VST3. Vital ships as VST3 on Linux
        # so "vst_chunk" may return empty; "vst3_chunk" is the correct key.
        # 512 KiB is more than enough for Vital's ~232 KB base64 chunk.
        # Using a tight buffer (instead of 16 MiB) avoids REAPER Python
        # binding memory accumulation when get_preset() is called repeatedly.
        _CHUNK_BUF = 524288  # 512 KiB

        raw_chunk = ""
        for parm_name in ("vst_chunk", "vst3_chunk"):
            result = self._call(
                "RPR_TrackFX_GetNamedConfigParm",
                track, fi, parm_name, "", _CHUNK_BUF
            )
            if isinstance(result, (tuple, list)):
                raw_chunk = result[4] if len(result) > 4 else ""
            else:
                raw_chunk = str(result)
            if raw_chunk:
                self._chunk_parm_name = parm_name  # remember for set_preset
                break

        if not raw_chunk:
            raise RuntimeError(
                "Could not read VST chunk from Vital — tried 'vst_chunk' and 'vst3_chunk'. "
                "Ensure Vital is loaded as VST2 or VST3 (not CLAP)."
            )

        prefix, preset, suffix = self._decode_chunk(raw_chunk)
        self._chunk_prefix = prefix
        self._chunk_suffix = suffix
        self._chunk_loaded = True
        # Explicitly release the large raw_chunk string so REAPER's Python
        # binding memory is freed before any subsequent RPR_ calls.
        del raw_chunk
        gc.collect()
        return preset

    # REAPER's ReaScript Python bridge segfaults when a string argument to
    # TrackFX_SetNamedConfigParm exceeds ~256 KiB. Vital's default chunk is
    # ~232 KiB; any real wavetable swap bumps it above the threshold. Above
    # this size we hand off via a file + in-REAPER ReaScript.
    _SAFE_RPC_CHUNK_BYTES: int = 200 * 1024  # 200 KiB — safely under 256 KiB

    _REAPER_SIDE_CHUNK_PATH: str = "/tmp/vital_chunk_payload.b64"
    _REAPER_SIDE_REQ_PATH: str = "/tmp/vital_chunk_req.json"
    _REAPER_SIDE_DONE_PATH: str = "/tmp/vital_chunk_done.json"

    def _set_preset_via_file(self, encoded: str, parm_name: str) -> None:
        """Large-chunk path: write chunk to disk and trigger an in-REAPER Lua script.

        We use a Lua ReaScript (not Python) because REAPER's ReaScript Python
        binding segfaults on large string arguments to TrackFX_SetNamedConfigParm
        even when called from inside REAPER's own process — the limit is in the
        C string marshalling, not the RPC layer. Lua's native length-prefixed
        strings sidestep that path entirely.
        """
        import os
        script_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..", "..", "skills", "vital", "scripts", "reaper_side_set_chunk.lua",
            )
        )
        if not os.path.exists(script_path):
            raise RuntimeError(
                f"reaper-side chunk-set script not found at {script_path}"
            )

        # Clean up any stale status file before triggering
        try:
            os.remove(self._REAPER_SIDE_DONE_PATH)
        except FileNotFoundError:
            pass

        with open(self._REAPER_SIDE_CHUNK_PATH, "w") as f:
            f.write(encoded)
        with open(self._REAPER_SIDE_REQ_PATH, "w") as f:
            json.dump(
                {
                    "chunk_path": self._REAPER_SIDE_CHUNK_PATH,
                    "track_idx": self._track_idx,
                    "fx_idx": self._fx_idx,
                    "parm_name": parm_name,
                },
                f,
            )

        cmd_id = self._call(
            "RPR_AddRemoveReaScript", True, 0, script_path, True
        )
        if isinstance(cmd_id, (tuple, list)):
            cmd_id = cmd_id[0] if cmd_id else 0
        if not cmd_id or int(cmd_id) <= 0:
            raise RuntimeError(
                f"AddRemoveReaScript failed to register {script_path}"
            )
        try:
            self._call("RPR_Main_OnCommand", int(cmd_id), 0)
        finally:
            self._call("RPR_AddRemoveReaScript", False, 0, script_path, True)

        # Verify the reaper-side script wrote a success status
        try:
            with open(self._REAPER_SIDE_DONE_PATH) as f:
                status = json.load(f)
        except Exception as exc:
            raise RuntimeError(
                f"reaper-side set_preset did not produce status file: {exc}"
            ) from exc
        if status.get("status") != "ok":
            raise RuntimeError(
                f"reaper-side set_preset failed: {status!r}"
            )

    def set_preset(self, preset: dict):
        """
        Write a modified preset dict back to Vital via the VST chunk.

        Requires that get_preset() has been called at least once (so that the
        binary header/suffix are cached).
        """
        self._require_discovered()
        if not self._chunk_loaded:
            raise RuntimeError(
                "call get_preset() before set_preset() — needed to cache binary chunk envelope"
            )

        encoded = self._encode_chunk(
            self._chunk_prefix, preset, self._chunk_suffix
        )
        parm_name = getattr(self, "_chunk_parm_name", "vst_chunk")
        # Force GC before the chunk write to release prior buffer refs.
        gc.collect()
        # Brief yield: releasing the Python GIL via time.sleep() allows
        # REAPER's internal VST3 processing thread to complete any pending
        # state updates before we issue the next SetNamedConfigParm write.
        # Without this, the 5th+ write blocks indefinitely.  The sleep does
        # NOT need to be at the top level of the ReaScript — any time.sleep()
        # call releases the GIL and lets REAPER's threads run.
        time.sleep(self._VITAL_WRITE_SETTLE_S)

        if len(encoded) > self._SAFE_RPC_CHUNK_BYTES:
            self._set_preset_via_file(encoded, parm_name)
            return

        self._call(
            "RPR_TrackFX_SetNamedConfigParm",
            self._track, self._fx_idx, parm_name, encoded
        )

    # ------------------------------------------------------------------
    # Modulations
    # ------------------------------------------------------------------

    def get_modulations(self) -> list:
        """
        Return the modulation routes from the current preset.

        Each entry is a dict with keys: "source", "destination", "amount".
        """
        preset = self.get_preset()
        raw = (
            preset.get("settings", {}).get("modulations")
            or preset.get("modulations")
            or []
        )
        result = []
        for entry in raw:
            result.append(
                {
                    "source": entry.get("source", ""),
                    "destination": entry.get("destination", ""),
                    "amount": float(entry.get("amount", 0.0)),
                }
            )
        return result

    def _get_modulations_key(self, preset: dict) -> tuple:
        """
        Return (parent_dict, key) where parent_dict[key] is the modulations
        list.  Handles both top-level and settings-nested layouts.
        """
        settings = preset.get("settings", {})
        if "modulations" in settings:
            return settings, "modulations"
        if "modulations" in preset:
            return preset, "modulations"
        # Create in settings (Vital's canonical location)
        if "settings" not in preset:
            preset["settings"] = {}
        preset["settings"]["modulations"] = []
        return preset["settings"], "modulations"

    def set_modulation(self, source: str, destination: str, amount: float):
        """
        Add or update a modulation route.

        Vital's modulation matrix has exactly 64 fixed slots.  This method
        updates an existing slot that matches source+destination, or fills the
        first empty slot (source=='' and destination=='').  Raising
        RuntimeError if all slots are occupied.

        Changes are written back to Vital immediately via a preset round-trip.
        """
        preset = self.get_preset()
        parent, key = self._get_modulations_key(preset)
        mods: list = parent[key]

        # Update existing matching slot
        for entry in mods:
            if entry.get("source") == source and entry.get("destination") == destination:
                entry["amount"] = amount
                self.set_preset(preset)
                return

        # Fill first empty slot (source='' AND destination='')
        for entry in mods:
            if not entry.get("source") and not entry.get("destination"):
                entry["source"] = source
                entry["destination"] = destination
                entry["amount"] = amount
                self.set_preset(preset)
                return

        raise RuntimeError(
            "All 64 Vital modulation slots are occupied — cannot add new route."
        )

    def remove_modulation(self, source: str, destination: str):
        """
        Clear a modulation route (no-op if it doesn't exist).

        The slot is cleared (source/destination set to '') rather than removed,
        because Vital requires exactly 64 modulation slots.
        """
        preset = self.get_preset()
        parent, key = self._get_modulations_key(preset)
        mods: list = parent[key]
        changed = False
        for entry in mods:
            if entry.get("source") == source and entry.get("destination") == destination:
                entry["source"] = ""
                entry["destination"] = ""
                entry["amount"] = 0.0
                changed = True
                break
        if changed:
            self.set_preset(preset)

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render_to(self, output_path: str):
        """
        Render the Vital track to a WAV file at *output_path*.

        Steps:
          1. Solo the track (I_SOLO = 2 → solo in place).
          2. Set render output path via RPR_GetSetProjectInfo_String.
          3. Set render source to selected tracks (RENDER_TRACKS flag).
          4. Run the "Render project, auto-close render dialog" command (41824).
          5. Un-solo the track.

        Note: ``Main_OnCommand(41824)`` is synchronous in REAPER's main thread —
        it returns only after the render completes.  The output file will exist
        immediately after this method returns.
        """
        self._require_discovered()
        track = self._track

        # Solo the Vital track so the master mix contains only its audio.
        self._call("RPR_SetOnlyTrackSelected", track)  # deselects all others
        self._call("RPR_SetMediaTrackInfo_Value", track, "I_SOLO", 2)

        # REAPER treats RENDER_FILE as the output *directory* and appends
        # RENDER_PATTERN (+ format extension) to form the actual filename.
        # Split the caller-supplied path so the file lands at exactly output_path.
        import os as _os
        _dir = _os.path.dirname(_os.path.abspath(output_path))
        _stem = _os.path.splitext(_os.path.basename(output_path))[0]
        self._call("RPR_GetSetProjectInfo_String", 0, "RENDER_FILE", _dir, True)
        self._call("RPR_GetSetProjectInfo_String", 0, "RENDER_PATTERN", _stem, True)
        # RENDER_SETTINGS=0 → master mix (track is soloed, so only Vital is audible)
        self._call("RPR_GetSetProjectInfo", 0, "RENDER_SETTINGS", 0, True)
        # RENDER_BOUNDSFLAG=1 → entire project length (not custom time selection)
        self._call("RPR_GetSetProjectInfo", 0, "RENDER_BOUNDSFLAG", 1, True)

        # Command 41824 = "Render project, using the most recent render settings,
        # auto-close render dialog" — renders without user interaction and
        # returns only after the render completes.
        self._call("RPR_Main_OnCommand", 41824, 0)

        # Unsolo immediately after render completes (synchronous, safe here).
        self._call("RPR_SetMediaTrackInfo_Value", track, "I_SOLO", 0)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def write_result(
        self,
        path: str,
        ok: bool,
        data=None,
        error: str = None,
    ):
        """
        Write a JSON result envelope to *path*.

        Format::

            {"ok": true|false, "data": <any>, "error": "<msg>"|null}
        """
        payload = {
            "ok": ok,
            "data": data,
            "error": error,
        }
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
