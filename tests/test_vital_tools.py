"""
Unit tests for maestro.reaper.vital_tools.VitalController.

All REAPER API calls are intercepted via the _rpr mock dict.
No real REAPER process is needed.
"""

import base64
import json
import os
import tempfile

import pytest

from maestro.reaper.vital_tools import VitalController, _find_json_start, _find_json_end


# ---------------------------------------------------------------------------
# Helpers — build a minimal mock RPR environment
# ---------------------------------------------------------------------------

def _make_preset(extra_mods=None) -> dict:
    """Return a minimal Vital preset dict.

    Always includes one empty modulation slot so set_modulation() can add
    new routes without raising "all slots occupied".
    """
    mods = list(extra_mods or [])
    # Ensure at least one empty slot (Vital requires fixed-length list)
    mods.append({"source": "", "destination": "", "amount": 0.0})
    return {
        "synth_version": "1.5.5",
        "preset_name": "Test Preset",
        "settings": {
            "osc_1_level": 0.8,
            "filter_cutoff": 0.5,
            "modulations": mods,
        },
    }


def _encode_preset(preset: dict, prefix: bytes = b"VITALHEADER\x00") -> str:
    """Encode a preset dict into a fake VST chunk (base64)."""
    json_bytes = json.dumps(preset, separators=(",", ":")).encode("utf-8")
    blob = prefix + json_bytes
    return base64.b64encode(blob).decode("ascii")


def _make_rpr(
    n_tracks=1,
    n_fx=1,
    fx_name="Vital",
    track_name="Synth",
    n_params=3,
    param_names=None,
    param_values=None,
    preset=None,
):
    """
    Build a minimal _rpr mock dict.

    Mocked functions:
      RPR_CountTracks, RPR_GetTrack, RPR_TrackFX_GetCount,
      RPR_TrackFX_GetFXName, RPR_GetTrackName,
      RPR_TrackFX_GetNumParams, RPR_TrackFX_GetParamName,
      RPR_TrackFX_GetParam, RPR_TrackFX_SetParam,
      RPR_TrackFX_GetNamedConfigParm, RPR_TrackFX_SetNamedConfigParm.
    """
    if param_names is None:
        param_names = [f"param_{i}" for i in range(n_params)]
    if param_values is None:
        param_values = [float(i) / max(n_params - 1, 1) for i in range(n_params)]

    # Tracks are just integers for simplicity
    tracks = list(range(n_tracks))
    # Per-track FX names: list of lists
    fx_names_per_track = [[fx_name] * n_fx for _ in range(n_tracks)]

    # Mutable state for set/get operations
    state = {
        "params": list(param_values),
        "chunk": _encode_preset(preset or _make_preset()),
        "set_param_calls": [],
        "set_chunk_calls": [],
    }

    def RPR_CountTracks(proj):
        return n_tracks

    def RPR_GetTrack(proj, idx):
        return tracks[idx]

    def RPR_TrackFX_GetCount(track):
        return n_fx

    def RPR_TrackFX_GetFXName(track, fi, buf, bufsz):
        name = fx_names_per_track[track][fi] if track < n_tracks and fi < n_fx else ""
        return (True, track, fi, name, len(name))

    def RPR_GetTrackName(track, buf, bufsz):
        # Real signature: (retval, track, buf_filled, bufsz)
        return (True, track, track_name, bufsz)

    def RPR_TrackFX_GetNumParams(track, fi):
        return n_params

    def RPR_TrackFX_GetParamName(track, fi, pi, buf, bufsz):
        # Real signature: (retval, track, fxindex, param_idx, buf_filled, bufsz)
        # param_idx (int) at [3], name string at [4]
        name = param_names[pi] if pi < len(param_names) else f"param_{pi}"
        return (True, track, fi, pi, name, len(name))

    def RPR_TrackFX_GetParam(track, fi, pi, minv, maxv):
        val = state["params"][pi] if pi < len(state["params"]) else 0.0
        return (val, minv, maxv)

    def RPR_TrackFX_SetParam(track, fi, pi, value):
        state["set_param_calls"].append((track, fi, pi, value))
        if pi < len(state["params"]):
            state["params"][pi] = value
        return True

    def RPR_TrackFX_GetNamedConfigParm(track, fi, parmname, buf, bufsz):
        if parmname in ("vst_chunk", "vst3_chunk"):
            return (True, track, fi, parmname, state["chunk"], bufsz)
        return (False, track, fi, parmname, "", bufsz)

    def RPR_SetOnlyTrackSelected(track):
        return None

    def RPR_TrackFX_SetNamedConfigParm(track, fi, parmname, value):
        if parmname == "vst_chunk":
            state["chunk"] = value
            state["set_chunk_calls"].append(value)
        return True

    rpr = {
        "RPR_CountTracks": RPR_CountTracks,
        "RPR_GetTrack": RPR_GetTrack,
        "RPR_TrackFX_GetCount": RPR_TrackFX_GetCount,
        "RPR_TrackFX_GetFXName": RPR_TrackFX_GetFXName,
        "RPR_GetTrackName": RPR_GetTrackName,
        "RPR_TrackFX_GetNumParams": RPR_TrackFX_GetNumParams,
        "RPR_TrackFX_GetParamName": RPR_TrackFX_GetParamName,
        "RPR_TrackFX_GetParam": RPR_TrackFX_GetParam,
        "RPR_TrackFX_SetParam": RPR_TrackFX_SetParam,
        "RPR_TrackFX_GetNamedConfigParm": RPR_TrackFX_GetNamedConfigParm,
        "RPR_TrackFX_SetNamedConfigParm": RPR_TrackFX_SetNamedConfigParm,
        "RPR_SetOnlyTrackSelected": RPR_SetOnlyTrackSelected,
    }
    return rpr, state


# ---------------------------------------------------------------------------
# Internal helper tests
# ---------------------------------------------------------------------------

class TestChunkHelpers:
    def test_find_json_start_with_header(self):
        data = b"BINARY\x00\x01\x02" + b'{"key": 1}'
        assert _find_json_start(data) == 9

    def test_find_json_start_no_brace(self):
        assert _find_json_start(b"no brace here") == -1

    def test_find_json_end_balanced(self):
        data = b'{"a":{"b":1}}'
        start = _find_json_start(data)
        end = _find_json_end(data, start)
        assert end == len(data)

    def test_find_json_end_with_suffix(self):
        data = b'{"x":1}TRAILING'
        start = 0
        end = _find_json_end(data, start)
        assert end == 7  # just past closing '}'


# ---------------------------------------------------------------------------
# 1. discover()
# ---------------------------------------------------------------------------

class TestDiscover:
    def test_finds_vital_track(self):
        rpr, _ = _make_rpr(fx_name="Vital Synth", track_name="Lead")
        vc = VitalController(_rpr=rpr)
        result = vc.discover()
        assert result["track_idx"] == 0
        assert result["fx_idx"] == 0
        assert result["track_name"] == "Lead"
        assert "vital" in result["fx_name"].lower()

    def test_case_insensitive(self):
        rpr, _ = _make_rpr(fx_name="VITAL (VST3)")
        vc = VitalController(_rpr=rpr)
        result = vc.discover()
        assert result["fx_idx"] == 0

    def test_raises_when_absent(self):
        rpr, _ = _make_rpr(fx_name="Serum")
        vc = VitalController(_rpr=rpr)
        with pytest.raises(RuntimeError, match="No Vital FX found"):
            vc.discover()

    def test_skips_non_vital_fx(self):
        """Two FX per track; Vital is second (fx index 1)."""
        # Use a name that does NOT contain "vital" as a substring
        rpr, _ = _make_rpr(n_fx=2, fx_name="Serum")
        # Override so only fx index 1 is Vital
        original_get_fx_name = rpr["RPR_TrackFX_GetFXName"]

        def patched(track, fi, buf, bufsz):
            if fi == 1:
                return (True, track, fi, "Vital", 5)
            return original_get_fx_name(track, fi, buf, bufsz)

        rpr["RPR_TrackFX_GetFXName"] = patched
        vc = VitalController(_rpr=rpr)
        result = vc.discover()
        assert result["fx_idx"] == 1

    def test_caches_state(self):
        rpr, _ = _make_rpr()
        vc = VitalController(_rpr=rpr)
        vc.discover()
        assert vc._track == 0
        assert vc._track_idx == 0
        assert vc._fx_idx == 0

    def test_clears_param_cache_on_rediscover(self):
        rpr, _ = _make_rpr()
        vc = VitalController(_rpr=rpr)
        vc._param_cache = {"stale": 99}
        vc.discover()
        assert vc._param_cache == {}


# ---------------------------------------------------------------------------
# 2. get_params()
# ---------------------------------------------------------------------------

class TestGetParams:
    def setup_method(self):
        # get_params now reads from the preset JSON, not REAPER's param API
        preset = _make_preset()  # has osc_1_level: 0.8, filter_cutoff: 0.5
        self.rpr, self.state = _make_rpr(preset=preset)
        self.vc = VitalController(_rpr=self.rpr)
        self.vc.discover()

    def test_returns_scalar_params_from_preset(self):
        params = self.vc.get_params()
        assert "osc_1_level" in params
        assert "filter_cutoff" in params

    def test_excludes_lists(self):
        params = self.vc.get_params()
        assert "modulations" not in params  # list in preset["settings"]

    def test_values_match_preset(self):
        params = self.vc.get_params()
        assert abs(params["osc_1_level"] - 0.8) < 1e-6
        assert abs(params["filter_cutoff"] - 0.5) < 1e-6

    def test_raises_before_discover(self):
        rpr, _ = _make_rpr()
        vc = VitalController(_rpr=rpr)
        with pytest.raises(RuntimeError, match="discover"):
            vc.get_params()


# ---------------------------------------------------------------------------
# 3. set_param()
# ---------------------------------------------------------------------------

class TestSetParam:
    def setup_method(self):
        preset = _make_preset()  # has osc_1_level: 0.8, filter_cutoff: 0.5
        # param_names must match the display names that _json_key_to_display()
        # generates: osc_1_level → "Oscillator 1 Level", filter_cutoff → "Filter Cutoff"
        self.rpr, self.state = _make_rpr(
            preset=preset,
            n_params=3,
            param_names=["Oscillator 1 Level", "Filter Cutoff", "Extra Param"],
            param_values=[0.8, 0.5, 0.0],
        )
        self.vc = VitalController(_rpr=self.rpr)
        self.vc.discover()

    def test_sets_known_param(self):
        result = self.vc.set_param("osc_1_level", 0.95)
        assert result is True
        # set_param uses TrackFX_SetParam; verify the normalized value was written
        assert len(self.state["set_param_calls"]) == 1
        _, _, idx, norm_val = self.state["set_param_calls"][0]
        assert idx == 0  # "Oscillator 1 Level" is param 0
        assert abs(norm_val - 0.95) < 1e-6  # osc_1_level range 0-1 → norm==native

    def test_returns_false_for_unknown(self):
        result = self.vc.set_param("nonexistent_param", 0.5)
        assert result is False

    def test_raises_before_discover(self):
        rpr, _ = _make_rpr()
        vc = VitalController(_rpr=rpr)
        with pytest.raises(RuntimeError, match="discover"):
            vc.set_param("osc_1_level", 0.5)

    def test_uses_set_param_not_chunk(self):
        """set_param routes through TrackFX_SetParam, not the chunk write."""
        before_chunks = len(self.state["set_chunk_calls"])
        self.vc.set_param("osc_1_level", 0.7)
        assert len(self.state["set_chunk_calls"]) == before_chunks  # no chunk writes
        assert len(self.state["set_param_calls"]) == 1             # one SetParam call

    def test_normalizes_value(self):
        """Native value is converted to 0-1 using param_ranges min/max."""
        # osc_1_level range is 0-1, so norm == native
        self.vc.set_param("osc_1_level", 0.3)
        _, _, _, norm_val = self.state["set_param_calls"][-1]
        assert 0.0 <= norm_val <= 1.0


# ---------------------------------------------------------------------------
# 4. set_params()
# ---------------------------------------------------------------------------

class TestSetParams:
    def setup_method(self):
        preset = _make_preset()  # has osc_1_level: 0.8, filter_cutoff: 0.5
        self.rpr, self.state = _make_rpr(
            preset=preset,
            n_params=3,
            param_names=["Oscillator 1 Level", "Filter Cutoff", "Extra Param"],
            param_values=[0.8, 0.5, 0.0],
        )
        self.vc = VitalController(_rpr=self.rpr)
        self.vc.discover()

    def test_applies_all_found(self):
        result = self.vc.set_params({"osc_1_level": 0.9, "filter_cutoff": 64.0})
        assert result["applied"] == 2
        assert result["not_found"] == []

    def test_reports_not_found(self):
        result = self.vc.set_params({"osc_1_level": 0.5, "missing_x": 0.9, "missing_y": 0.1})
        assert result["applied"] == 1
        assert set(result["not_found"]) == {"missing_x", "missing_y"}

    def test_empty_dict(self):
        result = self.vc.set_params({})
        assert result["applied"] == 0
        assert result["not_found"] == []

    def test_uses_set_param_per_key(self):
        """Each found key triggers one TrackFX_SetParam call, no chunk writes."""
        self.vc.set_params({"osc_1_level": 0.9, "filter_cutoff": 64.0})
        assert len(self.state["set_param_calls"]) == 2
        assert len(self.state["set_chunk_calls"]) == 0

    def test_normalized_values_in_range(self):
        """All values sent to TrackFX_SetParam are clamped to [0, 1]."""
        self.vc.set_params({"osc_1_level": 0.42})
        _, _, _, norm_val = self.state["set_param_calls"][-1]
        assert 0.0 <= norm_val <= 1.0


# ---------------------------------------------------------------------------
# 5. set_modulation() round-trip
# ---------------------------------------------------------------------------

class TestModulations:
    def setup_method(self):
        self.preset = _make_preset(
            extra_mods=[{"source": "lfo_1", "destination": "filter_cutoff", "amount": 0.5}]
        )
        self.rpr, self.state = _make_rpr(preset=self.preset)
        self.vc = VitalController(_rpr=self.rpr)
        self.vc.discover()

    def _decode_current_chunk(self) -> dict:
        """Decode whatever chunk is currently stored in state."""
        chunk = self.state["chunk"]
        prefix, preset, suffix = self.vc._decode_chunk(chunk)
        return preset

    def test_get_modulations_returns_list(self):
        mods = self.vc.get_modulations()
        # _make_preset includes lfo_1→filter_cutoff + one empty slot = 2 total
        assert len(mods) == 2
        active = [m for m in mods if m["source"]]
        assert len(active) == 1
        assert active[0]["source"] == "lfo_1"
        assert active[0]["destination"] == "filter_cutoff"
        assert abs(active[0]["amount"] - 0.5) < 1e-6

    def test_set_modulation_add_new(self):
        """Adding fills the first empty slot, not appends (Vital fixed-length list)."""
        # _make_preset includes one empty slot alongside lfo_1→filter_cutoff
        self.vc.set_modulation("env_2", "osc_1_level", 0.75)
        preset = self._decode_current_chunk()
        mods = preset["settings"]["modulations"]
        sources = [m["source"] for m in mods]
        assert "env_2" in sources
        env_mod = next(m for m in mods if m["source"] == "env_2")
        assert abs(env_mod["amount"] - 0.75) < 1e-6
        # Slot count unchanged — empty slot was filled, not a new one appended
        assert len(mods) == 2  # lfo_1 + env_2 (formerly empty slot)

    def test_set_modulation_update_existing(self):
        self.vc.set_modulation("lfo_1", "filter_cutoff", 0.9)
        preset = self._decode_current_chunk()
        mods = preset["settings"]["modulations"]
        lfo_mod = next(m for m in mods if m["source"] == "lfo_1")
        assert abs(lfo_mod["amount"] - 0.9) < 1e-6
        # Count unchanged (2 slots: lfo_1 + empty) — updated, not duplicated
        assert len(mods) == 2

    def test_set_modulation_roundtrip_preserves_preset_name(self):
        self.vc.set_modulation("env_1", "pitch", 0.1)
        preset = self._decode_current_chunk()
        assert preset["preset_name"] == "Test Preset"

    def test_remove_modulation_existing(self):
        """Slot is cleared (source/dest='') not removed — Vital requires fixed-length list."""
        self.vc.remove_modulation("lfo_1", "filter_cutoff")
        preset = self._decode_current_chunk()
        mods = preset["settings"]["modulations"]
        # Length unchanged — slot is cleared, not deleted (was 2: lfo_1 + empty)
        assert len(mods) == 2
        cleared = next(m for m in mods if m.get("source") == "" and m.get("destination") == "")
        assert cleared["amount"] == 0.0

    def test_remove_modulation_nonexistent_noop(self):
        # Should not raise and not write chunk
        initial_calls = len(self.state["set_chunk_calls"])
        self.vc.remove_modulation("ghost_lfo", "nowhere")
        assert len(self.state["set_chunk_calls"]) == initial_calls  # no write

    def test_set_chunk_called_on_set_modulation(self):
        initial_calls = len(self.state["set_chunk_calls"])
        self.vc.set_modulation("env_3", "volume", 0.3)
        assert len(self.state["set_chunk_calls"]) == initial_calls + 1


# ---------------------------------------------------------------------------
# 6. write_result()
# ---------------------------------------------------------------------------

class TestWriteResult:
    def test_writes_success(self):
        vc = VitalController()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name
        try:
            vc.write_result(path, ok=True, data={"presets": 3})
            with open(path, "r") as fh:
                payload = json.load(fh)
            assert payload["ok"] is True
            assert payload["data"] == {"presets": 3}
            assert payload["error"] is None
        finally:
            os.unlink(path)

    def test_writes_failure(self):
        vc = VitalController()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name
        try:
            vc.write_result(path, ok=False, error="Vital not found")
            with open(path, "r") as fh:
                payload = json.load(fh)
            assert payload["ok"] is False
            assert payload["data"] is None
            assert payload["error"] == "Vital not found"
        finally:
            os.unlink(path)

    def test_writes_valid_json(self):
        vc = VitalController()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name
        try:
            vc.write_result(path, ok=True, data=[1, 2, 3])
            with open(path, "r") as fh:
                content = fh.read()
            # Must parse without error
            obj = json.loads(content)
            assert isinstance(obj, dict)
            assert set(obj.keys()) == {"ok", "data", "error"}
        finally:
            os.unlink(path)

    def test_null_data_and_error(self):
        vc = VitalController()
        with tempfile.NamedTemporaryFile(mode="r", suffix=".json", delete=False) as f:
            path = f.name
        try:
            vc.write_result(path, ok=True)
            with open(path, "r") as fh:
                payload = json.load(fh)
            assert payload["data"] is None
            assert payload["error"] is None
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# get_preset / set_preset direct tests
# ---------------------------------------------------------------------------

class TestPresetRoundTrip:
    def setup_method(self):
        self.original_preset = _make_preset()
        self.rpr, self.state = _make_rpr(preset=self.original_preset)
        self.vc = VitalController(_rpr=self.rpr)
        self.vc.discover()

    def test_get_preset_returns_dict(self):
        preset = self.vc.get_preset()
        assert isinstance(preset, dict)
        assert preset["preset_name"] == "Test Preset"
        assert "settings" in preset

    def test_get_preset_caches_prefix(self):
        self.vc.get_preset()
        assert self.vc._chunk_prefix == b"VITALHEADER\x00"

    def test_set_preset_roundtrip(self):
        preset = self.vc.get_preset()
        preset["preset_name"] = "Modified"
        self.vc.set_preset(preset)

        # Decode what was written back
        new_chunk = self.state["set_chunk_calls"][-1]
        _, decoded, _ = self.vc._decode_chunk(new_chunk)
        assert decoded["preset_name"] == "Modified"
        # Binary prefix preserved
        raw = base64.b64decode(new_chunk)
        assert raw[:len(b"VITALHEADER\x00")] == b"VITALHEADER\x00"

    def test_set_preset_calls_set_named_config_parm(self):
        preset = self.vc.get_preset()
        self.vc.set_preset(preset)
        assert len(self.state["set_chunk_calls"]) == 1


# ---------------------------------------------------------------------------
# 8. VitalController.from_reapy()
# ---------------------------------------------------------------------------

class TestFromReapy:
    """Test the from_reapy() classmethod using a mocked reapy module."""

    def _make_mock_reapy(self):
        """Build a minimal mock reapy with a reascript_api namespace exposing RPR_ names."""
        class _MockReascriptApi:
            RPR_CountTracks = staticmethod(lambda proj: 0)
            RPR_GetTrack = staticmethod(lambda proj, idx: None)
            RPR_TrackFX_GetCount = staticmethod(lambda track: 0)
            RPR_TrackFX_GetFXName = staticmethod(lambda *a: (False, None, 0, "", 0))
            RPR_TrackFX_GetNamedConfigParm = staticmethod(lambda *a: (False, None, 0, "", "", 0))
            RPR_TrackFX_SetNamedConfigParm = staticmethod(lambda *a: False)

        class _MockReapy:
            reascript_api = _MockReascriptApi()

        return _MockReapy()

    def test_from_reapy_returns_vital_controller(self, monkeypatch):
        import sys
        mock_reapy = self._make_mock_reapy()
        monkeypatch.setitem(sys.modules, "reapy", mock_reapy)

        vc = VitalController.from_reapy()

        assert isinstance(vc, VitalController)

    def test_from_reapy_populates_rpr_dict(self, monkeypatch):
        import sys
        mock_reapy = self._make_mock_reapy()
        monkeypatch.setitem(sys.modules, "reapy", mock_reapy)

        vc = VitalController.from_reapy()

        assert len(vc._rpr) > 0, "_rpr should be populated with RPR_ functions"
        for key in vc._rpr:
            assert key.startswith("RPR_"), f"Non-RPR_ key in _rpr: {key}"

    def test_from_reapy_rpr_values_are_callable(self, monkeypatch):
        import sys
        mock_reapy = self._make_mock_reapy()
        monkeypatch.setitem(sys.modules, "reapy", mock_reapy)

        vc = VitalController.from_reapy()

        for key, fn in vc._rpr.items():
            assert callable(fn), f"_rpr['{key}'] is not callable"

    def test_from_reapy_missing_reapy_raises_import_error(self, monkeypatch):
        import sys
        # Remove reapy from sys.modules if present, block import
        monkeypatch.delitem(sys.modules, "reapy", raising=False)

        import builtins
        real_import = builtins.__import__

        def block_reapy(name, *args, **kwargs):
            if name == "reapy":
                raise ImportError("No module named 'reapy'")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", block_reapy)

        with pytest.raises(ImportError, match="reapy"):
            VitalController.from_reapy()
