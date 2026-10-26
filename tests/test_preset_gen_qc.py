"""preset_gen ground-truth QC: silent wavetables never sampled, values inside Vital's ranges, level QC/normalization."""
import json
import random
from pathlib import Path

import numpy as np

from maestro.synth import preset_gen as pg

PARAMS = json.loads((Path(__file__).resolve().parents[1] / "opencode/skills/vital/data/params.json").read_text())


def _wt(name: str) -> dict:
    return {"name": name, "groups": [{"components": [{"keyframes": [{}, {}, {}, {}]}]}]}


def test_silent_wavetables_never_sampled():
    lib = [_wt(n) for n in sorted(pg._SILENT_WAVETABLES)] + [_wt("Basic Shapes"), _wt("Squish Flange")]
    for i in range(200):
        s = pg.generate_preset(pg.ARCHETYPES[i % len(pg.ARCHETYPES)], random.Random(i), wavetable_lib=lib)["settings"]
        for n in (1, 2, 3):
            if s[f"osc_{n}_on"]:
                assert s["wavetables"][n - 1]["name"] not in pg._SILENT_WAVETABLES


def test_values_inside_vital_ranges_and_live_cross_mod():
    for i in range(200):
        s = pg.generate_preset(pg.ARCHETYPES[i % len(pg.ARCHETYPES)], random.Random(i))["settings"]
        for k, v in s.items():
            if k in PARAMS and isinstance(v, (int, float)):
                assert PARAMS[k]["min"] - 1e-9 <= v <= PARAMS[k]["max"] + 1e-9, (k, v)
        for n in (1, 2, 3):
            if not s[f"osc_{n}_on"]:
                continue
            d = int(s[f"osc_{n}_distortion_type"])
            src_a, src_b = n % 3 + 1, (n + 1) % 3 + 1
            assert d not in (9, 12)  # <- Sample (sample layer is off)
            assert d not in (7, 10) or s[f"osc_{src_a}_on"]
            assert d not in (8, 11) or s[f"osc_{src_b}_on"]
        assert -12.0 - 1e-6 <= pg._volume_to_db(s["volume"]) <= 6.1


def test_level_qc_and_normalize():
    sr = 44100
    t = np.arange(sr * 2) / sr
    loud = 0.1 * np.sin(2 * np.pi * 220 * t)
    assert pg.level_qc(pg.render_level(loud, sr, 1.5)) == (True, "")
    assert pg.level_qc(pg.render_level(np.zeros(sr), sr, 0.5))[1] == "digital silence"
    nan = loud.copy()
    nan[10] = np.nan
    assert not pg.level_qc(pg.render_level(nan, sr, 1.5))[0]
    quiet = pg.render_level(1e-3 * loud, sr, 1.5)  # ~ -63 dBFS
    assert not pg.level_qc(quiet)[0]
    settings = {"volume": pg._db_to_volume(-6.0)}
    new = pg.normalize_volume(settings, quiet)
    assert new is not None and abs(pg._volume_to_db(new) - 6.0) < 0.1  # wants +37 dB, clamped at Vital's +6 dB max
    assert pg.normalize_volume(settings, pg.render_level(loud, sr, 1.5)) is None  # -23 dBFS: inside the band


def test_no_value_reaches_vital_clamped():
    """Raw generator output (no range clamp) already lies inside Vital's ranges, except keys whose clamp is intended."""
    saved = pg._PARAM_RANGES
    pg._PARAM_RANGES = {}
    try:
        intended = {"chorus_delay_2", "reverb_low_shelf_gain", "voice_priority"}  # draws reaching past Vital's max/min
        stereo, delay, fade = [], [], []
        for i in range(300):
            s = pg.generate_preset(pg.ARCHETYPES[i % len(pg.ARCHETYPES)], random.Random(i))["settings"]
            for k, v in s.items():
                if k in PARAMS and k not in intended and isinstance(v, (int, float)):
                    assert PARAMS[k]["min"] - 1e-9 <= v <= PARAMS[k]["max"] + 1e-9, (k, v)
            stereo += [s[f"lfo_{n}_stereo"] for n in range(1, 9)]
            delay += [s[f"lfo_{n}_delay_time"] for n in range(1, 9)]
            fade += [s[f"lfo_{n}_fade_time"] for n in range(1, 9)]
            for k in ("chorus_voices", "compressor_enabled_bands", "delay_tempo", "chorus_tempo"):
                assert s[k] == int(s[k])
        assert min(delay) > 0 and min(fade) > 0
        assert len({round(x, 3) for x in stereo if x}) > 50  # a spread of offsets, not a constant
    finally:
        pg._PARAM_RANGES = saved


def test_portamento_is_deliberate():
    times = []
    for i in range(400):
        s = pg.generate_preset(pg.ARCHETYPES[i % len(pg.ARCHETYPES)], random.Random(i))["settings"]
        if s["portamento_on"]:
            assert 0.03 - 1e-9 <= 2 ** s["portamento_time"] <= 0.3 + 1e-9
            times.append(s["portamento_time"])
        else:
            assert s["portamento_time"] == -10.0  # no glide (portamento_on is not a gate in this Vital build)
    assert 0.05 < len(times) / 400 < 0.16


def test_manifest_quantizes_to_skill_precision():
    import copy
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_v5_manifest import quantize_to_skill
    from maestro.v5.planner import parsed
    p = pg.generate_preset("pad", random.Random(3))
    q = quantize_to_skill(copy.deepcopy(p))
    assert quantize_to_skill(copy.deepcopy(q)) == q  # idempotent: the GT is a fixed point of what the agent can type
    s = q["settings"]
    assert all(parsed(k, s[k]) == s[k] for k in ("reverb_size", "filter_1_cutoff", "env_1_attack", "volume"))


def test_glide_forced_and_audibility_rule():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import build_v5_manifest as bm
    for i in range(300):  # a deliberate glide forces portamento (Vital otherwise glides only between overlapping notes)
        s = pg.generate_preset(pg.ARCHETYPES[i % len(pg.ARCHETYPES)], random.Random(i))["settings"]
        assert s["portamento_force"] == s["portamento_on"]
    n = lambda t, d: {"pitch": 60, "velocity": 100, "start_s": t, "dur_s": d}  # noqa: E731
    assert bm.overlap_share([n(0, 0.4), n(0.5, 0.4), n(1.0, 0.4)]) == 0.0
    assert bm.overlap_share([n(0, 0.6), n(0.5, 0.6), n(1.0, 0.4)]) == 1.0
    off = bm.glide_off(pg.generate_preset("lead", random.Random(1)))["settings"]
    assert (off["portamento_on"], off["portamento_force"], off["portamento_time"]) == (0.0, 0.0, -10.0)
    sr, rng = 44100, np.random.default_rng(0)
    t = np.arange(sr) / sr
    tone = lambda f: np.stack([np.sin(2 * np.pi * f * t)] * 2, 1).astype(np.float32) * 0.1  # noqa: E731
    takes = [tone(220) + 1e-4 * rng.standard_normal((sr, 2)).astype(np.float32) for _ in range(4)]
    assert not bm.glide_audible(takes, takes[0].copy())["audible"]
    assert bm.glide_audible(takes, tone(330))["audible"]
