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
