import importlib.util
from pathlib import Path

import numpy as np
import soundfile as sf


def _load_module():
    script_path = Path(__file__).resolve().parent.parent / "scripts" / "smoke_test_lua_vs_vita.py"
    spec = importlib.util.spec_from_file_location("smoke_test_lua_vs_vita", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_should_check_match_modes():
    mod = _load_module()
    assert mod._should_check_match("none", "reaper") is False
    assert mod._should_check_match("exact", "vital") is True
    assert mod._should_check_match("auto", "reaper") is True
    assert mod._should_check_match("auto", "vital") is False
    assert mod._should_check_match("auto", None) is False


def test_exact_wav_match(tmp_path):
    mod = _load_module()
    a = np.array([[0.0, 0.1], [0.2, -0.1], [0.0, 0.0]], dtype=np.float32)
    b = a.copy()
    c = np.array([[0.0, 0.1], [0.2, -0.2], [0.0, 0.0]], dtype=np.float32)

    ref = tmp_path / "ref.wav"
    same = tmp_path / "same.wav"
    diff = tmp_path / "diff.wav"

    sf.write(ref, a, 44100, subtype="PCM_24")
    sf.write(same, b, 44100, subtype="PCM_24")
    sf.write(diff, c, 44100, subtype="PCM_24")

    ok_same, info_same = mod._exact_wav_match(str(ref), str(same))
    ok_diff, info_diff = mod._exact_wav_match(str(ref), str(diff))

    assert ok_same is True
    assert info_same == "match"
    assert ok_diff is False
    assert info_diff == "pcm_mismatch"
