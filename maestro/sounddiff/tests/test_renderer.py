from maestro.sounddiff.renderer import _key, clean_notes

NOTES = clean_notes({"bpm": 120, "notes": [{"pitch": 60, "velocity": 100, "start_s": 0.0, "dur_s": 0.5}]})


def test_cache_key_is_settings_only():
    s = {"osc_1_on": 1.0, "volume": 5000.0}
    full = {"preset_name": "x", "author": "y", "comments": "z", "settings": s}
    assert _key(full, NOTES, "") == _key({"settings": s}, NOTES, "") == _key(s, NOTES, "")
    assert _key(full, NOTES, "2") != _key(full, NOTES, "")
    assert _key({"settings": dict(s, volume=5001.0)}, NOTES, "") != _key(full, NOTES, "")
