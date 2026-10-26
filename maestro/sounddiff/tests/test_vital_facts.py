from maestro.sounddiff import vital_facts as vf
from maestro.sounddiff.delta import edit_units


def test_bpm_is_host_owned():
    s = {"beats_per_minute": 2.0, "osc_1_on": 1.0, "osc_1_level": 0.7}
    assert vf.why_dead(s, "beats_per_minute") == "tempo comes from the host"
    t = dict(s, beats_per_minute=2.5)
    assert vf.diff(s, t) == []          # a tempo-only change is no edit
    assert edit_units(s, t) == {}       # and nothing to ablate
