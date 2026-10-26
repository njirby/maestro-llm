from maestro.sounddiff import vital_facts as vf
from maestro.sounddiff.delta import edit_units


def test_bpm_is_host_owned():
    s = {"beats_per_minute": 2.0, "osc_1_on": 1.0, "osc_1_level": 0.7}
    assert vf.why_dead(s, "beats_per_minute") == "tempo comes from the host"
    t = dict(s, beats_per_minute=2.5)
    assert vf.diff(s, t) == []          # a tempo-only change is no edit
    assert edit_units(s, t) == {}       # and nothing to ablate


def test_portamento_follows_time_not_on():  # rendered: pilot20 s0008 (portamento_on 0, time -6.1, overlapping notes)
    s = {"portamento_on": 0.0, "portamento_time": -6.09, "portamento_slope": 0.0, "portamento_force": 0.0}
    assert vf.why_dead(s, "portamento_time") is None
    assert vf.why_dead(s, "portamento_slope") is None
    assert vf.why_dead(s, "portamento_on") is not None
    off = dict(s, portamento_time=-10.0)
    assert vf.why_dead(off, "portamento_slope") is not None
    assert vf.why_dead(off, "portamento_time") is None      # moving it off the minimum is an audible edit
    assert [d["param"] for d in vf.diff(off, s) if d["live"]] == ["portamento_time"]


def test_glide_is_ablated():
    from maestro.sounddiff.verified import ablations
    assert "glide:portamento" in ablations({"portamento_time": -6.09})
    assert "glide:portamento" not in ablations({"portamento_time": -10.0})
