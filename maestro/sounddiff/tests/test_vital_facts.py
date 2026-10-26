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


def test_onset_local_units():
    from maestro.sounddiff import verified as V
    assert V.ONSET_LOCAL
    for u in ("glide:portamento", "env:1", "other:portamento_time", "mod:env_2->osc_1_tune",
              "mod:env_3->osc_2_level,filter_1_cutoff"):
        assert V.onset_local(u), u
    for u in ("mod:lfo_1->osc_1_tune", "mod:env_2->filter_1_cutoff", "fx:reverb", "osc:1", "mod"):
        assert not V.onset_local(u), u


def test_local_excess_sees_onset_change():
    import numpy as np
    from maestro.sounddiff import verified as V
    sr = 44100
    notes = [{"pitch": 60 + 2 * i, "velocity": 100, "start_s": 0.5 * i, "dur_s": 0.45} for i in range(8)]
    t = np.arange(int(4.2 * sr)) / sr
    rng = np.random.default_rng(0)

    def render(glide):
        y = np.zeros_like(t)
        for n in notes:
            f = 440 * 2 ** ((n["pitch"] - 69) / 12)
            m = (t >= n["start_s"]) & (t < n["start_s"] + n["dur_s"])
            tt = t[m] - n["start_s"]
            ff = f * (1 - 0.3 * np.exp(-tt / 0.03)) if glide else f
            y[m] = 0.3 * np.sin(2 * np.pi * np.cumsum(np.broadcast_to(ff, tt.shape)) / sr)
        y = y + 1e-3 * rng.standard_normal(len(t))
        return np.stack([y, y], 1)
    a, a2, b = render(False), render(False), render(True)
    whole = V.mel_d(a, b) - V.mel_d(a, a2)
    local = V.local_excess(a, b, [(a, a2)], notes)
    assert local is not None and local > whole
