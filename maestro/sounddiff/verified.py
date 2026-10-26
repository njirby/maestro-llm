"""Ablation-verified sound state: what a patch audibly does, checked by re-rendering it with each module switched off.

The building block of every verified-notes mode (first_impression, self_edit, target_vs_current):

    st = verify_state(preset, notes, renderer)    # renders full, full#2 (render-noise floor), one render per ablation
    st.aud        # {"fx:reverb": ("clear", "reverb adding space and a tail"), "mod:lfo_1->filter_1_cutoff": ...}
    st.dims()     # every core dimension in listener words: part, space, movement, pitch, distortion, stereo, tone
    st.flags()    # booleans the checkers test narrations against (dry, steady, narrow, bright, plucky, ...)

A module counts as audible only if switching it off changes the render (log-mel distance) by more than two renders of
the identical patch differ; strength subtle / clear / strong from the excess. Tone, width and note behaviour are
measured on the render and ranked against the reference renders (calib.json, 480 refset patches). Nothing here is read
from raw parameter values except to decide what can be switched off.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from . import vital_facts as vf
from .facts import pct
from .measure import ENV_HOP, SR, db, descriptors, envelope, logmel
from .renderer import clean_notes

FX = ["reverb", "delay", "chorus", "distortion", "phaser", "flanger", "compressor", "eq"]
STRENGTH_ORDER = {"strong": 0, "clear": 1, "subtle": 2}
STRENGTH = {"subtle": "a touch of", "clear": "noticeable", "strong": "lots of"}
NOTE_NAMES = "C C# D D# E F F# G G# A A# B".split()
# sound between notes relative to the notes (dB, gap_fill_db): above CONT the notes run into each other, below SEP the
# gaps are clean silence (probe1 f038: short notes, release filling the gaps, heard as one unbroken line)
CONT_DB, SEP_DB = -12.0, -40.0


def settings_of(p) -> dict:
    return p["settings"] if "settings" in p else p


def ablations(s: dict) -> dict:
    """{name: settings-modifier} for every module that is on / routed in the patch (settings dict)."""
    out = {}
    for fx in FX:
        if s.get(f"{fx}_on", 0) > 0.5:
            out[f"fx:{fx}"] = lambda t, fx=fx: t.__setitem__(f"{fx}_on", 0.0)
    by_src = {}
    for r in vf.live_routes(s):
        if vf.why_dead(s, r["destination"]) is None:
            by_src.setdefault(r["source"], []).append(r)
    for src, rs in by_src.items():
        if src.startswith(("lfo_", "random_", "env_")) and src != "env_1":
            def kill(t, rs=rs):
                for r in rs:
                    t[f"modulation_{r['slot']}_amount"] = 0.0
            out[f"mod:{src}->" + ",".join(sorted({r['destination'] for r in rs}))] = kill
    for n in (1, 2, 3):
        if vf.why_dead(s, f"osc_{n}_wave_frame") is None:
            if s.get(f"osc_{n}_unison_voices", 1) > 1:
                out[f"unison:osc{n}"] = lambda t, n=n: t.__setitem__(f"osc_{n}_unison_voices", 1.0)
            if n > 1 or any(vf.why_dead(s, f"osc_{m}_wave_frame") is None for m in (2, 3)):
                out[f"osc:{n}"] = lambda t, n=n: t.__setitem__(f"osc_{n}_on", 0.0)
    if float(s.get("portamento_time", -10.0)) > -10.0:
        out["glide:portamento"] = lambda t: t.__setitem__("portamento_time", -10.0)
    for f in ("1", "2"):
        if vf.why_dead(s, f"filter_{f}_cutoff") is None:
            out[f"filter:{f}"] = lambda t, f=f: t.__setitem__(f"filter_{f}_on", 0.0)
    return out


def mel_d(a, b) -> float:
    la, lb = logmel(a.mean(1)), logmel(b.mean(1))
    n = min(len(la), len(lb))
    return float(np.abs(la[:n] - lb[:n]).mean())


def strength(x: float) -> str | None:
    return "strong" if x > 6 else "clear" if x > 2 else "subtle" if x > 0.8 else None


def rms_dbfs(y) -> float:
    m = y.mean(1)
    fr = m[: len(m) // 2048 * 2048].reshape(-1, 2048)
    e = (fr ** 2).mean(1)
    act = e > e.max() * 1e-4
    return float(10 * np.log10(e[act].mean() + 1e-20)) if act.any() else -240.0


def describe_module(name: str, d_full: dict, d_abl: dict) -> str:
    """Listener phrase for an audible module, from what changes when it is switched off."""
    kind, _, what = name.partition(":")
    if kind == "fx":
        return {"reverb": "reverb adding space and a tail", "delay": "delay echoes",
                "chorus": "chorus adding width and shimmer", "distortion": "distortion adding grit and bite",
                "phaser": "phaser sweep", "flanger": "flanger sweep", "compressor": "compression evening out the level",
                "eq": "EQ shaping the tone"}[what]
    if kind == "mod":
        src, _, dsts = what.partition("->")
        per_note = src.startswith("env")
        ds = dsts.split(",")
        if any("cutoff" in x for x in ds):
            return "each note's brightness changes as it plays" if per_note else "a cyclic wobble in the brightness"
        if any(x.endswith(("_level", "volume")) for x in ds):
            return "each note's volume swells or dips as it plays" if per_note else "a pulsing in the volume"
        if any(x.endswith(("_tune", "_transpose")) for x in ds):
            return "a pitch bend at the start of notes" if per_note else "a wobble in the pitch (vibrato)"
        if any("wave_frame" in x or "spectral" in x or "distortion_amount" in x for x in ds):
            return "each note's tone colour changes as it plays" if per_note else "the tone colour shifts back and forth"
        return "the effects swell and recede as it plays"
    if kind == "glide":
        return "a pitch glide from one note into the next"
    if kind == "unison":
        return "a slightly detuned, thick quality"
    if kind == "osc":
        return "a second oscillator layer"
    if kind == "filter":
        return "darker with it" if (d_full.get("centroid_w_hz") or 0) < (d_abl.get("centroid_w_hz") or 0) \
            else "brighter with it"
    return name


def note_name(p: int) -> str:
    return NOTE_NAMES[p % 12] + str(p // 12 - 1)


def heard_length(d: dict) -> str | None:
    sr = d.get("sustain_ratio")
    if sr is None:
        return None
    return "notes die away quickly (plucky)" if sr < 0.3 else "notes fade a bit as they hold" if sr < 0.7 else \
        "notes hold steady"


def midi_shape(notes: list[dict]) -> dict:
    """Musical facts from the MIDI alone: shape (single line / chords), register, written length, density."""
    n = notes
    ends = sorted([(x["start_s"], 1) for x in n] + [(x["start_s"] + x["dur_s"], -1) for x in n])
    cur = mx = 0
    for _, d in ends:
        cur += d
        mx = max(mx, cur)
    total = max(x["start_s"] + x["dur_s"] for x in n)
    overlap_t = 0.0
    for i, a in enumerate(n):
        for b in n[i + 1:]:
            overlap_t += max(0.0, min(a["start_s"] + a["dur_s"], b["start_s"] + b["dur_s"]) - max(a["start_s"], b["start_s"]))
    share = overlap_t / total
    shape = ("a single line" if mx == 1 else
             "chords throughout" if share > 0.5 else
             f"chords of up to {mx} notes" if share > 0.15 else "mostly a single line with some overlapping notes")
    med = sorted(x["pitch"] for x in n)[len(n) // 2]
    reg = "low" if med < 52 else "mid" if med < 67 else "high" if med < 80 else "very high"
    avg = sum(x["dur_s"] for x in n) / len(n)
    length = "very short, staccato notes" if avg < 0.15 else "short notes" if avg < 0.35 else \
        "medium-length notes" if avg < 0.8 else "long held notes"
    dens = len(n) / total
    # share of note-to-note transitions written back to back (next note starts < 80 ms after the last one ends: the
    # shortest gap gap_level measures)
    on = sorted(n, key=lambda x: x["start_s"])
    held_to, tight, k = 0.0, 0, 0
    for a, b in zip(on, on[1:]):
        held_to = max(held_to, a["start_s"] + a["dur_s"])
        if b["start_s"] > a["start_s"] + 0.03:
            k += 1
            tight += b["start_s"] - held_to < 0.08
    return {"legato": tight / k if k else 0.0,"shape": shape, "register": reg, "range": f"{note_name(min(x['pitch'] for x in n))}-"
                                                      f"{note_name(max(x['pitch'] for x in n))}",
            "length": length, "busy": "sparse" if dens < 1.5 else "busy" if dens > 4 else None,
            "single": mx == 1, "chords": "chords" in shape}


def gap_level(y, notes) -> float | None:
    """Median over the gaps in the part (>= 80 ms with no note held) of the level just before the next note starts,
    relative to the peak of the note(s) before the gap, in dB. None when the part has fewer than two such gaps."""
    env = envelope(y.mean(1))
    hop = ENV_HOP / SR
    on = sorted(notes, key=lambda n: n["start_s"])
    out, held_to = [], 0.0
    for i, n in enumerate(on[:-1]):
        held_to = max(held_to, n["start_s"] + n["dur_s"])
        nxt = on[i + 1]["start_s"]
        if nxt - held_to < 0.08:
            continue
        a = int(max(held_to - 1.0, on[max(0, i - 3)]["start_s"]) / hop)
        pk = env[a:int(held_to / hop) + 1].max(initial=0.0)
        g0, g1 = int((held_to + 0.6 * (nxt - held_to)) / hop), int((nxt - 0.01) / hop)
        if g1 > g0 and pk > 0:
            out.append(db(np.median(env[g0:g1])) - db(pk))
    return float(np.median(out)) if len(out) >= 2 else None


def continuity(g: float | None) -> str | None:
    """'continuous' when the sound carries on through the gaps between notes, 'separated' when the gaps are silent,
    None in between or when the part has no gaps to measure."""
    if g is None:
        return None
    return "continuous" if g > CONT_DB else "separated" if g < SEP_DB else None


@dataclass
class State:
    """One patch, verified: the full render, its floor, which modules are audible, measured descriptors."""
    settings: dict
    notes: dict
    full: np.ndarray            # normalised (peak 0.5)
    full2: np.ndarray
    raw_peak_db: float
    raw_rms_dbfs: float
    floor: float
    desc: dict
    desc2: dict
    gap_db: float | None = None  # gap_level of the full render
    aud: dict = field(default_factory=dict)  # {ablation: (strength, phrase)}

    # ---- per-dimension facts --------------------------------------------------------------------------------------
    @property
    def silent(self) -> bool:
        return self.raw_peak_db < -100

    def best(self, keys):
        return min((self.aud[k][0] for k in keys), key=STRENGTH_ORDER.get) if keys else None

    def space(self) -> dict:
        """{"reverb": strength|None, "delay": strength|None}"""
        return {fx: (self.aud[f"fx:{fx}"][0] if f"fx:{fx}" in self.aud else None) for fx in ("reverb", "delay")}

    def movement(self) -> dict:
        """{phrase: strength}, one entry per heard effect (several LFOs on one target collapse to the strongest)."""
        out = {}
        for k, (st, ph) in self.aud.items():
            if k.startswith(("mod:", "glide:")):
                pass
            elif k in ("fx:phaser", "fx:flanger"):
                ph = f"a {k[3:]} sweep"
            else:
                continue
            if ph not in out or STRENGTH_ORDER[st] < STRENGTH_ORDER[out[ph]]:
                out[ph] = st
        return out

    def pitch_mod(self) -> bool:
        return any(("tune" in k or "transpose" in k) for k in self.aud if k.startswith("mod:")) or \
            any(k.startswith("glide:") for k in self.aud)

    def distortion(self) -> str | None:
        return self.aud["fx:distortion"][0] if "fx:distortion" in self.aud else None

    def width(self) -> str:
        w = pct("width_db", self.desc["width_db"])
        return "narrow" if w < 25 else "wide" if w > 75 else "medium"

    def brightness(self) -> str:
        b = pct("centroid_w_hz", self.desc["centroid_w_hz"])
        return "dark" if b < 25 else "bright" if b > 75 else "medium"

    def harmonics(self) -> str:
        hr = pct("harm_rich_db", self.desc["harm_rich_db"]) if self.desc.get("harm_rich_db") is not None else 50
        return "pure" if hr < 25 else "rich" if hr > 75 else "moderate"

    def noisy(self) -> bool:
        return pct("flatness", self.desc["flatness"]) > 85

    def thick(self) -> list[str]:
        out = []
        if self.best([k for k in self.aud if k.startswith("osc:")]) == "strong":
            out.append("thick, layered")
        if any(k.startswith("unison") for k in self.aud):
            out.append("slightly detuned")
        if "fx:chorus" in self.aud:
            out.append(f"{STRENGTH[self.aud['fx:chorus'][0]]} chorus shimmer")
        return out

    def rolled_off(self) -> bool:
        return any(k.startswith("filter:") and "darker" in ph for k, (_, ph) in self.aud.items())

    def continuity(self) -> str | None:
        """Measured on the gaps; a part written back to back (no gaps to measure) whose notes hold is continuous
        (probe1 f038: heard as one unbroken line)."""
        c = continuity(self.gap_db)
        if c is None and self.gap_db is None and midi_shape(self.notes["notes"])["legato"] >= 0.8 \
                and self.length() in ("held", "fading"):
            return "continuous"
        return c

    def length(self) -> str | None:
        """'plucky' / 'fading' / 'held' from the render (sustain within notes)."""
        sr = self.desc.get("sustain_ratio")
        return None if sr is None else "plucky" if sr < 0.3 else "fading" if sr < 0.7 else "held"

    # ---- text --------------------------------------------------------------------------------------------------------
    def part_line(self) -> str:
        m = midi_shape(self.notes["notes"])
        hl = heard_length(self.desc)
        length = m["length"] + (f" that {hl.split(' ', 1)[1]}" if hl else "")
        cont = {"continuous": "the notes run into each other (no silence between them: one unbroken line)",
                "separated": "clean silence between the notes"}.get(self.continuity())
        return "; ".join(x for x in [m["shape"], f"{m['register']} register ({m['range']})",
                                     length + (f", {m['busy']}" if m["busy"] else ""), cont] if x)

    def dims(self) -> dict:
        """Every core dimension stated, in listener words (the v4 first-impression wording)."""
        sp = [f"{STRENGTH[st]} {w}" for fx, w in (("reverb", "reverb"), ("delay", "delay echoes"))
              if (st := self.space()[fx])]
        mv = sorted(self.movement().items(), key=lambda kv: STRENGTH_ORDER[kv[1]])

        def mv_phrase(ph, st):
            if ph.startswith("each note's"):
                return ph.replace(" as it plays", {"subtle": " slightly as it plays", "clear": " as it plays",
                                                   "strong": " a lot as it plays"}[st])
            core = ph[2:] if ph.startswith("a ") else ph
            return {"subtle": "a slight ", "clear": "a noticeable ", "strong": "a strong "}[st] + core
        dist = self.distortion()
        harm = {"pure": "pure, smooth (few overtones)", "moderate": "moderately rich",
                "rich": "rich, buzzy overtones" + ("" if dist else " from the sound itself, not distortion")}[self.harmonics()]
        bright = {"dark": "dark", "bright": "bright", "medium": "medium brightness"}[self.brightness()]
        if self.rolled_off() and bright != "bright":
            bright += ", top end rolled off"
        return {
            "part": self.part_line(),
            "space": ", ".join(sp) if sp else "dry: no reverb or delay",
            "movement": ("; ".join(mv_phrase(ph, st) for ph, st in mv) if mv else "steady: no wobble, pulsing or sweeps")
                        + ("" if self.pitch_mod() else ". Pitch is steady"),
            "distortion": f"{STRENGTH[dist]} distortion adding grit and bite" if dist else "no distortion",
            "stereo": {"narrow": "narrow, centered", "wide": "wide stereo", "medium": "medium width"}[self.width()],
            "tone": ", ".join(x for x in [bright, harm, "noisy / hissy" if self.noisy() else None] + self.thick() if x),
        }

    def flags(self) -> dict:
        m = midi_shape(self.notes["notes"])
        return {"dry": not any(self.space().values()), "no_dist": self.distortion() is None,
                "steady": not self.movement(), "pitch_steady": not self.pitch_mod(), "narrow": self.width() == "narrow",
                "wide": self.width() == "wide", "chords": m["chords"], "single": m["single"],
                "bright": self.brightness() == "bright", "dark": self.brightness() == "dark",
                "plucky": self.length() == "plucky", "held": self.length() == "held" and m["length"] in (
                    "medium-length notes", "long held notes"),
                "short_notes": m["length"] in ("very short, staccato notes", "short notes"),
                "continuous": self.continuity() == "continuous", "separated": self.continuity() == "separated",
                "high": m["register"] in ("high", "very high"), "low": m["register"] == "low"}


def render_state_jobs(preset) -> dict:
    """{render name: preset dict} needed to verify one patch: full, a second take (floor), one per ablation."""
    p = preset if "settings" in preset else {"settings": preset}
    jobs = {"full": p, "full#2": p}
    for name, fn in ablations(p["settings"]).items():
        q = copy.deepcopy(p)
        fn(q["settings"])
        jobs[name] = q
    return jobs


def build_state(preset, notes, wavs: dict) -> State:
    """State from rendered jobs (render_state_jobs names -> arrays). Everything is level-normalised by one gain (peak
    0.5 for the full render) so the measurements describe the sound, not its level; the raw level is kept."""
    p = preset if "settings" in preset else {"settings": preset}
    notes = clean_notes(notes)
    full = wavs["full"]
    peak = float(np.abs(full).max())
    raw_peak_db = float(20 * np.log10(peak + 1e-12))
    raw_rms = rms_dbfs(full)
    g = 0.5 / max(peak, 1e-9)
    w = {k: v * g for k, v in wavs.items()}
    n = notes["notes"]
    d = descriptors(w["full"], n)
    st = State(settings=p["settings"], notes=notes, full=w["full"], full2=w["full#2"], raw_peak_db=raw_peak_db,
               raw_rms_dbfs=raw_rms, floor=mel_d(w["full"], w["full#2"]), desc=d, desc2=descriptors(w["full#2"], n),
               gap_db=gap_level(w["full"], n))
    if st.silent:
        return st
    for k, y in w.items():
        if k in ("full", "full#2"):
            continue
        s_ = strength(mel_d(w["full"], y) - st.floor)
        if s_:
            st.aud[k] = (s_, describe_module(k, d, descriptors(y, n)))
    return st


def verify_state(preset, notes, renderer) -> State:
    """Render a patch with each module switched off and measure what is audible (see module docstring)."""
    jobs = render_state_jobs(preset if isinstance(preset, dict) else _load(preset))
    return build_state(jobs["full"], notes, renderer(jobs, notes))


def _load(path):
    import json
    from pathlib import Path
    return json.loads(Path(path).read_text())
