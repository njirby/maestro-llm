"""Ablation-verified session notes for the two comparison modes (self_edit, target_vs_current) and their narrators.

    from maestro.sounddiff.renderer import ContainerRenderer
    from maestro.sounddiff.delta import (notes_self_edit, narrate_self_edit,
                                         notes_target_vs_current, narrate_target_vs_current)

    with ContainerRenderer() as r:
        n = notes_self_edit(before_preset, after_preset, notes, renderer=r)
        out = narrate_self_edit(n, url="http://localhost:8001/v1")            # 27B on the notes (default arm)

        n = notes_target_vs_current(target, before, after, notes, renderer=r, next_subsystem="filter")
        out = narrate_target_vs_current(n, url="http://localhost:8001/v1")

notes_self_edit(before, after, notes, *, renderer) -> {text, check, overall, units, audio: {before, after}}
    before / after: preset path or dict (full .vital json or bare settings); notes as in renderer.clean_notes.
    The edit is verified by audio: before (rendered twice = render-noise floor) vs after; every edited unit (one FX
    module, oscillator, envelope, LFO, filter, the mod matrix) is switched back to its 'before' value on its own and
    counts only if that audibly changes the 'after' render; both patches are ablation-verified (verified.verify_state),
    so space / movement / distortion compare what is heard, not parameter values. Every core dimension is stated
    (changed with a size, or unchanged); an inaudible edit is stated as such.
notes_target_vs_current(target, before, after, notes, *, renderer, next_subsystem=None) -> {text, check, overall,
        trajectory, next, audio: {target, before, now}}
    before / after: the agent's patch before and after its last edit. Remaining gaps are measured target vs your patch
    now, per dimension, all stated; trajectory (closer / same / further) from the log-mel distance to the target before
    vs after the edit, beyond the floor. next_subsystem (osc/env/filter/lfo/fx/mod, the GT plan's next edit, e.g.
    items.next_subsystem) is mentioned only if copying the target's settings of that subsystem into your patch
    audibly closes the gap (rendered).
narrate_self_edit(notes, *, url, model=None, arm="text", retries=1) -> {text, attempts, problems, arm}
narrate_target_vs_current(notes, *, url, model=None, arm="text", retries=1) -> {text, attempts, problems, arm}
    notes: the dict from the builder. arm: "text" (notes only; the 27B on :8001, reasoning_effort xhigh; chosen in the
    probe1 head-to-head, outputs/sounddiff/delta/*/score.json), "omni" (labelled clips + notes; Qwen3-Omni-30B-A3B-
    Instruct on :8000; usable when only Omni is up), "audio" (clips only, prompted; baseline).
    Checked with check.check_delta; a flagged sample gets one fresh regenerate (same prompt, never post-edited).
    No sampling params are sent.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path

import numpy as np

from . import vital_facts as vf
from .check import check_delta
from .facts import DIMS, _change
from .omni import audio_parts, chat
from .renderer import clean_notes
from .verified import STRENGTH_ORDER, build_state, mel_d, render_state_jobs, strength

HEADER = "Session notes (verified against the session):"
RANK = {None: 0, "subtle": 1, "clear": 2, "strong": 3}
SIZE = {1: "slightly", 2: "noticeably", 3: "much"}     # descriptor change sizes (facts.DIMS thresholds)
AUD = {"subtle": "subtle: small but audible", "clear": "clearly audible", "strong": "a big, obvious change"}
_DIM = {key: (kind, th, abs_min) for key, _, kind, th, _, abs_min in DIMS}


def _load(p) -> dict:
    if isinstance(p, (str, Path)):
        p = json.loads(Path(p).read_text())
    return p if "settings" in p else {"settings": p}


# ---------------------------------------------------------------------------------------------------- edit units
_UNIT = re.compile(r"(osc|env|lfo|random|filter)_(\d|fx)_")


def unit_of(key: str) -> str | None:
    if key.startswith(("view_", "beats_per_minute")) or key.endswith("_view_2d"):
        return None
    for fx in vf.FX:
        if key.startswith(fx + "_"):
            return f"fx:{fx}"
    if key.startswith("modulation_"):
        return "mod"
    m = _UNIT.match(key)
    return f"{m.group(1)}:{m.group(2)}" if m else f"other:{key}"


def edit_units(a: dict, b: dict) -> dict:
    """{unit: [keys]} of settings that differ a -> b. List-valued state: 'wavetables:<i>', 'modulations', 'lfos'."""
    out = {}
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if k in ("wavetables", "modulations", "lfos", "sample") or va == vb:
            continue
        if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and abs(va - vb) < 1e-9:
            continue
        u = unit_of(k)
        if u:
            out.setdefault(u, []).append(k)
    for i, (wa, wb) in enumerate(zip(a.get("wavetables", []), b.get("wavetables", []))):
        if wa != wb:
            out.setdefault(f"osc:{i + 1}", []).append(f"wavetables:{i}")
    if a.get("modulations") != b.get("modulations"):
        out.setdefault("mod", []).append("modulations")
    if a.get("lfos") != b.get("lfos"):
        out.setdefault("lfo:shapes", []).append("lfos")
    return out


def copy_keys(dst: dict, src: dict, keys) -> None:
    for k in keys:
        if k.startswith("wavetables:"):
            i = int(k.split(":")[1])
            dst["wavetables"][i] = copy.deepcopy(src["wavetables"][i])
        elif k in src:
            dst[k] = copy.deepcopy(src[k])
        else:
            dst.pop(k, None)


def unit_label(u: str) -> str:
    kind, _, what = u.partition(":")
    return {"fx": f"the {what} change", "osc": f"the oscillator {what} change",
            "env": "the amp envelope change" if what == "1" else f"the envelope {what} change",
            "lfo": "the LFO shape change" if what == "shapes" else f"the LFO {what} change",
            "random": f"the random modulator {what} change", "filter": f"the filter {what} change",
            "mod": "the modulation routing change"}.get(kind, f"the {what} change")


# ---------------------------------------------------------------------------------------------------- dimensions
def _desc_dir(key: str, a, b) -> tuple[int | None, int]:
    """(direction, size 1-3) of descriptor `key` from state a to state b vs the facts.DIMS thresholds, required to hold
    across both takes of each (a/a2 x b/b2: the descriptors have their own render noise). direction 0 = confidently
    unchanged, None = too small or inconsistent to call either way."""
    kind, th, abs_min = _DIM[key]
    chs = [_change(kind, x.get(key), y.get(key)) for x in (a.desc, a.desc2) for y in (b.desc, b.desc2)]
    if any(c is None for c in chs):
        return None, 0
    raw = abs((b.desc.get(key) or 0) - (a.desc.get(key) or 0))
    if all(c >= th[0] for c in chs) or all(c <= -th[0] for c in chs):
        if raw < abs_min:
            return None, 0
        return (1 if chs[0] > 0 else -1), sum(min(abs(c) for c in chs) >= t for t in th)
    if all(abs(c) < th[0] / 2 for c in chs):
        return 0, 0
    return None, 0


def _modules_dir(ra: dict, rb: dict, edited=()) -> tuple[int | None, int]:
    """Direction from per-module audibility ranks (0 = not heard .. 3 = strong) in a vs b. A module appearing or
    disappearing at clear+ strength counts, as does a 2-step strength change; a 1-step change (incl. subtle <-> not
    heard) is within what the rest of the patch does to a module's audibility (a longer release masking echoes, a new
    wavetable showing an LFO): too small to call, unless the module itself was edited (`edited`)."""
    sig = set()
    big = 0
    amb = False
    for k in set(ra) | set(rb):
        x, y = ra.get(k, 0), rb.get(k, 0)
        if x == y:
            continue
        if ((x == 0) != (y == 0) and max(x, y) >= 2) or abs(y - x) >= 2 or k in edited:
            sig.add(1 if y > x else -1)
            big = max(big, abs(y - x))
        else:
            amb = True
    if len(sig) == 1:
        return sig.pop(), min(3, big)
    return (None, -1) if len(sig) > 1 else (None, 0) if amb else (0, 0)   # -1: different in kind


def _mv_ranks(s) -> dict:
    return {ph: RANK[v] for ph, v in s.movement().items()}


def dim_moves(a, b, edited=()) -> dict:
    """{dim: (dir, size)} from state a to state b (verified.State), dir in {+1, -1, 0, None}. Dims: brightness, level,
    space, movement, distortion, width, attack (+ slower), sustain (+ holds longer), overtones, noise."""
    out = {}
    for dim, key in (("brightness", "centroid_w_hz"), ("width", "width_db"), ("attack", "attack_ms"),
                     ("sustain", "sustain_ratio"), ("overtones", "harm_rich_db")):
        out[dim] = _desc_dir(key, a, b)
    # flatness moves with tails and gaps as much as with noise: only a change of the noisy / hissy class counts
    out["noise"] = (0, 0) if a.noisy() == b.noisy() else ((1 if b.noisy() else -1), 2)
    lv = b.raw_rms_dbfs - a.raw_rms_dbfs
    out["level"] = ((1 if lv > 0 else -1), 1 + (abs(lv) >= 4) + (abs(lv) >= 8)) if abs(lv) >= 1.5 else \
        (0, 0) if abs(lv) < 0.75 else (None, 0)
    sa, sb = a.space(), b.space()
    out["space"] = _modules_dir({k: RANK[v] for k, v in sa.items()}, {k: RANK[v] for k, v in sb.items()}, edited)
    # the ring-out after notes is its own dimension: with the same audible reverb / delay, a longer tail is the notes'
    # own release (amp envelope), and calling it 'space' made narrators hear echo on dry clips (v5 env rounds)
    out["tail"] = _desc_dir("tail_s", a, b) if out["space"][0] == 0 else (None, 0)
    out["movement"] = _modules_dir(_mv_ranks(a), _mv_ranks(b))
    out["distortion"] = _modules_dir({"distortion": RANK[a.distortion()]}, {"distortion": RANK[b.distortion()]},
                                     edited)
    return out


def _sz(n: int) -> str:
    return SIZE.get(max(1, min(3, n)), "slightly")


def _space_words(s) -> str:
    sp = s.space()
    parts = [f"{ {'subtle': 'a touch of', 'clear': 'noticeable', 'strong': 'lots of'}[v]} {w}"
             for w, v in (("reverb", sp["reverb"]), ("delay echoes", sp["delay"])) if v]
    return ", ".join(parts) if parts else "dry: no reverb or echo"


def _mv_words(s) -> str:
    mv = sorted(s.movement().items(), key=lambda kv: STRENGTH_ORDER[kv[1]])
    return "; ".join(f"{ {'subtle': 'slight', 'clear': 'noticeable', 'strong': 'strong'}[v]}: {ph}" for ph, v in mv) \
        if mv else "steady"


def _len_word(s) -> str:
    return {"plucky": "plucky, notes die away quickly", "fading": "notes fade a bit as they hold",
            "held": "notes hold steady", None: "note length unclear"}[s.length()]


def _flags_both(*states) -> dict:
    fl = [s.flags() for s in states]
    return {k: all(f[k] for f in fl) for k in fl[0]}


# ---------------------------------------------------------------------------------------------------- renders
def _render_states(renderer, groups: dict, notes, extra: dict | None = None):
    """Render several patches' verification jobs (+ extra single renders) in one batch; returns (states, raw wavs)."""
    jobs = {}
    for g, p in groups.items():
        for k, q in render_state_jobs(p).items():
            jobs[f"{g}/{k}"] = q
    for k, q in (extra or {}).items():
        jobs[f"x/{k}"] = q
    w = renderer(jobs, notes)
    states = {g: build_state(groups[g], notes, {k.split("/", 1)[1]: v for k, v in w.items() if k.startswith(g + "/")})
              for g in groups}
    return states, w


def _overall(x: float) -> str | None:
    return strength(x)


def _d(w: dict, xs, ys) -> float:
    """Mean log-mel distance over all take pairs (renders of patches with random phase / noise differ take to take;
    averaging over takes keeps a real change from drowning in one unlucky pair)."""
    return float(np.mean([mel_d(w[x], w[y]) for x in xs for y in ys]))


def _takes(g: str) -> tuple[str, str]:
    return f"{g}/full", f"{g}/full#2"


# ---------------------------------------------------------------------------------------------------- self_edit
def notes_self_edit(before, after, notes, *, renderer) -> dict:
    a, b = _load(before), _load(after)
    notes = clean_notes(notes)
    units = edit_units(a["settings"], b["settings"])
    extra = {}
    if len(units) > 1:
        for u, keys in units.items():
            q = copy.deepcopy(b)
            copy_keys(q["settings"], a["settings"], keys)
            extra[u] = q
    st, w = _render_states(renderer, {"A": a, "B": b}, notes, extra)
    A, B = st["A"], st["B"]
    floor = (_d(w, ["A/full"], ["A/full#2"]) + _d(w, ["B/full"], ["B/full#2"])) / 2
    overall = _overall(_d(w, _takes("A"), _takes("B")) - floor)
    unit_aud = {u: _overall(_d(w, _takes("B"), [f"x/{u}"]) - floor) for u in extra}
    if len(units) == 1:
        unit_aud = {next(iter(units)): overall}
    silent = A.silent or B.silent
    # a one-step audibility change of an FX module counts when that module was edited (and audibly so)
    edited = [u[3:] for u, v in unit_aud.items() if u.startswith("fx:") and v]
    moves = dim_moves(A, B, edited) if overall and not silent else {}
    L = [HEADER]
    if silent:
        L.append("- " + ("Both renders are silent: nothing can be heard." if A.silent and B.silent else
                         "The render is silent after the edit: the sound is gone." if B.silent else
                         "The render was silent before the edit; now it makes sound."))
    elif not overall:
        L.append("- Overall: no audible difference: before and after differ no more than two takes of the same patch.")
        L.append("- Every dimension (brightness, level, space, movement, distortion, stereo, note shape, tone): "
                 "unchanged, same as before.")
    else:
        L.append(f"- Overall: the edit is {AUD[overall]}.")
        if len(unit_aud) > 1:
            heard = [f"{unit_label(u)} ({AUD[v].split(':')[0]})" for u, v in
                     sorted(unit_aud.items(), key=lambda kv: -RANK[kv[1]]) if v]
            mute = [unit_label(u) for u, v in unit_aud.items() if not v]
            L.append("- What made the difference: " + ("; ".join(heard) if heard else
                                                     "no single part of the edit on its own, only all of it together")
                     + (f". Inaudible on its own: {', '.join(mute)}" if mute else "") + ".")
        L += _move_lines(moves, A, B)
    dirs = {d: v[0] for d, v in moves.items() if v[0] is not None and d in
            ("brightness", "level", "space", "width", "movement", "distortion", "attack", "sustain", "tail")}
    if not overall and not silent:
        dirs = {d: 0 for d in ("brightness", "level", "space", "width", "movement", "distortion", "attack", "sustain", "tail")}
    check = {"mode": "self_edit", "overall": overall or "none", "dirs": dirs,
             "flags": {} if silent else _flags_both(A, B)}
    path = getattr(renderer, "path_of", None)
    return {"text": "\n".join(L), "check": check, "overall": overall or "none", "units": unit_aud, "moves": moves,
            "floor": floor, "audio": {"before": str(path(a, notes)), "after": str(path(b, notes))} if path else None}


# per dimension: (label, more/up, less/down) for self_edit (after vs before) and tvc (the target vs your patch);
# {s} takes the size word
LINES = {
    "self_edit": {
        "brightness": ("Brightness", "{s} brighter", "{s} darker"),
        "level": ("Level", "{s} louder", "{s} quieter"),
        "space": ("Space", "{s} more reverb/echo", "{s} drier, less reverb/echo"),
        "movement": ("Movement", "{s} more movement", "{s} steadier, less movement"),
        "distortion": ("Distortion", "{s} more distortion, grittier", "{s} less distortion, cleaner"),
        "width": ("Stereo", "{s} wider", "{s} narrower"),
        "attack": ("Attack", "{s} slower, softer", "{s} snappier"),
        "sustain": ("Note length", "notes hold {s} longer", "notes die away {s} faster"),
        "tail": ("Ring-out after each note", "{s} longer: the notes themselves ring out longer after they're released "
                 "(the note's own release, not reverb or echo; the space is unchanged)",
                 "{s} shorter: the notes themselves stop sooner after they're released (the note's own release, not "
                 "less reverb or echo; the space is unchanged)"),
        "overtones": ("Overtones", "{s} more, buzzier", "{s} fewer, smoother"),
        "noise": ("Noise", "noisier, hissier", "cleaner, less hiss"),
    },
    "target_vs_current": {
        "brightness": ("Brightness", "the target is {s} brighter", "the target is {s} darker"),
        "level": ("Level", "the target is {s} louder", "the target is {s} quieter"),
        "space": ("Space", "the target has {s} more reverb/echo", "the target is {s} drier"),
        "movement": ("Movement", "the target moves {s} more", "the target is {s} steadier"),
        "distortion": ("Distortion", "the target is {s} more distorted", "the target is {s} cleaner"),
        "width": ("Stereo", "the target is {s} wider", "the target is {s} narrower"),
        "attack": ("Attack", "the target's is {s} slower, softer", "the target's is {s} snappier"),
        "sustain": ("Note length", "the target's notes hold {s} longer", "the target's notes die away {s} faster"),
        "tail": ("Ring-out after each note", "the target's notes themselves ring out {s} longer after they're "
                 "released (the note's own release, not reverb or echo)", "the target's notes themselves stop {s} "
                 "sooner after they're released (the note's own release, not reverb or echo)"),
        "overtones": ("Overtones", "the target has {s} more, buzzier", "the target has {s} fewer, smoother"),
        "noise": ("Noise", "the target is noisier", "the target is cleaner"),
    },
}
STATE_WORDS = {"space": _space_words, "movement": _mv_words,
               "distortion": lambda s: "distortion" if s.distortion() else "no distortion"}


def _move_lines(m: dict, A, B, *, tvc=False) -> list[str]:
    """One line per core dimension. self_edit: A = before, B = after. tvc: A = your patch now, B = the target
    (directions are the target relative to yours)."""
    mode = "target_vs_current" if tvc else "self_edit"
    words = dict(STATE_WORDS, sustain=lambda s: _len_word(s) if s.length() else None)
    L = []
    for dim, (label, up, down) in LINES[mode].items():
        d, n = m.get(dim, (None, 0))
        if dim == "tail" and d is None:  # covered by the space line (the reverb / delay changed), or too small
            continue
        wa, wb = (words[dim](A), words[dim](B)) if dim in words else (None, None)
        # unchanged / too-small lines carry the shared state only as 'same as before' / 'in both', so a description
        # of the state can't read as a change (v5: '(now: notes fade a bit as they hold)' narrated as 'now fade more')
        if d is None and n != -1 or d == 0:
            txt = ("unchanged" if d == 0 else "unchanged or too small to call") if not tvc else \
                  ("same as yours" if d == 0 else "about the same as yours (too small to call)")
            if wa and wa == wb:
                txt += f" (in both: {wb})" if tvc else f" ({wb}, same as before)"
        else:
            txt = "different in kind" if d is None else (up if d > 0 else down).replace("{s}", _sz(n)).replace("  ", " ")
            if wa and wb and wa != wb:
                txt += f" (target: {wb}; yours: {wa})" if tvc else f" (before: {wa}; now: {wb})"
            elif wb:
                txt += f" ({'target' if tvc else 'now'}: {wb})"
        L.append(f"- {label}: {txt}.")
    if all(m.get(k, (0,))[0] in (0, None) for k in LINES[mode]):
        L.append("- None of these moved clearly: the difference is small and in the overall tone colour.")
    return L


# ---------------------------------------------------------------------------------------------------- target_vs_current
SUB_PREFIX = {"osc": ("osc", "unison"), "env": ("env",), "filter": ("filter",), "lfo": ("lfo", "random"),
              "fx": ("fx",), "mod": ("mod",)}
SUB_LABEL = {"osc": "the oscillators", "env": "the envelopes (note shape)", "filter": "the filter",
             "lfo": "the LFOs", "fx": "the effects", "mod": "the modulation routing"}


def notes_target_vs_current(target, before, after, notes, *, renderer, next_subsystem: str | None = None) -> dict:
    t, c0, c1 = _load(target), _load(before), _load(after)
    notes = clean_notes(notes)
    extra = {"c0": c0}
    if next_subsystem:
        q = copy.deepcopy(c1)
        keys = [k for u, ks in edit_units(c1["settings"], t["settings"]).items()
                if u.split(":")[0] in SUB_PREFIX[next_subsystem] for k in ks]
        if keys:
            copy_keys(q["settings"], t["settings"], keys)
            extra["fix"] = q
    st, w = _render_states(renderer, {"T": t, "C": c1}, notes, extra)
    T, C = st["T"], st["C"]
    floor = (_d(w, ["T/full"], ["T/full#2"]) + _d(w, ["C/full"], ["C/full#2"])) / 2
    d0, d1 = _d(w, ["x/c0"], _takes("T")), _d(w, _takes("C"), _takes("T"))
    overall = _overall(d1 - floor)
    edit_aud = _overall(_d(w, ["x/c0"], _takes("C")) - floor)
    margin = max(0.8, floor)
    traj = "closer" if d0 - d1 > margin and d1 < 0.9 * d0 else \
        "further" if d1 - d0 > margin and d1 > 1.1 * d0 else "same"
    nxt = None
    if "fix" in extra and overall:
        df = _d(w, ["x/fix"], _takes("T"))
        if d1 - df > margin:
            nxt = next_subsystem
    silent = T.silent or C.silent
    moves = dim_moves(C, T) if overall and not silent else {}
    L = [HEADER]
    if silent:
        L.append("- " + ("The target render is silent." if T.silent else
                         "Your patch is silent now, the target is not: nothing of the target can be heard yet."))
    else:
        L.append("- Your last edit: " + (f"{AUD[edit_aud]}" if edit_aud else
                                        "no audible difference (before and after sound the same)") + ".")
        L.append("- Did it get you closer? " + {
            "closer": "yes: your patch is now clearly closer to the target than before the edit",
            "further": "no: your patch is now further from the target than before the edit",
            "same": "no real change: about the same distance from the target as before the edit"}[traj] + ".")
        L.append("- Your patch now vs the target: " + (
            {"subtle": "close, only subtly different", "clear": "clearly different",
             "strong": "very different"}[overall] if overall else
            "basically matches (no bigger difference than two takes of the target)") + ".")
        if overall:
            L += _move_lines(moves, C, T, tvc=True)
            if nxt:
                L.append(f"- Next: {SUB_LABEL[nxt]}: setting them like the target's audibly closes part of the gap.")
    dirs = {d: v[0] for d, v in moves.items() if v[0] is not None and d in
            ("brightness", "level", "space", "width", "movement", "distortion", "attack", "sustain", "tail")}
    check = {"mode": "target_vs_current", "overall": overall or "none", "dirs": dirs, "trajectory": traj,
             "flags": {} if silent else _flags_both(T, C)}
    path = getattr(renderer, "path_of", None)
    return {"text": "\n".join(L), "check": check, "overall": overall or "none", "trajectory": traj,
            "edit_audibility": edit_aud or "none", "next": nxt, "moves": moves, "floor": floor,
            "dist": {"before": d0, "after": d1},
            "audio": {"target": str(path(t, notes)), "before": str(path(c0, notes)), "now": str(path(c1, notes))}
            if path else None}


# ---------------------------------------------------------------------------------------------------- narrators
OPENER = "Start with the sound itself, not with an interjection or filler word."
STYLE = ("Keep subtle things subtle, don't claim anything the notes rule out, and don't read the notes back as a list. "
         "No numbers, parameter names or synth-engineering jargon. " + OPENER + " Reply with just what you'd say, in "
         "English.")
SE_TASK = ("say what you hear changed, the way a producer would to a collaborator right after making the edit, in 1-3 "
           "sentences, in your own words: lead with what changed most, and if the edit made no audible difference, "
           "say so.")
TVC_TASK = ("think out loud the way a producer would while recreating a sound, in 1-3 sentences, in your own words: did "
            "the last edit get you closer, and what's the biggest remaining difference from the target? If the notes "
            "name a next move, say that's what you'll go after next; if they don't, don't propose one. If it basically "
            "matches, say so.")
PROMPTS = {
    ("self_edit", "omni"): "Clip 1 is your synth patch before your last edit, clip 2 is after it; same part both "
                           "times. {notes}\n\nThe notes are accurate. Using them, " + SE_TASK + " " + STYLE,
    ("self_edit", "text"): "You just made an edit to your synth patch and played the part again to compare. {notes}"
                           "\n\nThe notes are accurate. Using them, " + SE_TASK + " " + STYLE,
    ("self_edit", "audio"): "Clip 1 is your synth patch before your last edit, clip 2 is after it; same part both "
                            "times. Listen carefully and " + SE_TASK + " Don't claim changes you can't hear. "
                            "No numbers, parameter names or synth-engineering jargon. " + OPENER + " Reply with just what you'd say, "
                            "in English.",
    ("target_vs_current", "omni"): "Clip 1 is the target sound you're recreating. Clip 2 is your synth patch before "
                                   "your last edit, clip 3 is your patch now; same part in all three. {notes}\n\nThe "
                                   "notes are accurate. Using them, " + TVC_TASK + " " + STYLE,
    ("target_vs_current", "text"): "You're recreating a target synth sound and just made an edit to your patch, then "
                                   "A/B'd it against the target. {notes}\n\nThe notes are accurate. Using them, "
                                   + TVC_TASK + " " + STYLE,
    ("target_vs_current", "audio"): "Clip 1 is the target sound you're recreating. Clip 2 is your synth patch before "
                                    "your last edit, clip 3 is your patch now; same part in all three. Listen carefully "
                                    "and " + TVC_TASK + " Don't claim differences you can't hear. No numbers, parameter "
                                    "names or synth-engineering jargon. " + OPENER + " Reply with just what you'd say, in English.",
}
CLIPS = {"self_edit": [("Clip 1 (before):", "before"), ("Clip 2 (after):", "after")],
         "target_vs_current": [("Clip 1 (target):", "target"), ("Clip 2 (your patch before the edit):", "before"),
                               ("Clip 3 (your patch now):", "now")]}
TEXT_EXTRA = {"chat_template_kwargs": {"reasoning_effort": "xhigh"}, "max_tokens": 16000}


def build_content(mode: str, n: dict, arm: str):
    prompt = PROMPTS[(mode, arm)].format(notes=n.get("text", ""))
    if arm == "text":
        return prompt
    labels = CLIPS[mode]
    aud = audio_parts([n["audio"][k] for _, k in labels])   # one gain: level differences stay audible
    parts = []
    for (lab, _), a in zip(labels, aud):
        parts += [{"type": "text", "text": lab}, a]
    return parts + [{"type": "text", "text": prompt}]


def _narrate(mode: str, n: dict, *, url, model=None, arm="omni", retries=1) -> dict:
    content = build_content(mode, n, arm)
    extra = TEXT_EXTRA if arm == "text" else None
    attempts = []
    for _ in range(1 + retries):
        text, rsn = chat(url, content, model=model, extra=extra)
        probs = check_delta(text, n["check"])
        attempts.append({"text": text, "problems": probs, "reasoning_chars": len(rsn)})
        if not probs:
            break
    return {"text": attempts[-1]["text"], "attempts": attempts, "problems": attempts[-1]["problems"], "arm": arm}


def narrate_self_edit(notes: dict, *, url, model=None, arm="text", retries=1) -> dict:
    return _narrate("self_edit", notes, url=url, model=model, arm=arm, retries=retries)


def narrate_target_vs_current(notes: dict, *, url, model=None, arm="text", retries=1) -> dict:
    return _narrate("target_vs_current", notes, url=url, model=model, arm=arm, retries=retries)
