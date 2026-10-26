"""Contradiction checks for narrations against verified notes (clause-level negation, idiom-aware).

    check_state(text, flags)                 # first impression: flags = State.flags() of the target
    check_delta(text, delta)                 # self_edit / target_vs_current: delta = notes["check"] from the builders

A problem is a claim the verified notes rule out. Anything the notes leave open is not flagged. The narration is never
edited: callers regenerate once on a flagged sample (see the narrate_* functions).
"""
from __future__ import annotations

import re

# (flag that must be true, label, claim pattern) - first-impression state checks
STATE_CHECKS = [
    ("dry", "reverb/echo on a dry clip",
     r"\breverb\w*|\becho\w*|\bdelay\w*|\bhall\b|cavern\w*|ambien\w*|reverberat\w*|wash(ed|y)? out"),
    ("no_dist", "distortion when none", r"distort\w*|overdriv\w*|saturat\w*|crunch\w*|fuzz\w*"),
    ("steady", "wobble/pulsing when steady",
     r"wobbl\w*|warbl\w*|tremolo|pulsing|pulses|throb\w*|sweep\w*|undulat\w*"),
    ("pitch_steady", "pitch movement when steady",
     r"vibrato|pitch (bend|shake|wobble|waver|drift|slide)\w*|glid\w*|slid(e|es|ing)\b|portamento|detun\w* pitch"),
    ("narrow", "wide when narrow",
     r"\bwide\b|\bwidth\b|stereo (spread|field|image) (is |feels )?(wide|big|broad)|spread out across"),
    ("wide", "narrow/mono when wide", r"\bnarrow\b|\bmono\b|dead[- ]cent(er|re)"),
    ("chords", "single line on a chord part", r"single (melodic )?line|single melod\w*|one note at a time|monophonic"),
    ("single", "chords on a single line", r"\bchords?\b|\bchordal\b"),
    # brightness (comparatives left to the delta checks; 'brightness' as a noun is not a claim)
    ("bright", "dark on a bright sound", r"\bdark(?!er\b|en)\w*|\bdull\b|muffled|murky"),
    ("dark", "bright on a dark sound", r"\bbright(?!ness|er\b|en)\w*|brilliant|sparkl\w*|sizzl\w*"),
    # note length / continuity
    ("plucky", "sustained/held on a plucky part",
     r"\bsustained\b|long[- ]held|held[- ]out|\bheld notes|notes? (that )?linger\w*|\bdron\w*|notes? ring(s|ing)? out"),
    ("held", "plucky/staccato on held notes",
     r"\bpluck\w*|staccato|percussive|\bstabs?\b|decays? (very )?(quickly|fast)|die\w* away (quickly|fast)"),
    ("separated", "notes running together when separated",
     r"seamless\w*|unbroken|\blegato\b|continuous\w*|blend\w* (into|together)|run (into|together)|flow\w* (into|together)|"
     r"notes? smear\w*|smear\w* (into each other|together)|\bdron\w*"),
    ("continuous", "silence between notes when they run together",
     r"\bchoppy\b|detached|silence between|gaps? between|clean breaks?|clipped off"),
    # genre labels the part rules out
    ("plucky", "'pad' on a plucky part", r"\bpads?\b"),
    ("held", "'pluck' label on held notes", r"\bpluck(s)?\b"),
    ("high", "'bass' on a high part", r"\bbass(line)?s?\b(?![- ](heavy|weight))|\bsub[- ]bass(?![- ]heavy)"),
]
# state checks that still hold when two or three clips are compared (no absolute brightness/length words)
DELTA_STATE_KEYS = ("dry", "no_dist", "steady", "pitch_steady", "chords", "single")

NEG = re.compile(r"\b(no|not|without|none|lack\w*|absent|free of|zero|never|isn.t|aren.t|doesn.t|don.t|didn.t|wasn.t|"
                 r"weren.t|won.t|can.t|couldn.t|hasn.t|haven.t|hardly|barely|rather than|"
                 r"instead of|nothing|minus|no longer|neither|nor|less)\b", re.I)
NON_LATIN = re.compile(r"[Ѐ-ӿ一-鿿぀-ヿ가-힯]")


def clause_before(text: str, i: int) -> str:
    start = max(text.rfind(c, 0, i) for c in ".;:!?—,") + 1
    return text[start:i]


def _basic(text: str) -> list[str]:
    probs = []
    if len(text.strip()) < 60:
        probs.append("too short / empty")
    if NON_LATIN.search(text):
        probs.append("non-English text")
    return probs


def _claimed(pat: str, text: str):
    """Non-negated matches of pat."""
    for m in re.finditer(pat, text, re.I):
        if not NEG.search(clause_before(text, m.start())):
            yield m


def check_state(text: str, flags: dict, keys=None) -> list[str]:
    probs = _basic(text)
    for key, label, pat in STATE_CHECKS:
        if (keys and key not in keys) or not flags.get(key) or label in probs:
            continue
        if next(_claimed(pat, text), None):
            probs.append(label)
    return probs


check = check_state  # first-impression name (first_impression/check.py, fi_notes.py)

# ---------------------------------------------------------------------------------------------------- deltas
# per dimension: (pattern for 'more / up', pattern for 'less / down'); a delta's dir is +1 / -1 / 0 (unchanged)
DIR_CLAIMS = {
    "brightness": (r"\bbrighter\b|more (top end|high end|air|sparkle|brightness|treble)|\bopen(s|ed)? up|crisper",
                   r"\bdarker\b|\bduller\b|more muffled|less (bright|top end|high end|brightness|air)|\bwarmer\b|"
                   r"\bmellower\b|top end (is |got |gets )?(rolled|tamed|cut|softened)"),
    "level": (r"\blouder\b|more volume|(level|volume) (went |goes |is |came )?up\b|(volume|level|it) up a (hair|touch|bit|notch)|"
              r"turned (it )?up|\bhotter\b",
              r"\bquieter\b|softer overall|less volume|lower volume|(level|volume) (dropped|went down|came down)|"
              r"turned (it )?down|(volume|level) down a (hair|touch|bit|notch)"),
    "space": (r"more (reverb|echo\w*|delay|space(?! between)|ambience|room(?! between))|\bwetter\b|\broomier\b|"
              r"(adds?|added|adding|introduc\w*) (a |some |more )?(reverb|echo\w*|delay|space|ambience)",
              r"\bdrier\b|less (reverb|echo\w*|delay|space|ambience|room)|"
              r"(reverb|echo\w*|delay|tail) (is |has )?(gone|been removed|removed|cut)"),
    "width": (r"\bwider\b|more (width|stereo|spread)|\bwiden\w*",
              r"\bnarrower\b|less (width|stereo|spread)|more (centered|centred|mono)"),
    "movement": (r"more (movement|wobble|pulsing|motion|vibrato)|(adds?|added|introduc\w*) (a |some )?"
                 r"(wobble|pulsing|pulse|movement|vibrato|tremolo|sweep)|now (wobbles|pulses|sweeps)",
                 r"\bsteadier\b|less (movement|wobble|pulsing|motion)|(wobble|pulsing|movement|vibrato) (is |has )?"
                 r"(gone|stopped|been removed)|no longer (wobbles|pulses|moves)"),
    "distortion": (r"more (grit|distortion|drive|crunch|dirt|saturation)|\bgrittier\b|\bdirtier\b|\bcrunchier\b|"
                   r"(adds?|added|adding) (some |a bit of )?(grit|distortion|drive|crunch|saturation)",
                   r"less (grit|distortion|drive|crunch)|(distortion|grit) (is |has )?(gone|been removed)"),
    "attack": (r"(slower|softer|gentler) attack|(fades?|swells?) in (more )?slowly",
               r"(snappier|faster|sharper|punchier) attack|\bsnappier\b"),
    "sustain": (r"(hold|sustain)s? longer|more sustain",
                r"die (away )?(faster|quicker|sooner)|less sustain|\bpluckier\b|more plucky"),
    # ring-out after the note is released (the amp envelope's release; reverb / echo claims are 'space')
    "tail": (r"ring(s|ing)? out longer|linger\w* longer|longer (tail|release|ring[- ]out)|trail\w* off longer",
             r"(stop|cut|end)s? (off )?(sooner|quicker|faster|shorter)|shorter (tail|release|ring[- ]out)"),
}
# 'unchanged' claims per dimension: flagged when the dimension measurably moved
SAME_CLAIMS = {
    "level": r"(level|volume|loudness) (is |stays |stayed |remains |remained )?(still |about |basically |pretty much )?"
             r"(the same|unchanged|untouched|where it was)|same (level|volume|loudness)|"
             r"\bstill (pretty |fairly |quite |very |a bit )?(loud|quiet)\b",
    "brightness": r"(brightness|top end) (is |stays |stayed |remains )?(still |basically )?(the same|unchanged|untouched)|"
                  r"same (brightness|top end)|\bstill (pretty |fairly |quite |very )?(bright|dark)\b",
    "space": r"(space|reverb|echo\w*|ambience|room) (is |stays |stayed |remains )?(still |basically )?"
             r"(the same|unchanged|untouched)|same (space|reverb|ambience|room)",
    "movement": r"(movement|wobble|pulsing|motion|modulation) (is |stays |stayed |remains )?(still |basically )?"
                r"(the same|unchanged|untouched|in (exactly )?the same place)|same (movement|wobble|pulsing|motion)",
    "width": r"(width|stereo( image| field| spread)?) (is |stays |stayed |remains )?(still |basically )?"
             r"(the same|unchanged|untouched)|same (width|stereo)",
}
NO_CHANGE = (r"no (audible |noticeable |real |discernible |clear )?(difference|change)|"
             r"sounds? (exactly |basically |essentially |pretty much )?the same|\bidentical\b|"
             r"can.t (really )?hear (a|any) (difference|change)|nothing (really |has )?changed|"
             r"\b(barely|hardly) (any|a) (change|difference)")
# 'everything else sounds the same' is about the other dimensions, not the edit
PARTIAL = re.compile(r"\b(else|otherwise|rest|apart from|other than|aside from|besides|beyond that|except)\b", re.I)
CLOSER = r"\bcloser\b|\bnearer\b|getting there|moving (it )?toward"
FURTHER = (r"\b(further|farther) (away )?from (the |my |our )?(target|it|where|what)|"
           r"(moved|moving|drifted|pushed (it|me|us)) (further |farther )?away from|"
           r"(further|farther) (away|off) (now|than (before|it was))")
MATCHES = r"\b(matches|nailed|spot[- ]on|identical to the target|right on (the )?target|indistinguishable)\b"
# a match claim about one dimension ('the movement is spot on') is not a claim about the whole patch
DIM_WORD = re.compile(r"\b(movement|wobble|brightness|tone|attack|level|volume|space|reverb|echo\w*|noise|width|"
                      r"stereo|decay|release|sustain|notes?|pitch|distortion|grit|character|timing|feel|energy|"
                      r"envelope|filter|texture|low end|top end|high end)\b", re.I)
# goals and plans ('to get closer', 'I need to ... to get it closer') are not trajectory claims
INTENT = re.compile(r"\b(to|need\w*|will|should|next|try\w*|could|would|help|hope|want\w*|going to|gonna)\b", re.I)
TARGET = re.compile(r"\b(target|reference)\b", re.I)
YOURS = re.compile(r"\b(yours|(your|my|our) (patch|sound|version|current|take)|current (patch|sound|version)|mine|"
                   r"ours)\b", re.I)


OBJECT = re.compile(r"\b(than|from|to|like|with|vs\.?|versus|against) (the |my |your |our )?$", re.I)


def _subject(pat, sent: str) -> int:
    """Start of the last mention of pat in sent that can be the subject: comparison objects ('than mine') don't
    count, possessives do ('compared to the target's darker tone')."""
    return max((x.start() for x in pat.finditer(sent)
                if not OBJECT.search(sent[:x.start()]) or sent[x.end():x.end() + 2] in ("'s", "’s")), default=-1)


def _sentence(text: str, i: int) -> tuple[int, str]:
    s = max(text.rfind(c, 0, i) for c in ".!?\n") + 1
    return s, text[s:i]


def dir_claims(text: str, mode: str) -> list[tuple[str, int]]:
    """(dim, +1/-1) for every non-negated direction claim, first per (dim, sign). tvc: oriented as the target relative
    to your patch, from whichever of the two the sentence names last before the claim (not counting comparison objects:
    'darker than mine'); claims with neither named are skipped."""
    out = []
    for dim, pats in DIR_CLAIMS.items():
        for sign, pat in ((1, pats[0]), (-1, pats[1])):
            for m in _claimed(pat, text):
                claim = sign
                if mode == "target_vs_current":
                    _, sent = _sentence(text, m.start())
                    t, y = _subject(TARGET, sent), _subject(YOURS, sent)
                    if t < 0 and y < 0:
                        continue
                    if y > t:
                        claim = -sign
                if (dim, claim) not in out:
                    out.append((dim, claim))
                break
    return out


def hit(text: str, delta: dict) -> bool:
    """Informativeness: the narration states at least one measured change correctly (self_edit: a moved dimension with
    its direction, or 'no difference' on an inaudible edit; tvc: a remaining gap with its direction, or that it
    matches when it does)."""
    if delta["overall"] == "none":
        return bool(re.search(NO_CHANGE if delta["mode"] == "self_edit" else MATCHES + r"|basically (matches|there)",
                              text, re.I))
    return any(delta["dirs"].get(dim) == c for dim, c in dir_claims(text, delta["mode"]))


# conversational filler opening a narration ("Yeah, that got brighter"): a tic, regenerated
OPENER_TIC = re.compile(r"^\W*(yeah|yep|yes|okay|ok|alright|all right|right|so|well|hey|hmm+|oh|sure)\b[,.!:—-]", re.I)


def check_delta(text: str, delta: dict) -> list[str]:
    """delta = {"mode": "self_edit"|"target_vs_current", "overall": none|subtle|clear|strong (the compared pair),
    "dirs": {dim: +1/-1/0} (self_edit: after vs before; tvc: target vs your patch now), "flags": state flags true in
    every clip, "trajectory": closer|same|further (tvc)}"""
    probs = _basic(text)
    if OPENER_TIC.search(text):
        probs.append("interjection opener")
    probs += [f"state: {p}" for p in check_state(text, delta.get("flags", {}), DELTA_STATE_KEYS) if p not in probs]
    tvc = delta["mode"] == "target_vs_current"
    overall = delta["overall"]
    if overall in ("clear", "strong") and not tvc and any(
            not PARTIAL.search(_sentence(text, m.end())[1]) for m in re.finditer(NO_CHANGE, text, re.I)):
        probs.append("says no change on an audible edit")
    if tvc:
        if overall in ("clear", "strong") and any(not DIM_WORD.search(clause_before(text, m.start()))
                                                   for m in _claimed(MATCHES, text)):
            probs.append("says it matches the target when clearly different")
        traj = delta.get("trajectory")
        if traj and traj != "closer" and any(not INTENT.search(clause_before(text, m.start()))
                                             for m in _claimed(CLOSER, text)):
            probs.append(f"says closer, measured {traj}")
        if traj and traj != "further" and any(not INTENT.search(clause_before(text, m.start()))
                                              for m in _claimed(FURTHER, text)):
            probs.append(f"says further, measured {traj}")
    if not tvc and overall != "none":
        for dim, pat in SAME_CLAIMS.items():
            if delta["dirs"].get(dim) and next(_claimed(pat, text), None):
                probs.append(f"says {dim} unchanged, measured {'up' if delta['dirs'][dim] > 0 else 'down'}")
    for dim, claim in dir_claims(text, delta["mode"]):
        if dim not in delta["dirs"]:
            continue
        meas = delta["dirs"][dim]
        if overall == "none" and not tvc:
            lab = f"claims {dim} changed on an inaudible edit"
        elif meas == 0:
            lab = f"claims {dim} {'up' if claim > 0 else 'down'}, measured unchanged"
        elif meas != claim:
            lab = f"claims {dim} {'up' if claim > 0 else 'down'}, measured {'up' if meas > 0 else 'down'}"
        else:
            continue
        if lab not in probs:
            probs.append(lab)
    return probs
