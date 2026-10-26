"""Contradiction check for a first impression against verified facts (clause-level negation, idiom-aware)."""
import re

CHECKS = [  # (fact key that must be true, label, claim pattern)
    ("dry", "reverb/echo on a dry clip", r"\breverb\w*|\becho\w*|\bdelay\w*|\bhall\b|cavern\w*|ambien\w*|reverberat\w*|wash(ed|y)? out"),
    ("no_dist", "distortion when none", r"distort\w*|overdriv\w*|saturat\w*|crunch\w*|fuzz\w*"),
    ("steady", "wobble/pulsing when steady", r"wobbl\w*|warbl\w*|tremolo|pulsing|pulses|throb\w*|sweep\w*|undulat\w*"),
    ("pitch_steady", "pitch movement when steady", r"vibrato|pitch (bend|shake|wobble|waver|drift|slide)\w*|glid\w*|slid(e|es|ing)\b|portamento|detun\w* pitch"),
    ("narrow", "wide when narrow", r"\bwide\b|\bwidth\b|stereo (spread|field|image) (is |feels )?(wide|big|broad)|spread out across"),
    ("chords", "single line on a chord part", r"single (melodic )?line|single melod\w*|one note at a time|monophonic"),
    ("single", "chords on a single line", r"\bchords?\b|\bchordal\b"),
]
NEG = re.compile(r"\b(no|not|without|none|lack\w*|absent|free of|zero|never|isn.t|aren.t|doesn.t|don.t|rather than|instead of|nothing|minus)\b", re.I)
NON_LATIN = re.compile(r"[Ѐ-ӿ一-鿿぀-ヿ가-힯]")


def clause_before(text, i):
    start = max(text.rfind(c, 0, i) for c in ".;:!?—,") + 1
    return text[start:i]


def check(text, facts):
    probs = []
    if len(text.strip()) < 60:
        probs.append("too short / empty")
    if NON_LATIN.search(text):
        probs.append("non-English text")
    for key, label, pat in CHECKS:
        if not facts.get(key):
            continue
        for m in re.finditer(pat, text, re.I):
            if NEG.search(clause_before(text, m.start())):
                continue
            probs.append(label)
            break
    return probs
