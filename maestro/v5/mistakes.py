"""Code mistakes on skill-CLI calls: a broken variant of a planned edit, run for real, then diagnosed and redone.

  m = code_mistake(edit, rng)      -> CodeMistake(kind, cmd, diagnosis) | None
The caller runs m.cmd in the session (the error text in the record is the CLI's own), checks that it failed without
touching preset.json (set.py / wavetable.py never half-write; a mod.py line that fails changes nothing), emits
m.diagnosis as assistant text, then runs the planned command. Kinds:

  bad_key       a typo'd or wrong-index parameter key            set.py: "unknown parameter ...; did you mean"
  out_of_range  a value outside the key's range                  set.py: "... out of range lo..hi"
  bad_label     a misspelled enum label                          set.py: "cannot parse value ...; labels: ..."
  mod_range     a mod amount past ±100%                         mod.py: "must be non-zero and within -1..1"
  bad_source    a mod source spelled without the underscore      mod.py: unknown source
  bad_osc       wavetable.py --osc 4                              argparse: invalid choice
"""
from __future__ import annotations

import random
import re
import shlex
from dataclasses import dataclass

from . import planner as pl


@dataclass
class CodeMistake:
    kind: str
    cmd: str
    diagnosis: str


def _join(argv: list[str]) -> str:
    """shlex.join that leaves the $VITAL_SKILL script path unquoted (it must expand)."""
    return " ".join(a if a.startswith("$VITAL_SKILL/") else shlex.quote(a) for a in argv)


def _typo(key: str, rng: random.Random) -> str | None:
    parts = key.split("_")
    words = [i for i, w in enumerate(parts) if len(w) >= 4 and not w.isdigit()]
    if not words:
        return None
    i = rng.choice(words)
    w = parts[i]
    j = rng.randrange(1, len(w))
    parts[i] = w[:j] + w[j + 1:] if rng.random() < 0.6 else w[:j - 1] + w[j] + w[j - 1] + w[j + 1:]
    bad = "_".join(parts)
    return None if bad == key or bad in pl.params() else bad


def _set_mistake(edit: pl.Edit, rng: random.Random) -> CodeMistake | None:
    argv = shlex.split(edit.cmd)
    pairs = [(i, a.split("=", 1)) for i, a in enumerate(argv) if "=" in a]
    if not pairs:
        return None
    kinds = ["bad_key", "out_of_range", "bad_label"]
    rng.shuffle(kinds)
    for kind in kinds:
        i, (key, val) = rng.choice(pairs)
        p = pl.params()[key]
        if kind == "bad_key":
            bad = _typo(key, rng)
            if bad:
                argv[i] = f"{bad}={val}"
                return CodeMistake(kind, _join(argv),
                                   f"`{bad}` isn't a key, it's `{key}`. set.py wrote nothing; running it again "
                                   f"with the right key.")
        elif kind == "out_of_range" and not p["discrete"] and "labels" not in p:
            span = p["max"] - p["min"]
            wrong = p["max"] + max(span * 0.5, 1.0)
            argv[i] = f"{key}={pl._num(wrong, 3)}"
            return CodeMistake(kind, _join(argv),
                               f"{key} only goes {pl._num(p['min'], 3)}..{pl._num(p['max'], 3)}; I meant {val}. "
                               f"Nothing was written, so I'll rerun the whole set with that value.")
        elif kind == "bad_label" and "labels" in p and val in p["labels"] and len(val) >= 5 and val.isalpha():
            j = rng.randrange(1, len(val) - 1)
            bad = val[:j] + val[j + 1:]
            argv[i] = f"{key}={bad}"
            return CodeMistake(kind, _join(argv),
                               f"The label is `{val}`, not `{bad}`. Rerunning the set with the right label.")
    return None


def _mod_mistake(edit: pl.Edit, rng: random.Random) -> CodeMistake | None:
    lines = edit.cmd.split("\n")
    li = rng.randrange(len(lines))
    argv = shlex.split(lines[li])
    try:
        k = argv.index("add")
    except ValueError:
        return None
    src, dst, amt = argv[k + 1], argv[k + 2], argv[k + 3]
    if amt.endswith("%") and rng.random() < 0.5:
        pct = float(amt[:-1])
        argv[k + 3] = f"{pl._num(pct + (100 if pct > 0 else -100), 1)}%"
        lines[li] = _join(argv)
        return CodeMistake("mod_range", "\n".join(lines),
                           f"Amounts stop at ±100%; that route should be {amt}. The other routes went in, and add "
                           f"only updates an existing route, so running the whole block again is safe.")
    if re.fullmatch(r"(lfo|env|random)_\d", src):
        argv[k + 1] = src.replace("_", "")
        lines[li] = _join(argv)
        return CodeMistake("bad_source", "\n".join(lines),
                           f"The source is `{src}`, with the underscore. add only updates existing routes, so "
                           f"running the whole block again is safe.")
    return None


def _wavetable_mistake(edit: pl.Edit, rng: random.Random) -> CodeMistake | None:
    m = re.search(r"--osc (\d)", edit.cmd)
    if not m:
        return None
    return CodeMistake("bad_osc", edit.cmd.replace(m.group(0), "--osc 4", 1),
                       f"Vital has oscillators 1-3; this table goes on osc {m.group(1)}.")


def code_mistake(edit: pl.Edit, rng: random.Random) -> CodeMistake | None:
    if edit.tool == "set":
        return _set_mistake(edit, rng)
    if edit.tool == "mod":
        return _mod_mistake(edit, rng)
    if edit.tool == "wavetable":
        return _wavetable_mistake(edit, rng)
    return None
