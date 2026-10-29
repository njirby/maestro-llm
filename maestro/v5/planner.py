"""GT diff -> the main agent's edit plan in skill-CLI form.

  plan = build_plan(gt_preset, init_preset, rng)
  plan.wavetables   [(osc, library entry)]       wavetable.py set per active oscillator (GT tables, oracle)
  plan.rounds       [Round]                      one subsystem per round, in the skill's order (rule 3):
                    oscillator -> envelope -> filter -> modulation -> fx -> global

Changed keys come from v3's build_batches_from_diff (scripts/build_main_agent_sft_v3.py), restricted to keys the skill
knows (data/params.json) that can affect the sound in the GT patch (vitallib.why_dead is None), and re-bucketed into the
skill's sections. Mod routes are mod.py add/flags (from the GT matrix's live routes, not the modulation_N_* scalars);
LFO shapes, which have no script, are a small JSON edit (SKILL.md: "edited in the JSON").

Param mistakes come from v3's inject_mistakes on normalized values (its spurious pool replaced by params.json keys of
the same section that the GT leaves at init); each mistaken round gets a correction edit with the GT values.
"""
from __future__ import annotations

import json
import math
import re
import shlex
import sys
import threading
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from scripts import build_main_agent_sft_v3 as v3
from maestro.synth.path_gen import _denormalize, _normalize

SKILL_DIR = Path(__file__).resolve().parents[2] / "opencode" / "skills" / "vital"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
import vitallib  # noqa: E402  (the skill's own library: params, liveness, wavetable index)

S = "$VITAL_SKILL/scripts"
# Family order (SKILL.md rule 3): static tone before its shape in time. Oscillators, then the filter (brightness), then
# the amp envelope (note shape), then modulation (movement), then effects, then global (level last). Syntorial and
# practitioner consensus; see docs/wavetable_search_heuristic.md's sibling docs/tuning_order.md.
ORDER = ("oscillator", "filter", "envelope", "modulation", "fx", "global")
FX_UNITS = tuple(vitallib.FX)
# effects in tuning order: those that rewrite the spectrum first, the ones that smear detail (delay, reverb) last
FX_TUNE = ("distortion", "eq", "compressor", "chorus", "flanger", "phaser", "delay", "reverb")
_GATES = re.compile(r"(_on|_type|_model|_style|_sync|_destination|_unison_voices)$")

# Within-family order (coarse, high-impact, discrete first; weakly identifiable fine-tuning last), first match wins.
# Ties within a rank go by measured audible impact (the key's one-key revert distance), biggest first.
KEY_ORDER = [
    (r"_on$", 0),
    # oscillators: pitch locked first, then balance, thickness, shaping; placement last
    (r"^osc_\d_(transpose|tune)$", 1), (r"^osc_\d_level$", 2), (r"^osc_\d_unison_voices$", 3),
    (r"^osc_\d_unison_detune$", 4), (r"^osc_\d_(unison_blend|stereo_spread|detune_range)$", 5),
    (r"^osc_\d_distortion_type$", 6), (r"^osc_\d_distortion_", 7), (r"^osc_\d_spectral_morph_type$", 8),
    (r"^osc_\d_spectral_morph_", 9), (r"^osc_\d_", 10),
    # filters: mode before amount, cutoff, resonance, drive, then the rest
    (r"^filter_\w+_(model|style)$", 1), (r"^filter_\w+_cutoff$", 2), (r"^filter_\w+_resonance$", 3),
    (r"^filter_\w+_drive$", 4), (r"^filter_\w+_keytrack$", 5), (r"^filter_", 6),
    # envelopes: what you hear at note start and end, then while held; curves last
    (r"^env_\d_attack$", 1), (r"^env_\d_release$", 2), (r"^env_\d_sustain$", 3), (r"^env_\d_decay$", 4),
    (r"^env_\d_(delay|hold)$", 5), (r"^env_\d_", 6),
    # LFOs / random: rate first, then everything else
    (r"^(lfo|random)_\d_(sync|sync_type|tempo|frequency)$", 1), (r"^(lfo|random)_\d_", 2),
    # effects: mode, mix, character, then their filters / tempo
    (r"^\w+_(type|model|style|mode)$", 1), (r"_(dry_wet|mix)$", 2),
    (r"_(drive|time|size|decay|feedback|frequency|rate|depth|delay|amount|ratio|threshold|attack|release)$", 3),
    # global: voice behaviour first, master volume last
    (r"^(portamento|legato|polyphony|voice)", 1), (r"^volume$", 9),
]
_KEY_ORDER = [(re.compile(rx), rank) for rx, rank in KEY_ORDER]
_IMPACT = threading.local()       # per build_plan call: {key: audible impact} for the tie-break and the split
COARSE_SHARE = 0.8                # a round's coarse step: the biggest-impact keys carrying >= 80% of its impact
SPLIT_MIN_KEYS = 6                # rounds smaller than this are never split


def key_rank(key: str) -> int:
    return next((rank for rx, rank in _KEY_ORDER if rx.search(key)), 8)


def key_sort(key: str) -> tuple:
    impact = getattr(_IMPACT, "v", None) or {}
    return key_rank(key), -(impact.get(key) or 0.0), key


@lru_cache(maxsize=1)
def params() -> dict:
    return vitallib.params()


# --------------------------------------------------------------------------- values as an agent would type them

def _num(x: float, nd: int = 4) -> str:
    s = f"{round(float(x), nd):.{nd}f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def fmt_value(key: str, v: float) -> str:
    """JSON value -> set.py argument: enum label, or a unit the skill documents, else the raw number."""
    p = params()[key]
    if "labels" in p:
        i = int(round(v - p["min"]))
        if 0 <= i < len(p["labels"]):
            lab = p["labels"][i]
            if not re.fullmatch(r"[-+]?\d*\.?\d+", lab):  # numeric labels ("1".."16") stay numbers
                return lab
    if p["discrete"]:
        return str(int(round(v)))
    if re.search(r"(cutoff|_center)$", key) and p["min"] >= 8 - 1e-6 and p["max"] <= 136 + 1e-6:
        hz = 440 * 2 ** ((v - 69) / 12)
        return f"{hz / 1000:.3g}kHz" if hz >= 1000 else f"{hz:.4g}Hz"
    if re.match(r"env_\d_(attack|decay|release|delay|hold)$", key):
        sec = max(float(v), 0.0) ** 4
        return f"{sec:.3g}s" if sec >= 1 else f"{sec * 1000:.3g}ms"
    if key.endswith("_transpose"):
        return f"{_num(v, 2)}st"
    if (p["min"], p["max"]) in ((0.0, 1.0), (-1.0, 1.0)):
        return f"{_num(v * 100, 1)}%"
    if key == "volume":
        return f"{_num(math.sqrt(max(v, 0.0)) - 80, 1)}dB"
    return _num(v)


def parsed(key: str, v: float) -> float:
    """What set.py stores for fmt_value(key, v) (unit rounding included)."""
    return float(vitallib.parse_value(key, fmt_value(key, v)))


def clamp(key: str, v: float) -> float:
    p = params()[key]
    v = min(max(float(v), p["min"]), p["max"])
    return float(round(v)) if p["discrete"] else v


# --------------------------------------------------------------------------- edits

@dataclass
class Edit:
    """One bash call of skill-CLI work on preset.json."""
    tool: str                       # "set" | "mod" | "wavetable" | "shape"
    cmd: str                        # the bash command
    values: dict = field(default_factory=dict)   # set: key -> JSON value it stores
    note: str = ""                  # what the edit is, in words (for the narration adapter / the index)

    def command(self) -> str:
        return self.cmd


def cli(script: str, args: list[str]) -> str:
    return f"python3 {S}/{script} " + " ".join(shlex.quote(a) for a in args)


def set_edit(values: dict[str, float], note: str = "") -> Edit:
    keys = sorted(values, key=key_sort)
    return Edit("set", cli("set.py", ["preset.json"] + [f"{k}={fmt_value(k, values[k])}" for k in keys]),
                {k: parsed(k, values[k]) for k in keys}, note)


def shape_edit(lfo_shapes: dict[int, dict]) -> Edit:
    """LFO shapes are data (settings.lfos[N-1]); a heredoc python edit, as SKILL.md says."""
    lines = ["python3 - <<'EOF'", "import json", "p = json.load(open('preset.json'))"]
    for n, shp in sorted(lfo_shapes.items()):
        shp = {k: ([round(x, 4) for x in v] if isinstance(v, list) else v) for k, v in shp.items()}
        lines.append(f"p['settings']['lfos'][{n - 1}] = {shp!r}")
    lines += ["json.dump(p, open('preset.json', 'w'))",
              "print(', '.join(f\"lfo {i + 1}: {l['name']}\" for i, l in enumerate(p['settings']['lfos'])"
              " if l['name'] != 'Triangle'))", "EOF"]
    names = ", ".join(f"lfo {n} {s.get('name')}" for n, s in sorted(lfo_shapes.items()))
    return Edit("shape", "\n".join(lines), {}, f"LFO shapes: {names}")


def mod_args(route: dict) -> list[str]:
    amt = f"{_num(route['amount'] * 100, 1)}%"
    args = ["preset.json", "add", route["source"], route["destination"], amt]
    if route.get("bipolar"):
        args += ["--bipolar", "on"]
    if route.get("power"):
        args += ["--power", _num(route["power"], 2)]
    if route.get("stereo"):
        args += ["--stereo", "on"]
    return args


def mod_edit(routes: list[dict], remove: list[tuple[str, str]] = (), explicit: frozenset = frozenset()) -> Edit:
    """All routes of the round in one bash call, one mod.py line each (they edit the same file: sequential): first
    `remove` for routes the starting patch has and the target doesn't, then `add`. Routes in `explicit` already
    exist in the starting patch, so their flags are spelled out (add keeps an existing slot's flags otherwise)."""
    lines = [cli("mod.py", ["preset.json", "remove", a, b]) for a, b in remove]
    for r in routes:
        args = mod_args(r)
        if (r["source"], r["destination"]) in explicit:
            args += [x for x in (["--bipolar", "off"] if not r.get("bipolar") else []) +
                     (["--power", "0"] if not r.get("power") else []) +
                     (["--stereo", "off"] if not r.get("stereo") else [])]
        lines.append(cli("mod.py", args))
    what = ", ".join([f"remove {a} -> {b}" for a, b in remove] + [f"{r['source']} -> {r['destination']}" for r in routes])
    return Edit("mod", "\n".join(lines), {}, "routes " + what)


def start_routes(init: dict) -> list[dict]:
    """Every used mod slot of a starting patch (live or not)."""
    s = init["settings"]
    out = []
    for i, m in enumerate(s.get("modulations", [])):
        if m.get("source") and m.get("destination"):
            n = i + 1
            out.append({"source": m["source"], "destination": m["destination"],
                        "amount": float(s.get(f"modulation_{n}_amount", 0)),
                        "bipolar": float(s.get(f"modulation_{n}_bipolar", 0)) > 0.5,
                        "power": float(s.get(f"modulation_{n}_power", 0)),
                        "stereo": float(s.get(f"modulation_{n}_stereo", 0)) > 0.5})
    return out


# --------------------------------------------------------------------------- GT wavetables

def _wt_hash(table: dict) -> str:
    import hashlib
    return hashlib.md5(json.dumps(table.get("groups"), sort_keys=True).encode()).hexdigest()


@lru_cache(maxsize=1)
def _wt_by_hash() -> dict[str, dict]:
    out: dict[str, dict] = {}
    for e in vitallib.wt_index():
        out.setdefault(_wt_hash(vitallib.wt_load(e)), e)
    return out


def wt_ref(e: dict) -> str:
    """How an agent names a table so wavetable.py set resolves exactly it: the name when it is the first entry with
    that name (and not "(unnamed)"), else the id."""
    first = next(x for x in vitallib.wt_index() if x["name"].lower() == e["name"].lower())
    return e["name"] if first["id"] == e["id"] and e["name"] != "(unnamed)" else e["id"]


def gt_wavetables(gt: dict, init: dict) -> list[tuple[int, dict]]:
    """[(osc, library entry)] for every active GT oscillator (init's table included: it is in the library too)."""
    s = gt["settings"]
    out = []
    for n in (1, 2, 3):
        if float(s.get(f"osc_{n}_on", 0)) > 0.5:
            e = _wt_by_hash().get(_wt_hash(s["wavetables"][n - 1]))
            if e is None:
                raise ValueError(f"GT osc {n} table {s['wavetables'][n - 1].get('name')!r} is not in the skill library")
            out.append((n, e))
    return out


# --------------------------------------------------------------------------- rounds

@dataclass
class Mistake:
    key: str
    kind: str              # overshoot | undershoot | omission | spurious
    wrong: float | None    # JSON value applied (None: omitted, the key keeps its previous value)
    true: float            # GT value (spurious: the value it had)


@dataclass
class Round:
    subsystem: str
    title: str
    edits: list[Edit]                  # as applied (mistakes included)
    gt_values: dict[str, float]        # GT JSON values this round is about
    mistakes: list[Mistake] = field(default_factory=list)
    fix: Edit | None = None            # set.py with the GT values of the mistaken keys


@dataclass
class Plan:
    wavetables: list[tuple[int, dict]]
    rounds: list[Round]
    changed: list[str]                 # all live changed keys


def section(key: str) -> str:
    if key.startswith(("osc_", "sample_")):
        return "oscillator"
    if key.startswith("env_1_"):
        return "envelope"
    if key.startswith("filter_"):
        return "filter"
    if re.match(r"(lfo_\d|random_\d|env_[2-6])_", key):
        return "modulation"
    if key.startswith(tuple(f"{u}_" for u in FX_UNITS)):
        return "fx"
    return "global"


def _group(key: str) -> str:
    """Sub-group inside a round: one set.py call per oscillator / filter / FX unit / mod source."""
    m = re.match(r"(osc_\d|filter_(?:1|2|fx)|lfo_\d|random_\d|env_\d)_", key)
    if m:
        return m.group(1)
    for u in FX_UNITS:
        if key.startswith(u + "_"):
            return u
    return "global"


def live_changed_keys(gt: dict, init: dict) -> list[str]:
    s, P = gt["settings"], params()
    keys = set()
    for b in v3.build_batches_from_diff(gt, init):
        keys.update(b.params)
    # keys path_gen's range table lacks (unnormalizable) are invisible to the v3 diff; add those that differ
    for k, v in s.items():
        if k in P and not isinstance(v, (list, dict)) and k not in keys and v != init["settings"].get(k, v):
            keys.add(k)
    out = []
    for k in sorted(keys):
        if k not in P or k.startswith("modulation_") or vitallib.why_dead(s, k) is not None:
            continue
        if abs(float(s[k]) - float(init["settings"].get(k, P[k]["default"]))) < 1e-9:
            continue
        out.append(k)
    return out


def gt_routes(gt: dict) -> list[dict]:
    s = gt["settings"]
    out = []
    for r in vitallib.live_routes(s):
        if vitallib.why_dead(s, r["destination"]) is not None:
            continue
        i = r["slot"]
        out.append({**r, "bipolar": float(s.get(f"modulation_{i}_bipolar", 0)) > 0.5,
                    "power": float(s.get(f"modulation_{i}_power", 0)),
                    "stereo": float(s.get(f"modulation_{i}_stereo", 0)) > 0.5})
    return out


def gt_shapes(gt: dict, init: dict, routes: list[dict]) -> dict[int, dict]:
    used = {int(r["source"].split("_")[1]) for r in routes if r["source"].startswith("lfo_")}
    s, i = gt["settings"]["lfos"], init["settings"]["lfos"]
    return {n: s[n - 1] for n in sorted(used) if s[n - 1] != i[n - 1]}


_INJECT_LOCK = threading.Lock()


def _inject(rounds: list[Round], gt: dict, init: dict, rng, per_param_rate: float, max_total: int | None) -> None:
    """v3.inject_mistakes on the rounds' continuous set.py keys; rewrites the affected edits and adds fixes."""
    P, s_gt, s_init = params(), gt["settings"], init["settings"]
    batches = []
    for r in rounds:
        cont = {k: _normalize(k, v) for k, v in r.gt_values.items() if not P[k]["discrete"]}
        cont = {k: float(v) for k, v in cont.items() if v is not None}
        batches.append(v3.SubsystemBatch(subsystem=r.subsystem, params=cont, params_applied=dict(cont)))
    # spurious pool: same-section continuous keys the GT leaves at init that are live in the GT patch
    pool: dict[str, list[str]] = {}
    for k, p in P.items():
        if p["discrete"] or k.startswith("modulation_") or _normalize(k, s_init.get(k, p["default"])) is None:
            continue
        if k in s_gt and s_gt[k] == s_init.get(k) and vitallib.why_dead(s_gt, k) is None:
            pool.setdefault(section(k), []).append(k)
    with _INJECT_LOCK:
        saved = dict(v3.SUBSYSTEM_PARAM_POOL)
        v3.SUBSYSTEM_PARAM_POOL.clear()
        v3.SUBSYSTEM_PARAM_POOL.update(pool)
        try:
            v3.inject_mistakes(batches, rng, per_param_rate=per_param_rate, max_mistakes_total=max_total,
                               init_preset_settings=s_init)
        finally:
            v3.SUBSYSTEM_PARAM_POOL.clear()
            v3.SUBSYSTEM_PARAM_POOL.update(saved)
    for r, b in zip(rounds, batches):
        if not b.mistakes:
            continue
        fix_vals = {}
        for m in b.mistakes:
            true = float(s_init.get(m.param, P[m.param]["default"])) if m.kind == "spurious" else float(s_gt[m.param])
            wrong = None if m.kind == "omission" else clamp(m.param, _denormalize(m.param, m.wrong_value))
            if wrong is not None and abs(parsed(m.param, wrong) - parsed(m.param, true)) < 1e-6:
                continue
            r.mistakes.append(Mistake(m.param, m.kind, wrong, true))
            fix_vals[m.param] = true
        if not r.mistakes:
            continue
        # rewrite the set.py edits of this round with the applied (wrong) values
        bad = {m.key: m for m in r.mistakes}
        new_edits = []
        spurious = [m for m in r.mistakes if m.kind == "spurious"]
        for e in r.edits:
            if e.tool != "set":
                new_edits.append(e)
                continue
            vals = {k: v for k, v in e.values.items() if not (k in bad and bad[k].kind == "omission")}
            vals.update({k: bad[k].wrong for k in vals if k in bad})
            for m in list(spurious):
                if _group(m.key) == _group(next(iter(e.values), "")):
                    vals[m.key] = m.wrong
                    spurious.remove(m)
            if vals:
                new_edits.append(set_edit(vals, e.note))
        for m in spurious:  # no edit of its group in this round: append to the last set.py edit
            idx = max((i for i, e in enumerate(new_edits) if e.tool == "set"), default=None)
            if idx is None:
                new_edits.append(set_edit({m.key: m.wrong}))
            else:
                vals = dict(new_edits[idx].values)
                vals[m.key] = m.wrong
                new_edits[idx] = set_edit(vals, new_edits[idx].note)
        r.edits = new_edits
        r.fix = set_edit(fix_vals, "correction")


def split_round(r: Round, impact: dict[str, float]) -> list[Round]:
    """Coarse then fine: when a set-only round has a long tail of small-impact keys, the coarse step takes the
    biggest-impact keys carrying >= COARSE_SHARE of the round's impact (plus every gate/mode key), the fine step the
    rest, after a listen that shows the remaining gap. Impact = the key's measured one-key revert distance."""
    keys = [k for e in r.edits if e.tool == "set" for k in e.values]
    if len(keys) < SPLIT_MIN_KEYS or any(e.tool != "set" for e in r.edits) or not impact:
        return [r]
    tot = sum(impact.get(k) or 0.0 for k in keys)
    if tot <= 0:
        return [r]
    coarse, acc = set(), 0.0
    for k in sorted(keys, key=lambda k: -(impact.get(k) or 0.0)):
        if acc >= COARSE_SHARE * tot:
            break
        coarse.add(k)
        acc += impact.get(k) or 0.0
    coarse |= {k for k in keys if key_rank(k) <= 1}
    fine = [k for k in keys if k not in coarse]
    if len(fine) < 2:
        return [r]
    by = lambda ks: {g: {k: r.gt_values[k] for k in ks if _group(k) == g}  # noqa: E731
                     for g in dict.fromkeys(_group(k) for k in ks)}
    mk = lambda title, ks: Round(r.subsystem, title, [set_edit(v, g) for g, v in by(ks).items() if v],  # noqa: E731
                                 {k: r.gt_values[k] for k in ks})
    return [mk(r.title, [k for k in keys if k in coarse]), mk(f"{r.title.split(':')[0]}: fine", fine)]


def first_impression_ok(fi_text: str, facts: dict):
    """Which oscillator keys the first impression (the only evidence before the first listen) supports. Keys it
    doesn't support wait for the oscillator's fine step, after a listen whose notes show their dimension."""
    low = fi_text.lower()
    wide = facts.get("wide") or re.search(r"thick|detuned|chorus|wide|lush", low)
    grit = not facts.get("no_dist", True) or re.search(r"grit|buzz|distort|bite|crunch", low)
    tone = re.search(r"- tone:.*\b(bright|dark|rich|pure|thin|hollow|metallic|smooth|buzzy|overtones)", low)
    rules = [(r"_(on|transpose|tune|level)$", True),            # the part, its register, its loudness: always heard
             (r"_unison_|_stereo_spread$|_detune_range$", wide), (r"_distortion_", grit),
             (r"_spectral_morph_", tone), (r"_pan$", facts.get("wide") or facts.get("narrow"))]

    def ok(key: str) -> bool:
        return bool(next((v for rx, v in rules if re.search(rx, key)), False))
    return ok


def defer_unsupported(parts: list[Round], ok) -> list[Round]:
    """The first oscillator step keeps only keys the first impression supports; the rest join the fine step (created
    right after it if the round wasn't split)."""
    first = parts[0]
    keep = {k: v for k, v in first.gt_values.items() if ok(k)}
    moved = {k: v for k, v in first.gt_values.items() if not ok(k)}
    if not moved or not keep:
        return parts
    def mk(title, vals):
        groups = {g: {k: vals[k] for k in vals if _group(k) == g} for g in dict.fromkeys(_group(k) for k in vals)}
        return Round(first.subsystem, title, [set_edit(v, g) for g, v in groups.items() if v], vals)
    out = [mk(first.title, keep)]
    if len(parts) > 1:
        out.append(mk(parts[1].title, {**moved, **parts[1].gt_values}))
        return out + parts[2:]
    return out + [mk(f"{first.title.split(':')[0]}: fine", moved)]


def build_plan(gt: dict, init: dict, rng, mistake_rate: float = 0.10, max_mistakes: int | None = None,
               audible: set[str] | None = None, impact: dict[str, float] | None = None,
               first_ok=None) -> Plan:
    """audible: the changed keys that can be heard (wt_rank.audible_changes); the others are never edited.
    impact: each key's audible impact (its one-key revert distance), for the within-rank order and coarse/fine split.
    first_ok: key -> bool, the first impression's support for an oscillator key (first_impression_ok); unsupported
    oscillator keys move to the oscillator's fine step."""
    _IMPACT.v = impact or {}
    s = gt["settings"]
    keys = live_changed_keys(gt, init)
    if audible is not None:
        keys = [k for k in keys if k in audible]
    by_sec: dict[str, dict[str, dict[str, float]]] = {}
    for k in keys:
        by_sec.setdefault(section(k), {}).setdefault(_group(k), {})[k] = float(s[k])
    routes = gt_routes(gt)
    shapes = gt_shapes(gt, init, routes)
    # a starting patch other than init: its routes the target lacks go, its routes the target keeps are re-set with
    # every flag; routes it already has exactly are not re-added
    have = {(r["source"], r["destination"]): r for r in start_routes(init)}
    want = {(r["source"], r["destination"]) for r in routes}
    remove = sorted(k for k in have if k not in want)
    same = lambda r, h: (abs(r["amount"] - h["amount"]) < 1e-6 and r["bipolar"] == h["bipolar"]  # noqa: E731
                         and abs(r["power"] - h["power"]) < 1e-6 and r["stereo"] == h["stereo"])
    routes = [r for r in routes if not ((k := (r["source"], r["destination"])) in have and same(r, have[k]))]
    explicit = frozenset(k for k in want if k in have)
    rounds: list[Round] = []
    for sec in ORDER:
        groups = by_sec.get(sec, {})
        if sec == "fx":
            # one round per FX unit pair at most (rule 3: one subsystem per listen), in tuning order (FX_TUNE)
            units = sorted(groups, key=lambda g: FX_TUNE.index(g) if g in FX_TUNE else 99)
            for i in range(0, len(units), 2):
                chunk = units[i:i + 2]
                vals = {k: v for g in chunk for k, v in groups[g].items()}
                rounds.append(Round("fx", "fx: " + " + ".join(chunk),
                                    [set_edit(groups[g], g) for g in chunk], vals))
            continue
        edits = [set_edit(groups[g], g) for g in sorted(groups)]
        if sec == "modulation":
            if not routes and not remove and not edits and not shapes:
                continue
            # what moves what, and how much, first; then the sources' rates and envelopes; their shapes last
            edits = ([mod_edit(routes, remove, explicit)] if routes or remove else []) + edits + \
                    ([shape_edit(shapes)] if shapes else [])
        if not edits and sec != "oscillator":
            continue
        vals = {k: v for g in groups.values() for k, v in g.items()}
        parts = split_round(Round(sec, sec, edits, vals), impact or {})
        if sec == "oscillator" and first_ok is not None:
            parts = defer_unsupported(parts, first_ok)
        rounds += parts
    _inject([r for r in rounds if r.subsystem != "modulation" or r.gt_values], gt, init, rng,
            mistake_rate, max_mistakes)
    return Plan(gt_wavetables(gt, init), rounds, keys)
