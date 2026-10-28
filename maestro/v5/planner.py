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
ORDER = ("oscillator", "envelope", "filter", "modulation", "fx", "global")
FX_UNITS = tuple(vitallib.FX)
_GATES = re.compile(r"(_on|_type|_model|_style|_sync|_destination|_unison_voices)$")


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
    keys = sorted(values, key=lambda k: (not _GATES.search(k), k))
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


def build_plan(gt: dict, init: dict, rng, mistake_rate: float = 0.10, max_mistakes: int | None = None) -> Plan:
    s = gt["settings"]
    keys = live_changed_keys(gt, init)
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
            # one round per FX unit pair at most (rule 3: one subsystem per listen; FX units are a chain)
            units = sorted(groups, key=lambda g: FX_UNITS.index(g) if g in FX_UNITS else 99)
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
            if shapes:
                edits.append(shape_edit(shapes))
            if routes or remove:
                edits.append(mod_edit(routes, remove, explicit))
        if not edits and sec != "oscillator":
            continue
        vals = {k: v for g in groups.values() for k, v in g.items()}
        rounds.append(Round(sec, sec, edits, vals))
    _inject([r for r in rounds if r.subsystem != "modulation" or r.gt_values], gt, init, rng,
            mistake_rate, max_mistakes)
    return Plan(gt_wavetables(gt, init), rounds, keys)
