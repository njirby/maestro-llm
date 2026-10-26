"""Narration adapter: the main agent's words about what it hears. The builder only calls the four functions below;
which narrator backs each mode is configured with configure(NarrationConfig(...)) (CLI flags of
scripts/build_rollouts_v5.py), so the per-mode choice can change without touching the builder.

Modes and backends:
  first_impression   WS-B maestro.sounddiff.first_impression: notes_first_impression (ablation-verified notes of the
                     GT patch) -> narrate_first_impression (Qwen3-Omni, target audio + notes), when fi_url is set
  self_edit          WS-B maestro.sounddiff.delta: notes_self_edit(before, after) -> narrate_self_edit(arm=edit_arm)
  target_vs_current  notes_target_vs_current(GT, before, after, next_subsystem) -> narrate_target_vs_current(arm=...)
                     when edit_url is set
  final_verdict      template only (no WS-B mode yet)
Deferred mode (NarrationConfig.defer): see the field comment; slot ids are "<sample>/<mode>/<n>".
Fallback: the template (below) when the mode's server is not configured, when the call fails, or when the narration
still has checker problems after WS-B's one retry. Every LLM use and every fallback is reported through `log`
(Ctx.note -> index.jsonl events: event="narration", mode, source llm|template, reason, problems, arm).

Interface (all keyword-only): paths are HOST paths of files that exist; presets are full Vital preset dicts; notes is
{"bpm", "notes": [{pitch, velocity, start_s, dur_s}]} (the GT notes); `renderer` is a ContainerRenderer-compatible
renderer on the rollout's own session (ctx.OracleRenderer: host-side only, nothing lands in the agents' workdir);
`log(**event)` records what happened. Ablation renders and narrator calls are builder-side; only the returned
sentence goes in the record.

  first_impression(target_wav, notes, gt_preset, renderer, log)
  self_edit(before_wav, after_wav, before_preset, after_preset, changed, subsystem, notes, renderer, log)
  target_vs_current(target_wav, current_wav, before_preset, current_preset, gt_preset, subsystem, next_subsystem,
                    notes, renderer, log)
  final_verdict(target_wav, final_wav, final_preset, gt_preset, notes, renderer, log)

`changed` = {key: (old, new)} for the round (routes / LFO shapes / wavetables as "route a -> b", "lfo_1 shape",
"osc_1 wavetable"); subsystem / next_subsystem are planner round names (oscillator, envelope, filter, modulation, fx,
global). Contract: never state anything about the sound that was not measured or is not implied by the notes / the
edit itself. The templates only state facts of that kind and make no perceptual claims.
"""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Callable

Log = Callable[..., None]
_NAMES = "C C# D D# E F F# G G# A A# B".split()
# planner round -> WS-B delta subsystem (next_subsystem); "global" has no counterpart
DELTA_SUB = {"oscillator": "osc", "envelope": "env", "filter": "filter", "modulation": "mod", "fx": "fx"}


def delta_next(next_subsystem: str | None) -> str | None:
    """Planner round title ("fx: delay + reverb") -> WS-B's next_subsystem ("fx"), or None."""
    return DELTA_SUB.get(next_subsystem.split(":")[0]) if next_subsystem else None


@dataclass
class NarrationConfig:
    fi_url: str | None = None          # Omni (first impression), e.g. http://localhost:8000/v1
    fi_model: str | None = None
    edit_arm: str = "text"             # self_edit / target_vs_current: "omni" (clips + notes) | "text" (notes only)
    edit_url: str | None = None        # e.g. http://localhost:8001/v1 for the text arm (27B)
    edit_model: str | None = None
    # deferred (two-pass) mode: compute the verified notes now (CPU, the rollout's session), put a slot token in the
    # record, and hand the slot to slot_sink; scripts/narrate_rollouts_v5.py narrates the slots later per mode (27B
    # pass for the edit modes, Omni pass for first impressions) and --finalize swaps the tokens for the text.
    defer: bool = False
    slot_sink: Callable[[dict], None] | None = None


CONFIG = NarrationConfig()
TOKEN = "\u27e6narr:{}\u27e7"           # ⟦narr:<slot id>⟧


def configure(cfg: NarrationConfig) -> None:
    global CONFIG
    CONFIG = cfg


def _notes_list(notes) -> list[dict]:
    return notes["notes"] if isinstance(notes, dict) else notes


def _llm(mode: str, log: Log | None, build, narrate, template: str, slot: str | None = None) -> str:
    """notes -> narration, or the template with a logged reason. Deferred mode: notes -> slot token."""
    log = log or (lambda **_: None)
    try:
        n = build()
        if CONFIG.defer:
            CONFIG.slot_sink({"slot": slot, "mode": mode, "notes": n, "template": template})
            log(event="narration", mode=mode, source="deferred", slot=slot)
            return TOKEN.format(slot)
        out = narrate(n)
    except Exception as e:  # server down, render failure, ...: never block a rollout on narration
        log(event="narration", mode=mode, source="template", reason=f"{type(e).__name__}: {str(e)[:300]}")
        return template
    if out["problems"] or not out["text"].strip():
        log(event="narration", mode=mode, source="template", reason="checker problems after retry",
            problems=out["problems"], rejected=out["text"][:500])
        return template
    log(event="narration", mode=mode, source="llm", arm=out.get("arm"), attempts=len(out["attempts"]))
    return out["text"].strip()
def _pitch(p: int) -> str:
    return f"{_NAMES[p % 12]}{p // 12 - 1}"


def note_facts(notes: list[dict]) -> list[str]:
    """Musical facts from the notes (vendored from sounddiff/first_impression/fi_notes.midi_facts, no tempo)."""
    n = notes
    ends = sorted([(x["start_s"], 1) for x in n] + [(x["start_s"] + x["dur_s"], -1) for x in n])
    cur = mx = 0
    for _, d in ends:
        cur += d
        mx = max(mx, cur)
    total = max(x["start_s"] + x["dur_s"] for x in n)
    overlap = sum(max(0.0, min(a["start_s"] + a["dur_s"], b["start_s"] + b["dur_s"]) - max(a["start_s"], b["start_s"]))
                  for i, a in enumerate(n) for b in n[i + 1:])
    share = overlap / total
    shape = ("a single line" if mx == 1 else "chords throughout" if share > 0.5 else
             f"chords of up to {mx} notes" if share > 0.15 else "mostly a single line with some overlapping notes")
    med = sorted(x["pitch"] for x in n)[len(n) // 2]
    reg = "low" if med < 52 else "mid" if med < 67 else "high" if med < 80 else "very high"
    avg = sum(x["dur_s"] for x in n) / len(n)
    length = ("very short, staccato notes" if avg < 0.15 else "short notes" if avg < 0.35 else
              "medium-length notes" if avg < 0.8 else "long held notes")
    dens = len(n) / total
    busy = "sparse" if dens < 1.5 else "busy" if dens > 4 else None
    lo, hi = min(x["pitch"] for x in n), max(x["pitch"] for x in n)
    return [shape, f"{reg} register ({_pitch(lo)}-{_pitch(hi)})", length + (f", {busy}" if busy else "")]


def first_impression(*, target_wav: Path, notes: dict, gt_preset: dict | None = None, renderer=None,
                     log: Log | None = None, slot: str | None = None) -> str:
    shape, reg, length = note_facts(_notes_list(notes))
    template = f"The target plays {shape} in the {reg}, {length}."
    if not (CONFIG.fi_url or CONFIG.defer) or renderer is None or gt_preset is None:
        (log or (lambda **_: None))(event="narration", mode="first_impression", source="template",
                                    reason="no server configured")
        return template
    from maestro.sounddiff.first_impression import narrate_first_impression, notes_first_impression
    return _llm("first_impression", log,
                lambda: notes_first_impression(gt_preset, notes, renderer=renderer),
                lambda n: narrate_first_impression(n["audio"], n, url=CONFIG.fi_url, model=CONFIG.fi_model),
                template, slot)


def _fmt_change(key: str, old, new) -> str:
    if old is None:
        return f"{key} {new}"
    return f"{key} {old} -> {new}"


def _edit_backend(mode: str, log: Log | None, renderer) -> bool:
    if (CONFIG.edit_url or CONFIG.defer) and renderer is not None:
        return True
    (log or (lambda **_: None))(event="narration", mode=mode, source="template", reason="no server configured")
    return False


def self_edit(*, before_wav: Path | None, after_wav: Path, before_preset: dict, after_preset: dict,
              changed: dict[str, tuple], subsystem: str, notes: dict, renderer=None, log: Log | None = None,
              slot: str | None = None) -> str:
    keys = list(changed)
    shown = ", ".join(keys[:4]) + (f" and {len(keys) - 4} more" if len(keys) > 4 else "")
    template = f"That round changed {shown}."
    if not _edit_backend("self_edit", log, renderer):
        return template
    from maestro.sounddiff.delta import narrate_self_edit, notes_self_edit
    return _llm("self_edit", log,
                lambda: notes_self_edit(before_preset, after_preset, notes, renderer=renderer),
                lambda n: narrate_self_edit(n, url=CONFIG.edit_url, model=CONFIG.edit_model, arm=CONFIG.edit_arm),
                template, slot)


def target_vs_current(*, target_wav: Path, current_wav: Path, before_preset: dict, current_preset: dict,
                      gt_preset: dict, subsystem: str, next_subsystem: str | None, notes: dict, renderer=None,
                      log: Log | None = None, slot: str | None = None) -> str:
    template = "Comparing this render with the target." + (f" Next: {next_subsystem}." if next_subsystem else "")
    if not _edit_backend("target_vs_current", log, renderer):
        return template
    from maestro.sounddiff.delta import narrate_target_vs_current, notes_target_vs_current
    nxt = delta_next(next_subsystem)
    return _llm("target_vs_current", log,
                lambda: notes_target_vs_current(gt_preset, before_preset, current_preset, notes, renderer=renderer,
                                                next_subsystem=nxt),
                lambda n: narrate_target_vs_current(n, url=CONFIG.edit_url, model=CONFIG.edit_model,
                                                    arm=CONFIG.edit_arm),
                template, slot)


def final_verdict(*, target_wav: Path, final_wav: Path, final_preset: dict, gt_preset: dict, notes: dict,
                  renderer=None, log: Log | None = None) -> str:
    return "The patch is in preset.json and the last render is in final.wav."
