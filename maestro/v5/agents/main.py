"""maestro (main agent): one full rollout, subagents included.

   skill vital -> listen to target.wav -> first impression -> cp init preset.json
   -> task transcriber -> wavetable.py list --limit 0 -> parallel task searchers, one per ~32-table chunk
      (full audition + oriented-pair beam; one message) -> task judge (recombines across the pooled configurations)
   -> rounds (planner.Plan): [code mistake + fix] edits, render, listen, self_edit + target_vs_current narration,
     param-mistake correction (fix edit, render, listen, self_edit)
  -> show.py, final render, listen to target then final, final verdict.
"""
from __future__ import annotations

import hashlib
import json
import shlex
from pathlib import Path

from scripts import opencode_contract as oc
from scripts.agent_sft_common import compute_round_coverage

from .. import search

from maestro import notation

from .. import gate, narration
from ..ctx import Ctx, OracleRenderer
from ..mistakes import code_mistake, render_mistake
from ..planner import Edit, Plan, build_plan, vitallib, wt_ref
from ..record import Record, final_text
from ..start import START_FILE
from .common import INIT, RENDER, listen, open_session, run, skill_response  # noqa: F401  (re-export)
from .judge import run_judge
from .searcher import WT, run_searcher
from .transcriber import run_transcriber

AGENT = "maestro"
USER_PROMPT = "Recreate the sound in target.wav in Vital."
START_PROMPT = "Recreate the sound in target.wav in Vital, starting from my patch in {file}."
_ORD = {1: "first", 2: "second", 3: "third", 4: "fourth", 5: "fifth", 6: "sixth", 7: "seventh", 8: "eighth",
        9: "ninth", 10: "tenth", 11: "eleventh", 12: "twelfth"}


def _ordinal(i: int) -> str:
    if i in _ORD:
        return _ORD[i]
    return f"{i}{'th' if 11 <= i % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th')}"


def residual_problem(info: dict) -> str:
    """What a single residual transcription error sounds like against the target (main agent's words)."""
    n = _ordinal(info["note_idx"] + 1)
    if info["type"] == "pitch_shift":
        size = "a semitone" if abs(info["delta"]) == 1 else "a whole step"
        return f"the {n} note is {size} {'high' if info['delta'] > 0 else 'low'}"
    slots = abs(info["shift_slots"])
    amt = "a sixteenth" if slots == 1 else ("an eighth" if slots == 2 else "a dotted eighth")
    return f"the {n} note comes in {amt} {'late' if info['shift_slots'] > 0 else 'early'}"


def recheck_dispatch_prompt(target_audio_path, notes_path, problem: str) -> str:
    """Parent -> transcriber, mid-rollout: the notes in notes.json sound wrong against the target."""
    return ("Check the transcription in notes.json against the target clip — with the patch getting close, the "
            f"melody itself stands out now: {problem}. Listen to what's there next to the target, fix notes.json "
            "in the format the `vital` skill documents, and make sure the rest still matches.\n\n"
            f"Target: {target_audio_path}\n"
            f"Notes: {notes_path}")
def _changed(before: dict, after: dict) -> dict[str, tuple]:
    b, a = before["settings"], after["settings"]
    out = {k: (b.get(k), v) for k, v in a.items() if not isinstance(v, (list, dict)) and b.get(k) != v
           and not k.startswith("modulation_")}
    for i, (mb, ma) in enumerate(zip(b.get("modulations", []), a.get("modulations", []))):
        amt_b, amt_a = b.get(f"modulation_{i + 1}_amount"), a.get(f"modulation_{i + 1}_amount")
        if ma != mb or amt_a != amt_b:
            out[f"route {ma.get('source')} -> {ma.get('destination')}"] = (amt_b, amt_a)
    for i, (lb, la) in enumerate(zip(b.get("lfos", []), a.get("lfos", []))):
        if lb != la:
            out[f"lfo_{i + 1} shape"] = (lb.get("name"), la.get("name"))
    for i, (wb, wa) in enumerate(zip(b.get("wavetables", []), a.get("wavetables", []))):
        if wb != wa:
            out[f"osc_{i + 1} wavetable"] = (wb.get("name"), wa.get("name"))
    return out


def _cfg_key(cfg: dict) -> tuple:
    return tuple(sorted((n, e["id"]) for n, e in cfg.items()))


class Main:
    def __init__(self, ctx: Ctx, code_mistake_rate: float = 0.08, mistake_rate: float = 0.10,
                 render_mistake_rate: float = 0.0, retranscribe_rate: float = 0.0,
                 transcriber_retry_weights: tuple[float, ...] = (1.0,), start: dict | None = None):
        """start: the user's starting patch (maestro/v5/start.py), already in the workdir as START_FILE."""
        self.ctx, self.rng, self.s = ctx, ctx.rng, ctx.sample
        self.render_rate, self.retranscribe_rate = render_mistake_rate, retranscribe_rate
        self.t_retry = transcriber_retry_weights
        self.gt, self.init, self.notes = self.s.gt_preset(), self.s.init_preset(), self.s.notes()
        self.start_cmd = f"cp {INIT} preset.json"
        prompt = USER_PROMPT
        if start is not None:
            self.init, self.start_cmd = start, f"cp {START_FILE} preset.json"
            prompt = START_PROMPT.format(file=START_FILE)
        self.code_rate = code_mistake_rate
        self.plan: Plan = build_plan(self.gt, self.init, self.rng, mistake_rate=mistake_rate)
        self.rec = Record(AGENT, f"{self.s.id}.main", ctx.wd, prompt)
        self.sub: dict[str, list[dict]] = {"transcriber": [], "searcher": [], "judge": []}
        self.links: list[dict] = []
        self._slots = 0
        self.notes_d = {"bpm": self.s.bpm, "notes": self.notes}
        self.renderer = OracleRenderer(ctx.sess, ctx.oracle_host_dir / "render_cache", jobs=4)  # host-side only

    # -- helpers -----------------------------------------------------------------------------------------------

    def task(self, agent: str, k: int, description: str, prompt: str, record: dict) -> tuple[str, str]:
        sid = oc.session_id(seed=f"{self.s.id}/{agent}/{k}")
        self.sub[agent].append(record)
        self.links.append({"agent": agent, "record_id": record["id"], "session_id": sid, "description": description})
        return oc.task_call(description, prompt, agent), oc.task_result(sid, final_text(record))

    def slot(self, mode: str, next_subsystem: str | None = None, **presets: dict) -> str:
        """A narration slot id; its inputs (presets by content hash + the notes) are saved host-side under
        <run>/oracle/<sample>/ (narration_inputs.jsonl, presets/, notes.json) so every narration can be rebuilt and
        re-narrated, e.g. after WS-B changes the notes wording. Never in the agents' workdir."""
        self._slots += 1
        sid = f"{self.s.id}/{mode}/{self._slots:02d}"
        d = self.ctx.oracle_host_dir
        (d / "presets").mkdir(parents=True, exist_ok=True)
        notes_path = d / "notes.json"
        if not notes_path.exists():
            notes_path.write_text(json.dumps(self.notes_d))
        paths = {}
        for role, p in presets.items():
            blob = json.dumps(p, separators=(",", ":"))
            f = d / "presets" / f"{hashlib.sha1(blob.encode()).hexdigest()[:16]}.vital"
            if not f.exists():
                f.write_text(blob)
            paths[role] = str(f.resolve())
        row = {"slot": sid, "mode": mode, "presets": paths, "notes": str(notes_path.resolve()),
               "next_subsystem": next_subsystem}
        with open(d / "narration_inputs.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
        return sid

    def edit(self, e: Edit) -> None:
        ctx, rec = self.ctx, self.rec
        if self.rng.random() < self.code_rate:
            m = code_mistake(e, self.rng)
            if m is not None:
                backup = ctx.host("preset.json").read_bytes()
                call, out, res = ctx.bash(m.cmd)
                if res.exit_code != 0 or "error:" in res.output:
                    rec.step([call], [out])
                    rec.say(m.diagnosis)
                    ctx.note(agent=AGENT, event="code_mistake", kind=m.kind)
                else:  # the mutation happened to be valid: undo it, record nothing
                    ctx.host("preset.json").write_bytes(backup)
        run(rec, ctx, e.command())

    def render(self, wav: str) -> None:
        """render.py preset.json notes.json <wav>; with render_mistake_rate a broken variant runs first (for real:
        render.py checks its arguments before rendering, so a failed call writes nothing)."""
        ctx, rec, cmd = self.ctx, self.rec, f"{RENDER} preset.json notes.json {wav}"
        if self.rng.random() < self.render_rate:
            m = render_mistake(cmd, self.rng)
            if m is not None:
                before = {p.name for p in ctx.host(".").iterdir()}
                call, out, res = ctx.bash(m.cmd)
                if res.exit_code != 0 or "error:" in res.output:
                    rec.step([call], [out])
                    rec.say(m.diagnosis)
                    ctx.note(agent=AGENT, event="code_mistake", kind=m.kind)
                else:  # it rendered after all: remove what it wrote, record nothing
                    for p in ctx.host(".").iterdir():
                        if p.name not in before and p.is_file():
                            p.unlink()
        run(rec, ctx, cmd)

    # -- the rollout ---------------------------------------------------------------------------------------------

    def run(self) -> dict:
        ctx, rec, rng, s = self.ctx, self.rec, self.rng, self.s
        target, notes_path = ctx.path("target.wav"), ctx.path("notes.json")

        target_host = open_session(rec, ctx)
        rec.say(narration.first_impression(
            target_wav=Path(target_host), notes=self.notes_d, gt_preset=self.gt, renderer=self.renderer, log=ctx.note,
            slot=self.slot("first_impression", target=self.gt)))
        run(rec, ctx, self.start_cmd)

        # transcriber. With retranscribe_rate it ends on notes with one residual error it doesn't catch (that record
        # is kept out of training: link train=false); the main agent hears it after a later round and sends it back.
        residual, recheck_after = None, None
        if self.retranscribe_rate and len(self.plan.rounds) >= 3 and self.rng.random() < self.retranscribe_rate:
            residual = self.residual_error()
            if residual:
                recheck_after = self.rng.randint(1, len(self.plan.rounds) - 2)
        t_prompt = oc.transcription_dispatch_prompt(target, notes_path)
        t_rec = run_transcriber(ctx, t_prompt, f"{s.id}.transcriber", retry_weights=self.t_retry,
                                final=residual[0] if residual else None)
        rec.say("First the notes.")
        call, resp = self.task("transcriber", 0, "Transcribe target melody", t_prompt, t_rec)
        if residual:
            self.links[-1]["train"] = False
            ctx.note(agent=AGENT, event="residual_transcription", mutations=residual[1], recheck_after=recheck_after)
        rec.step([call], [resp])

        # searchers: the window, chunked — every table in each chunk auditioned; each replies with its chunk's best
        # configurations by the superposition ranking (maestro/v5/wt_rank.py); the configurations are pooled
        run(rec, ctx, f"{WT} list --limit 0")
        gt_tables = [e for _, e in self.plan.wavetables]
        regions = self.coverage(gt_tables)
        a0, b0 = regions[0]
        n_tables = b0 - a0
        spans = [(a0 + a, a0 + b) for a, b in search.chunks(n_tables)]
        calls, resps, pool = [], [], []
        for k, (a, b) in enumerate(spans, 1):
            prompt = oc.search_dispatch_prompt(target, notes_path, a, b)
            r, cfgs = run_searcher(ctx, k, prompt, f"{s.id}.searcher{k}", a, b, self.gt, self.renderer, gt_tables)
            call, resp = self.task("searcher", k, f"Search wavetables {oc.wavetable_slice_label(a, b)}", prompt, r)
            calls.append(call)
            resps.append(resp)
            have = {_cfg_key(c) for c in pool}
            pool += [c for c in cfgs if _cfg_key(c) not in have]
        what = "the whole library" if n_tables == 499 else f"{n_tables} of the 499 tables"
        rec.say(f"Searching {what} — {len(spans)} chunks, every table auditioned.")
        rec.step(calls, resps)

        # judge: recombines across the pooled configurations and picks the oscillator tables
        j_prompt = oc.judge_dispatch_prompt(target, [[wt_ref(e) for _, e in sorted(c.items())] for c in pool],
                                            notes_path=notes_path)
        j_rec, choice = run_judge(ctx, j_prompt, f"{s.id}.judge", pool, self.gt, self.renderer, gt_tables)
        n_tab = len({e["id"] for c in pool for e in c.values()})
        rec.say(f"{n_tab} tables in {len(pool)} configurations; the judge recombines and picks the oscillator tables.")
        call, resp = self.task("judge", 0, "Judge wavetable pool", j_prompt, j_rec)
        rec.step([call], [resp])

        # rounds: the judge's tables go on (the GT's, or frame-0 duplicates of them, when the pool held them)
        wt_edit = Edit("wavetable", "\n".join(f"{WT} set preset.json --osc {n} {shlex.quote(wt_ref(e))}"
                                             for n, e in sorted(choice.items())), {}, "wavetables")
        rounds = self.plan.rounds
        if rounds and rounds[0].subsystem == "oscillator":
            rounds[0].edits.insert(0, wt_edit)
        prev_wav, prev = None, ctx.read_json("preset.json")
        for i, r in enumerate(rounds, 1):
            nxt = rounds[i].title if i < len(rounds) else None
            for e in r.edits:
                self.edit(e)
            wav = f"r{i:02d}_{r.subsystem}.wav"
            self.render(wav)
            host = Path(listen(rec, ctx, wav))
            cur = ctx.read_json("preset.json")
            rec.say(narration.self_edit(
                before_wav=prev_wav, after_wav=host, before_preset=prev, after_preset=cur, changed=_changed(prev, cur),
                subsystem=r.subsystem, notes=self.notes_d, renderer=self.renderer, log=ctx.note,
                slot=self.slot("self_edit", before=prev, after=cur)))
            tvc_next = r.title if r.fix else nxt
            rec.say(narration.target_vs_current(
                target_wav=Path(target_host), current_wav=host, before_preset=prev, current_preset=cur,
                gt_preset=self.gt, subsystem=r.subsystem, next_subsystem=tvc_next, notes=self.notes_d,
                renderer=self.renderer, log=ctx.note,
                slot=self.slot("target_vs_current", narration.delta_next(tvc_next), target=self.gt, before=prev,
                               after=cur)))
            prev_wav, prev = host, cur
            if r.fix:
                ctx.note(agent=AGENT, event="param_mistakes", round=i,
                         mistakes=[m.__dict__ for m in r.mistakes])
                self.edit(r.fix)
                wav = f"r{i:02d}_{r.subsystem}_fix.wav"
                self.render(wav)
                host = Path(listen(rec, ctx, wav))
                cur = ctx.read_json("preset.json")
                rec.say(narration.self_edit(
                    before_wav=prev_wav, after_wav=host, before_preset=prev, after_preset=cur,
                    changed=_changed(prev, cur), subsystem=r.subsystem, notes=self.notes_d, renderer=self.renderer,
                    log=ctx.note, slot=self.slot("self_edit", before=prev, after=cur)))
                prev_wav, prev = host, cur
            if i == recheck_after:
                prev_wav = self.recheck_notes(residual, i, r.subsystem, target, notes_path)

        # final check
        run(rec, ctx, "python3 $VITAL_SKILL/scripts/show.py preset.json")
        self.render("final.wav")
        listen(rec, ctx, "target.wav")
        final_host = listen(rec, ctx, "final.wav")
        main = rec.finish(narration.final_verdict(target_wav=Path(target_host), final_wav=Path(final_host),
                                                  final_preset=ctx.read_json("preset.json"), gt_preset=self.gt,
                                                  notes=self.notes_d, renderer=self.renderer, log=ctx.note))
        return {"main": main, **self.sub, "links": self.links, "gate": self.gt_gate()}

    def residual_error(self, tries: int = 10):
        """One wrong pitch or one note off the grid that leaves the note order intact (so "the Nth note" is the one
        you hear), with the shift as it actually landed. None if no such error fits the phrase."""
        gt_b = notation.to_beat_form(self.s.bpm, self.notes)["notes"]
        slot = notation._slot_of
        for _ in range(tries):
            mut = notation.mutate_transcription(gt_b, self.rng, n_muts=1, kinds=("pitch_shift", "timing_shift"))
            if mut is None or len(mut[1]) != 1:
                continue
            wrong, (info,), narr = mut
            if info["type"] == "timing_shift":
                i = info["note_idx"]
                moved = slot(wrong[i]) - slot(gt_b[i])
                order = [slot(n) for n in wrong]
                others = order[:i] + order[i + 1:]
                if moved == 0 or order != sorted(order) or order[i] in others:
                    continue
                info = {**info, "shift_slots": moved}
            return wrong, [info], narr
        return None

    def recheck_notes(self, residual, i: int, subsystem: str, target: str, notes_path: str) -> Path:
        """The main agent hears the residual transcription error, sends the transcriber back to notes.json, then
        re-renders the same patch with the fixed notes. Returns the new render (host copy)."""
        ctx, rec, s = self.ctx, self.rec, self.s
        problem = residual_problem(residual[1][0])
        rec.say(f"Something else is off, and it isn't the patch: {problem} compared to the target. That's the "
                f"transcription. Sending the transcriber back to check notes.json.")
        prompt = recheck_dispatch_prompt(target, notes_path, problem)
        what = f"{problem[0].upper()}{problem[1:]} compared to the target; the rest matches. Rewriting notes.json."
        t_rec = run_transcriber(ctx, prompt, f"{s.id}.transcriber2", existing=(residual[0], residual[1], what),
                                wav_prefix="transcription_check")
        call, resp = self.task("transcriber", 1, "Recheck melody transcription", prompt, t_rec)
        rec.step([call], [resp])
        ctx.note(agent=AGENT, event="retranscribed", round=i, mutations=residual[1])
        wav = f"r{i:02d}_{subsystem}_notes.wav"
        self.render(wav)
        host = Path(listen(rec, ctx, wav))
        rec.say("Same patch, with the corrected notes; the melody lines up with the target now.")
        return host

    def gt_gate(self) -> dict:
        """Does the rollout land on the GT? Structural + audio gates (maestro/v5/gate.py), host-side."""
        final = self.ctx.read_json("preset.json")
        d = self.ctx.oracle_host_dir   # host-side record of what the gate compared (for debugging dropped rollouts)
        d.mkdir(parents=True, exist_ok=True)
        (d / "final_preset.json").write_text(json.dumps(final))
        (d / "gt_preset.json").write_text(json.dumps(self.gt))
        g = gate.gate(self.renderer, self.gt, final, self.notes_d)
        takes = {"gt": (self.gt, "")} | {f"gt#{k}": (self.gt, str(k)) for k in range(2, gate.GT_TAKES + 1)}
        g["audio"]["wavs"] = {k: str(self.renderer.path_of(p, self.notes_d, v)) for k, (p, v) in
                              {**takes, "final": (final, "")}.items()}
        self.ctx.note(agent=AGENT, event="gt_gate", **g)
        return g

    def coverage(self, gt_tables: list[dict], tries: int = 64) -> list[tuple[int, int]]:
        """compute_round_coverage's round-1 window over the library (coverage 50-100%), redrawn until it is one
        contiguous range (which is then chunked for the searchers) holding every GT table (oracle: the GT reaches
        the pool)."""
        idx = [e["id"] for e in vitallib.wt_index()]
        need = {idx.index(e["id"]) for e in gt_tables}
        for _ in range(tries):
            r1, _ = compute_round_coverage(len(idx), self.rng.uniform(0.5, 1.0), self.rng)
            if len(r1) == 1 and all(any(a <= i < b for a, b in r1) for i in need):  # one contiguous window
                return r1
        return [(0, len(idx))]
