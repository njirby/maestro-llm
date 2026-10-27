"""maestro (main agent): one full rollout, subagents included.

  skill vital -> listen to target.wav -> first impression -> cp init preset.json
  -> task transcriber -> wavetable.py list --limit 0 -> 2-4 parallel task searcher (one message) -> task judge
  -> rounds (planner.Plan): [code mistake + fix] edits, render, listen, self_edit + target_vs_current narration,
     param-mistake correction (fix edit, render, listen, self_edit)
  -> show.py, final render, listen to target then final, final verdict.
"""
from __future__ import annotations

import hashlib
import json
import math
import shlex
from pathlib import Path

from scripts import opencode_contract as oc
from scripts.agent_sft_common import compute_round_coverage, slices_for_regions

from .. import gate, narration
from ..ctx import Ctx, OracleRenderer
from ..mistakes import code_mistake
from ..planner import Edit, Plan, build_plan, vitallib, wt_ref
from ..record import Record, final_text
from .common import INIT, RENDER, listen, open_session, run, skill_response  # noqa: F401  (re-export)
from .judge import run_judge
from .searcher import WT, run_searcher
from .transcriber import run_transcriber

AGENT = "maestro"
USER_PROMPT = "Recreate the sound in target.wav in Vital."
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


class Main:
    def __init__(self, ctx: Ctx, code_mistake_rate: float = 0.08, mistake_rate: float = 0.10):
        self.ctx, self.rng, self.s = ctx, ctx.rng, ctx.sample
        self.gt, self.init, self.notes = self.s.gt_preset(), self.s.init_preset(), self.s.notes()
        self.code_rate = code_mistake_rate
        self.plan: Plan = build_plan(self.gt, self.init, self.rng, mistake_rate=mistake_rate)
        self.rec = Record(AGENT, f"{self.s.id}.main", ctx.wd, USER_PROMPT)
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

    # -- the rollout ---------------------------------------------------------------------------------------------

    def run(self) -> dict:
        ctx, rec, rng, s = self.ctx, self.rec, self.rng, self.s
        target, notes_path = ctx.path("target.wav"), ctx.path("notes.json")

        target_host = open_session(rec, ctx)
        rec.say(narration.first_impression(
            target_wav=Path(target_host), notes=self.notes_d, gt_preset=self.gt, renderer=self.renderer, log=ctx.note,
            slot=self.slot("first_impression", target=self.gt)))
        run(rec, ctx, f"cp {INIT} preset.json")

        # transcriber
        t_prompt = oc.transcription_dispatch_prompt(target, notes_path)
        t_rec = run_transcriber(ctx, t_prompt, f"{s.id}.transcriber")
        rec.say("First the notes.")
        call, resp = self.task("transcriber", 0, "Transcribe target melody", t_prompt, t_rec)
        rec.step([call], [resp])

        # searchers: the library, sliced
        run(rec, ctx, f"{WT} list --limit 0")
        gt_tables = [e for _, e in self.plan.wavetables]
        regions = self.coverage(gt_tables)
        n_tables = sum(b - a for a, b in regions)
        n_agents = rng.choice([2, 3])
        slices = slices_for_regions(regions, math.ceil(n_tables / n_agents))
        gt_probes = {}
        for e in gt_tables:
            p = json.loads(json.dumps(self.init))
            p["settings"]["wavetables"][0] = vitallib.wt_load(e)
            if e.get("silent_frame0"):
                p["settings"]["osc_1_wave_frame"] = 128.0
            gt_probes[e["id"]] = str(ctx.oracle_render(p, f"gt_{e['id']}"))
        calls, resps, pool = [], [], []
        for k, (a, b) in enumerate(slices, 1):
            prompt = oc.search_dispatch_prompt(target, notes_path, a, b)
            r, short = run_searcher(ctx, k, prompt, f"{s.id}.searcher{k}", a, b, gt_tables, gt_probes)
            call, resp = self.task("searcher", k, f"Search wavetables {oc.wavetable_slice_label(a, b)}", prompt, r)
            calls.append(call)
            resps.append(resp)
            pool += [x for x in short if x not in pool]
        what = "the whole library" if n_tables == 499 else f"{n_tables} of the 499 tables"
        rec.say(f"Searching {what} in {len(slices)} slices.")
        rec.step(calls, resps)

        # judge
        gt_refs = [wt_ref(e) for e in gt_tables]
        j_prompt = oc.judge_dispatch_prompt(target, pool, notes_path=notes_path)
        j_rec = run_judge(ctx, j_prompt, f"{s.id}.judge", pool, gt_refs)
        rec.say(f"{len(pool)} candidates in the pool; the judge picks the oscillator tables.")
        call, resp = self.task("judge", 0, "Judge wavetable pool", j_prompt, j_rec)
        rec.step([call], [resp])

        # rounds
        wt_edit = Edit("wavetable", "\n".join(f"{WT} set preset.json --osc {n} {shlex.quote(wt_ref(e))}"
                                             for n, e in self.plan.wavetables), {}, "wavetables")
        rounds = self.plan.rounds
        if rounds and rounds[0].subsystem == "oscillator":
            rounds[0].edits.insert(0, wt_edit)
        prev_wav, prev = None, ctx.read_json("preset.json")
        for i, r in enumerate(rounds, 1):
            nxt = rounds[i].title if i < len(rounds) else None
            for e in r.edits:
                self.edit(e)
            wav = f"r{i:02d}_{r.subsystem}.wav"
            run(rec, ctx, f"{RENDER} preset.json notes.json {wav}")
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
                run(rec, ctx, f"{RENDER} preset.json notes.json {wav}")
                host = Path(listen(rec, ctx, wav))
                cur = ctx.read_json("preset.json")
                rec.say(narration.self_edit(
                    before_wav=prev_wav, after_wav=host, before_preset=prev, after_preset=cur,
                    changed=_changed(prev, cur), subsystem=r.subsystem, notes=self.notes_d, renderer=self.renderer,
                    log=ctx.note, slot=self.slot("self_edit", before=prev, after=cur)))
                prev_wav, prev = host, cur

        # final check
        run(rec, ctx, "python3 $VITAL_SKILL/scripts/show.py preset.json")
        run(rec, ctx, f"{RENDER} preset.json notes.json final.wav")
        listen(rec, ctx, "target.wav")
        final_host = listen(rec, ctx, "final.wav")
        main = rec.finish(narration.final_verdict(target_wav=Path(target_host), final_wav=Path(final_host),
                                                  final_preset=ctx.read_json("preset.json"), gt_preset=self.gt,
                                                  notes=self.notes_d, renderer=self.renderer, log=ctx.note))
        return {"main": main, **self.sub, "links": self.links, "gate": self.gt_gate()}

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
        contiguous range (slices map to one wtA-wtB range each) holding every GT table (oracle: the GT reaches the
        pool)."""
        idx = [e["id"] for e in vitallib.wt_index()]
        need = {idx.index(e["id"]) for e in gt_tables}
        for _ in range(tries):
            r1, _ = compute_round_coverage(len(idx), self.rng.uniform(0.5, 1.0), self.rng)
            if len(r1) == 1 and all(any(a <= i < b for a, b in r1) for i in need):  # one contiguous window
                return r1
        return [(0, len(idx))]
