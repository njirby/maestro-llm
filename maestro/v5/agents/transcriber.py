"""transcriber: loads the skill, listens to the target, writes notes.json, renders it through the init patch, listens, fixes, replies.

Notes are oracle (the GT clip, in the skill's measures-and-note-values form). With probability p_wrong
the first attempt is the GT corrupted by notation.mutate_transcription (wrong pitch / octave / dropped or
added note / note shifted off the grid); after listening to that render the agent says what is off (the
mutation's own description) and rewrites the notes from scratch. retry_weights[k] is the chance of k+1 wrong
attempts in a row (default one); every attempt after the first is closer (a single error).

existing=(notes, mutations, narration) is a re-check of a notes.json already in the workdir (the main agent heard
it was off): the agent renders and listens to what is there, says what is off, and writes the fix.
"""
from __future__ import annotations

from maestro import notation

from ..ctx import Ctx
from ..record import Record
from .common import INIT, RENDER, listen, open_session, run

AGENT = "transcriber"


def run_transcriber(ctx: Ctx, prompt: str, rid: str, p_wrong: float = 0.5, retry_weights: tuple[float, ...] = (1.0,),
                    existing: tuple[list[dict], list[dict], str] | None = None, final: list[dict] | None = None,
                    wav_prefix: str = "transcription") -> dict:
    """final: the beat-form notes the agent ends on (default the GT; a residual-error transcription the agent
    doesn't catch is only ever used for a record kept out of training, see agents/main.py)."""
    rng, s = ctx.rng, ctx.sample
    gt = s.notes()
    gt_b = notation.to_beat_form(s.bpm, gt)["notes"]
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)

    attempts = []
    if existing is not None:
        attempts.append((None, existing[2], existing[1]))   # already in notes.json: render it, don't rewrite it
    elif rng.random() < p_wrong:
        n_wrong = rng.choices(range(1, len(retry_weights) + 1), weights=retry_weights)[0] if len(retry_weights) > 1 else 1
        for j in range(n_wrong):
            mut = notation.mutate_transcription(gt_b, rng, n_muts=None if j == 0 else 1)
            if mut is not None:
                wrong, infos, narration = mut
                attempts.append((wrong, narration, infos))
    attempts.append((final if final is not None else gt_b, None, []))

    for k, (notes, narration, muts) in enumerate(attempts):
        if notes is not None:
            call, resp = ctx.write("notes.json", notation.beat_form_json(s.bpm, notes))
            rec.step([call], [resp])
        wav = f"{wav_prefix}.wav" if k == 0 else f"{wav_prefix}_{k + 1}.wav"
        run(rec, ctx, f"{RENDER} {INIT} notes.json {wav}")
        listen(rec, ctx, wav)
        if narration:
            rec.say(narration)
            ctx.note(agent=AGENT, event="wrong_transcription", mutations=muts)
    return rec.finish(f"{ctx.path('notes.json')}: {len(gt)} notes.")
