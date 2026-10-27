"""transcriber: loads the skill, listens to the target, writes notes.json, renders it through the init patch, listens, fixes, replies.

Notes are oracle (the GT clip, in the skill's measures-and-note-values form). With probability p_wrong
the first attempt is the GT corrupted by notation.mutate_transcription (wrong pitch / octave / dropped or
added note / note shifted off the grid); after listening to that render the agent says what is off (the
mutation's own description) and rewrites the notes from scratch.
"""
from __future__ import annotations

from maestro import notation

from ..ctx import Ctx
from ..record import Record
from .common import INIT, RENDER, listen, open_session, run

AGENT = "transcriber"


def run_transcriber(ctx: Ctx, prompt: str, rid: str, p_wrong: float = 0.5) -> dict:
    rng, s = ctx.rng, ctx.sample
    gt = s.notes()
    gt_b = notation.to_beat_form(s.bpm, gt)["notes"]
    rec = Record(AGENT, rid, ctx.wd, prompt)
    open_session(rec, ctx)

    attempts = []
    if rng.random() < p_wrong:
        mut = notation.mutate_transcription(gt_b, rng)
        if mut is not None:
            wrong, infos, narration = mut
            attempts.append((wrong, narration, infos))
    attempts.append((gt_b, None, []))

    for k, (notes, narration, muts) in enumerate(attempts):
        call, resp = ctx.write("notes.json", notation.beat_form_json(s.bpm, notes))
        rec.step([call], [resp])
        wav = "transcription.wav" if k == 0 else f"transcription_{k + 1}.wav"
        run(rec, ctx, f"{RENDER} {INIT} notes.json {wav}")
        listen(rec, ctx, wav)
        if narration:
            rec.say(narration)
            ctx.note(agent=AGENT, event="wrong_transcription", mutations=muts)
    return rec.finish(f"{ctx.path('notes.json')}: {len(gt)} notes.")
