"""transcriber: listens to the target, writes notes.json, renders it through the init patch, listens, fixes, replies.

Notes are oracle (the GT clip). With probability p_wrong the first attempt is the GT corrupted by
agent_sft_common.apply_transcription_mutations (pitch / octave / deleted / inserted / shifted notes); after listening
to that render the agent says what is off (the mutation's own description) and rewrites the notes from scratch.
"""
from __future__ import annotations

from scripts.agent_sft_common import apply_transcription_mutations

from ..ctx import Ctx
from ..record import Record
from .common import INIT, RENDER, listen, notes_json, run

AGENT = "transcriber"


def run_transcriber(ctx: Ctx, prompt: str, rid: str, p_wrong: float = 0.5) -> dict:
    rng, s = ctx.rng, ctx.sample
    gt = s.notes()
    rec = Record(AGENT, rid, ctx.wd, prompt)
    listen(rec, ctx, "target.wav")

    attempts = []
    if rng.random() < p_wrong:
        mut = apply_transcription_mutations(gt, rng)
        if mut is not None:
            wrong, infos, narration = mut
            attempts.append((wrong, narration, infos))
    attempts.append((gt, None, []))

    for k, (notes, narration, muts) in enumerate(attempts):
        call, resp = ctx.write("notes.json", notes_json(s.bpm, notes))
        rec.step([call], [resp])
        wav = "transcription.wav" if k == 0 else f"transcription_{k + 1}.wav"
        run(rec, ctx, f"{RENDER} {INIT} notes.json {wav}")
        listen(rec, ctx, wav)
        if narration:
            rec.say(narration)
            ctx.note(agent=AGENT, event="wrong_transcription", mutations=muts)
    return rec.finish(f"{ctx.path('notes.json')}: {len(gt)} notes.")
