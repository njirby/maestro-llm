# First-impression recipe (prototype, 2026-10-25)

Agreed recipe for the narration of a target sound's first impression:

1. **Verified session notes** (`fi_notes.py`): for each target preset,
   - musical facts from the ground-truth MIDI (single line vs chords, register, note length as heard, density; no tempo);
   - **ablation-verified modules**: re-render the target with each module switched off (FX, mod routes, unison,
     oscillators, filters); a module is audible only if the switch-off changes the render beyond the render-noise
     floor (target rendered twice); strength subtle / clear / strong from the log-mel distance;
   - tone from the target render ranked against 480 reference renders (brightness, harmonic richness kept separate from
     distortion, noisiness, stereo width);
   - every core dimension stated explicitly (space, movement, pitch, distortion, stereo, tone), in listener words.
2. **One Qwen3-Omni-30B-A3B-Instruct call**: the target audio (peak-normalised, mono) + the notes + a zero-shot
   instruction (no examples, no sampling params; server defaults).
3. **Check + one retry** (`check.py`): reject empty / non-English output and claims the notes rule out
   (clause-level negation); regenerate once with the same prompt (never post-edit).

Result on 39 probe1 targets x 2 samples: 1/78 flagged after retry (12% on the first try) vs 54% for the same model with
an audio-only prompt. Human spot checks: mostly true or partly true.

Scratch-era scripts, kept as they ran: `fi_notes.py` (prep / render / notes / run), `check.py`, `omni_fi.py` (side-by-side
page + audio-only arm), `oscshape.py` (waveform-representation head-to-head: text harmonic profile beat images),
`iterate.py` (27B narrator loop). Paths inside point at outputs/sounddiff/{probe1,fi_notes,omni_fi,oscshape}.

Library (2026-10-25): `from maestro.sounddiff.first_impression import notes_first_impression, narrate_first_impression`
(API in `__init__.py`; notes from `maestro/sounddiff/verified.py`, renders through the skill's `render.py` in the daw-farm
image via `maestro/sounddiff/renderer.py`, checker `maestro/sounddiff/check.py`). Probe CLI for regression:
`python -m maestro.sounddiff.first_impression notes|regress|run`. Round-4 follow-ups done there: movement items deduped
(one per heard effect), a continuity fact ("the notes run into each other" / "clean silence between the notes": gap level
between notes, or a part written back to back whose notes hold - f038), and checks for brightness, note length /
continuity and genre labels ("pad" on plucky parts, "pluck" on held notes, "bass" on high parts), tested in
`maestro/sounddiff/tests/`. Switching from render_pairs to the skill's render.py moved a few near-threshold facts
(`regress` lists them: e.g. f003 "a touch of reverb" -> dry).

Still open: relative (normalised) silence gate for other modes; ~25% of preset_gen targets are very quiet (generator
levels, WS-C); f000 renders silent through the skill's render.py (WS-C).
