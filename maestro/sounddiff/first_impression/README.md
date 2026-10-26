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

Known follow-ups: dedupe movement items from multiple LFOs on one target; a "notes run into each other" fact from the
between-notes measure; extend check.py to brightness / note length / genre labels; relative (normalised) silence gate
for other modes; ~25% of preset_gen targets are very quiet (generator levels); f000 renders silent through the skill's
render.py but not render_pairs (renderer bug).
