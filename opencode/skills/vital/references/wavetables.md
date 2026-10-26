# Vital wavetables — library, auditioning, frames

## The entry you edit

`settings.wavetables[N-1]` holds the **inline** table oscillator N plays. Schema:

```json
{"author": "Matt Tytel", "name": "Basic Shapes", "version": "1.5.5",
 "full_normalize": false, "remove_all_dc": true,
 "groups": [{"components": [{"type": "Wave Source", "interpolation": 1,
   "interpolation_style": 1,
   "keyframes": [{"position": 0, "wave_data": "<base64 float16>"}, ...]}]}]}
```

- Never hand-edit `wave_data`; swap whole tables with
  `wavetable.py set preset.json --osc N <name|id>` (copies the library entry verbatim).
- A table is a chain of components (Wave Source / Audio File Source / Line Source / Wave
  Warp / Wave Folder / Frequency Filter / Wave Window / Phase Shift / Slew Limiter) in
  1-4 groups. `keyframes` are the morph positions along the table.

## The library (`data/wavetables.idx.json` + `data/wavetables.bin`)

499 distinct tables (every distinct table of the generator's 568-entry library; ids
`wt001`..`wt499`). `wavetable.py list --query Q` searches name, author and source kind:

- `wave` — drawn / generated single-cycle frames (Basic Shapes, Sine, Classic Blend,
  Harmonic Series...): clean, pitched, the workhorses.
- `audio` — resampled audio (vocals, instruments, noise, "ESW ..." packs): grainy,
  formant-y, often noisy; frame position changes the sound a lot.
- `line` — Line Source shapes (Vital Sine, simple ramps): very plain.

Names repeat across authors and edits: "Init" alone has ~95 different tables (users'
edited init tables), "Basic Shapes" 25. A name in `set` picks the FIRST variant; when
you want a specific one, use its id from `list`. Author conventions: empty author =
factory/unknown imports; "Matt Tytel" = factory; Glorkglunk, woolf, Articks, David
Dumais, Mioseg, SNFK, Flamedragonz = third-party packs.

Name hints: `*Saw*` / Basic Shapes = rich harmonics; `Sine*` / `Sub*` = pure; `*Noise*`
/ `TTWT` = broadband; `*Blend*` / `*Fade*` = morphing collections (many frames);
`bell*` / `mallet*` = percussive; author packs = stylistic.

## Frames: silent at frame 0 (re-measured)

Every table in the library sounds at some frame — none is fully silent. 18 tables are
silent (or nearly) **at frame 0** and only sound further in (they start with a blank
keyframe, typical of resampled audio). `list` marks them `silent at frame 0`; with one
of these, set `osc_N_wave_frame` to e.g. 128. (Measured with this skill's render.py path: C4, 0.4 s,
osc 1 alone at level 1, frames 0 / 128 / 256.) The old "11 silent wavetables" list was
wrong: it was measured at frame 0 only.

## Auditioning (ears only)

- Render with the session's own notes.json so a table is judged in context.
- Judge one candidate at a time against the target, then move on. Shortlist by name /
  kind first (2-4 candidates), never page through the whole library one by one.
- For morphing tables (many frames) also try `osc_N_wave_frame` 64 / 128 / 192 — a table
  can be dull at frame 0 and right mid-table.
- If a swap "changes nothing", check `show.py preset.json osc` (osc off, level 0, or
  the table is silent at the current frame).

## Morph / position keys (per oscillator)

- `osc_N_wave_frame` (0..256): position along the table. LFOs/macros here = the #1
  animated parameter in the corpus ("wobble/movement").
- `osc_N_frame_spread` (-128..128): unison voices read different frames.
- `osc_N_unison_blend` (0..1): blend between the unison voices' frames.
- `osc_N_stereo_spread` (0..1): L/R offset of the wavetable read.
- `osc_N_random_phase` (0..1): randomize the start phase per note (repeats vary).
- `osc_N_spectral_morph_{type,amount,spread}`: resynthesize the current frame (Vocode,
  Formant Scale, Harmonic/Inharmonic Stretch, Smear, Random Amplitudes, Low/High Pass,
  Phase Disperse, Shepard Tone, Spectral Time Skew, ...); type 0 = None makes amount dead.
- The engine's 2-D "Oscillator N Blend" control has no preset JSON key in this build —
  it cannot be set from preset.json.

## Multi-osc assignment

- Each osc slot has its own inline table: assign different tables to osc 1-3 for a
  layered sound.
- Keep roles separate: e.g. saw/basic in osc 1 (mids), `Sine` in osc 2 with
  `osc_2_transpose=-24` + `osc_2_destination="DIRECT OUT"` as the sub, harmonics in osc 3
  an octave up (pattern from Banana Wob / Random Amp Growl).
- Unison/stack (`osc_N_unison_voices`, `osc_N_unison_detune`, `osc_N_stack_style`)
  thickens ONE table into a chord — prefer it over stacking 3 oscs for the same wave.
- FM/RM distortion types need their SOURCE osc switched on: `osc_1_distortion_type=
  "FM <- osc 2"` with `osc_2_on=0` = no FM.
