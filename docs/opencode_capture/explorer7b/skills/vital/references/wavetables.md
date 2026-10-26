# Vital wavetables — library conventions, auditioning, silent tables

## The entry you edit

`settings.wavetables[N]` (N = 0,1,2 → osc1, osc2, osc3) holds the **inline** table the
oscillator plays. Schema (from the library + init preset):

```json
{
  "author": "Matt Tytel",
  "name": "Basic Shapes",
  "version": "1.5.5",
  "full_normalize": false,
  "remove_all_dc": true,
  "groups": [
    {
      "components": [
        {
          "type": "Wave Source",            // or Audio File Source, Wave Warp, ...
          "interpolation": 1,
          "interpolation_style": 1,
          "keyframes": [
            {"position": 0.0, "wave_data": "<base64 float16 audio>"},
            {"position": 1.0, "wave_data": "<base64 float16 audio>"}
          ]
        }
      ]
    }
  ]
}
```

- `wave_data` is a **base64 string of float16 little-endian samples** (decode with
  `np.frombuffer(base64.b64decode(s), dtype='<f2')`). Never hand-edit it; swap whole
  entries.
- A table can be a chain of components (`type` = Wave Source / Audio File Source /
  Wave Warp / Line Source / Wave Folder / Frequency Filter / Wave Window / Phase Shift
  / Slew Limiter) inside one group, or multiple groups (X/Y-morphable, 1-4 groups in
  the library). `keyframes` position 0..1 = morph positions along the table.
- Library key variants: most entries have `full_normalize`; a few (v1.5.x exports) use
  `post_normalize` instead; 10 entries in the lib are completely empty `{}`. Copy
  entries verbatim — don't "normalize" the key set.

## The library: `/home/nate/Documents/maestro-llm/data/wavetable_lib.json`

568 entries; **301 unique (author, name) pairs, 282 unique names**; 10 empty entries.

Author conventions (who you can expect to find):
- **empty author `""`** = 351 entries — factory/unknown imports (names like "Sub",
  "Classic Blend", "Quad Saw", "TTWT", "Sine", "Harmonic Series"). These are the
  workhorses; the corpus uses them most (Sub, Classic Blend, Harmonic Series, TTWT).
- **Matt Tytel** (47) + the variant spelling **"Matt Tytelゑ"** (16) — factory content
  (Basic Shapes is the #1 table in the corpus; Sine, Classic Fade, ...).
- **Glorkglunk** (43), **woolf** (26), **Articks** (8), **David Dumais** (8),
  **Mioseg** (7), **SNFK** (5), **Flamedragonz** (5), **Yuli Yolo** (2), **Databroth**
  (1), **afro** (2), etc.
- Versions: 1.5.5 (360), 1.0.7 (81), 1.0.8 (31), 1.5.3 (28), 1.0.3 (27), 1.0.5 (11),
  older 0.x. Version is informational — all load into this 1.6.0 engine.
- Name collisions across authors exist: `Basic Shapes` ×3, `Init` ×13, `Resynthesize`
  ×4, `Sine to Saw` ×2, `Pink Noise` ×2. **Always match (author, name) as a pair** when
  swapping; if you have only a name, prefer the one the corpus actually uses (e.g.
  `Basic Shapes` → author "Matt Tytel").

Naming patterns that hint at content: `*Saw*`/`Basic Shapes` = rich harmonic;
`Sine*`/`Sub` = pure; `*Noise*`/`TTWT` (text-to-wavetable) = broadband; `*Blend*`/
`*Fade*` = morphing collections (many keyframes); `bell*`/`mallet*` = percussive;
`Resynthesize` = resampled sources; author packs (Databroth "Zine", woolf, Glorkglunk)
= stylistic.

## Auditioning a candidate (probe render)

```python
preset = json.load(open("preset.json"))
preset["settings"]["wavetables"][N] = lib_entry      # swap ONLY the target slot
preset["settings"]["osc_%d_on" % (N + 1)] = 1.0
preset["settings"]["osc_%d_level" % (N + 1)] = 1.0
preset["settings"]["osc_%d_wave_frame" % (N + 1)] = 0.0   # start of the table
# ... build_state → load_state → render the session notes → listen
```

- Render with the **session's notes.json** (not a single tone) so the table is judged
  in context. 0.4-0.8 s per note is enough.
- Judge one table at a time (SKILL RULE 5). Ask: does this table's character (harmonic
  density, noise content, morph movement) match what the target needs at this voice?
- For morphing tables (many keyframes), also probe `wave_frame = 0.5` — a table can be
  boring at frame 0 and interesting mid-table.

### Silent-table detection (MEASURED — 11 of 301)

Rendered every unique (author, name) in the library on a 0.4 s C4 note, osc1 only,
level 1.0. **These 11 render as digital silence (mono RMS < 1e-4) — label them, never
shortlist them:**

| author | name | RMS |
|---|---|---|
| (none) | meditation | 0.0 |
| (none) | Shimmer 2 | 0.0 |
| (none) | ICanHasKick | 0.0 |
| (none) | untitled | 0.0 |
| (none) | Ascension Ana 2 | 0.0 |
| (none) | SinII I | 0.0 |
| (none) | ESW BR Trumpeter 01 | 0.0 |
| MonalíciciaKenobii | Triangle - 8bit | 0.0 |
| (none) | exp | 3e-06 |
| (none) | pad | 4.5e-05 |
| (none) | synth | 2.9e-05 |

The rest (290) render non-silent. RMS rule for any render you do: `rms < 1e-4` ⇒
silent (dead table, osc level 0, or gated off) — re-check state before concluding
"the sound is empty". Two library entries also have corrupted component types (int
0/2 instead of a type string) — treat their names as suspect and verify by render.

## Morph / position keys (per oscillator)

- `osc_N_wave_frame` (0..256): position along the table; >1 is fine (wraps/extends).
  LFOs/macros here = "wobble/movement" — the #1 animated param in the corpus.
- X/Y 2D blend (between a table's groups when it has 2+): the engine exposes
  "Oscillator N Blend" (automatable, settable via `set_parameter`), but in this
  1.6.0 build it has **no preset JSON key** — it does not persist in preset.json.
  `osc_N_unison_blend` (0..1, in JSON) is a different thing: blends the unison
  voices' frames. `osc_N_view_2d` is the UI mode only.
- `osc_N_stereo_spread` (0..1): L/R phase offset of the wavetable read.
- `osc_N_random_phase` (0/1): randomize start frame per note (varies repeats).
- `osc_N_spectral_morph_{type,amount,spread}`: resynthesize the current frame with one
  of 17 spectral operations (Vocode, Formant Scale, Harmonic/Inharmonic Stretch,
  Smear, Random Amplitudes, Low/High Pass, Phase Disperse, Shepard Tone, ...) — amount
  0..1; type 0 None.

## Multi-osc assignment

- Each osc slot has its own inline table: assign different entries to
  `wavetables[0..2]` before one render (that's a "tuple" audition).
- Keep harmonic roles separate: e.g. saw/basic in osc1 (mids), sine in osc2 with
  `osc_2_transpose = -24` + `osc_2_destination = 4` (DIRECT OUT) as the sub, triangle/
  harmonics in osc3 an octave up. (Pattern verified in Banana Wob / Random Amp Growl.)
- Unison/stack (`osc_N_unison_voices`, `osc_N_unison_detune`, `osc_N_stack_style`)
  multiplies ONE table into a chord — use it instead of stacking 3 oscs when you want
  the same waveform thickened.
- Distortion-type FM/RM (see SKILL §1) needs a SOURCE osc to be on: e.g.
  `osc_1_distortion_type = 7` (FM←osc2) with `osc_2_on = 0` = no FM.
