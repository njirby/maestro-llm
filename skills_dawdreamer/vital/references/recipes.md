# Vital recipes — complete topology specs

Each recipe: sections to enable, keys with example values (mined = from the 77-preset
corpus unless marked), mod routes, expected character, and how to verify by ear.
All times in the JSON log-seconds units from SKILL.md (1.0 ≈ 1 s). Values are starting
points — move them toward the target and re-render.

Tag legend: [MINED] = topology copied/adapted from a named corpus preset ·
[VERIFIED] = rendered + measured on this engine (numbers below) · [MANUAL] =
derived from the manual/community docs + engine probing.

---

## growl_bass — 3-osc detuned growl with macro wobble [MINED: Random Amp Growl, Banana Wob] [VERIFIED]

Sections: osc1+2+3 on, filter_1 on, distortion on, compressor on, (eq/chorus optional).
- osc1: `Basic Shapes` (Matt Tytel) frame ~60; `osc_1_transpose -24`;
  `osc_1_unison_voices 4` (4 voices), `osc_1_unison_detune 3.0`, `osc_1_detune_range 4`,
  `osc_1_detune_power 1.5`; `osc_1_distortion_type 4` (Bend) + `osc_1_distortion_amount 0.7`.
- osc2 (sub): `Sine`, `osc_2_transpose -24`, `osc_2_level 0.5`, `osc_2_destination 4`
  (DIRECT OUT) — sub skips the filter.
- osc3 (bite): `Basic Shapes` frame 120, `osc_3_transpose -12`, `osc_3_level 0.4`,
  optional `osc_3_distortion_type 5` (Squeeze) amount 0.3.
- env1: `attack 0` (0.0), `decay 0.5`, `sustain 0.8`, `release 0.25`.
- filter1: Analog, `filter_1_cutoff 38` (dark), `filter_1_resonance 0.7`,
  `filter_1_drive 8`, `filter_1_mix 1`.
- distortion: `distortion_type 1` (Hard Clip), `distortion_drive 6`, `distortion_mix 1`,
  `distortion_filter_cutoff 60`, `distortion_filter_resonance 0.5`.
- compressor: `compressor_enabled_bands 3` (Single Band), attack 0.5, release 0.3.
- lfo1: `lfo_1_sync 1` (Tempo), `lfo_1_tempo 8` (1/8); **mods**: `lfo_1 → filter_1_cutoff`
  amount 0.25; `env_1 → filter_1_cutoff` amount 0.6 (scoop). Optional macro1 →
  `osc_1_wave_frame` 0.35 + `osc_1_distortion_amount` -0.2 (growl taste knob, mined).
Character: low end (centroid ~1 kHz), buzzy mid, pumping on each note; sub is clean.
Verify: b0-300 band > 0.35 of energy, zcr 0.05-0.1, centroid < 1.5k for bass register;
A/B the lfo route on/off = the "wobble" appears/disappears.
Measured (this build): rms 0.143, centroid 1007 Hz, bands 0-300: 0.498, zcr 0.081
(baseline init saw: centroid 4981) — distinct ✓.

## fm_bell — sine carrier + FM sidebands, long decay [MINED pattern: Crescendo Bells + FM Drum Circle] [VERIFIED]

Sections: osc1+2 on, filter_1 on (gentle), reverb on.
- osc1: `Sine`, `osc_1_level 1.0`, `osc_1_wave_frame 0`.
- osc2 (modulator): `Sine`, `osc_2_level 0.7`, `osc_2_transpose 12` (2:1 ratio =
  classic bell; +19 for inharmonic clang).
- `osc_1_distortion_type 7` (FM←osc2), `osc_1_distortion_amount 0.5-0.8`.
  (Type 8 = FM from the OTHER osc; 9 = FM←sample — needs a sample.)
- env1: `attack 0`, `decay 1.6` (~10 s region — start 1.2 and move), `sustain 0`,
  `release 1.2` — hard attack, exponential-looking decay = bell.
- filter1: Analog, `filter_1_cutoff 100`, `resonance 0.2`, `mix 1` (bright, open).
- reverb: `reverb_decay_time 1.3` (~2.5 s), `reverb_dry_wet 0.35`, `size 0.7`.
- Optional evolution: `lfo_1 (Seconds, ~2 s) → osc_1_distortion_amount` amount 0.3
  (FM index drifts = shimmer).
Character: instant ping, long metallic tail, rich mid + upper harmonics.
Verify: attack_tail_ratio (first-20% RMS / last-30% RMS) >> 1 (measured 135 = proper
decay); centroid above the plain-sine baseline; sidebands visible in b4k_10k/b10k_20k.
Measured: rms 0.135, centroid 5037 Hz, band-L1 vs baseline 0.419, mel-cos 0.995 ✓.

## evolving_pad — slow LFO drift, wide, long release [MINED: Float Chords] [VERIFIED]

Sections: osc1+2 on, filter_1 on, chorus on, delay on, reverb on, compressor on.
- osc1: `Basic Shapes` frame ~40, `osc_1_level 0.8`, `osc_1_unison_voices 8` (8 voices),
  `osc_1_unison_detune 4.0`, `osc_1_detune_range 2`, `osc_1_stereo_spread 1`.
- osc2: `Basic Shapes` frame ~120, `osc_2_transpose -12`, `osc_2_level 0.4`.
- env1: `attack 1.4` (~4-6 s swell), `sustain 1.0` (RULE 12!), `release 2.0` (~16 s).
- filter1: Analog, `filter_1_cutoff 55`, `resonance 0.3`, `mix 1`.
- lfo1: `lfo_1_sync 0` (Seconds), `lfo_1_frequency -2` (4 s) → **mod** `osc_1_wave_frame`
  amount 0.5 (table drift = "evolving").
- lfo2: Seconds, `frequency -3` (8 s) → **mod** `filter_1_cutoff` amount 0.3.
- chorus: `chorus_frequency -3` (8 s), `mod_depth 0.6`, `chorus_dry_wet 0.5`.
- delay: `delay_style 1` (Stereo), `delay_sync 1`, `delay_tempo 8` (1/8),
  `delay_feedback 0.35`, `delay_dry_wet 0.25`.
- reverb: `reverb_decay_time 1.8` (~3.5 s), `reverb_dry_wet 0.4`, `size 0.8`.
Character: slow-breathing wash, wide stereo, no transients. Verify: sustained RMS
across the note (attack_tail_ratio modest ~10, unlike bells); spectrum drifts over
time — render 5 s and compare first-1s vs last-1s centroids (should differ > 5%).
Measured: rms 0.069, centroid 2374 Hz, band-L1 vs baseline 0.648, mel-cos 0.986 ✓.

## supersaw_lead — unison stack + chorus, bright [MANUAL]

Sections: osc1 on, filter_1 on, chorus on, delay on (optional), compressor on.
- osc1: `Basic Shapes` (or a *Saw* table), `osc_1_level 0.9`,
  `osc_1_unison_voices 11-15` (8-16v), `osc_1_unison_detune 5-8`,
  `osc_1_detune_range 3`, `osc_1_stereo_spread 1`.
- Optional osc2: same table, `osc_2_unison_voices 7`, `osc_2_transpose 12`
  (octave spread) or `osc_2_stack_style 3` (Octave) instead.
- env1: `attack 0.05`, `decay 0.6`, `sustain 0.9`, `release 0.6`.
- filter1: `filter_1_style 1` (24dB), `cutoff 90-100`, `resonance 0.2-0.4`, `mix 1`.
- mods: `env_1 → filter_1_cutoff` 0.3 (initial open); `velocity → filter_1_cutoff` 0.2.
- chorus: `frequency -4` (16 s), depth 0.5, wet 0.4; delay: 1/4 dotted, fb 0.3, wet 0.2.
Character: thick, wide, aggressive lead. Verify: high unison density (low zcr but
very high b1k_4k), width (L/R correlation low), sustain holds.

## pluck — delayed attack, random per-note variation [MINED: Super Nice Pluck]

Sections: osc1 (+osc2 optional) on, filter_1 on, distortion on (light), reverb on,
compressor on, sample on (optional color).
- osc1: any harmonic table (e.g. `Basic Shapes` frame 90), `osc_1_level 0.8`.
- env1: `delay 0.32` (~0.25 s gap before note), `attack 0`, `decay 0.77` (~0.5 s),
  `sustain 0.005`, `release 0.92` — the "pluck" IS this envelope.
- mods (the mined pattern): `random_1 → osc_1_wave_frame` 0.62,
  `random_1 → osc_1_distortion_amount` 0.5, `random_1 → osc_1_spectral_morph_amount`
  0.14 (per-note variation); `env_1 → filter_1_cutoff` 0.26, `env_1 → osc_1_level` 0.19.
- filter1: Analog, `cutoff 48`, `resonance 0.5`, `mix 1`.
- distortion: Soft Clip, `drive 11`, `mix 1`, `distortion_filter_cutoff 80`.
- reverb: `reverb_decay_time 1.4` (~2.6 s), `reverb_dry_wet 0` base +
  **macro3 (PAD) → reverb_dry_wet amount 1.0** (space knob, mined).
- Optional macros: macro1 (LENGTH) → `env_1_decay` 0.15 + `env_1_release` 0.3 (mined).
Character: dry snap per note + per-note timbre jitter; reverb knob = "pad-ness".
Verify: each repeated note sounds slightly different (random routes), gap before attack
(delay), fast decay to silence.

## hard_bass — square/saw + sync distortion + 24dB lowpass sweep [MANUAL, corpus-informed]

Sections: osc1 on, filter_1 on, distortion on, compressor on.
- osc1: square-ish table (frame 0 of `Basic Shapes`), `osc_1_transpose -12`,
  `osc_1_level 0.9`, `osc_1_distortion_type 1` (Sync) `amount 0.5-0.8` (hard sync =
  aggressive), or type 4 (Bend) for softer growl.
- env1: `attack 0`, `decay 0.4`, `sustain 0.7`, `release 0.2`.
- filter1: `filter_1_style 1` (24dB), `cutoff 35` base, `resonance 0.4`, `mix 1`.
- mods: `env_1 → filter_1_cutoff` **0.5-0.8** (the sweep — this route is THE hard bass
  sound), `velocity → filter_1_resonance` 0.2 (optional).
- distortion: Soft Clip drive 4-6, mix 1, filter Post cutoff 50.
Character: punchy, defined low end, fast envelope. Verify: strong b0-300, fast
attack→decay per note (attack_tail_ratio 2-5), no sizzle above 3k unless you want it.

## noisy_drone — quantize/pulse distortion + very slow LFOs [MANUAL]

Sections: osc1+2 on, filter_1 on (dark), reverb on (big).
- osc1: any harmonic table, `osc_1_level 0.7`, `osc_1_distortion_type 3` (Quantize)
  or 6 (Pulse), `amount 0.5-0.9`, `osc_1_distortion_phase` to taste (0..1).
- osc2: same table detuned `osc_2_detune_range 8` (beating).
- env1: `attack 1.2` (~4 s), `sustain 1.0`, `release 2.0` (RULE 12).
- filter1: Analog, `cutoff 25-35`, `resonance 0.3`, `mix 1`.
- lfo1: Seconds `frequency -5` (32 s) → `osc_1_distortion_phase` 0.4 (creep).
- lfo2: Seconds `frequency -4` (16 s) → `filter_1_cutoff` 0.2.
- reverb: `decay_time 2.5` (~5.7 s), `wet 0.5`, `size 0.9`.
Character: granular/noisy sustained bed, slow evolution. Verify: high zcr (noise-like,
> 0.1), low centroid, near-constant RMS over 10 s (drone, not note).

## formant_voice — formant filter vowel morph [MINED: Factory "Ah Eh Ee Oh"]

Sections: osc1 on (vocal-ish or vowel table), filter_1 on **model 5 (Formant)**,
reverb on (small).
- osc1: any voiced/noisy table (or `sample_on` with a voice sample, `sample_loop 1`),
  `osc_1_level 0.9`.
- filter1: `filter_1_on 1`, `filter_1_model 5` (Formant),
  `filter_1_formant_resonance 0.85` (default), `filter_1_formant_spread 0.3`.
  **Vowels = `filter_1_formant_x` / `filter_1_formant_y`** (0..1 grid):
  (0.5, 0.5) ≈ "a/ah", (0.5, 0.9) ≈ "ee/eh", (0.5, 0.1) ≈ "oo/oh",
  (0.3, 0.7) ≈ "eh", (0.7, 0.5) ≈ "uh". (Verify each by render — the grid is
  not linearly named in the engine.)
- env1: `attack 0.3`, `sustain 1.0`, `release 0.8`.
- mods: LFO Seconds ~1-2 s → `filter_1_formant_y` 0.4 = "vocalizing" morph
  (mined pattern from Ah Eh Ee Oh + formant usage).
Character: vowel-tinted voice/moog-voice. Verify: strong 500-3k band with two visible
formant peaks (not a plain harmonic stack); moving formant_y changes "aah"→"eeeh"
audibly.

## lfo_wobble — tempo-synced LFO on cutoff, square shape [MINED: Banana Wob, LFO corpus patterns]

Sections: osc1 on, filter_1 on, distortion on (optional), compressor on.
- osc1: `Basic Shapes` or sub-sine, `osc_1_transpose -12` (bass register),
  `osc_1_level 0.9`.
- filter1: Analog, `cutoff 40` base (below the LFO peak), `resonance 0.6`, `mix 1`.
- lfo1: `lfo_1_sync 1` (Tempo — DEFAULT, keep), `lfo_1_tempo 8` (1/8) or 9 (1/16);
  `lfo_1_keytrack_transpose 0` (constant rate across keyboard — default -12 makes
  wobble faster on low notes, which can be desired).
- **LFO shape**: edit `settings.lfos[0]` to a square — 5 corner points (x,y pairs,
  all 0..1): `{"name": "Square", "num_points": 5,
  "points": [0.0, 0.0, 0.0, 1.0, 0.5, 1.0, 0.5, 0.0, 1.0, 0.0], "powers": [0,0,0,0,0],
  "smooth": false}` (rise at x=0, hold high to x=0.5, fall, hold low). Triangle
  (init default, 3 points) = smooth wobble; square = hard "dubstep" wobble.
- mods: `lfo_1 → filter_1_cutoff` amount **0.4-0.6** (12-25 unit swing).
- Optional: macro1 → `osc_1_wave_frame` 0.3 + `filter_1_cutoff` 0.2 (taste, mined).
Character: rhythmic filtered-pump in the bass. Verify: periodic brightness pulses at
the exact tempo subdivision (count peaks vs notes: 2 per note for 1/8 at the render's
BPM); A/B `lfo_1_tempo` 8↔9 = wobble doubles rate.
Measured (shape A/B on this engine, 1/8 wobble at 120 BPM on a low note): triangle
centroid 1822 Hz vs square 2331 Hz — the square shape is ~28% brighter because it
holds the filter open longer.

## ambient_wash — ping-pong delay + big reverb + slow randoms [MINED: Smoker's Lounge pattern]

Sections: osc1+2 on, reverb on, delay on, (phaser/flanger optional), compressor on.
- osc1: soft harmonic table, `osc_1_level 0.6`, `osc_1_unison_voices 4`,
  `osc_1_detune_range 3`; osc2: same table `transpose -7` (minor 3rd shimmer),
  `level 0.3`.
- env1: `attack 1.5`, `sustain 1.0`, `release 2.5`.
- delay: `delay_style 3` (Mid Ping Pong), `delay_sync 1`, `delay_tempo 7` (1/4) or
  aux at `delay_aux_tempo 5` (1/2), `delay_feedback 0.6`, `delay_dry_wet 0.3`.
- reverb: `decay_time 3.0` (8 s), `wet 0.5`, `size 0.9`, `pre_low_cutoff 20` (keep
  sub out of the reverb).
- lfo1: Seconds 16-32 s → `osc_1_wave_frame` 0.3; random_1 → `osc_2_level` 0.3
  (breathing).
- compressor: Single Band, gentle (ratio ~3) to glue the tails.
Character: huge, slow, panning space; individual notes barely visible. Verify: long
RMS tail (seconds after the note), alternating L/R energy (ping-pong), very low zcr.

## sub_bass — pure sine sub routed direct-out [MINED: Banana Wob / Random Amp Growl sub voice]

Sections: osc2 on (as sub) + whatever main osc; NO filter in the sub's path.
- osc2: `Sine` (or `Sub`), `osc_2_transpose -24`, `osc_2_level 0.4-0.6`,
  `osc_2_destination 4` (**DIRECT OUT** — bypasses both filters and reaches the FX
  chain end), `osc_2_unison_voices 1` (pure), `osc_2_detune_range 0`.
- env1 (shared): `attack 0.02`, `decay 0.5`, `sustain 0.8`, `release 0.2`.
- Optional: `velocity → osc_2_level` 0.2 (punch), macro → `osc_2_level` 1.0 with base
  `osc_2_level 0` (sub toggle, mined Banana Wob pattern).
Character: clean low-end under any top sound. Verify: b0-300 isolated energy, near-zero
harmonics above 1k from the sub itself (A/B: turn `osc_2_on 0` = sub gone),
no filter resonance color on it.

## fm_kit — FM/RM percussive tones [MINED: Factory "FM Drum Circle" template + Ceramic]

Sections: osc1+2+3 on (as needed), filter_1 on (bright), no reverb (or tiny).
- Per-voice template (bells/marimba): carrier osc = `Sine`; `osc_1_distortion_type 7`
  (FM←osc2), `osc_2` = `Sine` at `transpose 12` or `+19`; `amount 0.3-0.9`.
  Toms/kick color: RM — `osc_1_distortion_type 10` (RM←osc2), osc2 any table,
  amount 0.5-1.0 (mined: Ceramic uses osc1 t10 amt ~1.0 + osc2 t11 amt 0.5).
- env1: `attack 0`, `decay 0.5-0.8` (2-5 s? no — 0.5-0.8 ≈ 0.4-1 s), `sustain 0`,
  `release 0.3`.
- Velocity expression: `velocity → osc_1_distortion_amount` 0.3 (harder hit = more FM).
- Different pitch per note via the notes file (each drum = different MIDI pitch);
  `osc_N_transpose` sets the register.
Character: pitched metallic/wood hits with fast decay. Verify: per-note pitch follows
the MIDI (FM sidebands scale with the note), hard attack, quick decay
(attack_tail_ratio > 20), no sustain.

---

## Cross-recipe notes

- **Every recipe needs the gates** (RULE 1) — "filter_1 on + cutoff" is one edit, not two.
- **Bass register** = transpose -12..-24 on the osc(s); targets in the 30-60 Hz region
  use -24.
- **Clean vs gritty**: grit = osc distortion types (1-6) and/or the FX Distortion;
  clean = type 0 (None) + low drive.
- **Movement budget**: one LFO/route per perceptual feature (wobble, drift, swell).
  Three moving things at once = unjudgeable render.
- When a recipe's measured character doesn't match the target, change ONE subsystem per
  render (SKILL RULE 3) — do not re-tune a whole recipe at once.
