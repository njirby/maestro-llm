---
name: vital
description: Recreate a target synth sound in Vital (1.6.0 VST3) by editing preset.json and rendering it headless through DawDreamer, judging every render by ear. Use when the target was produced with (or should be recreated with) Vital, a wavetable synth with 3 wavetable oscillators + sample player, 6 envelopes, 8 LFOs, 4 random generators, two filters + FX filter, an 8-unit FX chain, a 64-slot modulation matrix and 4 macros. Ships scripts to render (render.py), set validated parameters (set.py), swap wavetables from a 499-table library (wavetable.py), edit modulation routes (mod.py) and show what is live in the patch (show.py). Invoke before listening to the target whenever the context indicates Vital or a wavetable-synthesis target.
metadata:
  plugin: vital
  paradigm: wavetable
  subsystems: oscillator,envelope,filter,lfo,fx,modulation,macro
---

# Vital skill

## Working model

- **`preset.json` (in your working directory) is the ONLY synth state** — a Vital 1.6.0 preset:
  top level `author, comments, macro1..4` (macro label strings), `preset_name, preset_style,
  synth_version, settings`. `settings` holds ~776 parameters plus `wavetables[3]`, `lfos[8]`
  (LFO shapes), `modulations[64]`, `sample`. Editing the file = changing the synth.
- **`notes.json`** = the melody and tempo, written the way a musician writes them:
  `{"bpm": 120, "time_signature": "4/4", "notes": [{"pitch": 60, "velocity": 100, "bar": 1,
  "beat": 1, "duration": "eighth"}, ...]}`. `bar`/`beat` place the note on the grid (beat 1 is the
  first beat of the bar; fractions land on sixteenths); `duration` is a note value:
  `whole`, `dotted half`, `half`, `dotted quarter`, `quarter`, `dotted eighth`, `eighth`,
  `dotted sixteenth`, `sixteenth`, `dotted thirty-second`, `thirty-second`, or a triplet
  (three in the time of two): `triplet quarter`, `triplet eighth`, `triplet sixteenth`,
  `triplet thirty-second`. `bpm` and `time_signature` are optional (120 and 4/4); a bare
  list of notes also works.
- **You judge by ear.** Render, then listen to the wav; compare with the target one audio at a
  time. There are no analysis tools on purpose.
- Scripts live in `$VITAL_SKILL/scripts/` (`$VITAL_SKILL` = this skill's base directory; it is
  set in the render environment). They use python3 + numpy + soundfile + dawdreamer only, every
  one has `--help`, exits 2 with `error: ...` on bad input and never half-writes a preset.

## Scripts

```bash
python3 $VITAL_SKILL/scripts/render.py preset.json notes.json out.wav [--bpm N] [--tail S]
#  -> out.wav  (notes 2.80 s, 128 bpm)
python3 $VITAL_SKILL/scripts/set.py preset.json filter_1_on=on filter_1_cutoff=900Hz env_1_release=400ms
#  -> filter_1_cutoff: 60 (262 Hz) -> 81.1 (900 Hz)          (one line per key)
python3 $VITAL_SKILL/scripts/wavetable.py list --query saw        # or --ids wt101-wt166 (a chunk)
python3 $VITAL_SKILL/scripts/wavetable.py set preset.json --osc 2 "Basic Shapes"   # or an id: wt012
python3 $VITAL_SKILL/scripts/mod.py preset.json add lfo_1 filter_1_cutoff 40%      # list | add | remove
python3 $VITAL_SKILL/scripts/show.py preset.json [osc|sample|env|filter|lfo|random|mod|fx|macro|global|KEY_PREFIX]
```

- **render.py** loads the full state (wavetables, LFO shapes, mod matrix), sets the host tempo
  from notes.json `bpm` (or `--bpm`), flushes the synth with a silent pre-roll (cut from the
  output), and writes 32-bit float stereo 44.1 kHz: the notes plus a fixed 3 s tail after the
  last note-off (`--tail S` to change; raise it for long reverbs/releases). About 1 s per render.
- **set.py** values: raw JSON numbers (units table below), enum labels (`filter_1_model=Formant`,
  `osc_1_destination="DIRECT OUT"`, `reverb_on=on`, `lfo_1_tempo=1/8`), or units: `Hz`/`kHz`
  (cutoff/center keys), `ms`/`s` (envelope times), `%` (0..1 and -1..1 keys), `dB` (volume,
  drive, gain), `st` (transposes). Unknown keys get suggestions; out-of-range values list the
  valid range/labels. `--dry-run` validates only.
- **show.py** with no section = one-screen overview of what can be heard; a section or key
  prefix (`filter_1_`, `reverb_`, `osc_2_unison`) lists keys as `key = raw (human)` with `*` for
  non-default values, ranges/labels, and `! <reason>` for keys that cannot affect the sound.
- **mod.py** `add` updates an existing SRC→DST route or takes the first free slot, sets the amount
  (non-zero, -1..1 or ±100%, e.g. `-30%`) and clears bypass; `remove` empties the slot. Optional
  `--bipolar on|off --power P --stereo on|off`.
- Things without a script (LFO shapes, sample import, macro labels) are edited in the JSON.

## Section map

**1. Oscillators 1-3** — gate `osc_N_on` (osc 1 on by default, 2/3 off). `osc_N_level`,
`osc_N_transpose` (st), `osc_N_tune`, `osc_N_wave_frame` (0..256 table position),
`osc_N_unison_voices` (1..16) + `osc_N_unison_detune` + `osc_N_detune_range` (cents) +
`osc_N_stereo_spread`, `osc_N_stack_style` (Unison, Center Drop 12/24, Octave, Octave + 7,
2x Octave, Power Chord, Major/Minor Chord, Harmonics, Odd/Sub Harmonics).
- **FM / RM are `osc_N_distortion_type` values** (no separate FM section): None, Sync, Formant,
  Quantize, Bend, Squeeze, Pulse, `FM <- osc A/B`, `FM <- Sample`, `RM <- osc A/B`,
  `RM <- Sample`. For osc 1 the other oscs are 2 and 3 (osc 2: 3 and 1; osc 3: 1 and 2) —
  set.py and show.py print the real osc number. The source osc must be on.
  `osc_N_distortion_amount` does nothing while the type is None.
- `osc_N_spectral_morph_type` (Vocode, Formant Scale, Harmonic/Inharmonic Stretch, Smear,
  Random Amplitudes, Low/High Pass, Phase Disperse, Shepard Tone, ...) + `_amount`.
- `osc_N_destination` routes the osc: FILTER 1, FILTER 2, FILTER 1+2, EFFECTS, DIRECT OUT,
  or straight into one FX unit. Defaults osc1→FILTER 1, osc2→FILTER 2, osc3→EFFECTS.
  A filter that is off passes its input unfiltered; DIRECT OUT skips filters and FX (sub trick).

**2. Sample player** — `sample_on` (off) + `sample_{level,pan,transpose,tune,loop,bounce,
keytrack,destination}`. Audio in `settings.sample = {length, name, sample_rate, samples}` with
`samples` = base64 of float16 little-endian. Init ships "White Noise".

**3. Envelopes 1-6** — `env_N_{delay,attack,hold,decay,sustain,release}` + `_power` curves
(-20..20). **env 1 is the amp envelope: when its release ends, the voice (and its LFO/random
motion) stops.** Pads: `env_1_sustain` near 1. env 2-6 only act through mod routes.

**4. Filters 1, 2, FX** — gates `filter_1_on`, `filter_2_on`, `filter_fx_on`, **all off by
default**. `filter_U_{model,style,cutoff,resonance,drive,mix,keytrack,blend,formant_x,formant_y,
...}`. Models: Analog, Dirty, Ladder, Digital, Diode, Formant, Comb, Phaser. Styles: 12dB,
24dB, Notch Blend, Notch Spread, B/P/N. With Formant, `formant_x/y` (0..1) pick the vowel
(center ≈ "ah", high y ≈ "ee", low y ≈ "oo" — confirm by ear). `filter_1_filter_input` = serial
routing: filter 2's output also feeds filter 1 (and vice versa for `filter_2_filter_input`).

**5. LFOs 1-8, random 1-4** — **shape is data, not a parameter**: `settings.lfos[N-1] =
{name, num_points, points: [x0,y0,x1,y1,...] (0..1), powers, smooth}` (init: Triangle; square =
`[0,0, 0,1, 0.5,1, 0.5,0, 1,0]`, 5 points). `lfo_N_sync`: Seconds | **Tempo (default)** |
Tempo Dotted | Tempo Triplets | Keytrack; tempo-synced rate = `lfo_N_tempo` (Freeze, 32/1 …
1/2 (default), 1/4, 1/8, 1/16, 1/32, 1/64); Seconds mode uses `lfo_N_frequency`.
`lfo_N_sync_type`: Trigger (default, restarts per note), Sync, Envelope (one-shot per note),
Sustain Envelope, Loop Point, Loop Hold, Point Cycle. `lfo_N_keytrack_transpose` default -12 (rate follows
the note); 0 = fixed rate. `random_N_style`: Perlin, Sample & Hold, Sine Interpolate, Lorenz
Attractor. LFOs/randoms only matter when routed (mod matrix).

**6. Modulation matrix** — below + `references/mod-matrix.md`.

**7. Macros 1-4** — knob values `settings.macro_control_1..4` (0..1); labels are the top-level
`macro1..4` strings. Macros are mod sources: they only act through routes.

**8. FX chain** — Chorus, Delay, Reverb, Phaser, Flanger, Distortion, EQ, Compressor, each gated
by `<fx>_on`, **all off by default**, each with its own dry/wet (`*_dry_wet` or `*_mix`).
Keys, classic settings and interactions: `references/fx.md`.

**9. Wavetables** — `settings.wavetables[3]`, one inline table per osc; swap with
`wavetable.py`. Library kinds, variants and frame advice: `references/wavetables.md`.

## Modulation matrix

`settings.modulations` = 64 `{"source": ..., "destination": ...}` slots; slot i (0-based) is
shaped by `settings.modulation_{i+1}_{amount,bipolar,power,stereo,bypass}`. **A route with
amount 0 or bypass on does nothing** — use `mod.py`, which keeps both halves in step.
Sources: `lfo_1..8`, `random_1..4`, `env_1..6`, `macro_control_1..4`, `mod_wheel`,
`pitch_wheel`, `velocity`, `aftertouch`, `note`, `note_in_octave`, `lift`, `stereo`
(renders send only note velocity; mod wheel, aftertouch, pitch wheel and lift stay at rest).
Destinations: continuous parameter keys (plus transposes, unison voices, `*_tempo`), including
`modulation_N_amount` (mod depth) and `macro_control_N`. Optional `line_mapping` curve per slot
(JSON only).

## Units table (JSON value → meaning)

Plain 0..1 fractions: levels, mixes, dry/wets, resonance, pan, positions, sustain, mod flags.

| key(s) | JSON range | meaning |
|---|---|---|
| `volume` | 0..7399 | dB = sqrt(value) − 80 (5473 = −6 dB default, 7399 = +6 dB) |
| `filter_{1,2,fx}_cutoff`, `eq_*_cutoff`, `*_filter_cutoff`, `chorus_cutoff`, `phaser_center`, `flanger_center` | 8..136 | MIDI note: Hz = 440·2^((v−69)/12) (60 = 262 Hz, 100 = 2.6 kHz) |
| `env_N_{delay,attack,hold,decay,release}` | 0..~2.4 | seconds = value⁴ (0.5 = 63 ms, 1 = 1 s, 1.5 = 5 s) |
| `osc_N_transpose`, `sample_transpose`, `voice_transpose` | -48..48 | semitones |
| `osc_N_tune`, `sample_tune`, `voice_tune` | -1..1 | ±100 cents |
| `osc_N_unison_detune` | 0..10 | % = (v/10)²·100 (4.47 = 20% default) |
| `osc_N_detune_range` | 0..48 | cents spread of the unison stack |
| `filter_*_drive` | 0..20 | dB |
| `filter_*_keytrack`, `velocity_track` | -1..1 | ±100% |
| `lfo_N_frequency`, `random_N_frequency`, `delay_frequency`, `chorus_frequency`, `phaser_frequency`, `flanger_frequency` | see `show.py` | period seconds = 2^(−value) (1 = 0.5 s, −3 = 8 s) |
| `reverb_decay_time`, `portamento_time`, `lfo_N_smooth_time` | -6..6 etc. | seconds = 2^value (reverb 0 = 1 s, 2 = 4 s) |
| `lfo_N_delay_time`, `lfo_N_fade_time`, `reverb_delay` | | plain seconds |
| `distortion_drive` | -30..30 | dB; `eq_*_gain` -15..15 dB; `reverb_*_shelf_gain` -6..0 dB |
| `delay_feedback`, `chorus_feedback`, `flanger_feedback` | -1..1 | ±100% |
| `modulation_N_amount` | -1..1 | ±100% depth (0 = dead route); `_power` -10..10 curve |
| `pitch_bend_range` | 0..48 | semitones; `polyphony` 1..32 voices |

Enum labels for any switch/selector: `show.py preset.json KEY` (e.g. `delay_tempo`, `chorus_voices`).

## Rules

1. **Gate first.** Sections off by default: `filter_1/2/fx`, every FX unit, `sample`, oscs 2-3.
   A parameter in an off section does nothing — turn the gate on in the same set.py call.
   Glide is the exception: `portamento_time` alone sets it (-10 = none; `portamento_on` changes nothing).
2. **Mode before amount.** `osc_N_distortion_type`, `osc_N_spectral_morph_type`,
   `distortion_filter_order` at index 0 make their amounts dead.
3. **One subsystem per round, then render and listen.** Order: oscillator/wavetable → filter →
   amp envelope → LFO/modulation → FX → macros: the static tone before its shape in time, effects
   last. Never change two subsystems before a listen — you cannot tell which change did what. List
   the section first (`show.py preset.json <prefix>`), then set the biggest, coarsest things before
   the fine ones: pitch before level before unison before shaping; filter mode, cutoff, resonance,
   then the rest; attack and release before sustain and decay; an effect's mix before its character.
4. **Render sounds unchanged ⇒ the edit didn't take.** Run `show.py` on the section: gate off,
   enum mode None, route amount 0, destination in an off section, nothing routed into the
   filter, osc level 0, table silent at frame 0. Fix the cause before trying another value.
5. **Listen to one audio at a time.** Target, then your render; state the difference in words
   (brighter, longer release, more motion...) before editing.
6. **Renders vary slightly** (random phase, unison, random generators): two renders of the same
   preset are near-identical but not bit-identical. Ignore differences you cannot hear.
7. **Tempo comes from notes.json `bpm`**, never from the preset (`beats_per_minute` is ignored
   by the host). LFOs, randoms and chorus/delay/phaser/flanger default to tempo sync; use
   `*_sync=Seconds` for absolute rates.
8. **env 1 owns the voice lifetime**: motion and sound stop when its release ends.
9. **Wavetables: audition the whole chunk, then combine.** Render every table in the chunk you
   were given with the session notes and listen to them in batches of ~8, saying what each batch
   is or isn't close to; a near-silent render is dropped. Then render pairs from the ~8 closest
   tables (both oscillator orders — osc 1 leads), listen to those in batches too, and reply with
   the 2-3 configurations that explain the target best. Tables listed as `silent at frame 0`
   need `osc_N_wave_frame` > 0.
10. **Copy, don't overwrite, to compare**: `cp preset.json try.json`, edit try.json, render both,
    keep the better one as preset.json.

## Recipes (`references/recipes.md`)

| recipe | one-liner |
|---|---|
| `growl_bass` | 3 detuned oscs + Bend distortion + driven lowpass + LFO wobble |
| `fm_bell` | sine carrier, FM ← osc 2 an octave up, instant attack, long decay, reverb |
| `supersaw_lead` | 8-16 unison voices, chorus + delay |
| `evolving_pad` | slow LFOs on wave_frame + cutoff, long envelope, reverb/delay wash |
| `pluck` | short decay envelope, random_1 per-note variation, macro → reverb |
| `hard_bass` | square/saw + Sync distortion + 24 dB lowpass envelope sweep |
| `noisy_drone` | Quantize/Pulse distortion, very slow LFOs, dark filter, big reverb |
| `formant_voice` | Formant filter model, formant_x/y vowels, LFO vowel morph |
| `lfo_wobble` | tempo-synced square LFO → cutoff in the bass |
| `ambient_wash` | ping-pong delay + big reverb + slow random level motion |
| `sub_bass` | sine osc at -24 st routed DIRECT OUT under the main sound |
| `fm_kit` | FM/RM percussive hits, short envelopes |

## Files

- `data/init_preset.json` — the init patch (start over: `cp $VITAL_SKILL/data/init_preset.json preset.json`).
- `data/params.json` — every settings key: min, max, default, discrete, enum labels.
- `data/engine_params.json`, `data/engine_enum_values.json` — the live Vital 1.6.0 parameter dump
  (display names, value texts), for keys this file doesn't cover.
- `data/wavetables.idx.json` + `data/wavetables.bin` — the wavetable library (via wavetable.py).
- When done, set `preset_name` / `preset_style` (Bass, Lead, Pad, Keys, Sequence,
  Percussion, SFX, Experiment) in preset.json.
