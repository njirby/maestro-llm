---
name: vital
description: Recreate a target synth sound in Vital (1.6.0 VST3), stateless, via
  preset.json + headless DawDreamer renders. Use when the target was produced with
  (or should be recreated with) Vital — a wavetable synth: 3 wavetable oscillators +
  sample player, 6 envelopes, 8 LFOs, 4 random generators, dual filters + FX filter,
  fixed 8-FX chain, 64-slot modulation matrix, 4 macros. preset.json is the ONLY synth
  state. Invoke before listening to the target whenever the context indicates Vital or
  a wavetable-based synthesis target.
metadata:
  plugin: vital
  paradigm: wavetable
  subsystems: [oscillator, envelope, filter, lfo, fx, modulation, macro]
---

# Vital skill

## Mental model

- **`preset.json` in your cwd is the ONLY synth state** — a Vital 1.6.0 preset JSON:
  top level `author, comments, macro1..4` (label strings), `preset_name, preset_style,
  synth_version, settings`. `settings` holds all ~776 params plus `lfos[8]` (LFO shapes),
  `modulations[64]`, `wavetables[3]`, `sample`. Editing the file = changing the synth.
- **`notes.json`** = the MIDI: `{"notes":[{"pitch","velocity","start_s","dur_s"}]}`.
- **Render**: preset.json + notes.json → wav via headless DawDreamer hosting
  `~/.vst3/Vital.vst3` (~2-3 s). Render, then listen — the wav is attached to your
  context. Render length is free (keep short; targets cap ~30 s).
- **Values/ranges/enum labels are ALWAYS live** — never trust a remembered number.
  `get_parameters_description()` → 2,983 entries (fields `index, name, numSteps,
  isDiscrete, min/max` (text), `text` (current), `defaultValueText`, `isAutomatable`);
  also `get_parameter_name(i)`, `get_parameter(i)`, `set_parameter(i, 0..1)`,
  `get_parameter_text(i)`, `load_state(path)/save_state(path)`, `clear_midi()`,
  `add_midi_note(pitch, vel, start_s, dur_s)`. This skill teaches **topology +
  semantics + rules**, not values; the UNITS TABLE below tells you how to interpret
  JSON values once you query them.

### Boot recipe (verified — copy exactly, do not "simplify" the X11 guard)

```python
import ctypes, ctypes.util
_x = ctypes.CDLL(ctypes.util.find_library("X11")); _x.XInitThreads()
_H = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)   # X11 handlers take ONE arg
_I = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
_eh = _H(lambda d: 0); _ih = _I(lambda d: 0)
_eh._keep = _eh; _ih._keep = _ih                        # KEEP REFERENCES ALIVE
_x.XSetErrorHandler(ctypes.cast(_eh, ctypes.c_void_p))
_x.XSetIOErrorHandler(ctypes.cast(_ih, ctypes.c_void_p))
import numpy as np, soundfile as sf, dawdreamer as daw
eng = daw.RenderEngine(44100, 512)
proc = eng.make_plugin_processor("vital", "/home/nate/.vst3/Vital.vst3")
eng.load_graph([(proc, [])])
```

Gotchas (all measured on this machine):
- The CFUNCTYPE handler objects MUST stay referenced (`_eh._keep = _eh`). As
  temporaries, any X error during `load_state` (e.g. `XGetWindowProperty`) segfaults
  the process in libffi — symptoms look like "load_state randomly crashes".
- `set_parameter` takes normalized 0..1 (discrete quantizes to `numSteps`); the
  `attempt to map invalid URI .../Vital.vst3` stderr line is harmless.
- `get_parameters_description()` → 2,983 entries; the last 2,080 (idx 903..2982) are
  `MIDI CC` meta params (`isAutomatable: false`) — ignore them.
- `load_state` is REQUIRED for wavetables/mod matrix/LFO shapes/samples (`set_parameter`
  can't set them); fresh process per render; cross-process RNG variance ≈ mel-cos
  0.99995, so byte-identical re-renders are impossible (RULES).
- `get_audio()` → `(channels, samples)`; do NOT mono-average `(L+R)/2` for RMS (phased
  stereo cancels) — use `sqrt(mean(L^2+R^2))/sqrt(2)` or per-channel RMS.

### Render recipe (stateless)

```python
import json, struct, tempfile
_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
def _b64(b):
    out = []
    for i in range(((len(b) * 8) + 5) // 6):
        bit = i * 6; y, o = bit >> 3, bit & 7
        w = b[y] | (b[y + 1] << 8) if y + 1 < len(b) else (b[y] if y < len(b) else 0)
        out.append(_B64[(w >> o) & 0x3F])
    return str(len(b)) + "." + "".join(out)
def build_state(preset):
    jb = json.dumps(preset, separators=(",", ":")).encode() + b"\x00"
    chunk = jb + b"\x00" * 16 + b"JUCEPrivateData"
    body = 4 + 4 + 4 + 4 + 4 + 128 + 4 + len(chunk)
    icomp = (b"VstW" + struct.pack(">III", 8, 1, 0) + b"CcnK" + struct.pack(">i", body)
             + b"FBCh" + struct.pack(">i", 2) + b"Vita" + struct.pack(">i", 0x00010600)
             + struct.pack(">i", 0) + b"\x00" * 128 + struct.pack(">i", len(chunk)) + chunk)
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n<VST3PluginState><IComponent>'
           + _b64(icomp) + "</IComponent></VST3PluginState>").encode() + b"\x00"
    return struct.pack("<II", 0x21324356, len(xml)) + xml
preset = json.load(open("preset.json"))
sp = tempfile.mktemp(suffix=".state"); open(sp, "wb").write(build_state(preset))
proc.load_state(sp)                       # boot recipe first
proc.clear_midi()
notes = [(n["pitch"], n["velocity"], n["start_s"], n["dur_s"])
         for n in json.load(open("notes.json"))["notes"]]
for p, v, s, d in notes:
    proc.add_midi_note(int(p), int(v), float(s), float(d))
eng.render(max(s + d for _, _, s, d in notes) + 0.5)
sf.write("out.wav", proc.get_audio().T, 44100)
```

## The 9-section map

### 1. Oscillators 1-3 (wavetable)
Gated by `osc_N_on` (osc1 ON by default, osc2/3 OFF). Position = `osc_N_wave_frame`
(0..256); width via `osc_N_detune_range` (cents) / `osc_N_stereo_spread` /
`osc_N_unison_*`. (The engine's "Oscillator N Blend" X/Y 2D parameter has NO preset
JSON key in this build — settable live via set_parameter only.) Full key set per osc
(all verified):
`osc_N_{on,level,pan,phase,random_phase,transpose,transpose_quantize,tune,wave_frame,
detune_range,detune_power,stereo_spread,unison_voices,unison_detune,unison_blend,
unison_stack_type,spectral_unison,frame_spread,distortion_type,distortion_amount,
distortion_phase,distortion_spread,spectral_morph_type,spectral_morph_amount,
spectral_morph_spread,stack_style,destination,midi_track,smooth_interpolation,view_2d}`.
Non-obvious:
- **FM and RM are `osc_N_distortion_type` values — no separate FM/RM section** (no
  `osc_N_fm` key exists in this build): 0 None, 1 Sync, 2 Formant, 3 Quantize, 4 Bend,
  5 Squeeze, 6 Pulse, **7/8 = FM from the two other oscillators, 9 = FM from sample,
  10/11 = RM from the two other oscillators, 12 = RM from sample** (verified: type 7
  on osc1 tracks osc2 — switch it off to confirm which osc a variant uses). Depth =
  `osc_N_distortion_amount`; amount does nothing until type ≠ 0.
- `osc_N_spectral_morph_type` (17: Vocode, Formant Scale, Harmonic/Inharmonic Stretch,
  Smear, Random Amplitudes, Low/High Pass, Phase Disperse, Shepard Tone, ...) — 0=None.
- `osc_N_stack_style` = chord voicing of unison voices (Unison, Center Drop 12/24,
  Octave, Octave+7, 2x Octave, Power Chord, Major/Minor Chord, Harmonics, Odd/
  Sub Harmonics).
- `osc_N_destination` routes the osc INTO a section: 0 FILTER 1, 1 FILTER 2, 2
  FILTER 1+2, 3 EFFECTS, 4 DIRECT OUT, 5 CHORUS, 6 COMPRESSOR, 7 DELAY, 8 DISTORTION,
  9 EQUALIZER, 10 FX FILTER, 11 FLANGER, 12 PHASER, 13 REVERB (defaults: osc1→0,
  osc2→1, osc3→3). Sub trick: dest 4 = direct out, bypasses the filters.

### 2. Sample player (4th source)
`sample_on` (default OFF) + `sample_{level,pan,transpose,transpose_quantize,tune,loop,
bounce,random_phase,keytrack,destination}`. Audio in `settings.sample = {length, name,
sample_rate, samples, samples_stereo?}` — `samples` = base64 of **float16 LE** (2
bytes/sample). Init ships "White Noise". `sample_destination` = same 15-route enum as
`osc_N_destination` (default 3=EFFECTS).

### 3. Envelopes 1-6
DAHDSR: `env_N_{delay,attack,attack_power,hold,decay,decay_power,sustain,release,
release_power}` (times in log-seconds — UNITS; sustain 0..1). **env1 is the master amp
envelope: when its release finishes, ALL sound/LFOs/random motion stop.** Long pads:
`env_1_sustain = 1` (or drive motion from env2-6 / LFOs). `*_power` = curve shape
(-20..20, decay default -2). No gate keys.

### 4. Filters (1, 2, fx)
Three identical filter units. Gated by `filter_1_on` / `filter_2_on` / `filter_fx_on` —
**all default OFF** (setting cutoff on an off filter is a silent no-op).
Keys: `filter_{1,2,fx}_{on,model,style,cutoff,resonance,mix,drive,keytrack,blend,
blend_transpose,comb_blend_offset,formant_x,formant_y,formant_resonance,formant_spread,
formant_transpose,filter_input}`.
Non-obvious:
- `cutoff` is note-scaled 8..136 (live "semitones" = value − 60; 60 = 0 st). NOT hertz.
- `model` (8): 0 Analog, 1 Dirty, 2 Ladder, 3 Digital, 4 Diode, **5 Formant, 6 Comb,
  7 Phaser**. With model 5, `formant_x/y` (0..1) is the vowel grid (center 0.5/0.5 ≈
  "a", high-y ≈ "ee", low-y ≈ "oo" — move and render to confirm);
  `formant_resonance` default 0.85.
- `style`: 12dB, 24dB, Notch Blend, Notch Spread, B/P/N. `filter_input` (Off/On) =
  pre/post input for the FX filter. `mix` = stage wet/dry; `drive` 0..20 dB.

### 5. LFOs 1-8 and random generators 1-4
Keys: `lfo_N_{frequency,phase,delay_time,fade_time,smooth_mode,smooth_time,stereo,sync,
sync_type,tempo,keytrack_transpose,keytrack_tune}` and `random_N_{frequency,style,sync,
sync_type,tempo,stereo,keytrack_transpose,keytrack_tune}`.
- **Shape is per-LFO data, not a param**: `settings.lfos[N] = {name, num_points, points
  ([x,y,...] pairs 0..1), powers, smooth}`. Init default = Triangle. A square/sine/
  custom LFO requires editing this list (load_state picks it up).
- `lfo_N_sync` (5): Seconds | Tempo | Tempo Dotted | Tempo Triplets | Keytrack —
  **default Tempo (tempo-synced!)**, rate note = `lfo_N_tempo`
  (Freeze, 32/1 … 1/64; default 7 = 1/2). In Seconds mode `frequency` (2^-x secs) rules.
- `lfo_N_sync_type` (7): Trigger [default], Sync, Envelope, Sustain Envelope, Loop
  Point, Loop Hold, Point Cycle. "Envelope" = one-shot per note — pluck movement.
- `random_N_style`: Perlin [default], Sample & Hold, Sine Interpolate, Lorenz.
- Keytracking: `lfo_N_keytrack_transpose` default -12 (LFO slows on high notes); 0 = fixed.

### 6. Modulation matrix (64 slots)
See next section + `references/mod-matrix.md`.

### 7. Macros 1-4
Top-level `macro1..4` = **label strings** ("CUTOFF", "TONE"...). Knob values =
`settings.macro_control_1..4` (0..1; settable in JSON or via engine "Macro 1..4").
Macros are mod SOURCES (routed through the matrix), not destinations — except presets
do route `velocity → macro_control_N`.

### 8. FX chain (8 units)
Chorus, Delay, Reverb, Phaser, Flanger, Distortion, EQ, Compressor — each gated by
`<fx>_on`, **all default OFF**. Key groups, ranges, classic settings, chain order
(`effect_chain_order`, permutation 0..362879) and interactions in `references/fx.md`.

### 9. Wavetables (the actual sound source)
`settings.wavetables[3]`, one inline entry per osc: `{author, name, version,
full_normalize, remove_all_dc, groups:[{components:[{type, interpolation,
interpolation_style, keyframes:[{position, wave_data}]}]}]}`. Library =
`wavetable_lib.json` (568 entries / 301 unique). Swap an entry in, render, listen =
audition. Conventions, the 11 silent tables, RMS rule: `references/wavetables.md`.

## Mod-matrix topology

`settings.modulations` = flat list of 64 slot objects; empty slot
`{"source": "", "destination": ""}`. A used slot:

```json
{"source": "lfo_1", "destination": "filter_1_cutoff"}
```

Optional third field `line_mapping` = curve over the source's 0..1 output (UI "MORPH"):
`{"name": "Linear", "num_points": 3, "points": [x0,y0,x1,y1,...], "powers": [...],
"smooth": false}`.
**Slot i (0-based) is controlled by `settings.modulation_{i+1}_*`**: `amount` (-1..1;
**0 = dead**, default 0), `bipolar`, `power` (-10..10 curve), `stereo`, `bypass`
(each Off/On). A route with `amount: 0` does nothing even if source+destination are
set — check BOTH.

Sources (exact strings): `lfo_1..8`, `random_1..4`, `env_1..6`, `macro_control_1..4`,
`mod_wheel`, `pitch_wheel`, `velocity`, `aftertouch`, `note`, `note_in_octave`, `lift`,
`stereo` (old presets may use bare `random`).
Destinations = any of the 903 `isAutomatable` engine params, in JSON-key form —
including `env_1_attack`, `lfo_1_delay_time`, `modulation_9_amount` (mod-on-mod) and
`macro_control_3` (velocity driving a macro). Full lists + routing semantics:
`references/mod-matrix.md`.

## UNITS TABLE (all non-0..1 params; JSON value → meaning)

`0..1` params (level, mix, resonance, pan, stereo, phase, positions, mod amounts,
bipolar/stereo/bypass flags...) are plain fractions. Everything else:

| key(s) | JSON range | units / meaning |
|---|---|---|
| `beats_per_minute` | 0.33..5.0 | **beats per second** (2.0 = 120 BPM; 20..300 BPM) |
| `volume` | 0..7399 | dB-encoded; 5473 ≈ -6 dB (default), 0 = -80 dB, 7399 = +6 dB |
| `filter_{1,2,fx}_cutoff` | 8..136 | note scale; live "semitones" = value−60 (60 = 0 st) |
| `eq_{low,band,high}_cutoff`, `distortion_filter_cutoff`, `delay_filter_cutoff`, `chorus_cutoff`, `phaser_center`, `flanger_center` | 8..136 | same note scale (live shows raw value as "semitones") |
| `filter_{1,2,fx}_{keytrack}` | -1..1 | ±100% |
| `filter_{1,2,fx}_{drive}` | 0..20 | dB |
| `filter_{1,2,fx}_{blend}` | 0..2 | low↔high filter blend |
| `filter_{1,2,fx}_{formant_x,formant_y}` | 0..1 | vowel grid (§4); `formant_resonance` 0.3..1 |
| `osc_N_transpose`, `sample_transpose`, `voice_transpose` | -48..48 | semitones (range table claims 96; live max 48) |
| `osc_N_tune`, `sample_tune`, `voice_tune`, `lfo_N_keytrack_tune` | -1..1 | ±100 cents |
| `lfo_N_keytrack_transpose`, `random_N_keytrack_transpose` | -60..96 | semitones of rate tracking (default -12; 0 = fixed rate) |
| `osc_N_detune_range` | 0..48 | cents; `osc_N_detune_power` -5..5 curve (default 1.5) |
| `osc_N_unison_voices` | 1..16 | write the voice count directly (range table says 1..15; measured 4 → 4v) |
| `osc_N_unison_detune` | 0..10 | 0..100% (square-law: 4.47 = 20%) |
| `osc_N_frame_spread` | -128..128 | frame offset spread; `osc_N_wave_frame` 0..256 position |
| `env_N_{attack,decay,release}` | 0..2.378 | log seconds: 0→0 s, 0.6→0.125 s, 1.19→2 s, 1.78→10 s, 2.38→32 s |
| `env_N_{delay,hold}` | 0..1.414 | log seconds: 0→0 s, 0.35→0.016 s, 0.71→0.25 s, 1.41→4 s |
| `env_N_{attack,decay,release}_power` | -20..20 | curve shape (decay default -2) |
| **2^(-x) second rates**: `lfo_N_frequency`, `random_N_frequency` (-7..9), `delay_frequency`, `delay_aux_frequency` (-2..9), `chorus_frequency` (-6..3), `phaser_frequency`, `flanger_frequency` (-5..2), `reverb_chorus_frequency` (-8..3) | as listed | **seconds = 2^(-x)** (1.0 = 0.5 s, -3 = 8 s, 5 = 31 ms) |
| **2^(x) second times**: `reverb_decay_time` (-6..6), `portamento_time`, `lfo_N_smooth_time` (-10..4) | as listed | **seconds = 2^(x)** (reverb: 0 = 1 s, 2 = 4 s, 6 = 64 s) |
| `lfo_N_delay_time` 0..4, `lfo_N_fade_time` 0..8, `reverb_delay` 0..0.3 | | plain seconds |
| `chorus_delay_1`, `chorus_delay_2` | -10..-5.64 | log seconds (internal) |
| `distortion_drive` | -30..30 | dB (negative = softer); `distortion_mix` 0..1 wet |
| `delay_feedback` | -1..1 | ±100%; `chorus_feedback` -0.95..0.95 |
| `eq_{low,band,high}_gain` | -15..15 | dB |
| `compressor_{attack,release}` | 0..1 | 0..100% (log); `compressor_{low,band,high}_gain` -30..30 dB makeup |
| `compressor_{low,band,high}_{lower,upper}_{threshold,ratio}` | various | dB / ratio (query live) |
| `pitch_bend_range` | 0..48 | semitones (default 2); `portamento_slope` -8..8 glide curve |
| `velocity_track` | -1..1 | ±100% velocity→level |
| `modulation_{1..64}_amount` | -1..1 | ±100% mod depth (**0 = route dead**); `modulation_{1..64}_power` -10..10 curve |
| `macro_control_{1..4}` | 0..1 | macro knob position |
| `mod_wheel` 0..1, `pitch_wheel` -1..1, `stereo_routing` 0..1 | | controller values / % |
| `polyphony` | 1..31 | voices (default 8); `oversampling` 0..3 = 1/2/4/8x |
| `effect_chain_order` | 0..362879 | FX stage permutation index (9!−1) |

Discrete enums (index → labels) are live-queryable for every switch/type key; the
important ones (gates, filter model, dist type incl. FM/RM, LFO sync/tempo, stack,
morph, destinations) are in the section map above and references/fx.md.

## RULES (hard-won; follow in order)

1. **Check the on-gate before trusting any edit.** Every gated section below defaults
   OFF: `filter_1_on, filter_2_on, filter_fx_on, distortion_on, chorus_on, delay_on,
   reverb_on, phaser_on, flanger_on, eq_on, compressor_on, sample_on, portamento_on`
   (osc gates: `osc_1_on`=ON default, `osc_2_on`/`osc_3_on`=OFF). A param set inside an
   off section does nothing. Turn the gate on in the SAME edit as the params.
2. **Set the enum mode before the amount.** Index-0-inert enums: `osc_N_distortion_type`
   (0=None), `osc_N_spectral_morph_type` (0=None), `distortion_filter_order` (0=None).
   Amounts with mode 0 are dead.
3. **One edit round = one subsystem, then render + listen.** Order:
   oscillator → envelope → filter → lfo → fx → modulation → macro. Never batch two
   subsystems into one unattributed change.
4. **Render sounds unchanged ⇒ the edit didn't take.** Causes, in order: gate off,
   wrong key name, enum mode 0, mod route with `modulation_N_amount` = 0, destination
   in an off section, section's `mix`=0 (filters), level 0. Re-read the JSON diff and
   the live `get_parameter_text` BEFORE retrying a value.
5. **Compare one audio at a time** — never ask the model to judge target+A+B
   simultaneously (it conflates). Judge A, then B, then state the delta.
6. **Target clips cap ~30 s** — longer material is transcribed in slices.
7. **Shortlist, don't page** for wavetable search: parallel slices → 2-4 candidates
   each → judge picks the final tuple. Never iterate the library one-by-one.
8. **Renders are ~0.5 s per note — use them for A/B**: same notes.json, two presets.
   Cross-process variance ≈ mel-cosine 0.99995: if mel-cos(A,B) > 0.9995 and
   |Δcentroid| < 1%, it is "no change" (rule 4).
9. **Detect silent wavetables by RMS** (< 1e-4 on a 0.4 s note = silent): label, never
   shortlist (11 known — references/wavetables.md).
10. **Mod routes need the slot amount**: `modulations[i]` + `modulation_{i+1}_amount = 0`
    = no-op.
11. **Tempo-synced by default**: LFOs/random/FX delays default `sync = Tempo`;
    `beats_per_minute` (beats/sec) sets the clock. Exact-second rates: `sync = Seconds`.
12. **Env1 is the lifecycle owner**: sound dies when env1 release ends —
    `env_1_sustain = 1` for pads, or route motion through env2-6/LFOs.

## Recipe INDEX

Full topology specs (sections, keys + example values, mod routes, character, how to
verify by ear) in `references/recipes.md`.

| recipe | one-liner |
|---|---|
| `growl_bass` | 3-osc detuned + Bend distortion + driven lowpass + macro wobble (mined: Random Amp Growl) |
| `fm_bell` | sine carrier + FM←osc2 type 7, fast attack/long decay (verified render) |
| `supersaw_lead` | unison 8-16 + octave stacks, chorus + delay (manual-derived) |
| `evolving_pad` | slow LFOs on wave_frame + cutoff, long env, reverb/delay wash (mined: Float Chords) |
| `pluck` | env1 delay+decay, random_1 variation, macro→release (mined: Super Nice Pluck) |
| `hard_bass` | square/saw + Sync distortion + 24dB lowpass, env sweep (manual-derived) |
| `noisy_drone` | Quantize/Pulse/Squeeze distortion + low LFO rate, dark filter (manual-derived) |
| `formant_voice` | filter model=Formant, formant_x/y vowels, sample/vocoded osc (mined: Ah Eh Ee Oh) |
| `lfo_wobble` | tempo LFO (1/8 or 1/16) → cutoff, square LFO shape, bass register (mined: Banana Wob) |
| `ambient_wash` | ping-pong delay + big reverb + slow random on levels (mined: Smoker's Lounge pattern) |
| `sub_bass` | sine osc transposed -24, destination 4 DIRECT OUT, separate from mids (mined: Banana Wob sub) |
| `fm_kit` | FM type 7-12 across 3 oscs, short envs, per-note transpose (mined: FM Drum Circle) |

## File ops

- **Save preset**: `preset.json` IS the save — set `preset_name`/`preset_style`/`author`
  when done. `preset_style` categories seen in the wild: Bass, Sequence, Pad, Keys,
  Lead, Percussion, SFX, Experiment, Template.
- **Load preset**: replace the whole file (corpus: `~/.local/share/vital/**/Presets/
  **/*.vital` are plain JSON).
- **Swap/import wavetable**: replace `settings.wavetables[N]` with the matching
  `wavetable_lib.json` entry verbatim, then render (load_state required). Never
  hand-edit `wave_data`.
- **Import sample**: `settings.sample = {length, name, sample_rate, samples}` —
  `samples` = base64 of float16 LE (`np.float16(audio).tobytes()` → b64); then
  `sample_on = 1`.
- **BPM**: `settings.beats_per_minute` = beats/sec (120 BPM = 2.0). Tempo-synced
  consumers: LFOs/random (`sync ≠ 0`), chorus/delay/phaser/flanger (`*_sync ≠ 0`).

## EXPLORATION appendix (cold start)

Anything not covered here — the engine answers, always:

- **Filter the live catalog** (boot once, cache to /tmp):
  `[p for p in proc.get_parameters_description() if 'filter' in p['name'].lower() and p['isAutomatable']]`.
  Reliable fields: `index, name, numSteps, isDiscrete, min, max, text, defaultValueText,
  isAutomatable`.
- **Label walk** (enum labels for a key): index from `maestro/synth/json_key_to_reaper.json`
  (key → {idx, name}); save `get_parameter(i)`, sweep `set_parameter(i, s/(n-1))` +
  `get_parameter_text(i)` for `s in range(numSteps)`, restore. Known table errors:
  `filter_fx_on` maps to idx 142 "Filter fx Model" (real switch = idx 143); `bypass`
  maps to 2983 (engine ends at 2982; real idx 727).
- **A/B for gates**: render the preset with `X_on=1` vs `X_on=0`, same notes; if the
  render is "unchanged" per rule 8, the gate (or your param) isn't doing what you think.
- **When to consult references/**: `recipes.md` = sound classes → topology;
  `mod-matrix.md` = sources/destinations + routing semantics; `fx.md` = FX key groups +
  classic settings; `wavetables.md` = library conventions + audition + silent tables.
  The full key→engine-index map and ranges live in
  `maestro/synth/{json_key_to_reaper.json, param_ranges.json}` — consult for any key
  you're unsure exists; every key named here was verified against those tables + the
  live 2,983-param dump.
