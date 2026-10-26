# Vital modulation matrix — sources, destinations, routing semantics

Authority: live 2,983-param dump (Vital 1.6.0 VST3) + 77 mined presets
(1,616 mod entries across 75 patches). JSON = the only state; everything here maps to
`settings.modulations[i]` + `settings.modulation_{i+1}_*`.

## Exact JSON entry format

`settings.modulations` is a list of exactly 64 slot objects:

```json
{"source": "", "destination": ""}                              // empty slot
{"source": "lfo_2", "destination": "osc_2_transpose"}          // live entry (mined: Mr Bill)
{"source": "macro_control_1", "destination": "filter_1_cutoff"}
```

Optional third field — `line_mapping` — is a freehand curve applied to the source's
0..1 signal (the UI "MORPH" column). Real example (mined: Disrupt.vital, lfo_1 →
osc_3_spectral_morph_amount):

```json
{
  "destination": "osc_3_spectral_morph_amount",
  "line_mapping": {
    "name": "Linear", "num_points": 5,
    "points": [0.0, 1.0, 0.25, 0.7276296019554138, 0.625, 1.0, 0.625, 0.40539824962615967, 1.0, 0.0],
    "powers": [0.0, 0.0, 0.0, 0.0, 0.0], "smooth": false
  },
  "source": "lfo_1"
}
```

`points` = [x0,y0, x1,y1, ...] pairs in 0..1. Keep `line_mapping` out of simple routes.

**The slot's own parameters live in `settings`, not in the entry.** Slot i (0-based)
is shaped by:

| key | range | meaning |
|---|---|---|
| `modulation_{i+1}_amount` | -1..1 | mod depth; **0 = route is dead** (default) |
| `modulation_{i+1}_bipolar` | 0/1 | Off: source 0..1 maps 0..amount; On: source centered at 0.5 |
| `modulation_{i+1}_power` | -10..10 | response curve (S-shape) |
| `modulation_{i+1}_stereo` | 0/1 | route to L/R split where applicable |
| `modulation_{i+1}_bypass` | 0/1 | kill this slot without deleting it |

A full working route = entry with source+destination **and** `modulation_{i+1}_amount ≠ 0`.
Real example (mined: Crescendo Bells.vital): `modulations[0] = {source: lfo_1,
destination: osc_1_level}` with `modulation_1_amount = 0.77`.

## Sources (exact strings, verified in corpus)

| source | semantics | notes |
|---|---|---|
| `lfo_1` .. `lfo_8` | the 8 LFOs' 0..1 output (per `settings.lfos[N]` shape) | rate/mode from `lfo_N_*`; bipolar flag centers |
| `random_1` .. `random_4` | per-note random value (style: Perlin/S&H/Sine/Lorenz) | "humanize" source; `random` (bare) appears in some old presets |
| `env_1` .. `env_6` | envelope 0..1 value | env_1 = amp env (decays per note); env_2-6 for one-shot motion |
| `macro_control_1` .. `macro_control_4` | macro knob 0..1 | most common source in the corpus (414/362/350/238 uses) |
| `mod_wheel` | MIDI CC1 0..1 | 180 uses; classic → cutoff |
| `pitch_wheel` | CC pitch bend -1..1 | rare (2) |
| `velocity` | note velocity 0..1 | 158 uses; → level, cutoff, reverb, wave_frame, even `macro_control_N` |
| `aftertouch` | key pressure 0..1 | 14 uses |
| `note` | note number across keyboard 0..1 (low→high) | keytracking; 44 uses |
| `note_in_octave` | position within current octave 0..1 | arpeggio-like transpose (16 uses) |
| `lift` | note-off velocity 0..1 | release expression (26 uses) |
| `stereo` | L=0/R=1 (or 0.5 center) | split-destinations for wide effects (80 uses) |

## Destinations

Any of the 903 `isAutomatable` engine params is a destination; name it in **JSON-key
form** (the key in `settings`, which matches `maestro/synth/json_key_to_reaper.json`).
Grouped list (JSON keys):

**Oscillators** (per N ∈ 1..3): `osc_N_level`, `osc_N_pan`, `osc_N_transpose`,
`osc_N_tune`, `osc_N_wave_frame`, `osc_N_phase`, `osc_N_detune_range`,
`osc_N_detune_power`, `osc_N_unison_voices`, `osc_N_unison_detune`, `osc_N_unison_blend`,
`osc_N_unison_stack_type`, `osc_N_spectral_unison`, `osc_N_stereo_spread`,
`osc_N_frame_spread`,
`osc_N_distortion_type`, `osc_N_distortion_amount`, `osc_N_distortion_phase`,
`osc_N_distortion_spread`, `osc_N_spectral_morph_type`, `osc_N_spectral_morph_amount`,
`osc_N_spectral_morph_spread`, `osc_N_stack_style`, `osc_N_destination`,
`osc_N_midi_track`, `osc_N_smooth_interpolation`.
**Sample**: `sample_level`, `sample_pan`, `sample_transpose`, `sample_tune`,
`sample_loop`, `sample_bounce`, `sample_keytrack`, `sample_destination`.
**Envelopes** (per N ∈ 1..6): `env_N_attack`, `env_N_attack_power`, `env_N_decay`,
`env_N_decay_power`, `env_N_delay`, `env_N_hold`, `env_N_sustain`, `env_N_release`,
`env_N_release_power`.
**Filters** (per U ∈ 1,2,fx): `filter_U_on`, `filter_U_cutoff`, `filter_U_resonance`,
`filter_U_mix`, `filter_U_drive`, `filter_U_keytrack`, `filter_U_model`, `filter_U_style`,
`filter_U_blend`, `filter_U_blend_transpose`, `filter_U_comb_blend_offset`,
`filter_U_formant_x`, `filter_U_formant_y`, `filter_U_formant_resonance`,
`filter_U_formant_spread`, `filter_U_formant_transpose`, `filter_U_filter_input`.
**LFOs** (per N ∈ 1..8): `lfo_N_frequency`, `lfo_N_phase`, `lfo_N_delay_time`,
`lfo_N_fade_time`, `lfo_N_smooth_mode`, `lfo_N_smooth_time`, `lfo_N_stereo`,
`lfo_N_sync`, `lfo_N_sync_type`, `lfo_N_tempo`, `lfo_N_keytrack_transpose`,
`lfo_N_keytrack_tune`.
**Random** (per N ∈ 1..4): `random_N_frequency`, `random_N_style`, `random_N_sync`,
`random_N_sync_type`, `random_N_tempo`, `random_N_stereo`, `random_N_keytrack_transpose`,
`random_N_keytrack_tune`.
**FX** — see references/fx.md key groups (all `*_dry_wet/mix`, `*_feedback`, `*_cutoff`,
`*_frequency`, `*_depth`, `*_on`, etc.).
**Global**: `volume`, `beats_per_minute`, `pitch_bend_range`, `portamento_time`,
`portamento_slope`, `portamento_force`, `portamento_scale`, `portamento_on`, `legato`,
`velocity_track`, `stereo_mode`, `stereo_routing`, `polyphony`, `oversampling`,
`effect_chain_order`, `bypass`, `voice_amplitude`, `voice_priority`, `voice_tune`,
`voice_transpose`, `voice_override`, `mpe_enabled`.
**Meta**: `modulation_{1..64}_amount` / `_bipolar` / `_bypass` / `_power` / `_stereo`
(modulating a mod route's depth — used 148× in corpus), and `macro_control_{1..4}`
(velocity/macro driving a macro — used in Crescendo Bells: `velocity → macro_control_3`).

## Routing semantics — the ones that matter

- **LFO → `filter_1_cutoff`** = wobble/acid motion. Rate: `lfo_N_sync=Tempo`
  (default) + `lfo_N_tempo` (e.g. 8 = 1/8) for musical wobble, or `sync=Seconds` +
  `frequency` for fixed Hz. Square LFO shape (edit `settings.lfos[N]`) = hard wobble;
  triangle (default) = smooth. Amount 0.2-0.5 on cutoff for a 15-unit swing.
- **env_1 (or env_2 one-shot) → `filter_1_resonance` / `filter_1_cutoff`** = the
  pluck/bass "scoop". Route env, `amount` 0.3-0.7, often bipolar OFF so the env's
  decay pulls the cutoff down. env_2-6 with `env_N_sustain=0` are cleaner one-shots.
- **Osc 2 as FM/RM source** = NOT via the matrix: set `osc_1_distortion_type` to
  7/8 (FM) or 10/11 (RM) + `osc_1_distortion_amount` (verified live: type 7 on osc1
  tracks osc2; osc2 transposition changes the FM index → sidebands spread). The matrix
  can still modulate `osc_1_distortion_amount` (LFO on FM depth = evolving bells).
- **macro_control_N → anything** = performance knob. Corpus pattern: one macro fans
  out to 5-10 destinations (wave_frame + cutoff + mix + FX wet) with small amounts
  (0.1-0.5) for a "taste" knob; single-destination amount 1.0 when the macro IS the
  parameter (e.g. Banana Wob: macro1 → osc_1_level, amount 1.0, base level 0).
- **velocity → `volume` / `*_dry_wet` / `sample_level`** = human dynamics (Crescendo
  Bells routes velocity to 6 places at 0.1-0.18).
- **`stereo` → `osc_N_pan` or FX center params** = auto-width: left channel low,
  right high (or opposite with negative amount).
- **`note` / `note_in_octave` → `osc_N_transpose`** = keytracking / arpeggio-like
  octave effects (community-doc trick: LFO → transpose quantize = arps).
- **`random_1` → `osc_1_wave_frame` / `_distortion_amount` / `_spectral_morph_amount`**
  = per-note variation without any LFO (Super Nice Pluck pattern, amounts 0.1-0.6).
- **`lift` → anything** = release expression (rare; needs a MIDI that sends note-off
  velocity — the DawDreamer `add_midi_note` path does not send it, so `lift`/
  `aftertouch`/`mod_wheel`/`pitch_wheel`/`velocity` variations are limited in renders;
  velocity is baked per-note, the others sit at 0).

## Compatibility rules (what actually matters)

1. Any source → any automatable destination is legal; the engine does not reject
   pairs. The constraint is **semantic**: routing `env_1` to `env_1_attack` is legal
   but self-referential and usually wrong.
2. Destinations inside gated sections are dead until the gate is on (see SKILL RULES 1).
3. A destination that is itself an enum (e.g. `filter_1_model`) is modulated
   numerically — LFOs crossing enum steps produce stepping, not smooth morph.
4. Two slots can target the same destination; amounts sum.
5. Slot order in the list does not matter; only the index ↔ `modulation_{i+1}_*`
   pairing matters. Prefer filling slots in order (0, 1, 2, ...) so the pairing stays
   obvious.
6. Delete a route by setting `source: ""` AND zeroing its `modulation_{i+1}_amount`
   (leaving amount non-zero on an empty slot is harmless but confusing).

## Corpus usage stats (what real patches do)

Top destinations: `filter_1_cutoff` (162), `filter_2_cutoff` (106),
`osc_1_spectral_morph_amount` (104), `osc_1_level` (102), `osc_2_level` (98),
`reverb_dry_wet` (96), `osc_1_wave_frame` (92), `distortion_drive` (66),
`sample_level` (66), `osc_1_distortion_amount` (62), `filter_fx_cutoff` (60),
`osc_2_transpose` (60), `env_1_attack` (30), `env_1_release` (20), `volume` (26),
`modulation_*_amount` (104 across slots). Top sources: macros (1,364 combined),
lfo_1..8 (966), `mod_wheel` (180), `velocity` (158), env_2 (128), env_3 (78),
random_1 (86), `stereo` (80), `note` (44), `lift` (26), `aftertouch` (14).
Bipolar is used rarely (~3% of routes); `line_mapping` curves on ~9% of routes.
