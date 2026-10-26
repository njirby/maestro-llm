# Vital FX chain (8 units) — keys, ranges, classic settings

Chain position in signal: [osc/sample] → filter 1 → filter 2 → **FX stages (orderable)**
→ out. FX stages: Chorus, Delay, Reverb, Phaser, Flanger, Distortion, EQ, Compressor.
`effect_chain_order` (0..362879 = 9!−1) permutes the 9 FX stages; default 0.
`bypass` (global, 0/1) mutes the whole plugin.

**Every unit is gated by `<fx>_on`, all default OFF.** Setting params on an off unit is
a silent no-op — enable the gate in the same edit. Each unit also has its own
dry/wet (`*_dry_wet` or `*_mix`), which defaults mid-range but only matters when on.

Key groups and units (JSON-native; SKILL.md units table for encodings; `show.py preset.json
<fx>_` lists a unit's keys with ranges and labels):

## Distortion — `distortion_on`
`distortion_type` (6: 0 Soft Clip [default], 1 Hard Clip, 2 Linear Fold, 3 Sine Fold,
4 Bit Crush, 5 Down Sample), `distortion_drive` (-30..30 dB), `distortion_mix` (0..1 wet).
The unit's built-in filter (no separate gate — it only acts when `distortion_on`):
`distortion_filter_order` (0 None, 1 Pre, 2 Post), `distortion_filter_cutoff`
(8..136 note scale), `distortion_filter_resonance` (0..1),
`distortion_filter_blend` (0..2 pre/post blend).
- When: grit for bass/lead, drive for pads.
- Classic: Soft Clip drive 6-12 dB, mix 1.0, filter Post cutoff ~60 res 0.5 (tames
  harshness); Bit Crush/Down Sample for lo-fi (corpus uses: Hard Clip most common
  among the non-None types; drive 8.2-11.4 typical).
- Interactions: put distortion BEFORE delay/reverb (default chain order does) so the
  tail is clean; LFO or macro on `distortion_filter_cutoff` = evolving crunch (mined:
  Banana Wob routes macro1 → distortion_filter_cutoff).

## Chorus — `chorus_on`
`chorus_frequency` (seconds=2^-x, -6..3; default -3 = 8 s), `chorus_mod_depth` (0..1),
`chorus_dry_wet`, `chorus_feedback` (-0.95..0.95), `chorus_cutoff` (8..136),
`chorus_spread` (0..1), `chorus_delay_1`, `chorus_delay_2` (log-sec),
`chorus_sync` (Seconds|Tempo|Tempo Dotted|Tempo Triplets, default Tempo),
`chorus_tempo` (Freeze..1/16), `chorus_voices` (JSON 1..4 = 4/8/12/16 voices, default 4 = 16).
- When: widening pads/leads, subtle movement on keys. Slow frequency (6-10 s) + depth
  0.5-0.7 + low wet (0.2-0.4) = natural; fast + deep = dubby.
- Interactions: chorus before reverb thickens the tail; on basses use very low wet or
  skip (corpus: 42/77 patches use it, mostly pads/keys).

## Delay — `delay_on`
`delay_style` (4: 0 Mono, 1 Stereo, 2 Ping Pong, 3 Mid Ping Pong [default Mono]),
`delay_frequency` (seconds=2^-x, -2..9; 0 = 1 s), `delay_sync` (Seconds|Tempo|Tempo
Dotted|Tempo Triplets, default Tempo), `delay_tempo` (4/1..1/64, default 1/8; JSON 4 = 4/1 … 12 = 1/64),
`delay_feedback` (-1..1, default 0.5), `delay_filter_cutoff` (8..136),
`delay_filter_spread` (0..1), `delay_dry_wet` (default 0.333), plus a **second delay
line**: `delay_aux_frequency`, `delay_aux_sync`, `delay_aux_tempo`.
- When: rhythmic echo (bass/sequence), stereo wash (pads).
- Classic: tempo 1/4 or 1/8 dotted, feedback 0.3-0.5, wet 0.2-0.35, filter cutoff ~70
  to darken repeats; Ping Pong (style 2/3) for wide ambient. Mined: Crescendo Bells
  (Mid Ping Pong, fb 0.5), Float Chords (Stereo, fb 0.95, wet driven by macro).
- Interactions: the second line (aux) lets you stack two echo times; keep aux dry_wet
  at the same `delay_dry_wet` — it shares the wet control.

## Reverb — `reverb_on`
`reverb_decay_time` (seconds=2^x, -6..6; 0 = 1 s, 2 = 4 s, 4 = 16 s), `reverb_size`
(0..1, default 0.5), `reverb_dry_wet` (default 0.25), `reverb_delay` (0..0.3 s pre-delay),
`reverb_pre_low_cutoff`, `reverb_pre_high_cutoff` (input HP/LP, 0..128 note scale),
`reverb_low_shelf_cutoff`, `reverb_low_shelf_gain` (-6..0 dB),
`reverb_high_shelf_cutoff`, `reverb_high_shelf_gain` (-6..0 dB), `reverb_chorus_amount`,
`reverb_chorus_frequency` (seconds=2^-x). (`reverb_feedback` appears in preset JSON but
has no engine parameter — it does nothing.)
- When: space for pads/keys; small (0.5-1.5 s, wet 0.15) for bass/lead glue; long
  (4-16 s, wet 0.3-0.5) for ambient.
- Interactions: macro → `reverb_dry_wet` amount 1.0 = "space" performance knob (mined:
  Super Nice Pluck PAD macro); velocity → reverb wet for dynamics.

## Phaser — `phaser_on`
`phaser_frequency` (seconds=2^-x, -5..2; default -3 = 8 s), `phaser_mod_depth` (0..1),
`phaser_center` (8..136), `phaser_blend` (0..2), `phaser_phase_offset`,
`phaser_feedback`, `phaser_dry_wet`, `phaser_sync`/`phaser_tempo` (Freeze..1/16,
default 8/1).
- When: '70s sweep on leads; slow + low wet = movement. Mined: Banana Wob routes
  macro1 → phaser_center + phaser_dry_wet.

## Flanger — `flanger_on`
`flanger_frequency` (seconds=2^-x, -5..2; default 2 = 0.25 s), `flanger_mod_depth`,
`flanger_center` (8..136), `flanger_phase_offset`, `flanger_feedback` (self-oscillation
  near 0.95), `flanger_dry_wet`, `flanger_sync`/`flanger_tempo` (default 4/1).
- When: jet sweep (fast, high wet), thickening (slow, low wet). Mined: Crescendo Bells
  macro2 (DISTORT) → flanger_dry_wet amount 0.78.

## EQ — `eq_on`
3 bands: `eq_low_{cutoff (8..136), gain (-15..15 dB), mode (Shelf|High Pass),
resonance}`, `eq_band_{cutoff, gain, mode (Shelf|Notch), resonance}`,
`eq_high_{cutoff, gain, mode (Shelf|Low Pass), resonance}`.
- When: cleanup (low-pass the highs on bright wavetables: high mode Low Pass, cutoff
  80-100), presence (band shelf +3 dB @ 2-4k region), sub control (low mode High Pass
  on non-bass oscs).
- Interactions: EQ runs in the chain (can be placed before/after distortion via
  `effect_chain_order`); mined Banana Wob routes macro1 → eq_low_cutoff +
  eq_band_gain (tone control knob).

## Compressor — `compressor_on`
`compressor_enabled_bands` (4: 0 Multiband [default], 1 Low Band, 2 High Band,
3 Single Band), per-band: `compressor_{low|band|high}_{lower_threshold, lower_ratio,
upper_threshold, upper_ratio, gain}`, `compressor_attack` (0..1 = 0..100% log),
`compressor_release` (0..1), `compressor_mix` (0..1).
- When: bass glue (single band, threshold -18..-10 dB, ratio 3-6, fast attack,
  release 0.25-0.5), level control after distortion. Corpus: 60/77 patches have it on
  (it's cheap glue). Mined classic: attack 0.75-1.0, release 0.25-0.5, modest gain.
- Interactions: env_1 → `compressor_mix` (mined: Super Nice Pluck, amount 0.395) =
  transient glue only while the note is alive.

## Order & global

- `effect_chain_order`: single integer = index into the permutation of the 9 FX slots.
  Leave it at the preset's value unless the target clearly needs a different order
  (e.g. EQ before distortion); change it only as its own A/B step.
- `bypass`: 0/1 global mute of the whole plugin — not an FX switch; leave it 0.
- Gated-section reminder: `filter_1_on`, `filter_2_on`, `filter_fx_on` are separate
  from these FX gates (see SKILL.md) — filters sit BEFORE the FX chain, so FX on
  but filter off means the FX hears the raw osc mix.

## Quick A/B verification of an FX edit

Same notes, two renders: `<fx>_on=0` vs `<fx>_on=1` (+ your params). Listen for:
distortion = added harmonics/noise floor up; delay = discrete repeats; reverb =
smear/tail; chorus = width + slow beating; phaser/flanger = comb sweep; EQ = band
gain tilt; compressor = reduced peaks, louder average. If the two renders sound the
same, the gate/param isn't taking — `show.py preset.json <fx>_` says why (SKILL rules).
