# Vital subsystem taxonomy

Mapping from Vital parameter families to the seven presentation subsystems used in
the DIAGNOSIS plan and subsystem-batched execution loop.

| Presentation subsystem | Vital parameter families | Audible role |
|---|---|---|
| `oscillator` | `osc_1_*`, `osc_2_*`, `osc_3_*` (level, pitch, unison voices/detune/blend, wavetable frame, phase, distortion) | Source material — layering, width, raw harmonic content |
| `envelope` | `env_1_*` … `env_6_*` (attack, decay, sustain, release, hold, shape curves) | Dynamic shaping over time — how the sound starts, holds, and fades |
| `filter` | `filter_1_*`, `filter_2_*` (cutoff, resonance, blend, drive, model) | Tonal coloring — brightness, resonance, filter type |
| `lfo` | `lfo_1_*` … `lfo_8_*` (rate, shape, phase, smoothing) | Periodic motion — wobble, tremolo, vibrato, grid-synced pulses |
| `fx` | `chorus_*`, `delay_*`, `reverb_*`, `distortion_*`, `compressor_*`, `eq_*`, `phaser_*`, `flanger_*` | Post-processing — space, grit, movement layered on top of the raw synth voice |
| `modulation` | `modulation_1_*` … `modulation_64_*` (source, destination, amount, bipolar, stereo) | Wiring between modulators and parameters — organises how LFOs/envelopes reshape the voice |
| `macro` | `macro_control_1` … `macro_control_4`, macro mapping | Top-level expressive handles mapped to multiple destinations |

## Batch ordering rationale

The execution loop applies subsystem batches in this fixed order:

```
oscillator → envelope → filter → lfo → fx → modulation → macro
```

Chosen so each batch's audible effect can be evaluated against the cumulative
state from prior batches: set the source first, shape its envelope, colour it
with a filter, layer motion, add space/FX, then wire up dynamic routings and
macro handles. Reordering would make intermediate renders confusing (e.g.
hearing LFO motion before the oscillator has a shaped envelope would obscure
what the LFO is doing).

## Common gotchas

- **Modulation slots point to params that may be off** — a `lfo_1 → filter_1_cutoff` route with amount 0.5 is silent until `filter_1_on=1`. Don't infer LFO motion from the modulation table alone; check target param on/off state.
- **Multiple envelopes target the same param** — `env_1` is typically the amp envelope but others can route to filter/pitch. Always check modulation routes when deciding which envelope owns a behavior.
- **FX chain order matters** — Vital's FX are fixed-slot: chorus → delay → reverb → distortion → compressor → EQ → phaser → flanger. A subtle distortion before reverb sounds different than after; the order isn't configurable.
- **Wavetable name is cosmetic, wavetable data is what sounds** — two presets with the same `osc_1_wavetable_frame_name` could have different actual wavetables if one was edited. In the maestro pipeline we apply wavetables by looking up the name in `data/wavetable_lib.json` so the audio matches.
