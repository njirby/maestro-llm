# DawDreamer Preset Loading — `set_parameter` vs `load_state`

[← Back to README](../README.md)

DawDreamer hosts VST3 plugins in-process for offline rendering. There are two mechanisms for applying preset state, with very different reliability on headless Linux.

## `set_parameter()` — the working path

`synth.set_parameter(index, normalized_value)` writes directly to the VST3 automation layer. It works headlessly on any platform, no display needed. DawDreamer exposes ~2,983 parameters for Vital; our mapping covers 758 of the 776 Vital JSON keys in `param_ranges.json`.

The mapping handles naming mismatches between Vital's JSON keys and DawDreamer's display names:

| Vital JSON key | DawDreamer name | Transform |
|---|---|---|
| `osc_1_on` | Oscillator 1 Switch | `osc` → `oscillator`, `on` → `switch` |
| `chorus_cutoff` | Chorus Filter Cutoff | insert `filter` |
| `*_dry_wet` | * Mix | `dry_wet` → `mix` |
| `env_1_attack` | Envelope 1 Attack | `env` → `envelope` |
| `lfo_1_delay_time` | LFO 1 Delay | drop `time` |
| `random_1_frequency` | Random LFO 1 Frequency | insert `lfo` |
| `compressor_band_lower_ratio` | Band Lower Ratio | drop `compressor_` prefix |
| `macro_control_1` | Macro 1 | `macro_control` → `macro` |

Values are normalized linearly: `(raw - min) / (max - min)` using bounds from `maestro/synth/param_ranges.json`.

**Limitation**: Cannot set wavetables, modulation routing, LFO shapes, or sample data — those live in the plugin's opaque internal state.

## `load_state()` — the broken path (on headless Linux)

`synth.load_state(path)` calls JUCE's `setStateInformation()` with a binary state blob. DawDreamer then creates a temporary editor window (`StandalonePluginWindow`) to force the plugin to commit the state — some plugins (including Vital) don't fully apply state without this GUI kick.

On headless Linux, the window creation either:
- **Fails silently** (no `DISPLAY`) → state never applies, all renders use default preset
- **Crashes with BadAtom X11 error** (Xvfb) → subprocess dies

The binary state format itself is correct (verified via round-trip: `save_state` → decode → re-encode → `load_state` preserves automation params). The issue is specifically that Vital requires the editor window to process the state blob.

## The 5-layer binary format (for reference)

DawDreamer's `load_state()` expects a file written by JUCE's `copyXmlToBinary()`:

```
Layer 1: VC2! header (8 bytes)
  LE uint32 0x21324356 + LE uint32 xml_byte_length

Layer 2: XML envelope
  <VST3PluginState><IComponent>{juce_base64}</IComponent></VST3PluginState>

Layer 3: JUCE custom base64
  Alphabet: .ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+
  Little-endian 6-bit packing, prefixed with "{byte_count}."

Layer 4: VstW + CcnK/FBCh wrapper (176 bytes header + payload)

Layer 5: Compact JSON + \0 + 16 zero bytes + "JUCEPrivateData"
```

The construction code is in `maestro/render/dawdreamer.py::_build_dawdreamer_state()`.

## Comparison with REAPER chunk format

| Layer | REAPER (`vst3_chunk`) | DawDreamer (`load_state`) |
|---|---|---|
| Outermost | Standard base64 of raw binary | VC2! header + XML + JUCE base64 |
| Wrapper start | 24-byte REAPER header + VstW | VstW (no REAPER header) |
| Inner | CcnK/FBCh/Vita + JSON + JUCE trailer | Same |

## Sources

- DawDreamer source: `loadStateInformation()` calls `setStateInformation()` then `StandalonePluginWindow`
- JUCE forum: [VST3 `setStateInformation` fails without prior `processBlock`](https://forum.juce.com/t/for-vst3-audioplugininstance-setstateinformation-does-not-work-unless-audio-has-processed-beforehand/68696) (April 2026)
- DawDreamer GitHub issues: #131 (Vital), #212 (mod matrix), #122 (TAL-NoiseMaker), #152 (load_state regression)

Verified 2026-04-27 against Vital 1.6.0 (Linux native) hosted by DawDreamer 0.8.3.
