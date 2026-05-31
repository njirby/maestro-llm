# Known Issues

## Cross-Platform Vital Parameter Mapping for Discrete Enums

**Discovered**: 2026-05-31
**Affects**: Rollout replay on macOS when rollouts were generated on Linux (or vice versa)

### Problem

Vital's discrete enum parameters (e.g. `osc_1_spectral_morph_type`, `osc_1_distortion_type`, `filter_1_model`, `beats_per_minute`) map normalized [0,1] REAPER param values to different integer values on macOS vs Linux.

Example: `osc_1_spectral_morph_type` has 12 values (0-11). The normalized value `0.636` maps to:
- **Linux**: type 7 (Bend) — correct
- **macOS**: type 10 (Quantize) — wrong

This causes audible differences when replaying Linux-generated rollouts on macOS. In the `bass_26fb984f` test case, the spectral morph mismatch eliminated the characteristic amplitude swell created by the "Bend" mode's wavetable frame interaction.

### Root Cause

REAPER's `TrackFX_SetParam()` takes a normalized [0,1] float. Vital internally maps this to a discrete integer, but the mapping formula differs between platform builds. The rollout batch steps use `TrackFX_SetParam` with normalized values, so discrete params are vulnerable to this mismatch.

The chunk apply step (which sets the full preset JSON via `vst_chunk`) uses native integer values and is NOT affected — it correctly sets `spectral_morph_type: 7` on both platforms.

### Affected Parameters

Any discrete-valued parameter in `maestro/synth/param_ranges.json` with `"discrete": true`. Key ones:
- `osc_N_spectral_morph_type` (12 values)
- `osc_N_distortion_type` (12 values)
- `filter_N_model` (8 values)
- `filter_N_style` (varies)
- `distortion_type` (varies)
- `compressor_enabled_bands` (4 values)
- `delay_style` (4 values)

### Workarounds

1. **For replay testing**: After each batch param set, read back discrete params and correct any that don't match expected values
2. **For training data**: Not affected — the model learns normalized values that are correct for the target platform
3. **Long-term fix**: Set discrete params via chunk JSON (native integers) instead of `TrackFX_SetParam` (normalized floats)
