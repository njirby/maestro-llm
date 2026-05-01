# VST Chunk Format — Constructing Vital Presets from Scratch

[← Back to README](../README.md)

Vital doesn't expose presets through REAPER's VST preset API (`TrackFX_SetPreset` returns false; `TrackFX_GetPresetIndex` reports 0 presets). The only programmatic way to change Vital's state is via `TrackFX_SetNamedConfigParm(track, fx, "vst_chunk", base64_encoded_chunk)`.

The chunk is a binary envelope around the `.vital` preset JSON:

```
[REAPER wrapper — 24 bytes]
  0-3    LE uint32  total_size - 16
  4-7    LE uint32  1 (constant)
  8-11   "VstW"

[VST standard header — 160 bytes]
  24-27  "CcnK"
  28-31  BE uint32  total_size - 40
  32-35  "FBCh"
  36-39  BE uint32  2 (version)
  40-43  "Vita"
  44-47  BE uint32  0x00010600 (Vital 1.6.0)
  48-179 zeros
  180-183 BE uint32  json_size + 32

[Preset JSON — variable]
  184..  compact JSON (separators=(',',':'))

[JUCE suffix — 40 bytes]
  17 zero bytes + "JUCEPrivateData" + 8 zero bytes
```

Three size fields depend on the JSON payload; everything else is constant. This means we can **construct a complete VST chunk from any `.vital` preset dict without reading the current chunk first**:

```python
import base64, json, struct

def build_vital_chunk(preset_json: dict) -> bytes:
    json_bytes = json.dumps(preset_json, separators=(',', ':')).encode('utf-8')
    json_size = len(json_bytes)
    suffix = b'\x00' * 17 + b'JUCEPrivateData' + b'\x00' * 8
    total = 184 + json_size + len(suffix)
    prefix = bytearray(184)
    struct.pack_into('<I', prefix, 0, total - 16)
    struct.pack_into('<I', prefix, 4, 1)
    prefix[8:12] = b'VstW'
    struct.pack_into('>I', prefix, 12, 8)
    struct.pack_into('>I', prefix, 16, 1)
    prefix[24:28] = b'CcnK'
    struct.pack_into('>I', prefix, 28, total - 40)
    prefix[32:36] = b'FBCh'
    struct.pack_into('>I', prefix, 36, 2)
    prefix[40:44] = b'Vita'
    struct.pack_into('>I', prefix, 44, 0x00010600)
    struct.pack_into('>I', prefix, 180, json_size + 32)
    return bytes(prefix) + json_bytes + suffix
```

This eliminates the old read→decode→parse→modify→re-encode round-trip. For wavetable swaps: take the preset dict, swap `preset["settings"]["wavetables"][osc_idx]`, call `build_vital_chunk()`, base64 encode, and apply via reapy's `RPR.TrackFX_SetNamedConfigParm(track, 0, "vst3_chunk", encoded)`. No chunk read step at all.

**Gotcha**: `synth_version` must be `"1.6.0"` (or whatever version the running Vital reports). Vital silently rejects chunks with unrecognized versions — `SetNamedConfigParm` returns success but the data is ignored. See [vital-reaper-gotchas.md](vital-reaper-gotchas.md) for the full story and other pitfalls.

Verified 2026-04-30 against Vital 1.6.0 (native Linux VST3) in REAPER on Linux.
