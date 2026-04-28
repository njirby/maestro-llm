# DawDreamer State Format — Loading Full Vital Presets Offline

[← Back to README](../README.md)

DawDreamer hosts VST3 plugins in-process for offline rendering. Unlike REAPER (which exposes `vst3_chunk` via `TrackFX_SetNamedConfigParm`), DawDreamer's only preset-loading API is `plugin.load_state(path)` — and it silently ignores files that aren't in the correct binary format.

## The problem

Calling `synth.load_state("preset.vital")` with a raw `.vital` JSON file **returns without error but does not apply the preset**. Every render uses the default Vital init sound. This is because JUCE's VST3 hosting layer expects a specific binary envelope, not raw plugin data.

`synth.set_parameter(idx, value)` works for the ~700 numeric parameters DawDreamer exposes, but **cannot set wavetables, modulation routing, LFO shapes, or sample data** — those live inside the plugin's opaque state blob.

## The 5-layer binary format

DawDreamer's `load_state()` expects a file written by JUCE's `copyXmlToBinary()`. The format wraps the Vital preset JSON in 5 nested layers:

```
Layer 1: VC2! header (8 bytes)
  0-3    LE uint32  0x21324356  ("VC2!" magic)
  4-7    LE uint32  xml_byte_length (including null terminator)

Layer 2: XML envelope (variable)
  <?xml version="1.0" encoding="UTF-8"?>
  <VST3PluginState>
    <IComponent>{base64_encoded_blob}</IComponent>
  </VST3PluginState>\0

Layer 3: JUCE custom base64 (variable)
  NOT standard base64 — JUCE uses its own alphabet and bit packing.
  Decodes to the VstW/FBCh binary blob (Layer 4).

Layer 4: VstW + FBCh wrapper (variable)
  0-3    "VstW"
  4-7    BE uint32  8
  8-11   BE uint32  1
  12-15  BE uint32  0
  16-19  "CcnK"
  20-23  BE int32   body_size
  24-27  "FBCh"
  28-31  BE int32   2  (version)
  32-35  "Vita"     (plugin ID)
  36-39  BE int32   0x00010600  (Vital 1.6.0)
  40-43  BE int32   0  (numPrograms)
  44-171 zeros      (128-byte future block)
  172-175 BE int32  chunk_size
  176..  chunk_data (Layer 5)

Layer 5: Vital JSON + JUCE trailer
  Compact JSON (separators=(',',':')) + \0
  + 16 zero bytes + "JUCEPrivateData"
```

## JUCE custom base64

JUCE's `MemoryBlock::toBase64Encoding()` uses a non-standard alphabet and little-endian bit packing:

```
Alphabet: .ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+
```

The encoding reads 6 bits at a time from the byte stream in little-endian order. The encoded string is prefixed with `{byte_count}.` (decimal size + dot separator).

```python
_JUCE_B64_TABLE = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"

def _juce_base64_encode(data: bytes) -> str:
    size = len(data)
    num_chars = ((size * 8) + 5) // 6
    chars = []
    for i in range(num_chars):
        bit_pos = i * 6
        byte_idx = bit_pos >> 3
        bit_off = bit_pos & 7
        if byte_idx + 1 < size:
            word = data[byte_idx] | (data[byte_idx + 1] << 8)
        else:
            word = data[byte_idx] if byte_idx < size else 0
        val = (word >> bit_off) & 0x3F
        chars.append(_JUCE_B64_TABLE[val])
    return f"{size}.{''.join(chars)}"
```

## Constructing the state from a preset dict

```python
def _build_dawdreamer_state(preset_dict: dict) -> bytes:
    json_bytes = json.dumps(preset_dict, separators=(",", ":")).encode("utf-8") + b"\x00"
    juce_trailer = b"\x00" * 16 + b"JUCEPrivateData"
    chunk_data = json_bytes + juce_trailer
    chunk_size = len(chunk_data)

    future = b"\x00" * 128
    body_size = 4 + 4 + 4 + 4 + 4 + 128 + 4 + chunk_size

    icomp = (
        b"VstW" + struct.pack(">III", 8, 1, 0)
        + b"CcnK" + struct.pack(">i", body_size)
        + b"FBCh" + struct.pack(">i", 2)
        + b"Vita" + struct.pack(">i", 0x00010600)
        + struct.pack(">i", 0) + future
        + struct.pack(">i", chunk_size) + chunk_data
    )

    b64 = _juce_base64_encode(icomp)
    xml_str = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f"<VST3PluginState><IComponent>{b64}</IComponent></VST3PluginState>"
    )
    xml_bytes = xml_str.encode("utf-8") + b"\x00"
    return struct.pack("<II", 0x21324356, len(xml_bytes)) + xml_bytes
```

Write the returned bytes to a temp file and pass to `synth.load_state(path)`. Vital's `setStateInformation()` will parse the JSON via `LoadSave::jsonToState()`, which handles wavetable data, modulation routing, LFO shapes, and all other state — not just numeric parameters.

## Comparison with REAPER chunk format

The [REAPER VST chunk format](vital-chunk-format.md) has a different outer wrapper:

| Layer | REAPER (`vst3_chunk`) | DawDreamer (`load_state`) |
|---|---|---|
| Outermost | Standard base64 of raw binary | VC2! header + XML + JUCE base64 |
| Wrapper start | 24-byte REAPER header + VstW | VstW (no REAPER header) |
| Inner | CcnK/FBCh/Vita + JSON + JUCE trailer | Same |

The inner CcnK/FBCh/Vita structure is identical — the difference is in how the outer envelope is constructed and encoded.

## Sources

- DawDreamer source: `ProcessorBase::loadStateFrom()` calls JUCE `setStateInformation()` after `copyBinaryToXml()`
- JUCE source: `juce_VST3Common.h` `writeStateToFile()` / `copyXmlToBinary()` / `MemoryBlock::toBase64Encoding()`
- Vital source: `SynthGuiInterface::setStateInformation()` → `LoadSave::jsonToState()`
- DawDreamer GitHub issues: #131, #212, #23, #152

Verified 2026-04-27 against Vital 1.6.0 (Linux native) hosted by DawDreamer.
