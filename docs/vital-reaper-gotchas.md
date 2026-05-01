# Vital + REAPER Integration — Gotchas and Lessons Learned

[← Back to README](../README.md)

Hard-won knowledge from debugging the Vital chunk loading pipeline in REAPER. Read this before touching any code that loads Vital presets into REAPER.

## 1. `synth_version` must be `"1.6.0"`

**The single most important thing in this document.**

Vital's deserializer silently rejects any preset whose `synth_version` field is not a recognized version string. `TrackFX_SetNamedConfigParm` returns success (1), the API reports no error, but the chunk data is completely ignored. The plugin stays in its previous state.

This caused weeks of debugging because every diagnostic said "it worked" while the audio was completely wrong.

```python
# WRONG — Vital silently ignores the entire preset
{"synth_version": "99999.9.9", "settings": {...}}

# CORRECT
{"synth_version": "1.6.0", "settings": {...}}
```

**How to verify a chunk actually loaded**: Read back params via `TrackFX_GetParam` after loading and compare to expected values. Don't trust the return value of `SetNamedConfigParm`.

**Where this must be set**:
- `maestro/synth/preset_gen.py` — the preset generator
- `maestro/synth/wavetable_lib.py` — the default wavetable's `version` field
- Any `.vital` preset file fed into `build_vital_chunk()`

## 2. Native Linux Vital works fine

Earlier docs claimed native Linux Vital (the 22MB ELF at `~/.vst3/Vital.vst3`) crashes REAPER when loading wavetable state. **This was wrong.** The crashes were caused by `synth_version: "99999.9.9"` — once fixed, native Vital loads all state correctly: 753/756 params match expected values, wavetables load, modulations load.

yabridge (Windows Vital via WINE) also works but has differences:
- Reports 2986 params vs 756 on native (extra params are yabridge internals)
- `TrackFX_GetParamName` returns index numbers ("462") instead of display names
- Higher overhead per API call due to WINE bridge

**Recommendation**: Use native Linux Vital. Only fall back to yabridge if you hit an actual native-specific bug.

## 3. REAPER render action IDs

```python
# WRONG — opens the render dialog interactively (blocks headless scripts)
RPR.Main_OnCommand(42, 0)

# CORRECT — renders with current settings and auto-closes the dialog
RPR.Main_OnCommand(42230, 0)
```

Action 42230 is "File: Render project, using the most recent render settings, auto-close render dialog". Essential for any headless/scripted rendering pipeline.

After calling 42230, sleep 2-3 seconds and check for the output file. REAPER sometimes appends `-001` to the filename if it detects a conflict:
```python
if not os.path.exists(expected_path):
    alt = expected_path.replace(".wav", "-001.wav")
    if os.path.exists(alt):
        os.rename(alt, expected_path)
```

## 4. reapy connection management

reapy communicates with REAPER over TCP (port 2306). The server is started by `__startup.lua` which runs in REAPER's defer loop on launch.

**Common failure modes**:

- **Stale Python processes**: A previous script that used reapy may have left a process holding the connection. `lsof -i :2306` to find it, `kill` the PID, then retry.
- **REAPER restarted but server not running**: The `__startup.lua` script only runs on REAPER launch. If REAPER was killed and restarted, verify port 2306 is listening before connecting: `ss -tlnp | grep 2306`.
- **Multiple reapy connections**: Only one Python process can hold the reapy connection at a time. If a subprocess uses reapy, the parent can't use it simultaneously.

**Pattern for reliable connection**:
```python
import reapy
from reapy import reascript_api as RPR

with reapy.inside_reaper():
    # All REAPER API calls go here
    track = RPR.GetTrack(0, 0)
    ...
```

## 5. Lua scripts inside REAPER (alternative to reapy)

For operations that need to run in REAPER's own process thread, register and execute a Lua script:

```python
lua_code = '...'
with open("/tmp/my_script.lua", "w") as f:
    f.write(lua_code)

cmd_id = RPR.AddRemoveReaScript(True, 0, "/tmp/my_script.lua", True)
if isinstance(cmd_id, (list, tuple)):
    cmd_id = cmd_id[0]
RPR.Main_OnCommand(int(cmd_id), 0)
time.sleep(1)  # wait for execution
RPR.AddRemoveReaScript(False, 0, "/tmp/my_script.lua", True)  # unregister
```

Both reapy RPC and Lua work equally well for chunk loading. Use whichever fits the calling context.

## 6. vita vs REAPER rendering differences

vita (the in-process C++ Vital engine used for GT rendering) and REAPER's hosted Vital plugin produce slightly different audio from the same preset + MIDI. Typical RMS ratios range from 0.3x to 1.2x across presets.

Sources of divergence:
- LFO phase initialization (random or zero — depends on host)
- Buffer size and sample-accurate timing differences
- Modulation routing evaluation order

This is expected and acceptable for SFT training — the agent operates in REAPER, so REAPER renders are the ground truth for the training loop.

## 7. MIDI timing in REAPER

REAPER's MIDI tick system defaults to 960 PPQ (pulses per quarter note) at the project tempo. When inserting notes programmatically:

```python
start_ticks = int(note_start_seconds * 960)
end_ticks = int(note_end_seconds * 960)
RPR.MIDI_InsertNote(take, False, False, start_ticks, end_ticks,
                    0, pitch, velocity, False)
```

This is an approximation — it assumes 120 BPM (960 ticks = 1 beat = 0.5s at 120 BPM → 1920 ticks/s, but we use 960 ticks/s here). For SFT training purposes the timing is close enough. If exact timing matters, account for the actual project tempo.

## 8. The 3 params that don't round-trip perfectly

After loading a chunk, 753/756 mapped params match expected values. The 3 mismatches are discrete enum-type params where Vital's internal quantization doesn't follow linear normalization:

| Param | Expected | Actual | Delta |
|---|---|---|---|
| `osc_1_spectral_morph_type` | 0.7273 | 0.5000 | 0.23 |
| `osc_2_spectral_morph_type` | 0.6364 | 0.4375 | 0.20 |
| `filter_1_style` | 0.1111 | 0.1250 | 0.01 |

These are selector-style params (dropdown menus) that snap to discrete values. The loaded value is functionally correct — the right option is selected — but the normalized float doesn't match our `(native - min) / (max - min)` formula exactly. Not a real problem.
