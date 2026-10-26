#!/usr/bin/env python3
"""Test whether TrackFX_SetPreset can load a .vital file from disk.

Requires REAPER running with the Lua dispatcher (maestro_boot.lua defer loop).

Test plan:
  1. Write a custom .vital preset with a known filter cutoff value to
     ~/.local/share/vital/User/Presets/maestro_test_preset.vital
  2. Dispatch Lua that calls TrackFX_SetPreset to load it
  3. Dispatch Lua that reads back the filter cutoff via TrackFX_GetParam
  4. Compare the readback value to what we wrote

If this works, we can replace the entire chunk encode/decode pipeline
with: write .vital JSON to disk → TrackFX_SetPreset → done.
"""
import json
import os
import shutil
import time
from pathlib import Path

VITAL_USER_PRESETS = Path.home() / ".local/share/vital/User/Presets"
TEST_PRESET_NAME = "maestro_test_preset"
TEST_PRESET_PATH = VITAL_USER_PRESETS / f"{TEST_PRESET_NAME}.vital"
INIT_PRESET = Path(__file__).resolve().parents[1] / "maestro/synth/init_preset.json"

TRIGGER = Path("/tmp/maestro_lua_dispatch.trigger")
ACTION = Path("/tmp/maestro_lua_action.lua")
RESULT = Path("/tmp/maestro_lua_result.json")

LUA_WRITE_JSON = """\
local function write_json(path, tbl)
  local f = io.open(path, "w")
  if not f then return end
  local parts = {}
  for k, v in pairs(tbl) do
    if type(v) == "string" then
      parts[#parts + 1] = string.format('"%s":"%s"', k, v)
    elseif type(v) == "table" then
      local items = {}
      for _, s in ipairs(v) do items[#items + 1] = '"' .. s .. '"' end
      parts[#parts + 1] = string.format('"%s":[%s]', k, table.concat(items, ","))
    else
      parts[#parts + 1] = string.format('"%s":%s', k, tostring(v))
    end
  end
  f:write("{" .. table.concat(parts, ",") .. "}")
  f:close()
end
"""


def dispatch_lua(lua_code: str, timeout: float = 10.0) -> dict:
    """Write Lua to action file, touch trigger, poll for result."""
    RESULT.unlink(missing_ok=True)
    ACTION.write_text(lua_code)
    TRIGGER.touch()
    t0 = time.time()
    while not RESULT.exists():
        if time.time() - t0 > timeout:
            raise TimeoutError(f"Lua dispatch timed out after {timeout}s")
        time.sleep(0.1)
    with open(RESULT) as f:
        return json.load(f)


def main():
    # ---- Step 0: Load init preset and modify filter cutoff ----
    with open(INIT_PRESET) as f:
        preset = json.load(f)

    # Set filter_1_cutoff to a distinctive value (native Vital units: MIDI note)
    # Default is ~60, let's set it to 90 (very high, easy to detect)
    test_cutoff = 90.0
    preset["settings"]["filter_1_cutoff"] = test_cutoff
    preset["preset_name"] = TEST_PRESET_NAME

    # ---- Step 1: Write preset to Vital's User directory ----
    VITAL_USER_PRESETS.mkdir(parents=True, exist_ok=True)
    with open(TEST_PRESET_PATH, "w") as f:
        json.dump(preset, f)
    print(f"[1] Wrote test preset to {TEST_PRESET_PATH}")
    print(f"    filter_1_cutoff = {test_cutoff}")

    # ---- Step 2: Check REAPER dispatcher is alive ----
    print("[2] Checking REAPER dispatcher...")
    try:
        r = dispatch_lua(
            LUA_WRITE_JSON
            + 'local RESULT = "/tmp/maestro_lua_result.json"\n'
            + 'write_json(RESULT, {status="ok", msg="dispatcher alive"})\n',
            timeout=5.0,
        )
        print(f"    Dispatcher: {r}")
    except TimeoutError:
        print("    ERROR: REAPER Lua dispatcher not responding. Is REAPER running?")
        return

    # ---- Step 3: Ensure track 0 has Vital loaded ----
    print("[3] Checking Vital on track 0...")
    r = dispatch_lua(
        LUA_WRITE_JSON
        + 'local RESULT = "/tmp/maestro_lua_result.json"\n'
        + 'local track = reaper.GetTrack(0, 0)\n'
        + 'if not track then\n'
        + '  write_json(RESULT, {status="error", msg="no track 0"})\n'
        + '  return\n'
        + 'end\n'
        + 'local n_fx = reaper.TrackFX_GetCount(track)\n'
        + 'if n_fx == 0 then\n'
        + '  write_json(RESULT, {status="error", msg="no FX on track 0"})\n'
        + '  return\n'
        + 'end\n'
        + 'local _, fx_name = reaper.TrackFX_GetFXName(track, 0, "")\n'
        + 'write_json(RESULT, {status="ok", fx_name=fx_name, n_fx=n_fx})\n'
    )
    print(f"    FX: {r}")
    if r.get("status") != "ok":
        print("    ERROR: Need Vital on track 0 FX slot 0")
        return

    # ---- Step 4: Read current filter cutoff (before) ----
    print("[4] Reading filter cutoff BEFORE preset load...")
    r = dispatch_lua(
        LUA_WRITE_JSON
        + 'local RESULT = "/tmp/maestro_lua_result.json"\n'
        + 'local track = reaper.GetTrack(0, 0)\n'
        + 'local n = reaper.TrackFX_GetNumParams(track, 0)\n'
        + 'local cutoff_idx = -1\n'
        + 'for i = 0, n - 1 do\n'
        + '  local _, name = reaper.TrackFX_GetParamName(track, 0, i, "")\n'
        + '  if name == "Filter 1 Cutoff" then cutoff_idx = i; break end\n'
        + 'end\n'
        + 'if cutoff_idx < 0 then\n'
        + '  write_json(RESULT, {status="error", msg="Filter 1 Cutoff not found"})\n'
        + '  return\n'
        + 'end\n'
        + 'local val, _, _ = reaper.TrackFX_GetParam(track, 0, cutoff_idx)\n'
        + 'write_json(RESULT, {status="ok", cutoff_idx=cutoff_idx, cutoff_norm=val})\n'
    )
    print(f"    Before: {r}")
    before_cutoff = r.get("cutoff_norm", -1)

    # ---- Step 5: Try TrackFX_SetPreset ----
    # Try several name formats to see what works
    attempts = [
        # Just the preset name (no extension, no path)
        TEST_PRESET_NAME,
        # With .vital extension
        f"{TEST_PRESET_NAME}.vital",
        # Full path
        str(TEST_PRESET_PATH),
        # Relative to Vital data dir: "User/Presets/name.vital"
        f"User/Presets/{TEST_PRESET_NAME}.vital",
        # Try with pack prefix: "User - name"
        f"User - {TEST_PRESET_NAME}",
    ]

    for i, name in enumerate(attempts):
        print(f"\n[5.{i}] Trying TrackFX_SetPreset with: {name!r}")
        r = dispatch_lua(
            LUA_WRITE_JSON
            + 'local RESULT = "/tmp/maestro_lua_result.json"\n'
            + 'local track = reaper.GetTrack(0, 0)\n'
            + f'local ret = reaper.TrackFX_SetPreset(track, 0, {json.dumps(name)})\n'
            + 'write_json(RESULT, {status="ok", ret=tostring(ret)})\n'
        )
        print(f"    Result: {r}")

        if r.get("ret") == "true" or r.get("ret") is True:
            print("    >>> SetPreset returned true!")
            # Give Vital a moment to load
            time.sleep(0.5)

            # Read back filter cutoff
            r2 = dispatch_lua(
                LUA_WRITE_JSON
                + 'local RESULT = "/tmp/maestro_lua_result.json"\n'
                + 'local track = reaper.GetTrack(0, 0)\n'
                + 'local n = reaper.TrackFX_GetNumParams(track, 0)\n'
                + 'local cutoff_idx = -1\n'
                + 'for i = 0, n - 1 do\n'
                + '  local _, name = reaper.TrackFX_GetParamName(track, 0, i, "")\n'
                + '  if name == "Filter 1 Cutoff" then cutoff_idx = i; break end\n'
                + 'end\n'
                + 'local val, _, _ = reaper.TrackFX_GetParam(track, 0, cutoff_idx)\n'
                + 'write_json(RESULT, {status="ok", cutoff_norm=val})\n'
            )
            after_cutoff = r2.get("cutoff_norm", -1)
            print(f"    Cutoff after: {after_cutoff} (before: {before_cutoff})")

            if abs(after_cutoff - before_cutoff) > 0.01:
                print(f"    >>> CUTOFF CHANGED! Preset load WORKED with format: {name!r}")
            else:
                print(f"    Cutoff didn't change — preset may not have loaded correctly")

    # ---- Step 6: Also try TrackFX_GetPreset to see current name ----
    print("\n[6] Reading current preset name via TrackFX_GetPreset...")
    r = dispatch_lua(
        LUA_WRITE_JSON
        + 'local RESULT = "/tmp/maestro_lua_result.json"\n'
        + 'local track = reaper.GetTrack(0, 0)\n'
        + 'local ret, name = reaper.TrackFX_GetPreset(track, 0, "")\n'
        + 'write_json(RESULT, {status="ok", ret=tostring(ret), preset_name=name or "nil"})\n'
    )
    print(f"    Current preset: {r}")

    # Cleanup
    print(f"\n[cleanup] Leaving test preset at {TEST_PRESET_PATH} for inspection")


if __name__ == "__main__":
    main()
