#!/usr/bin/env python3
"""Debug: check what FX format was actually loaded and its available parameters."""

import sys, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy

def debug_fx_format():
    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}

        proj = 0
        if int(rpr["RPR_CountTracks"](proj)) == 0:
            rpr["RPR_InsertTrackAtIndex"](0, True)
        track = rpr["RPR_GetTrack"](proj, 0)

        # Check existing FX
        fx_count = int(rpr["RPR_TrackFX_GetCount"](track))
        print(f"FX count: {fx_count}")

        if fx_count > 0:
            for i in range(fx_count):
                # Get FX name
                fx_name_result = rpr["RPR_TrackFX_GetFXName"](track, i, "", 256)
                if isinstance(fx_name_result, (tuple, list)):
                    fx_name = fx_name_result[1] if len(fx_name_result) > 1 else "unknown"
                else:
                    fx_name = str(fx_name_result)
                print(f"FX {i}: {fx_name}")

                # Try to get different parameter types
                for param_name in ["vst_chunk", "vst3_chunk", "lv2_state", "clap_state", ""]:
                    try:
                        result = rpr["RPR_TrackFX_GetNamedConfigParm"](track, i, param_name, "", 1024*1024)
                        if isinstance(result, (tuple, list)) and len(result) > 2:
                            value = result[2]
                            if value:
                                print(f"  {param_name}: {len(value)} bytes")
                            else:
                                print(f"  {param_name}: empty/failed")
                        else:
                            print(f"  {param_name}: failed ({result})")
                    except Exception as e:
                        print(f"  {param_name}: exception {e}")

        # Try to add VST2 Vital if not present
        if fx_count == 0:
            print("No FX loaded, trying to add Vital...")
            for name in ("VSTi: Vital", "VST: Vital", "LV2: Vital", "VST3i: Vital"):
                print(f"  Trying: {name}")
                idx = rpr["RPR_TrackFX_AddByName"](track, name, False, 1)
                if isinstance(idx, (tuple, list)):
                    idx = idx[0] if idx else -1
                if int(idx) >= 0:
                    print(f"  Success: loaded {name} at index {idx}")
                    time.sleep(2)
                    # Get the actual name
                    fx_name_result = rpr["RPR_TrackFX_GetFXName"](track, idx, "", 256)
                    if isinstance(fx_name_result, (tuple, list)):
                        actual_name = fx_name_result[1] if len(fx_name_result) > 1 else "unknown"
                    else:
                        actual_name = str(fx_name_result)
                    print(f"  Actual FX name: {actual_name}")
                    break
            else:
                print("  Failed to load any Vital format")

if __name__ == "__main__":
    debug_fx_format()