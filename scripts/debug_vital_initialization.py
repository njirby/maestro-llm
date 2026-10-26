#!/usr/bin/env python3
"""Debug: try to get Vital to initialize properly and provide state data."""

import sys, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy

def test_vital_initialization():
    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}

        proj = 0
        if int(rpr["RPR_CountTracks"](proj)) == 0:
            rpr["RPR_InsertTrackAtIndex"](0, True)
        track = rpr["RPR_GetTrack"](proj, 0)

        # Clear existing FX
        fx_count = int(rpr["RPR_TrackFX_GetCount"](track))
        for i in range(fx_count-1, -1, -1):
            rpr["RPR_TrackFX_Delete"](track, i)

        print("Loading Vital and trying to initialize state...")
        idx = rpr["RPR_TrackFX_AddByName"](track, "VST3i: Vital", False, 1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1

        if int(idx) < 0:
            print("Failed to load Vital")
            return

        print(f"Vital loaded at index {idx}")

        # Try different initialization strategies
        strategies = [
            ("Immediate", 0),
            ("Wait 2s", 2),
            ("Wait 5s", 5),
            ("Wait 10s", 10)
        ]

        for strategy_name, wait_time in strategies:
            print(f"\nStrategy: {strategy_name}")
            if wait_time > 0:
                time.sleep(wait_time)

            # Check parameter count first
            param_count = rpr["RPR_TrackFX_GetNumParams"](track, idx)
            if isinstance(param_count, (tuple, list)):
                param_count = param_count[0] if param_count else 0
            print(f"  Parameter count: {param_count}")

            # Try to read chunks
            for param_name in ["vst_chunk", "vst3_chunk"]:
                try:
                    result = rpr["RPR_TrackFX_GetNamedConfigParm"](track, idx, param_name, "", 2*1024*1024)
                    if isinstance(result, (tuple, list)):
                        success = result[0] if len(result) > 0 else False
                        value = result[2] if len(result) > 2 else ""
                        print(f"    {param_name}: success={success}, bytes={len(value) if value else 0}")
                        if value and len(value) > 100:
                            print(f"    SUCCESS! Found chunk data")
                            return True
                except Exception as e:
                    print(f"    {param_name}: exception - {e}")

            # Try to poke the plugin to wake it up
            if strategy_name == "Wait 2s":
                print("  Trying to activate plugin by setting a parameter...")
                try:
                    # Set volume parameter (usually parameter 0 or 1)
                    rpr["RPR_TrackFX_SetParam"](track, idx, 0, 0.5)
                    time.sleep(0.1)
                    rpr["RPR_TrackFX_SetParam"](track, idx, 0, 0.7)
                    print("  Set some parameters")
                except Exception as e:
                    print(f"  Parameter setting failed: {e}")

            if strategy_name == "Wait 5s":
                print("  Trying to open plugin UI...")
                try:
                    # Try to open the plugin UI to force initialization
                    rpr["RPR_TrackFX_SetOpen"](track, idx, 1)  # Open UI
                    time.sleep(1)
                    rpr["RPR_TrackFX_SetOpen"](track, idx, 0)  # Close UI
                    print("  Opened and closed UI")
                except Exception as e:
                    print(f"  UI open/close failed: {e}")

        print("  All strategies failed to get chunk data")
        return False

if __name__ == "__main__":
    success = test_vital_initialization()
    if success:
        print("\nSUCCESS: Found a way to get Vital state data")
    else:
        print("\nFAILED: Could not get Vital to provide state data")