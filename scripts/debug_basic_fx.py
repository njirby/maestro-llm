#!/usr/bin/env python3
"""Debug: test basic FX functionality with a simple built-in effect."""

import sys, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy

def test_basic_fx():
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

        # Try to add a simple built-in effect first
        print("Testing with built-in ReaEQ...")
        idx = rpr["RPR_TrackFX_AddByName"](track, "ReaEQ", False, 1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1

        if int(idx) >= 0:
            print(f"ReaEQ loaded at index {idx}")

            # Test FX name function with different buffer sizes
            for buf_size in [64, 256, 1024]:
                try:
                    name_result = rpr["RPR_TrackFX_GetFXName"](track, idx, "", buf_size)
                    print(f"  Buffer size {buf_size}: {name_result}")
                except Exception as e:
                    print(f"  Buffer size {buf_size}: exception - {e}")

            # Test parameter count
            param_count = rpr["RPR_TrackFX_GetNumParams"](track, idx)
            print(f"  Parameter count: {param_count}")

            # Test parameter names
            if isinstance(param_count, (tuple, list)):
                param_count = param_count[0] if param_count else 0
            param_count = int(param_count)

            for i in range(min(3, param_count)):  # Just first 3 params
                param_name_result = rpr["RPR_TrackFX_GetParamName"](track, idx, i, "", 64)
                print(f"  Param {i}: {param_name_result}")

            rpr["RPR_TrackFX_Delete"](track, idx)
        else:
            print(f"Failed to load ReaEQ: {idx}")

        # Now try Vital again with more investigation
        print("\nTesting Vital with VST2...")
        idx = rpr["RPR_TrackFX_AddByName"](track, "VSTi: Vital", False, 1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1

        if int(idx) >= 0:
            print(f"Vital loaded at index {idx}")
            time.sleep(3)  # Give it more time

            # Check parameter count
            param_count = rpr["RPR_TrackFX_GetNumParams"](track, idx)
            if isinstance(param_count, (tuple, list)):
                param_count = param_count[0] if param_count else 0
            print(f"  Vital parameter count: {param_count}")

            # Check if it's actually VST vs something else
            # Try to see if track state chunk contains info about the plugin
            try:
                state_result = rpr["RPR_GetTrackStateChunk"](track, "", 1024*1024, False)
                if isinstance(state_result, (tuple, list)) and len(state_result) > 2:
                    chunk_data = state_result[2]
                    print(f"  Track state chunk size: {len(chunk_data)} bytes")
                    # Look for plugin identifier in the chunk
                    if "vital" in chunk_data.lower():
                        print("  Found 'vital' in track state")
                    if "vst" in chunk_data.lower():
                        print("  Found 'vst' in track state")
                    # Print first few lines of the chunk to see format
                    lines = chunk_data.split('\n')[:10]
                    print("  First lines of state chunk:")
                    for line in lines:
                        print(f"    {line[:80]}")
            except Exception as e:
                print(f"  Track state chunk error: {e}")

if __name__ == "__main__":
    test_basic_fx()