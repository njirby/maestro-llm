#!/usr/bin/env python3
"""Debug: properly extract FX name and test VST chunk reading for Vital."""

import sys, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy

def test_vital_chunk_reading():
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

        # Try different Vital formats and extract names correctly
        vital_formats = [
            "VSTi: Vital",
            "VST3i: Vital",
            "VST: Vital",
            "VST3: Vital (Vital Audio)"
        ]

        for format_name in vital_formats:
            print(f"\nTesting format: {format_name}")
            idx = rpr["RPR_TrackFX_AddByName"](track, format_name, False, 1)
            if isinstance(idx, (tuple, list)):
                idx = idx[0] if idx else -1

            if int(idx) >= 0:
                print(f"  Loaded at index {idx}")
                time.sleep(3)  # Give it time to initialize

                # Extract FX name correctly (4th element in the tuple)
                name_result = rpr["RPR_TrackFX_GetFXName"](track, idx, "", 256)
                if isinstance(name_result, (tuple, list)) and len(name_result) >= 4:
                    actual_name = name_result[3]  # This is the actual name
                    print(f"  Actual FX name: '{actual_name}'")
                else:
                    print(f"  FX name result: {name_result}")

                # Test VST chunk reading with proper error handling
                for param_name in ["vst_chunk", "vst3_chunk"]:
                    try:
                        print(f"  Testing {param_name}...")
                        result = rpr["RPR_TrackFX_GetNamedConfigParm"](track, idx, param_name, "", 2*1024*1024)
                        if isinstance(result, (tuple, list)):
                            success = result[0] if len(result) > 0 else False
                            value = result[2] if len(result) > 2 else ""
                            print(f"    Success flag: {success}")
                            print(f"    Data length: {len(value) if value else 0} bytes")
                            if value and len(value) > 100:
                                print(f"    First 100 chars: {value[:100]}")
                                # This format should work with VitalController!
                                print(f"  SUCCESS: {format_name} works with {param_name}")
                                return format_name, param_name
                            else:
                                print(f"    No meaningful data")
                        else:
                            print(f"    Unexpected result format: {result}")
                    except Exception as e:
                        print(f"    Exception: {e}")

                # Clean up before next test
                rpr["RPR_TrackFX_Delete"](track, idx)
                time.sleep(0.5)
            else:
                print(f"  Failed to load")

        print("\nNo working VST format found")
        return None, None

if __name__ == "__main__":
    working_format, working_param = test_vital_chunk_reading()
    if working_format:
        print(f"\nSUCCESS: Use {working_format} with parameter {working_param}")
    else:
        print("\nFAILED: No format provided VST chunk data")