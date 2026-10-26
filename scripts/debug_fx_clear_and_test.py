#!/usr/bin/env python3
"""Debug: clear track and test loading different Vital formats."""

import sys, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy

def test_vital_formats():
    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}

        proj = 0
        # Create fresh track
        if int(rpr["RPR_CountTracks"](proj)) == 0:
            rpr["RPR_InsertTrackAtIndex"](0, True)
        track = rpr["RPR_GetTrack"](proj, 0)

        # Clear existing FX
        fx_count = int(rpr["RPR_TrackFX_GetCount"](track))
        print(f"Initial FX count: {fx_count}")
        for i in range(fx_count-1, -1, -1):  # Remove in reverse order
            rpr["RPR_TrackFX_Delete"](track, i)
        print("Cleared all FX")

        # Test different Vital formats
        formats_to_test = [
            "VSTi: Vital",         # VST2
            "VST: Vital",          # VST2
            "VST3i: Vital",        # VST3
            "VST3: Vital (Vital Audio)",  # VST3 full name
            "LV2: Vital",          # LV2
        ]

        for format_name in formats_to_test:
            print(f"\nTesting format: {format_name}")
            idx = rpr["RPR_TrackFX_AddByName"](track, format_name, False, 1)
            if isinstance(idx, (tuple, list)):
                idx = idx[0] if idx else -1

            if int(idx) >= 0:
                print(f"  Successfully loaded at index {idx}")
                time.sleep(2)  # Let it initialize

                # Try different ways to get the FX name
                try:
                    # Method 1: TrackFX_GetFXName
                    name_buf_size = 256
                    name_result = rpr["RPR_TrackFX_GetFXName"](track, idx, "", name_buf_size)
                    if isinstance(name_result, (tuple, list)) and len(name_result) >= 2:
                        actual_name = name_result[1]
                        print(f"  FX name: '{actual_name}'")
                    else:
                        print(f"  FX name failed: {name_result}")

                    # Method 2: Check if it has VST chunk capability
                    for param_name in ["vst_chunk", "vst3_chunk"]:
                        try:
                            result = rpr["RPR_TrackFX_GetNamedConfigParm"](track, idx, param_name, "", 1024)
                            if isinstance(result, (tuple, list)) and len(result) > 2:
                                value = result[2]
                                if value and len(value) > 10:  # Some meaningful data
                                    print(f"  {param_name}: SUCCESS ({len(value)} bytes)")
                                    return format_name, param_name  # Found working format!
                                else:
                                    print(f"  {param_name}: empty/small")
                        except Exception as e:
                            print(f"  {param_name}: exception - {e}")

                    # Clean up this FX before trying next
                    rpr["RPR_TrackFX_Delete"](track, idx)
                    time.sleep(0.5)

                except Exception as e:
                    print(f"  Error testing format: {e}")
                    # Clean up this FX
                    try:
                        rpr["RPR_TrackFX_Delete"](track, idx)
                    except:
                        pass
            else:
                print(f"  Failed to load (returned {idx})")

        print("\nNo working VST format found!")
        return None, None

if __name__ == "__main__":
    working_format, working_param = test_vital_formats()
    if working_format:
        print(f"\nWorking combination: {working_format} with {working_param}")
    else:
        print("\nNo working VST format found - all failed to provide chunk data")