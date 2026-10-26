#!/usr/bin/env python3
"""Test: load LV2 Vital and see if it can handle large presets without crashing."""

import sys, json, os, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy
from maestro.reaper.vital_tools import VitalController

def _load_lv2_vital(rpr: dict) -> None:
    proj = 0
    if int(rpr["RPR_CountTracks"](proj)) == 0:
        rpr["RPR_InsertTrackAtIndex"](0, True)
    track = rpr["RPR_GetTrack"](proj, 0)
    fx_count_fn = rpr.get("RPR_TrackFX_GetCount")
    if fx_count_fn is not None and int(fx_count_fn(track)) > 0:
        return

    # Try LV2 first, then fallback to VST formats
    for name in ("LV2: Vital", "Vital", "VST3i: Vital", "VSTi: Vital"):
        print(f"Trying to load: {name}")
        idx = rpr["RPR_TrackFX_AddByName"](track, name, False, 1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1
        if int(idx) >= 0:
            print(f"Successfully loaded {name} at fx index {idx}")
            time.sleep(2.0)  # Give it time to initialize
            return
    raise RuntimeError("Could not load any Vital format")

def test_lv2_vital_preset():
    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}

        _load_lv2_vital(rpr)

        # Check which format was actually loaded
        track = rpr["RPR_GetTrack"](0, 0)
        fx_name = rpr["RPR_TrackFX_GetFXName"](track, 0, "", 256)
        if isinstance(fx_name, (tuple, list)) and len(fx_name) > 1:
            fx_name = fx_name[1]
        print(f"Loaded FX: {fx_name}")

        # Try to initialize VitalController
        vc = VitalController(_rpr=rpr)
        print("VitalController created")

        try:
            vc.discover()
            print("VitalController.discover() successful")
            preset = vc.get_preset()
            print(f"Got preset, keys: {list(preset.keys()) if isinstance(preset, dict) else type(preset)}")

            # Test applying a large wavetable
            wt_lib = json.load(open('data/wavetable_lib.json'))
            target = next(w for w in wt_lib if w.get('name') == 'hola')
            print(f"Found hola wavetable, size: {len(json.dumps(target))} bytes")

            preset['settings']['wavetables'][0] = target
            print("Applied hola wavetable to preset")

            # This is the moment of truth - does LV2 crash like VST3?
            print("Calling set_preset...")
            vc.set_preset(preset)
            print("SUCCESS: set_preset completed without crash!")

        except Exception as e:
            print(f"ERROR in VitalController operation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    try:
        test_lv2_vital_preset()
    except Exception as e:
        print(f"FATAL ERROR: {e}")
        import traceback
        traceback.print_exc()