#!/usr/bin/env python3
"""Test: load a native .vital preset file via set_preset, not via lib swap."""
import sys, json, time, os, glob
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy
from maestro.reaper.vital_tools import VitalController


def find_preset_with_wt():
    """Find a .vital file that has wavetables embedded."""
    candidates = glob.glob(os.path.expanduser("~/Downloads/vital_presets/*.vital"))
    for path in candidates[:50]:
        try:
            d = json.load(open(path))
            wts = d.get("settings", {}).get("wavetables", [])
            if wts and any(w.get("groups") for w in wts if isinstance(w, dict)):
                raw_size = len(json.dumps(d))
                if raw_size > 50_000:  # non-trivial
                    return path, d, raw_size
        except Exception:
            continue
    return None, None, 0


def test_native_preset():
    path, preset, raw_size = find_preset_with_wt()
    if not preset:
        print("No suitable preset found")
        return
    print(f"Using: {os.path.basename(path)} (raw JSON: {raw_size:,} bytes)")
    print(f"  synth_version: {preset.get('synth_version')}")
    print(f"  preset_name: {preset.get('preset_name')}")
    print(f"  num wavetables: {len(preset.get('settings', {}).get('wavetables', []))}")

    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}
        if int(rpr["RPR_CountTracks"](0)) == 0:
            rpr["RPR_InsertTrackAtIndex"](0, True)
        track = rpr["RPR_GetTrack"](0, 0)
        for i in range(int(rpr["RPR_TrackFX_GetCount"](track)) - 1, -1, -1):
            rpr["RPR_TrackFX_Delete"](track, i)
        idx = rpr["RPR_TrackFX_AddByName"](track, "Vital", False, 1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1
        print(f"Loaded Vital at idx={idx}")
        time.sleep(2)

        vc = VitalController(_rpr=rpr)
        vc.discover()
        vc.get_preset()  # cache prefix/suffix

        encoded = vc._encode_chunk(vc._chunk_prefix, preset, vc._chunk_suffix)
        print(f"Encoded size: {len(encoded):,} bytes")

        try:
            print("Calling set_preset...")
            vc.set_preset(preset)
            print("SUCCESS - native preset loaded without crash")
        except Exception as e:
            print(f"FAIL: {type(e).__name__}: {e}")


if __name__ == "__main__":
    try:
        test_native_preset()
    except Exception as e:
        print(f"FATAL: {type(e).__name__}: {e}")