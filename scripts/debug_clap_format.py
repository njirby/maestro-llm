#!/usr/bin/env python3
"""Test if CLAP format Vital handles large/real wavetables without crashing."""
import sys, json, time
sys.path.insert(0, '/home/nate/Documents/maestro-llm')
import reapy

def _clear_fx(rpr, track):
    for i in range(int(rpr["RPR_TrackFX_GetCount"](track)) - 1, -1, -1):
        rpr["RPR_TrackFX_Delete"](track, i)

def test_clap():
    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f'RPR_{fn}': getattr(api, fn) for fn in dir(api) if not fn.startswith('_')}
        if int(rpr["RPR_CountTracks"](0)) == 0:
            rpr["RPR_InsertTrackAtIndex"](0, True)
        track = rpr["RPR_GetTrack"](0, 0)
        _clear_fx(rpr, track)

        # Try each CLAP prefix format REAPER uses
        clap_names = ["CLAP: Vital", "CLAPi: Vital", "CLAP:Vital Audio Vital"]
        loaded_idx = -1
        for name in clap_names:
            idx = rpr["RPR_TrackFX_AddByName"](track, name, False, 1)
            if isinstance(idx, (tuple, list)):
                idx = idx[0] if idx else -1
            print(f"Try '{name}' -> idx={idx}")
            if int(idx) >= 0:
                loaded_idx = int(idx)
                time.sleep(2)
                name_res = rpr["RPR_TrackFX_GetFXName"](track, loaded_idx, "", 256)
                print(f"  Loaded as: {name_res[3] if isinstance(name_res, (tuple, list)) and len(name_res) >= 4 else name_res}")
                break

        if loaded_idx < 0:
            print("Could not load CLAP Vital - scanning plugin list...")
            return

        # Try to find the state parameter name for CLAP
        for pname in ["clap_state", "vst_chunk", "vst3_chunk", "chunk", "state"]:
            try:
                r = rpr["RPR_TrackFX_GetNamedConfigParm"](track, loaded_idx, pname, "", 2*1024*1024)
                if isinstance(r, (tuple, list)):
                    sz = len(r[2]) if len(r) > 2 and r[2] else 0
                    ok = r[0] if len(r) > 0 else False
                    print(f"  {pname}: ok={ok}, bytes={sz}")
            except Exception as e:
                print(f"  {pname}: exc={e}")

if __name__ == "__main__":
    try:
        test_clap()
    except Exception as e:
        print(f"FATAL: {e}")