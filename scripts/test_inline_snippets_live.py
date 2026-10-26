#!/usr/bin/env python
"""Direct validation of inline SFT snippets against a live REAPER session.

Runs all key snippet types within a single reapy session to avoid the
per-subprocess overhead that makes the grader's live-exec path too slow.
"""
import json
import sys
import time
import base64
import gc
from pathlib import Path

PASS = 0
FAIL = 0

def report(name: str, ok: bool, detail: str = ""):
    global PASS, FAIL
    if ok:
        PASS += 1
        print(f"  PASS  {name}" + (f" — {detail}" if detail else ""))
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f" — {detail}" if detail else ""))


def test_list_wavetables_total():
    lib = json.load(open("data/wavetable_lib.json"))
    seen, names = set(), []
    for wt in lib:
        n = wt.get("name", "")
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    result = {"total": len(names)}
    report("list_wavetables_total", result["total"] > 200, f"total={result['total']}")


def test_list_wavetables_slice():
    lib = json.load(open("data/wavetable_lib.json"))
    seen, names = set(), []
    for wt in lib:
        n = wt.get("name", "")
        if n and n not in seen:
            seen.add(n)
            names.append(n)
    start, end = 0, min(48, len(names))
    rows = [{"idx": i, "name": names[i]} for i in range(start, end)]
    result = {"wavetables": rows, "start": start, "end": end, "count": len(rows), "total": len(names)}
    report("list_wavetables_slice", result["count"] == 48, f"count={result['count']}")


def test_set_params(track, rpr, fx_idx):
    """Test generic reapy parameter-setting pattern."""
    params = {"Filter 1 Cutoff": 0.5938, "Oscillator 1 Level": 0.778}
    n = rpr.TrackFX_GetNumParams(track.id, fx_idx)
    if isinstance(n, (list, tuple)):
        n = n[0]
    name_to_idx = {}
    for i in range(int(n)):
        result = rpr.TrackFX_GetParamName(track.id, fx_idx, i, "", 256)
        name = result[4] if isinstance(result, (list, tuple)) else str(result)
        if name:
            name_to_idx[name] = i
    applied = 0
    not_found = []
    for name, value in params.items():
        if name in name_to_idx:
            rpr.TrackFX_SetParam(track.id, fx_idx, name_to_idx[name], value)
            applied += 1
        else:
            not_found.append(name)
    report("set_params", applied == 2 and len(not_found) == 0,
           f"applied={applied}, not_found={not_found}")


def test_chunk_read_write(track, rpr, fx_idx):
    """Test VST chunk read → decode → mutate → re-encode → write."""
    raw = ""
    chunk_parm = ""
    for parm in ("vst_chunk", "vst3_chunk"):
        result = rpr.TrackFX_GetNamedConfigParm(track.id, fx_idx, parm, "", 524288)
        raw = result[4] if isinstance(result, (list, tuple)) else str(result)
        if raw:
            chunk_parm = parm
            break

    ok = bool(raw)
    report("chunk_read", ok, f"parm={chunk_parm}, len={len(raw)}")
    if not ok:
        return

    data = base64.b64decode(raw)
    start = data.index(b"{")
    depth, i = 0, start
    while i < len(data):
        if data[i:i+1] == b"{": depth += 1
        elif data[i:i+1] == b"}": depth -= 1
        if depth == 0: break
        i += 1
    prefix, suffix = data[:start], data[i+1:]
    preset = json.loads(data[start:i+1])
    report("chunk_decode", "settings" in preset and "wavetables" in preset.get("settings", {}),
           f"keys={list(preset.keys())[:5]}")

    lib = json.load(open("data/wavetable_lib.json"))
    name_to_wt = {wt["name"]: wt for wt in lib if "name" in wt}
    test_wt = "Sine"
    if test_wt in name_to_wt:
        preset["settings"]["wavetables"][0] = name_to_wt[test_wt]
    new_json = json.dumps(preset, separators=(",", ":")).encode("utf-8")
    encoded = base64.b64encode(prefix + new_json + suffix).decode("ascii")
    gc.collect()
    rpr.TrackFX_SetNamedConfigParm(track.id, fx_idx, chunk_parm, encoded)

    # Read back and verify
    result2 = rpr.TrackFX_GetNamedConfigParm(track.id, fx_idx, chunk_parm, "", 524288)
    raw2 = result2[4] if isinstance(result2, (list, tuple)) else str(result2)
    data2 = base64.b64decode(raw2)
    start2 = data2.index(b"{")
    depth2, j = 0, start2
    while j < len(data2):
        if data2[j:j+1] == b"{": depth2 += 1
        elif data2[j:j+1] == b"}": depth2 -= 1
        if depth2 == 0: break
        j += 1
    preset2 = json.loads(data2[start2:j+1])
    wt_name_readback = preset2["settings"]["wavetables"][0].get("name", "")
    report("chunk_write_roundtrip", wt_name_readback == test_wt,
           f"expected={test_wt}, got={wt_name_readback}")


def test_track_creation(rpr, project):
    """Test track creation + Vital FX add."""
    rpr.InsertTrackAtIndex(0, False)
    import reapy
    track = reapy.Project().tracks[0]
    idx = rpr.TrackFX_AddByName(track.id, "Vital", False, -1)
    ok = isinstance(idx, int) and idx >= 0
    report("track_creation", ok, f"fx_idx={idx}")
    return track, idx if ok else 0


def main():
    import reapy

    print("Connecting to REAPER...")
    t0 = time.time()

    with reapy.inside_reaper():
        rpr = reapy.reascript_api
        project = reapy.Project()

        # Clean project
        for track in reversed(list(project.tracks)):
            try: track.delete()
            except: pass

        print(f"Connected in {time.time()-t0:.1f}s\n")

        # 1. List wavetables (no reapy needed)
        test_list_wavetables_total()
        test_list_wavetables_slice()

        # 2. Track creation + Vital load
        t1 = time.time()
        track, fx_idx = test_track_creation(rpr, project)
        print(f"  (Vital load: {time.time()-t1:.1f}s)")

        # 3. Set params
        t2 = time.time()
        test_set_params(track, rpr, fx_idx)
        print(f"  (param discovery+set: {time.time()-t2:.1f}s)")

        # 4. Chunk read/write/roundtrip
        t3 = time.time()
        test_chunk_read_write(track, rpr, fx_idx)
        print(f"  (chunk roundtrip: {time.time()-t3:.1f}s)")

    print(f"\n{'='*40}")
    print(f"Results: {PASS} passed, {FAIL} failed")
    print(f"Total time: {time.time()-t0:.1f}s")
    sys.exit(0 if FAIL == 0 else 1)


if __name__ == "__main__":
    main()
