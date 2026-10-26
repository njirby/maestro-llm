"""Tests to isolate the reapy render deadlock hypothesis.

Run with REAPER open and reapy server active:
    DISPLAY=:1 python scripts/test_reapy_deadlock.py

Each test has a 15-second timeout. A deadlock shows as a timeout.
"""
import json
import os
import signal
import sys
import tempfile
import time


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def run_test(name, fn, timeout_s=15):
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(timeout_s)
    try:
        fn()
        signal.alarm(0)
        print(f"  ✓ PASS")
        return True
    except Timeout:
        print(f"  ✗ DEADLOCK (timed out after {timeout_s}s)")
        return False
    except Exception as e:
        signal.alarm(0)
        print(f"  ✗ ERROR: {e}")
        return False


# ---------------------------------------------------------------
# Test 1: Basic reapy round-trip (instant action)
# ---------------------------------------------------------------
def test_basic_reapy():
    import reapy
    with reapy.inside_reaper():
        rpr = reapy.reascript_api
        rpr.Main_OnCommand(40340, 0)  # "unsolo all tracks" — instant
    print("  Basic reapy RPC round-trip works")


# ---------------------------------------------------------------
# Test 2: Short Lua via reapy bridge (no render)
# ---------------------------------------------------------------
def test_lua_no_render():
    import reapy
    result_path = "/tmp/test_reapy_lua_norender.json"
    script_path = "/tmp/test_reapy_lua_norender.lua"
    try:
        os.remove(result_path)
    except FileNotFoundError:
        pass

    lua = f'''
local f = io.open("{result_path}", "w")
f:write('{{"status":"ok","msg":"lua ran"}}')
f:close()
'''
    with open(script_path, "w") as f:
        f.write(lua)

    with reapy.inside_reaper():
        rpr = reapy.reascript_api
        cmd = rpr.AddRemoveReaScript(True, 0, script_path, True)
        if isinstance(cmd, (list, tuple)):
            cmd = cmd[0]
        try:
            rpr.Main_OnCommand(int(cmd), 0)
        finally:
            rpr.AddRemoveReaScript(False, 0, script_path, True)

    with open(result_path) as f:
        r = json.load(f)
    print(f"  Lua result: {r}")
    assert r["status"] == "ok"


# ---------------------------------------------------------------
# Test 3: Direct render via reapy — NO Lua involved
# If this deadlocks, reapy can't handle any blocking action.
# If this passes, the problem is nesting (Lua→render inside reapy).
# ---------------------------------------------------------------
def test_direct_render_via_reapy():
    import reapy
    with reapy.inside_reaper():
        rpr = reapy.reascript_api
        # Make sure there's something to render (bounds = time selection or project)
        rpr.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 1, True)
        rpr.GetSetProjectInfo_String(0, "RENDER_FILE", "/tmp", True)
        rpr.GetSetProjectInfo_String(0, "RENDER_PATTERN", "test_direct_render", True)
        print("  Calling Main_OnCommand(41824, 0) directly via reapy...")
        rpr.Main_OnCommand(41824, 0)
    print("  Direct render via reapy returned")
    rendered = "/tmp/test_direct_render.wav"
    if os.path.exists(rendered):
        print(f"  Rendered file exists: {os.path.getsize(rendered)} bytes")
        os.remove(rendered)
    else:
        print("  WARNING: no rendered file (project may be empty)")


# ---------------------------------------------------------------
# Test 4: Render INSIDE Lua via reapy — the deadlock repro
# This is what the current snippet generators do.
# ---------------------------------------------------------------
def test_render_inside_lua():
    import reapy
    result_path = "/tmp/test_reapy_lua_render.json"
    script_path = "/tmp/test_reapy_lua_render.lua"
    try:
        os.remove(result_path)
    except FileNotFoundError:
        pass

    lua = f'''
reaper.GetSetProjectInfo(0, "RENDER_BOUNDSFLAG", 1, true)
reaper.GetSetProjectInfo_String(0, "RENDER_FILE", "/tmp", true)
reaper.GetSetProjectInfo_String(0, "RENDER_PATTERN", "test_lua_render", true)
reaper.Main_OnCommand(41824, 0)
local f = io.open("{result_path}", "w")
f:write('{{"status":"ok","msg":"render done"}}')
f:close()
'''
    with open(script_path, "w") as f:
        f.write(lua)

    with reapy.inside_reaper():
        rpr = reapy.reascript_api
        cmd = rpr.AddRemoveReaScript(True, 0, script_path, True)
        if isinstance(cmd, (list, tuple)):
            cmd = cmd[0]
        try:
            print("  Calling Lua that contains Main_OnCommand(41824)...")
            rpr.Main_OnCommand(int(cmd), 0)
        finally:
            rpr.AddRemoveReaScript(False, 0, script_path, True)

    with open(result_path) as f:
        r = json.load(f)
    print(f"  Lua result: {r}")


# ---------------------------------------------------------------
# Test 5: Long-blocking Lua via reapy — no render
# Tests whether ANY main-thread blocking causes deadlock,
# or only Main_OnCommand(41824) specifically.
# ---------------------------------------------------------------
def test_long_blocking_lua():
    import reapy
    result_path = "/tmp/test_reapy_lua_block.json"
    script_path = "/tmp/test_reapy_lua_block.lua"
    try:
        os.remove(result_path)
    except FileNotFoundError:
        pass

    lua = f'''
local t = os.clock()
while os.clock() - t < 3 do end
local f = io.open("{result_path}", "w")
f:write('{{"status":"ok","blocked_s":3}}')
f:close()
'''
    with open(script_path, "w") as f:
        f.write(lua)

    with reapy.inside_reaper():
        rpr = reapy.reascript_api
        cmd = rpr.AddRemoveReaScript(True, 0, script_path, True)
        if isinstance(cmd, (list, tuple)):
            cmd = cmd[0]
        try:
            t0 = time.time()
            print("  Calling Lua that busy-waits 3 seconds...")
            rpr.Main_OnCommand(int(cmd), 0)
            elapsed = time.time() - t0
            print(f"  Returned after {elapsed:.1f}s")
        finally:
            rpr.AddRemoveReaScript(False, 0, script_path, True)

    with open(result_path) as f:
        r = json.load(f)
    print(f"  Lua result: {r}")


if __name__ == "__main__":
    results = {}
    results["1_basic_reapy"] = run_test("Basic reapy round-trip", test_basic_reapy)
    results["2_lua_no_render"] = run_test("Short Lua via reapy (no render)", test_lua_no_render)
    results["3_direct_render"] = run_test("Direct render via reapy (no Lua)", test_direct_render_via_reapy)
    results["4_render_in_lua"] = run_test("Render INSIDE Lua via reapy (deadlock repro)", test_render_inside_lua)
    results["5_long_block_lua"] = run_test("Long-blocking Lua via reapy (no render)", test_long_blocking_lua)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL/DEADLOCK"
        print(f"  {name}: {status}")

    print()
    if results["3_direct_render"] and not results["4_render_in_lua"]:
        print("CONCLUSION: Deadlock is caused by nesting render inside Lua via reapy.")
        print("FIX: Split render out of Lua — call rpr.Main_OnCommand(41824) from Python.")
    elif not results["3_direct_render"] and not results["4_render_in_lua"]:
        print("CONCLUSION: reapy can't handle ANY long-blocking action.")
        print("FIX: File-watching dispatcher — eliminate reapy entirely.")
    elif results["3_direct_render"] and results["4_render_in_lua"]:
        print("CONCLUSION: No deadlock observed! The hypothesis may be wrong,")
        print("or the project needs content (tracks/items) to trigger the render path.")
    else:
        print("CONCLUSION: Unexpected pattern — investigate further.")
