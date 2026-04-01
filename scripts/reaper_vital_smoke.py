"""
Diagnostic script for verifying VitalController in a live REAPER session.

Run via a Lua launcher:
  local cmd = reaper.AddRemoveReaScript(true, 0, "/path/to/reaper_vital_smoke.py", true)
  reaper.Main_OnCommand(cmd, 0)

Prerequisites:
  - REAPER is running with a project that has Vital on at least one track
  - vital_tools.py is deployed to ~/.config/REAPER/Scripts/

Results are written to /tmp/vital_smoke_results.json
"""

from reaper_python import *
import sys
import os
import json
import gc
import time

_LOG = "/tmp/vital_smoke_trace.log"
def _log(msg):
    with open(_LOG, "a") as _f:
        _f.write(str(msg) + "\n")

_log("=== smoke test start ===")

sys.path.insert(0, os.path.expanduser("~/.config/REAPER/Scripts"))
from vital_tools import VitalController
_log("VitalController imported")

results = {}
v = VitalController()
_log("VitalController created")

# 1. discover() ---------------------------------------------------------------
_log("test 1: discover")
try:
    info = v.discover()
    results["discover"] = {"ok": True, "data": info}
    _log(f"discover ok: {info}")
except Exception as e:
    results["discover"] = {"ok": False, "error": str(e)}
    _log(f"discover FAILED: {e}")
gc.collect()

# 2. get_params() — reads from preset JSON now --------------------------------
_log("test 2: get_params")
try:
    params = v.get_params()
    sample = dict(list(params.items())[:5])
    results["get_params"] = {
        "ok": True,
        "count": len(params),
        "sample": sample,
        "note": "native Vital values (not normalized)",
    }
    _log(f"get_params ok: {len(params)} params")
except Exception as e:
    results["get_params"] = {"ok": False, "error": str(e)}
    params = {}
    _log(f"get_params FAILED: {e}")
gc.collect()

# 3. get_preset() — check vst_chunk vs vst3_chunk ----------------------------
_log("test 3: get_preset")
try:
    preset = v.get_preset()
    settings = preset.get("settings", {})
    results["get_preset"] = {
        "ok": True,
        "chunk_parm": getattr(v, "_chunk_parm_name", "unknown"),
        "top_keys": list(preset.keys()),
        "settings_key_count": len(settings),
        "has_modulations": "modulations" in settings,
        "sample_params": {k: settings[k] for k in list(settings.keys())[:3]
                          if isinstance(settings[k], (int, float))},
    }
    _log(f"get_preset ok: chunk_parm={v._chunk_parm_name}")
    del preset, settings  # release references — don't hold across later writes
except Exception as e:
    results["get_preset"] = {"ok": False, "error": str(e)}
    _log(f"get_preset FAILED: {e}")
gc.collect()

# 4. get_modulations() --------------------------------------------------------
_log("test 4: get_modulations")
try:
    mods = v.get_modulations()
    results["get_modulations"] = {
        "ok": True,
        "count": len(mods),
        "sample": mods[:3],
    }
    _log(f"get_modulations ok: {len(mods)} mods")
except Exception as e:
    results["get_modulations"] = {"ok": False, "error": str(e)}
    _log(f"get_modulations FAILED: {e}")
gc.collect()

# 5. set_param() — save, set, restore ----------------------------------------
# NOTE: time.sleep() must be called at the TOP LEVEL of this script (not inside
# vital_tools methods) to yield REAPER's event loop so Vital can process each
# async SetNamedConfigParm write before the next read or write.
_log("test 5: set_param")
try:
    if params:
        name = next(iter(params))
        orig = params[name]
        test_val = orig + 0.1
        ok = v.set_param(name, test_val)          # write #1
        gc.collect()
        time.sleep(0.1)
        v.set_param(name, orig)                    # write #2 (restore)
        gc.collect()
        time.sleep(0.1)
        results["set_param"] = {
            "ok": ok,
            "param": name,
            "orig": orig,
            "test_val": test_val,
            "restored": True,
        }
        _log(f"set_param ok: {name}={test_val} (restored to {orig})")
    else:
        results["set_param"] = {"ok": False, "error": "no params found"}
        _log("set_param skipped: no params")
except Exception as e:
    results["set_param"] = {"ok": False, "error": str(e)}
    _log(f"set_param FAILED: {e}")
gc.collect()

# 6. set_params() — batch update then restore ---------------------------------
_log("test 6: set_params")
try:
    if len(params) >= 2:
        names = list(params.keys())[:2]
        orig_vals = {n: params[n] for n in names}
        test_vals = {n: orig_vals[n] + 0.05 for n in names}
        result_info = v.set_params(test_vals)      # write #3
        gc.collect()
        time.sleep(0.1)
        v.set_params(orig_vals)                    # write #4 (restore)
        gc.collect()
        time.sleep(0.1)
        results["set_params"] = {
            "ok": result_info["applied"] == 2,
            "applied": result_info["applied"],
            "not_found": result_info["not_found"],
        }
        _log(f"set_params ok: applied={result_info['applied']}")
    else:
        results["set_params"] = {"ok": False, "error": "not enough params"}
        _log("set_params skipped: not enough params")
except Exception as e:
    results["set_params"] = {"ok": False, "error": str(e)}
    _log(f"set_params FAILED: {e}")
gc.collect()

# 7. set_modulation() — inline for debugging: add temp route, then remove ----
_log("test 7: set_modulation (inline)")
try:
    if results.get("get_preset", {}).get("ok"):
        # Read current preset, add test modulation, write back, verify, restore
        _log("7: get_preset")
        preset7 = v.get_preset()
        _log("7: got preset")

        settings7 = preset7.get("settings", preset7)
        mods7 = settings7.get("modulations", [])
        # Use a valid Vital destination (not a fake string — Vital ignores invalid dests)
        _TEST_SRC = "lfo_1"
        _TEST_DST = "osc_1_level"
        for entry in mods7:
            if not entry.get("source") and not entry.get("destination"):
                entry["source"] = _TEST_SRC
                entry["destination"] = _TEST_DST
                entry["amount"] = 0.1
                break
        _log("7: writing preset with modulation")
        v.set_preset(preset7)
        _log("7: write returned")
        gc.collect()
        time.sleep(0.2)

        mods_after = v.get_modulations()
        gc.collect()
        found = any(m["destination"] == _TEST_DST and m["source"] == _TEST_SRC
                    for m in mods_after)
        _log(f"7: route_appeared={found} (checked {len(mods_after)} mods)")

        # Restore: clear the modulation slot
        preset7r = v.get_preset()
        settings7r = preset7r.get("settings", preset7r)
        for entry in settings7r.get("modulations", []):
            if entry.get("destination") == _TEST_DST:
                entry["source"] = ""
                entry["destination"] = ""
                entry["amount"] = 0.0
                break
        v.set_preset(preset7r)
        _log("7: restore done")

        results["set_modulation"] = {"ok": found, "route_appeared": found}
        _log(f"set_modulation ok: route_appeared={found}")
    else:
        results["set_modulation"] = {"ok": False, "error": "skipped (get_preset failed)"}
        _log("set_modulation skipped")
except Exception as e:
    results["set_modulation"] = {"ok": False, "error": str(e)}
    _log(f"set_modulation FAILED: {e}")
gc.collect()

# 8. render_to() — check file appears (Main_OnCommand(42230) is synchronous in REAPER)
_log("test 8: render_to")
try:
    render_path = "/tmp/vital_smoke_render.wav"
    if os.path.exists(render_path):
        os.remove(render_path)
    v.render_to(render_path)
    gc.collect()
    # 42230 completes synchronously — check immediately
    appeared = os.path.exists(render_path) and os.path.getsize(render_path) > 0
    results["render_to"] = {
        "ok": appeared,
        "file_appeared": appeared,
        "size": os.path.getsize(render_path) if appeared else 0,
    }
    _log(f"render_to ok: file_appeared={appeared}")
except Exception as e:
    results["render_to"] = {"ok": False, "error": str(e)}
    _log(f"render_to FAILED: {e}")

# Write results ---------------------------------------------------------------
_log("writing results")
all_ok = all(r.get("ok") for r in results.values())
v.write_result(
    "/tmp/vital_smoke_results.json",
    ok=all_ok,
    data=results,
)
_log(f"=== done, all_ok={all_ok} ===")
