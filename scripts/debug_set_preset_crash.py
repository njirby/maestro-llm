#!/usr/bin/env python3
"""Diagnostic: find the failure mode of VitalController.set_preset() crashing REAPER.

Runs a single payload size variant per invocation. The outer driver
(debug_set_preset_crash.sh or manual) is expected to restart REAPER between runs
so each variant starts from a clean process.

Cases:
  baseline    — set_preset(preset) with no modification.
  tiny_tweak  — set_preset(preset) with one scalar param tweaked.
  tiny_wt     — wavetables[0] replaced with a synthetic ~1 KB entry.
  small_wt    — wavetables[0] replaced with a real entry at ~50 KB raw JSON.
  medium_wt   — wavetables[0] replaced with a real entry at ~200 KB raw JSON.
  huge_wt     — wavetables[0] replaced with the largest real entry (~914 KB raw).

Each run emits two JSON lines to stdout:
  {"case": ..., "size_before_b64": N, "size_after_b64": N, "wt_name": ...}
  {"case": ..., "result": "PASS"}  OR crash-before-line-two = FAIL.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import reapy  # noqa: E402

try:
    from maestro.reaper.vital_tools import VitalController  # noqa: E402
except Exception:
    sys.path.append("/home/nate/.config/REAPER/Scripts")
    from vital_tools import VitalController  # noqa: E402


def _load_wt_lib() -> list[dict]:
    wt_lib = json.load(open(ROOT / "data/wavetable_lib.json"))
    return [w for w in wt_lib if isinstance(w, dict) and "name" in w]


def _synth_tiny_wt() -> dict:
    return {
        "name": "debug_tiny",
        "groups": [
            {
                "components": [
                    {
                        "type": "wave_source",
                        "keyframes": [
                            {"name": "f0", "wave_data": [0.0] * 2}
                        ],
                    }
                ]
            }
        ],
        "version": "1.5.5",
    }


def _pick_by_size(wts: list[dict], target_raw_bytes: int) -> dict:
    """Return the wavetable whose JSON-serialized raw size is closest to target."""
    with_sizes = [(w, len(json.dumps(w))) for w in wts]
    with_sizes.sort(key=lambda t: abs(t[1] - target_raw_bytes))
    return with_sizes[0][0]


def _ensure_vital_loaded(rpr: dict) -> None:
    import time
    proj = 0
    if int(rpr["RPR_CountTracks"](proj)) == 0:
        rpr["RPR_InsertTrackAtIndex"](0, True)
    track = rpr["RPR_GetTrack"](proj, 0)
    fx_count_fn = rpr.get("RPR_TrackFX_GetCount")
    if fx_count_fn is not None and int(fx_count_fn(track)) > 0:
        return
    for name in (
        "Vital",
        "VST3i: Vital",
        "VST3: Vital (Vital Audio)",
        "VSTi: Vital",
    ):
        idx = rpr["RPR_TrackFX_AddByName"](track, name, False, 1)
        if isinstance(idx, (tuple, list)):
            idx = idx[0] if idx else -1
        if int(idx) >= 0:
            # Give Vital time to initialize its VST chunk.
            time.sleep(1.5)
            return
    raise RuntimeError("Could not load Vital on track 0")


def _discover_with_retry(vc, max_wait_s: float = 10.0) -> None:
    """Retry vc.discover() + vc.get_preset() until the chunk is ready."""
    import time
    start = time.monotonic()
    last_exc: Exception | None = None
    while time.monotonic() - start < max_wait_s:
        try:
            vc.discover()
            vc.get_preset()
            return
        except Exception as e:
            last_exc = e
            time.sleep(0.5)
    raise RuntimeError(f"Vital chunk never became available: {last_exc}")


def run_case(case: str) -> int:
    wts = _load_wt_lib()
    case_payload: dict | str | None
    if case == "baseline":
        case_payload = None
    elif case == "tiny_tweak":
        case_payload = "TWEAK"
    elif case == "tiny_wt":
        case_payload = _synth_tiny_wt()
    elif case == "small_wt":
        case_payload = _pick_by_size(wts, 50_000)
    elif case == "medium_wt":
        case_payload = _pick_by_size(wts, 200_000)
    elif case == "huge_wt":
        case_payload = max(wts, key=lambda w: len(json.dumps(w)))
    else:
        raise SystemExit(f"unknown case {case!r}")

    with reapy.inside_reaper():
        api = reapy.reascript_api
        rpr = {f"RPR_{fn}": getattr(api, fn) for fn in dir(api) if not fn.startswith("_")}
        _ensure_vital_loaded(rpr)
        vc = VitalController(_rpr=rpr)
        _discover_with_retry(vc)
        preset = vc.get_preset()

        size_before = len(
            vc._encode_chunk(vc._chunk_prefix, preset, vc._chunk_suffix)
        )

        if case_payload is None:
            pass
        elif case_payload == "TWEAK":
            preset.setdefault("settings", {})["volume"] = 0.5
        elif isinstance(case_payload, dict):
            preset.setdefault("settings", {}).setdefault("wavetables", [None, None, None])
            preset["settings"]["wavetables"][0] = case_payload

        encoded = vc._encode_chunk(vc._chunk_prefix, preset, vc._chunk_suffix)
        size_after = len(encoded)

        wt_name = (
            case_payload.get("name")
            if isinstance(case_payload, dict)
            else (case_payload if isinstance(case_payload, str) else None)
        )
        print(
            json.dumps(
                {
                    "case": case,
                    "size_before_b64": size_before,
                    "size_after_b64": size_after,
                    "wt_name": wt_name,
                }
            ),
            flush=True,
        )

        vc.set_preset(preset)

    print(json.dumps({"case": case, "result": "PASS"}), flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "case",
        choices=["baseline", "tiny_tweak", "tiny_wt", "small_wt", "medium_wt", "huge_wt"],
    )
    args = ap.parse_args()
    try:
        sys.exit(run_case(args.case))
    except Exception as e:
        print(
            json.dumps(
                {
                    "case": args.case,
                    "result": "EXCEPTION",
                    "error": type(e).__name__,
                    "msg": str(e)[:300],
                }
            ),
            flush=True,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
