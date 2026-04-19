"""ReaScript: read a VST chunk from disk and write it into a track's FX.

Invoked from outside REAPER via ``AddRemoveReaScript`` + ``Main_OnCommand``.
This exists because REAPER's ReaScript Python bridge crashes when a string
argument to ``TrackFX_SetNamedConfigParm`` exceeds ~256 KiB (the size Vital's
encoded VST chunk can easily exceed once a real wavetable is swapped in).

Invocation protocol:
  1. Outside caller writes the encoded base64 chunk to
     ``/tmp/vital_chunk_payload.b64``.
  2. Outside caller writes ``/tmp/vital_chunk_req.json`` with the fields:
       {
         "chunk_path": "/tmp/vital_chunk_payload.b64",
         "track_idx": int,
         "fx_idx": int,
         "parm_name": "vst_chunk" | "vst3_chunk"
       }
  3. Outside caller registers this script via ``AddRemoveReaScript(True,...)``
     and triggers it via ``Main_OnCommand(cmd_id, 0)``.
  4. This script reads both files, calls ``TrackFX_SetNamedConfigParm`` using
     REAPER's in-process Python binding (no RPC), and writes a status JSON to
     ``/tmp/vital_chunk_done.json``.
"""
import json
import os

try:
    import reaper  # type: ignore[import-not-found]  # noqa: F401
    _RPR_SET = reaper.TrackFX_SetNamedConfigParm  # type: ignore[attr-defined]
    _RPR_GET_TRACK = reaper.GetTrack  # type: ignore[attr-defined]
except Exception:
    from reaper_python import RPR_TrackFX_SetNamedConfigParm as _RPR_SET  # type: ignore[import-not-found]
    from reaper_python import RPR_GetTrack as _RPR_GET_TRACK  # type: ignore[import-not-found]


REQ_PATH = "/tmp/vital_chunk_req.json"
DONE_PATH = "/tmp/vital_chunk_done.json"


def _main() -> None:
    with open(REQ_PATH) as f:
        req = json.load(f)
    chunk_path = req["chunk_path"]
    track_idx = int(req["track_idx"])
    fx_idx = int(req["fx_idx"])
    parm_name = req.get("parm_name", "vst_chunk")

    with open(chunk_path) as f:
        encoded = f.read()

    track = _RPR_GET_TRACK(0, track_idx)
    _RPR_SET(track, fx_idx, parm_name, encoded)

    with open(DONE_PATH, "w") as f:
        json.dump({"status": "ok", "bytes": len(encoded)}, f)

    try:
        os.remove(chunk_path)
    except Exception:
        pass


try:
    _main()
except Exception as exc:
    try:
        with open(DONE_PATH, "w") as f:
            json.dump(
                {"status": "error", "error": type(exc).__name__, "msg": str(exc)[:400]},
                f,
            )
    except Exception:
        pass
