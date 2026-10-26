#!/usr/bin/env python3
"""Render every line of <dir>/renders.jsonl ({"name", "preset", "notes", "bpm"}) to <dir>/wav/<name>.wav through the
Vital VST3 in DawDreamer with full preset state. A render-noise floor is just a second line with the same preset (random
phase, free-running LFOs). Older dirs with only jobs.jsonl render A, A2 (floor) and B per job. Runs in the daw-farm image
(dawdreamer + Vital + Xvfb); self-contained because that image has no maestro install. Host tempo is set from the clip's
MIDI BPM (Vital takes tempo from the host).

  maestro/sounddiff/render_in_container.sh outputs/sounddiff/probe0
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from renderer_bakeoff_dawdreamer import build_state, install_x11_error_handler  # noqa: E402

VITAL = os.environ.get("VITAL_VST3", "/usr/lib/vst3/Vital.vst3")
SR, BLOCK, TAIL_S = 44100, 512, 2.0
# Vital keeps FX buffers and voices across load_state: without a pre-roll the previous job's tail bleeds into the next
# render. Notes are shifted by PREROLL_S of silence that is cut from the output.
PREROLL_S = 4.0


def main():
    out = Path(sys.argv[1])
    shard, nshards = int(sys.argv[2]), int(sys.argv[3])
    if (out / "renders.jsonl").exists():
        rows = [json.loads(l) for l in open(out / "renders.jsonl")]
    else:
        rows = [{"name": f"{jb['id']}_{tag}", "preset": f"presets/{jb['id']}_{p}.vital", "notes": jb["notes"],
                 "bpm": jb["bpm"]} for jb in map(json.loads, open(out / "jobs.jsonl"))
                for tag, p in (("a", "a"), ("a2", "a"), ("b", "b"))]
    rows = rows[shard::nshards]
    install_x11_error_handler()
    import dawdreamer as daw
    import numpy as np
    import soundfile as sf
    eng = daw.RenderEngine(SR, BLOCK)
    synth = eng.make_plugin_processor("v", VITAL)
    eng.load_graph([(synth, [])])
    state = f"/tmp/sd_state_{os.getpid()}"
    (out / "wav").mkdir(exist_ok=True)
    for r in rows:
        path = out / "wav" / f"{r['name']}.wav"
        if path.exists():
            continue
        dur = max(n["start_s"] + n["dur_s"] for n in r["notes"]) + TAIL_S
        t0 = time.time()
        with open(state, "wb") as f:
            f.write(build_state(json.load(open(out / r["preset"]))))
        synth.load_state(state)
        eng.set_bpm(float(r["bpm"]))
        synth.clear_midi()
        for n in r["notes"]:
            if float(n["dur_s"]) <= 0:  # Lakh has zero-length notes; DawDreamer rejects them
                continue
            synth.add_midi_note(int(n["pitch"]), int(n["velocity"]), PREROLL_S + float(n["start_s"]), float(n["dur_s"]))
        eng.render(PREROLL_S + dur)
        audio = np.asarray(synth.get_audio()).T[int(PREROLL_S * SR):]
        sf.write(path, audio, SR, subtype="FLOAT")
        print(json.dumps({"name": r["name"], "s": round(time.time() - t0, 3),
                          "peak": round(float(np.abs(audio).max()), 4)}), flush=True)


if __name__ == "__main__":
    main()
