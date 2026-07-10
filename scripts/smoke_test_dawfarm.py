#!/usr/bin/env python
"""Smoke test for the daw-farm real-rollout path.

Drives one daw-farm REAPER session through the same snippet builders that
build_main_agent_sft_v3 --daw-farm executes for real: project reset, track +
Vital creation, MIDI insertion, wavetable count, param search, chunk-based
param apply, a REAPER timeline render (fetched + checked non-silent), and a
DawDreamer in-container render.

Usage:
    python scripts/smoke_test_dawfarm.py [--spec docker] [--session NAME]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from maestro.reaper import dawfarm
from maestro.render.dawdreamer import make_probe_notes
from scripts.agent_sft_common import (
    _wrap_as_bash,
    build_list_wavetables_total_snippet,
    build_param_search_snippet,
    build_reaper_render_snippet,
    build_render_verify_snippet,
)

SAMPLE_ID = "dawfarm_smoke"


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docker")
    ap.add_argument("--session", default="", help="Explicit container/pod name.")
    ap.add_argument("--vital-data", default=str(Path.home() / ".local/share/vital"))
    args = ap.parse_args()

    spec = f"{args.spec.split(':')[0]}:{args.session}" if args.session else args.spec
    pool = dawfarm.DawFarmPool.from_spec(spec)
    failures = 0

    with pool.acquire() as s:
        print(f"session: {s.name}")
        rd = f"{dawfarm.ROLLOUT_ROOT}/{SAMPLE_ID}"

        # 1. infra setup (what build_record does before the conversation)
        dawfarm.reset_project(s)
        dawfarm.sync_vital_data(s, args.vital_data)
        dawfarm.prepare_sample_dirs(s, SAMPLE_ID)
        dawfarm.create_vital_track(s)
        notes = [
            {"pitch": p, "velocity": v, "start_s": st, "dur_s": d}
            for (p, v, st, d) in make_probe_notes("lead", clip_duration_s=10.0)
        ]
        dawfarm.insert_midi_notes(s, notes)
        notes_file = f"/tmp/agents/{SAMPLE_ID}/notes.json"
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tf:
            json.dump({"notes": notes}, tf)
        s.put(tf.name, notes_file)
        failures += not check("setup (reset/track/vital/midi)", True)

        # 2. wavetable library scan (real VITAL data dirs in the container)
        res = s.exec_bash(_wrap_as_bash(build_list_wavetables_total_snippet()))
        total = json.loads(res.stdout or "{}").get("total", 0) if res.ok else 0
        failures += not check("wavetable count", res.ok and total > 0, f"total={total}")

        # 3. param search via live reapy TrackFX API
        res = s.exec_bash(_wrap_as_bash(build_param_search_snippet("filter cutoff")))
        n_params = json.loads(res.stdout or "{}").get("count", 0) if res.ok else 0
        failures += not check("param search 'filter cutoff'", res.ok and n_params > 0,
                              f"count={n_params}, stderr={res.stderr.strip()[:200]}")

        # 4. chunk-based param apply (same path as build_batch_action_snippet)
        from scripts.build_main_agent_sft_v3 import build_batch_action_snippet
        res = s.exec_bash(build_batch_action_snippet({"filter_1_on": 1.0, "filter_1_cutoff": 60.0}))
        failures += not check("batch param apply", res.ok and '"status": "ok"' in res.stdout,
                              res.stdout.strip()[:120] or res.stderr.strip()[:200])

        # 5. REAPER timeline render → fetch → non-silence
        container_wav = f"{rd}/smoke_render.wav"
        res = s.exec_bash(_wrap_as_bash(build_reaper_render_snippet(out_path=container_wav)))
        probe = json.loads(res.stdout or "{}").get("listen_probe", {}) if res.ok else {}
        ok = res.ok and probe.get("exists") and s.wait_for_file(container_wav, timeout=30)
        host_wav = None
        peak = 0.0
        if ok:
            host_wav = Path(tempfile.gettempdir()) / "dawfarm_smoke_render.wav"
            s.get(container_wav, host_wav)
            audio, sr = sf.read(host_wav)
            peak = float(np.abs(audio).max())
            ok = peak > 1e-4 and len(audio) > sr
        failures += not check("REAPER render non-silent", bool(ok),
                              f"peak={peak:.4f}" if host_wav else
                              f"stdout={res.stdout.strip()[:200]} stderr={res.stderr.strip()[:200]}")

        # 6. DawDreamer in-container render (reads preset back from REAPER)
        dd_wav = f"{rd}/smoke_dawdreamer.wav"
        res = s.exec_bash(
            _wrap_as_bash(build_render_verify_snippet(out_path=dd_wav, midi_path=notes_file)),
            timeout=600,
        )
        ok = res.ok and s.wait_for_file(dd_wav, timeout=30)
        if ok:
            host_dd = Path(tempfile.gettempdir()) / "dawfarm_smoke_dd.wav"
            s.get(dd_wav, host_dd)
            audio, _sr = sf.read(host_dd)
            ok = float(np.abs(audio).max()) > 1e-4
        failures += not check("DawDreamer in-container render", bool(ok),
                              (res.stderr.strip()[:300] if not res.ok else ""))

        dawfarm.reset_project(s)

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
