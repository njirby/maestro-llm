#!/usr/bin/env python
"""Validate legacy search-agent records against a real daw-farm container.

Executes every Bash tool_call of N search records in sequence inside a
container (project reset + Vital track + melody seeded first), comparing
actual results to the recorded (fabricated) tool_responses. Focus points:
  - candidate name listing: does the env's wavetable index order match the
    recorded names? (index-order bugs poison every downstream selection)
  - batch renders: do all probe wavs get written, non-silent, non-stale?
  - any snippet errors (legacy snippets predate the daw-farm hardening).

Usage:
    python scripts/check_search_env_execution.py \
        --records 3 --session daw-farm-reaper-12
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.reaper.dawfarm import (
    DockerSession, reset_project, create_vital_track, set_project_tempo,
)

_NOTES_RE = re.compile(r"^notes = (\[.*\])$", re.M)


def seed_sample(s: DockerSession, sid: str, trans_by_sid: dict) -> bool:
    reset_project(s)
    create_vital_track(s)
    trans = trans_by_sid.get(sid)
    if not trans:
        return False
    tempo = trans.get("meta", {}).get("tempo") or 120
    try:
        set_project_tempo(s, float(tempo))
    except Exception:
        pass
    for m in trans["messages"]:
        if m["role"] == "tool_call" and "MIDI_InsertNote" in str(m.get("content", "")):
            cmd = json.loads(m["content"])["arguments"]["command"]
            nm = _NOTES_RE.search(cmd)
            if nm:
                notes_json = json.dumps({"notes": eval(nm.group(1))})
                s.exec_bash(f"mkdir -p /tmp/agents/{sid} && cat > /tmp/agents/{sid}/{sid}_notes.json <<'EOF'\n{notes_json}\nEOF")
                return True
    return False


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-file", type=Path,
                    default=Path("outputs/sft_32k_repaired/search_aligned.jsonl"))
    ap.add_argument("--trans-file", type=Path,
                    default=Path("outputs/sft_32k_repaired/transcription_aligned.jsonl"))
    ap.add_argument("--records", type=int, default=3)
    ap.add_argument("--session", default="daw-farm-reaper-12")
    ap.add_argument("--out", type=Path, default=Path("outputs/search_lora/env_check.json"))
    args = ap.parse_args()

    trans_by_sid = {}
    for l in open(args.trans_file):
        r = json.loads(l)
        trans_by_sid.setdefault(r["meta"]["sample_id"], r)

    s = DockerSession(args.session)
    reports = []
    taken = 0
    for l in open(args.search_file):
        if taken >= args.records:
            break
        rec = json.loads(l)
        sid = rec["meta"]["sample_id"]
        if not seed_sample(s, sid, trans_by_sid):
            continue
        taken += 1
        rep = {"id": rec["id"], "steps": [], "errors": 0, "divergent_names": None,
               "renders_expected": 0, "renders_found": 0, "renders_silent": 0}
        msgs = rec["messages"]
        for i, m in enumerate(msgs):
            if m["role"] != "tool_call":
                continue
            tc = json.loads(m["content"])
            if tc.get("name") != "Bash":
                continue
            cmd = tc["arguments"]["command"]
            res = s.exec_bash(cmd if cmd.lstrip().startswith("python") else f"python3 - <<'PY'\n{cmd}\nPY",
                              timeout=600)
            recorded = msgs[i + 1]["content"] if i + 1 < len(msgs) and msgs[i + 1]["role"] == "tool_response" else ""
            step = {"idx": i, "rc": res.returncode,
                    "stderr_tail": (res.stderr or "")[-150:] if res.returncode else ""}
            if res.returncode:
                rep["errors"] += 1
            # name-listing comparison
            if "wavetables" in (res.stdout or "") and '"name"' in (res.stdout or ""):
                try:
                    actual = [w["name"] for w in json.loads((res.stdout or "").strip().splitlines()[-1])["wavetables"]]
                    rec_names = re.findall(r'\\"name\\": \\"([^"\\\\]+)\\"', recorded)
                    if rec_names:
                        mismatch = [(a, b) for a, b in zip(actual, rec_names) if a != b]
                        rep["divergent_names"] = {"n_actual": len(actual), "n_recorded": len(rec_names),
                                                  "mismatches": mismatch[:5]}
                except Exception as exc:
                    step["name_parse_error"] = str(exc)[:100]
            rep["steps"].append(step)
        # render inventory: recorded audios are probe wav paths inside /tmp
        probe_paths = [a for a in rec.get("audios", []) if "/tmp/" in a]
        rep["renders_expected"] = len(probe_paths)
        for p in probe_paths:
            chk = s.exec_bash(
                f"python3 -c \"import soundfile as sf, os; "
                f"e=os.path.exists('{p}'); "
                f"print('E', e, end=' '); "
                f"d=sf.read('{p}')[0] if e else None; "
                f"import numpy as np; print('S', bool(e and float(np.abs(d).max())<1e-5))\"",
                timeout=60)
            out = (chk.stdout or "")
            if "E True" in out:
                rep["renders_found"] += 1
                if "S True" in out:
                    rep["renders_silent"] += 1
        reports.append(rep)
        print(f"[{taken}] {rec['id']}: errors={rep['errors']} "
              f"renders {rep['renders_found']}/{rep['renders_expected']} "
              f"(silent {rep['renders_silent']}) "
              f"names_div={bool(rep['divergent_names'] and rep['divergent_names']['mismatches'])}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(reports, open(args.out, "w"), indent=2)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
