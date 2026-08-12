#!/usr/bin/env python
"""Score swift-infer transcription generations: execute in REAPER, pitch/onset F1.

Companion to eval_transcription_f1.py for generations produced by `swift infer`
(which renders the exact training template). Marker-tolerant parsing: the
current checkpoints emit a corrupted rare token where the <tool_call> special
token belongs, so the command is extracted from the {"name": "Bash", ...} JSON
blob directly, regardless of the surrounding marker bytes.

Usage:
    python scripts/score_swift_transcription_results.py \
        --results <swift_infer_result.jsonl> --val outputs/transcription_lora/val.jsonl \
        --session daw-farm-reaper-12
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

from maestro.reaper.dawfarm import DockerSession, reset_project, create_vital_track
from scripts.eval_transcription_f1 import (  # noqa: E402
    READ_TAKE_LUA, f1, gt_notes_from_record, _NOTES_RE,
)

_BLOB_RE = re.compile(r'\{"name":\s*"Bash".*', re.S)


def extract_command(resp: str) -> str | None:
    m = _BLOB_RE.search(resp)
    if not m:
        return None
    blob = m.group(0)
    # balanced-brace scan (the response may trail garbage after the JSON)
    depth = 0
    in_str = False
    esc = False
    for i, ch in enumerate(blob):
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
        elif ch == '"' and not esc:
            in_str = not in_str
        elif not in_str:
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(blob[: i + 1])["arguments"]["command"]
                    except Exception:
                        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--slice", type=Path, required=True,
                    help="the opener jsonl fed to swift infer (provides id order)")
    ap.add_argument("--val", type=Path, default=Path("outputs/transcription_lora/val.jsonl"))
    ap.add_argument("--session", default="daw-farm-reaper-12")
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    gt_by_id = {}
    for line in open(args.val):
        r = json.loads(line)
        gt = gt_notes_from_record(r)
        if gt:
            gt_by_id[r["id"]] = gt

    slice_ids = [json.loads(l)["id"] for l in open(args.slice)]
    s = DockerSession(args.session)
    rows = []
    results = [json.loads(l) for l in open(args.results)]
    for i, res in enumerate(results):
        rid = res.get("id") or slice_ids[i]
        gt = gt_by_id.get(rid)
        resp = res.get("response") or ""
        cmd = extract_command(resp)
        row = {"id": rid, "n_gt": len(gt) if gt else None,
               "command_parsed": bool(cmd), "gen_len": len(resp)}
        if cmd and gt:
            reset_project(s)
            create_vital_track(s)
            r2 = s.exec_bash(cmd if cmd.lstrip().startswith("python") else f"python3 - <<'PY'\n{cmd}\nPY",
                             timeout=180)
            row["code_executed"] = r2.returncode == 0
            if r2.returncode == 0:
                lua = s.exec_lua(READ_TAKE_LUA, timeout=60)
                got = []
                body = (lua.stdout or "").strip().splitlines()
                if body and body[0]:
                    for tokn in body[0].split(";"):
                        if tokn:
                            p, st, en = tokn.split(",")
                            got.append((int(p), float(st), float(en)))
                row["n_got"] = len(got)
                row["pitch_f1"], row["onset_f1"] = f1(gt, got, args.onset_tol)
            else:
                row["exec_error"] = (r2.stderr or r2.stdout or "")[-200:]
        rows.append(row)
        print(f"[{i+1}/{len(results)}] {rid}: parsed={row['command_parsed']} "
              f"exec={row.get('code_executed')} pitch={row.get('pitch_f1', '—')} "
              f"onset={row.get('onset_f1', '—')}", flush=True)

    scored = [r for r in rows if "pitch_f1" in r]
    summary = {
        "n": len(rows),
        "command_parse_rate": sum(1 for r in rows if r["command_parsed"]) / max(1, len(rows)),
        "code_exec_rate": sum(1 for r in rows if r.get("code_executed")) / max(1, len(rows)),
        "mean_pitch_f1": sum(r["pitch_f1"] for r in scored) / max(1, len(scored)),
        "mean_onset_f1": sum(r["onset_f1"] for r in scored) / max(1, len(scored)),
        "n_scored": len(scored),
    }
    print(json.dumps(summary, indent=2))
    out = args.out or args.results.with_suffix(".scored.json")
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
