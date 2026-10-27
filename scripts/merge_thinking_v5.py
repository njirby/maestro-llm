"""Merge guessed thinking (scripts/add_thinking_v5.py) into v5 records as per-turn `reasoning_content`.

OpenCode re-sends every earlier assistant turn's reasoning as `reasoning_content` (docs/opencode_capture/goldens/reasoning),
so a turn's reasoning is stored on the first message of its step (the assistant message, or the first tool_call when the
turn has no visible text). Only turns whose thinking passed the checks get it; others stay without reasoning.

  .venv/bin/python scripts/merge_thinking_v5.py --run outputs/v5_rollouts/pilot20c \
      --thinking outputs/thinking_capture/hearing_lab/full_s0016 --out outputs/v5_rollouts/pilot20c_thinking

Writes <out>/{main,transcriber,searcher,judge}.jsonl for the rollouts that have thinking, and checks that the records
without the reasoning key are unchanged and still pass maestro.v5.validate.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from maestro.v5.validate import validate_record  # noqa: E402

FILES = ("main", "transcriber", "searcher", "judge")
STEP_ROLES = {"assistant", "tool_call"}


def steps(messages: list[dict]) -> list[int]:
    """Index of the first message of each assistant step (a run of assistant / tool_call messages)."""
    out, prev = [], False
    for i, m in enumerate(messages):
        cur = m["role"] in STEP_ROLES
        if cur and not prev:
            out.append(i)
        prev = cur
    return out


def route_of(rec_id: str) -> str:
    return "main" if rec_id.endswith(".main") else rec_id


def strip(rec: dict) -> dict:
    r = copy.deepcopy(rec)
    for m in r["messages"]:
        m.pop("reasoning_content", None)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--thinking", type=Path, required=True, help="dir of <sid>.jsonl from add_thinking_v5.py")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    th = {}
    for f in sorted(a.thinking.glob("*.jsonl")):
        for line in open(f):
            r = json.loads(line)
            th[(r["sid"], r["route"], r["step"])] = r
    sids = {k[0] for k in th}
    a.out.mkdir(parents=True, exist_ok=True)
    tot = {"turns": 0, "with_reasoning": 0, "records": 0, "invalid": 0}
    for name in FILES:
        rows = []
        for line in open(a.run / f"{name}.jsonl"):
            rec = json.loads(line)
            sid = rec["id"].split(".")[0]
            if sid not in sids:
                continue
            route = route_of(rec["id"])
            if route != "main" and not route.startswith(sid):
                route = f"{sid}.{route}"
            out = copy.deepcopy(rec)
            starts = steps(out["messages"])
            for k, i in enumerate(starts):
                tot["turns"] += 1
                r = th.get((sid, route, k))
                if r and r["ok"] and r["thinking"]:
                    out["messages"][i]["reasoning_content"] = r["thinking"]
                    tot["with_reasoning"] += 1
            assert strip(out) == rec, f"{rec['id']}: merge changed something besides reasoning_content"
            errs = validate_record(strip(out), check_files=False)
            tot["records"] += 1
            if errs:
                tot["invalid"] += 1
                print(rec["id"], "INVALID", errs[:2])
            rows.append(out)
        if rows:
            (a.out / f"{name}.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(tot)


if __name__ == "__main__":
    main()
