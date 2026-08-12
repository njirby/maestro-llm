#!/usr/bin/env python
"""Mechanical transcription eval: model code -> real REAPER -> pitch/onset F1.

For each val record: send the trained opening (dispatch prompt + target audio)
to the served model, extract the generated MIDI-insert code, execute it in a
daw-farm container (track 0 + Vital prepared), read the notes off the take,
and score against the record's ground-truth notes. First-attempt only — the
capacity signal, not the full verify loop.

Reported per record: tool_format_ok, code_executed, n_notes, pitch/onset F1.

Usage:
    python scripts/eval_transcription_f1.py \
        --val outputs/transcription_lora/val.jsonl \
        --server http://localhost:8010 --model sft-transcription \
        --session daw-farm-reaper-3 [--max-samples 50]
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.reaper.dawfarm import DockerSession, reset_project, create_vital_track

_NOTES_RE = re.compile(r"^notes = (\[.*\])$", re.M)
_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.S)
_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)

READ_TAKE_LUA = """
local tr = reaper.GetTrack(0, 0)
local out = {}
if tr and reaper.GetTrackNumMediaItems(tr) > 0 then
  local item = reaper.GetTrackMediaItem(tr, reaper.GetTrackNumMediaItems(tr) - 1)
  local take = reaper.GetActiveTake(item)
  local _, n = reaper.MIDI_CountEvts(take)
  for i = 0, n - 1 do
    local _, _, _, sppq, eppq, _, pitch, vel = reaper.MIDI_GetNote(take, i)
    local s = reaper.MIDI_GetProjTimeFromPPQPos(take, sppq)
    local e = reaper.MIDI_GetProjTimeFromPPQPos(take, eppq)
    out[#out + 1] = string.format("%d,%.4f,%.4f", pitch, s, e)
  end
end
LOG(table.concat(out, ";"))
return "ok"
"""


def gt_notes_from_record(rec: dict) -> list[dict] | None:
    """Ground truth = the notes literal in the record's FINAL insert snippet
    (the correct attempt; earlier attempts may carry injected mistakes)."""
    notes = None
    for m in rec["messages"]:
        if m["role"] == "tool_call" and "MIDI_InsertNote" in str(m.get("content", "")):
            cmd = json.loads(m["content"])["arguments"]["command"]
            nm = _NOTES_RE.search(cmd)
            if nm:
                try:
                    notes = eval(nm.group(1))
                except Exception:
                    pass
    return notes


def extract_code(msg: dict) -> str | None:
    """Model output -> python code, tolerating tool-call JSON, fences, or raw."""
    for tc in (msg.get("tool_calls") or []):
        try:
            args = json.loads(tc["function"]["arguments"])
            if "command" in args:
                return args["command"]
        except Exception:
            continue
    text = msg.get("content") or ""
    for m in _TOOL_CALL_RE.finditer(text):
        try:
            payload = json.loads(m.group(1))
            cmd = (payload.get("arguments") or {}).get("command")
            if cmd:
                return cmd
        except Exception:
            continue
    fm = _CODE_FENCE_RE.search(text)
    if fm:
        return fm.group(1)
    # raw-code emission (trained format): slice from the first import line
    im = re.search(r"^import \w+", text, re.M)
    if im and ("MIDI_InsertNote" in text or "notes = [" in text or "notes=[" in text):
        return text[im.start():]
    if "MIDI_InsertNote" in text:
        return text
    return None


def f1(gt: list[dict], got: list[tuple], onset_tol: float = 0.05) -> tuple[float, float]:
    """(pitch_f1, pitch_onset_f1). got = [(pitch, start_s, end_s)]."""
    def score(match_fn):
        used = set()
        tp = 0
        for g in gt:
            for j, o in enumerate(got):
                if j in used:
                    continue
                if match_fn(g, o):
                    used.add(j)
                    tp += 1
                    break
        p = tp / len(got) if got else 0.0
        r = tp / len(gt) if gt else 0.0
        return 2 * p * r / (p + r) if (p + r) else 0.0
    pitch = score(lambda g, o: g["pitch"] == o[0])
    onset = score(lambda g, o: g["pitch"] == o[0] and abs(g["start_s"] - o[1]) <= onset_tol)
    return pitch, onset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--val", type=Path, default=Path("outputs/transcription_lora/val.jsonl"))
    ap.add_argument("--server", default="http://localhost:8010")
    ap.add_argument("--model", default="sft-transcription")
    ap.add_argument("--session", default="daw-farm-reaper-3")
    ap.add_argument("--max-samples", type=int, default=50)
    ap.add_argument("--onset-tol", type=float, default=0.05)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    s = DockerSession(args.session)
    rows = []
    recs = [json.loads(l) for l in open(args.val)][: args.max_samples]
    for i, rec in enumerate(recs):
        gt = gt_notes_from_record(rec)
        if not gt:
            continue
        target_wav = rec["audios"][0]
        prompt_text = rec["messages"][0]["content"].replace("<audio>", "").strip()
        b64 = base64.b64encode(open(target_wav, "rb").read()).decode()
        try:
            r = httpx.post(f"{args.server}/v1/chat/completions", json={
                "model": args.model,
                "messages": [{"role": "user", "content": [
                    {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
                    {"type": "text", "text": prompt_text},
                ]}],
                "max_tokens": 2600, "temperature": 0.2,
            }, timeout=600)
            msg = r.json()["choices"][0]["message"]
        except Exception as exc:
            rows.append({"id": rec["id"], "error": f"llm: {exc}"})
            continue
        code = extract_code(msg)
        row = {"id": rec["id"], "n_gt": len(gt), "tool_format_ok": bool(msg.get("tool_calls") or "<tool_call>" in (msg.get("content") or "")),
               "code_found": bool(code)}
        if code:
            reset_project(s)
            create_vital_track(s)
            res = s.exec_bash(code if code.lstrip().startswith("python") else f"python3 - <<'PY'\n{code}\nPY",
                              timeout=180)
            row["code_executed"] = res.returncode == 0
            if res.returncode == 0:
                lua = s.exec_lua(READ_TAKE_LUA, timeout=60)
                got = []
                body = (lua.stdout or "").strip().splitlines()
                if body and body[0]:
                    for tok in body[0].split(";"):
                        if tok:
                            p, st, en = tok.split(",")
                            got.append((int(p), float(st), float(en)))
                row["n_got"] = len(got)
                row["pitch_f1"], row["onset_f1"] = f1(gt, got, args.onset_tol)
            else:
                row["exec_error"] = (res.stderr or "")[-200:]
        rows.append(row)
        print(f"[{i+1}/{len(recs)}] {rec['id']}: "
              f"fmt={row.get('tool_format_ok')} exec={row.get('code_executed')} "
              f"pitch_f1={row.get('pitch_f1', '—')} onset_f1={row.get('onset_f1', '—')}", flush=True)

    scored = [r for r in rows if "pitch_f1" in r]
    summary = {
        "n": len(rows),
        "tool_format_rate": sum(1 for r in rows if r.get("tool_format_ok")) / max(1, len(rows)),
        "code_found_rate": sum(1 for r in rows if r.get("code_found")) / max(1, len(rows)),
        "code_exec_rate": sum(1 for r in rows if r.get("code_executed")) / max(1, len(rows)),
        "mean_pitch_f1": sum(r["pitch_f1"] for r in scored) / max(1, len(scored)),
        "mean_onset_f1": sum(r["onset_f1"] for r in scored) / max(1, len(scored)),
        "n_scored": len(scored),
    }
    print(json.dumps(summary, indent=2))
    out = args.out or Path("outputs/transcription_lora") / f"eval_{args.model.replace('/', '_')}.json"
    json.dump({"summary": summary, "rows": rows}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
