#!/usr/bin/env python
"""Surgically repair legacy (pre-daw-farm) main SFT records for POC training.

Legacy records are internally coherent (narration/audio/intended edits agree)
but teach behaviors that malfunction in the real daw-farm environment. This
rewrites the offending tool_call snippet text in place — fabricated tool
responses remain valid because none of them depend on the changed code paths.

  R1  param-search discrete sweep -> format-only variant (the sweep stamps
      REAPER's VST3 param cache and reverts chunk-applied edits at render)
  R2  REAPER render snippet -> current version (stale-file remove + wait loop)
  R3  insert target LFO-shape application into the tuple-apply snippet
      (audio contains the LFO shapes; legacy actions never set them)
  R4  move track creation before the default render (render reads the synth)
  F1  drop records with the GT-slice teleport tell (non-arithmetic slices)
  F2  drop records failing ms-swift schema validation

Non-main records pass through untouched. Originals are never modified.

Usage:
    python scripts/repair_legacy_sft.py \
        --input outputs/sft_32k/sft_train_v3.jsonl \
        --presets-dir outputs/sft_32k/presets \
        --out-dir outputs/sft_32k_repaired
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

from scripts.agent_sft_common import (
    assert_valid_ms_swift_multiturn_record,
    build_param_search_snippet,
    build_reaper_render_snippet,
    _wrap_as_bash,
)

# --- R1 blocks: derive from the CURRENT builder so indentation always matches.
_cur_snippet = build_param_search_snippet("dummy")
_m = re.search(r"( *if step > 0:\n.*?entry\['type'\] = 'discrete'\n)", _cur_snippet, re.S)
assert _m, "current param-search discrete block not found"
CURRENT_BLOCK = _m.group(1)
# Legacy variant: no comment lines, SetParam sweep + restore, GetFormatted call.
_legacy_lines = []
for line in CURRENT_BLOCK.splitlines(keepends=True):
    if line.lstrip().startswith("#"):
        continue
    if "TrackFX_FormatParamValueNormalized" in line:
        indent = line[: len(line) - len(line.lstrip())]
        _legacy_lines.append(f"{indent}RPR.TrackFX_SetParam(track, 0, i, ov)\n")
        _legacy_lines.append(
            f"{indent}_od = RPR.TrackFX_GetFormattedParamValue(track, 0, i, '', 2048)\n")
        continue
    if "entry['type'] = 'discrete'" in line:
        indent = line[: len(line) - len(line.lstrip())]
        _legacy_lines.append(f"{indent}RPR.TrackFX_SetParam(track, 0, i, val)\n")
    _legacy_lines.append(line)
LEGACY_BLOCK = "".join(_legacy_lines)

_RENDER_OUT_RE = re.compile(r"out_path = os\.path\.abspath\('([^']+)'\)")
_TELEPORT_RE = re.compile(r"across slices \[([0-9,\- ]+)\]")
_MODS_ANCHOR = "preset['settings']['modulations'] = "


def _is_teleport(record: dict) -> bool:
    blob = "".join(str(m.get("content", "")) for m in record["messages"]
                   if m.get("role") == "assistant")
    m = _TELEPORT_RE.search(blob)
    if not m:
        return False
    starts = [int(s.split("-")[0]) for s in m.group(1).split(",")]
    if len(starts) < 3:
        return False
    d = starts[1] - starts[0]
    return any(starts[i + 1] - starts[i] != d for i in range(len(starts) - 1))


def repair_main(record: dict, presets_dir: Path, stats: dict) -> dict | None:
    msgs = record["messages"]
    lfos_literal = None
    sid = record.get("meta", {}).get("sample_id", "")
    pf = presets_dir / f"{sid}_target.vital"
    if pf.exists():
        try:
            lfos_literal = repr(json.load(open(pf)).get("settings", {}).get("lfos"))
        except Exception:
            pass
    if lfos_literal is None:
        stats["r3_missing_preset"] += 1

    r1 = r2 = r3 = 0
    for m in msgs:
        if m.get("role") != "tool_call":
            continue
        try:
            tc = json.loads(m["content"])
        except Exception:
            continue
        if tc.get("name") != "Bash":
            continue
        cmd = tc["arguments"].get("command", "")
        orig = cmd
        # R1
        if LEGACY_BLOCK in cmd:
            cmd = cmd.replace(LEGACY_BLOCK, CURRENT_BLOCK)
            r1 += 1
        # R2 — regenerate render snippets that lack the wait loop
        if ("Main_OnCommand(42230, 0)" in cmd and "listen_probe" in cmd
                and "prev_size" not in cmd and "GetTrackNumMediaItems" in cmd):
            om = _RENDER_OUT_RE.search(cmd)
            if om:
                cmd = _wrap_as_bash(build_reaper_render_snippet(out_path=om.group(1)))
                r2 += 1
        # R3 — tuple-apply snippet only (has the modulations literal + chunk write)
        if (_MODS_ANCHOR in cmd and "build_vital_chunk(preset)" in cmd
                and "_lfos" not in cmd and lfos_literal is not None):
            idx = cmd.index(_MODS_ANCHOR)
            eol = cmd.index("\n", idx) + 1
            cmd = (cmd[:eol]
                   + f"_lfos = {lfos_literal}\n"
                   + "if _lfos is not None:\n"
                   + "    preset['settings']['lfos'] = _lfos\n"
                   + cmd[eol:])
            r3 += 1
        if cmd != orig:
            tc["arguments"]["command"] = cmd
            m["content"] = json.dumps(tc, ensure_ascii=False)
    stats["r1"] += r1
    stats["r2"] += r2
    stats["r3"] += r3
    if r1 == 0:
        stats["r1_unmatched_records"] += 1

    # R4 — move the track-creation trio before the default-render pair.
    def _find(pred, start=0):
        for i in range(start, len(msgs)):
            if pred(msgs[i]):
                return i
        return -1

    di = _find(lambda m: m.get("role") == "tool_call"
               and "_default.wav" in str(m.get("content", ""))
               and "render_vital_preset" in str(m.get("content", "")))
    ci = _find(lambda m: m.get("role") == "tool_call"
               and "InsertTrackAtIndex" in str(m.get("content", "")), max(di, 0))
    if (di != -1 and ci > di + 1
            and msgs[di + 1].get("role") == "tool_response"
            and ci >= 1 and msgs[ci - 1].get("role") == "assistant"
            and ci + 1 < len(msgs) and msgs[ci + 1].get("role") == "tool_response"):
        trio = msgs[ci - 1:ci + 2]
        del msgs[ci - 1:ci + 2]
        msgs[di:di] = trio
        stats["r4"] += 1
    else:
        stats["r4_unmatched"] += 1

    try:
        assert_valid_ms_swift_multiturn_record(record)
    except Exception:
        stats["f2_schema_dropped"] += 1
        return None
    return record


_DISPATCH_KEY_RE = {
    "search_v2": lambda rid: ("search-r{}-a{}".format(*__import__("re").match(r".*_r(\d+)_agent(\d+)_search$", rid).groups())
                              if __import__("re").match(r".*_r(\d+)_agent(\d+)_search$", rid) else None),
    "judge": lambda rid: ("judge-{}".format(__import__("re").match(r".*_r(\d+)_judge$", rid).group(1))
                          if __import__("re").match(r".*_r(\d+)_judge$", rid) else "judge-1"),
    "melody_transcription": lambda rid: "transcribe",
}


def align_subagents(source: Path, out_dir: Path) -> dict:
    """D1: rewrite subagent record openers to the paired main's dispatch
    prompt text (audio attachment position preserved). Join: sample_id +
    dispatch name (search-rR-aN / judge-R / transcribe-<sid>)."""
    import re as _re
    # 1. dispatch prompt lookup from ALL original mains (prompts aren't the leak)
    lookup: dict[tuple, str] = {}
    for line in open(source):
        r = json.loads(line)
        if r.get("task_type") != "main":
            continue
        sid = r.get("meta", {}).get("sample_id", "")
        for m in r["messages"]:
            if m.get("role") != "tool_call" or '"Agent"' not in str(m.get("content", ""))[:20]:
                continue
            a = json.loads(m["content"]).get("arguments", {})
            name = a.get("name", "")
            if name.startswith("transcribe-"):
                key = (sid, "transcribe")
            else:
                key = (sid, name)
            lookup[key] = a.get("prompt", "")
    stats = {"aligned": 0, "fallback_kept": 0, "by_type": {}}
    outs = {t: open(out_dir / f"{short}_aligned.jsonl", "w")
            for t, short in (("search_v2", "search"), ("judge", "judge"),
                             ("melody_transcription", "transcription"))}
    for line in open(source):
        r = json.loads(line)
        t = r.get("task_type")
        if t not in outs:
            continue
        sid = r.get("meta", {}).get("sample_id", "")
        rid = r.get("id", "")
        if t == "search_v2":
            m = _re.match(r".*_r(\d+)_agent(\d+)_search$", rid)
            key = (sid, f"search-r{m.group(1)}-a{m.group(2)}") if m else None
        elif t == "judge":
            m = _re.match(r".*_r(\d+)_judge$", rid)
            key = (sid, f"judge-{m.group(1)}") if m else (sid, "judge-1")
        else:
            key = (sid, "transcribe")
        prompt = lookup.get(key) if key else None
        first = r["messages"][0]
        if prompt and first.get("role") == "user" and str(first.get("content", "")).startswith("<audio>"):
            first["content"] = "<audio>\n" + prompt
            stats["aligned"] += 1
        else:
            stats["fallback_kept"] += 1
        stats["by_type"][t] = stats["by_type"].get(t, 0) + 1
        try:
            assert_valid_ms_swift_multiturn_record(r)
            outs[t].write(json.dumps(r, ensure_ascii=False) + "\n")
        except Exception:
            stats.setdefault("schema_dropped", 0)
            stats["schema_dropped"] = stats.get("schema_dropped", 0) + 1
    for f in outs.values():
        f.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=Path)
    ap.add_argument("--presets-dir", type=Path,
                    default=Path("outputs/sft_32k/presets"))
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    stats = {k: 0 for k in ("mains", "kept", "r1", "r2", "r3", "r4",
                            "r1_unmatched_records", "r4_unmatched",
                            "r3_missing_preset", "f1_teleport_dropped",
                            "f2_schema_dropped", "passthrough")}
    out_main = open(args.out_dir / "main_repaired.jsonl", "w")
    for line in open(args.input):
        r = json.loads(line)
        if r.get("task_type") != "main":
            stats["passthrough"] += 1
            continue
        stats["mains"] += 1
        if _is_teleport(r):
            stats["f1_teleport_dropped"] += 1
            continue
        rep = repair_main(r, args.presets_dir, stats)
        if rep is not None:
            stats["kept"] += 1
            out_main.write(json.dumps(rep, ensure_ascii=False) + "\n")
    out_main.close()
    astats = align_subagents(args.input, args.out_dir)
    stats["align_subagents"] = astats
    with open(args.out_dir / "repair_report.json", "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
