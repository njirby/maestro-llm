#!/usr/bin/env python
"""Static container validation of generated unified rollouts.

Replays each sample's recorded tool calls in a FRESH daw-farm container, in
the true rollout order (main record's calls, with each subagent record's
calls spliced in at its `task` dispatch point), and compares actual results
to the recorded tool_responses. This is the gate before training: its job is
to find reasons NOT to train on a corpus.

Checks per sample:
  - container hygiene: recycle + assert_clean before the sample
  - bash: executes in-container; result classified match / acceptable / DIVERGENT
  - read: referenced file exists in container, is valid non-silent audio,
    duration within 10% of the recorded "Attached audio: ..." line
  - task: <task_result> payload equals the linked subagent record's final
    assistant message; dispatch prompt equals that record's opener text
  - host audio: every path in `audios` exists and is non-silent
  - final render vs GT: CLAP cosine, reported against the manifest's
    determinism ceiling for that sample
  - probe discriminability: spectral-centroid spread per search probe dir
Corpus-level:
  - opencode-contract validation of every record (validate_oc_record)
  - lowercase tool names from {bash, read, skill, task}; leading system turn

Usage:
    python scripts/validate_pilot_rollouts.py \
        --records-dir outputs/pilot_oc_v2 \
        --manifest outputs/dawfarm_run10/manifest.jsonl \
        --daw-farm docker --workers 4
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.reaper import dawfarm as _dawfarm  # noqa: E402
from maestro.reaper.dawfarm import DawFarmPool  # noqa: E402

VALID_TOOLS = {"bash", "read", "skill", "task"}
_ATTACHED_RE = re.compile(r"Attached audio: (\S+) \(([\d.]+)s, (\d+) Hz\)")
_NUM_RE = re.compile(r"-?\d+\.?\d*(?:[eE][-+]?\d+)?")


# ---------------------------------------------------------------------------
# result classification
# ---------------------------------------------------------------------------

def _floats(s: str) -> list[float]:
    out = []
    for m in _NUM_RE.finditer(s):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            pass
    return out


def classify_bash(recorded: str, actual: str, rc: int) -> tuple[str, str]:
    """-> (verdict, note). verdict in {match, acceptable, divergent, error}."""
    if rc != 0:
        return "error", f"exit {rc}"
    rec, act = recorded.strip(), actual.strip()
    if rec == act:
        return "match", ""
    # structured: same JSON keys, values differing only numerically / by path
    try:
        rj, aj = json.loads(rec), json.loads(act)
        if isinstance(rj, dict) and isinstance(aj, dict):
            if set(rj) == set(aj):
                same = all(rj[k] == aj[k] for k in rj
                           if not isinstance(rj[k], (int, float, dict, list)))
                return ("acceptable" if same else "acceptable",
                        "same JSON shape, differing values")
            return "divergent", f"JSON keys differ: {sorted(set(rj) ^ set(aj))[:5]}"
    except Exception:
        pass
    # numeric-only drift (durations, sizes)
    rf, af = _floats(rec), _floats(act)
    if rf and len(rf) == len(af):
        skeleton_r = _NUM_RE.sub("#", rec)
        skeleton_a = _NUM_RE.sub("#", act)
        if skeleton_r == skeleton_a:
            worst = max((abs(a - b) / max(1e-9, abs(b)) for a, b in zip(af, rf)), default=0.0)
            return ("acceptable" if worst < 0.25 else "divergent",
                    f"numeric drift {worst:.1%}")
    ratio = difflib.SequenceMatcher(None, rec[:500], act[:500]).ratio()
    return ("acceptable" if ratio > 0.85 else "divergent", f"text ratio {ratio:.2f}")


# ---------------------------------------------------------------------------
# host-side audio checks
# ---------------------------------------------------------------------------

def audio_stats(path: Path) -> dict:
    try:
        y, sr = sf.read(str(path))
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        peak = float(np.abs(y).max()) if len(y) else 0.0
        return {"exists": True, "peak": peak, "silent": peak < 1e-5,
                "dur_s": len(y) / sr if sr else 0.0, "sr": sr}
    except Exception as exc:
        return {"exists": False, "error": str(exc)[:120]}


def centroid(path: Path) -> float | None:
    try:
        y, sr = sf.read(str(path))
        if getattr(y, "ndim", 1) > 1:
            y = y.mean(axis=1)
        if not len(y):
            return None
        S = np.abs(np.fft.rfft(y))
        fr = np.fft.rfftfreq(len(y), 1 / sr)
        tot = S.sum()
        return float((S * fr).sum() / tot) if tot > 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# record helpers
# ---------------------------------------------------------------------------

def load_records(d: Path) -> dict[str, list[dict]]:
    out = {}
    for kind in ("main", "search", "judge", "transcription"):
        matches = sorted(d.glob(f"{kind}_*.jsonl"))
        recs = []
        for f in matches:
            recs.extend(json.loads(l) for l in open(f))
        out[kind] = recs
    return out


def tool_calls(rec: dict) -> list[tuple[int, dict]]:
    out = []
    for i, m in enumerate(rec["messages"]):
        if m.get("role") == "tool_call":
            try:
                out.append((i, json.loads(m["content"])))
            except Exception:
                out.append((i, {"name": "<unparseable>", "arguments": {}}))
    return out


def response_after(rec: dict, call_idx: int) -> str | None:
    """Positional pairing that respects parallel dispatch: the Nth tool_call in
    a consecutive run pairs with the Nth tool_response after that run."""
    msgs = rec["messages"]
    start = call_idx
    while start - 1 >= 0 and msgs[start - 1].get("role") == "tool_call":
        start -= 1
    end = call_idx
    while end + 1 < len(msgs) and msgs[end + 1].get("role") == "tool_call":
        end += 1
    pos = call_idx - start
    resp = [j for j in range(end + 1, len(msgs)) if msgs[j].get("role") == "tool_response"]
    # only the contiguous response run
    contiguous = []
    for j in resp:
        if not contiguous or j == contiguous[-1] + 1:
            contiguous.append(j)
        else:
            break
    if pos < len(contiguous):
        return msgs[contiguous[pos]].get("content")
    return None


def final_assistant(rec: dict) -> str:
    for m in reversed(rec["messages"]):
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
    return ""


def opener_text(rec: dict) -> str:
    for m in rec["messages"]:
        if m.get("role") == "user":
            return str(m.get("content", "")).replace("<audio>", "").strip()
    return ""


def link_subagent(call: dict, sample_id: str, by_kind: dict) -> dict | None:
    a = call.get("arguments", {})
    st = a.get("subagent_type")
    if st == "melody_transcription":
        return next((r for r in by_kind["transcription"]
                     if r["meta"].get("sample_id") == sample_id), None)
    if st == "wavetable_judge":
        return next((r for r in by_kind["judge"]
                     if r["meta"].get("sample_id") == sample_id), None)
    if st == "wavetable_search":
        m = re.search(r"wavetables (\d+)-(\d+)", a.get("description", ""))
        if not m:
            return None
        start = int(m.group(1))
        return next((r for r in by_kind["search"]
                     if r["meta"].get("sample_id") == sample_id
                     and int(r["meta"].get("shard_start", -1)) == start), None)
    return None


# ---------------------------------------------------------------------------
# per-sample replay
# ---------------------------------------------------------------------------

def replay_record(session, rec: dict, report: dict, label: str, timeout: float,
                  on_task=None, resolve_sub=None) -> None:
    """Execute a record's bash calls; verify its read calls.

    `on_task(call)` is invoked when a `task` dispatch is reached, so the
    linked subagent's calls run at the point they really ran in the rollout
    (a subagent depends on state the parent created before dispatching it).
    """
    for idx, call in tool_calls(rec):
        name = call.get("name")
        args = call.get("arguments", {})
        recorded = response_after(rec, idx) or ""
        if name == "bash":
            cmd = args.get("command", "")
            try:
                res = session.exec_bash(cmd, timeout=timeout)
                verdict, note = classify_bash(recorded, res.stdout or "", res.returncode)
                entry = {"record": label, "msg": idx, "verdict": verdict, "note": note}
                if verdict in ("divergent", "error"):
                    entry["cmd_head"] = cmd[:150]
                    entry["recorded_head"] = recorded[:180]
                    entry["actual_head"] = (res.stdout or "")[:180]
                    entry["stderr_tail"] = (res.stderr or "")[-180:]
                report["bash"].append(entry)
            except Exception as exc:
                report["bash"].append({"record": label, "msg": idx, "verdict": "error",
                                       "note": f"exec raised: {str(exc)[:140]}",
                                       "cmd_head": cmd[:150]})
        elif name == "read":
            fp = args.get("filePath", "")
            m = _ATTACHED_RE.search(recorded)
            entry = {"record": label, "msg": idx, "path": fp,
                     "contract_ok": bool(recorded.startswith("<audio>") and m)}
            try:
                chk = session.exec_bash(
                    "python3 - <<'PY'\n"
                    "import json,os\n"
                    f"p={fp!r}\n"
                    "d={'exists':os.path.exists(p)}\n"
                    "if d['exists']:\n"
                    "    import soundfile as sf, numpy as np\n"
                    "    y,sr=sf.read(p)\n"
                    "    y=y.mean(axis=1) if getattr(y,'ndim',1)>1 else y\n"
                    "    d.update(dur=len(y)/sr, peak=float(abs(y).max()))\n"
                    "print(json.dumps(d))\nPY", timeout=90.0)
                got = json.loads((chk.stdout or "{}").strip().splitlines()[-1])
                entry.update(exists=got.get("exists"), peak=got.get("peak"),
                             dur=got.get("dur"))
                entry["silent"] = bool(got.get("exists") and got.get("peak", 0) < 1e-5)
                if m and got.get("dur"):
                    rec_dur = float(m.group(2))
                    entry["dur_ok"] = abs(got["dur"] - rec_dur) <= 0.10 * max(rec_dur, 1e-6)
                    entry["recorded_dur"] = rec_dur
            except Exception as exc:
                entry.update(exists=False, error=str(exc)[:120])
            report["read"].append(entry)
        elif name == "task":
            # resolve via callable: tool_calls() re-parses JSON each time, so
            # annotations set on a previous parse would be lost.
            sub = resolve_sub(call) if resolve_sub is not None else None
            entry = {"record": label, "msg": idx,
                     "subagent_type": args.get("subagent_type")}
            if sub is None:
                entry["linked"] = False
            else:
                entry["linked"] = True
                payload = ""
                mm = re.search(r"<task_result>\n(.*)\n</task_result>", recorded, re.S)
                if mm:
                    payload = mm.group(1).strip()
                entry["result_matches_subagent_final"] = (
                    payload == final_assistant(sub).strip())
                entry["prompt_matches_opener"] = (
                    args.get("prompt", "").strip() == opener_text(sub))
                if not entry["result_matches_subagent_final"]:
                    entry["payload_head"] = payload[:150]
                    entry["subfinal_head"] = final_assistant(sub)[:150]
            report["task"].append(entry)
            if on_task is not None and sub is not None:
                on_task(sub)


def validate_sample(main: dict, by_kind: dict, pool, manifest: dict,
                    timeout: float, lock: threading.Lock) -> dict:
    sid = main["meta"]["sample_id"]
    report = {"sample_id": sid, "bash": [], "read": [], "task": [],
              "host_audio": [], "errors": []}
    try:
        with pool.acquire() as sess:
            report["session"] = sess.name
            sess.recycle()
            _dawfarm.assert_clean(sess)
            _dawfarm.sync_vital_data(sess, None)
            _dawfarm.prepare_sample_dirs(sess, sid)
            report["hygiene_ok"] = True

            # Replay the main record in order, splicing each subagent's calls
            # in at its dispatch point (subagents depend on parent-made state).
            def _resolve(call, _sid=sid):
                return link_subagent(call, _sid, by_kind)

            def _on_task(sub, _sess=sess):
                replay_record(_sess, sub, report, sub["id"], timeout)

            replay_record(sess, main, report, main["id"], timeout,
                          on_task=_on_task, resolve_sub=_resolve)
    except Exception as exc:
        report["errors"].append(f"{type(exc).__name__}: {exc}")
        report["hygiene_ok"] = report.get("hygiene_ok", False)
        report["traceback"] = traceback.format_exc()[-600:]

    # host-side audio existence / non-silence
    for kind in ("main", "search", "judge", "transcription"):
        for rec in by_kind[kind]:
            if rec["meta"].get("sample_id") != sid:
                continue
            for p in rec.get("audios", []):
                st = audio_stats(Path(p))
                if not st.get("exists") or st.get("silent"):
                    report["host_audio"].append({"record": rec["id"], "path": p, **st})
    report["host_audio_total_checked"] = sum(
        len(r.get("audios", [])) for k in by_kind for r in by_kind[k]
        if r["meta"].get("sample_id") == sid)

    # probe discriminability per search record
    spreads = []
    for rec in by_kind["search"]:
        if rec["meta"].get("sample_id") != sid:
            continue
        probes = [Path(p) for p in rec.get("audios", []) if "search_probe_audio" in p]
        cents = [c for c in (centroid(p) for p in probes[:48]) if c is not None]
        if len(cents) >= 4:
            spreads.append({"record": rec["id"], "n": len(cents),
                            "spread_hz": round(max(cents) - min(cents), 1)})
    report["probe_spreads"] = spreads
    return report


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-dir", type=Path, default=Path("outputs/pilot_oc_v2"))
    ap.add_argument("--manifest", type=Path, default=Path("outputs/dawfarm_run10/manifest.jsonl"))
    ap.add_argument("--daw-farm", default="docker")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=420.0)
    ap.add_argument("--skip-clap", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    by_kind = load_records(args.records_dir)
    manifest = {e["sample_id"]: e for e in
                (json.loads(l) for l in open(args.manifest))}
    print(f"records: " + ", ".join(f"{k}={len(v)}" for k, v in by_kind.items()), flush=True)

    # ---- corpus-level contract validation
    from scripts.agent_sft_common import validate_oc_record
    contract = {"checked": 0, "failures": []}
    for kind, recs in by_kind.items():
        for rec in recs:
            contract["checked"] += 1
            errs = list(validate_oc_record(rec))
            names = {c.get("name") for _, c in tool_calls(rec)}
            bad = names - VALID_TOOLS
            if bad:
                errs.append(f"non-contract tool names: {sorted(bad)}")
            if not rec["messages"] or rec["messages"][0].get("role") != "system":
                errs.append("no leading system message")
            if errs:
                contract["failures"].append({"id": rec["id"], "kind": kind, "errors": errs})
    print(f"contract: {contract['checked']} records, "
          f"{len(contract['failures'])} failing", flush=True)

    # ---- container replay
    pool = DawFarmPool.from_spec(args.daw_farm)
    lock = threading.Lock()
    mains = by_kind["main"]
    reports = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(validate_sample, m, by_kind, pool, manifest,
                          args.timeout, lock): m for m in mains}
        for fut in futs:
            try:
                rep = fut.result()
            except Exception as exc:
                rep = {"sample_id": futs[fut]["meta"]["sample_id"],
                       "errors": [f"validate_sample raised: {exc}"]}
            reports.append(rep)
            b = rep.get("bash", [])
            print(f"[{len(reports)}/{len(mains)}] {rep['sample_id']}: "
                  f"bash={len(b)} err={sum(1 for e in b if e['verdict']=='error')} "
                  f"div={sum(1 for e in b if e['verdict']=='divergent')} "
                  f"reads={len(rep.get('read', []))} "
                  f"hostaudio_bad={len(rep.get('host_audio', []))}", flush=True)

    # ---- final render vs GT (CLAP, cpu)
    if not args.skip_clap:
        try:
            from scripts.agent_sft_common import ClapEmbedder
            emb = ClapEmbedder.create("cpu")
            for rep in reports:
                sid = rep["sample_id"]
                rec = next(m for m in mains if m["meta"]["sample_id"] == sid)
                finals = [p for p in rec.get("audios", []) if Path(p).exists()]
                mf = manifest.get(sid, {})
                gt = mf.get("gt_wav")
                if finals and gt and Path(gt).exists():
                    rep["final_audio"] = finals[-1]
                    rep["final_vs_gt_clap"] = round(
                        float(emb.cosine_paths(Path(finals[-1]), Path(gt))), 4)
                    rep["determinism_clap"] = mf.get("determinism_clap")
        except Exception as exc:
            print(f"CLAP scoring skipped: {exc}", flush=True)

    out = args.out or args.records_dir / "validation_report.json"
    payload = {"contract": contract, "samples": reports}
    json.dump(payload, open(out, "w"), indent=2)

    # ---- summary
    tot_bash = sum(len(r.get("bash", [])) for r in reports)
    tot_err = sum(1 for r in reports for e in r.get("bash", []) if e["verdict"] == "error")
    tot_div = sum(1 for r in reports for e in r.get("bash", []) if e["verdict"] == "divergent")
    tot_read = sum(len(r.get("read", [])) for r in reports)
    bad_read = sum(1 for r in reports for e in r.get("read", [])
                   if not e.get("exists") or e.get("silent") or e.get("dur_ok") is False
                   or not e.get("contract_ok"))
    tot_task = sum(len(r.get("task", [])) for r in reports)
    bad_task = sum(1 for r in reports for e in r.get("task", [])
                   if e.get("linked") and (not e.get("result_matches_subagent_final")
                                           or not e.get("prompt_matches_opener")))
    bad_audio = sum(len(r.get("host_audio", [])) for r in reports)
    print("\n=== SUMMARY ===")
    print(f"contract failures : {len(contract['failures'])}/{contract['checked']}")
    print(f"bash calls        : {tot_bash} (errors {tot_err}, divergent {tot_div})")
    print(f"read calls        : {tot_read} (problems {bad_read})")
    print(f"task links        : {tot_task} (mismatches {bad_task})")
    print(f"host audio issues : {bad_audio}")
    print(f"{'sample':<22} {'bash':>5} {'err':>4} {'div':>4} {'clap':>6} {'ceil':>6} {'probe spread':>14}")
    for r in sorted(reports, key=lambda x: x["sample_id"]):
        sp = r.get("probe_spreads", [])
        spread = f"{min(s['spread_hz'] for s in sp):.0f}-{max(s['spread_hz'] for s in sp):.0f}" if sp else "-"
        print(f"{r['sample_id']:<22} {len(r.get('bash', [])):>5} "
              f"{sum(1 for e in r.get('bash', []) if e['verdict']=='error'):>4} "
              f"{sum(1 for e in r.get('bash', []) if e['verdict']=='divergent'):>4} "
              f"{r.get('final_vs_gt_clap', float('nan')):>6} "
              f"{r.get('determinism_clap', float('nan')):>6} {spread:>14}")
    blocking = []
    if contract["failures"]:
        blocking.append(f"{len(contract['failures'])} records fail contract validation")
    if tot_err:
        blocking.append(f"{tot_err} bash calls errored in-container")
    if bad_task:
        blocking.append(f"{bad_task} task links mismatch subagent records")
    if bad_audio:
        blocking.append(f"{bad_audio} referenced audio files missing/silent")
    low_spread = [s for r in reports for s in r.get("probe_spreads", []) if s["spread_hz"] < 500]
    if low_spread:
        blocking.append(f"{len(low_spread)} search records with probe spread < 500 Hz")
    print("\nVERDICT:", "FAIL" if blocking else "PASS")
    for b in blocking:
        print("  BLOCKING:", b)
    print("report:", out)


if __name__ == "__main__":
    main()
