#!/usr/bin/env python
"""Second pass of deferred narration for a v5 run (build_rollouts_v5.py --defer-narration).

The build wrote <run>/narration_slots.jsonl (one verified-notes dict per narration slot) and put ⟦narr:<slot>⟧ tokens in
the main records. This script narrates the slots one mode family at a time, so each pass needs only the server that
mode uses (Omni and the 27B never run together), then swaps the tokens for the text:

  # 27B up (text arm): the edit modes
  .venv/bin/python scripts/narrate_rollouts_v5.py --run R --modes self_edit,target_vs_current \\
      --url http://localhost:8001/v1 --arm text
  # Omni up: first impressions
  .venv/bin/python scripts/narrate_rollouts_v5.py --run R --modes first_impression --url http://localhost:8000/v1
  # any time (re-runnable): tokens -> narration, or the slot's template where there is none / it was flagged
  .venv/bin/python scripts/narrate_rollouts_v5.py --run R --finalize
  # after WS-B changes the notes wording: recompute the slots' verified notes from the saved inputs
  # (<run>/oracle/<sample>/narration_inputs.jsonl; one network-less pool session, renders cached), then narrate again
  .venv/bin/python scripts/narrate_rollouts_v5.py --run R --rebuild-notes [--rebuild-modes self_edit,...]

--rebuild-notes keeps the old slots as narration_slots.<n>.jsonl and the old narrations as narrations.<n>.jsonl, so
every rebuilt slot is narrated afresh by the next pass (slots not rebuilt keep their narrations).
Narrations are appended to <run>/narrations.jsonl (a slot already narrated is skipped, so passes resume). --finalize
keeps the token records as main.pending.jsonl, writes main.jsonl, re-validates every rollout, rebuilds the viewer
pages, and writes <run>/narration_report.json (sources and fallback reasons per mode, chars per mode).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.v5 import validate, viewer  # noqa: E402

TOKEN_RE = re.compile("⟦narr:([^⟧]+)⟧")
MODES = ("first_impression", "self_edit", "target_vs_current")


def jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text().splitlines() if l.strip()] if path.exists() else []


def rebuild_notes(run: Path, modes: list[str]) -> None:
    from maestro.dawfarm_dd import DawDreamerPool
    from maestro.sounddiff import delta
    from maestro.sounddiff.first_impression import notes_first_impression
    from maestro.v5.ctx import OracleRenderer
    from scripts import opencode_contract as oc

    slots = jsonl(run / "narration_slots.jsonl")
    inputs = {}
    for f in sorted((run / "oracle").glob("*/narration_inputs.jsonl")):
        inputs.update({r["slot"]: r for r in jsonl(f)})
    todo = [s for s in slots if s["mode"] in modes]
    missing = [s["slot"] for s in todo if s["slot"] not in inputs]
    assert not missing, f"no saved inputs for {len(missing)} slots, e.g. {missing[:3]} (built before inputs were saved)"
    k = 0
    while (run / f"narration_slots.{k}.jsonl").exists():
        k += 1
    (run / "narration_slots.jsonl").rename(run / f"narration_slots.{k}.jsonl")
    narr = jsonl(run / "narrations.jsonl")
    if narr:
        (run / "narrations.jsonl").rename(run / f"narrations.{k}.jsonl")
        rebuilt = {s["slot"] for s in todo}
        keep = [r for r in narr if r["slot"] not in rebuilt]
        (run / "narrations.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in keep))
    print(f"rebuilding notes for {len(todo)} slots ({', '.join(modes)})", flush=True)
    pool = DawDreamerPool.start(1, prefix="v5a-nar-", host_root=run / "_sessions",
                                skill_mount=oc.CONFIG_DIR + "/skills/vital")
    try:
        with pool.acquire() as sess, open(run / "narration_slots.jsonl", "w") as out:
            for s in slots:
                if s["mode"] in modes:
                    i = inputs[s["slot"]]
                    pr = {r: json.loads(Path(p).read_text()) for r, p in i["presets"].items()}
                    notes = json.loads(Path(i["notes"]).read_text())
                    r = OracleRenderer(sess, Path(i["notes"]).parent / "render_cache")
                    if s["mode"] == "first_impression":
                        n = notes_first_impression(pr["target"], notes, renderer=r)
                    elif s["mode"] == "self_edit":
                        n = delta.notes_self_edit(pr["before"], pr["after"], notes, renderer=r)
                    else:
                        n = delta.notes_target_vs_current(pr["target"], pr["before"], pr["after"], notes, renderer=r,
                                                          next_subsystem=i["next_subsystem"])
                    s = {**s, "notes": n}
                    print(f"  {s['slot']}: rebuilt", flush=True)
                out.write(json.dumps(s, ensure_ascii=False, default=_jsonable) + "\n")
    finally:
        pool.close()


def _jsonable(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    return str(x)


def narrate_one(slot: dict, url: str, model: str | None, arm: str) -> dict:
    n, mode = slot["notes"], slot["mode"]
    try:
        if mode == "first_impression":
            from maestro.sounddiff.first_impression import narrate_first_impression
            out = narrate_first_impression(n["audio"], n, url=url, model=model)
            out["arm"] = "omni"
        else:
            from maestro.sounddiff import delta
            fn = delta.narrate_self_edit if mode == "self_edit" else delta.narrate_target_vs_current
            out = fn(n, url=url, model=model, arm=arm)
    except Exception as e:
        return {"slot": slot["slot"], "mode": mode, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    return {"slot": slot["slot"], "mode": mode, "text": out["text"], "problems": out["problems"],
            "attempts": len(out["attempts"]), "arm": out.get("arm"), "url": url, "model": model}


def narrate(run: Path, modes: list[str], url: str, model: str | None, arm: str, workers: int) -> None:
    done = {r["slot"] for r in jsonl(run / "narrations.jsonl") if "error" not in r}
    todo = [s for s in jsonl(run / "narration_slots.jsonl") if s["mode"] in modes and s["slot"] not in done]
    print(f"{len(todo)} slots to narrate ({', '.join(modes)}) via {url} arm={arm}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as ex, open(run / "narrations.jsonl", "a") as f:
        for r in ex.map(lambda s: narrate_one(s, url, model, arm), todo):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            f.flush()
            state = r.get("error") or (f"problems {r['problems']}" if r["problems"] else "ok")
            print(f"  {r['slot']}: {state}", flush=True)


def finalize(run: Path) -> dict:
    slots = {s["slot"]: s for s in jsonl(run / "narration_slots.jsonl")}
    narr: dict[str, dict] = {}
    for r in jsonl(run / "narrations.jsonl"):   # last result per slot wins
        narr[r["slot"]] = r
    report = {m: {"llm": 0, "template": 0, "reasons": {}, "chars": []} for m in MODES}

    def text_for(slot_id: str) -> str:
        s, r = slots[slot_id], narr.get(slot_id)
        rep = report[s["mode"]]
        if r and "error" not in r and not r["problems"] and r["text"].strip():
            rep["llm"] += 1
            rep["chars"].append(len(r["text"].strip()))
            return r["text"].strip()
        why = "not narrated" if r is None else r.get("error", "").split(":")[0] or "checker problems after retry"
        rep["template"] += 1
        rep["reasons"][why] = rep["reasons"].get(why, 0) + 1
        return s["template"]

    pending = run / "main.pending.jsonl"
    if not pending.exists():
        (run / "main.jsonl").rename(pending)
    mains = []
    for rec in jsonl(pending):
        for m in rec["messages"]:
            if m["role"] == "assistant" and "⟦narr:" in m["content"]:
                m["content"] = TOKEN_RE.sub(lambda mo: text_for(mo.group(1)), m["content"])
        mains.append(rec)
    (run / "main.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mains))

    subs = {r["id"]: r for f in ("transcriber", "searcher", "judge") for r in jsonl(run / f"{f}.jsonl")}
    index = [r for r in jsonl(run / "index.jsonl") if "main_id" in r]
    by_id = {r["id"]: r for r in mains}
    errors = {}
    for row in index:
        errs = validate.validate_rollout(by_id[row["main_id"]], [subs[l["record_id"]] for l in row["links"]],
                                         row["links"])
        errors[row["sample_id"]] = errs
        viewer.write_page(run, row["sample_id"])
    for rep in report.values():
        c = rep.pop("chars")
        rep["median_chars"] = int(statistics.median(c)) if c else 0
    report["validation"] = {"valid": sum(not e for e in errors.values()), "rollouts": len(errors),
                            "errors": {k: v[:5] for k, v in errors.items() if v}}
    report["main_median_chars"] = int(statistics.median(sum(len(m["content"]) for m in r["messages"])
                                                        for r in mains)) if mains else 0
    (run / "narration_report.json").write_text(json.dumps(report, indent=1))
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--modes", help=f"comma-separated subset of {', '.join(MODES)}")
    ap.add_argument("--url")
    ap.add_argument("--model")
    ap.add_argument("--arm", choices=("omni", "text", "audio"), default="text", help="edit modes only")
    ap.add_argument("--workers", type=int, default=4, help="concurrent narrator requests")
    ap.add_argument("--finalize", action="store_true")
    ap.add_argument("--rebuild-notes", action="store_true")
    ap.add_argument("--rebuild-modes", default=",".join(MODES), help="modes whose notes --rebuild-notes recomputes")
    a = ap.parse_args()
    if a.rebuild_notes:
        rebuild_notes(a.run, a.rebuild_modes.split(","))
    if a.modes:
        modes = a.modes.split(",")
        assert set(modes) <= set(MODES), modes
        assert a.url, "--url is required to narrate"
        narrate(a.run, modes, a.url, a.model, a.arm, a.workers)
    if a.finalize:
        print(json.dumps(finalize(a.run), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
