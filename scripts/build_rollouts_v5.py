#!/usr/bin/env python
"""Builder v5: GT-driven synthetic rollouts in exact OpenCode v1.19.3 format, every tool call executed for real in
DawDreamer containers (see maestro/v5/__init__.py).

  .venv/bin/python scripts/build_rollouts_v5.py --n 1 --out outputs/v5_rollouts/smoke
  .venv/bin/python scripts/build_rollouts_v5.py --manifest outputs/v5_manifest/pilot/manifest.jsonl --n 20 \\
      --out outputs/v5_rollouts/pilot20 --workers 4

Without --manifest, small test inputs are generated into <out>/inputs (preset_gen GT + a Lakh clip; the target is
rendered in the session with the skill's render.py).

Outputs in --out: main.jsonl, transcriber.jsonl, searcher.jsonl, judge.jsonl (ms-swift {id, messages, audios, tools}),
index.jsonl (one row per rollout: sample, record ids, task session ids, workdir, init_dir, build events,
validation errors), init/<sample>/ (the workdir before the first step: WS-D replay --init-dir), run_meta.json,
audio/<sample>/ (every clip an agent listened to), view/<sample>.html, stats.json (chars per record type).

Target audio entry: the user message is text naming target.wav, which sits in the rollout workdir; the main agent
hears it through its first read (the maestro audio-read plugin), like every later listen. This is what the goldens
capture ("Recreate the sound in target.wav."); an audio file part on the first user message is not part of the
captured OpenCode contract.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from maestro.dawfarm_dd import DawDreamerPool, cleanup_stale  # noqa: E402
from maestro.v5 import narration, validate, viewer  # noqa: E402
from maestro.v5.inputs import load_manifest, make_test_samples  # noqa: E402
from maestro.v5.record import DATE  # noqa: E402
from maestro.v5.agents.main import MistakeMix  # noqa: E402
from maestro.v5.rollout import SkipSample, run_one  # noqa: E402
from scripts import opencode_contract as oc  # noqa: E402

AGENT_FILES = ("main", "transcriber", "searcher", "judge")
PREFIX = "v5a-"


def _jsonable(x):
    """numpy scalars/arrays, sets and paths in the verified-notes dicts."""
    if hasattr(x, "tolist"):
        return x.tolist()
    if isinstance(x, (set, frozenset, tuple)):
        return list(x)
    return str(x)


def chars(rec: dict) -> int:
    return sum(len(m["content"]) for m in rec["messages"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, help="WS-C manifest.jsonl (default: generate test inputs)")
    ap.add_argument("--workers", type=int, default=1, help="DawDreamer sessions in parallel")
    ap.add_argument("--session-prefix", default=PREFIX,
                    help="container name prefix; builds running side by side need different ones (stale cleanup "
                         "removes every container with this prefix)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--clean-share", type=float, default=0.35, help="main rollouts with no injected mistake")
    ap.add_argument("--param-share", type=float, default=0.40, help="rollouts with 1-3 misjudged param values")
    ap.add_argument("--code-share", type=float, default=0.20, help="rollouts with one broken edit/render command")
    ap.add_argument("--retranscribe-share", type=float, default=0.10,
                    help="rollouts where the main agent hears wrong notes late and sends the transcriber back")
    ap.add_argument("--transcriber-wrong", type=float, default=0.35, help="transcriber records with a wrong attempt")
    ap.add_argument("--start-patch-rate", type=float, default=0.0,
                    help="per rollout: the user hands a starting patch (partial or foreign, maestro/v5/start.py)")
    ap.add_argument("--no-view", action="store_true")
    ap.add_argument("--keep-gate-failures", action="store_true",
                    help="write rollouts that miss the GT gate (flagged in index.jsonl) instead of dropping them")
    ap.add_argument("--fi-url", help="first-impression narrator (Omni), e.g. http://localhost:8000/v1; unset = template")
    ap.add_argument("--fi-model")
    ap.add_argument("--edit-arm", choices=("omni", "text"), default="text",
                    help="self_edit / target_vs_current narrator arm")
    ap.add_argument("--edit-url", help="edit-mode narrator, e.g. http://localhost:8001/v1 (text arm); unset = template")
    ap.add_argument("--edit-model")
    ap.add_argument("--defer-narration", action="store_true",
                    help="two-pass: compute verified notes now, write <out>/narration_slots.jsonl and put slot tokens "
                         "in the records; then scripts/narrate_rollouts_v5.py narrates per mode and --finalize")
    a = ap.parse_args()

    out = a.out.resolve()
    slot_lock = threading.Lock()

    def slot_sink(slot: dict) -> None:
        with slot_lock, open(out / "narration_slots.jsonl", "a") as f:
            f.write(json.dumps(slot, ensure_ascii=False, default=_jsonable) + "\n")

    narration.configure(narration.NarrationConfig(fi_url=a.fi_url, fi_model=a.fi_model, edit_arm=a.edit_arm,
                                                  edit_url=a.edit_url, edit_model=a.edit_model,
                                                  defer=a.defer_narration, slot_sink=slot_sink))
    if any((out / f"{f}.jsonl").exists() for f in AGENT_FILES):
        sys.exit(f"{out} already holds records; pick a new --out")
    out.mkdir(parents=True, exist_ok=True)
    samples = (load_manifest(a.manifest)[:a.n] if a.manifest
               else make_test_samples(a.n, out / "inputs", seed=a.seed))
    lock = threading.Lock()
    files = {f: open(out / f"{f}.jsonl", "a") for f in AGENT_FILES}
    index = open(out / "index.jsonl", "a")
    sizes: dict[str, list[int]] = {f: [] for f in AGENT_FILES}
    gates: list[dict] = []
    n_ok = n_bad = n_gate_dropped = 0

    (out / "run_meta.json").write_text(json.dumps({
        "opencode_version": oc.OPENCODE_VERSION, "date": DATE, "skill_mount": oc.CONFIG_DIR + "/skills/vital",
        "data_dir": oc.DATA_DIR, "rollout_root": "/work/rollouts", "audio_base": str(out / "audio"),
        "audios": "absolute host paths (copies under audio_base/<sample>/)",
        "init_dirs": str(out / "init"), "seed": a.seed,
        "mistakes": {k: getattr(a, k) for k in ("clean_share", "param_share", "code_share", "retranscribe_share",
                                                "transcriber_wrong", "start_patch_rate")},
        "narration": {k: getattr(a, k) for k in ("fi_url", "fi_model", "edit_arm", "edit_url", "edit_model",
                                                 "defer_narration")}, "manifest": str(a.manifest) if a.manifest else None},
        indent=1))
    cleanup_stale(a.session_prefix)
    pool = DawDreamerPool.start(min(a.workers, len(samples)), prefix=a.session_prefix, host_root=out / "_sessions",
                                skill_mount=oc.CONFIG_DIR + "/skills/vital")

    def one(sample):
        with pool.acquire() as sess:
            return run_one(sess, sample, out, seed=a.seed, start_patch_rate=a.start_patch_rate,
                           mix=MistakeMix(clean=a.clean_share, param=a.param_share, code=a.code_share,
                                          retranscribe=a.retranscribe_share, transcriber_wrong=a.transcriber_wrong))

    try:
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            futs = {ex.submit(one, s): s for s in samples}
            for fut in as_completed(futs):
                s = futs[fut]
                try:
                    r = fut.result()
                except SkipSample as e:
                    print(f"skip {s.id}: {e}", flush=True)
                    continue
                except Exception:
                    n_bad += 1
                    print(f"FAILED {s.id}:\n{traceback.format_exc()}", flush=True)
                    with lock:
                        index.write(json.dumps({"sample_id": s.id, "error": traceback.format_exc()[-2000:]}) + "\n")
                    continue
                subs = r["transcriber"] + r["searcher"] + r["judge"]
                errs = validate.validate_rollout(r["main"], subs, r["links"])
                gate = r["gate"]
                gates.append({"sample_id": s.id, **gate})
                if not gate["pass"] and not a.keep_gate_failures:
                    n_gate_dropped += 1
                    print(f"{s.id}: GT gate FAILED {gate}; dropped", flush=True)
                    with lock:
                        index.write(json.dumps({"sample_id": s.id, "dropped": "gt_gate", "gate": gate,
                                                "events": r["events"], "sample": s.to_dict()}) + "\n")
                        index.flush()
                    continue
                n_ok += not errs
                with lock:
                    for f in AGENT_FILES:
                        for rec in ([r["main"]] if f == "main" else r[f]):
                            files[f].write(json.dumps(rec, ensure_ascii=False) + "\n")
                            sizes[f].append(chars(rec))
                    for fh in files.values():
                        fh.flush()
                    index.write(json.dumps({"sample_id": s.id, "main_id": r["main"]["id"], "links": r["links"],
                                                "mistakes": r.get("mistakes", {}),
                                            "workdir": f"/work/rollouts/{s.id}", "init_dir": r["init_dir"],
                                            "final_workdir": r["final_workdir"],
                                            "validation_errors": errs, "gate": gate, "events": r["events"],
                                            "sample": s.to_dict()}, ensure_ascii=False) + "\n")
                    index.flush()
                print(f"{s.id}: main {chars(r['main']):,} chars, {len(r['main']['audios'])} audios, "
                      f"{len(subs)} subagent records, validation {'OK' if not errs else errs[:5]}, GT gate "
                      f"{'pass' if gate['pass'] else 'FAIL'} (structural {gate['structural']['n_param_diffs']} param / "
                      f"{len(gate['structural']['route_diffs'])} route diffs; mel {gate['audio']['final_vs_gt_db']} "
                      f"vs floor {gate['audio']['gt_vs_gt_db']} dB; rms {gate['audio']['rms_diff_db']} dB)", flush=True)
                if not a.no_view:
                    viewer.write_page(out, s.id)
    finally:
        pool.close()
        for fh in [*files.values(), index]:
            fh.close()

    stats = {f: {"n": len(v), "median_chars": int(statistics.median(v)) if v else 0,
                 "mean_chars": int(statistics.mean(v)) if v else 0, "max_chars": max(v, default=0)}
             for f, v in sizes.items()}
    stats["rollouts"] = {"valid": n_ok, "failed": n_bad, "gate_dropped": n_gate_dropped, "requested": len(samples)}
    stats["gt_gate"] = {"pass": sum(g["pass"] for g in gates), "structural_pass": sum(g["structural"]["pass"] for g in gates),
                        "audio_pass": sum(g["audio"]["pass"] for g in gates), "n": len(gates), "rollouts": gates}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))
    return 0 if n_bad == 0 and n_gate_dropped == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
