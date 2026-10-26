#!/usr/bin/env python
"""Conformance replay (B6): run recorded rollouts through the real OpenCode 1.19.3 and diff every
request it sends against the records. See opencode/replay/README.md.

    # self-test: the golden capture conversation (must give 0 diffs)
    .venv/bin/python scripts/replay_in_opencode.py golden [--variant with_plugin] [--out outputs/replay/golden]

    # a rollout: main record + its subagent records (path.json, or path.jsonl#<id>)
    .venv/bin/python scripts/replay_in_opencode.py records MAIN SUB [SUB ...] --init-dir DIR \
        [--attach /work/rollouts/<sid>/target.wav] [--out outputs/replay/<name>]

    # a v5 builder output dir (outputs/v5_rollouts/<run>): one replay per sample, records from index.jsonl,
    # initial workdir from <run>/init/<sample>
    .venv/bin/python scripts/replay_in_opencode.py rollout outputs/v5_rollouts/smoke [--sample ID ...] \
        [--out outputs/replay/v5_smoke]

    # re-diff an existing run's request log (no OpenCode run)
    .venv/bin/python scripts/replay_in_opencode.py report outputs/replay/<name>

Exit status 0 iff every compared request matches (after the whitelist) and every step was requested.
CPU + Docker only: one throwaway daw-farm container per run (name prefix wsd-replay-).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from opencode.replay import audio_eq as AE  # noqa: E402
from opencode.replay import golden as G  # noqa: E402
from opencode.replay import harness as H  # noqa: E402


def load_record(spec: str) -> dict:
    path, _, rid = spec.partition("#")
    p = Path(path)
    if p.suffix == ".jsonl":
        recs = [json.loads(line) for line in p.open() if line.strip()]
        if rid:
            recs = [r for r in recs if r.get("id") == rid]
        if len(recs) != 1:
            raise SystemExit(f"{spec}: {len(recs)} records match (give path.jsonl#<id>)")
        return recs[0]
    return json.loads(p.read_text())


def summarize(rep: dict) -> int:
    print(f"requests {rep['requests']}  compared {rep['compared']}  match {rep['matches']}  diff {rep['diffs']}"
          f"  (opencode exit {rep.get('opencode_exit')}, {rep.get('elapsed_s')} s)")
    if rep.get("by_kind"):
        print("  by kind: " + ", ".join(f"{k} {v}" for k, v in sorted(rep["by_kind"].items())))
    for k, c in rep["coverage"].items():
        print(f"  {k}: {c['requested']}/{c['steps']} steps" + (f"  MISSING {c['missing']}" if c["missing"] else ""))
    for r in rep["rows"]:
        if r["status"] == "diff":
            print(f"  DIFF #{r['seq']} {r['route']} step {r['step']}: " + ", ".join(d["kind"] for d in r["diffs"]))
    if rep.get("workdir_check"):
        w = rep["workdir_check"]
        print(f"  final workdir: {w['files']} files, {w['identical']} identical, "
              f"{len(w['audio_rerendered'])} re-rendered audio ({len(w['audio_over_threshold'])} over threshold)")
    print(f"report: {rep.get('run_dir', '')}/report.md")
    return 0 if rep["diffs"] == 0 and rep["compared"] > 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("golden")
    g.add_argument("--variant", default="with_plugin", help="goldens subdir ('' = stock, no plugin)")
    g.add_argument("--out", default=str(REPO / "outputs/replay/golden"))
    g.add_argument("--port", type=int, default=8911)
    g.add_argument("--keep-container", action="store_true")
    r = sub.add_parser("records")
    r.add_argument("main")
    r.add_argument("subagents", nargs="*")
    r.add_argument("--init-dir", help="initial workdir contents (copied to /work/rollouts/<sid>)")
    r.add_argument("--attach", action="append", default=[], help="container path attached with -f")
    r.add_argument("--prompt", help="user text on stdin (default: main record's first user message)")
    r.add_argument("--base-dir", help="resolve relative `audios` paths against this dir")
    r.add_argument("--out", required=True)
    r.add_argument("--port", type=int, default=8911)
    r.add_argument("--timeout", type=int, default=1800)
    r.add_argument("--keep-container", action="store_true")
    r.add_argument("--max-mel-db", type=float, default=AE.DEFAULT_MAX_DB,
                   help="sanity cap for re-rendered clips (mean |log-mel diff|, dB; see opencode/replay/README.md)")
    ro = sub.add_parser("rollout")
    ro.add_argument("run")
    ro.add_argument("--sample", action="append", default=[], help="sample id (default: every sample in index)")
    ro.add_argument("--init-root", help="dir of <sample>/ initial workdirs (default: index init_dir, else <run>/init)")
    ro.add_argument("--out", help="default outputs/replay/v5_<run name>")
    ro.add_argument("--port", type=int, default=8911)
    ro.add_argument("--timeout", type=int, default=1800)
    ro.add_argument("--keep-container", action="store_true")
    ro.add_argument("--max-mel-db", type=float, default=AE.DEFAULT_MAX_DB)
    rp = sub.add_parser("report")
    rp.add_argument("run_dir")
    rp.add_argument("--max-mel-db", type=float, help="override the run's threshold")
    a = ap.parse_args()

    if a.cmd == "golden":
        out = Path(a.out).resolve()
        workdir = f"{H.ROLLOUT_ROOT}/golden"
        init = out.with_name(out.name + "_init")  # outside the run dir (replay() recreates that)
        shutil.rmtree(init, ignore_errors=True)
        G.make_workdir(init)
        recs = G.golden_records(workdir, init, a.port, a.variant)
        rep = H.replay(recs["maestro"], [recs[s] for s in G.SUBAGENTS], out, init_dir=init,
                       prompt=G.PROMPT, port=a.port, with_plugin=bool(a.variant),
                       keep_container=a.keep_container)
        return summarize(rep)
    if a.cmd == "records":
        main_rec = load_record(a.main)
        subs = [load_record(s) for s in a.subagents]
        rep = H.replay(main_rec, subs, a.out, init_dir=a.init_dir, prompt=a.prompt, attach=a.attach,
                       port=a.port, timeout_s=a.timeout, base_dir=a.base_dir, keep_container=a.keep_container,
                       max_mel_db=a.max_mel_db)
        return summarize(rep)
    if a.cmd == "rollout":
        run = Path(a.run).resolve()
        out = Path(a.out or REPO / "outputs/replay" / f"v5_{run.name}").resolve()
        recs = {}
        for agent in ("main", "transcriber", "searcher", "judge"):
            for line in (run / f"{agent}.jsonl").open():
                if line.strip():
                    r = json.loads(line)
                    recs[r["id"]] = r
        index = [json.loads(line) for line in (run / "index.jsonl").open() if line.strip()]

        def inputs_sig() -> dict:
            return {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(run.glob("*.jsonl"))}
        sig0 = inputs_sig()
        if a.sample:
            index = [e for e in index if e["sample_id"] in a.sample]
        init_root = Path(a.init_root) if a.init_root else run / "init"
        rc = 0
        for e in index:
            init = Path(e["init_dir"]) if e.get("init_dir") and not a.init_root else init_root / e["sample_id"]
            if not init.is_dir():
                raise SystemExit(f"{init}: missing initial workdir (WS-A writes <run>/init/<sample>)")
            subs = [recs[link["record_id"]] for link in e["links"]]
            finals = sorted(run.glob(f"_sessions/*/{e['sample_id']}"))
            print(f"== {e['sample_id']}: main + {len(subs)} subagent records")
            rep = H.replay(recs[e["main_id"]], subs, out / e["sample_id"], init_dir=init, port=a.port,
                           timeout_s=a.timeout, keep_container=a.keep_container, max_mel_db=a.max_mel_db,
                           final_dir=finals[0] if len(finals) == 1 else None)
            rc |= summarize(rep)
            if inputs_sig() != sig0:
                print(f"ERROR: {run} records changed during the replay (a rebuild in progress?); results invalid")
                return 2
        return rc
    run = Path(a.run_dir).resolve()
    records = json.loads((run / "io/records.json").read_text())
    meta_p = run / "io/meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    rep = H.build_report(records, run / "io/requests.jsonl", base_dir=meta.get("base_dir"),
                         init_shas=set(meta.get("init_shas", [])),
                         max_mel_db=a.max_mel_db if a.max_mel_db is not None else meta.get("max_mel_db", AE.DEFAULT_MAX_DB))
    rep["run_dir"] = str(run)
    return summarize(rep)


if __name__ == "__main__":
    sys.exit(main())
