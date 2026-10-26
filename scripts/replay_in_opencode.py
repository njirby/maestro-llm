#!/usr/bin/env python
"""Conformance replay (B6): run recorded rollouts through the real OpenCode 1.19.3 and diff every
request it sends against the records. See opencode/replay/README.md.

    # self-test: the golden capture conversation (must give 0 diffs)
    .venv/bin/python scripts/replay_in_opencode.py golden [--variant with_plugin] [--out outputs/replay/golden]

    # a rollout: main record + its subagent records (path.json, or path.jsonl#<id>)
    .venv/bin/python scripts/replay_in_opencode.py records MAIN SUB [SUB ...] --init-dir DIR \
        [--attach /work/rollouts/<sid>/target.wav] [--out outputs/replay/<name>]

    # re-diff an existing run's request log (no OpenCode run)
    .venv/bin/python scripts/replay_in_opencode.py report outputs/replay/<name>

Exit status 0 iff every compared request matches (after the whitelist) and every step was requested.
CPU + Docker only: one throwaway daw-farm container per run (name prefix wsd-replay-).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

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
    r.add_argument("--max-mel-db", type=float, default=1.5,
                   help="audio-equivalence threshold for re-rendered clips (mean |log-mel diff|, dB)")
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
    run = Path(a.run_dir).resolve()
    records = json.loads((run / "io/records.json").read_text())
    meta_p = run / "io/meta.json"
    meta = json.loads(meta_p.read_text()) if meta_p.is_file() else {}
    rep = H.build_report(records, run / "io/requests.jsonl", base_dir=meta.get("base_dir"),
                         init_shas=set(meta.get("init_shas", [])),
                         max_mel_db=a.max_mel_db if a.max_mel_db is not None else meta.get("max_mel_db", 1.5))
    rep["run_dir"] = str(run)
    return summarize(rep)


if __name__ == "__main__":
    sys.exit(main())
