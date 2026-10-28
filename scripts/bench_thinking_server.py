"""Benchmark the 27B server config on the real thinking workload: the same fixed set of thinking turns (one-call B2 +
the replay check, add_thinking_v5.attempt_native) from a finished run, at a fixed concurrency. Reports turns/min,
pass rate and the server's generation/prompt throughput. Run once per server config with a different --tag.

  .venv/bin/python scripts/bench_thinking_server.py --run outputs/v5_rollouts/pilot20e \
      --replay outputs/replay/v5_pilot20e --sids s0000 s0001 s0002 --n 96 --workers 48 --tag base
"""
import argparse
import concurrent.futures as cf
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import add_thinking_v5 as at  # noqa: E402
import hearing_prompt_lab as h  # noqa: E402
import native_thinking_lab as nl  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run", type=Path, required=True)
ap.add_argument("--replay", type=Path, required=True)
ap.add_argument("--sids", nargs="+", required=True)
ap.add_argument("--n", type=int, default=96)
ap.add_argument("--workers", type=int, default=48)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--tag", required=True)
ap.add_argument("--url", default="http://127.0.0.1:8001/v1")
ap.add_argument("--model", default="qwen3.8-27b-uncensored")
ap.add_argument("--effort", default="medium")
a = ap.parse_args()
a.no_verify = False   # attempt_native reads it: the replay check stays on, as in the pipeline
h.RUN, h.REPLAY = nl.RUN, nl.REPLAY = a.run, a.replay

jobs = []
for sid in a.sids:
    rows = [json.loads(l) for l in open(a.replay / sid / "io" / "requests.jsonl")]
    by = {}
    for r in rows:
        if r.get("kind") == "record":
            by.setdefault(r["route"], []).append(r)
    jobs += [(sid, route, k, r, rs[k + 1]) for route, rs in by.items() for k, r in enumerate(rs[:-1])]
jobs = random.Random(a.seed).sample(jobs, min(a.n, len(jobs)))


def one(job):
    sid, route, k, r, nxt = job
    t = time.time()
    try:
        x = at.attempt_native(at.for_server(a, sid, route), sid, route, r, nxt, a.run)
    except Exception as e:  # counted as a failure
        return {"ok": False, "err": str(e)[:200], "s": time.time() - t}
    return {"ok": not x["fails"], "fails": x["fails"], "s": time.time() - t, "chars": len(x["thinking"])}


t0 = time.time()
with cf.ThreadPoolExecutor(a.workers) as ex:
    res = list(ex.map(one, jobs))
wall = time.time() - t0
out = {"tag": a.tag, "turns": len(res), "wall_s": round(wall, 1), "turns_per_min": round(len(res) / wall * 60, 2),
       "pass": sum(r["ok"] for r in res), "errors": sum("err" in r for r in res),
       "median_turn_s": sorted(r["s"] for r in res)[len(res) // 2]}
print(json.dumps(out))
dst = Path("outputs/thinking_bench"); dst.mkdir(parents=True, exist_ok=True)
(dst / f"{a.tag}.json").write_text(json.dumps({**out, "results": res}, indent=1))
