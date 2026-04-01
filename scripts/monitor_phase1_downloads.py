#!/usr/bin/env python3
"""
Monitor long-running Phase 1 dataset downloads until completion.

It samples file sizes periodically, estimates ETA from recent speed,
writes JSONL snapshots, and exits when all targets reach expected bytes.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib as pl
import time
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class Target:
    name: str
    path: str
    total_bytes: int


DEFAULT_TARGETS = [
    Target(
        name="musdb18_hq",
        path="data/raw/phase1_sources/musdb18_hq/musdb18hq.zip",
        total_bytes=22_656_664_047,
    ),
    Target(
        name="slakh2100",
        path="data/raw/phase1_sources/slakh2100/slakh2100_flac_redux.tar.gz",
        total_bytes=104_322_767_708,
    ),
]


def _read_size(path: str) -> int:
    p = pl.Path(path)
    if not p.exists():
        return 0
    return p.stat().st_size


def _fmt_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    if seconds < 0:
        return None
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}"


def _snapshot(
    targets: List[Target],
    previous: Dict[str, Dict[str, float]],
    interval_s: float,
) -> Dict[str, object]:
    now = time.time()
    rows = []
    all_done = True

    for target in targets:
        size = _read_size(target.path)
        total = target.total_bytes
        done = size >= total > 0
        all_done = all_done and done
        pct = (100.0 * size / total) if total else 0.0

        prev = previous.get(target.name)
        speed_bps = None
        eta_s = None
        if prev is not None:
            delta_bytes = max(0, size - int(prev["size"]))
            delta_t = max(interval_s, now - prev["ts"])
            speed_bps = float(delta_bytes) / float(delta_t) if delta_t > 0 else 0.0
            if speed_bps > 1e-6 and size < total:
                eta_s = float(total - size) / speed_bps

        previous[target.name] = {"size": float(size), "ts": now}

        rows.append(
            {
                "name": target.name,
                "path": str(pl.Path(target.path).resolve()),
                "size_bytes": size,
                "total_bytes": total,
                "percent": round(pct, 4),
                "done": done,
                "speed_bps": speed_bps,
                "eta_s": eta_s,
                "eta_hms": _fmt_duration(eta_s),
            }
        )

    return {
        "timestamp": int(now),
        "iso_time": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(now)),
        "all_done": all_done,
        "targets": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Monitor phase1 download progress until completion.")
    parser.add_argument(
        "--interval-sec",
        type=int,
        default=300,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--log-path",
        type=str,
        default="data/raw/phase1_sources/logs/download_monitor.jsonl",
        help="JSONL status log path.",
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default="data/raw/phase1_sources/logs/download_monitor_latest.json",
        help="Latest status snapshot path.",
    )
    args = parser.parse_args()

    interval_s = max(5, int(args.interval_sec))
    log_path = pl.Path(args.log_path)
    summary_path = pl.Path(args.summary_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    previous: Dict[str, Dict[str, float]] = {}
    while True:
        snap = _snapshot(DEFAULT_TARGETS, previous, interval_s=float(interval_s))

        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snap, ensure_ascii=True) + "\n")
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2)

        print(
            f"[{snap['iso_time']}] "
            + " ".join(
                f"{row['name']}={row['percent']:.2f}%"
                for row in snap["targets"]  # type: ignore[index]
            )
        )

        if bool(snap["all_done"]):
            break
        time.sleep(interval_s)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
