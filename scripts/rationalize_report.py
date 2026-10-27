"""Summary table + reading page for hearing_prompt_lab rationalize runs (one dir per reasoning effort).

  .venv/bin/python scripts/rationalize_report.py rat_low rat_medium rat_xhigh
  -> outputs/thinking_capture/hearing_lab/view/index.html
"""
from __future__ import annotations

import glob
import html
import json
import re
import statistics
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[1] / "outputs/thinking_capture/hearing_lab"
NUM = re.compile(r"\d+(\.\d+)?\s?(ms|hz|khz|%|db|st|s\b)|\b\d+\.\d+\b", re.I)


def exact_values(rec) -> list[str]:
    """Numbers in the guessed thinking that also appear verbatim in the record turn's tool call (copied, not heard)."""
    call = rec["turn"].split("[tool call]", 1)[-1] if "[tool call]" in rec["turn"] else ""
    return sorted({m.group(0) for m in NUM.finditer(rec["guessed_thinking"]) if m.group(0) in call})


def load(tag):
    return [json.load(open(f)) for f in sorted(glob.glob(str(LAB / tag / "*.json")))]


def main():
    tags = sys.argv[1:]
    rows, cards = [], {}
    for tag in tags:
        rs = [r for r in load(tag) if "guessed_thinking" in r]
        n = len(rs)
        ch = [len(r["guessed_thinking"]) for r in rs]
        ex = [r for r in rs if exact_values(r)]
        rows.append([tag, n, f"{sum(r['match_with'] for r in rs)}/{n}", f"{sum(r['match_without'] for r in rs)}/{n}",
                     sum(not r["guessed_thinking"].strip() for r in rs), sum(bool(r["leaks"]) for r in rs), len(ex),
                     int(statistics.median(ch)) if ch else 0, max(ch) if ch else 0])
        for r in rs:
            cards.setdefault((r["sid"], r["point"]), {})[tag] = r
    head = ["effort", "points", "replay with thinking matches record", "replay without thinking matches", "empty",
            "leaks", "copies exact values from the call", "median chars", "max chars"]
    for row in rows:
        print("  ".join(f"{h}={v}" for h, v in zip(head, row)))
    e = html.escape
    css = """:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--ok:#2e8b57;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#34332f}}
body{background:var(--bg);color:var(--fg);font:14px/1.45 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1500px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}
.cols{display:grid;grid-template-columns:repeat(%d,minmax(0,1fr));gap:10px}@media(max-width:900px){.cols{grid-template-columns:1fr}}
pre{white-space:pre-wrap;font-size:12.5px;margin:4px 0;max-height:380px;overflow:auto;background:var(--bg);padding:6px;border-radius:6px;border:1px solid var(--line)}
.mut{color:var(--mut);font-size:12.5px}.ok{color:var(--ok)}.bad{color:var(--bad)}
table{border-collapse:collapse;font-size:13px}td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}
.tbl{overflow-x:auto}""" % max(len(tags), 1)
    body = ["<h1>Guessed thinking, by reasoning effort</h1><p class=mut>The 27B saw the session (audio replaced by what "
            "was heard) plus the record's next turn, and wrote the thinking that leads to it. Replay test: the same "
            "context with only that thinking, and the turn hidden. Does the model take the record's action again? "
            "The control replays with empty thinking.</p><div class=tbl><table><tr>"
            + "".join(f"<th>{e(h)}</th>" for h in head) + "</tr>"
            + "".join("<tr>" + "".join(f"<td>{e(str(v))}</td>" for v in row) + "</tr>" for row in rows) + "</table></div>"]
    for (sid, k), by in sorted(cards.items()):
        any_r = next(iter(by.values()))
        body.append(f"<div class=card><b>{sid} listen {k}</b><details><summary class=mut>record's next turn</summary>"
                    f"<pre>{e(any_r['turn'])}</pre></details><div class=cols>")
        for tag in tags:
            r = by.get(tag)
            if not r:
                body.append(f"<div><b>{e(tag)}</b><p class=mut>missing</p></div>")
                continue
            ok = "<span class=ok>replay matches</span>" if r["match_with"] else "<span class=bad>replay differs</span>"
            ex = exact_values(r)
            body.append(f"<div><b>{e(tag)}</b> {ok} <span class=mut>({len(r['guessed_thinking'])} chars"
                        f"{'; copies ' + e(', '.join(ex)) if ex else ''})</span><pre>{e(r['guessed_thinking']) or '(empty)'}</pre></div>")
        body.append("</div></div>")
    (LAB / "view").mkdir(exist_ok=True)
    (LAB / "view" / "index.html").write_text(
        f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Guessed thinking</title><style>{css}</style></head><body>{''.join(body)}</body></html>")
    print(LAB / "view" / "index.html")


if __name__ == "__main__":
    main()
