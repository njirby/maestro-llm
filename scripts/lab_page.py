"""Spot-check page for a hearing_prompt_lab lab2 run: each guessed thinking large, with flags, the record's turn, what
was heard at that point, and the replay from the thinking alone.

  .venv/bin/python scripts/lab_page.py sub_g3 "Subagent turns" -> outputs/thinking_capture/hearing_lab/view/sub_g3.html
"""
import glob
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hearing_prompt_lab as h  # noqa: E402

tag, title = sys.argv[1], sys.argv[2]
L = h.OUT
rs = [json.load(open(f)) for f in sorted(glob.glob(str(L / tag / "*.json")))]
e = html.escape
css = """:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--ok:#2e8b57;--bad:#c0392b;--acc:#2f6fdb}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#34332f;--acc:#6ea0ff}}
body{background:var(--bg);color:var(--fg);font:14px/1.5 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:1100px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:14px 0}
pre{white-space:pre-wrap;font-size:12.5px;margin:4px 0;max-height:360px;overflow:auto;background:var(--bg);padding:6px;border-radius:6px;border:1px solid var(--line)}
.think{font-size:14.5px;line-height:1.55;border-left:3px solid var(--acc);padding:4px 10px;margin:8px 0}
.mut{color:var(--mut);font-size:12.5px}.ok{color:var(--ok)}.bad{color:var(--bad)}
.tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);margin:0 3px}
nav button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:2px 9px;margin:2px;cursor:pointer}"""
groups = sorted({r["kind"].split(".")[0] for r in rs})
body = [f"<h1>{e(title)}</h1><p class=mut>{len(rs)} turns. The thinking was guessed by the 27B (medium effort) from what the "
        f"agent had heard and seen plus the record's turn. Replay with only the thinking matches the record's action "
        f"{sum(r['match_with'] for r in rs)}/{len(rs)} (empty thinking: {sum(r['match_without'] for r in rs)}/{len(rs)}).</p>",
        "<nav>" + "".join(f"<button onclick=\"document.querySelectorAll('.card').forEach(c=>c.style.display=('{g}'=='all'||c.dataset.g=='{g}')?'':'none')\">{g}</button>"
                          for g in ["all"] + groups) + "</nav>"]
for r in rs:
    flags = []
    if not r["match_with"]:
        flags.append(f"replay differs: {', '.join(r['with_subs'])}")
    if h.LEAK.search(r["guessed_thinking"]):
        flags.append("leak")
    if r.get("quotes"):
        flags.append("quotes hearing: " + "; ".join(r["quotes"])[:80])
    if r.get("claims_hearing_before_any_audio"):
        flags.append("hearing claim before any audio (check)")
    flags += [f"claim: {c[:60]}" for c in r.get("claim_problems", [])]
    ok = "<span class='tag ok'>replay matches</span>" if r["match_with"] else ""
    heard = ""
    # what was heard most recently in this turn's context
    ctx_last = None
    try:
        rows = [json.loads(l) for l in open(h.REPLAY / r["sid"] / "io" / "requests.jsonl")]
    except FileNotFoundError:
        rows = []
    body.append(f"<div class=card data-g={r['kind'].split('.')[0]}><b>{r['sid']}</b> <span class=tag>{e(r['kind'])}</span>{ok}"
                + "".join(f"<span class='tag bad'>{e(f)}</span>" for f in flags)
                + f" <span class=mut>{len(r['guessed_thinking'])} chars</span><div class=think>{e(r['guessed_thinking']) or '(empty)'}</div>"
                f"<details><summary class=mut>record's turn (what the thinking must lead to)</summary><pre>{e(r['turn'])}</pre></details>"
                f"<details><summary class=mut>replay from the thinking alone</summary><pre>{e(r['replay_with']['text'][:1500])}\n"
                f"{e(json.dumps(r['replay_with']['calls'])[:800])}</pre></details></div>")
(L / "view").mkdir(exist_ok=True)
(L / "view" / f"{tag}.html").write_text(
    f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
    f"<title>{e(title)}</title><style>{css}</style></head><body>{''.join(body)}</body></html>")
print(L / "view" / f"{tag}.html")
