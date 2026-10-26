"""Local listening page for a narration probe: per item the mode, players (target / before / after as the mode needs),
the facts the narrator saw, expected / surprise tags, every narrator's text (blind: tags hidden, order shuffled), the
text-only prediction, and true / partly / false ratings exportable as JSON. Summary stats at the top.

  .venv/bin/python -m maestro.sounddiff.report outputs/sounddiff/probe1   # -> outputs/sounddiff/probe1/report/index.html
"""
from __future__ import annotations

import glob
import html
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from .narrate import load_probe

CSS = """
:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--acc:#2f6fdb;--ok:#2e8b57;--mid:#c48a00;--bad:#c0392b}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#34332f;--acc:#6ea0ff}}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0;padding:16px;max-width:1100px;margin:auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px;margin:14px 0}
h1{font-size:20px} h2{font-size:16px;margin:0 0 6px} .mut{color:var(--mut);font-size:13px}
pre{white-space:pre-wrap;font-size:12.5px;background:var(--bg);padding:8px;border-radius:6px;border:1px solid var(--line);overflow-x:auto}
.n{border-top:1px solid var(--line);padding:8px 0} .n p{margin:2px 0 6px}
.r button,.flt button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:2px 10px;margin-right:4px;cursor:pointer}
.flt button.on{background:var(--acc);color:#fff}
.r button.on[data-v=true]{background:var(--ok);color:#fff}.r button.on[data-v=partly]{background:var(--mid);color:#fff}
.r button.on[data-v=false]{background:var(--bad);color:#fff}
.pl{display:flex;flex-wrap:wrap;gap:6px 14px;align-items:center;margin:6px 0} .pl span{font-size:13px;color:var(--mut)}
audio{height:32px;max-width:100%} .tag{display:inline-block;font-size:12px;padding:1px 7px;border-radius:9px;border:1px solid var(--line);margin:1px 2px}
.tag.exp{border-color:var(--ok);color:var(--ok)} .tag.sur{border-color:var(--bad);color:var(--bad)} .tag.mode{background:var(--acc);color:#fff;border-color:var(--acc)}
.who{display:none;font-size:12px;color:var(--mut)} body.reveal .who{display:inline} .pred{font-style:italic}
table{border-collapse:collapse;font-size:13px} td,th{border:1px solid var(--line);padding:3px 8px;text-align:left}
.tbl{overflow-x:auto}
"""

JS = """
const KEY='sounddiff-ratings-'+document.title;let R={};try{R=JSON.parse(localStorage.getItem(KEY)||'{}')}catch(e){}
function paint(){document.querySelectorAll('.r').forEach(r=>{r.querySelectorAll('button').forEach(b=>
 b.classList.toggle('on',R[r.dataset.k]===b.dataset.v))})}
document.addEventListener('click',e=>{const b=e.target.closest('.r button');if(!b)return;const k=b.parentNode.dataset.k;
 R[k]=R[k]===b.dataset.v?undefined:b.dataset.v;try{localStorage.setItem(KEY,JSON.stringify(R))}catch(e){};paint()});
function dl(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(R,null,1)],{type:'application/json'}));
 a.download='ratings.json';a.click()}
function flt(m){document.querySelectorAll('.flt button').forEach(b=>b.classList.toggle('on',b.dataset.m===m));
 document.querySelectorAll('.item').forEach(c=>c.style.display=(m==='all'||c.dataset.mode===m)?'':'none')}
paint();
"""

ROLES = {"target_vs_current": [("target", "target"), ("before", "yours, before the edit"), ("after", "yours, after the edit")],
         "self_edit_delta": [("before", "before"), ("after", "after")],
         "first_impression": [("target", "the patch")]}


def _write_audio(d, rep, item):
    names = [item["wavs"][r] for r, _ in ROLES[item["mode"]]]
    xs = [sf.read(d / "wav" / f"{n}.wav", dtype="float32")[0] for n in names]
    g = min(0.95 / max(max(np.abs(x).max() for x in xs), 1e-6), 4.0)  # one gain per item: keeps loudness differences
    for n, x in zip(names, xs):
        p = rep / "aud" / f"{n}.ogg"
        if not p.exists():
            sf.write(p, np.clip(x * g, -1, 1), 44100, format="OGG", subtype="VORBIS")


def _tags(f):
    t = [f"<span class='tag mode'>{f['mode'].replace('_', ' ')}</span>"]
    if f["mode"] != "first_impression":
        t.append(f"<span class=tag>audibility: {f['audibility']}</span>")
    if f["mode"] == "target_vs_current":
        t.append(f"<span class=tag>edit: {f['edit_audibility']}</span><span class=tag>trajectory: {f['trajectory']}</span>")
    if f["mode"] == "self_edit_delta":
        for c in f["changes"]:
            cls = "exp" if c["expected"] else "sur"
            t.append(f"<span class='tag {cls}'>{html.escape(c['dim'])} {'expected' if c['expected'] else 'SURPRISE'}</span>")
        for m in f["missing"]:
            t.append(f"<span class='tag sur'>{html.escape(m)}: expected, not heard</span>")
    return "".join(t)


def _stats_html(d):
    out = []
    p = d / "stats.json"
    if p.exists():
        s = json.loads(p.read_text())
        out.append("<div class=tbl><table><tr>" + "".join(f"<th>{html.escape(k)}</th>" for k in s["columns"]) + "</tr>"
                   + "".join("<tr>" + "".join(f"<td>{html.escape(str(v))}</td>" for v in row) + "</tr>" for row in s["rows"])
                   + "</table></div>")
    p = d / "predictability_summary.json"
    if p.exists():
        s = json.loads(p.read_text())
        out.append("<p class=mut>Predictability (text-only prediction vs grounded narration): " + "; ".join(
            f"{m}: beyond text {v['beyond_text']}, prediction wrong vs facts {v['pred_wrong_vs_facts']}" for m, v in s.items())
                   + "</p>")
    return "".join(out)


def main():
    d = Path(sys.argv[1])
    rep = d / "report"
    (rep / "aud").mkdir(parents=True, exist_ok=True)
    _, items = load_probe(d)
    narr = {}
    for p in sorted(glob.glob(str(d / "narr_*.jsonl"))):
        if p.endswith("_r1.jsonl"):  # earlier rounds are kept for reference, not shown
            continue
        for r in map(json.loads, open(p)):
            narr.setdefault((r["id"], r["mode"]), []).append(r)
    preds = {r["id"]: r for r in map(json.loads, open(d / "predictability.jsonl"))} \
        if (d / "predictability.jsonl").exists() else {}
    rng = random.Random(0)
    modes = ["all", "target_vs_current", "self_edit_delta", "first_impression"]
    body = [f"<h1>Narration probe: {html.escape(d.name)}</h1>",
            "<p class=mut>Listen, read the facts the narrator saw, and rate each narration true / partly / false "
            "(narrator names hidden, order shuffled). The italic line is a text-only prediction written without any "
            "measurement: if it says the same as the grounded narration, that item teaches little about listening. "
            "Ratings stay in this browser. <button onclick=dl()>export ratings.json</button> "
            "<button onclick=\"document.body.classList.toggle('reveal')\">reveal narrators</button></p>",
            _stats_html(d),
            "<div class=flt>" + "".join(f"<button data-m={m} onclick=\"flt('{m}')\" class={'on' if m == 'all' else ''}>"
                                        f"{m.replace('_', ' ')}</button>" for m in modes) + "</div>"]
    for it, f in items:
        _write_audio(d, rep, it)
        pl = "".join(f"<span>{lab}</span><audio controls preload=none src='aud/{it['wavs'][r]}.ogg'></audio>"
                     for r, lab in ROLES[it["mode"]])
        extra = html.escape(", ".join(it.get("moves", [])) or (
            f"{it['plan']['edit_kind']} {it['plan']['edit_subsystem']}" if "plan" in it else it["archetype"]))
        body.append(f"<div class='card item' data-mode={it['mode']}><h2>{it['id']} {_tags(f)} "
                    f"<span class=mut>{extra}</span></h2><div class=pl>{pl}</div>")
        if it["mode"] != "first_impression":
            body.append(f"<div class=r data-k='{it['id']}|audibility'>audibility verdict right? "
                        f"<button data-v=true>yes</button><button data-v=false>no</button></div>")
        body.append(f"<details><summary class=mut>facts the narrator saw</summary><pre>{html.escape(f['text'])}</pre></details>")
        rs = narr.get((it["id"], it["mode"]), [])
        rng.shuffle(rs)
        for r in rs:
            k = f"{it['id']}|{it['mode']}|{r['tag']}"
            flag = " (template fallback)" if r.get("fallback") else ""
            body.append(f"<div class=n><p>{html.escape(r['text'])} <span class=who>— {r['tag']}{flag}</span></p>"
                        f"<div class=r data-k='{k}'><button data-v=true>true</button><button data-v=partly>partly"
                        f"</button><button data-v=false>false</button></div></div>")
        if it["id"] in preds:
            pr = preds[it["id"]]
            body.append(f"<div class=n><p class='pred mut'>text-only prediction: {html.escape(pr['prediction'])} "
                        f"<span class=tag>{'beyond text' if pr['beyond_text'] else 'predictable'}</span></p></div>")
        body.append("</div>")
    page = (f"<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>sounddiff {html.escape(d.name)}</title><style>{CSS}</style></head><body>{''.join(body)}"
            f"<script>{JS}</script></body></html>")
    (rep / "index.html").write_text(page)
    print(rep / "index.html")


if __name__ == "__main__":
    main()
