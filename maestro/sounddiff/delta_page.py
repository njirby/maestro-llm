"""Blind rating page for one comparison mode: clips + every arm's narrations (shuffled, model hidden), true / partly /
false per narration, ratings kept in the browser and exported as JSON. Written to outputs/sounddiff/delta/<mode>/
(index.html + mp3/), served with `python3 -m http.server <port> --bind 0.0.0.0` from that dir."""
from __future__ import annotations

import html
import json
import random
import subprocess

from .delta import CLIPS
from .delta_probe import OUT, P1, selected

ARMS = ("omni", "text", "audio")
TITLE = {"self_edit": "Self-edit narration", "target_vs_current": "Target-vs-current narration"}
ABOUT = {"self_edit": "Each item: the patch before and after one edit (same part). Rate each description of what "
                      "changed against what you hear.",
         "target_vs_current": "Each item: the target, the agent's patch before its last edit, and its patch now. Rate "
                              "each think-out-loud line (did it get closer, what's still different, next move) "
                              "against what you hear."}


def _mp3(src: str, dst) -> None:
    if not dst.exists():
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "2", "-b:a", "160k", str(dst)],
                       check=True)


def page(mode: str) -> None:
    d = OUT / mode
    notes = json.loads((d / "notes.json").read_text())
    res = {}
    for arm in ARMS:
        p = d / f"arm_{arm}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                res.setdefault(r["id"], []).append(r)
    items = {it["id"]: it for it in map(json.loads, open(P1 / "items.jsonl"))}
    (d / "mp3").mkdir(exist_ok=True)
    cards = []
    keep = set(selected(mode))
    for iid, n in notes.items():
        if iid not in keep or iid not in res:
            continue
        clips = []
        for lab, key in CLIPS[mode]:
            _mp3(n["audio"][key], d / "mp3" / f"{iid}_{key}.mp3")
            clips.append(f"<div class=clip><span>{html.escape(lab.rstrip(':'))}</span>"
                         f"<audio controls preload=none src='mp3/{iid}_{key}.mp3'></audio></div>")
        rs = res[iid]
        random.Random(iid).shuffle(rs)
        lis = "".join(
            f"<div class=n><p>{html.escape(r['text'])}</p><p class=who>{r['arm']} #{r['k']}"
            f"{' · checker: ' + html.escape('; '.join(r['problems'])) if r['problems'] else ''}</p>"
            f"<div class=r data-k='{iid}|{r['arm']}|{r['k']}'><button data-v=true>true</button>"
            f"<button data-v=partly>partly</button><button data-v=false>false</button></div></div>" for r in rs)
        it = items[iid]
        edit = "; ".join(f"{e['param']}: {e['before']} → {e['after']}" + ("" if e["live"] else " (dead)")
                         for e in it["edit"])
        cards.append(f"<section class=card><h2>{iid}</h2>{''.join(clips)}{lis}"
                     f"<details><summary>session notes (given to two of the three arms) and the edit</summary>"
                     f"<pre>{html.escape(n['text'])}</pre><p class=mut>edit: {html.escape(edit)}</p></details>"
                     f"</section>")
    css = (":root{--bg:#fafaf8;--fg:#1d1d1b;--card:#fff;--line:#e3e2dc;--mut:#6b6b66;--ok:#2e7d4f;--mid:#a86f00;"
           "--bad:#b23a2e;--on:#fff}"
           "@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#161615;--fg:#ecebe6;--card:#1f1f1d;"
           "--line:#34332f;--mut:#9a9992;--ok:#4caf7a;--mid:#d9a227;--bad:#e06454;--on:#111}}"
           "*{box-sizing:border-box}body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;"
           "max-width:860px;margin:auto;padding:16px}h1{font-size:22px;margin:8px 0}h2{font-size:16px;margin:0 0 8px}"
           ".card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0}"
           ".clip{display:flex;gap:10px;align-items:center;margin:4px 0}.clip span{min-width:12em;font-size:13px;"
           "color:var(--mut)}audio{flex:1;min-width:0;height:36px}"
           ".n{border-top:1px solid var(--line);padding:6px 0}.n p{margin:4px 0}.mut{color:var(--mut);font-size:13px}"
           "pre{white-space:pre-wrap;font-size:12.5px;color:var(--mut)}"
           ".r button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;"
           "padding:3px 12px;margin-right:6px;font:inherit;cursor:pointer}"
           ".r button.on[data-v=true]{background:var(--ok);color:var(--on)}.r button.on[data-v=partly]"
           "{background:var(--mid);color:var(--on)}.r button.on[data-v=false]{background:var(--bad);color:var(--on)}"
           ".who{display:none;color:var(--mut);font-size:12px}body.reveal .who{display:block}"
           ".bar{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}"
           "@media (max-width:560px){.clip{flex-direction:column;align-items:stretch}.clip span{min-width:0}}")
    key = f"sounddiff-{mode}-v1"
    js = (f"const K='{key}';let R={{}};try{{R=JSON.parse(localStorage.getItem(K)||'{{}}')}}catch(e){{}}"
          "function paint(){let n=0;document.querySelectorAll('.r').forEach(r=>{if(R[r.dataset.k])n++;"
          "r.querySelectorAll('button').forEach(b=>b.classList.toggle('on',R[r.dataset.k]===b.dataset.v))});"
          "document.getElementById('cnt').textContent=n+' / '+document.querySelectorAll('.r').length+' rated'}"
          "document.addEventListener('click',e=>{const b=e.target.closest('.r button');if(!b)return;"
          "const k=b.parentNode.dataset.k;if(R[k]===b.dataset.v)delete R[k];else R[k]=b.dataset.v;"
          "try{localStorage.setItem(K,JSON.stringify(R))}catch(e){};paint()});"
          "function dl(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob("
          f"[JSON.stringify(R,null,1)],{{type:'application/json'}}));a.download='{mode}_ratings.json';a.click()}}"
          "paint();")
    doc = (f"<!doctype html><html lang=en><meta charset=utf-8><meta name=viewport "
           f"content='width=device-width,initial-scale=1'><title>{TITLE[mode]}</title><style>{css}</style><body>"
           f"<h1>{TITLE[mode]}: blind rating</h1><p class=mut>{ABOUT[mode]} Descriptions are shuffled and the "
           f"model is hidden: three arms x 2 samples each. Ratings stay in this browser; export when done.</p>"
           "<div class=bar><span id=cnt class=mut></span> "
           "<button onclick=\"document.body.classList.toggle('reveal')\">reveal arms</button> "
           "<button onclick=dl()>export ratings</button></div>" + "".join(cards) + f"<script>{js}</script></body></html>")
    (d / "index.html").write_text(doc)
    print(d / "index.html", len(cards), "items")
