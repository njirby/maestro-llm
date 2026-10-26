"""Omni Instruct vs Omni Thinking: first impression of a target synth clip, side by side.

  .venv/bin/python omni_fi.py run instruct     # with Instruct served on :8000
  .venv/bin/python omni_fi.py run thinking     # with Thinking served on :8000
  .venv/bin/python omni_fi.py page             # -> outputs/sounddiff/omni_fi/index.html
"""
import base64
import html
import io
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import soundfile as sf

REPO = Path("/home/nate/Documents/maestro-llm")
P1 = REPO / "outputs/sounddiff/probe1"
O = REPO / "outputs/sounddiff/omni_fi"
URL = "http://localhost:8000/v1/chat/completions"
IDS = ["f004", "f005", "f008", "f009", "f012", "f019", "f022", "f027"]
if __import__("os").environ.get("ALL"):
    IDS = [f"f{i:03d}" for i in range(40)]
SAMPLES = 2
USE_PROMPT = True
TAGS = ("instruct_p", "instruct_notes4")  # page shows these (no-prompt runs: instruct, thinking)
ITEMS = {json.loads(l)["id"]: json.loads(l) for l in open(P1 / "items.jsonl")}

PROMPT = ("This clip is a single synth patch playing a part. Give your first impression the way a producer would to a "
          "collaborator, in 2-4 sentences: the register and whether it's a single line or chords, how "
          "the notes behave (attack, decay, any wobble or pulsing), any space or effects (reverb, delay, chorus, "
          "distortion), and the core tone in general terms (bright or dark, pure or rich, hollow, buzzy, noisy, thin or "
          "thick). Hedge where you're unsure. Don't guess specific preset or wavetable names.")


def audio_b64(iid):
    y, sr = sf.read(P1 / "wav" / f"{ITEMS[iid]['wavs']['target']}.wav", dtype="float32")
    y = y.mean(1) if y.ndim > 1 else y
    y = y / max(np.abs(y).max(), 1e-6) * 0.9  # same peak level for every clip
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode(), y, sr


def call(model, b64):
    # audio only, no text prompt and no sampling params: see what the model volunteers
    parts = [{"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}}]
    if USE_PROMPT:
        parts.append({"type": "text", "text": PROMPT})
    body = {"model": model, "messages": [{"role": "user", "content": parts}]}
    for _ in range(40):
        try:
            r = requests.post(URL, json=body, timeout=900)
        except requests.ConnectionError as e:  # server restarting: wait; any other failure is raised
            err = e
            time.sleep(15)
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:500]}")
        if True:
            m = r.json()["choices"][0]["message"]
            txt = m.get("content") or ""
            rsn = m.get("reasoning") or m.get("reasoning_content") or ""
            if "</think>" in txt:  # no reasoning parser on the server: split ourselves
                rsn, txt = txt.split("</think>", 1)
            return txt.strip(), rsn.replace("<think>", "").strip()
    raise RuntimeError(err)


def run(tag):
    model = requests.get(URL.replace("chat/completions", "models")).json()["data"][0]["id"]
    O.mkdir(parents=True, exist_ok=True)
    keep = set(NOTES) if NOTES else set(IDS)
    jobs = [(iid, k) for iid in IDS if iid in keep for k in range(SAMPLES)]
    b64 = {iid: audio_b64(iid)[0] for iid, _ in jobs}
    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(lambda j: call(model, b64[j[0]]), jobs))
    with open(O / f"{tag}.jsonl", "w") as f:
        for (iid, k), (txt, rsn) in zip(jobs, outs):
            f.write(json.dumps({"id": iid, "k": k, "tag": tag, "model": model, "text": txt,
                                "reasoning_chars": len(rsn)}) + "\n")
            print(f"--- {iid} [{tag} #{k}] {txt}\n")
    print(f"{len(jobs)} in {time.time() - t0:.0f}s")


def truth(it):
    n = it["notes"]
    name = lambda p: "C C# D D# E F F# G G# A A# B".split()[p % 12] + str(p // 12 - 1)
    poly = any(a["start_s"] < b["start_s"] + b["dur_s"] and b["start_s"] < a["start_s"] + a["dur_s"]
               for i, a in enumerate(n) for b in n[i + 1:])
    s = it.get("summary_target") or {}
    fx = ", ".join(f"{k} {v}" for k, v in (s.get("fx") or {}).items()) or "none"
    env = s.get("amp_envelope", {})
    return (f"{it['bpm']:.0f} BPM; {len(n)} notes, {name(min(x['pitch'] for x in n))}-{name(max(x['pitch'] for x in n))}; "
            f"{'chords / overlapping notes' if poly else 'single line'}; avg note {sum(x['dur_s'] for x in n) / len(n):.2f} s"
            f"<br>amp env: attack {env.get('attack')}, decay {env.get('decay')}, sustain {env.get('sustain')}, "
            f"release {env.get('release')}<br>effects: {fx}<br>modulation: "
            + html.escape("; ".join((s.get("modulation") or [])[:6]) or "none"))


NOTES = {k: v["text"] for k, v in json.loads((REPO / "outputs/sounddiff/fi_notes/notes.json").read_text()).items()} \
    if (REPO / "outputs/sounddiff/fi_notes/notes.json").exists() else {}


def page():
    rows = []
    res = {}
    for tag in TAGS:
        p = O / f"{tag}.jsonl"
        if p.exists():
            for r in map(json.loads, open(p)):
                res.setdefault(r["id"], []).append(r)
    for iid in [i for i in IDS if not NOTES or i in NOTES]:
        _, y, sr = audio_b64(iid)
        sf.write(O / f"{iid}.wav", y, sr, subtype="PCM_16")
        rs = res.get(iid, [])
        random.Random(iid).shuffle(rs)
        lis = "".join(
            f"<div class=n><p>{html.escape(r['text'])} <span class=who>— {r['tag']} #{r['k']}</span></p>"
            f"<div class=r data-k='{iid}|{r['tag']}|{r['k']}'><button data-v=true>true</button>"
            f"<button data-v=partly>partly</button><button data-v=false>false</button></div></div>" for r in rs)
        rows.append(f"<div class=card><h3>{iid}</h3><audio controls preload=none src='{iid}.wav'></audio>{lis}"
                    f"<details><summary>ground truth (MIDI + patch)</summary><p class=mut>{truth(ITEMS[iid])}</p>"
                    f"<p class=mut><b>session notes given to the notes arm:</b><br>{NOTES.get(iid, '').replace(chr(10), '<br>')}</p></details></div>")
    css = (":root{--bg:#fafaf8;--fg:#1d1d1b;--card:#fff;--line:#e3e2dc;--mut:#6b6b66}"
           "@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--card:#1f1f1d;--line:#34332f;--mut:#9a9992}}"
           "body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui;max-width:900px;margin:auto;padding:16px}"
           ".card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 16px;margin:14px 0}"
           ".n{border-top:1px solid var(--line);padding:6px 0}.mut{color:var(--mut);font-size:13px}"
           ".r button{border:1px solid var(--line);background:var(--card);color:var(--fg);border-radius:6px;padding:2px 10px;margin-right:4px}"
           ".r button.on[data-v=true]{background:#2e8b57;color:#fff}.r button.on[data-v=partly]{background:#c48a00;color:#fff}"
           ".r button.on[data-v=false]{background:#c0392b;color:#fff}.who{display:none;color:var(--mut);font-size:12px}"
           "body.reveal .who{display:inline}audio{width:100%}")
    js = ("const K='omni-fi-notes4';let R={};try{R=JSON.parse(localStorage.getItem(K)||'{}')}catch(e){}"
          "function paint(){document.querySelectorAll('.r').forEach(r=>r.querySelectorAll('button').forEach(b=>"
          "b.classList.toggle('on',R[r.dataset.k]===b.dataset.v)))}"
          "document.addEventListener('click',e=>{const b=e.target.closest('.r button');if(!b)return;const k=b.parentNode.dataset.k;"
          "R[k]=R[k]===b.dataset.v?undefined:b.dataset.v;try{localStorage.setItem(K,JSON.stringify(R))}catch(e){};paint()});"
          "function dl(){const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(R,null,1)]));"
          "a.download='omni_fi_ratings.json';a.click()}paint();")
    doc = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
           f"<title>Omni first impressions</title><style>{css}</style><h1>Omni Instruct vs Thinking: first impressions</h1>"
           "<p class=mut>Two arms, model hidden: Omni-Instruct with the audio-only prompt, and Omni-Instruct with audio + verified session notes (notes shown under ground truth).</p><p class=mut>Each clip: 4 descriptions (2 per model), shuffled, model hidden. Rate each; open ground truth "
           "to check tempo/notes/effects. <button onclick=\"document.body.classList.toggle('reveal')\">reveal models</button> "
           "<button onclick=dl()>export ratings</button></p>" + "".join(rows) + f"<script>{js}</script>")
    (O / "index.html").write_text(doc)
    print(O / "index.html")


if __name__ == "__main__":
    {"run": lambda: run(sys.argv[2]), "page": page}[sys.argv[1]]()
