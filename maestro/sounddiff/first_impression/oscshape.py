"""Head-to-head: how should the narrator 'see' an oscillator's waveform?

R0 name only | R1 waveform image | R2 waveform + harmonic spectrum image | R3 text harmonic profile.
Objective test: pairs of lone-oscillator renders; the 27B answers brighter / buzzier / more hollow / purer from each
representation; scored against measurements of the renders. Plus a one-line description per table per rep, for ears.

  .venv/bin/python oscshape.py quiz --pairs 24
  .venv/bin/python oscshape.py describe
  .venv/bin/python oscshape.py page
"""
import base64
import io
import itertools
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import requests  # noqa: E402
import soundfile as sf  # noqa: E402

O = Path("/home/nate/Documents/maestro-llm/outputs/sounddiff/oscshape")
URL = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen3.8-27b-uncensored"
F0, NH = 110.0, 32
TABLES = [l.split(" ", 1) for l in (O / "tables.txt").read_text().splitlines() if l.strip()]
NAME = dict(TABLES)


REPS = ("R1", "R2", "R3")  # R0 (name only) dropped: names are known to mislead
STRANGE = ["wt088", "wt182", "wt242", "wt100", "wt248", "wt128", "wt053", "wt001"]


def analyze(tid):
    y, sr = sf.read(O / f"{tid}.wav", dtype="float64")
    y = y.mean(1)
    seg = y[int(0.4 * sr):int(1.4 * sr)]
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=len(seg) * 4))
    hz = np.fft.rfftfreq(len(seg) * 4, 1 / sr)
    amps = np.array([spec[(hz > h * F0 - 3) & (hz < h * F0 + 3)].max() for h in range(1, NH + 1)])
    e = amps ** 2
    n = np.arange(1, NH + 1)
    # 3 cycles from a rising zero crossing, resampled onto a fixed grid
    per = sr / F0
    i0 = int(0.5 * sr)
    zc = next(i for i in range(i0, i0 + int(per) * 2) if seg is not None and y[i - 1] < 0 <= y[i])
    t = zc + np.arange(0, 3 * per, per / 256)
    cyc = np.interp(t, np.arange(len(y)), y)
    truth = {"brightness": float((n * e).sum() / e.sum()),
             "buzz": float(e[5:].sum() / e.sum()),
             "hollow": float(e[2::2].sum() / max(e[1:].sum(), 1e-20)),  # odd harmonics 3,5,7.. share of h>=2
             "purity": float(e[0] / e.sum())}
    db = 20 * np.log10(amps / amps.max() + 1e-9)
    return {"amps": amps, "db": db, "cycles": cyc / (np.abs(cyc).max() + 1e-12), "truth": truth}


def classic_sim(amps):
    n = np.arange(1, NH + 1)
    ideal = {"sine": (n == 1) * 1.0, "saw": 1 / n, "square": (n % 2 == 1) / n, "triangle": (n % 2 == 1) / n ** 2}
    a = amps / np.linalg.norm(amps)
    return {k: float(a @ (v / np.linalg.norm(v))) for k, v in ideal.items()}


def img_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=80, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def r1_image(an):
    fig, ax = plt.subplots(figsize=(5, 2))
    ax.plot(an["cycles"], lw=1.5)
    ax.set_title("waveform, 3 cycles")
    ax.set_xticks([]); ax.set_yticks([-1, 0, 1]); ax.axhline(0, color="gray", lw=0.5)
    return img_b64(fig)


def r2_image(an):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 2.2))
    a1.plot(an["cycles"], lw=1.5); a1.set_title("waveform, 3 cycles"); a1.set_xticks([]); a1.axhline(0, color="gray", lw=0.5)
    d = np.maximum(an["db"], -60)
    a2.bar(np.arange(1, NH + 1), d + 60, bottom=-60, color=["C1" if h % 2 else "C0" for h in range(1, NH + 1)])
    a2.set_ylim(-60, 2); a2.set_title("harmonics 1-32, dB vs strongest (orange = odd)")
    a2.set_xticks([1, 2, 3, 4, 6, 8, 12, 16, 24, 32])
    return img_b64(fig)


def r3_text(an):
    d = an["db"]
    t = an["truth"]
    sims = classic_sim(an["amps"])
    slope = np.polyfit(np.log2(np.arange(1, NH + 1)), np.maximum(d, -80), 1)[0]
    return ("harmonic levels in dB vs the strongest, harmonics 1-16: "
            + ", ".join(f"{h}:{max(d[h - 1], -60):.0f}" for h in range(1, 17))
            + f"; harmonics 17-32 average {np.mean(np.maximum(d[16:], -60)):.0f} dB"
            + f"; overall falloff about {slope:.1f} dB per octave of harmonic number"
            + f"; similarity to classic shapes (1 = identical): "
            + ", ".join(f"{k} {v:.2f}" for k, v in sims.items()))


def rep_parts(rep, tid, an, label):
    """OpenAI content parts describing one oscillator in representation `rep`."""
    head = f"Oscillator {label}:"
    if rep == "R0":
        return [{"type": "text", "text": f"{head} wavetable named '{NAME[tid]}'."}]
    if rep == "R3":
        return [{"type": "text", "text": f"{head} {r3_text(an)}"}]
    img = r1_image(an) if rep == "R1" else r2_image(an)
    return [{"type": "text", "text": head},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img}"}}]


QUIZ = """Both are single Vital oscillators playing a sustained A2 (110 Hz) with no filter, effects or unison.
Answer by ear-imagination from what you were given. For each question answer "A" or "B":
- brighter: more energy higher up the spectrum
- buzzier: more strong high harmonics (buzzy/raspy rather than smooth)
- hollow: more hollow / square- or clarinet-like (odd harmonics dominate over even)
- purer: closer to a pure sine
End your reply with one line of JSON: {"brighter": "A|B", "buzzier": "A|B", "hollow": "A|B", "purer": "A|B"}"""
QKEY = {"brighter": "brightness", "buzzier": "buzz", "hollow": "hollow", "purer": "purity"}


def call(content):
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}],
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    for attempt in range(60):  # ride out server restarts (up to ~15 min)
        try:
            r = requests.post(URL, json=body, timeout=900).json()
            return r["choices"][0]["message"]["content"].strip()
        except Exception as e:  # noqa: BLE001
            err = e
            time.sleep(15)
    raise RuntimeError(f"server unavailable: {err}")


def quiz(npairs):
    AN = {t: analyze(t) for t, _ in TABLES}
    rng = random.Random(0)
    pairs = rng.sample(list(itertools.combinations([t for t, _ in TABLES], 2)), npairs)
    jobs = []
    for a, b in pairs:
        if rng.random() < 0.5:
            a, b = b, a
        for rep in REPS:
            content = rep_parts(rep, a, AN[a], "A") + rep_parts(rep, b, AN[b], "B") + [{"type": "text", "text": QUIZ}]
            jobs.append((rep, a, b, content))
    with ThreadPoolExecutor(12) as ex:
        outs = list(ex.map(lambda j: call(j[3]), jobs))
    score = {r: [0, 0] for r in REPS}
    perq = {(r, q): [0, 0] for r in score for q in QKEY}
    rows = []
    for (rep, a, b, _), out in zip(jobs, outs):
        m = re.findall(r"\{[^{}]*\}", out)
        ans = json.loads(m[-1]) if m else {}
        for q, k in QKEY.items():
            ta, tb = AN[a]["truth"][k], AN[b]["truth"][k]
            if abs(ta - tb) / max(abs(ta), abs(tb), 1e-9) < 0.15:  # too close to call: skip
                continue
            want = "A" if ta > tb else "B"
            ok = ans.get(q) == want
            score[rep][0] += ok; score[rep][1] += 1
            perq[(rep, q)][0] += ok; perq[(rep, q)][1] += 1
        rows.append({"rep": rep, "a": a, "b": b, "answer": ans, "raw_tail": out[-300:]})
    (O / "quiz.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    print("accuracy per representation (pairs where the measured difference is >15%):")
    for r, (k, n) in score.items():
        print(f"  {r}: {k}/{n} = {k / max(n, 1):.0%}   " +
              "  ".join(f"{q} {perq[(r, q)][0]}/{perq[(r, q)][1]}" for q in QKEY))


DESC = """This is a single Vital oscillator playing a sustained A2 (110 Hz) with no filter, effects or unison.
In one short sentence, describe what it sounds like the way a producer would (e.g. "a hollow, reedy square-ish tone"). Don't mention numbers."""


def describe():
    AN = {t: analyze(t) for t, _ in TABLES}
    jobs = [(rep, t) for t in STRANGE for rep in REPS]
    with ThreadPoolExecutor(12) as ex:
        outs = list(ex.map(lambda j: call(rep_parts(j[0], j[1], AN[j[1]], "")
                                          + [{"type": "text", "text": DESC}]), jobs))
    res = {}
    for (rep, t), o in zip(jobs, outs):
        res.setdefault(t, {})[rep] = o.strip().splitlines()[-1] if o.strip() else ""
    (O / "describe.json").write_text(json.dumps(res, indent=1))
    for t, d in res.items():
        print(f"== {t} {NAME[t]}")
        for rep, txt in d.items():
            print(f"   {rep}: {txt}")


def page():
    AN = {t: analyze(t) for t, _ in TABLES}
    res = json.loads((O / "describe.json").read_text())
    order = list(REPS)
    rows = []
    for t, name in TABLES:
        if t not in res:
            continue
        d = res.get(t, {})
        rnd = random.Random(t)
        reps = order[:]
        rnd.shuffle(reps)
        lis = "".join(f"<li>{d.get(r, '')} <span class=who>[{r}]</span></li>" for r in reps)
        rows.append(f"<div class=card><h3>{t} <span class=who>{name}</span></h3><audio controls preload=none src='{t}.wav'>"
                    f"</audio><br><img src='data:image/png;base64,{r2_image(AN[t])}'><ol>{lis}</ol></div>")
    css = ("body{font:15px system-ui;max-width:1000px;margin:auto;padding:16px;background:#fafaf8;color:#1d1d1b}"
           ".card{background:#fff;border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:12px 0}"
           ".who{display:none;color:#888;font-size:12px}body.reveal .who{display:inline}img{max-width:100%}"
           "@media (prefers-color-scheme:dark){body{background:#161615;color:#ecebe6}.card{background:#1f1f1d;border-color:#333}}")
    html = (f"<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>Oscillator shape test</title><style>{css}</style><h1>Oscillator descriptions</h1>"
            "<p>Each oscillator alone (A2, no filter/fx/unison). Four descriptions, one per representation, shuffled. "
            "<button onclick=\"document.body.classList.toggle('reveal')\">reveal names / reps</button></p>"
            + "".join(rows))
    (O / "index.html").write_text(html)
    print(O / "index.html")


if __name__ == "__main__":
    {"quiz": lambda: quiz(int(sys.argv[sys.argv.index("--pairs") + 1]) if "--pairs" in sys.argv else 24),
     "describe": describe, "page": page}[sys.argv[1]]()
