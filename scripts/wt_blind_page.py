"""Blind listening test: can you pick the GT osc-1 wavetable out of 5 in-context alternatives?

Per sample: the target (GT patch), and 5 options = the GT patch with osc 1's table swapped: the GT table (a separate
take, so it is not a copy of the target), the 2 tables measured closest to the target in context, and 2 random others
from the same 32-table chunk. Shuffled; the answer, each option's measured distance and the noise floor (target vs a
second take of itself) are revealed per question.

  .venv/bin/python scripts/wt_blind_page.py --sids s0001 s0002 ...  -> outputs/wt_rank/blind/index.html
"""
from __future__ import annotations

import argparse
import html
import json
import random
import shutil
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import wt_rank_test as t  # noqa: E402
from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.sounddiff.verified import mel_d  # noqa: E402

CSS = """:root{--bg:#fafaf8;--fg:#1d1d1b;--mut:#6b6b66;--card:#fff;--line:#e3e2dc;--ok:#2e8b57;--bad:#c0392b;--sel:#eef3ff}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ecebe6;--mut:#9a9992;--card:#1f1f1d;--line:#34332f;--sel:#1d2740}}
body{background:var(--bg);color:var(--fg);font:15px/1.45 system-ui,sans-serif;margin:0 auto;padding:16px;max-width:900px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px;margin:16px 0}
.opt{display:flex;align-items:center;gap:10px;flex-wrap:wrap;padding:6px 8px;border-radius:8px;border:1px solid transparent}
.opt.sel{background:var(--sel);border-color:var(--line)}
audio{width:100%;max-width:420px}
button{font:inherit;padding:6px 14px;border-radius:8px;border:1px solid var(--line);background:var(--card);color:var(--fg);cursor:pointer}
.mut{color:var(--mut);font-size:13px}.ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}
.reveal{display:none;margin-top:8px}.lab{min-width:22px;font-weight:600}
#score{position:sticky;top:0;background:var(--bg);padding:8px 0;border-bottom:1px solid var(--line)}"""

JS = """let right=0,done=0;
function pick(q,i){document.querySelectorAll('#q'+q+' .opt[onclick]').forEach((e,j)=>e.classList.toggle('sel',j===i));
 document.getElementById('q'+q).dataset.pick=i;}
function reveal(q){const c=document.getElementById('q'+q); if(c.dataset.done)return;
 const p=c.dataset.pick; if(p===undefined){alert('pick one first');return;}
 c.dataset.done=1; done++; const ok=+p===+c.dataset.ans; if(ok)right++;
 c.querySelector('.verdict').innerHTML=ok?'<span class=ok>correct</span>':'<span class=bad>wrong</span> — it was '+'ABCDE'[c.dataset.ans];
 c.querySelector('.reveal').style.display='block';
 document.getElementById('score').textContent=right+' / '+done+' correct ('+document.querySelectorAll('.card[id]').length+' questions)';}"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, default=ROOT / "outputs/v5_rollouts/pilot20d")
    ap.add_argument("--sids", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/wt_rank/blind")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    idx, byh = t.vl.wt_index(), t.pl._wt_by_hash()
    if a.out.exists():
        shutil.rmtree(a.out)
    (a.out / "audio").mkdir(parents=True)
    cards = []
    with ContainerRenderer(ROOT / "outputs/wt_rank/render_cache", jobs=8) as r:
        r._session()
        for qi, sid in enumerate(a.sids):
            d = a.run / "oracle" / sid
            gt = json.loads((d / "gt_preset.json").read_text())
            notes = json.loads((d / "notes.json").read_text())
            e = byh[t.pl._wt_hash(gt["settings"]["wavetables"][0])]
            pos = next(i for i, x in enumerate(idx) if x["id"] == e["id"])
            lo = pos // 32 * 32
            chunk = idx[lo:lo + 32]
            ref = r.render({"t": gt, "t#2": gt, "t#3": gt}, notes)
            floor = mel_d(ref["t"], ref["t#3"])
            ctx = r.render({x["id"]: t.probe(x, gt, "in_context") for x in chunk}, notes)
            dist = {k: mel_d(ref["t"], y) for k, y in ctx.items()}
            others = sorted((k for k in dist if k != e["id"]), key=dist.get)
            wrong = others[:2] + rng.sample(others[2:], 2)
            opts = [("gt", e["id"], ref["t#2"], mel_d(ref["t"], ref["t#2"]))] + [
                ("near" if i < 2 else "random", k, ctx[k], dist[k]) for i, k in enumerate(wrong)]
            rng.shuffle(opts)
            ans = next(i for i, o in enumerate(opts) if o[0] == "gt")
            n_close = sum(v <= 2 * max(floor, 0.3) for k, v in dist.items() if k != e["id"])

            def save(y, name):
                sf.write(a.out / "audio" / name, y, 44100, subtype="PCM_16")
                return f"audio/{name}"
            tgt = save(ref["t"], f"q{qi}_target.wav")
            rows = []
            for i, (kind, wid, y, dv) in enumerate(opts):
                src = save(y, f"q{qi}_{'ABCDE'[i]}.wav")
                name = next(x["name"] for x in idx if x["id"] == wid)
                rows.append(f"<div class=opt onclick='pick({qi},{i})'><span class=lab>{'ABCDE'[i]}</span>"
                            f"<audio controls preload=none src='{src}'></audio>"
                            f"<span class='reveal mut'>{html.escape(name)} ({wid}) — "
                            f"{'<b>GT</b> (another take)' if kind == 'gt' else kind}, distance {dv:.2f} dB</span></div>")
            cards.append(
                f"<div class=card id=q{qi} data-ans={ans}><b>Q{qi + 1}</b> <span class=mut>(click a row to pick)</span>"
                f"<div class=opt><span class=lab>T</span><audio controls preload=none src='{tgt}'></audio>"
                f"<span class=mut>target</span></div><hr style='border:0;border-top:1px solid var(--line)'>"
                f"{''.join(rows)}<p><button onclick='reveal({qi})'>Reveal</button> <span class=verdict></span></p>"
                f"<div class='reveal mut'>{sid}. Noise floor (target vs another take of itself): {floor:.2f} dB. "
                f"{n_close} of the chunk's 31 other tables measure within 2x the floor in this patch.</div></div>")
            print(sid, "done", flush=True)
    (a.out / "index.html").write_text(
        "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<title>Wavetable blind test</title><style>{CSS}</style></head><body><h1>Which one has the right wavetable?</h1>"
        "<p class=mut>Each option is the target's full patch with oscillator 1's wavetable swapped. Exactly one uses the "
        "target's own table (a separate take: Vital's random phase makes two renders of the same patch differ "
        "slightly). Wrong options: the two tables measured closest, plus two random ones from the same 32-table "
        "chunk.</p><div id=score>0 / 0 correct</div>"
        f"{''.join(cards)}<script>{JS}</script></body></html>")
    print(a.out / "index.html")


if __name__ == "__main__":
    main()
