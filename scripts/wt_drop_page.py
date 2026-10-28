"""Listening page: can you hear one oscillator missing? For each drop in outputs/wt_rank/metric_check.json: A = the
target (stripped patch: what the metric compares), B = the same with one oscillator removed, or identical (controls).
You answer "different" / "same"; reveal shows the oscillator's energy share and each metric variant's cost.

  .venv/bin/python scripts/wt_drop_page.py   -> outputs/wt_rank/blind/drops.html
"""
import copy
import json
import random
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from wt_blind_page import CSS  # noqa: E402
from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.synth.preset_gen import ARCHETYPES, generate_preset  # noqa: E402
from maestro.synth.wavetable_lib import load_wavetable_lib  # noqa: E402
from maestro.v5 import planner as pl, wt_rank as wr  # noqa: E402

rows = json.loads((ROOT / "outputs/wt_rank/metric_check.json").read_text())
lib = load_wavetable_lib(ROOT / "data/wavetable_lib.json")
byh = pl._wt_by_hash()
targets, rng = [], random.Random(1)
while len(targets) < 8:   # same targets as wt_metric_check.py
    gt = generate_preset(rng.choice(ARCHETYPES), random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
    if None not in [byh.get(pl._wt_hash(gt["settings"]["wavetables"][k - 1])) for k in wr.active_slots(gt)]:
        targets.append(gt)
targets += [json.loads(f.read_text()) for f in sorted((ROOT / "outputs/v5_rollouts/probe_wtrank/inputs").glob("*/gt_preset.json"))]
out = ROOT / "outputs/wt_rank/blind"
rng = random.Random(3)
items = [dict(x, control=False) for x in rows]
items += [dict(t=t, osc=None, share=None, costs=None, control=True) for t in rng.sample(sorted({x["t"] for x in rows}), 4)]
rng.shuffle(items)
cards = []
with ContainerRenderer(ROOT / "outputs/wt_rank/rank_cache", jobs=8) as r:
    r._session()
    for qi, it in enumerate(items):
        base = wr.stripped(targets[it["t"]])
        b = copy.deepcopy(base)
        if not it["control"]:
            b["settings"][f"osc_{it['osc']}_on"] = 0.0
        ys = r.render({"a": base, "b": b}, wr.PATTERN)
        sf.write(out / "audio" / f"d{qi}_a.wav", ys["a"], 44100, subtype="PCM_16")
        sf.write(out / "audio" / f"d{qi}_b.wav", ys["b"], 44100, subtype="PCM_16")
        truth = "same" if it["control"] else "different"
        info = ("control: B is identical to A" if it["control"] else
                f"B drops osc {it['osc']}, {it['share']} dB of the target's energy. Metric cost by weighting p: " +
                ", ".join(f"p={p}: {c}" for p, c in it["costs"].items()))
        cards.append(f"<div class=card id=q{qi} data-truth={truth}><b>Pair {qi + 1}</b>"
                     f"<div class=opt><span class=lab>A</span><audio controls preload=none src='audio/d{qi}_a.wav'></audio></div>"
                     f"<div class=opt><span class=lab>B</span><audio controls preload=none src='audio/d{qi}_b.wav'></audio></div>"
                     f"<p><button onclick=\"ans({qi},'different')\">I hear a difference</button> "
                     f"<button onclick=\"ans({qi},'same')\">Sounds the same</button> <span class=verdict></span></p>"
                     f"<div class='reveal mut'>{info}</div></div>")
        print(qi, flush=True)
JS2 = """let n=0,ok=0;function ans(q,a){const c=document.getElementById('q'+q);if(c.dataset.done)return;c.dataset.done=1;n++;
const t=c.dataset.truth;if(a===t)ok++;fetch('answer?page=drops&q='+q+'&a='+a).catch(()=>{});c.querySelector('.verdict').innerHTML='you said <b>'+a+'</b>'+(t==='same'?' (control: identical)':'');
c.querySelector('.reveal').style.display='block';document.getElementById('score').textContent=n+' answered; controls and '+
'drops are revealed as you go';}"""
(out / "drops.html").write_text(
    "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
    f"<title>Missing oscillator test</title><style>{CSS}</style></head><body><h1>Can you hear one oscillator missing?</h1>"
    "<p class=mut>A is a target patch (effects and modulation stripped, as the metric hears it). B is the same patch "
    "with one oscillator removed, or, for a few controls, identical. Say whether you hear a difference; the reveal shows "
    "how much of the sound that oscillator carried and what each metric variant charged for dropping it.</p>"
    f"<div id=score>0 answered</div>{''.join(cards)}<script>{JS2}</script></body></html>")
print(out / "drops.html")
