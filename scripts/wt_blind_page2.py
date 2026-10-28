"""Blind test v2: does the superposition ranking's "close" match your ears? Fresh POC targets (first frame, no
cross-mod), rendered in the FULL target patch (effects, modulation and random phase as the patch has them). Options:
the GT tables (a separate take), the ranking's two closest other configurations (frame-0 duplicates of the GT skipped:
they are equally correct), and two from deep in the ranking. Reveal shows each option's tables and ranking score.

  .venv/bin/python scripts/wt_blind_page2.py --n 10   -> outputs/wt_rank/blind/v2.html (served with the v1 page)
"""
import argparse
import copy
import html
import random
import sys
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from wt_blind_page import CSS, JS  # noqa: E402

JS = JS.replace("const ok=+p===+c.dataset.ans; if(ok)right++;",   # answers are logged by the page's server
                "const ok=+p===+c.dataset.ans; if(ok)right++; fetch('answer?page=v2&q='+q+'&pick='+p).catch(()=>{});")
from maestro.sounddiff.renderer import ContainerRenderer  # noqa: E402
from maestro.synth.preset_gen import ARCHETYPES, generate_preset  # noqa: E402
from maestro.synth.wavetable_lib import load_wavetable_lib  # noqa: E402
from maestro.v5 import planner as pl, wt_rank as wr  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=10)
ap.add_argument("--seed", type=int, default=11)
a = ap.parse_args()
out = ROOT / "outputs/wt_rank/blind"
(out / "audio").mkdir(parents=True, exist_ok=True)
lib = load_wavetable_lib(ROOT / "data/wavetable_lib.json")
idx, byh = pl.vitallib.wt_index(), pl._wt_by_hash()
name = {e["id"]: e["name"] for e in idx}
rng = random.Random(a.seed)
cards = []
with ContainerRenderer(ROOT / "outputs/wt_rank/rank_cache", jobs=8) as r:
    r._session()
    qi = 0
    while qi < a.n:
        gt = generate_preset(rng.choice(ARCHETYPES), random.Random(rng.randrange(1 << 30)), wavetable_lib=lib)
        slots = wr.active_slots(gt)
        gts = [byh.get(pl._wt_hash(gt["settings"]["wavetables"][k - 1])) for k in slots]
        if None in gts:
            continue
        pos = next(i for i, x in enumerate(idx) if x["id"] == gts[0]["id"])
        cands = {e["id"]: e for e in idx[pos // 32 * 32: pos // 32 * 32 + 32]}
        cands.update({e["id"]: e for e in gts})
        ranked = [x for x in wr.rank_subsets(r, gt, list(cands.values())) if None not in x.tables]
        want = tuple(e["id"] for e in gts)
        alts = [x for x in ranked if not all(wr.same_table(t, w) for t, w in zip(x.tables, want))
                and not wr.sounds_like_target(x.score_db)]   # sound-identical in this patch = the GT too
        if len(alts) < 10:
            continue
        near = alts[:2]
        deep = rng.sample(alts[len(alts) // 2:], 2)

        def full(tables):
            p = copy.deepcopy(gt)
            for k, t in zip(slots, tables):
                p["settings"]["wavetables"][k - 1] = pl.vitallib.wt_load(cands[t])
            return p
        ys = r.render({"target": gt, "gt#2": gt, **{f"o{i}": full(x.tables) for i, x in enumerate(near + deep)}},
                      wr.PATTERN)
        opts = [("GT (another take)", want, 0.0, ys["gt#2"])] + \
               [("ranking's closest" if i < 2 else "deep in the ranking", x.tables, x.score_db, ys[f"o{i}"])
                for i, x in enumerate(near + deep)]
        rng.shuffle(opts)
        ans = next(i for i, o in enumerate(opts) if o[0].startswith("GT"))

        def save(y, fn):
            sf.write(out / "audio" / fn, y, 44100, subtype="PCM_16")
            return f"audio/{fn}"
        tgt = save(ys["target"], f"v2_q{qi}_target.wav")
        rows = []
        for i, (kind, tables, sc, y) in enumerate(opts):
            src = save(y, f"v2_q{qi}_{'ABCDE'[i]}.wav")
            desc = " + ".join(f"{html.escape(name[t])} ({t}, osc {k})" for k, t in zip(slots, tables))
            rows.append(f"<div class=opt onclick='pick({qi},{i})'><span class=lab>{'ABCDE'[i]}</span>"
                        f"<audio controls preload=none src='{src}'></audio>"
                        f"<span class='reveal mut'>{desc} — <b>{kind}</b>, ranking score {sc:.2f}</span></div>")
        cards.append(f"<div class=card id=q{qi} data-ans={ans}><b>Q{qi + 1}</b> <span class=mut>({len(slots)} "
                     f"oscillator{'s' if len(slots) > 1 else ''}; click a row to pick)</span>"
                     f"<div class=opt><span class=lab>T</span><audio controls preload=none src='{tgt}'></audio>"
                     f"<span class=mut>target</span></div><hr style='border:0;border-top:1px solid var(--line)'>"
                     f"{''.join(rows)}<p><button onclick='reveal({qi})'>Reveal</button> <span class=verdict></span></p>"
                     f"<div class='reveal mut'>Ranking: 0 = identical (stripped patch); the closest wrong options "
                     f"scored {near[0].score_db:.2f} and {near[1].score_db:.2f}, the deep ones "
                     f"{deep[0].score_db:.2f} and {deep[1].score_db:.2f}.</div></div>")
        print("q", qi, len(slots), "oscs", flush=True)
        qi += 1
(out / "v2.html").write_text(
    "<!doctype html><html><head><meta charset=utf-8><meta name=viewport content='width=device-width,initial-scale=1'>"
    f"<title>Wavetable blind test v2</title><style>{CSS}</style></head><body><h1>Blind test v2: superposition ranking</h1>"
    "<p class=mut>Fresh targets (first frame only, no cross-modulation), each played through its full patch. Exactly "
    "one option has the target's own tables (another take). Two wrong options are the ranking's closest, two come from "
    "deep in the ranking. If the ranking matches your ears, the closest ones should be the hard ones.</p>"
    f"<div id=score>0 / 0 correct</div>{''.join(cards)}<script>{JS}</script></body></html>")
print(out / "v2.html")
