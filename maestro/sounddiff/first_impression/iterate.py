"""Scratch loop: build the new-format narrator prompt for a few probe1 items and send it to the hosted 27B.

  /home/nate/Documents/maestro-llm/.venv/bin/python iterate.py e031 e002 [--show-prompt] [--n 2]

Edit system.txt / user.txt (templates) and re-run. No sampling params are sent (server defaults);
only reasoning_effort via chat_template_kwargs.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

REPO = Path("/home/nate/Documents/maestro-llm")
sys.path.insert(0, str(REPO))
from maestro.sounddiff.measure import pair  # noqa: E402
from maestro.sounddiff.vital_facts import human, why_dead  # noqa: E402

FX_KEYS = {
    "distortion": ["distortion_type", "distortion_drive", "distortion_mix"],
    "reverb": ["reverb_dry_wet", "reverb_decay_time", "reverb_size"],
    "delay": ["delay_dry_wet", "delay_feedback"],
    "chorus": ["chorus_dry_wet", "chorus_voices"],
    "phaser": ["phaser_dry_wet", "phaser_feedback"],
    "flanger": ["flanger_dry_wet", "flanger_feedback"],
    "compressor": ["compressor_mix"],
    "eq": ["eq_low_gain", "eq_band_gain", "eq_high_gain"],
}
DIST_TYPES = ["soft clip", "hard clip", "linear fold", "sine fold", "bit crush", "down sample"]


def fx_line(settings):
    parts = []
    for fx, keys in FX_KEYS.items():
        if settings.get(f"{fx}_on", 0) < 0.5:
            continue
        vals = []
        for k in keys:
            if k not in settings:
                continue
            v = settings[k]
            if k.startswith("eq_") and k.endswith("_gain"):
                vals.append(f"{k[3:-5]} {v:+.1f} dB")
            elif k == "distortion_type":
                vals.append(DIST_TYPES[int(v)] if int(v) < len(DIST_TYPES) else f"type {int(v)}")
            else:
                vals.append(f"{k.split('_', 1)[1].replace('_', ' ')} {human(k, v)}")
        parts.append(f"{fx} ({', '.join(vals)})")
    return "effects: " + ("; ".join(parts) or "none")

P1 = REPO / "outputs/sounddiff/probe1"
HERE = Path(__file__).parent
URL = "http://localhost:8001/v1/chat/completions"
MODEL = "qwen3.8-27b-uncensored"

ITEMS = {json.loads(l)["id"]: json.loads(l) for l in open(P1 / "items.jsonl")}
DESC = {json.loads(l)["name"]: json.loads(l)["desc"] for l in open(P1 / "desc.jsonl")}

# (label, key, unit, fmt)
ROWS = [
    ("loudness (RMS)", "rms_dbfs", "dBFS", "{:.1f}"),
    ("peak level", "peak_db", "dBFS", "{:.1f}"),
    ("brightness (spectral centroid)", "centroid_w_hz", "Hz", "{:.0f}"),
    ("85% of energy below", "rolloff85_hz", "Hz", "{:.0f}"),
    ("upper harmonics vs fundamental area (grit)", "harm_rich_db", "dB", "{:.1f}"),
    ("noisiness (spectral flatness, 0=pure tone)", "flatness", "", "{:.4f}"),
    ("stereo width (side vs mid)", "width_db", "dB", "{:.1f}"),
    ("attack time to peak", "attack_ms", "ms", "{:.0f}"),
    ("level late in each note vs its peak", "sustain_ratio", "", "{:.2f}"),
    ("tail after last note", "tail_s", "s", "{:.2f}"),
    ("sound between notes vs during notes", "gap_fill_db", "dB", "{:.1f}"),
    ("brightness movement within notes", "mod_cent", "", "{:.2f}"),
    ("volume movement within notes", "mod_amp", "dB", "{:.2f}"),
]
BANDS = [("sub <120 Hz", "sub"), ("low 120-500", "low"), ("mid 500-2k", "mid"), ("high-mid 2k-6k", "high_mid"),
         ("air >6k", "air")]


REF = [json.loads(l)["desc"] for l in open(REPO / "outputs/sounddiff/refset/desc.jsonl")]
VARIANT = __import__("os").environ.get("V", "2")   # 1 = raw numbers, 2 = anchored + gated


def pct(key, v, band=None):
    xs = [(r["bands"][band] if band else r[key]) for r in REF if (r["bands"][band] if band else r[key]) is not None]
    return round(100 * sum(x < v for x in xs) / len(xs))


def audible(d):
    return d["rms_dbfs"] > -50 and d["peak_db"] > -40


def note_name(p):
    return "C C# D D# E F F# G G# A A# B".split()[p % 12] + str(p // 12 - 1)


def melody(it):
    n = it["notes"]
    lo, hi = min(x["pitch"] for x in n), max(x["pitch"] for x in n)
    avg = sum(x["dur_s"] for x in n) / len(n)
    return f"{len(n)} notes, {note_name(lo)}-{note_name(hi)}, {it['bpm']:.0f} BPM, average note {avg:.2f} s"


def patch_dump(s, settings=None):
    out = []
    for o in s["oscillators"]:
        line = (f"osc {o['osc']}: wavetable '{o['wavetable']}', level {o['level']}, {o['transpose']}, {o['unison']}"
                + (f" (detune {o['unison_detune']})" if "unison_detune" in o else "")
                + (f", warp {o['warp']}" if "warp" in o else "") + f" -> {o['route']}")
        out.append(line)
    for f in s["filters"] or []:
        out.append(f"filter {f['filter']}: {f['model']} {f['style']}, cutoff {f['cutoff']}, resonance {f['resonance']}")
    if not s["filters"]:
        out.append("filters: none active")
    a = s["amp_envelope"]
    out.append(f"amp envelope: attack {a['attack']}, decay {a['decay']}, sustain {a['sustain']}, release {a['release']}")
    out.append(fx_line(settings) if settings is not None else
               "effects: " + (", ".join(f"{k} {v}" for k, v in s["fx"].items()) or "none"))
    if s["modulation"]:
        out.append("modulation: " + "; ".join(s["modulation"]))
    return "\n".join("  " + l for l in out)


def edit_lines(it):
    return "\n".join(f"  {d['param']}: {d['before']} -> {d['after']}" for d in it["edit"])


def measurements(a, a2, b, it=None):
    if VARIANT == "2":
        return measurements2(a, a2, b, it)
    A, A2, B = DESC[a], DESC[a2], DESC[b]
    lines = []
    for label, k, unit, fmt in ROWS:
        va, va2, vb = A[k], A2[k], B[k]
        noise = abs(va - va2)
        u = f" {unit}" if unit else ""
        lines.append(f"  {label}: {fmt.format(va)} -> {fmt.format(vb)}{u}   (noise ±{fmt.format(noise)})")
    lines.append("  energy by band, before -> after: " + ", ".join(
        f"{lab} {A['bands'][k] * 100:.0f}% -> {B['bands'][k] * 100:.0f}%" for lab, k in BANDS))
    d, fl = pair(P1, a, b), pair(P1, a, a2)
    lines.append(f"  overall difference (log-mel, dB): {d['mel_db']:.1f}   (noise {fl['mel_db']:.1f})")
    lines.append(f"  overall difference, audio-model embeddings (0=same): CLAP {d['clap']:.3f} (noise {fl['clap']:.3f}), "
                 f"Omni {d['omni_frames']:.3f} (noise {fl['omni_frames']:.3f})")
    return "\n".join(lines)


ROWS2 = [  # (label, key, unit, fmt, higher-means)
    ("brightness (spectral centre)", "centroid_w_hz", "Hz", "{:.0f}", "brighter"),
    ("upper harmonics (6th and up) vs harmonics 1-3", "harm_rich_db", "dB", "{:.1f}", "more bite / grit"),
    ("noisiness (spectral flatness)", "flatness", "", "{:.4f}", "noisier / breathier"),
    ("stereo width (side vs mid)", "width_db", "dB", "{:.1f}", "wider"),
    ("attack: time to reach each note's peak", "attack_ms", "ms", "{:.0f}", "slower, softer attack"),
    ("level late in each note vs its peak", "sustain_ratio", "", "{:.2f}", "notes hold longer"),
    ("tail after the last note", "tail_s", "s", "{:.2f}", "longer tail"),
    ("sound between notes vs during notes", "gap_fill_db", "dB", "{:.1f}", "more ringing into the gaps"),
    ("brightness wobble within notes", "mod_cent", "", "{:.2f}", "more tonal movement"),
    ("volume wobble within notes", "mod_amp", "dB", "{:.2f}", "more volume pulsing"),
]


def measurements2(a, a2, b, it):
    A, A2, B = DESC[a], DESC[a2], DESC[b]
    f0 = sorted(440 * 2 ** ((n["pitch"] - 69) / 12) for n in it["notes"])[len(it["notes"]) // 2]
    au = {"before": audible(A), "after": audible(B)}

    def val(d, k, fmt, unit, ok):
        if not ok:
            return "n/a (barely audible)"
        v = d[k]
        if k == "width_db" and v <= -100:
            return "mono"
        extra = f", {v / f0:.1f}x the note pitch" if k == "centroid_w_hz" else ""
        return f"{fmt.format(v)}{(' ' + unit) if unit else ''} [p{pct(k, v)}{extra}]"

    lines = [f"  level (RMS): {A['rms_dbfs']:.1f} -> {B['rms_dbfs']:.1f} dBFS [p{pct('rms_dbfs', A['rms_dbfs'])} -> "
             f"p{pct('rms_dbfs', B['rms_dbfs'])}]   (noise ±{abs(A['rms_dbfs'] - A2['rms_dbfs']):.1f})"
             + ("" if all(au.values()) else "   -> " + ", ".join(f"{k} render is barely audible" for k, v in au.items() if not v))]
    for label, k, unit, fmt, hi in ROWS2:
        noise = abs(A[k] - A2[k])
        lines.append(f"  {label} (higher = {hi}): {val(A, k, fmt, unit, au['before'])} -> {val(B, k, fmt, unit, au['after'])}"
                     f"   (noise ±{fmt.format(noise)})")
    if all(au.values()):
        lines.append("  energy by band, before -> after: " + ", ".join(
            f"{lab} {A['bands'][k] * 100:.0f}% -> {B['bands'][k] * 100:.0f}%" for lab, k in BANDS))
    d, fl = pair(P1, a, b), pair(P1, a, a2)
    lines.append(f"  overall difference (log-mel, dB): {d['mel_db']:.1f}   (noise {fl['mel_db']:.1f})")
    lines.append(f"  overall difference, audio-model embeddings (0=same): CLAP {d['clap']:.3f} (noise {fl['clap']:.3f}), "
                 f"Omni {d['omni_frames']:.3f} (noise {fl['omni_frames']:.3f})")
    lines.append("  [pNN] = higher than NN% of 480 typical Vital patch renders; the note pitch here is about "
                 f"{f0:.0f} Hz")
    return "\n".join(lines)


def build(iid):
    it = ITEMS[iid]
    w = it["wavs"]
    settings = json.load(open(P1 / "presets" / f"{w['before']}.vital"))["settings"]
    fields = dict(arch=it["archetype"].lower(), melody=melody(it), patch=patch_dump(it["summary_before"], settings),
                  edit=edit_lines(it), meas=measurements(w["before"], w["before2"], w["after"], it))
    return (HERE / "user.txt").read_text().format(**fields)


def call(user):
    body = {"model": MODEL, "messages": [{"role": "system", "content": (HERE / "system.txt").read_text().strip()},
                                         {"role": "user", "content": user}],
            "chat_template_kwargs": {"reasoning_effort": "xhigh"}}
    t0 = time.time()
    r = requests.post(URL, json=body, timeout=900).json()
    m = r["choices"][0]["message"]
    return m["content"].strip(), (m.get("reasoning") or m.get("reasoning_content") or ""), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--show-prompt", action="store_true")
    ap.add_argument("--n", type=int, default=2, help="samples per item")
    ap.add_argument("--show-reasoning", action="store_true")
    a = ap.parse_args()
    jobs = [(i, build(i)) for i in a.ids for _ in range(a.n)]
    if a.show_prompt:
        for i in dict.fromkeys(a.ids):
            print(f"===== PROMPT {i}\n{build(i)}\n")
    with ThreadPoolExecutor(8) as ex:
        res = list(ex.map(lambda j: call(j[1]), jobs))
    log = HERE / "runs.jsonl"
    with open(log, "a") as f:
        for (i, p), (txt, rsn, sec) in zip(jobs, res):
            print(f"--- {i}  ({sec:.0f}s, reasoning {len(rsn)} chars)\n{txt}\n")
            if a.show_reasoning:
                print(f"[reasoning]\n{rsn}\n")
            f.write(json.dumps({"t": time.time(), "id": i, "prompt": p, "system": (HERE / 'system.txt').read_text(),
                                "out": txt, "reasoning": rsn}) + "\n")


if __name__ == "__main__":
    main()
