"""Verified 'session notes' for a target's first impression: MIDI facts + ablation-verified modules + ranked tone.

  .venv/bin/python fi_notes.py prep      # write ablation presets + notes.json per target
  (container) render all                 # see RENDER below
  .venv/bin/python fi_notes.py notes     # measure, print the session notes
  .venv/bin/python fi_notes.py run       # Omni (served on :8000) with audio + notes -> omni_fi/instruct_notes.jsonl
"""
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO = Path("/home/nate/Documents/maestro-llm")
sys.path.insert(0, str(REPO))
from maestro.sounddiff.measure import descriptors, logmel  # noqa: E402
from maestro.sounddiff.vital_facts import live_routes, why_dead  # noqa: E402

P1 = REPO / "outputs/sounddiff/probe1"
W = REPO / "outputs/sounddiff/fi_notes"
SKILL = REPO / "opencode/skills/vital"
IDS = ["f004", "f005", "f008", "f009", "f012", "f019", "f022", "f027"]
if __import__("os").environ.get("ALL"):
    IDS = [f"f{i:03d}" for i in range(40)]
ITEMS = {json.loads(l)["id"]: json.loads(l) for l in open(P1 / "items.jsonl")}
REF = [json.loads(l)["desc"] for l in open(REPO / "outputs/sounddiff/refset/desc.jsonl")]
FX = ["reverb", "delay", "chorus", "distortion", "phaser", "flanger", "compressor", "eq"]
FX_MIX = {"reverb": "reverb_dry_wet", "delay": "delay_dry_wet", "chorus": "chorus_dry_wet",
          "distortion": "distortion_mix", "phaser": "phaser_dry_wet", "flanger": "flanger_dry_wet"}


# ---------------------------------------------------------------- ablations
def ablations(s):
    """{name: settings-modifier} for every module that is on / routed in the target."""
    out = {}
    for fx in FX:
        if s.get(f"{fx}_on", 0) > 0.5:
            out[f"fx:{fx}"] = lambda t, fx=fx: t.__setitem__(f"{fx}_on", 0.0)
    by_src = {}
    for r in live_routes(s):
        if why_dead(s, r["destination"]) is None:
            by_src.setdefault(r["source"], []).append(r)
    for src, rs in by_src.items():
        if src.startswith(("lfo_", "random_", "env_")) and src != "env_1":
            def kill(t, rs=rs):
                for r in rs:
                    t[f"modulation_{r['slot']}_amount"] = 0.0
            out[f"mod:{src}->" + ",".join(sorted({r['destination'] for r in rs}))] = kill
    for n in (1, 2, 3):
        if why_dead(s, f"osc_{n}_wave_frame") is None:
            if s.get(f"osc_{n}_unison_voices", 1) > 1:
                out[f"unison:osc{n}"] = lambda t, n=n: t.__setitem__(f"osc_{n}_unison_voices", 1.0)
            if n > 1 or any(why_dead(s, f"osc_{m}_wave_frame") is None for m in (2, 3)):
                out[f"osc:{n}"] = lambda t, n=n: t.__setitem__(f"osc_{n}_on", 0.0)
    for f in ("1", "2"):
        if why_dead(s, f"filter_{f}_cutoff") is None:
            out[f"filter:{f}"] = lambda t, f=f: t.__setitem__(f"filter_{f}_on", 0.0)
    return out


def prep():
    (W / "presets").mkdir(parents=True, exist_ok=True)
    jobs = []
    for iid in IDS:
        it = ITEMS[iid]
        p = json.load(open(P1 / "presets" / f"{it['wavs']['target']}.vital"))
        (W / f"{iid}.notes.json").write_text(json.dumps({"bpm": it["bpm"], "notes": it["notes"]}))
        names = {"full": None, "full2": None, **ablations(p["settings"])}
        for k, (name, fn) in enumerate(names.items()):
            q = copy.deepcopy(p)
            if fn:
                fn(q["settings"])
            stem = f"{iid}__{k:02d}"
            (W / "presets" / f"{stem}.vital").write_text(json.dumps(q))
            jobs.append({"id": iid, "stem": stem, "abl": name})
    (W / "jobs.jsonl").write_text("\n".join(json.dumps(j) for j in jobs))
    print(len(jobs), "renders")


RENDER = """cd {w}; Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 & export DISPLAY=:99; sleep 2
ls presets/*.vital | xargs -P 6 -I{{}} sh -c 'b=$(basename {{}} .vital); i=${{b%%__*}}; [ -f wav/$b.wav ] || python3 {skill}/scripts/render.py {{}} $i.notes.json wav/$b.wav >/dev/null'
"""


def render():
    (W / "wav").mkdir(exist_ok=True)
    cmd = RENDER.format(w=W, skill=SKILL)
    subprocess.run(["docker", "run", "--rm", "--network", "none", "--user", "1000:1000", "-v", "/home/nate:/home/nate", "-e",
                    "VITAL_VST3=/usr/lib/vst3/Vital.vst3", "--entrypoint", "bash", "daw-farm/reaper:latest", "-c", cmd],
                   check=True)


# ---------------------------------------------------------------- measuring
def pct(key, v):
    xs = [r[key] for r in REF if r[key] is not None]
    return 100 * sum(x < v for x in xs) / len(xs)


def mel_d(a, b):
    la, lb = logmel(a.mean(1)), logmel(b.mean(1))
    n = min(len(la), len(lb))
    return float(np.abs(la[:n] - lb[:n]).mean())


def strength(x):
    return "strong" if x > 6 else "clear" if x > 2 else "subtle" if x > 0.8 else None


def describe_module(name, d_full, d_abl):
    """Phrase for an audible module from what changes when it is switched off (full vs ablated descriptors)."""
    kind, _, what = name.partition(":")
    dd = lambda k: (d_full.get(k) or 0) - (d_abl.get(k) or 0)
    if kind == "fx":
        return {"reverb": "reverb adding space and a tail", "delay": "delay echoes",
                "chorus": "chorus adding width and shimmer", "distortion": "distortion adding grit and bite",
                "phaser": "phaser sweep", "flanger": "flanger sweep", "compressor": "compression evening out the level",
                "eq": "EQ shaping the tone"}[what]
    if kind == "mod":
        src, _, dsts = what.partition("->")
        per_note = src.startswith("env")
        ds = dsts.split(",")
        if any("cutoff" in x for x in ds):
            return "each note's brightness changes as it plays" if per_note else "a cyclic wobble in the brightness"
        if any(x.endswith(("_level", "volume")) for x in ds):
            return "each note's volume swells or dips as it plays" if per_note else "a pulsing in the volume"
        if any(x.endswith(("_tune", "_transpose")) for x in ds):
            return "a pitch bend at the start of notes" if per_note else "a wobble in the pitch (vibrato)"
        if any("wave_frame" in x or "spectral" in x or "distortion_amount" in x for x in ds):
            return "each note's tone colour changes as it plays" if per_note else "the tone colour shifts back and forth"
        return "the effects swell and recede as it plays"
    if kind == "unison":
        return "a slightly detuned, thick quality"
    if kind == "osc":
        return "a second oscillator layer"
    if kind == "filter":
        return "darker with it" if dd("centroid_w_hz") < 0 else "brighter with it"
    return name


def heard_length(d):
    sr = d.get("sustain_ratio")
    if sr is None:
        return None
    return "notes die away quickly (plucky)" if sr < 0.3 else "notes fade a bit as they hold" if sr < 0.7 else \
        "notes hold steady"


def midi_facts(it, desc=None):
    n = it["notes"]
    name = lambda p: "C C# D D# E F F# G G# A A# B".split()[p % 12] + str(p // 12 - 1)
    ends = sorted([(x["start_s"], 1) for x in n] + [(x["start_s"] + x["dur_s"], -1) for x in n])
    cur = mx = 0
    for _, d in ends:
        cur += d
        mx = max(mx, cur)
    total = max(x["start_s"] + x["dur_s"] for x in n)
    overlap_t = 0.0
    for i, a in enumerate(n):
        for b in n[i + 1:]:
            overlap_t += max(0.0, min(a["start_s"] + a["dur_s"], b["start_s"] + b["dur_s"]) - max(a["start_s"], b["start_s"]))
    share = overlap_t / total
    shape = ("a single line" if mx == 1 else
             "chords throughout" if share > 0.5 else
             f"chords of up to {mx} notes" if share > 0.15 else "mostly a single line with some overlapping notes")
    med = sorted(x["pitch"] for x in n)[len(n) // 2]
    reg = "low" if med < 52 else "mid" if med < 67 else "high" if med < 80 else "very high"
    avg = sum(x["dur_s"] for x in n) / len(n)
    length = "very short, staccato notes" if avg < 0.15 else "short notes" if avg < 0.35 else \
        "medium-length notes" if avg < 0.8 else "long held notes"
    dens = len(n) / total
    busy = "sparse" if dens < 1.5 else "busy" if dens > 4 else None
    hl = heard_length(desc) if desc else None
    return [f"{shape}", f"{reg} register ({name(min(x['pitch'] for x in n))}-{name(max(x['pitch'] for x in n))})",
            length + (f" that {hl.split(' ', 1)[1]}" if hl and hl.startswith("notes ") else "") + (f", {busy}" if busy else "")]


STRENGTH = {"subtle": "a touch of", "clear": "noticeable", "strong": "lots of"}


def notes():
    """v4 notes: one line per core dimension, always stated (dry / steady / no distortion / width), listener words,
    subtle things said subtly, distortion kept separate from harmonic richness. Normalized audio; skip only silence."""
    jobs = [json.loads(l) for l in open(W / "jobs.jsonl")]
    res = {}
    for iid in IDS:
        it = ITEMS[iid]
        js = [j for j in jobs if j["id"] == iid]
        wav = {j["abl"] or ("full" if j["stem"].endswith("00") else "full2"):
               sf.read(W / "wav" / f"{j['stem']}.wav", dtype="float32")[0] for j in js}
        full = wav["full"]
        peak = float(np.abs(full).max())
        if peak < 1e-5:  # digital silence (< -100 dBFS): nothing to describe
            print(f"===== {iid} SKIPPED: silent render (peak {20 * np.log10(peak + 1e-12):.0f} dBFS)")
            continue
        g = 0.5 / peak  # level-normalize everything for this target by the same gain (keeps relative changes)
        wav = {k: v * g for k, v in wav.items()}
        full = wav["full"]
        floor = mel_d(full, wav["full2"])
        dfull = descriptors(full, it["notes"])
        aud = {}
        for j in js:
            if j["abl"]:
                st = strength(mel_d(full, wav[j["abl"]]) - floor)
                if st:
                    aud[j["abl"]] = (st, describe_module(j["abl"], dfull, descriptors(wav[j["abl"]], it["notes"])))
        order = {"strong": 0, "clear": 1, "subtle": 2}
        best = lambda keys: min((aud[k][0] for k in keys), key=order.get) if keys else None

        # space
        space = []
        for fx, word in (("reverb", "reverb"), ("delay", "delay echoes")):
            if f"fx:{fx}" in aud:
                space.append(f"{STRENGTH[aud[f'fx:{fx}'][0]]} {word}")
        space = ", ".join(space) if space else "dry: no reverb or delay"
        # movement
        mv = [(st, ph) for k, (st, ph) in aud.items() if k.startswith("mod:")]
        mv += [(aud[f"fx:{fx}"][0], f"a {fx} sweep") for fx in ("phaser", "flanger") if f"fx:{fx}" in aud]
        mv.sort(key=lambda a: order[a[0]])
        def mv_phrase(st, ph):
            if ph.startswith("each note's"):
                return ph.replace(" as it plays", {"subtle": " slightly as it plays", "clear": " as it plays",
                                                   "strong": " a lot as it plays"}[st])
            core = ph[2:] if ph.startswith("a ") else ph
            return {"subtle": "a slight ", "clear": "a noticeable ", "strong": "a strong "}[st] + core
        movement = "; ".join(mv_phrase(st, ph) for st, ph in mv) if mv else "steady: no wobble, pulsing or sweeps"
        pitch_mod = any(("tune" in k or "transpose" in k) for k in aud if k.startswith("mod:"))
        # distortion vs harmonic richness
        dist = f"{STRENGTH[aud['fx:distortion'][0]]} distortion adding grit and bite" if "fx:distortion" in aud else "no distortion"
        hr = pct("harm_rich_db", dfull["harm_rich_db"]) if dfull.get("harm_rich_db") is not None else 50
        harm = ("pure, smooth (few overtones)" if hr < 25 else
                "rich, buzzy overtones" + ("" if "fx:distortion" in aud else " from the sound itself, not distortion")
                if hr > 75 else "moderately rich")
        b = pct("centroid_w_hz", dfull["centroid_w_hz"])
        bright = "dark" if b < 25 else "bright" if b > 75 else "medium brightness"
        noisy = "noisy / hissy" if pct("flatness", dfull["flatness"]) > 85 else None
        w = pct("width_db", dfull["width_db"])
        width = "narrow, centered" if w < 25 else "wide stereo" if w > 75 else "medium width"
        thick = []
        if best([k for k in aud if k.startswith("osc:")]) == "strong":
            thick.append("thick, layered")
        if any(k.startswith("unison") for k in aud):
            thick.append("slightly detuned")
        if "fx:chorus" in aud:
            thick.append(f"{STRENGTH[aud['fx:chorus'][0]]} chorus shimmer")
        filt = [aud[k] for k in aud if k.startswith("filter:") and "darker" in aud[k][1]]
        if filt and bright != "bright":
            bright += ", top end rolled off"
        lines = ["Session notes (verified against the session):",
                 "- Part: " + "; ".join(midi_facts(it, dfull)) + ".",
                 f"- Space: {space}.",
                 f"- Movement: {movement}." + ("" if pitch_mod else " Pitch is steady."),
                 f"- Distortion: {dist}.",
                 f"- Stereo: {width}.",
                 "- Tone: " + ", ".join(x for x in [bright, harm, noisy] + thick if x) + "."]
        res[iid] = {"text": "\n".join(lines), "aud": aud, "floor": floor,
                    "facts": {"dry": "fx:reverb" not in aud and "fx:delay" not in aud,
                              "no_dist": "fx:distortion" not in aud, "steady": not mv, "pitch_steady": not pitch_mod,
                              "narrow": w < 25, "chords": "chords" in midi_facts(it)[0],
                              "single": midi_facts(it)[0] == "a single line"}}
        print(f"===== {iid}\n{res[iid]['text']}\n")
    (W / "notes.json").write_text(json.dumps(res, indent=1))


# ---------------------------------------------------------------- Omni
PROMPT = ("This clip is a single synth patch playing a part. {notes}\n\nThe notes are accurate. Using them, give your "
          "first impression the way a producer would to a collaborator, in 2-3 sentences, in your own words: the part, "
          "how the notes behave, the space and movement, and the core tone. Keep subtle things subtle, don't add "
          "anything the notes rule out, and don't read the notes back as a list. No preset or wavetable names and no "
          "synth-engineering jargon (oscillators, layers, polyphony, LFO names). Reply with just the impression, in "
          "English.")


def run(tag="instruct_notes4"):
    sys.path.insert(0, str(Path(__file__).parent))
    import omni_fi as o
    from check import check
    res = json.loads((W / "notes.json").read_text())
    model = o.requests.get(o.URL.replace("chat/completions", "models")).json()["data"][0]["id"]

    def gen(iid):
        b64 = o.audio_b64(iid)[0]
        body = {"model": model, "messages": [{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {"data": b64, "format": "wav"}},
            {"type": "text", "text": PROMPT.format(notes=res[iid]["text"])}]}]}
        r = o.requests.post(o.URL, json=body, timeout=600)
        if r.status_code != 200:
            raise RuntimeError(r.text[:400])
        t = r.json()["choices"][0]["message"]["content"].strip()
        return t.split("</think>", 1)[1].strip() if "</think>" in t else t

    def one(job):
        iid, k = job
        t1 = gen(iid)
        p1 = check(t1, res[iid]["facts"])
        if not p1:
            return t1, [], None
        t2 = gen(iid)  # one retry: a fresh sample, same prompt (no post-editing)
        return t2, p1, check(t2, res[iid]["facts"])

    from concurrent.futures import ThreadPoolExecutor
    jobs = [(iid, k) for iid in IDS if iid in res for k in range(2)]
    with ThreadPoolExecutor(16) as ex:
        outs = list(ex.map(one, jobs))
    with open(o.O / f"{tag}.jsonl", "w") as f:
        for (iid, k), (t, p1, p2) in zip(jobs, outs):
            f.write(json.dumps({"id": iid, "k": k, "tag": tag, "model": model, "text": t, "first_try_problems": p1,
                                "final_problems": p2 if p1 else [], "reasoning_chars": 0}) + "\n")
    n = len(outs)
    first_bad = sum(bool(p1) for _, p1, _ in outs)
    final_bad = sum(bool(p2) for _, p1, p2 in outs if p1)
    print(f"{n} impressions; first try flagged {first_bad} ({first_bad / n:.0%}); after one retry {final_bad} ({final_bad / n:.0%})")


if __name__ == "__main__":
    {"prep": prep, "render": render, "notes": notes, "run": run}[sys.argv[1]]()
