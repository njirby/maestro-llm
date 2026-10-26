#!/usr/bin/env python3
"""Edit Vital's 64-slot modulation matrix in preset.json.

  mod.py preset.json list
  mod.py preset.json add SRC DST AMOUNT [--bipolar on|off] [--power P] [--stereo on|off]
  mod.py preset.json remove SRC DST

list:   one line per used slot: `slot <n>  <src> -> <dst>  <amount %>[  bipolar][  power P][  bypassed]`
add:    routes SRC into DST with AMOUNT in -1..1 (or -100%..100%, e.g. -30%): updates the existing SRC->DST slot, else takes the
        first empty slot. Sets amount, clears bypass. Prints the slot line.
remove: empties the SRC->DST slot and resets its amount/bipolar/power/stereo/bypass to defaults.

Sources: lfo_1..8, random_1..4, env_1..6, macro_control_1..4, mod_wheel, pitch_wheel, velocity, aftertouch, note,
note_in_octave, lift, stereo. Destinations: any continuous parameter key (plus transposes, unison voices and
*_tempo); switches/enum selectors are not modulatable.
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vitallib import MOD_SOURCES, N_SLOTS, UsageError, load_preset, params, run_main, save_preset  # noqa

SLOT_DEFAULTS = {"amount": 0.0, "bipolar": 0.0, "power": 0.0, "stereo": 0.0, "bypass": 0.0}


def slot_line(s, i):
    m = s["modulations"][i]
    n = i + 1
    amt = float(s.get(f"modulation_{n}_amount", 0.0))
    line = f"slot {n}  {m['source']} -> {m['destination']}  {amt * 100:+.0f}%"
    if float(s.get(f"modulation_{n}_bipolar", 0)) > 0.5:
        line += "  bipolar"
    if float(s.get(f"modulation_{n}_power", 0)) != 0:
        line += f"  power {float(s[f'modulation_{n}_power']):g}"
    if float(s.get(f"modulation_{n}_bypass", 0)) > 0.5:
        line += "  bypassed"
    return line


def on_off(text):
    t = text.lower()
    if t not in ("on", "off", "1", "0"):
        raise UsageError(f"expected on/off, got {text!r}")
    return 1.0 if t in ("on", "1") else 0.0


def check_dest(dst):
    p = params().get(dst)
    if p is None:
        raise UsageError(f"unknown destination {dst!r} (use a parameter key, e.g. filter_1_cutoff)")
    if dst.endswith("_on") or dst == "bypass" or dst.startswith("view_") or dst.endswith("_view_2d") \
            or ("labels" in p and not dst.endswith("_tempo")):
        raise UsageError(f"{dst} is a switch/selector and cannot be modulated")


def find(s, src, dst):
    for i, m in enumerate(s["modulations"]):
        if m.get("source") == src and m.get("destination") == dst:
            return i
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("preset")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pa = sub.add_parser("add")
    pa.add_argument("src")
    pa.add_argument("dst")
    pa.add_argument("amount")
    pa.add_argument("--bipolar")
    pa.add_argument("--power", type=float)
    pa.add_argument("--stereo")
    pr = sub.add_parser("remove")
    pr.add_argument("src")
    pr.add_argument("dst")
    # argparse reads "-30%" as an option; a leading space keeps it a positional (the amount parser strips it)
    argv = [" " + t if re.fullmatch(r"-\d*\.?\d+%", t) else t for t in sys.argv[1:]]
    a = ap.parse_args(argv)

    preset = load_preset(a.preset)
    s = preset["settings"]
    mods = s.setdefault("modulations", [])
    while len(mods) < N_SLOTS:
        mods.append({"source": "", "destination": ""})

    if a.cmd == "list":
        used = [i for i, m in enumerate(mods) if m.get("source") or m.get("destination")]
        print(f"{len(used)} routes, {N_SLOTS - len(used)} free slots")
        for i in used:
            print(slot_line(s, i))
        return

    if a.src not in MOD_SOURCES:
        raise UsageError(f"unknown source {a.src!r}; sources: {', '.join(MOD_SOURCES)}")
    check_dest(a.dst)
    i = find(s, a.src, a.dst)

    if a.cmd == "remove":
        if i is None:
            raise UsageError(f"no route {a.src} -> {a.dst} (mod.py {a.preset} list)")
        mods[i] = {"source": "", "destination": ""}
        for k, v in SLOT_DEFAULTS.items():
            s[f"modulation_{i + 1}_{k}"] = v
        save_preset(a.preset, preset)
        print(f"slot {i + 1}  removed {a.src} -> {a.dst}")
        return

    m = re.fullmatch(r"([-+]?\d*\.?\d+)\s*(%?)", a.amount.strip())
    if not m:
        raise UsageError(f"amount must be a number in -1..1 or a percentage, got {a.amount!r}")
    amt = float(m.group(1)) / (100 if m.group(2) else 1)
    if not -1 <= amt <= 1 or amt == 0:
        raise UsageError(f"amount {amt:g} must be non-zero and within -1..1 (0 = dead route; use remove)")
    if a.power is not None and not -10 <= a.power <= 10:
        raise UsageError("--power must be within -10..10")
    if i is None:
        i = next((j for j, mm in enumerate(mods) if not mm.get("source") and not mm.get("destination")), None)
        if i is None:
            raise UsageError("all 64 modulation slots are used; remove one first")
        mods[i] = {"source": a.src, "destination": a.dst}
        for k, v in SLOT_DEFAULTS.items():
            s[f"modulation_{i + 1}_{k}"] = v
    n = i + 1
    s[f"modulation_{n}_amount"] = amt
    s[f"modulation_{n}_bypass"] = 0.0
    if a.bipolar is not None:
        s[f"modulation_{n}_bipolar"] = on_off(a.bipolar)
    if a.power is not None:
        s[f"modulation_{n}_power"] = a.power
    if a.stereo is not None:
        s[f"modulation_{n}_stereo"] = on_off(a.stereo)
    save_preset(a.preset, preset)
    print(slot_line(s, i))


if __name__ == "__main__":
    run_main(main)
