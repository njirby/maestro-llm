"""
Smoke test: verify that running a generated Lua script in REAPER produces
audible audio with the correct duration.

Vita and Vital are different synths, so cosine similarity is not meaningful.
Instead we check:
  1. REAPER WAV exists
  2. REAPER WAV is non-silent (peak amplitude > threshold)
  3. REAPER WAV duration is within DUR_TOLERANCE of expected clip duration
  4. Optional WAV-match check against tuple WAV:
     - auto mode: strict check only when tuple wav_engine == "reaper"
     - exact mode: strict sample-for-sample equality check
     - none mode: disabled

For each tuple in the manifest:
  1. Load the extracted MIDI clip and regenerate Lua pointing to a fresh temp path
  2. Run through REAPER (no overwrite dialog since path is new)
  3. Check existence, silence, duration, and optional match

Usage:
    python scripts/smoke_test_lua_vs_vita.py --manifest /tmp/tuple_clips/manifest.jsonl
"""

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent.parent))
from maestro.render.reaper import generate_lua, run_lua_batch

SILENCE_THRESHOLD = 0.001   # peak amplitude below this = silent
DUR_TOLERANCE_S   = 5.0     # REAPER dur must be within this of clip_dur_s


def _should_check_match(mode: str, wav_engine: str | None) -> bool:
    if mode == "none":
        return False
    if mode == "exact":
        return True
    # auto
    return (wav_engine or "").lower() == "reaper"


def _exact_wav_match(reference_wav: str, generated_wav: str) -> tuple[bool, str]:
    """
    Compare decoded PCM sample data exactly.
    Uses int32 read for stable integer-domain equality checks.
    """
    try:
        ref, ref_sr = sf.read(reference_wav, always_2d=True, dtype="int32")
        got, got_sr = sf.read(generated_wav, always_2d=True, dtype="int32")
    except Exception as exc:
        return False, f"read_error:{exc}"

    if ref_sr != got_sr:
        return False, f"sr_mismatch({ref_sr}!={got_sr})"
    if ref.shape != got.shape:
        return False, f"shape_mismatch({ref.shape}!={got.shape})"
    if not np.array_equal(ref, got):
        return False, "pcm_mismatch"
    return True, "match"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument(
        "--match-check",
        choices=["auto", "exact", "none"],
        default="auto",
        help=(
            "Tuple WAV match check mode: auto=only for wav_engine=reaper, "
            "exact=always strict sample equality, none=disable."
        ),
    )
    args = parser.parse_args()

    out_dir = Path("/tmp/smoke_reaper")
    out_dir.mkdir(exist_ok=True)

    entries = []
    with open(args.manifest) as f:
        for line in f:
            e = json.loads(line)
            if e["wav_ok"]:
                entries.append(e)
            if len(entries) >= args.limit:
                break

    if not entries:
        print("No valid entries in manifest.")
        sys.exit(1)

    print(f"Testing {len(entries)} tuple(s)...\n")

    # 1. Regenerate Lua with fresh output paths (avoids REAPER overwrite dialog)
    temp_lua_paths = []
    reaper_wav_paths = []

    for e in entries:
        reaper_wav = str(out_dir / f"{e['id']}.wav")
        reaper_rpp = str(out_dir / f"{e['id']}.rpp")
        reaper_wav_paths.append(reaper_wav)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pm = pretty_midi.PrettyMIDI(e["midi"])

        notes = pm.instruments[0].notes
        tempos = pm.get_tempo_changes()
        bpm = float(tempos[1][0]) if len(tempos[1]) > 0 else 120.0
        tail_s = float(e.get("tail_s", 0.5))
        expected_dur = float(e.get("render_dur_s", max(n.end for n in notes) + tail_s))

        lua_src = generate_lua(
            notes=notes, bpm=bpm,
            wav_path=reaper_wav, rpp_path=reaper_rpp,
            track_name=e.get("track_name", "Vital Synth"),
            tail_s=tail_s,
            min_duration_s=expected_dur,
        )
        temp_lua = str(out_dir / f"{e['id']}.lua")
        Path(temp_lua).write_text(lua_src)
        temp_lua_paths.append((temp_lua, expected_dur))
        print(
            f"  Prepared: {e['id']}  clip={e['clip_dur_s']:.1f}s"
            f"  expected={expected_dur:.1f}s  bpm={bpm:.0f}"
        )

    lua_only = [p for p, _ in temp_lua_paths]

    # 2. Run through REAPER
    print(f"\nRunning {len(lua_only)} Lua script(s) through REAPER...")
    run_lua_batch(lua_only, timeout=300)

    # 3. Check results
    print("\nResults:")
    print(f"  {'ID':<44} {'expected':>8}  {'dur_reaper':>10}  {'peak':>7}  {'match':>14}  pass?")
    print("  " + "-" * 102)
    all_pass = True
    for e, reaper_wav, (temp_lua, expected_dur) in zip(entries, reaper_wav_paths, temp_lua_paths):
        reaper_path = Path(reaper_wav)
        if not reaper_path.exists():
            print(f"  {e['id']:<44} REAPER WAV not found — FAIL")
            all_pass = False
            continue

        audio, sr = sf.read(reaper_wav, always_2d=True)
        dur_r = audio.shape[0] / sr
        peak  = float(np.abs(audio).max())

        silent = peak < SILENCE_THRESHOLD
        dur_ok = abs(dur_r - expected_dur) <= DUR_TOLERANCE_S
        match_required = _should_check_match(args.match_check, e.get("wav_engine"))
        match_ok = True
        match_info = "skipped"
        if match_required:
            ref_wav = e.get("wav")
            if not ref_wav:
                match_ok = False
                match_info = "missing_ref"
            else:
                match_ok, match_info = _exact_wav_match(str(ref_wav), reaper_wav)

        ok = (not silent) and dur_ok and match_ok

        flags = []
        if silent:     flags.append("SILENT")
        if not dur_ok: flags.append(f"DUR_MISMATCH(expected~{expected_dur:.1f}s)")
        if not match_ok: flags.append(f"MATCH_FAIL({match_info})")
        status = "PASS" if ok else "FAIL " + ",".join(flags)
        if not ok:
            all_pass = False
        print(
            f"  {e['id']:<44} {expected_dur:>6.1f}s  {dur_r:>8.1f}s  {peak:>7.4f}"
            f"  {match_info:>14}  {status}"
        )

    print()
    if all_pass:
        print("All passed.")
    else:
        print("Some failed.")
        sys.exit(1)


if __name__ == "__main__":
    main()
