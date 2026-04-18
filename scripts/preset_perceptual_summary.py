#!/usr/bin/env python3
"""Translate a Vital preset dict into a perceptual-bucket text summary.

Intentionally abstracts away numeric parameter values — the output uses only
the kind of vocabulary a producer would use describing a sound by ear. The
purpose is to ground Stage 1 OBSERVATIONS generation: pass this summary to the
audio-LM alongside the target clip so listening is anchored to truth instead
of pattern-matched to generic synth prose.

Usage as library:
    from scripts.preset_perceptual_summary import summarize_preset_perceptual
    text = summarize_preset_perceptual(preset_dict)

Usage as CLI (for inspection):
    python scripts/preset_perceptual_summary.py path/to/preset.vital
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _env_attack_bucket(seconds: float) -> str:
    if seconds < 0.02: return "snappy near-instant"
    if seconds < 0.08: return "plucked/fast"
    if seconds < 0.25: return "quick"
    if seconds < 0.8:  return "gradual"
    return "slow swelling"


def _env_decay_bucket(seconds: float) -> str:
    if seconds < 0.15: return "short"
    if seconds < 0.6:  return "moderate"
    if seconds < 1.5:  return "long"
    return "very long"


def _env_sustain_bucket(level: float) -> str:
    if level < 0.15: return "percussive (drops to near-silence)"
    if level < 0.45: return "low sustain"
    if level < 0.75: return "moderate sustain"
    return "held/high sustain"


def _env_release_bucket(seconds: float) -> str:
    if seconds < 0.15: return "tight cut-off"
    if seconds < 0.5:  return "natural release"
    if seconds < 1.2:  return "long tail"
    return "very long lingering tail"


def _cutoff_bucket(midi: float) -> str:
    # Vital filter cutoff is in MIDI note units (roughly)
    if midi < 45:  return "very dark"
    if midi < 70:  return "warm"
    if midi < 90:  return "bright"
    return "very bright / open"


def _resonance_bucket(r: float) -> str:
    if r < 0.15: return "no resonant peak"
    if r < 0.4:  return "mild resonance"
    if r < 0.7:  return "prominent resonance"
    return "sharp/vocal resonance"


def _filter_model_name(model: int) -> str:
    # Vital filter model enum — rough labels
    names = {
        0: "analog-style low-pass",
        1: "analog-style low-pass (alt)",
        2: "dirty low-pass",
        3: "ladder low-pass",
        4: "digital low-pass",
        5: "low-pass",
        6: "band-pass",
        7: "high-pass",
        8: "notch",
        9: "all-pass",
        10: "comb",
        11: "formant",
    }
    return names.get(int(model), f"filter model {int(model)}")


def _unison_bucket(voices: int, detune: float) -> str:
    if voices <= 1:
        return "single voice (no unison)"
    if detune < 0.1:
        return f"{voices} tight stacked voices"
    if detune < 0.3:
        return f"{voices} gently detuned unison voices"
    return f"{voices} widely detuned unison voices"


def _lfo_rate_bucket(rate: float | None) -> str:
    """Rate in Hz if freeform, or tempo-sync factor if synced. Rough bucketing."""
    if rate is None:
        return "unspecified rate"
    if rate < 0.5: return "very slow swell"
    if rate < 2:   return "slow modulation"
    if rate < 6:   return "mid-rate wobble"
    return "fast vibrato-rate"


def _mod_route_describe(src: str, dst: str, amount: float) -> str | None:
    """Translate a (source, destination, amount) tuple into a perceptual phrase."""
    if not src or not dst or abs(float(amount)) < 0.01:
        return None
    src = str(src).lower()
    dst = str(dst).lower()
    amt_word = "subtle" if abs(amount) < 0.3 else ("moderate" if abs(amount) < 0.7 else "strong")
    src_label = (
        "envelope" if src.startswith("env") else
        "LFO" if src.startswith("lfo") else
        "random" if "random" in src else
        "keytrack" if "note" in src or "key" in src else
        "velocity" if "velocity" in src else
        "macro" if "macro" in src else
        src.split("_")[0]
    )
    if "cutoff" in dst:
        return f"{amt_word} {src_label}-driven filter sweep"
    if "reson" in dst:
        return f"{amt_word} {src_label}-driven resonance modulation"
    if "pitch" in dst or "transpose" in dst:
        return f"{amt_word} {src_label}-driven pitch movement"
    if "wave_frame" in dst or "frame" in dst:
        return f"{amt_word} {src_label}-driven wavetable morphing"
    if "amp" in dst or "level" in dst:
        return f"{amt_word} {src_label}-driven amplitude movement"
    if "pan" in dst:
        return f"{amt_word} {src_label}-driven stereo motion"
    if "fm" in dst or "detune" in dst or "unison" in dst:
        return f"{amt_word} {src_label}-driven timbral color shift"
    return f"{amt_word} {src_label}-driven movement"


def summarize_preset_perceptual(preset: dict, wavetable_lib: list[dict] | None = None) -> str:
    """Produce a perceptual text summary of a Vital preset.

    No parameter names, no numeric values, no kHz. Purely perceptual buckets a
    producer would use by ear.
    """
    s = preset.get("settings", {}) or {}
    lines: list[str] = []

    # Oscillators — which are on, unison, wavetable character
    active_oscs: list[int] = [i for i in (1, 2, 3) if float(s.get(f"osc_{i}_on", 0) or 0) > 0.5]
    osc_phrases = []
    for i in active_oscs:
        voices = int(s.get(f"osc_{i}_unison_voices", 1) or 1)
        detune = float(s.get(f"osc_{i}_unison_detune", 0) or 0)
        wave_name = s.get(f"osc_{i}_wavetable_frame_name", "") or ""
        pieces = [_unison_bucket(voices, detune)]
        if wave_name:
            pieces.append(f"wavetable '{wave_name}'")
        osc_phrases.append(f"osc {i}: " + ", ".join(pieces))
    if osc_phrases:
        lines.append(f"- Oscillators: {len(active_oscs)} active — " + "; ".join(osc_phrases))
    else:
        lines.append("- Oscillators: none active (sound would be silent without modulation)")

    # Envelope (primary)
    if any(f"env_1_{k}" in s for k in ("attack", "decay", "sustain", "release")):
        atk = float(s.get("env_1_attack", 0) or 0)
        dec = float(s.get("env_1_decay", 0) or 0)
        sus = float(s.get("env_1_sustain", 0) or 0)
        rel = float(s.get("env_1_release", 0) or 0)
        lines.append(
            f"- Primary envelope: {_env_attack_bucket(atk)} attack, "
            f"{_env_decay_bucket(dec)} decay, {_env_sustain_bucket(sus)}, "
            f"{_env_release_bucket(rel)}"
        )

    # Extra envelopes with nontrivial shape (2+)
    extra_envs = []
    for ei in (2, 3, 4, 5, 6):
        # is envelope actually routed somewhere? check mod routes
        routed = False
        for mi in range(1, 65):
            src = str(s.get(f"modulation_{mi}_source", "") or "")
            amt = float(s.get(f"modulation_{mi}_amount", 0) or 0)
            if f"env_{ei}" in src and abs(amt) > 0.01:
                routed = True
                break
        if routed:
            atk = float(s.get(f"env_{ei}_attack", 0) or 0)
            sus = float(s.get(f"env_{ei}_sustain", 0) or 0)
            extra_envs.append(f"env {ei} ({_env_attack_bucket(atk)} attack, {_env_sustain_bucket(sus)})")
    if extra_envs:
        lines.append(f"- Secondary envelopes shaping movement: {'; '.join(extra_envs)}")

    # Filter
    if float(s.get("filter_1_on", 0) or 0) > 0.5:
        cutoff = float(s.get("filter_1_cutoff", 60) or 60)
        reson = float(s.get("filter_1_resonance", 0) or 0)
        model = int(s.get("filter_1_model", 0) or 0)
        lines.append(
            f"- Filter 1 active: {_cutoff_bucket(cutoff)}, {_resonance_bucket(reson)}, "
            f"{_filter_model_name(model)}"
        )
    else:
        lines.append("- No active filter (timbre is raw from the oscillators)")

    # Modulation routes (pick prominent ones)
    route_phrases: list[str] = []
    seen_routes: set[tuple[str, str]] = set()
    for mi in range(1, 65):
        src = str(s.get(f"modulation_{mi}_source", "") or "")
        dst = str(s.get(f"modulation_{mi}_destination", "") or "")
        amt = float(s.get(f"modulation_{mi}_amount", 0) or 0)
        phrase = _mod_route_describe(src, dst, amt)
        key = (src, dst)
        if phrase and key not in seen_routes:
            seen_routes.add(key)
            route_phrases.append(phrase)
        if len(route_phrases) >= 4:
            break
    if route_phrases:
        lines.append(f"- Prominent movement: {'; '.join(route_phrases)}")
    else:
        lines.append("- No prominent modulation movement (static timbre)")

    # LFOs — any with notable modulation depth
    lfos_with_depth = set()
    for mi in range(1, 65):
        src = str(s.get(f"modulation_{mi}_source", "") or "")
        amt = float(s.get(f"modulation_{mi}_amount", 0) or 0)
        for li in range(1, 9):
            if f"lfo_{li}" in src and abs(amt) > 0.05:
                lfos_with_depth.add(li)
                break
    if lfos_with_depth:
        lines.append(f"- Active LFOs contributing audible motion: {', '.join('lfo ' + str(i) for i in sorted(lfos_with_depth))}")

    # Effects
    fx_on = []
    for fx in ("chorus", "reverb", "delay", "distortion", "compressor", "eq", "phaser", "flanger"):
        if float(s.get(f"{fx}_on", 0) or 0) > 0.5:
            fx_on.append(fx)
    if fx_on:
        lines.append(f"- Effects active: {', '.join(fx_on)}")
    else:
        lines.append("- No active effects (dry signal)")

    return "\n".join(lines)


def summarize_residual_delta_perceptual(
    target: dict, final: dict, max_items: int = 5
) -> str:
    """Identify what still differs between target and final preset, in perceptual language.

    Scans envelope ADSR, filter cutoff/resonance/on-off, unison, and effect on/off for
    meaningful differences. Returns the top `max_items` by magnitude as human-readable
    bullets ready to inject into the FINAL ASSESSMENT prompt. Empty-string if the
    recreation closely matches the target.
    """
    t = target.get("settings", {}) or {}
    f = final.get("settings", {}) or {}
    items: list[tuple[float, str]] = []

    # Envelope ADSR per envelope (1-6)
    for ei in range(1, 7):
        for attr, thresh, perceptual in [
            ("attack", 0.05, "attack"),
            ("decay", 0.1, "decay"),
            ("sustain", 0.1, "sustain level"),
            ("release", 0.1, "release tail"),
        ]:
            key = f"env_{ei}_{attr}"
            if key in t and key in f:
                try:
                    tv, fv = float(t[key]), float(f[key])
                except (TypeError, ValueError):
                    continue
                diff = abs(tv - fv)
                if diff > thresh:
                    # Normalized magnitude for sort (thresh-relative)
                    mag = diff / max(thresh, 1e-6)
                    if attr == "sustain":
                        direction = "too low" if fv < tv else "too high"
                    else:
                        direction = "too short" if fv < tv else "too long"
                    items.append((mag, f"env {ei} {perceptual} is {direction} relative to target"))

    # Filter on/off + cutoff + resonance
    for fi in (1, 2):
        t_on = float(t.get(f"filter_{fi}_on", 0) or 0) > 0.5
        f_on = float(f.get(f"filter_{fi}_on", 0) or 0) > 0.5
        if t_on != f_on:
            items.append((10.0,
                f"filter {fi} should be {'engaged' if t_on else 'disengaged'} but is currently "
                f"{'engaged' if f_on else 'off'}"))
        if t_on and f_on:
            tc = float(t.get(f"filter_{fi}_cutoff", 60) or 60)
            fc = float(f.get(f"filter_{fi}_cutoff", 60) or 60)
            if abs(tc - fc) > 5:
                direction = "too dark" if fc < tc else "too bright"
                items.append((abs(tc - fc) / 10.0, f"filter {fi} cutoff is {direction}"))
            tr = float(t.get(f"filter_{fi}_resonance", 0) or 0)
            fr = float(f.get(f"filter_{fi}_resonance", 0) or 0)
            if abs(tr - fr) > 0.15:
                direction = "less resonant" if fr < tr else "more resonant"
                items.append((abs(tr - fr) * 3.0, f"filter {fi} needs to be {direction}"))

    # Unison voices and detune on active osc
    for oi in (1, 2, 3):
        if float(t.get(f"osc_{oi}_on", 0) or 0) > 0.5 and float(f.get(f"osc_{oi}_on", 0) or 0) > 0.5:
            tv = int(t.get(f"osc_{oi}_unison_voices", 1) or 1)
            fv = int(f.get(f"osc_{oi}_unison_voices", 1) or 1)
            if tv != fv:
                direction = "too thin (fewer layered voices)" if fv < tv else "too thick (more layered voices than needed)"
                items.append((abs(tv - fv) / 2.0, f"osc {oi} unison is {direction}"))
            td = float(t.get(f"osc_{oi}_unison_detune", 0) or 0)
            fd = float(f.get(f"osc_{oi}_unison_detune", 0) or 0)
            if abs(td - fd) > 0.1:
                direction = "too tight (not enough detuning)" if fd < td else "too wide (over-detuned)"
                items.append((abs(td - fd) * 2.0, f"osc {oi} detune is {direction}"))

    # Effect on/off mismatches
    for fx in ("chorus", "reverb", "delay", "distortion", "compressor", "eq", "phaser", "flanger"):
        t_on = float(t.get(f"{fx}_on", 0) or 0) > 0.5
        f_on = float(f.get(f"{fx}_on", 0) or 0) > 0.5
        if t_on != f_on:
            items.append((10.0,
                f"{fx} should be {'active' if t_on else 'disengaged'} but is currently "
                f"{'active' if f_on else 'off'}"))

    if not items:
        return "(recreation closely matches the target — no significant residual differences)"

    items.sort(key=lambda x: -x[0])
    top = [s for _, s in items[:max_items]]
    return "\n".join(f"- {s}" for s in top)


GROUNDED_OBSERVATIONS_PROMPT_TEMPLATE = """\
You are a producer listening to a synth clip and describing it to another producer who will
reverse-engineer the patch. Your description must be concrete enough that they can sketch
the signal chain from your words alone.

For reference only — a perceptual-bucket summary of the clip (no numbers, no parameter names):
{preset_summary}

Listen to the clip. Write 3-5 short sentences. Every sentence must carry actionable
sound-design information — not poetic atmosphere. Between them, cover ALL FIVE of these
aspects, weaving them into natural prose (not a bulleted list):

  1. ENVELOPE SHAPE — attack speed (snappy/plucky/gradual/slow-swell), decay & sustain
     behavior (percussive drop / held plateau / slow decline), release tail length.
  2. TONE & FILTER — brightness, harmonic character (clean / buzzy / resonant / vocal),
     any filter sweep or static coloration.
  3. OSCILLATOR BODY — single-voice vs. layered/detuned unison, sense of thickness or
     chorused width from the raw oscillators (before effects).
  4. MOVEMENT — static vs. LFO-driven wobble vs. envelope-driven evolution vs. filter
     mod sweep. Name the kind of motion if any.
  5. SPACE & EFFECTS — dry vs. wet, presence of chorus shimmer, reverb size, delay
     echoes, distortion severity (subtle grit vs. heavy crunch).

EXAMPLE (actionable, reverse-engineerable):
  "Quick plucky attack drops into a long held plateau with a smooth, lingering release.
  The tone is raw and bright with no filter coloration — just the oscillators' own
  harmonic edge. Multiple close-detuned voices stack into a thick chorused body. The
  timbre sits fixed once it hits — no LFO motion, no filter sweep. A prominent chorus
  adds shimmer and subtle distortion gives it grit, with no reverb or delay space."

COUNTER-EXAMPLE (too poetic, not actionable — DO NOT write like this):
  "Pristine, glass-like onset with a sustained, shimmering aura, layered with a faint
  organic warble that adds life without motion."

STRICT RULES:
  - Describe only what is AUDIBLY present. The reference is a safety net against
    hallucination — not a source of exact details to cite.
  - Do NOT state exact voice counts ("three voices") or name specific effect units
    ("chorus effect") — describe the audible signature ("thick detuned shimmer",
    "long wet tail").
  - Do NOT cite parameter names, numbers, Hz values, seconds, or preset fields.
  - Do NOT use comparative framing ("the target has", "unlike the default", "in
    contrast"). Describe the sound directly.
  - Do NOT contradict the reference (e.g. if reference says "static timbre", do not
    claim LFO wobble or evolution).
  - Keep it 3-5 sentences. Natural prose, no bullets, no headers.
"""
"""Template for grounded Stage-1 OBSERVATIONS. Format with:
    preset_summary=<output of summarize_preset_perceptual>
"""


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python scripts/preset_perceptual_summary.py path/to/preset.vital", file=sys.stderr)
        sys.exit(1)
    with open(sys.argv[1]) as f:
        preset = json.load(f)
    print(summarize_preset_perceptual(preset))


if __name__ == "__main__":
    main()
