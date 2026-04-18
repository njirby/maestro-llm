---
name: vital
description: Recreate a target synth sound using the Vital synthesizer. Use when
  the target audio was produced with (or should be recreated with) Vital — a
  wavetable synthesizer with 3 oscillators, 6 envelopes, 8 LFOs, a dual-filter
  section, a fixed FX chain (chorus, delay, reverb, distortion, compressor,
  EQ, phaser, flanger), a modulation matrix, and 4 macros. Invoke this skill
  before listening to the target whenever the context indicates Vital or a
  wavetable-based synthesis target.
metadata:
  plugin: vital
  paradigm: wavetable
  subsystems: [oscillator, envelope, filter, lfo, fx, modulation, macro]
---

# Vital skill

Domain-specific guidance and tooling for recreating sounds made in Vital.

## Helper tools

All invoked via the `bash` tool. Run from the repo root (paths are relative).

- `python skills/vital/scripts/list_wavetables.py --total` — library size (e.g. 282)
- `python skills/vital/scripts/list_wavetables.py --start N --end M` — names of wavetables at indices N..M
- `python skills/vital/scripts/render_probes.py --names "A,B,C" --out-dir DIR` — render each wavetable through the default Vital preset so they can be auditioned (bare probes; no envelope/filter/FX shaping applied)
- `python skills/vital/scripts/render_probes.py --idxs 12,45,67 --out-dir DIR` — same, by library index
- `python skills/vital/scripts/render_tuple.py --osc1 X [--osc2 Y] [--osc3 Z] --out WAV` — render a full preset with chosen wavetables assigned to each active oscillator (inactive oscillators muted)

Parameter application happens through the `VitalController` inside an inline Python bash snippet — see the diagnosis → subsystem batches flow for the pattern.

## Strategy

High-level process for recreating a target sound in Vital:

1. **Listen** to the target audio and the current default baseline.
2. **Search** — dispatch 4 `wavetable_search` sub-agents in parallel, each covering a slice (~48 indices) of the library. Each returns a shortlist of 2-4 wavetable names.
3. **Judge** — dispatch a `wavetable_judge` sub-agent with the combined pool (~12 unique names). The judge listens to the target + all pool candidates in one audition and selects the N best (N = number of active oscillators in the target preset).
4. **Audition tuple** — render the judge-selected tuple via `render_tuple.py`, listen, accept or trigger another search round.
5. **Diagnose** — write OBSERVATIONS (preset-grounded perceptual description) + PLAN (one qualitative bullet per subsystem that needs changes).
6. **Execute** subsystem-batched parameter edits in order: **oscillator → envelope → filter → lfo → fx → modulation → macro**. Render fresh audio after each batch; narration grounded in the plan bullet + actual before/after param deltas.
7. **Correct inline** if a batch overshoots — a single `set_params` call reverts the offending parameter, followed by a listen-and-confirm turn.
8. **Verdict** — FINAL ASSESSMENT grounded in the residual preset delta: what the recreation matches well, what specific residual (attack too plucky, filter too dark, etc.) remains.

## Subsystem taxonomy

See `references/subsystem_taxonomy.md` for the mapping from Vital parameter families to presentation subsystems. The seven-subsystem ordering above is what PLAN bullets cover and what batch-narration turns use.

## Important caveats

- **Multi-audio Omni comparison is out-of-distribution.** Do not ask Omni to compare target+default simultaneously — it conflates the two and hallucinates attack direction and modulation. For Stage 1 observations, use target-only audio with the preset-bucket summary as a grounding prior.
- **Filter cutoff is in MIDI note units** (not Hz). Resonance is 0-1.
- **Wavetable names can contain spaces, hyphens, and parentheses** — quote them in bash. Names never contain commas (safe to comma-join with `--names`).
- **The library has 282 unique wavetables** (post-dedup by name from a 568-entry raw library).
- **Edge case**: if the user invokes the pipeline without attaching an audio clip (no `<audio>` in the first user message), refuse with a prompt asking the user to select an audio item in REAPER first. Do not fabricate a target.

## No-audio variant

Sometimes the user sends "Recreate this sound in Vital." without attaching audio. Respond with a single-turn refusal that asks them to select an audio clip in REAPER, then stop. Do not fabricate a target sound.
