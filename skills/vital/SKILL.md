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

## Operations

All operations are Python, invoked through the `bash` tool as inline heredocs. Two backends:

- **reapy** (`import reapy; from reapy import reascript_api as RPR`) — TCP RPC bridge to REAPER for all live-project interaction: track creation, param set/get, MIDI insert, VST chunk apply.
- **DawDreamer** (`import dawdreamer`) — in-process JUCE-based VST3 host for all audio rendering. Loads Vital's VST3 directly, renders MIDI to WAV without touching REAPER's transport.

Concrete operations:

- **List wavetables**: Load `data/wavetable_lib.json`, deduplicate by name, print total or a slice. (Pure Python, no REAPER interaction.)
- **Render probes**: DawDreamer loads `maestro/synth/init_preset.json`, swaps `wavetables[0]` with each candidate, renders the target's MIDI notes via `add_midi_note()`, writes WAV with `soundfile`.
- **Render tuple**: Same as probes but assigns wavetables to multiple oscillator slots simultaneously before rendering.
- **Set parameters**: reapy scans `RPR.TrackFX_GetParamName` to find indices by REAPER display name, then calls `RPR.TrackFX_SetParam` with normalized 0-1 values.
- **Apply VST chunk**: Python builds the Vital preset JSON, base64-encodes it, and calls `RPR.TrackFX_SetNamedConfigParm` via reapy to apply to the live REAPER track.

## Sub-agents you coordinate

Before running the recreation loop you dispatch three kinds of sub-agents via the `Agent` tool. Each runs in its own fresh context and writes its result to a JSON file on disk that you then `cat` to consume.

- `melody_transcription` — listens to the target and uses reapy to insert the correct MIDI notes on a REAPER track via `RPR.MIDI_InsertNote`. Output: `/tmp/agents/<sample>/transcription.json` with `{notes, n_notes, duration_s}`. Runs **once** at the start, after you've created the track.
- `wavetable_search` — audits one slice (~48 wavetables) of the library and returns a shortlist of 2–4 names that could be building blocks. Dispatched **4 in parallel** per search round across disjoint library slices.
- `wavetable_judge` — takes the combined pool from all search agents (~12 unique names) and selects the final tuple (N names = target's active oscillator count). Dispatched **once per search round** after the search shortlists come back.

## Strategy

High-level process for recreating a target sound in Vital:

1. **Listen** to the target audio and the current default baseline.
2. **Create a REAPER MIDI track** (reapy: `RPR.InsertTrackAtIndex` + `RPR.TrackFX_AddByName`). This is where the transcribed MIDI will drive playback.
3. **Transcribe the melody** — dispatch `melody_transcription` with the target audio + track index. The subagent inserts notes on the track and writes the note list JSON for you to `cat`.
4. **Search** — dispatch 4 `wavetable_search` sub-agents in parallel, each covering a slice (~48 indices) of the library. Each returns a shortlist of 2-4 wavetable names.
5. **Judge** — dispatch a `wavetable_judge` sub-agent with the combined pool (~12 unique names). The judge listens to the target + all pool candidates in one audition and selects the N best (N = number of active oscillators in the target preset).
6. **Audition tuple** — render the judge-selected tuple via DawDreamer, listen, accept or trigger another search round.
7. **Diagnose** — write OBSERVATIONS (preset-grounded perceptual description) + PLAN (one qualitative bullet per subsystem that needs changes).
8. **Execute** subsystem-batched parameter edits in order: **oscillator → envelope → filter → lfo → fx → modulation → macro**. Render fresh audio after each batch; narration grounded in the plan bullet + actual before/after param deltas.
9. **Correct inline** if a batch overshoots — a single `set_params` call reverts the offending parameter, followed by a listen-and-confirm turn.
10. **Verdict** — FINAL ASSESSMENT grounded in the residual preset delta: what the recreation matches well, what specific residual (attack too plucky, filter too dark, etc.) remains.

## Subsystem taxonomy

See `references/subsystem_taxonomy.md` for the mapping from Vital parameter families to presentation subsystems. The seven-subsystem ordering above is what PLAN bullets cover and what batch-narration turns use.

## Important caveats

- **Multi-audio Omni comparison is out-of-distribution.** Do not ask Omni to compare target+default simultaneously — it conflates the two and hallucinates attack direction and modulation. For Stage 1 observations, use target-only audio with the preset-bucket summary as a grounding prior.
- **Filter cutoff is in MIDI note units** (not Hz). Resonance is 0-1.
- **Wavetable names can contain spaces, hyphens, and parentheses** — quote them in bash. Names never contain commas (safe to comma-join with `--names`).
- **The library has 282 unique wavetables** (post-dedup by name from a 568-entry raw library).
- **Edge case**: if the user invokes the pipeline without attaching an audio clip (no audio tag in the first user message), refuse with a prompt asking the user to select an audio item in REAPER first. Do not fabricate a target.
- **Audio duration cap**: target clips are limited to ~30 seconds by the rendering backend. Longer clips (for melody transcription of whole verses, bridges, etc.) aren't yet supported in a single transcription pass — they'd need parallel-slice transcription which is future work.

## No-audio variant

Sometimes the user sends "Recreate this sound in Vital." without attaching audio. Respond with a single-turn refusal that asks them to select an audio clip in REAPER, then stop. Do not fabricate a target sound.
