# Vital Synthetic Preset Pipeline

[← Back to README](../README.md)

Generates musically diverse Vital presets for training data at scale.

## Archetypes

| Archetype | Character | Envelope | Filter bias |
|---|---|---|---|
| `bass` | Sub/mid bass | Fast attack, medium release | Low cutoff (~400 Hz) |
| `lead` | Monophonic lead | Fast attack, medium-long release | Mid-high cutoff |
| `pad` | Sustained texture | Slow attack (≥0.5s), long release | Low-mid + resonance |
| `keys` | Piano/EP/organ | Medium attack, medium decay | Bright |
| `pluck` | Plucked transient | Very fast attack, short decay | Bandpass |
| `sequence` | Rhythmic/arpeggio | Short attack, varied decay | Sweeping |

## Wavetable Library

568 unique, deduplicated wavetables extracted from `.vitaltable` files and embedded wavetables in the real preset corpus. Build once:

```bash
python -m maestro.synth.wavetable_lib \
  --vital-dir /path/to/.vitaltable/files \
  --presets-dir ~/Downloads/vital_presets \
  --output data/wavetable_lib.json
```

## Quality

- Audibility rate: ~92–94% overall; pluck ~83–96% (probe gate catches silent presets at render time)
- CLAP diversity: **0.524 overall** vs real preset corpus 0.477 — generator exceeds real preset diversity
- Archetype signature routes guaranteed: `env_2→filter` for bass/pluck, `lfo→wave_frame` for pad

## Key constraints

- Only ONE `vita.Synth()` per process — creating multiple causes a segfault
- `maestro/synth/preset_gen.py` never creates a Synth internally — it returns dicts only
- All 777 Vital control parameters are bounded in `maestro/synth/param_ranges.json`

## Throughput (Threadripper PRO 7965WX, 48 logical CPUs)

| Workers | WAV/s | ETA for 1M renders |
|---|---|---|
| 8 | 1.8 | ~6.5 days |
| 16 | 3.1 | ~3.7 days |
| **24** | **3.9** | **~3 days** |
| 32 | 3.5 | ~3.3 days |

24 workers is the sweet spot. Performance drops at 32 due to NUMA boundary effects.

---

## Iterative Parameter Path Generation

`maestro/synth/path_gen.py` generates N-step parameter paths from Vital's default preset to a generated target. Each path is used to create one multi-turn training conversation.

**N is determined by the number of changed parameters** (normalized diff > 0.05):

| Changed params | N iterations |
|---|---|
| ≤56 | 8 |
| ≤72 | 10 |
| ≤96 | 13 |
| ≤120 | 17 |
| >120 | 20 |

**Param priority ordering** ensures the model learns to fix oscillators before envelopes before modulation routing — mirroring how a skilled sound designer approaches the problem.

**Mistake injection**: 8% of individual parameter assignments in each step are deliberately set to wrong values (`MISTAKE_PROB=0.08`), teaching error recovery without injecting full steps of noise. A correction step is appended at the end of paths that have mistakes.

**Final-step convergence**: the last cumulative preset is silently snapped to exactly the target so all final audio clips match the GT.

### Conversation structure

Each conversation follows this pattern:

```
user:       <audio> [GT clip]  "Recreate this {archetype} sound in Vital."
assistant:  → bash: render default preset (baseline listen)
tool:       <audio> [default clip]

# Per iteration (N times):
assistant:  [reasoning] → bash: search for keyword params
tool:       Param Name 1: 0.4063\nParam Name 2: 0.4000\n...
assistant:  [observation] → bash: set params by display name
tool:       Done
assistant:  [listen text] → bash: render probe
tool:       <audio> [step N clip]

assistant:  "Recreation complete."
```

Total audio in conversation: 2 + N clips (GT + default + N step renders).

---

## Offline Audio-Script Tuple Pipeline (Prior SFT Stage)

An earlier pipeline — still working — builds `(audio, script)` pairs for teaching the model to transcribe audio into REAPER scripts that recreate a melody. This is a distinct task from the iterative sound recreation pipeline above.

```bash
python scripts/generate_reaper_tuples.py \
  --source slakh2100 \
  --workers 16 \
  --out data/processed/reaper_tuples
```

Note representation: `n("C4", m(12,3,"8t"), "q.", 90)` — bar/beat/offset tokens, duration tokens.
