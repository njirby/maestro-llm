# Maestro-LLM

Training data pipeline for an LLM that acts as a **terminal coding agent for music production**. The agent's job: given a target synthesizer sound, iteratively recreate it by writing and executing Python reapy code against a live REAPER DAW session.

---

## What the Agent Does

The agent runs from a terminal. It has a live REAPER session with a Vital synthesizer on the first track. The agent can see neither the target preset nor the REAPER GUI.

```
listen to reference audio
→ reason: what differs? which params to change?
→ search: enumerate fx.params filtered by keyword to find current values + param names
→ set: write Python (reapy) to set specific params by display name
→ listen to result → iterate
```

Each bash tool call is real, executable Python against the reapy API — no fictional CLI wrappers or custom helper classes.

---

## Agent → REAPER Interface

The agent communicates with a live REAPER session via **reapy** — a Python library that installs a TCP server inside REAPER's defer loop, enabling full bidirectional ReaScript API access from any external Python process.

### IPC mechanisms

| Mechanism | Direction | Latency | Full API | Notes |
|---|---|---|---|---|
| **reapy** (TCP defer loop) | Bidirectional RPC | ~5–30ms/call | Yes (Python) | **Primary choice** |
| `reaper -nonewinst script.lua` | → REAPER (one-shot) | ~100–500ms | Yes (Lua) | No return value to shell without a file |
| HTTP web interface | Bidirectional (poll) | ~5–50ms | Actions + fixed cmds only | No arbitrary Lua/Python |
| File-bridge (MCP pattern) | Bidirectional (poll) | ~50–200ms | Yes (Lua) | Used by total-reaper-mcp |

### reapy setup (one-time)

```bash
pip install python-reapy
python -c "import reapy; reapy.configure_reaper()"
# Restart REAPER — the reapy server script starts automatically
```

### How the agent writes code

**Search turn** — find params by keyword, see current normalized values:

```python
import reapy
with reapy.inside_reaper():
    fx = reapy.Project().tracks[0].fxs[0]
    hits = [(p.name, round(p.normalized, 4)) for p in fx.params
            if 'filter 1' in p.name.lower()]
    for name, val in hits:
        print(f"{name}: {val:.4f}")
```

```
Filter 1 Blend: 0.5000
Filter 1 Cutoff: 0.4063
Filter 1 Resonance: 0.4000
Filter 1 Style: 0.0000
```

**Set turn** — set params by exact display name, normalized [0, 1]:

```python
import reapy
with reapy.inside_reaper():
    fx = reapy.Project().tracks[0].fxs[0]
    fx.params["Filter 1 Cutoff"].value = 0.719
    fx.params["Filter 1 Resonance"].value = 0.620
    fx.params["Oscillator 1 Level"].value = 0.778
    print("Done")
```

**Listen turn** — render and play back:

```bash
python scripts/reaper_render_probe.py && aplay /tmp/probe.wav
```

The search-before-set pattern keeps context lean: instead of dumping all ~800 parameters, the model searches by keyword (~5–20 results) and uses the exact display names it sees in the output.

---

## Training Data Pipeline (Offline)

Generating SFT data does not require REAPER or a GPU. The pipeline uses **vita** — Python bindings for the Vital synthesis engine — to render audio directly.

```
preset_gen.py       →  target Vital preset (6 archetypes)
path_gen.py         →  N-step parameter path: default → target
demo_iter_examples.py  →  renders all clips + builds agentic tool-call conversations
```

### Generate demo examples

```bash
python scripts/demo_iter_examples.py
```

Produces under `outputs/iter_demo/{archetype}/`:
- `{id}_gt.wav` — ground truth (target preset, probe notes)
- `{id}_default.wav` — default Vital preset baseline
- `{id}_step{N}.wav` — cumulative preset after step N
- `{id}_target.vital` — target preset JSON
- `{id}_step{N}.vital` — cumulative preset JSON at each step
- `{id}_conversation.json` — full agentic tool-call conversation
- `demo_train.jsonl` — combined ms-swift JSONL

All clips (GT, default, per-step) use the same note sequence so release and length are directly comparable across the conversation.

### Verify preset fidelity

```bash
python scripts/verify_preset_path.py --archetype lead --seed 42
python scripts/verify_preset_path.py --archetype bass --seed 7 --verbose
```

Categorises every parameter by why it differs between target and final cumulative preset:

```
Summary: 165 exact | 0 noisy | 462 below_thresh | 0 untracked | 0 missing
```

- **exact** — tracked and set correctly (within 0.01 norm)
- **noisy** — tracked but drifted across steps (should be 0 after final-step convergence)
- **below_thresh** — |target − init| ≤ 0.05 norm, intentionally skipped
- **untracked** — not in `param_ranges.json`
- **missing** — in target but absent from final cumulative

### Build full dataset (requires Omni at localhost:8000)

```bash
python scripts/build_iter_sft_dataset.py \
    --manifest outputs/iter_sft/manifest.jsonl \
    --omni-server http://localhost:8000 \
    --output data/prepared/iter_sft/train.jsonl \
    --concurrency 8
```

### Train

```bash
# Single-card packing test (3B model, 1×24GB)
CUDA_VISIBLE_DEVICES=0 python scripts/test_packing_multimodal.py

# Full training (7B, 4×24GB with DeepSpeed ZeRO-3)
bash scripts/train_qwen25_omni_lora.sh full

# Smoke run (dry run, no GPU needed)
python scripts/train_qwen25_omni_lora.py --profile smoke --dry-run
```

Always train inside tmux — bare SSH runs orphan processes and can OOM the machine.

---

## Vital Synthetic Preset Pipeline

Generates musically diverse Vital presets for training data at scale.

### Archetypes

| Archetype | Character | Envelope | Filter bias |
|---|---|---|---|
| `bass` | Sub/mid bass | Fast attack, medium release | Low cutoff (~400 Hz) |
| `lead` | Monophonic lead | Fast attack, medium-long release | Mid-high cutoff |
| `pad` | Sustained texture | Slow attack (≥0.5s), long release | Low-mid + resonance |
| `keys` | Piano/EP/organ | Medium attack, medium decay | Bright |
| `pluck` | Plucked transient | Very fast attack, short decay | Bandpass |
| `sequence` | Rhythmic/arpeggio | Short attack, varied decay | Sweeping |

### Wavetable Library

568 unique, deduplicated wavetables extracted from `.vitaltable` files and embedded wavetables in the real preset corpus. Build once:

```bash
python -m maestro.synth.wavetable_lib \
  --vital-dir /path/to/.vitaltable/files \
  --presets-dir ~/Downloads/vital_presets \
  --output data/wavetable_lib.json
```

### Quality

- Audibility rate: ~92–94% overall; pluck ~83–96% (probe gate catches silent presets at render time)
- CLAP diversity: **0.524 overall** vs real preset corpus 0.477 — generator exceeds real preset diversity
- Archetype signature routes guaranteed: `env_2→filter` for bass/pluck, `lfo→wave_frame` for pad

### Key constraints

- Only ONE `vita.Synth()` per process — creating multiple causes a segfault
- `maestro/synth/preset_gen.py` never creates a Synth internally — it returns dicts only
- All 777 Vital control parameters are bounded in `maestro/synth/param_ranges.json`

### Throughput (Threadripper PRO 7965WX, 48 logical CPUs)

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

**Noise and mistake injection**: 35% of params in each step get Gaussian noise (σ=0.08 norm), 25% of paths get a deliberate mistake step to teach error recovery.

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

## Offline Lua Tuple Pipeline (Prior SFT Stage)

An earlier pipeline — still working — builds `(audio, Lua)` pairs for teaching the model to transcribe audio into REAPER Lua scripts that recreate a melody. This is a distinct task from the iterative sound recreation pipeline above.

### Generate tuples

```bash
python scripts/generate_reaper_tuples.py \
  --source slakh2100 \
  --workers 16 \
  --out data/processed/reaper_tuples
```

Note representation: `n("C4", m(12,3,"8t"), "q.", 90)` — bar/beat/offset tokens, duration tokens.

---

## Audio Token Budget

- ~94.5 audio tokens/second (Qwen2.5-Omni, empirically measured)
- ~6,000 tokens/minute
- 30s clip = ~2,835 audio tokens
- At `max_length=8192`: fits roughly 2 × 30s clips + text overhead

---

## MS-Swift Training Notes

- Requires `--attn_impl flash_attn` when packing is enabled (hard error otherwise)
- `packing_length` defaults to `max_length` — set `max_length ≥ 2× avg sample tokens` for packing to combine samples
- Samples exceeding `max_length` are **dropped** (not truncated) when packing is enabled
- 3B fits on 1×24GB at `max_length≤5120`; 7B needs 4-GPU DeepSpeed ZeRO-3
- Confirmed 2x step reduction: 16 samples → 8 steps/epoch at `max_length=5120`

DeepSpeed config: `configs/deepspeed_zero3.json` — includes both `optimizer` and `scheduler` blocks to avoid HF/DeepSpeed LR scheduler group mismatches.

---

## Environment Setup

```bash
# Python venv
source .venv/bin/activate
pip install -e ".[test]"

# MS-Swift + Omni dependencies
bash scripts/setup_ms_swift_omni.sh

# Serve Omni locally (for JSONL assembly and inference)
bash scripts/serve_qwen3_omni.sh
```

Venv: `/home/nate/Documents/maestro-llm/.venv`

---

## Repository Layout

```
maestro/
  render/
    vital.py          # vita bindings: render, trim silence, probe audibility
    reaper.py         # Lua script generation + headless subprocess execution
  reaper/
    vital_tools.py    # VitalController: high-level Vital VST access via reapy (utility, not training data)
  synth/
    preset_gen.py     # Synthetic Vital preset generation (6 archetypes)
    path_gen.py       # N-step parameter paths, snippet generation, fidelity diff
    wavetable_lib.py  # 568-wavetable library builder and loader
    init_preset.json  # Vital default preset (used as path start point)
    param_ranges.json # Bounds for all 777 Vital parameters
  data/
    phase1.py         # Manifest utilities for stem/MIDI corpora

scripts/
  demo_iter_examples.py       # Generate demo conversations + render all audio
  verify_preset_path.py       # Diagnostic: compare target vs final cumulative preset
  build_iter_sft_dataset.py   # Assemble multi-turn JSONL with Omni commentary
  reaper_render_probe.py      # Standalone: render REAPER track to /tmp/probe.wav
  render_vital_wavs.py        # General-purpose preset render script
  check_preset_diversity.py   # CLAP diversity audit
  benchmark_render.py         # Worker sweep benchmarks
  train_qwen25_omni_lora.py   # MS-Swift LoRA training launcher

tests/
  test_path_gen_snippets.py   # Unit tests: display names, search/set snippets
  test_pipeline_v2.py         # Structural invariants: reapy API, no fictional CLI
  test_demo_iter.py           # Conversation builder tests
  test_reapy_live.py          # Live REAPER integration tests (--reaper flag)

configs/
  deepspeed_zero3.json
```

---

## What Exists vs. What Is Planned

**Implemented and working:**
- Synthetic preset generator (`preset_gen.py`) with 6 archetypes
- Wavetable library builder and loader
- N-step parameter path generator (`path_gen.py`) with noise/mistake injection and final-step convergence
- Search-before-set snippet generation: synthetic reapy search/set code for each iteration
- Preset fidelity diagnostic (`verify_preset_path.py`) categorising diffs by root cause
- Demo conversation builder (`demo_iter_examples.py`) with matching note sequences across all clips
- Multi-turn JSONL assembler (`build_iter_sft_dataset.py`) with Omni commentary
- Lua tuple pipeline for melody transcription SFT data
- MS-Swift LoRA training scripts for Qwen2.5-Omni

**Planned / not yet implemented:**
- Agent inference loop (the trained model running against a live REAPER session)
- REAPER-bench for RLVR
- RL training stage

---

## Related Work

- [DAWZY (Elkins et al., NeurIPS 2025)](https://arxiv.org/abs/2512.03289) — LLM-based natural language control of REAPER
- [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291) — skill library accumulation for open-ended embodied agents
- [Agent-RLVR (Scale AI, 2025)](https://arxiv.org/abs/2506.11425) — RLVR with guidance hints for software engineering agents
- [LAION CLAP](https://github.com/LAION-AI/CLAP) — contrastive language-audio pretraining
- [vita Python bindings](https://github.com/andrewjjenkins/vita) — direct C++ Vital engine access (offline rendering)
