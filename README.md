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

| Mechanism | Direction | Latency | Full API | Notes |
|---|---|---|---|---|
| **reapy** (TCP defer loop) | Bidirectional RPC | ~5–30ms/call | Yes (Python) | **Primary choice** — all REAPER interaction in generated conversations |
| **DawDreamer** (in-process VST3) | N/A (offline) | ~0.4s/10s clip | Vital only | Wavetable probe + tuple rendering (no REAPER needed) |
| `reaper -nonewinst script.lua` | → REAPER (one-shot) | ~100–500ms | Yes (Lua) | Not used in generated code |
| HTTP web interface | Bidirectional (poll) | ~5–50ms | Actions + fixed cmds only | No arbitrary Lua/Python |

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

**Set turn** — set params by exact display name, normalized [0, 1]:

```python
import reapy
with reapy.inside_reaper():
    fx = reapy.Project().tracks[0].fxs[0]
    fx.params["Filter 1 Cutoff"].value = 0.719
    fx.params["Filter 1 Resonance"].value = 0.620
    print("Done")
```

The search-before-set pattern keeps context lean: instead of dumping all ~800 parameters, the model searches by keyword (~5–20 results) and uses the exact display names it sees in the output.

---

## Quick Start

```bash
# Python venv
source .venv/bin/activate
pip install -e ".[test]"

# MS-Swift + Omni dependencies
bash scripts/setup_ms_swift_omni.sh

# Serve Omni locally (for JSONL assembly and inference)
bash scripts/serve_qwen3_omni.sh
```

Agent runtime deps (used in generated tool calls): `dawdreamer`, `numpy`, `soundfile`, `python-reapy`.

---

## Repository Layout

```
maestro/
  render/
    vital.py            # vita bindings: render, trim silence, probe audibility
  reaper/
    vital_tools.py      # VitalController: high-level Vital VST access via reapy
  synth/
    preset_gen.py       # Synthetic Vital preset generation (6 archetypes)
    path_gen.py         # N-step parameter paths, snippet generation, fidelity diff
    wavetable_lib.py    # 568-wavetable library builder and loader
    init_preset.json    # Vital default preset (used as path start point)
    param_ranges.json   # Bounds for all 777 Vital parameters

scripts/
  build_main_agent_sft_v3.py    # Primary SFT builder: diagnose → subsystem-batched execute
  build_search_agent_sft_v2.py  # Search agent: iterative batch-listening
  build_judge_agent_sft_v3.py   # Judge agent: audition combined pool, pick best wavetables
  build_transcription_agent_sft_v4.py  # Melody transcription with self-verification
  build_all_agent_sft.sh        # Orchestrator: build all 4 agents + grade + spot-check
  build_unified_sft_v4.py       # Unified builder: all 4 agents from one script
  grade_agent_sft.py            # Quality grader (structural + LLM-judge)
  render_iter_presets.py         # Batch-render GT + probe clips; write manifest.jsonl
  agent_sft_common.py           # Shared helpers: CLAP embedder, candidate pool, rendering

tests/
  test_agent_sft_contracts_v3.py  # Contract tests for v3 main agent
  test_search_agent_sft_v2.py    # Contract tests for search agent v2

configs/
  deepspeed_zero3.json
```

---

## What Exists vs. What Is Planned

**Implemented and working:**
- Synthetic preset generator with 6 archetypes + 568-wavetable library
- Primary SFT pipeline (v3): diagnose → subsystem-batched execute with per-batch audio, inline mistake correction, preset-grounded observations, residual-delta verdict
- 4-agent architecture: main, search, judge, melody transcription (see [SFT Pipeline](docs/sft-pipeline.md))
- Orchestrator script (`build_all_agent_sft.sh`) with full grading and HTML spot-check
- MS-Swift LoRA training for Qwen2.5-Omni (see [Training Notes](docs/ms-swift-training.md))

**Planned / not yet implemented:**
- Plugin-explorer agent — systematically probe unfamiliar plugins, write `skills/<plugin>/SKILL.md`
- Second-plugin proof-of-generalisation beyond Vital
- Longer-audio melody transcription (parallel-slice transcription for clips >30s)
- Agent inference loop against live REAPER (see `/home/nate/Documents/maestro-reaper-plugin/`)
- REAPER-bench for RLVR + RL training stage

---

## Documentation

| Doc | Contents |
|---|---|
| [SFT Pipeline](docs/sft-pipeline.md) | v3 pipeline architecture, conversation structure, grading dimensions, all builder commands |
| [Vital Synth](docs/vital-synth.md) | Preset generation, wavetable library, quality metrics, render throughput, path generation |
| [MS-Swift Training](docs/ms-swift-training.md) | Audio token budget, packing, Megatron sequence-parallel findings, LoRA rank sweeps |
| [VST Chunk Format](docs/vital-chunk-format.md) | Binary chunk layout, `build_vital_chunk()`, gotchas |
| [DawDreamer State Format](docs/dawdreamer-state-format.md) | 5-layer JUCE binary format for offline VST3 preset loading |
| [Vital yabridge Setup](docs/vital-yabridge-setup.md) | Why Linux-native Vital crashes REAPER, WINE + yabridge install |
| [Agent SFT Architecture](docs/agent_sft_architecture.md) | Deep dive on the 4-agent design and inter-agent protocol |
| [Megatron QLoRA Investigation](docs/megatron_qlora_investigation.md) | Detailed TP/BNB patching experiments |

---

## Related Work

- [DAWZY (Elkins et al., NeurIPS 2025)](https://arxiv.org/abs/2512.03289) — LLM-based natural language control of REAPER
- [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291) — skill library accumulation for open-ended embodied agents
- [Agent-RLVR (Scale AI, 2025)](https://arxiv.org/abs/2506.11425) — RLVR with guidance hints for software engineering agents
- [LAION CLAP](https://github.com/LAION-AI/CLAP) — contrastive language-audio pretraining
- [vita Python bindings](https://github.com/andrewjjenkins/vita) — direct C++ Vital engine access (offline rendering)
