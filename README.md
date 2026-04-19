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

# Dual-agent SFT outputs:
# - main agent orchestration conversations (delegates search)
# - step-level search-agent conversations (writes bash+reapy search code)
python scripts/build_iter_sft_dataset.py \
    --manifest outputs/iter_sft/manifest.jsonl \
    --omni-server http://localhost:8000 \
    --output data/prepared/iter_sft/main_agent_train.jsonl \
    --main-search-mode search_agent \
    --search-output data/prepared/iter_sft/search_agent_train.jsonl \
    --search-fanout 3 \
    --search-bad-result-prob 0.25 \
    --search-bad-result-max 1 \
    --concurrency 8
```

### Diagnose → subsystem-batched execute pipeline (main-agent SFT v3)

This is the primary SFT pipeline as of April 2026. It generates multi-turn conversations where an audio-language model listens to a target sound, writes an upfront subsystem plan, then executes by subsystem with one listen per batch. Replaces v2's per-step HEARD/HYPOTHESIS/PLAN narration, which produced shallow boilerplate commentary because the LLM was asked to post-hoc rationalize an oracle's 17-step plan one step at a time.

**Step 1 — generate path data and render audio** (no GPU needed, same as v2):

```bash
python scripts/render_iter_presets.py \
    --generate 1000 \
    --archetypes bass lead pad keys pluck sequence \
    --output-dir outputs/iter_sft \
    --wavetable-lib data/wavetable_lib.json \
    --jobs 24
```

**Step 2 — build SFT conversations** (requires Qwen3-Omni at localhost:8000):

```bash
python scripts/build_main_agent_sft_v3.py \
    --manifest outputs/iter_sft/manifest.jsonl \
    --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
    --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
    --wavetable-lib data/wavetable_lib.json \
    --out-jsonl data/prepared/agent_sft/main_v3.jsonl \
    --max-batches 8 \
    --mistake-rate 0.20 \
    --omni-server http://localhost:8000 \
    --omni-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --stage2-server http://localhost:8000 \
    --stage2-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --clap-device cuda:0
```

v3 does **not** use path_gen's per-step iterations for conversation structure. It diffs the target preset vs the init preset directly, buckets changed params by subsystem, and renders fresh per-batch audio using vita. This eliminates cross-subsystem leakage from path_gen's support-family fill mechanism.

**Conversation structure:**

```
user:       <audio> [GT clip]  "Recreate this sound in Vital."

# SKILL DISCOVERY + LOAD (claw-code-idiomatic, every normal record):
assistant:  Let me see which skills are available for this plugin.
tool_call:  bash ls skills/*/SKILL.md
tool_resp:  skills/vital/SKILL.md
assistant:  The vital skill matches. Loading it for plugin-specific instructions,
            helper-script paths, and recreation strategy.
tool_call:  Skill { skill: "vital", args: "" }
tool_resp:  { skill, path, description, prompt: <full SKILL.md contents> }

assistant:  Skill loaded. Listening to current default preset baseline.
tool_resp:  <audio> [default clip]

# MIDI TRANSCRIPTION BLOCK:
assistant:  Creating a REAPER track to hold the transcribed MIDI before search.
tool_call:  bash reapy → InsertTrackAtIndex + TrackFX_AddByName("Vital", ...)
tool_resp:  {"status":"ok","track_idx":0,"track_name":"target_melody"}
assistant:  Dispatching the transcription subagent to populate the track from the target.
tool_call:  Agent { subagent_type: "melody_transcription", ... }
tool_resp:  {agentId, outputFile: "/tmp/agents/<sid>/transcription.json", status: completed}
assistant:  Reading the transcribed note list.
tool_call:  bash cat /tmp/agents/<sid>/transcription.json
tool_resp:  {"notes": [...], "n_notes": 12, "duration_s": 5.0}

assistant:  MIDI ready on track 0 — 12 notes, ~5.0s. Checking wavetable library size.
tool_call:  bash python skills/vital/scripts/list_wavetables.py --total
tool_resp:  {"total": 282}
assistant:  Library has 282 wavetables. Dispatching 4 search agents in parallel...
            → 4 parallel Agent tool_calls (wavetable_search subagents)
            → bash cat to read all shortlist JSONs
            → dispatch Agent wavetable_judge to audition combined pool
            → bash cat judge output → render tuple → listen
            → accept (or re-search if target not found)
            → bash apply tuple via VitalController

# DIAGNOSIS (one audio turn, one text turn):
assistant:  OBSERVATIONS: [preset-grounded perceptual description of target, from Omni
              Stage 1 — target WAV + perceptual preset summary as grounding prior]
            PLAN:
              • Oscillator: [qualitative change needed]
              • Filter: [...]
              • Envelope: [...]
              ...
            Executing plan by subsystem.
            Applying oscillator changes.
tool_call:  bash [set ALL oscillator params in one call]
tool_resp:  {"status": "ok"}
assistant:  Listening after oscillator batch.
tool_call:  bash [render probe of cumulative state]
tool_resp:  {"status": "ok", "batch_audio": "<audio>", ...}
assistant:  [plan-aligned one-sentence narration grounded in param before→after
             deltas and the subsystem's plan bullet]

# Repeat for each non-empty subsystem: envelope, filter, lfo, fx, modulation, macro

# CORRECTION (inline, ~20% of samples):
assistant:  [batch check] Overshot on {subsystem} — backing off {param}.
tool_call:  bash [set corrective value]
tool_resp:  {"status": "ok"}
assistant:  Listening to the corrected preset.
tool_call:  bash [render probe]
tool_resp:  {"status": "ok", "corrected_audio": "<audio>", ...}

# FINAL ASSESSMENT (grounded in residual preset delta):
assistant:  FINAL ASSESSMENT (complete): [matches target — 2 sentences grounded in
            the residual delta summary]
            — or —
            FINAL ASSESSMENT (budget_exhausted): [what matches; specific remaining
            residual cited from the residual delta summary]
```

**Key design decisions:**
- **Claw-code-style skill architecture.** Per-plugin knowledge lives in `skills/<plugin>/` bundles (`SKILL.md` + `scripts/` + `references/`). Main agent opens each session with a discovery turn (`bash ls skills/*/SKILL.md`) followed by a `Skill(name)` tool call that injects the SKILL.md contents into the conversation. Files are the API — no `pyproject.toml` entry points, no install step — so agent-authored skills (future plugin-explorer task) will work identically to pre-shipped ones.
- **Three-tier agent hierarchy.** Main agent orchestrates; dispatches 4 parallel `wavetable_search` sub-agents across library slices; dispatches a `wavetable_judge` sub-agent to audition the combined pool and pick the final tuple. All file-based handoff (`/tmp/agents/<sample>/*.json`). Sub-agents start with fresh context; the main agent extracts what they need into their dispatch prompts (scripts, target audio path, n_osc_slots, etc.) rather than propagating the full SKILL.md.
- **Subsystem batches, not per-param steps.** One batch per presentation subsystem (oscillator, envelope, filter, lfo, fx, modulation, macro). Params are bucketed by `_param_family()` — guarantees 100% batch-param alignment.
- **Fresh vita-rendered audio per batch.** Each batch's audio reflects exactly the cumulative preset state after that batch.
- **Inline mistake correction.** ~20% of samples get a deliberate overshoot in one param of one batch. The correction fires immediately after the listen.
- **No-audio edge case (~5%).** User sends "Recreate this sound in Vital." without attaching audio → single-turn refusal asking them to select an audio clip in REAPER. Teaches the model to recognise the missing attachment instead of fabricating a target. Flagged via `meta.variant='no_audio_selected'`.
- **Preset-grounded Stage 1 observations.** Omni listens to the target WAV with a perceptual-bucket summary of the target preset injected as a grounding prior (`scripts/preset_perceptual_summary.py: summarize_preset_perceptual`). Avoids the multi-audio target/default comparison that was out-of-distribution for Qwen-Omni and produced hallucinated modulation and flipped attack shapes. The summary is no-numbers/no-param-names — just producer vocabulary.
- **Plan + param-delta driven batch narrations.** Each per-batch narration receives the plan bullet for its subsystem and the concrete before→after param deltas. Narrations are plan-aligned by construction (no more "LFO drifts with human-like unpredictability" when the plan says "disengage all modulation"). Natural variety comes from different presets having genuinely different deltas — no descriptor-lens bank needed.
- **Residual-delta-grounded verdict.** FINAL ASSESSMENT is anchored on `summarize_residual_delta_perceptual(target_preset, final_preset)` — the top 5 concrete residual differences between target and final, ranked by magnitude, rendered as perceptual bullets. Verdicts cite "attack is still too plucky" or "filter should be darker" instead of defaulting to "envelope 6".

Total audio per record: 2 + B clips (GT + default + B batch listens, where B ≈ 5–7) + optional correction listen. At ~94.5 tokens/sec, a 7-batch conversation runs ~7–9K audio tokens — ~40% less than v2.

**Step 3 — grade and filter:**

```bash
# Main agent — with LLM-as-judge (recommended) — ~35s for 8 samples at --workers 8:
python scripts/grade_agent_sft.py \
    --input data/prepared/agent_sft/main_v3.jsonl \
    --output data/prepared/agent_sft/main_v3_graded.jsonl \
    --llm-judge-server http://localhost:8000 \
    --llm-judge-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --workers 8 --verbose

# Main agent — structural-only (fast, no LLM calls):
python scripts/grade_agent_sft.py \
    --input data/prepared/agent_sft/main_v3.jsonl \
    --output data/prepared/agent_sft/main_v3_graded.jsonl \
    --no-llm-judge --verbose

# Search agent (task_type=search_v2) — structural/correctness scoring, no LLM:
python scripts/grade_agent_sft.py \
    --input data/prepared/agent_sft/search_v2.jsonl \
    --output data/prepared/agent_sft/search_v2_graded.jsonl \
    --no-llm-judge --verbose

# Judge agent (task_type=judge, v3_judge) — structural/correctness scoring, no LLM:
python scripts/grade_agent_sft.py \
    --input data/prepared/agent_sft/judge_v3.jsonl \
    --output data/prepared/agent_sft/judge_v3_graded.jsonl \
    --no-llm-judge --verbose
```

The grader dispatches on `task_type` + `meta.pipeline_version` — one entry point for all three agent types.

**Scoring dimensions for v3 `main` records:**

Structural axes (always computed):

| Dimension | Weight | What it measures |
|---|---|---|
| `batch_param_alignment` | 15% | Every param in each batch's set_params call matches the subsystem label |
| `diagnosis_subsystem_coverage` | 10% | F1 of subsystems named in PLAN vs subsystems that actually differ |
| `clap_net_improvement` | 15% | Net CLAP cosine improvement from first batch to last |
| `verdict_grounded` | 5% | FINAL ASSESSMENT names a residual subsystem |
| `snake_case_clean` | 2.5% | No raw snake_case param names in assistant text |
| `format_consistent` | 2.5% | No **BOLD:** headers |
| `mistake_recovery` | 5% (conditional) | Correction block appears and targets the correct param |

LLM-judge axes (added when `--llm-judge-server` is set):

| Dimension | Weight | What it measures |
|---|---|---|
| `llm_observations_audio_grounded` | 10% | Omni listens to the target WAV and judges whether the OBSERVATIONS text matches what's audibly present |
| `llm_narration_no_hallucination` | 10% | Per-batch binary check: does the narration reference any param family not in the batch's `param_names`? |
| `llm_narration_plan_ref` | 8% | Does the narration pick up intent from the plan's subsystem bullet? |
| `llm_narration_param_specific` | 7% | Does the narration name audible consequences tied to the specific params edited? |
| `llm_narration_templateness` | 5% | **Cross-sample binary** — compared against reference narrations from the same subsystem in other samples; scores 1 if phrasing is sample-specific, 0 if swappable template |
| `llm_verdict_residual_grounded` | 7% | Does the verdict cite a concrete residual difference (not just "envelope N")? |
| `llm_verdict_novelty` | 5% | **Cross-sample binary** — compared against 3 other samples' verdicts; scores 1 if sample-specific, 0 if template-shaped |

`execution_fidelity` (from `--live-exec-check`) validates that generated bash tool calls run against a live REAPER session.

**Scoring dimensions for `search_v2` records** (structural + correctness, no LLM needed):

| Dimension | Weight | What it measures |
|---|---|---|
| `gt_recovery` | 35% (conditional) | Fraction of `meta.gt_in_shard` that made it onto `meta.final_shortlist`. Only assessable when the slice contained a GT. |
| `shortlist_file_written` | 25% | Last bash tool_call writes `*_search_*.json` + matching tool_response returns `{status:"ok", file:...}` |
| `closing_assistant` | 10% | Last message is an assistant turn (signals task completion) |
| `has_render_probes` | 10% | ≥1 bash tool_call invokes `skills/vital/scripts/render_probes.py` (agent actually auditioned) |
| `shortlist_nonempty` | 10% | Final shortlist has ≥1 name |
| `snake_case_clean` | 5% | No raw snake_case param names in assistant prose |
| `format_consistent` | 5% | No `**BOLD:**` headers |

**Scoring dimensions for `judge` records (v3_judge)** (structural + correctness, no LLM needed):

| Dimension | Weight | What it measures |
|---|---|---|
| `judge_correct` | 30% | `meta.judge_correct`: selection matches GTs present in pool (oracle) |
| `tuple_size_correct` | 10% | `len(selected_tuple) == n_osc_slots` |
| `tuple_names_in_pool` | 10% | Every selected name is in the pool (no hallucinated wavetables) |
| `output_file_written` | 25% | Last bash tool_call writes judge JSON (`{tuple, n_osc_slots, reasoning}`) + ok tool_response |
| `pool_candidates_discussed` | 10% | Fraction of pool names mentioned in the judge's deliberation |
| `has_render_probes` | 5% | Agent actually rendered pool probes |
| `closing_assistant` | 5% | Last message is assistant |
| `format_consistent` | 2.5% | No `**BOLD:**` headers |
| `snake_case_clean` | 2.5% | No raw snake_case (note: agent IDs with underscores are a known false-positive source) |

**Scoring dimensions for `melody_transcription` records** (structural + oracle correctness, no LLM needed):

| Dimension | Weight | What it measures |
|---|---|---|
| `has_reapy_midi_insert` | 20% | ≥1 bash tool_call contains `MIDI_InsertNote` (or `RPR_MIDI_InsertNote`) — the subagent actually emits reapy insert code |
| `output_file_written` | 25% | Final bash tool_call writes `transcription.json` with `{notes, n_notes, duration_s}` schema + matching ok tool_response |
| `note_count_match` | 20% | N notes in the payload matches `meta.n_notes` (oracle count from the source MIDI) |
| `pitch_coverage` | 10% | Fraction of oracle MIDI pitches mentioned in deliberation or insert command |
| `has_render_listen` | 10% | First user message carries `<audio>` (subagent received the target for listening) |
| `closing_assistant` | 5% | Last message is assistant |
| `snake_case_clean` | 5% | No snake_case in assistant prose |
| `format_consistent` | 5% | No `**BOLD:**` headers |

**Benchmarks on smoke_test_v10 (n=8 samples, full four-agent pipeline):**

Main agent (`--llm-judge-server`, LLM-judge axes):

| Build | overall | audio_grounded | verdict_novelty | verdict_grounded | templateness |
|---|---|---|---|---|---|
| v2 original (pre-grounding, lens bank off) | 0.708 | 0.125 | 0.375 | 0.688 | 0.323 |
| v5 (grounded Stage 1) | 0.776 | 0.438 | 0.500 | 0.750 | 0.583 |
| v7 (plan+delta narrations, lens bank removed) | 0.762 | 0.625 | 0.500 | 0.625 | 0.542 |
| v8 (+ residual-delta verdict) | 0.781 | 0.312 | 0.750 | 0.812 | 0.771 |
| **v13** (+ Skill discovery + load protocol) | **0.839** | — | — | — | — |

Sub-agent rollouts (structural + correctness grading, n=32 search × n=8 judge):

| Rollout | overall | gt_recovery / judge_correct | file_written | pool_discussed |
|---|---|---|---|---|
| Search v12 | **0.994** | gt_recovery 1.00 on all 10 assessable samples | 1.00 | — |
| Judge v12 | **0.992** | judge_correct 1.00 on all 8 samples | 1.00 | 1.00 |

Structural grader on legacy smoke_v3 (n=32, older pre-grounding, main agent only): overall mean 0.930, batch_param_alignment 1.0, execution_fidelity 1.0, diagnosis_subsystem_coverage 1.0. That number was dominated by structural floors; the LLM-judge scores above give a truer picture of text-quality.

**Audio-grounding spot-check HTML:** `scripts/build_audio_grounding_spotcheck.py` renders a self-contained HTML (audio embedded as base64) with target + default playback, preset summary, observations, per-batch narrations with judge badges. Supports side-by-side comparison between two graded files (`--compare-grades`).

**Smoke test** (8 samples end-to-end, including LLM-judge grading):

```bash
python scripts/render_iter_presets.py --generate 8 \
    --output-dir outputs/smoke_test --wavetable-lib data/wavetable_lib.json

python scripts/build_main_agent_sft_v3.py \
    --manifest outputs/smoke_test/manifest.jsonl \
    --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
    --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
    --wavetable-lib data/wavetable_lib.json \
    --out-jsonl outputs/smoke_test/main_v3.jsonl \
    --max-samples 8 --workers 8 --max-batches 8 --mistake-rate 0.20 \
    --force-research-rate 0.80 \
    --omni-server http://localhost:8000 \
    --omni-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --stage2-server http://localhost:8000 \
    --stage2-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --clap-device cuda:0

python scripts/grade_agent_sft.py \
    --input outputs/smoke_test/main_v3.jsonl \
    --output outputs/smoke_test/graded_v3.jsonl \
    --llm-judge-server http://localhost:8000 \
    --workers 8 --verbose

# Optional: self-contained HTML with audio + per-sample scores for spot-check
python scripts/build_audio_grounding_spotcheck.py \
    --grades outputs/smoke_test/graded_v3.jsonl \
    --out outputs/smoke_test/spotcheck.html
```

Expected runtime on a single 4×3090 rig with vLLM serving Qwen3-Omni at
MAX_NUM_SEQS=24: build ~2 min, grade ~35s.

### Iterative preset recreation pipeline (main-agent SFT v2, legacy)

Legacy pipeline. Generates multi-turn conversations where an audio-language model listens to a target sound, reasons about what parameters differ, and iteratively adjusts a Vital preset step-by-step until it converges. Superseded by v3 which eliminates per-step commentary confabulation.

**Step 1 — generate path data and render audio** (no GPU needed):

```bash
python scripts/render_iter_presets.py \
    --generate 1000 \
    --archetypes bass lead pad keys pluck sequence \
    --output-dir outputs/iter_sft \
    --wavetable-lib data/wavetable_lib.json \
    --jobs 24
```

This generates `N` synthetic (target preset, parameter path, rendered audio) tuples. For each sample it produces:
- `{id}_gt.wav` — ground truth audio (target preset, musical note sequence)
- `{id}_default.wav` — Vital default preset baseline probe
- `{id}_step{N}.wav` — cumulative preset probe after step N
- `{id}_target.vital` — target preset (saved separately, not embedded in path JSON)
- A `manifest.jsonl` with one entry per sample pointing to all the above paths

Path lengths scale with the number of differing parameters (8–20 steps). Params are applied in priority order (oscillators → envelopes → filters → modulation routing) to mirror how a skilled sound designer approaches the problem.

**Step 2 — build SFT conversations** (requires Qwen3-Omni at localhost:8000):

```bash
python scripts/build_main_agent_sft_v2.py \
    --manifest outputs/iter_sft/manifest.jsonl \
    --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
    --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
    --wavetable-lib data/wavetable_lib.json \
    --out-jsonl data/prepared/agent_sft/main.jsonl \
    --max-steps 24 \
    --omni-server http://localhost:8000 \
    --omni-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --commentary-mode two_stage \
    --stage2-server http://localhost:8000 \
    --stage2-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --audio-gate-threshold 0.01 \
    --clap-device cuda:0
```

Each output record is a multi-turn ms-swift JSONL conversation. `--max-steps` caps how many iteration turns appear per conversation. If the path has more steps than the cap, the conversation ends with a `FINAL ASSESSMENT (budget exhausted):` verdict; if the path runs to its planned end, it ends with `FINAL ASSESSMENT (complete):`. Aim for ~25–35% complete examples for balanced training — use `--max-steps` relative to path lengths in your manifest.

**Conversation structure:**

```
user:       <audio> [GT clip]  "Recreate this {archetype} sound in Vital."
assistant:  Spawning search shards...
            ...wavetable search/judge turns...
assistant:  → tool_call: render default preset (baseline listen)
tool_resp:  <audio> [default clip]

# Per iteration step — two variants:

# LISTENING step (|CLAP delta| >= threshold — audio changed materially):
# If planning steps preceded this listen, a catch-up probe is emitted first:
assistant:  "Listening to accumulated changes after step(s) K–N-1."  [only if steps_since_last_listen > 1]
tool_call:  bash [render probe of accumulated state]
tool_resp:  {"status": "ok", "iter_audio": "<audio>", ...}
# HEARD always immediately follows <audio>:
assistant:  HEARD: [what changed since last listen + remaining timbral gap]
            HYPOTHESIS: [perceptual quality still separating current from target — audio observation, not a prescription]
            PLAN: Adjusting {param list}. [one sentence future-tense rationale]
            "Executing step N parameter updates now."
tool_call:  bash [set params]
tool_resp:  {"status": "ok"}
assistant:  "Listening to updated preset after step N."
tool_call:  bash [render probe]
tool_resp:  {"status": "ok", "iter_audio": "<audio>", ...}

# PLANNING step (|CLAP delta| < threshold — sub-perceptual edit):
assistant:  PLAN: Adjusting {param list}. [rationale for structural setup]
            "Executing step N parameter updates now."
tool_call:  bash [set params]
tool_resp:  {"status": "ok"}

# After all steps:
assistant:  FINAL ASSESSMENT (complete): [2-sentence perceptual summary]
            — or —
            FINAL ASSESSMENT (budget exhausted): [summary of progress and remaining gap]
```

The **audio gate** (`--audio-gate-threshold`, default 0.01) decides which variant each step gets. Steps where |CLAP cosine delta| < threshold are planning-only — no audio listen, no HEARD/HYPOTHESIS confabulation. This mirrors how Claude Code narrates a plan before writing code, and eliminates the "I heard nothing change" hallucination pattern on sub-perceptual edits (~40–50% of steps, mostly LFO/modulation routing).

Total audio per record: 2 + L clips (GT + default probe + L *listening* step probes, where L ≤ N). GT audio is heard exactly once — at the start of block 0. At ~94.5 tokens/sec, a 20-step conversation with ~50% listening rate runs ~10–13K tokens total.

**Two-stage commentary** is the recommended mode. Stage 1 calls Omni with audio to produce perceptual observations (what changed, what gap remains). Stage 2 calls a text model to write the three commentary sections. Planning steps skip Stage 1 entirely — a text-only Stage 2 call writes PLAN only. This separation keeps audio-grounded reasoning clean and avoids audio API calls for steps where nothing changed.

**Commentary section roles** (listening steps):
- **HEARD**: what the last edit (or accumulated edits since last listen) changed + the single most important remaining timbral gap, described as a positive quality the target has.
- **HYPOTHESIS**: one sentence describing the most important perceptual quality still separating current from target, grounded in what was just heard. Uses hedged language ("appears to", "seems to"). Does NOT prescribe a next step or name a parameter family — that's PLAN's job.
- **PLAN**: Sentence 1 is the exact param inventory ("Adjusting Filter 1 Cutoff, Resonance."). Sentence 2 is a future-tense rationale for how these changes address the remaining gap.

**GT-preset grounding**: at each step, the builder computes which subsystems still differ most between the cumulative preset and the ground-truth target. The remaining delta is injected into the Stage 2 prompt to ground HEARD Sentence 2 in parameter-space truth. These are stored in `meta.step_labels[].remaining_top_2` for offline grading.

**GT re-anchor** (`--reanchor-gt-audio`, default off): by default the GT audio is only included once in the conversation (block 0 preamble). Subsequent chunked blocks start with current-state audio only — the model is expected to hold the target in memory. Pass `--reanchor-gt-audio` to re-attach GT at the start of every block (old behaviour).

**Step 3 — grade and filter** (requires Qwen3-Omni at localhost:8000):

```bash
python scripts/grade_agent_sft.py \
    --input data/prepared/agent_sft/main.jsonl \
    --output data/prepared/agent_sft/main_graded.jsonl \
    --llm-judge-server http://localhost:8000 \
    --min-score 0.85 \
    --verbose
```

**Scoring dimensions for `main` records:**

| Dimension | Weight | What it measures |
|---|---|---|
| `clap_net_improvement` | 30% | Did the path make net progress toward GT audio? (final − initial CLAP cosine delta) |
| `plan_rationale_unique` | 25% | PLAN Sentence 2 rationale is distinct across steps (not boilerplate) |
| `hypothesis_grounding` | 25% | HYPOTHESIS contains a perceptual quality term AND hedged language, does not prescribe a next step (listening steps only) |
| `section_structure` | 10% | All 3 headers present in every listening step (planning steps are exempt by design) |
| `snake_case_clean` | 5% | No raw snake_case param names in commentary text |
| `format_consistent` | 5% | No **BOLD:** headers |

`overall` is a weighted sum. `commentary_diversity` (1 − mean pairwise Jaccard) and `plan_param_alignment` are computed and stored for diagnostics. Benchmark on smoke_coherence_v3 (n=4, with causal-coherence fixes): overall mean **0.882**, commentary_diversity 0.761, hypothesis_grounding 0.880.

**Smoke test** (12 samples, fast iteration):

```bash
# Generate 12 path/audio samples
python scripts/render_iter_presets.py --generate 12 \
    --output-dir outputs/smoke_test --wavetable-lib data/wavetable_lib.json

# Build conversations
python scripts/build_main_agent_sft_v2.py \
    --manifest outputs/smoke_test/manifest.jsonl \
    --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
    --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
    --wavetable-lib data/wavetable_lib.json \
    --out-jsonl outputs/smoke_test/main_agent_smoke.jsonl \
    --max-samples 12 --max-steps 24 \
    --omni-server http://localhost:8000 \
    --omni-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --commentary-mode two_stage \
    --stage2-server http://localhost:8000 \
    --stage2-model Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --audio-gate-threshold 0.01 \
    --clap-device cuda:0

# Grade
python scripts/grade_agent_sft.py \
    --input outputs/smoke_test/main_agent_smoke.jsonl \
    --output outputs/smoke_test/graded.jsonl \
    --llm-judge-server http://localhost:8000 \
    --verbose
```

### Search agent SFT v2 (iterative batch-listening wavetable search)

Trains a search agent to evaluate wavetable candidates by ear through iterative batch listening. Each search agent hears the target audio + batches of candidate wavetable probes (audio + name), builds a running shortlist, and returns candidate names for the main agent to assemble into tuples.

```bash
python scripts/build_search_agent_sft_v2.py \
    --manifest outputs/iter_sft/manifest.jsonl \
    --index-npy outputs/wt_retrieval_baseline/wt_index.npz \
    --index-meta outputs/wt_retrieval_baseline/wt_index_meta.json \
    --wavetable-lib data/wavetable_lib.json \
    --out-jsonl data/prepared/agent_sft/search_v2.jsonl \
    --pool-top-k 48 --num-agents 4 --candidates-per-batch 8 \
    --omni-server http://localhost:8000 \
    --omni-model Qwen/Qwen3-Omni-30B-A3B-Instruct
```

**Key design features:**
- **Candidate pool** built from GT wavetable CLAP embeddings (apples-to-apples index comparison, top-K=48). CLAP is used only at build time for pool construction — the model never sees or uses CLAP.
- **Iterative batch listening**: target + 6-8 candidates per round, multiple rounds, shortlist evolves across rounds.
- **GT grounding**: when a GT wavetable appears, Stage 2 receives the target preset's processing chain (`describe_key_transforms`) and writes reasoning about how the raw wavetable transforms under that processing. The model learns to reason about raw→processed transformation.
- **Names in token space**: each candidate has both `<audio>` and its wavetable name, so the model can reference it in code.
- **No CLAP, no role-tagging, no tuple assembly** — just a shortlist of wavetable names that "sound like they belong."

### Legacy hierarchical builders (search v1 / judge — superseded)

Earlier search and judge agent builders used pre-computed CLAP rankings with template proposals. Superseded by search agent v2 (above) which teaches genuine perceptual comparison. Legacy scripts retained for reference:
- `scripts/build_search_agent_sft.py` (v1, template proposals)
- `scripts/build_judge_agent_sft.py` (listwise ranking from CLAP scores)

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

### Megatron Sequence-Parallel Findings (April 8, 2026)

All runs below used 4x24GB GPUs with LoRA (`rank=8`, `alpha=32`, `target_modules=all-linear`), `tensor_model_parallel_size=4`, `sequence_parallel=true`, `micro_batch_size=1`, packing enabled, and flash attention.

`gbs` = `global_batch_size`.

#### Qwen2.5-Omni-3B

- 1-step probes (`gbs=4`) validated long context beyond 32k.
- Verified passes at `32768`, `36864`, and `40960`.
- `45056` was interrupted (SIGTERM), so the true upper bound was not finalized in that sweep.
- Reference logs: `outputs/qwen25_omni_lora_megatron_probe/probe_20260408_171119/`.

#### Qwen2.5-Omni-7B

- `32768` succeeded for both:
  - 1 step (`train_iters=1`), and
  - 4 steps (`train_iters=4`) with checkpoint-4 written.
- Observed training memory at 32k was ~`10.6 GiB` per GPU in these runs.
- Artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_single/len32768_localhf/v0-20260408-173018/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni7b_single/len32768_localhf_4steps/v0-20260408-174044/`

#### Qwen3-Omni-30B-A3B-Instruct (MoE)

- Important: on 4 GPUs, this model required `--expert_model_parallel_size 4` for stable Megatron execution.
- With `gbs=4`, 1-step search:
  - `16384` pass
  - `18432` pass
  - `18688` OOM
- Separate `gbs=4`, 4-step attempts at `16384` and `18432` did not complete to checkpoint in this session (runs ended early), so stability at `gbs=4` was not confirmed.
- With `gbs=1`, 4-step stability search:
  - `16896` pass
  - `17152` pass
  - `17280` pass
  - `17408` OOM
  - `17344` interrupted before completion
- Current best verified 4-step stable context for 30B (`gbs=1`): **`17280`**.
- Additional 4-step checks (same setup) that failed under memory pressure:
  - `20000` OOM (failed after step 1)
  - `19456` OOM (failed after step 1)
  - `18432` failed at/after step 2 (rank crash under pressure)
- Reference logs:
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_seqfind_ep4/20260408_175533/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_maxctx_gbs1_4steps/20260408_194108/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_single/`

#### QLoRA Note (Megatron Backend Behavior)

- We tested `--quant_method bnb --quant_bits 4 --bnb_4bit_quant_type nf4 --bnb_4bit_use_double_quant true` on the same 30B Megatron setup.
- In an apples-to-apples A/B at `max_length=16000`, `train_iters=1`:
  - LoRA run: `memory(GiB)=21.21`
  - "QLoRA" flag run: `memory(GiB)=21.23`
- The live model structure still showed Megatron/TransformerEngine linear layers (`TE*` with `LoraParallelLinear`), not bitsandbytes `Linear4bit` modules.
- Practical takeaway for this repo right now: on Megatron Omni path, treat `bnb` quant flags as offering no reliable VRAM savings unless backend support is explicitly confirmed.
- A/B artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/lora_len16000_1step/`
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_lora_vs_qlora_ab/qlora_len16000_1step/`

#### TP/CP Topology Probe (TP=2, CP=2)

- We also tested `tensor_model_parallel_size=2` + `context_parallel_size=2` on the same 30B setup.
- Outcome in this environment was unstable:
  - `max_length=20000`, `train_iters=1` failed before a completed train-step metric with
    `torch.distributed.DistBackendError` and `Failed to CUDA calloc async 4 bytes` (rank 2).
  - `max_length=16000`, `train_iters=1` reached `iteration 1/1` (`memory(GiB)=21.84`) but then failed during distributed collectives with
    `Failed to CUDA calloc async 40 bytes`.
- Practical takeaway for now: this TP/CP topology is not yet a reliable path to longer context on this box without deeper NCCL/memory tuning.
- Why this can happen even though CP should help long-context in theory:
  - `CP=2` introduces additional distributed collectives; failures occurred inside those collectives (`broadcast`/`all_reduce`), not in forward math.
  - `TP=2` changes per-rank tensor shard shapes vs `TP=4`; some temporary/comm buffers can become less favorable.
  - With MoE (`expert_model_parallel_size=4`) layered on top, communicator complexity is higher, and this stack appears fragile in this topology on this machine.
- Artifacts:
  - `outputs/qwen25_omni_lora_megatron_probe/omni30b_tp_cp_experiments/`

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

### Vital plugin setup (required for live-exec grading)

**Vital's Linux-native builds (VST2/VST3/CLAP, all versions through 1.6.0) crash REAPER** when loading preset state containing any real wavetable. The crash fires inside Vital's own state deserializer — it's not a transport issue and can't be worked around via chunked writes or file-handoff. Any `VitalController.set_preset()` call with a non-trivial wavetable will segfault REAPER.

**Workaround:** run Vital as a Windows plugin bridged through [yabridge](https://github.com/robbert-vdh/yabridge) + WINE. This is a **hard dependency** for live-exec grading (`scripts/grade_agent_sft.py --live-exec-check`) and for any agent rollout that applies wavetables to a live REAPER session.

```bash
# 1. WINE staging
sudo dpkg --add-architecture i386
sudo mkdir -pm755 /etc/apt/keyrings
sudo wget -O /etc/apt/keyrings/winehq-archive.key https://dl.winehq.org/wine-builds/winehq.key
sudo wget -NP /etc/apt/sources.list.d/ https://dl.winehq.org/wine-builds/ubuntu/dists/noble/winehq-noble.sources
sudo apt update
sudo apt install --install-recommends winehq-staging

# 2. yabridge (user-local)
mkdir -p ~/.local/share/yabridge ~/.local/bin
# Download latest yabridge release tarball from https://github.com/robbert-vdh/yabridge/releases
tar xzf yabridge-*.tar.gz -C ~/.local/share/yabridge --strip-components=1
ln -sf ~/.local/share/yabridge/yabridgectl ~/.local/bin/yabridgectl

# 3. Install Windows Vital via WINE (requires VitalInstaller.exe from vital.audio)
#    In the installer component screen, check: VST3, VST, CLAP
wine ~/VitalInstaller.exe

# 4. Bridge the plugins into Linux plugin paths
yabridgectl add "$HOME/.wine/drive_c/Program Files/Common Files/VST3"
yabridgectl add "$HOME/.wine/drive_c/Program Files/Common Files/CLAP"
yabridgectl add "$HOME/.wine/drive_c/Program Files/Steinberg/VstPlugins"
yabridgectl sync

# 5. Move aside any Linux-native Vital so REAPER picks the bridged one
for d in ~/.vst3/Vital.vst3 ~/.vst/Vital.so ~/.clap/Vital.clap; do
    [ -e "$d" ] && mv "$d" "$d.linux-native-bak"
done
```

Restart REAPER; it will scan and register the bridged plugins under the usual "Vital (Vital Audio)" names.

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
  render_iter_presets.py      # Batch-render GT + probe clips; write manifest.jsonl
  build_main_agent_sft_v3.py  # Primary SFT builder: diagnose → subsystem-batched execute
  build_main_agent_sft_v2.py  # Legacy SFT builder: per-step two-stage commentary
  grade_agent_sft.py          # Quality grader: main (structural + 7-axis LLM-judge), search_v2, judge_v3
  preset_perceptual_summary.py # Perceptual-bucket preset summaries (grounding prior for Stage 1) + residual-delta summary (grounding for verdict)
  build_audio_grounding_spotcheck.py # Self-contained HTML spot-check (audio embedded, compare mode)
  validate_grounded_observations.py  # A/B runner: current vs preset-grounded Stage 1 observations
  agent_sft_common.py         # Shared helpers: CLAP embedder, candidate pool, GT-similarity pool
  build_search_agent_sft_v2.py # Search-agent SFT v2: iterative batch-listening (--workers for entry + agent parallelism)
  build_judge_agent_sft_v3.py # Judge-agent SFT v3: audition combined search pool in one Omni call, pick final tuple, write output file
  build_transcription_agent_sft_v3.py # Melody-transcription SFT: listen to target, write reapy code that inserts MIDI notes on a REAPER track, save note list JSON
  build_search_agent_sft.py   # Search-agent SFT v1 (legacy: template proposals)
  build_judge_agent_sft.py    # Judge-agent SFT (legacy: listwise ranking from CLAP)
  merge_agent_sft.py          # Merge task JSONL files, shuffle
  experiment_clap_wt_threshold.py  # CLAP GT-to-index threshold experiment
  experiment_omni_batch_listen.py  # Omni batch audio comparison experiment
  build_iter_sft_dataset.py   # Earlier single-agent SFT assembler (Omni commentary)
  demo_iter_examples.py       # Generate demo conversations + render all audio
  verify_preset_path.py       # Diagnostic: compare target vs final cumulative preset
  reaper_render_probe.py      # Standalone: render REAPER track to /tmp/probe.wav
  render_vital_wavs.py        # General-purpose preset render script
  check_preset_diversity.py   # CLAP diversity audit
  benchmark_render.py         # Worker sweep benchmarks
  train_qwen25_omni_lora.py   # MS-Swift LoRA training launcher

tests/
  test_search_agent_sft_v2.py  # Contract tests for search agent v2 (batch listening, shortlists)
  test_agent_sft_contracts_v3.py # Contract tests for v3 main agent (subsystem batches, WT scaffold)
  test_agent_sft_contracts.py # Contract tests for v2 main agent (legacy)
  test_agent_sft_grading.py   # Tests for grade_agent_sft scoring logic (v2 + v3)
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
- Wavetable library builder and loader (568 unique wavetables)
- N-step parameter path generator (`path_gen.py`) with noise/mistake injection and final-step convergence
- `render_iter_presets.py` — batch render of GT, default, and per-step probe clips; writes `manifest.jsonl`
- `build_main_agent_sft_v3.py` — **primary pipeline**: diagnose → subsystem-batched execute with fresh vita-rendered per-batch audio, inline mistake correction, producer-style plan-then-execute flow, GT-CLAP-similarity pool + tuple render+listen WT scaffold. **Preset-grounded Stage 1** (target-only audio + perceptual preset summary), **plan + param-delta driven narrations** (no descriptor-lens bank), **residual-delta grounded verdict**. Entry-level `--workers` concurrency (8×4 agents saturates vLLM's 24 slots). Overall 0.781 on n=8 LLM-judge smoke.
- `preset_perceptual_summary.py` — perceptual-bucket preset summaries (no numbers, no param names) used as a grounding prior by Stage 1 observations; `summarize_residual_delta_perceptual` feeds the final verdict
- `build_search_agent_sft_v2.py` — search agent with iterative batch-listening, GT-grounded processing-chain reasoning, dynamic per-sample transform descriptions; entry-level + agent-level concurrency via `--workers`
- `build_judge_agent_sft_v3.py` — judge agent's own SFT rollouts. Takes combined pool from all search agents (simulated at build time), renders probes, listens to target + all pool candidates in one Omni call (each `<audio>` labelled by wavetable name), picks the best N (=active oscillator slots), writes selection to output file that the main agent consumes via `cat`. Build-time oracle: GT-if-in-pool + CLAP-best-proxy
- `build_transcription_agent_sft_v3.py` — melody-transcription subagent's SFT rollouts. Loads `source_midi_path` from the manifest as the oracle note list, combines with an Omni perceptual impression (contour + rhythm feel) to form a per-note narration, then writes the bash `reapy` code that inserts the notes via `RPR_MIDI_InsertNote` at PPQ positions (same convention as the legacy Lua example). Output JSON `{notes, n_notes, duration_s}` lands at `/tmp/agents/<sample_id>/transcription.json` for the main agent to `cat`. The main agent dispatches this subagent right after the skill-load turn, before the library-size check.
- `build_main_agent_sft_v2.py` — legacy per-step pipeline (superseded by v3)
- `grade_agent_sft.py` — grades all three agent types via one entry point (dispatched on `task_type` + `meta.pipeline_version`). Main-agent v3: structural axes (batch_param_alignment, diagnosis_subsystem_coverage, clap_net_improvement, verdict_grounded, mistake_recovery) + 7-axis LLM judge (narration plan_ref / param_specific / templateness-cross-sample / no-hallucination, observations audio-grounded, verdict residual_grounded / novelty-cross-sample). Search v2: gt_recovery (conditional), shortlist_file_written, has_render_probes, closing_assistant. Judge v3: judge_correct (oracle), tuple_size_correct, tuple_names_in_pool, output_file_written, pool_candidates_discussed. Live-exec-check against REAPER available for main-agent records.
- `build_audio_grounding_spotcheck.py` — self-contained HTML spot-check with embedded audio + judge-badge breakdown; `--compare-grades` renders two graded files side-by-side
- Legacy wavetable-retrieval SFT builders (search v1 / judge — superseded by search v2)
- Lua tuple pipeline for melody transcription SFT data
- MS-Swift LoRA training scripts for Qwen2.5-Omni

**Planned / not yet implemented:**
- **Plugin-explorer agent task** — SFT data for an agent that meets an unfamiliar plugin, systematically probes it (parameter sweeps, structured listens, hypothesis-forming), then writes its own `skills/<plugin>/SKILL.md` + bundled helper scripts. Closes the loop on the agent-writes-its-own-skill vision and unlocks generalisation beyond Vital without per-plugin training curation.
- Second-plugin proof-of-generalisation — even a simple subtractive synth's SFT data would validate whether the strategy layer (`listen → decompose → search → audition → apply → verify`) actually transfers across plugins or whether we're over-fit to Vital's vocabulary.
- **Longer-audio melody transcription** — `melody_transcription` currently ships with a ~30 s audio cap (the `vita` render ceiling). Longer clips would need parallel-slice transcription (multiple subagents on overlapping chunks, merged results) or a streaming-chunk backend.
- **Error-correction transcription review** — main agent could re-invoke `melody_transcription` after hearing a wrong note in the recreation, to fix individual notes. Currently transcription is one-shot.
- Agent inference loop (the trained model running against a live REAPER session; see `/home/nate/Documents/maestro-reaper-plugin/`)
- REAPER-bench for RLVR
- RL training stage

---

## Related Work

- [DAWZY (Elkins et al., NeurIPS 2025)](https://arxiv.org/abs/2512.03289) — LLM-based natural language control of REAPER
- [Voyager (Wang et al., 2023)](https://arxiv.org/abs/2305.16291) — skill library accumulation for open-ended embodied agents
- [Agent-RLVR (Scale AI, 2025)](https://arxiv.org/abs/2506.11425) — RLVR with guidance hints for software engineering agents
- [LAION CLAP](https://github.com/LAION-AI/CLAP) — contrastive language-audio pretraining
- [vita Python bindings](https://github.com/andrewjjenkins/vita) — direct C++ Vital engine access (offline rendering)
