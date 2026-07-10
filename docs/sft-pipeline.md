# SFT Training Data Pipeline

[← Back to README](../README.md)

Generating SFT data does not require REAPER or a GPU. The pipeline uses **vita** (Python bindings for the Vital synthesis engine) for build-time audio rendering, and **DawDreamer** (in-process VST3 host) for the probe/tuple rendering snippets embedded in generated conversations.

```
preset_gen.py       →  target Vital preset (6 archetypes)
path_gen.py         →  N-step parameter path: default → target
demo_iter_examples.py  →  renders all clips + builds agentic tool-call conversations
```

## Generate demo examples

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

## Verify preset fidelity

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

## Build full dataset (requires Omni at localhost:8000)

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

## Diagnose → subsystem-batched execute pipeline (main-agent SFT v3)

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

Target melodies come from the Lakh clip catalog by default
(`outputs/midi_clips/lakh_catalog.jsonl`); generation fails loudly if it's
missing. `--synthetic-melodies` opts into the legacy 4-triad pattern for
debug/smoke runs.

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

v3 does **not** use path_gen's per-step iterations for conversation structure. It diffs the target preset vs the init preset directly, buckets changed params by subsystem, and renders fresh per-batch audio using vita (build time). This eliminates cross-subsystem leakage from path_gen's support-family fill mechanism.

**Step 2b — real-DAW rollouts via daw-farm (recommended):**

With `--daw-farm`, every emitted tool-call snippet executes inside a real
REAPER instance from the [daw-farm](~/daw-farm) fleet instead of being
simulated: tool responses carry actual stdout/stderr from the container,
param searches hit the live TrackFX API, chunk applies mutate the live Vital,
and all listen audio is real REAPER timeline renders (or in-container
DawDreamer renders) fetched out of the session. Snippet paths point at the
container filesystem (`/work/rollouts/<sample_id>/`, `/tmp/agents/<sid>/`) —
the same view the model has at inference time.

```bash
# bring up sessions (one per --workers is ideal)
cd ~/daw-farm && docker compose up -d --scale reaper=8
# k8s alternative: ./bin/dawfarm new -c 8

# re-render stage A's model-visible audio (gt_wav/default_wav) through the
# environment itself — training targets must come from the same engine and
# render path the model acts in, not from vita
python scripts/rerender_manifest_dawfarm.py --manifest outputs/iter_sft/manifest.jsonl

python scripts/build_main_agent_sft_v3.py \
    ... same flags as above ... \
    --daw-farm docker \
    --workers 8
```

- `--daw-farm docker[:name1,name2]` / `--daw-farm k8s[:pod1,pod2]` — backend
  + optional explicit session list (default: discover all healthy sessions).
- `--daw-farm-vital-data` — host wavetable dir synced into each session.
  Point it at the generation library, not a personal Vital dir (same-name
  wavetables with different content silently corrupt rollout audio):
  `python scripts/export_wavetable_lib_dir.py` materializes
  `data/wavetable_lib.json` as `data/prepared/wavetable_lib_vital_dir`.
- `--daw-farm-timeout` (default 300 s) — per-snippet exec timeout.
- Sanity-check the fleet first: `python scripts/smoke_test_dawfarm.py`.

A sample whose snippet fails in the real environment (nonzero exit) is
dropped with a warning — surfacing environment bugs instead of baking
fabricated successes into training data. `meta.daw_farm_session` records
which session built each record.

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
tool_call:  bash python reapy → InsertTrackAtIndex + TrackFX_AddByName("Vital", ...)
tool_resp:  {"status":"ok","track_idx":0,"track_name":"target_melody"}
assistant:  Dispatching the transcription subagent to populate the track from the target.
tool_call:  Agent { subagent_type: "melody_transcription", ... }
tool_resp:  {agentId, outputFile: "/tmp/agents/<sid>/transcription.json", status: completed, n_notes, duration_s}

assistant:  MIDI ready on track 0 — 12 notes, ~5.0s. Checking wavetable library size.
tool_call:  bash python skills/vital/scripts/list_wavetables.py --total
tool_resp:  {"total": 282}
assistant:  Library has 282 wavetables. Dispatching 4 search agents in parallel...
            → 4 parallel Agent tool_calls (wavetable_search subagents)
            → bash cat to read all shortlist JSONs
            → dispatch Agent wavetable_judge to audition combined pool
            → bash cat judge output → render tuple → listen
            → accept (or re-search if target not found)
            → bash python reapy → apply VST chunk via TrackFX_SetNamedConfigParm

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
- **Fresh audio per batch.** Each batch's audio reflects exactly the cumulative preset state after that batch (vita at build time; DawDreamer in the generated snippets the model learns to write).
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

**Scoring dimensions for `search_v2` records** (structural + correctness + optional LLM audio grounding):

| Dimension | Weight | What it measures |
|---|---|---|
| `gt_recovery` | 35% (conditional) | Fraction of `meta.gt_in_shard` that made it onto `meta.final_shortlist`. Only assessable when the slice contained a GT. |
| `shortlist_file_written` | 20% | Last bash tool_call writes `*_search_*.json` + matching tool_response returns `{status:"ok", file:...}` |
| `llm_candidates_audio_grounded` | 15% (conditional) | Mean single-audio Omni grounding check across (candidate, description) pairs. Each check sends ONE candidate audio + its written description and asks whether the timbral claims are audible. Catches position-confusion hallucinations from multi-audio build-time calls. Requires `--llm-judge-server`. Sample rate controllable via `--audio-grounding-sample-rate`. |
| `has_render_probes` | 10% | ≥1 bash tool_call invokes `skills/vital/scripts/render_probes.py` (agent actually auditioned) |
| `shortlist_nonempty` | 10% | Final shortlist has ≥1 name |
| `closing_assistant` | 5% | Last message is an assistant turn (signals task completion) |
| `snake_case_clean` | 2.5% | No raw snake_case param names in assistant prose |
| `format_consistent` | 2.5% | No `**BOLD:**` headers |

**Scoring dimensions for `judge` records (v3_judge)** (structural + correctness + optional LLM audio grounding):

| Dimension | Weight | What it measures |
|---|---|---|
| `judge_correct` | 25% | `meta.judge_correct`: selection matches GTs present in pool (oracle) |
| `output_file_written` | 20% | Last bash tool_call writes judge JSON (`{tuple, n_osc_slots, reasoning}`) + ok tool_response |
| `llm_candidates_audio_grounded` | 15% (conditional) | Same single-audio grounding check as search, applied to per-pool-candidate descriptions in the deliberation. Requires `--llm-judge-server`. |
| `tuple_size_correct` | 10% | `len(selected_tuple) == n_osc_slots` |
| `tuple_names_in_pool` | 10% | Every selected name is in the pool (no hallucinated wavetables) |
| `pool_candidates_discussed` | 10% | Fraction of pool names mentioned in the judge's deliberation |
| `has_render_probes` | 5% | Agent actually rendered pool probes |
| `closing_assistant` | 2.5% | Last message is assistant |
| `format_consistent` | 1.25% | No `**BOLD:**` headers |
| `snake_case_clean` | 1.25% | No raw snake_case (note: agent IDs with underscores are a known false-positive source) |

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

## Iterative preset recreation pipeline (main-agent SFT v2, legacy)

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
            FINAL ASSESSMENT (budget_exhausted): [summary of progress and remaining gap]
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

## Search agent SFT v2 (iterative batch-listening wavetable search)

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
- **Sequential batching**: each search agent walks its 48-candidate slice in index order and splits into 6 fixed batches of 8. No shuffling — the model learns to cover a slice systematically rather than relying on batch-position heuristics. GT candidates land wherever they fall in the slice; `is_clap_selected` guarantees they're sticky once on the shortlist (append-only, never removed).
- **Single-audio policy at build time** *(anti-hallucination)*: every Omni call that takes audio sends exactly ONE clip. The build-time flow is: (1) describe the target once, (2) describe each candidate in isolation (parallelized), (3) text-only synthesis of the per-candidate building-block assessments against the cached target description. This replaces the earlier multi-audio batch call (target + 8 candidates in one message), which produced position-confusion hallucinations where the model scrambled which description belonged to which clip. Measured hallucination rate: 53.6% → 17.7% zero-grounding on a search smoke, 59.4% → 19.8% on judge.
- **GT grounding**: when a GT wavetable appears, Stage 2 receives the target preset's processing chain (`describe_key_transforms`) and writes reasoning about how the raw wavetable transforms under that processing. The model learns to reason about raw→processed transformation.
- **Names in token space**: each candidate has both `<audio>` and its wavetable name, so the model can reference it in code.
- **No CLAP, no role-tagging, no tuple assembly** — just a shortlist of wavetable names that "sound like they belong."

## Legacy hierarchical builders (search v1 / judge — superseded)

Earlier search and judge agent builders used pre-computed CLAP rankings with template proposals. Superseded by search agent v2 (above) which teaches genuine perceptual comparison. Legacy scripts retained for reference:
- `scripts/build_search_agent_sft.py` (v1, template proposals)
- `scripts/build_judge_agent_sft.py` (listwise ranking from CLAP scores)

## Train

```bash
# Single-card packing test (3B model, 1×24GB)
CUDA_VISIBLE_DEVICES=0 python scripts/test_packing_multimodal.py

# Full training (7B, 4×24GB with DeepSpeed ZeRO-3)
bash scripts/train_qwen25_omni_lora.sh full

# Smoke run (dry run, no GPU needed)
python scripts/train_qwen25_omni_lora.py --profile smoke --dry-run

# Megatron 7B LoRA rank-capacity sweep at 48k + 64k (4-step viability)
bash scripts/sweep_qwen25_omni_7b_lora_rank.sh
```

Always train inside tmux — bare SSH runs orphan processes and can OOM the machine.
