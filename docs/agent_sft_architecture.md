# Agent SFT Architecture (Search v2 → Judge v3 → Transcription v3 → Main v3)

This document captures the current training-data architecture for agentic Vital sound recreation in REAPER.

## Goal
Train a terminal coding agent that can:
1. Listen to a target audio clip.
2. Dispatch search sub-agents in parallel to search a wavetable library by ear.
3. Audition the combined search pool via a judge sub-agent to pick the final wavetable tuple.
4. Render and listen to the chosen tuple, then iteratively apply subsystem-batched parameter edits grounded in the ground-truth preset delta.
5. Recognise edge cases (missing audio attachment → ask user to select a clip).

The objective is realistic listen → plan → act → listen trajectories with genuine audio reasoning, explicit tool use, and a claw-code-style hierarchical sub-agent architecture.

## Agent Hierarchy

```
┌──────────────────┐
│   Main Agent     │  (task_type=main, pipeline_version=v3)
│                  │
│ user selects     │
│ audio in REAPER  │
└────────┬─────────┘
         │ dispatches
         │  (parallel Agent tool_calls)
         ▼
┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐      ┌──────────────────┐
│  Search Agent 1  │      │  Search Agent 2  │      │  Search Agent 3  │      │  Search Agent 4  │
│  (slice 0-47)    │      │  (slice 70-117)  │      │  (slice 140-187) │      │  (slice 210-257) │
└────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘      └────────┬─────────┘
         │                          │                         │                         │
         └──────────────────────────┴─────────────────────────┴─────────────────────────┘
                                              │
                                              │ each writes shortlist JSON to
                                              │ /tmp/agents/<sample_id>/*.json
                                              ▼
                                    ┌──────────────────┐
                                    │  Main Agent      │
                                    │  cats all 4 files│
                                    │  → combined pool │
                                    └────────┬─────────┘
                                             │ dispatches
                                             ▼
                                    ┌──────────────────┐
                                    │  Judge Agent     │  (task_type=judge, v3_judge)
                                    │                  │
                                    │ Audions target + │
                                    │ all pool cands   │
                                    │ in ONE Omni call │
                                    │ Picks final tuple│
                                    │ → writes JSON    │
                                    └────────┬─────────┘
                                             │
                                             ▼
                                    ┌──────────────────┐
                                    │  Main Agent      │
                                    │  reads judge out │
                                    │  renders tuple   │
                                    │  → listens       │
                                    │  → diagnose/plan │
                                    │  → apply batches │
                                    │  → verdict       │
                                    └──────────────────┘
```

### 1) Search Agent (`task_type=search_v2`)

**Builder:** `scripts/build_search_agent_sft_v2.py`

Scope:
- Receives one slice (48 consecutive wavetables by index) of the library.
- Iterates through batches of 8 candidates per round. For each batch: renders probe audios, Omni listens to target + 8 candidate probes, Stage 2 writes per-candidate one-sentence assessment.
- CLAP-grounded selection (threshold 0.92): the builder decides which candidates are "Selected" / "Not selected" at build time; Stage 2 only writes reasoning that aligns with the correct labels.
- Returns a shortlist of wavetable names (no ranking, no role-tagging).

Key features:
- **Iterative batch listening**: 6 batches × 8 candidates = 48 evaluated per agent.
- **GT-grounded reasoning**: when a GT wavetable appears in a batch, Stage 2 receives the target preset's processing chain (filter, envelope, modulation, FX) and writes reasoning about how the raw wavetable transforms under that processing.
- **Audio-labelled names**: each candidate probe is paired with its wavetable name in the Omni prompt so the model learns the audio↔name mapping.
- **Concurrency**: `--workers` parallelises at the entry level; each sample also runs its 4 search agents in parallel (entry × 4 agents saturates vLLM at ~24 in-flight requests).

Output:
- Shortlist of 2-4 wavetable names per slice.
- Per-agent JSON file written to `/tmp/agents/<sample_id>/<agent_id>.json` (mirrors the runtime executor's file-handoff protocol).

### 2) Judge Agent (`task_type=judge`, `pipeline_version=v3_judge`)

**Builder:** `scripts/build_judge_agent_sft_v3.py`

Scope:
- Receives the combined pool from all 4 search agents in a single auditory view (each `<audio>` labelled by wavetable name in the prompt).
- Renders probes for all pool candidates in one batch.
- Omni listens to target + all N pool candidates in ONE audio call; Stage 2 formats per-candidate reasoning + final "SELECTED: [...]" line.
- Writes the final tuple + reasoning to a JSON file.

Why the judge exists:
- Each search agent only sees its own slice. When ground-truth wavetables are scattered across slices, each agent finds one GT + some false positives. None has the global view to pick the correct combination.
- The judge has the whole pool in one auditory view — it's the only agent that can reliably pick N complementary wavetables (N = target's active oscillator count).

Build-time oracle:
- Selected tuple = GT-if-in-pool + CLAP-best-proxy per osc slot.
- Stage 2 is told the oracle answer and writes per-candidate reasoning that aligns with the selection.
- `meta.judge_correct` flags whether the selection matches the GTs present in the pool.

Output:
- Shortlist → tuple of 1-3 wavetable names (matching `n_osc_slots`).
- JSON file: `{"tuple": [...], "n_osc_slots": N, "reasoning": "..."}` written via bash Python heredoc during the conversation.

### 3) Melody Transcription Agent (`task_type=melody_transcription`, `pipeline_version=v3_transcription`)

**Builder:** `scripts/build_transcription_agent_sft_v3.py`

Scope:
- Receives the target audio + a REAPER track index (created by the main agent just before dispatch).
- Listens to the target (Omni Stage 1 — perceptual impression: contour, rhythm feel, rough note count).
- Formats oracle-grounded per-note narration (Stage 2 gets the note list from the manifest's `source_midi_path` as ground truth and renders it as a readable numbered list).
- Writes the Python (reapy) code that inserts the notes via `RPR_MIDI_InsertNote` at PPQ positions — same convention as the legacy Lua example (`bpm=120, ppb=960`).
- Saves the final note list JSON to `/tmp/agents/<sample_id>/transcription.json` via a visible bash tool_call (so the main agent can `cat` it).

Why the transcription agent exists:
- Closes the "where does the MIDI come from?" gap in the main agent's rollout. Previously the MIDI was pure oracle — the model never saw it get transcribed or populated onto a REAPER track.
- At inference, the main agent creates a track and dispatches this subagent to actually write notes on it via `reapy.inside_reaper()`. Without this task, the trained model would have no idea how to do that.

Build-time oracle:
- Notes come from `pretty_midi.PrettyMIDI(source_midi_path)` — the same MIDI that drives `vita` to render the target. Transcription is exactly correct by construction.
- Stage 1 Omni provides a perceptual impression only (no numbers); Stage 2 then renders the oracle per-note list inside a coherent narration.

Output:
- JSON file: `{"notes": [{"pitch","start_s","end_s","velocity"}, ...], "n_notes": N, "duration_s": X}`.
- Scope is **one-shot**: transcription runs once at the start of the main-agent flow. Error-correction reviews (re-invoke transcription after hearing a wrong note) are deferred.
- Audio cap: target clips must fit in `vita`'s 30 s ceiling. Longer clips would need parallel-slice transcription (future work).

### 4) Main Agent (`task_type=main`, `pipeline_version=v3`)

**Builder:** `scripts/build_main_agent_sft_v3.py`

Scope:
- Listens to the target + renders/listens to default baseline.
- Checks library size (single bash ceremony turn).
- Dispatches 4 search agents in parallel (all `Agent` tool_calls emitted back-to-back, then all tool_responses — represents parallel dispatch in the claw-code tool-use protocol).
- Reads search shortlists via `cat`, pools candidates.
- Dispatches judge agent. Reads judge's output JSON via `cat`.
- Renders the selected tuple via `render_wavetable_tuple.py`, listens, decides.
- If tuple doesn't match target (or GTs missing from pool): triggers another search round with shifted slices, then another judge pass.
- Once tuple matches: applies wavetables via VitalController, writes DIAGNOSIS (preset-grounded OBSERVATIONS + subsystem PLAN), executes subsystem-batched parameter edits with per-batch audio and plan-aligned narration, optional inline mistake correction, FINAL ASSESSMENT grounded in residual delta.
- Edge case (~5% of samples): user message arrives without `<audio>` attachment → single-turn refusal asking user to select an audio clip in REAPER.

Conversation structure:
```
user:        <audio>  "Recreate this sound in Vital."
             (OR: no <audio> → early-return refusal)

# SKILL DISCOVERY + LOAD (claw-code-idiomatic, every normal record)
assistant:   Let me see which skills are available for this plugin.
tool_call:   bash ls skills/*/SKILL.md
tool_resp:   skills/vital/SKILL.md
assistant:   The vital skill matches. Loading it for plugin-specific instructions,
             helper-script paths, and recreation strategy.
tool_call:   Skill { skill: "vital", args: "" }
tool_resp:   { skill, path, description, prompt: <full SKILL.md contents> }

assistant:   Skill loaded. Listening to current default preset baseline.
tool_call:   bash listen probe
tool_resp:   {"baseline_audio": "<audio>", ...}

# MIDI TRANSCRIPTION BLOCK
assistant:   Creating a REAPER track to hold the transcribed MIDI before I search
             the wavetable library.
tool_call:   bash python (reapy: InsertTrackAtIndex + TrackFX_AddByName("Vital"))
tool_resp:   {"status":"ok","track_idx":0,"track_name":"target_melody"}
assistant:   Dispatching the transcription subagent to listen to the target and
             populate the track with MIDI notes.
tool_call:   Agent { subagent_type: "melody_transcription", ... }
tool_resp:   {agentId, outputFile: "/tmp/agents/<sid>/transcription.json",
              status: "completed"}
assistant:   Reading the transcribed note list.
tool_call:   bash cat /tmp/agents/<sid>/transcription.json
tool_resp:   {"notes":[...],"n_notes":N,"duration_s":X}

assistant:   MIDI ready on track 0 — N notes, ~Xs. Checking wavetable library size.
tool_call:   bash python skills/vital/scripts/list_wavetables.py --total
tool_resp:   {"total": 282}

# PARALLEL SEARCH DISPATCH
assistant:   Library has 282 wavetables. Dispatching 4 search agents in parallel...
tool_call:   Agent wavetable_search #1    ┐
tool_call:   Agent wavetable_search #2    │  emitted back-to-back
tool_call:   Agent wavetable_search #3    │  (parallel tool_use blocks)
tool_call:   Agent wavetable_search #4    ┘
tool_resp:   agent_1 outputFile           ┐
tool_resp:   agent_2 outputFile           │  matching parallel responses
tool_resp:   agent_3 outputFile           │
tool_resp:   agent_4 outputFile           ┘

assistant:   Reading shortlists from 4 search agents.
tool_call:   bash cat ...search_1.json ...search_4.json
tool_resp:   {4 concatenated shortlist JSONs}

# JUDGE DISPATCH
assistant:   Pool has 12 candidates across 4 slices. Dispatching judge agent...
tool_call:   Agent wavetable_judge
tool_resp:   {agentId: wavetable_judge_..., outputFile: .../judge_r1.json}

assistant:   Reading judge's tuple selection.
tool_call:   bash cat .../judge_r1.json
tool_resp:   {"tuple": ["X", "Y"], "n_osc_slots": 2, "reasoning": "..."}

# TUPLE VERIFY
assistant:   Judge selected [X, Y]. Rendering the tuple to verify the combination.
tool_call:   bash python skills/vital/scripts/render_tuple.py --osc1 X --osc2 Y --out ...
tool_resp:   {"tuple_audio": "<audio>", ...}

assistant:   This tuple matches the target well. Applying.
             OR: The rendered tuple doesn't capture the target's character closely
             enough. Expanding to different library regions... (re-search, re-judge)

# APPLY + DIAGNOSE + EXECUTE + VERDICT
assistant:   (apply tuple via VitalController)
assistant:   OBSERVATIONS: [preset-grounded perceptual description from Omni Stage 1
             with preset summary as grounding prior]
             PLAN:
               • Oscillator: ...
               • Envelope: ...
               ...
             Executing plan by subsystem.
             Applying oscillator changes.
tool_call:   bash VitalController set_params (oscillator batch)
tool_resp:   {"status": "ok"}
assistant:   Listening after oscillator batch.
tool_call:   bash listen probe (fresh vita render of cumulative state)
tool_resp:   {"batch_audio": "<audio>", ...}
assistant:   [plan-aligned narration grounded in this batch's plan bullet +
             before→after param deltas]

# (repeat for envelope, filter, lfo, fx, modulation, macro)

# CORRECTION (inline, ~20% of samples)
assistant:   Overshot on {subsystem} — backing off {param}.
tool_call:   bash VitalController corrective set_param
tool_resp:   {"status": "ok"}
assistant:   Listening to the corrected preset.
tool_call:   bash listen probe
tool_resp:   {"corrected_audio": "<audio>", ...}

# VERDICT (grounded in residual preset delta)
assistant:   FINAL ASSESSMENT (complete|budget_exhausted): [what matches well; one
             specific concrete residual cited from summarize_residual_delta_perceptual]
```

## Data Contract (MS-Swift-Friendly)

All three task types follow one strict schema:
- JSONL (one JSON object per line).
- `messages[*].content` is always a string.
- Allowed roles only: `user`, `assistant`, `tool_call`, `tool_response`.
- First message role is `user`, last is `assistant`.
- No adjacent duplicate conversational roles, **except** for consecutive `tool_call` or consecutive `tool_response` messages, which represent parallel tool dispatch (multiple tool_use blocks from a single assistant turn in the claw-code / Anthropic protocol).
- Max one `<audio>` tag per `user` or `assistant` message (tool_response may have multiple).
- Total `<audio>` tag count equals `len(audios)`.

Tools are represented in-message via:
- `tool_call` content: JSON string like `{"name": "bash", "arguments": {...}}` or `{"name": "Agent", "arguments": {"subagent_type": ..., "description": ..., "prompt": ..., "name": ...}}`.
- `tool_response` content: JSON/string output payload.

## Search → Judge → Main Handoff

1. **Main agent** starts from target audio + default baseline; determines `n_osc_slots` from target preset.
2. **Main dispatches 4 search agents in parallel** (one `Agent` tool_call per slice, all emitted before any tool_response). Each search agent runs as its own independent SFT task at training time; at inference, the harness executes them concurrently.
3. **Search agents** each evaluate 48 candidates across 6 batches by ear, write shortlist JSON files (2-4 names per slice).
4. **Main cats all 4 shortlist files** and forms the combined pool (~12 unique candidates).
5. **Main dispatches judge agent** with the pool. The judge auditions target + all pool candidates in one Omni call, picks the final tuple matching `n_osc_slots`, writes output JSON.
6. **Main cats judge output**, reads the tuple, renders it via `render_wavetable_tuple.py`, listens.
7. **If tuple doesn't match target** (or GTs missing from pool): main agent triggers a new search round with shifted slices, then a new judge pass. Up to `max_search_rounds` total.
8. **Once tuple matches**: main applies wavetables via VitalController, writes diagnosis, executes subsystem batches, produces verdict.

## Preset-Grounded Stage 1 Observations

`scripts/preset_perceptual_summary.py`

Multi-audio target/default comparison was out-of-distribution for Qwen-Omni (hallucinated modulation, flipped attack shape, wrote generic differential prose). The pipeline now relies on what text Qwen actually knows — sound-design vocabulary and param-to-perception mapping — while keeping audio in the loop as a perceptual grounding prior.

Flow:
1. Extract perceptual buckets from the target preset (no numbers, no param names): envelope ADSR (plucky/gradual/swell + short/moderate/long/lingering), filter (brightness + resonance + model), unison (voice count + detune), active LFOs, active effects, prominent modulation routes.
2. Inject the bucket summary into Stage 1's Omni prompt alongside the target audio.
3. Stage 1 prompt enforces coverage of five sound-design aspects (envelope shape, tone+filter, oscillator body, motion, space+effects) so observations are reverse-engineerable by another producer.

`summarize_residual_delta_perceptual(target, final)` does the same for the FINAL ASSESSMENT — identifies the top 5 concrete differences between target and final preset (envelope ADSR mismatches, filter cutoff/resonance/on-off, unison voices/detune, effect on/off) ranked by magnitude. Verdicts cite these residuals instead of defaulting to "envelope N".

## Plan + Param-Delta Driven Narrations

The per-batch narration (assistant turn after each subsystem batch) used to pull from a 28-lens descriptor bank that rotated across batches. Problem: lenses could push the narration to describe motion on a batch that actually disengaged motion, contradicting the plan.

Current approach (no lens bank):
- Narration generator receives the plan's bullet for this subsystem + the concrete before→after values for the params changed in this batch (with direction words: substantially increased / slightly decreased / disengaged / unchanged).
- Prompt instructs: stay consistent with the plan direction; ground claims in the actual param deltas; translate numeric change into perceptual effect.
- Natural variety comes from different presets having genuinely different deltas.

## No-Audio Edge Case

`--no-audio-rate` (default 0.05): for ~5% of samples, the user message has no `<audio>` tag. The assistant responds with a single-turn refusal:

> "I don't see an audio clip attached to your message. Please select the audio item in REAPER that you want me to recreate..."

Record is flagged `meta.variant='no_audio_selected'` so the grader skips the normal metrics.

## Candidate Pool Construction (Build-Time Only)

Three independent mechanisms that together determine what each search agent sees and what ends up in the pool:

### 1) Library slicing + GT-coverage rotation

1. Extract GT wavetable names from target preset (1-3 active oscillators).
2. Compute CLAP embeddings for each GT wavetable's bare probe (used only for the selection step below, never shown to the model).
3. Slice the 282-wavetable library into N=4 agent slices of ~48 wavetables each. The **base offset** is chosen so at least one slice covers a GT — the builder rotates the offset until this holds (reliable round-1 success by default).

### 2) `--force-research-rate` — deliberately miss the GT in round 1

`scripts/build_main_agent_sft_v3.py:863-864` — for this fraction of samples, **skip the rotation step above**. Slices land wherever the random offset puts them, possibly with no GT in any slice.

Why: the re-search branch of the main agent (dispatch a second round of search agents at shifted offsets when the tuple doesn't match) needs training examples where round 1 genuinely failed. Without forced misses, every sample would find the GT in round 1 and the model would never learn to expand its search. Default 0.30, smoke tests use 0.80 for higher re-search coverage.

### 3) CLAP-grounded per-candidate selection inside the search agent

`scripts/agent_sft_common.py: is_clap_selected(threshold=0.92)` — invoked by the search-agent builder for each candidate in each batch.

What it does: for each candidate in a search agent's slice, the builder labels it **"Selected"** or **"Not selected"** by this rule:
- If candidate name is exactly a GT wavetable → Selected
- Else if `cosine(candidate_embedding, any_GT_embedding) >= 0.92` → Selected
- Otherwise → Not selected

The 0.92 threshold came from `scripts/experiment_clap_wt_threshold.py` — CLAP-similarity distributions showed 0.92 is the cutoff where candidates are acoustically close enough to the GT to serve as plausible building blocks.

Stage 2 (the text model writing the shortlist reasoning) is **told** which label each candidate got and just writes a one-sentence rationale that aligns with the label. This prevents the old contradiction where Stage 2 wrote "Skip" but the builder added the GT back anyway.

### Re-search trigger (exact-name match, NOT CLAP)

`scripts/build_main_agent_sft_v3.py:1286`:
```python
all_gt_found = all(gt in pool for gt in gt_names_list)
if all_gt_found or rounds_used >= max_rounds:
    break
```

The main agent keeps dispatching search rounds until **every GT wavetable name** is present in the pool (exact match, not CLAP-similar), or `max_search_rounds=3` is hit. Exact match is the reliable build-time proxy for "the tuple will sound right" — at inference, the model judges by listening to the rendered tuple.

An earlier design used `_pool_covers_gt(threshold=0.92)` as the re-search trigger, but we moved to exact-GT-in-pool because at inference the model can't check CLAP coverage.

### CLAP retrieval was tried differently and failed

CLAP retrieval against fully-processed target audio was evaluated earlier and failed (R@5=4.95%) because it compared processed target audio vs bare wavetable probes. The current usage (GT embedding vs candidate embedding, both bare probes through the same default preset) is apples-to-apples and works.

## Parallel Tool Dispatch in the JSONL

The claw-code / Anthropic tool-use protocol represents parallel tool calls as multiple `tool_use` blocks in a single assistant message. Our ms-swift JSONL simplifies this to per-message roles, but represents the same parallel semantics by emitting **all N `tool_call` messages back-to-back, then all N `tool_response` messages**. The validator (`agent_sft_common.assert_valid_ms_swift_multiturn_record`) permits consecutive `tool_call` / `tool_response` as an explicit exception to the no-adjacent-duplicate rule.

Used for:
- 4 parallel search-agent dispatches at the start of each search round.

Serial (tool_call → tool_response → tool_call → tool_response) is used when calls have data dependencies (e.g. cat shortlists → dispatch judge → cat judge output).

## Current Builders

- `scripts/build_search_agent_sft_v2.py` — search agent (iterative batch listening, CLAP-grounded selection, entry + agent concurrency)
- `scripts/build_judge_agent_sft_v3.py` — judge agent (audition combined pool in one Omni call, pick final tuple, write output file)
- `scripts/build_main_agent_sft_v3.py` — main agent (parallel search dispatch → judge → tuple render+listen → diagnose → subsystem batches → residual-grounded verdict)
- `scripts/preset_perceptual_summary.py` — perceptual-bucket preset summaries (grounding prior for Stage 1 and verdict)
- `scripts/grade_agent_sft.py` — v2 + v3 scoring; v3 LLM-as-judge across 7 axes; cross-sample template detection
- `scripts/build_audio_grounding_spotcheck.py` — self-contained HTML spot-check (audio embedded, optional compare mode)
- `scripts/validate_grounded_observations.py` — A/B runner for Stage 1 observation variants
- `scripts/agent_sft_common.py` — shared helpers including `build_gt_similarity_pool()`, `build_name_embedding_map()`, `is_clap_selected()`
- `scripts/experiment_clap_wt_threshold.py` — CLAP threshold experiment
- `scripts/experiment_omni_batch_listen.py` — Omni batch listening validation

### Legacy (superseded)
- `scripts/build_search_agent_sft.py` — v1 search (template proposals from CLAP rankings)
- `scripts/build_judge_agent_sft.py` — v1 judge (listwise ranking from CLAP scores, no audio listening; superseded by v3 audition-based judge)
- `scripts/build_main_agent_sft_v2.py` — v2 main (per-step HEARD/HYPOTHESIS/PLAN)

## Grading

`scripts/grade_agent_sft.py` grades all three agent types through one entry point, dispatching on `task_type` + `meta.pipeline_version`.

### Main agent — LLM-as-Judge (v3 path)

With `--llm-judge-server` the grader runs 7 LLM-judge axes in addition to the structural checks:

| Axis | Purpose |
|---|---|
| `llm_observations_audio_grounded` | Omni listens to target WAV and judges whether OBSERVATIONS matches audio |
| `llm_narration_no_hallucination` | Per-batch binary: does narration reference param families not in the batch? |
| `llm_narration_plan_ref` | Does narration pick up plan's subsystem bullet? |
| `llm_narration_param_specific` | Does narration cite audible consequences of the edited params? |
| `llm_narration_templateness` | **Cross-sample binary** — compares phrasing against 3 other samples' same-subsystem narrations |
| `llm_verdict_residual_grounded` | Does verdict cite concrete residual (not generic "envelope N")? |
| `llm_verdict_novelty` | **Cross-sample binary** — compares against 3 other samples' verdicts |

Weight split: structural metrics ~52%, LLM-judge ~48% when enabled. Runtime: ~35s for 8 samples at `--workers 8`.

Latest n=8 smoke benchmarks:
- **v9**: overall 0.841 on normal records; both verdict axes at 1.00
- **v13** (with Skill discovery + load): overall 0.839

### Search agent v2 — structural + correctness (no LLM)

| Axis | Weight | Purpose |
|---|---|---|
| `gt_recovery` | 35% (conditional) | Fraction of `meta.gt_in_shard` that made it onto the final shortlist |
| `shortlist_file_written` | 25% | Final bash tool_call writes `*_search_*.json` + matching ok tool_response |
| `closing_assistant` | 10% | Last message is assistant (task completion signal) |
| `has_render_probes` | 10% | ≥1 bash tool_call invokes `skills/vital/scripts/render_probes.py` |
| `shortlist_nonempty` | 10% | Final shortlist has ≥1 name |
| `snake_case_clean` | 5% | No snake_case in assistant prose |
| `format_consistent` | 5% | No `**BOLD:**` headers |

Search v12 smoke (n=32): overall **0.994**, gt_recovery 1.00 on all 10 gt-in-shard cases.

### Judge agent v3 — structural + correctness (no LLM)

| Axis | Weight | Purpose |
|---|---|---|
| `judge_correct` | 30% | `meta.judge_correct`: selection matches GTs present in pool (oracle) |
| `tuple_size_correct` | 10% | `len(selected_tuple) == n_osc_slots` |
| `tuple_names_in_pool` | 10% | Every selected name is in the pool (no hallucinated wavetables) |
| `output_file_written` | 25% | Last bash writes `{tuple, n_osc_slots, reasoning}` + ok tool_response |
| `pool_candidates_discussed` | 10% | Fraction of pool names mentioned in judge's deliberation |
| `has_render_probes` | 5% | Agent rendered pool probes |
| `closing_assistant` | 5% | Last message is assistant |
| `format_consistent` | 2.5% | No `**BOLD:**` headers |
| `snake_case_clean` | 2.5% | No snake_case in assistant prose (agent IDs can trigger false positives) |

Judge v12 smoke (n=8): overall **0.992**, judge_correct 1.00 across all samples.

### Melody transcription v3 — structural + oracle correctness (no LLM)

| Axis | Weight | Purpose |
|---|---|---|
| `has_midi_insert` | 20% | ≥1 bash tool_call contains `MIDI_InsertNote` (or `RPR_MIDI_InsertNote`) |
| `output_file_written` | 25% | Final bash writes `transcription.json` with `{notes,n_notes,duration_s}` + ok tool_response |
| `note_count_match` | 20% | Payload n_notes matches `meta.n_notes` (oracle count) |
| `pitch_coverage` | 10% | Fraction of oracle MIDI pitches mentioned in deliberation or insert cmd |
| `has_render_listen` | 10% | First user message carries `<audio>` (subagent actually received target audio) |
| `closing_assistant` | 5% | Last message is assistant |
| `snake_case_clean` | 5% | No snake_case in assistant prose |
| `format_consistent` | 5% | No `**BOLD:**` headers |

## Validation + Tests
- Contract validator: `validate_ms_swift_multiturn_record(...)` — allows consecutive tool_call/tool_response for parallel dispatch.
- Test coverage (169 tests total):
  - `tests/test_search_agent_sft_v2.py` — search agent v2 structural invariants
  - `tests/test_agent_sft_contracts_v3.py` — main agent v3 structural invariants
  - `tests/test_agent_sft_grading.py` — main-agent v2 + v3 grading logic
  - `tests/test_agent_sft_grading_search_judge.py` — search_v2 + judge_v3 grading logic (15 tests)
  - `tests/test_agent_sft_grading_transcription.py` — transcription grading logic (10 tests)
  - `tests/test_agent_sft_contracts.py` — v2 legacy (regression guard)
  - *(Missing: contract tests for `build_judge_agent_sft_v3.py` + `build_transcription_agent_sft_v3.py` record shapes — follow-up work)*

Recommended checks (159 tests):
```bash
pytest tests/test_search_agent_sft_v2.py tests/test_agent_sft_contracts_v3.py \
       tests/test_agent_sft_grading.py tests/test_agent_sft_grading_search_judge.py -x
```
