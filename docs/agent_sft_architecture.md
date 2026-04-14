# Agent SFT Architecture (Search v2 → Main v3)

This document captures the current training-data architecture for agentic Vital sound recreation in REAPER.

## Goal
Train a terminal coding agent that can:
1. Listen to a target audio clip.
2. Search a wavetable library by ear to find matching building blocks.
3. Assemble and evaluate wavetable combinations (tuples) by rendering and listening.
4. Apply iterative subsystem-batched parameter edits with inline correction.

The objective is realistic listen → plan → act → listen trajectories with genuine audio reasoning and explicit tool use.

## Agent Hierarchy

### 1) Search Agent (`task_type=search_v2`)

**Builder:** `scripts/build_search_agent_sft_v2.py`

Scope:
- Receives a disjoint shard of the candidate wavetable pool.
- Iterates through batches of candidates (target audio + 6-8 candidate probes per round).
- Maintains a running shortlist across rounds, evaluating each candidate's raw character as potential raw material for the target.
- Returns a shortlist of wavetable names (not tuples, not role-tagged).

Key features:
- **Iterative batch listening**: multiple rounds, shortlist evolves.
- **GT-grounded reasoning**: when a GT wavetable appears, Stage 2 receives the target preset's processing chain (filter, envelope, modulation, FX) and writes reasoning about how the raw wavetable transforms under that specific processing.
- **Names in token space**: each candidate has `<audio>` + its wavetable name, so the model can reference it in code.
- **No CLAP at inference**: CLAP is used only at build time for candidate pool construction (GT-to-index apples-to-apples similarity, top-K=48).

Output:
- Final shortlist of 2-4 wavetable names per search agent.

### 2) Main Agent (`task_type=main`, `pipeline_version=v3`)

**Builder:** `scripts/build_main_agent_sft_v3.py`

Scope:
- Dispatches search agents and collects pooled candidate wavetable names.
- Renders 2-3 tuple combinations through Vital and listens to pick the best.
- Applies chosen wavetables via library lookup by name (inference-compatible).
- Writes an upfront DIAGNOSIS (subsystem plan) grounded in GT-vs-init preset diff.
- Executes subsystem-batched parameter edits with fresh vita-rendered per-batch audio.
- Inline mistake correction (~20% of samples).
- Final assessment comparing recreation vs target.

Conversation structure:
```
Block 0  — Baseline listen + WT search dispatch + collect
Block 1  — Tuple render + listen + select + library-lookup apply
Block 2  — DIAGNOSIS (Omni Stage 1 + Stage 2 subsystem plan)
Block 3..K — Subsystem batches (oscillator → envelope → filter → lfo → fx → modulation → macro)
Block K+1 — CORRECTION (inline, if mistake was injected)
Block K+2 — FINAL ASSESSMENT
```

## Data Contract (MS-Swift-Friendly)

Both task types follow one strict schema:
- JSONL (one JSON object per line).
- `messages[*].content` is always a string.
- Allowed roles only: `user`, `assistant`, `tool_call`, `tool_response`.
- First message role is `user`, last is `assistant`.
- No adjacent duplicate conversational roles.
- Max one `<audio>` tag per `user` or `assistant` message (tool_response may have multiple).
- Total `<audio>` tag count equals `len(audios)`.

Tools are represented in-message via:
- `tool_call` content: JSON string like `{ "name": "bash", "arguments": {...} }`.
- `tool_response` content: JSON/string output payload.

## Search → Main Handoff

1. Main agent starts from target audio + default baseline.
2. Main dispatches search agents (tool_call: `spawn_search_agents`).
3. Search agents (trained separately) evaluate candidates by ear across multiple rounds.
4. Main collects pooled shortlist (tool_call: `collect_search_reports`).
5. Main renders 2-3 tuple combinations through Vital, listens, selects the best.
6. Main applies chosen wavetables via library lookup (inference-compatible).
7. Main writes DIAGNOSIS and executes subsystem-batched parameter edits.

## Candidate Pool Construction

At build time (not model-visible):
1. Extract GT wavetable names from target preset (1-3 active oscillators).
2. Look up GT wavetable embeddings in the CLAP wavetable index (bare probes through default preset).
3. Compute cosine similarity of GT embeddings against all 568 wavetable index entries (apples-to-apples).
4. Take top-K=48 most similar as hard negatives. GT always included.
5. Shuffle and split into disjoint shards for search agents.

CLAP retrieval against fully-processed target audio was evaluated and failed (R@5=4.95%) because it compared processed target audio vs bare wavetable probes. GT-to-index comparison works because both sides are bare probes through the same default preset.

## Current Builders
- `scripts/build_search_agent_sft_v2.py` — search agent (iterative batch listening)
- `scripts/build_main_agent_sft_v3.py` — main agent (diagnose → subsystem batches)
- `scripts/grade_agent_sft.py` — v2 + v3 scoring paths
- `scripts/agent_sft_common.py` — shared helpers including `build_gt_similarity_pool()`
- `scripts/experiment_clap_wt_threshold.py` — CLAP threshold experiment
- `scripts/experiment_omni_batch_listen.py` — Omni batch listening validation

### Legacy (superseded)
- `scripts/build_search_agent_sft.py` — v1 search (template proposals from CLAP rankings)
- `scripts/build_judge_agent_sft.py` — judge (listwise ranking from CLAP scores)
- `scripts/build_main_agent_sft_v2.py` — v2 main (per-step HEARD/HYPOTHESIS/PLAN)

## Validation + Tests
- Contract validator: `validate_ms_swift_multiturn_record(...)`
- Test coverage:
  - `tests/test_search_agent_sft_v2.py` — search agent v2 structural invariants
  - `tests/test_agent_sft_contracts_v3.py` — main agent v3 structural invariants
  - `tests/test_agent_sft_grading.py` — v2 + v3 grading logic
  - `tests/test_agent_sft_contracts.py` — v2 legacy (regression guard)

Recommended checks:
```bash
pytest tests/test_search_agent_sft_v2.py tests/test_agent_sft_contracts_v3.py tests/test_agent_sft_grading.py -x
```
