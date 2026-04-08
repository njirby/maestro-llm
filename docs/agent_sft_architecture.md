# Hierarchical Agent SFT Architecture (Search -> Judge -> Main)

This document captures the current training-data architecture for agentic Vital sound recreation in REAPER.

## Goal
Train a terminal coding agent that can:
1. Listen to target/current audio.
2. Search candidate wavetable directions in parallel.
3. Judge/select a small candidate set (<=3).
4. Apply iterative parameter edits and re-listen.

The objective is realistic listen -> act -> listen trajectories with explicit tool use and minimal fluff.

## Agent Hierarchy

### 1) Search Agent (`task_type=search`)
Scope:
- Works on a disjoint shard of candidate wavetables.
- Runs tool-based search (bash/python code).
- Auditions top candidates and returns proposal JSON.

Output:
- Ranked shard candidates.
- Proposal list with candidate IDs, wavetable names, confidence, and reasons.

### 2) Judge Agent (`task_type=judge`)
Scope:
- Receives target/current context + candidate bundle.
- Ranks candidates listwise.
- Selects final <=3 candidates for downstream use.

Output:
- `ranking` (ordered candidate IDs).
- `selected` (<=3 IDs).
- Reason string.

### 3) Main Agent (`task_type=main`)
Scope:
- Orchestrates search fanout and report collection.
- Calls judge to collapse candidate pool.
- Executes iterative Vital parameter updates.
- **Audio-gated listening**: re-listens after steps where |CLAP delta| ≥ threshold (default 0.01). Sub-perceptual steps get PLAN-only turns — no audio, no HEARD/HYPOTHESIS.

Output:
- End-to-end orchestration trajectory suitable for SFT warm start.

## Data Contract (MS-Swift-Friendly)
All three datasets follow one strict schema:
- JSONL (one JSON object per line).
- `messages[*].content` is always a string.
- Allowed roles only: `user`, `assistant`, `tool_call`, `tool_response`.
- First message role is `user`, last is `assistant`.
- No adjacent duplicate conversational roles (`user/user`, `assistant/assistant`).
- Max one `<audio>` tag per `user` or `assistant` message (tool_response may have one per audio result).
- Total `<audio>` tag count equals `len(audios)`.
- GT audio appears exactly once per record (block 0 preamble only, unless `--reanchor-gt-audio` is passed).

Tools are represented in-message via:
- `tool_call` content: JSON string like `{ "name": "bash", "arguments": {...} }`.
- `tool_response` content: JSON/string output payload.

## Why This Contract
This fixes the biggest data-quality failures we saw:
- Mixed content types (object vs string) breaking loaders.
- Non-JSONL files masquerading as JSONL.
- Misaligned analysis/tool turns.
- Overloaded search responses and noisy message ordering.

## Search/Judge/Main Handoff
1. Main agent starts from target (+ baseline current audio when available).
2. Main spawns disjoint search shards.
3. Search agents return local proposals from real rendered/auditioned candidates.
4. Main collects reports and calls judge.
5. Judge returns <=3 candidates.
6. Main executes iterative edits, listening after each step.

This gives modular supervision now and cleaner RL trajectory decomposition later.

## Current Builders
- `scripts/build_search_agent_sft.py`
- `scripts/build_judge_agent_sft.py`
- `scripts/build_main_agent_sft_v2.py`
- `scripts/merge_agent_sft.py` (`--drop-invalid` validation gate)
- Shared validator/helpers: `scripts/agent_sft_common.py`

## Validation + Tests
- Contract validator: `validate_ms_swift_multiturn_record(...)`
- Test coverage:
  - `tests/test_agent_sft_contracts.py`

Recommended checks:
- Build smoke JSONL for search/judge/main.
- Load with `datasets.load_dataset('json')`.
- Run `pytest -q tests/test_agent_sft_contracts.py`.
