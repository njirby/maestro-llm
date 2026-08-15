# Which generation path is current? (read this first)

Several generations of SFT builders coexist in `scripts/`. This has already
caused real bugs — a helper imported from the wrong module, a function
reimplemented that already existed, and confusion over which artifact
determines search labels. This file is the map. **Last verified: 2026-08-15.**

## The canonical path

```
scripts/build_unified_sft_v4.py  --daw-farm docker
```

That is the ONLY entry point that should be used to generate new training
data. It produces all four record kinds (main / search / judge /
transcription) from one manifest, executing every tool call in a real REAPER
container.

It depends on:

| module | role | status |
|---|---|---|
| `scripts/opencode_contract.py` | **the contract** — tool schemas, output shapes, system messages, `task_result`, agent prompts, dispatch prompts | current, single source of truth |
| `scripts/agent_sft_common.py` | shared machinery: snippet templates, record emission (`oc_*` helpers), validator, mistake catalogs, `compute_round_coverage`, `ClapEmbedder` | current |
| `scripts/build_search_agent_sft_v3.py` | search records: evidence labels, terse verdicts, discriminability gate | current |
| `scripts/build_judge_agent_sft_v3.py` | judge records | current |
| `scripts/build_transcription_agent_sft_v4.py` | transcription records | current |
| `scripts/build_main_agent_sft_v3.py` | **library only** — v4 imports `_batch_search_queries`, `_extract_plan_bullet`, `_init_preset`, batch/diff construction | do not run standalone |
| `scripts/build_main_agent_sft_v2.py` | **library only** — v4 imports `_llm_post`, `_check_server_reachable`, `_build_listen_probe_command` | do not run standalone |
| `scripts/build_search_agent_sft_v2.py` | **library only, simulated mode** — v4 imports `build_search_record`/`build_judge_record` for the non-daw-farm path | superseded by v3 in daw-farm mode |

Everything else matching `scripts/build_*_agent_sft*.py` (no version suffix,
or `_v2` main, or `build_iter_sft_dataset.py`, `build_omni_lua_sft_dataset.py`)
is **legacy** — kept because older datasets were built with it, not because it
should be used again.

## Two execution modes, one output shape

- **daw-farm mode** (`--daw-farm docker`): every tool call really executes in a
  container; audio is really rendered. This is what training data must come
  from — rollouts should come from the exact environment the model will act in.
- **simulated mode** (no flag): fabricates tool responses from static dumps.
  Retained for fast iteration on conversation shape only. Its formatters are
  kept byte-identical to the live snippets on purpose; if you change one, change
  both.

## Things that look like they matter but don't (and vice versa)

- **`outputs/wt_retrieval_baseline/wt_index*`** does NOT determine search
  shortlists. v3 renders its own candidate AND GT-ingredient probes with the
  sample's transcribed melody and compares those. The index is used by v4
  startup (`selected_by_name`) and the simulated path only. See
  `transcription_lora_arm1_findings.md`.
- **Probe rendering must apply full preset state.** `maestro/render/dawdreamer.py`
  applies numeric params via `set_parameter` AND full state via `load_state`.
  If `load_state` ever fails it falls back to params-only and prints a LOUD
  warning — wavetables silently would not apply, which invalidated an entire
  corpus once. Never remove that warning.
- **`--force-research-rate` is a documented no-op.** Re-search is driven by
  `--round1-coverage` (round 1 auditions a random window; GT outside it means a
  genuine miss and an honest second round).

## If you are adding a builder

Don't. Extend v4 or the v3 sub-builders. If you must add one, add it to the
table above in the same commit, and route all tool-call/tool-response emission
through `opencode_contract` so training data and the harness stay byte-identical.

## Cleanup backlog

The legacy builders should eventually be collapsed: move the handful of helpers
v4 imports from `build_main_agent_sft_v2/v3` and `build_search_agent_sft_v2`
into `agent_sft_common.py`, then delete or archive the standalone scripts. Not
done yet because the legacy 8k corpus tooling still references them.
