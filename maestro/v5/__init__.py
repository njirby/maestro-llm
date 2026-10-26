"""Builder v5: ground-truth-driven synthetic rollouts in exact OpenCode v1.19.3 format.

Every tool call in a record is executed for real in a DawDreamer session (maestro/dawfarm_dd.py); the agents'
choices come from the ground truth (GT preset, GT notes, GT wavetables), so the rollouts are off-policy by design.

  rollout.py        one sample end to end: main record + transcriber / searcher / judge records
  record.py         ms-swift record assembly on top of scripts/opencode_contract.py
  ctx.py            per-rollout execution context (session, audio copies, bash with OpenCode formatting)
  audio.py          host-side spectral distance (oracle ranking of searcher probes), peak level
  planner.py        GT diff -> edit rounds (skill CLI commands), param mistakes and corrections
  mistakes.py       code mistakes on skill CLI calls (bad key, out-of-range value, wrong flag)
  narration.py      narration adapter (first_impression / self_edit / target_vs_current / final_verdict ...)
  agents/           the four agents' record builders
  inputs.py         sample schema + a small test-input generator (WS-C's manifest replaces it)
  validate.py       the record contract (single record + linked rollout)
  viewer.py         one rollout as an HTML page with inline audio
"""
