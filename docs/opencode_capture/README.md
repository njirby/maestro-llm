# OpenCode capture evidence (copied from /tmp/opencode on 2026-10-25, v1.19.3)

- `captured.jsonl` — requests OpenCode sent to the mock OpenAI server (`mock_openai.py`); record 1 holds the full
  system prompt (`body.messages[0]`) and the stock tool schemas (`body.tools`).
- `captured2.jsonl.gz` — 2,550 later captured requests (task results).
- `opencode.json` — the agent/provider config used for those captures (permissions restricting tools, agents).
- `mock_openai.py`, `mock2.py` — capturing mock servers.
- `explorer7b/` — staged agent setup.
- `oc_strings_v1.19.3.txt.gz` — strings extracted from the bun-compiled binary (no source checkout exists); system-prompt
  selection is function `ud(e)` (Qwen/other -> prompt `Fi`), result-format strings (read wrapper, shell_metadata, ...).
- `OPENCODE_VERSION` — `opencode --version` at copy time.
