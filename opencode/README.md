# maestro OpenCode config (repo-owned)

The agents run in OpenCode 1.19.3 with this directory as the **only** config source.
`scripts/opencode_contract.py` reproduces what this config makes OpenCode send, and the goldens in
`docs/opencode_capture/goldens/` pin that behaviour.

```
opencode.json      provider "maestro" (openai-compatible -> vLLM), defaults, permissions
agent/*.md         maestro (primary) + transcriber, searcher, judge (subagents)
skills/<name>/     SKILL.md skills (vital), auto-discovered from this dir
plugin/            maestro plugins (audio read), auto-loaded from this dir
```

## Running isolated

OpenCode never loads this dir directly. It loads a throwaway **staging dir of symlinks** into it,
with HOME and the XDG dirs moved under the same root. Everything OpenCode writes (`.gitignore`,
npm install attempts, the session DB, spill files) lands under the root, never in the repo, and
nothing from `~/.config/opencode`, `~/.claude` or `~/.agents` is read:

```bash
REPO=/path/to/maestro-llm
R=/tmp/maestro-oc                     # fixed: CONFIG_DIR and DATA_DIR paths are model-visible
rm -rf $R && mkdir -p $R/config $R/home $R/data $R/cache/opencode/bin
ln -s $REPO/opencode/opencode.json $REPO/opencode/agent $REPO/opencode/skills $REPO/opencode/plugin $R/config/
# ripgrep 15.1.0 on PATH, or: cp <rg 15.1.0> $R/cache/opencode/bin/rg
env -i PATH=/usr/local/bin:/usr/bin:/bin LANG=C.UTF-8 \
  HOME=$R/home XDG_CONFIG_HOME=$R/home/.config XDG_DATA_HOME=$R/data \
  XDG_CACHE_HOME=$R/cache XDG_STATE_HOME=$R/home/.local/state \
  OPENCODE_CONFIG_DIR=$R/config \
  OPENCODE_DISABLE_AUTOUPDATE=1 OPENCODE_DISABLE_MODELS_FETCH=1 OPENCODE_DISABLE_LSP_DOWNLOAD=1 \
  OPENCODE_DISABLE_CLAUDE_CODE=1 OPENCODE_DISABLE_EXTERNAL_SKILLS=1 \
  NPM_CONFIG_REGISTRY=http://127.0.0.1:9/ \
  MAESTRO_BASE_URL=http://127.0.0.1:8000/v1 MAESTRO_MODEL=maestro \
  opencode run --agent maestro --format json < prompt.txt      # run from the rollout cwd
```

`docs/opencode_capture/goldens/capture/capture.sh` stages the config the same way (with
`OFFLINE=1 RG_BIN=...` for the no-network part). The goldens were captured with it, verifying:

- **(a) No personal config.** No `/home/...` path, personal skill or `Instructions from:` appears in
  any captured request. With the plugin loaded, system prompts and tools are byte-identical to
  stock.
- **(b) No writes into the repo.** The symlink targets are unchanged after a run, online or
  offline. Online, OpenCode's npm install writes `package.json`, `package-lock.json` and a 63 MB
  `node_modules/` into the staging dir and `$R/home/.config/opencode`, not through the symlinks.
- **(c) No network.** At startup OpenCode runs a background
  `npm install @opencode-ai/plugin@1.19.3` into every config dir; no config setting turns it off.
  The plugin doesn't need it, because its only import of that package is `import type` (erased at
  load). With `NPM_CONFIG_REGISTRY` pointing at a dead port, the install fails at once with a
  logged WARN ("background dependency install failed") and writes nothing. Requests are
  byte-identical to an online run, apart from orders OpenCode doesn't fix: the skill-file sample
  and glob results.
- **(d) The plugin loads once.** It is picked up only by auto-discovery of `plugin/*.ts` in the
  config dir; opencode.json has no `plugin` entry. A probe plugin staged next to it ran its init
  exactly once per run.

Notes:
- **Model-visible paths.** Two paths show up in what the model sees. The skill `<location>` and
  base directory are under `$R/config` (the symlink path, not the repo path). Bash spill files
  are under `$R/data/opencode/tool-output/`. `opencode_contract.CONFIG_DIR` / `DATA_DIR` assume
  `R=/tmp/maestro-oc`; keep the two in sync.
- **Concurrency is untested.** Concurrent runs sharing one `$R` would share the session DB. Use one
  root per container.
- **Model name.** `MAESTRO_BASE_URL` / `MAESTRO_MODEL` fill `{env:...}` in opencode.json. The
  system prompt then says `The exact model ID is maestro/<MAESTRO_MODEL>`.
- **Prompt on stdin.** `opencode run "two words"` wraps every positional argument that contains a
  space in double quotes before it reaches the model.
- **Instruction files.** OpenCode picks up `AGENTS.md`/`CLAUDE.md` from the cwd up to the
  workspace root, which is `/` outside a git repo. Keep rollout dirs and their parents free of
  them.
- **ripgrep.** glob, grep and skill (its `<skill_files>` sample) shell out to `rg`. OpenCode uses
  `rg` from PATH or `$XDG_CACHE_HOME/opencode/bin/rg`; otherwise it downloads ripgrep 15.1.0 from
  GitHub. Offline without it, those tools return `ripgrep execution failed`.

## Config choices

- Full stock tool set; nothing is denied. Primary agents get bash, edit, glob, grep, read, skill,
  task, todowrite, webfetch, write. Subagents get the same minus task and todowrite (OpenCode
  default).
- `general` / `explore` are disabled, so the task tool lists only our subagents.
- The built-in `customize-opencode` skill is denied, which hides it from `<available_skills>`.
  The skill tool's "not found" message still names it.
- `shell: /bin/bash` pins the bash tool description (`Shell: bash`), whatever `$SHELL` is.
- `compaction.auto: false`, so history is never rewritten mid-rollout. `external_directory: allow`,
  because a headless run can't answer permission prompts.
- `opencode run` also sends one title-generation request per session to the same model
  (`docs/opencode_capture/goldens/title_request.json`).

## Re-capturing the goldens

After editing agents, skills or opencode.json, or after upgrading OpenCode:

```bash
docs/opencode_capture/goldens/capture/capture.sh /tmp/oc_capture
.venv/bin/python docs/opencode_capture/goldens/capture/extract.py /tmp/oc_capture
.venv/bin/python -m pytest tests/test_opencode_contract.py -q
```

The capture stages this dir without `plugin/`, so the top-level goldens record stock
OpenCode behaviour. The plugin goldens (`goldens/with_plugin/`, plus the image-modality check) come
from:

```bash
WITH_PLUGIN=1 docs/opencode_capture/goldens/capture/capture.sh /tmp/oc_plugin
.venv/bin/python docs/opencode_capture/goldens/capture/extract.py /tmp/oc_plugin with_plugin
CONFIG_CONTENT='{"provider":{"maestro":{"models":{"maestro":{"modalities":{"input":["text","audio","image"],"output":["text"]}}}}}}' \
  WITH_PLUGIN=1 docs/opencode_capture/goldens/capture/capture.sh /tmp/oc_img
.venv/bin/python docs/opencode_capture/goldens/capture/extract.py /tmp/oc_img with_plugin_image_modality
#   (keep only media_messages.json and meta.json in with_plugin_image_modality/)
```

The committed goldens were captured with `OFFLINE=1 RG_BIN=<ripgrep 15.1.0>` (no network).
