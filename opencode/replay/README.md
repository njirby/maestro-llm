# Conformance replay (B6)

Proves that synthetic records (ms-swift `{"id","messages","audios","tools"}`) are what real OpenCode
1.19.3 sends a model: a mock OpenAI endpoint plays the records' assistant steps back verbatim, the real
binary executes every tool call for real, and each request it sends is diffed against the record.

```bash
.venv/bin/python scripts/replay_in_opencode.py golden                      # self-test, with plugin
.venv/bin/python scripts/replay_in_opencode.py golden --variant '' --out outputs/replay/golden_stock
.venv/bin/python scripts/replay_in_opencode.py records MAIN.json SUB.json ... --init-dir DIR --out outputs/replay/<name>
.venv/bin/python scripts/replay_in_opencode.py report outputs/replay/<name>  # re-diff a run's log
.venv/bin/python -m pytest tests/test_replay.py -q                          # offline; MAESTRO_REPLAY_LIVE=1 adds the live run
```

Exit 0 iff every compared request matches after the whitelist and every record step was requested.
Each run writes `<out>/report.{json,md}` and `<out>/io/` (scenario, records, `requests.jsonl` with media
as sha256 digests, OpenCode's `run.jsonl`/`run.err`).

## Execution model

One throwaway `daw-farm/reaper:latest` container per replay (`wsd-replay-*`, `--network none`, uid
1000, Xvfb :99) runs the mock (`mock_server.py`, stdlib only), OpenCode and every tool it executes, so
paths and skill scripts are those of `maestro.dawfarm_dd.DawDreamerSession`:

| container path | host | |
|---|---|---|
| `/work/rollouts/<sid>` | `<out>/rollouts/<sid>` (copy of `--init-dir`) | cwd; `<sid>` taken from the record's `Working directory:` |
| `/tmp/maestro-oc` | `<out>/oc_root` | staging root of `opencode/README.md`, fresh per run |
| `<repo>/opencode` | same (ro) | targets of the staging symlinks; skill = `/tmp/maestro-oc/config/skills/vital` |
| `/opt/opencode/bin/opencode` | `~/.opencode/bin/opencode` (ro) | |
| `/usr/local/bin/rg` | `outputs/replay/bin/rg-15.1.0` (ro, sha256-checked) | also copied to `$XDG_CACHE_HOME/opencode/bin/rg` |

OpenCode runs as `env -i <README env> opencode run --agent maestro --format json < prompt` with
`DISPLAY`, `VITAL_VST3`, `VITAL_SKILL` added (as DawDreamerSession sets them container-wide). Since the
staging root is a bind mount, the host's `/tmp` is not used and concurrent replays never share a root.

## Routing

A session's first request is routed by (first line of the system prompt, normalized first user
message): the agent prompt names the agent, the first user message (the task prompt, verbatim) the
session. Records with identical keys are told apart by the assistant steps already in the request,
and a step-0 request takes the first unclaimed one. Later requests stick to the record via the
`x-opencode-session-id` header; a header-less request whose messages equal an earlier one's (a retry)
gets that request's record. The step is the number of assistant messages in the request. Title
requests get a fixed title and are diffed exactly against `goldens/title_request.json` (prompt and
model filled in).

## Diff

Requests are converted to record form (`records.wire_to_record`) and compared to the record prefix
before the requested step, after normalization; also compared: `tools` (exact), media (sha256 of
decoded bytes vs the record's `audios` files, in order) and request params (golden
`request_params.json`, model from the record). Whitelist (`records.WHITELIST`): the `<env>` date,
`ses_` ids, `tool_` spill ids, `/tmp/tmp*` paths, the `<skill_files>` sample (count and base dir kept),
and glob result order. Tool-call ids are not compared (records carry none).

Re-rendered audio: Vital renders are not bit-identical (oscillator phase, free-running LFOs). A media
item whose sha256 differs passes (`whitelisted: media_render` in the report) when the record's clip is
not a file of the initial workdir (those must match exactly), lengths agree within 512 samples, and the
mean |log-mel diff| (`audio_eq.py`, the metric of `scripts/check_render_parity.py`) is at most
`--max-mel-db` (default 1.5 dB; same-preset render noise is 0.2-0.7 dB). The mock saves every media
payload to `io/media/<sha256>.<ext>` for this. A message diff that disappears when all numbers are
masked is reported as kind `numeric` (timing/randomness in tool output), separate from `messages`.
`report.md` lists diff counts by kind.

## Findings

- `opencode run -f <file.wav>` does **not** attach audio: run mode turns `-f` into a synthetic Read in
  the first user message (`Called the Read tool with...` / `Cannot read binary file`), bypassing the
  audio plugin. User-turn audio would need the serve API's file parts (untested); v5 records don't
  need it (audio arrives via `read`).
