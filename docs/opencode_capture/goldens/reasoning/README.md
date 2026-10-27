# Reasoning echo (OpenCode 1.19.3, `@ai-sdk/openai-compatible`)

What OpenCode sends back to the model for earlier assistant turns that carried reasoning. Captured by
`opencode/replay/reasoning_probe.py` (replay mock + the real binary, offline, `--network none`); pinned
by `tests/test_replay.py::test_reasoning_echo_rule`.

The mock streams every assistant turn with reasoning, for the main agent (stock `build`: reasoning +
text + tool call, reasoning + tool call, reasoning + 2 parallel calls, reasoning + `task`, reasoning +
final text) and the `transcriber` subagent it dispatches (reasoning + call, reasoning + 2 parallel
calls, reasoning + final text). The reasoning is multi-line with a leading and trailing newline. Three
variants, one file each:

| file | how the mock streams reasoning |
|---|---|
| `reasoning_content.json` | `delta.reasoning_content` (vLLM reasoning parser), two deltas |
| `reasoning.json` | `delta.reasoning`, two deltas |
| `think_tag.json` | no parser: `<think>\n…\n</think>\n\n` at the head of `content` |

Each file lists every captured request (title request excluded) with, per earlier assistant message:
its keys, which fields hold the probe reasoning, whether it equals what was streamed, and its content;
plus the `<task ...>` results in that request.

## Rule

- **Every** earlier assistant turn of the same session is re-sent with its reasoning, verbatim
  (whitespace included), in the field **`reasoning_content`**, whichever delta field streamed it
  (`reasoning_content` or `reasoning`). Nothing is dropped and it is not only the latest turn; this
  holds for text, tool-call and parallel-tool-call turns alike.
- `content` stays the visible text only: the streamed text, or `""` for a turn with only tool calls.
  The reasoning never appears in `content` and is never wrapped in `<think>`.
- If reasoning arrives as a `<think>` block inside `content` (no reasoning parser), OpenCode keeps it
  verbatim in `content` and sends no `reasoning_content`.
- Sessions are separate: a subagent's requests hold only its own turns (none of the parent's
  reasoning). Across the task boundary the parent receives only the subagent's final `content` inside
  `<task_result>`; reasoning sent as `reasoning_content` does not cross, a `<think>` block in content
  does.

Re-capture after an OpenCode or provider upgrade:

```bash
.venv/bin/python -m opencode.replay.reasoning_probe --write-goldens
.venv/bin/python -m pytest tests/test_replay.py -q -k reasoning
```
