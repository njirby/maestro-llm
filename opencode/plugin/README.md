# OpenCode plugins

## `audio-read.ts` — let the stock `read` tool listen

OpenCode 1.19.3 attaches images/PDFs read with `read` but fails audio with `Cannot read binary file: <path>`.
This plugin makes `read` on `.wav` / `.mp3` / `.flac` / `.ogg` succeed and attach the audio, mirroring the
stock image path. No fork, and the advertised `read` schema/description is untouched (verified byte-identical
to `docs/opencode_capture/captured.jsonl`).

### Registration

`opencode.json`, path relative to the declaring config:

```json
"plugin": ["./plugin/audio-read.ts"]
```

(Or rely on auto-discovery: OpenCode loads `{plugin,plugins}/*.{ts,js}` from every config directory, e.g.
`OPENCODE_CONFIG_DIR=<repo>/opencode` or `<project>/.opencode/`. Use one or the other, not both.)
The plugin only imports types from `@opencode-ai/plugin`, so no `npm install` is needed.

The model must declare audio input, or OpenCode replaces the audio with
`ERROR: Cannot read audio (this model does not support audio input). Inform the user.`:

```json
"modalities": { "input": ["text", "audio"], "output": ["text"] }
```

### Wire shape (OpenAI-compatible provider, captured)

Model calls `read {"filePath": ".../tone.wav"}`; the next request carries:

```json
{"role": "assistant", "content": "", "tool_calls": [{"id": "call_0", "type": "function",
  "function": {"name": "read", "arguments": "{\"filePath\":\".../tone.wav\"}"}}]}
{"role": "tool", "tool_call_id": "call_0", "content": "Audio read successfully"}
{"role": "user", "content": [
  {"type": "text", "text": "Attached media from tool result:"},
  {"type": "input_audio", "input_audio": {"data": "UklGRhIFAgBXQVZF...", "format": "wav"}}]}
```

This is exactly the stock image layout (`"Image read successfully"` tool message, then a user message
`[{"type":"text","text":"Attached media from tool result:"},{"type":"image_url",...}]`).

- `data` is the raw file bytes, base64 (no `data:` prefix). `.wav` is sent byte-for-byte; `.mp3` byte-for-byte
  with `format: "mp3"`; `.flac`/`.ogg` are decoded with ffmpeg to 16-bit PCM WAV (`format: "wav"`), keeping
  sample rate and channel count. Nothing is resampled or downmixed; that is the server's job.
- Parallel reads in one assistant step: one tool message per call (in call order), then one user message with
  all audio parts in call order. If the same step also read images, OpenCode's image message comes first and
  the audio message follows it (two consecutive user messages).
- History: the audio is stored on the tool part (as images are), so later turns resend the same message.
- Tool part as stored/shown: `output` and `metadata.preview` = `Audio read successfully`, `title` =
  worktree-relative path, `attachments` = `[{type:"file", mime:"audio/wav"|"audio/mpeg", url:"data:...;base64,..."}]`.

### Behaviour

- Only existing regular files with those extensions are handled. Everything else (text, directories, images,
  PDFs, missing files incl. missing `.wav`, other binaries) goes through stock `read` unchanged; the test matrix
  shows byte-identical requests for those cases.
- Size cap: files over `MAESTRO_AUDIO_READ_MAX_BYTES` (default 64 MiB, ~6 min of 44.1 kHz stereo 16-bit) fail
  the call with `Audio file too large: <path> (<n> bytes, limit <m>)`. OpenCode itself has no audio cap. There
  is no duration cap; limits like vLLM's max audio length apply server-side.

### How it works

1. `tool.execute.before` (read, audio file exists): rewrite `filePath` to the file's parent directory so the
   stock tool succeeds with a directory listing instead of throwing.
2. `tool.execute.after`: put the original `filePath` back (the args object is the tool part's recorded input,
   so history and `tool_calls` show the real path), replace the result with `Audio read successfully`, and
   attach the audio. OpenCode 1.19.3 passes `attachments` through the after-hook output.
3. `experimental.chat.messages.transform`: OpenCode hoists only `image/*` and PDF tool attachments into the
   `Attached media from tool result:` user message. Other attachments stay inside the tool result, which
   `@ai-sdk/openai-compatible` JSON-stringifies into the tool message content (verified, unusable). The
   transform moves audio attachments into a synthetic user message with the same text label, placed right after
   the assistant step. It works on cloned parts, so the stored session is unchanged.

### Limitations

- `read` permission rules are evaluated against the parent directory, not the audio file (a rule like
  `"*.wav": "deny"` would not match). The default `read: allow` is unaffected.
- Depends on 1.19.3 internals: after-hook `attachments`, the media hoisting rule, and the
  `Attached media from tool result:` label (constant `t8` in the binary). Re-run the test matrix after any
  OpenCode upgrade.
- If the after-hook fails (too large, ffmpeg error), the call becomes a tool error with that message, and the
  path has already been restored.

### Test

`test/run_matrix.sh [workdir]` (CPU only, ~1 min): runs 12 cases (txt, dir, png, missing wav/txt, binary, wav,
relative wav, mp3, flac, ogg, parallel mix) through `opencode run` with and without the plugin. Each run uses
an isolated `HOME`, so `~/.config/opencode` is never read. A scripted capturing mock OpenAI server
(`test/mock_openai.py`) is used. `test/compare.py` asserts the tool schemas are identical, the non-audio
requests are byte-identical to stock, and the decoded audio matches the source.
