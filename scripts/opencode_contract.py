#!/usr/bin/env python
"""The opencode-conformant tool contract — single source of truth.

Every rollout builder AND the serving harness must express tool calls,
tool results, errors, the system message, and subagent returns through
these helpers, so training data and deployment stay byte-compatible.

Decisions encoded here (2026-08-15, from the opencode conformance map):
  - lowercase tool ids with opencode signatures: bash / read / skill / task
  - bash args: {command, timeout?: MILLISECONDS, workdir?}
  - read args: {filePath, offset?, limit?}; output is line-numbered text
  - task args: {description, prompt, subagent_type}; result returns INLINE
    as <task id=... state=...><task_result>...</task_result></task>
  - tool outputs are plain strings, never JSON envelopes
  - errors are the error string alone (no is_error flag in content), and
    invalid arguments use opencode's canonical recovery wording
  - a leading system message is always present (env block + agent prompt)
  - audio stays as input_audio parts on the tool message at serving time;
    in ms-swift records that is the <audio> placeholder inside the
    tool_response content (positionally aligned with the `audios` list)
"""

from __future__ import annotations

import json
from typing import Any

# ---------------------------------------------------------------------------
# Tool schemas (sent to the model as OpenAI function schemas at serving time;
# embedded in records' `tools` field so the training template renders them)
# ---------------------------------------------------------------------------

TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Executes the given shell command and returns its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The command to execute"},
                    "timeout": {"type": "integer",
                                "description": "Optional timeout in milliseconds"},
                    "workdir": {"type": "string",
                                "description": "Working directory for the command"},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Reads a file from the filesystem. Text files return "
                           "line-numbered content; audio files are attached so you can "
                           "listen to them.",
            "parameters": {
                "type": "object",
                "properties": {
                    "filePath": {"type": "string", "description": "Absolute path to the file"},
                    "offset": {"type": "integer", "description": "Line to start from"},
                    "limit": {"type": "integer", "description": "Max lines to read"},
                },
                "required": ["filePath"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "skill",
            "description": "Load a skill's instructions by name.",
            "parameters": {
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Skill name"}},
                "required": ["name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "task",
            "description": "Delegate a task to a subagent. Returns the subagent's "
                           "final result inline when it completes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string",
                                    "description": "Short (3-5 word) task description"},
                    "prompt": {"type": "string", "description": "The task for the subagent"},
                    "subagent_type": {"type": "string",
                                      "description": "The type of subagent to use"},
                },
                "required": ["description", "prompt", "subagent_type"],
            },
        },
    },
]

TOOL_NAMES = {t["function"]["name"] for t in TOOLS}


# ---------------------------------------------------------------------------
# Message-side helpers (ms-swift record roles: tool_call / tool_response)
# ---------------------------------------------------------------------------

def tool_call(name: str, arguments: dict[str, Any]) -> str:
    """Content string for a role=tool_call message."""
    assert name in TOOL_NAMES, f"unknown tool {name!r}"
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False)


def bash_call(command: str, timeout_ms: int | None = None,
              workdir: str | None = None) -> str:
    args: dict[str, Any] = {"command": command}
    if timeout_ms is not None:
        args["timeout"] = int(timeout_ms)
    if workdir is not None:
        args["workdir"] = workdir
    return tool_call("bash", args)


def read_call(file_path: str) -> str:
    return tool_call("read", {"filePath": file_path})


def task_call(description: str, prompt: str, subagent_type: str) -> str:
    return tool_call("task", {"description": description, "prompt": prompt,
                              "subagent_type": subagent_type})


def bash_output(stdout: str, exit_code: int = 0, stderr: str = "") -> str:
    """Plain-string bash result. Success -> raw stdout. Failure -> error shape."""
    if exit_code == 0:
        return stdout if stdout.strip() else "(no output)"
    parts = [f"Command failed with exit code {exit_code}."]
    if stderr.strip():
        parts.append(stderr.rstrip())
    if stdout.strip():
        parts.append(stdout.rstrip())
    return "\n".join(parts)


def read_output_text(content: str, offset: int = 0) -> str:
    """Line-numbered text file content, opencode style: `{n}: {line}`."""
    lines = content.splitlines()
    return "\n".join(f"{i + 1 + offset}: {l}" for i, l in enumerate(lines))


def read_output_audio(filename: str, duration_s: float, sample_rate: int) -> str:
    """Text half of an audio read; the harness attaches the audio itself as an
    input_audio part, which in ms-swift records is the <audio> placeholder
    prepended to this string in the tool_response content."""
    return f"Attached audio: {filename} ({duration_s:.2f}s, {sample_rate} Hz)"


def task_result(session_id: str, text: str, state: str = "completed") -> str:
    """Inline subagent return — the tool_response content of a task call."""
    return (f'<task id="{session_id}" state="{state}">\n'
            f"<task_result>\n{text}\n</task_result>\n</task>")


def invalid_arguments_error(tool: str, detail: str) -> str:
    """opencode's canonical model-facing invalid-arguments wording."""
    return (f"The {tool} tool was called with invalid arguments: {detail}.\n"
            "Please rewrite the input so it satisfies the expected schema.")


# ---------------------------------------------------------------------------
# System message
# ---------------------------------------------------------------------------

def system_message(agent_prompt: str, cwd: str, platform: str = "linux") -> str:
    """Leading system turn: agent prompt + environment block (opencode-style)."""
    env = (f"<env>\n  Working directory: {cwd}\n  Platform: {platform}\n"
           f"  Shell: bash\n</env>")
    return f"{agent_prompt.rstrip()}\n\n{env}"


AGENT_PROMPTS = {
    "main": (
        "You are a synthesizer sound-design agent working inside REAPER with the "
        "Vital synthesizer. Recreate the target sound you are given: transcribe its "
        "melody, search for matching wavetables, judge candidates, shape parameters, "
        "then render and listen to verify your work against the target. Use the "
        "tools available to you; always verify audibly before concluding."
    ),
    "melody_transcription": (
        "You are a melody-transcription agent. Listen to the target audio and write "
        "Python (reapy) that inserts the melody's MIDI notes on the given track, "
        "then render and listen to verify the transcription against the target."
    ),
    "wavetable_search": (
        "You are a wavetable-search agent. Given a target sound and a slice of the "
        "wavetable library, render each candidate with the transcribed melody, listen, "
        "and shortlist the candidates whose character best matches the target. Give a "
        "terse verdict per candidate and end with your shortlist."
    ),
    "wavetable_judge": (
        "You are a wavetable-judge agent. Given the target sound and the search "
        "agents' shortlisted candidates, listen and select the final wavetable(s) "
        "for the patch, explaining briefly what each selected candidate contributes."
    ),
}
